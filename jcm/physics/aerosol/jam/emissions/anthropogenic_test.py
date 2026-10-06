"""Tests for prescribed CEDS anthropogenic emissions (#498, Phase A)."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.emissions.anthropogenic import (
    AnthropogenicEmissions,
    EmissionParameters,
)
from jcm.physics.aerosol.jam.emissions.sectors import (
    OM_OC_RATIO,
    SO2_TO_SO4_MASS,
    SO4_PRIMARY_FRACTION,
    SUPER_SECTORS,
)
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.species import SPECIES
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name
from jcm.physics_interface import PhysicsState

_NLEV, _NCOLS = 5, 2
_F_SO2, _F_BC, _F_OC = 2.0e-9, 1.0e-9, 3.0e-9   # kg/m²/s


def _setup(**fluxes):
    state = PhysicsState.zeros((_NLEV, _NCOLS)).copy(
        temperature=jnp.full((_NLEV, _NCOLS), 280.0),
    )
    diagnostics = {
        "air_density": jnp.full((_NLEV, _NCOLS), 1.0),
        "layer_thickness": jnp.full((_NLEV, _NCOLS), 200.0),
        "height_full": jnp.broadcast_to(
            jnp.asarray([4000.0, 2000.0, 1000.0, 300.0, 50.0])[:, None],
            (_NLEV, _NCOLS),
        ),
    }
    # The term reads per-channel fluxes from the ``anthropogenic_emissions``
    # mapping on ``ForcingData`` (keyed ``emis_<sector>_<species>``); mirror
    # that here so the test exercises the real forcing contract.
    forcing = types.SimpleNamespace(
        anthropogenic_emissions={
            k: jnp.full((_NCOLS,), v) for k, v in fluxes.items()
        }
    )
    return state, diagnostics, forcing


def _column_integral(tend, rho, dz):
    """Σ ρ_k Δz_k · tend_k  → the recovered surface flux [/m²/s], (ncols,)."""
    return np.asarray(jnp.sum(tend * rho * dz, axis=0))


class AnthropogenicEmissionsTest(unittest.TestCase):
    def test_zero_without_forcing(self):
        state, diagnostics, _ = _setup()
        tend, _ = AnthropogenicEmissions()(state, diagnostics, None, None)
        for v in tend.tracers.values():
            self.assertTrue(np.all(np.asarray(v) == 0.0))

    def test_primary_so4_split_and_gas_remainder(self):
        state, diagnostics, forcing = _setup(emis_surface_combustion_so2=_F_SO2)
        tend, _ = AnthropogenicEmissions()(state, diagnostics, forcing, None)
        rho, dz = diagnostics["air_density"], diagnostics["layer_thickness"]

        so4 = (_column_integral(tend.tracers[mass_name("so4", "ait")], rho, dz)
               + _column_integral(tend.tracers[mass_name("so4", "acc")], rho, dz))
        gso2 = _column_integral(tend.tracers[gas_name("so2")], rho, dz)
        # 2.5% of S → primary SO4 mass; 97.5% → SO2 gas.
        np.testing.assert_allclose(
            so4, SO4_PRIMARY_FRACTION * _F_SO2 * SO2_TO_SO4_MASS, rtol=1e-5
        )
        np.testing.assert_allclose(
            gso2, (1.0 - SO4_PRIMARY_FRACTION) * _F_SO2, rtol=1e-5
        )

    def test_sulfur_conserved(self):
        state, diagnostics, forcing = _setup(emis_elevated_industrial_so2=_F_SO2)
        tend, _ = AnthropogenicEmissions()(state, diagnostics, forcing, None)
        rho, dz = diagnostics["air_density"], diagnostics["layer_thickness"]
        m_so2 = GAS_SPECIES["so2"].molar_mass
        m_so4 = SPECIES["so4"].molar_mass
        s_so4 = (
            _column_integral(tend.tracers[mass_name("so4", "ait")], rho, dz)
            + _column_integral(tend.tracers[mass_name("so4", "acc")], rho, dz)
        ) / m_so4
        s_gas = _column_integral(tend.tracers[gas_name("so2")], rho, dz) / m_so2
        np.testing.assert_allclose(s_so4 + s_gas, _F_SO2 / m_so2, rtol=1e-5)

    def test_bc_and_oc_to_primary_carbon(self):
        state, diagnostics, forcing = _setup(
            emis_shipping_bc=_F_BC, emis_shipping_oc=_F_OC,
        )
        tend, _ = AnthropogenicEmissions()(state, diagnostics, forcing, None)
        rho, dz = diagnostics["air_density"], diagnostics["layer_thickness"]
        bc = _column_integral(tend.tracers[mass_name("bc", "pcm")], rho, dz)
        poa = _column_integral(tend.tracers[mass_name("poa", "pcm")], rho, dz)
        np.testing.assert_allclose(bc, _F_BC, rtol=1e-5)
        np.testing.assert_allclose(poa, _F_OC * OM_OC_RATIO, rtol=1e-5)  # OM:OC

    def test_grad_through_injection_and_so4_params_finite(self):
        state, diagnostics, forcing = _setup(
            emis_surface_combustion_so2=_F_SO2,
            emis_surface_combustion_bc=_F_BC,
        )

        def loss(height, thickness, frac):
            base = EmissionParameters.default()
            p = EmissionParameters(
                injection_height=base.injection_height.at[0].set(height),
                injection_thickness=base.injection_thickness.at[0].set(thickness),
                so4_primary_fraction=base.so4_primary_fraction.at[0].set(frac),
                scale=base.scale,
            )
            tend, _ = AnthropogenicEmissions(params=p)(
                state, diagnostics, forcing, None
            )
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss, argnums=(0, 1, 2))(
            jnp.asarray(40.0), jnp.asarray(50.0), jnp.asarray(0.025)
        )
        for gi in g:
            self.assertTrue(np.isfinite(float(gi)))
        # Injection height genuinely matters (non-zero gradient).
        self.assertGreater(abs(float(g[0])), 0.0)


class MAM4IgnoresExtraChannelsTest(unittest.TestCase):
    """MAM4 (default ``spec``, ``sector_policy is None``) must be completely

    unaffected by a bundle that also carries the residential/energy
    HAM-sizing subset channels (#1017 F6): MAM4's branch only ever looks up
    ``emis_<super_sector>_<species>`` (:func:`AnthropogenicEmissions._flux`
    called with exactly those names in the ``sector_policy is None``
    branch), never reads ``forcing.anthropogenic_emissions`` as a whole, so
    extra keys it never names must change nothing it computes.
    """

    def test_extra_residential_and_energy_channels_are_bit_identical(self):
        base_fluxes = dict(
            emis_surface_combustion_so2=_F_SO2,
            emis_surface_combustion_bc=_F_BC,
            emis_surface_combustion_oc=_F_OC,
            emis_elevated_industrial_so2=_F_SO2 * 0.5,
            emis_elevated_industrial_bc=_F_BC * 0.5,
        )
        extra_fluxes = dict(
            base_fluxes,
            # Present in a bundle built with the new subset channels, but
            # never named by MAM4's lookup -- so2 subsets use the SO2 name
            # per the forcing contract (``anthropogenic.py``'s own
            # ``channel = f"emis_{subset_name}_{'so2' if species=='so4' ...}"``).
            emis_residential_so2=_F_SO2 * 0.3,
            emis_residential_bc=_F_BC * 0.3,
            emis_residential_oc=_F_OC * 0.3,
            emis_energy_so2=_F_SO2 * 0.2,
            emis_energy_bc=_F_BC * 0.2,
        )
        state_a, diagnostics_a, forcing_a = _setup(**base_fluxes)
        state_b, diagnostics_b, forcing_b = _setup(**extra_fluxes)
        term = AnthropogenicEmissions()
        tend_a, _ = term(state_a, diagnostics_a, forcing_a, None)
        tend_b, _ = term(state_b, diagnostics_b, forcing_b, None)

        self.assertEqual(set(tend_a.tracers), set(tend_b.tracers))
        for name in tend_a.tracers:
            np.testing.assert_array_equal(
                np.asarray(tend_a.tracers[name]), np.asarray(tend_b.tracers[name]),
                err_msg=f"{name} differs: MAM4 must ignore the extra subset channels")


def _mass_weighted_height(tend, rho, dz, height, col=0):
    """Column mass-weighted injection height [m] of a tracer tendency."""
    w = np.asarray((tend * rho * dz)[:, col])
    h = np.asarray(height[:, col])
    return float(np.sum(w * h) / np.sum(w))


class BiomassBurningTest(unittest.TestCase):
    """Phase E: open biomass burning as a 4th super-sector with FIRE injection."""

    def test_biomass_bc_to_primary_carbon_mass_conserving(self):
        # Biomass BC/OC use the same speciation as anthropogenic carbon (MAM4
        # routes both to the single primary-carbon mode), just a deeper profile.
        state, diagnostics, forcing = _setup(
            emis_biomass_burning_bc=_F_BC, emis_biomass_burning_oc=_F_OC)
        tend, _ = AnthropogenicEmissions()(state, diagnostics, forcing, None)
        rho, dz = diagnostics["air_density"], diagnostics["layer_thickness"]
        bc = _column_integral(tend.tracers[mass_name("bc", "pcm")], rho, dz)
        poa = _column_integral(tend.tracers[mass_name("poa", "pcm")], rho, dz)
        np.testing.assert_allclose(bc, _F_BC, rtol=1e-5)
        np.testing.assert_allclose(poa, _F_OC * OM_OC_RATIO, rtol=1e-5)

    def test_fire_profile_is_deeper_than_surface(self):
        # The FIRE default injects much higher than surface combustion: the
        # mass-weighted height of the emitted BC must be substantially larger.
        state, diagnostics, _ = _setup()
        rho, dz = diagnostics["air_density"], diagnostics["layer_thickness"]
        h = diagnostics["height_full"]

        _, _, f_fire = _setup(emis_biomass_burning_bc=_F_BC)
        _, _, f_surf = _setup(emis_surface_combustion_bc=_F_BC)
        t_fire, _ = AnthropogenicEmissions()(state, diagnostics, f_fire, None)
        t_surf, _ = AnthropogenicEmissions()(state, diagnostics, f_surf, None)
        bc = mass_name("bc", "pcm")
        z_fire = _mass_weighted_height(t_fire.tracers[bc], rho, dz, h)
        z_surf = _mass_weighted_height(t_surf.tracers[bc], rho, dz, h)
        self.assertGreater(z_fire, z_surf + 300.0)

    def test_grad_through_biomass_injection_height_finite(self):
        i = SUPER_SECTORS.index("biomass_burning")
        state, diagnostics, forcing = _setup(emis_biomass_burning_bc=_F_BC)

        def loss(height):
            base = EmissionParameters.default()
            p = EmissionParameters(
                injection_height=base.injection_height.at[i].set(height),
                injection_thickness=base.injection_thickness,
                so4_primary_fraction=base.so4_primary_fraction,
                scale=base.scale,
            )
            tend, _ = AnthropogenicEmissions(params=p)(
                state, diagnostics, forcing, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = float(jax.grad(loss)(jnp.asarray(1000.0)))
        self.assertTrue(np.isfinite(g))
        self.assertGreater(abs(g), 0.0)


class M7SectorEmissionTest(unittest.TestCase):
    """``AnthropogenicEmissions`` on the M7 population (jax-gcm#1017)."""

    def _run(self, **fluxes):
        from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC

        state, diagnostics, forcing = _setup(**fluxes)
        tend, _ = AnthropogenicEmissions(spec=M7_SPEC)(
            state, diagnostics, forcing, None)
        rho, dz = diagnostics["air_density"], diagnostics["layer_thickness"]
        return tend, rho, dz

    def _zm2n(self, mode_short, species, cmr_m):
        from jcm.physics.aerosol.jam.emissions.distributors import particle_mean_mass
        from jcm.physics.aerosol.jam.emissions.ham_sectors import cmr_to_emission_diameter
        from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC

        mode = M7_SPEC.mode(mode_short)
        d = cmr_to_emission_diameter(cmr_m, mode.geom_std_dev)
        density = M7_SPEC.species_props(species).density
        return 1.0 / particle_mean_mass(mode, density, emission_diameter=d)

    def test_fossil_so2_splits_ks_as_sulfur_closes_number_matches_zm2n(self):
        from jcm.physics.aerosol.jam.emissions.ham_sectors import CMR_SA, CMR_SK
        from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
        from jcm.physics.aerosol.jam.tracer_layout import (
            gas_name, mass_name, number_name,
        )

        tend, rho, dz = self._run(emis_surface_combustion_so2=_F_SO2)
        so2_to_so4 = (M7_SPEC.species_props("so4").molar_mass
                      / GAS_SPECIES["so2"].molar_mass)
        total_so4 = SO4_PRIMARY_FRACTION * _F_SO2 * so2_to_so4

        m_ks = _column_integral(tend.tracers[mass_name("so4", "ks")], rho, dz)
        m_as = _column_integral(tend.tracers[mass_name("so4", "as")], rho, dz)
        np.testing.assert_allclose(m_ks, 0.5 * total_so4, rtol=1e-5)
        np.testing.assert_allclose(m_as, 0.5 * total_so4, rtol=1e-5)

        gso2 = _column_integral(tend.tracers[gas_name("so2")], rho, dz)
        np.testing.assert_allclose(
            gso2 + total_so4 / so2_to_so4, _F_SO2, rtol=1e-5,
            err_msg="SO2 -> SO4 sulfur mass must close")

        n_ks = _column_integral(tend.tracers[number_name("ks")], rho, dz)
        np.testing.assert_allclose(
            n_ks, 0.5 * total_so4 * self._zm2n("ks", "so4", CMR_SK), rtol=1e-5)
        n_as = _column_integral(tend.tracers[number_name("as")], rho, dz)
        np.testing.assert_allclose(
            n_as, 0.5 * total_so4 * self._zm2n("as", "so4", CMR_SA), rtol=1e-5)

        # No mass lands anywhere else that could carry so4 (ns, cs).
        for short in ("ns", "cs"):
            self.assertTrue(
                mass_name("so4", short) not in tend.tracers
                or np.all(np.asarray(tend.tracers[mass_name("so4", short)]) == 0.0))

    def test_shipping_so4_splits_as_cs_not_ks(self):
        from jcm.physics.aerosol.jam.emissions.ham_sectors import CMR_SC
        from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
        from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

        tend, rho, dz = self._run(emis_shipping_so2=_F_SO2)
        so2_to_so4 = (M7_SPEC.species_props("so4").molar_mass
                      / GAS_SPECIES["so2"].molar_mass)
        total_so4 = SO4_PRIMARY_FRACTION * _F_SO2 * so2_to_so4

        m_as = _column_integral(tend.tracers[mass_name("so4", "as")], rho, dz)
        m_cs = _column_integral(tend.tracers[mass_name("so4", "cs")], rho, dz)
        np.testing.assert_allclose(m_as, 0.5 * total_so4, rtol=1e-5)
        np.testing.assert_allclose(m_cs, 0.5 * total_so4, rtol=1e-5)
        self.assertTrue(
            mass_name("so4", "ks") not in tend.tracers
            or np.all(np.asarray(tend.tracers[mass_name("so4", "ks")]) == 0.0))

        n_cs = _column_integral(tend.tracers[number_name("cs")], rho, dz)
        np.testing.assert_allclose(
            n_cs, 0.5 * total_so4 * self._zm2n("cs", "so4", CMR_SC), rtol=1e-5)

    def test_biomass_burning_oc_splits_ki_ks_at_cmr_bb(self):
        from jcm.physics.aerosol.jam.emissions.ham_sectors import BB_WSOC_FRACTION, CMR_BB
        from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

        tend, rho, dz = self._run(emis_biomass_burning_oc=_F_OC)
        om = _F_OC * OM_OC_RATIO

        m_ki = _column_integral(tend.tracers[mass_name("oc", "ki")], rho, dz)
        m_ks = _column_integral(tend.tracers[mass_name("oc", "ks")], rho, dz)
        np.testing.assert_allclose(m_ki, (1.0 - BB_WSOC_FRACTION) * om, rtol=1e-5)
        np.testing.assert_allclose(m_ks, BB_WSOC_FRACTION * om, rtol=1e-5)

        n_ki = _column_integral(tend.tracers[number_name("ki")], rho, dz)
        np.testing.assert_allclose(
            n_ki, (1.0 - BB_WSOC_FRACTION) * om * self._zm2n("ki", "oc", CMR_BB),
            rtol=1e-5)

    def test_residential_subset_channel_is_sized_as_biomass_cmr_bb(self):
        """The residential share of surface_combustion BC gets cmr_bb, not cmr_ff."""
        from jcm.physics.aerosol.jam.emissions.ham_sectors import CMR_BB, CMR_FF
        from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

        f_main, f_residential = 4.0e-9, 1.5e-9
        # #1017 F6: when the subset channel IS present, the policy reads it
        # silently for THAT (sector, species) -- other (sector, species)
        # pairs this fixture leaves genuinely unforced still warn (e.g.
        # residential so4/oc, the whole energy triple), so this checks the
        # one channel under test is absent from the warnings rather than
        # asserting no warnings fired at all.
        with self.assertLogs(
                "jcm.physics.aerosol.jam.emissions.anthropogenic",
                level="WARNING") as cm:
            tend, rho, dz = self._run(
                emis_surface_combustion_bc=f_main, emis_residential_bc=f_residential)
        self.assertFalse(any("emis_residential_bc" in m for m in cm.output))

        m_ki = _column_integral(tend.tracers[mass_name("bc", "ki")], rho, dz)
        np.testing.assert_allclose(m_ki, f_main, rtol=1e-5,
                                   err_msg="mass is unaffected by the size split")

        n_ki = _column_integral(tend.tracers[number_name("ki")], rho, dz)
        expected = ((f_main - f_residential) * self._zm2n("ki", "bc", CMR_FF)
                    + f_residential * self._zm2n("ki", "bc", CMR_BB))
        np.testing.assert_allclose(n_ki, expected, rtol=1e-5)

    def test_missing_residential_channel_falls_back_to_fossil_size_and_warns(self):
        from jcm.physics.aerosol.jam.emissions.ham_sectors import CMR_FF
        from jcm.physics.aerosol.jam.tracer_layout import number_name

        with self.assertLogs(
                "jcm.physics.aerosol.jam.emissions.anthropogenic",
                level="WARNING") as cm:
            tend, rho, dz = self._run(emis_surface_combustion_bc=_F_BC)
        self.assertTrue(any("emis_residential_bc" in m for m in cm.output))

        n_ki = _column_integral(tend.tracers[number_name("ki")], rho, dz)
        np.testing.assert_allclose(n_ki, _F_BC * self._zm2n("ki", "bc", CMR_FF),
                                   rtol=1e-5)

    def test_biogenic_oc_channel_splits_ki_ks_as_no_om_oc_scaling(self):
        from jcm.physics.aerosol.jam.emissions.ham_sectors import BG_WSOC_FRACTION, CMR_BG
        from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

        f_bg_oc = 2.0e-9
        tend, rho, dz = self._run(emis_biogenic_oc=f_bg_oc)

        m_ki = _column_integral(tend.tracers[mass_name("oc", "ki")], rho, dz)
        m_ks = _column_integral(tend.tracers[mass_name("oc", "ks")], rho, dz)
        m_as = _column_integral(tend.tracers[mass_name("oc", "as")], rho, dz)
        # No OM_OC_RATIO scaling: the three fractions sum to exactly f_bg_oc.
        np.testing.assert_allclose(m_ki, (1.0 - BG_WSOC_FRACTION) * f_bg_oc, rtol=1e-5)
        np.testing.assert_allclose(m_ks, 0.5 * BG_WSOC_FRACTION * f_bg_oc, rtol=1e-5)
        np.testing.assert_allclose(m_as, 0.5 * BG_WSOC_FRACTION * f_bg_oc, rtol=1e-5)
        np.testing.assert_allclose(m_ki + m_ks + m_as, f_bg_oc, rtol=1e-5)

        n_ki = _column_integral(tend.tracers[number_name("ki")], rho, dz)
        np.testing.assert_allclose(
            n_ki,
            (1.0 - BG_WSOC_FRACTION) * f_bg_oc * self._zm2n("ki", "oc", CMR_BG),
            rtol=1e-5)
        # No explicit number tendency on the soluble biogenic targets.
        self.assertTrue(
            number_name("ks") not in tend.tracers
            or np.all(np.asarray(tend.tracers[number_name("ks")]) == 0.0))
        self.assertTrue(
            number_name("as") not in tend.tracers
            or np.all(np.asarray(tend.tracers[number_name("as")]) == 0.0))

    def test_mam4_never_reads_biogenic_channel(self):
        # MAM4's population has no "biogenic" class, and sector_emission is
        # None, so AnthropogenicEmissions() never even looks at the channel.
        state, diagnostics, forcing = _setup(emis_biogenic_oc=2.0e-9)
        tend, _ = AnthropogenicEmissions()(state, diagnostics, forcing, None)
        for v in tend.tracers.values():
            self.assertTrue(np.all(np.asarray(v) == 0.0))
    def test_energy_subset_channel_splits_so4_into_as_cs_not_ks_as(self):
        """``elevated_industrial``'s ``energy`` subset (#1017 F6): its SO4

        share routes to accumulation+coarse (``energy_ships``'s own
        targets), while the REMAINING ``elevated_industrial`` flux keeps
        fossil's Aitken+accumulation split -- present with no warning.
        """
        from jcm.physics.aerosol.jam.emissions.ham_sectors import (
            CMR_SA, CMR_SC, CMR_SK,
        )
        from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
        from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

        f_main_so2, f_energy_so2 = 5.0e-9, 2.0e-9
        # Other (sector, species) pairs this fixture leaves unforced still
        # warn (e.g. energy bc/oc) -- check the one channel under test is
        # absent from the warnings, not that none fired at all.
        with self.assertLogs(
                "jcm.physics.aerosol.jam.emissions.anthropogenic",
                level="WARNING") as cm:
            tend, rho, dz = self._run(
                emis_elevated_industrial_so2=f_main_so2,
                emis_energy_so2=f_energy_so2)
        self.assertFalse(any("emis_energy_so2" in m for m in cm.output))

        so2_to_so4 = (M7_SPEC.species_props("so4").molar_mass
                      / GAS_SPECIES["so2"].molar_mass)
        main_so4 = SO4_PRIMARY_FRACTION * (f_main_so2 - f_energy_so2) * so2_to_so4
        energy_so4 = SO4_PRIMARY_FRACTION * f_energy_so2 * so2_to_so4

        m_ks = _column_integral(tend.tracers[mass_name("so4", "ks")], rho, dz)
        m_as = _column_integral(tend.tracers[mass_name("so4", "as")], rho, dz)
        m_cs = _column_integral(tend.tracers[mass_name("so4", "cs")], rho, dz)
        np.testing.assert_allclose(m_ks, 0.5 * main_so4, rtol=1e-5)
        np.testing.assert_allclose(m_as, 0.5 * main_so4 + 0.5 * energy_so4, rtol=1e-5)
        np.testing.assert_allclose(m_cs, 0.5 * energy_so4, rtol=1e-5)

        n_ks = _column_integral(tend.tracers[number_name("ks")], rho, dz)
        n_cs = _column_integral(tend.tracers[number_name("cs")], rho, dz)
        np.testing.assert_allclose(
            n_ks, 0.5 * main_so4 * self._zm2n("ks", "so4", CMR_SK), rtol=1e-5)
        np.testing.assert_allclose(
            n_cs, 0.5 * energy_so4 * self._zm2n("cs", "so4", CMR_SC), rtol=1e-5)
        # "as" number mixes two different cmr-implied number factors (fossil's
        # CMR_SA share from the main flux, energy_ships' own CMR_SA share from
        # the subset -- same cmr here, so this is really just additivity).
        n_as = _column_integral(tend.tracers[number_name("as")], rho, dz)
        np.testing.assert_allclose(
            n_as,
            0.5 * main_so4 * self._zm2n("as", "so4", CMR_SA)
            + 0.5 * energy_so4 * self._zm2n("as", "so4", CMR_SA),
            rtol=1e-5)

    def test_missing_energy_channel_falls_back_to_fossil_size_and_warns(self):
        from jcm.physics.aerosol.jam.emissions.ham_sectors import CMR_SA, CMR_SK
        from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

        with self.assertLogs(
                "jcm.physics.aerosol.jam.emissions.anthropogenic",
                level="WARNING") as cm:
            tend, rho, dz = self._run(emis_elevated_industrial_so2=_F_SO2)
        self.assertTrue(any("emis_energy_so2" in m for m in cm.output))

        from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
        so2_to_so4 = (M7_SPEC.species_props("so4").molar_mass
                      / GAS_SPECIES["so2"].molar_mass)
        total_so4 = SO4_PRIMARY_FRACTION * _F_SO2 * so2_to_so4
        # Absent the subset, the WHOLE elevated_industrial flux keeps
        # fossil's Aitken+accumulation split -- no coarse mode at all.
        m_cs = tend.tracers.get(mass_name("so4", "cs"))
        self.assertTrue(m_cs is None or np.all(np.asarray(m_cs) == 0.0))
        n_ks = _column_integral(tend.tracers[number_name("ks")], rho, dz)
        np.testing.assert_allclose(
            n_ks, 0.5 * total_so4 * self._zm2n("ks", "so4", CMR_SK), rtol=1e-5)
        n_as = _column_integral(tend.tracers[number_name("as")], rho, dz)
        np.testing.assert_allclose(
            n_as, 0.5 * total_so4 * self._zm2n("as", "so4", CMR_SA), rtol=1e-5)

    def test_mam4_unaffected_by_m7_sector_policy_path(self):
        # The default (no spec override) path is bit-for-bit the one
        # exercised throughout the rest of this file.
        state, diagnostics, forcing = _setup(emis_surface_combustion_so2=_F_SO2)
        tend, _ = AnthropogenicEmissions()(state, diagnostics, forcing, None)
        rho, dz = diagnostics["air_density"], diagnostics["layer_thickness"]
        so4 = (_column_integral(tend.tracers[mass_name("so4", "ait")], rho, dz)
               + _column_integral(tend.tracers[mass_name("so4", "acc")], rho, dz))
        np.testing.assert_allclose(
            so4, SO4_PRIMARY_FRACTION * _F_SO2 * SO2_TO_SO4_MASS, rtol=1e-5)


class FactoryWiringTest(unittest.TestCase):
    def test_default_excludes_anthropogenic(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        names = [t.name for t in jam_aerosol_physics()]
        self.assertNotIn("jam_anthropogenic_emissions", names)

    def test_flag_includes_anthropogenic(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        names = [t.name for t in jam_aerosol_physics(anthropogenic=True)]
        self.assertIn("jam_anthropogenic_emissions", names)


if __name__ == "__main__":
    unittest.main()
