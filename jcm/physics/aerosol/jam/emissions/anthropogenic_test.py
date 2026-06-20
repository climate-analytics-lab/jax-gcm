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
