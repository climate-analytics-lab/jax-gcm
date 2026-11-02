"""Tests for the volcanic SO₂ emission term (#1190)."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate

from jcm.physics.aerosol.jam.emissions.distributors import particle_mean_mass
from jcm.physics.aerosol.jam.emissions.ham_sectors import (
    cmr_to_emission_diameter)
from jcm.physics.aerosol.jam.emissions.sectors import SO4_PRIMARY_FRACTION
from jcm.physics.aerosol.jam.emissions.volcanic import (
    HAM_VOLCANIC_CLASS,
    VolcanicEmissions,
    VolcanicParameters,
    gaussian_layer_fractions,
    layer_overlap_fractions,
)
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name
from jcm.physics_interface import PhysicsState
from jcm.testing import check_gradients

# Five layers, top first, over two columns: the second stands on a 1.5 km hill.
_Z_HALF = jnp.asarray([[12000.0, 12000.0], [8000.0, 8000.0], [4000.0, 4000.0],
                       [2000.0, 2500.0], [800.0, 1800.0], [0.0, 1500.0]])
_NLEV, _NCOLS = 5, 2
_RHO = jnp.asarray([0.3, 0.6, 0.8, 1.0, 1.2])[:, None] * jnp.ones((1, _NCOLS))


def _diagnostics():
    return {
        "air_density": _RHO,
        "layer_thickness": _Z_HALF[:-1] - _Z_HALF[1:],
        "height_half": _Z_HALF,
    }


def _state():
    return PhysicsState.zeros((_NLEV, _NCOLS)).copy(
        temperature=jnp.full((_NLEV, _NCOLS), 260.0))


def _volcanic(*, degassing_flux=((1.0e-9, 3.0e-9),), eruptions=()):
    """One degassing source in column 1 on two layers (0-1 km, 1-3 km).

    ``eruptions`` is a sequence of ``(cell, flux, height, depth, year, day)``.
    """
    e = np.asarray(eruptions, dtype=float).reshape(-1, 6)
    return {
        "degassing_cell": jnp.asarray([1], dtype=jnp.int32),
        "degassing_flux": jnp.asarray(degassing_flux),
        "degassing_altitude_interface": jnp.asarray([0.0, 1000.0, 3000.0]),
        "eruption_cell": jnp.asarray(e[:, 0], dtype=jnp.int32),
        "eruption_flux": jnp.asarray(e[:, 1]),
        "eruption_height": jnp.asarray(e[:, 2]),
        "eruption_depth": jnp.asarray(e[:, 3]),
        "eruption_year": jnp.asarray(e[:, 4]),
        "eruption_day": jnp.asarray(e[:, 5]),
    }


def _forcing(volcanic, *, year=2010.0, day=100.0):
    solar = types.SimpleNamespace(calendar_year=jnp.asarray(year),
                                  day_of_year=jnp.asarray(day))
    return types.SimpleNamespace(volcanic_emissions=volcanic, solar=solar)


def _column(tend):
    """Σ ρΔz·tendency per column: the emitted flux it carries [/m²/s]."""
    d = _diagnostics()
    return np.asarray(jnp.sum(tend * d["air_density"] * d["layer_thickness"], axis=0))


class LayerFractionTest(unittest.TestCase):
    def test_overlap_shares_sum_to_one_per_source_layer(self):
        source = jnp.asarray([-500.0, 1000.0, 1600.0, 9000.0, 20000.0])
        share = layer_overlap_fractions(_Z_HALF, source)
        np.testing.assert_allclose(np.asarray(share.sum(axis=0)), 1.0, rtol=1e-6)

    def test_overlap_puts_below_surface_and_above_lid_in_end_layers(self):
        share = np.asarray(layer_overlap_fractions(
            _Z_HALF, jnp.asarray([0.0, 1000.0, 13000.0, 14000.0])))
        # Column 1's surface is at 1.5 km: the 0-1 km layer is all underground.
        np.testing.assert_allclose(share[:, 0, 1], [0, 0, 0, 0, 1], atol=1e-7)
        # 13-14 km is above both lids (12 km).
        np.testing.assert_allclose(share[:, 2, :], [[1, 1]] + [[0, 0]] * 4, atol=1e-7)
        # 1-13 km in column 0: none in layer 4 (0-0.8 km), 1 km in 3, 2 in 2,
        # 4 in 1, 4 in 0 plus the 1 km above the lid.
        np.testing.assert_allclose(share[:, 1, 0],
                                   np.asarray([5.0, 4.0, 2.0, 1.0, 0.0]) / 12.0,
                                   rtol=1e-6, atol=1e-7)

    def test_gaussian_fractions_are_the_exact_layer_integrals(self):
        height, depth = 3000.0, 700.0
        frac = np.asarray(gaussian_layer_fractions(
            _Z_HALF, jnp.full((_NCOLS,), height), jnp.full((_NCOLS,), depth)))
        np.testing.assert_allclose(frac.sum(axis=0), 1.0, rtol=1e-6)
        norm = np.sqrt(np.pi) * depth
        z = np.asarray(_Z_HALF[:, 0])
        expect = [integrate.quad(lambda x: np.exp(-((x - height) / depth) ** 2),
                                 z[k + 1], z[k])[0] / norm for k in range(_NLEV)]
        expect[0] += integrate.quad(lambda x: np.exp(-((x - height) / depth) ** 2),
                                    z[0], np.inf)[0] / norm
        expect[-1] += integrate.quad(lambda x: np.exp(-((x - height) / depth) ** 2),
                                     -np.inf, z[-1])[0] / norm
        np.testing.assert_allclose(frac[:, 0], expect, rtol=1e-4, atol=1e-7)

    def test_thin_plume_stays_in_its_layer(self):
        frac = np.asarray(gaussian_layer_fractions(
            _Z_HALF, jnp.full((_NCOLS,), 6000.0), jnp.full((_NCOLS,), 50.0)))
        np.testing.assert_allclose(frac[1], 1.0, rtol=1e-6)


class VolcanicEmissionsTest(unittest.TestCase):
    def test_inert_without_volcanic_forcing(self):
        tend, _ = VolcanicEmissions()(_state(), _diagnostics(),
                                      _forcing(None), None)
        self.assertEqual(tend.tracers, {})

    def test_degassing_column_sulfur_and_primary_split(self):
        tend, diag = VolcanicEmissions()(_state(), _diagnostics(),
                                         _forcing(_volcanic()), None)
        total = 4.0e-9   # kg SO2 m-2 s-1 in column 1
        frac = SO4_PRIMARY_FRACTION
        gas = _column(tend.tracers[gas_name("so2")])
        np.testing.assert_allclose(gas, [0.0, (1.0 - frac) * total], rtol=1e-6)
        so4_factor = (MAM4_SPEC.species_props("so4").molar_mass
                      / GAS_SPECIES["so2"].molar_mass)
        so4 = {m: _column(tend.tracers[mass_name("so4", m)]) for m in ("ait", "acc")}
        for m in ("ait", "acc"):   # MAM4's own primary split, 50/50
            np.testing.assert_allclose(so4[m], [0.0, 0.5 * frac * total * so4_factor],
                                       rtol=1e-6)
            mode = MAM4_SPEC.mode(m)
            density = MAM4_SPEC.species_props("so4").density
            np.testing.assert_allclose(
                _column(tend.tracers[number_name(m)]),
                so4[m] * mode.number_factor / density, rtol=1e-6)
        # Column 1 stands at 1.5 km: the 0-1 km layer and the lowest quarter
        # of the 1-3 km one are underground and go to the lowest layer
        # (1.5-1.8 km), which also takes 0.15 of 1-3 km; 0.35 lands in 1.8-2.5
        # km and 0.25 in 2.5-4 km.
        g = np.asarray(tend.tracers[gas_name("so2")])[:, 1] * np.asarray(
            _RHO[:, 1] * (_Z_HALF[:-1, 1] - _Z_HALF[1:, 1]))
        np.testing.assert_allclose(
            g / ((1.0 - frac) * total),
            np.asarray([0, 0, 3e-9 * 0.25, 3e-9 * 0.35, 1e-9 + 3e-9 * 0.40]) / total,
            rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.asarray(diag["emi_so2"]), gas, rtol=1e-6)

    def test_m7_takes_ham_default_class_sizes(self):
        tend, _ = VolcanicEmissions(spec=M7_SPEC)(
            _state(), _diagnostics(), _forcing(_volcanic()), None)
        so4_factor = (M7_SPEC.species_props("so4").molar_mass
                      / GAS_SPECIES["so2"].molar_mass)
        density = M7_SPEC.species_props("so4").density
        targets = M7_SPEC.sector_emission.targets[HAM_VOLCANIC_CLASS]["so4"]
        self.assertEqual({t.mode for t in targets}, {"ks", "as"})
        for t in targets:
            mode = M7_SPEC.mode(t.mode)
            mass = _column(tend.tracers[mass_name("so4", mode.short)])
            np.testing.assert_allclose(
                mass[1], t.mass_fraction * SO4_PRIMARY_FRACTION * 4.0e-9 * so4_factor,
                rtol=1e-6)
            m_p = particle_mean_mass(mode, density, emission_diameter=(
                cmr_to_emission_diameter(t.cmr_m, mode.geom_std_dev)))
            np.testing.assert_allclose(
                _column(tend.tracers[number_name(mode.short)]), mass / m_p, rtol=1e-6)

    def test_eruption_fires_only_on_its_day_and_year(self):
        eruptions = [(0, 2.0e-8, 9000.0, 900.0, 2010.0, 100.0),
                     (0, 5.0e-8, 9000.0, 900.0, 2011.0, 100.0),
                     (1, 7.0e-8, 6000.0, 600.0, -1.0, 101.0)]
        volc = _volcanic(degassing_flux=((0.0, 0.0),), eruptions=eruptions)
        cases = {(2010.0, 100.0): [2.0e-8, 0.0], (2011.0, 100.0): [5.0e-8, 0.0],
                 (2012.0, 101.0): [0.0, 7.0e-8], (2010.0, 99.0): [0.0, 0.0]}
        for (year, day), expect in cases.items():
            tend, _ = VolcanicEmissions()(_state(), _diagnostics(),
                                          _forcing(volc, year=year, day=day), None)
            np.testing.assert_allclose(_column(tend.tracers[gas_name("so2")]),
                                       expect, rtol=1e-6, atol=1e-20)
            # Eruptions emit no primary sulfate.
            np.testing.assert_allclose(
                _column(tend.tracers[mass_name("so4", "acc")]), 0.0, atol=1e-25)

    def test_eruption_profile_peaks_at_injection_height(self):
        volc = _volcanic(degassing_flux=((0.0, 0.0),),
                         eruptions=[(0, 2.0e-8, 6000.0, 600.0, 2010.0, 100.0)])
        tend, _ = VolcanicEmissions()(_state(), _diagnostics(), _forcing(volc), None)
        per_layer = np.asarray(tend.tracers[gas_name("so2")][:, 0]
                               * _RHO[:, 0] * (_Z_HALF[:-1, 0] - _Z_HALF[1:, 0]))
        self.assertEqual(int(np.argmax(per_layer)), 1)   # the 4-8 km layer

    def test_follows_the_state_dtype(self):
        state = PhysicsState.zeros((_NLEV, _NCOLS)).copy(
            temperature=jnp.full((_NLEV, _NCOLS), 260.0, dtype=jnp.float32))
        tend, _ = VolcanicEmissions()(state, _diagnostics(), _forcing(_volcanic()), None)
        for v in tend.tracers.values():
            self.assertEqual(v.dtype, jnp.float32)

    def test_whole_grid_layout_matches_columns(self):
        grid = lambda x: x.reshape(x.shape[0], 1, _NCOLS)  # noqa: E731
        d = {k: grid(v) for k, v in _diagnostics().items()}
        state = PhysicsState.zeros((_NLEV, 1, _NCOLS)).copy(
            temperature=jnp.full((_NLEV, 1, _NCOLS), 260.0))
        on_grid, _ = VolcanicEmissions()(state, d, _forcing(_volcanic()), None)
        on_cols, _ = VolcanicEmissions()(_state(), _diagnostics(),
                                         _forcing(_volcanic()), None)
        for k, v in on_cols.tracers.items():
            np.testing.assert_allclose(np.asarray(on_grid.tracers[k][:, 0]),
                                       np.asarray(v), rtol=1e-7)

    def test_gradients_of_parameters_and_injection_height(self):
        volc = _volcanic(eruptions=[(0, 2.0e-8, 5000.0, 700.0, 2010.0, 100.0)])

        def emitted(degassing_scale, eruption_scale, frac, height):
            p = VolcanicParameters(degassing_scale=degassing_scale,
                                   eruption_scale=eruption_scale,
                                   so4_primary_fraction=frac)
            v = {**volc, "eruption_height": height}
            tend, _ = VolcanicEmissions(params=p)(
                _state(), _diagnostics(), _forcing(v), None)
            return {k: x * 1.0e9 for k, x in tend.tracers.items()}

        check_gradients(emitted, (jnp.asarray(1.0), jnp.asarray(1.0),
                                  jnp.asarray(SO4_PRIMARY_FRACTION),
                                  jnp.asarray([5000.0])), rtol=1e-3)
        g = jax.grad(lambda h: jnp.sum(
            emitted(1.0, 1.0, SO4_PRIMARY_FRACTION, h)[gas_name("so2")][1]))(
                jnp.asarray([5000.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(g))) and float(g[0]) != 0.0)


class VolcanicCompositionTest(unittest.TestCase):
    def test_flag_composes_the_term(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        self.assertNotIn("jam_volcanic_emissions",
                         [t.name for t in jam_aerosol_physics()])
        self.assertIn("jam_volcanic_emissions",
                      [t.name for t in jam_aerosol_physics(volcanic=True)])

    def test_params_without_the_flag_are_refused(self):
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, r"\['volcanic_params'\] would be ignored"):
            echam_physics(checkpoint_terms=False, aerosol_module="jam",
                          cloud_scheme="2m", jam_microphysics="placeholder",
                          volcanic_params={"eruption_scale": 0.5})

    def test_composed_term_warns_without_its_forcing(self):
        with self.assertLogs("jcm.physics.aerosol.jam.emissions.volcanic",
                             level="WARNING") as logs:
            VolcanicEmissions().validate_forcing(types.SimpleNamespace(
                volcanic_emissions=None))
        self.assertIn("no volcanic SO2", logs.output[0])


if __name__ == "__main__":
    unittest.main()
