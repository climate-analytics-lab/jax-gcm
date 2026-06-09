"""Tests for refractive indices and the JamOpticsTerm."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.optics.refractive_index import refractive_index_at
from jcm.physics.radiation.band_config import RadiationBandConfig


class RefractiveIndexTest(unittest.TestCase):
    def test_black_carbon_strongly_absorbing(self):
        _, k = refractive_index_at("bc", jnp.array([550.0]))
        self.assertGreater(float(k[0]), 0.5)

    def test_sulfate_transparent_in_sw_absorbing_in_lw(self):
        _, k_sw = refractive_index_at("so4", jnp.array([550.0]))
        _, k_lw = refractive_index_at("so4", jnp.array([10000.0]))
        self.assertLess(float(k_sw[0]), 1e-6)
        self.assertGreater(float(k_lw[0]), 1e-2)

    def test_interp_shape(self):
        n, k = refractive_index_at("du", jnp.array([300.0, 550.0, 10000.0]))
        self.assertEqual(n.shape, (3,))
        self.assertTrue(np.all(np.isfinite(np.asarray(n))))


def _setup(nlev=4, ncols=3, n_sw=3, n_lw=2):
    from jcm.physics.aerosol.aerosol_types import AerosolData
    from jcm.physics_interface import PhysicsState

    n_modes = MAM4_SPEC.n_modes()
    shape = (n_modes, nlev, ncols)
    aer = JamAerosolState(
        r_dry=jnp.full(shape, 0.1e-6),
        r_wet=jnp.full(shape, 0.2e-6),
        rho=jnp.full(shape, 1800.0),
        kappa=jnp.full(shape, 0.5),
        mass=jnp.full(shape, 1e-9),
        number=jnp.full(shape, 1.0e8),
    )
    tracers = {}
    for mode in MAM4_SPEC.modes:
        tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
        for sp in mode.species:
            tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1e-9)
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 285.0), tracers=tracers,
    )
    sw = tuple(float(x) for x in np.linspace(400.0, 1000.0, n_sw))
    lw = tuple(float(x) for x in np.linspace(8000.0, 20000.0, n_lw))
    band = RadiationBandConfig(lw_band_centers_nm=lw, sw_band_centers_nm=sw)
    aerosol = AerosolData.zeros((ncols,), nlev, n_bnd_sw=n_sw, n_bnd_lw=n_lw)
    diagnostics = {
        "_jam_state": aer,
        "aerosol": aerosol,
        "air_density": jnp.full((nlev, ncols), 1.0),
        "layer_thickness": jnp.full((nlev, ncols), 500.0),
        "_band_config": band,
    }
    return state, diagnostics, band, n_sw, n_lw


class JamOpticsTermTest(unittest.TestCase):
    def _term(self, band):
        term = JamOpticsTerm()
        term.cache_band_config(band)
        return term

    def test_writes_finite_bounded_optics(self):
        state, diagnostics, band, n_sw, n_lw = _setup()
        term = self._term(band)
        _, diag = term(state, diagnostics, None, None)
        a = diag["aerosol"]
        self.assertEqual(a.aod_sw_per_band.shape[0], n_sw)
        self.assertEqual(a.aod_lw_per_band.shape[0], n_lw)
        for arr in (a.aod_sw_per_band, a.aod_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            self.assertTrue(bool(jnp.all(arr >= 0.0)))
        for arr in (a.ssa_sw_per_band, a.ssa_lw_per_band):
            self.assertTrue(bool(jnp.all((arr >= 0.0) & (arr <= 1.0 + 1e-5))))
        for arr in (a.asy_sw_per_band, a.asy_lw_per_band):
            self.assertTrue(bool(jnp.all((arr >= -1.0 - 1e-5) & (arr <= 1.0 + 1e-5))))

    def test_more_aerosol_more_aod(self):
        state, diagnostics, band, *_ = _setup()
        term = self._term(band)
        _, d1 = term(state, diagnostics, None, None)
        # double the mass tracers
        state2 = state.copy(tracers={k: 2.0 * v for k, v in state.tracers.items()})
        _, d2 = term(state2, diagnostics, None, None)
        self.assertGreater(
            float(jnp.sum(d2["aerosol"].aod_sw_per_band)),
            float(jnp.sum(d1["aerosol"].aod_sw_per_band)),
        )

    def test_grad_through_mass(self):
        state, diagnostics, band, *_ = _setup()
        term = self._term(band)
        key = mass_name("bc", "acc")

        def loss(scale):
            tr = {k: (v * scale if k == key else v) for k, v in state.tracers.items()}
            s = state.copy(tracers=tr)
            _, d = term(s, diagnostics, None, None)
            return jnp.sum(d["aerosol"].aod_sw_per_band)

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))


if __name__ == "__main__":
    unittest.main()
