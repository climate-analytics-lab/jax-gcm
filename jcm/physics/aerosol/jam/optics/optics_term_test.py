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

    def test_negative_number_ringing_stays_bounded(self):
        """Negative modal number (spectral Gibbs ringing on the growing aerosol
        field) must not blow the extinction-weighted SSA/asymmetry out of
        range. Without flooring the number at 0 the band AOD goes ≤ 0, the SSA
        (= scat / AOD) and asymmetry diverge to ±huge, and RRTMGP's two-stream
        solver NaNs — this is the step-10 echam-jam blow-up. Drive some cells
        negative and assert every per-band optic stays finite and physical.
        """
        state, diagnostics, band, n_sw, n_lw = _setup()
        aer = diagnostics["_jam_state"]
        # Flip the sign of the modal number in a subset of cells (ringing).
        num = aer.number
        num = num.at[:, 0, :].set(-jnp.abs(num[:, 0, :]))
        num = num.at[:, :, 1].set(-1.0e7)
        diagnostics = {**diagnostics, "_jam_state": aer.copy(number=num)}
        # And the corresponding number tracers (used for the water volume).
        tracers = dict(state.tracers)
        for mode in MAM4_SPEC.modes:
            nm = number_name(mode.short)
            t = tracers[nm].at[0, :].set(-1.0e7)
            tracers[nm] = t.at[:, 1].set(-1.0e7)
        state = state.copy(tracers=tracers)

        term = self._term(band)
        _, diag = term(state, diagnostics, None, None)
        a = diag["aerosol"]
        for arr in (a.aod_sw_per_band, a.aod_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            self.assertTrue(bool(jnp.all(arr >= 0.0)))           # AOD floored
        for arr in (a.ssa_sw_per_band, a.ssa_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            self.assertTrue(bool(jnp.all((arr >= 0.0) & (arr <= 1.0 + 1e-5))))
        for arr in (a.asy_sw_per_band, a.asy_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            # Asymmetry is physically [-1, 1] (negative g = back-scattering).
            self.assertTrue(bool(jnp.all((arr >= -1.0 - 1e-5) & (arr <= 1.0 + 1e-5))))
        self.assertTrue(np.all(np.isfinite(np.asarray(diag["aerosol_optical_depth"]))))

    def test_empty_levels_carry_exactly_zero_tau(self):
        """POSITIVE-side ringing: tiny +ve number with a garbage wet radius
        must give EXACTLY zero tau where the mode has no mass. n*q_ext*
        pi*r^2 was finite at empty levels, and the 1/dp amplification at
        the 1 Pa model top produced 13,000 K/day of spurious SW
        absorption — +90 K in 6 h and a global NaN by day 10 of the
        first coupled JAM year. The mass gate (vol_tot > 1e-24 m3/kg)
        pins tau to zero there.
        """
        state, diagnostics, band, n_sw, n_lw = _setup()
        aer = diagnostics["_jam_state"]
        # Ringing-like state: tiny positive number, inflated wet radius,
        # zero species mass everywhere.
        diagnostics = {**diagnostics, "_jam_state": aer.copy(
            number=jnp.full_like(aer.number, 1.0e-6),
            r_wet=jnp.full_like(aer.r_wet, 5.0e-5),
        )}
        tracers = {k: (jnp.zeros_like(v) if k.startswith("m_") else v)
                   for k, v in state.tracers.items()}
        state = state.copy(tracers=tracers)

        term = self._term(band)
        _, diag = term(state, diagnostics, None, None)
        a = diag["aerosol"]
        self.assertEqual(float(jnp.max(jnp.abs(a.aod_sw_per_band))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(a.aod_lw_per_band))), 0.0)

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

    def test_column_aod_550_diagnostic(self):
        from jcm.physics.aerosol.aerosol_types import AerosolData
        from jcm.physics_interface import PhysicsState

        # Bands chosen so the 550 nm pick is unambiguous (500 nm is closest).
        nlev, ncols = 4, 3
        sw = (350.0, 500.0, 900.0)
        lw = (8000.0, 20000.0)
        band = RadiationBandConfig(lw_band_centers_nm=lw, sw_band_centers_nm=sw)
        term = self._term(band)
        self.assertEqual(term._cache.aod_band_idx, 1)   # 500 nm is closest to 550
        self.assertEqual(term._cache.aod_band_nm, 500.0)

        n_modes = MAM4_SPEC.n_modes()
        shape = (n_modes, nlev, ncols)
        aer = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6), r_wet=jnp.full(shape, 0.2e-6),
            rho=jnp.full(shape, 1800.0), kappa=jnp.full(shape, 0.5),
            mass=jnp.full(shape, 1e-9), number=jnp.full(shape, 1.0e8),
        )
        tracers = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1e-9)
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 285.0), tracers=tracers,
        )
        diagnostics = {
            "_jam_state": aer,
            "aerosol": AerosolData.zeros((ncols,), nlev, n_bnd_sw=3, n_bnd_lw=2),
            "air_density": jnp.full((nlev, ncols), 1.0),
            "layer_thickness": jnp.full((nlev, ncols), 500.0),
            "_band_config": band,
        }
        _, diag = term(state, diagnostics, None, None)

        aod = diag["aerosol_optical_depth"]
        # Column field, one value per column; finite and physical.
        self.assertEqual(aod.shape, (ncols,))
        self.assertTrue(np.all(np.isfinite(np.asarray(aod))))
        self.assertTrue(bool(jnp.all(aod > 0.0)))
        # It is exactly the column sum of the 550 nm-band per-layer AOD.
        per_layer_550 = diag["aerosol"].aod_sw_per_band[1]
        np.testing.assert_allclose(
            np.asarray(aod), np.asarray(jnp.sum(per_layer_550, axis=0)),
            rtol=1e-6,
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
