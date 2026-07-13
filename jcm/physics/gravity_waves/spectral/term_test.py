"""Tests for the FrontalGravityWaveDrag PhysicsTerm."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.gravity_waves.spectral.params import (
    FrontalGWParameters,
    SpectrumShape,
)
from jcm.physics.gravity_waves.spectral.term import FrontalGravityWaveDrag
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData
from jcm.utils import get_coords

KX = 20
IX, IL = 64, 32
DT = 1800.0


def _make_setup(params=None):
    coords = get_coords(np.linspace(0.0, 1.0, KX + 1), nodal_shape=(IX, IL))
    term = FrontalGravityWaveDrag(params)
    term.cache_coords(coords)

    sigma_full = 0.5 * (np.linspace(0, 1, KX + 1)[:-1]
                        + np.linspace(0, 1, KX + 1)[1:])
    # Plausible temperature/wind columns, uniform in the horizontal plus a
    # small deterministic perturbation so columns are not all identical.
    T = 210.0 + 80.0 * sigma_full[:, None, None] * np.ones((KX, IX, IL))
    u = 30.0 * np.exp(-((sigma_full[:, None, None] - 0.25) / 0.15) ** 2)
    u = u * np.ones((KX, IX, IL))
    u = u + 0.1 * np.sin(np.arange(IX))[None, :, None]
    v = 2.0 * np.ones((KX, IX, IL))
    state = PhysicsState.zeros(
        (KX, IX, IL),
        u_wind=jnp.asarray(u),
        v_wind=jnp.asarray(v),
        temperature=jnp.asarray(T),
        specific_humidity=jnp.full((KX, IX, IL), 1e-3),
        normalized_surface_pressure=jnp.ones((IX, IL)),
    )
    forcing = ForcingData.zeros((IX, IL))
    terrain = TerrainData.aquaplanet(coords)
    return term, state, forcing, terrain


class LevelSelectionTest(unittest.TestCase):
    def test_static_levels_from_reference_pressures(self):
        term, _, _, _ = _make_setup()
        # Uniform sigma, 20 levels: interfaces at multiples of 5000 Pa.
        # count(p < 50000) = 10 -> ksrc = 8 (midpoint ~42.5 kPa, source
        # interface 9 at 45 kPa, the deepest interface above 500 hPa).
        self.assertEqual(term._ksrc, 8)
        # count(p < 60000) = 12 -> kfront = 11 (upper interface 55 kPa).
        self.assertEqual(term._kfront, 11)
        # Newtonian cooling: interface profile, floored at 1e-6 s^-1.
        alpha = np.asarray(term._alpha.get_value())
        self.assertEqual(alpha.shape, (KX + 1,))
        self.assertGreaterEqual(alpha.min(), 1e-6 - 1e-12)


class InertWithoutProviderTest(unittest.TestCase):
    def test_exactly_zero_without_frontogenesis(self):
        term, state, forcing, terrain = _make_setup()
        tend, diag = term(state, {"_dt_seconds": DT}, forcing, terrain)
        # No "frontogenesis" diagnostic and default fallback 0.0: the term
        # must be inert — tendencies exactly zero, not merely small.
        self.assertEqual(float(jnp.abs(tend.u_wind).max()), 0.0)
        self.assertEqual(float(jnp.abs(tend.v_wind).max()), 0.0)
        self.assertEqual(float(jnp.abs(tend.temperature).max()), 0.0)
        self.assertEqual(float(jnp.abs(tend.specific_humidity).max()), 0.0)

    def test_fallback_trigger_can_force_launch(self):
        # Opt-in fallback above the threshold launches everywhere even
        # without a provider (a testing aid; disabled by default).
        params = FrontalGWParameters(fallback_frontogenesis=1.0e-14)
        term, state, forcing, terrain = _make_setup(params)
        tend, _ = term(state, {"_dt_seconds": DT}, forcing, terrain)
        self.assertGreater(float(jnp.abs(tend.u_wind).max()), 0.0)


class ActiveWithProviderTest(unittest.TestCase):
    def _diags(self, value):
        frontgf = jnp.full((KX, IX, IL), value)
        return {"_dt_seconds": DT, "frontogenesis": frontgf}

    def test_nonzero_finite_above_threshold(self):
        term, state, forcing, terrain = _make_setup()
        tend, diag = term(state, self._diags(1.0e-14), forcing, terrain)
        for field in (tend.u_wind, tend.v_wind, tend.temperature):
            self.assertTrue(bool(jnp.all(jnp.isfinite(field))))
        self.assertGreater(float(jnp.abs(tend.u_wind).max()), 0.0)
        self.assertGreater(float(jnp.abs(tend.temperature).max()), 0.0)
        # Moisture untouched.
        self.assertEqual(float(jnp.abs(tend.specific_humidity).max()), 0.0)
        # Tendency limiter (applies above the source; the momentum fixer
        # adds a uniform increment below it, so allow its small excess).
        ksrc = term._ksrc
        ubt = jnp.sqrt(tend.u_wind[: ksrc + 1] ** 2
                       + tend.v_wind[: ksrc + 1] ** 2)
        self.assertLessEqual(float(ubt.max()),
                             400.0 / 86400.0 * (1.0 + 1e-5))

    def test_zero_below_threshold(self):
        term, state, forcing, terrain = _make_setup()
        tend, _ = term(state, self._diags(1.0e-16), forcing, terrain)
        self.assertEqual(float(jnp.abs(tend.u_wind).max()), 0.0)
        self.assertEqual(float(jnp.abs(tend.temperature).max()), 0.0)

    def test_flat_spectrum_option(self):
        params = FrontalGWParameters(spectrum=SpectrumShape.FLAT)
        term, state, forcing, terrain = _make_setup(params)
        tend, _ = term(state, self._diags(1.0e-14), forcing, terrain)
        self.assertTrue(bool(jnp.all(jnp.isfinite(tend.u_wind))))
        self.assertGreater(float(jnp.abs(tend.u_wind).max()), 0.0)

    def test_gradient_through_params(self):
        term, state, forcing, terrain = _make_setup()
        diags = self._diags(1.0e-14)

        def loss(taubgnd):
            p = FrontalGWParameters(taubgnd=taubgnd)
            t = FrontalGravityWaveDrag(p)
            t._a_half = term._a_half
            t._b_half = term._b_half
            t._ksrc = term._ksrc
            t._kfront = term._kfront
            t._alpha = term._alpha
            tend, _ = t(state, diags, forcing, terrain)
            return jnp.sum(tend.u_wind**2) * 1e12

        g = jax.grad(loss)(jnp.asarray(1.25e-3))
        self.assertTrue(bool(jnp.isfinite(g)))

    def test_heating_consistent_with_dissipation(self):
        # Column-integrated dycore-visible energy change should be ~0 with
        # the energy fixer on (KE loss is returned as heat; the fixer
        # removes the residual). Check the column integral of
        # cpd*dT/dt + u*du/dt + v*dv/dt is small relative to its parts.
        term, state, forcing, terrain = _make_setup()
        tend, _ = term(state, self._diags(1.0e-14), forcing, terrain)
        ps = np.asarray(state.normalized_surface_pressure) * c.p0
        p_half = np.linspace(0, 1, KX + 1)[:, None, None] * ps[None]
        dp = np.diff(p_half, axis=0)
        dsdt = c.cpd * np.asarray(tend.temperature)
        u = np.asarray(state.u_wind)
        v = np.asarray(state.v_wind)
        du = np.asarray(tend.u_wind)
        dv = np.asarray(tend.v_wind)
        de = np.sum(dp / c.grav * (dsdt + du * (u + 0.5 * DT * du)
                                   + dv * (v + 0.5 * DT * dv)), axis=0)
        scale = np.sum(np.abs(dp / c.grav * dsdt), axis=0).max()
        self.assertLess(np.abs(de).max(), 1e-3 * max(scale, 1e-30) + 1e-8)


class Ne30HeatingRegressionTest(unittest.TestCase):
    """Real winter-jet columns that blew the ne30 v5 run up (123 K/day).

    Fixture ``jcm/data/test/gw_frontal_repro_cols.npz`` holds the eight
    worst heating columns (ECHAM L47 hybrid grid, forced launch) from the
    run's last clean state. On that grid the lid layer is ~2 Pa thick and
    ``rho -> 0`` makes every wave saturate there; waves on both sides of
    ``ubm`` cancel in the net, so CAM's net-only tndmax limiter left the
    frictional heating unbounded (dttke up to 123 K/day) while |du/dt|
    sat exactly at tndmax. The production default
    ``limit_tendency_sum=True`` (solver deviation 6) must bound the
    heating; ``False`` must reproduce the blowup (guarding the exact-CAM
    path's documented behaviour).
    """

    @classmethod
    def setUpClass(cls):
        from importlib import resources
        from types import SimpleNamespace

        from jcm.physics.echam.echam_levels import get_echam_levels

        path = resources.files("jcm.data.test") / "gw_frontal_repro_cols.npz"
        with resources.as_file(path) as f:
            d = np.load(f)
            cls.u, cls.v = d["u"], d["v"]
            cls.T, cls.q, cls.ps = d["T"], d["q"], d["ps"]
        cls.coords = SimpleNamespace(vertical=get_echam_levels(47))

    def _run(self, params=None):
        term = FrontalGravityWaveDrag(params)
        term.cache_coords(self.coords)
        ncols = self.ps.shape[0]
        state = PhysicsState.zeros(
            self.T.shape,
            u_wind=jnp.asarray(self.u), v_wind=jnp.asarray(self.v),
            temperature=jnp.asarray(self.T),
            specific_humidity=jnp.asarray(self.q),
            normalized_surface_pressure=jnp.asarray(self.ps) / c.p0,
        )
        diags = {"_dt_seconds": 900.0,
                 "frontogenesis": jnp.full(self.T.shape, 1.0e-12)}
        return term(state, diags, ForcingData.zeros((ncols,)), None)

    def test_heating_bounded_with_default_params(self):
        tend, _ = self._run(FrontalGWParameters())
        dT_day = np.abs(np.asarray(tend.temperature)) * 86400.0
        du_day = np.hypot(np.asarray(tend.u_wind),
                          np.asarray(tend.v_wind)) * 86400.0
        self.assertTrue(np.all(np.isfinite(dT_day)))
        self.assertTrue(np.all(np.isfinite(du_day)))
        # tndmax-consistent heating bound (was 123 K/day before the fix).
        self.assertLess(dT_day.max(), 30.0)
        # Momentum limiter untouched (fixer adds a tiny uniform increment
        # below the source, hence the small headroom).
        self.assertLessEqual(du_day.max(), 400.0 * (1.0 + 1e-3))

    def test_exact_cam_limiter_reproduces_blowup(self):
        # The exact-CAM path (limit_tendency_sum=False) is kept available
        # and must still show the unbounded-heating behaviour this
        # fixture was built from — if this starts passing the 30 K/day
        # bound, the fixture no longer exercises the cancellation regime.
        tend, _ = self._run(FrontalGWParameters(limit_tendency_sum=False))
        dT_day = np.abs(np.asarray(tend.temperature)) * 86400.0
        self.assertGreater(dT_day.max(), 80.0)

    def test_publishes_tendency_diagnostics(self):
        tend, diag = self._run(FrontalGWParameters())
        for key in ("gw_frontal_dudt", "gw_frontal_dvdt", "gw_frontal_dtdt"):
            self.assertIn(key, diag)
        np.testing.assert_array_equal(np.asarray(diag["gw_frontal_dudt"]),
                                      np.asarray(tend.u_wind))
        np.testing.assert_array_equal(np.asarray(diag["gw_frontal_dvdt"]),
                                      np.asarray(tend.v_wind))
        np.testing.assert_array_equal(np.asarray(diag["gw_frontal_dtdt"]),
                                      np.asarray(tend.temperature))

    def test_gradient_finite_on_real_columns(self):
        term = FrontalGravityWaveDrag()
        term.cache_coords(self.coords)
        state = PhysicsState.zeros(
            self.T.shape,
            u_wind=jnp.asarray(self.u), v_wind=jnp.asarray(self.v),
            temperature=jnp.asarray(self.T),
            specific_humidity=jnp.asarray(self.q),
            normalized_surface_pressure=jnp.asarray(self.ps) / c.p0,
        )
        diags = {"_dt_seconds": 900.0,
                 "frontogenesis": jnp.full(self.T.shape, 1.0e-12)}
        forcing = ForcingData.zeros((self.ps.shape[0],))

        def loss(taubgnd):
            t = FrontalGravityWaveDrag(FrontalGWParameters(taubgnd=taubgnd))
            t._a_half = term._a_half
            t._b_half = term._b_half
            t._ksrc = term._ksrc
            t._kfront = term._kfront
            t._alpha = term._alpha
            tend, _ = t(state, diags, forcing, None)
            return (jnp.sum(tend.temperature**2)
                    + jnp.sum(tend.u_wind**2)) * 1e8

        g = jax.grad(loss)(jnp.asarray(1.25e-3))
        self.assertTrue(bool(jnp.isfinite(g)))


if __name__ == "__main__":
    unittest.main()
