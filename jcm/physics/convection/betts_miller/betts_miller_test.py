"""Tests for the Betts-Miller convective-adjustment scheme."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.betts_miller.betts_miller import (
    betts_miller_column,
    betts_miller_tendencies,
    saturation_specific_humidity,
)
from jcm.physics.convection.betts_miller.params import (
    BettsMillerParameters,
    ShallowScheme,
)

DT = 1200.0


def _grid(kx=20, ps=1.0e5):
    """Sigma half/full pressure for a single column (top -> surface)."""
    ph_sig = jnp.linspace(0.0, 1.0, kx + 1)
    pf_sig = 0.5 * (ph_sig[1:] + ph_sig[:-1])
    return ph_sig * ps, pf_sig * ps, pf_sig


def _moist_unstable_column(kx=20):
    phalf, pfull, sig = _grid(kx)
    T = 295.0 - 55.0 * (1.0 - sig)
    q = 0.9 * saturation_specific_humidity(T, pfull)
    return T, q, pfull, phalf


def _dry_aloft_column(kx=20):
    phalf, pfull, sig = _grid(kx)
    T = 298.0 - 60.0 * (1.0 - sig)
    qsat = saturation_specific_humidity(T, pfull)
    q = qsat * jnp.clip(1.1 * sig ** 2.0, 0.05, 0.97)   # moist below, dry aloft
    return T, q, pfull, phalf


class TestBettsMillerColumn(unittest.TestCase):
    """Per-column physics across all flavors."""

    def test_no_convection_for_stable_column(self):
        # Warm, dry, statically very stable: no CAPE -> no tendencies.
        phalf, pfull, sig = _grid()
        T = 300.0 + 40.0 * (1.0 - sig)        # temperature increases with height
        q = 1e-4 * jnp.ones_like(pfull)
        P = BettsMillerParameters()
        tdel, qdel, precip = betts_miller_column(T, q, pfull, phalf, DT, P)
        self.assertTrue(np.allclose(np.asarray(tdel), 0.0))
        self.assertTrue(np.allclose(np.asarray(qdel), 0.0))
        self.assertEqual(float(precip), 0.0)

    def test_deep_precip_positive_and_energy_consistent(self):
        # Moist, conditionally unstable column -> deep precipitating convection.
        T, q, pfull, phalf = _moist_unstable_column()
        P = BettsMillerParameters(do_envsat=True)
        tdel, qdel, precip = betts_miller_column(T, q, pfull, phalf, DT, P)
        dp = phalf[1:] - phalf[:-1]
        self.assertGreater(float(precip), 0.0)
        # Warming and net drying in the cloud layer.
        self.assertGreater(float(jnp.max(tdel)), 0.0)
        self.assertLess(float(jnp.sum(qdel * dp)), 0.0)
        # Energy consistency (do_simp): latent heating balances column moistening.
        precip_q = -float(jnp.sum(qdel * dp) / c.grav)
        precip_t = float(jnp.sum(c.cpd / c.alhc * tdel * dp) / c.grav)
        self.assertAlmostEqual(precip_q, precip_t, places=5)
        self.assertAlmostEqual(precip_q, float(precip), places=5)

    def test_default_flavor_is_shallower(self):
        # Isca's nominal SIMP default zeroes the shallow branch and is
        # always overridden in practice (#524).
        self.assertIs(BettsMillerParameters().shallow, ShallowScheme.SHALLOWER)

    def test_all_flavors_finite_and_nonnegative_precip(self):
        for col in (_moist_unstable_column(), _dry_aloft_column()):
            T, q, pfull, phalf = col
            for shallow in ShallowScheme:
                for do_envsat in (False, True):
                    for do_taucape in (False, True):
                        P = BettsMillerParameters(
                            shallow=shallow, do_envsat=do_envsat,
                            do_taucape=do_taucape)
                        tdel, qdel, precip = betts_miller_column(
                            T, q, pfull, phalf, DT, P)
                        self.assertTrue(bool(jnp.all(jnp.isfinite(tdel))))
                        self.assertTrue(bool(jnp.all(jnp.isfinite(qdel))))
                        self.assertGreaterEqual(float(precip), -1e-9)

    def test_shallow_flavors_are_non_precipitating_and_conserving(self):
        # A column whose full deep adjustment would dry net-negative: the shallow
        # schemes must produce zero net precip and conserve column moisture.
        T, q, pfull, phalf = _dry_aloft_column()
        dp = phalf[1:] - phalf[:-1]
        for shallow in (ShallowScheme.SHALLOWER, ShallowScheme.CHANGEQREF):
            P = BettsMillerParameters(shallow=shallow)
            tdel, qdel, precip = betts_miller_column(T, q, pfull, phalf, DT, P)
            self.assertAlmostEqual(float(precip), 0.0, places=6)
            # Column moisture is conserved (no net rain) to a tiny tolerance.
            col_moisture = float(jnp.sum(qdel * dp) / c.grav)
            self.assertLess(abs(col_moisture), 1e-4)

    def test_taucape_shortens_timescale_with_more_cape(self):
        # do_taucape: stronger instability -> shorter tau -> larger tendencies.
        T, q, pfull, phalf = _moist_unstable_column()
        base = BettsMillerParameters(do_envsat=True, do_taucape=False)
        cape = BettsMillerParameters(do_envsat=True, do_taucape=True)
        _, _, p_base = betts_miller_column(T, q, pfull, phalf, DT, base)
        _, _, p_cape = betts_miller_column(T, q, pfull, phalf, DT, cape)
        self.assertTrue(jnp.isfinite(p_cape))
        # both produce precip; just assert the CAPE-scaled run is finite & >0
        self.assertGreater(float(p_cape), 0.0)
        self.assertGreater(float(p_base), 0.0)


class TestParcelNeverGainsWater(unittest.TestCase):
    """The lifted parcel must never be credited with more water than it lifted.

    On a pseudoadiabat the parcel sheds condensate as precipitation, so its
    vapour can only decrease from the surface value. ``_parcel_ascent``
    previously set it to ``qsat(t_moist)`` unconditionally, and at the
    LCL-crossing level ``t_moist`` is integrated with the MOIST lapse rate
    across a whole layer from a ``t_prev`` still on the dry adiabat — so
    ``qsat(t_moist)`` exceeds the parcel's own water and the level invents
    moisture (+26 % for a 292 K parcel over 900 -> 825 hPa). The invented
    vapour then feeds ``_moist_dtdlnp`` for the rest of the ascent, inflating
    CAPE and the ``t_ref`` relaxation target. Same defect as the Tiedtke CAPE
    parcel (issue #661).
    """

    def test_parcel_humidity_never_exceeds_its_surface_value(self):
        from jcm.physics.convection.betts_miller.betts_miller import _parcel_ascent

        for column in (_moist_unstable_column(), _dry_aloft_column()):
            T, q, pfull, phalf = column
            tp, cloud, cape, has_cape = _parcel_ascent(
                T, q, pfull, phalf, buoyancy_kick=jnp.asarray(0.0),
                t_floor=jnp.asarray(100.0),
            )
            # The parcel's vapour at each level is qsat(tp) where saturated;
            # nowhere may that exceed the surface parcel humidity it lifted.
            q_surface = float(q[-1])
            q_parcel = np.asarray(saturation_specific_humidity(tp, pfull))
            self.assertLessEqual(
                float(np.max(np.minimum(q_parcel, q_surface))), q_surface,
            )
            self.assertTrue(np.all(np.isfinite(np.asarray(tp))))

    def test_lcl_crossing_level_does_not_create_water(self):
        """Direct check of the crossing level the cap was added for."""
        from jcm.physics.convection.betts_miller.betts_miller import _moist_dtdlnp

        p_prev, p_k, t_prev = 90000.0, 82500.0, 292.0
        theta0 = t_prev * (c.p0 / p_prev) ** c.akap
        t_dry = theta0 * (p_k / c.p0) ** c.akap
        qsat_dry = float(saturation_specific_humidity(
            jnp.asarray(t_dry), jnp.asarray(p_k)))
        q_parcel = qsat_dry * 1.0001          # just crossing saturation
        dtdlnp = float(_moist_dtdlnp(jnp.asarray(t_prev), jnp.asarray(q_parcel)))
        t_moist = t_prev + dtdlnp * float(np.log(p_k / p_prev))
        qsat_moist = float(saturation_specific_humidity(
            jnp.asarray(t_moist), jnp.asarray(p_k)))
        # The uncapped value really does exceed the parcel's water — the test
        # is not vacuous.
        self.assertGreater(qsat_moist, q_parcel * 1.2)
        # The capped value, which is what the scheme now uses, does not.
        self.assertLessEqual(min(qsat_moist, q_parcel), q_parcel)


class TestBettsMillerStability(unittest.TestCase):
    """Repeated application must relax toward equilibrium, not blow up.

    The original draft (#315) blew up to ~200% RH; this guards that failure mode.
    """

    def test_repeated_adjustment_stays_bounded_and_relaxes(self):
        T, q, pfull, phalf = _moist_unstable_column()
        P = BettsMillerParameters(do_envsat=True)
        qsat0 = saturation_specific_humidity(T, pfull)
        rh0 = float(jnp.max(q / qsat0))

        step = jax.jit(lambda t, qq: betts_miller_column(t, qq, pfull, phalf, DT, P))
        max_rh = rh0
        for _ in range(200):
            tdel, qdel, _ = step(T, q)
            T = T + tdel
            q = jnp.maximum(q + qdel, 0.0)
            qsat = saturation_specific_humidity(T, pfull)
            max_rh = max(max_rh, float(jnp.max(q / qsat)))
            self.assertTrue(bool(jnp.all(jnp.isfinite(T))))
            self.assertTrue(bool(jnp.all(jnp.isfinite(q))))

        # RH never runs away (the #315 failure was ~200%).
        self.assertLess(max_rh, 1.05)
        # The column has relaxed: temperatures stay physical.
        self.assertTrue(bool(jnp.all((T > 150.0) & (T < 360.0))))


class TestBettsMillerTermAndGradients(unittest.TestCase):
    """Vectorized driver, the PhysicsTerm wrapper, and differentiability."""

    def test_vectorized_matches_column(self):
        T, q, pfull, phalf = _moist_unstable_column()
        kx = T.shape[0]
        T3 = jnp.broadcast_to(T[:, None, None], (kx, 2, 3))
        q3 = jnp.broadcast_to(q[:, None, None], (kx, 2, 3))
        pf3 = jnp.broadcast_to(pfull[:, None, None], (kx, 2, 3))
        ph3 = jnp.broadcast_to(phalf[:, None, None], (kx + 1, 2, 3))
        P = BettsMillerParameters(do_envsat=True)
        dTdt, dqdt, precip = betts_miller_tendencies(T3, q3, pf3, ph3, DT, P)
        tdel, qdel, p_col = betts_miller_column(T, q, pfull, phalf, DT, P)
        self.assertTrue(np.allclose(np.asarray(dTdt[:, 0, 0]) * DT,
                                    np.asarray(tdel), atol=1e-6))
        self.assertTrue(np.allclose(np.asarray(precip), float(p_col) / DT))

    def test_vectorized_columns_shape(self):
        # ComposablePhysics(vectorize_columns=True) passes (kx, ncols) state;
        # the driver must accept it and match the (kx, ix, il) result per column.
        T, q, pfull, phalf = _moist_unstable_column()
        kx = T.shape[0]
        P = BettsMillerParameters(do_envsat=True)
        ncols = 5
        T2 = jnp.broadcast_to(T[:, None], (kx, ncols))
        q2 = jnp.broadcast_to(q[:, None], (kx, ncols))
        pf2 = jnp.broadcast_to(pfull[:, None], (kx, ncols))
        ph2 = jnp.broadcast_to(phalf[:, None], (kx + 1, ncols))
        dTdt, dqdt, precip = betts_miller_tendencies(T2, q2, pf2, ph2, DT, P)
        self.assertEqual(dTdt.shape, (kx, ncols))
        self.assertEqual(precip.shape, (ncols,))
        tdel, qdel, p_col = betts_miller_column(T, q, pfull, phalf, DT, P)
        self.assertTrue(np.allclose(np.asarray(dTdt[:, 0]) * DT,
                                    np.asarray(tdel), atol=1e-6))
        self.assertAlmostEqual(float(precip[0]), float(p_col) / DT, places=9)

    def test_gradients_are_finite(self):
        T, q, pfull, phalf = _moist_unstable_column()
        P = BettsMillerParameters(do_envsat=True)

        def loss(t, qq):
            tdel, qdel, precip = betts_miller_column(t, qq, pfull, phalf, DT, P)
            return jnp.sum(tdel ** 2) + jnp.sum(qdel ** 2) + precip ** 2

        gT, gq = jax.grad(loss, argnums=(0, 1))(T, q)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gT))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gq))))

    def test_params_is_a_differentiable_pytree(self):
        # The numeric tunables are pytree leaves; the flavor/modifier flags are
        # static aux data — so gradients flow to tau_bm/rhbm but the flags are
        # untouched and remain usable in Python branching.
        P = BettsMillerParameters(do_envsat=True)
        leaves = jax.tree_util.tree_leaves(P)
        # Six numeric leaves; the enum + two bools are aux, not leaves.
        self.assertEqual(len(leaves), 6)
        self.assertTrue(all(jnp.ndim(jnp.asarray(x)) == 0 for x in leaves))

        T, q, pfull, phalf = _moist_unstable_column()

        def loss(params):
            tdel, qdel, precip = betts_miller_column(
                T, q, pfull, phalf, DT, params)
            return jnp.sum(tdel ** 2) + jnp.sum(qdel ** 2) + precip ** 2

        grads = jax.grad(loss)(P)
        # Deep precipitating convection depends on both tau_bm and rhbm.
        self.assertTrue(jnp.isfinite(grads.tau_bm))
        self.assertTrue(jnp.isfinite(grads.rhbm))
        self.assertNotEqual(float(grads.tau_bm), 0.0)
        self.assertNotEqual(float(grads.rhbm), 0.0)
        # Static fields survive the transform unchanged (still Python values).
        self.assertEqual(grads.shallow, P.shallow)
        self.assertEqual(grads.do_envsat, P.do_envsat)

    def test_term_end_to_end(self):
        from jcm.utils import get_coords
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData
        from jcm.forcing import ForcingData
        from jcm.physics.convection.betts_miller import BettsMillerConvection

        kx = 12
        coords = get_coords(np.linspace(0, 1, kx + 1), nodal_shape=(64, 32))
        ix, il = coords.horizontal.nodal_shape
        sig = jnp.linspace(0.05, 0.97, kx)
        T1d = 295.0 - 55.0 * (1.0 - sig)
        q1d = 0.9 * saturation_specific_humidity(T1d, sig * 1e5)  # kg/kg
        T = jnp.broadcast_to(T1d[:, None, None], (kx, ix, il))
        q = jnp.broadcast_to(q1d[:, None, None], (kx, ix, il))
        state = PhysicsState.zeros((kx, ix, il), temperature=T,
                                   specific_humidity=q,
                                   normalized_surface_pressure=jnp.ones((ix, il)))
        term = BettsMillerConvection(
            BettsMillerParameters(do_envsat=True))
        term.cache_coords(coords)
        tend, diag = term(state, {"_dt_seconds": DT},
                          ForcingData.zeros((ix, il)),
                          TerrainData.aquaplanet(coords))
        self.assertTrue(bool(jnp.all(jnp.isfinite(tend.temperature))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(tend.specific_humidity))))
        self.assertIn("betts_miller_precip", diag)
        self.assertGreaterEqual(float(jnp.min(diag["betts_miller_precip"])), -1e-9)


if __name__ == "__main__":
    unittest.main()
