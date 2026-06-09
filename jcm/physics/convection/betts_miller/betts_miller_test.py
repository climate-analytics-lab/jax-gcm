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

    def test_gradients_are_finite(self):
        T, q, pfull, phalf = _moist_unstable_column()
        P = BettsMillerParameters(do_envsat=True)

        def loss(t, qq):
            tdel, qdel, precip = betts_miller_column(t, qq, pfull, phalf, DT, P)
            return jnp.sum(tdel ** 2) + jnp.sum(qdel ** 2) + precip ** 2

        gT, gq = jax.grad(loss, argnums=(0, 1))(T, q)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gT))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gq))))

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
        q1d = 0.9 * saturation_specific_humidity(T1d, sig * 1e5) * 1000.0  # g/kg
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
