"""The Tiedtke ledger on ECHAM's half levels: cuini, the cuflx taper, cudtdq.

The scheme is a finite-volume ledger on the model's own layers (#530): plume
fluxes live on the interfaces, each layer's tendency is the difference of
the fluxes through its two interfaces over its TRUE air mass ``Δp/g``. These
tests pin the consequences a host can check from outside:

* column water changes by exactly minus the reported convective
  precipitation when integrated with the host's layer mass — on a stretched
  hybrid grid, where a dual-grid (centre-to-centre) ledger leaks;
* column moist static energy is conserved in a warm column;
* below cloud base the updraft fluxes follow cuflx's linear-in-pressure
  taper, spreading the cloud-base flux divergence through the sub-cloud
  layer;
* the ``cuini`` half-level environment has the reference's structure.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.flux_tendencies import (
    calculate_tendencies,
    subcloud_taper,
)
from jcm.physics.convection.tiedtke_nordeng.half_levels import (
    half_level_environment,
)
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    tiedtke_nordeng_convection,
)
from jcm.physics.convection.tiedtke_nordeng.types import ConvectionTendencies
from jcm.physics.convection.tiedtke_nordeng.updraft import UpdatedraftState
from jcm.physics.convection.tiedtke_nordeng.downdraft import DowndraftState
from jcm.physics.thermodynamics import moist_isobaric_heat_capacity


def _l47_tropical_column(sst=302.0, rh=0.8):
    """Build the seeded RCE column on ECHAM's stretched L47 hybrid grid."""
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.rce import _pressure_centers, rce_initial_state

    vertical = get_echam_levels(47)
    ic = rce_initial_state(vertical, sst=sst, relative_humidity=rh)
    p_full = _pressure_centers(vertical, jnp.asarray(c.p0))
    p_half = jnp.asarray(
        np.asarray(vertical.a_boundaries)
        + np.asarray(vertical.b_boundaries) * float(c.p0))
    T, q = ic.temperature, ic.specific_humidity
    rho = p_full / (c.rd * T)
    dz = jnp.diff(p_half) / (rho * c.grav)
    return T, q, p_full, p_half, dz, rho


def _run(T, q, p, p_half, dz, rho, deep=True, supply=1.5e-4, dt=900.0):
    nlev = T.shape[0]
    mass = jnp.diff(p_half) / c.grav
    kw = dict(moisture_supply=jnp.asarray(supply))
    if deep:
        # Resolved moisture convergence beyond 1.1x the supply makes ECHAM's
        # zdqcv test classify the column deep.
        sl = slice(nlev // 2, nlev - 4)
        kw["qte_dynamics"] = jnp.zeros(nlev).at[sl].set(
            1.5 * supply / jnp.sum(mass[sl]))
    z = jnp.zeros(nlev)
    return tiedtke_nordeng_convection(
        T, q, p, dz, rho, z, z, z, z, dt, ConvectionParameters.default(),
        pressure_half=p_half, **kw,
    )


def _water(tend: ConvectionTendencies):
    return np.asarray(tend.dqdt + tend.dqc_dt + tend.dqi_dt, dtype=np.float64)


class TestColumnWaterOnTrueLayerMass(unittest.TestCase):
    """Σ (dq + dqc + dqi)·Δp/g = −P with the host's Δp (#530)."""

    def _check(self, deep):
        # The shallow plume entrains at ``entrscv`` and stops at the first
        # interface where the diluted parcel does not condense; at 80 % RH
        # that is its first layer, and it does not rain. A 95 % boundary
        # layer lets it rise and precipitate.
        T, q, p, p_half, dz, rho = _l47_tropical_column(
            rh=0.8 if deep else 0.95)
        tend, state = _run(T, q, p, p_half, dz, rho, deep=deep)
        precip = float(tend.precip_conv)
        self.assertGreater(precip, 0.0, "fixture did not precipitate")
        self.assertEqual(int(state.ktype), 1 if deep else 2)
        mass = np.diff(np.asarray(p_half, dtype=np.float64)) / c.grav
        water = _water(tend)
        gross = float(np.sum(np.abs(water) * mass))
        residual = float(np.sum(water * mass)) + precip
        self.assertLess(abs(residual), 1e-5 * gross,
                        f"water budget open by {residual:.3e} kg/m2/s "
                        f"(precip {precip:.3e}, gross {gross:.3e})")
        # Not vacuous: the stretched grid's centre-to-centre spacing — the
        # mass a dual-grid ledger would be conservative in — leaves the
        # same tendencies open by far more than the closure tolerance.
        dpa = np.abs(np.diff(np.asarray(p, dtype=np.float64)))
        dual = np.concatenate([dpa, dpa[-1:]]) / c.grav
        dual_residual = float(np.sum(water * dual)) + precip
        self.assertGreater(abs(dual_residual), 1e-3 * precip)
        return tend, state

    def test_deep_column_closes(self):
        tend, _ = self._check(deep=True)
        # A deep plume here reaches the ice phase, so the budget above
        # covers the phase-keyed paths too.
        self.assertGreater(float(np.max(np.asarray(tend.dqi_dt))), 0.0)

    def test_shallow_column_closes(self):
        self._check(deep=False)


class TestWarmColumnMoistStaticEnergy(unittest.TestCase):
    """In a column warmer than the melting point everywhere, cudtdq conserves
    moist static energy exactly: Σ (cp·dT + L_v·dq)·Δp/g = 0.

    Every latent-heat factor is then ``alv`` (``zalv``, ``palvsh``) and
    nothing melts, so the heating of the per-layer sources,
    ``L_v·(plude + pdmfup + pdmfdp)``, is exactly ``−L_v`` times the vapour
    they remove, and every flux difference telescopes to zero.
    """

    def test_mse_conserved(self):
        nlev = 30
        p_half = jnp.linspace(62_000.0, 101_325.0, nlev + 1)
        p = 0.5 * (p_half[1:] + p_half[:-1])
        z = -7.6e3 * jnp.log(p / p_half[-1])
        mixed = z < 600.0
        T = jnp.where(mixed, 303.0 - 9.8e-3 * z,
                      303.0 - 9.8e-3 * 600.0 - 6.0e-3 * (z - 600.0))
        self.assertGreater(float(T.min()), c.tmelt + 5.0)
        from jcm.physics.convection.saturation import saturation_mixing_ratio
        q = jnp.where(mixed, 0.9, 0.75) * saturation_mixing_ratio(p, T)
        rho = p / (c.rd * T)
        dz = jnp.diff(p_half) / (rho * c.grav)
        tend, state = _run(T, q, p, p_half, dz, rho, deep=True)
        self.assertGreater(float(tend.precip_conv), 0.0)
        mass = np.diff(np.asarray(p_half, dtype=np.float64)) / c.grav
        cp = np.asarray(moist_isobaric_heat_capacity(q), dtype=np.float64)
        heat = cp * np.asarray(tend.dtedt, dtype=np.float64) * mass
        latent = c.alhc * np.asarray(tend.dqdt, dtype=np.float64) * mass
        scale = float(np.sum(np.abs(heat)))
        self.assertGreater(scale, 0.0)
        self.assertLess(abs(float(np.sum(heat + latent))), 1e-5 * scale)


class TestSubCloudTaper(unittest.TestCase):
    """cuflx: below kcbot the updraft fluxes are the cloud-base values times
    ``(p_s − p_half)/(p_s − p_half(kcbot))`` — so the cloud-base flux
    divergence heats every sub-cloud layer at the same rate per unit mass,
    ``−g·F(kcbot)/(p_s − p_half(kcbot))``, instead of landing on one layer.
    """

    def test_uniform_subcloud_heating(self):
        nlev = 8
        p_half = jnp.array([1.0e4, 2.0e4, 3.5e4, 5.0e4, 6.5e4, 8.0e4, 8.8e4,
                            9.5e4, 1.0e5])
        p = 0.5 * (p_half[1:] + p_half[:-1])
        T = jnp.linspace(230.0, 295.0, nlev)
        q = jnp.full(nlev, 1.0e-3)
        env = half_level_environment(
            T, q, p, p_half, moist_isobaric_heat_capacity(q))
        kbase = 5
        z = jnp.zeros(nlev)
        mfu = jnp.zeros(nlev).at[2:kbase + 1].set(0.05)
        # A plume 1 K warmer than the half-level environment at and above
        # cloud base: a pure dry-static-energy flux, no moisture/condensate.
        up = UpdatedraftState(
            tu=env.tenh + 1.0, qu=env.qenh, lu=z, mfu=mfu, entr=z, detr=z,
            buoy=z, pdmfup=z, plude=z, uu=z, vu=z,
        )
        dn = DowndraftState(td=env.tenh, qd=env.qenh, mfd=z, pdmfdp=z, ud=z,
                            vd=z, lfs=0, active=False)
        tend = calculate_tendencies(
            T, q, z, z, p, p / (c.rd * T), z + 1.0, up, dn, kbase, 2, 900.0,
            ConvectionParameters.default(), ktype=jnp.array(2),
            pressure_half=p_half,
        )
        cp = np.asarray(moist_isobaric_heat_capacity(q))
        flux_cb = float(env.cpcu[kbase]) * 1.0 * 0.05
        expected = -c.grav * flux_cb / float(p_half[-1] - p_half[kbase])
        heating = cp * np.asarray(tend.dtedt)       # W/kg
        np.testing.assert_allclose(heating[kbase:], expected, rtol=1e-5)
        # The mid-level taper is squared: no longer uniform.
        tend3 = calculate_tendencies(
            T, q, z, z, p, p / (c.rd * T), z + 1.0, up, dn, kbase, 2, 900.0,
            ConvectionParameters.default(), ktype=jnp.array(3),
            pressure_half=p_half,
        )
        h3 = cp * np.asarray(tend3.dtedt)
        self.assertFalse(np.allclose(h3[kbase:], expected, rtol=1e-3))
        # Either way the sub-cloud layer takes the whole cloud-base flux.
        mass = np.diff(np.asarray(p_half)) / c.grav
        for h in (heating, h3):
            np.testing.assert_allclose(
                np.sum(h[kbase:] * mass[kbase:]), -flux_cb, rtol=1e-5)


class TestFinalAscentRespectsZmfmax(unittest.TestCase):
    """The closure's amplitude is applied by a second ascent (cumastr), so
    ECHAM's ``zmfmax`` limiter — no interface may pass more than the air
    mass of the layer above it per step — holds for the FINAL plume, to the
    precision ECHAM's own ordering of the limiters gives it. A linear
    rescale of the first ascent would not keep it.
    """

    def test_interface_fluxes_within_layer_mass_per_step(self):
        T, q, p, p_half, dz, rho = _l47_tropical_column()
        dt = 3600.0
        for supply in (1.5e-4, 5.0e-4):
            _, state = _run(T, q, p, p_half, dz, rho, deep=True,
                            supply=supply, dt=dt)
            mfu = np.asarray(state.mfu, dtype=np.float64)
            kb = int(state.kbase)
            self.assertGreater(mfu.max(), 0.0)
            dp = np.diff(np.asarray(p_half, dtype=np.float64))
            cap = dp[:-1] / (c.grav * dt)          # layer above interface k
            ratio = mfu[1:kb + 1] / cap[:kb]
            # cuasc applies the organized limiter with the organized
            # detrainment BEFORE ``zodmax`` reduces it (mo_cuascent.f90:
            # 365-383), so where ``zodmax`` binds the final flux may pass
            # the layer mass by the detrainment ``zodmax`` removed — a
            # fraction of a percent here, as in ECHAM.
            self.assertLessEqual(float(ratio.max()), 1.01,
                                 f"supply {supply}: {ratio.max():.4f}")


class TestPublishedSubCloudTaper(unittest.TestCase):
    """``subcloud_taper`` — the cuflx taper the ledger applies and the term
    publishes on ``mass_flux_up`` — is broadcasting-native and exact.
    """

    def test_column_and_block_agree_and_match_cuflx(self):
        nlev, ncols = 6, 3
        p_half = jnp.array([0.0, 2.0e4, 4.0e4, 6.0e4, 8.0e4, 9.0e4, 1.0e5])
        flux = jnp.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.0])
        kbase = 3
        col = np.asarray(subcloud_taper(flux, kbase, 1, p_half))
        # At and above the base: unchanged. Below: F(kb)·(ps−p)/(ps−p_kb).
        np.testing.assert_allclose(col[:4], np.asarray(flux)[:4])
        np.testing.assert_allclose(
            col[4:], 0.3 * np.array([2.0e4, 1.0e4]) / 4.0e4, rtol=1e-6)
        mid = np.asarray(subcloud_taper(flux, kbase, 3, p_half))
        np.testing.assert_allclose(
            mid[4:], 0.3 * (np.array([2.0e4, 1.0e4]) / 4.0e4) ** 2,
            rtol=1e-6)
        block = np.asarray(subcloud_taper(
            jnp.tile(flux[:, None], (1, ncols)),
            jnp.array([kbase, kbase, nlev - 1]),
            jnp.array([1, 3, 0]),
            jnp.tile(p_half[:, None], (1, ncols)),
        ))
        np.testing.assert_allclose(block[:, 0], col, rtol=1e-6)
        np.testing.assert_allclose(block[:, 1], mid, rtol=1e-6)
        # A column without a plume base (kbase = nlev − 1) is untouched.
        np.testing.assert_allclose(block[:, 2], np.asarray(flux))


class TestHalfLevelEnvironment(unittest.TestCase):
    """Structure of ECHAM ``cuini`` (mo_cuinitialize.f90:31-230)."""

    def setUp(self):
        T, q, p, p_half, _, _ = _l47_tropical_column()
        self.T, self.q, self.p, self.p_half = T, q, p, p_half
        self.cp = moist_isobaric_heat_capacity(q)
        self.env = half_level_environment(T, q, p, p_half, self.cp)

    def test_boundaries_and_geopotential(self):
        env = self.env
        # Model top: the full-level values; bottom interface: the bottom
        # full level's humidity and its dry static energy carried up.
        self.assertEqual(float(env.tenh[0]), float(self.T[0]))
        self.assertEqual(float(env.qenh[-1]), float(self.q[-1]))
        np.testing.assert_allclose(
            float(env.cpcu[-1]) * 0 + float(self.cp[-1]) * float(env.tenh[-1])
            + float(env.geoh[-1]),
            float(env.dse[-1]), rtol=1e-6)
        # Interfaces sit between full levels, all above the surface.
        geoh, geo = np.asarray(env.geoh), np.asarray(env.geo)
        self.assertTrue(np.all(geoh[1:] < geo[:-1]))
        self.assertTrue(np.all(geoh[1:] > geo[1:]))
        self.assertTrue(np.all(geo > 0.0))

    def test_half_level_dse_never_decreases_upward(self):
        # The ``zzs`` running maximum: the interface profile is never
        # dry-unstable (below the model-top interface).
        env = self.env
        sh = np.asarray(env.cpcu * env.tenh + env.geoh)[1:]
        self.assertTrue(np.all(np.diff(sh) <= 1e-3))

    def test_half_level_values_bracketed_by_neighbours(self):
        env = self.env
        T, q = np.asarray(self.T), np.asarray(self.q)
        tenh, qenh = np.asarray(env.tenh), np.asarray(env.qenh)
        self.assertTrue(np.all(np.isfinite(tenh)) and np.all(qenh >= 0.0))
        # Tropospheric interfaces (below 200 hPa — the plume's domain) lie
        # within a couple of kelvin of the two full levels they separate.
        # Higher up the layers are kilometres thick and statically stable,
        # so the upper level's dry static energy brought down half a layer
        # (cuini's choice there) sits several kelvin above both neighbours;
        # and near the model top, at a few Pa, the saturation adjustment
        # works with a saturation humidity capped at 0.5, so the interface
        # values carry no physical meaning. No plume reaches either.
        trop = np.asarray(env.paph)[1:-1] > 2.0e4
        lo = np.minimum(T[:-1], T[1:]) - 2.0
        hi = np.maximum(T[:-1], T[1:]) + 2.0
        ok = (tenh[1:] >= lo) & (tenh[1:] <= hi)
        self.assertTrue(np.all(ok[trop]))
        # The humidity is cuini's moist interpolation: the level above's
        # (sub)saturated humidity plus the change of saturation humidity
        # from that level to the interface.
        qs, qsh = np.asarray(env.qsen), np.asarray(env.qsenh)
        expected = np.maximum(
            np.minimum(q[:-2], qs[:-2]) + (qsh[1:-1] - qs[:-2]), 0.0)
        np.testing.assert_allclose(qenh[1:-1], expected, rtol=1e-5,
                                   atol=1e-12)


if __name__ == "__main__":
    unittest.main()
