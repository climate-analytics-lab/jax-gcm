"""Tests for the convective bulk-plume tracer transport (#602, #621, #622)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.convection.tracer_transport import (
    ConvTransportParameters,
    ConvectiveTracerTransport,
    convective_tracer_tendency,
    release_scavenged,
)
from jcm.physics_interface import PhysicsState


def _plume(nlev=10, ncols=1, base=8, top=3, mf=0.05):
    """Synthetic updraft: base supply at ``base``, detrainment near ``top``.

    ``mfu[k]`` is the flux at each layer's TOP interface: constant ``mf``
    from the base layer up to (and including) ``top``, zero above.
    Entrainment: the base supply plus a small lateral pickup per layer.
    """
    mfu = jnp.zeros((nlev, ncols))
    lev = jnp.arange(nlev)[:, None]
    inside = (lev >= top) & (lev <= base)
    mfu = jnp.where(inside, mf, 0.0) * jnp.ones((1, ncols))
    entrain = jnp.zeros((nlev, ncols))
    entrain = entrain.at[base].set(mf)          # cloud-base supply
    return mfu, entrain


def _downdraft(nlev=10, ncols=1, lfs=4, mf=0.02, entrdd=2.0e-4, dz=400.0):
    """Synthetic downdraft mirroring the Tiedtke (ECHAM half-level) conventions.

    ``mfd[k]`` (≤ 0) is the flux through the TOP interface of layer k (the
    same interface as ``mfu[k]``): the LFS seed at interface ``lfs``,
    constant through the bulk down to ``itopde = nlev − 3``, then the
    cuddraf taper, linear towards zero at the surface interface (which
    carries no flux). ``entrain_down`` is the cuddraf turbulent ledger
    ``entrdd·|mfd|·dz`` of the layers the descent crosses in the bulk,
    zero in the taper.
    """
    lev = jnp.arange(nlev)[:, None]
    mfd = jnp.where((lev >= lfs) & (lev <= nlev - 3), -mf, 0.0)
    mfd = mfd.at[nlev - 2].set(-mf * 2.0 / 3.0).at[nlev - 1].set(-mf / 3.0)
    mfd = jnp.broadcast_to(mfd, (nlev, ncols))
    mfd_out = jnp.concatenate([mfd[1:], jnp.zeros((1, ncols))], axis=0)
    e_dn = jnp.where(
        (mfd < 0) & (mfd_out < 0) & (lev < nlev - 3),
        entrdd * jnp.abs(mfd) * dz, 0.0,
    )
    return mfd, e_dn


def _column_budgets(dq, dm):
    net = float(jnp.sum(dq * dm[jnp.newaxis]))
    gross = float(jnp.sum(jnp.abs(dq) * dm[jnp.newaxis]))
    return net, gross


class ConvectiveTendencyTest(unittest.TestCase):
    def _grid(self, nlev=10, ncols=1):
        rho = jnp.linspace(0.4, 1.2, nlev)[:, None] * jnp.ones((1, ncols))
        dz = jnp.full((nlev, ncols), 400.0)
        return rho, dz

    def test_conserves_column_mass_exactly(self):
        rho, dz = self._grid()
        mfu, entrain = _plume()
        q = jnp.stack([
            jnp.linspace(2.0, 0.1, 10)[:, None] ** 2 * 1e-9
            * jnp.ones((1, 1)),
        ])
        dq, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        net, gross = _column_budgets(dq, rho * dz)
        self.assertGreater(gross, 0.0, "plume did nothing — fixture off")
        self.assertLessEqual(abs(net), 1e-6 * gross)

    def test_lofts_boundary_layer_tracer(self):
        # A surface-concentrated tracer entrained at cloud base must be
        # deposited at the detrainment levels aloft and reduced below.
        rho, dz = self._grid()
        mfu, entrain = _plume(base=8, top=3)
        q = jnp.zeros((1, 10, 1)).at[0, 8:].set(1.0e-9)
        dq, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        # Detrainment lands in the layer ABOVE the last carrying interface
        # (mfu is the layer-TOP flux, so the plume dies inside level top-1).
        self.assertGreater(float(dq[0, 2, 0]), 0.0)
        # The base layer loses to the updraft.
        self.assertLess(float(dq[0, 8, 0]), 0.0)

    def test_plume_concentration_bounded_by_environment(self):
        # The plume is a convex mix of what it entrained, and subsidence
        # advects environment values — no new extrema can appear.
        rho, dz = self._grid()
        mfu, entrain = _plume()
        q0 = jnp.linspace(0.0, 1.0e-9, 10)[:, None][jnp.newaxis]
        dq, _ = convective_tracer_tendency(q0, mfu, entrain, rho, dz, 1800.0)
        q1 = q0 + 1800.0 * dq
        self.assertGreaterEqual(float(q1.min()), -1e-25)
        self.assertLessEqual(float(q1.max()), 1.0e-9 * (1.0 + 1e-6))

    def test_plume_through_model_top_still_conserves(self):
        # A pathological mass-flux profile that reaches the top layer must
        # detrain there (no flux through the model top), not leak tracer.
        rho, dz = self._grid()
        mfu = jnp.full((10, 1), 0.05)
        entrain = jnp.zeros((10, 1)).at[9].set(0.05)
        q = jnp.stack([jnp.linspace(1.0, 0.1, 10)[:, None] * 1e-9])
        dq, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        net, gross = _column_budgets(dq, rho * dz)
        self.assertLessEqual(abs(net), 1e-6 * max(gross, 1e-30))

    def test_thin_detrainment_layer_stays_bounded(self):
        # Adversarial-review repro: the environment sink in a
        # net-detrainment layer is (E_eff + mfu_below)·dt/dm — the flux
        # from the layer BELOW over this layer's OWN mass. With thin
        # layers aloft (hybrid-coordinate dm shrinking with height) and a
        # near-CFL base flux, a guard formed per layer from (mfu + E)
        # missed that cross-level ratio: the sink reached 1.556 and a
        # bounded tracer went to -0.556. The guard must bound the DERIVED
        # sink, keeping the update positivity-preserving here.
        nlev = 6
        dm_prof = jnp.array([60.0, 90.0, 300.0, 700.0, 1000.0, 1300.0])
        rho = jnp.ones((nlev, 1))
        dz = dm_prof[:, None]                     # rho = 1 → dm = dz
        lev = jnp.arange(nlev)[:, None]
        mfu = jnp.where((lev >= 1) & (lev <= 5), 0.55, 0.0) * jnp.ones((1, 1))
        entrain = jnp.zeros((nlev, 1)).at[5].set(0.55)
        dt = 1800.0
        # Tracer concentrated in the detrainment layer (level 0): the
        # over-drained case.
        q = jnp.zeros((1, nlev, 1)).at[0, 0].set(1.0)
        dq, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, dt)
        q1 = q + dt * dq
        self.assertGreaterEqual(float(q1.min()), -1e-9)
        # Bounded-in-[0,1] tracer cannot exceed 1 either.
        q = jnp.ones((1, nlev, 1))
        dq, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, dt)
        q1 = q + dt * dq
        self.assertLessEqual(float(q1.max()), 1.0 + 1e-9)
        self.assertGreaterEqual(float(q1.min()), -1e-9)

    def test_conserves_with_distinct_tracers_and_columns(self):
        # K=2 tracers x 2 distinct columns, both legs active: a transposed
        # (K, ncols) carry in either plume scan mixes them and breaks the
        # per-tracer, per-column budgets.
        nlev = 10
        rho = jnp.linspace(0.4, 1.2, nlev)[:, None] * jnp.ones((1, 2))
        dz = jnp.full((nlev, 2), 400.0)
        mfu, entrain = _plume(ncols=2)
        mfu = mfu * jnp.asarray([1.0, 0.5])[None, :]
        entrain = entrain * jnp.asarray([1.0, 0.5])[None, :]
        mfd, e_dn = _downdraft(ncols=2)
        mfd = mfd * jnp.asarray([1.0, 0.5])[None, :]
        e_dn = e_dn * jnp.asarray([1.0, 0.5])[None, :]
        q = jnp.stack([
            jnp.zeros((nlev, 2)).at[8:].set(1.0e-9),
            jnp.linspace(1.0, 0.2, nlev)[:, None] * jnp.ones((1, 2)) * 1e-8,
        ])
        dq, _ = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0, mfd=mfd, entrain_down=e_dn,
        )
        dm = rho * dz
        for k in range(2):
            for col in range(2):
                net = float(jnp.sum(dq[k, :, col] * dm[:, col]))
                gross = float(jnp.sum(jnp.abs(dq[k, :, col]) * dm[:, col]))
                self.assertLessEqual(
                    abs(net), 1e-6 * max(gross, 1e-30), (k, col),
                )

    def test_huge_mass_flux_stays_positive(self):
        # The per-column CFL rescale bounds the explicit update: even a
        # mass flux that would empty a layer many times over cannot drive
        # a tracer negative — with the downdraft leg adding to the sink.
        rho, dz = self._grid()
        mfu, entrain = _plume(mf=50.0)
        mfd, e_dn = _downdraft(mf=20.0)
        q = jnp.zeros((1, 10, 1)).at[0, 8:].set(1.0e-9)
        dq, _ = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0, mfd=mfd, entrain_down=e_dn,
        )
        q1 = q + 1800.0 * dq
        self.assertGreaterEqual(float(q1.min()), -1e-25)
        self.assertTrue(bool(jnp.all(jnp.isfinite(dq))))

    def test_no_mass_flux_no_tendency(self):
        rho, dz = self._grid()
        q = jnp.ones((1, 10, 1)) * 1e-9
        dq, scav = convective_tracer_tendency(
            q, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
            mfd=jnp.zeros((10, 1)), entrain_down=jnp.zeros((10, 1)),
        )
        np.testing.assert_array_equal(np.asarray(dq), 0.0)
        np.testing.assert_array_equal(np.asarray(scav), 0.0)


class DowndraftLegTest(unittest.TestCase):
    """The mfd side of the transport (jax-gcm#622)."""

    def _grid(self, nlev=10, ncols=1):
        rho = jnp.linspace(0.4, 1.2, nlev)[:, None] * jnp.ones((1, ncols))
        dz = jnp.full((nlev, ncols), 400.0)
        return rho, dz

    def test_downdraft_conserves_column_mass_exactly(self):
        rho, dz = self._grid()
        mfd, e_dn = _downdraft()
        q = jnp.stack([
            (jnp.linspace(0.3, 2.0, 10)[:, None]) ** 2 * 1e-9
            * jnp.ones((1, 1)),
        ])
        dq, _ = convective_tracer_tendency(
            q, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
            mfd=mfd, entrain_down=e_dn,
        )
        net, gross = _column_budgets(dq, rho * dz)
        self.assertGreater(gross, 0.0, "downdraft did nothing — fixture off")
        self.assertLessEqual(abs(net), 1e-6 * gross)

    def test_downdraft_carries_lfs_air_into_subcloud(self):
        # The LFS interface is the top of layer 4, so the seed mass is drawn
        # from the layer above it (3) — the layer ECHAM's flux-form ledger
        # debits through that interface. A tracer confined there must be
        # reduced and must appear in the sub-cloud taper layers, where the
        # descent detrains.
        rho, dz = self._grid()
        mfd, e_dn = _downdraft(lfs=4)
        q = jnp.zeros((1, 10, 1)).at[0, 3].set(1.0e-9)
        dq, _ = convective_tracer_tendency(
            q, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
            mfd=mfd, entrain_down=e_dn,
        )
        self.assertLess(float(dq[0, 3, 0]), 0.0)
        self.assertGreater(float(dq[0, 8, 0]), 0.0)
        self.assertGreater(float(dq[0, 9, 0]), 0.0)

    def test_downdraft_plume_bounded_by_environment(self):
        # The descent is a convex mix all the way down: no new extrema.
        rho, dz = self._grid()
        mfd, e_dn = _downdraft()
        q0 = jnp.linspace(1.0e-9, 0.0, 10)[:, None][jnp.newaxis]
        dq, _ = convective_tracer_tendency(
            q0, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
            mfd=mfd, entrain_down=e_dn,
        )
        q1 = q0 + 1800.0 * dq
        self.assertGreaterEqual(float(q1.min()), -1e-25)
        self.assertLessEqual(float(q1.max()), 1.0e-9 * (1.0 + 1e-6))

    def test_downdraft_dying_middescent_still_conserves(self):
        # Buoyancy shut-off mid-column (mfd -> 0 with inflow above):
        # continuity must dump the arriving flux as detrainment there.
        rho, dz = self._grid()
        lev = jnp.arange(10)[:, None]
        mfd = jnp.where((lev >= 3) & (lev <= 5), -0.02, 0.0)
        e_dn = jnp.where((lev >= 3) & (lev < 5), 2.0e-4 * 0.02 * 400.0, 0.0)
        q = jnp.stack([jnp.linspace(0.5, 1.5, 10)[:, None] * 1e-9])
        dq, _ = convective_tracer_tendency(
            q, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
            mfd=mfd, entrain_down=e_dn,
        )
        net, gross = _column_budgets(dq, rho * dz)
        self.assertGreater(gross, 0.0)
        self.assertLessEqual(abs(net), 1e-6 * gross)


class ScavengingTest(unittest.TestCase):
    """HAMMOZ-parameterised in-plume removal in a closed plume (jax-gcm#621)."""

    def _setup(self, nlev=10, ncols=1, peff=0.5):
        rho = jnp.linspace(0.4, 1.2, nlev)[:, None] * jnp.ones((1, ncols))
        dz = jnp.full((nlev, ncols), 400.0)
        mfu, entrain = _plume(nlev=nlev, ncols=ncols)
        # Condensate and precipitation efficiency inside the cloudy layers.
        lev = jnp.arange(nlev)[:, None]
        cloudy = (lev >= 4) & (lev <= 7)
        cond = jnp.where(cloudy, 5.0e-4, 0.0) * jnp.ones((1, ncols))
        eff = jnp.where(cloudy, peff, 0.0) * jnp.ones((1, ncols))
        q = jnp.stack([
            jnp.zeros((nlev, ncols)).at[8:].set(1.0e-9),
            jnp.zeros((nlev, ncols)).at[8:].set(1.0e-9),
        ])
        return q, mfu, entrain, rho, dz, cond, eff

    def test_budget_closes_to_scavenged_flux(self):
        # Column change must equal MINUS the scavenged surface flux,
        # per tracer, exactly — with and without the release below.
        q, mfu, entrain, rho, dz, cond, eff = self._setup()
        evap = jnp.zeros_like(cond).at[8:].set(0.3)
        for ev in (None, evap):
            dq, scav = convective_tracer_tendency(
                q, mfu, entrain, rho, dz, 1800.0,
                csr_conv=jnp.asarray([0.99, 0.2]),
                precip_efficiency=eff, plume_condensate=cond,
                evap_fraction=ev,
            )
            dm = rho * dz
            for k in range(2):
                net = float(jnp.sum(dq[k] * dm))
                self.assertGreater(float(scav[k, 0]), 0.0)
                self.assertLessEqual(
                    abs(net + float(scav[k, 0])),
                    1e-6 * float(jnp.sum(jnp.abs(dq[k]) * dm)),
                )

    def test_closes_to_round_off_without_a_mass_fixer(self):
        # HAMMOZ needs ``xt_conv_massfix`` because its post-ascent removal
        # and total-flux overwrite are not a plume budget; here the removal
        # is taken from the plume itself, so column change + deposition is
        # zero to float round-off in float64.
        prior = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            q, mfu, entrain, rho, dz, cond, eff = (
                jnp.asarray(a, jnp.float64) for a in self._setup())
            mfd, e_dn = _downdraft()
            evap = jnp.zeros_like(cond).at[8:].set(0.3)
            dq, scav = convective_tracer_tendency(
                q, mfu, entrain, rho, dz, 1800.0,
                mfd=jnp.asarray(mfd, jnp.float64),
                entrain_down=jnp.asarray(e_dn, jnp.float64),
                csr_conv=jnp.asarray([0.99, 0.2], jnp.float64),
                precip_efficiency=eff, plume_condensate=cond,
                evap_fraction=evap,
            )
            dm = rho * dz
            for k in range(2):
                gross = float(jnp.sum(jnp.abs(dq[k]) * dm))
                resid = float(jnp.sum(dq[k] * dm) + scav[k, 0])
                self.assertLess(abs(resid), 1e-13 * gross)
        finally:
            jax.config.update("jax_enable_x64", prior)

    def test_scavenging_thins_what_detrains_aloft(self):
        q, mfu, entrain, rho, dz, cond, eff = self._setup()
        dq0, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        dq1, _ = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.asarray([0.99, 0.99]),
            precip_efficiency=eff, plume_condensate=cond,
        )
        self.assertLess(float(dq1[0, 2, 0]), float(dq0[0, 2, 0]))

    def test_zero_fractions_recover_conservative_plume(self):
        q, mfu, entrain, rho, dz, cond, eff = self._setup()
        dq0, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        dq1, scav = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.zeros(2),
            precip_efficiency=eff, plume_condensate=cond,
        )
        np.testing.assert_allclose(np.asarray(dq1), np.asarray(dq0))
        np.testing.assert_array_equal(np.asarray(scav), 0.0)

    def test_negative_lobe_never_scavenged(self):
        # Spectral ringing leaves negative lobes on near-zero tracers;
        # scavenging a negative in-plume concentration would inject mass
        # and turn the deposition flux negative (Codex P1 on #636). The
        # flux must stay >= 0 and the budget must still close to it.
        q, mfu, entrain, rho, dz, cond, eff = self._setup()
        q = q.at[0].set(-1.0e-10)                 # all-negative tracer 0
        dq, scav = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.ones(2),
            precip_efficiency=eff, plume_condensate=cond,
        )
        self.assertGreaterEqual(float(scav.min()), 0.0)
        np.testing.assert_array_equal(np.asarray(scav[0]), 0.0)
        dm = rho * dz
        for k in range(2):
            net = float(jnp.sum(dq[k] * dm))
            self.assertLessEqual(
                abs(net + float(scav[k, 0])),
                1e-6 * max(float(jnp.sum(jnp.abs(dq[k]) * dm)), 1e-30),
            )

    def test_bare_column_matches_ncols_one(self):
        # Broadcasting-native: (K, nlev) column inputs must agree with the
        # same column as a (K, nlev, 1) block (the SCM driver shape).
        q, mfu, entrain, rho, dz, cond, eff = self._setup()
        evap = jnp.zeros_like(cond).at[8:].set(0.3)
        dq3, scav3 = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.asarray([0.99, 0.2]), precip_efficiency=eff,
            plume_condensate=cond, evap_fraction=evap)
        dq1, scav1 = convective_tracer_tendency(
            q[..., 0], mfu[:, 0], entrain[:, 0], rho[:, 0], dz[:, 0],
            1800.0, csr_conv=jnp.asarray([0.99, 0.2]),
            precip_efficiency=eff[:, 0], plume_condensate=cond[:, 0],
            evap_fraction=evap[:, 0])
        np.testing.assert_allclose(np.asarray(dq1), np.asarray(dq3[..., 0]),
                                   rtol=1e-6, atol=1e-30)
        np.testing.assert_allclose(np.asarray(scav1),
                                   np.asarray(scav3[:, 0]), rtol=1e-6)

    def test_dry_plume_scavenges_nothing(self):
        # No condensate above HAMMOZ's zmin -> nothing activates, nothing
        # is removed even where a conversion fraction is supplied.
        q, mfu, entrain, rho, dz, _, eff = self._setup()
        _, scav = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.ones(2), precip_efficiency=eff,
            plume_condensate=jnp.full(mfu.shape, 1.0e-11),
        )
        np.testing.assert_array_equal(np.asarray(scav), 0.0)

    def test_in_condensate_share_is_taken_once(self):
        # ``csr`` of the aerosol joins the condensate ONCE, where the plume
        # first meets cloud; only that share loses ``peff`` per cloudy
        # layer and the ``1 − csr`` share rides to the top. Four cloudy
        # layers remove csr·(1 − (1 − peff)^4) of the cloud-base supply —
        # not 1 − (1 − csr·peff)^4, the result of offering the leftover to
        # the fixed fraction again at every level.
        csr, peff = 0.9, 0.6
        q, mfu, entrain, rho, dz, cond, eff = self._setup(peff=peff)
        mf, q_base = 0.05, 1.0e-9
        dq, scav = convective_tracer_tendency(
            q[:1], mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.asarray([csr]),
            precip_efficiency=eff, plume_condensate=cond,
        )
        expected = mf * q_base * csr * (1.0 - (1.0 - peff) ** 4)
        np.testing.assert_allclose(float(scav[0, 0]), expected, rtol=1e-5)
        compounded = mf * q_base * (1.0 - (1.0 - csr * peff) ** 4)
        self.assertGreater(compounded / expected, 1.05)
        dm = rho * dz
        survive = (1.0 - csr) + csr * (1.0 - peff) ** 4
        np.testing.assert_allclose(
            float(dq[0, 2, 0] * dm[2, 0]), mf * q_base * survive, rtol=1e-5,
        )

    def test_air_entrained_in_cloud_joins_the_condensate(self):
        # Aerosol entrained above cloud base joins the condensate at csr
        # where it enters. The layer detrains the same mass flux, but at
        # the INCOMING plume concentration (cuasc's flux form), which holds
        # none of this tracer, so all of it continues and loses peff at
        # that layer and at the two cloudy layers above.
        csr, peff, lateral = 0.9, 0.6, 0.01
        q, mfu, entrain, rho, dz, cond, eff = self._setup(peff=peff)
        entrain = entrain.at[6].set(lateral)
        qe = jnp.zeros((1, 10, 1)).at[0, 6].set(1.0e-9)
        _, scav = convective_tracer_tendency(
            qe, mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.asarray([csr]),
            precip_efficiency=eff, plume_condensate=cond,
        )
        expected = lateral * 1.0e-9 * csr * (1.0 - (1.0 - peff) ** 3)
        np.testing.assert_allclose(float(scav[0, 0]), expected, rtol=1e-5)

    def test_detrained_air_leaves_before_this_layers_removal(self):
        # One cloudy layer where half the incoming plume detrains
        # (mo_cuascent.f90:421-424) and the rest continues and
        # precipitates (cuasc 446-462 on pmfu(jk)); HAMMOZ deposits
        # zdep = pxtu·csr_conv·peff·pmfu(jk) (mo_ham_wetdep.f90:250, 325).
        # So the deposition is peff·csr·x·M_k on the CONTINUING flux, the
        # detrained half leaves unscavenged at x, and mass closes.
        nlev, mf, x, csr, peff = 6, 0.04, 1.0e-9, 0.99, 0.6
        rho = jnp.full((nlev, 1), 1.0)
        dz = jnp.full((nlev, 1), 400.0)
        # Plume enters layer 3 from below at mf (base supply in layer 4),
        # half detrains in layer 3, the rest leaves through its top and
        # detrains entirely in layer 2.
        mfu = jnp.zeros((nlev, 1)).at[4, 0].set(mf).at[3, 0].set(0.5 * mf)
        entrain = jnp.zeros((nlev, 1)).at[4, 0].set(mf)
        q = jnp.zeros((1, nlev, 1)).at[0, 4, 0].set(x)
        cond = jnp.zeros((nlev, 1)).at[3, 0].set(5.0e-4)
        eff = jnp.zeros((nlev, 1)).at[3, 0].set(peff)
        dq, scav = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            csr_conv=jnp.asarray([csr]),
            precip_efficiency=eff, plume_condensate=cond,
        )
        np.testing.assert_allclose(float(scav[0, 0]),
                                   peff * csr * x * 0.5 * mf, rtol=1e-5)
        dm = rho * dz
        # Layer 3 receives the detrained half unscavenged (x·mf/2) less
        # the compensating subsidence it passes down (none: the
        # environment above is clean); layer 2 the scavenged remainder.
        np.testing.assert_allclose(float(dq[0, 3, 0] * dm[3, 0]),
                                   0.5 * mf * x, rtol=1e-5)
        np.testing.assert_allclose(float(dq[0, 2, 0] * dm[2, 0]),
                                   0.5 * mf * x * (1.0 - csr * peff),
                                   rtol=1e-5)
        net = float(jnp.sum(dq[0] * dm))
        self.assertLess(abs(net + float(scav[0, 0])), 1e-6 * mf * x)

    def test_evaporation_releases_the_falling_deposit(self):
        # HAMMOZ's re-evaporation ledger: top to bottom, prevap of the
        # running deposit returns to the environment at each level; what
        # is left reaches the surface. Two sub-cloud layers evaporating
        # 30 % and 50 % leave 0.7·0.5 of the in-cloud removal.
        q, mfu, entrain, rho, dz, cond, eff = self._setup()
        kw = dict(csr_conv=jnp.asarray([0.99, 0.2]), precip_efficiency=eff,
                  plume_condensate=cond)
        dq0, scav0 = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0, **kw)
        evap = jnp.zeros_like(cond).at[8].set(0.3).at[9].set(0.5)
        dq1, scav1 = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0, evap_fraction=evap, **kw)
        np.testing.assert_allclose(np.asarray(scav1),
                                   0.35 * np.asarray(scav0), rtol=1e-5)
        dm = rho * dz
        released = (dq1 - dq0) * dm[jnp.newaxis]
        np.testing.assert_allclose(np.asarray(released[:, 8, 0]),
                                   0.3 * np.asarray(scav0[:, 0]), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(released[:, 9, 0]),
                                   0.35 * np.asarray(scav0[:, 0]), rtol=1e-5)

    def test_release_scavenged_conserves(self):
        removed = jnp.asarray(np.random.default_rng(0).uniform(
            0, 1, (3, 6, 2)))
        evap = jnp.asarray(np.random.default_rng(1).uniform(0, 1, (6, 2)))
        released, surface = release_scavenged(removed, evap)
        np.testing.assert_allclose(
            np.asarray(released.sum(axis=1) + surface),
            np.asarray(removed.sum(axis=1)), rtol=1e-6)
        self.assertGreaterEqual(float(released.min()), 0.0)

    def test_gradients_finite_precipitating_and_dry(self):
        # The removal and release are products of clipped fractions with
        # the plume pools; the pools' guarded divisions keep reverse-mode
        # finite in a precipitating column and in one with no plume at all.
        q, mfu, entrain, rho, dz, cond, eff = self._setup()
        evap = jnp.zeros_like(cond).at[8:].set(0.3)
        cases = {
            "precipitating": (mfu, entrain, cond, eff, evap),
            "dry": (jnp.zeros_like(mfu), jnp.zeros_like(entrain),
                    jnp.zeros_like(cond), jnp.zeros_like(eff),
                    jnp.zeros_like(evap)),
        }
        for name, (m, e, c_, pe, ev) in cases.items():
            def loss(qq, csr, pe_, ev_):
                dq, scav = convective_tracer_tendency(
                    qq, m, e, rho, dz, 1800.0, csr_conv=csr,
                    precip_efficiency=pe_, plume_condensate=c_,
                    evap_fraction=ev_)
                return jnp.sum(dq ** 2) * 1e20 + jnp.sum(scav) * 1e9
            grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
                q, jnp.asarray([0.99, 0.2]), pe, ev)
            for g in grads:
                self.assertTrue(bool(jnp.all(jnp.isfinite(g))), name)


class _Conv:
    def __init__(self, mfu, entrain, mfd=None, e_dn=None, eff=None, cond=None,
                 evap=None):
        zeros = jnp.zeros_like(mfu)
        self.mass_flux_up = mfu
        self.entrain_up = entrain
        self.mass_flux_down = mfd if mfd is not None else zeros
        self.entrain_down = e_dn if e_dn is not None else zeros
        self.precip_efficiency = eff if eff is not None else zeros
        self.precip_evap_fraction = evap if evap is not None else zeros
        self.qc_conv = cond if cond is not None else zeros
        self.qi_conv = zeros


class ConvectiveTracerTransportTermTest(unittest.TestCase):
    def _setup(self, with_conv=True, with_downdraft=False, with_scav=False,
               nlev=10, ncols=1):
        shape = (nlev, ncols)
        tracers = {"m_so4_acc": jnp.zeros(shape).at[8:].set(1.0e-9)}
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 280.0), tracers=tracers,
        )
        diagnostics = {
            "air_density": jnp.full(shape, 1.0),
            "layer_thickness": jnp.full(shape, 400.0),
            "_dt_seconds": 1800.0,
        }
        if with_conv:
            mfu, entrain = _plume(nlev=nlev, ncols=ncols)
            kwargs = {}
            if with_downdraft:
                kwargs["mfd"], kwargs["e_dn"] = _downdraft(
                    nlev=nlev, ncols=ncols,
                )
            if with_scav:
                lev = jnp.arange(nlev)[:, None]
                cloudy = (lev >= 4) & (lev <= 7)
                kwargs["cond"] = jnp.where(cloudy, 5.0e-4, 0.0) * jnp.ones(
                    (1, ncols))
                kwargs["eff"] = jnp.where(cloudy, 0.5, 0.0) * jnp.ones(
                    (1, ncols))
            diagnostics["convection"] = _Conv(mfu, entrain, **kwargs)
        return state, diagnostics

    def test_transports_and_conserves(self):
        state, diagnostics = self._setup(with_downdraft=True)
        term = ConvectiveTracerTransport(("m_so4_acc",))
        tend, _ = term(state, diagnostics, None, None)
        dq = np.asarray(tend.tracers["m_so4_acc"])
        self.assertGreater(float(dq[2, 0]), 0.0)
        self.assertLess(float(dq[8, 0]), 0.0)
        net = float(np.sum(dq) * 400.0)
        gross = float(np.sum(np.abs(dq)) * 400.0)
        self.assertLessEqual(abs(net), 1e-6 * gross)

    def test_noop_without_convection_diagnostic(self):
        state, diagnostics = self._setup(with_conv=False)
        term = ConvectiveTracerTransport(("m_so4_acc",))
        tend, _ = term(state, diagnostics, None, None)
        np.testing.assert_array_equal(
            np.asarray(tend.tracers["m_so4_acc"]), 0.0,
        )

    def test_scavenging_publishes_surface_flux(self):
        # With weights, the term must remove exactly its published
        # ``_conv_scav_flux`` from the column, and only list weighted
        # tracers there.
        state, diagnostics = self._setup(with_scav=True)
        term = ConvectiveTracerTransport(
            ("m_so4_acc",), csr_conv=(0.99,),
        )
        tend, diag_out = term(state, diagnostics, None, None)
        flux = diag_out["_conv_scav_flux"]["m_so4_acc"]
        self.assertGreater(float(flux[0]), 0.0)
        net = float(np.sum(np.asarray(tend.tracers["m_so4_acc"])) * 400.0)
        self.assertLessEqual(abs(net + float(flux[0])), 1e-6 * float(flux[0]))

    def test_scav_flux_published_without_convection_too(self):
        # The diagnostics dict is a lax.scan carry: the key set must be
        # identical whether or not convection ran this step, or the
        # structural probe's carry mismatches the stepped one (the
        # aerocom end-to-end repro: 73- vs 74-child carry TypeError).
        state, diagnostics = self._setup(with_conv=False)
        term = ConvectiveTracerTransport(("m_so4_acc",), csr_conv=(0.99,))
        _, diag_out = term(state, diagnostics, None, None)
        np.testing.assert_array_equal(
            np.asarray(diag_out["_conv_scav_flux"]["m_so4_acc"]), 0.0,
        )

    def test_unweighted_tracer_not_in_scav_flux(self):
        state, diagnostics = self._setup(with_scav=True)
        state = state.copy(tracers={
            **state.tracers, "so2": jnp.full((10, 1), 1.0e-9),
        })
        term = ConvectiveTracerTransport(
            ("m_so4_acc", "so2"), csr_conv=(0.99, 0.0),
        )
        _, diag_out = term(state, diagnostics, None, None)
        self.assertIn("m_so4_acc", diag_out["_conv_scav_flux"])
        self.assertNotIn("so2", diag_out["_conv_scav_flux"])

    def test_grad_through_transport_scale(self):
        state, diagnostics = self._setup(with_downdraft=True)

        def loss(scale):
            term = ConvectiveTracerTransport(
                ("m_so4_acc",),
                params=ConvTransportParameters(
                    transport_scale=scale, csr_conv=jnp.zeros(1),
                ),
            )
            tend, _ = term(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_so4_acc"] ** 2)

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)

    def test_grad_through_csr_conv(self):
        state, diagnostics = self._setup(with_scav=True)

        def loss(csr):
            term = ConvectiveTracerTransport(
                ("m_so4_acc",),
                params=ConvTransportParameters(
                    transport_scale=jnp.asarray(1.0), csr_conv=csr,
                ),
                csr_conv=(0.99,),
            )
            tend, _ = term(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_so4_acc"] ** 2)

        g = jax.grad(loss)(jnp.asarray([0.99]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))
        self.assertNotEqual(float(g[0]), 0.0)

    def test_default_params_take_the_given_fractions(self):
        term = ConvectiveTracerTransport(("a", "b"), csr_conv=(0.6, 0.2))
        np.testing.assert_allclose(
            np.asarray(term.params.get_value().csr_conv), [0.6, 0.2])

    def test_misshapen_param_fractions_rejected(self):
        with self.assertRaises(ValueError):
            ConvectiveTracerTransport(
                ("a", "b"), params=ConvTransportParameters.default((0.5,)))

    def test_empty_tracer_list_rejected(self):
        with self.assertRaises(ValueError):
            ConvectiveTracerTransport(())

    def test_misaligned_csr_conv_rejected(self):
        with self.assertRaises(ValueError):
            ConvectiveTracerTransport(("a", "b"), csr_conv=(1.0,))


class ComposedColumnScavengingTest(unittest.TestCase):
    """The convective aerosol sink through the FULL composed ECHAM column.

    Every unit test above hands the transport term a synthetic
    ``ConvectionData``, so all of them stayed green while the composed
    column produced no plume at all and ``_conv_scav_flux`` was exactly
    zero for every tracer (#773). This runs the same stack the release
    validation does — Tiedtke + convective tracer transport + JAM wet
    deposition — and asserts the STATE it is supposed to produce.

    One prescribed day at 47 levels. ~34 s against a warm JAX compilation
    cache but ~203 s cold, and CI's cache is job-local (run_test.yaml), so
    by the cost CI actually pays this is a slow test.
    """

    @pytest.mark.slow
    def test_soluble_tracer_is_scavenged_out_of_the_convective_column(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.rce import (
            JAM_COLUMN_FT_WINDOW,
            convergent_initial_physics_data,
            jam_scavenging_column,
        )
        from jcm.single_column_model import SingleColumnModel

        nlev, dt, nsteps = 47, 900.0, 96          # one day
        vertical = get_echam_levels(nlev)
        physics = echam_physics(cloud_scheme="2m", aerosol_module="jam",
                                radiation_scheme="grey")
        scm = SingleColumnModel(physics=physics, vertical=vertical,
                                lat_deg=0.0, lon_deg=150.0, dt_seconds=dt)
        # The SAME prescribed column and seeds the release-validation check
        # runs, so the guard cannot drift away from what it guards.
        state, seed, p = jam_scavenging_column(vertical, physics)
        states = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (nsteps,) + jnp.shape(x)), state,
        )

        preds = scm.run(states, initial_tracers=seed,
                        initial_physics_data=convergent_initial_physics_data(
                            scm, state),
                        times=jnp.arange(nsteps) * dt / 86400.0)

        conv = preds.physics_data["convection"]
        self.assertGreater(float(jnp.max(conv.mass_flux_up)), 0.0,
                           "no convection in the composed column")
        self.assertGreater(
            float(jnp.max(conv.qc_conv + conv.qi_conv)), 0.0,
            "convection ran but published no in-plume condensate",
        )
        self.assertGreater(float(jnp.max(conv.precip_flux)), 0.0)

        scav = preds.physics_data["_conv_scav_flux"]["m_so4_acc"]
        self.assertGreater(float(jnp.sum(scav)), 0.0,
                           "in-plume scavenging removed nothing")

        # State assertion, not a ledger check: with equal seeds the soluble
        # tracer must end up far less abundant aloft than the insoluble one.
        ft_lo, ft_hi = JAM_COLUMN_FT_WINDOW
        ft = (p > ft_lo) & (p < ft_hi)
        so4 = np.asarray(preds.tracer_states["m_so4_acc"])[-1][ft].mean()
        pom = np.asarray(preds.tracer_states["m_poa_pcm"])[-1][ft].mean()
        self.assertGreater(pom, 1e-20, "nothing was lofted at all")
        self.assertLess(so4, 0.5 * pom,
                        f"soluble {so4:.2e} not depleted vs insoluble {pom:.2e}")
        # ...but some of it IS lofted: the 1 − csr_conv share outside the condensate
        # rides the plume to the free troposphere, so relative to its own
        # boundary-layer loading the soluble tracer aloft sits near the
        # percent level (~0.1 after a day here), not at the ~1e-5 a plume
        # that re-activates its interstitial tail at every level leaves.
        bl = p > 850.0e2
        so4_bl = np.asarray(preds.tracer_states["m_so4_acc"])[-1][bl].mean()
        self.assertGreater(so4, 1.0e-3 * so4_bl,
                           f"soluble FT {so4:.2e} vs its BL {so4_bl:.2e}: "
                           "the interstitial share is being scavenged")

if __name__ == "__main__":
    unittest.main()
