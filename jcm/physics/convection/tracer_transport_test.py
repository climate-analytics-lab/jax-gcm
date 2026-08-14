"""Tests for the convective bulk-plume tracer transport (#602, #621, #622)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.convection.tracer_transport import (
    ConvTransportParameters,
    ConvectiveTracerTransport,
    convective_tracer_tendency,
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
    """Synthetic downdraft mirroring the Tiedtke scan's conventions.

    ``mfd[k]`` (≤ 0) is the flux leaving layer k through its BOTTOM
    interface: the LFS seed at ``lfs``, constant through the bulk, halved
    in the second-to-last layer and zero in the last (the cuddraf
    ``itopde`` taper). ``entrain_down`` is the cuddraf turbulent ledger
    ``entrdd·|mfd_in|·dz`` in the bulk, zero in the taper.
    """
    lev = jnp.arange(nlev)[:, None]
    mfd = jnp.where((lev >= lfs) & (lev < nlev - 2), -mf, 0.0)
    mfd = mfd.at[nlev - 2].set(-0.5 * mf)
    mfd = jnp.broadcast_to(mfd, (nlev, ncols))
    mfd_in = jnp.concatenate([jnp.zeros((1, ncols)), mfd[:-1]], axis=0)
    e_dn = jnp.where(
        (mfd < 0) & (lev < nlev - 2), entrdd * jnp.abs(mfd_in) * dz, 0.0
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
        # A tracer confined to the LFS layer must appear in the two
        # sub-cloud taper layers (where the descent detrains) and be
        # reduced at the LFS (where the seed mass is entrained).
        rho, dz = self._grid()
        mfd, e_dn = _downdraft(lfs=4)
        q = jnp.zeros((1, 10, 1)).at[0, 4].set(1.0e-9)
        dq, _ = convective_tracer_tendency(
            q, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
            mfd=mfd, entrain_down=e_dn,
        )
        self.assertLess(float(dq[0, 4, 0]), 0.0)
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
        e_dn = jnp.where((lev > 3) & (lev <= 5), 2.0e-4 * 0.02 * 400.0, 0.0)
        q = jnp.stack([jnp.linspace(0.5, 1.5, 10)[:, None] * 1e-9])
        dq, _ = convective_tracer_tendency(
            q, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
            mfd=mfd, entrain_down=e_dn,
        )
        net, gross = _column_budgets(dq, rho * dz)
        self.assertGreater(gross, 0.0)
        self.assertLessEqual(abs(net), 1e-6 * gross)


class ScavengingTest(unittest.TestCase):
    """CAM aero_convproc-style in-plume removal (jax-gcm#621)."""

    def _setup(self, nlev=10, ncols=1):
        rho = jnp.linspace(0.4, 1.2, nlev)[:, None] * jnp.ones((1, ncols))
        dz = jnp.full((nlev, ncols), 400.0)
        mfu, entrain = _plume(nlev=nlev, ncols=ncols)
        # Condensate and precip formation inside the cloudy layers only.
        lev = jnp.arange(nlev)[:, None]
        cloudy = (lev >= 4) & (lev <= 7)
        cond = jnp.where(cloudy, 5.0e-4, 0.0) * jnp.ones((1, ncols))
        pf = jnp.where(cloudy, 1.0e-5, 0.0) * jnp.ones((1, ncols))
        q = jnp.stack([
            jnp.zeros((nlev, ncols)).at[8:].set(1.0e-9),
            jnp.zeros((nlev, ncols)).at[8:].set(1.0e-9),
        ])
        return q, mfu, entrain, rho, dz, cond, pf

    def test_budget_closes_to_scavenged_flux(self):
        # Column change must equal MINUS the scavenged surface flux,
        # per tracer, exactly.
        q, mfu, entrain, rho, dz, cond, pf = self._setup()
        dq, scav = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            scav_weights=jnp.asarray([1.0, 0.5]),
            precip_formation=pf, plume_condensate=cond,
        )
        dm = rho * dz
        for k in range(2):
            net = float(jnp.sum(dq[k] * dm))
            self.assertGreater(float(scav[k, 0]), 0.0)
            self.assertLessEqual(
                abs(net + float(scav[k, 0])),
                1e-6 * float(jnp.sum(jnp.abs(dq[k]) * dm)),
            )

    def test_scavenging_thins_what_detrains_aloft(self):
        # With removal inside the ascent, less tracer survives to the
        # detrainment layers than in the conservative plume.
        q, mfu, entrain, rho, dz, cond, pf = self._setup()
        dq0, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        dq1, _ = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            scav_weights=jnp.asarray([1.0, 1.0]),
            precip_formation=pf, plume_condensate=cond,
        )
        self.assertLess(float(dq1[0, 2, 0]), float(dq0[0, 2, 0]))

    def test_zero_weights_recover_conservative_plume(self):
        q, mfu, entrain, rho, dz, cond, pf = self._setup()
        dq0, _ = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        dq1, scav = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            scav_weights=jnp.zeros(2),
            precip_formation=pf, plume_condensate=cond,
        )
        np.testing.assert_allclose(np.asarray(dq1), np.asarray(dq0))
        np.testing.assert_array_equal(np.asarray(scav), 0.0)

    def test_dry_plume_scavenges_nothing(self):
        # No condensate (below CAM's clw_cut) -> gate closed everywhere.
        q, mfu, entrain, rho, dz, _, pf = self._setup()
        _, scav = convective_tracer_tendency(
            q, mfu, entrain, rho, dz, 1800.0,
            scav_weights=jnp.ones(2),
            precip_formation=pf,
            plume_condensate=jnp.full(mfu.shape, 1.0e-7),
        )
        np.testing.assert_array_equal(np.asarray(scav), 0.0)


class _Conv:
    def __init__(self, mfu, entrain, mfd=None, e_dn=None, pf=None, cond=None):
        zeros = jnp.zeros_like(mfu)
        self.mass_flux_up = mfu
        self.entrain_up = entrain
        self.mass_flux_down = mfd if mfd is not None else zeros
        self.entrain_down = e_dn if e_dn is not None else zeros
        self.precip_formation = pf if pf is not None else zeros
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
                kwargs["pf"] = jnp.where(cloudy, 1.0e-5, 0.0) * jnp.ones(
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
            ("m_so4_acc",), scav_weights=(1.0,),
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
        term = ConvectiveTracerTransport(("m_so4_acc",), scav_weights=(1.0,))
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
            ("m_so4_acc", "so2"), scav_weights=(1.0, 0.0),
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
                    transport_scale=scale, scav_ratio=jnp.asarray(0.99),
                ),
            )
            tend, _ = term(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_so4_acc"] ** 2)

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)

    def test_grad_through_scav_ratio(self):
        state, diagnostics = self._setup(with_scav=True)

        def loss(ratio):
            term = ConvectiveTracerTransport(
                ("m_so4_acc",),
                params=ConvTransportParameters(
                    transport_scale=jnp.asarray(1.0), scav_ratio=ratio,
                ),
                scav_weights=(1.0,),
            )
            tend, _ = term(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_so4_acc"] ** 2)

        g = jax.grad(loss)(jnp.asarray(0.99))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)

    def test_empty_tracer_list_rejected(self):
        with self.assertRaises(ValueError):
            ConvectiveTracerTransport(())

    def test_misaligned_scav_weights_rejected(self):
        with self.assertRaises(ValueError):
            ConvectiveTracerTransport(("a", "b"), scav_weights=(1.0,))


if __name__ == "__main__":
    unittest.main()
