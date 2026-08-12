"""Tests for the convective bulk-plume tracer transport (#602 item 2)."""

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
        dq = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        dm = rho * dz
        net = float(jnp.sum(dq * dm[jnp.newaxis]))
        gross = float(jnp.sum(jnp.abs(dq) * dm[jnp.newaxis]))
        self.assertGreater(gross, 0.0, "plume did nothing — fixture off")
        self.assertLessEqual(abs(net), 1e-6 * gross)

    def test_lofts_boundary_layer_tracer(self):
        # A surface-concentrated tracer entrained at cloud base must be
        # deposited at the detrainment levels aloft and reduced below.
        rho, dz = self._grid()
        mfu, entrain = _plume(base=8, top=3)
        q = jnp.zeros((1, 10, 1)).at[0, 8:].set(1.0e-9)
        dq = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
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
        dq = convective_tracer_tendency(q0, mfu, entrain, rho, dz, 1800.0)
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
        dq = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        dm = rho * dz
        net = float(jnp.sum(dq * dm[jnp.newaxis]))
        gross = float(jnp.sum(jnp.abs(dq) * dm[jnp.newaxis]))
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
        dq = convective_tracer_tendency(q, mfu, entrain, rho, dz, dt)
        q1 = q + dt * dq
        self.assertGreaterEqual(float(q1.min()), -1e-9)
        # Bounded-in-[0,1] tracer cannot exceed 1 either.
        q = jnp.ones((1, nlev, 1))
        dq = convective_tracer_tendency(q, mfu, entrain, rho, dz, dt)
        q1 = q + dt * dq
        self.assertLessEqual(float(q1.max()), 1.0 + 1e-9)
        self.assertGreaterEqual(float(q1.min()), -1e-9)

    def test_conserves_with_distinct_tracers_and_columns(self):
        # K=2 tracers x 2 distinct columns: a transposed (K, ncols) carry
        # in the plume scan mixes them and breaks the per-tracer,
        # per-column budgets.
        nlev = 10
        rho = jnp.linspace(0.4, 1.2, nlev)[:, None] * jnp.ones((1, 2))
        dz = jnp.full((nlev, 2), 400.0)
        mfu, entrain = _plume(ncols=2)
        mfu = mfu * jnp.asarray([1.0, 0.5])[None, :]
        entrain = entrain * jnp.asarray([1.0, 0.5])[None, :]
        q = jnp.stack([
            jnp.zeros((nlev, 2)).at[8:].set(1.0e-9),
            jnp.linspace(1.0, 0.2, nlev)[:, None] * jnp.ones((1, 2)) * 1e-8,
        ])
        dq = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
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
        # a tracer negative.
        rho, dz = self._grid()
        mfu, entrain = _plume(mf=50.0)
        q = jnp.zeros((1, 10, 1)).at[0, 8:].set(1.0e-9)
        dq = convective_tracer_tendency(q, mfu, entrain, rho, dz, 1800.0)
        q1 = q + 1800.0 * dq
        self.assertGreaterEqual(float(q1.min()), -1e-25)
        self.assertTrue(bool(jnp.all(jnp.isfinite(dq))))

    def test_no_mass_flux_no_tendency(self):
        rho, dz = self._grid()
        q = jnp.ones((1, 10, 1)) * 1e-9
        dq = convective_tracer_tendency(
            q, jnp.zeros((10, 1)), jnp.zeros((10, 1)), rho, dz, 1800.0,
        )
        np.testing.assert_array_equal(np.asarray(dq), 0.0)


class _Conv:
    def __init__(self, mfu, entrain):
        self.mass_flux_up = mfu
        self.entrain_up = entrain


class ConvectiveTracerTransportTermTest(unittest.TestCase):
    def _setup(self, with_conv=True, nlev=10, ncols=1):
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
            diagnostics["convection"] = _Conv(mfu, entrain)
        return state, diagnostics

    def test_transports_and_conserves(self):
        state, diagnostics = self._setup()
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

    def test_grad_through_transport_scale(self):
        state, diagnostics = self._setup()

        def loss(scale):
            term = ConvectiveTracerTransport(
                ("m_so4_acc",),
                params=ConvTransportParameters(transport_scale=scale),
            )
            tend, _ = term(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_so4_acc"] ** 2)

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)

    def test_empty_tracer_list_rejected(self):
        with self.assertRaises(ValueError):
            ConvectiveTracerTransport(())


if __name__ == "__main__":
    unittest.main()
