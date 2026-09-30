"""Tests for the cloud-scheme input helper (``cloud_scheme_inputs``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.clouds.cloud_inputs import (
    CONVECTIVE_DETRAINMENT_KEY,
    cloud_scheme_inputs,
)
from jcm.physics_interface import POST_PHYSICS_STATE_KEY, PhysicsState

NLEV, NCOLS = 6, 4
DT = 900.0
TRACERS = ("qc", "qi", "qnc", "qni")


def _field(seed, scale, offset=0.0, shape=(NLEV, NCOLS)):
    return offset + scale * jax.random.uniform(jax.random.PRNGKey(seed), shape)


def _state(shape=(NLEV, NCOLS)):
    z = jnp.zeros(shape)
    return PhysicsState(
        u_wind=z, v_wind=z,
        temperature=_field(1, 40.0, 230.0, shape),
        specific_humidity=_field(2, 1e-2, 0.0, shape),
        geopotential=z,
        normalized_surface_pressure=jnp.ones(shape[1:]),
        tracers={
            "qc": _field(3, 1e-4, 0.0, shape),
            "qi": _field(4, 1e-5, 0.0, shape),
            "qnc": _field(5, 1e8, 0.0, shape),
            "qni": _field(6, 1e5, 0.0, shape),
        },
    )


def _run(shape=(NLEV, NCOLS)):
    return {
        "u_wind": jnp.zeros(shape), "v_wind": jnp.zeros(shape),
        "temperature": _field(11, 2e-4, -1e-4, shape),
        "specific_humidity": _field(12, 2e-8, -1e-8, shape),
        "tracers": {
            "qc": _field(13, 2e-9, 0.0, shape),
            "qi": _field(14, 2e-10, 0.0, shape),
            "qnc": jnp.zeros(shape),
            "qni": jnp.zeros(shape),
        },
    }


def _detrainment(shape=(NLEV, NCOLS)):
    return {"qc": _field(21, 1e-9, 0.0, shape), "qi": _field(22, 1e-10, 0.0, shape)}


def _carry(valid, shape=(NLEV, NCOLS)):
    state = _state(shape)
    return {
        "temperature": state.temperature - _field(31, 1.0, -0.5, shape),
        "specific_humidity": state.specific_humidity - _field(32, 1e-4, -5e-5, shape),
        "tracers": {
            "qc": state.tracers["qc"] - _field(33, 1e-5, -5e-6, shape),
            "qi": state.tracers["qi"] - _field(34, 1e-6, -5e-7, shape),
            "qnc": state.tracers["qnc"] - _field(35, 1e6, -5e5, shape),
            "qni": state.tracers["qni"] - _field(36, 1e3, -5e2, shape),
        },
        "valid": jnp.asarray(valid, jnp.float32),
    }


def _diagnostics(carry=None, detrainment=True, run=True):
    diagnostics = {"_dt_seconds": DT}
    if run:
        diagnostics["_tendency_run"] = _run()
    if detrainment:
        diagnostics[CONVECTIVE_DETRAINMENT_KEY] = _detrainment()
    if carry is not None:
        diagnostics[POST_PHYSICS_STATE_KEY] = carry
    return diagnostics


def _all_fields(fields):
    out = {"temperature": fields.temperature,
           "specific_humidity": fields.specific_humidity}
    out.update(fields.tracers)
    return out


def _assert_sum_identity(inputs):
    """Assert provisional == anchor + (increment + detrained), bit for bit."""
    anchor = _all_fields(inputs.anchor)
    increment = _all_fields(inputs.increment)
    provisional = _all_fields(inputs.provisional)
    detrained = {"qc": inputs.detrained_qc, "qi": inputs.detrained_qi}
    for name in anchor:
        if name in detrained:
            expected = anchor[name] + (increment[name] + detrained[name])
        else:
            expected = anchor[name] + increment[name]
        np.testing.assert_array_equal(provisional[name], expected, err_msg=name)


class TestFirstStage:
    """No valid carry: anchor = received state, increments = dt·run − detrainment."""

    @pytest.mark.parametrize("carry", [None, "invalid"])
    def test_anchor_is_received_state_and_dynamics_is_zero(self, carry):
        state = _state()
        slot = _carry(0.0) if carry == "invalid" else None
        inputs = cloud_scheme_inputs(state, _diagnostics(slot), TRACERS)
        assert float(inputs.dynamics_valid) == 0.0
        np.testing.assert_array_equal(inputs.anchor.temperature, state.temperature)
        np.testing.assert_array_equal(inputs.anchor.specific_humidity,
                                      state.specific_humidity)
        for name in TRACERS:
            np.testing.assert_array_equal(inputs.anchor.tracers[name],
                                          state.tracers[name])
        run = _run()
        detr = _detrainment()
        np.testing.assert_array_equal(inputs.increment.temperature,
                                      0.0 + DT * run["temperature"])
        np.testing.assert_array_equal(inputs.increment.specific_humidity,
                                      0.0 + DT * run["specific_humidity"])
        for name in ("qc", "qi"):
            np.testing.assert_array_equal(
                inputs.increment.tracers[name],
                0.0 + (DT * run["tracers"][name] - DT * detr[name]))
        np.testing.assert_array_equal(inputs.detrained_qc, DT * detr["qc"])
        np.testing.assert_array_equal(inputs.detrained_qi, DT * detr["qi"])
        _assert_sum_identity(inputs)

    def test_invalid_carry_values_never_reach_the_result(self):
        """A garbage (even non-finite) slot under valid = 0 changes nothing."""
        state = _state()
        clean = cloud_scheme_inputs(state, _diagnostics(_carry(0.0)), TRACERS)
        garbage = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), _carry(0.0))
        garbage["valid"] = jnp.asarray(0.0, jnp.float32)
        dirty = cloud_scheme_inputs(state, _diagnostics(garbage), TRACERS)
        for a, b in zip(jax.tree.leaves(clean), jax.tree.leaves(dirty)):
            np.testing.assert_array_equal(a, b)

    def test_provisional_is_received_plus_upstream(self):
        state = _state()
        inputs = cloud_scheme_inputs(state, _diagnostics(), TRACERS)
        run = _run()
        np.testing.assert_allclose(inputs.provisional.temperature,
                                   state.temperature + DT * run["temperature"],
                                   rtol=1e-6)
        np.testing.assert_allclose(
            inputs.provisional.tracers["qc"],
            state.tracers["qc"] + DT * run["tracers"]["qc"], rtol=1e-5, atol=1e-12)

    def test_no_run_no_detrainment_is_the_state_itself(self):
        state = _state()
        inputs = cloud_scheme_inputs(
            state, _diagnostics(detrainment=False, run=False), TRACERS)
        for a, b in zip(jax.tree.leaves(inputs.provisional),
                        jax.tree.leaves((state.temperature, state.specific_humidity,
                                         {k: state.tracers[k] for k in TRACERS}))):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(inputs.detrained_qc, 0.0)
        np.testing.assert_array_equal(inputs.detrained_qi, 0.0)

    def test_missing_tracer_reads_as_zero(self):
        state = _state()
        state = state.copy(tracers={"qc": state.tracers["qc"]})
        inputs = cloud_scheme_inputs(
            state, _diagnostics(detrainment=False, run=False), ("qc", "qi"))
        np.testing.assert_array_equal(inputs.anchor.tracers["qi"], 0.0)
        np.testing.assert_array_equal(inputs.provisional.tracers["qi"], 0.0)


class TestSecondStage:
    """Valid carry: anchor = carried post-physics state, dynamics in the increment."""

    def test_anchor_is_carry_and_increment_holds_dynamics(self):
        state = _state()
        carry = _carry(1.0)
        inputs = cloud_scheme_inputs(state, _diagnostics(carry), TRACERS)
        assert float(inputs.dynamics_valid) == 1.0
        np.testing.assert_array_equal(inputs.anchor.temperature,
                                      carry["temperature"])
        for name in TRACERS:
            np.testing.assert_array_equal(inputs.anchor.tracers[name],
                                          carry["tracers"][name])
        run = _run()
        detr = _detrainment()
        np.testing.assert_array_equal(
            inputs.increment.temperature,
            (state.temperature - carry["temperature"]) + DT * run["temperature"])
        np.testing.assert_array_equal(
            inputs.increment.tracers["qc"],
            (state.tracers["qc"] - carry["tracers"]["qc"])
            + (DT * run["tracers"]["qc"] - DT * detr["qc"]))
        np.testing.assert_array_equal(
            inputs.increment.tracers["qnc"],
            (state.tracers["qnc"] - carry["tracers"]["qnc"]) + 0.0)
        _assert_sum_identity(inputs)

    def test_provisional_state_does_not_depend_on_the_stage(self):
        """Only the split into anchor and increment moves; the sum stays x_n + dt·run."""
        state = _state()
        first = cloud_scheme_inputs(state, _diagnostics(), TRACERS)
        second = cloud_scheme_inputs(state, _diagnostics(_carry(1.0)), TRACERS)
        for a, b in zip(jax.tree.leaves(first.provisional),
                        jax.tree.leaves(second.provisional)):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-12)

    def test_detrainment_split_is_exact_against_the_running_tendency(self):
        """The increment plus the detrainment is what convection put into the run.

        To the rounding of one subtraction and one addition: the detrained
        part is taken out of and put back into the same array.
        """
        state = _state()
        inputs = cloud_scheme_inputs(state, _diagnostics(), TRACERS)
        run = _run()
        eps = np.finfo(np.float32).eps
        for name, detr in (("qc", inputs.detrained_qc), ("qi", inputs.detrained_qi)):
            upstream = DT * run["tracers"][name]
            np.testing.assert_allclose(
                inputs.increment.tracers[name] + detr, upstream,
                rtol=0.0, atol=4 * eps * float(jnp.max(jnp.abs(upstream))))


def test_block_matches_single_columns():
    """The helper is broadcasting-native: a (nlev, ncols) block == each column."""
    state = _state()
    diagnostics = _diagnostics(_carry(1.0))
    block = cloud_scheme_inputs(state, diagnostics, TRACERS)
    for col in range(NCOLS):
        col_state = jax.tree.map(
            lambda x: x[..., col] if getattr(x, "ndim", 0) >= 1 else x, state)
        col_diag = jax.tree.map(
            lambda x: x[..., col] if getattr(x, "ndim", 0) >= 1 else x, diagnostics)
        single = cloud_scheme_inputs(col_state, col_diag, TRACERS)
        for a, b in zip(jax.tree.leaves(single), jax.tree.leaves(block)):
            if getattr(b, "ndim", 0) >= 1:
                np.testing.assert_array_equal(a, b[..., col])


@pytest.mark.parametrize("valid", [0.0, 1.0])
def test_gradients_are_finite_on_both_stages(valid):
    """No NaN in the gradient, whether or not the carry is used.

    On the fallback path the carry receives an exactly-zero gradient: the
    anchor is the received state there, not a masked mix.
    """
    state = _state()
    carry = _carry(valid)

    def loss(state, carry):
        inputs = cloud_scheme_inputs(state, _diagnostics(carry), TRACERS)
        # anchor + increment would telescope to x_n and hide the carry.
        return sum(jnp.sum(x ** 2) for x in jax.tree.leaves(
            (inputs.anchor, inputs.increment, inputs.provisional)))

    g_state, g_carry = jax.grad(loss, argnums=(0, 1))(state, carry)
    for leaf in jax.tree.leaves((g_state, g_carry)):
        assert np.all(np.isfinite(leaf))
    if valid == 0.0:
        for name in ("temperature", "specific_humidity"):
            np.testing.assert_array_equal(g_carry[name], 0.0)
    else:
        assert float(jnp.max(jnp.abs(g_carry["temperature"]))) > 0.0
