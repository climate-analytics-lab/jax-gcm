"""Focused tests for the exact core run clock."""

import jax.numpy as jnp
import jax_datetime as jdt

from jcm.dycore.base import Predictions
from jcm.model import _op_split_trajectory


def _post_process(state, physics):
    # Deliberately nonlinear: the interval mean must be mean(x**2), not
    # mean(x)**2.
    return Predictions(state**2, physics, None)


def test_nonmidnight_average_uses_exact_clock_bounds_and_physical_values():
    start = jdt.to_datetime("2000-02-28 23:59:30")

    def step(state, physics, date):
        del date
        return state + 1, physics

    integrate = _op_split_trajectory(
        step_fn=step,
        initial_physics_state={},
        empty_diagnostics={},
        outer_steps=1,
        inner_steps=2,
        post_process_fn=_post_process,
        output_averages=True,
        initial_time=start,
        initial_step=jnp.int32(7),
        dt_seconds=30,
    )
    (_, _, final_time, final_step, predictions, _, _, times,
     bounds) = integrate(jnp.asarray(0.0))

    assert float(predictions.dynamics[0]) == 2.5
    assert int(final_step) == 9
    assert int(final_time.delta.days) == int(jdt.to_datetime("2000-02-29").delta.days)
    assert int(final_time.delta.seconds) == 30
    assert int(times.delta.seconds[0]) == 0  # midpoint is leap-day midnight
    assert int(bounds.delta.seconds[0, 0]) == 86370
    assert int(bounds.delta.seconds[0, 1]) == 30


def test_snapshot_labels_are_exact_end_instants():
    start = jdt.to_datetime("2001-01-01 12:00:00")

    def step(state, physics, date):
        assert date.dt is not None
        return state + 1, physics

    integrate = _op_split_trajectory(
        step_fn=step,
        initial_physics_state={},
        empty_diagnostics={},
        outer_steps=2,
        inner_steps=1,
        post_process_fn=_post_process,
        initial_time=start,
        initial_step=jnp.int32(0),
        dt_seconds=3600,
    )
    *_, times, bounds = integrate(jnp.asarray(0.0))
    assert times.delta.seconds.tolist() == [13 * 3600, 14 * 3600]
    assert bounds.delta.seconds.tolist() == [
        [12 * 3600, 13 * 3600],
        [13 * 3600, 14 * 3600],
    ]


def test_averaging_keeps_last_categorical_diagnostic():
    def step(state, physics, date):
        del date
        return state + 1, {"counter": physics["counter"] + 1}

    integrate = _op_split_trajectory(
        step_fn=step,
        initial_physics_state={"counter": jnp.int32(4)},
        empty_diagnostics={"counter": jnp.int32(0)},
        outer_steps=1,
        inner_steps=3,
        post_process_fn=_post_process,
        output_averages=True,
        initial_time=jdt.to_datetime("2000-01-01"),
        initial_step=jnp.int32(0),
        dt_seconds=60,
    )
    *_, predictions, _, _, _, _ = integrate(jnp.asarray(0.0))
    assert predictions.physics["counter"].dtype == jnp.int32
    assert int(predictions.physics["counter"][0]) == 7
