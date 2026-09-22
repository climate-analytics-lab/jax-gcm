"""Regression coverage for the v3 Gregorian clock contract."""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import pytest

from jcm.date import (
    DateData,
    fraction_of_year_elapsed,
    gregorian_ymd_from_days,
    parse_duration_seconds,
)
from jcm.dycore.base import Predictions
from jcm.model import Model, RunState, _op_split_trajectory


def _ymd(value):
    date = jdt.to_datetime(value)
    return tuple(int(x) for x in gregorian_ymd_from_days(date.delta.days))


def test_gregorian_century_leap_boundaries():
    """1900/2100 are common years; 2000 remains a Gregorian leap year."""
    one_day = jdt.Timedelta(days=jnp.int32(1))

    feb_28_1900 = jdt.to_datetime("1900-02-28")
    feb_28_2000 = jdt.to_datetime("2000-02-28")
    feb_28_2100 = jdt.to_datetime("2100-02-28")

    assert tuple(int(x) for x in gregorian_ymd_from_days(
        (feb_28_1900 + one_day).delta.days)) == (1900, 3, 1)
    feb_29_2000 = feb_28_2000 + one_day
    assert tuple(int(x) for x in gregorian_ymd_from_days(
        feb_29_2000.delta.days)) == (2000, 2, 29)
    assert tuple(int(x) for x in gregorian_ymd_from_days(
        (feb_29_2000 + one_day).delta.days)) == (2000, 3, 1)
    assert tuple(int(x) for x in gregorian_ymd_from_days(
        (feb_28_2100 + one_day).delta.days)) == (2100, 3, 1)

    assert jnp.isclose(fraction_of_year_elapsed(
        jdt.to_datetime("1900-03-01")), 59 / 365, rtol=0, atol=1e-7)
    assert jnp.isclose(fraction_of_year_elapsed(
        jdt.to_datetime("2000-03-01")), 60 / 366, rtol=0, atol=1e-7)
    assert jnp.isclose(fraction_of_year_elapsed(
        jdt.to_datetime("2100-03-01")), 59 / 365, rtol=0, atol=1e-7)


def test_january_first_phase_is_zero_across_decades_issue_449():
    # The former fixed-365 epoch phase drifted with distance from 1970.
    dates = jdt.Datetime(jdt.Timedelta(
        days=jnp.stack([
            jdt.to_datetime(f"{year}-01-01").delta.days
            for year in (1900, 1950, 1970, 2000, 2026, 2100)
        ]),
        seconds=jnp.zeros(6, dtype=jnp.int32),
    ))
    phase = jax.jit(jax.vmap(fraction_of_year_elapsed))(dates)
    assert jnp.array_equal(phase, jnp.zeros_like(phase))


def test_fixed_day_is_exactly_86400_si_seconds():
    assert parse_duration_seconds("1 day") == 86_400
    assert parse_duration_seconds("24 hours") == 86_400
    assert parse_duration_seconds("1440 minutes") == 86_400
    assert parse_duration_seconds("86400 seconds") == 86_400


@pytest.mark.parametrize(("year", "expected_seconds"), [
    (1900, 86_400),
    (2000, 172_800),
    (2100, 86_400),
])
def test_end_time_resolves_gregorian_century_boundaries(
        year, expected_seconds):
    duration = Model._resolve_run_duration(
        None, f"{year}-03-01", jdt.to_datetime(f"{year}-02-28"))
    assert parse_duration_seconds(duration) == expected_seconds


def test_end_time_resolves_nonmidnight_exact_seconds():
    duration = Model._resolve_run_duration(
        None,
        "2000-03-01 01:02:03",
        jdt.to_datetime("2000-02-28 23:59:58"),
    )
    assert parse_duration_seconds(duration) == 90_125


def _trajectory(start_time, start_step, outer_steps):
    def step(state, physics, date):
        # The value depends on the authoritative step, making a reset at the
        # seam visible even when the dynamics scalar itself is continuous.
        return state + date.model_step.astype(state.dtype), physics

    def post_process(state, physics):
        return Predictions(state, physics, None)

    return _op_split_trajectory(
        step_fn=step,
        initial_physics_state={},
        empty_diagnostics={},
        outer_steps=outer_steps,
        inner_steps=2,
        post_process_fn=post_process,
        initial_time=start_time,
        initial_step=jnp.int32(start_step),
        dt_seconds=45,
    )


def test_nonmidnight_restart_seam_matches_uninterrupted_exact_clock():
    start = jdt.to_datetime("1999-12-31 23:59:15")
    whole = _trajectory(start, 0, outer_steps=2)(jnp.asarray(0.0))

    first = _trajectory(start, 0, outer_steps=1)(jnp.asarray(0.0))
    resumed = RunState(
        dynamics=first[0], physics=first[1], time=first[2], step=first[3])
    second = _trajectory(
        resumed.time, int(resumed.step), outer_steps=1)(resumed.dynamics)

    assert float(second[0]) == float(whole[0])
    assert int(second[3]) == int(whole[3]) == 4
    assert int(second[2].delta.days) == int(whole[2].delta.days)
    assert int(second[2].delta.seconds) == int(whole[2].delta.seconds)
    assert second[7].delta.seconds.tolist() == whole[7].delta.seconds[1:].tolist()


def test_one_jit_handles_dates_from_different_decades():
    traces = []

    @jax.jit
    def phase(date):
        traces.append(None)
        return DateData(date, jnp.int32(0), 1800).tyear()

    assert float(phase(jdt.to_datetime("1980-07-01"))) > 0.49
    assert float(phase(jdt.to_datetime("2080-07-01"))) > 0.49
    assert len(traces) == 1
