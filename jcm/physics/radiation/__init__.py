"""Radiation parameterisations and shared helpers.

The three radiation terms (:class:`~grey_two_stream.GreyTwoStreamRadiation`,
:class:`~rrtmgp.RRTMGPRadiation`, :class:`~nn_emulator_scheme.NNEmulatorRadiation`)
each gate their compute on a configurable sub-stepping interval —
``parameters.radiation.radiation_interval`` — and re-emit the previous
step's cached heating rates from ``diagnostics["radiation"]`` on
non-radiation steps. The gate and the cache-replay tendency are both
exposed here so the three terms share a single source of truth.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.radiation.radiation_types import (
    RadiationData,
    RadiationParameters,
)
from jcm.physics_interface import PhysicsTendency


def radiation_should_compute(
    diagnostics: dict, parameters: RadiationParameters,
) -> jnp.ndarray:
    """Return a scalar bool: should we recompute radiation this step?

    If ``radiation_interval > 0``, recompute every
    ``round(interval / dt)`` steps; otherwise (the default) recompute
    every step.
    """
    date = diagnostics["_date"]
    dt = date.dt_seconds
    step = date.model_step
    interval = parameters.radiation_interval
    steps_per_call = jnp.where(
        interval > 0,
        jnp.int32(jnp.round(interval / dt)),
        jnp.int32(1),
    )
    return jnp.mod(step, steps_per_call) == 0


def cached_radiation_tendency(
    radiation: RadiationData, shape: tuple,
) -> PhysicsTendency:
    """Build the tendency that re-emits the cached SW + LW heating rates."""
    nlev, ncols = shape
    return PhysicsTendency(
        u_wind=jnp.zeros(shape),
        v_wind=jnp.zeros(shape),
        temperature=radiation.sw_heating_rate + radiation.lw_heating_rate,
        specific_humidity=jnp.zeros(shape),
        tracers={},
    )


__all__ = [
    "cached_radiation_tendency",
    "radiation_should_compute",
]
