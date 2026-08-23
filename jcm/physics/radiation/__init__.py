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
    every step. The step counter is the radiation term's own
    ``RadiationData.step`` carry slot — incremented each call by the
    radiation term — so this gate no longer depends on the model-wide
    date/step plumbing.
    """
    step = diagnostics["radiation"].step
    dt = diagnostics["_dt_seconds"]
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


# Shortwave leaves of ``RadiationData``. All scale linearly with the incoming
# solar beam, so one ratio rescales the lot. The ``noa_frac_*`` slots are
# deliberately absent: they are ratios of two fluxes and so are already
# zenith-independent. Longwave is absent because it does not see the sun.
_CACHED_SW_FIELDS = (
    "sw_flux_up", "sw_flux_down", "sw_heating_rate",
    "surface_sw_down", "surface_sw_up",
    "toa_sw_up", "toa_sw_down", "toa_sw_up_clear", "toa_sw_up_noa",
    "toa_sw_up_clear_noa",
)

# Below this cosine (~88 deg) the compute-step column is treated as dark and
# the ratio is held at zero rather than dividing by a near-zero denominator.
_MIN_COS_ZENITH_FOR_RESCALE = 1.0e-3


def current_cos_zenith(solar, longitude, latitude) -> jnp.ndarray:
    """Cosine of the solar zenith angle now, for ``longitude``/``latitude``.

    Pure trigonometry -- no radiative transfer -- so it is cheap enough to
    evaluate on cached steps, which is what makes the zenith rescaling in
    :func:`rescale_cached_radiation` affordable. Both angles are in degrees,
    matching the per-column values the radiation terms cache in
    ``cache_coords``.
    """
    from jax_solar import OrbitalTime, get_solar_sin_altitude

    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    # sin(altitude) == cos(zenith).
    return get_solar_sin_altitude(orbital_time, longitude, latitude)


def rescale_cached_radiation(
    radiation: RadiationData, cos_zenith_now: jnp.ndarray,
) -> RadiationData:
    """Rescale cached shortwave from the compute-step sun to the current sun.

    Radiation is solved every ``radiation_interval`` (7200 s by default) but
    applied every step, so with a 20-minute step the same shortwave is reused
    six times. Without a solar factor a column crossing the terminator gets
    either zero or full daylight for up to two hours — worst at the equinoxes
    and at high latitude, where the terminator sweeps fastest in local time.

    Every shortwave quantity is linear in the incoming beam at fixed
    atmospheric transmissivity, so multiplying them all by
    ``mu0_now / mu0_at_compute`` is equivalent to ECHAM psrad's scheme of
    caching transmissivity and rescaling by the instantaneous solar flux.
    Rescaling the whole set, rather than only the heating rate and
    ``surface_sw_down`` that feed back into the model, keeps the saved flux
    diagnostics consistent with the heating actually applied.

    The rescaled fluxes are written back into the carry, so the reference
    ``cos_zenith_for_fluxes`` MUST advance with them. Successive cached steps
    then telescope --
    ``(mu_1/mu_0)(mu_2/mu_1)...(mu_k/mu_{k-1}) = mu_k/mu_0`` -- which is the
    intended factor against the compute step. Holding the reference fixed at
    the compute-step value instead makes each step rescale an already-rescaled
    flux, so the ratio COMPOUNDS: eight cached steps through a sunrise turned
    100 W/m2 into 708,750 and NaN'd the model within a day.

    Longwave is untouched. So is a column whose stored shortwave is already
    zero -- either it was dark when radiation last ran, or it went dark during
    the interval. Multiplication cannot recover a flux that has been driven to
    zero, so such a column stays dark until the next compute step even if the
    sun comes back up within the interval. That residual error is bounded by
    the interval, and is the reason ``radiation_interval`` should not be
    pushed far beyond a couple of hours.
    """
    mu0_ref = radiation.cos_zenith_for_fluxes
    mu0_now = jnp.maximum(cos_zenith_now, 0.0)
    ratio = jnp.where(
        mu0_ref > _MIN_COS_ZENITH_FOR_RESCALE,
        mu0_now / jnp.maximum(mu0_ref, _MIN_COS_ZENITH_FOR_RESCALE),
        0.0,
    )
    scaled = {
        name: getattr(radiation, name) * ratio
        for name in _CACHED_SW_FIELDS
    }
    return radiation.copy(
        cos_zenith=cos_zenith_now,
        cos_zenith_for_fluxes=mu0_now,
        **scaled,
    )


__all__ = [
    "cached_radiation_tendency",
    "current_cos_zenith",
    "radiation_should_compute",
    "rescale_cached_radiation",
]
