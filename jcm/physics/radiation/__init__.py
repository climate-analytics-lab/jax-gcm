"""Radiation parameterisations and shared helpers.

The three radiation terms (:class:`~grey_two_stream.GreyTwoStreamRadiation`,
:class:`~rrtmgp.RRTMGPRadiation`, :class:`~nn_emulator_scheme.NNEmulatorRadiation`)
each gate their compute on a configurable sub-stepping interval —
``parameters.radiation.radiation_interval``. A solve stores the shortwave as
transmissivities (:func:`sw_solve_to_transmissivity`), and every step, solve
or not, the shortwave fluxes and heating are rebuilt from them and the
instantaneous sun (:func:`apply_insolation`), as ECHAM's ``radheat`` does with
``trsol``. The gate, the solve geometry and the replay tendency are exposed
here so the three terms share a single source of truth.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.radiation.radiation_types import (
    RadiationData,
    RadiationParameters,
)
from jcm.physics_interface import PhysicsTendency


#: Diagnostics key under which the boundary-condition term publishes the
#: surface optical properties of the CURRENT step (``albedo_vis``,
#: ``albedo_nir``, ``emissivity``, each ``(ncols,)``). A radiation term reads
#: them only when it solves, and publishes the values it solved with in
#: ``RadiationData.surface_albedo_*`` / ``surface_emissivity``. Between solves
#: those carry fields keep the last solve's values, so the published albedo,
#: the reflected flux and the heating rate always describe one solve — as in
#: ECHAM, whose radiation reads the surface albedo only at a radiation step
#: (``trigrad``) and replays the transmissivities in between (``radheat``).
#: The land tile's own ``land_albedo`` rides in the same dict and is held the
#: same way, on ``surface.land_albedo_at_solve`` (:func:`hold_land_albedo`), so
#: the land's net shortwave is the solve's.
#:
#: Step-local: ``ComposablePhysics`` removes it before the diagnostics become
#: the cross-step carry (``_STEP_LOCAL_KEYS``), so it is never checkpointed
#: and a radiation term only ever sees the value published earlier in the
#: same step.
SURFACE_OPTICS_KEY = "_surface_optics"


def surface_optics_for_solve(diagnostics: dict, ncols: int):
    """``(albedo_vis, albedo_nir, emissivity)`` a radiation solve should use.

    The current step's :data:`SURFACE_OPTICS_KEY` when a boundary-condition
    term published it, else the values already on the ``radiation`` carry
    (hosts that set the surface optics there directly, e.g. single-column
    tests).
    """
    optics = diagnostics.get(SURFACE_OPTICS_KEY)
    if optics is None:
        radiation = diagnostics["radiation"]
        return (radiation.surface_albedo_vis.reshape(ncols),
                radiation.surface_albedo_nir.reshape(ncols),
                radiation.surface_emissivity.reshape(ncols))
    return (optics["albedo_vis"].reshape(ncols),
            optics["albedo_nir"].reshape(ncols),
            optics["emissivity"].reshape(ncols))


def surface_optics_bands_for_solve(diagnostics: dict, ncols: int):
    """``(vis_dir, vis_dif, nir_dir, nir_dif)`` a per-band RRTMGP solve should use.

    Companion to :func:`surface_optics_for_solve` for the per-band
    direct/diffuse quartet (jax-gcm#959): ``EchamBoundaryConditions``
    publishes it on :data:`SURFACE_OPTICS_KEY` alongside the broadband pair
    (open water's Fresnel direct albedo is not its diffuse albedo; land and
    sea ice repeat the same value in both slots).

    Falls back to the BLENDED BROADBAND scalar (``0.46*vis + 0.54*nir``,
    the same TOA-spectrum split ``rrtmgp.py`` blends with) in all four
    slots whenever the current step's :data:`SURFACE_OPTICS_KEY` carries no
    quartet — a boundary-condition term that predates #959, or a host that
    sets the surface optics some other way (e.g. a single-column test).
    This deliberately does NOT fall back to
    ``radiation.surface_albedo_{vis,nir}_{dir,dif}``: those leaves exist on
    every ``RadiationData`` (zero-initialised, like every other field), so a
    carry built by ``RadiationData.zeros().copy(surface_albedo_vis=...)``
    without also setting the quartet would read back as an all-zero albedo
    rather than falling through to the broadband value it DID set -- the
    trap this fallback order avoids.

    The fallback is the BLEND, not the raw ``(vis, vis, nir, nir)`` pair:
    passing the raw pair through to a per-band RRTMGP solve gives every
    band below 0.7 um the visible value and every band at/above it the
    near-infrared one, an unintended per-band split whenever ``vis != nir``
    that no caller predating #959 asked for (Codex review on this PR caught
    the identical bug one layer down, in ``rrtmgp.py``'s own kwarg
    fallback). The blended scalar in every slot instead reproduces today's
    single-albedo RRTMGP behaviour exactly, which is also the #959
    regression pin: with no quartet published, the per-band solve must
    match the old broadband one bit for bit regardless of whether
    ``vis == nir``.
    """
    optics = diagnostics.get(SURFACE_OPTICS_KEY)
    if optics is not None and "albedo_vis_dir" in optics:
        return (optics["albedo_vis_dir"].reshape(ncols),
                optics["albedo_vis_dif"].reshape(ncols),
                optics["albedo_nir_dir"].reshape(ncols),
                optics["albedo_nir_dif"].reshape(ncols))
    vis, nir, _ = surface_optics_for_solve(diagnostics, ncols)
    broadband = 0.46 * vis + 0.54 * nir
    return broadband, broadband, broadband, broadband


def hold_land_albedo(diagnostics: dict, solved) -> dict:
    """Hold the land tile's albedo of the last radiation solve on the ``surface`` carry.

    ``solved`` is the radiation term's own gate (:func:`radiation_should_compute`,
    read before the step counter advances). On a solve the land albedo this
    step's :data:`SURFACE_OPTICS_KEY` carries, the one the grid albedo the
    solve used was built from, goes to ``surface.land_albedo_at_solve``;
    between solves the carried value stays, as ``surface_sw_down`` and
    ``surface_sw_up`` do. The land energy balance then absorbs the held
    downward shortwave through the albedo that shortwave was solved with
    (ECHAM's JSBACH takes the radiation's net shortwave, and changes its
    interactive albedo only at a radiation step).

    A no-op where there is nothing to hold: no boundary-condition term
    published the land albedo, or the ``surface`` carry has no such slot.
    """
    surface = diagnostics.get("surface")
    optics = diagnostics.get(SURFACE_OPTICS_KEY)
    held = getattr(surface, "land_albedo_at_solve", None)
    if held is None or optics is None or "land_albedo" not in optics:
        return diagnostics
    fresh = optics["land_albedo"].reshape(held.shape).astype(held.dtype)
    return {**diagnostics, "surface": surface.copy(
        land_albedo_at_solve=jnp.where(solved, fresh, held))}


def radiation_should_compute(
    diagnostics: dict, parameters: RadiationParameters,
) -> jnp.ndarray:
    """Return a scalar bool: should we recompute radiation this step?

    If ``radiation_interval > 0``, recompute every
    ``round(interval / dt)`` steps; otherwise (the default) recompute
    every step. The step counter is the radiation term's own
    ``RadiationData.step`` carry slot — incremented each call by the
    radiation term — so this gate does not depend on the model-wide
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


# The shortwave leaves of ``RadiationData`` that scale linearly with the
# incoming solar beam, and so are carried between solves as transmissivities
# (``<name>_trans``) and rebuilt every step by :func:`apply_insolation`. This is
# the complete list of shortwave quantities any consumer reads off the
# radiation carry:
#
# - TOA: ``toa_sw_up``, ``toa_sw_up_clear``, ``toa_sw_up_noa``,
#   ``toa_sw_up_clear_noa`` (all-sky, clear-sky and aerosol-free outgoing);
#   ``toa_sw_down`` is the incoming beam itself, not a transmissivity;
# - surface: ``surface_sw_down``, ``surface_sw_up`` (the surface and land
#   energy balances, ``rsds``, the albedo-weighted net);
# - profiles: ``sw_flux_up``/``sw_flux_down`` and their clear-sky partners
#   ``sw_flux_up_clear``/``sw_flux_down_clear``, and ``sw_heating_rate`` (the
#   temperature tendency).
#
# The ``noa_frac_*`` slots are absent on purpose: they are ratios of two fluxes
# and so already independent of the sun. Longwave is absent because it does not
# see the sun. A test (``radiation_caching_test``) checks that every ``sw``
# field of ``RadiationData`` is either listed here or is ``toa_sw_down``.
_CACHED_SW_FIELDS = (
    "sw_flux_up", "sw_flux_down", "sw_heating_rate",
    "sw_flux_up_clear", "sw_flux_down_clear",
    "surface_sw_down", "surface_sw_up",
    "toa_sw_up", "toa_sw_up_clear", "toa_sw_up_noa",
    "toa_sw_up_clear_noa",
)

# ECHAM's atmospheric-curvature constant ``rae`` = ratio of the atmosphere's
# scale height to the Earth's radius
# (``mo_physical_constants.f90``: ``rae = 0.1277E-2``).
_RAE = 0.1277e-2

# ECHAM's lower bound on the cosine of the zenith the radiation SOLVE is made
# at (``mo_radiation.f90::pre_radiation``: ``amu0m_x = MAX(amu0m_x, 0.1)``), so
# that a column which is dark at the solve still gets a finite transmissivity
# to apply when the sun reaches it before the next solve.
_MIN_SOLVE_COS_ZENITH = 0.1

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.25


def echam_solve_cos_zenith(cos_zenith: jnp.ndarray) -> jnp.ndarray:
    """Cosine of the zenith a radiation solve is made at, from the geometric one.

    ECHAM (``mo_radiation.f90::pre_radiation``) corrects the geometric cosine
    for the curvature of the atmosphere at high zenith angles,
    ``rae / (sqrt(mu**2 + rae*(rae + 2)) - mu)``, then floors the result at
    0.1 (a ~84 degree maximum zenith) so night columns are solved too. A column
    that is dark when radiation runs is not a zero flux but a transmissivity at
    this floored geometry, rescaled every step by the real insolation.
    """
    zrae = _RAE * (_RAE + 2.0)
    # rae / (s - mu) with s = sqrt(mu^2 + zrae) equals (s + mu) / (rae + 2)
    # (since s^2 - mu^2 = zrae = rae (rae + 2)): the same value without the
    # float32 cancellation of s - mu at large mu.
    curved = (jnp.sqrt(cos_zenith**2 + zrae) + cos_zenith) / (_RAE + 2.0)
    return jnp.maximum(curved, _MIN_SOLVE_COS_ZENITH)


def solve_solar_geometry(solar, dt_seconds, parameters: RadiationParameters):
    """Return the ``solar`` geometry a radiation solve is evaluated at.

    ECHAM solves at the MIDDLE of the radiation interval, not at the step the
    solve happens to run on (``mo_time_control.f90::radiation_time``:
    ``radiation_date = current_date + 0.5*(trigrad - delta_time)``), because
    the transmissivity it produces is then applied over the whole interval.
    Taking the midpoint makes the solve geometry the best single representative
    of the steps the result serves. ``radiation_interval <= 0`` (solve every
    step) is an offset of zero.
    """
    import dataclasses

    interval = parameters.radiation_interval
    steps_per_call = jnp.where(
        interval > 0, jnp.round(interval / dt_seconds), 1.0)
    offset = 0.5 * (steps_per_call - 1.0) * dt_seconds
    two_pi = 2.0 * jnp.pi
    return dataclasses.replace(
        solar,
        synodic_phase=solar.synodic_phase + two_pi * offset / _SECONDS_PER_DAY,
        orbital_phase=solar.orbital_phase
        + two_pi * offset / (_SECONDS_PER_DAY * _DAYS_PER_YEAR),
    )


def current_cos_zenith(solar, longitude, latitude) -> jnp.ndarray:
    """Cosine of the solar zenith angle now, for ``longitude``/``latitude``.

    Pure trigonometry -- no radiative transfer -- so it is evaluated every
    step by :func:`apply_insolation`. Both angles are in degrees, matching the
    per-column values the radiation terms cache in ``cache_coords``. Not
    clipped: negative at night.
    """
    from jax_solar import OrbitalTime, get_solar_sin_altitude

    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    # sin(altitude) == cos(zenith).
    return get_solar_sin_altitude(orbital_time, longitude, latitude)


def sw_solve_to_transmissivity(radiation: RadiationData) -> RadiationData:
    """Store a solve's shortwave as transmissivities (ECHAM ``trsol``/``trsof``).

    Divides every shortwave field of :data:`_CACHED_SW_FIELDS` by the incoming
    TOA flux of the SAME solve (``toa_sw_down``, ``S * mu_solve`` with
    ``mu_solve >= 0.1``, so strictly positive; the guard only protects a
    scheme that was handed a zero beam). The fluxes themselves are left at the
    solve geometry; :func:`apply_insolation` replaces them every step.
    """
    inc = radiation.toa_sw_down
    ok = inc > 0.0
    # Double-where: the unused branch must not divide by zero, or its
    # reverse-mode cotangent is NaN even though the value is masked.
    safe = jnp.where(ok, inc, 1.0)
    return radiation.copy(**{
        f"{name}_trans": jnp.where(ok, getattr(radiation, name) / safe, 0.0)
        for name in _CACHED_SW_FIELDS
    })


def apply_insolation(
    radiation: RadiationData, solar, longitude, latitude, solar_constant,
) -> RadiationData:
    """Rebuild every shortwave field from the transmissivities and the sun NOW.

    ``field = trans * S(t) * max(cos_zenith(t), 0)``: ECHAM's
    ``physc.f90``: ``zi0 = flx_ratio_cur * solc * amu0_x * rdayl_x`` followed by
    ``radheat.f90``: ``zflxs = zi0 * trsol``, applied on every step whether or
    not a solve ran. ``S(t)`` carries the Earth-Sun distance factor
    (``jax_solar.direct_solar_irradiance``), ``max(., 0)`` is the day/night
    mask. The incoming beam is exactly the instantaneous insolation, so the
    time mean of ``toa_sw_down`` is the exact insolation integral however
    coarse the radiation interval. ``cos_zenith`` is published as the current
    clipped value.
    """
    from jax_solar import direct_solar_irradiance

    dtype = radiation.toa_sw_down.dtype
    mu_now = jnp.maximum(
        current_cos_zenith(solar, longitude, latitude), 0.0).astype(dtype)
    inc = (direct_solar_irradiance(solar.orbital_phase, solar_constant)
           .astype(dtype) * mu_now)
    scaled = {
        name: getattr(radiation, f"{name}_trans") * inc
        for name in _CACHED_SW_FIELDS
    }
    return radiation.copy(cos_zenith=mu_now, toa_sw_down=inc, **scaled)


__all__ = [
    "SURFACE_OPTICS_KEY",
    "cached_radiation_tendency",
    "surface_optics_for_solve",
    "surface_optics_bands_for_solve",
    "apply_insolation",
    "current_cos_zenith",
    "echam_solve_cos_zenith",
    "hold_land_albedo",
    "radiation_should_compute",
    "solve_solar_geometry",
    "sw_solve_to_transmissivity",
]
