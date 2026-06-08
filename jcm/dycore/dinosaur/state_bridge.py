"""Gridpoint ↔ modal conversions for the dinosaur dynamical core.

These functions used to live in :mod:`jcm.physics_interface`. They are
dinosaur-specific (they call into ``dinosaur.spherical_harmonic`` and
``dinosaur.primitive_equations``) and so they belong on the dycore side of the
``DynamicalCore`` protocol — outside this subpackage, the rest of jax-gcm only
sees the dycore-agnostic :class:`PhysicsState` / :class:`PhysicsTendency`
types.

The three functions here are pure JAX (no side effects, no Python conditionals
on traced values) and the implementations match what was in
``physics_interface.py`` line-for-line up to import paths; the dinosaur-side
refactor regression in :mod:`jcm.dycore.dinosaur.regression_test` is the bit-
level guardrail.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dinosaur import scales
from dinosaur.hybrid_coordinates import HybridCoordinates
from dinosaur.primitive_equations import (
    State, PrimitiveEquations,
    compute_diagnostic_state, compute_diagnostic_state_hybrid,
    get_geopotential_on_sigma, get_geopotential_on_hybrid,
)
from dinosaur.scales import units
from dinosaur.spherical_harmonic import (
    uv_nodal_to_vor_div_modal, vor_div_to_uv_nodal,
)
import logging

from jcm.physics_interface import PhysicsState, PhysicsTendency


logger = logging.getLogger(__name__)


def dynamics_state_to_physics_state(
    state: State,
    dynamics: PrimitiveEquations,
    tracer_specs: dict | None = None,
) -> PhysicsState:
    """Convert a dinosaur modal ``State`` into a gridpoint :class:`PhysicsState`.

    Args:
        state: Dinosaur ``State`` in spectral space.
        dynamics: ``PrimitiveEquations`` carrying the reference temperature,
            orography, and physics specs.
        tracer_specs: Optional ``name -> TracerSpec`` mapping. Tracers whose
            spec has ``nondimensionalize=False`` (e.g. number concentrations,
            VMRs) bypass the gram/kg conversion. Default ``None`` applies the
            gram/kg conversion to every non-``specific_humidity`` tracer.

    Returns:
        Gridpoint :class:`PhysicsState`.

    """
    jax.debug.callback(lambda: logger.debug("Converting state variables from dynamics to physics state variables"))

    u, v = vor_div_to_uv_nodal(dynamics.coords.horizontal, state.vorticity, state.divergence)

    # Z, X, Y — dispatch to the hybrid variant when the vertical coord is hybrid.
    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        nodal_state = compute_diagnostic_state_hybrid(state, dynamics.coords)
    else:
        nodal_state = compute_diagnostic_state(state, dynamics.coords)
    t = nodal_state.temperature_variation
    q = nodal_state.tracers['specific_humidity']

    nodal_orography = dynamics.coords.horizontal.to_nodal(dynamics.orography)
    log_sp = dynamics.coords.horizontal.to_nodal(state.log_surface_pressure)
    sp = jnp.exp(log_sp)

    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        # Hybrid coords store ``log(P_s in nondim Pa)`` directly; ``exp(log_sp)``
        # is the surface pressure in the same units as ``a_boundaries``.
        # ``get_geopotential_on_hybrid`` uses method='sparse' which needs ``sp``
        # with a leading vertical-1 axis (shape ``(1, lon, lat)``).
        full_temperature = nodal_state.temperature_variation + dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
        phi = get_geopotential_on_hybrid(
            temperature=full_temperature,
            surface_pressure=sp,
            specific_humidity=None,
            nodal_orography=nodal_orography,
            coordinates=dynamics.nondim_levels,
            gravity_acceleration=dynamics.physics_specs.nondimensionalize(scales.GRAVITY_ACCELERATION),
            ideal_gas_constant=dynamics.physics_specs.nondimensionalize(scales.IDEAL_GAS_CONSTANT),
            sharding=None,
        )
    else:
        full_temperature = nodal_state.temperature_variation + dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
        phi = get_geopotential_on_sigma(
            temperature=full_temperature,
            specific_humidity=None,
            nodal_orography=nodal_orography,
            sigma=dynamics.coords.vertical,
            gravity_acceleration=dynamics.physics_specs.nondimensionalize(scales.GRAVITY_ACCELERATION),
            ideal_gas_constant=dynamics.physics_specs.nondimensionalize(scales.IDEAL_GAS_CONSTANT),
            sharding=None,
        )

    t += dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    q = dynamics.physics_specs.dimensionalize(q, units.gram / units.kilogram).m

    # Extra tracers — those with ``nondimensionalize=False`` (e.g. number
    # concentrations) pass through untouched; everything else is treated as a
    # mass mixing ratio in gram/kilogram.
    all_tracers = {}
    for tracer_name, tracer_value in nodal_state.tracers.items():
        if tracer_name == 'specific_humidity':
            continue
        spec = tracer_specs.get(tracer_name) if tracer_specs else None
        if spec is not None and not spec.nondimensionalize:
            all_tracers[tracer_name] = tracer_value
        else:
            all_tracers[tracer_name] = dynamics.physics_specs.dimensionalize(
                tracer_value, units.gram / units.kilogram,
            ).m

    # Produce ``normalized_surface_pressure = P_s / p0`` on a common scale
    # regardless of coord family.
    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        from jcm.constants import p0 as P0_PA
        p0_nondim = dynamics.physics_specs.nondimensionalize(P0_PA * units.pascal)
        nsp = jnp.squeeze(sp, axis=-3) / p0_nondim
    else:
        nsp = jnp.squeeze(sp, axis=-3)

    return PhysicsState(u, v, t, q, phi, nsp, all_tracers)


def physics_state_to_dynamics_state(
    physics_state: PhysicsState,
    dynamics: PrimitiveEquations,
    tracer_specs: dict | None = None,
) -> State:
    """Convert a gridpoint :class:`PhysicsState` back into a dinosaur ``State``.

    The inverse of :func:`dynamics_state_to_physics_state`. Used at model
    initialization when the user supplies a gridpoint ``PhysicsState`` rather
    than letting the dycore build its own default.
    """
    modal_vorticity, modal_divergence = uv_nodal_to_vor_div_modal(
        dynamics.coords.horizontal, physics_state.u_wind, physics_state.v_wind,
    )

    q = dynamics.physics_specs.nondimensionalize(physics_state.specific_humidity * units.gram / units.kilogram)
    q_modal = dynamics.coords.horizontal.to_modal(q)

    temperature = physics_state.temperature - dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    temperature_modal = dynamics.coords.horizontal.to_modal(temperature)

    # ``normalized_surface_pressure`` is P_s / p0 regardless of coord family
    # (see :func:`dynamics_state_to_physics_state`). dinosaur stores
    # ``log(P_s / p0)`` for sigma but ``log(P_s)`` (in nondim Pa) for hybrid, so
    # the hybrid branch must multiply by ``p0_nondim`` before the log — the exact
    # inverse of the division done on the forward path. Without this, an injected
    # hybrid PhysicsState collapses surface pressure by a factor of ~p0.
    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        from jcm.constants import p0 as P0_PA
        p0_nondim = dynamics.physics_specs.nondimensionalize(P0_PA * units.pascal)
        sp_nondim = physics_state.normalized_surface_pressure * p0_nondim
    else:
        sp_nondim = physics_state.normalized_surface_pressure
    log_surface_pressure = jnp.log(sp_nondim)
    modal_log_sp = dynamics.coords.horizontal.to_modal(log_surface_pressure)

    tracers_modal = {'specific_humidity': q_modal}
    for tracer_name, tracer_value in physics_state.tracers.items():
        spec = tracer_specs.get(tracer_name) if tracer_specs else None
        if spec is not None and not spec.nondimensionalize:
            tracer_nd = tracer_value
        else:
            tracer_nd = dynamics.physics_specs.nondimensionalize(
                tracer_value * units.gram / units.kilogram,
            )
        tracers_modal[tracer_name] = dynamics.coords.horizontal.to_modal(tracer_nd)

    return State(
        vorticity=modal_vorticity,
        divergence=modal_divergence,
        temperature_variation=temperature_modal,
        log_surface_pressure=modal_log_sp[..., jnp.newaxis, :, :],
        tracers=tracers_modal,
    )


def physics_tendency_to_dynamics_tendency(
    physics_tendency: PhysicsTendency,
    dynamics: PrimitiveEquations,
    tracer_specs: dict | None = None,
) -> State:
    """Convert gridpoint physics tendencies into a dinosaur dynamics-tendency ``State``.

    The returned ``State`` is intended to be forward-Euler-added to the
    dycore's current modal state (operator-split Lie a).
    """
    u_tend = physics_tendency.u_wind
    v_tend = physics_tendency.v_wind
    t_tend = physics_tendency.temperature
    q_tend = physics_tendency.specific_humidity

    q_tend = dynamics.physics_specs.nondimensionalize(q_tend * units.gram / units.kilogram / units.second)

    vor_tend_modal, div_tend_modal = uv_nodal_to_vor_div_modal(
        dynamics.coords.horizontal, u_tend, v_tend,
    )
    t_tend_modal = dynamics.coords.horizontal.to_modal(t_tend)
    q_tend_modal = dynamics.coords.horizontal.to_modal(q_tend)

    # The dinosaur ``State.log_surface_pressure`` is shape ``(1, n_lat_modes,
    # n_lon_modes)`` (a leading vertical axis of size 1 for broadcasting). The
    # op-split path adds the tendency directly to the state via tree_math,
    # which requires *exact* shape matches — keep the leading axis.
    log_sp_tend_modal = jnp.zeros_like(t_tend_modal[:1, ...])

    tracers_tend_modal = {'specific_humidity': q_tend_modal}
    for tracer_name, tracer_tend in physics_tendency.tracers.items():
        spec = tracer_specs.get(tracer_name) if tracer_specs else None
        if spec is not None and not spec.nondimensionalize:
            tracer_tend_nd = tracer_tend
        else:
            tracer_tend_nd = dynamics.physics_specs.nondimensionalize(
                tracer_tend * units.gram / units.kilogram / units.second,
            )
        tracers_tend_modal[tracer_name] = dynamics.coords.horizontal.to_modal(tracer_tend_nd)

    return State(
        vor_tend_modal,
        div_tend_modal,
        t_tend_modal,
        log_sp_tend_modal,
        sim_time=0.,
        tracers=tracers_tend_modal,
    )
