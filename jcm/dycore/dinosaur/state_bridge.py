"""Gridpoint ↔ modal conversions for the dinosaur dynamical core.

These functions used to live in :mod:`jcm.physics_interface`. They are
dinosaur-specific (they call into ``dinosaur.spherical_harmonic`` and
``dinosaur.primitive_equations``) and so they belong on the dycore side of the
``DynamicalCore`` protocol — outside this subpackage, the rest of jax-gcm only
sees the dycore-agnostic :class:`PhysicsState` / :class:`PhysicsTendency`
types.

The three functions here are pure JAX (no side effects, no Python conditionals
on traced values). This module also owns Dinosaur's nondimensionalisation
boundary. Every mass mixing ratio — specific humidity, cloud condensate,
aerosol and gas mass — crosses it as the dimensionless kg/kg value, because
Dinosaur's moist primitive equations consume the stored humidity and
condensate tracers directly in their virtual-temperature terms. Tracers
declaring ``nondimensionalize=False`` (number concentrations, VMRs) are
passed through untouched.
"""

from __future__ import annotations

import jax.numpy as jnp
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



#: Prognostic condensate species that load the virtual temperature. ECHAM6
#: computes ``Tv = T (1 + vtmpc1 q - (xl + xi))`` in both its dynamics
#: (``dyn.f90::ztv``) and the geopotential it hands physics
#: (``physc.f90::ztvm1``); jcm adds prognostic rain and snow, which ECHAM6's
#: one-moment scheme does not carry, because suspended precipitation loads a
#: column exactly as suspended cloud does. Names absent from a composition's
#: tracer set are simply skipped.
CONDENSATE_TRACERS = ("qc", "qi", "qr", "qs")


def _condensate_loading(tracers):
    """Sum the condensate mass mixing ratios present in ``tracers``.

    Returns ``None`` when the composition carries no condensate, which is the
    signal Dinosaur's geopotential helpers take to mean "no cloud loading".
    """
    present = [tracers[name] for name in CONDENSATE_TRACERS if name in tracers]
    if not present:
        return None
    total = present[0]
    for value in present[1:]:
        total = total + value
    return total


def dynamics_state_to_physics_state(
    state: State,
    dynamics: PrimitiveEquations,
    tracer_specs: dict | None = None,
    nodal_tracers: tuple = (),
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
        nodal_tracers: Names of tracers the (semi-Lagrangian) dycore carries
            as NODAL arrays inside ``State.tracers`` — they skip every
            spectral transform (that is their whole point: no per-step Gibbs
            ringing for sharp sources), so they must bypass the modal
            diagnostic pipeline here and only get the spec-driven
            dimensionalization. Empty for the Eulerian spectral core.

    Returns:
        Gridpoint :class:`PhysicsState`.

    """
    # No logging here: this runs per timestep inside the model's scan, and a
    # host callback makes the whole integration uncacheable by XLA.

    # Nodal tracers must not enter the modal->nodal diagnostic pipeline;
    # split them off and merge them (dimensionalized) into the output below.
    nodal_direct = {k: v for k, v in state.tracers.items() if k in nodal_tracers}
    if nodal_direct:
        state = state.replace(tracers={
            k: v for k, v in state.tracers.items() if k not in nodal_tracers
        })

    u, v = vor_div_to_uv_nodal(dynamics.coords.horizontal, state.vorticity, state.divergence)

    # Z, X, Y — dispatch to the hybrid variant when the vertical coord is hybrid.
    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        nodal_state = compute_diagnostic_state_hybrid(state, dynamics.coords)
    else:
        nodal_state = compute_diagnostic_state(state, dynamics.coords)
    t = nodal_state.temperature_variation
    # Dinosaur's moisture-aware primitive equations consume q numerically as a
    # dimensionless mass fraction. Keep that representation through the
    # diagnostic calculations; treating the same value as g/kg suppresses the
    # virtual-temperature contribution by a factor of 1000.
    q = nodal_state.tracers['specific_humidity']

    # Condensate loading for the virtual temperature, from whichever frame
    # each species is carried in. Both dicts hold the dimensionless kg/kg
    # value, the same representation the dynamics' own Tv term reads.
    clouds = _condensate_loading({**nodal_state.tracers, **nodal_direct})

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
            specific_humidity=q,
            clouds=clouds,
            nodal_orography=nodal_orography,
            coordinates=dynamics.nondim_levels,
            gravity_acceleration=dynamics.physics_specs.gravity_acceleration,
            ideal_gas_constant=dynamics.physics_specs.R,
            water_vapor_gas_constant=dynamics.physics_specs.R_vapor,
            sharding=None,
        )
    else:
        full_temperature = nodal_state.temperature_variation + dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
        phi = get_geopotential_on_sigma(
            temperature=full_temperature,
            specific_humidity=q,
            clouds=clouds,
            nodal_orography=nodal_orography,
            sigma=dynamics.coords.vertical,
            gravity_acceleration=dynamics.physics_specs.gravity_acceleration,
            ideal_gas_constant=dynamics.physics_specs.R,
            water_vapor_gas_constant=dynamics.physics_specs.R_vapor,
            sharding=None,
        )

    t += dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    q = dynamics.physics_specs.dimensionalize(q, units.dimensionless).m

    # Extra tracers — those with ``nondimensionalize=False`` (e.g. number
    # concentrations) pass through untouched; every other tracer is a mass
    # mixing ratio and shares specific humidity's contract: kg/kg, which is
    # dimensionless, stored as the physical value. Dinosaur reads the stored
    # condensate directly for the virtual-temperature loading term, so any
    # rescaling here would weaken that coupling by exactly the scale factor.
    # Nodal tracers (split off above) are already gridpoint arrays and only
    # need the same dimensionalization.
    all_tracers = {}
    for tracer_name, tracer_value in {**nodal_state.tracers, **nodal_direct}.items():
        if tracer_name == 'specific_humidity':
            continue
        spec = tracer_specs.get(tracer_name) if tracer_specs else None
        if spec is not None and not spec.nondimensionalize:
            all_tracers[tracer_name] = tracer_value
        else:
            all_tracers[tracer_name] = dynamics.physics_specs.dimensionalize(
                tracer_value, units.dimensionless,
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
    nodal_tracers: tuple = (),
) -> State:
    """Convert a gridpoint :class:`PhysicsState` back into a dinosaur ``State``.

    The inverse of :func:`dynamics_state_to_physics_state`. Used at model
    initialization when the user supplies a gridpoint ``PhysicsState`` rather
    than letting the dycore build its own default.
    """
    modal_vorticity, modal_divergence = uv_nodal_to_vor_div_modal(
        dynamics.coords.horizontal, physics_state.u_wind, physics_state.v_wind,
    )

    # kg/kg is dimensionless. Dinosaur must store the physical mass fraction,
    # because its hybrid primitive equations use this tracer directly in the
    # virtual-temperature and moist thermodynamic terms.
    q = dynamics.physics_specs.nondimensionalize(
        physics_state.specific_humidity * units.dimensionless
    )
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
            # Mass mixing ratios share specific humidity's kg/kg contract.
            tracer_nd = dynamics.physics_specs.nondimensionalize(
                tracer_value * units.dimensionless,
            )
        # Nodal tracers stay gridpoint (the semi-Lagrangian core transports
        # them without any spectral round trip — see
        # ``dynamics_state_to_physics_state``).
        if tracer_name in nodal_tracers:
            tracers_modal[tracer_name] = jnp.asarray(tracer_nd)
        else:
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
    nodal_tracers: tuple = (),
) -> State:
    """Convert gridpoint physics tendencies into a dinosaur dynamics-tendency ``State``.

    The returned ``State`` is intended to be forward-Euler-added to the
    dycore's current modal state (operator-split Lie a).
    """
    u_tend = physics_tendency.u_wind
    v_tend = physics_tendency.v_wind
    t_tend = physics_tendency.temperature
    q_tend = physics_tendency.specific_humidity

    q_tend = dynamics.physics_specs.nondimensionalize(
        q_tend * units.dimensionless / units.second
    )

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
                tracer_tend * units.dimensionless / units.second,
            )
        # Nodal tracer tendencies stay gridpoint so the operator-split
        # forward-Euler add matches the nodal state entries shape-for-shape.
        if tracer_name in nodal_tracers:
            tracers_tend_modal[tracer_name] = jnp.asarray(tracer_tend_nd)
        else:
            tracers_tend_modal[tracer_name] = dynamics.coords.horizontal.to_modal(tracer_tend_nd)

    return State(
        vor_tend_modal,
        div_tend_modal,
        t_tend_modal,
        log_sp_tend_modal,
        sim_time=0.,
        tracers=tracers_tend_modal,
    )
