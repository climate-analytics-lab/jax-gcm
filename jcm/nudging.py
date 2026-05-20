"""Newtonian relaxation toward an external reference state.

Implements nudging as a composable :class:`PhysicsTerm`. The relaxation
tendency ``dX/dt = inv_tau · (X_ref − X)`` is computed in gridpoint space
on the physics-facing :class:`PhysicsState`, returned as a
:class:`PhysicsTendency`, and folded into the same forward-Euler op-split
add the rest of the physics stack uses. The dycore stays nudging-free.

Per-variable, per-level relaxation timescales are configurable so the
common case ("nudge winds above the PBL") is expressible without
subclassing. Time-varying targets are supported via
:class:`jcm.forcing.TimeSeries` leaves and the existing ``select(date,
calendar)`` machinery; the current ``DateData`` is plumbed into the
physics path through the diagnostics dict (``ComposablePhysics`` injects
``_date`` at the top of every ``compute_tendencies`` call). The calendar
is held on :class:`NudgingTerm` itself — a Python ``str`` can't ride
through ``jax.checkpoint`` in the diagnostics dict.

The previous version of this module composed the relaxation tendency
directly into the dinosaur IMEX-RK substages by transforming everything
to modal space at load time. That coupled the relaxation API to a single
dycore. Moving to a :class:`PhysicsTerm` keeps it physics-agnostic — any
dycore that satisfies the :class:`DynamicalCore` protocol gets nudging
for free.

Reference: Krishnamurti et al. (1991), *Tellus 43AB*, 53–81.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import jax
import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.date import DateData, DEFAULT_CALENDAR
from jcm.forcing import ForcingData, TimeSeries, BY_DATE, _select_time_series, make_time_series
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


# ---------------------------------------------------------------------------
# Reference fields
# ---------------------------------------------------------------------------


@tree_math.struct
class NudgingTarget:
    """Gridpoint reference fields the relaxation drives the state toward.

    All fields are dimensional in the model's native conventions: ``u_wind``
    and ``v_wind`` in m/s, ``temperature`` in K, on the
    ``(nlev, *horizontal_shape)`` layout the dycore exposes via
    ``coords.horizontal.nodal_shape``. Each leaf can be a bare
    ``jnp.ndarray`` (static target) or a :class:`jcm.forcing.TimeSeries`
    leaf with a leading time axis that ``select(date, calendar)`` slices
    per step.

    Use :meth:`from_dataset` to build one from an xarray Dataset of
    nodal (lat / lon / level) reference fields.
    """

    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray

    def select(self, date: DateData, calendar: str = DEFAULT_CALENDAR) -> "NudgingTarget":
        """Collapse every TimeSeries leaf to its current-step slice.

        No-op for static targets. Called by :class:`NudgingTerm` once per
        physics step so downstream tendency math sees plain arrays.
        """
        def slice_leaf(leaf):
            if isinstance(leaf, TimeSeries):
                return _select_time_series(leaf, date, calendar=calendar)
            return leaf
        return jax.tree_util.tree_map(
            slice_leaf, self,
            is_leaf=lambda x: isinstance(x, TimeSeries),
        )

    @classmethod
    def from_dataset(cls, ds, *,
                     u_var: str = "u", v_var: str = "v",
                     T_var: str = "T",
                     time_var: Optional[str] = "time") -> "NudgingTarget":
        """Build a :class:`NudgingTarget` from an xarray Dataset.

        Args:
            ds: ``xarray.Dataset`` carrying ``u``, ``v``, ``T`` (or names
                overridden by the ``*_var`` kwargs). Each is expected with
                axes ``(time, lev, lat, lon)`` — time is optional, see
                ``time_var``. Loaded verbatim onto the model grid;
                regridding is the user's responsibility before loading.
            u_var, v_var, T_var: netCDF variable names.
            time_var: Time coord name. ``None`` for static (climatology)
                reference data.

        Returns:
            A :class:`NudgingTarget` ready to attach to a
            :class:`NudgingTerm`.

        """
        import numpy as np
        from jcm.forcing import _time_axis_seconds_from_ds

        is_time_varying = time_var is not None and time_var in ds.coords

        def to_jax(name):
            return jnp.asarray(np.asarray(ds[name].values))

        u = to_jax(u_var)
        v = to_jax(v_var)
        T = to_jax(T_var)

        if is_time_varying:
            time_seconds = _time_axis_seconds_from_ds(ds.rename({time_var: "time"}))
            return cls(
                u_wind=make_time_series(u, time_seconds, align_mode=BY_DATE),
                v_wind=make_time_series(v, time_seconds, align_mode=BY_DATE),
                temperature=make_time_series(T, time_seconds, align_mode=BY_DATE),
            )
        return cls(u_wind=u, v_wind=v, temperature=T)


# ---------------------------------------------------------------------------
# What to nudge, with what timescale
# ---------------------------------------------------------------------------


@tree_math.struct
class NudgingConfig:
    """Per-variable, per-level inverse relaxation timescales (1 / s).

    All values are dimensional (per second). Zero entries mean "no nudging"
    for that variable / level — that's how the common "winds above the PBL
    only" pattern is expressed: ``inv_tau_wind`` non-zero from the free
    troposphere upwards, zero below; ``inv_tau_temperature`` zero everywhere.

    Wind nudging applies symmetrically to ``u_wind`` and ``v_wind`` (one
    inverse-timescale profile covers both). Surface-pressure nudging is
    deliberately not supported through the physics path — the dycore
    advances surface pressure via the continuity equation, and a ps
    nudging tendency would require extending :class:`PhysicsTendency` with
    a ``normalized_surface_pressure`` field. Add it only when a concrete
    use case lands.
    """

    inv_tau_wind: jnp.ndarray         # (nlev,) — applied to both u and v
    inv_tau_temperature: jnp.ndarray  # (nlev,)

    @classmethod
    def winds_only(cls, nlev: int, *, tau_seconds: float = 21600.0,
                   pbl_levels: int = 0) -> "NudgingConfig":
        """Nudge winds everywhere except the bottom ``pbl_levels`` layers.

        Args:
            nlev: Number of vertical levels.
            tau_seconds: Relaxation timescale in seconds (default 6 h).
            pbl_levels: Number of levels at the bottom of the column
                (highest sigma) where wind nudging is suppressed. Default
                0 (nudge all levels).

        """
        inv_tau = 1.0 / float(tau_seconds)
        # Convention: level 0 is TOA, level nlev-1 is the surface.
        mask = jnp.ones(nlev).at[nlev - pbl_levels:].set(0.0) if pbl_levels else jnp.ones(nlev)
        return cls(
            inv_tau_wind=inv_tau * mask,
            inv_tau_temperature=jnp.zeros(nlev),
        )


# ---------------------------------------------------------------------------
# Tendency helper (split out so unit tests can call it without a Model)
# ---------------------------------------------------------------------------


def nudging_tendency(state: PhysicsState, target: NudgingTarget,
                     config: NudgingConfig) -> PhysicsTendency:
    """Newtonian relaxation tendency in gridpoint space.

    ``dX/dt = inv_tau · (X_ref − X)`` per relaxed variable. Variables with
    zero ``inv_tau`` get zero tendency. Tracers and specific humidity are
    not nudged — the tendency carries zeros for them so the
    :class:`PhysicsTendency` pytree shape matches the state's tracer dict.

    Args:
        state: Current gridpoint state.
        target: Reference fields. Call ``target.select(date)`` *before*
            this for time-varying references.
        config: Per-variable inverse-tau profiles.

    Returns:
        A :class:`PhysicsTendency` with the relaxation tendencies. Caller
        is responsible for converting it to the dycore's native tendency
        representation if needed.

    """
    inv_tau_wind = config.inv_tau_wind[:, jnp.newaxis, jnp.newaxis]
    inv_tau_temp = config.inv_tau_temperature[:, jnp.newaxis, jnp.newaxis]

    u_t = inv_tau_wind * (target.u_wind - state.u_wind)
    v_t = inv_tau_wind * (target.v_wind - state.v_wind)
    T_t = inv_tau_temp * (target.temperature - state.temperature)
    q_zeros = jnp.zeros_like(state.specific_humidity)
    tracer_zeros = {name: jnp.zeros_like(t) for name, t in state.tracers.items()}

    return PhysicsTendency(
        u_wind=u_t, v_wind=v_t, temperature=T_t,
        specific_humidity=q_zeros, tracers=tracer_zeros,
    )


# ---------------------------------------------------------------------------
# Composable PhysicsTerm
# ---------------------------------------------------------------------------


class NudgingTerm(PhysicsTerm):
    """Composable Newtonian-relaxation physics term.

    Holds a :class:`NudgingTarget`, a :class:`NudgingConfig`, and the
    calendar string used to align time-varying targets. On every physics
    step it slices the target against the current date (read from
    ``diagnostics["_date"]`` — injected by
    :class:`~jcm.physics.composable_physics.ComposablePhysics`) and emits
    the gridpoint relaxation tendency.

    Drop it into any ``ComposablePhysics`` term list to add nudging::

        physics = speedy_physics() + NudgingTerm(target, config)

    The calendar is held on the term rather than threaded through the
    diagnostics dict because the dict rides through ``jax.checkpoint``,
    which only accepts traceable types (not Python ``str``). Match it to
    the calendar you pass to :class:`Model` for time-varying targets.
    """

    name: ClassVar[str] = "nudging"
    category: ClassVar[str] = "nudging"

    # ``target`` and ``config`` are tree_math structs containing JAX
    # arrays — annotate them with ``nnx.data`` so flax's pytree machinery
    # traverses them (rather than rejecting them as unrecognised types).
    # ``calendar`` is a Python ``str`` and stays as a non-pytree class
    # attribute by default.
    target: NudgingTarget = nnx.data(None)
    config: NudgingConfig = nnx.data(None)

    def __init__(self, target: NudgingTarget, config: NudgingConfig,
                 calendar: str = DEFAULT_CALENDAR):
        """Initialise the term with a reference target, timescales, and calendar."""
        self.target = target
        self.config = config
        self.calendar = calendar

    def __call__(self, state: PhysicsState, diagnostics: dict,
                 forcing: ForcingData, terrain: TerrainData):
        """Compute the per-step relaxation tendency.

        Reads ``_date`` from ``diagnostics`` if present; falls back to the
        static target otherwise — that's the right behaviour when the
        target carries no ``TimeSeries`` leaves.
        """
        date = diagnostics.get("_date")
        target_now = (
            self.target.select(date, calendar=self.calendar)
            if date is not None else self.target
        )
        return nudging_tendency(state, target_now, self.config), diagnostics


# ---------------------------------------------------------------------------
# Convenience helper for the common "free-running physics + nudging" case
# ---------------------------------------------------------------------------


def with_nudging(physics, target: NudgingTarget, config: NudgingConfig,
                 calendar: str = DEFAULT_CALENDAR):
    """Return ``physics`` extended with a :class:`NudgingTerm`.

    Equivalent to ``physics + NudgingTerm(target, config, calendar)`` for any
    :class:`ComposablePhysics`; the explicit helper exists so callers can
    discover the pattern via :mod:`jcm.nudging` rather than having to know
    that ``ComposablePhysics`` overloads ``+``.
    """
    if not isinstance(physics, ComposablePhysics):
        raise TypeError(
            f"with_nudging expects a ComposablePhysics, got {type(physics).__name__}"
        )
    return physics + NudgingTerm(target, config, calendar=calendar)
