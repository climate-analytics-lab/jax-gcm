"""Newtonian relaxation toward an external reference state (#129).

Implements model nudging as a spectral-space tendency that gets composed
with the dycore + physics. Per-variable, per-level relaxation timescales
are configurable so the common case ("nudge winds above the PBL") is
expressible without subclassing or special-casing the rest of the model.

Architecture:

    dynamics + physics + nudging   (composed via ``compose_equations``)
                                              │
                                              ▼
                              ``dX/dt|_nudge = (X_ref - X) / τ``

The tendency is built directly in spectral space, matching the dycore's
own state representation (``vorticity``, ``divergence``,
``temperature_variation``, ``log_surface_pressure``). Reference fields
are loaded as nodal data and transformed to modal once at construction
time, so the per-step cost is just an array slice + element-wise
multiplication. Time-varying targets are supported via
:class:`jcm.forcing.TimeSeries` leaves and the existing
``select(date, calendar)`` machinery.

The scheme is physics-agnostic by construction — it acts on the dycore
state, so any physics package (SPEEDY, ICON, future ones) gets nudging
"for free" as long as it composes with the dynamical core.

Reference: Krishnamurti et al. (1991), *Tellus 43AB*, 53–81. Implementation
patterned after ECHAM ``mo_nudging.f90``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.primitive_equations import State
from dinosaur.scales import units
from dinosaur.spherical_harmonic import uv_nodal_to_vor_div_modal
from typing import Optional

from jcm.constants import p0
from jcm.date import DateData, DEFAULT_CALENDAR
from jcm.forcing import TimeSeries, BY_DATE, _select_time_series, make_time_series


# ---------------------------------------------------------------------------
# Reference fields
# ---------------------------------------------------------------------------


@tree_math.struct
class NudgingTarget:
    """Modal-space reference fields the nudging relaxes toward.

    Each field is either a bare ``jnp.ndarray`` (static, applied at every
    step) or a :class:`jcm.forcing.TimeSeries` leaf (with a leading time
    axis that ``select(date, calendar)`` slices per step). Fields are in
    spectral form, matching the dycore's own state.

    Use :meth:`from_dataset` to build one from an xarray Dataset of
    nodal (lat / lon / level) reference fields — the loader handles the
    nodal-to-modal transform once at construction time.
    """

    vorticity: jnp.ndarray              # (nlev, m, n) or TimeSeries((n_time, nlev, m, n))
    divergence: jnp.ndarray
    temperature_variation: jnp.ndarray
    log_surface_pressure: jnp.ndarray   # (1, m, n) or TimeSeries leaf

    def select(self, date: DateData, calendar: str = DEFAULT_CALENDAR) -> "NudgingTarget":
        """Collapse every TimeSeries leaf to its current-step slice.

        No-op for static targets. The Model calls this once per step so
        downstream tendency code sees plain modal arrays.
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
    def from_dataset(cls, ds, coords: CoordinateSystem, *,
                     reference_temperature: jnp.ndarray,
                     physics_specs,
                     u_var: str = "u", v_var: str = "v",
                     T_var: str = "T", ps_var: str = "ps",
                     time_var: Optional[str] = "time"):
        """Build a NudgingTarget from an xarray Dataset of nodal reference
        fields. Performs the nodal-to-modal transform at load time.

        Args:
            ds: ``xarray.Dataset`` carrying ``u``, ``v``, ``T``, ``ps`` (or
                names overridden by the *_var keyword args). u/v/T are
                expected with axes ``(time, lev, lat, lon)`` (time is
                optional — see ``time_var``); ps is ``(time, lat, lon)``.
            coords: The same ``CoordinateSystem`` the Model will run on.
                Used to look up the spectral transform.
            reference_temperature: Per-level reference profile that the
                dycore subtracts before storing temperature; pull from
                ``Model.primitive.reference_temperature`` so the nudging
                target is consistent with the dycore's representation.
            physics_specs: ``PhysicsSpecs`` for nondimensionalisation
                (also from ``Model.primitive.physics_specs``).
            u_var, v_var, T_var, ps_var: netCDF variable names.
            time_var: Time coord name. ``None`` for static (climatology)
                reference data.

        Returns:
            A :class:`NudgingTarget` ready to attach to ``Nudging``.

        """
        import numpy as np
        from jcm.forcing import _time_axis_seconds_from_ds

        is_time_varying = time_var is not None and time_var in ds.coords

        def to_jax(name):
            return jnp.asarray(np.asarray(ds[name].values))

        u = to_jax(u_var)   # (time?, lev, lat, lon)
        v = to_jax(v_var)
        T = to_jax(T_var)
        ps = to_jax(ps_var)  # (time?, lat, lon)

        if is_time_varying:
            time_seconds = _time_axis_seconds_from_ds(ds.rename({time_var: "time"}))

            def transform_step(u_t, v_t, T_t, ps_t):
                vor, div = uv_nodal_to_vor_div_modal(coords.horizontal, u_t, v_t)
                T_var_nodal = T_t - reference_temperature[:, jnp.newaxis, jnp.newaxis]
                T_modal = coords.horizontal.to_modal(T_var_nodal)
                ps_norm = ps_t / physics_specs.nondimensionalize(p0 * units.pascal)
                log_ps_modal = coords.horizontal.to_modal(jnp.log(ps_norm))
                return vor, div, T_modal, log_ps_modal[jnp.newaxis, ...]

            vors, divs, Ts, log_sps = jax.vmap(transform_step)(u, v, T, ps)

            return cls(
                vorticity=make_time_series(vors, time_seconds, align_mode=BY_DATE),
                divergence=make_time_series(divs, time_seconds, align_mode=BY_DATE),
                temperature_variation=make_time_series(Ts, time_seconds, align_mode=BY_DATE),
                log_surface_pressure=make_time_series(log_sps, time_seconds, align_mode=BY_DATE),
            )

        # Static case: collapse a single timestep through the same transform.
        vor, div = uv_nodal_to_vor_div_modal(coords.horizontal, u, v)
        T_var_nodal = T - reference_temperature[:, jnp.newaxis, jnp.newaxis]
        T_modal = coords.horizontal.to_modal(T_var_nodal)
        ps_norm = ps / physics_specs.nondimensionalize(p0 * units.pascal)
        log_ps_modal = coords.horizontal.to_modal(jnp.log(ps_norm))
        return cls(
            vorticity=vor,
            divergence=div,
            temperature_variation=T_modal,
            log_surface_pressure=log_ps_modal[jnp.newaxis, ...],
        )


# ---------------------------------------------------------------------------
# What to nudge, with what timescale
# ---------------------------------------------------------------------------


@tree_math.struct
class NudgingConfig:
    """Per-variable, per-level inverse relaxation timescales (1 / s).

    Zero entries mean "no nudging" for that variable / level — that's how
    the common "winds above the PBL only" pattern is expressed: keep
    ``inv_tau_vorticity`` and ``inv_tau_divergence`` non-zero from the
    free troposphere upwards and zero below, and zero out everything else.

    All timescales are *nondimensional* under the Model's ``physics_specs``
    — the Model converts per-second values from the user-facing
    constructors before storing them so the spectral tendency math works
    in the same units the dycore uses.
    """

    inv_tau_vorticity: jnp.ndarray              # (nlev,)
    inv_tau_divergence: jnp.ndarray             # (nlev,)
    inv_tau_temperature: jnp.ndarray            # (nlev,)
    inv_tau_log_surface_pressure: jnp.ndarray   # scalar — ps has no level axis

    @classmethod
    def winds_only(cls, nlev: int, *, tau_seconds: float = 21600.0,
                   pbl_levels: int = 0, physics_specs=None) -> "NudgingConfig":
        """Nudge vorticity and divergence everywhere except the bottom
        ``pbl_levels`` layers; leave temperature and surface pressure free.

        Args:
            nlev: Number of vertical levels.
            tau_seconds: Relaxation timescale in seconds (default 6 h).
            pbl_levels: Number of levels at the *bottom* of the column
                (highest sigma) where wind nudging is suppressed. Default
                0 (nudge all levels).
            physics_specs: Required — used to nondimensionalise ``tau``.
                Pass ``Model.primitive.physics_specs``.

        """
        if physics_specs is None:
            raise ValueError("`physics_specs` is required so τ can be nondimensionalised "
                             "consistently with the dycore.")
        tau_nd = physics_specs.nondimensionalize(tau_seconds * units.second)
        inv_tau = 1.0 / tau_nd

        # Convention: level 0 is TOA, level nlev-1 is the surface.
        mask = jnp.ones(nlev).at[nlev - pbl_levels:].set(0.0) if pbl_levels else jnp.ones(nlev)
        inv_tau_winds = inv_tau * mask
        zero_lev = jnp.zeros(nlev)
        return cls(
            inv_tau_vorticity=inv_tau_winds,
            inv_tau_divergence=inv_tau_winds,
            inv_tau_temperature=zero_lev,
            inv_tau_log_surface_pressure=jnp.array(0.0),
        )


# ---------------------------------------------------------------------------
# Tendency
# ---------------------------------------------------------------------------


def nudging_tendency(state: State, target: NudgingTarget,
                     config: NudgingConfig) -> State:
    """Newtonian relaxation tendency in spectral space.

    For each relaxed variable: ``dX/dt = inv_tau · (X_ref − X)``. Inside
    the dycore time integration this is composed with the primitive-
    equation tendencies and the physics tendencies.

    Args:
        state: Current dynamics state (modal).
        target: Reference fields (modal). Use ``target.select(date)``
            *before* calling this for time-varying references.
        config: Per-variable inverse-tau profiles.

    Returns:
        A ``State`` whose fields are the relaxation tendencies. Variables
        with zero ``inv_tau`` get zero tendency.

    """
    # Per-level inverse-tau broadcasts across the spectral (m, n) axes.
    def _level_relax(inv_tau_lev, x, x_ref):
        return inv_tau_lev[:, jnp.newaxis, jnp.newaxis] * (x_ref - x)

    vor_t = _level_relax(config.inv_tau_vorticity, state.vorticity, target.vorticity)
    div_t = _level_relax(config.inv_tau_divergence, state.divergence, target.divergence)
    temp_t = _level_relax(
        config.inv_tau_temperature, state.temperature_variation, target.temperature_variation,
    )
    log_sp_t = config.inv_tau_log_surface_pressure * (
        target.log_surface_pressure - state.log_surface_pressure
    )

    # Tracers are not nudged (they're physics-package-specific). Pass
    # zero tendencies to keep the State pytree shape consistent.
    tracer_zeros = {name: jnp.zeros_like(t) for name, t in state.tracers.items()}

    return State(
        vorticity=vor_t,
        divergence=div_t,
        temperature_variation=temp_t,
        log_surface_pressure=log_sp_t,
        tracers=tracer_zeros,
    )


# ---------------------------------------------------------------------------
# Top-level container the Model accepts
# ---------------------------------------------------------------------------


class Nudging:
    """User-facing handle bundling the relaxation target and config.

    Build once, pass to ``Model(..., nudging=...)``; the Model takes care
    of slicing the (possibly time-varying) target and adding the
    relaxation tendency to the dycore time integration.
    """

    def __init__(self, target: NudgingTarget, config: NudgingConfig):
        """Initialize the Nudging container."""
        self.target = target
        self.config = config

    def tendency(self, state: State, date: DateData,
                 calendar: str = DEFAULT_CALENDAR) -> State:
        """Compute the per-step nudging tendency. Slices the target via
        ``target.select(date, calendar)`` so static and time-varying
        references take the same code path.
        """
        target_now = self.target.select(date, calendar=calendar)
        return nudging_tendency(state, target_now, self.config)
