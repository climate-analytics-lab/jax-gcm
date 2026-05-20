"""Dinosaur-backed implementation of the :class:`DynamicalCore` protocol.

Wraps the spectral primitive-equations dycore from the external ``dinosaur``
package. Owns the IMEX-RK SIL3 step, the three diffusion filter closures,
the global-mean ps-conservation filter, the modal-orography truncation,
and the gridpoint↔modal conversions. Outside this subpackage the rest of
jax-gcm only sees the gridpoint :class:`PhysicsState` projection.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import dinosaur
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.filtering import horizontal_diffusion_filter
from dinosaur.hybrid_coordinates import HybridCoordinates
from dinosaur.primitive_equations import State
from dinosaur.scales import SI_SCALE, units

from jcm.constants import p0
from jcm.diffusion import DiffusionFilter, level_dependent_scaling
from jcm.dycore.base import DynamicalCore, Predictions
from jcm.dycore.dinosaur.state_bridge import (
    dynamics_state_to_physics_state,
    physics_state_to_dynamics_state,
    physics_tendency_to_dynamics_tendency,
)
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale=SI_SCALE)


class DinosaurDycore(DynamicalCore):
    """Spectral dynamical core backed by the ``dinosaur`` package.

    Args:
        coords: A dinosaur :class:`CoordinateSystem`.
        terrain: Boundary conditions (orography, land-sea mask, SSO descriptors).
        dt_seconds: Integration timestep in seconds.
        tracer_specs: Mapping ``name -> TracerSpec`` declaring every tracer the
            attached physics package needs. Used to seed the initial state and
            to drive the nondimensionalisation flag in
            :mod:`jcm.dycore.dinosaur.state_bridge`. May be ``None`` for
            dry-only runs. The user-facing :class:`jcm.model.Model` writes
            this every time it is constructed.
        diffusion: :class:`DiffusionFilter` describing horizontal hyperdiffusion
            scaling. Defaults to :meth:`DiffusionFilter.default`.

    """

    def __init__(
        self,
        coords: CoordinateSystem,
        terrain: TerrainData,
        dt_seconds: float,
        *,
        tracer_specs: Mapping[str, Any] | None = None,
        diffusion: DiffusionFilter | None = None,
    ):
        """Initialise the dinosaur backend; see the class docstring for argument semantics."""
        self.coords = coords
        self.terrain = terrain
        self.dt_seconds = float(dt_seconds)
        self.diffusion = diffusion or DiffusionFilter.default()
        self.tracer_specs = dict(tracer_specs) if tracer_specs else {}

        # Nondimensional timestep used throughout the dinosaur path.
        self._physics_specs = PHYSICS_SPECS
        self._dt_si = (self.dt_seconds * units.second).to(units.second)
        self._dt = self._physics_specs.nondimensionalize(self._dt_si)

        # Build the dycore's primitive-equations operator + initial-state
        # helper. The reference-temperature profile that comes out of
        # ``isothermal_rest_atmosphere`` is what dinosaur's hybrid / sigma
        # State expects in ``temperature_variation``.
        self._default_state_fn, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self._physics_specs,
            p0=p0 * units.pascal,
            p1=0.01 * p0 * units.pascal,
        )

        # Orography is truncated against the spectral basis here — the SE
        # backend (pyses) will project against its own basis instead.
        self._truncated_orography = primitive_equations.truncated_modal_orography(
            self.terrain.orog, self.coords, wavenumbers_to_clip=2,
        )

        # Dispatch on the vertical-coordinate family. Hybrid coords carry
        # ``a_boundaries`` in Pa; tell the dycore to interpret ``hpa_quantity``
        # accordingly. Hybrid is the only family that currently accepts a
        # ``humidity_key`` (q ↔ Tv coupling).
        if isinstance(self.coords.vertical, HybridCoordinates):
            self._primitive = primitive_equations.PrimitiveEquationsHybrid(
                reference_temperature=aux_features[dinosaur.xarray_utils.REF_TEMP_KEY],
                orography=self._truncated_orography,
                coords=self.coords,
                physics_specs=self._physics_specs,
                hpa_quantity=units.pascal,
                humidity_key='specific_humidity',
            )
        else:
            self._primitive = primitive_equations.PrimitiveEquations(
                reference_temperature=aux_features[dinosaur.xarray_utils.REF_TEMP_KEY],
                orography=self._truncated_orography,
                coords=self.coords,
                physics_specs=self._physics_specs,
            )

        self._filters = self._build_filters()
        self._dynamics_step_fn = self._build_dynamics_step_fn()

    # ------------------------------------------------------------------
    # ABC properties / metadata
    # ------------------------------------------------------------------

    @property
    def primitive(self) -> primitive_equations.PrimitiveEquations:
        """The wrapped dinosaur primitive-equations operator.

        Exposed so that callers that legitimately need the dinosaur-side
        object (e.g. nudging-target construction reading
        ``primitive.reference_temperature``) don't have to reach through a
        private attribute. Not part of the :class:`DynamicalCore` protocol.
        """
        return self._primitive

    @property
    def physics_specs(self) -> primitive_equations.PrimitiveEquationsSpecs:
        """The dinosaur physics specs (SI nondimensionalisation)."""
        return self._physics_specs

    @property
    def dt_nondim(self):
        """Nondimensional timestep used by the dinosaur integrator."""
        return self._dt

    @property
    def dt_si(self):
        """Dimensional timestep as a pint quantity (seconds)."""
        return self._dt_si

    # ------------------------------------------------------------------
    # Filter construction (lifted from Model._make_diffusion_fn)
    # ------------------------------------------------------------------

    def _conserve_global_mean_ps(self, u, u_next):
        return u_next.replace(
            log_surface_pressure=u_next.log_surface_pressure.at[0, 0, 0].set(
                u.log_surface_pressure[0, 0, 0],
            ),
        )

    def _make_diffusion_fn(self, timescale, order, replace_fn, level_orders=None):
        """Hyperdiffusion filter closure for one of the three state slots.

        Lifted unchanged from :meth:`jcm.model.Model._make_diffusion_fn` — the
        Phase-1 baseline asserts the bit-level invariance.
        """
        if level_orders is None:
            def diffusion_filter(u, u_next):
                eigenvalues = self.coords.horizontal.laplacian_eigenvalues
                scale = self._dt / (timescale * abs(eigenvalues[-1]) ** order)
                filter_fn = horizontal_diffusion_filter(self.coords.horizontal, scale, order)
                u_temp = filter_fn(u_next)
                return replace_fn(u_next, u_temp)
            return diffusion_filter

        eigenvalues = self.coords.horizontal.laplacian_eigenvalues
        scaling_const = np.asarray(level_dependent_scaling(
            eigenvalues, timescale, level_orders, self._dt,
        ))

        def diffusion_filter(u, u_next):
            def rescale(x):
                if not hasattr(x, "shape"):
                    return x
                target_shape = np.shape(x)
                if target_shape != np.broadcast_shapes(target_shape, scaling_const.shape):
                    return x
                return scaling_const * x
            u_temp = jax.tree_util.tree_map(rescale, u_next)
            return replace_fn(u_next, u_temp)
        return diffusion_filter

    def _build_filters(self):
        diffuse_div = self._make_diffusion_fn(
            self.diffusion.div_timescale,
            self.diffusion.div_order,
            replace_fn=lambda u_next, u_temp: u_next.replace(divergence=u_temp.divergence),
            level_orders=self.diffusion.level_orders_div,
        )
        diffuse_vor_q = self._make_diffusion_fn(
            self.diffusion.vor_q_timescale,
            self.diffusion.vor_q_order,
            replace_fn=lambda u_next, u_temp: u_next.replace(
                vorticity=u_temp.vorticity,
                tracers=dict(u_temp.tracers),
            ),
            level_orders=self.diffusion.level_orders_vor_q,
        )
        diffuse_temp = self._make_diffusion_fn(
            self.diffusion.temp_timescale,
            self.diffusion.temp_order,
            replace_fn=lambda u_next, u_temp: u_next.replace(
                temperature_variation=u_temp.temperature_variation,
            ),
            level_orders=self.diffusion.level_orders_temp,
        )
        return [
            self._conserve_global_mean_ps,
            diffuse_div,
            diffuse_vor_q,
            diffuse_temp,
        ]

    # ------------------------------------------------------------------
    # Dynamics step (IMEX-RK SIL3)
    # ------------------------------------------------------------------

    def _build_dynamics_step_fn(self):
        """Build the IMEX-RK SIL3 step over pure dynamics.

        The op-split caller adds the physics dynamics-tendency to the state
        forward-Euler-style before invoking this; the integrator advances
        ``state.sim_time`` by ``dt`` and applies the implicit-explicit RK
        stages.
        """
        return dinosaur.time_integration.imex_rk_sil3(self._primitive, self._dt)

    # ------------------------------------------------------------------
    # DynamicalCore protocol implementation
    # ------------------------------------------------------------------

    def initial_state(
        self,
        physics_state: PhysicsState | None,
        *,
        sim_time: float = 0.0,
        random_seed: int = 0,
        tracer_specs: Mapping[str, Any] | None = None,
    ) -> State:
        """Build a dinosaur :class:`State` to seed the integration.

        Identical semantics to :meth:`Model._prepare_initial_dycore_state`. If
        ``physics_state`` is provided it is round-tripped through
        :func:`physics_state_to_dynamics_state`; otherwise the
        ``isothermal_rest_atmosphere`` default state is used with a small
        per-cell pressure perturbation seeded by ``random_seed``.
        """
        specs = dict(tracer_specs) if tracer_specs is not None else dict(self.tracer_specs)

        if physics_state is not None:
            state = physics_state_to_dynamics_state(
                physics_state, self._primitive, tracer_specs=specs,
            )
        else:
            state = self._default_state_fn(jax.random.PRNGKey(random_seed))
            # Sigma coords store ``log(P_s / p0)``; hybrid coords store
            # ``log(P_s in Pa)`` directly. Normalize only on the sigma path.
            if not isinstance(self.coords.vertical, HybridCoordinates):
                state.log_surface_pressure = self.coords.horizontal.to_modal(
                    self.coords.horizontal.to_nodal(state.log_surface_pressure)
                    - jnp.log(self._physics_specs.nondimensionalize(p0 * units.pascal))
                )
            state.tracers = {
                'specific_humidity': 0.0 * primitive_equations_states.gaussian_scalar(
                    self.coords, self._physics_specs,
                ),
            }

        # Seed any required tracers not already present in ``state.tracers``.
        for spec in specs.values():
            if spec.name in state.tracers:
                continue
            state.tracers[spec.name] = (
                spec.initial_value
                * jnp.ones_like(state.tracers['specific_humidity'])
            )

        return State(**state.asdict(), sim_time=sim_time)

    def to_physics_state(self, state: State) -> PhysicsState:
        return dynamics_state_to_physics_state(
            state, self._primitive, tracer_specs=self.tracer_specs,
        )

    def step(
        self,
        state: State,
        physics_tendency: PhysicsTendency | None,
    ) -> State:
        """Advance ``state`` by one ``dt``.

        Order: forward-Euler add of the physics dynamics-tendency →
        IMEX-RK SIL3 dynamics step → spectral filters.
        """
        if physics_tendency is not None:
            dyn_tendency = physics_tendency_to_dynamics_tendency(
                physics_tendency, self._primitive, tracer_specs=self.tracer_specs,
            )
            state_after_physics = state + self._dt * dyn_tendency
        else:
            state_after_physics = state
        state_after_dyn = self._dynamics_step_fn(state_after_physics)
        state_next = state_after_dyn
        for f in self._filters:
            state_next = f(state, state_next)
        return state_next

    def sim_time(self, state: State) -> jnp.ndarray:
        return state.sim_time

    def with_sim_time(self, state: State, sim_time) -> State:
        return State(**state.asdict(), sim_time=sim_time)

    # ------------------------------------------------------------------
    # Output & terrain (Phase-1 thin shims; full relocation in a follow-up)
    # ------------------------------------------------------------------

    def to_xarray(self, predictions: Predictions, times, *, additional_coords=None):
        """Convert a saved trajectory to an :class:`xarray.Dataset`.

        Phase-1 implementation delegates to :func:`jcm.utils.data_to_xarray`
        unchanged — the modal-axis dispatch in
        :func:`jcm.utils._infer_dims_shape_and_coords` still runs against the
        dinosaur ``CoordinateSystem``. A subsequent PR moves that logic in
        here so a future cubed-sphere backend can supply its own version
        without monkey-patching ``utils``.
        """
        # Avoid the otherwise-circular import (utils does not currently depend
        # on dycore, but a top-level import here would still be fine; deferred
        # to keep import-time cost on this module low).
        from jcm.utils import data_to_xarray

        return data_to_xarray(
            predictions.dynamics.asdict() | predictions.physics,
            coords=self.coords,
            serialize_coords_to_attrs=False,
            times=times - times[0],
            additional_coords=additional_coords or {},
        )

    def build_terrain(self, *, source_file=None, **kwargs) -> TerrainData:
        """Construct a :class:`TerrainData` against the dinosaur basis.

        Phase-1 implementation forwards to :class:`TerrainData` classmethods.
        The dinosaur-flavoured spectral truncation already lives in
        ``TerrainData.from_coords``/``from_file`` today; a subsequent PR will
        move that logic in here so the symmetric pyses backend can do its own
        SE projection in its own ``build_terrain``.
        """
        if source_file is None:
            return TerrainData.aquaplanet(self.coords)
        envelope = kwargs.pop("orog_envelope_wavenumber", None)
        if envelope is not None:
            return TerrainData.from_file(
                source_file, coords=self.coords,
                orog_envelope_wavenumber=envelope, **kwargs,
            )
        return TerrainData.from_coords(
            self.coords, terrain_file=source_file, **kwargs,
        )

    def required_tracers_ok(self, specs: Sequence[Any]) -> None:
        # No native restriction; dinosaur can carry any TracerSpec the
        # physics package declares.
        return None
