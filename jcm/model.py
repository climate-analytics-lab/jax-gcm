import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import tree_math
import jax_datetime as jdt
from numpy import timedelta64
import dinosaur
from typing import Callable, Any
from dinosaur import typing
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.constants import p0
from jcm.terrain import TerrainData
from jcm.date import DateData, parse_duration_days
from jcm.forcing import ForcingData, default_forcing
from jcm.nudging import Nudging
from jcm.physics_interface import PhysicsState, Physics, dynamics_state_to_physics_state, compute_physics_step
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.utils import DYNAMICS_UNITS_TABLE_CSV_PATH
from jcm.diffusion import DiffusionFilter
import pandas as pd
from functools import partial
import logging

# logging.basicConfig(format='%(name)s: %(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

@tree_math.struct
class Predictions:
    """Internal container for model prediction outputs (JAX pytree).

    This is the internal pytree-compatible struct used during JAX transformations
    (scan, tree_map, etc.). Users should interact with ModelPredictions instead,
    which wraps this struct and provides to_xarray() conversion.

    Attributes:
        dynamics (PhysicsState): The physical state variables converted from
            the dynamical state.
        physics (Any): Diagnostic physics data computed by the physics package.
        times (Any): Timestamps of the predictions.

    """

    dynamics: PhysicsState
    physics: Any
    times: Any


class ModelPredictions:
    """User-facing container for model prediction outputs.

    Wraps the internal Predictions pytree with the coordinate system and
    physics module needed for xarray conversion. Returned by Model.run(),
    Model.resume(), and Model.run_from_state().

    Attributes:
        dynamics (PhysicsState): The physical state variables.
        physics (Any): Diagnostic physics data.
        times (Any): Timestamps of the predictions.

    """

    def __init__(self, predictions: Predictions, coords: 'CoordinateSystem', physics: Physics):  # noqa: D107
        self._predictions = predictions
        self._coords = coords
        self._physics = physics

    @property
    def dynamics(self):
        return self._predictions.dynamics

    @property
    def physics(self):
        return self._predictions.physics

    @property
    def times(self):
        return self._predictions.times

    def to_xarray(self):
        """Convert the full prediction trajectory to an xarray.Dataset.

        Returns:
            An xarray.Dataset ready for analysis and plotting.

        """
        from jcm.utils import data_to_xarray


        # float0s are placeholders representing the lack of tangent space for non-differentiable variables
        # jax.numpy arrays cannot have float0 dtype, so jcm handles them with numpy arrays
        # substituting jax.numpy arrays here allows us to handle Predictions objects that contain derivatives
        float0s_to_nans = lambda pytree: tree_map(lambda x: jnp.full_like(x, jnp.nan, dtype=float) if x.dtype == jax.dtypes.float0 else x, pytree)

        dynamics_predictions = float0s_to_nans(self.dynamics)
        physics_predictions = float0s_to_nans(self.physics)

        nodal_shape = dynamics_predictions.u_wind.shape[1:]

        # prepare physics predictions for xarray conversion
        physics_preds_dict = self._physics.data_struct_to_dict(physics_predictions, nodal_shape=nodal_shape)

        times = jax.device_get(self.times)
        coords = jax.device_get(self._coords)

        # get additional coords from physics-specific cached coords (e.g. SpeedyCoords, EchamCoords)
        additional_coords = {}
        if self._physics.cached_coords is not None and hasattr(self._physics.cached_coords, 'xarray_additional_coords'):
            additional_coords = self._physics.cached_coords.xarray_additional_coords()

        pred_ds = data_to_xarray(dynamics_predictions.asdict() | physics_preds_dict,
                                 coords=coords, serialize_coords_to_attrs=False,
                                 times=times - times[0],
                                 additional_coords=additional_coords)

        # Import units attribute associated with each xarray output from units_table.csv
        units_df = pd.read_csv(DYNAMICS_UNITS_TABLE_CSV_PATH)
        if self._physics.UNITS_TABLE_CSV_PATH is not None:
            units_df = pd.concat([units_df, pd.read_csv(self._physics.UNITS_TABLE_CSV_PATH)], ignore_index=True)
        for var, unit, desc in zip(units_df["Variable"], units_df["Units"], units_df["Description"]):
            if var in pred_ds:
                pred_ds[var].attrs["units"] = unit
                pred_ds[var].attrs["description"] = desc

        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # convert time in days to datetime
        pred_ds['time'] = (
            times*(timedelta64(1, 'D')/timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')

        return pred_ds


def _model_predictions_flatten(mp):
    """Flatten ModelPredictions for JAX pytree operations (tree_map, etc.).

    Only the internal Predictions pytree is treated as array data.
    Coords and physics are not included in aux_data so that tree_map
    works across ModelPredictions from different Model instances.
    """
    children = (mp._predictions,)
    return children, None


def _model_predictions_unflatten(aux_data, children):
    return ModelPredictions(children[0], None, None)


jax.tree_util.register_pytree_node(
    ModelPredictions,
    _model_predictions_flatten,
    _model_predictions_unflatten,
)


def _op_split_trajectory(
    step_fn: Callable[[Any, Any], tuple[Any, Any]],
    initial_physics_state: Any,
    empty_diagnostics: Any,
    outer_steps: int,
    inner_steps: int,
    post_process_fn: Callable[[Any, Any], Any] = lambda x, ps: x,
    output_averages: bool = False,
) -> Callable[[Any], tuple[Any, Any, Any]]:
    """Trajectory builder for the operator-split path.

    The op-split ``step_fn`` has signature ``(state, physics_state) ->
    (state_next, physics_state_next)``. ``physics_state`` is the
    cross-step physics carry (radiation flux for sub-cycling, prev
    TKE for the analytic source update, etc.) and flows through the
    ``lax.scan`` as a first-class pytree.

    ``post_process_fn`` takes ``(state, physics_state)``. In snapshot
    mode the saved physics carry is exactly the one used by the
    integration — radiation sub-cycle cache, TKE memory, etc. — so
    diagnostics reported in ``predictions.physics`` match what the
    dycore actually consumed.

    In averaged mode, the per-step physics dict (the same dict that
    becomes ``physics_state_next``) is accumulated as a running mean
    across the inner steps and saved per outer step. The running mean
    uses POST-step states (``x_next``); this matches the snapshot
    path's end-of-step samples, so ``mean(snapshots)`` and
    ``averaged(...)`` agree to numerical roundoff.

    Args:
        step_fn: Operator-split per-``dt`` step.
        initial_physics_state: Cross-step carry initial value (built
            via :meth:`ComposablePhysics.initial_carry_state` unioned
            with a structural template from
            :meth:`ComposablePhysics.get_empty_data`).
        empty_diagnostics: Zero-shaped diagnostics dict used to seed
            the running-mean accumulator in averaged mode. Same
            structure as the per-step ``physics_state_next``.
        outer_steps: Number of saved frames.
        inner_steps: Inner ``dt`` steps between saved frames.
        post_process_fn: Applied to the state at save time (snapshot
            mode) or to the running mean (averaged mode).
        output_averages: When True, the saved frame is the running
            mean of ``post_process_fn(state)`` over the inner steps.

    Returns:
        A function ``initial_state -> (final_state, final_physics_state,
        saved_trajectory)`` where ``saved_trajectory`` has a leading
        axis of length ``outer_steps``. ``final_physics_state`` is the
        cross-step carry coming out of the last ``dt`` — exposing it
        lets callers (e.g. ``Model.resume``) thread a continuous carry
        across API boundaries so a 5d + resume(5d) integration matches
        a single 10d integration. In averaged mode the returned
        trajectory's ``physics`` field is the time-averaged diagnostics
        dict.

    """
    # Snapshot and averaged modes only differ in what the inner scan
    # accumulates and what the outer step saves; the surrounding outer
    # ``lax.scan`` over ``(state, physics_state)`` and the
    # ``(x_final, ps_final, preds)`` return are identical, so define
    # them once.
    def _averaged_outer_step():
        @jax.checkpoint
        def inner_step(carry, _):
            x, physics_state, x_sum, diag_sum = carry
            x_next, physics_state_next = step_fn(x, physics_state)
            # Sum POST-step states so that mean(state_1..state_N)
            # matches the snapshot path (which saves state_N at outer
            # steps). Summing pre-step states would be off by one
            # timestep — tolerable for slow fields, but the op-split
            # per-step transient is large enough to surface as test
            # failures at the rtol=1e-3 the averaging test runs at.
            x_sum = tree_map(lambda a, b: a + b, x_sum, x_next)
            diag_sum = tree_map(
                lambda acc, new: acc + new / inner_steps,
                diag_sum, physics_state_next,
            )
            return (x_next, physics_state_next, x_sum, diag_sum), None

        def outer_step(carry, _, empty_sum, empty_diag_sum):
            x, physics_state = carry
            init = (x, physics_state, empty_sum, empty_diag_sum)
            (x_next, ps_next, x_sum, diag_sum), _ = jax.lax.scan(
                inner_step, init, None, length=inner_steps,
            )
            averaged_state = tree_map(lambda s: s / inner_steps, x_sum)
            preds = post_process_fn(averaged_state, ps_next)
            # Attach this outer-step's mean diagnostics dict to the
            # Predictions saved for the frame. Stacked along the
            # outer-step leading axis by the surrounding scan.
            preds = preds.replace(physics=diag_sum)
            return (x_next, ps_next), preds

        return outer_step

    def _snapshot_outer_step():
        @jax.checkpoint
        def inner_step(carry, _):
            x, physics_state = carry
            x_next, physics_state_next = step_fn(x, physics_state)
            return (x_next, physics_state_next), None

        def outer_step(carry, _):
            (x_final, ps_final), _ = jax.lax.scan(
                inner_step, carry, None, length=inner_steps,
            )
            # Save the carried physics state alongside the dynamics
            # state. Calling ``post_process_fn`` with ``ps_final``
            # lets snapshot diagnostics reflect the sub-cycled
            # radiation cache / TKE memory the dycore actually
            # consumed — recomputing physics at save time with a
            # freshly-seeded carry would zero out radiation on
            # non-radiation outer steps (default 2-hour
            # ``radiation_interval``).
            return (x_final, ps_final), post_process_fn(x_final, ps_final)

        return outer_step

    def integrate(x_initial):
        if output_averages:
            empty_sum = tree_map(jnp.zeros_like, x_initial)
            # Cast accumulator leaves to float so that ``acc + new /
            # N`` doesn't promote dtype mid-scan — jax.lax.scan
            # rejects type changes in the carry.
            empty_diag_sum = tree_map(
                lambda x: jnp.zeros(jnp.shape(x), dtype=float),
                empty_diagnostics,
            )
            outer_step_fn = _averaged_outer_step()
            outer_step = lambda c, _: outer_step_fn(
                c, _, empty_sum, empty_diag_sum,
            )
        else:
            outer_step = _snapshot_outer_step()

        (x_final, ps_final), preds = jax.lax.scan(
            outer_step,
            (x_initial, initial_physics_state),
            None, length=outer_steps,
        )
        return x_final, ps_final, preds

    return integrate


class Model:
    """Top level class for a JAX-GCM configuration using the Speedy physics on an aquaplanet."""

    def __init__(self, coords: CoordinateSystem, time_step=30.0, terrain: TerrainData=None,
                 physics: Physics=None, diffusion: DiffusionFilter=None,
                 start_date: jdt.Datetime=jdt.to_datetime('2000-01-01'),
                 calendar: str = "365_day",
                 nudging: Nudging | None = None,
                 radiation_chunk_size: int | None = None,
                 log_level=logging.CRITICAL) -> None:
        """Initialize the model with the given time step, save interval, and total time.

        Args:
            coords:
                CoordinateSystem object describing the model coordinates. To enable SPMD
                parallelization, pass ``spmd_mesh`` to the coords helper (e.g. ``get_speedy_coords``).
            time_step:
                Model time step in minutes
            terrain:
                TerrainData object describing boundary conditions (orography, land-sea mask, etc.)
            physics:
                Physics object describing the model physics
            diffusion:
                DiffusionFilter object describing horizontal diffusion filter params
            start_date:
                jax_datetime.Datetime object containing start date of the simulation (default January 1, 2000)
            calendar:
                Calendar used for `tyear` / `model_year` / forcing-axis alignment.
                ``"365_day"`` (no leap years) matches SPEEDY's climatology
                tables and solar lookup; ``"gregorian"`` (365.2425 days/year)
                is available for ICON-style runs that align against real
                Gregorian timestamps in their forcing files.
            nudging:
                Optional :class:`jcm.nudging.Nudging` handle. When provided,
                the dycore time integration is augmented with a Newtonian
                relaxation tendency toward the configured reference state
                (``dX/dt|_nudge = (X_ref − X) / τ``). Physics-agnostic — it
                acts on the dycore state, so the same nudging works under
                SPEEDY, ICON, or any future physics package. Default
                ``None`` (no nudging, no extra cost).
            radiation_chunk_size:
                Override the RRTMGP chunked-vmap chunk size (cells per
                chunk). Default ``None`` auto-detects from the JAX
                device's HBM (largest chunk that fits at ~55 % of the
                XLA bytes_limit). Set to a positive integer to fix the
                chunk count — useful on shared GPUs with reduced free
                memory or for reproducible kernel launches. Has no
                effect when the radiation backend is not RRTMGP.
            log_level:
                (int) indicates what level of messages will be output, use logging.INFO (20) for verbose (defaults logging.CRITICAL)

        """
        # Set root logging level to be log_level so it propagates to other modules
        logging.getLogger().setLevel(log_level)
        self.calendar = calendar
        self.nudging = nudging

        # Wire the RRTMGP chunked-vmap chunk-size override (only takes
        # effect if the physics actually uses RRTMGP — the setter is a
        # no-op for other radiation backends). ``None`` means auto-detect.
        if radiation_chunk_size is not None:
            from jcm.physics.radiation import rrtmgp as _rrtmgp_mod
            _rrtmgp_mod.set_chunk_size(radiation_chunk_size)

        self.physics_specs = PHYSICS_SPECS
        self.dt_si = (time_step * units.minute).to(units.second)
        self.dt = self.physics_specs.nondimensionalize(self.dt_si)

        # Store coords - used by dynamics and physics
        self.coords = coords

        # Store terrain (boundary conditions)
        self.terrain = terrain if terrain is not None else TerrainData.aquaplanet(self.coords)

        # Get the reference temperature and orography. This also returns the initial state function (if wanted to start from rest)
        self.default_state_fn, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self.physics_specs,
            p0=p0*units.pascal,
            p1=0.01*p0*units.pascal,
        )
        
        self.physics = physics or speedy_physics()
        self.physics.cache_coords(self.coords)
        # Hand the model's timestep to the physics. ``ComposablePhysics``
        # injects it into the diagnostics dict every step under
        # ``"_dt_seconds"`` so any term that integrates by ``dt``
        # (chemistry, microphysics, vertical diffusion, …) reads a
        # single source of truth instead of going through date plumbing.
        if hasattr(self.physics, "dt_seconds"):
            self.physics.dt_seconds = float(self.dt_si.m)

        self.diffusion = diffusion or DiffusionFilter.default()

        # TODO: make the truncation number a parameter consistent with the grid shape
        self.truncated_orography = primitive_equations.truncated_modal_orography(self.terrain.orog, self.coords, wavenumbers_to_clip=2)

        # Dispatch to sigma vs hybrid primitive-equations class based on the
        # vertical coordinate type (`PrimitiveEquations` defaults to sigma).
        # ICON-style hybrid coords store `a_boundaries` in Pa, so we override
        # `hpa_quantity` so dinosaur's internal nondimensionalization treats
        # them as Pa (not hPa which is the dinosaur default).
        # Wire up the ``specific_humidity`` tracer as the dynamics' humidity
        # so that moisture contributes to virtual temperature in the dycore
        # (and the moisture-related divergence/vorticity corrections fire).
        # Without this, q is transported as an inert passive tracer and the
        # dynamics are effectively dry — inconsistent with the moist physics.
        from dinosaur.hybrid_coordinates import HybridCoordinates
        if isinstance(self.coords.vertical, HybridCoordinates):
            self.primitive = primitive_equations.PrimitiveEquationsHybrid(
                reference_temperature=aux_features[dinosaur.xarray_utils.REF_TEMP_KEY],
                orography=self.truncated_orography,
                coords=self.coords,
                physics_specs=self.physics_specs,
                hpa_quantity=units.pascal,
                humidity_key='specific_humidity',
            )
        else:
            # Sigma-coord dinosaur ``PrimitiveEquations`` does not accept
            # ``humidity_key``; moisture-Tv coupling is only available on the
            # hybrid variant in this version.
            self.primitive = primitive_equations.PrimitiveEquations(
                reference_temperature=aux_features[dinosaur.xarray_utils.REF_TEMP_KEY],
                orography=self.truncated_orography,
                coords=self.coords,
                physics_specs=self.physics_specs,
            )
        
        def conserve_global_mean_surface_pressure(u, u_next):
            return u_next.replace(
                # prevent global mean (0th spectral component) surface pressure drift by setting it to its value before timestep
                log_surface_pressure=u_next.log_surface_pressure.at[0, 0, 0].set(u.log_surface_pressure[0, 0, 0])
            )
        
        # create diffusion filter function handles
        diffuse_div = self._make_diffusion_fn(
            self.diffusion.div_timescale,
            self.diffusion.div_order,
            replace_fn=lambda u_next, u_temp: u_next.replace(divergence=u_temp.divergence),
            level_orders=self.diffusion.level_orders_div,
        )

        diffuse_vor_q = self._make_diffusion_fn(
            self.diffusion.vor_q_timescale,
            self.diffusion.vor_q_order,
            # Apply the filter to vorticity and every tracer (specific_humidity + any
            # extras like qc/qi/qnc). Keeping only specific_humidity silently zeros
            # microphysics tracers over time.
            replace_fn=lambda u_next, u_temp: u_next.replace(
                vorticity=u_temp.vorticity,
                tracers=dict(u_temp.tracers),
            ),
            level_orders=self.diffusion.level_orders_vor_q,
        )

        diffuse_temp = self._make_diffusion_fn(
            self.diffusion.temp_timescale,
            self.diffusion.temp_order,
            replace_fn=lambda u_next, u_temp: u_next.replace(temperature_variation=u_temp.temperature_variation),
            level_orders=self.diffusion.level_orders_temp,
        )

        self.filters = [
            conserve_global_mean_surface_pressure,
            diffuse_div,
            diffuse_vor_q,
            diffuse_temp,
        ]

        self.start_date = start_date

        # grid space PhysicsState set upon calling model.run
        self.initial_nodal_state = None

        # spectral space primitive_equations.State updated by model.run and model.resume
        self._final_modal_state = None

        # Cross-step physics carry threaded through op-split run/resume.
        # ``None`` means "build a fresh carry on the next call" — set
        # to ``self._build_initial_physics_carry()`` lazily inside
        # ``run_from_state`` (see Issue #471 P1). Holding the final
        # carry on the model is what makes a ``run() + resume()``
        # bisection numerically equivalent to a single ``run()`` of
        # the combined duration; sub-cycled radiation, prior-step TKE,
        # etc. would otherwise reset at every API seam.
        self._final_physics_state = None
    
    def _make_diffusion_fn(self, timescale, order, replace_fn, level_orders=None):
        """Return a diffusion filter closure for one of the three state slots.

        Args:
            timescale: base hyperdiffusion timescale (s).
            order: uniform-order spectral power (used when level_orders is None).
            replace_fn: picks which state variables get overwritten by the filter.
            level_orders: optional 1-D array of per-level orders (length nlev)
                enabling the ECHAM-style level-dependent hyperdiffusion.

        """
        from dinosaur.filtering import horizontal_diffusion_filter
        from jcm.diffusion import level_dependent_scaling
        import jax

        if level_orders is None:
            def diffusion_filter(u, u_next):
                eigenvalues = self.coords.horizontal.laplacian_eigenvalues
                scale = self.dt / (timescale * abs(eigenvalues[-1]) ** order)
                filter_fn = horizontal_diffusion_filter(self.coords.horizontal, scale, order)
                u_temp = filter_fn(u_next)
                return replace_fn(u_next, u_temp)
            return diffusion_filter

        import numpy as np
        # Precompute the scaling once (pure constant, not traced). This avoids
        # JIT-time issues observed when the 3-D scaling was rebuilt inside the
        # filter closure at high hyperdiffusion orders.
        eigenvalues = self.coords.horizontal.laplacian_eigenvalues
        scaling_const = np.asarray(level_dependent_scaling(
            eigenvalues, timescale, level_orders, self.dt,
        ))  # (nlev, 1, lat_modes), numpy array → inlined as JIT constant

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
    
    def _prepare_initial_modal_state(self, physics_state: PhysicsState=None, random_seed=0, sim_time=0.0, humidity_perturbation=False) -> primitive_equations.State:
        """Prepare initial dinosaur.primitive_equations.State for a model run.

        Args:
            physics_state:
                Optional nodal PhysicsState from which to generate the modal state. If none provided, initial state will be isothermal atmosphere with random noise surface pressure perturbation.
            random_seed:
                Seed for pressure perturbation (default 0).
            sim_time:
                Optionally specify the sim_time attribute for the state (default 0.0).
            humidity_perturbation:
                If True and using the default state, adds a horizontally localized perturbation to specific humidity.

        Returns:
            A `primitive_equations.State` object ready for integration.

        """
        from jcm.physics_interface import physics_state_to_dynamics_state

        tracer_specs = {spec.name: spec for spec in self.physics.required_tracers()}

        # Either use the designated initial state, or generate one. The initial state to the dycore is a modal primitive_equations.State,
        # but the optional initial state from the user is a nodal PhysicsState
        if physics_state is not None:
            state = physics_state_to_dynamics_state(physics_state, self.primitive, tracer_specs=tracer_specs)
        else:
            state = self.default_state_fn(jax.random.PRNGKey(random_seed))
            # For sigma coords, we want log(P_s / p0) so that `exp(log_sp)` gives
            # the normalized surface pressure ≈ 1. For hybrid coords, the dynamics
            # combines a_boundaries (in Pa) with exp(log_sp), so log_sp must be
            # `log(P_s_actual_in_Pa)` — no normalization by p0.
            from dinosaur.hybrid_coordinates import HybridCoordinates
            if not isinstance(self.coords.vertical, HybridCoordinates):
                state.log_surface_pressure = self.coords.horizontal.to_modal(
                    self.coords.horizontal.to_nodal(state.log_surface_pressure)
                    - jnp.log(self.physics_specs.nondimensionalize(p0 * units.pascal))
                )

            # need to add specific humidity as a tracer
            state.tracers = {
                'specific_humidity': (1e-2 if humidity_perturbation else 0.0) * primitive_equations_states.gaussian_scalar(self.coords, self.physics_specs)
            }

        # Seed modal tracers for every TracerSpec the physics declares so that the
        # dynamics core advects them. Shape matches specific_humidity (modal).
        for spec in tracer_specs.values():
            if spec.name in state.tracers:
                continue
            state.tracers[spec.name] = (
                spec.initial_value
                * jnp.ones_like(state.tracers['specific_humidity'])
            )

        return primitive_equations.State(**state.asdict(), sim_time=sim_time)

    def _date_from_sim_time(self, sim_time) -> DateData:
        # Stop gradient: date/calendar computations use non-differentiable ops
        # (floor, round, int casts) and should not be part of the AD graph.
        sim_time = jax.lax.stop_gradient(sim_time)
        return DateData.set_date(
            model_time=self.start_date + jdt.Timedelta(
                days=jnp.floor(sim_time / 86400).astype(jnp.int32),
                seconds=jnp.round(sim_time % 86400).astype(jnp.int32)
            ),
            model_step=jnp.int32(sim_time / self.dt_si.m),
            dt_seconds=float(self.dt_si.m),
            calendar=self.calendar,
        )

    def _get_dynamics_step_fn(self) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
        """Build the IMEX-RK SIL3 step over pure dynamics.

        The operator-split path (issue #471) calls this once per ``dt``
        from inside ``_get_op_split_step_fn``, with physics tendencies
        already added forward-Euler to the state.

        Sponge and nudging stay inside the RK stages — they are stiff /
        fast-linear couplings that benefit from intermediate-state
        evaluation. Physics is the only thing that leaves the stage
        loop.

        """
        equations = [self.primitive]
        if self.nudging is not None:
            nudging_eqn = ExplicitODE.from_functions(
                lambda state: self.nudging.tendency(
                    state,
                    date=self._date_from_sim_time(state.sim_time),
                    calendar=self.calendar,
                )
            )
            equations.append(nudging_eqn)
        composed = dinosaur.time_integration.compose_equations(equations)
        return dinosaur.time_integration.imex_rk_sil3(composed, self.dt)

    def _get_op_split_step_fn(
        self, forcing: ForcingData,
    ) -> Callable[[primitive_equations.State, Any], tuple[primitive_equations.State, Any]]:
        """Build the operator-split single-step function (Lie split a).

        One call: ``state, physics_state -> state_next, physics_state_next``.

        Order: ``state -> physics_tendency -> apply forward-Euler ->
        IMEX-RK dynamics -> filters``. This is the structure described
        in ``docs/design/operator_split_physics.md`` and mirrors
        ECHAM6's ``physc`` → ``sccd``/``scctp`` → ``hdiff`` chain
        (`stepon.f90:271,280,309`).

        ``physics_state`` is the cross-step carry — the dict returned
        by the previous step's :meth:`ComposablePhysics.compute_tendencies`
        (or the initial value from
        :meth:`ComposablePhysics.initial_carry_state` on the first step).
        """
        dynamics_step = self._get_dynamics_step_fn()

        def step(state, physics_state):
            date = self._date_from_sim_time(state.sim_time)
            forcing_now = forcing.select(date, calendar=self.calendar)
            dyn_tendency, new_physics_state = compute_physics_step(
                state=state,
                dynamics=self.primitive,
                time_step=self.dt_si.m,
                physics=self.physics,
                forcing=forcing_now,
                terrain=self.terrain,
                physics_state=physics_state,
            )
            # Forward-Euler add of the physics dynamics tendency. The
            # dinosaur State is a tree_math.struct so + and * lift
            # leaf-wise; physics tendency has ``sim_time = 0`` so the
            # state's ``sim_time`` is not perturbed here — the dynamics
            # IMEX-RK below is what advances sim_time by ``dt``.
            state_after_physics = state + self.dt * dyn_tendency
            state_after_dynamics = dynamics_step(state_after_physics)
            # Run the same filters used in the legacy path. They receive
            # the pre-step state and the post-dynamics state, matching
            # the ``step_with_filters`` contract.
            state_next = state_after_dynamics
            for f in self.filters:
                state_next = f(state, state_next)
            return state_next, new_physics_state

        return step

    def _post_process(
        self,
        state: primitive_equations.State,
        physics_state: Any,
        output_averages: bool,
    ) -> Predictions:
        """Post-process a single saved state from the op-split trajectory.

        The op-split scan threads ``physics_state`` — the cross-step
        carry returned by the prior ``compute_tendencies`` call — into
        this function at save time. We use it directly as the
        ``predictions.physics`` payload in snapshot mode rather than
        re-running physics with a freshly-seeded carry. That avoids
        the bug where sub-cycled radiation diagnostics (default
        ``radiation_interval=7200s``) would be reported as zero on
        non-radiation outer steps because the recompute path didn't
        see the cached radiation fields the dycore was actually
        consuming.

        In averaged mode the caller overrides ``predictions.physics``
        with the inner-step running mean, so the value attached here
        is discarded — we leave it as ``physics_state`` for symmetry
        and to keep the pytree structure stable.

        Non-negative tracers (specific_humidity, qc/qi, GHG VMRs) get
        a final ``verify_state`` clamp at the modal→nodal output
        boundary so spectral Gibbs ringing of the physics tendency
        doesn't leak negative values into user-visible output. Cheap
        (one ``max``) and complementary to the ``verify_state`` that
        runs on the physics input side inside ``compute_physics_step``.

        """
        from jcm.physics_interface import verify_state
        jax.debug.callback(lambda t: logger.info("Post processing: %s simulated seconds", t), state.sim_time)

        tracer_specs = {spec.name: spec for spec in self.physics.required_tracers()}
        return Predictions(
            dynamics=verify_state(dynamics_state_to_physics_state(
                state, self.primitive, tracer_specs=tracer_specs,
            )),
            physics=physics_state if not output_averages else None,
            times=None,
        )

    def _build_initial_physics_carry(self) -> Any:
        """Build the cross-step physics carry seed for an op-split run.

        Pulls per-term initial state from
        :meth:`Physics.initial_carry_state` (deterministic, no
        zero-state probe). Unions with the *structural template*
        from :meth:`Physics.get_empty_data` so the ``lax.scan`` carry
        pytree matches the post-step ``compute_tendencies`` output
        structure on iteration 1 (within-step diagnostic keys terms
        write are zero-filled). ``get_empty_data`` is internal-only
        in this role: it discovers the post-step output pytree
        structure via a one-shot probe — never used as live state.
        """
        template = self.physics.get_empty_data(self.coords)
        initial_carry = self.physics.initial_carry_state(self.coords)
        if isinstance(initial_carry, dict) and isinstance(template, dict):
            return {**template, **initial_carry}
        # Explicit ``is None`` check: ``initial_carry or template``
        # would trigger ``bool(carry)`` and raise an ambiguous-truth
        # ``ValueError`` if a ``Physics`` subclass returns a JAX
        # array (or any object with non-scalar truth semantics).
        return template if initial_carry is None else initial_carry

    def _get_op_split_integrate_fn(
        self,
        step_fn,
        outer_steps,
        inner_steps,
        post_process_fn,
        output_averages,
    ):
        """Integrate-fn builder for the operator-split path.

        Returns a closure ``(state, initial_physics_state) ->
        (final_state, final_physics_state, predictions)``. The
        running-mean accumulator template comes from
        :meth:`Physics.get_empty_data` — a zero-filled snapshot of
        the dict ``compute_tendencies`` produces, which is exactly the
        pytree structure the scan carries.
        """
        template = self.physics.get_empty_data(self.coords)

        def _integrate_fn(state, initial_physics_state):
            trajectory = _op_split_trajectory(
                step_fn=step_fn,
                initial_physics_state=initial_physics_state,
                empty_diagnostics=template,
                outer_steps=outer_steps,
                inner_steps=inner_steps,
                post_process_fn=post_process_fn,
                output_averages=output_averages,
            )
            return trajectory(state)

        return _integrate_fn

    @partial(jax.jit, static_argnums=(0, 4, 5, 6)) # Note: if model fields assumed to be static are changed, the changes will not be picked up here
    def _run_from_state(self,
                        initial_state: primitive_equations.State,
                        initial_physics_state: Any,
                        forcing: ForcingData,
                        save_interval=10.0,
                        total_time=120.0,
                        output_averages=False,
    ) -> tuple[primitive_equations.State, Any, Predictions]:
        """JIT-compiled simulation loop. Returns raw Predictions pytree.

        Physics is computed once per ``dt`` outside the IMEX-RK stages
        and applied as a forward-Euler add to the dynamical state
        (operator-split Lie a from #471). The cross-step physics carry
        is first-class — threaded in as ``initial_physics_state`` and
        returned as the final carry so callers can continue a run
        across API boundaries without re-seeding (e.g.
        ``Model.resume``).
        """
        inner_steps = int(save_interval / self.dt_si.to(units.day).m)
        outer_steps = int(total_time / save_interval)
        # Op-split saves end-of-step states (snapshot mode) or
        # post-step running means (averaged mode), so the first saved
        # frame is at ``initial_state.sim_time + save_interval``, not
        # ``+ 0``. Index by ``arange(outer_steps) + 1`` to label the
        # frames at the times they actually correspond to.
        times = self.start_date.delta.days \
                + (initial_state.sim_time*units.second).to(units.day).m \
                + save_interval * (jnp.arange(outer_steps) + 1)

        op_split_step = self._get_op_split_step_fn(forcing)
        integrate = self._get_op_split_integrate_fn(
            op_split_step,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            post_process_fn=lambda state, physics_state: self._post_process(
                state, physics_state, output_averages,
            ),
            output_averages=output_averages,
        )
        final_modal_state, final_physics_state, predictions = integrate(
            initial_state, initial_physics_state,
        )

        return final_modal_state, final_physics_state, predictions.replace(times=times)

    def run_from_state(self,
                       initial_state: primitive_equations.State,
                       forcing: ForcingData,
                       save_interval=10.0,
                       total_time=120.0,
                       output_averages=False,
    ) -> tuple[primitive_equations.State, ModelPredictions]:
        """Run the full simulation forward in time starting from given initial state.

        Alternative to ``model.run`` / ``model.resume`` which does not
        read or write the model's internal state.

        Note: the operator-split path carries a cross-step physics
        state (radiation cache, prior-step TKE, …). This method
        rebuilds that carry from scratch at every call. For chaining
        runs continuously (so the carry persists across API
        boundaries), use ``run`` / ``resume`` — those thread
        ``self._final_physics_state`` automatically. For an advanced
        caller that wants explicit control of the carry, use
        :meth:`run_from_state_with_carry`.

        Args:
            initial_state:
                dinosaur.primitive_equations.State containing initial state of the run.
            forcing:
                ForcingData containing forcing conditions for the run.
            save_interval:
                Interval at which to save model outputs. Either a number
                of days (float) or a calendar string like ``'1 month'`` /
                ``'1 year'``; calendar strings are converted to days via
                ``Model.calendar`` (so ``'1 month'`` is 365/12 days under
                ``'365_day'``). Default 10.0 days.
            total_time:
                Total time to run the model. Same units as ``save_interval``.
                Default 120.0 days.
            output_averages:
                Whether to output time-averaged quantities (default False).

        Returns:
            A tuple containing (final dinosaur.primitive_equations.State, ModelPredictions object containing trajectory of post-processed model states).

        """
        final_modal_state, _, predictions = self.run_from_state_with_carry(
            initial_state,
            forcing,
            save_interval=save_interval,
            total_time=total_time,
            output_averages=output_averages,
        )
        return final_modal_state, predictions

    def run_from_state_with_carry(self,
                                  initial_state: primitive_equations.State,
                                  forcing: ForcingData,
                                  save_interval=10.0,
                                  total_time=120.0,
                                  output_averages=False,
                                  initial_physics_state: Any = None,
    ) -> tuple[primitive_equations.State, Any, ModelPredictions]:
        """Lower-level ``run_from_state`` that exposes the cross-step physics carry.

        Same semantics as :meth:`run_from_state` but also accepts a
        cross-step physics carry seed and returns the final carry —
        useful for callers that need to chain runs while bypassing the
        ``self._final_*`` state on ``Model``.

        Args:
            initial_state: See :meth:`run_from_state`.
            forcing: See :meth:`run_from_state`.
            save_interval: See :meth:`run_from_state`.
            total_time: See :meth:`run_from_state`.
            output_averages: See :meth:`run_from_state`.
            initial_physics_state:
                Optional cross-step physics carry to seed the
                operator-split integration. When ``None`` (default),
                seeded from :meth:`ComposablePhysics.initial_carry_state`
                (deterministic, see :meth:`_build_initial_physics_carry`).
                Threading the previous call's final carry here keeps
                sub-cycled radiation, prior-step TKE, etc. continuous
                across API boundaries — so two 5d calls match a single
                10d call.

        Returns:
            ``(final_modal_state, final_physics_state, model_predictions)``.
            Pass ``final_physics_state`` back in as
            ``initial_physics_state`` on the next call to continue the
            run without re-seeding physics.

        """
        save_interval_days = parse_duration_days(save_interval, calendar=self.calendar)
        total_time_days = parse_duration_days(total_time, calendar=self.calendar)
        if initial_physics_state is None:
            initial_physics_state = self._build_initial_physics_carry()
        final_modal_state, final_physics_state, predictions = self._run_from_state(
            initial_state, initial_physics_state, forcing,
            save_interval_days, total_time_days,
            output_averages,
        )
        return (
            final_modal_state,
            final_physics_state,
            ModelPredictions(predictions, self.coords, self.physics),
        )

    def resume(self,
               forcing: ForcingData=None,
               save_interval=10.0,
               total_time=120.0,
               output_averages=False,
    ) -> ModelPredictions:
        """Run the full simulation forward in time starting from end of previous call to model.run or model.resume.

        Continues the cross-step physics carry across the call
        boundary: ``self._final_physics_state`` from the previous
        ``run``/``resume`` is threaded back in so sub-cycled radiation,
        prior-step TKE, etc. don't reset at the API seam. A run that
        is broken into ``run()`` then ``resume()`` for the same total
        duration therefore matches a single ``run()`` of the combined
        duration (to numerical roundoff).

        Args:
            forcing:
                ForcingData containing forcing conditions for the run.
            save_interval:
                Interval at which to save model outputs. Number of days
                (float) or a calendar string like ``'1 month'`` /
                ``'1 year'`` (resolved against ``Model.calendar``).
            total_time:
                Total time to run the model. Same units as ``save_interval``.
            output_averages:
                Whether to output time-averaged quantities (default False).

        Returns:
            A ModelPredictions object containing the trajectory of post-processed model states.

        """
        # starts from preexisting self._final_modal_state, then updates self._final_modal_state
        jax.debug.callback(
            lambda: logger.info("Model starting with params: save_interval: %s, total_time: %s, output_averages: %s",
                                save_interval, total_time, output_averages)
        )
        final_modal_state, final_physics_state, predictions = self.run_from_state_with_carry(
            initial_state=self._final_modal_state,
            forcing=forcing or default_forcing(self.coords.horizontal),
            save_interval=save_interval,
            total_time=total_time,
            output_averages=output_averages,
            initial_physics_state=self._final_physics_state,
        )
        jax.debug.callback(lambda: logger.info("Run completed."))
        self._final_modal_state = final_modal_state
        self._final_physics_state = final_physics_state
        return predictions

    def run(self,
            initial_state: PhysicsState | primitive_equations.State = None,
            forcing: ForcingData=None,
            save_interval=10.0,
            total_time=120.0,
            output_averages=False,
    ) -> ModelPredictions:
        """Set model.initial_nodal_state and model.start_date and run the full simulation forward in time.

        Args:
            initial_state:
                PhysicsState or dinosaur.primitive_equations.State containing initial state of the model (default isothermal atmosphere).
            forcing:
                ForcingData containing forcing conditions for the run (default aquaplanet).
            save_interval:
                Interval at which to save model outputs. Number of days
                (float) or a calendar string like ``'1 month'`` /
                ``'1 year'`` (resolved against ``Model.calendar``).
                Default 10.0 days.
            total_time:
                Total time to run the model. Same units as ``save_interval``.
                Default 120.0 days.
            output_averages:
                Whether to output time-averaged quantities (default False).

        Returns:
            A ModelPredictions object containing the trajectory of post-processed model states.

        """
        self.bootstrap_state(initial_state)
        return self.resume(
            forcing=forcing, save_interval=save_interval,
            total_time=total_time, output_averages=output_averages,
        )

    def bootstrap_state(
        self,
        initial_state: PhysicsState | primitive_equations.State | None = None,
    ) -> None:
        """Populate ``_final_modal_state`` and ``_final_physics_state`` without integrating.

        Equivalent to the prep that ``run`` does before its first
        ``resume`` call, but exposed as a standalone method so callers
        that need only the initial pytrees — checkpoint restore (where
        ``flax.serialization.from_bytes`` requires a template), state
        introspection, or a bring-your-own-stepper workflow — don't have
        to spin up a zero-length integration to get them.
        """
        if isinstance(initial_state, primitive_equations.State):
            tracer_specs = {spec.name: spec for spec in self.physics.required_tracers()}
            self.initial_nodal_state = dynamics_state_to_physics_state(
                initial_state, self.primitive, tracer_specs=tracer_specs)
            self._final_modal_state = initial_state
        else:
            self.initial_nodal_state = initial_state
            self._final_modal_state = self._prepare_initial_modal_state(initial_state)

        # Eagerly build the physics carry. ``resume`` would otherwise
        # build it lazily on first call, but materialising it here
        # makes the pytree available as a checkpoint-restore template
        # and to any caller that wants to inspect / mutate the seed
        # state before stepping. Equivalent to leaving it ``None`` and
        # letting ``resume`` build it: ``resume`` skips its lazy-build
        # branch when ``_final_physics_state`` is already populated.
        self._final_physics_state = self._build_initial_physics_carry()
