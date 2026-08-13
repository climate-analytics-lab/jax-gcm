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

import jcm.constants as jcm_constants
from jcm.constants import PhysicalConstants
from jcm.diffusion import DiffusionFilter, level_dependent_scaling
from jcm.dycore.base import DynamicalCore, Predictions
from jcm.dycore.dinosaur.state_bridge import (
    dynamics_state_to_physics_state,
    physics_state_to_dynamics_state,
    physics_tendency_to_dynamics_tendency,
)
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


def physics_specs_from_constants(
    constants: PhysicalConstants,
) -> primitive_equations.PrimitiveEquationsSpecs:
    """Build dinosaur's nondimensionalisation/equation constants from ``constants``.

    This is the single bridge that makes the dynamical core honour the same
    :class:`~jcm.constants.PhysicalConstants` as the physics, instead of
    dinosaur's own defaults. ``Cp`` is derived inside dinosaur as ``R/kappa``,
    so passing ``rd`` (= ``akap·cpd``) and ``akap`` reproduces ``cpd`` exactly.
    """
    return primitive_equations.PrimitiveEquationsSpecs.from_si(
        radius_si=constants.rearth * units.meter,
        angular_velocity_si=constants.omega / units.second,
        gravity_acceleration_si=constants.grav * units.meter / units.second**2,
        ideal_gas_constant_si=constants.rd * units.joule / units.kilogram / units.kelvin,
        water_vapor_gas_constant_si=constants.rv * units.joule / units.kilogram / units.kelvin,
        water_vapor_isobaric_heat_capacity_si=(
            constants.cpv * units.joule / units.kilogram / units.kelvin
        ),
        kappa_si=constants.akap * units.dimensionless,
        scale=SI_SCALE,
    )


# Default specs, built from the default (shared) PhysicalConstants so the
# dynamical core and physics agree on values out of the box. Consumers that
# need a fixed module-level handle (e.g. Held-Suarez) reference this; the
# dycore itself rebuilds per-instance from its injected ``constants``.
PHYSICS_SPECS = physics_specs_from_constants(PhysicalConstants.default())


#: Semi-Lagrangian transport classes jcm requires from dinosaur. They live in
#: neuralgcm/dinosaur#135 and are not in a released dinosaur yet, so until that
#: lands the backend needs the fork (``shoyer/dinosaur`` @ ``semi-lagrangian``)
#: on the path. Pin a minimum dinosaur here once it ships.
_SL_CLASSES = (
    "SemiLagrangianPrimitiveEquations",
    "SemiLagrangianPrimitiveEquationsHybrid",
)


def semi_lagrangian_available() -> bool:
    """Return True when the installed dinosaur provides the SL transport."""
    return all(hasattr(primitive_equations, name) for name in _SL_CLASSES)


def _require_semi_lagrangian() -> None:
    """Fail with an actionable message when the SL core is missing.

    jcm transports tracers semi-Lagrangian only — the Eulerian spectral
    transport it replaced rang negative on sharp emission sources and NaN'd
    the aerosol microphysics (#521), so there is no fallback to offer.
    """
    if semi_lagrangian_available():
        return
    missing = [n for n in _SL_CLASSES if not hasattr(primitive_equations, n)]
    raise RuntimeError(
        "the installed dinosaur has no semi-Lagrangian transport "
        f"({', '.join(missing)} missing), and jcm's dinosaur backend now "
        "requires it — the Eulerian path has been removed because it rang "
        "negative on sharp sources and NaN'd aerosol microphysics (#521). "
        "Install the fork until neuralgcm/dinosaur#135 is released:\n"
        "    pip install 'dinosaur @ git+https://github.com/shoyer/dinosaur"
        "@semi-lagrangian'\n"
        "or put a clone of that branch on PYTHONPATH."
    )


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
        constants: PhysicalConstants | None = None,
        tracer_specs: Mapping[str, Any] | None = None,
        diffusion: DiffusionFilter | None = None,
        tracer_filter: Any | None = None,
        compute_frontogenesis: bool = False,
        compute_omega: bool = False,
        sl_options: Mapping[str, Any] | None = None,
    ):
        """Initialise the dinosaur backend; see the class docstring for argument semantics.

        Transport is dinosaur's semi-Lagrangian core
        (neuralgcm/dinosaur#135) — departure-point transport with the
        Bermejo-Staniforth quasi-monotone limiter, integrated with
        ``semi_lagrangian_crank_nicolson_rk2`` (self-starting, so jcm's
        chunk/resume structure needs no special first step). Every jcm
        extra tracer (aerosol mass/number, gases, cloud condensate — all
        of ``tracer_specs``) is carried as a NODAL tracer: it never
        round-trips through the spectral basis, which removes the
        per-step Gibbs ringing that made sharp emission sources go
        negative and NaN the aerosol microphysics (#521), and makes the
        limiter's non-negativity exact. ``specific_humidity`` stays
        modal (it participates in the implicit q<->Tv coupling).

        There is no Eulerian option. The classic spectral-transform
        transport rang negative on sharp sources and was the documented
        cause of aerosol blow-ups, so keeping it selectable only offered
        a way to run a configuration nobody should choose; it was also
        the silent default, which is how whole investigations ended up
        run on it by accident.

        ``sl_options`` forwards extras: ``interpolation_order``
        ('cubic'), ``monotone_tracers`` (True), ``departure_iterations``
        (1), ``off_centering`` (0.0), ``vertical_interpolation_order``
        ('linear').
        """
        _require_semi_lagrangian()
        self._sl_options = dict(sl_options or {})
        self.coords = coords
        self.terrain = terrain
        self.dt_seconds = float(dt_seconds)
        self.diffusion = diffusion or DiffusionFilter.default()
        # Optional gridpoint tracer filter (e.g. mass-conserving positivity)
        # applied as this spectral core projects to the physics gridpoint state
        # — see ``to_physics_state``. ``None`` disables it (the default). The
        # half-level (a, b) coefficients let ``to_physics_state`` reconstruct the
        # per-layer air mass ``Δp`` the filter weights by, dycore-side, from the
        # core's own vertical coordinate (hybrid a/b, or pure sigma → a=0, b=σ).
        self.tracer_filter = tracer_filter
        if isinstance(self.coords.vertical, HybridCoordinates):
            self._a_half = jnp.asarray(self.coords.vertical.a_boundaries)
            self._b_half = jnp.asarray(self.coords.vertical.b_boundaries)
        else:
            sigma_b = jnp.asarray(self.coords.vertical.boundaries)
            self._a_half = jnp.zeros_like(sigma_b)
            self._b_half = sigma_b
        self.tracer_specs = dict(tracer_specs) if tracer_specs else {}
        # Opt-in per-step frontogenesis diagnostic for the spectral frontal
        # GW source (CAM computes the analogous field inside its SE dycore).
        # Off by default: it costs horizontal finite differences of
        # (u, v, theta) every dt, which only the frontal GW term consumes.
        self.compute_frontogenesis = bool(compute_frontogenesis)
        # Opt-in pressure vertical velocity (omega = Dp/Dt) diagnostic on
        # the physics grid, for AeroCom's wap/w500/w700 (jax-gcm#409). Off
        # by default: it re-runs the diagnostic-state projection (a few
        # modal->nodal transforms) every step, which only output
        # consumers need.
        self.compute_omega = bool(compute_omega)

        # Physical constants drive both nondimensionalisation and the primitive
        # equations. Default to the *live* module singleton (read here at
        # construction time) so a prior jcm.constants.set_constants(...) is
        # honoured; an explicit ``constants`` argument overrides that.
        self.constants = constants if constants is not None else jcm_constants.physical_constants

        # Nondimensional timestep used throughout the dinosaur path.
        self._physics_specs = physics_specs_from_constants(self.constants)
        self._dt_si = (self.dt_seconds * units.second).to(units.second)
        self._dt = self._physics_specs.nondimensionalize(self._dt_si)

        # Build the dycore's primitive-equations operator + initial-state
        # helper. The reference-temperature profile that comes out of
        # ``isothermal_rest_atmosphere`` is what dinosaur's hybrid / sigma
        # State expects in ``temperature_variation``.
        self._default_state_fn, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self._physics_specs,
            p0=self.constants.p0 * units.pascal,
            p1=0.01 * self.constants.p0 * units.pascal,
        )

        # Orography is truncated against the spectral basis here — the SE
        # backend (pyses) will project against its own basis instead.
        self._truncated_orography = primitive_equations.truncated_modal_orography(
            self.terrain.orog, self.coords, wavenumbers_to_clip=2,
        )

        # Every jcm extra tracer rides nodally under semi-Lagrangian
        # transport (see the constructor docstring); the Eulerian core has
        # no nodal tracers.
        self._nodal_tracers = tuple(self.tracer_specs)

        # Dispatch on the vertical-coordinate family. Hybrid coords carry
        # ``a_boundaries`` in Pa; tell the dycore to interpret
        # ``hpa_quantity`` accordingly. Hybrid is the only family that
        # currently accepts a ``humidity_key`` (q <-> Tv coupling).
        sl_kwargs = dict(
            interpolation_order=self._sl_options.get("interpolation_order", "cubic"),
            monotone_tracers=self._sl_options.get("monotone_tracers", True),
            nodal_tracers=self._nodal_tracers,
            departure_iterations=self._sl_options.get("departure_iterations", 1),
            vertical_interpolation_order=self._sl_options.get(
                "vertical_interpolation_order", "linear"),
        )
        if isinstance(self.coords.vertical, HybridCoordinates):
            self._primitive = primitive_equations.SemiLagrangianPrimitiveEquationsHybrid(
                reference_temperature=aux_features[dinosaur.xarray_utils.REF_TEMP_KEY],
                orography=self._truncated_orography,
                coords=self.coords,
                physics_specs=self._physics_specs,
                hpa_quantity=units.pascal,
                humidity_key='specific_humidity',
                **sl_kwargs,
            )
        else:
            self._primitive = primitive_equations.SemiLagrangianPrimitiveEquations(
                reference_temperature=aux_features[dinosaur.xarray_utils.REF_TEMP_KEY],
                orography=self._truncated_orography,
                coords=self.coords,
                physics_specs=self._physics_specs,
                **sl_kwargs,
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
                # ``abs(...).max()`` not ``abs(eigenvalues[-1])``: under SPMD
                # the modal axis is zero-padded to divide across devices, so the
                # last entry is a padding 0 and ``[-1]`` would give ``scale =
                # dt/0 = inf`` (NaN-ing the filtered field). The max is the true
                # largest-wavenumber eigenvalue with or without padding.
                scale = self._dt / (timescale * abs(eigenvalues).max() ** order)
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
        filters = [
            self._conserve_global_mean_ps,
            diffuse_div,
            diffuse_vor_q,
            diffuse_temp,
        ]
        if self._nodal_tracers:
            # Modal hyperdiffusion masks are shaped like the spectral basis —
            # meaningless (and shape-incompatible) for the nodal tracers the
            # SL core carries. Wrap every filter so nodal tracers pass
            # through untouched (their smoothing is the SL interpolation +
            # quasi-monotone limiter, by design).
            filters = [
                primitive_equations.step_filter_excluding_nodal_tracers(
                    f, self._nodal_tracers,
                )
                for f in filters
            ]
        return filters

    # ------------------------------------------------------------------
    # Dynamics step (IMEX-RK SIL3)
    # ------------------------------------------------------------------

    def _build_dynamics_step_fn(self):
        """Build the dynamics step (IMEX-RK SIL3, or SL Crank–Nicolson RK2).

        The op-split caller adds the physics dynamics-tendency to the state
        forward-Euler-style before invoking this; the integrator advances
        ``state.sim_time`` by ``dt`` and applies its stages. The
        semi-Lagrangian path uses the self-starting two-stage
        ``semi_lagrangian_crank_nicolson_rk2`` (not SETTLS): it carries no
        cross-step departure memory, so jcm's chunked ``lax.scan`` /
        checkpoint-resume structure works unchanged.
        """
        return dinosaur.time_integration.semi_lagrangian_crank_nicolson_rk2(
            self._primitive, self._dt,
            off_centering=self._sl_options.get("off_centering", 0.0),
        )

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
                nodal_tracers=self._nodal_tracers,
            )
        else:
            state = self._default_state_fn(jax.random.PRNGKey(random_seed))
            # Sigma coords store ``log(P_s / p0)``; hybrid coords store
            # ``log(P_s in Pa)`` directly. Normalize only on the sigma path.
            if not isinstance(self.coords.vertical, HybridCoordinates):
                state.log_surface_pressure = self.coords.horizontal.to_modal(
                    self.coords.horizontal.to_nodal(state.log_surface_pressure)
                    - jnp.log(self._physics_specs.nondimensionalize(self.constants.p0 * units.pascal))
                )
            state.tracers = {
                'specific_humidity': 0.0 * primitive_equations_states.gaussian_scalar(
                    self.coords, self._physics_specs,
                ),
            }

        # Seed any required tracers not already present in ``state.tracers``.
        # Nodal tracers live on the gridpoint lat-lon grid (the whole point:
        # no spectral representation), so their seed is a nodal constant.
        nodal_ones = None
        if self._nodal_tracers:
            nlev = self.coords.vertical.layers
            nodal_ones = jnp.ones((nlev,) + self.coords.horizontal.nodal_shape)
        for spec in specs.values():
            if spec.name in state.tracers:
                continue
            if spec.name in self._nodal_tracers:
                state.tracers[spec.name] = spec.initial_value * nodal_ones
            else:
                state.tracers[spec.name] = (
                    spec.initial_value
                    * jnp.ones_like(state.tracers['specific_humidity'])
                )

        return State(**state.asdict(), sim_time=sim_time)

    def to_physics_state(self, state: State) -> PhysicsState:
        physics_state = dynamics_state_to_physics_state(
            state, self._primitive, tracer_specs=self.tracer_specs,
            nodal_tracers=self._nodal_tracers,
        )
        # Clean the gridpoint tracers as we hand them to the physics. A spectral
        # projection of a sharp, near-zero tracer source rings into negatives
        # (unphysical for aerosol microphysics / activation / optics); the
        # optional ``tracer_filter`` floors them here, dycore-side, so neither
        # the operator split nor the physics need to know about it. ``dp`` is the
        # per-layer air mass (∝ Δp) the mass-conserving rescale weights by, built
        # from this core's own (a, b) half-levels and the column surface
        # pressure. No-op (returns the state unchanged) when no filter is set or
        # there are no tracers.
        if self.tracer_filter is not None and physics_state.tracers:
            ps = physics_state.normalized_surface_pressure * self.constants.p0
            bcast = (slice(None),) + (jnp.newaxis,) * ps.ndim
            pressure_half = self._a_half[bcast] + self._b_half[bcast] * ps[jnp.newaxis, ...]
            dp = jnp.diff(pressure_half, axis=0)            # (nlev, *horiz)
            physics_state = physics_state.copy(
                tracers=self.tracer_filter(physics_state.tracers, dp),
            )
        # Then pin the cleaned gridpoint state to dinosaur's "physics" sharding
        # before it crosses into the (dycore-agnostic) physics packages. That
        # spec (``P(None, ('x', 'z'), 'y')``) replicates the vertical axis so
        # every column lives wholly on one device, and carries the device split
        # on longitude/latitude instead — the layout column physics wants, since
        # each column is independent. The dycore itself runs on
        # ``dycore_partition_spec`` (``P('z', 'x', 'y')``); the modal→nodal
        # transform here is where the two layouts meet. Under the recommended
        # longitude-only mesh ``(N, 1, 1)`` the two specs coincide, so this is
        # a free relabelling rather than a reshard. No-op without an
        # ``spmd_mesh``. See docs/source/design/parallelization.md.
        return self.coords.with_physics_sharding(physics_state)

    def physics_field_names(self) -> tuple[str, ...]:
        """Declare the enabled per-step dycore-side diagnostic fields."""
        names = []
        if self.compute_frontogenesis:
            names.append("frontogenesis")
        if self.compute_omega:
            names.append("omega")
        return tuple(names)

    def _compute_omega(self, state: State) -> jnp.ndarray:
        """Pressure vertical velocity omega = Dp/Dt [Pa/s] at level centres.

        Diagnosed from the modal state via dinosaur's own diagnostic-state
        computation, so the divergence, ``v . grad(ln ps)`` and vertical
        mass flux are exactly the ones the dynamics integrates. On hybrid
        levels (p = a + b ps):

            omega_k = b_k ps (v . grad ln ps)_k + b_k (dps/dt)
                      + (M_{k-1/2} + M_{k+1/2}) / 2

        with ``M = eta_dot dp/deta`` the boundary mass flux and
        ``dps/dt = -sum_k div(v dp)_k`` from continuity. On pure sigma
        levels (a = 0, b = sigma, M = sigma_dot ps) the same expression
        reduces to the classic ``omega = ps (sigma_dot + sigma d ln ps/dt)``,
        which is what the sigma branch computes directly. Returned
        dimensionalized, shape ``(nlev, nlon, nlat)``.
        """
        to_nodal = self.coords.horizontal.to_nodal
        vertical = self.coords.vertical
        # Omega needs no tracers, and under semi-Lagrangian advection the
        # extra tracers are stored NODAL, so the diagnostic-state helpers'
        # unconditional to_nodal over state.tracers would both crash (shape
        # mismatch) and waste one transform per tracer. Strip them.
        state = state.replace(tracers={})
        # Surface pressure in NONDIM PRESSURE units, (1, nlon, nlat) so the
        # leading axis broadcasts against (nlev, ...). The two coordinate
        # families store different conventions (see state_bridge): hybrid
        # keeps ``log(P_s)`` in nondim Pa directly, but sigma keeps the
        # NORMALIZED ``log(P_s / p0)`` (the sigma dynamics only ever use
        # grad/d-dt of log ps, which the scale cancels out of), so the
        # sigma path must restore the p0 factor here or omega comes out
        # ~1e5 too small (Codex P1 on #606).
        ps = jnp.exp(to_nodal(state.log_surface_pressure))
        if not isinstance(vertical, HybridCoordinates):
            ps = ps * self._physics_specs.nondimensionalize(
                self.constants.p0 * units.pascal)
        if isinstance(vertical, HybridCoordinates):
            # The primitive operator's nondim_coords, not self.coords: the
            # (a, b) tables must be nondimensionalized consistently with
            # the state's log_surface_pressure. Under jcm's SI scale the
            # two coincide numerically, but only nondim_coords is correct
            # by construction under any scale.
            ds = primitive_equations.compute_diagnostic_state_hybrid(
                state, self._primitive.nondim_coords)
            delta_b = vertical.sigma_thickness[:, jnp.newaxis, jnp.newaxis]
            b_bounds = jnp.asarray(vertical.b_boundaries)
            b_full = 0.5 * (b_bounds[:-1] + b_bounds[1:])[
                :, jnp.newaxis, jnp.newaxis]
            div_v_dp = (ds.layer_pressure_thickness * ds.divergence
                        + delta_b * ps * ds.u_dot_grad_log_sp)
            dps_dt = -jnp.sum(div_v_dp, axis=0, keepdims=True)
            mass_flux = ds.mass_flux_full  # (nlev+1, ...), zero at both ends
            omega = (b_full * ps * ds.u_dot_grad_log_sp
                     + b_full * dps_dt
                     + 0.5 * (mass_flux[:-1] + mass_flux[1:]))
        else:
            ds = primitive_equations.compute_diagnostic_state_sigma(
                state, self.coords)
            sigma = jnp.asarray(vertical.centers)[:, jnp.newaxis, jnp.newaxis]
            thickness = jnp.asarray(vertical.layer_thickness)[
                :, jnp.newaxis, jnp.newaxis]
            dlnps_dt = -jnp.sum(
                thickness * (ds.divergence + ds.u_dot_grad_log_sp),
                axis=0, keepdims=True)
            # sigma_dot lives on the nlev-1 inner boundaries; zero at the
            # top and bottom, centred average to the layer midpoints.
            sigma_dot = jnp.pad(ds.sigma_dot_full, [(1, 1), (0, 0), (0, 0)])
            sigma_dot_c = 0.5 * (sigma_dot[:-1] + sigma_dot[1:])
            omega = ps * (sigma_dot_c
                          + sigma * (ds.u_dot_grad_log_sp + dlnps_dt))
        return self._physics_specs.dimensionalize(
            omega, units.pascal / units.second).m

    def physics_fields(self, state, physics_state) -> dict:
        """Compute the enabled dycore-side diagnostics on the nodal grid.

        Frontogenesis (when ``compute_frontogenesis``):

        Uses the already-projected gridpoint ``physics_state`` (SI units):
        theta = T (p0/p)^kappa with the hybrid/sigma mid-level pressures
        from this core's own (a, b) coefficients, then the centred
        finite-difference frontogenesis of :func:`jcm.physics.
        gravity_waves.spectral.frontogenesis.frontogenesis_function`
        (a pure lat-lon function; importing it here is a deliberate
        dycore->shared-math dependency, mirroring CAM where the SE dycore
        owns ``compute_frontogenesis``). A spectral-gradient
        implementation is a possible upgrade — the FD version is
        second-order, and fields are smooth at the truncation scale.

        Omega (when ``compute_omega``): see :meth:`_compute_omega`.
        """
        if not (self.compute_frontogenesis or self.compute_omega):
            return {}
        out: dict = {}
        dtype = physics_state.temperature.dtype
        if self.compute_frontogenesis:
            from jcm.physics.gravity_waves.spectral.frontogenesis import (
                frontogenesis_function,
            )
            p0 = float(self.constants.p0)
            ps = physics_state.normalized_surface_pressure * p0
            a_full = 0.5 * (self._a_half[:-1] + self._a_half[1:])
            b_full = 0.5 * (self._b_half[:-1] + self._b_half[1:])
            shape = (-1,) + (1,) * ps.ndim
            p_full = (a_full.reshape(shape)
                      + b_full.reshape(shape) * ps[jnp.newaxis])
            kappa = float(self.constants.akap)
            theta = physics_state.temperature * (p0 / p_full) ** kappa
            frontgf = frontogenesis_function(
                physics_state.u_wind, physics_state.v_wind, theta,
                lons=jnp.asarray(self.coords.horizontal.longitudes),
                lats=jnp.asarray(self.coords.horizontal.latitudes),
            )
            out["frontogenesis"] = frontgf.astype(dtype)
        if self.compute_omega:
            out["omega"] = self._compute_omega(state).astype(dtype)
        return out

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
            # The tendency comes back from physics in the "physics" sharding;
            # pin it explicitly so the gridpoint→modal transform inside
            # ``physics_tendency_to_dynamics_tendency`` reshards from a known
            # layout rather than whatever GSPMD happened to infer. No-op
            # without an ``spmd_mesh``.
            physics_tendency = self.coords.with_physics_sharding(physics_tendency)
            dyn_tendency = physics_tendency_to_dynamics_tendency(
                physics_tendency, self._primitive, tracer_specs=self.tracer_specs,
                nodal_tracers=self._nodal_tracers,
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
