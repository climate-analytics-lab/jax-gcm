"""Single-column physics driver.

``SingleColumnModel`` evolves prognostic tracers (cloud water/ice, aerosols,
chemistry species, etc.) with ``lax.scan`` while large-scale atmospheric
state is supplied externally as a time series for *one column at one
location*. The dynamical core does not run; physics tendencies decide
state evolution.

The user supplies a vertical coordinate (``SigmaCoordinates`` or
``HybridCoordinates``) and a single ``(lat_deg, lon_deg)`` location, plus a
single-column ``TerrainData`` and ``ForcingData``. Internally the SCM
builds a duck-typed ``(1, 1)`` coords stub so column-based physics can
cache its coord-dependent transforms (lat, vertical-level transforms,
etc.) without dragging in a full horizontal grid.

Multiple columns at unrelated locations should be run in parallel one
layer above the SCM (e.g. ``jax.vmap`` over a list of
``(lat, lon, column_state)`` triples).

The companion ``PrescribedStateModel`` (in ``jcm.prescribed_state_model``)
is the multi-column equivalent: it accepts a full-grid prescribed state
and computes tendencies for every cell with ``vmap``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from jcm import column_coordinates
import tree_math
from jax import lax
from jax.tree_util import tree_map

from jcm.forcing import ForcingData
from jcm.physics_interface import (
    Physics,
    PhysicsState,
    PhysicsTendency,
    compute_physics_step_gridpoint,
    has_non_negative_tendency,
)

from jcm.terrain import TerrainData

#: Prognostic state variables, as opposed to tracers. ``free_evolve`` accepts
#: either, but the two take different paths through a step: prognostics go
#: through the nudged/free integrator, tracers through the tendency update.
_PROGNOSTIC_VARS = ("u_wind", "v_wind", "temperature", "specific_humidity")


@tree_math.struct
class SCMPredictions:
    """Container for ``SingleColumnModel.run`` outputs.

    All array fields are 1-D in the level dimension with a leading time
    axis: ``(n_times, nlev)`` for column profiles, ``(n_times,)`` for
    surface scalars.

    Attributes:
        prescribed_states: Time series of input column states (1-D).
        tracer_states: Time series of evolved tracers (dict of 1-D arrays).
        relaxed_states: Time series of *evolved* prognostic variables (dict;
            empty when neither relaxation nor free-evolution is configured).
            Holds both nudged (``relaxation_timescales``) and freely evolving
            (``free_evolve``) variables — e.g. the equilibrium temperature
            profile of an RCE run is read back here under ``"temperature"``.
        tendencies: Physics tendencies at each step (1-D).
        physics_data: Per-step diagnostics dict from the physics package.
        times: Times in days since the start of the run.

    """

    prescribed_states: PhysicsState
    tracer_states: dict
    relaxed_states: dict
    tendencies: PhysicsTendency
    physics_data: Any
    times: Any


def _vertical_nlev(vertical) -> int:
    # Retained as an alias: the implementation moved to
    # ``column_coordinates`` with the first-class coordinate type.
    return column_coordinates._vertical_nlev(vertical)


def _make_single_column_coords(vertical, lat_deg: float, lon_deg: float):
    """Column coordinates at the user's location.

    Previously a duck-typed ``SimpleNamespace``; now the first-class
    :class:`jcm.column_coordinates.ColumnCoordinates`, which implements
    the same consumed surface (``vertical``, ``nodal_shape``,
    ``horizontal.{latitudes, longitudes, nodal_shape, nodal_axes}``)
    with a type to test against and explanatory errors for spectral
    attributes a column cannot provide.
    """
    return column_coordinates.ColumnCoordinates.at_location(
        vertical, lat_deg, lon_deg
    )


def _expand_field(value: jnp.ndarray, nlev: int) -> jnp.ndarray:
    """Reshape a 1-D column field to ``(nlev, 1, 1)`` (or scalar surface to ``(1, 1)``)."""
    arr = jnp.asarray(value)
    if arr.ndim == 1:
        return arr.reshape(nlev, 1, 1)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    return arr


def _column_state_to_grid(column_state: PhysicsState, nlev: int) -> PhysicsState:
    """Reshape a 1-D column ``PhysicsState`` to the internal ``(nlev, 1, 1)`` grid."""
    grid_args = {}
    for field, value in column_state.asdict().items():
        if field == "tracers":
            grid_args["tracers"] = {
                k: _expand_field(v, nlev) for k, v in value.items()
            }
        elif field == "normalized_surface_pressure":
            arr = jnp.asarray(value)
            grid_args[field] = arr.reshape(1, 1) if arr.ndim == 0 else arr
        else:
            grid_args[field] = _expand_field(value, nlev)
    return type(column_state)(**grid_args)


def _squeeze_field(value: jnp.ndarray) -> jnp.ndarray:
    """Squeeze the ``(1, 1)`` grid axes off a per-cell array."""
    arr = jnp.asarray(value)
    if arr.ndim >= 2:
        return arr[..., 0, 0]
    return arr


def _squeeze_tendency(tend: PhysicsTendency) -> PhysicsTendency:
    args = {}
    for field, value in tend.asdict().items():
        if field == "tracers":
            args["tracers"] = {k: _squeeze_field(v) for k, v in value.items()}
        else:
            args[field] = _squeeze_field(value)
    return type(tend)(**args)


#: Coordinate agreement treated as exact, in degrees (~11 m). A state file
#: with float32 ``lat``/``lon`` (common from ``cdo``/``ncks``; jcm itself
#: writes float64) stores a coordinate a few 1e-6 degrees from the value it
#: names, which must not count as a request falling off the grid.
_COORDINATE_TOLERANCE_DEG = 1.0e-4


def _latitude_coverage(lat) -> tuple[float, float]:
    """``(south, north)`` bounds of the latitudes an axis's cells cover.

    Each end of the axis extends by half the spacing to its inner
    neighbour: a request within half a cell of the outermost row is in that
    row's cell. The exception is an end that already reaches its pole, i.e.
    whose distance to the pole is less than one spacing, so that no further
    row of that spacing fits in between. Its cell then runs to the pole,
    whatever the grid: a Gaussian axis stops ~0.77 spacings short of the pole
    at every truncation (T63's outermost centre, 88.57, is 1.43 degrees from
    the pole: beyond its half spacing of 0.92, inside its spacing of 1.85), a
    regular cell-centred axis half a spacing short, and a pole-centred one
    not at all, while an extraction from any of them stops at least one full
    spacing short. That keeps every global file
    accepting the whole of [-90, 90] and a regional one (``lat = 20..80``)
    refusing a pole ten degrees beyond its last row. A single-row axis has
    no spacing and covers only its own latitude.
    """
    import numpy as np

    rows = np.unique(np.asarray(lat, dtype=np.float64))
    if rows.size == 1:
        return float(rows[0]), float(rows[0])
    tol = _COORDINATE_TOLERANCE_DEG
    south_step = rows[1] - rows[0]
    north_step = rows[-1] - rows[-2]
    south = (-90.0 if rows[0] + 90.0 < south_step - tol
             else max(-90.0, rows[0] - south_step / 2.0))
    north = (90.0 if 90.0 - rows[-1] < north_step - tol
             else min(90.0, rows[-1] + north_step / 2.0))
    return float(south), float(north)


def _longitude_coverage(lon) -> tuple[float, float] | None:
    """Return the arc of longitudes an axis's cells cover, ``None`` for all.

    Returned as ``(west, east)`` degrees in [0, 360), the arc running east
    from ``west`` to ``east``. Longitude is periodic only when the file's
    cells close the circle: every point's cell extends half the spacing to
    its neighbour, so coverage can open only at a gap between consecutive
    longitudes on the circle that is wider than every other — the one an
    extraction leaves — and there only beyond its two end points' outward
    half cells, each half the spacing on that end's inner side. A global
    axis (JCM writes 0-360) has no such gap; a regional or zonal-sector
    extraction does. A single-column axis has no spacing and covers only its
    own longitude.
    """
    import numpy as np

    points = np.unique(np.mod(np.asarray(lon, dtype=np.float64), 360.0))
    if points.size == 1:
        return float(points[0]), float(points[0])
    tol = _COORDINATE_TOLERANCE_DEG
    gaps = np.diff(np.append(points, points[0] + 360.0))
    hole = int(np.argmax(gaps))
    n = points.size
    west_extension = gaps[(hole + 1) % n] / 2.0
    east_extension = gaps[hole - 1] / 2.0
    # A widest gap no wider than some other gap is just the grid's spacing
    # (a uniform global axis ties every gap), not a missing sector.
    others = np.delete(gaps, hole)
    if (gaps[hole] <= others.max() + tol
            or gaps[hole] - west_extension - east_extension <= tol):
        return None
    west = (points[(hole + 1) % n] - west_extension) % 360.0
    east = (points[hole] + east_extension) % 360.0
    return float(west), float(east)


def _on_arc(value, west, east) -> bool:
    """Whether longitude ``value`` lies on the eastward arc ``west`` → ``east``."""
    tol = _COORDINATE_TOLERANCE_DEG
    length = (east - west) % 360.0
    offset = (value - west) % 360.0
    return offset <= length + tol or offset >= 360.0 - tol


def _great_circle_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points, in km."""
    import math

    import jcm.constants as c

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
    return 2.0 * c.rearth * math.asin(min(1.0, math.sqrt(a))) / 1000.0


def select_column(states, ds, lat_deg: float, lon_deg: float):
    """Return the column of ``states`` nearest to ``(lat_deg, lon_deg)``.

    The state's xarray ``ds`` carries ``lat`` / ``lon`` coordinates from the
    JCM run that wrote it; pick by nearest neighbour so users can give
    physical degrees rather than grid indices.

    Longitude is matched on the circle, so any convention works: JCM writes
    its ``lon`` axis as 0-360, and a plain ``argmin(abs(lon - lon_deg))``
    would answer ``lon_deg=-120`` — the ordinary way to write 120W — with
    the column at 0, 120 degrees away, and would also mis-pick across the
    0/360 seam. Latitude is not periodic, so an out-of-range value cannot be
    folded into meaning and is refused instead.

    The request must also lie on the grid the file covers, or the nearest
    column can be arbitrarily far from it: a regional or single-column
    state file holds only part of the globe. The rule is the file's axis
    span extended by half a grid cell at each end — so a request is refused
    exactly when no cell of the file contains it — with longitude periodic
    only when the file's longitudes close the circle, and a latitude end
    that already reaches its pole covering it (see
    :func:`_latitude_coverage`; a global file therefore accepts every
    request, as before). A length-1 axis has no spacing to take half of, so
    a single-column file accepts only its own coordinates. Agreement to
    1e-4 degrees counts as exact, which absorbs float32 storage of the
    axes.

    Raises:
        ValueError: For a non-finite coordinate, a latitude outside
            [-90, 90], or a request outside the file's coverage; the last
            names the request, the coverage and the distance to the nearest
            column.

    """
    import numpy as np

    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    import math

    for name, value in (("lat_deg", lat_deg), ("lon_deg", lon_deg)):
        # NaN would win no comparison, so argmin would quietly return 0 and
        # the separation check below could not fire either.
        if not math.isfinite(float(value)):
            raise ValueError(f"{name}={value!r} is not a finite coordinate.")
    if not -90.0 <= float(lat_deg) <= 90.0:
        raise ValueError(
            f"lat_deg={lat_deg} is not a latitude; expected [-90, 90]. "
            "Longitude wraps, so lon_deg needs no such range — check the "
            "two have not been swapped.")
    i_lat = int(np.argmin(np.abs(lat - lat_deg)))
    # Signed separation folded into [-180, 180): the shorter way round.
    i_lon = int(np.argmin(np.abs((lon - lon_deg + 180.0) % 360.0 - 180.0)))

    tol = _COORDINATE_TOLERANCE_DEG
    south, north = _latitude_coverage(lat)
    lon_arc = _longitude_coverage(lon)
    off_lat = not south - tol <= float(lat_deg) <= north + tol
    off_lon = lon_arc is not None and not _on_arc(float(lon_deg), *lon_arc)
    if off_lat or off_lon:
        lat_text = (f"latitude {south:g}" if south == north
                    else f"latitudes {south:g} to {north:g}")
        if lon_arc is None:
            lon_text = "every longitude"
        elif lon_arc[0] == lon_arc[1]:
            lon_text = f"longitude {lon_arc[0]:g}"
        else:
            lon_text = f"longitudes {lon_arc[0]:g} east to {lon_arc[1]:g}"
        nearest_lat, nearest_lon = float(lat[i_lat]), float(lon[i_lon])
        distance = _great_circle_km(float(lat_deg), float(lon_deg),
                                    nearest_lat, nearest_lon)
        raise ValueError(
            f"The requested column (lat_deg={lat_deg}, lon_deg={lon_deg}) is "
            f"outside the state file's grid, which covers {lat_text} and "
            f"{lon_text} ({lat.size} x {lon.size} lat/lon points: each "
            "axis's span extended by half a grid cell, and a latitude end "
            "within one row spacing of its pole extended to the pole). Its "
            "nearest column, "
            f"at (lat={nearest_lat:g}, lon={nearest_lon:g}), is "
            f"{distance:.0f} km away. Request a point the file covers, or "
            "use a state file whose grid contains this one.")

    def slice_field(arr):
        # JCM xarray output is laid out (time, level, lon, lat) for column
        # variables and (time, lon, lat) for surface scalars.
        if arr.ndim == 4:
            return arr[:, :, i_lon, i_lat]
        if arr.ndim == 3:
            return arr[:, i_lon, i_lat]
        return arr

    return tree_map(slice_field, states), (i_lon, i_lat, float(lat[i_lat]), float(lon[i_lon]))


class SingleColumnModel:
    """Evolve physics tracers for one column at one ``(lat, lon)`` location.

    Example::

        from dinosaur.sigma_coordinates import SigmaCoordinates
        from jcm.physics.echam.echam_terms import echam_physics
        scm = SingleColumnModel(
            physics=echam_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=0.0, lon_deg=180.0,
        )
        # column_state is a PhysicsState whose array fields are 1-D (nlev,)
        # and normalized_surface_pressure is a scalar.
        predictions = scm.run([column_state, column_state, ...])

    Args:
        physics: Physics package whose ``compute_tendencies`` drives evolution.
        vertical: Vertical coordinate (``SigmaCoordinates`` or
            ``HybridCoordinates``) — the only required spatial input.
        lat_deg: Column latitude in degrees (default 0).
        lon_deg: Column longitude in degrees (default 0).
        terrain: Optional single-column ``TerrainData`` (shape ``(1, 1)``);
            defaults to ``TerrainData.single_column()`` (flat, all ocean).
        forcing: Optional single-column ``ForcingData`` (shape ``(1, 1)``);
            defaults to ``ForcingData.zeros((1, 1))``.
        dt_seconds: Physics timestep in seconds (default 1800).
        apply_tracer_tendencies: When ``False`` tracers are reported
            diagnostically but not advanced — except any named in
            ``free_evolve``, which still evolve. Set it ``False`` and list the
            tracers of interest to hold the column's other fields fixed: a
            prescribed-state column has no ascent, so a seeded ``qc`` rains out
            within hours and never re-forms, and any cloud-mediated aerosol
            sink then goes untested. Prescribing the cloud and freeing the
            aerosol is the configuration that tests one.
        relaxation_timescales: Optional ``{var_name: tau_seconds}`` mapping.
            Listed prognostic variables (``u_wind``, ``v_wind``,
            ``temperature``, ``specific_humidity``) are nudged toward the
            prescribed state with timescale ``tau`` while still receiving
            their physics tendency. The SCM prevents explicit humidity
            nudging from producing negative ``specific_humidity``, but that
            local guard is outside the physics positivity-correction ledger.
        free_evolve: Optional tuple of names that evolve under their physics
            tendency alone — no nudging toward the prescribed state. Accepts
            both prognostic variables and tracers. For a prognostic, e.g.
            ``free_evolve=("temperature",)`` lets temperature seek
            radiative-convective equilibrium; a variable may be in
            ``free_evolve`` *or* ``relaxation_timescales`` but not both (free
            evolution is just relaxation with no nudging term). For a tracer it
            only has an effect alongside ``apply_tracer_tendencies=False``,
            where it exempts that tracer from being held fixed.
        state_closure: Optional ``f(state, forcing) -> state`` applied to the
            assembled column *each step, before physics*. Use it to re-derive
            diagnostic fields from the freely evolving prognostics so the
            terms see a consistent state — e.g. a fixed-relative-humidity
            closure ``q = rh · qsat(p, T)`` (see :func:`jcm.rce.fixed_rh_closure`).
            Must be pure / ``jit``-compatible.

    """

    def __init__(
        self,
        physics: Physics,
        vertical,
        lat_deg: float = 0.0,
        lon_deg: float = 0.0,
        terrain: TerrainData | None = None,
        forcing: ForcingData | None = None,
        dt_seconds: float = 1800.0,
        apply_tracer_tendencies: bool = True,
        relaxation_timescales: dict[str, float] | None = None,
        free_evolve: tuple[str, ...] = (),
        state_closure: Callable[[PhysicsState, ForcingData], PhysicsState] | None = None,
    ) -> None:
        """Initialise (see class docstring for argument descriptions)."""
        self.physics = physics
        self.vertical = vertical
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.dt_seconds = float(dt_seconds)
        self.apply_tracer_tendencies = apply_tracer_tendencies
        self.relaxation_timescales = dict(relaxation_timescales or {})
        self.free_evolve = tuple(free_evolve)
        self.state_closure = state_closure

        # Free evolution and relaxation share one integrator (see
        # ``_make_step_fn``): a freely evolving variable is relaxation with no
        # nudging term, marked here by ``tau is None``. A variable is therefore
        # either nudged toward the prescribed target or left free, never both.
        overlap = set(self.free_evolve) & set(self.relaxation_timescales)
        if overlap:
            raise ValueError(
                f"Variables {sorted(overlap)} appear in both free_evolve and "
                "relaxation_timescales; a variable is either nudged or free, "
                "not both."
            )
        # ``free_evolve`` spans prognostics and tracers; only the prognostic
        # ones take part in the nudged/free integrator below. Which names are
        # tracers is not known until ``run`` sees the column, so the split
        # happens there.
        self._evolving_timescales: dict[str, float | None] = {
            **{name: None for name in self.free_evolve
               if name in _PROGNOSTIC_VARS},
            **self.relaxation_timescales,
        }

        self.coords = _make_single_column_coords(vertical, lat_deg, lon_deg)
        self.terrain = terrain if terrain is not None else TerrainData.single_column()
        self.forcing = forcing if forcing is not None else ForcingData.zeros((1, 1))

        self.physics.cache_coords(self.coords)
        # Hand the SCM's timestep down to the composable-physics container so
        # its terms read a single ``dt`` source (the same plumbing the full
        # ``Model`` uses).
        if hasattr(self.physics, "dt_seconds"):
            self.physics.dt_seconds = self.dt_seconds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_states(states: list[PhysicsState]) -> PhysicsState:
        return tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *states)

    def _make_step_fn(
        self,
        forcing: ForcingData,
        apply_tendencies: bool,
        tracer_names: tuple[str, ...],
        free_tracers: tuple[str, ...],
        evolving_var_params: tuple[tuple[str, float | None], ...],
        state_closure: Callable | None,
    ) -> Callable:
        physics = self.physics
        terrain = self.terrain
        nlev = self.coords.nodal_shape[0]
        dt_seconds = self.dt_seconds

        def step_fn(prescribed_column, tracers, evolving_vars, physics_data, time_idx):
            full_state_args = prescribed_column.asdict()
            full_state_args.pop("tracers", None)
            for name, _ in evolving_var_params:
                full_state_args[name] = evolving_vars[name]
            full_state_args["tracers"] = tracers
            column_state = type(prescribed_column)(**full_state_args)

            # Per-step diagnostic closure (e.g. fixed-RH ``q = rh·qsat(p, T)``).
            # Applied to the freshly assembled column *before* physics so the
            # terms (radiation, convection) see a thermodynamically consistent
            # (T, q) pair derived from the current — possibly freely evolving —
            # temperature. Terms communicate only through tendencies and the
            # diagnostics dict, so a closure like this cannot be a PhysicsTerm:
            # it has to overwrite the state the term loop reads.
            if state_closure is not None:
                column_state = state_closure(column_state, forcing)

            grid_state = _column_state_to_grid(column_state, nlev)
            tendencies_grid, new_physics_data = compute_physics_step_gridpoint(
                grid_state,
                forcing,
                terrain,
                physics_data,
                physics=physics,
                time_step=dt_seconds,
            )
            tendencies = _squeeze_tendency(tendencies_grid)

            updated_tracers = {}
            for name in tracer_names:
                tracer = tracers[name]
                if not (apply_tendencies or name in free_tracers):
                    # Held at the value the column prescribes.
                    updated_tracers[name] = tracer
                    continue
                tracer_tend = tendencies.tracers.get(name, jnp.zeros_like(tracer))
                # Integrate exactly the verified tendency returned above: it
                # is already bounded by the common interface, and clipping the
                # RESULT would make SCMPredictions.tendencies disagree with
                # the state it produced and hide the water source from the
                # common positivity diagnostics.
                #
                # Clip the ENTRY value instead, for the positive-definite
                # tracers only. ``compute_physics_step_gridpoint`` bounds the
                # tendency against a clamped copy of the state, so a column
                # that arrives with a negative ``qc``/``qi`` (numerical noise
                # in an externally supplied profile) gets a tendency that is
                # merely non-negative — which never repairs the value it is
                # added to, leaving the carry negative indefinitely. The full
                # model has no equivalent exposure: its prognostic state is
                # the dycore's, which spectral filtering and the optional
                # ``tracer_filter`` clean each step. The SCM carry has
                # neither and is what ``SCMPredictions`` reports.
                #
                # Aerosol and gas tracers are excluded exactly as they are in
                # the interface: their tendencies sum conservative
                # redistributions, so clamping either end creates mass.
                entry = (
                    jnp.maximum(tracer, 0.0)
                    if has_non_negative_tendency(name) else tracer
                )
                updated_tracers[name] = entry + dt_seconds * tracer_tend

            updated_evolving_vars = {}
            for name, tau in evolving_var_params:
                current_val = evolving_vars[name]
                phys_tend = getattr(tendencies, name)
                if tau is None:
                    # Free evolution: physics tendency only, no nudging. This
                    # is the RCE / free-running path.
                    nudging_tend = 0.0
                else:
                    target_val = getattr(prescribed_column, name)
                    nudging_tend = (target_val - current_val) / tau
                updated = current_val + dt_seconds * (phys_tend + nudging_tend)
                # ``phys_tend`` is the common interface's applied tendency, so
                # free evolution lands exactly on its predicted state. A
                # sufficiently strong explicit q nudge can still overdraw the
                # layer after that cap, so keep the nudged path non-negative.
                # This user-configured truncation is intentionally outside the
                # physics positivity ledger and cannot occur for ``tau=None``.
                if name == "specific_humidity" and tau is not None:
                    updated = jnp.maximum(updated, 0.0)
                updated_evolving_vars[name] = updated

            return tendencies, updated_tracers, updated_evolving_vars, new_physics_data

        return step_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prescribed_states: list[PhysicsState] | PhysicsState,
        forcing: ForcingData | None = None,
        initial_tracers: dict | None = None,
        initial_physics_data: Any = None,
        times: jnp.ndarray | None = None,
        initial_relaxed_vars: dict | None = None,
    ) -> SCMPredictions:
        """Run the SCM with evolving tracers.

        Args:
            prescribed_states: List of column ``PhysicsState`` snapshots, or
                a single ``PhysicsState`` whose leading axis is time. Array
                fields must be 1-D ``(nlev,)`` per snapshot.
            forcing: Optional override for the single-column forcing supplied
                at construction.
            initial_tracers: Initial values for prognostic tracers
                (1-D ``(nlev,)`` per tracer). Defaults to the first
                prescribed state's tracers (or ``{}``).
            initial_physics_data: Optional initial diagnostics dict.
            times: Optional days-since-start array.
            initial_relaxed_vars: Initial values for the evolved prognostic
                variables — both nudged (``relaxation_timescales``) and freely
                evolving (``free_evolve``) — 1-D ``(nlev,)`` per variable.
                Defaults to the first prescribed state's values.

        Returns:
            ``SCMPredictions``.

        """
        if forcing is None:
            forcing = self.forcing
        # The same two-way forced-mode contract every run entry point
        # enforces (#301): a forced-mode term without its ``prescribed_*``
        # fields must not run on a zero flux, and prescribed fluxes no term
        # consumes must not be silently ignored. No run window: the column
        # has no absolute start date, only a relative time axis.
        from jcm.physics.surface.prescribed_flux import validate_run_forcing
        validate_run_forcing(self.physics, forcing, run_window=None)

        if isinstance(prescribed_states, list):
            prescribed_states = self._stack_states(prescribed_states)

        n_times = prescribed_states.u_wind.shape[0]

        if initial_tracers is None:
            first_tracers = tree_map(lambda x: x[0], prescribed_states.tracers)
            initial_tracers = first_tracers if first_tracers else {}

        # Sort by name only — ``tau`` may be ``None`` (free-evolving), which is
        # not orderable against the float relaxation timescales.
        evolving_var_params = tuple(
            sorted(self._evolving_timescales.items(), key=lambda kv: kv[0])
        )
        if initial_relaxed_vars is None:
            first_state_slice = tree_map(lambda x: x[0], prescribed_states)
            initial_relaxed_vars = {
                name: getattr(first_state_slice, name)
                for name, _ in evolving_var_params
            }

        # Seed the diagnostics-dict carry the same way ``Model`` does:
        # the structural template comes from
        # :meth:`ComposablePhysics.get_empty_data` (a zero-filled
        # snapshot of the post-step output pytree) unioned with the
        # declarative cross-step carry slots from
        # :meth:`ComposablePhysics.initial_carry_state` (e.g. TKE
        # floored at ECHAM's 0.01 m²/s² lower bound). Using a
        # live ``compute_tendencies`` result here was the architectural
        # bug `#470 <https://github.com/climate-analytics-lab/jax-gcm/issues/470>`_
        # tracks — among other things, the radiation carry's ``step``
        # counter gets advanced before the first real scan step, which
        # shifts the sub-stepping cadence by one under ``nstrad > 1``.
        if initial_physics_data is None:
            template = self.physics.get_empty_data(self.coords)
            initial_carry = self.physics.initial_carry_state(self.coords)
            if isinstance(initial_carry, dict) and isinstance(template, dict):
                initial_physics_data = {**template, **initial_carry}
            else:
                initial_physics_data = (
                    template if initial_carry is None else initial_carry
                )

        if times is None:
            times = jnp.arange(n_times) * (self.dt_seconds / 86400.0)

        # Names in free_evolve that are neither prognostics nor tracers of this
        # column are a typo, not a silent no-op.
        free_tracers = tuple(n for n in self.free_evolve if n in initial_tracers)
        unknown = sorted(set(self.free_evolve) - set(initial_tracers)
                         - set(_PROGNOSTIC_VARS))
        if unknown:
            raise ValueError(
                f"free_evolve names {unknown} are neither prognostic variables "
                f"{sorted(_PROGNOSTIC_VARS)} nor tracers of this column "
                f"({sorted(initial_tracers)})"
            )

        step_fn = self._make_step_fn(
            forcing=forcing,
            apply_tendencies=self.apply_tracer_tendencies,
            tracer_names=tuple(initial_tracers.keys()),
            free_tracers=free_tracers,
            evolving_var_params=evolving_var_params,
            state_closure=self.state_closure,
        )

        def scan_step(carry, time_idx):
            tracers, evolving_vars, physics_data = carry
            prescribed_column = tree_map(lambda x: x[time_idx], prescribed_states)
            prescribed_column = prescribed_column.copy(tracers={})
            tendencies, new_tracers, new_evolving_vars, new_physics_data = step_fn(
                prescribed_column, tracers, evolving_vars, physics_data, time_idx,
            )
            new_carry = (new_tracers, new_evolving_vars, new_physics_data)
            return new_carry, (tendencies, new_tracers, new_evolving_vars, new_physics_data)

        initial_carry = (initial_tracers, initial_relaxed_vars, initial_physics_data)

        @jax.jit
        def run_scan():
            return lax.scan(scan_step, initial_carry, jnp.arange(n_times))

        _, (tendencies, tracer_history, relaxed_vars_history, physics_data_history) = run_scan()

        return SCMPredictions(
            prescribed_states=prescribed_states,
            tracer_states=tracer_history,
            relaxed_states=relaxed_vars_history,
            tendencies=tendencies,
            physics_data=physics_data_history,
            times=times,
        )
