"""User-facing prediction container and its xarray serialization.

:class:`ModelPredictions` wraps the internal :class:`~jcm.dycore.base.Predictions`
pytree with the coordinate system and physics module needed to turn a run into an
analysis-ready :class:`xarray.Dataset`. Returned by :meth:`jcm.model.Model.run`,
:meth:`~jcm.model.Model.resume`, and :meth:`~jcm.model.Model.run_from_state`.
"""

from __future__ import annotations

import logging

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import pandas as pd

from jcm import cf_metadata, provenance, temporal_aggregation
from jcm.dycore.base import Predictions
from jcm.physics_interface import Physics
from jcm.utils import DYNAMICS_UNITS_TABLE_CSV_PATH, data_to_xarray

logger = logging.getLogger(__name__)

_MISSING = object()
_LIVE_CONTEXT_PARAMS_KEY = "parameters_rederived_from_live_context"
_LIVE_CONTEXT_PARAMS_DESCRIPTION = (
    "values were read from the live physics passed to "
    "ModelPredictions.with_context; they are not a trace-time record of the "
    "parameters that produced this trajectory"
)


def _exact_datetime64(values):
    """Transfer a ``jax_datetime.Datetime`` array to exact host datetimes."""
    host = jax.device_get(values)
    if hasattr(host, "to_datetime64"):
        return np.asarray(host.to_datetime64()).astype("datetime64[ms]")
    array = np.asarray(host)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ms]")
    raise TypeError(
        "Prediction timestamps must be jax_datetime.Datetime or datetime64; "
        f"got {array.dtype}."
    )


def _has_cell_method(existing: str, requested: str) -> bool:
    """Avoid duplicating an already-present CF cell-method declaration."""
    normalize = lambda value: " ".join(value.replace(":", " : ").split())
    return normalize(requested) in normalize(existing)


def _time_cell_operations(cell_methods: str) -> set[str]:
    """Extract every operation following a CF ``time:`` cell method."""
    words = cell_methods.replace(":", " : ").split()
    return {
        words[index + 2]
        for index in range(len(words) - 2)
        if words[index:index + 2] == ["time", ":"]
    }


def _apply_term_output_attrs(ds, physics):
    """Stamp per-term ``output_attrs`` onto matching variables of ``ds`` (#740).

    Each :class:`~jcm.physics.physics_term.PhysicsTerm` declares CF/units
    attributes for the diagnostics it computes, keyed by their dotted output
    names; :meth:`ComposablePhysics.output_attrs` merges them for the package.
    Factored out so BOTH trajectory paths — the dinosaur lat/lon build and the
    delegated non-modal (pySES) build — apply the identical merge rather than
    one silently omitting it. A physics predating ``output_attrs`` is tolerated
    via ``getattr``. Returns ``ds`` (mutated in place).
    """
    term_attrs = getattr(physics, "output_attrs", None)
    if callable(term_attrs):
        for var, attrs in term_attrs().items():
            if var in ds:
                ds[var].attrs.update(attrs)
    return ds


class ModelPredictions:
    """User-facing container for model prediction outputs.

    Wraps the internal :class:`Predictions` pytree with the coordinate system
    and physics module needed for xarray conversion. Returned by
    :meth:`Model.run`, :meth:`Model.resume`, and :meth:`Model.run_from_state`.

    Attributes:
        dynamics (PhysicsState): The physical state variables.
        physics (Any): Diagnostic physics data.
        times (Any): Timestamps of the predictions.

    """

    def __init__(self, predictions: Predictions, coords, physics: Physics,  # noqa: D107
                 dycore=None, observations=None, observers=(),
                 observer_start_time=None, obs_dt_seconds=None,
                 snapshots=None, snapshot_variables=(),
                 snapshot_times=None, params=None):
        self._predictions = predictions
        self._coords = coords
        self._physics = physics
        self._dycore = dycore
        self._observations = observations
        self._observers = tuple(observers)
        self._observer_start_time = observer_start_time
        self._obs_dt_seconds = obs_dt_seconds
        self._snapshots = snapshots
        self._snapshot_variables = tuple(snapshot_variables)
        self._snapshot_times = snapshot_times
        # The parameters this trajectory was produced with (#732).
        #
        # ``params`` is the record captured at TRACE time by
        # ``Model._run_from_state`` and is authoritative when present:
        # ``self`` is a static argument to that jit, so the parameters are
        # constants inside the executable, and reading the live module here
        # can report values that never reached the computation. Falling
        # back to a live read covers a ModelPredictions built directly,
        # where there is no trace to have captured. With no physics at all
        # there is nothing to record: the pytree unflatten rebuilds without
        # it by design, and runs on every tree_map over a ModelPredictions.
        self._params = {}
        try:
            if params is not None:
                self._params = self._check_live_matches_traced(params, physics)
            elif physics is not None:
                self._params = provenance.describe_params(physics)
        except Exception:  # noqa: BLE001 — never fail a completed run
            logger.warning("provenance: parameter capture failed",
                           exc_info=True)

    @staticmethod
    def _check_live_matches_traced(traced, physics):
        """Return *traced*, flagging a live/compiled parameter divergence.

        A mismatch means the caller edited a parameter in place after the
        model was first compiled with a different value. jcm binds physics
        parameters when the physics is first traced (``Model`` is a static
        jit argument), and whether a later edit reaches a later run depends
        on which of JAX's compilation caches that run hits (#735), so the
        results are unreliable either way. That is a scientific error the
        user needs told about, not something for provenance to paper over
        by quietly recording the compiled values and moving on.
        """
        live = provenance.describe_params(physics)
        if live == traced:
            return traced
        logger.warning(
            "provenance: the live parameters differ from those this model "
            "was first compiled with. jcm binds physics parameters when the "
            "physics is first traced (Model._run_from_state takes `self` as "
            "a static argument), and an in-place parameter change afterwards "
            "may or may not reach a later run, depending on JAX's "
            "compilation caches. Results after such a change are therefore "
            "unreliable, whatever this record says. Rebuild the Model to "
            "change parameters.")
        flagged = dict(traced)
        flagged["live_parameters_differ_from_compiled"] = (
            "parameters were edited in place after this model was first "
            "compiled; the record holds the first-compiled values and the "
            "run's results are unreliable")
        return flagged

    @property
    def params(self):
        """The physics parameter values behind this trajectory (#732).

        Flat ``<term>.<variable>.<field>`` keys, read off the built model
        rather than the requested config, so they reflect what ran. Empty
        for predictions reconstructed by a pytree ``tree_map``, which
        carries no physics.
        """
        return self._params

    @property
    def dynamics(self):
        return self._predictions.dynamics

    @property
    def physics(self):
        return self._predictions.physics

    @property
    def times(self):
        return self._predictions.times

    @property
    def observations(self):
        """Raw per-timestep observer samples (tuple of dicts), or ``None``."""
        return self._observations

    @property
    def snapshots(self):
        """Raw interval-instantaneous snapshot arrays, or ``None``."""
        return self._snapshots

    def with_context(
        self,
        model_or_coords,
        physics: Physics | None = None,
        *,
        dycore=_MISSING,
        observations=_MISSING,
        observers=_MISSING,
        observer_start_time=_MISSING,
        obs_dt_seconds=_MISSING,
        snapshots=_MISSING,
        snapshot_variables=_MISSING,
        snapshot_times=_MISSING,
    ) -> "ModelPredictions":
        """Return a copy with host-side model/output context re-attached.

        JAX pytree operations deliberately carry only prediction and
        observation arrays. Consequently, a ``ModelPredictions`` returned by
        ``jax.tree.map`` (or reconstructed after another JAX boundary) needs
        its static context restored before its Dataset accessors can be used.
        The common spelling is::

            restored = transformed.with_context(model)

        ``model`` supplies coordinates, physics, dycore, observers, and the
        model timestep. Alternatively, pass ``coords`` and ``physics`` as the
        first two arguments and supply any remaining context by keyword.

        Snapshot arrays are not pytree children, so a transformed object
        cannot recover them. Pass ``snapshots``, ``snapshot_variables``, and
        ``snapshot_times`` when restoring a snapshot stream. Omitted
        keyword metadata is preserved when it is still present on ``self``.

        Parameter values are always re-read from the supplied *live* physics
        and marked in :attr:`params` as re-derived. They must not be presented
        as the trace-time parameters that produced the transformed arrays,
        because pytree operations may combine predictions from a different
        model instance.

        Args:
            model_or_coords: A :class:`~jcm.model.Model`, or the coordinate
                system to attach.
            physics: Physics package when ``model_or_coords`` is a coordinate
                system. Must be omitted when passing a Model.
            dycore: Optional dynamical core. The model-bound form obtains this
                from the Model.
            observations: Raw observer samples. Defaults to the samples that
                survived the pytree operation.
            observers: Observer definitions. The model-bound form obtains
                these from the Model.
            observer_start_time: Exact start instant for the observer window.
            obs_dt_seconds: Observation cadence. The model-bound form uses the
                model timestep.
            snapshots: Raw interval-instantaneous snapshot arrays.
            snapshot_variables: Names requested for the snapshot stream.
            snapshot_times: Exact timestamp for every snapshot.

        Returns:
            A new ``ModelPredictions`` with the same pytree children and the
            supplied host-side context.

        Raises:
            TypeError: If neither a Model nor ``(coords, physics)`` is passed.
            ValueError: If the prediction, observation, or snapshot arrays are
                obviously incompatible with the supplied context.

        """
        is_model = all(
            hasattr(model_or_coords, name)
            for name in ("coords", "physics", "dycore")
        )
        if is_model:
            if physics is not None:
                raise TypeError(
                    "Pass either with_context(model) or "
                    "with_context(coords, physics), not both.")
            model = model_or_coords
            coords = model.coords
            physics = model.physics
            if dycore is _MISSING:
                dycore = model.dycore
            if observers is _MISSING:
                observers = getattr(model, "observers", ())
            if obs_dt_seconds is _MISSING:
                dt_si = getattr(model, "dt_si", None)
                obs_dt_seconds = getattr(
                    dt_si, "m", getattr(model.dycore, "dt_seconds", None))
        else:
            coords = model_or_coords
            if physics is None:
                raise TypeError(
                    "with_context(coords, physics) requires a physics "
                    "argument; pass a Model as the sole positional argument "
                    "for the model-bound form.")

        dycore = self._dycore if dycore is _MISSING else dycore
        observations = (
            self._observations if observations is _MISSING else observations
        )
        observers = self._observers if observers is _MISSING else observers
        observer_start_time = (
            self._observer_start_time
            if observer_start_time is _MISSING else observer_start_time
        )
        obs_dt_seconds = (
            self._obs_dt_seconds
            if obs_dt_seconds is _MISSING else obs_dt_seconds
        )
        snapshots = self._snapshots if snapshots is _MISSING else snapshots
        snapshot_variables = (
            self._snapshot_variables
            if snapshot_variables is _MISSING else snapshot_variables
        )
        snapshot_times = (
            self._snapshot_times
            if snapshot_times is _MISSING
            else snapshot_times
        )

        self._validate_context_shapes(
            coords, observations, observers, snapshots)
        restored = ModelPredictions(
            self._predictions,
            coords,
            physics,
            dycore=dycore,
            observations=observations,
            observers=observers,
            observer_start_time=observer_start_time,
            obs_dt_seconds=obs_dt_seconds,
            snapshots=snapshots,
            snapshot_variables=snapshot_variables,
            snapshot_times=snapshot_times,
        )
        # ``__init__`` has just captured the live values. Label that record
        # after construction rather than passing it as ``params=``: the latter
        # means "trace-time record" and intentionally invokes the live-vs-
        # compiled mismatch check, which is not the claim this API can make.
        restored._params = dict(restored._params)
        restored._params[_LIVE_CONTEXT_PARAMS_KEY] = (
            _LIVE_CONTEXT_PARAMS_DESCRIPTION)
        return restored

    def _validate_context_shapes(
        self, coords, observations, observers, snapshots,
    ) -> None:
        """Reject context that is plainly incompatible with retained arrays."""
        expected = getattr(coords, "nodal_shape", None)
        dynamics = getattr(self._predictions, "dynamics", None)
        u_wind = getattr(dynamics, "u_wind", None)
        if expected is not None and u_wind is not None:
            expected = tuple(int(size) for size in expected)
            actual = tuple(int(size) for size in u_wind.shape[1:])
            if actual != expected:
                raise ValueError(
                    "Prediction grid shape is incompatible with the supplied "
                    f"coordinates: retained u_wind frames have {actual}, but "
                    f"coords.nodal_shape is {expected}.")

        if observations and len(observations) != len(observers):
            raise ValueError(
                f"Retained observations contain {len(observations)} stream(s), "
                f"but the supplied context has {len(observers)} observer(s).")

        horizontal_shape = getattr(
            getattr(coords, "horizontal", None), "nodal_shape", None)
        if snapshots and horizontal_shape is not None:
            horizontal_shape = tuple(int(size) for size in horizontal_shape)
            for name, values in snapshots.items():
                actual = tuple(int(size) for size in values.shape[1:])
                if actual != horizontal_shape:
                    raise ValueError(
                        f"Snapshot {name!r} has horizontal shape {actual}, "
                        "but the supplied coordinates require "
                        f"{horizontal_shape}.")

    def snapshot_dataset(self):
        """Interval-instantaneous 2-D snapshots as one xarray Dataset.

        The AeroCom 3-hourly stream (jax-gcm#586): each requested variable
        comes back as ``(snap_time, lon, lat)`` at the snapshot cadence,
        with ``snap_time`` in days from the window start (first snapshot
        one interval in, matching the post-step sampling convention).
        Empty dict semantics: returns ``None`` when the run requested no
        snapshots.
        """
        if not self._snapshots or self._snapshot_times is None:
            return None
        import xarray as xr

        snaps = jax.device_get(self._snapshots)
        nlon, nlat = self._coords.horizontal.nodal_shape
        first = next(iter(snaps.values()))
        n = first.shape[0]
        t = _exact_datetime64(self._snapshot_times)
        if t.shape != (n,):
            raise ValueError(
                f"snapshot_times must have shape ({n},); got {t.shape}.")
        data = {}
        for name, arr in snaps.items():
            arr = np.asarray(arr).reshape(n, nlon, nlat)
            data[name.replace(".", "_")] = (("snap_time", "lon", "lat"), arr)
        lon = self._coords.horizontal.nodal_axes[0] * 180.0 / np.pi
        lat = np.arcsin(self._coords.horizontal.nodal_axes[1]) * 180.0 / np.pi
        ds = xr.Dataset(
            data,
            coords={"snap_time": t, "lon": lon, "lat": lat},
            attrs={"sampling": "instantaneous (post-step)",
                   **provenance.params_attrs(self._params)},
        )
        return temporal_aggregation.set_cf_datetime_encoding(ds, "snap_time")

    def observation_datasets(self):
        """Per-timestep virtual-observation output as xarray Datasets.

        Stamped with the run's parameters like the trajectory and the
        snapshots, since an observer stream is often persisted on its own.

        Returns:
            Dict ``{observer_name: xarray.Dataset}`` — one Dataset per
            attached :class:`jcm.observers.Observer`, with dims
            ``(time, point)`` (``(time, level, point)`` in profile mode),
            a per-``dt`` time axis, and the sampling positions as
            coordinates. Empty dict when the run had no observers.

        """
        if not self._observations:
            return {}
        if self._observer_start_time is None:
            raise ValueError(
                "This trajectory has observation samples but no window start "
                "time, so the per-timestep time axis cannot be built. That "
                "happens when a run inside a JAX transformation was given "
                "prepared sampling tables (observer_xs) and no exact "
                "observer_start_time, leaving nothing concrete to date the "
                "samples by. Pass observer_start_time alongside observer_xs to "
                "record it; the raw samples are on `.observations` either "
                "way.")
        samples_host = jax.device_get(self._observations)
        stamp = provenance.params_attrs(self._params)
        datasets = {}
        for obs, samples in zip(self._observers, samples_host):
            ds = obs.to_dataset(samples, self._observer_start_time,
                                self._obs_dt_seconds)
            ds.attrs.update(stamp)
            datasets[obs.name] = ds
        return datasets

    def to_xarray(self):
        """Convert the full prediction trajectory to an xarray.Dataset.

        The parameters the run used are stamped into the dataset's global
        attributes here (#732), so they survive a bare
        ``model.run(...).to_xarray().to_netcdf(...)`` that never goes near
        the Hydra runners. Wrapping rather than stamping inside
        :meth:`_trajectory_dataset` keeps a per-backend return path from
        being able to skip it.

        Returns:
            An xarray.Dataset ready for analysis and plotting.

        Integer and boolean time-dependent diagnostics are categorical unless
        a diagnostic defines a separate numerical statistic. They are omitted
        from interval-mean trajectories and listed in the Dataset attribute
        ``omitted_interval_mean_variables``; instantaneous output retains them.

        """
        ds = self._trajectory_dataset()
        bounds = getattr(self._predictions, "time_bounds", None)
        cell_method = getattr(self._predictions, "time_cell_method", None)
        if bounds is not None:
            bounds = _exact_datetime64(bounds)
            if bounds.shape != (ds.sizes["time"], 2):
                raise ValueError(
                    "Prediction time_bounds must have shape (time, 2); got "
                    f"{bounds.shape}."
                )
            ds["time_bounds"] = (("time", "bounds"), bounds)
            ds["time_bounds"].attrs.update(
                long_name="time interval bounds",
                description=(
                    "lower and upper bounds of each represented time interval"),
            )
            ds["time"].attrs["bounds"] = "time_bounds"
        is_mean = (cell_method is not None
                   and bool(np.asarray(jax.device_get(cell_method))))
        if is_mean:
            if bounds is None:
                raise ValueError("Interval-mean predictions require time_bounds.")
            # The traced whole-second clock cannot represent a half-second
            # midpoint for odd-duration intervals.  Bounds are authoritative
            # and datetime64[ms] preserves the exact midpoint on the host.
            ds["time"] = ("time", bounds[:, 0]
                          + (bounds[:, 1] - bounds[:, 0]) // 2)
            ds["time"].attrs["bounds"] = "time_bounds"
            cell_method = "time: mean"
            categorical = sorted(
                name for name, var in ds.data_vars.items()
                if (name != "time_bounds" and "time" in var.dims
                    and (np.issubdtype(var.dtype, np.integer)
                         or np.issubdtype(var.dtype, np.bool_)))
            )
            if categorical:
                ds = ds.drop_vars(categorical)
                ds.attrs["omitted_interval_mean_variables"] = ",".join(
                    categorical)
            for var in ds.data_vars.values():
                if "time" in var.dims and var.name != "time_bounds":
                    existing = var.attrs.get("cell_methods", "")
                    operations = _time_cell_operations(existing)
                    if operations - {"mean"}:
                        raise ValueError(
                            f"Variable {var.name!r} declares incompatible "
                            f"time cell methods {sorted(operations)} but the "
                            "trajectory contains interval means."
                        )
                    if not _has_cell_method(existing, cell_method):
                        var.attrs["cell_methods"] = " ".join(
                            item for item in (existing, cell_method) if item)
        temporal_aggregation.set_cf_datetime_encoding(
            ds, "time", "time_bounds")
        ds.attrs.update(provenance.params_attrs(self._params))
        return ds

    def monthly_means(self):
        """Return bounds-aware Gregorian monthly means as an xarray Dataset.

        The trajectory must contain interval averages with exact bounds.
        Instantaneous output and intervals spanning a month boundary are
        rejected because they cannot be converted into true monthly means.
        This host-only convenience method leaves the differentiable raw
        prediction arrays untouched.
        """
        return temporal_aggregation.monthly_means(self.to_xarray())

    def _trajectory_dataset(self):
        """Build the trajectory Dataset, before provenance stamping."""
        # Backends whose native horizontal layout is not the separable
        # lat/lon grid the legacy path below assumes (pySES cubed-sphere
        # columns) own their trajectory conversion per the DynamicalCore
        # protocol; delegate whenever the grid has no modal axes.
        if self._dycore is not None and not hasattr(
                self._coords.horizontal, "modal_axes"):
            times = _exact_datetime64(self.times)
            ds = self._dycore.to_xarray(self._predictions, times)
            # The dycore's ``to_xarray`` has already run
            # ``cf_metadata.finalize_output`` (CSV attrs and the curated
            # ``_VARIABLE_ATTRS`` are on). Apply the per-term output metadata
            # here too, so pySES output carries the radiation/cloud/convection
            # units the dinosaur path gets (#740). Ordering is safe: the
            # term-declared names (``radiation.*`` and other diagnostics) are
            # disjoint from ``cf_metadata._VARIABLE_ATTRS`` (vertical coords,
            # core prognostics), so stamping term attrs after finalize does not
            # upset the documented CSV < term < cf_metadata precedence.
            return _apply_term_output_attrs(ds, self._physics)

        # float0s are placeholders representing the lack of tangent space for non-differentiable variables.
        # jax.numpy arrays cannot have float0 dtype, so jcm handles them with numpy arrays;
        # substituting jax.numpy arrays here allows us to handle Predictions objects that contain derivatives.
        float0s_to_nans = lambda pytree: tree_map(
            lambda x: jnp.full_like(x, jnp.nan, dtype=float) if x.dtype == jax.dtypes.float0 else x,
            pytree,
        )

        dynamics_predictions = float0s_to_nans(self.dynamics)
        physics_predictions = float0s_to_nans(self.physics)

        nodal_shape = dynamics_predictions.u_wind.shape[1:]

        # Per-physics flattening of the diagnostic struct into a dict of named fields.
        physics_preds_dict = self._physics.data_struct_to_dict(physics_predictions, nodal_shape=nodal_shape)

        times = _exact_datetime64(self.times)
        coords = jax.device_get(self._coords)

        additional_coords = {}
        if self._physics.cached_coords is not None and hasattr(self._physics.cached_coords, 'xarray_additional_coords'):
            additional_coords = dict(self._physics.cached_coords.xarray_additional_coords())
        # Aerosol-mode coordinate so per-mode JAM state fields (``jam_state.*``,
        # shaped ``(mode, level, lon, lat)``) serialize with a named ``mode`` dim
        # rather than failing the shape→dims lookup. Sourced from the aerosol
        # population spec carried by the microphysics term.
        for _term in getattr(self._physics, 'terms', []):
            _spec = getattr(_term, 'spec', None)
            if _spec is not None and hasattr(_spec, 'mode_shorts'):
                _mode_shorts = list(_spec.mode_shorts)
                # data_to_xarray assigns dims purely by array shape, so a mode
                # axis whose length equals the vertical layer count is genuinely
                # indistinguishable from the level axis — a (mode, level, lon,
                # lat) field can't be disambiguated from (level, …). This only
                # bites the unphysical case n_modes == n_levels (MAM4 has 4
                # modes, so only an L4 run). Fail early and specifically rather
                # than deep inside data_to_xarray's generic shape lookup.
                if len(_mode_shorts) == coords.vertical.layers:
                    raise ValueError(
                        f"Aerosol mode count ({len(_mode_shorts)}) equals the "
                        f"vertical layer count ({coords.vertical.layers}); the "
                        "per-mode aerosol state can't be given a distinct 'mode' "
                        "dimension because data_to_xarray infers dims from shape "
                        "alone. Use a vertical resolution other than "
                        f"{coords.vertical.layers} levels to serialize jam_state."
                    )
                additional_coords['mode'] = np.asarray(_mode_shorts)
                break
        # Spectral-band coordinates for the JAM per-band optics fields
        # (#584): ``*_sw_per_band`` / ``*_lw_per_band`` are
        # ``(time, band, level, lon, lat)`` and need a named band dim or
        # the shape→dims lookup fails (first hit by the first full-output
        # echam-jam run after #584). Lengths come from the arrays
        # themselves (RRTMGP: 14 SW / 16 LW); the additional_coords
        # collision check still guards a band count equal to the layer
        # count.
        # Band count 1 (grey radiation) is skipped: a length-1 coord here
        # would shadow the existing ``(1, ...)`` surface-axis mappings for
        # every other field; those fields already serialize via that axis.
        for _key, _val in physics_preds_dict.items():
            for _suffix, _dim in (('_sw_per_band', 'sw_band'),
                                  ('_lw_per_band', 'lw_band')):
                if (_key.endswith(_suffix) and _dim not in additional_coords
                        and getattr(_val, 'ndim', 0) >= 2
                        and _val.shape[1] > 1):
                    additional_coords[_dim] = np.arange(_val.shape[1])

        pred_ds = data_to_xarray(
            dynamics_predictions.asdict() | physics_preds_dict,
            coords=coords, serialize_coords_to_attrs=False,
            times=np.arange(times.shape[0]),
            additional_coords=additional_coords,
        )

        # Attach units / descriptions from the physics-specific units tables.
        # ``Physics`` is a structural contract, so a physics predating
        # ``units_table_paths`` still produces output, just undocumented.
        table_paths = getattr(self._physics, "units_table_paths", tuple)()
        units_df = pd.concat(
            [pd.read_csv(p) for p in (DYNAMICS_UNITS_TABLE_CSV_PATH, *table_paths)],
            ignore_index=True)
        # First table listed wins a duplicated variable name: the dynamics
        # table is authoritative, then terms in composition order.
        units_df = units_df.drop_duplicates(subset="Variable", keep="first")
        for var, unit, desc in zip(units_df["Variable"], units_df["Units"], units_df["Description"]):
            if var in pred_ds:
                pred_ds[var].attrs["units"] = unit
                pred_ds[var].attrs["description"] = desc

        # Per-term output metadata (#740). Each PhysicsTerm declares CF/units
        # attributes for the diagnostics it computes (``output_attrs``, keyed by
        # the dotted output names) — the home for metadata the per-physics CSVs
        # never listed, notably the whole radiation flux set. Applied AFTER the
        # CSV loop so a term declaration overrides the CSV (more specific wins),
        # but BEFORE ``cf_metadata.finalize_output`` so its own curated names
        # (vertical coordinates, core prognostics) still win last. Shared with
        # the non-modal delegation branch above.
        _apply_term_output_attrs(pred_ds, self._physics)

        # Exact model timestamps replace the temporary positional coordinate.
        pred_ds["time"] = ("time", times)

        # Put the file into the output convention: BOTH vertical axes
        # surface-first, with the sigma/hybrid coordinates and CF attributes
        # that say so. ``cf_metadata`` owns the flip — doing it inline here is
        # how ``level`` came to be flipped while ``level_i`` was not (#710).
        return cf_metadata.finalize_output(pred_ds, vertical=coords.vertical)


def _model_predictions_flatten(mp):
    """Flatten ModelPredictions for JAX pytree operations (tree_map, etc.).

    Only the internal Predictions pytree is treated as array data. Coords and
    physics are not in aux_data so that ``tree_map`` works across ModelPredictions
    from different Model instances.
    """
    children = (mp._predictions, mp._observations)
    return children, None


def _model_predictions_unflatten(aux_data, children):
    return ModelPredictions(children[0], None, None, observations=children[1])


jax.tree_util.register_pytree_node(
    ModelPredictions,
    _model_predictions_flatten,
    _model_predictions_unflatten,
)
