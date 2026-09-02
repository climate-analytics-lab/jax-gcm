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
from numpy import timedelta64
import pandas as pd

from jcm import cf_metadata, provenance
from jcm.dycore.base import Predictions
from jcm.physics_interface import Physics
from jcm.utils import DYNAMICS_UNITS_TABLE_CSV_PATH, data_to_xarray

logger = logging.getLogger(__name__)


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
                 obs_t0_days=None, obs_dt_seconds=None,
                 snapshots=None, snapshot_variables=(),
                 snapshot_interval_days=None, params=None):
        self._predictions = predictions
        self._coords = coords
        self._physics = physics
        self._dycore = dycore
        self._observations = observations
        self._observers = tuple(observers)
        self._obs_t0_days = obs_t0_days
        self._obs_dt_seconds = obs_dt_seconds
        self._snapshots = snapshots
        self._snapshot_variables = tuple(snapshot_variables)
        self._snapshot_interval_days = snapshot_interval_days
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

    def snapshot_dataset(self):
        """Interval-instantaneous 2-D snapshots as one xarray Dataset.

        The AeroCom 3-hourly stream (jax-gcm#586): each requested variable
        comes back as ``(snap_time, lon, lat)`` at the snapshot cadence,
        with ``snap_time`` in days from the window start (first snapshot
        one interval in, matching the post-step sampling convention).
        Empty dict semantics: returns ``None`` when the run requested no
        snapshots.
        """
        if not self._snapshots or self._snapshot_interval_days is None:
            return None
        import xarray as xr

        snaps = jax.device_get(self._snapshots)
        nlon, nlat = self._coords.horizontal.nodal_shape
        first = next(iter(snaps.values()))
        n = first.shape[0]
        t = (np.arange(1, n + 1) * self._snapshot_interval_days)
        data = {}
        for name, arr in snaps.items():
            arr = np.asarray(arr).reshape(n, nlon, nlat)
            data[name.replace(".", "_")] = (("snap_time", "lon", "lat"), arr)
        lon = self._coords.horizontal.nodal_axes[0] * 180.0 / np.pi
        lat = np.arcsin(self._coords.horizontal.nodal_axes[1]) * 180.0 / np.pi
        return xr.Dataset(
            data,
            coords={"snap_time": t, "lon": lon, "lat": lat},
            attrs={"snapshot_interval_days": self._snapshot_interval_days,
                   "sampling": "instantaneous (post-step)",
                   **provenance.params_attrs(self._params)},
        )

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
        samples_host = jax.device_get(self._observations)
        stamp = provenance.params_attrs(self._params)
        datasets = {}
        for obs, samples in zip(self._observers, samples_host):
            ds = obs.to_dataset(samples, self._obs_t0_days,
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

        """
        ds = self._trajectory_dataset()
        ds.attrs.update(provenance.params_attrs(self._params))
        return ds

    def _trajectory_dataset(self):
        """Build the trajectory Dataset, before provenance stamping."""
        # Backends whose native horizontal layout is not the separable
        # lat/lon grid the legacy path below assumes (pySES cubed-sphere
        # columns) own their trajectory conversion per the DynamicalCore
        # protocol; delegate whenever the grid has no modal axes.
        if self._dycore is not None and not hasattr(
                self._coords.horizontal, "modal_axes"):
            times = jax.device_get(self.times)
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

        times = jax.device_get(self.times)
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
            times=times - times[0],
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

        # Convert sim-day timestamps to datetimes. Done before the CF pass so
        # ``time`` is already a datetime axis when its attributes are set.
        pred_ds['time'] = (
            times * (timedelta64(1, 'D') / timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')

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
