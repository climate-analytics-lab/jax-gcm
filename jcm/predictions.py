"""User-facing prediction container and its xarray serialization.

:class:`ModelPredictions` wraps the internal :class:`~jcm.dycore.base.Predictions`
pytree with the coordinate system and physics module needed to turn a run into an
analysis-ready :class:`xarray.Dataset`. Returned by :meth:`jcm.model.Model.run`,
:meth:`~jcm.model.Model.resume`, and :meth:`~jcm.model.Model.run_from_state`.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from numpy import timedelta64
import pandas as pd

from jcm.dycore.base import Predictions
from jcm.physics_interface import Physics
from jcm.utils import DYNAMICS_UNITS_TABLE_CSV_PATH, data_to_xarray


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

    def __init__(self, predictions: Predictions, coords, physics: Physics,
                 dycore=None):  # noqa: D107
        self._predictions = predictions
        self._coords = coords
        self._physics = physics
        self._dycore = dycore

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
        # Backends whose native horizontal layout is not the separable
        # lat/lon grid the legacy path below assumes (pySES cubed-sphere
        # columns) own their trajectory conversion per the DynamicalCore
        # protocol; delegate whenever the grid has no modal axes.
        if self._dycore is not None and not hasattr(
                self._coords.horizontal, "modal_axes"):
            times = jax.device_get(self.times)
            return self._dycore.to_xarray(self._predictions, times)

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

        pred_ds = data_to_xarray(
            dynamics_predictions.asdict() | physics_preds_dict,
            coords=coords, serialize_coords_to_attrs=False,
            times=times - times[0],
            additional_coords=additional_coords,
        )

        # Attach units / descriptions from the physics-specific units table.
        units_df = pd.read_csv(DYNAMICS_UNITS_TABLE_CSV_PATH)
        if self._physics.UNITS_TABLE_CSV_PATH is not None:
            units_df = pd.concat([units_df, pd.read_csv(self._physics.UNITS_TABLE_CSV_PATH)], ignore_index=True)
        for var, unit, desc in zip(units_df["Variable"], units_df["Units"], units_df["Description"]):
            if var in pred_ds:
                pred_ds[var].attrs["units"] = unit
                pred_ds[var].attrs["description"] = desc

        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere.
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # Convert sim-day timestamps to datetimes.
        pred_ds['time'] = (
            times * (timedelta64(1, 'D') / timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')

        return pred_ds


def _model_predictions_flatten(mp):
    """Flatten ModelPredictions for JAX pytree operations (tree_map, etc.).

    Only the internal Predictions pytree is treated as array data. Coords and
    physics are not in aux_data so that ``tree_map`` works across ModelPredictions
    from different Model instances.
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
