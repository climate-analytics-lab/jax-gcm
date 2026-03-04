import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import tree_map
from importlib import resources
import dinosaur
import functools
from dinosaur.coordinate_systems import CoordinateSystem, HorizontalGridTypes
from dinosaur.primitive_equations import PrimitiveEquationsSpecs
from dinosaur.scales import SI_SCALE
from dinosaur import typing
from typing import Any, Mapping, MutableMapping
from dinosaur.xarray_utils import _maybe_update_shape_and_dim_with_realization_time_sample
import xarray

DYNAMICS_UNITS_TABLE_CSV_PATH = resources.files('jcm') / 'dynamics_units_table.csv'

TRUNCATION_FOR_NODAL_SHAPE = {
    (64, 32): 21,
    (96, 48): 31,
    (128, 64): 42,
    (256, 128): 85,
    (320, 160): 106,
    (360, 180): 119,
    (512, 256): 170,
    (640, 320): 213,
    (1024, 512): 340,
    (1280, 640): 425,
}

VALID_NODAL_SHAPES = tuple(TRUNCATION_FOR_NODAL_SHAPE.keys())
VALID_TRUNCATIONS = tuple(TRUNCATION_FOR_NODAL_SHAPE.values())

def get_coords(sigma_boundaries, spectral_truncation=31, nodal_shape=None, spmd_mesh=None) -> CoordinateSystem:
    """Return a CoordinateSystem object for the given sigma boundaries and horizontal resolution.

    This is a physics-agnostic function. Use physics-specific helpers for default sigma boundaries:
    - jcm.physics.speedy.utils.get_speedy_coords()
    - jcm.physics.held_suarez.utils.get_held_suarez_coords()

    Args:
        sigma_boundaries: Array of sigma layer boundaries (required)
        spectral_truncation: Spectral truncation number (default 31)
        nodal_shape: Optional nodal shape (ix, il) to infer spectral_truncation
        spmd_mesh: Optional SPMD mesh for parallelization

    Returns:
        CoordinateSystem object
        
    """
    from dinosaur.spherical_harmonic import FastSphericalHarmonics, RealSphericalHarmonics

    if nodal_shape is not None:
        if nodal_shape not in VALID_NODAL_SHAPES:
            raise ValueError(f"Invalid nodal shape: {nodal_shape}. Must be one of: {VALID_NODAL_SHAPES}.")
        spectral_truncation = TRUNCATION_FOR_NODAL_SHAPE[nodal_shape]
    elif spectral_truncation not in VALID_TRUNCATIONS:
        raise ValueError(f"Invalid horizontal resolution: {spectral_truncation}. Must be one of: {VALID_TRUNCATIONS}.")
    horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f'T{spectral_truncation}')

    physics_specs = PrimitiveEquationsSpecs.from_si(scale=SI_SCALE)

    if spmd_mesh is not None:
        spmd_mesh = jax.make_mesh(spmd_mesh, ('x', 'y', 'z'))
        spherical_harmonics_impl = FastSphericalHarmonics
    else:
        spherical_harmonics_impl = RealSphericalHarmonics

    return CoordinateSystem(
        horizontal=horizontal_grid(radius=physics_specs.radius,
                                   spherical_harmonics_impl=spherical_harmonics_impl),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(sigma_boundaries),
        spmd_mesh=spmd_mesh
    )

# Function to take a field in grid space and truncate it to a given wavenumber
def spectral_truncation(grid: HorizontalGridTypes, grid_field):
    """grid_field: field in grid space
    trunc: truncation level, # of wavenumbers to keep
    """
    spectral_field = grid.to_modal(grid_field)
    nx,mx = spectral_field.shape
    n_indices, m_indices = jnp.meshgrid(jnp.arange(nx), jnp.arange(mx), indexing='ij')
    total_wavenumber = m_indices + n_indices

    # the spectral resolution is total wavenumbers - 2
    truncation_number = (grid.total_wavenumbers - 2)

    spectral_field = jnp.where(total_wavenumber > truncation_number, 0.0, spectral_field)

    truncated_grid_field = grid.to_nodal(spectral_field)

    return truncated_grid_field

def validate_ds(ds, expected_structure):
    """Validate that an xarray Dataset has the expected variables and dimensions.

    Args:
        ds (xr.Dataset): The dataset to validate.
        expected_structure (dict): A dictionary where keys are variable names and values are tuples of expected dimension names.

    """
    missing_vars = set(expected_structure) - set(ds.data_vars)
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")
    for var, expected_dims in expected_structure.items():
        actual_dims = ds[var].dims
        if actual_dims != expected_dims:
            raise ValueError(
                f"Variable '{var}' has dims {actual_dims}, expected {expected_dims}"
            )

@jit
def pass_fn(operand):
    return operand

def ones_like(x):
    return tree_map(jnp.ones_like, x)

def _index_if_3d(arr, key):
    return arr[:, :, key] if arr.ndim > 2 else arr

def tree_index_3d(tree, key):
    return tree_map(lambda arr: _index_if_3d(jnp.array(arr), key), tree)

def _check_type_ones_like_tangent(x):
        if jnp.result_type(x) == jnp.result_type(float):
            return jnp.ones_like(x)
        # in case of a bool or int, return a float0 denoting the lack of tangent space
        # jax requires that we use numpy to construct the float0 scalar
        # because it is a semantic placeholder not backed by any array data / memory allocation
        return np.ones((), dtype=jax.dtypes.float0)

def ones_like_tangent(pytree):
    return tree_map(_check_type_ones_like_tangent, pytree)

def _check_type_zeros_like_tangent(x):
        if jnp.result_type(x) == jnp.result_type(float):
            return jnp.zeros_like(x)
        return np.zeros((), dtype=jax.dtypes.float0)

def zeros_like_tangent(pytree):
    return tree_map(_check_type_zeros_like_tangent, pytree)

def _check_type_convert_to_float(x):
    return jnp.asarray(x, dtype=float)

def convert_to_float(x): 
    return tree_map(_check_type_convert_to_float, x)

# Revert object with type float back to true type
def _check_type_convert_back(x, x0):
    return x if jnp.result_type(x0) == jnp.result_type(float) else x0

def convert_back(x, x0):
    return tree_map(_check_type_convert_back, x, x0)

def _infer_dims_shape_and_coords(
    coords: CoordinateSystem,
    times: typing.Array | None,
    sample_ids: typing.Array,
    additional_coords: typing.Mapping[str, typing.Array],
) -> tuple[dict[str, typing.Array], dict[tuple[int, ...], tuple[int, ...]]]:
    """Return full coordinates for given grids and default shape to dims mapping.

    Args:
        coords: horizontal and vertical descritization.
        times: expected time values. If `None` time shape/dim is not added.
        sample_ids: expected sample values. If `None` sample shape/dim is not added.
        additional_coords: additional coordinates to include.

    Returns:
        all_coords: mapping that represents all supported coordinates.
        shape_to_dims: mapping from array shape to dimensions. `sample` is assumed
        to come prior to `time`.

    """
    # Axes and coordinate names
    XR_SAMPLE_NAME = 'sample'
    XR_TIME_NAME = 'time'
    XR_LEVEL_NAME = 'level'
    XR_LON_NAME = 'lon'
    XR_LAT_NAME = 'lat'
    XR_LON_MODE_NAME = 'longitudinal_mode'
    XR_LAT_MODE_NAME = 'total_wavenumber'
    XR_REALIZATION_NAME = 'realization'

    # Axes for `Dataset`s in the nodal/spatial harmonic basis.
    NODAL_AXES_NAMES = (
        XR_LON_NAME,
        XR_LAT_NAME,
    )

    MODAL_AXES_NAMES = (
        XR_LON_MODE_NAME,
        XR_LAT_MODE_NAME,
    )
    
    lon_k, lat_k = coords.horizontal.modal_axes  # k stands for wavenumbers
    lon, sin_lat = coords.horizontal.nodal_axes
    all_xr_coords = {
        XR_LON_NAME: lon * 180 / np.pi,
        XR_LAT_NAME: np.arcsin(sin_lat) * 180 / np.pi,
        XR_LON_MODE_NAME: lon_k,
        XR_LAT_MODE_NAME: lat_k,
        XR_LEVEL_NAME: coords.vertical.centers,
        **additional_coords,
    }
    if times is not None:
        all_xr_coords[XR_TIME_NAME] = times
    if sample_ids is not None:
        all_xr_coords[XR_SAMPLE_NAME] = sample_ids
    basic_shape_to_dims = {}
    basic_shape_to_dims[tuple()] = tuple()  # scalar variables
    modal_shape = coords.horizontal.modal_shape
    nodal_shape = coords.horizontal.nodal_shape
    basic_shape_to_dims[(coords.vertical.layers,) + modal_shape] = (
        XR_LEVEL_NAME,
    ) + MODAL_AXES_NAMES
    basic_shape_to_dims[(coords.vertical.layers,) + nodal_shape] = (
        XR_LEVEL_NAME,
    ) + NODAL_AXES_NAMES
    basic_shape_to_dims[nodal_shape] = NODAL_AXES_NAMES
    basic_shape_to_dims[modal_shape] = MODAL_AXES_NAMES
    basic_shape_to_dims[(coords.vertical.layers,)] = (XR_LEVEL_NAME,)
    basic_shape_to_dims[sin_lat.shape] = (XR_LAT_NAME,)
    # Add unconventional shape for nodal covariate surface data, which have dim=2
    # (lon, lat) in xarray. The singleton dimension for level is added when
    # converting to covariate data.
    basic_shape_to_dims[coords.surface_nodal_shape] = NODAL_AXES_NAMES
    for dim, value in additional_coords.items():
        if dim == XR_REALIZATION_NAME:
            continue  # Handled in _maybe_update_shape_and_dim_with_time_sample
        if value.ndim != 1:
            raise ValueError(
                '`additional_coords` must be 1d vectors, but got: '
                f'{value.shape=} for {dim=}'
            )
        if value.shape == (coords.vertical.layers,):
            raise ValueError(
                f'`additional_coords` {dim=} has shape={value.shape} that collides '
                f'with {XR_LEVEL_NAME=}. Since matching of axes is done using shape, '
                'consider renaming after the fact.'
            )
        basic_shape_to_dims[value.shape + modal_shape] = (dim,) + MODAL_AXES_NAMES
        basic_shape_to_dims[value.shape + nodal_shape] = (dim,) + NODAL_AXES_NAMES
        basic_shape_to_dims[value.shape] = (dim,)
        basic_shape_to_dims[(coords.vertical.layers,) + value.shape] = (XR_LEVEL_NAME,) + (dim,)

    update_shape_dims_fn = functools.partial(
        _maybe_update_shape_and_dim_with_realization_time_sample,
        times=times,
        sample_ids=sample_ids,
        include_realization=XR_REALIZATION_NAME in additional_coords,
    )
    shape_to_dims = {}
    for shape, dims in basic_shape_to_dims.items():
        full_shape, full_dims = update_shape_dims_fn(shape, dims)
        shape_to_dims[full_shape] = full_dims
    return all_xr_coords, shape_to_dims  # pytype: disable=bad-return-type

def data_to_xarray(
    data: dict,
    *,
    coords: CoordinateSystem,
    times: typing.Array | None,
    sample_ids: typing.Array | None = None,
    additional_coords: MutableMapping[str, typing.Array] | None = None,
    attrs: Mapping[str, Any] | None = None,
    serialize_coords_to_attrs: bool = True,
) -> xarray.Dataset:
  """Return a sample/time referenced xarray.Dataset of primitive equation data.

  Args:
    data: dictionary representation of the primitive equation states.
    coords: horizontal and vertical descritization.
    times: xarray coordinates to use for `time` axis.
    sample_ids: xarray coordinates to use for `sample` axis.
    additional_coords: additional coordinates to include.
    attrs: additional attributes to include in the xarray.Dataset metadata.
    serialize_coords_to_attrs: whether to save serialized coords to attrs.

  Returns:
    xarray.Dataset with containing `data`.

  """
  XR_SURFACE_NAME = 'surface'
  # check that prognostic and tracer names do not collide;
  prognostic_keys = set(data.keys()) - {'tracers'} - {'diagnostics'}
  tracer_keys = data['tracers'].keys() if 'tracers' in data else set()
  diagnostic_keys = (
      data['diagnostics'].keys() if 'diagnostics' in data else set()
  )
  if not prognostic_keys.isdisjoint(tracer_keys):
    raise ValueError(
        'Tracer names collide with prognostic variables',
        f'Tracers: {tracer_keys}; prognostics: {prognostic_keys}',
    )
  if not prognostic_keys.isdisjoint(diagnostic_keys):
    raise ValueError(
        'Diagnostic names collide with prognostic variables',
        f'Diagnostic: {diagnostic_keys}; ',
        f'prognostics: {prognostic_keys}',
    )

  if additional_coords is None:
    additional_coords = {}
  # if XR_SURFACE_NAME is not specified manually, set by default.
  if (coords.vertical.layers != 1) and (
      XR_SURFACE_NAME not in additional_coords
  ):
    additional_coords[XR_SURFACE_NAME] = np.ones(1)
  all_coords, shape_to_dims = _infer_dims_shape_and_coords(
      coords, times, sample_ids, additional_coords
  )

  dims_in_state = set()  # keep track which coordinates should be included.
  data_vars = {}
  for key in prognostic_keys:
    value = data[key]
    if value.shape not in shape_to_dims:
      raise ValueError(
          f'Value of shape {value.shape} is not in {shape_to_dims=}'
      )
    else:
      dims = shape_to_dims[value.shape]
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  for key in tracer_keys:
    value = data['tracers'][key]
    if value.shape not in shape_to_dims:
      raise ValueError(f'Value of shape {value.shape} is not recognized.')
    else:
      dims = shape_to_dims[value.shape]
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  for key in diagnostic_keys:
    value = data['diagnostics'][key]
    if value.shape not in shape_to_dims:
      raise ValueError(f'Value of shape {value.shape} is not recognized.')
    else:
      dims = shape_to_dims[value.shape]
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  dataset_attrs = coords.asdict() if serialize_coords_to_attrs else {}
  if attrs is not None:
    for key in dataset_attrs.keys():
      if key in attrs:
        raise ValueError(f'Key {key} is not allowed in `attrs`.')
    dataset_attrs.update(attrs)
  # only include coordinates for dimensions that are present in the dataset.
  coords = {k: v for k, v in all_coords.items() if k in dims_in_state}
  return xarray.Dataset(data_vars, coords, attrs=dataset_attrs)