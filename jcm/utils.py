import jax
import jax.numpy as jnp
import logging
import numpy as np
from jax.tree_util import tree_map
from importlib import resources
import dinosaur
import functools
from dinosaur.coordinate_systems import CoordinateSystem, HorizontalGridTypes
from dinosaur.hybrid_coordinates import HybridCoordinates
from dinosaur.sigma_coordinates import SigmaCoordinates
from dinosaur import typing
from typing import Any, Mapping, MutableMapping, Union
from dinosaur.xarray_utils import _maybe_update_shape_and_dim_with_realization_time_sample
import xarray

from jcm import cf_metadata

DYNAMICS_UNITS_TABLE_CSV_PATH = resources.files('jcm') / 'dynamics_units_table.csv'

TRUNCATION_FOR_NODAL_SHAPE = {
    (64, 32): 21,
    (96, 48): 31,
    (128, 64): 42,
    (192, 96): 63,    # T63 — ECHAM standard production grid
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

def get_coords(
    vertical_coords: Union[typing.Array, SigmaCoordinates, HybridCoordinates],
    spectral_truncation=31,
    nodal_shape=None,
    spmd_mesh=None,
    constants=None,
) -> CoordinateSystem:
    """Return a CoordinateSystem object for the given vertical and horizontal resolution.

    This is a physics-agnostic function. Use physics-specific helpers for default
    vertical coordinates:
    - jcm.physics.speedy.utils.get_speedy_coords()
    - jcm.physics.held_suarez.utils.get_held_suarez_coords()
    - jcm.physics.echam.echam_levels.get_echam_levels()

    Args:
        vertical_coords: Vertical coordinate specification. Can be one of:
            - An array of sigma layer boundaries (wrapped in SigmaCoordinates).
            - A SigmaCoordinates instance (passed through).
            - A HybridCoordinates instance (passed through), e.g. from
              jcm.physics.echam.echam_levels.get_echam_levels().
        spectral_truncation: Spectral truncation number (default 31)
        nodal_shape: Optional nodal shape (ix, il) to infer spectral_truncation
        spmd_mesh: Optional tuple ``(x, y, z)`` describing the SPMD device mesh
            for sharding (longitude, latitude, vertical). The product must equal
            ``len(jax.devices())``. When set, a ``FastSphericalHarmonics`` implementation
            is used; otherwise the model runs on a single device. This is the only
            place to configure SPMD — ``Model`` consumes the sharding via ``coords``.

            Recommended layout (see docs/source/design/parallelization.md): a
            longitude-only mesh ``(N, 1, 1)``. The dycore shards longitude for
            the spectral transforms while column physics keeps whole columns
            local, so the two layouts coincide and no per-step reshard is
            needed. Splitting latitude (``y > 1``) forces a reshard at the
            physics column-flatten, so keep ``y = 1`` unless you have measured
            a reason not to.

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
    # Most truncations have a dedicated dinosaur factory (Grid.T31, T42, …);
    # T63 doesn't, so build it directly via Grid.construct with
    # gaussian_nodes=(max_wavenumber+1)/2-rounded so the nodal grid matches
    # the ECHAM T63 file (192 lon × 96 lat). For all other supported
    # truncations the dedicated Grid.T* factory uses the same convention,
    # so we use it where available.
    if spectral_truncation == 63:
        def horizontal_grid(**kwargs):
            return dinosaur.spherical_harmonic.Grid.construct(
                max_wavenumber=63, gaussian_nodes=48, **kwargs)
    else:
        horizontal_grid = getattr(
            dinosaur.spherical_harmonic.Grid, f'T{spectral_truncation}')

    # The horizontal grid's radius is part of the physical-constants set and
    # must match the dycore's physics_specs.radius (dinosaur enforces this).
    # Source it from the live PhysicalConstants singleton so coords and dynamics
    # agree and a prior set_constants(...) is honoured.
    import jcm.constants as _jcm_constants
    constants = constants if constants is not None else _jcm_constants.physical_constants
    grid_radius = constants.rearth

    if spmd_mesh is not None:
        # Build the mesh with Auto axis types. dinosaur (and the physics-
        # boundary wiring in jax-gcm) drive sharding with
        # ``jax.lax.with_sharding_constraint``, which requires Auto axes —
        # JAX's ``make_mesh`` otherwise defaults to Explicit axes, under which
        # those constraints raise and demand the ``reshard`` API instead.
        from jax.sharding import AxisType
        spmd_mesh = jax.make_mesh(
            spmd_mesh, ('x', 'y', 'z'),
            axis_types=(AxisType.Auto,) * 3,
        )
        spherical_harmonics_impl = FastSphericalHarmonics
    else:
        spherical_harmonics_impl = RealSphericalHarmonics

    if isinstance(vertical_coords, (SigmaCoordinates, HybridCoordinates)):
        vertical = vertical_coords
    else:
        vertical = SigmaCoordinates(vertical_coords)

    return CoordinateSystem(
        horizontal=horizontal_grid(radius=grid_radius,
                                   spherical_harmonics_impl=spherical_harmonics_impl),
        vertical=vertical,
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

def ones_like(x):
    return tree_map(jnp.ones_like, x)

def _check_type_ones_like_tangent(x):
        if jnp.result_type(x) == jnp.result_type(float):
            return jnp.ones_like(x)
        # in case of a bool or int, return a float0 denoting the lack of tangent space
        # jax requires that we use numpy to construct the float0 scalar
        # because it is a semantic placeholder not backed by any array data / memory allocation
        return np.ones((), dtype=jax.dtypes.float0)

def ones_like_tangent(pytree):
    return tree_map(_check_type_ones_like_tangent, pytree)

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

    # HybridCoordinates uses `get_sigma_centers(p_ref)` and doesn't expose
    # a .centers attribute; fall back to that for xarray's level axis.
    vertical = coords.vertical
    if hasattr(vertical, 'centers'):
        level_coords = vertical.centers
    else:
        from jcm.constants import p0
        level_coords = np.asarray(vertical.get_sigma_centers(p0))

    all_xr_coords = {
        XR_LON_NAME: lon * 180 / np.pi,
        XR_LAT_NAME: np.arcsin(sin_lat) * 180 / np.pi,
        XR_LON_MODE_NAME: lon_k,
        XR_LAT_MODE_NAME: lat_k,
        XR_LEVEL_NAME: level_coords,
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
    # Column-vectorized layout: physics terms running under
    # ``ComposablePhysics(vectorize_columns=True)`` write per-column
    # scalars and profiles in flattened ``(ncols,)`` / ``(nlev, ncols)``
    # shape rather than ``(lon, lat)`` / ``(nlev, lon, lat)``. Map the
    # flat shapes back to the same xarray dims so downstream reshape is
    # a no-op axis relabel.
    nlon, nlat = nodal_shape
    nlev = coords.vertical.layers
    basic_shape_to_dims[(nlon * nlat,)] = NODAL_AXES_NAMES
    basic_shape_to_dims[(nlev, nlon * nlat)] = (
        XR_LEVEL_NAME,
    ) + NODAL_AXES_NAMES
    # Half-level fields (e.g. fluxes, half-pressure) — emit them on a
    # ``level_i`` (interface) axis so they don't clash with the full-level
    # ``level`` coord, which has length nlev.
    #
    # Both vertical axes are in the physics-internal TOA-first frame here, and
    # the interface axis is a bare index. ``jcm.cf_metadata.finalize_output``
    # is what turns a trajectory into the *file* convention (both axes
    # surface-first, real sigma values, CF attributes) — this function stays
    # the low-level shape->dims mapper.
    XR_LEVEL_INTERFACE_NAME = cf_metadata.LEVEL_INTERFACE_DIM
    if XR_LEVEL_INTERFACE_NAME not in all_xr_coords:
        all_xr_coords[XR_LEVEL_INTERFACE_NAME] = np.arange(nlev + 1)
    basic_shape_to_dims[(nlev + 1,) + nodal_shape] = (
        XR_LEVEL_INTERFACE_NAME,
    ) + NODAL_AXES_NAMES
    basic_shape_to_dims[(nlev + 1, nlon * nlat)] = (
        XR_LEVEL_INTERFACE_NAME,
    ) + NODAL_AXES_NAMES
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
        # ``(extra, level, horizontal)`` fields — e.g. JAM's per-aerosol-mode
        # state (mode, level, lon/lat), incl. the column-vectorized layout.
        basic_shape_to_dims[
            value.shape + (coords.vertical.layers,) + nodal_shape
        ] = (dim, XR_LEVEL_NAME) + NODAL_AXES_NAMES
        basic_shape_to_dims[
            value.shape + (coords.vertical.layers, nlon * nlat)
        ] = (dim, XR_LEVEL_NAME) + NODAL_AXES_NAMES

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

  def _maybe_reshape_to_dims(value, dims, all_coords):
    """If ``value`` has a flattened ncols axis but ``dims`` calls for
    separate (lon, lat) axes, reshape it. Otherwise pass through.
    """
    if 'lon' not in dims or 'lat' not in dims:
      return value
    expected = tuple(len(all_coords[d]) for d in dims)
    if value.shape == expected:
      return value
    return value.reshape(expected)
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
      value = _maybe_reshape_to_dims(value, dims, all_coords)
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  for key in tracer_keys:
    value = data['tracers'][key]
    if value.shape not in shape_to_dims:
      raise ValueError(f'Value of shape {value.shape} is not recognized.')
    else:
      dims = shape_to_dims[value.shape]
      value = _maybe_reshape_to_dims(value, dims, all_coords)
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  for key in diagnostic_keys:
    value = data['diagnostics'][key]
    if value.shape not in shape_to_dims:
      raise ValueError(f'Value of shape {value.shape} is not recognized.')
    else:
      dims = shape_to_dims[value.shape]
      value = _maybe_reshape_to_dims(value, dims, all_coords)
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


# ---------------------------------------------------------------------------
# Helpers for ``SingleColumnModel`` / ``PrescribedStateModel``
# ---------------------------------------------------------------------------

def _tracer_names(required_tracers) -> tuple[str, ...]:
    """Names from ``required_tracers``, which may be specs or bare strings.

    ``ComposablePhysics.required_tracers()`` returns ``TracerSpec`` objects;
    callers that only know names (a notebook, a test) may pass strings.
    """
    return tuple(t if isinstance(t, str) else t.name for t in required_tracers)


def _resolve_tracer_vars(ds, tracer_vars, required_tracers) -> dict[str, str]:
    """Decide which file variables become ``PhysicsState.tracers`` (#718).

    An explicit ``tracer_vars`` mapping always wins — the empty mapping
    included, which is how a caller says "load no tracers". ``None`` means
    *infer*, and inference is only possible when the caller declares what it
    needs: the file alone cannot tell a prognostic tracer from a diagnostic,
    since both are written as ordinary level-dimensioned variables. So the
    names come from the configured physics' ``required_tracers()``, and the
    identically-named variables the file actually carries are picked up.

    Inferring from the physics rather than matching a hardcoded ``qc``/``qi``
    list is what makes this work for every scheme: the two-moment cloud
    scheme also needs ``qnc``/``qni``, and JAM declares a tracer per aerosol
    species. Those declarations already exist; this just reads them.

    Loading nothing when nothing is declared is correct. Loading nothing when
    the physics *does* declare ``qc``/``qi`` and the file has them is the #718
    failure — radiation then sees a cloud-free sky and returns plausible,
    wrong fluxes with no error — so both the pick-up and every drop are
    logged.
    """
    names = _tracer_names(required_tracers or ())

    if tracer_vars is None:
        resolved = {name: name for name in names if name in ds}
        if resolved:
            logging.info(
                "load_states_from_xarray: loading tracer(s) %s, declared by "
                "the physics and present in the state file.",
                ", ".join(resolved),
            )
        missing = [name for name in names if name not in resolved]
        if missing:
            logging.warning(
                "load_states_from_xarray: the physics declares tracer(s) %s "
                "but the state file carries no such variable(s); physics will "
                "see them as absent (zero).",
                ", ".join(missing),
            )
        return resolved

    dropped = [name for name in names if name not in tracer_vars and name in ds]
    if dropped:
        logging.warning(
            "load_states_from_xarray: tracer(s) %s are declared by the physics "
            "and present in the state file, but the explicit tracer_vars "
            "mapping omits them; physics will see them as zero.",
            ", ".join(dropped),
        )
    return dict(tracer_vars)


def _check_top_first_pressures(ds, surface_pressure_var: str) -> None:
    """Verify the converted Dataset really is in the top-first physics frame.

    Orientation is *inferred* from the ``level`` coordinate, and a file whose
    ``level`` is a bare index carries no orientation at all — so the
    inference needs an independent check (#718). In the top-first frame
    ``pressure_full`` must increase with index, and its last (surface) entry
    must be within a factor of two of the surface pressure the state itself
    carries. Both hold to within a percent or so for any real hybrid or sigma
    column — the lowest full level sits just above the surface — while an
    inverted column misses by the ratio of surface to model-top pressure,
    four or five orders of magnitude. The factor of two is deliberately far
    looser than any physical column needs: this is a wrong-orientation
    detector, not a precision check on the vertical grid.

    Skipped when the file has no ``pressure_full`` (a trimmed restart file
    holding only the prognostic fields, e.g. ``spinup_state.nc``) or when the
    sampled column is not finite; those fall back to the ``level``-coordinate
    inference alone.

    Args:
      ds: Dataset already converted to the top-first frame.
      surface_pressure_var: name of the variable holding
        ``normalized_surface_pressure`` — p_s / p0 by definition of the
        ``PhysicsState`` field, whatever the file calls it.

    Raises:
      ValueError: the state is not top-first, so physics would pair each
        level's fields with another level's pressure.

    """
    if cf_metadata.PRESSURE_FULL not in ds:
        return

    from jcm import constants as c

    def _first_column(da):
        """One column: every non-level dimension reduced to its first index."""
        return da.isel({
            dim: 0 for dim in da.dims if dim != cf_metadata.LEVEL_DIM
        })

    pressure = np.asarray(
        _first_column(ds[cf_metadata.PRESSURE_FULL]).values, dtype=float)
    if (pressure.ndim != 1 or pressure.size < 2
            or not np.all(np.isfinite(pressure))):
        return

    if not np.all(np.diff(pressure) > 0):
        raise ValueError(
            f"{cf_metadata.PRESSURE_FULL!r} does not increase with level index "
            f"after conversion to the top-first physics frame "
            f"({pressure[0]:.4g} Pa at index 0, {pressure[-1]:.4g} Pa at the "
            f"last index). The file's vertical orientation could not be read "
            f"from its {cf_metadata.LEVEL_DIM!r} coordinate, so physics would "
            f"be handed each level's fields paired with another level's "
            f"pressure (#718). Reverse the level axis before loading, or write "
            f"the file through jcm.cf_metadata.finalize_output."
        )

    if surface_pressure_var not in ds:
        return
    surface_pressure = c.p0 * float(np.asarray(
        _first_column(ds[surface_pressure_var]).values, dtype=float).ravel()[0])
    if not np.isfinite(surface_pressure) or surface_pressure <= 0.0:
        return
    if not 0.5 < pressure[-1] / surface_pressure < 2.0:
        raise ValueError(
            f"the loaded state's surface {cf_metadata.PRESSURE_FULL} "
            f"({pressure[-1]:.4g} Pa) disagrees with the surface pressure it "
            f"carries ({surface_pressure:.4g} Pa) by more than a factor of "
            f"two. The vertical axis is inconsistent with the state's own "
            f"surface pressure — most likely an inverted column (#718)."
        )


def load_states_from_xarray(
    ds,
    u_wind_var: str = 'u_wind',
    v_wind_var: str = 'v_wind',
    temperature_var: str = 'temperature',
    specific_humidity_var: str = 'specific_humidity',
    geopotential_var: str = 'geopotential',
    surface_pressure_var: str = 'normalized_surface_pressure',
    tracer_vars: Mapping[str, str] | None = None,
    required_tracers=None,
):
    """Load a ``PhysicsState`` time series from an xarray ``Dataset``.

    The returned state is **always** in the top-first physics-internal frame
    (index 0 = model top), which is what ``PhysicsState`` and the dycore
    expect. Because every JCM output file is written surface-first (see
    :mod:`jcm.cf_metadata`), the file's vertical orientation is detected and
    the Dataset is flipped to top-first before values are extracted (#741).

    Orientation is inferred from the ``level`` coordinate **values**, not its
    attributes — attributes do not survive all xarray operations. A ``level``
    coordinate that is numeric, longer than one element, and *descending*
    (``level[0] > level[-1]``, i.e. sigma falling from ~1 at the surface to ~0
    at the top) marks a surface-first file, which is reversed via
    :func:`jcm.cf_metadata.orient_top_first`. An ascending ``level`` is already
    top-first and passes through. If the ``level`` coordinate is absent or
    non-numeric (a bare integer index), orientation cannot be determined; the
    data passes through unflipped, top-first is assumed, and a warning is
    logged. That inference is then *checked* against the file's own
    ``pressure_full`` where it has one, and a state that is not top-first is
    rejected rather than handed to physics (#718).

    Tracers beyond ``specific_humidity`` are resolved from ``tracer_vars`` when
    given and otherwise inferred from ``required_tracers`` — see
    :func:`_resolve_tracer_vars`. Passing neither loads no tracers, which for
    cloud-aware physics means a cloud-free sky, so callers that have a physics
    object should pass its ``required_tracers()``.

    Args:
      ds: Dataset containing the required variables.
      *_var: Variable names in ``ds`` for each ``PhysicsState`` field.
      tracer_vars: Explicit mapping ``tracer_name → ds_variable_name`` for
        tracers beyond ``specific_humidity``. ``None`` (the default) infers
        from ``required_tracers``; an empty mapping loads no tracers.
      required_tracers: What the consuming physics needs — an iterable of
        ``TracerSpec`` (as ``ComposablePhysics.required_tracers()`` returns)
        or of bare names. Used only to infer ``tracer_vars`` and to warn about
        declared tracers the file has but the caller drops.

    Returns:
      ``PhysicsState`` whose leading axis is time, in the top-first physics
      frame.

    Raises:
      ValueError: the Dataset's vertical orientation is inconsistent with its
        own pressures, so no faithful top-first state can be built.

    """
    from jcm.physics_interface import PhysicsState

    # Detect the file's vertical orientation from the ``level`` coordinate
    # values (attributes do not survive all xarray ops) and convert to the
    # top-first physics frame. A descending numeric ``level`` (sigma falling
    # from the surface to the top) is a surface-first file; anything else is
    # left as-is with top-first assumed.
    if cf_metadata.LEVEL_DIM in ds.coords:
        level = np.asarray(ds[cf_metadata.LEVEL_DIM].values)
        if np.issubdtype(level.dtype, np.number) and level.size > 1:
            if level[0] > level[-1]:
                ds = cf_metadata.orient_top_first(ds)
            # else: ascending sigma is already top-first; pass through.
        else:
            logging.warning(
                "load_states_from_xarray: %r coordinate is non-numeric or "
                "size<=1; cannot determine vertical orientation. Assuming the "
                "Dataset is already top-first (physics frame).",
                cf_metadata.LEVEL_DIM,
            )
    else:
        logging.warning(
            "load_states_from_xarray: no %r coordinate present; cannot "
            "determine vertical orientation. Assuming the Dataset is already "
            "top-first (physics frame).",
            cf_metadata.LEVEL_DIM,
        )

    # The orientation above is inferred; this is the independent check on it.
    _check_top_first_pressures(ds, surface_pressure_var)

    tracers = {
        tracer_name: jnp.asarray(ds[var_name].values)
        for tracer_name, var_name
        in _resolve_tracer_vars(ds, tracer_vars, required_tracers).items()
    }

    return PhysicsState(
        u_wind=jnp.asarray(ds[u_wind_var].values),
        v_wind=jnp.asarray(ds[v_wind_var].values),
        temperature=jnp.asarray(ds[temperature_var].values),
        specific_humidity=jnp.asarray(ds[specific_humidity_var].values),
        geopotential=jnp.asarray(ds[geopotential_var].values),
        normalized_surface_pressure=jnp.asarray(ds[surface_pressure_var].values),
        tracers=tracers,
    )


def create_single_column_state(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    u_wind: jnp.ndarray | None = None,
    v_wind: jnp.ndarray | None = None,
    surface_pressure: float = 101325.0,
    nlev: int | None = None,
):
    """Build a 1-D column ``PhysicsState`` for ``SingleColumnModel``.

    Geopotential is approximated hydrostatically from the column-mean
    temperature and ``surface_pressure``.

    Args:
      temperature: Temperature profile [K], shape ``(nlev,)``.
      specific_humidity: Specific humidity profile [kg/kg], shape ``(nlev,)``.
      u_wind: Optional zonal wind profile [m/s]; defaults to zeros.
      v_wind: Optional meridional wind profile [m/s]; defaults to zeros.
      surface_pressure: Surface pressure [Pa].
      nlev: Optional explicit level count (otherwise inferred).

    Returns:
      ``PhysicsState`` whose array fields are 1-D ``(nlev,)`` and
      ``normalized_surface_pressure`` is a scalar.

    """
    from jcm.physics_interface import PhysicsState
    from jcm.constants import grav, rd, p0

    if nlev is None:
        nlev = temperature.shape[0]

    temperature = jnp.asarray(temperature).reshape(nlev)
    specific_humidity = jnp.asarray(specific_humidity).reshape(nlev)
    u_wind = jnp.zeros(nlev) if u_wind is None else jnp.asarray(u_wind).reshape(nlev)
    v_wind = jnp.zeros(nlev) if v_wind is None else jnp.asarray(v_wind).reshape(nlev)

    p_levels = surface_pressure * jnp.linspace(0.1, 1.0, nlev)[::-1]
    scale_height = rd * jnp.mean(temperature) / grav
    z_approx = -scale_height * jnp.log(p_levels / surface_pressure)
    geopotential = (grav * z_approx).reshape(nlev)

    return PhysicsState(
        u_wind=u_wind,
        v_wind=v_wind,
        temperature=temperature,
        specific_humidity=specific_humidity,
        geopotential=geopotential,
        normalized_surface_pressure=jnp.asarray(surface_pressure / p0),
        tracers={},
    )


def create_initial_tracers(
    shape: tuple | int,
    tracer_names: list[str] | None = None,
    cloud_water: float = 0.0,
    cloud_ice: float = 0.0,
) -> dict:
    """Build a tracer dict for SCM/prescribed-state runs.

    ``qc`` and ``qi`` are populated with ``cloud_water`` and ``cloud_ice``;
    any other listed tracer names get zero arrays. ``shape`` is normally a
    tuple but ``int`` is accepted for the 1-D SCM case
    (``shape=nlev`` ⇒ ``(nlev,)``).
    """
    if isinstance(shape, int):
        shape = (shape,)
    if tracer_names is None:
        tracer_names = ['qc', 'qi']
    tracers = {}
    for name in tracer_names:
        if name == 'qc':
            tracers[name] = jnp.full(shape, cloud_water)
        elif name == 'qi':
            tracers[name] = jnp.full(shape, cloud_ice)
        else:
            tracers[name] = jnp.zeros(shape)
    return tracers