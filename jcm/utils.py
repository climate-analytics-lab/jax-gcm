import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import tree_map
from pathlib import Path
import dinosaur
from dinosaur.coordinate_systems import CoordinateSystem, HorizontalGridTypes
from dinosaur.primitive_equations import PrimitiveEquationsSpecs
from dinosaur.scales import SI_SCALE
from jcm.physics.speedy.physical_constants import SIGMA_LAYER_BOUNDARIES

DYNAMICS_UNITS_TABLE_CSV_PATH = Path(__file__).parent / 'dynamics_units_table.csv'

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

def get_coords(layers=8, spectral_truncation=31, nodal_shape=None, spmd_mesh=None) -> CoordinateSystem:
    f"""
    Returns a CoordinateSystem object for the given number of layers and one of the following horizontal resolutions: {VALID_TRUNCATIONS}.
    """
    from dinosaur.spherical_harmonic import FastSphericalHarmonics, RealSphericalHarmonics

    if nodal_shape is not None:
        if nodal_shape not in VALID_NODAL_SHAPES:
            raise ValueError(f"Invalid nodal shape: {nodal_shape}. Must be one of: {VALID_NODAL_SHAPES}.")
        spectral_truncation = TRUNCATION_FOR_NODAL_SHAPE[nodal_shape]
    elif spectral_truncation not in VALID_TRUNCATIONS:
        raise ValueError(f"Invalid horizontal resolution: {spectral_truncation}. Must be one of: {VALID_TRUNCATIONS}.")
    horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f'T{spectral_truncation}')

    if layers not in SIGMA_LAYER_BOUNDARIES:
        raise ValueError(f"Invalid number of layers: {layers}. Must be one of: {tuple(SIGMA_LAYER_BOUNDARIES.keys())}")

    physics_specs = PrimitiveEquationsSpecs.from_si(scale=SI_SCALE)

    if spmd_mesh is not None:
        spmd_mesh = jax.make_mesh(spmd_mesh, ('x', 'y', 'z'))
        spherical_harmonics_impl = FastSphericalHarmonics
    else:
        spherical_harmonics_impl = RealSphericalHarmonics

    return CoordinateSystem(
        horizontal=horizontal_grid(radius=physics_specs.radius, 
                                   spherical_harmonics_impl=spherical_harmonics_impl),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(SIGMA_LAYER_BOUNDARIES[layers]),
        spmd_mesh=spmd_mesh
    )

# Function to take a field in grid space and truncate it to a given wavenumber
def spectral_truncation(grid: HorizontalGridTypes, grid_field, truncation_number=None):
    """grid_field: field in grid space
    trunc: truncation level, # of wavenumbers to keep
    """
    spectral_field = grid.to_modal(grid_field)
    nx,mx = spectral_field.shape
    n_indices, m_indices = jnp.meshgrid(jnp.arange(nx), jnp.arange(mx), indexing='ij')
    total_wavenumber = m_indices + n_indices

    # truncate to grid truncation if no truncation number is given
    truncation_number = truncation_number or (grid.total_wavenumbers - 2)

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

def stack_trees(trees):
    return tree_map(lambda *arrays: jnp.stack(arrays, axis=0).astype(jnp.float32), *trees)

def _index_if_3d(arr, key):
    return arr[:, :, key] if arr.ndim > 2 else arr

def tree_index_3d(tree, key):
    return tree_map(lambda arr: _index_if_3d(jnp.array(arr), key), tree)

def _check_type_ones_like_tangent(x):
        if jnp.result_type(x) == jnp.float32:
            return jnp.ones_like(x)
        # in case of a bool or int, return a float0 denoting the lack of tangent space
        # jax requires that we use numpy to construct the float0 scalar
        # because it is a semantic placeholder not backed by any array data / memory allocation
        return np.ones((), dtype=jax.dtypes.float0)

def ones_like_tangent(pytree):
    return tree_map(_check_type_ones_like_tangent, pytree)

def _check_type_zeros_like_tangent(x):
        if jnp.result_type(x) == jnp.float32:
            return jnp.zeros_like(x)
        return np.zeros((), dtype=jax.dtypes.float0)

def zeros_like_tangent(pytree):
    return tree_map(_check_type_zeros_like_tangent, pytree)

def _check_type_convert_to_float(x):
    return jnp.asarray(x, dtype=jnp.float32)

def convert_to_float(x): 
    return tree_map(_check_type_convert_to_float, x)

# Revert object with type float back to true type
def _check_type_convert_back(x, x0):
    return x if jnp.result_type(x0) == jnp.float32 else x0

def convert_back(x, x0):
    return tree_map(_check_type_convert_back, x, x0)


# =============================================================================
# State creation utilities for prescribed state / single column models
# =============================================================================

def load_states_from_xarray(
    ds,
    u_wind_var: str = 'u_wind',
    v_wind_var: str = 'v_wind',
    temperature_var: str = 'temperature',
    specific_humidity_var: str = 'specific_humidity',
    geopotential_var: str = 'geopotential',
    surface_pressure_var: str = 'normalized_surface_pressure',
    tracer_vars: dict[str, str] = None,
):
    """Load PhysicsState time series from an xarray Dataset.

    Args:
        ds: xarray Dataset containing atmospheric state variables.
        u_wind_var: Variable name for zonal wind.
        v_wind_var: Variable name for meridional wind.
        temperature_var: Variable name for temperature.
        specific_humidity_var: Variable name for specific humidity.
        geopotential_var: Variable name for geopotential.
        surface_pressure_var: Variable name for normalized surface pressure.
        tracer_vars: Dict mapping tracer names to variable names in ds.

    Returns:
        PhysicsState with time dimension as the first axis.
    """
    from jcm.physics_interface import PhysicsState

    # Extract variables and convert to JAX arrays
    u_wind = jnp.asarray(ds[u_wind_var].values)
    v_wind = jnp.asarray(ds[v_wind_var].values)
    temperature = jnp.asarray(ds[temperature_var].values)
    specific_humidity = jnp.asarray(ds[specific_humidity_var].values)
    geopotential = jnp.asarray(ds[geopotential_var].values)
    normalized_surface_pressure = jnp.asarray(ds[surface_pressure_var].values)

    # Load tracers
    tracers = {}
    if tracer_vars:
        for tracer_name, var_name in tracer_vars.items():
            tracers[tracer_name] = jnp.asarray(ds[var_name].values)

    return PhysicsState(
        u_wind=u_wind,
        v_wind=v_wind,
        temperature=temperature,
        specific_humidity=specific_humidity,
        geopotential=geopotential,
        normalized_surface_pressure=normalized_surface_pressure,
        tracers=tracers,
    )


def create_single_column_state(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    u_wind: jnp.ndarray = None,
    v_wind: jnp.ndarray = None,
    surface_pressure: float = 101325.0,
    nlev: int = None,
):
    """Create a single-column PhysicsState for SCM simulations.

    Args:
        temperature: Temperature profile [K] (nlev,).
        specific_humidity: Specific humidity profile [kg/kg] (nlev,).
        u_wind: Zonal wind profile [m/s] (nlev,). Default zeros.
        v_wind: Meridional wind profile [m/s] (nlev,). Default zeros.
        surface_pressure: Surface pressure [Pa]. Default 101325.
        nlev: Number of levels. Inferred from temperature if not provided.

    Returns:
        PhysicsState with shape (nlev, 1, 1) suitable for single-column runs.
    """
    from jcm.physics_interface import PhysicsState
    from jcm.constants import grav, rd, p0

    if nlev is None:
        nlev = temperature.shape[0]

    # Reshape to (nlev, 1, 1) for single column
    temperature = jnp.asarray(temperature).reshape(nlev, 1, 1)
    specific_humidity = jnp.asarray(specific_humidity).reshape(nlev, 1, 1)

    if u_wind is None:
        u_wind = jnp.zeros((nlev, 1, 1))
    else:
        u_wind = jnp.asarray(u_wind).reshape(nlev, 1, 1)

    if v_wind is None:
        v_wind = jnp.zeros((nlev, 1, 1))
    else:
        v_wind = jnp.asarray(v_wind).reshape(nlev, 1, 1)

    # Compute approximate geopotential (hydrostatic)
    # Simple approximation: phi = g * z, where z is computed from hypsometric equation
    p_levels = surface_pressure * jnp.linspace(0.1, 1.0, nlev)[::-1]
    scale_height = rd * jnp.mean(temperature) / grav
    z_approx = -scale_height * jnp.log(p_levels / surface_pressure)
    geopotential = (grav * z_approx).reshape(nlev, 1, 1)

    # Normalized surface pressure
    normalized_surface_pressure = jnp.array([[surface_pressure / p0]])

    return PhysicsState(
        u_wind=u_wind,
        v_wind=v_wind,
        temperature=temperature,
        specific_humidity=specific_humidity,
        geopotential=geopotential,
        normalized_surface_pressure=normalized_surface_pressure,
        tracers={},
    )


def create_initial_tracers(
    shape: tuple,
    tracer_names: list[str] = None,
    cloud_water: float = 0.0,
    cloud_ice: float = 0.0,
) -> dict:
    """Create initial tracer fields for SCM simulation.

    Args:
        shape: Shape of tracer arrays (nlev, nlon, nlat).
        tracer_names: List of tracer names to initialize.
        cloud_water: Initial cloud water mixing ratio (kg/kg).
        cloud_ice: Initial cloud ice mixing ratio (kg/kg).

    Returns:
        Dict of initialized tracer arrays.
    """
    if tracer_names is None:
        tracer_names = ['qc', 'qi']  # Default cloud tracers

    tracers = {}
    for name in tracer_names:
        if name == 'qc':
            tracers[name] = jnp.full(shape, cloud_water)
        elif name == 'qi':
            tracers[name] = jnp.full(shape, cloud_ice)
        else:
            tracers[name] = jnp.zeros(shape)

    return tracers
