import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import tree_map
import dinosaur
from dinosaur.coordinate_systems import CoordinateSystem, HorizontalGridTypes
from dinosaur.primitive_equations import PrimitiveEquationsSpecs
from dinosaur.scales import SI_SCALE
from jcm.physics.speedy.physical_constants import SIGMA_LAYER_BOUNDARIES

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

def get_coords(layers=8, spectral_truncation=31, spmd_mesh=None) -> CoordinateSystem:
    f"""
    Returns a CoordinateSystem object for the given number of layers and one of the following horizontal resolutions: {VALID_TRUNCATIONS}.
    """
    if spectral_truncation not in VALID_TRUNCATIONS:
        raise ValueError(f"Invalid horizontal resolution: {spectral_truncation}. Must be one of: {VALID_TRUNCATIONS}.")
    horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f'T{spectral_truncation}')
    if layers not in SIGMA_LAYER_BOUNDARIES:
        raise ValueError(f"Invalid number of layers: {layers}. Must be one of: {tuple(SIGMA_LAYER_BOUNDARIES.keys())}")

    physics_specs = PrimitiveEquationsSpecs.from_si(scale=SI_SCALE)

    if spmd_mesh is not None:
        spmd_mesh = jax.make_mesh(spmd_mesh, ('x', 'y', 'z'))

    return CoordinateSystem(
        horizontal=horizontal_grid(radius=physics_specs.radius),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(SIGMA_LAYER_BOUNDARIES[layers]),
        spmd_mesh=spmd_mesh
    )

# Function to take a field in grid space and truncate it to a given wavenumber
def spectral_truncation(grid: HorizontalGridTypes, grid_field, truncation_number=None):
    """
        grid_field: field in grid space
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
    """
    Validate that an xarray Dataset has the expected variables and dimensions.

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
    return tree_map(lambda arr: _index_if_3d(arr, key), tree)

def _check_type_ones_like_tangent(x):
        if jnp.result_type(x) == jnp.float32:
            return jnp.ones_like(x)
        # in case of a bool or int, return a float0 denoting the lack of tangent space
        # jax requires that we use numpy to construct the float0 scalar
        # because it is a semantic placeholder not backed by any array data / memory allocation
        return np.ones((), dtype=jax.dtypes.float0)

def ones_like_tangent(pytree):
    return tree_map(_check_type_ones_like_tangent, pytree)

def _check_type_convert_to_float(x):
    return jnp.asarray(x, dtype=jnp.float32)

def convert_to_float(x): 
    return tree_map(_check_type_convert_to_float, x)

# Revert object with type float back to true type
def _check_type_convert_back(x, x0):
    return x if jnp.result_type(x0) == jnp.float32 else x0

def convert_back(x, x0):
    return tree_map(_check_type_convert_back, x, x0)