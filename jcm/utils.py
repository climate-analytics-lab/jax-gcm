import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import tree_map
from dinosaur.coordinate_systems import HorizontalGridTypes

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