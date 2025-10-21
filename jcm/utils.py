import jax.numpy as jnp
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

# Convert object to float 
def check_type_convert_to_float(x):
    return jnp.asarray(x, dtype=jnp.float32)

def convert_to_float(x): 
    return tree_map(check_type_convert_to_float, x)

# Revert object with type float back to true type
def check_type_convert_back(x, x0):
    return x if jnp.result_type(x0) == jnp.float32 else x0

def convert_back(x, x0):
    return tree_map(check_type_convert_back, x, x0)