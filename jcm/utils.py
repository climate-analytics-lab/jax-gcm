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
    import jax.tree_util as jtu
    return jtu.tree_map(lambda y: jnp.ones_like(y), x)

# Convert object to float 
def check_type_convert_to_float(x): 
    print(type(x))
    try:
        return x.astype(jnp.float32)
    except AttributeError:
        return jnp.float32(x)
def convert_to_float(x): 
    return tree_map(check_type_convert_to_float, x)

# Revery object with type float back to true type
def check_type_convert_back(x, x0):
    try: 
        if x0.dtype == jnp.float32:
            return x
        else:
            return x0
    except AttributeError:
        if type(x0) == jnp.float32:
            return x
        else:
            return x0
def convert_back(x, x0):
    return tree_map(check_type_convert_back, x, x0)