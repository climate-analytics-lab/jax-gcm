import jax.numpy as jnp
from dinosaur.primitive_equations import PrimitiveEquations 

# Function to take a field in grid space and truncate it to a given wavenumber
def spectral_truncation(dynamics: PrimitiveEquations, grid_field, trunc):
    '''
        grid_field: field in grid space
        trunc: truncation level, # of wavenumbers to keep
    '''
    spectral_field = dynamics.coords.horizontal.to_modal(grid_field)
    nx,mx = spectral_field.shape
    n_indices, m_indices = jnp.meshgrid(jnp.arange(nx), jnp.arange(mx), indexing='ij')
    total_wavenumber = m_indices + n_indices

    spectral_field = jnp.where(total_wavenumber > trunc, 0.0, spectral_field)

    truncated_grid_field = dynamics.coords.horizontal.to_nodal(spectral_field)

    return truncated_grid_field