import jax.numpy as jnp
import tree_math
from jax import tree_util

@tree_math.struct
class DiffusionFilter:
    vor_q_timescale: jnp.float32 # Diffusion timescale (s)
    vor_q_order: jnp.int32 # Order of diffusion operator for tendencies
    temp_timescale: jnp.float32 # Diffusion timescale (s)
    temp_order: jnp.int32  # Order of diffusion operator for state variables
    div_timescale: jnp.float32 # Diffusion timescale (s)
    div_order: jnp.int32  # Order of diffusion operator for state variables

    @classmethod
    def default(cls):
        return cls(
            div_timescale = 2*60*60, # Diffusion timescale (s)
            div_order = 1, # Order of diffusion operator for tendencies
            vor_q_timescale = 12*60*60, # Diffusion timescale (s)
            vor_q_order = 2,  # Order of diffusion operator for state variables
            temp_timescale = 24*60*60, # Diffusion timescale (s)
            temp_order = 2,  # Order of diffusion operator for state variables
        )

    @classmethod
    def make_diffusion_fn(cls,grid,dt,timescale, order, replace_fn):
        '''
        Returns diffusion filter function handle for use in the model time step.
        '''
        from dinosaur.filtering import horizontal_diffusion_filter

        def diffusion_filter(u, u_next):
            eigenvalues = grid.laplacian_eigenvalues
            scale = dt / (timescale * abs(eigenvalues[-1]) ** order)

            filter_fn = horizontal_diffusion_filter(grid, scale, order)

            u_temp = filter_fn(u_next)
            return replace_fn(u_next, u_temp)
        return diffusion_filter
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

