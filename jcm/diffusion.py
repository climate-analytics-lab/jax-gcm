import jax.numpy as jnp
import tree_math
from jax import tree_util

@tree_math.struct
class DiffusionFilter:
    vor_q_timescale: jnp.float_ # Diffusion timescale (s)
    vor_q_order: jnp.int_ # Order of diffusion operator for tendencies
    temp_timescale: jnp.float_ # Diffusion timescale (s)
    temp_order: jnp.int_  # Order of diffusion operator for state variables
    div_timescale: jnp.float_ # Diffusion timescale (s)
    div_order: jnp.int_  # Order of diffusion operator for state variables

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
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

