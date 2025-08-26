import jax.numpy as jnp
import tree_math
from jax import tree_util

@tree_math.struct
class DiffusionFilter:
    tendency_diff_timescale: jnp.ndarray # Diffusion timescale (s)
    tendency_diff_order: jnp.ndarray # Order of diffusion operator for tendencies
    state_diff_timescale: jnp.ndarray # Diffusion timescale (s)
    state_diff_order: jnp.ndarray  # Order of diffusion operator for state variables

    @classmethod
    def default(cls):
        return cls(
            tendency_diff_timescale = jnp.array(2.4*60*60), # Diffusion timescale (s)
            tendency_diff_order = jnp.array(4), # Order of diffusion operator for tendencies
            state_diff_timescale = jnp.array(2.4*60*60), # Diffusion timescale (s)
            state_diff_order = jnp.array(1),  # Order of diffusion operator for state variables
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)