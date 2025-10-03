import jax.numpy as jnp
import tree_math
from jax import tree_util

@tree_math.struct
class DiffusionFilter:
    tendency_diff_timescale: float # Diffusion timescale (s)
    tendency_diff_order: int # Order of diffusion operator for tendencies
    strat_tendency_diff_timescale: float # Diffusion timescale (s)
    strat_tendency_diff_order: int # Order of diffusion operator for tendencies
    state_diff_timescale: float # Diffusion timescale (s)
    state_diff_order: int  # Order of diffusion operator for state variables
    stratosphere_level: int # Number of upper levels for stratospheric diffusion

    @classmethod
    def default(cls):
        return cls(
            tendency_diff_timescale = 2.4*60*60, # Diffusion timescale (s)
            tendency_diff_order = 4, # Order of diffusion operator for tendencies
            strat_tendency_diff_timescale = 2.4*60*60, # Diffusion timescale (s)
            strat_tendency_diff_order = 2, # Order of diffusion operator for tendencies
            state_diff_timescale = 2.4*60*60, # Diffusion timescale (s)
            state_diff_order = 1,  # Order of diffusion operator for state variables
            stratosphere_level = 2, # number of upper levels for stratospheric diffusion
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)