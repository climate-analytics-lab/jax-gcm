import tree_math
import jax.numpy as jnp
from jax import tree_util


@tree_math.struct
class SlaboceanModelParameters:

    tau       : jnp.ndarray # relaxation time of SST (unit: day)
    d_omin    : jnp.ndarray # mixed layer thickness of low  latitude (unit: meter)
    d_omax    : jnp.ndarray # mixed layer thickness of high latitude (unit: meter)
    thrsh     : float
    
    @classmethod
    def default(self):
        return SlaboceanModelParameters(
            tau = jnp.array(60.0),
            d_omin = jnp.array(40.0),
            d_omax = jnp.array(120.0),
            thrsh   = 0.5,
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)


