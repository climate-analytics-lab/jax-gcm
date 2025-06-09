import tree_math
import jax.numpy as jnp
from jax import tree_util


@tree_math.struct
class SlaboceanModelParameters:

    tau       : jnp.ndarray # relaxation time of SST (unit: day)
    h_lolat   : jnp.ndarray # mixed layer thickness of low  latitude (unit: meter)
    h_hilat   : jnp.ndarray # mixed layer thickness of high latitude (unit: meter)

    @classmethod
    def default(self):
        return SlaboceanModelParameters(
            tau = jnp.array(60.0),
            h_lolat = jnp.array(40.0),
            h_hilat = jnp.array(120.0),
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)


