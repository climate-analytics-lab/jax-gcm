import jax.numpy as jnp
import tree_math
from jax import tree_util
import jcm.slabocean_model.slabocean_model_tools as tools

@tree_math.struct
class SlaboceanModelData:
    
    sst : jnp.ndarray
    sic  : jnp.ndarray
    
    @classmethod
    def __consts(cls, nodal_shape, const, sst = None, sic = None):
        
        
        return SlaboceanModelData(
            sst = sst if sst is not None else jnp.full((nodal_shape), const),
            sic  = sic  if sic  is not None else jnp.full((nodal_shape), const),
        )
    
    @classmethod
    def zeros(cls, nodal_shape, **kwargs):
        return cls.__consts(nodal_shape, 0.0, **kwargs)
    
    @classmethod
    def ones(cls, nodal_shape, **kwargs):
        return cls.__consts(nodal_shape, 1.0, **kwargs)
    
    def copy(self, **kwargs):
        return SlaboceanModelData(
            sst = tools.getDefault(kwargs, "sst", self.sst),
            sic  = tools.getDefault(kwargs, "sic", self.sic),
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
