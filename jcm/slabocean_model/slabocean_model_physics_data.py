import jax.numpy as jnp
import tree_math
from jax import tree_util
import jcm.slabocean_model.slabocean_model_tools as tools

@tree_math.struct
class SlaboceanModelData:
    
    sst_anom : jnp.ndarray
    sic_anom  : jnp.ndarray
   

    @classmethod
    def __consts(cls, nodal_shape, const, sst_anom = None, sic_anom = None):
        
        
        return SlaboceanModelData(
            sst_anom = sst_anom if sst_anom is not None else jnp.full((nodal_shape), const),
            sic_anom  = sic_anom  if sic_anom  is not None else jnp.full((nodal_shape), const),
        )
    
     
    @classmethod
    def zeros(cls, nodal_shape, **kwargs):
        return cls.__consts(nodal_shape, 0.0, **kwargs)
 
    @classmethod
    def ones(cls, nodal_shape, **kwargs):
        return cls.__consts(nodal_shape, 1.0, **kwargs)
    
    def copy(self, **kwargs):
        return SlaboceanModelData(
            sst_anom = tools.getDefault(kwargs, "sst_anom", self.sst_anom),
            sic_anom  = tools.getDefault(kwargs, "sic_anom", self.sic_anom),
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
