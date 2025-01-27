import jax.numpy as jnp
import tree_math
from jax import tree_util

@tree_math.struct
class BoundaryData:
    fmask: jnp.ndarray # fractional land-sea mask (ix,il)
    forog: jnp.ndarray
    phi0: jnp.ndarray  # surface geopotential (ix, il)
    phis0: jnp.ndarray # spectrally-filtered surface geopotential
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    fmask: jnp.ndarray
    fmask_l: jnp.ndarray # land mask - set by land_model_init()
    rhcapl: jnp.ndarray # 1/heat capacity (land)
    cdland: jnp.ndarray # 1/dissipation time (land)
    stlcl_ob: jnp.ndarray # climatology for land temperature - might not need this and stl_lm
    snowd_am: jnp.ndarray # used to be snowcl_ob in fortran - but one day of that was snowd_am
    soilw_am: jnp.ndarray # used to be soilwcl_ob in fortran - but one day of that was soilw_am
    lfluxland: jnp.bool
    land_coupling_flag: jnp.bool # 0 or 1

    fmask_s: jnp.ndarray # sea mask - set bt sea_model_init() once we have a model (instead of fixed ssts)

    @classmethod
    def zeros(self,nodal_shape,fmask=None,forog=None,phi0=None,phis0=None,alb0=None,fmask_l=None,rhcapl=None,cdland=None,stlcl_ob=None,snowd_am=None,soilw_am=None,fmask_s=None,lfluxland=None, land_coupling_flag=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else jnp.zeros((nodal_shape)),
            forog=forog if forog is not None else jnp.zeros((nodal_shape)),
            phi0=phi0 if phi0 is not None else jnp.zeros((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.zeros((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            fmask_l=fmask_l if fmask_l is not None else jnp.zeros((nodal_shape)),
            rhcapl=rhcapl if rhcapl is not None else jnp.zeros((nodal_shape)),
            cdland=cdland if cdland is not None else jnp.zeros((nodal_shape)),
            stlcl_ob=stlcl_ob if stlcl_ob is not None else jnp.zeros((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.zeros((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)+(365,)),
            land_coupling_flag=land_coupling_flag if land_coupling_flag is not None else False,
            lfluxland=lfluxland if lfluxland is not None else True,
            fmask_s=fmask_s if fmask_s is not None else jnp.zeros((nodal_shape)),
        )
    
    @classmethod
    def ones(self,nodal_shape,fmask=None,forog=None,phi0=None,phis0=None,alb0=None,fmask_l=None,rhcapl=None,cdland=None,stlcl_ob=None,snowd_am=None,soilw_am=None,fmask_s=None,lfluxland=None, land_coupling_flag=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else jnp.ones((nodal_shape)),
            forog=forog if forog is not None else jnp.ones((nodal_shape)),
            phi0=phi0 if phi0 is not None else jnp.ones((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.ones((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            fmask_l=fmask_l if fmask_l is not None else jnp.ones((nodal_shape)),
            rhcapl=rhcapl if rhcapl is not None else jnp.ones((nodal_shape)),
            cdland=cdland if cdland is not None else jnp.ones((nodal_shape)),
            stlcl_ob=stlcl_ob if stlcl_ob is not None else jnp.ones((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.ones((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)+(365,)),
            land_coupling_flag=land_coupling_flag if land_coupling_flag is not None else False,
            lfluxland=lfluxland if lfluxland is not None else True,
            fmask_s=fmask_s if fmask_s is not None else jnp.ones((nodal_shape)),
        )

    def copy(self,fmask=None,phi0=None,forog=None,phis0=None,alb0=None,fmask_l=None,rhcapl=None,cdland=None,stlcl_ob=None,snowd_am=None,soilw_am=None,fmask_s=None,lfluxland=None, land_coupling_flag=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else self.fmask,
            forog=forog if forog is not None else self.forog,
            phi0=phi0 if phi0 is not None else self.phi0,
            phis0=phis0 if phis0 is not None else self.phis0,
            alb0=alb0 if alb0 is not None else self.alb0,
            fmask_l=fmask_l if fmask_l is not None else self.fmask_l,
            rhcapl=rhcapl if rhcapl is not None else self.rhcapl,
            cdland=cdland if cdland is not None else self.cdland,
            stlcl_ob=stlcl_ob if stlcl_ob is not None else self.stlcl_ob,
            snowcl_ob=snowd_am if snowd_am is not None else self.snowd_am,
            soilwcl_ob=soilw_am if soilw_am is not None else self.soilw_am,
            land_coupling_flag=land_coupling_flag if land_coupling_flag is not None else self.land_coupling_flag,
            lfluxland=lfluxland if lfluxland is not None else self.lfluxland,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            fmask_s=fmask_s if fmask_s is not None else self.fmask_s
        )
    
    def has_nans(self):
        def check_nans(x):
            if isinstance(x, jnp.ndarray):
                return jnp.any(jnp.isnan(x))
            return False
        return tree_util.tree_reduce(lambda x, y: x or y,
                                   tree_util.tree_map(check_nans, self),
                                   False)
    


#this function calls land_model_init and eventually will call init for sea and ice models
def initialize_boundaries(surface_filename, primitive, truncation_number):
    """
    Initialize the boundary conditions
    """
    from jcm.physical_constants import grav
    from jcm.physics import spectral_truncation
    from jcm.land_model import land_model_init
    from jcm.surface_flux import set_orog_land_sfc_drag
    import xarray as xr
    import numpy as np

    # Read surface geopotential (i.e. orography)
    phi0 = grav* jnp.asarray(xr.open_dataset(surface_filename)["orog"])

    # Also store spectrally truncated surface geopotential for the land drag term
    #TODO: See if we can get the truncation number from the primitive equation object
    phis0 = spectral_truncation(primitive, phi0, truncation_number)
    forog = set_orog_land_sfc_drag(phis0)

    # Read land-sea mask
    fmask = jnp.asarray(xr.open_dataset(surface_filename)["lsm"])

    # Annual-mean surface albedo
    alb0 = jnp.asarray(xr.open_dataset(surface_filename)["alb"])

    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all(fmask >= 0.0), "Land-sea mask must be between 0 and 1"
    assert jnp.all(fmask <= 1.0), "Land-sea mask must be between 0 and 1"

    nodal_shape = fmask.shape
    boundaries = BoundaryData.zeros(nodal_shape,fmask=fmask,forog=forog,phi0=phi0, phis0=phis0, alb0=alb0)
    boundaries = land_model_init(surface_filename,boundaries)
    # call sea model init 
    # call ice model init

    return boundaries