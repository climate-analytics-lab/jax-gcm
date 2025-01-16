import jax.numpy as jnp
import tree_math

# we might want to separate this into sub-dataclasses (sea-model and land-model) for clarity
# need to pull sea-model out of PhysicsData
@tree_math.struct
class BoundaryData:
    fmask: jnp.ndarray # fractional land-sea mask (ix,il)
    phi0: jnp.ndarray  # surface geopotential (ix, il)
    phis0: jnp.ndarray # spectrally-filtered surface geopotential
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    fmask_l: jnp.ndarray # land mask - set by land_model_init()
    stl_am: jnp.ndarray # land surface temperature (ix,il)
    stl_lm: jnp.ndarray # land surface temperature (ix,il)
    rhcapl: jnp.ndarray # 1/heat capacity (land)
    cdland: jnp.ndarray # 1/dissipation time (land)
    stlcl_ob: jnp.ndarray
    snowcl_ob: jnp.ndarray # -- for all days - don't need _am, just grab from _ob
    soilwcl_ob: jnp.ndarray

    fmask_s: jnp.ndarray # sea mask - set bt sea_model_init()

    def copy(self,fmask=None,phi0=None,phis0=None,alb0=None,fmask_l=None,stl_am=None,stl_lm=None,rhcapl=None,cdland=None,stlcl_ob=None,snowcl_ob=None,soilwcl_ob=None,fmask_s=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else self.fmask,
            phi0=phi0 if phi0 is not None else self.phi0,
            phis0=phis0 if phis0 is not None else self.phis0,
            alb0=alb0 if alb0 is not None else self.alb0,
            fmask_l=fmask_l if fmask_l is not None else self.fmask_l,
            stl_am=stl_am if stl_am is not None else self.stl_am,
            stl_lm=stl_lm if stl_lm is not None else self.stl_lm,
            rhcapl=rhcapl if rhcapl is not None else self.rhcapl,
            cdland=cdland if cdland is not None else self.cdland,
            stlcl_ob=stlcl_ob if stlcl_ob is not None else self.stlcl_ob,
            snowcl_ob=snowcl_ob if snowcl_ob is not None else self.snowcl_ob,
            soilwcl_ob=soilwcl_ob if soilwcl_ob is not None else self.soilwcl_ob,
            fmask_s=fmask_s if fmask_s is not None else self.fmask_s
        )

# this function should probably call the land_model_init and sea_model_init functions
def initialize_boundaries(surface_filename, primitive, truncation_number):
    """
    Initialize the boundary conditions
    """
    from physical_constants import grav
    from jcm.physics import spectral_truncation
    import xarray as xr
    import numpy as np

    # Read surface geopotential (i.e. orography)
    phi0 = grav* jnp.asarray(xr.open_dataset(surface_filename)["orog"])

    # Also store spectrally truncated surface geopotential for the land drag term
    #TODO: See if we can get the truncation number from the primitive equation object
    phis0 = spectral_truncation(primitive, phi0, truncation_number)

    # Read land-sea mask
    fmask = jnp.asarray(xr.open_dataset(surface_filename)["lsm"])

    # Annual-mean surface albedo
    alb0 = jnp.asarray(xr.open_dataset(surface_filename)["alb"])

    # Apply some sanity checks
    assert jnp.all(fmask >= 0.0), "Land-sea mask must be between 0 and 1"
    assert jnp.all(fmask <= 1.0), "Land-sea mask must be between 0 and 1"

    return BoundaryData(fmask=fmask, phi0=phi0, phis0=phis0, alb0=alb0)
