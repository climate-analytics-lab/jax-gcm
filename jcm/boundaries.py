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
    snowd_am: jnp.ndarray # snow depth (water equivalent) (ix,il)
    soilw_am: jnp.ndarray # soil water availability (ix,il)

    fmask_s: jnp.ndarray # sea mask - set bt sea_model_init()

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
