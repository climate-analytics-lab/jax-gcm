
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData
import jax.numpy as jnp
import jax

# land_filename = 'land.nc'
# snow_filename = 'snow.nc'
# soil_filename = 'soil.nc'

def land_model_init(surface_filename, parameters: Parameters, boundaries):
    import xarray as xr
    from jcm.params import delt
    # =========================================================================
    # Initialize land-surface boundary conditions
    # =========================================================================

    # Fractional and binary land masks
    fmask_l = boundaries.fmask
    bmask_l = jnp.where(fmask_l >= parameters.land_model.thrsh, 1.0, 0.0)

    # Update fmask_l based on the conditions
    fmask_l = jnp.where(fmask_l >= parameters.land_model.thrsh, 
                        jnp.where(boundaries.fmask > (1.0 - parameters.land_model.thrsh), 1.0, fmask_l), 0.0)

    # Land-surface temperature
    stlcl_ob = jnp.asarray(xr.open_dataset(surface_filename)["stl"])
    # instead of using a check forchk, we check that 0.0 < stl12 < 400 and if it isn't we set it to 273
    # this might not work exactly because stl12 is ix,il,12 and bmask is ix,il
    stlcl_ob = jnp.where((bmask_l[:,:,jnp.newaxis] > 0.0) & ((stlcl_ob < 0.0) | (stlcl_ob > 400.0)), 273.0, stlcl_ob)

    # Snow depth
    snowd_am = jnp.asarray(xr.open_dataset(surface_filename)["snowd"])
    # simple sanity check - same method ras above for stl12 
    snowd_am = jnp.where((bmask_l[:,:,jnp.newaxis] > 0.0) & ((snowd_am < 0.0) | (snowd_am > 20000.0)), 0.0, snowd_am)
    # Read soil moisture and compute soil water availability using vegetation fraction
    # Read vegetation fraction
    veg_high = jnp.asarray(xr.open_dataset(surface_filename)["vegh"])
    veg_low  = jnp.asarray(xr.open_dataset(surface_filename)["vegl"])

    # Combine high and low vegetation fractions
    veg = jnp.maximum(0.0, veg_high + 0.8*veg_low)

    # Read soil moisture
    sdep1 = 70.0
    idep2 = 3
    # sdep2 = idep2*sdep1

    swwil2 = idep2*parameters.land_model.swwil
    rsw    = 1.0/(parameters.land_model.swcap + idep2*(parameters.land_model.swcap - parameters.land_model.swwil))

    # Combine soil water content from two top layers
    swl1 = jnp.asarray(xr.open_dataset(surface_filename)["swl1"])
    swl2 = jnp.asarray(xr.open_dataset(surface_filename)["swl2"])
    
    swroot = idep2 * swl2
    # Compute the intermediate max term
    max_term = jnp.maximum(0.0, swroot - swwil2)
    # Compute the soil water content

    soilw_am = jnp.minimum(1.0, rsw * (swl1 + veg[:,:,jnp.newaxis] * max_term))

    # simple sanity check - same method ras above for stl12 
    soilw_am = jnp.where((bmask_l[:,:,jnp.newaxis] > 0.0) & ((soilw_am < 0.0) | (soilw_am > 10.0)), 0.0, soilw_am)

    # =========================================================================
    # Set heat capacities and dissipation times for soil and ice-sheet layers
    # =========================================================================

    # 2. Compute constant fields
    # Set domain mask (blank out sea points)
    dmask = jnp.ones_like(fmask_l)
    dmask = jnp.where(fmask_l < parameters.land_model.flandmin, 0, dmask)

    # Set time_step/heat_capacity and dissipation fields
    rhcapl = jnp.where(boundaries.alb0 < 0.4, delt / parameters.land_model.hcapl, delt / parameters.land_model.hcapli)
    cdland = dmask*parameters.land_model.tdland/(1.0+dmask*parameters.land_model.tdland)

    boundaries_new = boundaries.copy(rhcapl=rhcapl, cdland=cdland, fmask_l=fmask_l, stlcl_ob=stlcl_ob, snowd_am=snowd_am, soilw_am=soilw_am)
    return boundaries_new

# Exchanges fluxes between land and atmosphere.
def couple_land_atm(state: PhysicsState, physics_data: PhysicsData, parameters: Parameters, boundaries: BoundaryData=jnp.newaxis):
    from jcm.date import days_year

    day = jnp.round(physics_data.date.tyear*days_year).astype(jnp.int32)
    stl_lm=None
    # Run the land model if the land model flags is switched on
    if (boundaries.land_coupling_flag):
        # stl_lm need to persist from time step to time step? what does this get from the model?
        stl_lm = run_land_model(physics_data.stlcl_lm, boundaries.stlcl_ob[:,:,day], boundaries.cdland, boundaries.rhcapl)
        stl_am = stl_lm
    # Otherwise get the land surface from climatology
    else:
        stl_am = boundaries.stlcl_ob

    # update land physics data
    land_model_data = physics_data.land_model.copy(stl_am=stl_am, stl_lm=stl_lm)
    physics_data = physics_data.copy(land_model=land_model_data)
    physics_tendency = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendency, physics_data

#Integrates slab land-surface model for one day.
def run_land_model(stl_lm, stlcl_ob, cdland, rhcapl):
    from jcm.auxiliaries import hfluxn

    # Land-surface (soil/ice-sheet) layer
    # Anomaly w.r.t. final-time climatological temperature
    tanom = stl_lm - stlcl_ob

    # Time evolution of temperature anomaly
    tanom = cdland*(tanom + rhcapl*hfluxn[:,:,0])

    # Full surface temperature at final time
    stl_lm = tanom + stlcl_ob

    return stl_lm

