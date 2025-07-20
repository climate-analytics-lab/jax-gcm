from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import Geometry
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData
from jcm import physical_constants

import jax.numpy as jnp
from jax import jit

from dinosaur.scales import units


# Questions: 
# 1. PhysicsTendency includes sst and sic?
# 2. Ocean model also includes sea ice?
# 3. Does slabocean model (a) compute both sst and sst tendency, or (b) sst tendency?
#
# 4. Where to obtain length of time step
# 5. Mask deterimination for land and sea
# 6. Why it seems couple_sea_atm only run once?
#

def slabocean_model_init(
    surface_filename,
    parameters: Parameters,
    boundaries: BoundaryData,
    grid,
    time_step,
):

    print("Called: slabocean_model_init")

    """
        surface_filename: filename storing boundary data
        parameters: initialized model parameters
        boundaries: partially initialized boundary data
        time_step: time step - model timestep in minutes
    """
    import xarray as xr
    # =========================================================================
    # Initialize slab ocean model boundary conditions
    # =========================================================================

    # Fractional and binary land masks
    fmask_l = boundaries.fmask
    bmask_l = jnp.where(fmask_l > 0, 0.0, 1.0) # ocean grid needs to be fully ocean
    
    # Update fmask_l based on the conditions
    fmask_l = jnp.where(
        fmask_l >= parameters.slabocean_model.thrsh,
        1.0,
        0.0,
    )

    # Sea surface temperature
    sst_clim = jnp.asarray(xr.open_dataset(surface_filename)["sst"])
    
    # =========================================================================
    # Set heat capacities and dissipation times for slab ocean model
    # =========================================================================
    
    # 2. Compute constant fields
    # Set domain mask (blank out sea points)
    
    # Set time_step/heat_capacity and dissipation fields

    time_step_s = time_step.to(units.s).magnitude
    print("time_step = ", time_step)
    print("time_step in seconds = ", time_step_s)
    
    lat_rad = grid.latitudes
    d_min = parameters.slabocean_model.d_min
    d_max = parameters.slabocean_model.d_max
    d_ocn = d_max + (d_min - d_max) * jnp.cos(lat_rad)**3.0

    sst_init = sst_clim[:, :, 0].copy()

    cd = physical_constants.cp_sw * d_ocn
    tau = jnp.ones_like(cd) * parameters.slabocean_model.tau_ocn

    ocn_time_factor = 1.0 + time_step_s / tau
    ocn_cd_factor = time_step_s / cd
    
    #parameters.slabocean_model.c_0land/(1.0+dmask*parameters.slabocean_model.tdland)

    return boundaries.copy(
        fmask_l   = fmask_l,
        sst_clim  = sst_clim,
        sst = sst_init,
        ocn_time_factor = ocn_time_factor,
        ocn_cd_factor = ocn_cd_factor,
    )



# Exchanges fluxes between land and atmosphere.
def couple_sea_atm(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:
    
    print("Called: couple_sea_atm")

    day = physics_data.date.model_day()
    """
    # Run the slabocean model if the slabocean model flags is switched on
    if boundaries.sea_coupling_flag == True:

        sst, sic = run_slabocean_model(
            physics_data.surface_flux.hfluxn,
            physics_data.stlcl_lm,
            boundaries.stlcl_ob[:,:,day],
            boundaries.cdland, 
            boundaries.rhcapl,
        )

    # Otherwise get the land surface from climatology
    else:
        sst = boundaries.sst_clim[:,:,day]
        sic = boundaries.sic_clim[:,:,day]

    """

    sst_new = run_slabocean_model(
        physics_data.slabocean_model.sst,
        physics_data.surface_flux.hfluxn[:, :, 0],
        boundaries.ocn_time_factor,
        boundaries.ocn_cd_factor,
        boundaries.sst_clim[:,:,day],
        boundaries.sst_clim[:,:,day+1],
        physics_data.surface_flux.hfluxn[:, :, 0] * 0,
    )

    # update land physics data
    slabocean_model_data = physics_data.slabocean_model.copy(sst=sst_new)
    physics_data = physics_data.copy(slabocean_model=slabocean_model_data)
    physics_tendency = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendency, physics_data

#Integrates slab land-surface model for one day.
@jit
def run_slabocean_model(
    sst,
    hfluxn,
    time_factor,
    cd_factor,
    sst_clim_1,
    sst_clim_2,
    hfluxn_clim,
):

    sst_anom = sst - sst_clim_1
    hfluxn_anom = hfluxn - hfluxn_clim
    
    new_sst_anom = time_factor * ( sst_anom + hfluxn_anom * cd_factor )

    new_sst = new_sst_anom + sst_clim_2

 
    return new_sst

