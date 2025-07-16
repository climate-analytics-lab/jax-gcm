from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import Geometry
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData
import jax.numpy as jnp
from jax import jit

def slabocean_model_init(
    surface_filename,
    parameters: Parameters,
    boundaries: BoundaryData,
):

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
    bmask_l = jnp.where(fmask_l >= parameters.slabocean_model.thrsh, 0.0, 1.0)
    
    # Update fmask_l based on the conditions
    fmask_l = jnp.where(
        fmask_l >= parameters.slabocean_model.thrsh,
        1.0,
        0.0,
    )

    # Sea surface temperature
    sst_clim = jnp.asarray(xr.open_dataset(surface_filename)["sst"])
    
    # =========================================================================
    # Set heat capacities and dissipation times for soil and ice-sheet layers
    # =========================================================================
    
    # 2. Compute constant fields
    # Set domain mask (blank out sea points)
    dmask = jnp.ones_like(fmask_l)
    dmask = jnp.where(fmask_l < parameters.slabocean_model.flandmin, 0, dmask)
    
    # Set time_step/heat_capacity and dissipation fields
    cdland = dmask*parameters.slabocean_model.tdland/(1.0+dmask*parameters.slabocean_model.tdland)

    return boundaries.copy(
        sst_clim = sst_clim,
        fmask_l   = fmask_l,
        sst_clim  = sst_clim,
        sic_clim  = sic_clim,
        sst = sst,
        sic = sic,
        ocn_d0 = ocn_d0,
    )

# Exchanges fluxes between land and atmosphere.
def couple_land_atm(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:

    day = physics_data.date.model_day()
    stl_lm=None
    # Run the land model if the land model flags is switched on
    if (boundaries.land_coupling_flag):
        # stl_lm need to persist from time step to time step? what does this get from the model?
        stl_lm = run_slabocean_model(physics_data.surface_flux.hfluxn, physics_data.stlcl_lm, boundaries.stlcl_ob[:,:,day], boundaries.cdland, boundaries.rhcapl)
        stl_am = stl_lm
    # Otherwise get the land surface from climatology
    else:
        stl_am = boundaries.stlcl_ob[:,:,day]

    # update land physics data
    slabocean_model_data = physics_data.slabocean_model.copy(stl_am=stl_am, stl_lm=stl_lm)
    physics_data = physics_data.copy(slabocean_model=slabocean_model_data)
    physics_tendency = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendency, physics_data

#Integrates slab land-surface model for one day.
@jit
def run_slabocean_model(hfluxn, stl_lm, stlcl_ob, cdland, rhcapl):
    # Land-surface (soil/ice-sheet) layer
    # Anomaly w.r.t. final-time climatological temperature
    tanom = stl_lm - stlcl_ob

    # Time evolution of temperature anomaly
    tanom = cdland*(tanom + rhcapl*hfluxn[:,:,0])

    # Full surface temperature at final time
    stl_lm = tanom + stlcl_ob

    return stl_lm

