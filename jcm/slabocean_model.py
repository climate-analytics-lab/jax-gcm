from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import Geometry
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData
import jax.numpy as jnp
from jax import jit

# Questions: 
# 1. PhysicsTendency includes sst and sic?
# 2. Ocean model also includes sea ice?
# 3. Does slabocean model (a) compute both sst and sst tendency, or (b) sst tendency?


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
        fmask_l   = fmask_l,
        sst_clim  = sst_clim,
        sic_clim  = sic_clim,
        sst = sst,
        sic = sic,
        ocn_d0 = ocn_d0,
    )



# Exchanges fluxes between land and atmosphere.
def couple_sea_atm(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:

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

    dsstdt, dsicdt = run_slabocean_model(
        physics_data.slabocean_model.sst,
        physics_data.slabocean_model.sic,
        physics_data.surface_flux.hfluxn,
        physics_data.surface_flux.hfluxn,
        boundaries.cd_ocn,
        boundaries.cd_ice,
        boundaries.tau_ocn,
        boundaries.tau_ice,
        boundaries.sst_clim[:,:,day],
        boundaries.sic_clim[:,:,day],
    )


    # update land physics data
    #slabocean_model_data = physics_data.slabocean_model.copy(sst=sst, sic=sic)
    physics_data = physics_data.copy()
    physics_tendency = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendency, physics_data

#Integrates slab land-surface model for one day.
@jit
def run_slabocean_model(
    sst,
    sic,
    hfluxn_ocn,
    hfluxn_ice,
    cd_ocn,
    cd_ice,
    tau_ocn,
    tau_ice,
    sst_clim,
    sic_clim,
):

    dsstdt = hfluxn_ocn[:, :, 0] / cd_ocn - ( sst - sst_clim ) / tau_ocn
    dsicdt = hfluxn_ice[:, :, 0] / cd_ice - ( sic - sic_clim ) / tau_ice
 
    return dsstdt, dsicdt

