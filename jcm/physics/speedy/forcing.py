from jcm.geometry import Geometry
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import ablco2_ref, PhysicsData
from jcm.boundaries import BoundaryData
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.speedy.shortwave_radiation import get_zonal_average_fields
import jax.numpy as jnp
# linear trend of co2 absorptivity (del_co2: rate of change per year)
del_co2   = 0.005

def set_forcing(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:
    """Calculates and sets time-dependent radiative forcings for the model.

    This function updates the physical state of the model by calculating
    key radiative forcing components that vary with time. It performs two main tasks:

    1.  **Surface Albedo Calculation**: It computes the total surface albedo by
        combining contributions from bare land, snow cover, and sea ice. The
        snow and sea ice amounts are determined from the daily climatology
        stored in the `BoundaryData`.
    2.  **CO2 Concentration Update**: If enabled via the `increase_co2` parameter,
        it updates the CO2 absorptivity based on an exponential growth trend
        from a reference year.


    Args:
        state: The current physical state of the atmosphere (e.g., temperature).
        physics_data: A container for various physical parameters and data fields
            that are passed between physics routines.
        parameters: A container for the model's configurable parameters.
        boundaries: A container for static boundary conditions like orography,
            land-sea masks, and climatological data.
        geometry: A container for grid and geometric information.

    Returns:
        A tuple containing:
        - `physics_tendencies`: A zero tendency object, returned for API
          consistency with other physics parameterizations. This function
          only updates `PhysicsData`, it does not compute tendencies directly.
        - `physics_data`: The updated `PhysicsData` object containing the newly
          calculated forcing values.
    """
    # 2. daily-mean radiative forcing
    physics_data = get_zonal_average_fields(state, physics_data, boundaries=boundaries, geometry=geometry)
    tyear = physics_data.date.tyear
    day = physics_data.date.model_day()
    model_year = physics_data.date.model_year

    # total surface albedo
    snowd_am = boundaries.snowd_am[:,:,day]
    fmask_l = boundaries.fmask_l
    sice_am = boundaries.sice_am[:,:,day]

    alb0 = boundaries.alb0

    snowc = jnp.minimum(1.0, snowd_am / parameters.land_model.sd2sc)
    alb_l = alb0 + snowc * (parameters.mod_radcon.albsn - alb0)
    alb_s = parameters.mod_radcon.albsea + sice_am * (parameters.mod_radcon.albice - parameters.mod_radcon.albsea)
    albsfc = alb_s + fmask_l * (alb_l - alb_s)

    increase_co2 = parameters.forcing.increase_co2
    iyear_ref = parameters.forcing.co2_year_ref

    mod_radcon = physics_data.mod_radcon.copy(snowc=snowc, alb_l=alb_l, alb_s=alb_s, albsfc=albsfc)
    if increase_co2:
        ablco2 = ablco2_ref * jnp.exp(del_co2 * (model_year + tyear - iyear_ref))
        mod_radcon = mod_radcon.copy(ablco2=ablco2)

    physics_data = physics_data.copy(mod_radcon=mod_radcon)
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendencies, physics_data
