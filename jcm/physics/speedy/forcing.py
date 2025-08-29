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

    iyear_ref = parameters.forcing.co2_year_ref
    ablco2 = ablco2_ref * jnp.exp(parameters.forcing.increase_co2 * del_co2 * (model_year + tyear - iyear_ref))

    mod_radcon = physics_data.mod_radcon.copy(snowc=snowc, alb_l=alb_l, alb_s=alb_s, albsfc=albsfc, ablco2=ablco2)

    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendencies, physics_data
