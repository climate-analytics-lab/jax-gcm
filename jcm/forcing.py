from jcm.params import Parameters
from jcm.physics_data import ablco2_ref, PhysicsData
from jcm.boundaries import BoundaryData
from jcm.physics import PhysicsState
import jax.numpy as jnp
# linear trend of co2 absorptivity (del_co2: rate of change per year)
del_co2   = 0.005

def set_forcing(state: PhysicsState, physics_data: PhysicsData, parameters: Parameters, boundaries: BoundaryData=None):

    from jcm.date import days_year
    from jcm.shortwave_radiation import get_zonal_average_fields
    from jcm.physics import PhysicsTendency
    from jcm.physical_constants import rgas
    from jcm.land_model import sd2sc
    from jcm.mod_radcon import albsea, albice, albsn

    # 2. daily-mean radiative forcing
    # incoming solar radiation
    tyear = physics_data.date.tyear
    get_zonal_average_fields(state, physics_data)
    day = jnp.round(physics_data.date.tyear*days_year).astype(jnp.int32)

    # total surface albedo
    snowd_am = boundaries.snowd_am
    fmask_l = boundaries.fmask_l
    snowc = physics_data.mod_radcon.snowc
    sice_am = physics_data.ice_model.sice_am

    alb0 = boundaries.alb0

    snowc = jnp.minimum(1.0, snowd_am / sd2sc)
    alb_l = alb0 + snowc * (albsn - alb0)
    alb_s = albsea + sice_am * (albice - albsea)
    albsfc = alb_s + fmask_l * (alb_l - alb_s)

    increase_co2 = physics_data.shortwave_rad.increase_co2
    iyear_ref = physics_data.shortwave_rad.co2_year_ref
    model_year = physics_data.date.model_year
    if increase_co2:
        ablco2 = ablco2_ref * jnp.exp(del_co2 * (model_year + tyear - iyear_ref))
        mod_radcon = physics_data.mod_radcon.copy(ablco2=ablco2,albsfc=albsfc, alb_l=alb_l, alb_s=alb_s)
        physics_data = physics_data.copy(mod_radcon=mod_radcon)
    # update alb_l, alb_s, alsfc, etc
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendencies, physics_data
