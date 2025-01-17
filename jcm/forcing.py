import jax.numpy as jnp
from jcm.shortwave_radiation import ablco2
# linear trend of co2 absorptivity (del_co2: rate of change per year)
del_co2   = 0.005
ablco2_ref = ablco2

# need to call once on initialization, (or just need to know if its the first call)
def set_forcing(state, physics_data, boundaries):
    '''
    imode = Mode -> 0 = initialization step, 1 = daily update
    '''
    from jcm.date import days_year
    from jcm.shortwave_radiation import get_zonal_average_fields
    from jcm.physics import PhysicsTendency
    from jcm.physical_constants import rgas
    from jcm.land_model import sd2sc
    from jcm.mod_radcon import albsea, albice, albsn
    from jcm.shortwave_radiation import ablco2

    # time variables for interpolation are set by newdate

    # 2. daily-mean radiative forcing
    # incoming solar radiation
    tyear = physics_data.date.tyear
    get_zonal_average_fields(tyear)
    day = jnp.round(physics_data.date.tyear*days_year).astype(jnp.int32)

    # total surface albedo
    snowd_am = boundaries.snowd_am[:,:,day]
    fmask_l = boundaries.fmaks_l
    snowc = physics_data.mod_radcon.snowc
    sice_am = physics_data.ice_model.sice_am

    alb0 = boundaries.alb0

    snowc = jnp.minimum(1.0, snowd_am / sd2sc)
    alb_l = alb0 + snowc * (albsn - alb0)
    alb_s = albsea + sice_am * (albice - albsea)
    albsfc = alb_s + fmask_l * (alb_l - alb_s)

    # CO2 effect calculation
    increase_co2 = physics_data.shortwave_radiation.increase_co2
    iyear_ref = physics_data.shortwave_radiation.iyear_ref
    model_year = physics_data.date.model_year
    if increase_co2:
        ablco2 = ablco2_ref * jnp.exp(del_co2 * (model_year + tyear - iyear_ref))

    # update alb_l, alb_s, alsfc, etc
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    mod_radcon = physics_data.mod_radcon.copy(albsfc=albsfc, alb_l=alb_l, alb_s=alb_s)
    physics_data = physics_data.copy(mod_radcon=mod_radcon)

    return physics_tendencies, physics_data
