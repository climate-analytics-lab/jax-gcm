import jax.numpy as jnp

# linear trend of co2 absorptivity (del_co2: rate of change per year)
iyear_ref = 1950
del_co2   = 0.005
imode = 0 # this won't work if you want to re-run the model without restarting the program entirely

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
    from jcm.mod_radcon import ablco2_ref, albsea, albice, albsn
    from jcm.shortwave_radiation import ablco2, increase_co2

    # time variables for interpolation are set by newdate

    # 1. time-independent parts of physical parametrizations
    if (imode == 0):
        imode = 1
        ablco2_ref = ablco2

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
    if increase_co2:
        ablco2 = ablco2_ref * jnp.exp(del_co2 * (model_datetime%year + tyear - iyear_ref))

    # POSSIBLE WE DON"T NEED THIS BECAUSE ITS HANDLED IN DINOSAUR
    # # 3. temperature correction term for horizontal diffusion
    # setgam(gamlat)

    # do j = 1, il
    #     do i = 1, ix
    #         corh(i,j) = gamlat(j) * phis0(i,j)

    # tcorh = grid_to_spec(corh)

    # # 4. humidity correction term for horizontal diffusion
    # do j = 1, il
    #     pexp = 1./(rgas * gamlat(j))
    #     do i = 1, ix
    #         tsfc(i,j) = fmask_l(i,j) * stl_am(i,j) + fmask_s(i,j) * sst_am(i,j)
    #         tref(i,j) = tsfc(i,j) + corh(i,j)
    #         psfc(i,j) = (tsfc(i,j)/tref(i,j))**pexp

    # qref = get_qsat(tref, psfc/psfc, -1.0_p)
    # qsfc = get_qsat(tsfc, psfc, 1.0_p)

    # corh = refrh1 * (qref - qsfc)

    # qcorh = grid_to_spec(corh)

    # update alb_l, alb_s, alsfc, etc
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    mod_radcon = physics_data.mod_radcon.copy(albsfc=albsfc, alb_l=alb_l, alb_s=alb_s)
    physics_data = physics_data.copy(mod_radcon=mod_radcon)

    return physics_tendencies, physics_data

# Compute reference lapse rate as a function of latitude and date
# def setgam(gamlat):
#     from physical_constants import grav, gamma

#     gamlat(1) = gamma/(1000. * grav)
#     gamlat = gamlat.at[1:il].set(gamlat[0])
    
#     return gamlat
