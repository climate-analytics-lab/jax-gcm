import jax.numpy as jnp

# linear trend of co2 absorptivity (del_co2: rate of change per year)
iyear_ref = 1950
del_co2   = 0.005

# need to call once on initialization, (or just need to know if its the first call)
def set_forcing(state, physics_data, boundaries):
    '''
    imode = Mode -> 0 = initialization step, 1 = daily update
    '''
    from dynamical_constants import refrh1
    from jcm.params import ?
    from jcm.date import days_year
    from horizontal_diffusion import tcorh, qcorh
    from jcm.shortwave_radiation import get_zonal_average_fields
    from jcm.physical_constants import rgas
    from jcm.land_model import sd2sc
    from jcm.mod_radcon import ablco2_ref, albsea, albice, albsn
    from jcm.shortwave_radiation import ablco2, increase_co2

    # time variables for interpolation are set by newdate

    # 1. time-independent parts of physical parametrizations
    if (imode == 0):
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

    alb0 = boundaries.alb0

    do i = 1, ix
        do j = 1, il
            snowc(i,j)  = min(1.0, snowd_am(i,j)/sd2sc)
            alb_l(i,j)  = alb0(i,j) + snowc(i,j) * (albsn - alb0(i,j))
            alb_s(i,j)  = albsea + sice_am(i,j) * (albice - albsea)
            albsfc(i,j) = alb_s(i,j) + fmask_l(i,j) * (alb_l(i,j) - alb_s(i,j))

    if (increase_co2):
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

    return physics_tendencies, physics_data

# Compute reference lapse rate as a function of latitude and date
def setgam(gamlat):
    use dynamical_constants, only: gamma
    use params
    use physical_constants, only: grav

    gamlat(1) = gamma/(1000. * grav)
    do j = 2, il
        gamlat(j) = gamlat(1)
    
    return gamlat
