import jax.numpy as jnp


def set_forcing(imode):
    '''
    imode = Mode -> 0 = initialization step, 1 = daily update
    '''
    use dynamical_constants, only: refrh1
    use params
    use horizontal_diffusion, only: tcorh, qcorh
    use physical_constants, only: rgas
    use boundaries, only: phis0, alb0
    use surface_fluxes, only: set_orog_land_sfc_drag
    use date, only: model_datetime, tyear
    use land_model, only: stl_am, snowd_am, fmask_l, sd2sc
    use sea_model, only: fmask_s, sst_am, sice_am
    use mod_radcon, only: ablco2_ref, albsea, albice, snowc, albsn, alb_l, alb_s, albsfc
    use shortwave_radiation, only: get_zonal_average_fields, ablco2, increase_co2
    use longwave_radiation, only: radset
    use humidity, only: get_qsat
    use spectral, only: grid_to_spec

    # time variables for interpolation are set by newdate

    # 1. time-independent parts of physical parametrizations
    if (imode == 0):
        radset()
        set_orog_land_sfc_drag(phis0)
        ablco2_ref = ablco2

    # 2. daily-mean radiative forcing
    # incoming solar radiation
    get_zonal_average_fields(tyear)

    # total surface albedo

    do i = 1, ix
        do j = 1, il
            snowc(i,j)  = min(1.0, snowd_am(i,j)/sd2sc)
            alb_l(i,j)  = alb0(i,j) + snowc(i,j) * (albsn - alb0(i,j))
            alb_s(i,j)  = albsea + sice_am(i,j) * (albice - albsea)
            albsfc(i,j) = alb_s(i,j) + fmask_l(i,j) * (alb_l(i,j) - alb_s(i,j))

    # linear trend of co2 absorptivity (del_co2: rate of change per year)
    iyear_ref = 1950
    del_co2   = 0.005
    # del_co2   = 0.0033

    if (increase_co2):
        ablco2 = ablco2_ref * exp(del_co2 * (model_datetime%year + tyear - iyear_ref))

    # 3. temperature correction term for horizontal diffusion
    setgam(gamlat)

    do j = 1, il
        do i = 1, ix
            corh(i,j) = gamlat(j) * phis0(i,j)

    tcorh = grid_to_spec(corh)

    # 4. humidity correction term for horizontal diffusion
    do j = 1, il
        pexp = 1./(rgas * gamlat(j))
        do i = 1, ix
            tsfc(i,j) = fmask_l(i,j) * stl_am(i,j) + fmask_s(i,j) * sst_am(i,j)
            tref(i,j) = tsfc(i,j) + corh(i,j)
            psfc(i,j) = (tsfc(i,j)/tref(i,j))**pexp

    qref = get_qsat(tref, psfc/psfc, -1.0_p)
    qsfc = get_qsat(tsfc, psfc, 1.0_p)

    corh = refrh1 * (qref - qsfc)

    qcorh = grid_to_spec(corh)

# Compute reference lapse rate as a function of latitude and date
def setgam(gamlat):
    use dynamical_constants, only: gamma
    use params
    use physical_constants, only: grav

    gamlat(1) = gamma/(1000. * grav)
    do j = 2, il
        gamlat(j) = gamlat(1)
    
    return gamlat
