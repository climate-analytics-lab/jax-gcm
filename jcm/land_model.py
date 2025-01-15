import jax.numpy as jnp

    use types, only: p

    public stl_am, snowd_am, soilw_am
    public land_model_init, couple_land_atm
    public fmask_l
    public land_coupling_flag
    public sd2sc

    real(p) :: rhcapl(ix,il) !! 1/heat capacity (land)
    real(p) :: cdland(ix,il) !! 1/dissipation time (land)

    ! Daily observed climatological fields over land
    real(p) :: stlcl_ob(ix,il)   !! Climatological land surface temperature
    real(p) :: snowdcl_ob(ix,il) !! Climatological snow depth (water equivalent)
    real(p) :: soilwcl_ob(ix,il) !! Climatological soil water availability

    ! Land surface fields used by atmospheric model
    real(p) :: stl_am(ix,il)   !! Land surface temperature
    real(p) :: snowd_am(ix,il) !! Snow depth (water equivalent)
    real(p) :: soilw_am(ix,il) !! Soil water availability

    ! Land surface fields from land model
    real(p) :: stl_lm(ix,il) !! Land-model surface temperature

    ! Land masks
    real(p) :: fmask_l(ix,il) !! Fraction of land
    real(p) :: bmask_l(ix,il) !! Binary land mask

    real(p) :: stl12(ix,il,12)   !! Land surface temperature monthly-mean climatology
    real(p) :: snowd12(ix,il,12) !! Snow depth (water equivalent) monthly-mean climatology
    real(p) :: soilw12(ix,il,12) !! Soil water availability monthly-mean climatology

    integer :: land_coupling_flag = 1 !! Flag for land-coupling (0: off, 1: on)

    real(p), parameter :: sd2sc = 60.0 !! Snow depth (mm water) corresponding to snow cover = 1

# need to pass this function the boundaries data object -- this should get called from either boundaries.py
# or model.py
def land_model_init(land_filename, snow_filename, soil_filename, surface_filename):
    # import some sort of boundaires check function
    from boundaries import forchk, fillsf

    # Auxiliary variables
    real(p) :: dmask(ix,il) ! domain mask
    real(p) :: depth_soil, depth_lice, tdland, hcapl, hcapli, flandmin

    # Soil moisture parameters
    # Soil wetness at field capacity (volume fraction)
    swcap = 0.30

    # Soil wetness at wilting point  (volume fraction)
    swwil = 0.17

    # Threshold for land-sea mask definition (i.e. minimum fraction of
    # either land or sea)
    thrsh = 0.1

    real(p) :: rsw, sdep1, sdep2, swroot, swwil2
    real(p), dimension(ix,il) :: veg_low, veg_high, veg, swl1, swl2
    integer :: idep2

    # =========================================================================
    # Initialize land-surface boundary conditions
    # =========================================================================

    # Fractional and binary land masks
    fmask_l = fmask
    do j = 1, il
        do i = 1, ix
            if (fmask_l(i,j) >= thrsh) then
                bmask_l(i,j) = 1.0
                if (fmask(i,j) > (1.0 - thrsh)) fmask_l(i,j) = 1.0
            else
                bmask_l(i,j) = 0.0
                fmask_l(i,j) = 0.0

    # Land-surface temperature
    do month = 1, 12
        stl12(:,:,month) = load_boundary_file(land_filename, "stl", month)

        # simple sanity check
        fillsf(stl12(:,:,month), 0.0_p)

    # simple sanity check
    forchk(bmask_l, 12, 0.0_p, 400.0_p, 273.0_p, stl12)

    # Snow depth
    do month = 1, 12
        snowd12(:,:,month) = load_boundary_file(snow_filename, "snowd", month)

    # simple sanity check
    forchk(bmask_l, 12, 0.0_p, 20000.0_p, 0.0_p, snowd12)

    # Read soil moisture and compute soil water availability using vegetation fraction
    # Read vegetation fraction
    veg_high = load_boundary_file(surface_filename, "vegh")
    veg_low  = load_boundary_file(surface_filename, "vegl")

    # Combine high and low vegetation fractions
    veg = max(0.0, veg_high + 0.8*veg_low)

    # Read soil moisture
    sdep1 = 70.0
    idep2 = 3
    sdep2 = idep2*sdep1

    swwil2 = idep2*swwil
    rsw    = 1.0/(swcap + idep2*(swcap - swwil))

    do month = 1, 12
        # Combine soil water content from two top layers
        swl1 = load_boundary_file(soil_filename, "swl1", month)
        swl2 = load_boundary_file(soil_filename, "swl2", month)

        do j = 1, il
            do i = 1, ix
                swroot = idep2*swl2(i,j)
                soilw12(i,j,month) = min(1.0, rsw*(swl1(i,j) + veg(i,j) &
                    & *max(0.0, swroot - swwil2)))

    # replace this with whatever check function duncan comes up with
    forchk(bmask_l, 12, 0.0_p, 10.0_p, 0.0_p, soilw12)

    # =========================================================================
    # Set heat capacities and dissipation times for soil and ice-sheet layers
    # =========================================================================

    # Model parameters (default values)

    # Soil layer depth (m)
    depth_soil = 1.0

    # Land-ice depth (m)
    depth_lice = 5.0

    # Dissipation time (days) for land-surface temp. anomalies
    tdland  = 40.

    # Minimum fraction of land for the definition of anomalies
    flandmin = 1./3.

    # Heat capacities per m^2 (depth*heat_cap/m^3)
    hcapl  = depth_soil*2.50e+6
    hcapli = depth_lice*1.93e+6

    # 2. Compute constant fields
    # Set domain mask (blank out sea points)
    dmask(:,:) = 1.

    do j=1,il
        do i=1,ix
            if (fmask_l(i,j).lt.flandmin) dmask(i,j) = 0

    ! Set time_step/heat_capacity and dissipation fields
    do j=1,il
        do i=1,ix
            if (alb0(i,j).lt.0.4) then
                rhcapl(i,j) = delt/hcapl
            else
                rhcapl(i,j) = delt/hcapli

    cdland(:,:) = dmask(:,:)*tdland/(1.+dmask(:,:)*tdland)

#Exchanges fluxes between land and atmosphere.
def couple_land_atm(day):
    # day = the day (starting at 0 for the first time step)
    # Fortran has these imports, we don't necessarily have them
    from date import imont1 
    from interpolation import forin5, forint

    # Interpolate climatological fields to actual date
    # Climatological land surface temperature
    forin5(imont1, stl12, stlcl_ob)

    # Climatological snow depth
    forint(imont1, snowd12, snowdcl_ob)

    # Climatological soil water availability
    forint(imont1, soilw12, soilwcl_ob)

    # If it's the first day then initialise the land surface
    # temperature from climatology
    if (day == 0) then
        stl_lm = stlcl_ob
        stl_am = stlcl_ob
    else
        # Run the land model if the land model flags is switched on
        if (land_coupling_flag == 1) then
            call run_land_model

            stl_am = stl_lm
        # Otherwise get the land surface from climatology
        else
            stl_am = stlcl_ob

    # Always get snow depth and soil water availability from climatology
    snowd_am = snowdcl_ob
    soilw_am = soilwcl_ob

#Integrates slab land-surface model for one day.
def run_land_model():
    from jcm.auxiliaries import hfluxn

    # Surface temperature anomaly
    real(p) :: tanom(ix,il)

    # Land-surface (soil/ice-sheet) layer
    # Anomaly w.r.t. final-time climatological temperature
    tanom = stl_lm - stlcl_ob

    # Time evolution of temperature anomaly
    tanom = cdland*(tanom + rhcapl*hfluxn(:,:,1))

    # Full surface temperature at final time
    stl_lm = tanom + stlcl_ob

