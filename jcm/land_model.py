import jax.numpy as jnp

# should these be moved to params.py?
land_coupling_flag = 1 # Flag for land-coupling (0: off, 1: on)
sd2sc = 60.0 # Snow depth (mm water) corresponding to snow cover = 1

# need to pass this function the boundaries data object -- this should get called from either boundaries.py
# or model.py
def land_model_init(land_filename, snow_filename, soil_filename, surface_filename, boundaries):
    import xarray as xr

    # Soil moisture parameters
    swcap = 0.30 # Soil wetness at field capacity (volume fraction)
    swwil = 0.17 # Soil wetness at wilting point  (volume fraction)
    thrsh = 0.1 # Threshold for land-sea mask definition (i.e. minimum fraction of either land or sea)

    # =========================================================================
    # Initialize land-surface boundary conditions
    # =========================================================================

    # Fractional and binary land masks
    fmask_l = boundaries.fmask
    bmask_l = jnp.where(fmask_l >= thrsh, 1.0, 0.0)

    # Update fmask_l based on the conditions
    fmask_l = jnp.where(fmask_l >= thrsh, 
                        jnp.where(boundaries.fmask > (1.0 - thrsh), 1.0, fmask_l), 0.0)

    # Land-surface temperature
    stl12 = jnp.asarray(xr.open_dataset(surface_filename)["stl"])
    # simple sanity check
    fillsf(stl12(:,:,month), 0.0)
    forchk(bmask_l, 12, 0.0, 400.0, 273.0, stl12)

    # Snow depth
    snowd12 = jnp.asarray(xr.open_dataset(snow_filename)["snowd"])
    # simple sanity check
    forchk(bmask_l, 12, 0.0, 20000.0, 0.0, snowd12)

    # Read soil moisture and compute soil water availability using vegetation fraction
    # Read vegetation fraction
    veg_high = jnp.asarray(xr.open_dataset(surface_filename)["vegh"])
    veg_low  = jnp.asarray(xr.open_dataset(surface_filename)["vegl"])

    # Combine high and low vegetation fractions
    veg = max(0.0, veg_high + 0.8*veg_low)

    # Read soil moisture
    sdep1 = 70.0
    idep2 = 3
    sdep2 = idep2*sdep1

    swwil2 = idep2*swwil
    rsw    = 1.0/(swcap + idep2*(swcap - swwil))

    # Combine soil water content from two top layers
    swl1 = jnp.asarray(xr.open_dataset(soil_filename)["swl1"])
    swl2 = jnp.asarray(xr.open_dataset(soil_filename)["swl2"])
    
    swroot = idep2 * swl2
    # Compute the intermediate max term
    max_term = jnp.maximum(0.0, swroot - swwil2)
    # Compute the soil water content
    soilw12 = jnp.minimum(1.0, rsw * (swl1 + veg * max_term))

    # Repeat the computed 2D array along a new axis for 12 months - dont need to do this we 
    # will have all 12 months at once
    # soilw12 = jnp.tile(soilw12[:, :, None], (1, 1, 12))

    # sanity check
    forchk(bmask_l, 12, 0.0, 10.0, 0.0, soilw12)

    # =========================================================================
    # Set heat capacities and dissipation times for soil and ice-sheet layers
    # =========================================================================

    # Model parameters (default values)
    depth_soil = 1.0 # Soil layer depth (m)
    depth_lice = 5.0 # Land-ice depth (m)
    tdland  = 40.0 # Dissipation time (days) for land-surface temp. anomalies
    flandmin = 1.0/3.0 # Minimum fraction of land for the definition of anomalies
    hcapl  = depth_soil*2.50e+6 # Heat capacities per m^2 (depth*heat_cap/m^3)
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

