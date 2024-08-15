import jax as jnp

# importing custom functions from library
from physical_constants import p0, rgas, cp, alhc, sbc, sigl, wvi, grav
from geometry import coa
# These have not yet been defined - TS 08/14/24
#from mod_radcon import emisfc, alb_l, alb_s, snowc
#from land_model import stl_am, soilw_am
from humidity import get_qsat, rel_hum_to_spec_hum

# constants for sufrace fluxes
fwind0 = 0.95 # Ratio of near-sfc wind to lowest-level wind

# Weight for near-sfc temperature extrapolation (0-1) :
# 1 : linear extrapolation from two lowest levels
# 0 : constant potential temperature ( = lowest level)
ftemp0 = 1.0

# Weight for near-sfc specific humidity extrapolation (0-1) :
# 1 : extrap. with constant relative hum. ( = lowest level)
# 0 : constant specific hum. ( = lowest level)
fhum0 = 0.0

cdl = 2.4e-3   # Drag coefficient for momentum over land
cds = 1.0e-3   # Drag coefficient for momentum over sea
chl = 1.2e-3   # Heat exchange coefficient over land
chs = 0.9e-3   # Heat exchange coefficient over sea
vgust = 5.0    # Wind speed for sub-grid-scale gusts
ctday = 1.0e-2 # Daily-cycle correction (dTskin/dSSRad)
dtheta = 3.0   # Potential temp. gradient for stability correction
fstab = 0.67   # Amplitude of stability correction (fraction)
hdrag = 2000.0 # Height scale for orographic correction
clambda = 7.0  # Heat conductivity in skin-to-root soil layer
clambsn = 7.0  # Heat conductivity in soil for snow cover = 1



def get_surface_fluxes(forog, psa, ua, va, ta, qa, rh , phi, phi0, fmask,  \
                 tsea, ssrd, slrd):
    '''

    Parameters
    ----------
    il - latitude
    ix - longitudes

    forog : 2D array
        - adjustments for drag coefficient. Originally calculated in set_orog_land_sfc_drag
        subroutine. Now used in main

    psa : 2D array
        - Normalised surface pressure
    ua : 3D array
        - u-wind
    va : 3D array
        - v-wind
    ta :  3D array
        - Temperature
    qa : 3D array
        - Specific humidity [g/kg]
    rh : 3D array
        - Relative humidity
    phi : 3D array
        - Geopotential
    phi0 : 2D array
        - Surface geopotential
    fmask : 2D array
        - Fractional land-sea mask
    tsea : 2D array
        - Sea-surface temperature
    ssrd : 2D array
        - Downward flux of short-wave radiation at the surface
    slrd : 2D array
        - Downward flux of long-wave radiation at the surface
    lfluxland : boolean

    Returns
    -------
    '''
    
    ''' 
    # variable was initially declared in the set_orog_land_sfc_drag subroutine
    rhdrag = 1.0/(grav*hdrag)

    forog = 1.0 + rhdrag*(1.0 - jnp.exp(-jnp.max(phi0, 0.0)*rhdrag))
    '''

     

