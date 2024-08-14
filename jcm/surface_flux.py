import jax as jnp

# importing custom functions from library
import physical_constants
import geometry
# These have not yet been defined - TS 08/14/24
#import mod_radcon
#import land_model
from humidity import get_qsat

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



def get_surface_fluxes(ix,il, psa, ua, va, ta, qa, rh , phi, phi0, fmask,  \
                 tsea, ssrd, slrd):
    '''

    Parameters
    ----------
    il - latitude
    ix - longitudes
    '''
    
    #(\___/)
    #(=^.^=) In the fortran code this is a variable that is used in both subroutines 
    #(")_(") but it requires initialization from il,ix. Need to determine if get_surface_flux is always declared first.
    global forog 
    forog = jnp.zeros([il,ix]) # Time-invariant fields (initial. in SFLSET)
