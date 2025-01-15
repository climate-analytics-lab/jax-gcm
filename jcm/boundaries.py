import jax.numpy as jnp
from jax import jit
from jcm.physical_constants import grav
from jcm.params import mx, nx

#global variables 
fmask = None #fractional land-sea mask (ix,il)
phi0 = None  #surface geopotential (ix, il)
phis0 = None #spectrally-filtered surface geopotential
alb0 = None #bare-land annual mean albedo (ix,il)

def initialize_boundaries(input_filename):
    global phi0, phis0, fmask, alb0

    # Read surface geopotential (i.e. orography)
    phi0 = grav*load_boundary_file(input_filename, "orog")

    # Also store spectrally truncated surface geopotential
    phis0 = spectral_truncation(phi0)

    # Read land-sea mask
    fmask = load_boundary_file(input_filename, "lsm")

    # Annual-mean surface albedo
    alb0 = load_boundary_file(input_filename, "alb")


def spectral_truncation(fsp, trunc):
    # given fsp, a spectral representation of a field, return a truncated version
    nx = trunc+2 # Number of total wavenumbers for spectral storage arrays
    mx = trunc+1 # Number of zonal wavenumbers for spectral storage arrays

    n_indices, m_indices = jnp.meshgrid(jnp.arange(nx), jnp.arange(mx), indexing='ij')
    total_wavenumber = m_indices + n_indices
    fsp = jnp.where(total_wavenumber > trunc, 0.0, fsp)

    return fsp

def load_boundary_file(file_name, field_name):
    # implement whatever python method we want for loading a variable into ix, il array from ncfile
    field = None

    # Fix undefined values
    # where (field <= -999) then field = 0.0 

    return field

# land model and sea model call this 
# Check consistency of surface fields with land-sea mask and set undefined 
# values to a constant (to avoid over/underflow).
def forchk(fmask, nf, fmin, fmax, fset, field):
    '''
        nf              # The number of input 2D fields
        fmin            # The minimum allowable value
        fmax            # The maximum allowable value
        fset            # Replacement for undefined values
    '''
    
    # need to adjust indices -- these are the fortran version 
    ix, il = fmask.shape
    for jf in range(nf):
        nfault = 0

        for i in range(ix):
            for j in range(il):
                if (fmask[i,j] > 0.0):
                    if ((field[i,j,jf] < fmin) | (field[i,j,jf] > fmax)):
                        nfault = nfault + 1
                else:
                    field[i,j,jf] = fset

    
    return field

# Replace missing values in surface fields.
# It is assumed that non-missing values exist near the Equator.
def fillsf(sf, fmis):
    '''
        sf(ix,il) !! Field to replace missing values in - return this same thing
        fmis      !! Replacement for missing values
    '''

    # real(p) :: sf2(0:ix+1)
    # need to set up j here

    # need to adjust indices -- these are the fortran version 
    for hemisphere in range(2):
        if (hemisphere == 1):
            j1 = il/2
            j2 = 1
            j3 = -1
        else:
            j1 = j1+1
            j2 = il
            j3 = 1

        for j in j1, j2, j3:
            sf2[1:ix] = sf[:,j]

            nmis = 0
            for i in range(ix):
                if (sf[i,j] < fmis):
                    nmis = nmis + 1
                    sf2[i] = 0.0

            if (nmis < ix): 
                fmean = sum(sf2[1:ix])/float(ix - nmis)

            for i in range(ix):
                if (sf[i,j] < fmis): 
                    sf2[i] = fmean

            sf2[0] = sf2[ix]
            sf2[ix+1] = sf2[1]
            for i in range[ix]:
                if (sf[i,j] < fmis):
                    sf[i,j] = 0.5*(sf2[i-1] + sf2[i+1])
