import jax.numpy as jnp
import tree_math

@tree_math.struct
class Boundaries:
    fmask: jnp.ndarray # fractional land-sea mask (ix,il)
    phi0: jnp.ndarray  # surface geopotential (ix, il)
    phis0: jnp.ndarray # spectrally-filtered surface geopotential
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)


def initialize_boundaries(input_filename):
    """
    Initialize the boundary conditions
    """
    from physical_constants import grav
    import xarray as xr

    # Read surface geopotential (i.e. orography)
    phi0 = grav* xr.open_dataset(input_filename)["orog"]

    # Also store spectrally truncated surface geopotential
    phis0 = spectral_truncation(phi0)

    # Read land-sea mask
    fmask = xr.open_dataset(input_filename, mask_and_scale=)["lsm"]

    # Annual-mean surface albedo
    alb0 = xr.open_dataset(input_filename)["alb"]

    # The original code has a fortran function that does the following:
    # where (field <= -999) then field = 0.0 
    # TODO: I don't think we need to do this because of the fcheck's that get done, but should double check

    return Boundaries(fmask=fmask, phi0=phi0, phis0=phis0, alb0=alb0)


def spectral_truncation(fg1):
    # what shouldl we be using for grid_to_spec/spec_to_grid? should that be from dinosaur?
    fsp # (mx,nx)

    fsp = grid_to_spec(fg1)

    for n in range(nx):
        for m in range(mx):
            total_wavenumber = m + n - 2
            if (total_wavenumber > trunc):
                fsp[m,n] = (0.0, 0.0) #--> change to jax.at

    fg2 = spec_to_grid(fsp, 1)

    return fg2


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
    
    field = jnp.where(field <= fmin, fset, field)
    field = jnp.where(field >= fmax, fset, field)

    

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
