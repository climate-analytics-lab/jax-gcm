'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''

import jax

# Model geometry parameters
trunc = 30 # Spectral truncation total wavenumber
ix = 96 # Number of longitudes
iy = 24 # Number of latitudes in hemisphere
il = 2 * iy # Number of latitudes in full sphere
kx = 8 # Number of vertical levels
nx = trunc + 2 # Number of total wavenumbers for spectral storage arrays
mx = trunc + 1 # Number of zonal wavenumbers for spectral storage arrays
ntr = 1 # Number of tracers (specific humidity is considered a tracer)

# Time stepping parameters
nsteps = 36 # Number of time steps in one day
delt = 86400.0 / nsteps # Time step in seconds
rob = 0.05 # Damping factor in Robert time filter
wil = 0.53 # Parameter of Williams filter
alph = 0.5 # Coefficient for semi-implicit computations

# Physics parameters
iseasc = 1 # Seasonal cycle flag (0=no, 1=yes)
nstrad = 3 # Period (number of steps) for shortwave radiation
sppt_on = False # Turn on SPPT?
issty0 = 1979 # Starting year for SST anomaly file

# User-specified parameters
nstdia = 36 * 5 # Period (number of steps) for diagnostic print-out
nsteps_out = 1 # Number of time steps between outputs

# Function to initialize user-defined parameters from namelist file
@jax.jit
def initialize_params():

    '''REMOVED NAMELIST. UNSURE OF PURPOSE.'''

    # Set default values
    nsteps_out = 1
    nstdia = 36 * 5

    # Read namelist file, if it exists
    try:
        with open("namelist.nml", "r") as file:
            content = file.read()
            params_dict = dict([line.split() for line in content.split('\n') if line])
            nsteps_out = int(params_dict.get('nsteps_out', nsteps_out))
            nstdia = int(params_dict.get('nstdia', nstdia))
    except FileNotFoundError:
        pass

    # Print values to screen
    print(f'nsteps_out (frequency of output)  = {nsteps_out}')
    print(f'nstdia (frequency of diagnostics) = {nstdia}')
