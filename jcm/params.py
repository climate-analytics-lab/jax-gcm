'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''

# Time stepping parameters
nsteps = 36     # Number of time steps in one day
delt = 86400.0 / nsteps # Time step in seconds
rob = 0.05      # Damping factor in Robert time filter
wil = 0.53      # Parameter of Williams filter
alph = 0.5      # Coefficient for semi-implicit computations

# Physics parameters
iseasc = 1      # Seasonal cycle flag (0=no, 1=yes)
nstrad = 3      # Period (number of steps) for shortwave radiation
sppt_on = False # Turn on SPPT?
issty0 = 1979   # Starting year for SST anomaly file

# User-specified parameters
nstdia = 36 * 5 # Period (number of steps) for diagnostic print-out
nsteps_out = 1  # Number of time steps between outputs

trunc = 30   # Spectral truncation total wavenumber
nx = trunc+2 # Number of total wavenumbers for spectral storage arrays
mx = trunc+1 # Number of zonal wavenumbers for spectral storage arrays