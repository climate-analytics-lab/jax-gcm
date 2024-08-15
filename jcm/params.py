'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''

import jax

# Model geometry parameters
trunc = 30      # Spectral truncation total wavenumber
ix = 96         # Number of longitudes
iy = 24         # Number of latitudes in hemisphere
il = 2 * iy     # Number of latitudes in full sphere
kx = 8          # Number of vertical levels
nx = trunc + 2  # Number of total wavenumbers for spectral storage arrays
mx = trunc + 1  # Number of zonal wavenumbers for spectral storage arrays
ntr = 1         # Number of tracers (specific humidity is considered a tracer)

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

##### surface_flux Parameters ###### 
'''
Waiting to see if these variables should be moved over
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

hdrag = 2000.0 # Height scale for orographic correction
clambda = 7.0  # Heat conductivity in skin-to-root soil layer
clambsn = 7.0  # Heat conductivity in soil for snow cover = 1

# fstab is used in more functions other than surface_flux
fstab = 0.67   # Amplitude of stability correction (fraction)
'''