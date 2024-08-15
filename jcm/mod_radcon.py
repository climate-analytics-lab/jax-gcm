import jax.numpy as jnp

# Radiation and cloud constants
# Albedo values
albsea = 0.07  # Albedo over sea
albice = 0.60  # Albedo over sea ice (for ice fraction = 1)
albsn  = 0.60  # Albedo over snow (for snow cover = 1)

# Longwave parameters
epslw  = 0.05  # Fraction of blackbody spectrum absorbed/emitted by PBL only
emisfc = 0.98  # Longwave surface emissivity

# Placeholder for CO2-related variable
ablco2_ref = None  # To be initialized elsewhere

# Time-invariant fields (arrays)
# fband = energy fraction emitted in each LW band = f(T)
fband = jnp.zeros((301, 4))  # Example shape (100:400, 4)

# Radiative properties of the surface (updated in fordate)
# Albedo and snow cover arrays
alb_l = jnp.zeros((ix, il))  # Daily-mean albedo over land (bare-land + snow)
alb_s = jnp.zeros((ix, il))  # Daily-mean albedo over sea (open sea + sea ice)
albsfc = jnp.zeros((ix, il)) # Combined surface albedo (land + sea)
snowc = jnp.zeros((ix, il))  # Effective snow cover (fraction)

# Transmissivity and blackbody radiation (updated in radsw/radlw)
tau2 = jnp.zeros((ix, il, kx, 4))     # Transmissivity of atmospheric layers
st4a = jnp.zeros((ix, il, kx, 2))     # Blackbody emission from full and half atmospheric levels
stratc = jnp.zeros((ix, il, 2))       # Stratospheric correction term
flux = jnp.zeros((ix, il, 4))         # Radiative flux in different spectral bands


