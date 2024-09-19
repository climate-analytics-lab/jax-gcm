import jax.numpy as jnp
from jcm.params import ix, il, kx
from jcm.physics import ModRadConData

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

mod_radcon_data = ModRadConData(
    alb_l=jnp.zeros((ix, il)),
    alb_s=jnp.zeros((ix, il)),
    albsfc=jnp.zeros((ix, il)),
    snowc=jnp.zeros((ix, il)),
    tau2=jnp.zeros((ix, il, kx, 4)),
    st4a=jnp.zeros((ix, il, kx, 2)),
    stratc=jnp.zeros((ix, il, 2)),
    flux=jnp.zeros((ix, il, 4))
)


