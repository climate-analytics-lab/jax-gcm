'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''

import jax.numpy as jnp

from jcm.params import ix, il
from speedyf90_types import p

# Physical variables shared among all physics schemes
precnv = jnp.zeros((ix, il), dtype=p) # Convective precipitation [g/(m^2 s)], total
precls = jnp.zeros((ix, il), dtype=p) # Large-scale precipitation [g/(m^2 s)], total
snowcv = jnp.zeros((ix, il), dtype=p) # Convective precipitation [g/(m^2 s)], snow only
snowls = jnp.zeros((ix, il), dtype=p) # Large-scale precipitation [g/(m^2 s)], snow only
cbmf = jnp.zeros((ix, il), dtype=p) # Cloud-base mass flux
tsr = jnp.zeros((ix, il), dtype=p) # Top-of-atmosphere shortwave radiation (downward)
ssrd = jnp.zeros((ix, il), dtype=p) # Surface shortwave radiation (downward-only)
ssr = jnp.zeros((ix, il), dtype=p) # Surface shortwave radiation (net downward)
slrd = jnp.zeros((ix, il), dtype=p) # Surface longwave radiation (downward-only)
slr = jnp.zeros((ix, il), dtype=p) # Surface longwave radiation (net upward)
olr = jnp.zeros((ix, il), dtype=p) # Outgoing longwave radiation (upward)

# Third dimension -> 1:land, 2:sea, 3: weighted average
slru = jnp.zeros((ix, il, 3), dtype=p) # Surface longwave emission (upward)
ustr = jnp.zeros((ix, il, 3), dtype=p) # U-stress
vstr = jnp.zeros((ix, il, 3), dtype=p) # V-stress
shf = jnp.zeros((ix, il, 3), dtype=p) # Sensible heat flux
evap = jnp.zeros((ix, il, 3), dtype=p) # Evaporation [g/(m^2 s)]
hfluxn = jnp.zeros((ix, il, 3), dtype=p) # Net heat flux into surface
