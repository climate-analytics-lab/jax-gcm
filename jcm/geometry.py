'''
Date: 2/1/2024
For storing all variables related to the model's grid space.
'''

import jax.numpy as jnp
from jax import jit

from jcm.physical_constants import akap, omega

# Initializes all of the model geometry variables.
@jit
def initialize_geometry(self, kx = 8, il = 64, iy = 32):
    # Definition of model levels
    if kx == 5:
        self.hsg = jnp.array([0.000, 0.150, 0.350, 0.650, 0.900, 1.000])
    elif kx == 7:
        self.hsg = jnp.array([0.020, 0.140, 0.260, 0.420, 0.600, 0.770, 0.900, 1.000])
    elif kx == 8:
        self.hsg = jnp.array([0.000, 0.050, 0.140, 0.260, 0.420, 0.600, 0.770, 0.900, 1.000])

    # Layer thicknesses and full (u,v,T) levels
    self.dhs = self.hsg[1:] - self.hsg[:-1]
    self.fsg = 0.5 * (self.hsg[1:] + self.hsg[:-1])

    # Additional functions of sigma
    self.dhsr = 0.5 / self.dhs
    self.fsgr = akap / (2.0 * self.fsg)

    # Horizontal functions

    # Latitudes and functions of latitude
    # NB: J=1 is Southernmost point!
    j = jnp.arange(1, iy + 1)

    self.sia_half = jnp.cos(3.141592654 * (j - 0.25) / (il + 0.5))
    self.coa_half = jnp.sqrt(1.0 - self.sia_half ** 2.0)

    self.sia = jnp.concatenate((-self.sia_half, self.sia_half[::-1]), axis=0).ravel()
    self.coa = jnp.concatenate((self.coa_half, self.coa_half[::-1]), axis=0).ravel()
    self.radang = jnp.concatenate((-jnp.arcsin(self.sia_half), jnp.arcsin(self.sia_half)[::-1]), axis=0)

    # Expand cosine and its reciprocal to cover both hemispheres
    self.cosg = jnp.repeat(self.coa_half, 2)
    self.cosgr = 1. / self.cosg
    self.cosgr2 = 1. / (self.cosg * self.cosg)

    self.coriol = 2.0 * omega * self.sia
