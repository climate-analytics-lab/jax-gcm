'''
Date: 2/1/2024
For storing all variables related to the model's grid space.
'''

import jax.numpy as jnp

# Declare global variables
hsg = None
dhs = None
fsg = None
dhsr = None
fsgr = None
radang = None
coriol = None
sia = None
coa = None
sia_half = None
coa_half = None
cosg = None
cosgr = None
cosgr2 = None

# Initializes all of the model geometry variables.
def initialize_geometry(kx = 8, il = 64):
    iy = (il + 1)//2
    
    from jcm.physical_constants import akap, omega
    
    global hsg, dhs, fsg, dhsr, fsgr, radang, coriol, sia, coa, sia_half, coa_half, cosg, cosgr, cosgr2
    # Definition of model levels
    if kx == 5:
        hsg = jnp.array([0.000, 0.150, 0.350, 0.650, 0.900, 1.000])
    elif kx == 7:
        hsg = jnp.array([0.020, 0.140, 0.260, 0.420, 0.600, 0.770, 0.900, 1.000])
    elif kx == 8:
        hsg = jnp.array([0.000, 0.050, 0.140, 0.260, 0.420, 0.600, 0.770, 0.900, 1.000])

    # Layer thicknesses and full (u,v,T) levels
    dhs = hsg[1:] - hsg[:-1]
    fsg = 0.5 * (hsg[1:] + hsg[:-1])

    # Additional functions of sigma
    dhsr = 0.5 / dhs
    fsgr = akap / (2.0 * fsg)

    # Horizontal functions

    # Latitudes and functions of latitude
    # NB: J=1 is Southernmost point!
    j = jnp.arange(1, iy + 1)

    sia_half = jnp.cos(3.141592654 * (j - 0.25) / (il + 0.5))
    coa_half = jnp.sqrt(1.0 - sia_half ** 2.0)

    sia = jnp.concatenate((-sia_half, sia_half[::-1]), axis=0).ravel()
    coa = jnp.concatenate((coa_half, coa_half[::-1]), axis=0).ravel()
    radang = jnp.concatenate((-jnp.arcsin(sia_half), jnp.arcsin(sia_half)[::-1]), axis=0)

    # Expand cosine and its reciprocal to cover both hemispheres
    cosg = jnp.repeat(coa_half, 2)
    cosgr = 1. / cosg
    cosgr2 = 1. / (cosg * cosg)

    coriol = 2.0 * omega * sia
