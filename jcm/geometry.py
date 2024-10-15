'''
Date: 2/1/2024
For storing all variables related to the model's grid space.
'''

import jax.numpy as jnp
from jax import jit

from jcm.params import kx, il, iy
from jcm.physical_constants import akap, omega, grav, p0, cp

# Initializes all of the model geometry variables.
@jit
def initialize_geometry(kx=kx, il=il, iy=iy):
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

    # 1.2 Functions of sigma and latitude from physics.f90 initialization
    sigl = jnp.log(fsg) # Logarithm of full-level sigma
    sigh = hsg # Half-level sigma
    grdsig = grav/(dhs*p0) # g/(d_sigma p0): to convert fluxes of u,v,q into d(u,v,q)/dt
    grdscp = grdsig/cp # g/(d_sigma p0 c_p): to convert energy fluxes into dT/dt
    
    # Note that for phys.par. half-lev(k) is between full-lev k and k+1
    # Weights for vertical interpolation at half-levels(1,kx) and surface
    # Fhalf(k) = Ffull(k)+WVI(K,2)*(Ffull(k+1)-Ffull(k))
    # Fsurf = Ffull(kx)+WVI(kx,2)*(Ffull(kx)-Ffull(kx-1))
    wvi = jnp.zeros((kx, 2)) # Weights for vertical interpolation
    wvi = wvi.at[:-1,0].set(1./(sigl[1:]-sigl[:-1]))
    wvi = wvi.at[:-1,1].set((jnp.log(sigh[1:-1])-sigl[:-1])*wvi[:-1,0])
    wvi = wvi.at[-1, 0].set((jnp.log(0.99)-sigl[-1])*wvi[-2,0])
    wvi = wvi.at[-1, 1].set(0.)

    return hsg, dhs, fsg, dhsr, fsgr, sia_half, coa_half, sia, coa, radang, cosg, cosgr, cosgr2, coriol, sigl, sigh, grdsig, grdscp, wvi

hsg, dhs, fsg, dhsr, fsgr, sia_half, coa_half, sia, coa, radang, cosg, cosgr, cosgr2, coriol, sigl, sigh, grdsig, grdscp, wvi = initialize_geometry()