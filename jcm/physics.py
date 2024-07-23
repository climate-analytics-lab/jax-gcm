'''
Date: 2/7/2024
Physics module.
'''

import jax
import jax.numpy as jnp

# Initialize physical parametrization routines
def initialize_physics():
    from physical_constants import grav, cp, p0, sigl, sigh, grdsig, grdscp, wvi
    from geometry import hsg, fsg, dhs

    '''
    kx = len(fsg)
    sigh = jnp.append(hsg[1:], hsg[-1]) # Slight adjustment to match the Fortran code

    for k in range(kx):
        sigl[k] = jnp.log(fsg[k])
        grdsig[k] = grav / (dhs[k] * p0)
        grdscp[k] = grdsig[k] / cp
    '''

    # 1.2 Functions of sigma and latitude
    sigh = jnp.concatenate((jnp.array([hsg[1]]), hsg[1:]))
    sigl = jnp.log(fsg)
    grdsig = grav / (dhs * p0)
    grdscp = grdsig / cp

    '''
    UNSURE ABOUT BUILT IN ARRAY OPERATIONS OUTPUT
    Code in loops
    # Weights for vertical interpolation at half-levels(1,kx) and surface
    for k in range(kx - 1):
        wvi[k, 0] = 1.0 / (sigl[k + 1] - sigl[k])
        wvi[k, 1] = (jnp.log(sigh[k]) - sigl[k]) * wvi[k, 0]

    wvi = jax.ops.index_update(wvi, kx - 1, jnp.array([0., (jnp.log(0.99) - sigl[kx - 1]) * wvi[kx - 2, 0]]))
    '''

    # Weights for vertical interpolation at half-levels(1,kx) and surface
    # Note that for phys.par. half-lev(k) is between full-lev k and k+1
    # Fhalf(k) = Ffull(k)+WVI(K,2)*(Ffull(k+1)-Ffull(k))
    # Fsurf = Ffull(kx)+WVI(kx,2)*(Ffull(kx)-Ffull(kx-1))
    wvi_1 = 1.0 / (sigl[1:] - sigl[:-1])
    wvi_2 = (jnp.log(sigh[:-1]) - sigl[:-1]) * wvi_1
    wvi = jnp.column_stack((wvi_1, wvi_2))


    wvi_last = jnp.array([0., (jnp.log(0.99) - sigl[-1]) * wvi[-1, 0]])
    wvi = jnp.vstack((wvi[:-1], wvi_last))

    return sigl, sigh, grdsig, grdscp, wvi

