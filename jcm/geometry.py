'''
Date: 2/1/2024
For storing all variables related to the model's grid space.
'''
import jax.numpy as jnp

# to prevent blowup of gradients
epsilon = 1e-9

# Declare global variables
hsg = None
dhs = None
fsg = None
sigl = None
radang = None
sia = None
coa = None
sia_half = None
coa_half = None

# Initializes all of the model geometry variables.
def initialize_geometry(kx = 8, il = 64, coords=None):
    global hsg, dhs, fsg, sigl, radang, sia, coa, sia_half, coa_half

    if coords is not None:
        kx = len(coords.vertical.boundaries)-1
        il = coords.horizontal.nodal_shape[1]

    # Definition of model levels
    # Layer thicknesses and full (u,v,T) levels
    # FIXME: if coords is not None, hsg, fsg, dhs should be coords.vertical.boundaries, centers, layer_thickness respectively, but there is some issue with coords being jitted
    sigma_layer_boundaries = {
        5: jnp.array([0.0, 0.15, 0.35, 0.65, 0.9, 1.0]),
        7: jnp.array([0.02, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
        8: jnp.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
    }
    if kx not in sigma_layer_boundaries:
        raise ValueError(f"Invalid number of vertical levels: {kx}")
    hsg = sigma_layer_boundaries[kx]
    fsg = 0.5 * (hsg[1:] + hsg[:-1])
    dhs = hsg[1:] - hsg[:-1]

    sigl = jnp.log(fsg) # Moved here from physical_constants

    # Horizontal functions
    # Latitudes and functions of latitude
    if coords is not None:
        radang = coords.horizontal.latitudes
        sia = jnp.sin(radang)
        coa = jnp.cos(radang)
    else:
        # NB: J=1 is Southernmost point!
        iy = (il + 1)//2
        j = jnp.arange(1, iy + 1)
        sia_half = jnp.cos(jnp.pi * (j - 0.25) / (il + 0.5))
        coa_half = jnp.sqrt(1.0 - sia_half ** 2.0)
        sia = jnp.concatenate((-sia_half, sia_half[::-1]), axis=0).ravel()
        coa = jnp.concatenate((coa_half, coa_half[::-1]), axis=0).ravel()
        radang = jnp.concatenate((-jnp.arcsin(sia_half), jnp.arcsin(sia_half)[::-1]), axis=0)
