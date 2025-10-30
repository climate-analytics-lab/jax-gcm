import jax.numpy as jnp
from jcm.geometry import Geometry

def convert_to_speedy_latitudes(geometry: Geometry) -> Geometry:
    # Recompute horizontal fields for speedy latitudes
    il = geometry.nodal_shape[2]
    iy = (il + 1)//2
    j = jnp.arange(1, iy + 1)
    sia_half = jnp.cos(jnp.pi * (j - 0.25) / (il + 0.5))
    radang = jnp.concatenate((-jnp.arcsin(sia_half), jnp.arcsin(sia_half)[::-1]), axis=0)
    sia = jnp.concatenate((-sia_half, sia_half[::-1]), axis=0).ravel()
    coa = jnp.cos(radang)

    # Changing latitudes makes phis0 incorrect unless orography is flat
    phis0 = geometry.phis0 if jnp.allclose(geometry.orog, geometry.orog[0,0]) else jnp.full_like(geometry.phis0, jnp.nan)
    
    return geometry.replace(phis0=phis0, radang=radang, sia=sia, coa=coa)