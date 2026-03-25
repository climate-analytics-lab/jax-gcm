import jax.numpy as jnp
from jcm.terrain import TerrainData
from jcm.physics.speedy.speedy_coords import SpeedyCoords
from typing import Tuple

def convert_to_speedy_latitudes(terrain: TerrainData, speedy_coords: SpeedyCoords) -> Tuple[TerrainData, SpeedyCoords]:
    # Recompute horizontal fields for speedy latitudes
    il = terrain.orog.shape[1]
    iy = (il + 1)//2
    j = jnp.arange(1, iy + 1)
    sia_half = jnp.cos(jnp.pi * (j - 0.25) / (il + 0.5))
    radang = jnp.concatenate((-jnp.arcsin(sia_half), jnp.arcsin(sia_half)[::-1]), axis=0)
    sia = jnp.concatenate((-sia_half, sia_half[::-1]), axis=0).ravel()
    coa = jnp.cos(radang)

    # Changing latitudes makes phis0 incorrect unless orography is flat
    phis0 = terrain.phis0 if jnp.allclose(terrain.orog, terrain.orog[0,0]) else jnp.full_like(terrain.phis0, jnp.nan)
    
    return terrain.copy(phis0=phis0), speedy_coords.copy(radang=radang, sia=sia, coa=coa)