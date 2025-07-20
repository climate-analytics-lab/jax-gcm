
from jcm.slabocean_model import Env, Core, State


from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import sigma_layer_boundaries, Geometry

#from jcm.slabocean_model.slabocean_model_physics import PhysicsState
import tree_math

from jcm.physics_data import PhysicsData
from jcm import physical_constants

import jax.numpy as jnp
from jax import jit

from jax.tree_util import Partial

import dinosaur
from dinosaur.scales import SI_SCALE, units
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur import primitive_equations, primitive_equations_states

#import tree_util.Partial as Partial
#import functools
import pandas as pd
import xarray as xr
from datetime import datetime

def jit2(func):
    """ This is the decorator to apply jit on member functions that take the first argument as calling object itself. """
    #@functools.wraps(func)
    @jax.tree_util.Partial(jit, static_argnums=0)
    def wrapped_func(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    return wrapped_func


PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

def get_coords(layers=8, horizontal_resolution=31) -> CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (31, 42, 85, or 213).
    """
    resolution_map = {
        31: dinosaur.spherical_harmonic.Grid.T31,
        42: dinosaur.spherical_harmonic.Grid.T42,
        85: dinosaur.spherical_harmonic.Grid.T85,
        213: dinosaur.spherical_harmonic.Grid.T213,
    }

    if horizontal_resolution not in resolution_map:
        raise ValueError(f"Invalid resolution: {horizontal_resolution}. Must be one of: {list(resolution_map.keys())}")

    if layers not in sigma_layer_boundaries:
        raise ValueError(f"Invalid number of layers: {layers}. Must be one of: {list(sigma_layer_boundaries.keys())}")

    # Define the coordinate system
    return dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=resolution_map[horizontal_resolution](radius=PHYSICS_SPECS.radius), # truncation
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(sigma_layer_boundaries[layers])
    )



def initBoundaries(
    filename,
    parameters: Parameters,
    boundaries: BoundaryData,
):
    """
        filename: filename storing boundary data
        parameters: initialized model parameters
        boundaries: partially initialized boundary data
    """
    
    # =========================================================================
    # Initialize ocean surface boundary conditions
    # =========================================================================
   
    print("Reading file containing boundary information: ", filename)

    with xr.open_dataset(filename) as ds_bc:
        sst_clim = jnp.asarray(ds_bc["sst"])
        sic_clim = jnp.asarray(ds_bc["icec"])
    
    return boundaries.copy(
        sst_clim = sst_clim,
        sic_clim = sic_clim,
    )

