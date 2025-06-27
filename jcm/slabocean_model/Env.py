
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

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

class Env:
    
    def __init__(
        self,
        time_step,
        save_interval,
        total_time,
        start_date,
        horizontal_resolution,
        coords: CoordinateSystem,
        boundaries: BoundaryData,
        parameters: Parameters,
        physics_data: PhysicsData,
        post_process=True, 
        checkpoint_terms=True,
    ) -> None:
 
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            time_step: Model time step in minutes
            save_interval: Save interval in days
            total_time: Total integration time in days
            start_date: Start date of the simulation
            horizontal_resolution: Horizontal resolution of the model (31, 42, 85, or 213)
            coords: CoordinateSystem object describing model grid
            boundaries: BoundaryData object describing surface boundary conditions
            parameters: Parameters object describing model parameters
            physics_specs: PrimitiveEquationsSpecs object describing the model physics
            post_process: Whether to post-process the model output
            checkpoint_terms: Whether to jax.checkpoint each physics term
        """
        
        # Integration settings
        self.start_date = start_date or pd.Timestamp("2000-01-01")
        self.save_interval = save_interval
        self.total_time = total_time

        self.physics_specs = PHYSICS_SPECS

        self.coords = coords
        horizontal_resolution = coords.horizontal.total_wavenumbers - 2
        self.geometry = Geometry.from_coords(self.coords)
        
        parameters.slabocean_model.dt = jnp.array(time_step)
        self.parameters = parameters
        
        
        self.post_process_physics = post_process
        self.boundaries = boundaries
        self.physics_data = physics_data


