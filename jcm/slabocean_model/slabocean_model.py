
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import sigma_layer_boundaries, Geometry

from jcm.slabocean_model.slabocean_model_physics import PhysicsState

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


#
# Questions:
# 1. Mask conditions
# 2. For the `run` function, because of JAX, I need
#    to write out arguments explicitly instead of 
#    calling model.run(). So, there needs a class function
#    that has all the differentiable parameters needed.

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

class SlaboceanModel:

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
        initial_state: PhysicsState=None,
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
            initial_state: Initial state of the model (PhysicsState object), optional
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
        if initial_state is not None:
            self.initial_state = initial_state
        else:
            self.initState()
        


    def initState(self):

        # =========================================================================
        # Initialize land-surface boundary conditions
        # =========================================================================
        print("Initialize state...")

        boundaries = self.boundaries
        parameters = self.parameters
        som_params = parameters.slabocean_model
        geometry = self.geometry

        # Fractional and binary ocean masks
        fmask_l = boundaries.fmask
        bmask_l = jnp.where(fmask_l >= 0.0, 1.0, 0.0)

        # Update fmask_l based on the conditions
        #fmask_l = jnp.where(bmask_l == 1.0,
        #                    jnp.where(boundaries.fmask > (1.0 - parameters.slabocean_model.thrsh), 1.0, fmask_l), 0.0)

        # State
        self.state = PhysicsState.zeros(boundaries.sst_clim.shape[0:2]) # This is pretty ad-hoc. Need better solution

        self.state.d_o = self.state.d_o.at[:].set(10.0)

        # =========================================================================
        # Set heat capacities and dissipation times for soil and ice-sheet layers
        # =========================================================================

        # 2. Compute constant fields
        # Set domain mask (blank out sea points)
        #dmask = jnp.ones_like(fmask_l)
        #dmask = jnp.where(bmask_l < parameters.slabocean_model.flandmin, 0, dmask)

        # Set time_step/heat_capacity and dissipation fields
        #cdland = dmask*parameters.slabocean_model.tdland/(1.0+dmask*parameters.slabocean_model.tdland)
        
        return self
    
    # Exchanges fluxes between ocean and atmosphere.
    def couple_ocn_atm(
        self,
#        state: PhysicsState,
#        physics_data: PhysicsData,
#        parameters: Parameters,
#        boundaries: BoundaryData=None,
#        geometry: Geometry=None
    ) -> tuple[PhysicsState, PhysicsData]:
        
        state = self.state
        physics_data = self.physics_data
        parameters = self.parameters
        boundaries = self.boundaries
        
        day = physics_data.date.model_day()
        sst_anom = None
        
        # Run the ocn model if the flags is switched on
        if boundaries.ocn_coupling_flag:
           
            print("Run!") 
            #state.sst_anom = state.sst_anom.at[:].set(1)


            #print(hash(state.sst_anom))
            #print(state.sst_anom)
            sst_anom, si_anom = run(
                state.sst_anom,
                state.si_anom,
                state.d_o,
                physics_data.surface_flux.hfluxn, # net downward heat flux
                parameters.slabocean_model.dt,
                parameters.slabocean_model.tau0,
            )
            
        # Otherwise get from climatology
        else:
            sst_anom = boundaries.sst_clim[:, :, day]

        # update physics data
        #slabocean_model_data = physics_data.slabocean_model.copy(sst_anom=sst_anom, si_anom=si_anom)
        physics_data = physics_data.copy(slabocean_model=physics_data)
        self.state = state.copy(sst_anom=sst_anom, si_anom=si_anom)
        #physics_tendency = PhysicsTendency.zeros(state.temperature.shape)
        
        #return physics_tendency, physics_data
        return physics_data
        
        
#Integrates slab land-surface model for one day.
@jit
def run(sst_anom, si_anom, d_o, hfluxn, dt, tau0):
#def run(cls, sst_anom, si_anom, d_o, hfluxn, dt, tau0):
    
    factor = 1.0 + dt / tau0 
    # Time evolution of temperature anomaly
    sst_anom = sst_anom / factor + dt / ( factor * physical_constants.cp_ocn * d_o ) * hfluxn[:, :, 0]

    return sst_anom, si_anom




