from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData
import jax.numpy as jnp
from jax import jit


import dinosaur
from dinosaur.scales import SI_SCALE, units
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur import primitive_equations, primitive_equations_states

import functools
import pandas as pd

def jit2(func):
    """ This is the decorator to apply jit on member functions that take the first argument as calling object itself. """

    @functools.wraps(func)
    @functools.partial(jit, static_argnums=(0,))
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
        time_step=30.0,
        save_interval=10.0,
        total_time=1200,
        start_date=None,
        horizontal_resolution=31,
        coords: CoordinateSystem=None,
        boundaries: BoundaryData=None,
        initial_state: PhysicsState=None,
        parameters: Parameters=None,
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
        from datetime import datetime

        # Integration settings
        self.start_date = start_date or pd.Timestamp("2000-01-01")
        self.save_interval = save_interval * units.day
        self.total_time = total_time * units.day
        dt_si = time_step * units.minute

        self.physics_specs = PHYSICS_SPECS

        if coords is not None:
            self.coords = coords
            horizontal_resolution = coords.horizontal.total_wavenumbers - 2
        else:
            self.coords = get_coords(layers=8, horizontal_resolution=horizontal_resolution)
        self.geometry = Geometry.from_coords(self.coords)

        self.inner_steps = int(self.save_interval.to(units.minute) / dt_si)
        self.outer_steps = int(self.total_time / self.save_interval)
        self.dt = self.physics_specs.nondimensionalize(dt_si)

        self.parameters = parameters or Parameters.default()
        self.post_process_physics = post_process

        if initial_state is not None:
            self.initial_state = initial_state
        else:
            self.initial_state = None

        self.boundaries = boundaries

    def init(self):
        """
            surface_filename: filename storing boundary data
            parameters: initialized model parameters
            boundaries: partially initialized boundary data
            time_step: time step - model timestep in minutes <= ??? implementation
        """
        import xarray as xr
        # =========================================================================
        # Initialize land-surface boundary conditions
        # =========================================================================

        boundaries = self.boundaries
        parameters = self.parameters
        som_params = parameters.slabocean_model

        # Fractional and binary ocean masks
        fmask_l = boundaries.fmask
        bmask_l = jnp.where(fmask_l >= 0.0, 1.0, 0.0)

        # Update fmask_l based on the conditions
        #fmask_l = jnp.where(bmask_l == 1.0,
        #                    jnp.where(boundaries.fmask > (1.0 - parameters.slabocean_model.thrsh), 1.0, fmask_l), 0.0)

        # State
        sst_clim = jnp.asarray(boundaries.sst_clim)
        si_clim  = jnp.asarray(boundaries.sice_am)

        print("d_omax = ", som_params.d_omax)
        d_o      = jnp.asarray(boundaries.sst_clim) * 0 + som_params.d_omax # this way we also copy nan together 
        
         
        # =========================================================================
        # Set heat capacities and dissipation times for soil and ice-sheet layers
        # =========================================================================

        # 2. Compute constant fields
        # Set domain mask (blank out sea points)
        #dmask = jnp.ones_like(fmask_l)
        #dmask = jnp.where(bmask_l < parameters.slabocean_model.flandmin, 0, dmask)

        # Set time_step/heat_capacity and dissipation fields
        #cdland = dmask*parameters.slabocean_model.tdland/(1.0+dmask*parameters.slabocean_model.tdland)

        return boundaries.copy()

    # Exchanges fluxes between ocean and atmosphere.
    def couple_ocn_atm(
        self,
        state: PhysicsState,
        physics_data: PhysicsData,
        parameters: Parameters,
        boundaries: BoundaryData=None,
        geometry: Geometry=None
    ) -> tuple[PhysicsTendency, PhysicsData]:
        
        day = physics_data.date.model_day()
        
        stl_lm=None

        # Run the ocn model if the flags is switched on
        if (boundaries.ocn_coupling_flag):
            
            sst_anom = runExplicit(
                physics_data.surface_flux.hfluxn, # net downward heat flux
                physics_data.slabocean_model.sst_anom,
                physics_data.slabocean_model.si_anom,
                boundaries.ocn_d0,
            )

            stl_am = stl_lm
       

         # Otherwise get the land surface from climatology
        else:
            stl_am = boundaries.stlcl_ob[:,:,day]

        # update land physics data
        slabocean_model_data = physics_data.slabocean_model.copy(stl_am=stl_am, stl_lm=stl_lm)
        physics_data = physics_data.copy(slabocean_model=slabocean_model_data)
        physics_tendency = PhysicsTendency.zeros(state.temperature.shape)

        return physics_tendency, physics_data


    def run(self):
        
        slabocean_model.runExplicit(self.xxx)
        

    #Integrates slab land-surface model for one day.
    @jit2
    @classmethod
    def runExplicit(cls, sst_a, d_o, c_o):
        
        # Anomaly w.r.t. final-time climatological temperature
        tanom = stl_lm - stlcl_ob

        # Time evolution of temperature anomaly
        tanom = cdland*(tanom + rhcapl*hfluxn[:,:,0])

        # Full surface temperature at final time
        stl_lm = tanom + stlcl_ob

        return stl_lm

