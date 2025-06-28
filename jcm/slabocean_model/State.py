from jcm.slabocean_model import Env

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

import pandas as pd
import xarray as xr
from datetime import datetime

@tree_math.struct
class State:

    sst: jnp.ndarray
    sic: jnp.ndarray
    d_o: jnp.ndarray
    
    @classmethod
    def zeros(
        self,
        shape,
        sst = None,
        sic  = None,
        d_o      = None,
    ):
        return State(
            sst if sst is not None else jnp.zeros(shape),
            sic if sic is not None else jnp.zeros(shape),
            d_o if d_o is not None else jnp.zeros(shape),
        )

    @classmethod
    def ones(
        self,
        shape,
        sst = None,
        sic  = None,
        d_o = None,
    ):
        
        return State(
            sst if sst is not None else jnp.ones(shape),
            sic if sic is not None else jnp.ones(shape),
            d_o if d_o is not None else jnp.ones(shape),
        )

    def copy(
        self,
        sst = None,
        sic  = None,
        d_o = None,
    ):
        return State(
            sst if sst is not None else self.sst,
            sic if sic is not None else self.sic,
            d_o if d_o is not None else self.d_o,
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))


    @classmethod
    def createStateWithEnv(
        cls,
        ev: Env,
    ):
        
        # =========================================================================
        # Initialize land-surface boundary conditions
        # =========================================================================
        print("Initialize state...")
        
        boundaries = ev.boundaries
        parameters = ev.parameters
        som_params = parameters.slabocean_model
        geometry = ev.geometry
        
        # Fractional and binary ocean masks
        fmask_l = boundaries.fmask
        bmask_o = jnp.where(fmask_l == 0.0, 1.0, 0.0)
        
        # Create state
        st = State.zeros(boundaries.sst_clim.shape[0:2]) # This is pretty ad-hoc. Need better solution

        # Define initial sst
        st.sst = jnp.array(boundaries.sst_clim[:, :, 0] + 5)
        st.sic = jnp.array(boundaries.sic_clim[:, :, 0])
        
        d_o = jnp.zeros_like(st.sst) + som_params.d_omax + (som_params.d_omin - som_params.d_omax) * (geometry.coa**3.0)[jnp.newaxis, :]
        #self.st.d_o = self.st.d_o.at[:].set(10.0)
        print("Shape of d_o: ", d_o.shape)
        st = st.copy(
            sst = st.sst.at[bmask_o == 0.0].set(jnp.nan),
            sic = st.sic.at[bmask_o == 0.0].set(jnp.nan),
            d_o = d_o.at[bmask_o == 0.0].set(jnp.nan),
        )

        # =========================================================================
        # Set heat capacities and dissipation times for soil and ice-sheet layers
        # =========================================================================

        # 2. Compute constant fields
        # Set domain mask (blank out sea points)
        #dmask = jnp.ones_like(fmask_l)
        #dmask = jnp.where(bmask_l < parameters.slabocean_model.flandmin, 0, dmask)

        # Set time_step/heat_capacity and dissipation fields
        #cdland = dmask*parameters.slabocean_model.tdland/(1.0+dmask*parameters.slabocean_model.tdland)
        
        return st


