
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

class Model:

    ev : Env
    st : State
    co : Core

    def __init__(
        self,
        ev,
        init_co = False,
    ) -> None:

        self.ev = ev
        if init_co:
            self.initCore()

        self.st = State.createStateWithEnv(ev)
    
    def initCore(self):
        self.co = Core(self.ev)

    # Exchanges fluxes between ocean and atmosphere.
    def couple_ocn_atm(
        self,
#        st: State,
#        physics_data: PhysicsData,
#        parameters: Parameters,
#        boundaries: BoundaryData=None,
#        geometry: Geometry=None
    ) -> tuple[State, PhysicsData]:
        
        ev = self.ev
        st = self.st
        
        physics_data = ev.physics_data
        parameters   = ev.parameters
        boundaries   = ev.boundaries
        
        day = physics_data.date.model_day()
        # Run the ocn model if the flags is switched on
        if boundaries.ocn_coupling_flag:
            
            print("Run!") 
            #st.sst = st.sst.at[:].set(1)

            print(parameters.slabocean_model)
            #print(hash(st.sst))
            #print(st.sst)
            sst, sic = run(
                st.sst,
                st.sic,
                st.d_o,
                physics_data.surface_flux.hfluxn, # net downward heat flux
                parameters.slabocean_model.dt,
                parameters.slabocean_model.tau0,
                boundaries.sst_clim[:, :, day],
                boundaries.sst_clim[:, :, day+1],
                boundaries.sic_clim[:, :, day],
                boundaries.sic_clim[:, :, day+1],
            )
        
         
        # Otherwise get from climatology
        else:
            sst = boundaries.sst_clim[:, :, day+1]
            sic = boundaries.sst_clim[:, :, day+1]
            
        # update physics data
        #slabocean_model_data = physics_data.slabocean_model.copy(sst=sst, si=si)
        physics_data = physics_data.copy(slabocean_model=physics_data)
        self.st = st.copy(sst=sst, sic=sic)
        #physics_tendency = PhysicsTendency.zeros(st.temperature.shape)
        
        #return physics_tendency, physics_data
        return physics_data
        
        
#Integrates slab land-surface model for one day.
@jit
def run(
    sst,
    sic,
    d_o,
    hfluxn,
    dt,
    tau0,
    sst_clim_0,
    sst_clim_1,
    sic_clim_0,
    sic_clim_1,
):
    
    sst_anom = sst - sst_clim_0
    sic_anom = sic - sic_clim_0
 
    factor = 1.0 + dt / tau0 
    sst_anom = sst_anom / factor +  dt * hfluxn[:, :, 0] / ( factor * physical_constants.cp_sw * physical_constants.rho_sw * d_o )
    
    new_sst = sst_anom + sst_clim_1
    new_sic = sic_anom + sic_clim_1
    
    return new_sst, new_sic




