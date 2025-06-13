"""
Date: 2025/06/13
"""

from collections import abc
import jax.numpy as jnp
import tree_math
from typing import Callable
from jcm.physics_data import PhysicsData
from jcm.geometry import Geometry
from jcm.params import Parameters
from dinosaur import scales
from dinosaur.scales import units
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, compute_diagnostic_state, State, PrimitiveEquations
from jax import tree_util
from jcm.boundaries import BoundaryData
from jcm.physical_constants import p0

@tree_math.struct
class PhysicsState:

    sst_anom: jnp.ndarray
    si_anom:  jnp.ndarray
    d_o:      jnp.ndarray
    
    @classmethod
    def zeros(self, shape,
        sst_anom = None,
        si_anom  = None,
        d_o      = None,
    ):
        return PhysicsState(
            sst_anom if sst_anom is not None else jnp.zeros(shape),
            si_anom if si_anom is not None else jnp.zeros(shape),
            d_o if d_o is not None else jnp.zeros(shape),
        )

    @classmethod
    def ones(self, shape,
        sst_anom = None,
        si_anom  = None,
        d_o = None,
    ):
        
        return PhysicsState(
            sst_anom if sst_anom is not None else jnp.ones(shape),
            si_anom if si_anom is not None else jnp.ones(shape),
            d_o if d_o is not None else jnp.ones(shape),
        )

    def copy(self,
        sst_anom = None,
        si_anom  = None,
        d_o = None,
    ):
        return PhysicsState(
            sst_anom if sst_anom is not None else self.sst_anom,
            si_anom if si_anom is not None else self.si_anom,
            d_o if d_o is not None else self.d_o,
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))


