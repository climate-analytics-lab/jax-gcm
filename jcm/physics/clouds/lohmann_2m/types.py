"""Shared data structures and timestep constants for the Lohmann 2M scheme.

Split out of the monolithic ``lohmann_2m.py`` module (pure move, no
numerical change) so the process submodules can share the tendency
container and the ECHAM timestep-derived constants without cycles.
This module must not import any other module from this package.
"""

import jax.numpy as jnp
from typing import NamedTuple
from math import pi

import jcm.constants as c

from ..lohmann_2m_params import CloudParams2M


class MicrophysicsTendencies_2M(NamedTuple):
    """Tendencies from microphysics processes"""
    
    dtedt: jnp.ndarray          # Temperature tendency (K/s)
    dqdt: jnp.ndarray           # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray          # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray          # Cloud ice tendency (kg/kg/s)
    dqncdt: jnp.ndarray         # Cloud droplet number tendency (1/kg/s)
    dqnidt: jnp.ndarray         # Cloud ice crystal number tendency (1/kg/s)
    dqrdt: jnp.ndarray          # Rain water tendency (kg/kg/s)
    dqsdt: jnp.ndarray          # Snow tendency (kg/kg/s)

def microphysics_dt_constants(dt: jnp.ndarray, params: CloudParams2M) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constants that depend on the microphysics timestep. Here for consistency with ECHAM6,
    where they cannot be parameters.
    Constants are defined locally in each subroutine where needed.
    """
    ztmst = dt
    ztmst_rcp = 1.0 / jnp.maximum(ztmst, params.eps)
    zcons1 = c.cpd*c.vtmpc2
    # Match the ECHAM Fortran (mo_cloud_micro_2m.f90 line 535):
    # ``zcons2 = ztmst_rcp * rgrav = 1 / (dt * g)``. The earlier port had
    # ``ztmst * rgrav`` which was dt^2 too large in every site that uses
    # zcons2 to convert ``pdp * mmr`` into a flux (kg/m^2/s) — so the
    # large-scale surface precip diagnostic came out ~dt^2 (~5x10^5 at
    # dt=12 min) too large, and the latent heat in melt/sub paths was
    # similarly mis-scaled.
    zcons2 = ztmst_rcp * c.rgrav
    zcons3 = 1.0 / ( pi*params.crhosno*params.cn0s*params.cvtfall**(1.0/1.16) )**0.25
    
    return ztmst, ztmst_rcp, zcons1, zcons2, zcons3
