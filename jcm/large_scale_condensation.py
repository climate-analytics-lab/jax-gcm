'''
Date: 2/11/2024
Parametrization of large-scale condensation.
'''

import jax.numpy as jnp
from jcm.physics import PhysicsData, PhysicsTendency, PhysicsState
from jcm.convection import ConvectionData
from jcm.physical_constants import p0, cp, alhc, grav
from jcm.geometry import fsg, dhs
import tree_math
from jcm.params import ix, il, kx

# Constants for large-scale condensation
trlsc = 4.0   # Relaxation time (in hours) for specific humidity
rhlsc = 0.9   # Maximum relative humidity threshold (at sigma=1)
drhlsc = 0.1  # Vertical range of relative humidity threshold
rhblsc = 0.95 # Relative humidity threshold for boundary layer

@tree_math.struct
class CondensationData:
    precls = jnp.zeros((ix,il))
    dtlsc = jnp.zeros((ix,il,kx))
    dqlsc = jnp.zeros((ix,il,kx))

# Compute large-scale condensation and associated tendencies of temperature and 
# moisture
def get_large_scale_condensation_tendencies(physics_data: PhysicsData, state: PhysicsState):
    humidity = physics_data.humidity
    conv = physics_data.convection

    ix, il, _ = humidity.qa.shape

    # 1. Initialization

    # Initialize outputs
    dtlsc = jnp.zeros_like(humidity.qa)
    dqlsc = jnp.zeros_like(humidity.qa)
    precls = jnp.zeros((ix, il))

    # Constants for computation
    qsmax = 10.0
    rtlsc = 1.0 / (trlsc * 3600.0)
    tfact = alhc / cp
    prg = p0 / grav

    psa2 = conv.psa ** 2.0

    # Tendencies of temperature and moisture
    # NB. A maximum heating rate is imposed to avoid grid-point-storm 
    # instability
    
    # Compute sig2, rhref, and dqmax arrays
    sig2 = fsg**2.0
    rhref = rhlsc + drhlsc * (sig2 - 1.0)
    rhref = rhref.at[-1].set(jnp.maximum(rhref[-1], rhblsc))
    dqmax = qsmax * sig2 * rtlsc

    # Compute dqa array
    dqa = rhref[jnp.newaxis, jnp.newaxis, :] * humidity.qsat - humidity.qa

    # Calculate dqlsc and dtlsc where dqa < 0
    negative_dqa_mask = dqa < 0
    dqlsc = jnp.where(negative_dqa_mask, dqa * rtlsc, dqlsc)
    dtlsc = jnp.where(negative_dqa_mask, tfact * jnp.minimum(-dqlsc, dqmax[jnp.newaxis, jnp.newaxis, :] * psa2[:, :, jnp.newaxis]), dtlsc)

    # Update itop
    def update_iptop(iptop, indices, values):
        for idx, val in zip(zip(*indices), values):
            iptop = iptop.at[idx[:2]].set(jnp.minimum(iptop[idx[:2]], val)) # should this be iptop=? or can we just use iptop.at[idx[:2]].set(val)?
        return iptop

    iptop_update_indices = jnp.where(negative_dqa_mask)
    iptop = update_iptop(conv.iptop, iptop_update_indices, iptop_update_indices[2])

    # Large-scale precipitation
    pfact = dhs * prg
    precls -= jnp.sum(pfact[jnp.newaxis, jnp.newaxis, :] * dqlsc, axis=2)
    precls *= conv.psa

    condensation_out = CondensationData(precls, dtlsc, dqlsc)   
    conv_out = ConvectionData()
    conv_out = conv 
    conv_out.iptop = iptop

    physics_data = PhysicsData(physics_data.shortwave_rad, conv_out, physics_data.modradcon, physics_data.humidity, condensation_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_tendencies, physics_data
