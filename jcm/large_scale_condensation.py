'''
Date: 2/11/2024
Parametrization of large-scale condensation.
'''
from jax import jit
import jax.numpy as jnp
from jcm.boundaries import BoundaryData
from jcm.physics import PhysicsTendency, PhysicsState
from jcm.physics_data import PhysicsData
from jcm.physical_constants import p0, cp, alhc, grav
from jcm.geometry import fsg, dhs

# Constants for large-scale condensation
trlsc = 4.0   # Relaxation time (in hours) for specific humidity
rhlsc = 0.9   # Maximum relative humidity threshold (at sigma=1)
drhlsc = 0.1  # Vertical range of relative humidity threshold
rhblsc = 0.95 # Relative humidity threshold for boundary layer

# Compute large-scale condensation and associated tendencies of temperature and 
# moisture
@jit
def get_large_scale_condensation_tendencies(state: PhysicsState, physics_data: PhysicsData, boundaries: BoundaryData = None):
    """
    Compute large-scale condensation and associated tendencies of temperature and moisture

    Args:
        psa: Normalized surface pressure - convection.psa
        qa: Specific humidity [g/kg] - state.specific_humidity
        qsat: Saturation specific humidity [g/kg] - humidity.qsat
        iptop: Cloud top diagnosed from precipitation due to convection and large-scale condensation conv.iptop

    Returns:
        iptop: Cloud top diagnosed from precipitation due to convection and large-scale condensation
        precls: Precipitation due to large-scale condensation
        dtlsc: Temperature tendency due to large-scale condensation
        dqlsc: Specific humidity tendency due to large-scale condensation

    """
    # 1. Initialization
    humidity = physics_data.humidity
    conv = physics_data.convection
    
    # Initialize outputs
    dtlsc = jnp.zeros_like(state.specific_humidity)
    dqlsc = jnp.zeros_like(state.specific_humidity)
    
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
    rhref = jnp.maximum(rhref, rhblsc)
    dqmax = qsmax * sig2 * rtlsc

    # Compute dqa array
    dqa = rhref[jnp.newaxis, jnp.newaxis, :] * humidity.qsat[..., :] - state.specific_humidity[..., :]

    # Calculate dqlsc and dtlsc where dqa < 0
    negative_dqa_mask = dqa < 0
    dqlsc = dqlsc.at[..., 1:].set(jnp.where(negative_dqa_mask[..., 1:], dqa[..., 1:] * rtlsc, 0.0))
    dtlsc = dtlsc.at[..., 1:].set(jnp.where(negative_dqa_mask[..., 1:], tfact * jnp.minimum(-dqlsc[..., 1:], dqmax[jnp.newaxis, jnp.newaxis, 1:] * psa2[:, :, jnp.newaxis]), 0.))

    # The +1 here is because the first element of negative_dqa_mask is not included in the argmin
    iptop = jnp.minimum(jnp.argmin(dqa[..., 1:]>=0, axis=2)+1, conv.iptop)

    # Large-scale precipitation
    pfact = dhs * prg
    precls = 0. - jnp.sum(pfact[jnp.newaxis, jnp.newaxis, 1:] * dqlsc[..., 1:], axis=2)
    precls *= conv.psa

    condensation_out = physics_data.condensation.copy(precls=precls)
    convection_out = physics_data.convection.copy(iptop=iptop)
    physics_data = physics_data.copy(condensation=condensation_out, convection=convection_out)
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape,temperature=dtlsc, specific_humidity=dqlsc)
    
    return physics_tendencies, physics_data
