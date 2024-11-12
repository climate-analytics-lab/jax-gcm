'''
Date: 2/11/2024
For converting between specific and relative humidity, and computing the 
saturation specific humidity.
'''

import jax 
import jax.numpy as jnp
from jcm.physics_data import PhysicsData
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physical_constants import cp
from jcm.geometry import fsg

def spec_hum_to_rel_hum(state: PhysicsState, physics_data: PhysicsData):
    """
    Converts specific humidity to relative humidity, and also returns saturation 
     specific humidity.

    Args:
        ta: Absolute temperature [K] - PhysicsState.temperature
        ps: Normalized pressure (p/1000 hPa) - Convection.psa
        sig: Sigma level - fsg from jcm.geometry 
        qa: Specific humidity - PhysicsState.specific_humidity

    Returns:
        rh: Relative humidity
        qsat: Saturation specific humidity
    """

    # compute thermodynamic variables: logic from physics.f90:110-114
    psa = state.surface_pressure
    se = cp * state.temperature + state.geopotential
    convection_out = physics_data.convection.copy(psa=psa, se=se)
    
    # spec_hum_to_rel_hum logic
    map_qsat = jax.vmap(get_qsat, in_axes=(2, jnp.newaxis, 0), out_axes=2) # map over each input's z-axis and output to z-axis
    qsat = map_qsat(state.temperature, psa, fsg)
    rh = state.specific_humidity / qsat
    humidity_out = physics_data.humidity.copy(rh=rh, qsat=qsat)

    physics_data = physics_data.copy(convection=convection_out, humidity=humidity_out)
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)
    
    return physics_tendencies, physics_data


def rel_hum_to_spec_hum(ta, ps, sig, rh):
    """
    Converts relative humidity to specific humidity, and also returns saturation 
    specific humidity.

    Args:
        ta: Absolute temperature 
        ps: Normalized pressure (p/1000 hPa)
        sig: Sigma level
        rh: Relative humidity

    Returns:
        qa: Specific humidity
        qsat: Saturation specific humidity
    """
    qsat = get_qsat(ta, ps, sig)
    qa = rh * qsat
    return qa, qsat


def get_qsat(ta, ps, sig):
    """
    Computes saturation specific humidity.
    
    Args:
        ta: Absolute temperature [K]
        ps: Normalized pressure (p/1000 hPa)
        sig: Sigma level
        
    Returns:
        qsat: Saturation specific humidity (g/kg)
    """
    
    e0 = 6.108e-3
    c1 = 17.269
    c2 = 21.875
    t0 = 273.16
    t1 = 35.86
    t2 = 7.66

    # Computing qsat for each grid point
    # 1. Compute Qsat (g/kg) from T (degK) and normalized pres. P (= p/1000_hPa)
    
    qsat = jnp.where(ta >= t0, e0 * jnp.exp(c1 * (ta - t0) / (ta - t1)), 
                      e0 * jnp.exp(c2 * (ta - t0) / (ta - t2)))
    
    # If sig > 0, P = Ps * sigma, otherwise P = Ps(1) = const
    qsat = jnp.where(sig <= 0.0, 622.0 * qsat / (ps[0,0] - 0.378 * qsat), 
                      622.0 * qsat / (sig * ps - 0.378 * qsat))

    return qsat
