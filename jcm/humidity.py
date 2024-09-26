'''
Date: 2/11/2024
For converting between specific and relative humidity, and computing the 
saturation specific humidity.
'''

import jax.numpy as jnp
from jcm.physics import PhysicsData, PhysicsState, PhysicsTendency
from jcm.geometry import fsg
import jax 

#def spec_hum_to_rel_hum(ta, ps, sig, qa):
def spec_hum_to_rel_hum(physics_data: PhysicsData, state: PhysicsState):
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

    # vectorize get_qsat to be over all sigma levels instead of taking sig as an input - doing this will break existing tests which used to be for one sigma level at a time
    get_qsat_lambda = lambda ta, ps, fsg: get_qsat(ta, ps, fsg)
    map_qsat = jax.vmap(get_qsat_lambda, in_axes=(2, 2, 0), out_axes=2) # mapping over dim 2 for arguments ta, ps and over dim 0 (the only dim) for fsg, mapping over dim 2 of the output
    qsat = map_qsat(state.temperature, physics_data.convection.psa, fsg) #need to check that this produces ix x il x kx array

    rh = state.specific_humidity / qsat
    
    humidity_out = physics_data.humidity.copy(rh=rh, qsat=qsat)
    physics_data = physics_data.copy(humidity=humidity_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
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
    
    # If sig > 0, P = Ps * sigma, otherwise P = Ps(1) = const.
    if sig <= 0.0:
        qsat = 622.0 * qsat / (ps[0, 0] - 0.378 * qsat)
    else:
        qsat = 622.0 * qsat / (sig * ps - 0.378 * qsat)

    return qsat
