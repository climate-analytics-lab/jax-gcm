'''
Date: 2/11/2024
For converting between specific and relative humidity, and computing the 
saturation specific humidity.
'''

import jax.numpy as jnp


def spec_hum_to_rel_hum(ta, ps, sig, qa):
    """
    Converts specific humidity to relative humidity, and also returns saturation 
     specific humidity.

    Args:
        ta: Absolute temperature [K]
        ps: Normalized pressure (p/1000 hPa)
        sig: Sigma level
        qa: Specific humidity

    Returns:
        rh: Relative humidity
        qsat: Saturation specific humidity
    """

    qsat = get_qsat(ta, ps, sig)
    rh = qa / qsat
    return rh, qsat


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
