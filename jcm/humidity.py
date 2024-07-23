'''
Date: 2/11/2024
For converting between specific and relative humidity, and computing the 
saturation specific humidity.
'''

import jax.numpy as jnp

# Converts specific humidity to relative humidity, and also returns saturation 
# specific humidity.
def spec_hum_to_rel_hum(ta, ps, sig, qa):
    qsat = get_qsat(ta, ps, sig)
    rh = qa / qsat
    return rh, qsat

# Converts relative humidity to specific humidity, and also returns saturation 
# specific humidity.
def rel_hum_to_spec_hum(ta, ps, sig, rh):
    qsat = get_qsat(ta, ps, sig)
    qa = rh * qsat
    return qa, qsat

# Computes saturation specific humidity.
def get_qsat(ta, ps, sig):
    # 1. Compute Qsat (g/kg) from T (degK) and normalized pres. P (= p/1000_hPa)
    # If sig > 0, P = Ps * sigma, otherwise P = Ps(1) = const.
    e0 = 6.108e-3
    c1 = 17.269
    c2 = 21.875
    t0 = 273.16
    t1 = 35.86
    t2 = 7.66

    # Computing qsat for each grid point
    qsat = jnp.where(ta >= t0, e0 * jnp.exp(c1 * (ta - t0) / (ta - t1)), 
                      e0 * jnp.exp(c2 * (ta - t0) / (ta - t2)))

    if sig <= 0.0:
        qsat = 622.0 * qsat / (ps[0, 0] - 0.378 * qsat)
    else:
        qsat = 622.0 * qsat / (sig * ps - 0.378 * qsat)

    return qsat
