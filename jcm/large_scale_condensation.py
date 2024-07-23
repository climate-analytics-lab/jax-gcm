'''
Date: 2/11/2024
Parametrization of large-scale condensation.
'''

import jax.numpy as jnp

from physical_constants import p0, cp, alhc, grav
from geometry import fsg, dhs

# Constants for large-scale condensation
trlsc = 4.0   # Relaxation time (in hours) for specific humidity
rhlsc = 0.9   # Maximum relative humidity threshold (at sigma=1)
drhlsc = 0.1  # Vertical range of relative humidity threshold
rhblsc = 0.95 # Relative humidity threshold for boundary layer

# Compute large-scale condensation and associated tendencies of temperature and 
# moisture
def get_large_scale_condensation_tendencies(psa, qa, qsat, itop):
    ix, il, _ = qa.shape

    # 1. Initialization

    # Initialize outputs
    dtlsc = jnp.zeros_like(qa)
    dqlsc = jnp.zeros_like(qa)
    precls = jnp.zeros((ix, il))

    # Constants for computation
    qsmax = 10.0
    rtlsc = 1.0 / (trlsc * 3600.0)
    tfact = alhc / cp
    prg = p0 / grav

    psa2 = psa ** 2.0

    # Tendencies of temperature and moisture
    # NB. A maximum heating rate is imposed to avoid grid-point-storm 
    # instability
    
    # Compute sig2, rhref, and dqmax arrays
    sig2 = fsg**2.0
    rhref = rhlsc + drhlsc * (sig2 - 1.0)
    rhref = rhref.at[-1].set(jnp.maximum(rhref[-1], rhblsc))
    dqmax = qsmax * sig2 * rtlsc

    # Compute dqa array
    dqa = rhref[jnp.newaxis, jnp.newaxis, :] * qsat - qa

    # Calculate dqlsc and dtlsc where dqa < 0
    negative_dqa_mask = dqa < 0
    dqlsc = jnp.where(negative_dqa_mask, dqa * rtlsc, dqlsc)
    dtlsc = jnp.where(negative_dqa_mask, tfact * jnp.minimum(-dqlsc, dqmax[jnp.newaxis, jnp.newaxis, :] * psa2[:, :, jnp.newaxis]), dtlsc)

    # Update itop
    def update_itop(itop, indices, values):
        for idx, val in zip(zip(*indices), values):
            itop = itop.at[idx[:2]].set(jnp.minimum(itop[idx[:2]], val))
        return itop

    itop_update_indices = jnp.where(negative_dqa_mask)
    itop = update_itop(itop, itop_update_indices, itop_update_indices[2])

    # Large-scale precipitation
    pfact = dhs * prg
    precls -= jnp.sum(pfact[jnp.newaxis, jnp.newaxis, :] * dqlsc, axis=2)
    precls *= psa

    return itop, precls, dtlsc, dqlsc
