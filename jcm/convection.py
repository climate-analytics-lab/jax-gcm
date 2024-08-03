'''
Date: 2/11/2024
Parametrization of convection. Convection is modelled using a simplified 
version of the Tiedke (1993) mass-flux convection scheme.
'''

import jax.numpy as jnp

from jcm.physical_constants import p0, alhc, wvi, grav
from jcm.geometry import dhs, fsg


psmin = jnp.array(0.8) # Minimum (normalised) surface pressure for the occurrence of convection
trcnv = jnp.array(6.0) # Time of relaxation (in hours) towards reference state
rhil = jnp.array(0.7) # Relative humidity threshold in intermeduate layers for secondary mass flux
rhbl = jnp.array(0.9) # Relative humidity threshold in the boundary layer
entmax = jnp.array(0.5) # Maximum entrainment as a fraction of cloud-base mass flux
smf = jnp.array(0.8) # Ratio between secondary and primary mass flux at cloud-base

def diagnose_convection(psa, se, qa, qsat):
    """
    Diagnose convectively unstable gridboxes  

    Convection is activated in gridboxes with conditional instability. This
    is diagnosed by checking for any tropopsheric half level where the
    saturation moist static energy is lower than in the boundary-layer level.
    In gridboxes where this is true, convection is activated if either: there
    is convective instability - the actual moist static energy at the
    tropospheric level is lower than in the boundary-layer level, or, the
    relative humidity in the boundary-layer level and lowest tropospheric
    level exceed a set threshold (rhbl).

    Args:
    psa: Normalised surface pressure [p/p0]
    se: Dry static energy [c_p.T + g.z]
    qa: Specific humidity [g/kg]
    qsat: Saturation specific humidity [g/kg]

    Returns:
    itop: Top of convection (layer index)
    qdif: Excess humidity in convective gridboxes

    """
    ix, il, kx = se.shape
    itop = jnp.full((ix, il), kx + 1, dtype=int)  # Initialize itop with nlp
    qdif = jnp.zeros((ix, il), dtype=float)

    # Saturation moist static energy
    mss = se + alhc * qsat

    rlhc = 1.0 / alhc

    # Minimum of moist static energy in the lowest two levels
    # Mask for psa > psmin
    mask_psa = psa > psmin 

    mse0 = jnp.where(mask_psa, 0, se[:, :, kx-1] + alhc * qa[:, :, kx-1]) #se[:, :, kx-1] + alhc * qa[:, :, kx-1]
    mse1 = jnp.where(mask_psa, 0, se[:, :, kx-2] + alhc * qa[:, :, kx-2]) #se[:, :, kx-2] + alhc * qa[:, :, kx-2]
    mse1 = jnp.minimum(mse0, mse1)

    # Saturation (or super-saturated) moist static energy in PBL
    mss0 = jnp.maximum(mse0, mss[:, :, kx-1])

    # Compute mss2 array for all k layers (3 to kx-3)
    k_indices = jnp.arange(3, kx-3, dtype=int)
    mss2 = mss[:, :, k_indices] + wvi[k_indices, 1] * (mss[:, :, k_indices + 1] - mss[:, :, k_indices])
    
    # Check 1: conditional instability (MSS in PBL > MSS at top level)
    mask_conditional_instability = mss0[:, :, None] > mss2
    ktop1 = jnp.full((ix, il), kx, dtype=int)
    ktop1 = k_indices[jnp.argmax(mask_conditional_instability, axis=2)]

    # Check 2: gradient of actual moist static energy between lower and upper 
    # troposphere
    mask_mse1_greater_mss2 = mse1[:, :, None] > mss2
    ktop2 = jnp.full((ix, il), kx, dtype=int)
    ktop2 = k_indices[jnp.argmax(mask_mse1_greater_mss2, axis=2)]
    msthr = jnp.zeros((ix, il), dtype=float)
    msthr = mss2[jnp.arange(ix)[:, None], jnp.arange(il), jnp.argmax(mask_mse1_greater_mss2, axis=2)]

    # Check 3: RH > RH_c at both k=kx and k=kx-1
    qthr0 = rhbl * qsat[:, :, kx-1]
    qthr1 = rhbl * qsat[:, :, kx-2]
    lqthr = (qa[:, :, kx-1] > qthr0) & (qa[:, :, kx-2] > qthr1)

    # Applying masks to itop and qdif
    mask_ktop1_less_kx = ktop1 < kx
    mask_ktop2_less_kx = ktop2 < kx

    combined_mask1 = mask_ktop1_less_kx & mask_ktop2_less_kx
    itop = jnp.where(combined_mask1, ktop1, itop)
    qdif = jnp.where(combined_mask1, jnp.maximum(qa[:, :, kx-1] - qthr0, (mse0 - msthr) * rlhc), qdif)

    combined_mask2 = mask_ktop1_less_kx & lqthr & ~combined_mask1
    itop = jnp.where(combined_mask2, ktop1, itop)
    qdif = jnp.where(combined_mask2, qa[:, :, kx-1] - qthr0, qdif)

    return itop, qdif

def get_convection_tendencies(psa, se, qa, qsat):
    """
    Compute convective fluxes of dry static energy and moisture using a simplified mass-flux scheme.

    Args:
    psa: Normalised surface pressure [p/p0]
    se: Dry static energy [c_p.T + g.z]
    qa: Specific humidity [g/kg]
    qsat: Saturation specific humidity [g/kg]

    Returns:
    itop: Top of convection (layer index)
    cbmf: Cloud-base mass flux
    precnv: Convective precipitation [g/(m^2 s)]
    dfse:  Net flux of dry static energy into each atmospheric layer
    dfqa: Net flux of specific humidity into each atmospheric layer

    """
    _, _, kx = se.shape

    # 1. Initialization of output and workspace arrays

    # Entrainment profile (up to sigma = 0.5)
    entr = jnp.maximum(0.0, fsg[1:kx-1] - 0.5)**2.0
    sentr = jnp.sum(entr)
    entr *= entmax / sentr

    # 2. Check of conditions for convection
    itop, qdif = diagnose_convection(psa, se, qa, qsat)

    # 3. Convection over selected grid-points
    # 3.1 Boundary layer (cloud base)
    # Maximum specific humidity in the PBL
    mask = itop < kx
    qmax = jnp.maximum(1.01 * qa[:, :, -1], qsat[:, :, -1])

    # Dry static energy and moisture at upper boundary
    sb = se[:, :, -2] + wvi[-2, 1] * (se[:, :, -1] - se[:, :, -2])
    qb = jnp.minimum(qa[:, :, -2] + wvi[-2, 1] * (qa[:, :, -1] - qa[:, :, -2]), qa[:, :, -1])

    # Cloud-base mass flux, computed to satisfy:
    # fmass*(qmax-qb)*(g/dp)=qdif/trcnv
    fqmax = 5.0
    fm0 = p0 * dhs[-1] / (grav * trcnv * 3600.0)
    rdps = 2.0 / (1.0 - psmin)

    fpsa = psa * jnp.minimum(1.0, (psa - psmin) * rdps)
    fmass = fm0 * fpsa * jnp.minimum(fqmax, qdif / (qmax - qb))
    cbmf = jnp.where(mask, fmass, cbmf)

    # Upward fluxes at upper boundary
    fus = fmass * se[:, :, -1]
    fuq = fmass * qmax

    # Downward fluxes at upper boundary
    fds = fmass * sb
    fdq = fmass * qb

    # Net flux of dry static energy and moisture
    dfse = dfse.at[:, :, -1].set(fds - fus)
    dfqa = dfqa.at[:, :, -1].set(fdq - fuq)

    # Create an array of k values to use for broadcasting
    k_vals = jnp.arange(kx-2, 0, -1)

    # Initialize fmass, fus, and fuq arrays for broadcasting
    fmass_broadcast = jnp.tile(fmass[:, :, jnp.newaxis], (1, 1, len(k_vals)))
    fus_broadcast = jnp.tile(fus[:, :, jnp.newaxis], (1, 1, len(k_vals)))
    fuq_broadcast = jnp.tile(fuq[:, :, jnp.newaxis], (1, 1, len(k_vals)))

    # Calculate sb and qb for each layer in the loop using broadcasting
    sb_vals = se[:, :, k_vals-1] + wvi[k_vals-1, 1] * (se[:, :, k_vals] - se[:, :, k_vals-1])
    qb_vals = qa[:, :, k_vals-1] + wvi[k_vals-1, 1] * (qa[:, :, k_vals] - qa[:, :, k_vals-1])

    # Mass entrainment
    enmass = entr[k_vals-1] * psa[:, :, jnp.newaxis] * cbmf[:, :, jnp.newaxis]

    # Upward fluxes at upper boundary
    fmass_broadcast += enmass
    fus_broadcast += enmass * se[:, :, k_vals]
    fuq_broadcast += enmass * qa[:, :, k_vals]

    # Downward fluxes at upper boundary
    fds_vals = fmass_broadcast * sb_vals
    fdq_vals = fmass_broadcast * qb_vals

    # Net flux of dry static energy and moisture
    dfse = dfse.at[:, :, k_vals].set(fus_broadcast - fds_vals)
    dfqa = dfqa.at[:, :, k_vals].set(fuq_broadcast - fdq_vals)

    # Secondary moisture flux
    delq_vals = rhil * qsat[:, :, k_vals] - qa[:, :, k_vals]
    fsq_vals = jnp.where(delq_vals > 0, smf * cbmf[:, :, jnp.newaxis] * delq_vals, 0.0)

    dfqa = dfqa.at[:, :, k_vals].add(fsq_vals)
    dfqa = dfqa.at[:, :, -1].add(-jnp.sum(fsq_vals, axis=-1))

    # 3.3 Top layer (condensation and detrainment)
    k = itop

    # Flux of convective precipitation
    qsatb = qsat[:, :, k] + wvi[k, 1] *(qsat[:, :, k+1]-qsat[:, :, k])
    precnv = jnp.where(mask, jnp.maximum(fuq - fmass * qsatb, 0.0), precnv)

    # Net flux of dry static energy and moisture
    dfse = fus - fds + alhc * precnv
    dfqa = fuq - fdq - precnv

    return itop, dfse, dfqa, cbmf, precnv