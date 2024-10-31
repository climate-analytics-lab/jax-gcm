'''
Date: 2/11/2024
Parametrization of convection. Convection is modelled using a simplified 
version of the Tiedke (1993) mass-flux convection scheme.
'''
import jax
import jax.numpy as jnp
from jcm.physics import PhysicsTendency, PhysicsState
from jcm.physics_data import PhysicsData
from jcm.physical_constants import p0, alhc, wvi, grav, grdscp, grdsig
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
    iptop: Top of convection (layer index)
    qdif: Excess humidity in convective gridboxes

    """
    ix, il, kx = se.shape
    iptop = jnp.full((ix, il), kx + 1, dtype=int)  # Initialize iptop with nlp
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

    mask_ktop1_less_kx = ktop1 < kx    
    # Check 3: RH > RH_c at both k=kx and k=kx-1
    qthr0 = mask_ktop1_less_kx * rhbl * qsat[:, :, kx-1]
    qthr1 = mask_ktop1_less_kx * rhbl * qsat[:, :, kx-2]
    lqthr = jnp.where(mask_ktop1_less_kx, (qa[:, :, kx-1] > qthr0) & (qa[:, :, kx-2] > qthr1), lqthr)

    mask_ktop2_less_kx = ktop2 < kx
    combined_mask1 = mask_ktop1_less_kx & mask_ktop2_less_kx
    iptop = jnp.where(combined_mask1, ktop1, iptop)
    qdif = jnp.where(combined_mask1, jnp.maximum(qa[:, :, kx-1] - qthr0, (mse0 - msthr) * rlhc), qdif)

    combined_mask2 = mask_ktop1_less_kx & lqthr & ~combined_mask1
    iptop = jnp.where(combined_mask2, ktop1, iptop)
    qdif = jnp.where(combined_mask2, qa[:, :, kx-1] - qthr0, qdif)

    return iptop, qdif

def get_convection_tendencies(state: PhysicsState, physics_data: PhysicsData):
    """
    Compute convective fluxes of dry static energy and moisture using a simplified mass-flux scheme.

    Args:
    psa: Normalised surface pressure [p/p0] 
    se: Dry static energy [c_p.T + g.z]
    qa: Specific humidity [g/kg] - state.specific_humidity
    qsat: Saturation specific humidity [g/kg] - humidity.qsat

    Returns:
    iptop: Top of convection (layer index)
    cbmf: Cloud-base mass flux
    precnv: Convective precipitation [g/(m^2 s)]
    dfse:  Net flux of dry static energy into each atmospheric layer
    dfqa: Net flux of specific humidity into each atmospheric layer

    """
    conv = physics_data.convection
    humidity = physics_data.humidity
    se = conv.se
    qa = state.specific_humidity
    qsat = humidity.qsat
    ix, il, kx = se.shape
    psa = conv.psa
    
    # 1. Initialization of output and workspace arrays

    dfse = jnp.zeros_like(se)
    dfqa = jnp.zeros_like(state.specific_humidity)

    cbmf = jnp.zeros_like(psa)
    precnv = jnp.zeros_like(psa)

    #keep indexing consistent with original Speedy
    nl1 = kx - 1 
    nlp = kx + 1

    # Entrainment profile (up to sigma = 0.5)
    entr = jnp.maximum(0.0, fsg[1:kx-1] - 0.5)**2.0
    sentr = jnp.sum(entr)
    entr *= entmax / sentr

    fqmax = 5.0 #maximum mass flux, not sure why this is needed
    fm0 = p0*dhs[-1]/(grav*trcnv*3600.0) #prefactor for mass fluxes
    rdps=2.0/(1.0 - psmin)

    # 2. Check of conditions for convection
    iptop, qdif = diagnose_convection(psa, se, state.specific_humidity, humidity.qsat)

    # 3. Convection over selected grid-points
    mask = iptop < kx
    #3.1 Boundary layer (cloud base)
    k = kx - 1
    k1 = k - 1

    # Maximum specific humidity in the PBL
    qmax = jnp.maximum(1.01 * state.specific_humidity[:, :, -1], humidity.qsat[:, :, -1])

    # Dry static energy and moisture at upper boundary
    sb = se[:, :, k1] + wvi[k1, 1] * (se[:, :, k] - se[:, :, k1])
    qb = jnp.minimum(state.specific_humidity[:, :, k1] + wvi[k1, 1] * (state.specific_humidity[:, :, k] - state.specific_humidity[:, :, k1]), state.specific_humidity[:, :, k])
    
    # Cloud-base mass flux
    fpsa = psa * jnp.minimum(1.0, (psa - psmin) * rdps)
    fmass = fm0 * fpsa * jnp.minimum(fqmax, qdif / (qmax - qb))
    cbmf = jnp.where(mask, fmass, cbmf)

    # Upward fluxes at upper boundary
    fus = fmass * se[:, :, -1]
    fuq = fmass * qmax

    # Downward fluxes at upper boundary
    fds = fmass*sb
    fdq = fmass*qb

    #Net flux of dry static energy and moisture
    dfse = dfse.at[:, :, k].set(fds - fus)
    dfqa = dfqa.at[:, :, k].set(fdq - fuq)

    # 3.2 intermediate layers (entrainment)
    # Loop runs on reversed(range(iptop, kx-1)) but we slice only as necessary to keep indices within bounds, and use loop_mask to restrict the logic
    loop_mask = (jnp.arange(kx)[jnp.newaxis, jnp.newaxis, :] >= iptop[:, :, jnp.newaxis]) and (jnp.arange(kx)[jnp.newaxis, jnp.newaxis, :] < kx-1)

    # Loop body: the only thing reordered from the f90 is that fluxes at lower boundary are now computed later

    # Mass entrainment (fmass) and upward flux at upper boundary (fus, fuq) can be done first
    _enmass_array = loop_mask * entr[jnp.newaxis, jnp.newaxis, :] * psa[:, :, jnp.newaxis] * cbmf[:, :, jnp.newaxis]
    
    _fmass_array = jnp.cumsum(_enmass_array.at[:, :, -1].add(fmass)[::-1], axis=-1)[::-1]
    _fus_array = jnp.cumsum((_enmass_array*se).at[:, :, -1].add(fus)[::-1], axis=-1)[::-1]
    _fuq_array = jnp.cumsum((_enmass_array*qa).at[:, :, -1].add(fuq)[::-1], axis=-1)[::-1]

    # Downward fluxes at upper boundary can now be calculated using fmass
    sb = jnp.zeros_like(loop_mask).at[:, :, 1:].set(se[:, :, :-1] + wvi[jnp.newaxis, jnp.newaxis, :-1, 1] * (se[:, :, 1:] - se[:, :, :-1]))
    qb = jnp.zeros_like(loop_mask).at[:, :, 1:].set(qa[:, :, :-1] + wvi[jnp.newaxis, jnp.newaxis, :-1, 1] * (qa[:, :, 1:] - qa[:, :, :-1]))
    _fds_array = (fmass * sb).at[:, :, -1].set(fds)
    _fdq_array = (fmass * qb).at[:, :, -1].set(fdq)

    # With fus, fds, fuq, fdq we can calculate dfse and dfqa.

    # Fluxes at lower boundary: use index offset to access values corresponding to start of f90 loop body
    dfse = dfse.at[:, :, :-1].set(jnp.where(loop_mask[:, :, :-1], _fus_array[:, :, 1:] - _fds_array[:, :, 1:], dfse[:, :, :-1]))
    dfqa = dfqa.at[:, :, :-1].set(jnp.where(loop_mask[:, :, :-1], _fuq_array[:, :, 1:] - _fdq_array[:, :, 1:], dfqa[:, :, :-1]))

    #Net flux of dry static energy and moisture
    dfse = dfse.add(jnp.where(loop_mask, _fds_array - _fus_array, 0))
    dfqa = dfqa.add(jnp.where(loop_mask, _fdq_array - _fuq_array, 0))

    # Secondary moisture flux
    delq = rhil * qsat - qa
    fsq = smf * cbmf[:, :, jnp.newaxis] * delq
    secondary_moisture_flux_mask = loop_mask and (delq > 0.)
    dfqa = dfqa.add(jnp.where(secondary_moisture_flux_mask, fsq, 0))
    dfqa = dfqa.at[:, :, -1].add(-1 * jnp.sum(secondary_moisture_flux_mask * fsq, axis=-1))

    if iptop < kx - 1: # Only update these fields if they would have been updated in the fortran
        fmass = _fmass_array[:, :, iptop]
        fus, fuq, fds, fdq = _fus_array[:, :, iptop], _fuq_array[:, :, iptop], _fds_array[:, :, iptop], _fdq_array[:, :, iptop]

    # 3.3 Top layer (condensation and detrainment)
    k = iptop - 1

    # Flux of convective precipitation
    qsatb = qsat[:, :, k] + wvi[jnp.newaxis, jnp.newaxis, k, 1] * (qsat[:, :, k + 1] - qsat[:, :, k])
    precnv = jnp.maximum(fuq - fmass * qsatb, 0.0)

    # Net flux of dry static energy and moisture
    dfse = dfse.at[:, :, k].set(fus - fds + alhc * precnv)
    dfqa = dfqa.at[:, :, k].set(fuq - fdq - precnv)

    # make a new physics_data struct. overwrite the appropriate convection bits that were calculated in this function
    # pass on the rest of physics_data that was not updated or needed in this function
    # convection in Speedy generates net *flux* -- not tendencies, so we convert dfse and dfqa to tendencies here
    # Another important note is that this goes from 2:kx in the fortran

    rps = 1/psa 
    ttend = dfse 
    qtend = dfqa
    ttend = ttend.at[:,:,1:].set(dfse[:,:,1:] * rps[:,:,jnp.newaxis] * grdscp[jnp.newaxis, jnp.newaxis, 1:])
    qtend = qtend.at[:,:,1:].set(dfqa[:,:,1:] * rps[:,:,jnp.newaxis] * grdsig[jnp.newaxis, jnp.newaxis, 1:])

    convection_out = physics_data.convection.copy(psa=psa, se=se, iptop=iptop, cbmf=cbmf, precnv=precnv)
    physics_data = physics_data.copy(convection=convection_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),ttend,qtend)
    
    return physics_tendencies, physics_data