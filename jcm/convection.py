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
    k_indices = jnp.arange(2, kx-3, dtype=int)
    mss2 = mss[:, :, k_indices] + wvi[k_indices, 1] * (mss[:, :, k_indices + 1] - mss[:, :, k_indices])

    # Check 1: conditional instability (MSS in PBL > MSS at top level)
    mask_conditional_instability = mss0[:, :, jnp.newaxis] > mss2
    ktop1 = jnp.full((ix, il), kx, dtype=int)
    ktop1 = (k_indices+1)[jnp.argmax(mask_conditional_instability, axis=2)]

    # Check 2: gradient of actual moist static energy between lower and upper 
    # troposphere
    mask_mse1_greater_mss2 = mse1[:, :, jnp.newaxis] > mss2
    ktop2 = jnp.full((ix, il), kx, dtype=int)
    ktop2 = (k_indices+1)[jnp.argmax(mask_mse1_greater_mss2, axis=2)]
    msthr = jnp.zeros((ix, il), dtype=float)
    msthr = mss2[jnp.arange(ix)[:, jnp.newaxis], jnp.arange(il), jnp.argmax(mask_mse1_greater_mss2, axis=2)]

    # Check 3: RH > RH_c at both k=kx and k=kx-1
    qthr0 = rhbl * qsat[:, :, kx-1]
    qthr1 = rhbl * qsat[:, :, kx-2]
    lqthr = (qa[:, :, kx-1] > qthr0) & (qa[:, :, kx-2] > qthr1)

    # Applying masks to iptop and qdif
    mask_ktop1_less_kx = ktop1 < kx
    mask_ktop2_less_kx = ktop2 < kx

    combined_mask1 = mask_ktop1_less_kx & mask_ktop2_less_kx
    iptop = jnp.where(combined_mask1, ktop1, iptop)
    qdif = jnp.where(combined_mask1, jnp.maximum(qa[:, :, kx-1] - qthr0, (mse0 - msthr) * rlhc), qdif)

    combined_mask2 = mask_ktop1_less_kx & (~mask_ktop2_less_kx) & lqthr
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
    se = conv.se
    qa = state.specific_humidity
    qsat = physics_data.humidity.qsat
    ix, il, kx = se.shape
    _zeros_3d = lambda: jnp.zeros((ix,il,kx))
    psa = conv.psa
    
    # 1. Initialization of output and workspace arrays

    dfse, dfqa = _zeros_3d(), _zeros_3d()

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
    iptop, qdif = diagnose_convection(psa, se, qa, qsat)

    # 3. Convection over selected grid-points
    mask = iptop < kx
    # 3.1 Boundary layer (cloud base)
    k = kx - 1

    # Maximum specific humidity in the PBL
    qmax = jnp.maximum(1.01 * qa[:, :, -1], qsat[:, :, -1])

    interpolate = lambda tracer: tracer[:, :, :-1] + wvi[jnp.newaxis, jnp.newaxis, :-1, 1] * jnp.diff(tracer, axis=-1)
    _sb_3d, _qb_3d = (_zeros_3d().at[:, :, 1:].set(interpolate(tracer)) for tracer in (se, qa))
    
    # Dry static energy and moisture at upper boundary
    sb, qb = _sb_3d[:, :, k], jnp.minimum(_qb_3d, qa)[:, :, k]
    
    # Cloud-base mass flux
    fpsa = psa * jnp.minimum(1.0, (psa - psmin) * rdps)
    fmass = fm0 * fpsa * jnp.minimum(fqmax, qdif / (qmax - qb))
    cbmf = mask * fmass

    # Upward fluxes at upper boundary
    fus, fuq = fmass * se[:, :, k], fmass * qmax

    # Downward fluxes at upper boundary
    fds, fdq = fmass * sb, fmass * qb

    # Net flux of dry static energy and moisture
    dfse, dfqa = dfse.at[:, :, k].set(fds - fus), dfqa.at[:, :, k].set(fdq - fuq)

    # 3.2 Intermediate layers (entrainment)

    # replace loop with masking
    loop_mask = (kx - 2 >= jnp.arange(kx)[jnp.newaxis, jnp.newaxis, :]) & \
                (jnp.arange(kx)[jnp.newaxis, jnp.newaxis, :] >= iptop[:, :, jnp.newaxis])
    
    #start by making entrainment profile:
    _enmass_3d = loop_mask * _zeros_3d().at[:, :, 1:-1].set(entr[jnp.newaxis, jnp.newaxis, :] * (psa * cbmf)[:, :, jnp.newaxis])

    # Upward fluxes at upper boundary of mass, energy, moisture
    _fmass_3d, _fus_3d, _fuq_3d = (
        base_flux[:, :, jnp.newaxis] + jnp.cumsum((_enmass_3d * tracer)[:, :, ::-1], axis=-1)[:, :, ::-1]
        for base_flux, tracer in ((fmass, 1), (fus, se), (fuq, qa))
    )

    #Downward fluxes
    _fds_3d, _fdq_3d = (_fmass_3d * _sb_3d).at[:, :, -1].set(fds), (_fmass_3d * _qb_3d).at[:, :, -1].set(fdq)

    # Calculate flux convergence
    dfse = dfse.at[:, :, :-1].add(loop_mask[:, :, :-1] * (jnp.diff(_fus_3d - _fds_3d, axis=-1)))
    dfqa = dfqa.at[:, :, :-1].add(loop_mask[:, :, :-1] * (jnp.diff(_fuq_3d - _fdq_3d, axis=-1)))

    # Secondary moisture flux
    delq = loop_mask * (rhil * qsat - qa)
    moisture_flux_mask = delq > 0.
    fsq_masked = moisture_flux_mask * smf * cbmf[:, :, jnp.newaxis] * delq
    dfqa += fsq_masked
    dfqa = dfqa.at[:, :, -1].add(-jnp.sum(fsq_masked, axis=-1))

    # assuming that take_along_axis is at least as well-optimized as any workaround via masking
    index_array = lambda array, index: jnp.squeeze(jnp.take_along_axis(array, index[:, :, jnp.newaxis], axis=-1), axis=-1)
    fmass, fus, fuq, fds, fdq = (index_array(_flux_3d, iptop) for _flux_3d in (_fmass_3d, _fus_3d, _fuq_3d, _fds_3d, _fdq_3d))
    
    # 3.3 Top layer (condensation and detrainment)
    k = iptop - 1

    # Flux of convective precipitation
    qsatb = index_array(interpolate(qsat), k)
    precnv = jnp.maximum(fuq - fmass * qsatb, 0.0)

    # Net flux of dry static energy and moisture
    dfse = dfse.at[:, :, k].set(fus - fds + alhc * precnv)
    dfqa = dfqa.at[:, :, k].set(fuq - fdq - precnv)

    # convection in Speedy generates net *flux* -- not tendencies, so we convert dfse and dfqa to tendencies here
    # Another important note is that this goes from 2:kx in the fortran

    # Compute tendencies due to convection. Logic from physics.f90:127-130
    rps = 1/psa
    ttend = dfse.at[:,:,1:].set(dfse[:,:,1:] * rps[:,:,jnp.newaxis] * grdscp[jnp.newaxis, jnp.newaxis, 1:])
    qtend = dfqa.at[:,:,1:].set(dfqa[:,:,1:] * rps[:,:,jnp.newaxis] * grdsig[jnp.newaxis, jnp.newaxis, 1:])

    convection_out = physics_data.convection.copy(convection=conv,psa=psa, se=se, iptop=iptop, cbmf=cbmf, precnv=precnv)
    physics_data = physics_data.copy(phys_data=physics_data,convection=convection_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),ttend,qtend)
    
    return physics_tendencies, physics_data