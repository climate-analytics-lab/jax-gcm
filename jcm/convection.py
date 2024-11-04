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
    qa: Specific humidity [g/kg] - qa
    qsat: Saturation specific humidity [g/kg] - qsat

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
    _zeros_3d = jnp.zeros_like(se)
    _zeros_2d = _zeros_3d[:, :, 0]
    _, _, kx = _zeros_3d.shape
    psa = conv.psa
    
    # 1. Initialization of output and workspace arrays

    dfse, dfqa = _zeros_3d, _zeros_3d
    cbmf, precnv = _zeros_2d, _zeros_2d

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
    k1 = k - 1

    # Maximum specific humidity in the PBL
    qmax = jnp.maximum(1.01 * qa[:, :, -1], qsat[:, :, -1])
    
    # # Dry static energy and moisture at upper boundary
    sb = se[:, :, k1] + wvi[k1, 1] * (se[:, :, k] - se[:, :, k1])
    qb = jnp.minimum(qa[:, :, k1] + wvi[k1, 1] * (qa[:, :, k] - qa[:, :, k1]), qa[:, :, k])
    
    # Cloud-base mass flux
    fpsa = psa * jnp.minimum(1.0, (psa - psmin) * rdps)
    fmass = fm0 * fpsa * jnp.minimum(fqmax, qdif / (qmax - qb))
    cbmf = jnp.where(mask, fmass, cbmf)

    # Upward fluxes at upper boundary
    fus = fmass * se[:, :, k]
    fuq = fmass * qmax

    # Downward fluxes at upper boundary
    fds = fmass*sb
    fdq = fmass*qb

    # Net flux of dry static energy and moisture
    dfse = dfse.at[:, :, k].set(fds - fus)
    dfqa = dfqa.at[:, :, k].set(fdq - fuq)
    
    # 3.2 intermediate layers (entrainment)
    # replace loop with masking
    loop_mask = (jnp.arange(1, kx+1)[jnp.newaxis, jnp.newaxis, :] >= iptop[:, :, jnp.newaxis] + 1) & (jnp.arange(1, kx+1)[jnp.newaxis, jnp.newaxis, :] <= kx - 1)
    #start by making entrainment profile:
    _enmass_3d = loop_mask*_zeros_3d.at[:,:,1:-1].set(entr[jnp.newaxis, jnp.newaxis, :] * (psa * cbmf)[:, :, jnp.newaxis])

    #now get mass entrainment
    #mass flux
    _fmass_3d = fmass[:, :, jnp.newaxis] + loop_mask*(jnp.cumsum(_enmass_3d[:, :, ::-1], axis=-1)[:, :, ::-1])

    #upwards static energy flux
    _fus_3d = fus[:, :, jnp.newaxis] + loop_mask*(jnp.cumsum((_enmass_3d*se)[:, :, ::-1], axis=-1)[:, :, ::-1])
    
    #upwards moisture flux
    _fuq_3d = fuq[:, :, jnp.newaxis] + loop_mask*(jnp.cumsum((_enmass_3d*qa)[:, :, ::-1], axis=-1)[:, :, ::-1])
    
    #Downward fluxes
    interpolate_tracer = lambda tracer_density: tracer_density[:, :, :-1] + wvi[jnp.newaxis, jnp.newaxis, :-1, 1] * (tracer_density[:, :, 1:] - tracer_density[:, :, :-1])
    _sb_3d, _qb_3d = (
        _zeros_3d.at[:, :, 1:].set(interpolate_tracer(tracer_density))
        for tracer_density in (se, qa)
    )
    _fds_3d = (_fmass_3d * _sb_3d).at[:, :, -1].set(fds)
    _fdq_3d = (_fmass_3d * _qb_3d).at[:, :, -1].set(fdq)

    # Fluxes at lower boundary
    dfse = dfse.at[:, :, :-1].set(jnp.where(loop_mask[:, :, :-1], _fus_3d[:, :, 1:] - _fds_3d[:, :, 1:], dfse[:, :, :-1]))
    dfqa = dfqa.at[:, :, :-1].set(jnp.where(loop_mask[:, :, :-1], _fuq_3d[:, :, 1:] - _fdq_3d[:, :, 1:], dfqa[:, :, :-1]))

    # Net flux of dry static energy and moisture
    dfse += loop_mask*(_fds_3d - _fus_3d)
    dfqa += loop_mask*(_fdq_3d - _fuq_3d)

    # Secondary moisture flux
    delq = loop_mask * (rhil * qsat - qa)
    moisture_flux_mask = delq > 0.
    fsq_masked = moisture_flux_mask * smf * cbmf[:, :, jnp.newaxis] * delq
    dfqa += fsq_masked
    dfqa = dfqa.at[:, :, -1].add(-jnp.sum(fsq_masked, axis=-1))

    # assuming that take_along_axis is at least as well-optimized as any workaround via masking
    index_array = lambda array, index: jnp.squeeze(jnp.take_along_axis(array, index[:, :, jnp.newaxis], axis=-1), axis=-1)
    fus, fuq, fds, fdq = index_array(_fus_3d, iptop), index_array(_fuq_3d, iptop), index_array(_fds_3d, iptop), index_array(_fdq_3d, iptop)

    # 3.3 Top layer (condensation and detrainment)
    k = iptop - 1

    # Flux of convective precipitation
    qsatb = index_array(qsat[:, :, :-1] + wvi[jnp.newaxis, jnp.newaxis, :-1, 1] * (qsat[:, :, 1:]-qsat[:, :, :-1]), k)
    precnv = jnp.maximum(fuq - index_array(_fmass_3d, iptop) * qsatb, 0.0)

    # Net flux of dry static energy and moisture
    dfse = dfse.at[:,:,k].set(fus - fds + alhc * precnv)
    dfqa = dfqa.at[:,:,k].set(fuq - fdq - precnv)

    print(iptop[0, 0], qdif[0, 0], cbmf[0, 0], precnv[0, 0], dfse[0, 0], dfqa[0, 0])

    # make a new physics_data struct. overwrite the appropriate convection bits that were calculated in this function
    # pass on the rest of physics_data that was not updated or needed in this function
    # convection in Speedy generates net *flux* -- not tendencies, so we convert dfse and dfqa to tendencies here
    # Another important note is that this goes from 2:kx in the fortran

    # Compute tendencies due to convection. Logic from physics.f90:127-130
    rps = 1/psa
    ttend = dfse.at[:,:,1:].set(dfse[:,:,1:] * rps[:,:,jnp.newaxis] * grdscp[jnp.newaxis, jnp.newaxis, 1:])
    qtend = dfqa.at[:,:,1:].set(dfqa[:,:,1:] * rps[:,:,jnp.newaxis] * grdsig[jnp.newaxis, jnp.newaxis, 1:])

    convection_out = physics_data.convection.copy(psa=psa, se=se, iptop=iptop, cbmf=cbmf, precnv=precnv)
    physics_data = physics_data.copy(convection=convection_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),ttend,qtend)
    
    return physics_tendencies, physics_data