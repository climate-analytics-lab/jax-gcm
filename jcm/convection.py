"""
Date: 2/11/2024
Parametrization of convection. Convection is modelled using a simplified 
version of the Tiedke (1993) mass-flux convection scheme.
"""
from jax import jit
import jax.numpy as jnp
from jcm.geometry import Geometry
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.physics import PhysicsTendency, PhysicsState
from jcm.physics_data import PhysicsData
from jcm.physical_constants import p0, alhc, grav, cp

@jit
def diagnose_convection(
    psa, se, qa, qsat,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    kx, ix, il = se.shape
    iptop = jnp.full((ix, il), kx + 1, dtype=int)  # Initialize iptop with nlp
    qdif = jnp.zeros((ix, il), dtype=float)

    # Saturation moist static energy
    mss = se + alhc * qsat

    rlhc = 1.0 / alhc

    # Minimum of moist static energy in the lowest two levels
    # Mask for psa > psmin
    mask_psa = psa > parameters.convection.psmin

    mse0 = se[kx-1] + alhc * qa[kx-1]
    mse1 = se[kx-2] + alhc * qa[kx-2]
    mse1 = jnp.minimum(mse0, mse1)

    # Saturation (or super-saturated) moist static energy in PBL
    mss0 = jnp.maximum(mse0, mss[kx-1])

    mss2 = jnp.pad(
        mss[:-1] + geometry.wvi[:-1, 1, jnp.newaxis, jnp.newaxis] * jnp.diff(mss, axis=0),
        ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0 # adding a 'surface' mss2 of 0 to capture ktop2 = kx case
    )

    # If there is any instability, cloud top is the first unstable level (from top down)
    # Otherwise kx (surface)
    # Note ktop1 and ktop2 are 1-indexed to match icltop convention
    possible_cltop_levels = jnp.arange(2, kx-3, dtype=int)
    get_cloud_top = lambda instability_mask: jnp.where(
        jnp.any(instability_mask, axis=0),
        (possible_cltop_levels+1)[jnp.argmax(instability_mask, axis=0)],
        jnp.array(kx, dtype=int)
    )

    # Check 1: conditional instability (MSS in PBL > MSS at top level)
    ktop1 = get_cloud_top(mss0[jnp.newaxis] > mss2[2:kx-3])

    # Check 2: gradient of actual moist static energy between lower and upper troposphere
    ktop2 = get_cloud_top(mse1[jnp.newaxis] > mss2[2:kx-3])
    msthr = jnp.squeeze(jnp.take_along_axis(mss2, ktop2[jnp.newaxis] - 1, axis=0), axis=0)

    # Check 3: RH > RH_c at both k=kx and k=kx-1
    qthr0 = parameters.convection.rhbl * qsat[kx-1]
    qthr1 = parameters.convection.rhbl * qsat[kx-2]
    lqthr = (qa[kx-1] > qthr0) & (qa[kx-2] > qthr1)

    case_1 = mask_psa & (ktop1 < kx) & (ktop2 < kx)
    case_2 = mask_psa & (ktop1 < kx) & ~(ktop2 < kx) & lqthr

    iptop = jnp.where(case_1 | case_2, ktop1, iptop)
    qdif = jnp.where(case_1, jnp.maximum(qa[kx-1] - qthr0, (mse0 - msthr) * rlhc), qdif)
    qdif = jnp.where(case_2, qa[kx-1] - qthr0, qdif)
    return iptop, qdif

@jit
def get_convection_tendencies(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:
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
    se = cp * state.temperature + state.geopotential
    qa = state.specific_humidity
    qsat = physics_data.humidity.qsat
    kx, ix, il = se.shape
    _zeros_3d = lambda: jnp.zeros((kx,ix,il))
    psa = state.surface_pressure
    
    # 1. Initialization of output and workspace arrays

    dfse, dfqa = _zeros_3d(), _zeros_3d()

    #keep indexing consistent with original Speedy
    nl1 = kx - 1
    nlp = kx + 1

    # Entrainment profile (up to sigma = 0.5)
    entr = jnp.maximum(0.0, geometry.fsg[1:kx-1] - 0.5)**2.0
    sentr = jnp.sum(entr)
    entr *= parameters.convection.entmax / sentr

    fqmax = 5.0 #maximum mass flux, not sure why this is needed
    fm0 = p0*geometry.dhs[-1]/(grav*parameters.convection.trcnv*3600.0) #prefactor for mass fluxes
    rdps=2.0/(1.0 - parameters.convection.psmin)

    # 2. Check of conditions for convection
    iptop, qdif = diagnose_convection(psa, se, qa, qsat, parameters, boundaries, geometry)

    # 3. Convection over selected grid-points
    mask = iptop < kx
    # 3.1 Boundary layer (cloud base)
    k = kx - 1

    # Maximum specific humidity in the PBL
    qmax = jnp.maximum(1.01 * qa[-1], qsat[-1])

    interpolate = lambda tracer: tracer[:-1] + geometry.wvi[:-1, 1, jnp.newaxis, jnp.newaxis] * jnp.diff(tracer, axis=0)
    _sb_3d, _qb_3d = (_zeros_3d().at[1:].set(interpolate(tracer)) for tracer in (se, qa))
    
    # Dry static energy and moisture at upper boundary
    sb, qb = _sb_3d[k], jnp.minimum(_qb_3d, qa)[k]
    
    # Cloud-base mass flux
    fpsa = psa * jnp.minimum(1.0, (psa - parameters.convection.psmin) * rdps)
    fmass = fm0 * fpsa * jnp.minimum(fqmax, qdif / (qmax - qb))
    cbmf = mask * fmass

    # Upward fluxes at upper boundary
    fus, fuq = fmass * se[k], fmass * qmax

    # Downward fluxes at upper boundary
    fds, fdq = fmass * sb, fmass * qb

    # Net flux of dry static energy and moisture
    dfse, dfqa = dfse.at[k].set(fds - fus), dfqa.at[k].set(fdq - fuq)

    # 3.2 Intermediate layers (entrainment)

    # replace loop with masking
    loop_mask = (kx - 2 >= jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis]) & \
                (jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis] >= iptop[jnp.newaxis])
    
    #start by making entrainment profile:
    _enmass_3d = loop_mask * _zeros_3d().at[1:-1].set(entr[:, jnp.newaxis, jnp.newaxis] * (psa * cbmf)[jnp.newaxis])

    # Upward fluxes at upper boundary of mass, energy, moisture
    _fmass_3d, _fus_3d, _fuq_3d = (
        base_flux[jnp.newaxis] + jnp.cumsum((_enmass_3d * tracer)[::-1], axis=0)[::-1]
        for base_flux, tracer in ((fmass, 1), (fus, se), (fuq, qa))
    )

    #Downward fluxes
    _fds_3d, _fdq_3d = (_fmass_3d * _sb_3d).at[-1].set(fds), (_fmass_3d * _qb_3d).at[-1].set(fdq)

    # Calculate flux convergence
    dfse = dfse.at[:-1].set(loop_mask[:-1] * (jnp.diff(_fus_3d - _fds_3d, axis=0)))
    dfqa = dfqa.at[:-1].set(loop_mask[:-1] * (jnp.diff(_fuq_3d - _fdq_3d, axis=0)))

    # Secondary moisture flux
    delq = loop_mask * (parameters.convection.rhil * qsat - qa)
    moisture_flux_mask = delq > 0.
    fsq_masked = moisture_flux_mask * parameters.convection.smf * cbmf[jnp.newaxis] * delq
    dfqa += fsq_masked
    dfqa = dfqa.at[-1].add(-jnp.sum(fsq_masked, axis=0))

    # assuming that take_along_axis is at least as well-optimized as any workaround via masking
    index_array = lambda array, index: jnp.squeeze(jnp.take_along_axis(array, index[jnp.newaxis], axis=0), axis=0)
    pad_array = lambda array: jnp.pad(array, ((0, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)
    fmass, fus, fuq, fds, fdq = (index_array(pad_array(_flux_3d), iptop)
                                 for _flux_3d in (_fmass_3d, _fus_3d, _fuq_3d, _fds_3d, _fdq_3d))
    
    # 3.3 Top layer (condensation and detrainment)
    k = iptop - 1

    # Flux of convective precipitation
    qsatb = index_array(pad_array(interpolate(qsat)), k)
    precnv = jnp.maximum(fuq - fmass * qsatb, 0.0)

    # Net flux of dry static energy and moisture
    i, j = jnp.meshgrid(jnp.arange(ix), jnp.arange(il), indexing="ij")
    dfse = dfse.at[k, i, j].set(fus - fds + alhc * precnv)
    dfqa = dfqa.at[k, i, j].set(fuq - fdq - precnv)

    # convection in Speedy generates net *flux* -- not tendencies, so we convert dfse and dfqa to tendencies here
    # Another important note is that this goes from 2:kx in the fortran

    # Compute tendencies due to convection. Logic from physics.f90:127-130
    rps = 1/psa
    ttend = dfse.at[1:].set(dfse[1:] * rps[jnp.newaxis] * geometry.grdscp[1:, jnp.newaxis, jnp.newaxis])
    qtend = dfqa.at[1:].set(dfqa[1:] * rps[jnp.newaxis] * geometry.grdsig[1:, jnp.newaxis, jnp.newaxis])

    convection_out = physics_data.convection.copy(psa=psa, se=se, iptop=iptop, cbmf=cbmf, precnv=precnv)
    physics_data = physics_data.copy(convection=convection_out)
    physics_tendencies = PhysicsTendency.zeros(
        shape=state.temperature.shape,
        temperature=ttend,
        specific_humidity=qtend
    )
    
    return physics_tendencies, physics_data
