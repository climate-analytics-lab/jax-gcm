"""Date: 2/11/2024
Parametrization of convection. Convection is modelled using a simplified 
version of the Tiedtke (1993) mass-flux convection scheme.
"""
from jax import jit
import jax.numpy as jnp
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics_interface import PhysicsTendency, PhysicsState
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.speedy_coords import (
    PBL_TOP_SIGMA, interp_to_sigma, stratosphere_mask,
)
import jcm.constants as c
# alhc is the SPEEDY latent heat in J/g (consistent with q in g/kg); it is a
# SPEEDY-specific value, not the shared SI constant. Shared constants (cpd, p0,
# grav) are read as module attributes from jcm.constants.
from jcm.physics.speedy.physical_constants import alhc
from jcm.physics.speedy.smoothing import smooth_gate, smooth_pos

@jit
def diagnose_convection(
    psa, se, qa, qsat,
    parameters: Parameters,
    physics_data: PhysicsData,
    forcing: ForcingData=None,
    terrain: TerrainData=None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Diagnose convectively unstable gridboxes

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
    iptop = jnp.full((ix, il), kx + 1)  # Initialize iptop with nlp
    qdif = jnp.zeros((ix, il))

    # Saturation moist static energy
    mss = se + alhc * qsat

    rlhc = 1.0 / alhc

    # Minimum of moist static energy between the surface layer and the PBL top
    # Mask for psa > psmin
    mask_psa = psa > parameters.convection.psmin

    # The trigger compares the surface layer against the top of the sub-cloud
    # layer. That "PBL-top" reference is a *physical* depth (~150 hPa above the
    # surface), so it is evaluated at a fixed sigma rather than at an
    # index-relative level, keeping the trigger independent of the vertical
    # grid. On the 8-level reference grid the fixed sigma is the second-lowest
    # layer centre, so the validated behaviour is reproduced exactly there.
    fsg = physics_data.speedy_coords.fsg
    se_sc = interp_to_sigma(se, fsg, PBL_TOP_SIGMA)
    qa_sc = interp_to_sigma(qa, fsg, PBL_TOP_SIGMA)
    qsat_sc = interp_to_sigma(qsat, fsg, PBL_TOP_SIGMA)

    mse0 = se[kx-1] + alhc * qa[kx-1]
    mse1 = se_sc + alhc * qa_sc
    mse1 = jnp.minimum(mse0, mse1)

    # Saturation (or super-saturated) moist static energy in PBL
    mss0 = jnp.maximum(mse0, mss[kx-1])

    mss2 = jnp.pad(
        mss[:-1] + physics_data.speedy_coords.wvi[:-1, 1, jnp.newaxis, jnp.newaxis] * jnp.diff(mss, axis=0),
        ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0 # adding a 'surface' mss2 of 0 to capture ktop2 = kx case
    )

    # Cloud top is the highest (least-sigma) unstable level. SPEEDY searched
    # k=3..kx-3 (1-indexed), i.e. indices 2..kx-4 here, which hardcodes "the top
    # two levels are stratosphere and cannot hold a cloud top". We replace that
    # upper bound with the sigma<0.2 stratosphere mask so the search range scales
    # with nlev: candidate levels run from the first interface below the very top
    # down to kx-4, and any candidate falling in the stratosphere is masked out
    # of the instability test so it is never chosen. The cloud *base* is the PBL
    # (lowest layer), which is already physical. strat_mask is static (fsg).
    possible_cltop_levels = jnp.arange(1, kx-3)
    strat_mask = stratosphere_mask(physics_data.speedy_coords.fsg)
    not_strat_cand = (~strat_mask[possible_cltop_levels])[:, jnp.newaxis, jnp.newaxis]
    get_cloud_top = lambda instability_mask: jnp.where(
        jnp.any(instability_mask, axis=0),
        (possible_cltop_levels+1)[jnp.argmax(instability_mask, axis=0)],
        jnp.array(kx)
    )

    # Check 1: conditional instability (MSS in PBL > MSS at top level)
    ktop1 = get_cloud_top((mss0 > mss2[1:kx-3]) & not_strat_cand)

    # Check 2: gradient of actual moist static energy between lower and upper troposphere
    ktop2 = get_cloud_top((mse1 > mss2[1:kx-3]) & not_strat_cand)
    msthr = jnp.squeeze(jnp.take_along_axis(mss2, ktop2[jnp.newaxis] - 1, axis=0), axis=0)

    # Check 3: RH > RH_c at both the surface layer and the PBL top
    qthr0 = parameters.convection.rhbl * qsat[kx-1]
    qthr1 = parameters.convection.rhbl * qsat_sc

    case_1 = mask_psa & (ktop1 < kx) & (ktop2 < kx)

    # The humidity trigger is a value jump: when the PBL-top RH condition
    # flips, qdif switches between 0 and the finite surface-layer excess.
    # With trigger_smoothing > 0 (an RH fraction) the case-2 excess is
    # instead scaled by sigmoid gates on both RH criteria, ramping
    # convection in over ~2 widths of relative humidity. The iptop
    # assignment keeps a hard mask (a level index has no smooth
    # counterpart) but widens its humidity condition by 6 widths so the
    # discrete activation happens out on the gate's skirt, where the gated
    # mass flux is at most sigmoid(-6) ~ 2.5e-3 of the excess: the value
    # jump survives only at that negligible amplitude. Width 0 reproduces
    # the hard trigger exactly.
    w_rh = parameters.convection.trigger_smoothing
    trigger_gate = (
        smooth_gate(qa[kx-1], qthr0, w_rh * qsat[kx-1])
        * smooth_gate(qa_sc, qthr1, w_rh * qsat_sc)
    )
    lqthr_wide = (
        (qa[kx-1] > qthr0 - 6.0 * w_rh * qsat[kx-1])
        & (qa_sc > qthr1 - 6.0 * w_rh * qsat_sc)
    )
    case_2_soft = mask_psa & (ktop1 < kx) & ~(ktop2 < kx) & lqthr_wide

    iptop = jnp.where(case_1 | case_2_soft, ktop1, iptop)
    qdif = jnp.where(case_1, jnp.maximum(qa[kx-1] - qthr0, (mse0 - msthr) * rlhc), qdif)
    qdif = jnp.where(
        case_2_soft,
        trigger_gate * jnp.maximum(qa[kx-1] - qthr0, 0.0),
        qdif,
    )
    return iptop, qdif

@jit
def get_convection_tendencies(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData=None,
    terrain: TerrainData=None
) -> tuple[PhysicsTendency, PhysicsData]:
    """Compute convective fluxes of dry static energy and moisture using a simplified mass-flux scheme.

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
    se = c.cpd * state.temperature + state.geopotential
    qa = state.specific_humidity
    qsat = physics_data.humidity.qsat
    kx, ix, il = se.shape
    _zeros_3d = lambda: jnp.zeros((kx,ix,il))
    psa = state.normalized_surface_pressure
    
    # 1. Initialization of output and workspace arrays

    dfse, dfqa = _zeros_3d(), _zeros_3d()

    # Entrainment profile (up to sigma = 0.5)
    entr = jnp.maximum(0.0, physics_data.speedy_coords.fsg[1:kx-1] - 0.5)**2.0
    sentr = jnp.sum(entr)
    entr *= parameters.convection.entmax / sentr

    fqmax = 5.0 #maximum mass flux, not sure why this is needed
    fm0 = c.p0*physics_data.speedy_coords.dhs[-1]/(c.grav*parameters.convection.trcnv*3600.0) #prefactor for mass fluxes
    rdps=2.0/(1.0 - parameters.convection.psmin)

    # 2. Check of conditions for convection
    iptop, qdif = diagnose_convection(psa, se, qa, qsat, parameters, physics_data, forcing, terrain)

    # 3. Convection over selected grid-points
    mask = ~(iptop == kx+1)
    # 3.1 Boundary layer (cloud base)
    k = kx - 1

    # Maximum specific humidity in the PBL
    qmax = jnp.maximum(1.01 * qa[-1], qsat[-1])

    interpolate = lambda tracer: tracer[:-1] + physics_data.speedy_coords.wvi[:-1, 1, jnp.newaxis, jnp.newaxis] * jnp.diff(tracer, axis=0)
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
    _k_3d = jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis]
    loop_mask = (kx - 2 >= _k_3d) & (_k_3d >= iptop)
    
    #start by making entrainment profile:
    _enmass_3d = loop_mask * _zeros_3d().at[1:-1].set(entr[:, jnp.newaxis, jnp.newaxis] * psa * cbmf)

    # Upward fluxes at upper boundary of mass, energy, moisture
    _fmass_3d, _fus_3d, _fuq_3d = (
        base_flux + jnp.cumsum((_enmass_3d * tracer)[::-1], axis=0)[::-1]
        for base_flux, tracer in ((fmass, 1), (fus, se), (fuq, qa))
    )

    # Downward fluxes
    _fds_3d, _fdq_3d = (_fmass_3d * _sb_3d).at[-1].set(fds), (_fmass_3d * _qb_3d).at[-1].set(fdq)

    # Calculate flux convergences
    dfse = dfse.at[:-1].set(loop_mask[:-1] * (jnp.diff(_fus_3d - _fds_3d, axis=0)))
    dfqa = dfqa.at[:-1].set(loop_mask[:-1] * (jnp.diff(_fuq_3d - _fdq_3d, axis=0)))

    # Secondary moisture flux
    delq = loop_mask * (parameters.convection.rhil * qsat - qa)
    moisture_flux_mask = delq > 0.
    fsq_masked = moisture_flux_mask * parameters.convection.smf * cbmf * delq
    dfqa += fsq_masked
    dfqa = dfqa.at[-1].add(-jnp.sum(fsq_masked, axis=0))

    # assuming that take_along_axis is at least as well-optimized as any workaround via masking
    index_array = lambda array, index: jnp.squeeze(jnp.take_along_axis(array, index[jnp.newaxis], axis=0), axis=0)
    pad_array = lambda array: jnp.pad(array, ((0, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)
    fmass, fus, fuq, fds, fdq = (index_array(pad_array(_flux_3d), iptop)
                                 for _flux_3d in (_fmass_3d, _fus_3d, _fuq_3d, _fds_3d, _fdq_3d))
    
    # 3.3 Top layer (condensation and detrainment)
    k = iptop - 1

    # Flux of convective precipitation. The onset hinge (moisture flux
    # crossing cloud-top saturation) zeroes the gradient of every
    # parameter through non-precipitating columns; precnv_smoothing > 0
    # [g/(m^2 s)] replaces it with a softplus of that half-width.
    qsatb = index_array(pad_array(interpolate(qsat)), k)
    precnv = smooth_pos(fuq - fmass * qsatb, parameters.convection.precnv_smoothing)

    # Net flux of dry static energy and moisture
    i, j = jnp.meshgrid(jnp.arange(ix), jnp.arange(il), indexing="ij")
    dfse = dfse.at[k, i, j].set(fus - fds + alhc * precnv)
    dfqa = dfqa.at[k, i, j].set(fuq - fdq - precnv)

    # convection in Speedy generates net *flux* -- not tendencies, so we convert dfse and dfqa to tendencies here
    # Another important note is that this goes from 2:kx in the fortran

    # Compute tendencies due to convection. Logic from physics.f90:127-130
    rps = 1/psa
    ttend = dfse.at[1:].set(dfse[1:] * rps * physics_data.speedy_coords.grdscp[1:, jnp.newaxis, jnp.newaxis])
    qtend = dfqa.at[1:].set(dfqa[1:] * rps * physics_data.speedy_coords.grdsig[1:, jnp.newaxis, jnp.newaxis])

    convection_out = physics_data.convection.copy(se=se, iptop=iptop, cbmf=cbmf, qdif=qdif, precnv=precnv)
    physics_data = physics_data.copy(convection=convection_out)
    physics_tendencies = PhysicsTendency.zeros(
        shape=state.temperature.shape,
        temperature=ttend,
        specific_humidity=qtend
    )
    
    return physics_tendencies, physics_data
