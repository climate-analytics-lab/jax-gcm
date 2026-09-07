import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
import jax
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physical_constants import epssw, solc, epsilon
from jcm.physics_interface import PhysicsTendency, PhysicsState
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.speedy_coords import (
    PBL_TOP_SIGMA, SpeedyCoords, interp_to_sigma, ozone_sigma_weight,
    stratosphere_mask,
)
from jcm.physics.speedy.smoothing import (
    smooth_clip01, smooth_min, smooth_max, smooth_pos,
)

# Reference sigma surfaces for the cloud diagnostics in :func:`clouds`. Both
# diagnostics are tuned to sample the atmosphere at particular *physical*
# depths, so they are evaluated at fixed sigmas (the 8-level reference grid's
# layer centres, where the scheme was validated) rather than at index-relative
# model levels, keeping them independent of the vertical grid:
#   * the stratocumulus stability gradient ``gse`` measures lower-tropospheric
#     stability across the interval spanning the marine boundary-layer
#     inversion (sigma ~0.835-0.95). Measured over the two lowest *levels* it
#     collapses at high nlev, where both levels sit inside the near-neutral
#     surface mixed layer, and the stratocumulus SW feedback is lost.
#   * the total-cloud RH maximum ``cloudc`` samples free-troposphere RH. A max
#     over all model levels samples the profile more finely as nlev grows, so
#     cloudc inflates with resolution — and, through the
#     ``clsmax - clfact*cloudc`` competition below, spuriously suppresses the
#     stratocumulus deck.
_GSE_SIGMA_TOP, _GSE_SIGMA_BOT = PBL_TOP_SIGMA, 0.95
_CLOUDC_REF_SIGMAS = (0.34, 0.51, 0.685)  # free-troposphere layer centres

@jit
def get_shortwave_rad_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:

    # SPEEDY computes shortwave radiation only every `nstrad` steps (physics.f90) but keeps
    # the resulting heating rate (tt_rsw) and applies it on every step. Do the same here:
    # on a radiation step compute the fluxes and cache the temperature tendency in the
    # shortwave carry; on the other steps skip the (expensive) computation and re-apply the
    # cached tendency. The cached radiative diagnostics (ftop, rsds, rsns, dfabs) are carried
    # unchanged between radiation steps, consistent with the heating actually applied.
    shape = state.temperature.shape

    def _compute(_):
        zero_tendencies = PhysicsTendency.zeros(shape=shape)
        _, new_physics_data, _, _, _, tendencies = shortwave_rad_fluxes(
            (state, physics_data, parameters, forcing, terrain, zero_tendencies))
        shortwave_rad = new_physics_data.shortwave_rad.copy(heating_rate=tendencies.temperature)
        return tendencies, new_physics_data.copy(shortwave_rad=shortwave_rad)

    def _replay(_):
        tendencies = PhysicsTendency.zeros(shape=shape, temperature=physics_data.shortwave_rad.heating_rate)
        return tendencies, physics_data

    return jax.lax.cond(physics_data.shortwave_rad.compute_shortwave, _compute, _replay, None)


@jit
def shortwave_rad_fluxes(operand):
    """psa(ix,il)       # Normalised surface pressure [p/p0]
    qa(ix,il,kx)     # Specific humidity [g/kg]
    icltop(ix,il)    # Cloud top level
    cloudc(ix,il)    # Total cloud cover
    clstr(ix,il)     # Stratiform cloud cover
    rsds(ix,il)    # Total downward flux of short-wave radiation at the surface
    rsns(ix,il)     # Net downward flux of short-wave radiation at the surface
    ftop(ix,il)     # Net downward flux of short-wave radiation at the top of the atmosphere
    dfabs(ix,il,kx) # Flux of short-wave radiation absorbed in each atmospheric layer
    """
    state, physics_data, parameters, forcing, terrain, tendencies = operand

    kx, ix, il = state.temperature.shape
    dhs = physics_data.speedy_coords.dhs
    fsg = physics_data.speedy_coords.fsg

    psa = state.normalized_surface_pressure
    qa = state.specific_humidity
    icltop = physics_data.shortwave_rad.icltop
    cloudc = physics_data.shortwave_rad.cloudc
    clstr = physics_data.shortwave_rad.cloudstr

    # mod_radcon inputs
    albsfc = physics_data.mod_radcon.albsfc

    nl1 = kx - 1

    fband2 = 0.05
    fband1 = 1.0 - fband2

    #  Initialization
    mask = icltop <= kx
    clamped_icltop = jnp.clip(icltop, 1, kx).astype(int) # Clamp icltop to avoid invalid indices, for vectorizing indexing operation
    
    # Start with tau2
    # Create arrays of i and j indices that will broadcast correctly alongside clamped_icltop
    i_idx, j_idx = jnp.meshgrid(jnp.arange(ix), jnp.arange(il), indexing='ij')
    # Update values at cloud top
    tau2 = jnp.zeros((kx, ix, il, 4))
    tau2 = tau2.at[clamped_icltop-1, i_idx, j_idx, 2].set(
        mask * parameters.shortwave_radiation.albcl * cloudc
    ) # equivalent to updating tau2 only where mask is true
    # Update the tau2 values for the second condition (kx index) across the entire array
    tau2 = tau2.at[kx - 1, :, :, 2].set(parameters.shortwave_radiation.albcls * clstr)

    # 2. Shortwave transmissivity:
    # function of layer mass, ozone (in the statosphere),
    # abs. humidity and cloud cover (in the troposphere)
    psaz = psa*physics_data.shortwave_rad.zenit
    acloud = cloudc*jnp.minimum(
        parameters.shortwave_radiation.abscl1*physics_data.shortwave_rad.qcloud,
        parameters.shortwave_radiation.abscl2
    )
    tau2 = tau2.at[0,:,:,0].set(jnp.exp(-psaz*dhs[0]*parameters.shortwave_radiation.absdry))

    abs1 = parameters.shortwave_radiation.absdry + parameters.shortwave_radiation.absaer * fsg[1:nl1] ** 2
    cloudy = jnp.arange(2, nl1+1)[:, jnp.newaxis, jnp.newaxis] >= icltop
    
    tau2 = tau2.at[1:nl1, :, :, 0].set(
        jnp.exp(-psaz * dhs[1:nl1, jnp.newaxis, jnp.newaxis] * (
            abs1[:, jnp.newaxis, jnp.newaxis] +
            parameters.shortwave_radiation.abswv1 * qa[1:nl1] +
            cloudy * acloud
        ))
    )

    abs1 = parameters.shortwave_radiation.absdry + parameters.shortwave_radiation.absaer*fsg[kx - 1]**2
    tau2 = tau2.at[kx-1,:,:,0].set(jnp.exp(-psaz*dhs[kx - 1]*(abs1 + parameters.shortwave_radiation.abswv1*qa[kx - 1])))
    tau2 = tau2.at[1:kx,:,:,1].set(
        jnp.exp(-psaz*dhs[1:kx, jnp.newaxis, jnp.newaxis]*parameters.shortwave_radiation.abswv2*qa[1:kx])
    )

    # 3. Shortwave downward flux
    # 3.1 Initialization of fluxes
    
    rsns = jnp.zeros((ix, il)) # Net downward flux of short-wave radiation at the surface
    dfabs = jnp.zeros((kx,ix,il)) # Flux of short-wave radiation absorbed in each atmospheric layer
    ftop = physics_data.shortwave_rad.fsol # Net downward flux of short-wave radiation at the top of the atmosphere

    flux_1, flux_2 = jnp.zeros((kx, ix, il)), jnp.zeros((kx, ix, il))
    flux_1 = flux_1.at[0].set(physics_data.shortwave_rad.fsol*fband1)
    flux_2 = flux_2.at[0].set(physics_data.shortwave_rad.fsol*fband2)

    # 3.2 Ozone absorption in the stratosphere, distributed by sigma.
    #
    # Original SPEEDY hardcodes ozone absorption to the top two levels: the
    # `ozupp` field at k=0 and the `ozone` field at k=1. To scale with nlev we
    # instead spread the *total* stratospheric ozone absorption
    # (ozupp + ozone) over every layer with sigma < 0.2, weighted by
    # SpeedyWeather.jl's ozone distribution 50*max(0, 1/5 - sigma) times the
    # layer thickness dhs. The weights are normalised to sum to 1 over the
    # column so the column-integrated ozone absorption is preserved and equals
    # ozupp + ozone exactly (matching SPEEDY's total at nlev=8, where the top
    # two levels are the only ones with sigma<0.2). oz_lev is a (kx,ix,il)
    # per-layer ozone absorption field; fsg/dhs are static so the weights are
    # compile-time constants.
    oz_weight = ozone_sigma_weight(fsg) * dhs                      # (kx,)
    oz_weight = oz_weight / jnp.maximum(jnp.sum(oz_weight), epsilon)
    oz_total = physics_data.shortwave_rad.ozupp + physics_data.shortwave_rad.ozone  # (ix,il)
    oz_lev = oz_weight[:, jnp.newaxis, jnp.newaxis] * oz_total[jnp.newaxis] * psa    # (kx,ix,il)

    # 3.3 Single downward-flux pass over the whole column.
    #
    # At each level the beam loses ozone absorption (non-zero only in the
    # stratosphere) and is then attenuated by the layer transmissivity and
    # cloud reflection (cloud reflection tau2[...,2] is non-zero only in the
    # troposphere). The unified propagator therefore reduces to SPEEDY's
    # stratosphere update tau*(flux - oz) where there is no cloud, and to its
    # troposphere update flux*tau*(1-cloud) where there is no ozone.
    propagate_flux_1 = lambda flux, tau, oz: tau[:, :, 0] * (flux - oz) * (1 - tau[:, :, 2])

    # The per-layer outputs are the flux leaving the bottom of each layer. Flux
    # entering layer k equals fsol*fband1 for k=0 and the flux leaving layer k-1
    # otherwise.
    _, flux_out = lax.scan(
        jax.checkpoint(lambda carry, xs: (propagate_flux_1(carry, xs[0], xs[1]),) * 2),
        flux_1[0],
        (tau2[:, :, :, :], oz_lev),
    )
    flux_in = jnp.concatenate([flux_1[:1], flux_out[:-1]], axis=0)
    flux_1 = flux_out

    # Absorbed flux per layer = (incoming - outgoing) minus the cloud-reflected
    # fraction (flux_in - oz)*tau_cloud. In the stratosphere tau_cloud=0 so all
    # attenuation (including ozone) is absorbed; in the troposphere oz=0 and the
    # reflected fraction is excluded from absorption.
    reflected = (flux_in - oz_lev) * tau2[:, :, :, 2]
    dfabs = dfabs.at[:].set((flux_in - flux_1) - reflected)
    tau2 = tau2.at[:, :, :, 2].multiply(flux_in - oz_lev)

    flux_2 = flux_2.at[1].set(flux_2[0])
    propagate_flux_2 = lambda flux, tau: flux * tau[:, :, 1]
    _, flux_2_scan = lax.scan(
        jax.checkpoint(lambda carry, i: (propagate_flux_2(carry, i),)*2),
        flux_2[1],
        tau2[1:kx])
    flux_2 = flux_2.at[1:kx].set(flux_2_scan)
    dfabs = dfabs.at[1:kx].add(flux_2[:kx-1]*(1 - tau2[1:kx,:,:,1])) # changed k to kx double check this

    # 4. Shortwave upward flux

    # 4.1  Absorption and reflection at the surface
    rsds = flux_1[kx-1] + flux_2[kx-1]
    flux_1 = flux_1.at[kx-1].multiply(albsfc)
    rsns = rsds - flux_1[kx-1]

    # 4.2  Absorption of upward flux

    propagate_flux_up = lambda flux, tau: flux * tau[:,:,0] + tau[:,:,2]
    _, flux_1_scan = lax.scan(
        jax.checkpoint(lambda carry, tau: (propagate_flux_up(carry, tau),) * 2),
        flux_1[-1],
        tau2[1:kx][::-1]
    )
    flux_1 = flux_1.at[:-1].set(flux_1_scan[::-1])
    
    dfabs += flux_1*(1 - tau2[:,:,:,0])

    flux_1 = flux_1.at[1:].set(flux_1[:-1])
    flux_1 = flux_1.at[0].set(tau2[0,:,:,0]*flux_1[0] + tau2[0,:,:,2])

    # 4.3  Net solar radiation = incoming - outgoing
    ftop = ftop - flux_1[0]

    # 5. Initialization of longwave radiation model
    # 5.1 Longwave transmissivity:
    # function of layer mass, abs. humidity and cloud cover.

    # Base absorptivities
    absorptivity = jnp.stack([
        parameters.shortwave_radiation.ablwin * jnp.ones_like(qa),
        physics_data.mod_radcon.ablco2 * jnp.ones_like(qa),
        parameters.shortwave_radiation.ablwv1 * qa,
        parameters.shortwave_radiation.ablwv2 * qa
    ], axis=-1)

    # Topmost stratospheric layer: no water-vapour longwave absorption.
    # SPEEDY zeroed the water-vapour bands at k=0 only; we zero them for every
    # layer with sigma<0.2 (the stratosphere). qa is already ~0 there so this is
    # mostly a clean-up, but it keeps the "stratosphere has no water vapour"
    # statement nlev-independent. strat_mask is static.
    strat_mask = stratosphere_mask(fsg)
    absorptivity = absorptivity.at[:, :, :, 2:].set(
        jnp.where(strat_mask[:, jnp.newaxis, jnp.newaxis, jnp.newaxis], 0.0, absorptivity[:, :, :, 2:])
    )

    # Cloud absorptivity is added only in the free troposphere: below the
    # stratosphere (sigma>=0.2) and above the PBL (the lowest layer, which SPEEDY
    # leaves cloud-free). SPEEDY hardcoded this range as k=2..kx-2; we replace
    # the top boundary with the stratosphere sigma mask so it scales with nlev,
    # and keep the single-layer PBL exclusion at the bottom.
    trop_cloud_mask = (~strat_mask)[:, jnp.newaxis, jnp.newaxis]
    trop_cloud_mask = trop_cloud_mask.at[kx - 1].set(False)  # PBL: cloud-free
    acloud = cloudc * parameters.shortwave_radiation.ablcl2
    acloud1 = jnp.where(jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis] + 1 < icltop, acloud, cloudc * parameters.shortwave_radiation.ablcl1)
    absorptivity = absorptivity.at[:, :, :, 0].add(jnp.where(trop_cloud_mask, acloud1, 0.0))
    absorptivity = absorptivity.at[:, :, :, 2:].set(
        jnp.where(
            trop_cloud_mask[:, :, :, jnp.newaxis],
            jnp.maximum(absorptivity[:, :, :, 2:], acloud[:, :, jnp.newaxis]),
            absorptivity[:, :, :, 2:],
        )
    )

    # Now compute tau2
    deltap = psa*dhs[:,jnp.newaxis,jnp.newaxis]
    tau2 = jnp.exp(-deltap[:,:,:,jnp.newaxis] * absorptivity)
        
    # 5.2  Stratospheric correction terms
    # eps1 spreads the longwave stratospheric-cooling correction over the mass
    # of the stratosphere. SPEEDY used dhs[0]+dhs[1] (the top two layers); we use
    # the total thickness of all layers with sigma<0.2 so the correction scales
    # with the (now nlev-dependent) stratosphere depth. strat_mask/dhs are static.
    strat_mask = stratosphere_mask(fsg)
    strat_dsig = jnp.sum(jnp.where(strat_mask, dhs, 0.0))
    eps1 = parameters.mod_radcon.epslw / jnp.maximum(strat_dsig, epsilon)
    stratc = jnp.zeros((ix, il, 2))
    stratc = stratc.at[:,:,0].set(physics_data.shortwave_rad.stratz*psa)
    stratc = stratc.at[:,:,1].set(eps1*psa)

    flux = physics_data.mod_radcon.flux.at[:,:,0].set(flux_1[0]).at[:,:,1].set(flux_2[kx-1])
    mod_radcon_out = physics_data.mod_radcon.copy(tau2=tau2, stratc=stratc, flux=flux)
    shortwave_rad_out = physics_data.shortwave_rad.copy(rsns=rsns, ftop=ftop, dfabs=dfabs, rsds=rsds)
    physics_data = physics_data.copy(shortwave_rad=shortwave_rad_out, mod_radcon=mod_radcon_out)

    # Get temperature tendency due to absorbed shortwave flux. Logic from physics.f90:160-162
    ttend_swr = dfabs*physics_data.speedy_coords.grdscp[:, jnp.newaxis, jnp.newaxis]/state.normalized_surface_pressure # physics.f90:160-162
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape, temperature=ttend_swr)

    return (state, physics_data, parameters, forcing, terrain, physics_tendencies)


@jit
def get_zonal_average_fields(
    state: PhysicsState,
    physics_data: PhysicsData,
    forcing: ForcingData,
    terrain: TerrainData
) -> PhysicsData:
    """Calculate zonal average fields including solar radiation, ozone depth,
    and polar night cooling in the stratosphere using JAX.

    Reads the fraction of year off ``forcing.solar.tyear`` (populated by
    `Model._get_step_fn_factory` ↔ `ForcingData.select(date)`).

    Returns
    -------
    fsol : jnp.ndarray
        Solar radiation at the top
    ozupp : jnp.ndarray
        Ozone depth in upper stratosphere
    ozone : jnp.ndarray
        Ozone concentration in lower stratosphere
    stratz : jnp.ndarray
        Polar night cooling in the stratosphere
    zenit : jnp.ndarray
        The zenith angle

    """
    kx, ix, il = state.temperature.shape

    # `forcing.solar` is precomputed by `Model._get_step_fn_factory` ↔
    # `ForcingData.select(date)`, so this routine never has to read the
    # date object directly. tyear here matches the SPEEDY convention used
    # by the `solar` lookup below.
    tyear = forcing.solar.tyear

    # Alpha = year phase (0 - 2pi, 0 = winter solstice = 22 Dec)
    alpha = 4.0 * jnp.arcsin(1.0) * (tyear + 10.0 / 365.0)
    dalpha = 0.0

    coz1 = jnp.maximum(0.0, jnp.cos(alpha - dalpha))
    coz2 = 1.8

    azen = 1.0
    nzen = 2

    rzen = -jnp.cos(alpha) * 23.45 * jnp.arcsin(1.0) / 90.0

    fs0 = 6.0

    # Solar radiation at the top
    topsr = jnp.zeros(il)
    topsr = solar(tyear, physics_data.speedy_coords, 4*solc)

    def compute_fields(sia_j, coa_j, topsr_j):
        flat2 = 1.5 * sia_j ** 2 - 0.5

        # Solar radiation at the top
        fsol_i_j = topsr_j

        # Ozone depth in upper stratosphere
        ozupp_i_j = 0.5 * epssw
        ozone_i_j = 0.4 * epssw * (1.0 + coz1 * sia_j + coz2 * flat2)

        # Zenith angle correction to (downward) absorptivity
        zenit_i_j = 1.0 + azen * (1.0 - (coa_j * jnp.cos(rzen) + sia_j * jnp.sin(rzen))) ** nzen

        # Ozone absorption in upper and lower stratosphere
        ozupp_i_j = fsol_i_j * ozupp_i_j * zenit_i_j
        ozone_i_j = fsol_i_j * ozone_i_j * zenit_i_j

        # Polar night cooling in the stratosphere
        stratz_i_j = jnp.maximum(fs0 - fsol_i_j, 0.0)

        return *(jnp.full(ix, field) for field in (fsol_i_j, ozupp_i_j, ozone_i_j, zenit_i_j, stratz_i_j)),

    vectorized_compute_fields = vmap(compute_fields, in_axes=0, out_axes=1)

    fsol, ozupp, ozone, zenit, stratz = vectorized_compute_fields(physics_data.speedy_coords.sia, physics_data.speedy_coords.coa, topsr)

    swrad_out = physics_data.shortwave_rad.copy(fsol=fsol, ozupp=ozupp, ozone=ozone, zenit=zenit, stratz=stratz)
    physics_data = physics_data.copy(shortwave_rad=swrad_out)
    
    return physics_data

@jit
def get_clouds(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:

    # Clouds are only needed on shortwave radiation steps; skip the computation otherwise
    # and carry the previous cloud fields (they produce no tendency of their own).
    zero_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)

    def _compute(_):
        _, new_physics_data, _, _, _, _ = clouds((state, physics_data, parameters, forcing, terrain, zero_tendencies))
        return new_physics_data

    return zero_tendencies, jax.lax.cond(physics_data.shortwave_rad.compute_shortwave, _compute, lambda _: physics_data, None)


@jit
def clouds(operand):
    """Simplified cloud cover scheme based on relative humidity and precipitation.

    Args:
        qa: Specific humidity [g/kg] - PhysicsState.specific_humidity
        rh: Relative humidity - PhysicsData.Humidity
        precnv: Convection precipitation - PhysicsData.Convection
        precls: Large-scale condensational precipitation - PhysicsData.Condensation
        iptop: Cloud top level - PhysicsData.Convection
        gse: Vertical gradient of dry static energy - 
        fmask: Fraction land-sea mask

    Returns:
        icltop: Cloud top level
        cloudc: Total cloud cover
        clstr: Stratiform cloud cover

    """
    state, physics_data, parameters, forcing, terrain, tendencies = operand

    # Stratocumulus stability gradient: dry static energy gradient across the
    # fixed sigma interval spanning the boundary-layer inversion (see the
    # module-level note on reference sigmas). On the 8-level reference grid the
    # interval endpoints are the two lowest layer centres, reproducing the
    # validated behaviour exactly.
    se = physics_data.convection.se
    phig = state.geopotential
    fsg = physics_data.speedy_coords.fsg
    se_top = interp_to_sigma(se, fsg, _GSE_SIGMA_TOP)
    se_bot = interp_to_sigma(se, fsg, _GSE_SIGMA_BOT)
    ph_top = interp_to_sigma(phig, fsg, _GSE_SIGMA_TOP)
    ph_bot = interp_to_sigma(phig, fsg, _GSE_SIGMA_BOT)
    # Safety check to prevent division by zero (can happen during initialization)
    dphi = ph_top - ph_bot
    gse = jnp.where(jnp.abs(dphi) > 1e-10, (se_top - se_bot)/dphi, 0.0)

    humidity = physics_data.humidity
    conv = physics_data.convection
    condensation = physics_data.condensation
    kx = state.temperature.shape[0]

    # Constants
    nl1  = kx-2
    nlp  = kx
    rrcl = 1./(parameters.shortwave_radiation.rhcl2-parameters.shortwave_radiation.rhcl1)

    # 1.  Cloud cover, defined as the sum of:
    #     - a term proportional to the square-root of precip. rate
    #     - a quadratic function of the max. relative humidity
    #       in tropospheric layers above PBL where Q > QACL :
    #       ( = 0 for RHmax < RHCL1, = 1 for RHmax > RHCL2 )
    #     Cloud-top level: defined as the highest (i.e. least sigma)
    #       between the top of convection/condensation and
    #       the level of maximum relative humidity.

    # First for loop (2 levels)
    mask = humidity.rh[nl1] > parameters.shortwave_radiation.rhcl1  # Create a mask where the condition is true
    cloudc = jnp.where(mask, humidity.rh[nl1] - parameters.shortwave_radiation.rhcl1, 0.0)  # Compute cloudc values where the mask is true
    icltop = jnp.where(mask, nl1+1, nlp+1) # Assign icltop values based on the mask

    # Vectorized implementation of the second for loop.
    #
    # Search the free troposphere for the level of maximum relative humidity.
    # SPEEDY restricted this to k=2..kx-3 (skipping the top-two stratosphere and
    # the two PBL layers). We replace the upper (stratosphere) bound with the
    # sigma<0.2 mask so it scales with nlev, and keep the lower PBL exclusion
    # (the two lowest layers, handled by the "first for loop" above). The search
    # is done over the whole column with invalid layers masked out, so no
    # fixed-index slice survives. fsg-derived masks are static.
    strat_mask = stratosphere_mask(physics_data.speedy_coords.fsg)
    drh = humidity.rh - parameters.shortwave_radiation.rhcl1
    level = jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis]
    search_mask = (
        (state.specific_humidity > parameters.shortwave_radiation.qacl)
        & (~strat_mask)[:, jnp.newaxis, jnp.newaxis]
        & (level < kx - 2)
    )

    # Set invalid entries to -1 so they are not chosen by argmax
    max_valid_rh_layer = jnp.argmax(jnp.where(search_mask, humidity.rh, -1), axis=0)
    max_drh = jnp.squeeze(jnp.take_along_axis(drh, max_valid_rh_layer[jnp.newaxis], axis=0), axis=0)

    valid_column = jnp.any(search_mask, axis=0) # Ensures that max_drh is from a valid layer
    icltop = jnp.where(valid_column & (max_drh > cloudc), max_valid_rh_layer + 1, icltop)

    # The cloud-cover *magnitude* takes the RH maximum on a fixed reference
    # sigma grid (the 8-level free-troposphere layer centres plus the PBL top)
    # rather than over the model levels searched above, so it is independent of
    # the vertical grid (see the module-level note on reference sigmas). The
    # cloud-top level ``icltop`` keeps the actual-level argmax from the search
    # above, since the radiation needs a real level index for the cloud top.
    # cover_smoothing > 0 (an RH/cover fraction) rounds every hinge and
    # clip in the cover diagnosis below; 0 keeps the original hard
    # branches. The RH hinge at rhcl1 and the max over reference levels
    # zero d(cover)/d(rhcl1) and the state gradient wherever a level is
    # sub-critical or non-dominant; the smooth forms keep exponentially
    # decaying tails instead.
    w_cov = parameters.shortwave_radiation.cover_smoothing
    rh = humidity.rh
    rhcl1 = parameters.shortwave_radiation.rhcl1
    cloudc = smooth_pos(interp_to_sigma(rh, fsg, PBL_TOP_SIGMA) - rhcl1, w_cov)
    for sig_ref in _CLOUDC_REF_SIGMAS:
        cloudc = smooth_max(cloudc, interp_to_sigma(rh, fsg, sig_ref) - rhcl1, w_cov)

    # Third for loop (two levels)
    # Perform the calculations (Two Loops)
    # The precipitation term has three sharp sites: the pmaxcl cap, the
    # sqrt corner at zero precipitation (slope 1/(2*sqrt(eps)) ~ 1.6e4
    # with the hard epsilon floor), and the saturation of the total cover
    # at 1. With cover_smoothing on, the cap and saturation become
    # hyperbolic minima and the sqrt is regularized as sqrt(pr1 + delta)
    # with delta = (w*pmaxcl)^2, bounding the corner slope at
    # 1/(2*w*pmaxcl).
    pmaxcl = parameters.shortwave_radiation.pmaxcl
    pr1 = smooth_min(pmaxcl, 86.4 * (conv.precnv + condensation.precls), w_cov * pmaxcl)
    sqrt_arg = jnp.where(
        w_cov > 0.0,
        smooth_pos(pr1, w_cov) + (w_cov * pmaxcl) ** 2,
        jnp.maximum(epsilon, pr1),
    )
    cloudc = smooth_min(
        1.0,
        parameters.shortwave_radiation.wpcl * jnp.sqrt(sqrt_arg)
        + smooth_min(1.0, cloudc * rrcl, w_cov)**2.0,
        w_cov,
    )
    cloudc = jnp.where(jnp.isnan(cloudc), 1.0, cloudc)
    icltop = jnp.minimum(conv.iptop, icltop)

    # 2.  Equivalent specific humidity of clouds
    qcloud = state.specific_humidity[nl1]

    # 3. Stratiform clouds at the top of PBL
    clfact = 1.2
    rgse   = 1.0/(parameters.shortwave_radiation.gse_s1 - parameters.shortwave_radiation.gse_s0)

    # Fourth for loop (Two Loops)
    # 2. Stratocumulus clouds over sea and land
    fstab = smooth_clip01(rgse * (gse - parameters.shortwave_radiation.gse_s0), w_cov)
    # Stratocumulus clouds over sea
    clstr = fstab * smooth_pos(parameters.shortwave_radiation.clsmax - clfact * cloudc, w_cov)
    # Stratocumulus clouds over land
    clstrl = smooth_max(clstr, parameters.shortwave_radiation.clsminl, w_cov) * humidity.rh[kx - 1]
    clstr = clstr + terrain.fmask * (clstrl - clstr)
    # Cloud cover is a fraction: cap at 1. The land-branch RH amplification
    # above is unbounded when the lowest layer supersaturates (rh > 1), which
    # the thin surface layers of high-nlev grids do routinely over cold land
    # (clstr up to ~5-7 over wintertime Antarctica at nlev >= 24). The
    # shortwave scheme applies this cover as a reflectivity albcls*clstr whose
    # layer transmission (1 - albcls*clstr) turns negative for
    # clstr > 1/albcls = 2 — numerically explosive (it flips the sign of the
    # downward flux and NaNs the integration within hours). On the validated
    # 8-level grid clstr never exceeds clsmax = 0.6, so this cap is a no-op
    # there.
    clstr = smooth_min(clstr, 1.0, w_cov)

    swrad_out = physics_data.shortwave_rad.copy(gse=gse, icltop=icltop, cloudc=cloudc, cloudstr=clstr, qcloud=qcloud)
    physics_data = physics_data.copy(shortwave_rad=swrad_out)

    # This function doesn't directly produce tendencies
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)

    return (state, physics_data, parameters, forcing, terrain, physics_tendencies)

@jit
def solar(tyear, speedy_coords: SpeedyCoords, csol=4.*solc):
    """Calculate the daily-average insolation at the top of the atmosphere as a function of latitude.
    
    Parameters
    ----------
    tyear : float
        Time as a fraction of the year (0-1, where 0 corresponds to January 1st at midnight).

    Returns
    -------
    topsr : array-like
        Daily-average insolation at the top of the atmosphere for each latitude band.

    """
    # Constants and precomputed values
    pigr = 2.0 * jnp.arcsin(1.0)
    alpha = 2.0 * pigr * tyear
    
    # Calculate declination angle and Earth-Sun distance factor
    ca1 = jnp.cos(alpha)
    sa1 = jnp.sin(alpha)
    ca2 = ca1**2 - sa1**2
    sa2 = 2.0 * sa1 * ca1
    ca3 = ca1 * ca2 - sa1 * sa2
    sa3 = sa1 * ca2 + sa2 * ca1

    decl = (0.006918 - 0.399912 * ca1 + 0.070257 * sa1 -
            0.006758 * ca2 + 0.000907 * sa2 -
            0.002697 * ca3 + 0.001480 * sa3)

    fdis = 1.000110 + 0.034221 * ca1 + 0.001280 * sa1 + 0.000719 * ca2 + 0.000077 * sa2

    cdecl = jnp.cos(decl)
    sdecl = jnp.sin(decl)
    tdecl = sdecl / cdecl

    # Compute daily-average insolation at the top of the atmosphere
    csolp = csol / pigr

    # Calculate the solar radiation at the top of the atmosphere for each latitude
    ch0 = jnp.clip(-tdecl * speedy_coords.sia / speedy_coords.coa, -1+epsilon, 1-epsilon) # Clip to prevent blowup of gradients
    h0 = jnp.arccos(ch0)
    sh0 = jnp.sin(h0)

    topsr = csolp * fdis * (h0 * speedy_coords.sia * sdecl + sh0 * speedy_coords.coa * cdecl)

    return topsr