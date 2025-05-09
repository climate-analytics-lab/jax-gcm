import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
import jax
from jcm.geometry import Geometry
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.physical_constants import epssw, solc, epsilon
from jcm.physics import PhysicsTendency, PhysicsState
from jcm.physics_data import PhysicsData

@jit 
def pass_fxn(operand):
    return operand

@jit
def get_shortwave_rad_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:

    # if compute_shortwave is true, then compute shortwave radiation
    # otherwise return the same physics_data and empty tendencies
    tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)
    state, physics_data, parameters, boundaries, geometry, tendencies = jax.lax.cond(
        physics_data.shortwave_rad.compute_shortwave, 
        shortwave_rad_fluxes, 
        pass_fxn, 
        operand=(state, physics_data, parameters, boundaries, geometry, tendencies)
    )

    return tendencies, physics_data

@jit
def shortwave_rad_fluxes(operand):
    """
    psa(ix,il)       # Normalised surface pressure [p/p0]
    qa(ix,il,kx)     # Specific humidity [g/kg]
    icltop(ix,il)    # Cloud top level
    cloudc(ix,il)    # Total cloud cover
    clstr(ix,il)     # Stratiform cloud cover
    rsds(ix,il)    # Total downward flux of short-wave radiation at the surface
    rsns(ix,il)     # Net downward flux of short-wave radiation at the surface
    ftop(ix,il)     # Net downward flux of short-wave radiation at the top of the atmosphere
    dfabs(ix,il,kx) # Flux of short-wave radiation absorbed in each atmospheric layer
    """

    state, physics_data, parameters, boundaries, geometry, tendencies = operand

    kx, ix, il = state.temperature.shape
    dhs = geometry.dhs
    fsg = geometry.fsg

    psa = state.surface_pressure
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
    tau2 = jnp.zeros((kx, ix, il, 4))
    mask = icltop < kx
    clamped_icltop = jnp.clip(icltop, 0, tau2.shape[0] - 1).astype(int) # Clamp icltop - 1 to be within the valid index range for tau2
    # Start with tau2 and update the values where the mask is true
    tau2 = tau2.at[clamped_icltop, :, :, 2].set(
        jnp.where(mask, parameters.shortwave_radiation.albcl * cloudc, tau2[clamped_icltop, :, :, 2])
    )
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
    cloudy = jnp.arange(1, nl1)[:, jnp.newaxis, jnp.newaxis] >= icltop[jnp.newaxis]
    
    tau2 = tau2.at[1:nl1, :, :, 0].set(
        jnp.exp(-psaz[jnp.newaxis] * dhs[1:nl1, jnp.newaxis, jnp.newaxis] * (
            abs1[:, jnp.newaxis, jnp.newaxis] +
            parameters.shortwave_radiation.abswv1 * qa[1:nl1] +
            cloudy * acloud[jnp.newaxis]
        ))
    )

    abs1 = parameters.shortwave_radiation.absdry + parameters.shortwave_radiation.absaer*fsg[kx - 1]**2
    tau2 = tau2.at[kx-1,:,:,0].set(jnp.exp(-psaz*dhs[kx - 1]*(abs1 + parameters.shortwave_radiation.abswv1*qa[kx - 1])))
    tau2 = tau2.at[1:kx,:,:,1].set(
        jnp.exp(-psaz[jnp.newaxis]*dhs[1:kx, jnp.newaxis, jnp.newaxis]*parameters.shortwave_radiation.abswv2*qa[1:kx])
    )

    # 3. Shortwave downward flux
    # 3.1 Initialization of fluxes
    
    rsns = jnp.zeros((ix, il)) # Net downward flux of short-wave radiation at the surface
    dfabs = jnp.zeros((kx,ix,il)) # Flux of short-wave radiation absorbed in each atmospheric layer
    ftop = physics_data.shortwave_rad.fsol # Net downward flux of short-wave radiation at the top of the atmosphere

    flux_1, flux_2 = jnp.zeros((kx, ix, il)), jnp.zeros((kx, ix, il))
    flux_1 = flux_1.at[0].set(physics_data.shortwave_rad.fsol*fband1)
    flux_2 = flux_2.at[0].set(physics_data.shortwave_rad.fsol*fband2)

    # 3.2 Ozone and dry-air absorption in the stratosphere
    k = 0
    dfabs = dfabs.at[k].set(flux_1[k])
    flux_1 = flux_1.at[k].set(tau2[k, :, :, 0] * (flux_1[k] - physics_data.shortwave_rad.ozupp * psa))
    dfabs = dfabs.at[k].add(- flux_1[k])

    k = 1
    flux_1 = flux_1.at[k].set(flux_1[k - 1])
    dfabs = dfabs.at[k].set(flux_1[k])
    flux_1 = flux_1.at[k].set(tau2[k, :, :, 0] * (flux_1[k] - physics_data.shortwave_rad.ozone * psa))
    dfabs = dfabs.at[k].add(- flux_1[k])
    
    # 3.3 Absorption and reflection in the troposphere
    # scan alert!
    # here's the function that will compute the flux
    propagate_flux_1 = lambda flux, tau: flux * tau[:,:,0] * (1 - tau[:,:,2])
    
    # scan over k = 2:kx
    _, flux_1_scan = lax.scan(
        jax.checkpoint(lambda carry, i: (propagate_flux_1(carry, i),)*2),
        flux_1[1], #initial value
        tau2[2:kx] #pass tau2 directly rather than indexing
    )
    
    # put results in flux_1
    flux_1 = flux_1.at[2:kx].set(flux_1_scan)

    # at each k, dfabs and tau2 only depend on the updated value of flux_1 and the non-updated value of tau2
    dfabs = dfabs.at[2:kx].set(flux_1[1:kx-1] * (1 - tau2[2:kx, :, :, 2]) * (1 - tau2[2:kx, :, :, 0]))
    tau2 = tau2.at[2:kx, :, :, 2].multiply(flux_1[1:kx-1])

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
    ablco2 = physics_data.mod_radcon.ablco2

    # Base absorptivities
    absorptivity = jnp.stack([
        parameters.shortwave_radiation.ablwin * jnp.ones_like(qa),
        ablco2 * jnp.ones_like(qa),
        parameters.shortwave_radiation.ablwv1 * qa,
        parameters.shortwave_radiation.ablwv2 * qa
    ], axis=-1)

    # Upper stratosphere (k = 0): no water vapor
    absorptivity = absorptivity.at[0, :, :, 2:].set(0)
    
    # Cloud-free layers: lower stratosphere (k = 1) and PBL (k = kx - 1)
    # Leave absorptivity unchanged

    # Cloudy layers: free troposphere (2 <= k <= kx - 2)
    acloud1, acloud2 = (cloudc[jnp.newaxis, :, :, jnp.newaxis]*a for a in (parameters.shortwave_radiation.ablcl1, parameters.shortwave_radiation.ablcl2))

    absorptivity = absorptivity.at[2:kx-1, :, :, 0].add(jnp.where(jnp.arange(2, kx-1)[:, jnp.newaxis, jnp.newaxis] > icltop[jnp.newaxis], acloud1[:, :, :, 0], acloud2[:, :, :, 0]))
    absorptivity = absorptivity.at[2:kx-1, :, :, 2:].set(jnp.maximum(absorptivity[2:kx-1, :, :, 2:], jnp.tile(acloud2, (kx-3, 1, 1, 2))))

    # Compute transmissivity
    tau2 = jnp.exp(-absorptivity*psa[jnp.newaxis, :, :, jnp.newaxis]*dhs[:, jnp.newaxis, jnp.newaxis, jnp.newaxis])
    
    # 5.2  Stratospheric correction terms
    eps1 = parameters.mod_radcon.epslw/(dhs[0] + dhs[1])
    stratc = jnp.zeros((ix, il, 2))
    stratc = stratc.at[:,:,0].set(physics_data.shortwave_rad.stratz*psa)
    stratc = stratc.at[:,:,1].set(eps1*psa)

    flux = physics_data.mod_radcon.flux.at[:,:,0].set(flux_1[0]).at[:,:,1].set(flux_2[kx-1])
    mod_radcon_out = physics_data.mod_radcon.copy(tau2=tau2, stratc=stratc, flux=flux)
    shortwave_rad_out = physics_data.shortwave_rad.copy(rsns=rsns, ftop=ftop, dfabs=dfabs, rsds=rsds)
    physics_data = physics_data.copy(shortwave_rad=shortwave_rad_out, mod_radcon=mod_radcon_out)

    # Get temperature tendency due to absorbed shortwave flux. Logic from physics.f90:160-162
    ttend_swr = dfabs*geometry.grdscp[:, jnp.newaxis, jnp.newaxis]/state.surface_pressure[jnp.newaxis] # physics.f90:160-162
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape, temperature=ttend_swr)

    return (state, physics_data, parameters, boundaries, geometry, physics_tendencies)


@jit
def get_zonal_average_fields(
    state: PhysicsState,
    physics_data: PhysicsData,
    boundaries: BoundaryData,
    geometry: Geometry
) -> PhysicsData:
    """
    Calculate zonal average fields including solar radiation, ozone depth, 
    and polar night cooling in the stratosphere using JAX.
    
    Parameters:
    tyear : float - physics_data.date.tyear
        Time as fraction of year (0-1, 0 = 1 Jan)

    Returns:
    fsol : jnp.ndarray
        Solar radiation at the top
    ozupp : jnp.ndarray
        Ozone depth in upper stratosphere
    ozone : jnp.ndarray
        Ozone concentration in lower stratosphere
    stratz : jnp.ndarray
        Polar night cooling in the stratosphere
    zenit : jnp.ndarray
        The Zenit angle
    """
    kx, ix, il = state.temperature.shape

    # Alpha = year phase (0 - 2pi, 0 = winter solstice = 22 Dec)
    alpha = 4.0 * jnp.arcsin(1.0) * (physics_data.date.tyear + 10.0 / 365.0)
    dalpha = 0.0

    coz1 = jnp.maximum(0.0, jnp.cos(alpha - dalpha))
    coz2 = 1.8

    azen = 1.0
    nzen = 2

    rzen = -jnp.cos(alpha) * 23.45 * jnp.arcsin(1.0) / 90.0

    fs0 = 6.0

    # Solar radiation at the top
    topsr = jnp.zeros(il)
    topsr = solar(physics_data.date.tyear,4*solc,geometry=geometry)

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

    fsol, ozupp, ozone, zenit, stratz = vectorized_compute_fields(geometry.sia, geometry.coa, topsr)

    swrad_out = physics_data.shortwave_rad.copy(fsol=fsol, ozupp=ozupp, ozone=ozone, zenit=zenit, stratz=stratz)
    physics_data = physics_data.copy(shortwave_rad=swrad_out)
    
    return physics_data

@jit
def get_clouds(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:

    # if compute_shortwave is true, then clouds
    # otherwise return the same physics_data and empty tendencies
    tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)
    state, physics_data, parameters, boundaries, geometry, tendencies = jax.lax.cond(
        physics_data.shortwave_rad.compute_shortwave, 
        clouds, 
        pass_fxn, 
        operand=(state, physics_data, parameters, boundaries, geometry, tendencies)
    )

    return tendencies, physics_data

@jit
def clouds(operand):
    """
    Simplified cloud cover scheme based on relative humidity and precipitation.

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

    state, physics_data, parameters, boundaries, geometry, tendencies = operand

    # Compute gradient of static energy: logic from physics.f90:147
    se = physics_data.convection.se
    phig = state.geopotential
    gse = (se[-2] - se[-1])/(phig[-2] - phig[-1])

    humidity = physics_data.humidity
    conv = physics_data.convection
    condensation = physics_data.condensation
    kx = state.temperature.shape[0]
    compute_shortwave = physics_data.shortwave_rad.compute_shortwave

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

    #First for loop (2 levels)
    mask = humidity.rh[nl1] > parameters.shortwave_radiation.rhcl1  # Create a mask where the condition is true
    cloudc = jnp.where(mask, humidity.rh[nl1] - parameters.shortwave_radiation.rhcl1, 0.0)  # Compute cloudc values where the mask is true
    icltop = jnp.where(mask, nl1, nlp) # Assign icltop values based on the mask

    #Second for loop (three levels)
    drh = humidity.rh[2:kx-2] - parameters.shortwave_radiation.rhcl1 # Calculate drh for the relevant range of k (2D slices of 3D array)
    mask = (drh > cloudc[jnp.newaxis]) & (state.specific_humidity[2:kx-2] > parameters.shortwave_radiation.qacl)  # Create a boolean mask where the conditions are met
    cloudc_update = jnp.where(mask, drh, cloudc[jnp.newaxis])  # Update cloudc where the mask is True
    cloudc = jnp.max(cloudc_update, axis=0)   # Only update cloudc when the condition is met; use np.max along axis 2

    # Update icltop where the mask is True
    k_indices = jnp.arange(2, kx-2)  # Generate the k indices (since range starts from 2)
    icltop_update = jnp.where(mask, k_indices[:, jnp.newaxis, jnp.newaxis], icltop[jnp.newaxis])  # Use the mask to update icltop only where the cloudc was updated
    icltop = jnp.where(cloudc[jnp.newaxis] == cloudc_update, icltop_update, icltop[jnp.newaxis]).max(axis=0)

    #Third for loop (two levels)
    # Perform the calculations (Two Loops)
    pr1 = jnp.minimum(parameters.shortwave_radiation.pmaxcl, 86.4 * (conv.precnv + condensation.precls))
    cloudc = jnp.minimum(1.0, parameters.shortwave_radiation.wpcl * jnp.sqrt(jnp.maximum(epsilon, pr1)) + jnp.minimum(1.0, cloudc * rrcl)**2.0)
    cloudc = jnp.where(jnp.isnan(cloudc), 1.0, cloudc)
    icltop = jnp.minimum(conv.iptop, icltop)

    # 2.  Equivalent specific humidity of clouds
    qcloud = state.specific_humidity[nl1]

    # 3. Stratiform clouds at the top of PBL
    clfact = 1.2
    rgse   = 1.0/(parameters.shortwave_radiation.gse_s1 - parameters.shortwave_radiation.gse_s0)

    #Fourth for loop (Two Loops)
    # 2. Stratocumulus clouds over sea and land
    fstab = jnp.clip(rgse * (gse - parameters.shortwave_radiation.gse_s0), 0.0, 1.0)
    # Stratocumulus clouds over sea
    clstr = fstab * jnp.maximum(parameters.shortwave_radiation.clsmax - clfact * cloudc, 0.0)
    # Stratocumulus clouds over land
    clstrl = jnp.maximum(clstr, parameters.shortwave_radiation.clsminl) * humidity.rh[kx - 1]
    clstr = clstr + boundaries.fmask_l * (clstrl - clstr)

    swrad_out = physics_data.shortwave_rad.copy(gse=gse, icltop=icltop, cloudc=cloudc, cloudstr=clstr, qcloud=qcloud)
    physics_data = physics_data.copy(shortwave_rad=swrad_out)

    # This function doesn't directly produce tendencies
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)

    return (state, physics_data, parameters, boundaries, geometry, physics_tendencies)

@jit
def solar(tyear, csol=4.*solc, geometry: Geometry=None):
    """
    Calculate the daily-average insolation at the top of the atmosphere as a function of latitude.
    
    Parameters:
    tyear : float
        Time as a fraction of the year (0-1, where 0 corresponds to January 1st at midnight).

    Returns:
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
    ch0 = jnp.clip(-tdecl * geometry.sia / geometry.coa, -1+epsilon, 1-epsilon) # Clip to prevent blowup of gradients
    h0 = jnp.arccos(ch0)
    sh0 = jnp.sin(h0)

    topsr = csolp * fdis * (h0 * geometry.sia * sdecl + sh0 * geometry.coa * cdecl)

    return topsr