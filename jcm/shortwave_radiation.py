import jax.numpy as jnp
from jax import jit
from jax import vmap
from jcm.physical_constants import epssw, solc
from jcm.physics import PhysicsTendency, PhysicsState
from jcm.physics_data import PhysicsData
from jcm.geometry import sia, coa, fsg, dhs
from jcm.mod_radcon import epslw
from jax import lax

# @jit
def get_shortwave_rad_fluxes(state: PhysicsState, physics_data: PhysicsData):
    ''''
    psa(ix,il)       # Normalised surface pressure [p/p0]
    qa(ix,il,kx)     # Specific humidity [g/kg]
    icltop(ix,il)    # Cloud top level
    cloudc(ix,il)    # Total cloud cover
    clstr(ix,il)     # Stratiform cloud cover
    rsds(ix,il)    # Total downward flux of short-wave radiation at the surface
    ssr(ix,il)     # Net downward flux of short-wave radiation at the surface
    ftop(ix,il)     # Net downward flux of short-wave radiation at the top of the atmosphere
    dfabs(ix,il,kx) # Flux of short-wave radiation absorbed in each atmospheric layer
    '''

    ix, il, kx = state.temperature.shape
    psa = physics_data.convection.psa
    qa = state.specific_humidity
    icltop = physics_data.shortwave_rad.icltop
    cloudc = physics_data.shortwave_rad.cloudc
    clstr = physics_data.shortwave_rad.cloudstr

    # mod_radcon inputs
    albsfc = physics_data.mod_radcon.albsfc

    # Shortwave radiation and cloud constants
    albcl   = 0.43  # Cloud albedo (for cloud cover = 1)
    albcls  = 0.50  # Stratiform cloud albedo (for st. cloud cover = 1)
    
    # Shortwave absorptivities (for dp = 10^5 Pa)
    absdry =  0.033 # Absorptivity of dry air (visible band)
    absaer =  0.033 # Absorptivity of aerosols (visible band)
    abswv1 =  0.022 # Absorptivity of water vapour
    abswv2 = 15.000 # Absorptivity of water vapour
    abscl1 =  0.015 # Absorptivity of clouds (visible band, maximum value)
    abscl2 =  0.15  # Absorptivity of clouds

    # Longwave absorptivities (for dp = 10^5 Pa)
    ablwin = 0.3   # Absorptivity of air in "window" band
    ablco2 = 6.0   # Absorptivity of air in CO2 band
    ablwv1 = 0.7   # Absorptivity of water vapour in H2O band 1 (weak) (for dq = 1 g/kg)
    ablwv2 = 50.0  # Absorptivity of water vapour in H2O band 2 (strong) (for dq = 1 g/kg)
    ablcl1 = 12.0  # Absorptivity of "thick" clouds in window band (below cloud top)
    ablcl2 = 0.6   # Absorptivity of "thin" upper clouds in window and H2O bands

    nl1 = kx - 1

    fband2 = 0.05
    fband1 = 1.0 - fband2

    #  Initialization
    tau2 = jnp.zeros((ix, il, kx, 4))
    mask = icltop < kx  
    clamped_icltop = jnp.clip(icltop, 0, tau2.shape[2] - 1).astype(int) # Clamp icltop - 1 to be within the valid index range for tau2
    tau2 = tau2.at[:, :, clamped_icltop, 2].set(
        jnp.where(mask, albcl * cloudc, tau2[:, :, clamped_icltop, 2])   # Start with tau2 and update the values where the mask is true
    )
    
    tau2 = tau2.at[:, :, kx - 1, 2].set(albcls * clstr)  # Update the tau2 values for the second condition (kx index) across the entire array

    # 2. Shortwave transmissivity:
    # function of layer mass, ozone (in the statosphere),
    # abs. humidity and cloud cover (in the troposphere)
    psaz = psa*physics_data.shortwave_rad.zenit
    acloud = cloudc*jnp.minimum(abscl1*physics_data.shortwave_rad.qcloud, abscl2)
    tau2 = tau2.at[:,:,0,0].set(jnp.exp(-psaz*dhs[0]*absdry))

    abs1 = absdry + absaer * fsg[1:nl1] ** 2
    cloudy = jnp.arange(1, nl1)[jnp.newaxis, jnp.newaxis, :] >= icltop[:, :, jnp.newaxis]
    
    tau2 = tau2.at[:, :, 1:nl1, 0].set(
        jnp.exp(-psaz[:, :, jnp.newaxis] * dhs[jnp.newaxis, jnp.newaxis, 1:nl1] * (
            abs1[jnp.newaxis, jnp.newaxis, :] +
            abswv1 * qa[:, :, 1:nl1] +
            cloudy * acloud[:, :, jnp.newaxis]
        ))
    )

    abs1 = absdry + absaer*fsg[kx - 1]**2
    tau2 = tau2.at[:,:,kx-1,0].set(jnp.exp(-psaz*dhs[kx - 1]*(abs1 + abswv1*qa[:,:,kx - 1])))

    tau2 = tau2.at[:,:,1:kx,1].set(jnp.exp(-psaz[:, :, jnp.newaxis]*dhs[jnp.newaxis, jnp.newaxis, 1:kx]*abswv2*qa[:,:,1:kx]))

    # 3. Shortwave downward flux
    # 3.1 Initialization of fluxes
    
    ssr = jnp.zeros((ix, il)) # Net downward flux of short-wave radiation at the surface
    dfabs = jnp.zeros((ix,il,kx)) # Flux of short-wave radiation absorbed in each atmospheric layer
    ftop = physics_data.shortwave_rad.fsol # Net downward flux of short-wave radiation at the top of the atmosphere

    flux_1, flux_2 = jnp.zeros((ix, il, kx)), jnp.zeros((ix, il, kx))
    flux_1 = flux_1.at[:,:,0].set(physics_data.shortwave_rad.fsol*fband1)
    flux_2 = flux_2.at[:,:,0].set(physics_data.shortwave_rad.fsol*fband2)

    # 3.2 Ozone and dry-air absorption in the stratosphere
    k = 0
    dfabs = dfabs.at[:, :, k].set(flux_1[:, :, k])
    flux_1 = flux_1.at[:, :, k].set(tau2[:, :, k, 0] * (flux_1[:, :, k] - physics_data.shortwave_rad.ozupp * psa))
    dfabs = dfabs.at[:, :, k].add(- flux_1[:, :, k])

    k = 1
    flux_1 = flux_1.at[:, :, k].set(flux_1[:, :, k - 1])
    dfabs = dfabs.at[:, :, k].set(flux_1[:, :, k])
    flux_1 = flux_1.at[:, :, k].set(tau2[:, :, k, 0] * (flux_1[:, :, k] - physics_data.shortwave_rad.ozone * psa))
    dfabs = dfabs.at[:, :, k].add(- flux_1[:, :, k])
    
    # 3.3 Absorption and reflection in the troposphere
    # scan alert!
    # here's the function that will compute the flux
    propagate_flux_1 = lambda flux, tau: flux * tau[:,:,0] * (1 - tau[:,:,2])
    
    # transpose because scan uses the first axis
    flux_1_t = jnp.moveaxis(flux_1, 2, 0)

    # scan over k = 2:kx
    _, flux_1_scan = lax.scan(
        lambda carry, i: (propagate_flux_1(carry, i),)*2, #scan wants a tuple of carry and output for the next iteration, I'm just returning the output for both?
        flux_1_t[1], #initial value
        jnp.moveaxis(tau2, 2, 0)[2:kx]) #pass tau2 directly rather than indexing
    
    # put results in flux_1
    flux_1 = flux_1.at[:,:,2:kx].set(
        jnp.moveaxis(flux_1_scan, 0, 2)
    )

    # at each k, dfabs and tau2 only depend on the updated value of flux_1 and the non-updated value of tau2
    dfabs = dfabs.at[:, :, 2:kx].set(flux_1[:, :, 1:kx-1] * (1 - tau2[:, :, 2:kx, 2]) * (1 - tau2[:, :, 2:kx, 0]))
    tau2 = tau2.at[:, :, 2:kx, 2].multiply(flux_1[:, :, 1:kx-1])

    flux_2 = flux_2.at[:,:,1].set(flux_2[:,:,0])
    propagate_flux_2 = lambda flux, tau: flux * tau[:, :, 1]
    flux_2_t = jnp.moveaxis(flux_2, 2, 0)
    _, flux_2_scan = lax.scan(
        lambda carry, i: (propagate_flux_2(carry, i),)*2,
        flux_2_t[1],
        jnp.moveaxis(tau2, 2, 0)[1:kx])
    flux_2 = flux_2.at[:,:,1:kx].set(
        jnp.moveaxis(flux_2_scan, 0, 2)
    )
    dfabs = dfabs.at[:,:,1:kx].add(flux_2[:,:,:kx-1]*(1 - tau2[:,:,1:kx,1])) # changed k to kx double check this

    # 4. Shortwave upward flux

    # 4.1  Absorption and reflection at the surface
    rsds = flux_1[:,:,kx-1] + flux_2[:,:,kx-1]
    flux_1 = flux_1.at[:,:,kx-1].multiply(albsfc)
    ssr = rsds - flux_1[:,:,kx-1]

    # 4.2  Absorption of upward flux

    propagate_flux_up = lambda flux, tau: flux * tau[:,:,0] + tau[:,:,2]
    _, flux_1_scan = lax.scan(
        lambda carry, tau: (propagate_flux_up(carry, tau),) * 2,
        jnp.moveaxis(flux_1, 2, 0)[-1],
        jnp.moveaxis(tau2, 2, 0)[1:kx][::-1]
    )
    flux_1 = flux_1.at[:, :, :-1].set(jnp.moveaxis(flux_1_scan[::-1], 0, 2))
        
    dfabs = dfabs.at[:,:,:].add(flux_1*(1 - tau2[:,:,:,0]))

    flux_1 = flux_1.at[:, :, 1:].set(flux_1[:, :, :-1])
    flux_1 = flux_1.at[:,:,0].set(tau2[:,:,0,0]*flux_1[:,:,0] + tau2[:,:,0,2])

    # 4.3  Net solar radiation = incoming - outgoing
    ftop = ftop - flux_1[:,:,0]

    # 5. Initialization of longwave radiation model
    # 5.1 Longwave transmissivity:
    # function of layer mass, abs. humidity and cloud cover.

    # Base absorptivities
    absorptivity = jnp.stack([
        ablwin * jnp.ones_like(qa),
        ablco2 * jnp.ones_like(qa),
        ablwv1 * qa,
        ablwv2 * qa
    ], axis=-1)

    # Upper stratosphere (k = 0): no water vapor
    absorptivity = absorptivity.at[:, :, 0, 2:].set(0)
    
    # Cloud-free layers: lower stratosphere (k = 1) and PBL (k = kx - 1)
    #   Leave absorptivity unchanged

    # Cloudy layers: free troposphere (2 <= k <= kx - 2)
    acloud1, acloud2 = (cloudc[:, :, jnp.newaxis, jnp.newaxis]*a for a in (ablcl1, ablcl2))

    absorptivity = absorptivity.at[:, :, 2:kx-1, 0].add(jnp.where(jnp.arange(2, kx-1)[jnp.newaxis, jnp.newaxis, :] > icltop[:, :, jnp.newaxis], acloud1.reshape(ix, il, 1), acloud2.reshape(ix, il, 1)))
    absorptivity = absorptivity.at[:, :, 2:kx-1, 2:].set(jnp.maximum(absorptivity[:, :, 2:kx-1, 2:], jnp.tile(acloud2, (1, 1, 5, 2))))

    # Compute transmissivity
    tau2 = jnp.exp(-absorptivity*psa[:, :, jnp.newaxis, jnp.newaxis]*dhs[jnp.newaxis, jnp.newaxis, :, jnp.newaxis])
    
    # 5.2  Stratospheric correction terms
    eps1 = epslw/(dhs[0] + dhs[1])
    stratc = jnp.zeros((ix, il, 2))
    stratc = stratc.at[:,:,0].set(physics_data.shortwave_rad.stratz*psa)
    stratc = stratc.at[:,:,1].set(eps1*psa)

    flux = physics_data.mod_radcon.flux.at[:,:,0].set(flux_1[:,:,0]).at[:,:,1].set(flux_2[:,:,kx-1])
    mod_radcon_out = physics_data.mod_radcon.copy(tau2=tau2, stratc=stratc, flux=flux)
    shortwave_rad_out = physics_data.shortwave_rad.copy(ssr=ssr, ftop=ftop, dfabs=dfabs, rsds=rsds)
    physics_data = physics_data.copy(shortwave_rad=shortwave_rad_out, mod_radcon=mod_radcon_out)

    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))

    return physics_tendencies, physics_data


# @jit
def get_zonal_average_fields(state: PhysicsState, physics_data: PhysicsData):
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
    ix, il, _ = state.temperature.shape

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
    topsr = solar(physics_data.date.tyear,4*solc)
    
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

        return jnp.full(ix, fsol_i_j), jnp.full(ix, ozupp_i_j), jnp.full(ix, ozone_i_j), jnp.full(ix, zenit_i_j), jnp.full(ix, stratz_i_j)

    vectorized_compute_fields = vmap(compute_fields, in_axes=0, out_axes=1)

    fsol, ozupp, ozone, zenit, stratz = vectorized_compute_fields(sia, coa, topsr)

    swrad_out = physics_data.shortwave_rad.copy(fsol=fsol, ozupp=ozupp, ozone=ozone, zenit=zenit, stratz=stratz)
    physics_data = physics_data.copy(shortwave_rad=swrad_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_tendencies, physics_data
    
# @jit
def clouds(state: PhysicsState, physics_data: PhysicsData):
    '''
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
        
    '''

    humidity = physics_data.humidity
    conv = physics_data.convection
    condensation = physics_data.condensation
    swrad = physics_data.shortwave_rad
    kx = state.temperature.shape[2]

    # Constants
    rhcl1   = 0.30  # Relative humidity threshold corresponding to cloud cover = 0
    rhcl2   = 1.00  # Relative humidity correponding to cloud cover = 1
    qacl    = 0.20  # Specific humidity threshold for cloud cover
    wpcl    = 0.2   # Cloud cover weight for the square-root of precipitation (for p = 1 mm/day)
    pmaxcl  = 10.0  # Maximum value of precipitation (mm/day) contributing to cloud cover
    clsmax  = 0.60  # Maximum stratiform cloud cover
    clsminl = 0.15  # Minimum stratiform cloud cover over land (for RH = 1)
    gse_s0  = 0.25  # Gradient of dry static energy corresponding to stratiform cloud cover = 0
    gse_s1  = 0.40  # Gradient of dry static energy corresponding to stratiform cloud cover = 1

    nl1  = kx-2
    nlp  = kx
    rrcl = 1./(rhcl2-rhcl1)

    # 1.  Cloud cover, defined as the sum of:
    #     - a term proportional to the square-root of precip. rate
    #     - a quadratic function of the max. relative humidity
    #       in tropospheric layers above PBL where Q > QACL :
    #       ( = 0 for RHmax < RHCL1, = 1 for RHmax > RHCL2 )
    #     Cloud-top level: defined as the highest (i.e. least sigma)
    #       between the top of convection/condensation and
    #       the level of maximum relative humidity.

    #First for loop (2 levels)
    mask = humidity.rh[:, :, nl1] > rhcl1  # Create a mask where the condition is true
    cloudc = jnp.where(mask, humidity.rh[:, :, nl1] - rhcl1, 0.0)  # Compute cloudc values where the mask is true
    icltop = jnp.where(mask, nl1, nlp) # Assign icltop values based on the mask

    #Second for loop (three levels)
    drh = humidity.rh[:, :, 2:kx-2] - rhcl1 # Calculate drh for the relevant range of k (2D slices of 3D array)
    mask = (drh > cloudc[:, :, jnp.newaxis]) & (state.specific_humidity[:, :, 2:kx-2] > qacl)  # Create a boolean mask where the conditions are met
    cloudc_update = jnp.where(mask, drh, cloudc[:, :, jnp.newaxis])  # Update cloudc where the mask is True
    cloudc = jnp.max(cloudc_update, axis=2)   # Only update cloudc when the condition is met; use np.max along axis 2

    # Update icltop where the mask is True
    k_indices = jnp.arange(2, kx-2)  # Generate the k indices (since range starts from 2)
    icltop_update = jnp.where(mask, k_indices, icltop[:, :, jnp.newaxis])  # Use the mask to update icltop only where the cloudc was updated
    icltop = jnp.where(cloudc[:, :, jnp.newaxis] == cloudc_update, icltop_update, icltop[:, :, jnp.newaxis]).max(axis=2)

    #Third for loop (two levels)
    # Perform the calculations (Two Loops)
    pr1 = jnp.minimum(pmaxcl, 86.4 * (conv.precnv + condensation.precls))
    cloudc = jnp.minimum(1.0, wpcl * jnp.sqrt(pr1) + jnp.minimum(1.0, cloudc * rrcl)**2.0)
    cloudc = jnp.where(jnp.isnan(cloudc), 1.0, cloudc)
    icltop = jnp.minimum(conv.iptop, icltop)

    # 2.  Equivalent specific humidity of clouds
    qcloud = state.specific_humidity[:,:,nl1]

    # 3. Stratiform clouds at the top of PBL
    clfact = 1.2
    rgse   = 1.0/(gse_s1 - gse_s0)

    #Fourth for loop (Two Loops)
    # 2. Stratocumulus clouds over sea and land
    fstab = jnp.clip(rgse * (swrad.gse - gse_s0), 0.0, 1.0)
    # Stratocumulus clouds over sea
    clstr = fstab * jnp.maximum(clsmax - clfact * cloudc, 0.0)
    # Stratocumulus clouds over land
    clstrl = jnp.maximum(clstr, clsminl) * humidity.rh[:, :, kx - 1]
    clstr = clstr + physics_data.surface_flux.fmask * (clstrl - clstr)

    swrad_out = physics_data.shortwave_rad.copy(icltop=icltop, cloudc=cloudc, cloudstr=clstr, qcloud=qcloud) 
    physics_data = physics_data.copy(shortwave_rad=swrad_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_tendencies, physics_data

# @jit
def solar(tyear, csol=4.0*solc):

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
    ch0 = jnp.clip(-tdecl * sia / coa, -1.0, 1.0)
    h0 = jnp.arccos(ch0)
    sh0 = jnp.sin(h0)

    topsr = csolp * fdis * (h0 * sia * sdecl + sh0 * coa * cdecl)

    return topsr