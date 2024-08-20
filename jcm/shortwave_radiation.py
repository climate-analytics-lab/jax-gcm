import jax.numpy as jnp
from jax import jit
from jax import vmap
from jcm.physical_constants import epssw
from jcm.params import il, ix
from jcm.physics import PhysicsData, PhysicsTendency, PhysicsState
from jcm.geometry import sia, coa, fmask
from jcm.date import tyear # maybe this can come from somewhere else? like the model instance tracks it? it comes from date.f90 in speedy
import tree_math

@tree_math.struct
class SWRadiationData:
    qcloud = jnp.zeros((ix,il))
    fsol= jnp.zeros((ix,il))
    ozone = jnp.zeros((ix,il))
    ozupp = jnp.zeros((ix,il))
    zenit = jnp.zeros((ix,il))
    stratz = jnp.zeros((ix,il))
    gse = jnp.zeros((ix,il))
    icltop = jnp.zeros((ix,il))
    cloudc = jnp.zeros((ix,il))
    cloudstr = jnp.zeros((ix,il))

@jit
def get_zonal_average_fields(physics_data: PhysicsData, state: PhysicsState):
    """
    Calculate zonal average fields including solar radiation, ozone depth, 
    and polar night cooling in the stratosphere using JAX.
    
    Parameters:
    tyear : float - where should this be defined? 
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
    topsr = solar(tyear)
    
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

    swrad_out = SWRadiationData()
    swrad_out = physics_data.shortwave_rad
    swrad_out.fsol = fsol
    swrad_out.ozupp = ozupp
    swrad_out.ozone = ozone
    swrad_out.zenit = zenit
    swrad_out.stratz = stratz

    physics_data = PhysicsData(swrad_out, physics_data.convection, physics_data.modradcon, physics_data.humidity, physics_data.condensation)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_tendencies, physics_data
    

def clouds(physics_data: PhysicsData, state: PhysicsState):
    #import params as p 
    from jcm.params import kx 
    '''
    Simplified cloud cover scheme based on relative humidity and precipitation.

    Args:
        qa: Specific humidity [g/kg] - PhysicsData.Convection
        rh: Relative humidity - PhysicsData.Humidity
        precnv: Convection precipitation - PhysicsData.Convection
        precls: Large-scale condensational precipitation - PhysicsData.Condensation
        iptop: Cloud top level - PhysicsData.Convection
        gse: Vertical gradient of dry static energy - this gets created and set in phyiscs.f90 (line 147) - it is added to the shortwave rad dataclass 
                                                        but the setting needs to be done in physics.py or updated to be done inside this function
        fmask: Fraction land-sea mask - FIX ME! should this come from geometry.initialize()? It gets set on start up
                                        See boundaries.f90:38 fmask = load_boundary_file("surface.nc", "lsm")

    Returns:
        icltop: Cloud top level
        cloudc: Total cloud cover
        clstr: Stratiform cloud cover
        
    '''

    humidity = physics_data.humidity
    conv = physics_data.convection
    condensation = physics_data.condensation
    swrad = physics_data.shortwave_rad

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
    mask = (drh > cloudc[:, :, jnp.newaxis]) & (humidity.qa[:, :, 2:kx-2] > qacl)  # Create a boolean mask where the conditions are met
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
    qcloud = humidity.qa[:,:,nl1]

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
    clstr = clstr + fmask * (clstrl - clstr)

    swrad_out = SWRadiationData()
    swrad_out = swrad
    swrad_out.icltop = icltop
    swrad_out.cloudc = cloudc
    swrad_out.clstr = clstr
    swrad_out.qcloud = qcloud

    physics_data = PhysicsData(swrad_out, physics_data.convection, physics_data.modradcon, physics_data.humidity, physics_data.condensation)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_tendencies, physics_data

def solar(tyear):
    """
    Calculate the daily-average insolation at the top of the atmosphere as a function of latitude.
    
    Parameters:
    tyear : float
        Time as a fraction of the year (0-1, where 0 corresponds to January 1st at midnight).

    Returns:
    topsr : array-like
        Daily-average insolation at the top of the atmosphere for each latitude band.
    """
    from jcm.geometry import coa, sia
    csol = 1368.0
    
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