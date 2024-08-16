import jax.numpy as jnp
from jax import jit
from jax import vmap
from jcm.physical_constants import epssw
from jcm.params import il, ix
from jcm.geometry import sia, coa
import tree_math

@tree_math.struct
class SWRadiationData:
    qcloud: jnp.ndarray
    fsol: jnp.ndarray
    ozone: jnp.ndarray
    ozupp: jnp.ndarray
    zenit: jnp.ndarray
    stratz: jnp.ndarray

def get_shortwave_rad_fluxes(psa, qa, icltop, cloudc, clstr):
    from jcm.params import kx 
    from geometry import fsg, dhs
    import mod_radcon
    ''''
    psa(ix,il)       # Normalised surface pressure [p/p0]
    qa(ix,il,kx)     # Specific humidity [g/kg]
    icltop(ix,il)    # Cloud top level
    cloudc(ix,il)    # Total cloud cover
    clstr(ix,il)     # Stratiform cloud cover
    fsfcd(ix,il)    # Total downward flux of short-wave radiation at
                                            # the surface
    fsfc(ix,il)     # Net downward flux of short-wave radiation at the
                                            # surface
    ftop(ix,il)     # Net downward flux of short-wave radiation at the
                                            # top of the atmosphere
    dfabs(ix,il,kx) # Flux of short-wave radiation absorbed in each
                                            # atmospheric layer
    '''
    nl1 = kx - 1

    fband2 = 0.05
    fband1 = 1.0 - fband2
    #  Initialization
    tau2 = 0.0

    #return fsfcd, fsfc, ftop, dfabs

    #     integer :: i, j, k
    #     real(kind=8) :: acloud(ix,il), psaz(ix,il), abs1, acloud1, deltap, eps1

        

    #     ! 1.  Initialization
    #     tau2 = 0.0

    #     do i = 1, ix
    #         do j = 1, il
    #             if (icltop(i,j) <= kx) then
    #                 tau2(i,j,icltop(i,j),3) = albcl*cloudc(i,j)
    #             end if
    #             tau2(i,j,kx,3) = albcls*clstr(i,j)
    #         end do
    #     end do

    #     ! 2. Shortwave transmissivity:
    #     ! function of layer mass, ozone (in the statosphere),
    #     ! abs. humidity and cloud cover (in the troposphere)
    #     psaz = psa*zenit
    #     acloud = cloudc*min(abscl1*qcloud, abscl2)
    #     tau2(:,:,1,1) = exp(-psaz*dhs(1)*absdry)

    #     do k = 2, nl1
    #         abs1 = absdry + absaer*fsg(k)**2

    #         do i = 1, ix
    #             do j = 1, il
    #                 if (k >= icltop(i,j)) then
    #                     tau2(i,j,k,1) = exp(-psaz(i,j)*dhs(k)*(abs1 + abswv1*qa(i,j,k) + acloud(i,j)))
    #                 else
    #                     tau2(i,j,k,1) = exp(-psaz(i,j)*dhs(k)*(abs1 + abswv1*qa(i,j,k)))
    #                 end if
    #             end do
    #         end do
    #     end do

    #     abs1 = absdry + absaer*fsg(kx)**2
    #     tau2(:,:,kx,1) = exp(-psaz*dhs(kx)*(abs1 + abswv1*qa(:,:,kx)))

    #     do k = 2, kx
    #         tau2(:,:,k,2) = exp(-psaz*dhs(k)*abswv2*qa(:,:,k))
    #     end do

    #     ! 3. Shortwave downward flux
    #     ! 3.1 Initialization of fluxes
    #     ftop = fsol
    #     flux(:,:,1) = fsol*fband1
    #     flux(:,:,2) = fsol*fband2

    #     ! 3.2 Ozone and dry-air absorption in the stratosphere
    #     k = 1
    #     dfabs(:,:,k) = flux(:,:,1)
    #     flux(:,:,1)  = tau2(:,:,k,1)*(flux(:,:,1) - ozupp*psa)
    #     dfabs(:,:,k) = dfabs(:,:,k) - flux(:,:,1)

    #     k = 2
    #     dfabs(:,:,k) = flux(:,:,1)
    #     flux(:,:,1)  = tau2(:,:,k,1)*(flux(:,:,1) - ozone*psa)
    #     dfabs(:,:,k) = dfabs(:,:,k) - flux(:,:,1)

    #     ! 3.3  Absorption and reflection in the troposphere
    #     do k = 3, kx
    #         tau2(:,:,k,3) = flux(:,:,1)*tau2(:,:,k,3)
    #         flux (:,:,1)  = flux(:,:,1) - tau2(:,:,k,3)
    #         dfabs(:,:,k)  = flux(:,:,1)
    #         flux (:,:,1)  = tau2(:,:,k,1)*flux(:,:,1)
    #         dfabs(:,:,k)  = dfabs(:,:,k) - flux(:,:,1)
    #     end do

    #     do k = 2, kx
    #         dfabs(:,:,k) = dfabs(:,:,k) + flux(:,:,2)
    #         flux(:,:,2)  = tau2(:,:,k,2)*flux(:,:,2)
    #         dfabs(:,:,k) = dfabs(:,:,k) - flux(:,:,2)
    #     end do

    #     ! 4. Shortwave upward flux
    #     ! 4.1  Absorption and reflection at the surface
    #     fsfcd       = flux(:,:,1) + flux(:,:,2)
    #     flux(:,:,1) = flux(:,:,1)*albsfc
    #     fsfc        = fsfcd - flux(:,:,1)

    #     ! 4.2  Absorption of upward flux
    #     do k=kx,1,-1
    #         dfabs(:,:,k) = dfabs(:,:,k) + flux(:,:,1)
    #         flux(:,:,1)  = tau2(:,:,k,1)*flux(:,:,1)
    #         dfabs(:,:,k) = dfabs(:,:,k) - flux(:,:,1)
    #         flux(:,:,1)  = flux(:,:,1) + tau2(:,:,k,3)
    #     end do

    #     ! 4.3  Net solar radiation = incoming - outgoing
    #     ftop = ftop - flux(:,:,1)

    #     ! 5.  Initialization of longwave radiation model
    #     ! 5.1  Longwave transmissivity:
    #     ! function of layer mass, abs. humidity and cloud cover.

    #     ! Cloud-free levels (stratosphere + PBL)
    #     k = 1
    #     tau2(:,:,k,1) = exp(-psa*dhs(k)*ablwin)
    #     tau2(:,:,k,2) = exp(-psa*dhs(k)*ablco2)
    #     tau2(:,:,k,3) = 1.0
    #     tau2(:,:,k,4) = 1.0

    #     do k = 2, kx, kx - 2
    #         tau2(:,:,k,1) = exp(-psa*dhs(k)*ablwin)
    #         tau2(:,:,k,2) = exp(-psa*dhs(k)*ablco2)
    #         tau2(:,:,k,3) = exp(-psa*dhs(k)*ablwv1*qa(:,:,k))
    #         tau2(:,:,k,4) = exp(-psa*dhs(k)*ablwv2*qa(:,:,k))
    #     end do

    #     ! Cloudy layers (free troposphere)
    #     acloud = cloudc * ablcl2

    #     do k = 3, nl1
    #         do i = 1, ix
    #             do j = 1, il
    #                  deltap = psa(i,j)*dhs(k)

    #                  if (k < icltop(i,j)) then
    #                    acloud1 = acloud(i,j)
    #                  else
    #                    acloud1 = ablcl1*cloudc(i,j)
    #                  endif

    #                  tau2(i,j,k,1) = exp(-deltap*(ablwin+acloud1))
    #                  tau2(i,j,k,2) = exp(-deltap*ablco2)
    #                  tau2(i,j,k,3) = exp(-deltap*max(ablwv1*qa(i,j,k), acloud(i,j)))
    #                  tau2(i,j,k,4) = exp(-deltap*max(ablwv2*qa(i,j,k), acloud(i,j)))
    #             end do
    #         end do
    #     end do

    #     ! 5.2  Stratospheric correction terms
    #     eps1 = epslw/(dhs(1) + dhs(2))
    #     stratc(:,:,1) = stratz*psa
    #     stratc(:,:,2) = eps1*psa
    # end



@jit
def get_zonal_average_fields(tyear):
    """
    Calculate zonal average fields including solar radiation, ozone depth, 
    and polar night cooling in the stratosphere using JAX.
    
    Parameters:
    tyear : float
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

    return fsol, ozupp, ozone, zenit, stratz
    

def clouds(qa ,rh,precnv,precls,iptop,gse,fmask):
    #import params as p 
    from jcm.params import kx 
    '''
    Simplified cloud cover scheme based on relative humidity and precipitation.

    Args:
        qa: Specific humidity [g/kg]
        rh: Relative humidity
        precnv: Convection precipitation
        precls: Large-scale condensational precipitation
        iptop: Cloud top level
        gse: Vertical gradient of dry static energy
        fmask: Fraction land-sea mask

    Returns:
        icltop: Cloud top level
        cloudc: Total cloud cover
        clstr: Stratiform cloud cover
        
    '''

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
    mask = rh[:, :, nl1] > rhcl1  # Create a mask where the condition is true
    cloudc = jnp.where(mask, rh[:, :, nl1] - rhcl1, 0.0)  # Compute cloudc values where the mask is true
    icltop = jnp.where(mask, nl1, nlp) # Assign icltop values based on the mask

    #Second for loop (three levels)
    drh = rh[:, :, 2:kx-2] - rhcl1 # Calculate drh for the relevant range of k (2D slices of 3D array)
    mask = (drh > cloudc[:, :, jnp.newaxis]) & (qa[:, :, 2:kx-2] > qacl)  # Create a boolean mask where the conditions are met
    cloudc_update = jnp.where(mask, drh, cloudc[:, :, jnp.newaxis])  # Update cloudc where the mask is True
    cloudc = jnp.max(cloudc_update, axis=2)   # Only update cloudc when the condition is met; use np.max along axis 2

    # Update icltop where the mask is True
    k_indices = jnp.arange(2, kx-2)  # Generate the k indices (since range starts from 2)
    icltop_update = jnp.where(mask, k_indices, icltop[:, :, jnp.newaxis])  # Use the mask to update icltop only where the cloudc was updated
    icltop = jnp.where(cloudc[:, :, jnp.newaxis] == cloudc_update, icltop_update, icltop[:, :, jnp.newaxis]).max(axis=2)

    #Third for loop (two levels)
    # Perform the calculations (Two Loops)
    pr1 = jnp.minimum(pmaxcl, 86.4 * (precnv + precls))
    cloudc = jnp.minimum(1.0, wpcl * jnp.sqrt(pr1) + jnp.minimum(1.0, cloudc * rrcl)**2.0)
    cloudc = jnp.where(jnp.isnan(cloudc), 1.0, cloudc)
    icltop = jnp.minimum(iptop, icltop)

    # 2.  Equivalent specific humidity of clouds
    qcloud = qa[:,:,nl1]

    # 3. Stratiform clouds at the top of PBL
    clfact = 1.2
    rgse   = 1.0/(gse_s1 - gse_s0)

    #Fourth for loop (Two Loops)
    # 2. Stratocumulus clouds over sea and land
    fstab = jnp.clip(rgse * (gse - gse_s0), 0.0, 1.0)
    # Stratocumulus clouds over sea
    clstr = fstab * jnp.maximum(clsmax - clfact * cloudc, 0.0)
    # Stratocumulus clouds over land
    clstrl = jnp.maximum(clstr, clsminl) * rh[:, :, kx - 1]
    clstr = clstr + fmask * (clstrl - clstr)

    return icltop, cloudc, clstr, qcloud

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

