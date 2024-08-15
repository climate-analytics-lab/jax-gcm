import jax.numpy as jnp
import tree_math
@tree_math.struct
class SWRadiationData:
    qcloud: jnp.ndarray
    fsol: jnp.ndarray
    ozone: jnp.ndarray
    ozupp: jnp.ndarray
    zenit: jnp.ndarray
    stratz: jnp.ndarray
    
def clouds(qa ,rh,precnv,precls,iptop,gse,fmask):
    #import params as p 
    from jcm.params import kx 
    '''
    qa(ix,il,kx)   # Specific humidity [g/kg]
    rh(ix,il,kx)   # Relative humidity
    precnv(ix,il)  # Convection precipitation
    precls(ix,il)  # Large-scale condensational precipitation
    iptop(ix,il)
    gse(ix,il)     # Vertical gradient of dry static energy
    fmask(ix,il)   # Fraction land-sea mask
    icltop(ix,il)  # Cloud top level
    cloudc(ix,il)  # Total cloud cover
    clstr(ix,il)   # Stratiform cloud cover

    integer :: i, j, k, nl1, nlp
    real(kind=8) :: clfact, clstrl, drh, fstab, pr1, rgse, rrcl
    '''
    # these are all just for clouds - none are shared. clouds function doesn't need any shared variables (i dont think)
    # these should get moved to shortwave_radiation_clouds and this file can be deleted

    rhcl1   = 0.30  # Relative humidity threshold corresponding to cloud cover = 0
    rhcl2   = 1.00  # Relative humidity correponding to cloud cover = 1
    qacl    = 0.20  # Specific humidity threshold for cloud cover
    wpcl    = 0.2   # Cloud cover weight for the square-root of precipitation (for p = 1 mm/day)
    pmaxcl  = 10.0  # Maximum value of precipitation (mm/day) contributing to cloud cover
    clsmax  = 0.60  # Maximum stratiform cloud cover
    clsminl = 0.15  # Minimum stratiform cloud cover over land (for RH = 1)
    gse_s0  = 0.25  # Gradient of dry static energy corresponding to stratiform cloud cover = 0
    gse_s1  = 0.40  # Gradient of dry static energy corresponding to stratiform cloud cover = 1

    # icltop(p.ix,p.il)  # Cloud top level (make sure returns integer)
    # cloudc(p.ix,p.il)  # Total cloud cover
    # clstr(p.ix,p.il)   # Stratiform cloud cover

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