import jax.numpy as jnp
def clouds(qa ,rh,precnv,precls,iptop,gse,fmask):
    import shortwave_constants as shconst
    import params as p 
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
    lsminl = 0.15  # Minimum stratiform cloud cover over land (for RH = 1)
    gse_s0  = 0.25  # Gradient of dry static energy corresponding to stratiform cloud cover = 0
    gse_s1  = 0.40  # Gradient of dry static energy corresponding to stratiform cloud cover = 1

    # icltop(p.ix,p.il)  # Cloud top level
    # cloudc(p.ix,p.il)  # Total cloud cover
    # clstr(p.ix,p.il)   # Stratiform cloud cover

    nl1  = p.kx-1
    nlp  = p.kx+1
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
    mask = rh[:, :, nl1-1] > rhcl1  # Create a mask where the condition is true

    cloudc = jnp.where(mask, rh[:, :, nl1-1] - rhcl1, 0.0)  # Compute cloudc values where the mask is true

    icltop = jnp.where(mask, nl1, nlp) # Assign icltop values based on the mask

    #Second for loop (three levels)
    # do k = 3, kx - 2
    #         do i = 1, ix
    #             do j = 1, il
    #                 drh = rh(i,j,k) - rhcl1
    #                 if (drh > cloudc(i,j) .and. qa(i,j,k) > qacl) then
    #                     cloudc(i,j) = drh
    #                     icltop(i,j) = k
    #                 end if
    #             end do
    #         end do
    #     end do

    #return icltop, cloudc, clstr