import jax.numpy as jnp
def clouds(qa ,rh,precnv,precls,iptop,gse,fmask,icltop,cloudc,clstr):
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

    nl1  = p.kx-1
    nlp  = p.kx+1
    rrcl = 1./(shconst.rhcl2-shconst.rhcl1)

    # 1.  Cloud cover, defined as the sum of:
    #     - a term proportional to the square-root of precip. rate
    #     - a quadratic function of the max. relative humidity
    #       in tropospheric layers above PBL where Q > QACL :
    #       ( = 0 for RHmax < RHCL1, = 1 for RHmax > RHCL2 )
    #     Cloud-top level: defined as the highest (i.e. least sigma)
    #       between the top of convection/condensation and
    #       the level of maximum relative humidity.

    # do i = 1, ix
    #         do j = 1, il
    #             if (rh(i,j,nl1) > rhcl1) then
    #                 cloudc(i,j) = rh(i,j,nl1) - rhcl1
    #                 icltop(i,j) = nl1
    #             else
    #                 cloudc(i,j) = 0.0
    #                 icltop(i,j) = nlp
    #             end if
    #         end do
    #     end do