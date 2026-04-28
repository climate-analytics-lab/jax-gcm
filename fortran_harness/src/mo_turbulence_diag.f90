#include "fsel.inc"
!>
!! @brief Subroutines for computing turbulent exchange coefficients.
!!
!! @author Authors of ECHAM code
!! @author Hui Wan, MPI-M
!!
!! @par Revision History
!! Code originates from ECHAM5/6
!! Re-organized by Hui Wan (2010-09).
!! Changed to solve the total turbulent energy scheme by Felix Pithan and 
!! Thorsten Mauritsen (2016-01), and later revised and bug-fixed (2017-12).
!!
!! @par Copyright and License
!!
!! This code is subject to the DWD and MPI-M-Software-License-Agreement in
!! its most recent form.
!! Please see the file LICENSE in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the
!! headers of the routines.
!!
MODULE mo_turbulence_diag

  USE mo_kind,              ONLY: wp
  USE mo_convect_tables,    ONLY: prepare_ua_index_spline, lookup_ua_spline, &
    &                             compute_qsat
!!$  USE mo_echam_convect_tables, ONLY: prepare_ua_index_spline, lookup_ua_spline
  USE mo_echam_vdf_config,  ONLY: echam_vdf_config
  USE mo_echam_vdiff_params,ONLY: ckap, cb,cc, chneu, da1,                  &
    &                             eps_shear, eps_corio, totte_min, cons5,   &
    &                             lmix_max
  USE mo_physical_constants,ONLY: grav, rd, cpd, cpv, rd_o_cpd, rv,         &
    &                             vtmpc1, tmelt, alv, als, p0ref,           &
    &                             earth_angular_velocity

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: atm_exchange_coeff, sfc_exchange_coeff

CONTAINS
  !>
  !! Compute various thermodynamic variables for all (full) vertical levels;
  !! Diagnose PBL extension;
  !! Diagnose wind shear, buoyancy, Ri-number, mixing length, then compute
  !! the turbulent exchange coefficients of momentum, dry static energy,
  !! tracers, TTE, variance of virtual optential temperation at half levels
  !! [1+1/2, klev-1/2].
  !!
  !! Note that
  !! - for all coeffcient arrays, vertical index k in this subroutine
  !!   correspond to interface (half level) k+1/2;
  !! - the exchange coefficient at model top (level 1/2) is zero, thus does
  !!   not need computing;
  !! Therefore a number of variables are defined on what shall be referred to
  !! as "mid-levels", which are defined on levels with indices xmid(k) corre-
  !! sponding to xh(k+1), where x is some quantity at half-level k+1/2.
  !!
  !! @par Revision History
  !! Separated from vdiff.f90 of ECHAM6 and re-organized by Hui Wan (2010-09).
  !!  updated to echam-6.3.01 by Monika Esch (2014-11)
  !!
  SUBROUTINE atm_exchange_coeff( jg,                                      &! in
                               & jb,                                      &! in, for debugging only
                               & jcs, kproma, kbdim,                      &! in
                               & klev, klevm1, klevp1,     &! in
                               & pdtime, pcoriol,                         &! in
                               & pghf, pghh,                              &! in
                               & pum1, pvm1, ptm1, ptvm1,                 &! in
                               & pqm1, pxm1,                              &! in
                               & papm1, paphm1, paclc,                    &! in
                               & pustarm, pthvvar,                        &! in
                               & ptottem1,                                &! in
                               & pcptgz, phdtcbl,                         &! out
                               & pzthvvar, ptottevn,                      &! out
                               & pcfm, pcfh, pcfv, pcftotte, pcfthv,      &! out
                               & pprfac, ptheta_b, pthetav_b, pthetal_b,  &! out
                               & pqsat_b, plh_b,                          &! out
                               & pri, pmixlen                             )
    ! Arguments

    INTEGER, INTENT(IN) :: jg
    INTEGER, INTENT(IN) :: jb
    INTEGER, INTENT(IN) :: jcs, kproma, kbdim
    INTEGER, INTENT(IN) :: klev, klevm1, klevp1
    REAL(wp),INTENT(IN) :: pdtime
    REAL(wp),INTENT(IN) :: pcoriol(kbdim)
    REAL(wp),INTENT(IN) :: pghf(kbdim,klev)
    REAL(wp),INTENT(IN) :: pghh(kbdim,klevp1)
    REAL(wp),INTENT(IN) :: pxm1(kbdim,klev)
    REAL(wp),INTENT(IN) :: ptvm1(kbdim,klev)
    REAL(wp),INTENT(IN) :: pqm1(kbdim,klev),   pum1(kbdim,klev),  pvm1(kbdim,klev)
    REAL(wp),INTENT(IN) :: ptm1(kbdim,klev)
    REAL(wp),INTENT(IN) :: paclc(kbdim,klev)
    REAL(wp),INTENT(IN) :: papm1(kbdim,klev),  paphm1(kbdim,klevp1)
    REAL(wp),INTENT(IN) :: pthvvar(kbdim,klev)
    REAL(wp),INTENT(IN) :: pustarm(kbdim)

    REAL(wp),INTENT(IN) :: ptottem1 (kbdim,klev)

    REAL(wp),INTENT(OUT) :: phdtcbl (kbdim)        !< top height of dry convective boundary layer
    REAL(wp),INTENT(OUT) :: ptottevn(kbdim,klevm1) !< TTE at intermediate time step
    REAL(wp),INTENT(OUT) :: pcftotte(kbdim,klevm1) !< exchange coeff. for TTE
    REAL(wp),INTENT(OUT) :: pcfthv  (kbdim,klevm1) !< exchange coeff. for var. of theta_v
    REAL(wp),INTENT(OUT) :: pcfm    (kbdim,klevm1) !< exchange coeff. for u, v
    REAL(wp),INTENT(OUT) :: pcfh    (kbdim,klevm1) !< exchange coeff. for cptgz and tracers
    REAL(wp),INTENT(OUT) :: pcfv    (kbdim,klevm1) !< exchange coeff. for variance of qx
    REAL(wp),INTENT(OUT) :: pzthvvar(kbdim,klevm1) !< variance of theta_v at interm. step
    REAL(wp),INTENT(OUT) :: pcptgz  (kbdim,klev)   !< dry static energy
    REAL(wp),INTENT(OUT) :: pprfac  (kbdim,klevm1) !< prefactor for the exchange coeff.

    ! _b denotes the values at the bottom level (the klev-th full level)
    REAL(wp),INTENT(OUT) :: ptheta_b (kbdim)  !< potential temperature
    REAL(wp),INTENT(OUT) :: pthetav_b(kbdim)  !< virtual potential temperature
    REAL(wp),INTENT(OUT) :: pthetal_b(kbdim)  !< liquid (and ice?) potential temperature
    REAL(wp),INTENT(OUT) :: pqsat_b  (kbdim)  !< specific humidity at saturation
    REAL(wp),INTENT(OUT) :: plh_b    (kbdim)  !< latent heat

    ! Just for output
    REAL(wp),INTENT(OUT) :: pri     (kbdim,klevm1) !< moist Richardson number at mid-levels
    REAL(wp),INTENT(OUT) :: pmixlen (kbdim,klevm1) !< mixing length

    ! Local variables
    ! - Variables defined at full levels

    REAL(wp) :: zlh    (kbdim,klev)  !< latent heat at full levels
    REAL(wp) :: ztheta (kbdim,klev)  !< potential temperature
    REAL(wp) :: zthetav(kbdim,klev)  !< virtual potential temperature
    REAL(wp) :: zthetal(kbdim,klev)  !< liquid (and ice?) potential temperature
    REAL(wp) :: zqsat  (kbdim,klev)  !< specific humidity at saturation
    REAL(wp) :: km     (kbdim,klev)  !< turbulent viscosity
    REAL(wp) :: kh     (kbdim,klev)  !< turbulent conductivity

    ! - Variables defined at mid-levels

    REAL(wp) :: zlhmid  (kbdim,klevm1) !< latent heat at mid-levels
    REAL(wp) :: zdgmid  (kbdim,klevm1) !< geopotential height difference between two full levels

    REAL(wp) :: zccovermid (kbdim,klevm1), zqxmid     (kbdim,klevm1),       &
                zqmid      (kbdim,klevm1), zqsatmid   (kbdim,klevm1),       &
                zthetamid  (kbdim,klevm1), zthetavmid (kbdim,klevm1),       &
                ztmid      (kbdim,klevm1), ztvmid     (kbdim,klevm1)

    ! - 1D variables and scalars

    INTEGER  :: ihpbl (kbdim)        !< grid level index of PBL top
    INTEGER  :: ihpblc(kbdim),          ihpbld(kbdim),          idx(kbdim)
    INTEGER  :: jk, jl
    REAL(wp) :: za(kbdim),              zhdyn(kbdim)                          &
               ,zpapm1i(kbdim),         zua(kbdim)
    REAL(wp) :: f_tau, f_theta, e_kin, e_pot, lmix, imix_neutral, ldis, lmc
    REAL(wp) :: zalh2, zbuoy, zcor, zdisl, zdusq, zdvsq
    REAL(wp) :: zdqtot, zds, zdus1, zdus2, zdz
    REAL(wp) :: zes, zfox, zfux, zktest
    REAL(wp) :: zmult1, zmult2, zmult3, zmult4
    REAL(wp) :: zmult5, zqddif, zqtmid, zrdp
    REAL(wp) :: zrdrv, zri, zrvrd, zsdep1
    REAL(wp) :: zsdep2, zshear, zteldif, ztottesq
    REAL(wp) :: zthvprod, zthvdiss, zthvirdif
    REAL(wp) :: zucf, zusus1, zzb

    REAL(wp) :: zonethird

    ! Shortcuts to components of echam_vdf_config
    !
    REAL(wp), POINTER :: f_tau0, f_theta0, c_f, c_n, c_e, pr0, fbl
    !
    f_tau0   => echam_vdf_config(jg)% f_tau0
    f_theta0 => echam_vdf_config(jg)% f_theta0
    c_f      => echam_vdf_config(jg)% c_f
    c_n      => echam_vdf_config(jg)% c_n
    c_e      => echam_vdf_config(jg)% c_e
    pr0      => echam_vdf_config(jg)% pr0
    fbl      => echam_vdf_config(jg)% fbl

    !-------------------------------------
    ! 1. Some constants
    !-------------------------------------
    zrvrd     = rv/rd
    zrdrv     = rd/rv
    zonethird = 1._wp/3._wp

    !-------------------------------------
    ! 2. NEW THERMODYNAMIC VARIABLES
    !-------------------------------------

    DO 212 jk=1,klev
      ! Harness adapts to the older mo_echam_convect_tables signature
      ! that takes (jg, name, jcs, size, ...).
      CALL prepare_ua_index_spline(1,'vdiff (1)',jcs,kproma,ptm1(:,jk),idx,za       &
      &                                       ,klev=jk,kblock=jb,kblock_size=kbdim)
      CALL lookup_ua_spline(jcs,kproma,idx,za,zua)

      zpapm1i(jcs:kproma) = 1._wp/papm1(jcs:kproma,jk)
      ztheta(jcs:kproma,jk) = (p0ref*zpapm1i(jcs:kproma))**rd_o_cpd

      DO 211 jl=jcs,kproma

        ! Virtual dry static energy, potential temperature, virtual potential
        ! temperature

        pcptgz (jl,jk) = pghf(jl,jk)*grav+ptm1(jl,jk)*(cpd+(cpv-cpd)*pqm1(jl,jk))
        ztheta (jl,jk) = ptm1(jl,jk)*ztheta(jl,jk)
        zthetav(jl,jk) = ztheta(jl,jk)*(1._wp+vtmpc1*pqm1(jl,jk)-pxm1(jl,jk))

        ! Latent heat, liquid (and ice) potential temperature

        zlh(jl,jk)     = FSEL(ptm1(jl,jk)-tmelt,alv,als) ! latent heat
        zusus1         = zlh(jl,jk)/cpd*ztheta(jl,jk)/ptm1(jl,jk)*pxm1(jl,jk)
        zthetal(jl,jk) = ztheta(jl,jk)-zusus1

        zes=zua(jl)*zpapm1i(jl)              ! (sat. vapour pressure)*Rd/Rv/ps
        zes=MIN(zes,0.5_wp)
        zqsat(jl,jk)=zes/(1._wp-vtmpc1*zes)  ! specific humidity at saturation
211   END DO
212 END DO

    ! Copy bottom-level values to dummy arguments

    ptheta_b (jcs:kproma) = ztheta (jcs:kproma,klev)
    pthetav_b(jcs:kproma) = zthetav(jcs:kproma,klev)
    pthetal_b(jcs:kproma) = zthetal(jcs:kproma,klev)
    pqsat_b  (jcs:kproma) = zqsat  (jcs:kproma,klev)
    plh_b    (jcs:kproma) = zlh    (jcs:kproma,klev)

    !---------------------------------------------------------
    ! 3. Preparation for computation of exchange coefficients
    !---------------------------------------------------------
    ! Vertical interpolation from full levels to mid-levels
    ! using linear interpolation in pressure coordinate

    DO 214 jk=1,klevm1
      DO 213 jl=jcs,kproma
        zdgmid(jl,jk) = pghf(jl,jk) - pghf(jl,jk+1)

        ! interpolation coefficients
        zrdp   = 1._wp/(papm1(jl,jk) - papm1(jl,jk+1))
        zsdep1 = (papm1(jl,jk)  - paphm1(jl,jk+1))*zrdp
        zsdep2 = (paphm1(jl,jk+1)- papm1(jl,jk+1))*zrdp

        ! vertical interpolation for various variables
        zqsatmid(jl,jk)  =zsdep1*zqsat(jl,jk)+zsdep2*zqsat(jl,jk+1)
        ztmid(jl,jk)=zsdep1*ptm1(jl,jk)+zsdep2*ptm1(jl,jk+1)
        ztvmid(jl,jk)=zsdep1*ptvm1(jl,jk)+zsdep2*ptvm1(jl,jk+1)
        zthetavmid(jl,jk)=zsdep1*zthetav(jl,jk)+zsdep2*zthetav(jl,jk+1)
        zlhmid(jl,jk)=zsdep1*zlh(jl,jk)+zsdep2*zlh(jl,jk+1)
        zqxmid(jl,jk)=zsdep1*pxm1(jl,jk)+zsdep2*pxm1(jl,jk+1)
        zqmid(jl,jk)=zsdep1*pqm1(jl,jk)+zsdep2*pqm1(jl,jk+1)
        zthetamid(jl,jk)=zsdep1*ztheta(jl,jk)+zsdep2*ztheta(jl,jk+1)
        zccovermid(jl,jk)=paclc(jl,jk)*zsdep1+paclc(jl,jk+1)*zsdep2

        ! Air density at mid levels, p/(Tv*R)/dz = air density/dz, and the prefactor 
        ! that will be multiplied later to the exchange coeffcients to build a linear 
        ! algebraic equation set.
        pprfac(jl,jk) = paphm1(jl,jk+1)/(ztvmid(jl,jk)*rd*zdgmid(jl,jk))

213   END DO
214 END DO

    !------------------------------------------------
    ! 4. Compute convective dry boundary layer height
    !------------------------------------------------

    DO jl = jcs,kproma
      zcor=MAX(ABS(pcoriol(jl)),eps_corio)
      zhdyn(jl)=MIN(pghf(jl,1),chneu*pustarm(jl)/zcor)
      ihpblc(jl)=klev
      ihpbld(jl)=klev
    END DO

    DO jk=klevm1,1,-1
      DO jl=jcs,kproma
        zds=pcptgz(jl,jk)-pcptgz(jl,klev)
        zdz=pghf(jl,jk)-zhdyn(jl)
        ihpblc(jl)=MERGE(jk,ihpblc(jl),ihpblc(jl).EQ.klev.AND.zds.GT.0._wp)
        ihpbld(jl)=MERGE(jk,ihpbld(jl),ihpbld(jl).EQ.klev.AND.zdz.GE.0._wp)
      END DO
    END DO

    DO jl=jcs,kproma
      ihpbl (jl) = MIN(ihpblc(jl),ihpbld(jl))
      phdtcbl(jl) = pghf(jl,ihpblc(jl))                                        &
                  & -((pghf(jl,ihpblc(jl))-pghf(jl,ihpblc(jl)+1))              &
                  & /(pcptgz(jl,ihpblc(jl))-pcptgz(jl,ihpblc(jl)+1)))          &
                  & *(pcptgz(jl,ihpblc(jl))-pcptgz(jl,klev)) 

      IF( (pcptgz(jl,klevm1).GT.pcptgz(jl,klev)).AND.(ihpbld(jl).LT.ihpblc(jl))) THEN
         phdtcbl(jl)=0._wp
      END IF
    END DO

    !--------------------------------------------
    ! 5. Compute exchange coefficients
    !--------------------------------------------
    DO 372 jk=1,klevm1
      DO 361 jl=jcs,kproma

        ! gradient of specific humidity, wind shear, buoyancy, Ri-number
        ! according to Brinkop and Roeckner (1995, Tellus A)

        zqtmid=zqxmid(jl,jk)+zqmid(jl,jk)                               ! qt
        zfux=zlhmid(jl,jk)/(cpd*ztmid(jl,jk))                           ! L/(cpd*T)
        zfox=zlhmid(jl,jk)/(rd*ztmid(jl,jk))                            ! L/(Rd*T)
        zmult1=1._wp+vtmpc1*zqtmid                                      ! (1+0.61*qt) = 
                                                                        !  A in clear sky
        zmult2=zfux*zmult1-zrvrd
        zmult3=zrdrv*zfox*zqsatmid(jl,jk)/(1._wp+zrdrv*zfux*zfox*zqsatmid(jl,jk))
        zmult5=zmult1-zmult2*zmult3                                     ! A in cloud
        zmult4=zfux*zmult5-1._wp                                        ! D in cloud
        zdus1=zccovermid(jl,jk)*zmult5+(1._wp-zccovermid(jl,jk))*zmult1 ! A avg
        zdus2=zccovermid(jl,jk)*zmult4+(1._wp-zccovermid(jl,jk))*vtmpc1 ! D avg
        zteldif  =(zthetal(jl,jk)-zthetal(jl,jk+1))/zdgmid(jl,jk)       ! d theta_l
        zthvirdif=(zthetav(jl,jk)-zthetav(jl,jk+1))/zdgmid(jl,jk)       ! d theta_v
        zdqtot=(pqm1(jl,jk)+pxm1(jl,jk))-(pqm1(jl,jk+1)+pxm1(jl,jk+1))  ! d qt
        zqddif=zdqtot/zdgmid(jl,jk)                                     ! (d qt)/(d z)
        zbuoy  = (zteldif*zdus1+zthetamid(jl,jk)*zdus2*zqddif)            &
                *grav/zthetavmid(jl,jk)
        zdusq  = (pum1(jl,jk)-pum1(jl,jk+1))**2
        zdvsq  = (pvm1(jl,jk)-pvm1(jl,jk+1))**2
        zshear = (zdusq+zdvsq)/(zdgmid(jl,jk))**2
        zri    = zbuoy/MAX(zshear,eps_shear)

        pri(jl,jk) = zri          ! save for output

        ! stability functions for heat and momentum (Mauritsen et al. 2007) 

        IF(zri.GT.0._wp) THEN
           f_tau   = f_tau0*(0.25_wp+0.75_wp/(1._wp+4._wp*zri))
           f_theta = f_theta0/(1._wp+4._wp*zri)
        ELSE
           f_tau   = f_tau0
           f_theta = f_theta0
        END IF

        ! turbulent kinetic and turbulent potential energy

        IF(zri.GT.0._wp) THEN
           e_kin=ptottem1(jl,jk)/(1._wp+zri/(f_tau0**2/(2._wp*f_theta0**2)+3._wp*zri))
           e_pot=e_kin*zri/(f_tau0**2/(2._wp*f_theta0**2)+3._wp*zri)
        ELSE
           e_kin=ptottem1(jl,jk)/(1._wp+zri/(2._wp*zri-f_tau0**2/(2._wp*f_theta0**2)))
           e_pot=e_kin*zri/(2._wp*zri-f_tau0**2/(2._wp*f_theta0**2)) 
        END IF

        ! mixing length for neutral and stably stratified conditions

        imix_neutral =   1._wp/(ckap*pghh(jl,jk+1))                                            &
                     & + 2._wp*earth_angular_velocity/(c_f*SQRT(f_tau*e_kin))

        lmix =   imix_neutral + SQRT(MAX(0._wp,zbuoy))/(c_n*SQRT(f_tau*e_kin))
        lmix =   MIN(lmix_max,1._wp/lmix)

        ldis=lmix
        pmixlen(jl,jk)=lmix

        ! mixing coefficients 
        
        km(jl,jk)=f_tau**2*e_kin**2                                                       &
                 & /((c_e*e_kin*SQRT(ptottem1(jl,jk))/lmix)-grav/zthetavmid(jl,jk)        &
                 & *f_theta*SQRT(e_kin*2._wp*e_pot*abs(zbuoy)/(grav/zthetavmid(jl,jk))**2))
        kh(jl,jk)=2._wp*f_theta**2*e_kin*lmix/(c_e*SQRT(ptottem1(jl,jk)))

        ! convective bl mixing coefs, and merge in the upper half of convective boundary layer

        lmc =   imix_neutral + fbl/(ckap*(phdtcbl(jl)-pghh(jl,jk+1)))
        lmc =   1._wp/lmc

        IF(pghf(jl,jk).LE.0.5_wp*phdtcbl(jl)) THEN
           km(jl,jk)=f_tau0**2/c_e*lmc*SQRT(e_kin)
           kh(jl,jk)=km(jl,jk)/pr0
        ELSE IF (pghf(jl,jk).LE.phdtcbl(jl)) THEN
           km(jl,jk)=max(km(jl,jk),f_tau0**2/c_e*lmc*SQRT(e_kin))
           kh(jl,jk)=max(kh(jl,jk),km(jl,jk)/pr0)
        END IF

        ! introduce unstable stability function (Louis 79)

        IF (zri.LT.0._wp) THEN 
           zalh2=lmix*lmix

           zucf =   1._wp                                             &
                & /(1._wp+cons5*zalh2*SQRT(ABS(zri)*( ((pghf(jl,jk)/pghf(jl,jk+1))**zonethird-1._wp)         &
                &                                    /(pghf(jl,jk)-pghf(jl,jk+1))                    )**3    &
                &                                  /pghf(jl,jk+1)                                       ))

           kh(jl,jk)=kh(jl,jk)*(1._wp-3._wp*cb*zri*zucf)
           km(jl,jk)=km(jl,jk)*(1._wp-2._wp*cb*zri*zucf)
        END IF

        ! TTE at intermediate time step, obtained by solving a prognostic equation
        ! of TTE, the right-hand side of which consists of shear production,
        ! buoyancy production, and dissipation of TTE.
        ! See Appendix A in Brinkop and Roeckner (1995, Tellus) for the numerics.

        IF (zri.GT.0._wp) THEN 
           zzb=km(jl,jk)*zshear
        ELSE 
           zzb=km(jl,jk)*zshear-2._wp*kh(jl,jk)*zbuoy
        END IF
        zdisl=ldis/(c_e*pdtime)
        zktest=1._wp+(zzb*pdtime+SQRT(ptottem1(jl,jk))*2._wp)/zdisl
        IF (zktest.LE.1._wp) THEN
           ptottevn(jl,jk)=totte_min
        ELSE
           ptottevn(jl,jk)=MAX(totte_min,(zdisl*(SQRT(zktest)-1._wp))**2)
        END IF

        ! Square root of TTE at the old time step

        ztottesq = SQRT(MAX(totte_min,ptottem1(jl,jk)))

        ! Virtual potential temperature variance at intermediate step,
        ! obtained by solving a prognostic equation the variance, the right-hand side
        ! of which consists of a production term and dissipation.
        ! An explicit time stepping method is used here.

        zthvprod = 2._wp*kh(jl,jk)*zthvirdif**2        ! production rate
        zthvdiss = pthvvar(jl,jk)*ztottesq/(da1*lmix)    ! dissipation rate
        pzthvvar(jl,jk) = pthvvar(jl,jk)+(zthvprod-zthvdiss)*pdtime
        pzthvvar(jl,jk) = MAX(totte_min,pzthvvar(jl,jk))

        ! Exchange coefficients for
        ! - momentum  (variable pcfm),
        ! - heat and tracers (variable pcfh),
        ! - variance of hydrometeors (variable pcfv).

        pcfm(jl,jk) = km(jl,jk)
        pcfh(jl,jk) = kh(jl,jk)
        pcfv(jl,jk) = kh(jl,jk)*0.5_wp

        ! Exchange coefficients for
        ! - TTE (variable pfctotte), note that this deviates from Pithan et al. (2015)

        pcftotte(jl,jk) = km(jl,jk)
        pcfthv(jl,jk) = kh(jl,jk)

361   END DO
372 END DO

  END SUBROUTINE atm_exchange_coeff
  !-------------
  !>
  !!
  SUBROUTINE sfc_exchange_coeff( jg,                                     &! in
                               & jcs, kproma, kbdim, ksfc_type,          &! in
                               & idx_wtr, idx_ice, idx_lnd,              &! in
                               & pz0m, ptsfc,                            &! in
                               & pfrc, phdtcbl,                          &! in
                               & pocu, pocv, ppsfc,                      &! in
                               & pghf_b,                                 &! in
                               & pum1_b, pvm1_b,                         &! in
                               & ptm1_b,                                 &! in
                               & pqm1_b, pqxm1_b,                        &! in
                               & pqsat_b, plh_b,                         &! in
                               & ptheta_b, pthetav_b,                    &! in
                               & pthetal_b, paclc_b,                     &! in
                               & ptotte_b, pthvvar_b,                    &! in
                               & pthvsig_b,                              &! inout
                               & pwstar, pwstar_tile,                    &! inout
                               & pqsat_tile, pcpt_tile,                  &! out
                               & pri_gbm, pri_tile,                      &! out
                               & pcfm_gbm, pcfm_tile,                    &! out
                               & pcfh_gbm, pcfh_tile,                    &! out
                               & pcfv_sfc,                               &! out
                               & pcftotte_sfc, pcfthv_sfc,               &! out
                               & pprfac_sfc,                             &! out
                               & ptottevn_sfc, pthvvar_sfc,              &! out
                               & pustarm,                                &! out
                               & pch_tile,                               &! out
                               & pbn_tile, pbhn_tile, pbm_tile, pbh_tile,&! out
                               & paz0lh,                                 &! in, optional
                               & pcsat, pcair                            &! in, optional
                               & )

    INTEGER, INTENT(IN) :: jg
    INTEGER, INTENT(IN) :: jcs, kproma, kbdim
    INTEGER, INTENT(IN) :: ksfc_type, idx_wtr, idx_ice, idx_lnd

    REAL(wp),INTENT(IN) :: pz0m     (kbdim,ksfc_type) !< aerodynamic roughness length
    REAL(wp),INTENT(IN) :: ptsfc    (kbdim,ksfc_type) !< temp. at surface
    REAL(wp),INTENT(IN) :: pfrc     (kbdim,ksfc_type) !< fraction of the grid box occupied
                                                      !< by each surface type
    REAL(wp),INTENT(IN) :: phdtcbl  (kbdim)  !< height of the top of the atmospheric dry convective boundary layer
    REAL(wp),INTENT(IN) :: pocu     (kbdim)  !< ocean surface velocity
    REAL(wp),INTENT(IN) :: pocv     (kbdim)  !< ocean surface velocity
    REAL(wp),INTENT(IN) :: ppsfc    (kbdim)  !< surface pressure

    ! "_b" denotes value at the bottom level (the klev-th full level)

    REAL(wp),INTENT(IN) :: pghf_b   (kbdim)  !< geopot. height above ground
    REAL(wp),INTENT(IN) :: pum1_b   (kbdim)  !< u-wind
    REAL(wp),INTENT(IN) :: pvm1_b   (kbdim)  !< v-wind
    REAL(wp),INTENT(IN) :: ptm1_b   (kbdim)  !< temperature
    REAL(wp),INTENT(IN) :: pqm1_b   (kbdim)  !< specific humidity
    REAL(wp),INTENT(IN) :: pqxm1_b  (kbdim)  !< total concentration of hydrometeors
    REAL(wp),INTENT(IN) :: pqsat_b  (kbdim)  !< saturation specific humidity
    REAL(wp),INTENT(IN) :: plh_b    (kbdim)  !< latent heat
    REAL(wp),INTENT(IN) :: ptheta_b (kbdim)  !< potential temp.
    REAL(wp),INTENT(IN) :: pthetav_b(kbdim)  !< virtual potential temp.
    REAL(wp),INTENT(IN) :: pthetal_b(kbdim)  !< liquid water (?) pot. temp.
    REAL(wp),INTENT(IN) :: paclc_b  (kbdim)  !< cloud cover at lowest model level
    REAL(wp),INTENT(IN) :: ptotte_b (kbdim)  !< TTE 

    ! For the variance of theta_v, "_b" denotes the lowest computational level
    ! above surface, i.e., the interface between full levels klev-1 and klev.

    REAL(wp),INTENT(IN) :: pthvvar_b (kbdim)  !< variance of theta_v

    ! "_tile" denotes value at surface

    REAL(wp),INTENT(OUT) :: pqsat_tile (kbdim,ksfc_type) !< saturation specific humidity 
                                                         !<  at surface
    REAL(wp),INTENT(OUT) :: pcpt_tile  (kbdim,ksfc_type) !< dry static energy
    REAL(wp),INTENT(OUT) :: pri_gbm    (kbdim)           !< moist Richardson number
    REAL(wp),INTENT(OUT) :: pri_tile   (kbdim,ksfc_type) !< moist Richardson number

    REAL(wp),INTENT(OUT) :: pcfm_gbm   (kbdim)           !< exchange coeff. of momentum
    REAL(wp),INTENT(OUT) :: pcfm_tile  (kbdim,ksfc_type) !< exchange coeff. of momentum,
                                                         !< for each type of surface
    REAL(wp),INTENT(OUT) :: pcfh_gbm   (kbdim)           !< exchange coeff. of heat and 
                                                         !<  vapor
    REAL(wp),INTENT(OUT) :: pcfh_tile  (kbdim,ksfc_type) !< exchange coeff. of heat and 
                                                         !<  vapor for each surface type
    REAL(wp),INTENT(OUT) :: pcfv_sfc    (kbdim)  !< exchange coeff. of total water variance
    REAL(wp),INTENT(OUT) :: pcftotte_sfc(kbdim)  !< exchange coeff. of TTE
    REAL(wp),INTENT(OUT) :: pcfthv_sfc  (kbdim)  !< exchange coeff. of the variance of 
                                                 !<  theta_v
    REAL(wp),INTENT(OUT) :: pprfac_sfc (kbdim)   !< prefactor for exchange coefficients
    REAL(wp),INTENT(OUT) :: ptottevn_sfc(kbdim)  !< boundary condition (sfc value) of TTE
    REAL(wp),INTENT(OUT) :: pthvvar_sfc(kbdim)   !< boundary condition (sfc value)
                                                 !< of the variance of theta_v
    REAL(wp),INTENT(OUT) :: pustarm    (kbdim)   !< friction velocity, grid-box mean
    REAL(wp),INTENT(OUT) :: pwstar     (kbdim)   !< convective velocity scale, grid-box mean
    REAL(wp),INTENT(INOUT) ::pwstar_tile(kbdim,ksfc_type)!< convective velocity scale, 
                                                         !<  each sfc type
    REAL(wp),INTENT(OUT) :: pthvsig_b (kbdim)
    REAL(wp),INTENT(OUT) :: pbn_tile  (kbdim,ksfc_type)  !< for diagnostics
    REAL(wp),INTENT(OUT) :: pbhn_tile (kbdim,ksfc_type)  !< for diagnostics
    REAL(wp),INTENT(OUT) :: pbm_tile  (kbdim,ksfc_type)  !< for diagnostics
    REAL(wp),INTENT(OUT) :: pbh_tile  (kbdim,ksfc_type)  !< for diagnostics
    REAL(wp),INTENT(OUT) :: pch_tile  (kbdim,ksfc_type)  !< for TTE boundary condition
    !
    ! optional arguments for use with jsbach
    REAL(wp),OPTIONAL,INTENT(IN) :: paz0lh (kbdim)  !< roughness length for heat over land
    REAL(wp),OPTIONAL,INTENT(IN) :: pcsat  (kbdim)  !< area fraction with wet land surface
    REAL(wp),OPTIONAL,INTENT(IN) :: pcair  (kbdim)  !< area fraction with wet land surface (air)

    REAL(wp) :: pchn_tile (kbdim,ksfc_type) !<
    REAL(wp) :: pcdn_tile (kbdim,ksfc_type) !<
    REAL(wp) :: pcfnc_tile(kbdim,ksfc_type) !<
    REAL(wp) :: pthvsig_tile(kbdim,ksfc_type)

    ! Local variables

    REAL(wp) :: zdu2      (kbdim,ksfc_type) !<
    REAL(wp) :: zcfnch    (kbdim,ksfc_type) !<
    REAL(wp) :: zustar    (kbdim,ksfc_type) !< friction velocity
    REAL(wp) :: zqts      (kbdim,ksfc_type)
    REAL(wp) :: zthetavmid(kbdim,ksfc_type) ! virtual potential temperature at mid-level
    REAL(wp) :: zdthetal  (kbdim,ksfc_type) !
    REAL(wp) :: lmix      (kbdim,ksfc_type) !< mixing length
    REAL(wp) :: e_kin     (kbdim,ksfc_type) !< turbulent kinetic energy
    REAL(wp) :: e_pot     (kbdim,ksfc_type) !< turbulent potential energy
    REAL(wp) :: f_tau     (kbdim,ksfc_type) !< stability finction for momentum
    REAL(wp) :: f_theta   (kbdim,ksfc_type) !< stability finction for heat
    REAL(wp) :: z0h       (kbdim,ksfc_type) !
 
    INTEGER  :: loidx     (kbdim,ksfc_type) !< counter for masks
    INTEGER  :: is        (ksfc_type)       !< counter for masks

    
    REAL(wp) :: zrdrv, zrvrd, zonethird, ztwothirds
    REAL(wp) :: z2b, z3b, z3bc, zcons17, zepdu2, zepsec
    REAL(wp) :: zqtl, zqtmid, zdqt, zqsmid
    REAL(wp) :: ztmid, ztheta, zthetav, zthetamid, zfux, zfox
    REAL(wp) :: zmult1, zmult2, zmult3, zmult4, zmult5
    REAL(wp) :: zdus1, zdus2, zbuoy, ztottev
    REAL(wp) :: zucf, zucfh
    REAL(wp) :: zust
    REAL(wp) :: w1, ws
    INTEGER  :: jsfc, jl, jls, js

    ! Shortcuts to components of echam_vdf_config
    !
    LOGICAL , POINTER :: lsfc_mom_flux, lsfc_heat_flux
    REAL(wp), POINTER :: f_tau0, f_theta0, c_f, c_n, pr0, wmc, fsl, fbl
    !
    lsfc_mom_flux  => echam_vdf_config(jg)% lsfc_mom_flux
    lsfc_heat_flux => echam_vdf_config(jg)% lsfc_heat_flux
    !
    f_tau0   => echam_vdf_config(jg)% f_tau0
    f_theta0 => echam_vdf_config(jg)% f_theta0
    c_f      => echam_vdf_config(jg)% c_f
    c_n      => echam_vdf_config(jg)% c_n
    pr0      => echam_vdf_config(jg)% pr0
    wmc      => echam_vdf_config(jg)% wmc
    fsl      => echam_vdf_config(jg)% fsl
    fbl      => echam_vdf_config(jg)% fbl

    !-------------------
    ! Some constants
    !-------------------
    zepsec     = 1.E-2_wp
    zrvrd      = rv/rd
    zrdrv      = rd/rv
    z2b        = 2._wp*cb
    z3b        = 3._wp*cb
    z3bc       = 3._wp*cb*cc
    zcons17    = 1._wp / ckap**2
    zepdu2     = 1._wp
    zonethird  = 1._wp/3._wp
    ztwothirds = 2._wp/3._wp
    ws         = 1._wp-fsl
    w1         = fsl

    !------------------------------------------------------------------------------
    ! The prefactor (= air density) that will be multiplied to the exchange
    ! coefficients when building the linear algebraic equations. The density here is
    ! computed using air temperature of the lowest model level at time step n-1.
    !------------------------------------------------------------------------------
    pprfac_sfc(jcs:kproma) =  ppsfc(jcs:kproma)                                     &
                         & /( rd*ptm1_b(jcs:kproma)                               &
                         &   *(1._wp+vtmpc1*pqm1_b(jcs:kproma)-pqxm1_b(jcs:kproma)) )

    !-------------------------------------------------------------
    ! COMPUTATION OF BASIC QUANTITIES: WIND SHEAR,
    ! RICHARDSON NUMBER,SQUARED MIXING LENGTHS, UNSTABLE
    ! AND STABLE CASE COMMON FACTORS AND NEUTRAL CASE
    ! COMMON PART OF THE DRAG COEFFICIENTS.
    !-------------------------------------------------------------
    !    preset values to zero
     pcpt_tile (:,:) = 0._wp
     pqsat_tile(:,:) = 0._wp
     pri_tile  (:,:) = 0._wp

     DO jsfc = 1,ksfc_type

     ! check for masks
     !
      is(jsfc) = 0
      DO jl = jcs,kproma
        IF(pfrc(jl,jsfc).GT.0.0_wp) THEN
          is(jsfc) = is(jsfc) + 1
          loidx(is(jsfc),jsfc) = jl
        ENDIF
      ENDDO

      CALL compute_qsat( kproma, is(jsfc), loidx(1,jsfc), ppsfc, ptsfc(1,jsfc), pqsat_tile(1,jsfc) )

     ! loop over mask only
     !
      DO jls = 1,is(jsfc)
        js=loidx(jls,jsfc)
        ! dry static energy pcpt_tile
        !
        IF(jsfc == idx_lnd) THEN
          zqts(js,jsfc) = pcsat(js) * pqsat_tile(js,jsfc) + (1._wp - pcair(js))*pqm1_b(js) ! q_total at land surface
        ELSE
          zqts(js,jsfc) = pqsat_tile(js,jsfc)                                              ! q_total at non-land surface
        END IF
        pcpt_tile(js,jsfc) = ptsfc(js,jsfc) * (cpd + (cpv - cpd) * zqts(js,jsfc))

        ztheta      = ptsfc(js,jsfc)*(p0ref/ppsfc(js))**rd_o_cpd
        zthetav     = ztheta*(1._wp+vtmpc1*zqts(js,jsfc))

        zqtl       = pqm1_b(js) + pqxm1_b(js)                     ! q_total at lowest model level

!  Weighted interpolation to surface-layer mid-level (fsl):
        zqtmid              = w1*zqtl + ws*zqts(js,jsfc)
        zqsmid              = w1*pqsat_b(js) + ws*pqsat_tile(js,jsfc)
        ztmid               = w1*ptm1_b(js) + ws*ptsfc(js,jsfc)
        zthetamid           = w1*ptheta_b(js) + ws*ztheta
        zthetavmid(js,jsfc) = w1*pthetav_b(js) + ws*zthetav

        zfux = plh_b(js)/(cpd*ztmid)
        zfox = plh_b(js)/(rd*ztmid)

        zmult1 = 1._wp+vtmpc1*zqtmid                              ! A in clear sky
        zmult2 = zfux*zmult1-zrvrd
        zmult3 = zrdrv*zfox*zqsmid/(1._wp+zrdrv*zfox*zfux*zqsmid)
        zmult5 = zmult1-zmult2*zmult3                             ! A in cloud
        zmult4 = zfux*zmult5-1._wp                                ! D in cloud

        zdus1 = paclc_b(js)*zmult5 + (1._wp-paclc_b(js))*zmult1   ! A avg
        zdus2 = paclc_b(js)*zmult4 + (1._wp-paclc_b(js))*vtmpc1   ! D avg

        zdqt     = zqtl - zqts(js,jsfc)                           ! d qt
        zdthetal(js,jsfc) = pthetal_b(js) - ztheta                ! d theta_l
        IF (jsfc == idx_lnd) THEN                                 ! over land
          zdu2(js,jsfc) = MAX(zepdu2,(pum1_b(js)**2+pvm1_b(js)**2 &
                                    +(wmc*pwstar_tile(js,jsfc))**2))
        ELSE                                                      ! over water or ice
          zdu2(js,jsfc) = MAX(zepdu2,(pum1_b(js)-pocu(js))**2 &   ! (d u)^2
                                    +(pvm1_b(js)-pocv(js))**2 &   ! (d v)^2
                                    +(wmc*pwstar_tile(js,jsfc))**2 )
        END IF

        zbuoy        = zdus1*zdthetal(js,jsfc) + zdus2*zthetamid*zdqt
        pri_tile(js,jsfc) = pghf_b(js)*grav*zbuoy/(zthetavmid(js,jsfc)*zdu2(js,jsfc))

 !  Stability functions for heat and momentum (Mauritsen et al. 2007) 

        IF(pri_tile(js,jsfc).GT.0._wp) THEN
           f_tau(js,jsfc)   = f_tau0*(0.25_wp+0.75_wp/(1._wp+4._wp*pri_tile(js,jsfc)))
           f_theta(js,jsfc) = f_theta0               /(1._wp+4._wp*pri_tile(js,jsfc))
        ELSE
           f_tau(js,jsfc)   = f_tau0
           f_theta(js,jsfc) = f_theta0
        END IF

 !  Diagnose turbulent kinetic and turbulent potential energy from total turbulent energy:

        IF(pri_tile(js,jsfc).GT.0._wp) THEN
           e_kin(js,jsfc) = ptotte_b(js)/(1._wp+pri_tile(js,jsfc)/(pr0+3._wp*pri_tile(js,jsfc)))
           e_pot(js,jsfc) = e_kin(js,jsfc)     *pri_tile(js,jsfc)/(pr0+3._wp*pri_tile(js,jsfc))
        ELSE
           e_kin(js,jsfc) = ptotte_b(js)/(1._wp+pri_tile(js,jsfc)/(2._wp*pri_tile(js,jsfc)-pr0))
           e_pot(js,jsfc) = e_kin(js,jsfc)     *pri_tile(js,jsfc)/(2._wp*pri_tile(js,jsfc)-pr0) 
        END IF

!   Near-surface virtual potential temperature variance used by convection scheme:

        pthvsig_tile(js,jsfc) =  zthetav/grav*SQRT(2.0_wp*e_pot(js,jsfc)*ABS(zbuoy))

 !  Compute mixing length for neutrally stratified conditions:

        lmix(js,jsfc) = 1._wp/(ckap*fsl*pghf_b(js))                                       &
                      & + 2._wp*earth_angular_velocity                                    &
                      & /(c_f*SQRT(f_tau(js,jsfc)*e_kin(js,jsfc)))

!   Add term for stably stratified conditions:

        lmix(js,jsfc) = lmix(js,jsfc)                                                     &
                      & + SQRT(grav*MAX(0._wp,zbuoy)/(zthetavmid(js,jsfc)*pghf_b(js)))    &
                      &   /(c_n*SQRT(f_tau(js,jsfc)*e_kin(js,jsfc)))

!   Add term for convectively unstable conditions and invert. Here it is assumed that the
!   lowest model level is in the lower half of the convectively unstable boundary layer
!   which simplifies the code.

        lmix(js,jsfc) = lmix(js,jsfc)                                                     &
                      & + MAX(0._wp, fbl/(ckap*(phdtcbl(js)-fsl*pghf_b(js))))

        lmix(js,jsfc) = 1._wp/lmix(js,jsfc)

 !  Neutral drag coefficient for momentum, z0m is effectively limited to half 
 !  the first level height.
       
        pcdn_tile(js,jsfc) = lmix(js,jsfc)**2/((fsl*pghf_b(js))**2                        &
                          & *((LOG(MAX(2._wp,pghf_b(js)/pz0m(js,jsfc))))**2))

        pcfnc_tile(js,jsfc)= SQRT(zdu2(js,jsfc))*pcdn_tile(js,jsfc)

 !  REMARK:
 !  Compared to the original (precalc_land,ocean,ice) the factor zcons is missing
 !  this factor is included as "prefactor for the exchange coefficients" in
 !  subroutine matrix_setup_elim

 !  Compute/extract roughness length for heat over each surface, currently equal to z0m over ice

        IF ( jsfc == idx_wtr ) THEN         ! over water
           z0h(js,jsfc)=pz0m(js,jsfc)*EXP(2._wp-86.276_wp*pz0m(js,jsfc)**0.375_wp)
        ELSE IF ( jsfc == idx_ice ) THEN
           z0h(js,jsfc)=pz0m(js,jsfc)
        ELSE IF ( jsfc == idx_lnd ) THEN
           z0h(js,jsfc)=paz0lh(js)
        END IF

 ! Neutral drag coefficient for heat/scalars, z0h is effectively limited 
 !  to half the first level height! 

          pchn_tile(js,jsfc) = lmix(js,jsfc)/((fsl*pghf_b(js))                            &
                            & *LOG(MAX(2._wp,pghf_b(js)/z0h(js,jsfc))))                   &
                            & *1._wp/pr0*SQRT(pcdn_tile(js,jsfc))
          zcfnch(js,jsfc)   = SQRT(zdu2(js,jsfc))*pchn_tile(js,jsfc)

      ENDDO ! 1:is
    ENDDO   ! 1:ksfc_type

    !-------------------------------------------------------------------------
    ! Compute the exchange coefficients for momentum, heat and water vapour,
    ! for each type of surface
    !-------------------------------------------------------------------------

    IF (lsfc_mom_flux.OR.lsfc_heat_flux) THEN  ! Surface flux is considered

!  Preset values to zero
     pcfh_tile(:,:) = 0.0_wp
     pcfm_tile(:,:) = 0.0_wp

!  Multiply neutral coefficients by stability functions

      DO jsfc = 1,ksfc_type
        DO jls = 1,is(jsfc)
          js=loidx(jls,jsfc)
          IF ( pri_tile(js,jsfc) > 0._wp ) THEN
            pcfm_tile(js,jsfc) = pcfnc_tile(js,jsfc)*f_tau  (js,jsfc)/f_tau0
            pcfh_tile(js,jsfc) = zcfnch    (js,jsfc)*f_theta(js,jsfc)/f_theta0*SQRT(f_tau(js,jsfc)/f_tau0)  
            pch_tile (js,jsfc) = pcfh_tile(js,jsfc)/zcfnch(js,jsfc)*pchn_tile(js,jsfc)   
          ENDIF
        
 
          IF ( pri_tile(js,jsfc) <= 0._wp ) THEN        ! retain Louis stability functions
                                                        ! functions for the unstable case
            zucf =  SQRT( -pri_tile(js,jsfc)*(1._wp+ pghf_b(js)/pz0m(js,jsfc)) ) 
                                                        ! sqrt in (5.4)
            zucf =  1._wp+z3bc*pcdn_tile(js,jsfc)*zucf  ! denominator in (5.4)
            zucf =  1._wp/zucf
            zucfh=  SQRT( -pri_tile(js,jsfc)*(1._wp+ pghf_b(js)/z0h(js,jsfc)) ) 
                                                        ! sqrt in (5.4)
            zucfh=  1._wp+z3bc*pchn_tile(js,jsfc)*zucfh ! denominator in (5.4)
            zucfh=  1._wp/zucfh
            pcfm_tile(js,jsfc) = pcfnc_tile (js,jsfc)*(1._wp-z2b*pri_tile(js,jsfc)*zucf)  
                                                        ! (5.2), (5.4)
            pcfh_tile(js,jsfc) = zcfnch(js,jsfc)*(1._wp-z3b*pri_tile(js,jsfc)*zucfh) 
                                                        ! (5.2), (5.4)
            pch_tile (js,jsfc) = pchn_tile  (js,jsfc)*(1._wp-z3b*pri_tile(js,jsfc)*zucfh)
          ENDIF
          

        ENDDO
      ENDDO
    
    END IF

    IF (.NOT.lsfc_mom_flux ) THEN  ! Surface momentum flux is switched off
      pcfm_tile(jcs:kproma,1:ksfc_type) = 0._wp
    END IF

    IF (.NOT.lsfc_heat_flux) THEN  ! Surface heat flux is switched off
      pcfh_tile(jcs:kproma,1:ksfc_type) = 0._wp
    END IF

    !------------------------------------------------------------------------------
    ! Store values
    ! to be used in subroutine "nsurf_diag" to compute
    ! new t2m, 2m dew point, 10m wind components
    !------------------------------------------------------------------------------

    pbn_tile (:,:) = 0._wp
    pbhn_tile(:,:) = 0._wp
    pbm_tile (:,:) = 0._wp
    pbh_tile (:,:) = 0._wp

    DO jsfc = 1,ksfc_type
      DO jls = 1,is(jsfc)
      js=loidx(jls,jsfc)
        pbn_tile(js,jsfc) = ckap / MAX(zepsec, SQRT(pcdn_tile(js,jsfc)))
        pbhn_tile(js,jsfc) = ckap / MAX(zepsec, SQRT(pchn_tile(js,jsfc)))
        pbm_tile(js,jsfc) = MAX(zepsec, SQRT(pcfm_tile(js,jsfc)*pcdn_tile(js,jsfc)*zcons17 / pcfnc_tile(js,jsfc)))
        pbh_tile(js,jsfc) = MAX(zepsec, pch_tile(js,jsfc)/pbm_tile(js,jsfc)       *zcons17)
        pbm_tile(js,jsfc) = 1._wp / pbm_tile(js,jsfc)
        pbh_tile(js,jsfc) = 1._wp / pbh_tile(js,jsfc)
      END DO
    END DO

    !-------------------------------------------------------------------------
    ! Get the aggregated exchange coefficient for momentum
    !-------------------------------------------------------------------------
    ! For sensible heat and water vapour, it is not the exchange coefficient
    ! but the solution at the lowest model level that is aggregated.
    ! The exchange coeffcients (cfh_tile) computed in this subroutine
    ! are returned to the calling subroutine for each surface type.
    ! They are used later for solving the discretized vertical diffusion
    ! equation at the lowest grid level (klev) for each surface type
    ! separately. Then the solutions are aggregated using
    ! (fraction of type)*(cfh_tile of type) as the weighting factor, which
    ! ensures conservation of the area-weighted total flux.
    !   Here we compute the aggregated exchange coefficient for output
    ! and for solving the vertical diffusion equation of the variance of
    ! virtual potential temperature (theta_v).
    !-------------------------------------------------------------------------
    ! Add aggregated Richardson number for the surface
    !-------------------------------------------------------------------------

    pcfm_gbm(:) = 0._wp
    pcfh_gbm(:) = 0._wp
    pri_gbm (:) = 0._wp

    DO jsfc = 1,ksfc_type
      DO jls = 1,is(jsfc)
      js=loidx(jls,jsfc)
        pcfm_gbm(js) = pcfm_gbm(js) + pfrc(js,jsfc)*pcfm_tile(js,jsfc)
        pcfh_gbm(js) = pcfh_gbm(js) + pfrc(js,jsfc)*pcfh_tile(js,jsfc)
        pri_gbm (js) = pri_gbm (js) + pfrc(js,jsfc)*pri_tile (js,jsfc)
      ENDDO
    ENDDO

    !-------------------------------------------------------------------------
    ! Hydrometeors and the other tracers share the same exchange coefficient
    ! with heat and moisture, but have no turbulence-induced surface flux.
    ! These are taken care of in subroutine matrix_setup.
    !   The total water variance has a different exchange coefficient
    ! (variable cfv), and no surface flux. Set the surface exchange coefficient
    ! to zero.
    !-------------------------------------------------------------------------
    pcfv_sfc(:) = 0._wp

    !-------------------------------------------------------------------------
    ! Diagnose friction velocity. The values of each individual surface type
    ! are used immediately below for computing the surface value of TTE;
    ! The grid-box mean is used in the next time step in subroutine
    ! "atm_exchange_coeff" for finding the PBL height.
    !-------------------------------------------------------------------------
    IF (lsfc_mom_flux) THEN  ! Surface momentum flux is switched on

      DO jsfc = 1,ksfc_type
        DO jls = 1,is(jsfc)
        js=loidx(jls,jsfc)
          zust   = pcfm_tile(js,jsfc)*SQRT(zdu2(js,jsfc)-(wmc*pwstar_tile(js,jsfc))**2)
          zustar(js,jsfc) = SQRT(zust)
         IF (pri_tile(js,jsfc).LT. 0._wp) THEN
           pwstar_tile(js,jsfc)= 0.5_wp*(pwstar_tile(js,jsfc)                                 &
                   &                +(phdtcbl(js)*grav/zthetavmid(js,jsfc)*pcfh_tile(js,jsfc)  & 
                   &                *abs(zdthetal(js,jsfc)))**zonethird)
         ELSE
           pwstar_tile(js,jsfc) = 0._wp
         END IF
        END DO
      END DO
 
 !  REMARK:
 !  compared to the original (precalc_land,ocean,ice) the factor 1/zcons is missing
 !  this factor is included as "prefactor for the exchange coefficients" in
 !  subroutine matrix_setup_elim

      pustarm(:) = 0._wp
      pwstar(:)= 0._wp
      pthvsig_b(:) = 0._wp

      DO jsfc = 1,ksfc_type
        DO jls = 1,is(jsfc)
        js=loidx(jls,jsfc)
          pustarm(js) = pustarm(js) + pfrc(js,jsfc)*zustar(js,jsfc)
          pwstar(js)  = pwstar(js)  + pfrc(js,jsfc)*pwstar_tile(js,jsfc)
          pthvsig_b(js)= pthvsig_b(js)+ pfrc(js,jsfc)*pthvsig_tile(js,jsfc)
        END DO
      END DO

    ELSE ! Surface momentum flux is off. Friction velocity is by definition zero.
      zustar (1:kproma,1:ksfc_type)     = 0._wp
      pustarm(1:kproma)                 = 0._wp
      pwstar_tile(1:kproma,1:ksfc_type) = 0._wp
      pwstar(1:kproma)                  = 0._wp
      pthvsig_b(1:kproma)               = 0._wp 
    END IF

    !---------------------------------------------------
    ! Surface value of TTE and its exchange coefficient
    !---------------------------------------------------
    ! The exchange coefficient of TTE is set to the same value as for momentum
    pcftotte_sfc(jcs:kproma) = pcfm_gbm(jcs:kproma)

    ! TTE at the surface
    ptottevn_sfc(:) = 0._wp  ! initialize the weighted average

    DO jsfc = 1,ksfc_type
      DO jls = 1,is(jsfc)
      js=loidx(jls,jsfc)

       IF(pri_tile(js,jsfc).GT.0._wp) THEN
         ztottev  = (1._wp+e_pot(js,jsfc)/e_kin(js,jsfc))/f_tau(js,jsfc)*(pustarm(js)**2)
       ELSE
         ztottev  = (1._wp+e_pot(js,jsfc)/e_kin(js,jsfc))/f_tau(js,jsfc)                 &
                    & *(pustarm(js)**3+lmix(js,jsfc)*2._wp*grav/zthetavmid(js,jsfc)      &
                    & *pcfh_tile(js,jsfc)*abs(zdthetal(js,jsfc)))**ztwothirds
       END IF

       ptottevn_sfc(js) = ptottevn_sfc(js) + ztottev*pfrc(js,jsfc)
      END DO
    END DO
    ptottevn_sfc(jcs:kproma) = MAX( totte_min,ptottevn_sfc(jcs:kproma) )

    !----------------------------------------------------------------
    ! Surface value and exchange coefficient of theta_v variance
    !----------------------------------------------------------------
    ! The exchange coefficient is set to the aggregated coefficient
    ! of heat and moisture.
    pcfthv_sfc(jcs:kproma) = pcfh_gbm(jcs:kproma)

    ! thvvar at the surface
    pthvvar_sfc(jcs:kproma) = pthvvar_b(jcs:kproma)

  END SUBROUTINE sfc_exchange_coeff
  !-------------

END MODULE mo_turbulence_diag
