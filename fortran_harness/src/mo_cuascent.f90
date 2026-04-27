#ifdef __xlC__
@PROCESS STRICT
#endif
#include "fsel.inc"

!>
!! @brief
!!       *cuasc*
!!        Routine produces cloud ascents for cumulus-parameterization
!!        (vertical profiles of t,q,l,u and v and corresponding fluxes
!!        as well as precipitation rates)
!!        *cubasmc*
!!        Routine calculates cloud base values for midlevel convection
!!        *cuentr*
!!        Routine calculates entrainment/detrainment rates for updrafts
!!
!! @remarks
!!       *cuasc*
!!       This routine is called from subroutine *cumastr*
!!       Lift surface air dry-adiabatically to cloud base and then calculate
!!       moist ascent for entraining/detraining plume.
!!       Entrainment and detrainment rates differ for shallow and deep cumulus
!!       convection. In case there is no penetrative or shallow convection
!!       check for possibility of mid level convection
!!       (cloud base values calculated in *cubasmc*)
!!
!!       *cubasmc*
!!       This routine is called from subroutine *cuasc*
!!       Input are environmental values t,q etc
!!       It returns cloud base values for midlevel convection
!!
!!       *cuentr*
!!       This routine is called from subroutine *cuasc*
!!       Input are environmental values t,q etc and updraft
!!       values t,q etc. It returns entrainment/detrainment rates
!!
!! @author M. Tiedtke, ECMWF,    Dec 1989
!!
!! @references.
!!       Tiedtke, 1989: Mon. Wea. Rev., 117, 1779-1800
!!
!! @par Revision History
!! - Taken from ECHAM6.3, unified for ICON/ECHAM by Monika Esch, MPI-M (2015-06)
!!
!! @par Copyright
!! This code is subject to the MPI-M-Software - License - Agreement in it's most recent form.
!! Please see URL http://www.mpimet.mpg.de/en/science/models/model-distribution.html and the
!! file COPYING in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the headers of the routines.
!!
MODULE mo_cuascent
  USE mo_kind,                 ONLY : wp
  USE mo_physical_constants,   ONLY : grav, tmelt, vtmpc1, rv, rd, alv, als
  USE mo_echam_cnv_config,     ONLY : echam_cnv_config
  USE mo_cuadjust,             ONLY : cuadjtq

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: cuasc, cubasmc, cuentr

CONTAINS
  !>
  !!
  SUBROUTINE cuasc(    jg,       jb,                                                 &
    &        jcs,      kproma,   kbdim,    klev,     klevp1,   klevm1,               &
    &        pzf,      pzh,      pmref,                                              &
    &        ptenh,    pqenh,    puen,     pven,                                     &
    &        ktrac,                                                                  &
    &        pdtime,                                                                 &
    &        pxtenh,   pxten,    pxtu,     pmfuxt,                                   &
    &        pten,     pqen,     pqsen,                                              &
    &        pgeo,     pgeoh,    paphp1,   pthvsig,                                  &
    &        pqte,     pverv,    klwmin,                                             &
    &        ldcum,    ldland,   ktype,    klab,                                     &
    &        ptu,      pqu,      plu,      puu,      pvu,                            &
    &        pmfu,     pmfub,    pentr,                                              &
    &        pmfus,    pmfuq,                                                        &
    &        pmful,    plude,    pqude,    pdmfup,                                   &
    &        khmin,    phhatt,   phcbase,  pqsenh,                                   &
    &        pcpen,    pcpcu,                                                        &
    &        kcbot,    kctop,    kctop0                                              &
    &        )

    INTEGER, INTENT (IN) :: jg, jb
    INTEGER, INTENT (IN) :: jcs, kproma, kbdim, klev, klevp1, klevm1, ktrac
    INTEGER :: jl, jk, jt, ik, icall, ikb, ikt, n, locnt
    REAL(wp),INTENT (IN) :: pdtime
    REAL(wp),INTENT (IN) :: pzf(kbdim,klev),  pzh(kbdim,klevp1)
    REAL(wp),INTENT (IN) :: pmref(kbdim,klev)
    
    REAL(wp) :: ptenh(kbdim,klev),       pqenh(kbdim,klev),                          &
      &         puen(kbdim,klev),        pven(kbdim,klev),                           &
      &         pten(kbdim,klev),        pqen(kbdim,klev),                           &
      &         pgeo(kbdim,klev),        pgeoh(kbdim,klev),                          &
      &         paphp1(kbdim,klevp1),    pthvsig(kbdim),                             &
      &         pqsen(kbdim,klev),       pqte(kbdim,klev),                           &
      &         pverv(kbdim,klev)
    !
    REAL(wp) :: ptu(kbdim,klev),         pqu(kbdim,klev),                            &
      &         puu(kbdim,klev),         pvu(kbdim,klev),                            &
      &         pmfu(kbdim,klev),                                                    &
      &         pmfub(kbdim),            pentr(kbdim),                               &
      &         pmfus(kbdim,klev),       pmfuq(kbdim,klev),                          &
      &         plu(kbdim,klev),         plude(kbdim,klev),                          &
      &         pqude(kbdim,klev),                                                   &
      &         pmful(kbdim,klev),       pdmfup(kbdim,klev)
    REAL(wp) :: pcpen(kbdim,klev),       pcpcu(kbdim,klev)
    INTEGER  :: klwmin(kbdim),           ktype(kbdim),                               &
      &         klab(kbdim,klev),        kcbot(kbdim),                               &
      &         kctop(kbdim),            kctop0(kbdim)
    INTEGER  :: khmin(kbdim),            loidx(kbdim)
    REAL(wp) :: phhatt(kbdim,klev)
    REAL(wp) :: phcbase(kbdim)
    REAL(wp) :: pqsenh(kbdim,klev)
    LOGICAL  :: ldcum(kbdim),            ldland(kbdim)
    !
    REAL(wp) :: zdmfen(kbdim),           zdmfde(kbdim),                              &
      &         zmfuu(kbdim),            zmfuv(kbdim),                               &
      &         zpbase(kbdim),           zqold(kbdim)
    REAL(wp) :: zph(kbdim)
    REAL(wp) :: zodetr(kbdim,klev)
    REAL(wp) :: zoentr(kbdim,klev)
    REAL(wp) :: zbuoy(kbdim)
    REAL(wp) :: pxtenh(kbdim,klev,ktrac),pxten(kbdim,klev,ktrac),                    &
      &         pxtu(kbdim,klev,ktrac),  pmfuxt(kbdim,klev,ktrac)
    REAL(wp) :: zcons, zmfmax, zfac, zmftest, zqeen, zseen                           &
      &       , zqude, zmfusk, zmfuqk, zmfulk, zxteen, zxtude, zmfuxtk               &
      &       , zbuo, zdnoprc, zprcon, zlnew, zz, zdmfeu, zdmfdu, zzdmf              &
      &       , zdz, zdrodz, zdprho, zalvs, zmse, znevn, zodmax, zga, zdt            &
      &       , zscod, zqcod, zbuoyz, zscde, zlift
    !
    !
    !      Intrinsic functions
    INTRINSIC MAX, MIN, LOG

    ! Shortcuts to components of echam_cnv_config
    !
    LOGICAL , POINTER :: lmfmid, lmfdudv
    INTEGER , POINTER :: nmctop
    REAL(wp), POINTER :: cprcon, cmfctop, cmfcmin, cminbuoy, cmaxbuoy, cbfac, centrmax
    REAL(wp), POINTER :: dlev_land, dlev_ocean
    !
    lmfmid     => echam_cnv_config(jg)% lmfmid
    lmfdudv    => echam_cnv_config(jg)% lmfdudv
    nmctop     => echam_cnv_config(jg)% nmctop
    cprcon     => echam_cnv_config(jg)% cprcon
    cmfctop    => echam_cnv_config(jg)% cmfctop
    cmfcmin    => echam_cnv_config(jg)% cmfcmin
    cminbuoy   => echam_cnv_config(jg)% cminbuoy
    cmaxbuoy   => echam_cnv_config(jg)% cmaxbuoy
    cbfac      => echam_cnv_config(jg)% cbfac
    centrmax   => echam_cnv_config(jg)% centrmax
    dlev_land  => echam_cnv_config(jg)% dlev_land
    dlev_ocean => echam_cnv_config(jg)% dlev_ocean

    !---------------------------------------------------------------------------------
    !
    !*    1.           Specify parameters
    !                  ------------------
    !
    zcons=1._wp/pdtime
    zqold(jcs:kproma) = 0.0_wp
    !
    !---------------------------------------------------------------------------------
    !
    !     2.           Set default values
    !                  ------------------
    !
    DO jl=jcs,kproma
     zmfuu(jl)=0._wp
     zmfuv(jl)=0._wp
     IF(.NOT.ldcum(jl)) ktype(jl)=0
    END DO
    DO jk=1,klev
      DO jl=jcs,kproma
        plu(jl,jk)=0._wp
        pmfu(jl,jk)=0._wp
        pmfus(jl,jk)=0._wp
        pmfuq(jl,jk)=0._wp
        pmful(jl,jk)=0._wp
        plude(jl,jk)=0._wp
        pqude(jl,jk)=0._wp
        pdmfup(jl,jk)=0._wp
        IF(.NOT.ldcum(jl).OR.ktype(jl).EQ.3) klab(jl,jk)=0
        IF(jk.LT.kcbot(jl)) klab(jl,jk)=0
      END DO
      DO jt=1,ktrac
        DO jl=jcs,kproma
           pmfuxt(jl,jk,jt)=0._wp
        END DO
      END DO
      !
    END DO
    DO jk=1,klev
      DO jl=jcs,kproma
        zoentr(jl,jk)=0._wp
        zodetr(jl,jk)=0._wp
      ENDDO
    ENDDO
    !
    !---------------------------------------------------------------------------------
    !
    !     3.0          Initialize values at lifting level
    !                  ----------------------------------
    !
    DO jl=jcs,kproma
      kctop(jl)=klevm1
      IF(.NOT.ldcum(jl)) THEN
        kcbot(jl)=klevm1
        pmfub(jl)=0._wp
        pqu(jl,klev)=0._wp
      END IF
      pmfu(jl,klev)=pmfub(jl)
      pmfus(jl,klev)=pmfub(jl)*(pcpcu(jl,klev)*ptu(jl,klev)+pgeoh(jl,klev))
      pmfuq(jl,klev)=pmfub(jl)*pqu(jl,klev)
      IF(lmfdudv) THEN
        zmfuu(jl)=pmfub(jl)*puu(jl,klev)
        zmfuv(jl)=pmfub(jl)*pvu(jl,klev)
      END IF
    END DO
    !
    DO jt=1,ktrac
      DO jl=jcs,kproma
        IF(.NOT.ldcum(jl)) THEN
          pxtu(jl,klev,jt)=0._wp
        ENDIF
        pmfuxt(jl,klev,jt)=pmfub(jl)*pxtu(jl,klev,jt)
      END DO
    END DO
    !
    DO jl=jcs,kproma
      ldcum(jl)=.FALSE.
    END DO
    !
    !---------------------------------------------------------------------------------
    !
    !     3.5          Find organized entrainment at cloud base
    !                  ----------------------------------------
    !
    DO jl=jcs,kproma
      IF(ktype(jl).EQ.1) THEN
        ikb=kcbot(jl)
        zbuoy(jl)=grav*(ptu(jl,ikb)-ptenh(jl,ikb))/ptenh(jl,ikb) +                   &
          &               grav*vtmpc1*(pqu(jl,ikb)-pqenh(jl,ikb))
        IF(zbuoy(jl).GT.0._wp) THEN
          zdz=pzf(jl,ikb-1)-pzf(jl,ikb)
          zdrodz=-LOG(pten(jl,ikb-1)/pten(jl,ikb))/zdz                               &
            &       -grav/(rd*ptenh(jl,ikb)*(1._wp+vtmpc1*pqenh(jl,ikb)))
          ! nb zoentr is here a fractional value
          zoentr(jl,ikb-1)=zbuoy(jl)*0.5_wp/(1._wp+zbuoy(jl)*zdz) + zdrodz
          zoentr(jl,ikb-1)=MIN(zoentr(jl,ikb-1),centrmax)
          zoentr(jl,ikb-1)=MAX(zoentr(jl,ikb-1),0._wp)
        ENDIF
      ENDIF
    ENDDO
    !
    !---------------------------------------------------------------------------------
    !
    !     4.  Do ascent: subcloud layer (klab=1), clouds (klab=2) by doing
    !         first dry-adiabatic ascent and then by adjusting t, q and l
    !         accordingly in *cuadjtq*, then check for buoyancy and set
    !         flags accordingly
    !         -------------------------------------------------------------
    !
    DO jk=klevm1,2,-1
    !
    !         Specify cloud base values for midlevel convection in
    !         *cubasmc* in case there is not already convection
    !         -----------------------------------------------------
    !
      ik=jk
      IF(lmfmid.AND.ik.LT.klevm1.AND.ik.GT.nmctop) THEN
        CALL cubasmc(jg,                                                             &
          &          jcs,      kproma,   kbdim,    klev,     ik,       klab,         &
          &          pten,     pqen,     pqsen,    puen,     pven,                   &
          &          ktrac,                                                          &
          &          pxten,    pxtu,     pmfuxt,                                     &
          &          pverv,    pgeo,     pgeoh,    ldcum,    ktype,                  &
          &          pmfu,     pmfub,    pentr,    kcbot,                            &
          &          ptu,      pqu,      plu,      puu,      pvu,                    &
          &          pmfus,    pmfuq,    pmful,    pdmfup,   zmfuu,                  &
          &          pcpen,                                                          &
          &          zmfuv                                                           )
      ENDIF
      !
      locnt = jcs-1
      DO jl=jcs,kproma
        IF(klab(jl,jk+1).EQ.0) klab(jl,jk)=0
        IF(klab(jl,jk+1).GT.0) THEN
          locnt = locnt + 1
          loidx(locnt) = jl
        END IF
        zph(jl)=paphp1(jl,jk)
        IF(ktype(jl).EQ.3.AND.jk.EQ.kcbot(jl)) THEN
          zmfmax=pmref(jl,jk-1)*zcons
          IF(pmfub(jl).GT.zmfmax) THEN
            zfac=zmfmax/pmfub(jl)
            pmfu(jl,jk+1)=pmfu(jl,jk+1)*zfac
            pmfus(jl,jk+1)=pmfus(jl,jk+1)*zfac
            pmfuq(jl,jk+1)=pmfuq(jl,jk+1)*zfac
            zmfuu(jl)=zmfuu(jl)*zfac
            zmfuv(jl)=zmfuv(jl)*zfac
          END IF
        END IF
      END DO
      DO jt=1,ktrac
        DO jl=jcs,kproma
          IF(ktype(jl).EQ.3.AND.jk.EQ.kcbot(jl)) THEN
            zmfmax=pmref(jl,jk-1)*zcons
            IF(pmfub(jl).GT.zmfmax) THEN
              zfac=zmfmax/pmfub(jl)
              pmfuxt(jl,jk+1,jt)=pmfuxt(jl,jk+1,jt)*zfac
            END IF
          END IF
        END DO
      END DO
      !
      ! Reset pmfub if necessary
      !
      DO jl=jcs,kproma
        IF(ktype(jl).EQ.3.AND.jk.EQ.kcbot(jl)) THEN
          zmfmax=pmref(jl,jk-1)*zcons
          pmfub(jl)=MIN(pmfub(jl),zmfmax)
        END IF
      END DO
      !
      !*   Specify turbulent entrainment and detrainment rates plus
      !    organized detrainment rates in *cuentr*
      !    --------------------------------------------------------
      !
      ik=jk
      CALL cuentr(    jg,                                                             &
        &   jcs,      kproma,   kbdim,    klev,     klevp1,   ik,                     &
        &   pzh,      pmref,                                                          &
        &   ptenh,    pqenh,    pqte,     paphp1,                                     &
        &   klwmin,   ldcum,    ktype,    kcbot,    kctop0,                           &
        &   zpbase,   pmfu,     pentr,    zodetr,                                     &
        &   khmin,                                                                    &
        &   zdmfen,   zdmfde)
      !
      !     Do adiabatic ascent for entraining/detraining plume
      !     The cloud ensemble entrains environmental values
      !     In turbulent detrainment cloud ensemble values are detrained
      !     In organized detrainment the dry static energy and moisture
      !     that are neutral compared to the environmental air are detrained
      !     ----------------------------------------------------------------
      !
      DO n=jcs,locnt
        jl = loidx(n)

        IF(jk.LT.kcbot(jl)) THEN
          zmftest=pmfu(jl,jk+1)+zdmfen(jl)-zdmfde(jl)
          zmfmax=MIN(zmftest,pmref(jl,jk-1)*zcons)
          zdmfen(jl)=MAX(zdmfen(jl)-MAX(zmftest-zmfmax,0._wp),0._wp)
        END IF
        zdmfde(jl)=MIN(zdmfde(jl),0.75_wp*pmfu(jl,jk+1))
        pmfu(jl,jk)=pmfu(jl,jk+1)+zdmfen(jl)-zdmfde(jl)
        IF (ktype(jl).EQ.1 .AND. jk.LT.kcbot(jl)) THEN
          zdprho=pzh(jl,jk)-pzh(jl,jk+1)
          zoentr(jl,jk)=zoentr(jl,jk)*zdprho*pmfu(jl,jk+1)
          zmftest=pmfu(jl,jk)+zoentr(jl,jk)-zodetr(jl,jk)
          zmfmax=MIN(zmftest,pmref(jl,jk-1)*zcons)
          zoentr(jl,jk)=MAX(zoentr(jl,jk)-MAX(zmftest-zmfmax,0._wp),0._wp)
        ELSE
          zoentr(jl,jk)=0._wp
        ENDIF
        IF(ktype(jl).EQ.1.AND.jk.LT.kcbot(jl).AND.jk.LE.khmin(jl)) THEN
        !          limit organized detrainment to prevent too
        !          deep clouds
          zalvs=MERGE(alv,als,ptu(jl,jk+1)>tmelt)
          zmse=pcpcu(jl,jk+1)*ptu(jl,jk+1)+zalvs*pqu(jl,jk+1)+pgeoh(jl,jk+1)
          ikt=kctop0(jl)
          znevn=(pzh(jl,ikt)-pzh(jl,jk+1))*(zmse-phhatt(jl,jk+1))
          IF(znevn.LE.0._wp) znevn=1._wp
          zdprho=pzh(jl,jk)-pzh(jl,jk+1)
          zodmax=((phcbase(jl)-zmse)/znevn)*zdprho*pmfu(jl,jk+1)
          zodmax=MAX(zodmax,0._wp)
          zodetr(jl,jk)=MIN(zodetr(jl,jk),zodmax)
        ENDIF
        zodetr(jl,jk)=MIN(zodetr(jl,jk),0.75_wp*pmfu(jl,jk))
        pmfu(jl,jk)=pmfu(jl,jk)+zoentr(jl,jk)-zodetr(jl,jk)
        zqeen=pqenh(jl,jk+1)*zdmfen(jl)
        zqeen=zqeen+pqenh(jl,jk+1)*zoentr(jl,jk)
        zseen=(pcpcu(jl,jk+1)*ptenh(jl,jk+1)+pgeoh(jl,jk+1))*zdmfen(jl)
        zseen=zseen+(pcpcu(jl,jk+1)*ptenh(jl,jk+1)+pgeoh(jl,jk+1))*zoentr(jl,jk)
        zscde=(pcpcu(jl,jk+1)*ptu(jl,jk+1)+pgeoh(jl,jk+1))*zdmfde(jl)
        ! find moist static energy that give nonbuoyant air
        zalvs=MERGE(alv,als,ptenh(jl,jk+1)>tmelt)
        zga=zalvs*pqsenh(jl,jk+1)/(rv*(ptenh(jl,jk+1)**2))
        zdt=(plu(jl,jk+1)-vtmpc1*(pqsenh(jl,jk+1)-pqenh(jl,jk+1)))/                  &
          &            (1._wp/ptenh(jl,jk+1) + vtmpc1*zga)
        zscod=pcpcu(jl,jk+1)*ptenh(jl,jk+1)+pgeoh(jl,jk+1)+pcpcu(jl,jk+1)*zdt
        zscod=MAX(zscod,0._wp)
        zscde=zscde+zodetr(jl,jk)*zscod
        zqude=pqu(jl,jk+1)*zdmfde(jl)
        zqcod=pqsenh(jl,jk+1)+zga*zdt
        zqcod=MAX(zqcod,0._wp)
        zqude=zqude+zodetr(jl,jk)*zqcod
        pqude(jl,jk)=zqude
        plude(jl,jk)=plu(jl,jk+1)*zdmfde(jl)
        plude(jl,jk)=plude(jl,jk)+plu(jl,jk+1)*zodetr(jl,jk)
        zmfusk=pmfus(jl,jk+1)+zseen-zscde
        zmfuqk=pmfuq(jl,jk+1)+zqeen-zqude
        zmfulk=pmful(jl,jk+1)    -plude(jl,jk)
        plu(jl,jk)=zmfulk*(1._wp/MAX(cmfcmin,pmfu(jl,jk)))
        pqu(jl,jk)=zmfuqk*(1._wp/MAX(cmfcmin,pmfu(jl,jk)))
        ptu(jl,jk)=(zmfusk*(1._wp/MAX(cmfcmin,pmfu(jl,jk)))-pgeoh(jl,jk))/pcpcu(jl,jk)
        ptu(jl,jk)=MAX(100._wp,ptu(jl,jk))
        ptu(jl,jk)=MIN(400._wp,ptu(jl,jk))
        zqold(jl)=pqu(jl,jk)
      END DO
      !
      DO jt=1,ktrac
!IBM* ASSERT(NODEPS)
        DO n=jcs,locnt
          jl = loidx(n)
          zxteen=pxtenh(jl,jk+1,jt)*(zdmfen(jl)+zoentr(jl,jk))
          zxtude=pxtu(jl,jk+1,jt)*(zdmfde(jl)+zodetr(jl,jk))
          zmfuxtk=pmfuxt(jl,jk+1,jt)+zxteen-zxtude
          pxtu(jl,jk,jt)=zmfuxtk*(1._wp/MAX(cmfcmin,pmfu(jl,jk)))
        END DO
      END DO
      !
      !    Do corrections for moist ascent by adjusting t,q and l in *cuadjtq*
      !    -------------------------------------------------------------------
      !
      ik=jk
      icall=1

      IF (locnt .GE. jcs) THEN
      CALL cuadjtq(jb, jcs, kproma, kbdim, klev, ik,                                 &
        &          zph,      ptu,      pqu,      loidx, locnt,  icall)
      END IF

      !
!DIR$ IVDEP
!OCL NOVREC
!IBM* ASSERT(NODEPS)
      DO n=jcs,locnt
        jl = loidx(n)
        IF (pqu(jl,jk).LT.zqold(jl)) THEN
          klab(jl,jk)=2
          zlift=MAX(cminbuoy,MIN(cmaxbuoy,pthvsig(jl)*cbfac))
          zlift=MIN(zlift,1.0_wp)
          plu(jl,jk)=plu(jl,jk)+zqold(jl)-pqu(jl,jk)
          zbuo=ptu(jl,jk)*(1._wp+vtmpc1*pqu(jl,jk)-plu(jl,jk))-                      &
            &       ptenh(jl,jk)*(1._wp+vtmpc1*pqenh(jl,jk))
          IF(klab(jl,jk+1).EQ.1) zbuo=zbuo+zlift
          IF(zbuo.GT.0._wp.AND.pmfu(jl,jk).GE.0.01_wp*pmfub(jl).AND.                 &
            &          jk.GE.kctop0(jl)) THEN
            kctop(jl)=jk
            ldcum(jl)=.TRUE.
            zdnoprc=MERGE(dlev_land,dlev_ocean,ldland(jl))
            zprcon=MERGE(0._wp,cprcon,zpbase(jl)-paphp1(jl,jk).LT.zdnoprc)
            zlnew=plu(jl,jk)/(1._wp+zprcon*(pgeoh(jl,jk)-pgeoh(jl,jk+1)))
            pdmfup(jl,jk)=MAX(0._wp,(plu(jl,jk)-zlnew)*pmfu(jl,jk))
            plu(jl,jk)=zlnew
          ELSE
            klab(jl,jk)=0
            pmfu(jl,jk)=0._wp
          END IF
        END IF
      END DO

!IBM* ASSERT(NODEPS)
      DO n=jcs,locnt
        jl = loidx(n)
        pmful(jl,jk)=plu(jl,jk)*pmfu(jl,jk)
        pmfus(jl,jk)=(pcpcu(jl,jk)*ptu(jl,jk)+pgeoh(jl,jk))*pmfu(jl,jk)
        pmfuq(jl,jk)=pqu(jl,jk)*pmfu(jl,jk)
      END DO
      DO jt=1,ktrac
!IBM* ASSERT(NODEPS)
        DO n=jcs,locnt
          jl = loidx(n)
          pmfuxt(jl,jk,jt)=pxtu(jl,jk,jt)*pmfu(jl,jk)
        END DO
      END DO
      !
      IF(lmfdudv) THEN
        DO jl=jcs,kproma
          zdmfen(jl)=zdmfen(jl)+zoentr(jl,jk)
          zdmfde(jl)=zdmfde(jl)+zodetr(jl,jk)
        ENDDO
!IBM* ASSERT(NODEPS)
        DO n=jcs,locnt
          jl = loidx(n)
          IF(ktype(jl).EQ.1.OR.ktype(jl).EQ.3) THEN
            zz=MERGE(3._wp,2._wp,zdmfen(jl).EQ.0._wp)
          ELSE
            zz=MERGE(1._wp,0._wp,zdmfen(jl).EQ.0._wp)
          END IF
          zdmfeu=zdmfen(jl)+zz*zdmfde(jl)
          zdmfdu=zdmfde(jl)+zz*zdmfde(jl)
          zdmfdu=MIN(zdmfdu,0.75_wp*pmfu(jl,jk+1))
          zmfuu(jl)=zmfuu(jl)+zdmfeu*puen(jl,jk)-zdmfdu*puu(jl,jk+1)
          zmfuv(jl)=zmfuv(jl)+zdmfeu*pven(jl,jk)-zdmfdu*pvu(jl,jk+1)
          IF(pmfu(jl,jk).GT.0._wp) THEN
             puu(jl,jk)=zmfuu(jl)*(1._wp/pmfu(jl,jk))
             pvu(jl,jk)=zmfuv(jl)*(1._wp/pmfu(jl,jk))
          END IF
        END DO
      END IF
      !
      !   Compute organized entrainment for use at next level
      !   ---------------------------------------------------
      !
!IBM* ASSERT(NODEPS)
      DO n=jcs,locnt
        jl = loidx(n)
        IF(ktype(jl).EQ.1) THEN
          zbuoyz=grav*(ptu(jl,jk)-ptenh(jl,jk))/ptenh(jl,jk) +                       &
            &   grav*vtmpc1*(pqu(jl,jk)-pqenh(jl,jk))-grav*plu(jl,jk)
          zbuoyz=MAX(zbuoyz,0.0_wp)
          zdz=pzf(jl,jk-1)-pzf(jl,jk)
          zdrodz=-LOG(pten(jl,jk-1)/pten(jl,jk))/zdz                                 &
            &          -grav/(rd*ptenh(jl,jk)*(1._wp+vtmpc1*pqenh(jl,jk)))
          zbuoy(jl)=zbuoy(jl)+zbuoyz*zdz
          zoentr(jl,jk-1)=zbuoyz*0.5_wp/(1._wp+zbuoy(jl)) + zdrodz
          zoentr(jl,jk-1)=MIN(zoentr(jl,jk-1),centrmax)
          zoentr(jl,jk-1)=MAX(zoentr(jl,jk-1),0._wp)
        ENDIF
      ENDDO
      !
    END DO
    !
    !---------------------------------------------------------------------------------
    !
    !     5.       Determine convective fluxes above non-buoyancy level
    !              ----------------------------------------------------
    !              (Note: cloud variables like t,q and l are not affected by
    !               detrainment and are already known from previous calculations
    !               above
    !
    DO jl=jcs,kproma
      IF(kctop(jl).EQ.klevm1) ldcum(jl)=.FALSE.
      kcbot(jl)=MAX(kcbot(jl),kctop(jl))
    END DO
!DIR$ IVDEP
    DO jl=jcs,kproma
      IF(ldcum(jl)) THEN
        jk=kctop(jl)-1
        zzdmf=cmfctop
        zdmfde(jl)=(1._wp-zzdmf)*pmfu(jl,jk+1)
        plude(jl,jk)=zdmfde(jl)*plu(jl,jk+1)
        pqude(jl,jk)=zdmfde(jl)*pqu(jl,jk+1)
        pmfu(jl,jk)=pmfu(jl,jk+1)-zdmfde(jl)
        pdmfup(jl,jk)=0._wp
        pmfus(jl,jk)=(pcpcu(jl,jk)*ptu(jl,jk)+pgeoh(jl,jk))*pmfu(jl,jk)
        pmfuq(jl,jk)=pqu(jl,jk)*pmfu(jl,jk)
        pmful(jl,jk)=plu(jl,jk)*pmfu(jl,jk)
        IF(jk.GE.2) THEN
          plude(jl,jk-1)=pmful(jl,jk)
          pqude(jl,jk-1)=pmfuq(jl,jk)
        ELSE
          plude(jl,jk)=plude(jl,jk)+pmful(jl,jk)
          pqude(jl,jk)=pqude(jl,jk)+pmfuq(jl,jk)
        END IF
      END IF
    END DO
    DO jt=1,ktrac
      DO jl=jcs,kproma
        IF(ldcum(jl)) THEN
          jk=kctop(jl)-1
          pmfuxt(jl,jk,jt)=pxtu(jl,jk,jt)*pmfu(jl,jk)
        ENDIF
      END DO
    END DO
    !
    IF(lmfdudv) THEN
!DIR$      IVDEP
      DO jl=jcs,kproma
        IF(ldcum(jl)) THEN
          jk=kctop(jl)-1
          puu(jl,jk)=puu(jl,jk+1)
          pvu(jl,jk)=pvu(jl,jk+1)
        END IF
      END DO
    END IF
    !

  END SUBROUTINE cuasc
  !!
  !!
  SUBROUTINE cubasmc(  jg,                                                           &
    &        jcs,      kproma,   kbdim,    klev,     kk,       klab,                 &
    &        pten,     pqen,     pqsen,    puen,     pven,                           &
    &        ktrac,                                                                  &
    &        pxten,    pxtu,     pmfuxt,                                             &
    &        pverv,    pgeo,     pgeoh,    ldcum,    ktype,                          &
    &        pmfu,     pmfub,    pentr,    kcbot,                                    &
    &        ptu,      pqu,      plu,      puu,      pvu,                            &
    &        pmfus,    pmfuq,    pmful,    pdmfup,   pmfuu,                          &
    &        pcpen,                                                                  &
    &        pmfuv                                                                 )
    !
    INTEGER, INTENT (IN) :: jg
    INTEGER, INTENT (IN) :: kbdim, klev, ktrac, jcs, kproma, kk
    REAL(wp) :: pten(kbdim,klev),        pqen(kbdim,klev),                           &
      &         puen(kbdim,klev),        pven(kbdim,klev),                           &
      &         pqsen(kbdim,klev),       pverv(kbdim,klev),                          &
      &         pgeo(kbdim,klev),        pgeoh(kbdim,klev)
    !
    REAL(wp) :: ptu(kbdim,klev),         pqu(kbdim,klev),                            &
      &         puu(kbdim,klev),         pvu(kbdim,klev),                            &
      &         plu(kbdim,klev),         pmfu(kbdim,klev),                           &
      &         pmfub(kbdim),            pentr(kbdim),                               &
      &         pmfus(kbdim,klev),       pmfuq(kbdim,klev),                          &
      &         pmful(kbdim,klev),       pdmfup(kbdim,klev),                         &
      &         pmfuu(kbdim),            pmfuv(kbdim)
    REAL(wp) :: pcpen(kbdim,klev)
    INTEGER  :: ktype(kbdim),            kcbot(kbdim),                               &
      &         klab(kbdim,klev)
    LOGICAL  :: ldcum(kbdim)
    !
    REAL(wp) :: pxten(kbdim,klev,ktrac), pxtu(kbdim,klev,ktrac),                     &
      &         pmfuxt(kbdim,klev,ktrac)
    LOGICAL  :: llo3(kbdim)
    INTEGER  :: jl, jt
    REAL(wp) :: zzzmb

    ! Shortcuts to components of echam_cnv_config
    !
    LOGICAL , POINTER :: lmfdudv
    REAL(wp), POINTER :: entrmid, cmfcmin, cmfcmax
    !
    lmfdudv  => echam_cnv_config(jg)% lmfdudv
    entrmid  => echam_cnv_config(jg)% entrmid
    cmfcmin  => echam_cnv_config(jg)% cmfcmin
    cmfcmax  => echam_cnv_config(jg)% cmfcmax

    !---------------------------------------------------------------------------------
    !
    !*    1.    Calculate entrainment and detrainment rates
    !           -------------------------------------------
    !
!DIR$ IVDEP
!OCL NOVREC
    DO jl=jcs,kproma
      llo3(jl)=.FALSE.
      IF(.NOT.ldcum(jl) .AND. klab(jl,kk+1) .EQ. 0                                   &
        &      .AND. pqen(jl,kk)    .GT. 0.90_wp*pqsen(jl,kk)                        &
        &      .AND. pverv(jl,kk)   .LT. 0.0_wp                                      &
        &      .AND. pgeoh(jl,kk)/grav .GT. 1500.0_wp) THEN
        llo3(jl)=.TRUE.
        ptu(jl,kk+1)=(pcpen(jl,kk)*pten(jl,kk)                                       &
          &             +pgeo(jl,kk)-pgeoh(jl,kk+1))/pcpen(jl,kk)
        pqu(jl,kk+1)=pqen(jl,kk)
        plu(jl,kk+1)=0._wp
        zzzmb=MAX(cmfcmin,-pverv(jl,kk)/grav)
        zzzmb=MIN(zzzmb,cmfcmax)
        pmfub(jl)=zzzmb
        pmfu(jl,kk+1)=pmfub(jl)
        pmfus(jl,kk+1)=pmfub(jl)*(pcpen(jl,kk+1)*ptu(jl,kk+1)+pgeoh(jl,kk+1))
        pmfuq(jl,kk+1)=pmfub(jl)*pqu(jl,kk+1)
        pmful(jl,kk+1)=0._wp
        pdmfup(jl,kk+1)=0._wp
        kcbot(jl)=kk
        klab(jl,kk+1)=1
        ktype(jl)=3
        pentr(jl)=entrmid
        IF(lmfdudv) THEN
          puu(jl,kk+1)=puen(jl,kk)
          pvu(jl,kk+1)=pven(jl,kk)
          pmfuu(jl)=pmfub(jl)*puu(jl,kk+1)
          pmfuv(jl)=pmfub(jl)*pvu(jl,kk+1)
        END IF
      END IF
    END DO
!DIR$ IVDEP
!OCL NOVREC
    DO jt=1,ktrac
      DO jl=jcs,kproma
        IF (llo3(jl)) THEN
          pxtu(jl,kk+1,jt)=pxten(jl,kk,jt)
          pmfuxt(jl,kk+1,jt)=pmfub(jl)*pxtu(jl,kk+1,jt)
        ENDIF
      END DO
    END DO
    !
  END SUBROUTINE cubasmc
  !!
  !!
  SUBROUTINE cuentr(   jg,                                                           &
    &        jcs,      kproma,   kbdim,    klev,     klevp1,   kk,                   &
    &        pzh,      pmref,                                                        &
    &        ptenh,    pqenh,    pqte,     paphp1,                                   &
    &        klwmin,   ldcum,    ktype,    kcbot,    kctop0,                         &
    &        ppbase,   pmfu,     pentr,    podetr,                                   &
    &        khmin,                                                                  &
    &        pdmfen,   pdmfde)
    !
    INTEGER, INTENT (IN) :: jg
    INTEGER, INTENT (IN) :: kbdim, klev, klevp1, jcs, kproma, kk
    !
    REAL(wp),INTENT (IN) :: pzh(kbdim,klevp1), pmref(kbdim,klev)

    REAL(wp) :: ptenh(kbdim,klev),       pqenh(kbdim,klev),                          &
      &         paphp1(kbdim,klevp1),                                                &
      &         pmfu(kbdim,klev),        pqte(kbdim,klev),                           &
      &         pentr(kbdim),            ppbase(kbdim)
    REAL(wp) :: podetr(kbdim,klev)
    INTEGER  :: khmin (kbdim)
    INTEGER  :: klwmin(kbdim),           ktype(kbdim),                               &
      &         kcbot(kbdim),            kctop0(kbdim)
    LOGICAL  :: ldcum(kbdim)
    !
    REAL(wp) :: pdmfen(kbdim),           pdmfde(kbdim)
    !
    LOGICAL  :: llo1,llo2
    !
    INTEGER  :: jl, ikt, ikh, iklwmin, n, ncnt
    REAL(wp) :: zpmid, zentr, zentest, zzmzk, ztmzk, zorgde, zarg
    REAL(wp) :: zrrho(kbdim),zdprho(kbdim)
    INTEGER  :: icond1(kbdim),icond2(kbdim),icond3(kbdim),idx(kbdim)

    ! Shortcuts to components of echam_cnv_config
    !
    REAL(wp), POINTER :: cmfcmin, centrmax
    !
    cmfcmin  => echam_cnv_config(jg)% cmfcmin
    centrmax => echam_cnv_config(jg)% centrmax

    !
    !---------------------------------------------------------------------------------
    !
    !*    1.    Calculate entrainment and detrainment rates
    !           Specify entrainment rates for shallow clouds
    !           Specify entrainment rates for deep clouds
    !           --------------------------------------------
    !
!IBM* NOVECTOR
    DO jl=jcs,kproma
      ppbase(jl) = paphp1(jl,kcbot(jl))
      zrrho(jl)  = (rd*ptenh(jl,kk+1)*(1._wp+vtmpc1*pqenh(jl,kk+1)))/paphp1(jl,kk+1)
      zdprho(jl) = pmref(jl,kk)
      zpmid      = 0.5_wp*(ppbase(jl)+paphp1(jl,kctop0(jl)))
      icond1(jl) = FSEL(zpmid-paphp1(jl,kk),0._wp,1._wp)
      icond2(jl) = FSEL(0.2e5_wp - (ppbase(jl)-paphp1(jl,kk)),0._wp,1._wp)
      icond3(jl) = FSEL(1.e-5_wp-pqenh(jl,kk+1),0._wp,1._wp)
      pdmfde(jl)=0._wp
      pdmfen(jl)=0._wp
      podetr(jl,kk)=0._wp
    END DO

    ncnt = 0
    DO jl=jcs,kproma
      llo1=kk.LT.kcbot(jl).AND.ldcum(jl)
      IF (llo1) THEN
        ncnt = ncnt+1
        idx(ncnt) = jl
      END IF
    END DO

!IBM* ASSERT(NODEPS)
    DO n=1,ncnt
      jl = idx(n)
      zentr=pentr(jl)*pmfu(jl,kk+1)*zdprho(jl)*zrrho(jl)
      pdmfde(jl)=zentr

      llo2=ktype(jl).EQ.2.AND.(icond2(jl).LT.1.OR.icond1(jl).GT.0)
      IF (llo2) pdmfen(jl)=zentr

      iklwmin=MAX(klwmin(jl),kctop0(jl)+2)
      llo2=ktype(jl).EQ.3 .AND. kk .GE. iklwmin
      IF(llo2) pdmfen(jl)=zentr

      IF(llo2 .AND. icond3(jl).GT.0) THEN
        pmfu(jl,kk+1) = MAX(pmfu(jl,kk+1),cmfcmin)
        zentest = MAX(pqte(jl,kk),0._wp)/pqenh(jl,kk+1)
        zentest = MIN(centrmax,zentest/(pmfu(jl,kk+1)*zrrho(jl)))
        pdmfen(jl) = zentr+zentest*pmfu(jl,kk+1)*zrrho(jl)*zdprho(jl)
      ENDIF

      llo2=ktype(jl).EQ.1 .AND.(kk.GE.iklwmin.OR.icond1(jl).GT.0)
      IF(llo2) pdmfen(jl)=zentr
    END DO
    !
    !    organized detrainment, detrainment starts at khmin
    !
    IF (kproma > 0) THEN
      ! kproma>0: mpuetz's workaround to avoid combining of two loops
!IBM* ASSERT(NODEPS)
      DO n=1,ncnt
        jl = idx(n)
        llo2=ktype(jl).EQ.1
        IF(llo2.AND.kk.LE.khmin(jl).AND.kk.GE.kctop0(jl)) THEN
          ikt=kctop0(jl)
          ikh=khmin(jl)
          IF(ikh.GT.ikt) THEN
            zzmzk  =-(pzh(jl,ikh)-pzh(jl,kk))
            ztmzk  =-(pzh(jl,ikh)-pzh(jl,ikt))
            zarg  =3.1415_wp*(zzmzk/ztmzk)*0.5_wp
            zorgde=TAN(zarg)*3.1415_wp*0.5_wp/ztmzk
            zdprho(jl)=pmref(jl,kk)*zrrho(jl)
            podetr(jl,kk)=MIN(zorgde,centrmax)*pmfu(jl,kk+1)*zdprho(jl)
          ENDIF
        ENDIF
      END DO
    END IF
    !
  END SUBROUTINE cuentr

END MODULE mo_cuascent
