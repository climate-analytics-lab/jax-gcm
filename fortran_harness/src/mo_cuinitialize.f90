#ifdef __xlC__
@PROCESS HOT
#endif
 
!>
!! @brief
!!       *cuini*
!!       This routine interpolates large-scale flieds of t,q etc.
!!       to half levels, determines level of maximum vertical velocity and
!!       initializes values for updrafts and downdrafts
!!       *cubase*
!!       This routine produces cloud base values for cu-parameterization
!! @remarks
!!       *cuini*
!!       This routine is called from subroutine *cumastr*
!!       *cubase*
!!       This routine is called from subroutine *cumastr*
!!       Input are environmental values t,q,p,phi at half levels.
!!       It returns cloud base values and flags as follows
!!          klab=1 for subcloud levels
!!          klab=2 for condensation level
!!       Method: lift surface air dry-adiabatically to cloud base
!!               (non-entraining plume, i.e. constant massflux)
!!
!! @author M. Tiedtke, ECMWF,    Dec 1989
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
MODULE mo_cuinitialize
  USE mo_kind,                 ONLY: wp
  USE mo_physical_constants,   ONLY: cpd, cpv, vtmpc1, alv, als, tmelt
  USE mo_echam_cnv_config,     ONLY: echam_cnv_config
  USE mo_echam_convect_tables, ONLY: prepare_ua_index_spline,lookup_ua_spline
  USE mo_cuadjust,             ONLY: cuadjtq

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: cuini, cubase

CONTAINS 
  !>
  !!
  SUBROUTINE cuini(    jg,       jb,                                                 &
    &        jcs,      kproma,   kbdim,    klev,     klevp1,   klevm1,               &
    &        pten,     pqen,     pqsen,    pxen,     puen,     pven,                 &
    &        ktrac,                                                                  &
    &        pxten,    pxtenh,   pxtu,     pxtd,     pmfuxt,   pmfdxt,               &
    &        pverv,    papp1,    pgeo,     paphp1,   pgeoh,                          &
    &        ptenh,    pqenh,    pqsenh,   pxenh,    klwmin,                         &
    &        ptu,      pqu,      ptd,      pqd,                                      &
    &        puu,      pvu,      pud,      pvd,                                      &
    &        pmfu,     pmfd,     pmfus,    pmfds,                                    &
    &        pmfuq,    pmfdq,    pdmfup,   pdmfdp,                                   &
    &        pcpen,    pcpcu,    palvsh,                                             &
    &        pdpmel,   plu,      plude,    pqude,    klab                            )
    INTEGER, INTENT (IN) :: jg, jb
    INTEGER, INTENT (IN) :: jcs, kproma, kbdim, klev, klevp1, klevm1, ktrac
    REAL(wp):: pten(kbdim,klev),          pqen(kbdim,klev),                          &
      &        puen(kbdim,klev),          pven(kbdim,klev),                          &
      &        pqsen(kbdim,klev),         pverv(kbdim,klev),                         &
      &        pgeo(kbdim,klev),          pgeoh(kbdim,klev),                         &
      &        papp1(kbdim,klev),         paphp1(kbdim,klevp1),                      &
      &        ptenh(kbdim,klev),                                                    &
      &        pxenh(kbdim,klev),         pxen(kbdim,klev),                          &
      &        palvsh(kbdim,klev),                                                   &
      &        pqenh(kbdim,klev),         pqsenh(kbdim,klev)
    REAL(wp):: pcpen(kbdim,klev),         pcpcu(kbdim,klev)
    REAL(wp):: ptu(kbdim,klev),           pqu(kbdim,klev),                           &
      &        ptd(kbdim,klev),           pqd(kbdim,klev),                           &
      &        puu(kbdim,klev),           pud(kbdim,klev),                           &
      &        pvu(kbdim,klev),           pvd(kbdim,klev),                           &
      &        pmfu(kbdim,klev),          pmfd(kbdim,klev),                          &
      &        pmfus(kbdim,klev),         pmfds(kbdim,klev),                         &
      &        pmfuq(kbdim,klev),         pmfdq(kbdim,klev),                         &
      &        pdmfup(kbdim,klev),        pdmfdp(kbdim,klev),                        &
      &        plu(kbdim,klev),           plude(kbdim,klev),                         &
      &        pqude(kbdim,klev)
    REAL(wp):: pdpmel(kbdim,klev)
    INTEGER :: klab(kbdim,klev),          klwmin(kbdim)
    REAL(wp):: zwmax(kbdim)
    REAL(wp):: zph(kbdim)
    INTEGER :: loidx(kbdim)
    REAL(wp):: pxten(kbdim,klev,ktrac),   pxtenh(kbdim,klev,ktrac),                  &
      &        pxtu(kbdim,klev,ktrac),    pxtd(kbdim,klev,ktrac),                    &
      &        pmfuxt(kbdim,klev,ktrac),  pmfdxt(kbdim,klev,ktrac)
    REAL(wp):: za(kbdim),                 ua(kbdim)
    INTEGER :: idx(kbdim)
    INTEGER :: jk, jl, jt, ik, icall
    REAL(wp):: zcpm, zzs
    LOGICAL :: llo1
    !
    !  INTRINSIC FUNCTIONS
    INTRINSIC MAX, MIN
    !
    !---------------------------------------------------------------------------------
    !
    !*    1.  Specify large scale parameters at half levels, adjust temperature
    !*        fields if staticly unstable, find level of maximum vert. velocity
    !         -----------------------------------------------------------------
    !
    DO jk=1,klev

      CALL prepare_ua_index_spline(jg,'cuini',jcs,kproma,pten(1,jk),idx(1),za(1))
      CALL lookup_ua_spline(jcs,kproma,idx(1),za(1),ua(1))


!IBM* NOVECTOR
      DO jl=jcs,kproma

        pqsen(jl,jk)=ua(jl)/papp1(jl,jk)
        pqsen(jl,jk)=MIN(0.5_wp,pqsen(jl,jk))
        pqsen(jl,jk)=pqsen(jl,jk)/(1._wp-vtmpc1*pqsen(jl,jk))

        pcpen(jl,jk)=cpd+(cpv-cpd)*pqen(jl,jk) ! cp of moist air for comp. of fluxes
      END DO
    END DO
    !
    DO jk=2,klev
!IBM* NOVECTOR
      DO jl=jcs,kproma
        zcpm=(pcpen(jl,jk)+pcpen(jl,jk-1))*0.5_wp
        pcpcu(jl,jk)=zcpm
        ptenh(jl,jk)=(MAX(pcpen(jl,jk-1)*pten(jl,jk-1)+pgeo(jl,jk-1),                &
          &        pcpen(jl,jk)*pten(jl,jk)+pgeo(jl,jk))-pgeoh(jl,jk))
        ptenh(jl,jk) = ptenh(jl,jk)/zcpm
        pqsenh(jl,jk)=pqsen(jl,jk-1)
        zph(jl)=paphp1(jl,jk)
        loidx(jl)=jl
      END DO
      !
      DO jt=1,ktrac
        DO jl=jcs,kproma
          pxtenh(jl,jk,jt)=(pxten(jl,jk,jt)+pxten(jl,jk-1,jt))*0.5_wp
        END DO
      END DO
      !
      ik=jk
      icall=0
      CALL cuadjtq(jb, jcs, kproma, kbdim, klev, ik, zph, ptenh, pqsenh, loidx, kproma, icall)
      !
      DO jl=jcs,kproma
        pxenh(jl,jk)=(pxen(jl,jk)+pxen(jl,jk-1))*0.5_wp
        pqenh(jl,jk)=MIN(pqen(jl,jk-1),pqsen(jl,jk-1))+(pqsenh(jl,jk)-pqsen(jl,jk-1))
        pqenh(jl,jk)=MAX(pqenh(jl,jk),0._wp)
      END DO
    END DO
    !
!IBM* NOVECTOR
    DO jl=jcs,kproma
      ptenh(jl,klev)=pcpen(jl,klev)*pten(jl,klev)+pgeo(jl,klev)-pgeoh(jl,klev)
      ptenh(jl,klev)=ptenh(jl,klev)/pcpen(jl,klev)
      pxenh(jl,klev)=pxen(jl,klev)
      pqenh(jl,klev)=pqen(jl,klev)
      pcpcu(jl,1)=pcpen(jl,1)
      ptenh(jl,1)=pten(jl,1)
      pxenh(jl,1)=pxen(jl,1)
      pqenh(jl,1)=pqen(jl,1)
      pgeoh(jl,1)=pgeo(jl,1)
      klwmin(jl)=klev
      zwmax(jl)=0._wp
    END DO
    !
    DO jt=1,ktrac
      DO jl=jcs,kproma
        pxtenh(jl,klev,jt)=pxten(jl,klev,jt)
        pxtenh(jl,1,jt)=pxten(jl,1,jt)
      END DO
    END DO
    !
    DO jk=klevm1,2,-1
!IBM* NOVECTOR
      DO jl=jcs,kproma
        zzs=MAX(pcpcu(jl,jk)*ptenh(jl,jk)+pgeoh(jl,jk),                              &
          &           pcpcu(jl,jk+1)*ptenh(jl,jk+1)+pgeoh(jl,jk+1))
        ptenh(jl,jk)=(zzs-pgeoh(jl,jk))/pcpcu(jl,jk)
      END DO
    END DO
    !
    DO jk=1,klev
! NOVECTOR ?
      DO jl=jcs,kproma
        llo1 = (ptenh(jl,jk)-tmelt) .GT. 0.0_wp
        palvsh(jl,jk) = MERGE(alv,als,llo1)
      END DO
    END DO
    !
    DO jk=klev,3,-1
!DIR$ IVDEP
!OCL NOVREC
      DO jl=jcs,kproma
        IF(pverv(jl,jk).LT.zwmax(jl)) THEN
          zwmax(jl)=pverv(jl,jk)
          klwmin(jl)=jk
        END IF
      END DO
    END DO
    !
    !-----------------------------------------------------------------------
    !*    2.0   Initialize values for updrafts and downdrafts
    !*          ---------------------------------------------
    !
    DO jk=1,klev
      ik=jk-1
      IF(jk.EQ.1) ik=1
      DO jl=jcs,kproma
        ptu(jl,jk)=ptenh(jl,jk)
        ptd(jl,jk)=ptenh(jl,jk)
        pqu(jl,jk)=pqenh(jl,jk)
        pqd(jl,jk)=pqenh(jl,jk)
        plu(jl,jk)=0._wp
        puu(jl,jk)=puen(jl,ik)
        pud(jl,jk)=puen(jl,ik)
        pvu(jl,jk)=pven(jl,ik)
        pvd(jl,jk)=pven(jl,ik)
        pmfu(jl,jk)=0._wp
        pmfd(jl,jk)=0._wp
        pmfus(jl,jk)=0._wp
        pmfds(jl,jk)=0._wp
        pmfuq(jl,jk)=0._wp
        pmfdq(jl,jk)=0._wp
        pdmfup(jl,jk)=0._wp
        pdmfdp(jl,jk)=0._wp
        pdpmel(jl,jk)=0._wp
        plude(jl,jk)=0._wp
        pqude(jl,jk)=0._wp
        klab(jl,jk)=0
      END DO
      DO jt=1,ktrac
        DO jl=jcs,kproma
          pxtu(jl,jk,jt)=pxtenh(jl,jk,jt)
          pxtd(jl,jk,jt)=pxtenh(jl,jk,jt)
          pmfuxt(jl,jk,jt)=0._wp
          pmfdxt(jl,jk,jt)=0._wp
        END DO
      END DO
      !
    END DO
    !
  END SUBROUTINE cuini
  !>
  !!
  SUBROUTINE cubase(   jg,       jb,                                                 &
    &        jcs,      kproma,   kbdim,    klev,     klevp1, klevm1,                 &
    &        ptenh,    pqenh,    pgeoh,    paph,   pthvsig,                          &
    &        ptu,      pqu,      plu,                                                &
    &        puen,     pven,     puu,      pvu,                                      &
    &        pcpcu,                                                                  &
    &        ldcum,    kcbot,    klab)

    INTEGER, INTENT (IN) :: jg, jb
    INTEGER, INTENT (IN) :: jcs, kproma, kbdim, klev, klevp1, klevm1
    REAL(wp):: ptenh(kbdim,klev),       pqenh(kbdim,klev),                           &
      &        pgeoh(kbdim,klev),       paph(kbdim,klevp1),                          &
      &        pthvsig(kbdim)
    REAL(wp):: ptu(kbdim,klev),         pqu(kbdim,klev),                             &
      &        plu(kbdim,klev)
    REAL(wp):: puen(kbdim,klev),        pven(kbdim,klev),                            &
      &        puu(kbdim,klev),         pvu(kbdim,klev)
    REAL(wp):: pcpcu(kbdim,klev)
    INTEGER :: klab(kbdim,klev),        kcbot(kbdim)
    LOGICAL :: ldcum(kbdim)
    REAL(wp):: zqold(kbdim)
    REAL(wp):: zph(kbdim)
    INTEGER :: loidx(kbdim)
    INTEGER :: jl, jk, nl, is, ik, ikb, icall
    REAL(wp):: zbuo, zz, zlift

    ! Shortcuts to components of echam_cnv_config
    !
    LOGICAL , POINTER :: lmfdudv
    REAL(wp), POINTER :: cbfac, cminbuoy, cmaxbuoy
    !
    lmfdudv  => echam_cnv_config(jg)% lmfdudv
    cbfac    => echam_cnv_config(jg)% cbfac
    cminbuoy => echam_cnv_config(jg)% cminbuoy
    cmaxbuoy => echam_cnv_config(jg)% cmaxbuoy

    !
    !---------------------------------------------------------------------------------
    !
    !     1.       Initialize values at lifting level
    !              ----------------------------------
    !
    DO jl=jcs,kproma
      klab(jl,klev)=1
      kcbot(jl)=klevm1
      ldcum(jl)=.FALSE.
      puu(jl,klev)=puen(jl,klev)*(paph(jl,klevp1)-paph(jl,klev))
      pvu(jl,klev)=pven(jl,klev)*(paph(jl,klevp1)-paph(jl,klev))
    END DO
    !
    !---------------------------------------------------------------------------------
    !
    !     2.0  Do ascent in subcloud layer, check for existence of condensation 
    !          level, adjust t,q and l accordingly in *cuadjtq*, check for
    !          buoyancy and set flags
    !          ----------------------------------------------------------------
    DO jk=klevm1,2,-1
      is=jcs-1
      DO jl=jcs,kproma
        IF (klab(jl,jk+1).EQ.1) THEN
          is = is + 1
          loidx(is) = jl
        END IF
        zph(jl)=paph(jl,jk)
      END DO
      IF(is.EQ.jcs-1) CYCLE !GOTO 290
!IBM* ASSERT(NODEPS)
      DO nl=jcs,is
        jl = loidx(nl)
        zlift=MAX(cminbuoy,MIN(cmaxbuoy,pthvsig(jl)*cbfac))
        zlift=MIN(zlift,1.0_wp)
        pqu(jl,jk)=pqu(jl,jk+1)
        ptu(jl,jk)=(pcpcu(jl,jk+1)*ptu(jl,jk+1)+pgeoh(jl,jk+1)                       &
          &                  -pgeoh(jl,jk))/pcpcu(jl,jk)
        zbuo=ptu(jl,jk)*(1._wp+vtmpc1*pqu(jl,jk))-ptenh(jl,jk)                       &
          &                  *(1._wp+vtmpc1*pqenh(jl,jk))+zlift
        IF(zbuo.GT.0._wp) klab(jl,jk)=1
        zqold(jl)=pqu(jl,jk)
      END DO
      !
      ik=jk
      icall=1

      ! call cuadjtq for level kk=ik
      CALL cuadjtq(jb, jcs, kproma, kbdim, klev, ik, zph, ptu, pqu, loidx, is, icall)
      !
!DIR$ IVDEP
!OCL NOVREC
      DO nl=jcs,is
        jl = loidx(nl)
        IF(pqu(jl,jk).LT.zqold(jl)) THEN
          klab(jl,jk)=2
          zlift=MAX(cminbuoy,MIN(cmaxbuoy,pthvsig(jl)*cbfac))
          zlift=MIN(zlift,1.0_wp)
          plu(jl,jk)=plu(jl,jk)+zqold(jl)-pqu(jl,jk)
          zbuo=ptu(jl,jk)*(1._wp+vtmpc1*pqu(jl,jk)-plu(jl,jk))-                      &
            &          ptenh(jl,jk)*(1._wp+vtmpc1*pqenh(jl,jk))+zlift
          IF(zbuo.GT.0.) THEN
            kcbot(jl)=jk
            ldcum(jl)=.TRUE.
          END IF
        END IF
      END DO
      !
      !    Calculate averages of u and v for subcloud area, the values will
      !    be used to define cloud base values.
      !
      IF(lmfdudv) THEN
        DO jl=jcs,kproma
          IF(jk.GE.kcbot(jl)) THEN
            puu(jl,klev)=puu(jl,klev)+puen(jl,jk)*(paph(jl,jk+1)-paph(jl,jk))
            pvu(jl,klev)=pvu(jl,klev)+pven(jl,jk)*(paph(jl,jk+1)-paph(jl,jk))
          END IF
        END DO
      END IF
      !
    END DO  !290
    !
    !
    IF(lmfdudv) THEN
      DO jl=jcs,kproma
        IF(ldcum(jl)) THEN
          ikb=kcbot(jl)
          zz=1._wp/(paph(jl,klevp1)-paph(jl,ikb))
          puu(jl,klev)=puu(jl,klev)*zz
          pvu(jl,klev)=pvu(jl,klev)*zz
        ELSE
          puu(jl,klev)=puen(jl,klevm1)
          pvu(jl,klev)=pven(jl,klevm1)
        END IF
      END DO
    END IF
    !
  END SUBROUTINE cubase
END MODULE mo_cuinitialize
