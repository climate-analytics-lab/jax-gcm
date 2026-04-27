!>
!! @brief  
!!     *cudlfs*
!!     This routine calculates level of free sinking for cumulus 
!!     downdrafts and specifies t,q,u and v values
!!     *cuddraf*
!!     This routine calculates cumulus downdraft descent
!!
!! @remarks
!!     *cudlfs*
!!     This routine is called from subroutine *cumastr*
!!     Input are environmental values t,q,u and v and also cloud base massflux and
!!     cu-precipitation rates. It returns t,q,u and v values and massflux at lfs.
!!     Method: check for negative buoyancy of air and equal parts of moist 
!!     environmental air and cloud air.
!!     *cuddraf*
!!     This routine is called from subroutine *cumastr*
!!     Input is t,q,p,phi,u,v at half levels. 
!!     It returns fluxes of s,q and evaporation rate and u,v at levels where 
!!     downdraft occurs
!!     Method: calculate moist descent for entraining/detraining plume
!!     A) Moving air dry-adiabatically to next level below and
!!     B) Correcting for evaporation to obtain saturated state.
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
MODULE mo_cudescent

  USE mo_kind,                 ONLY : wp
  USE mo_physical_constants,   ONLY : grav, rd, vtmpc1
  USE mo_echam_cnv_config,     ONLY : echam_cnv_config
  USE mo_cuadjust,             ONLY : cuadjtq
 
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: cudlfs, cuddraf

CONTAINS
  !>
  !!
  SUBROUTINE cudlfs(   jg,       jb,                                                 &
    &        jcs,      kproma,   kbdim,    klev,     klevp1,                         &
    &        ptenh,    pqenh,    puen,     pven,                                     &
    &        ktrac,                                                                  &
    &        pxtenh,   pxtu,     pxtd,     pmfdxt,                                   &
    &        pgeoh,    paphp1,                                                       &
    &        ptu,      pqu,      puu,      pvu,                                      &
    &        ldcum,    kcbot,    kctop,    pmfub,    prfl,                           &
    &        ptd,      pqd,      pud,      pvd,                                      &
    &        pmfd,     pmfds,    pmfdq,    pdmfdp,                                   &
    &        pcpcu,                                                                  &
    &        kdtop,    lddraf                                                       )
    !
    INTEGER, INTENT (IN) :: jg, jb
    INTEGER, INTENT (IN) :: kbdim, klev, ktrac, jcs, kproma, klevp1
    REAL(wp) :: ptenh(kbdim,klev),       pqenh(kbdim,klev),                          &
      &         puen(kbdim,klev),        pven(kbdim,klev),                           &
      &         pgeoh(kbdim,klev),       paphp1(kbdim,klevp1),                       &
      &         ptu(kbdim,klev),         pqu(kbdim,klev),                            &
      &         puu(kbdim,klev),         pvu(kbdim,klev),                            &
      &         pmfub(kbdim),            prfl(kbdim)
    !
    REAL(wp) :: ptd(kbdim,klev),         pqd(kbdim,klev),                            &
      &         pud(kbdim,klev),         pvd(kbdim,klev),                            &
      &         pmfd(kbdim,klev),        pmfds(kbdim,klev),                          &
      &         pmfdq(kbdim,klev),       pdmfdp(kbdim,klev)
    REAL(wp) :: pcpcu(kbdim,klev)
    INTEGER  :: kcbot(kbdim),            kctop(kbdim),                               &
      &         kdtop(kbdim)
    LOGICAL  :: ldcum(kbdim),            lddraf(kbdim)
    !
    REAL(wp) :: ztenwb(kbdim,klev),      zqenwb(kbdim,klev),                         &
      &         zcond(kbdim)
    REAL(wp) :: zph(kbdim)
    INTEGER  :: loidx(kbdim)
    LOGICAL  :: llo2(kbdim)
    REAL(wp) :: pxtenh(kbdim,klev,ktrac),pxtu(kbdim,klev,ktrac),                     &
      &         pxtd(kbdim,klev,ktrac),  pmfdxt(kbdim,klev,ktrac)
    LOGICAL  :: llo3(kbdim)
    INTEGER  :: jl, jk, ke, is, ik, icall, jt
    REAL(wp) :: zttest, zqtest, zbuo, zmftop

    ! Shortcuts to components of echam_cnv_config
    !
    LOGICAL , POINTER :: lmfdd, lmfdudv
    REAL(wp), POINTER :: cmfdeps
    !
    lmfdudv  => echam_cnv_config(jg)% lmfdudv
    lmfdd    => echam_cnv_config(jg)% lmfdd
    cmfdeps  => echam_cnv_config(jg)% cmfdeps

    !---------------------------------------------------------------------------------
    !
    !     1.           Set default values for downdrafts
    !                  ---------------------------------
    !
    DO jl = jcs, kproma
      lddraf(jl) = .FALSE.
      kdtop(jl) = klevp1
    END DO
    !
    IF (.NOT.lmfdd) RETURN
      !
      !-------------------------------------------------------------------------------
      !
      !     2.  Determine level of free sinking by doing a scan from top to base
      !         of cumulus clouds for every point and proceed as follows:
      !             (1) Determine wet bulb environmental t and q
      !             (2) Do mixing with cumulus cloud air
      !             (3) Check for negative buoyancy
      !         The assumption is that air of downdrafts is mixture of 50% cloud
      !         air and 50% environmental air at wet bulb temperature (i.e. which
      !         became saturated due to evaporation of rain and cloud water)
      !         -----------------------------------------------------------------
      !
      ke=klev-3
      level: DO jk=3,ke
      !
      !     2.1 Calculate wet-bulb temperature and moisture for environmental air
      !         -----------------------------------------------------------------
      !
      is = jcs-1
      DO jl = jcs, kproma
        llo2(jl) = .FALSE.
        IF (ldcum(jl) .AND. prfl(jl) > 0.0_wp .AND. .NOT. lddraf(jl) .AND.           &
          &                         (jk < kcbot(jl) .AND. jk > kctop(jl))) THEN
          is = is+1
          loidx(is) = jl
          llo2(jl) = .TRUE.
        ENDIF
      ENDDO
      IF (is == jcs-1) CYCLE level
      DO jl=jcs,kproma
        ztenwb(jl,jk)=ptenh(jl,jk)
        zqenwb(jl,jk)=pqenh(jl,jk)
        zph(jl)=paphp1(jl,jk)
      END DO
      !
      ik=jk
      icall=2

      IF (is .GE. jcs) THEN
        CALL cuadjtq( jb, jcs, kproma, kbdim, klev, ik, zph, ztenwb, zqenwb, loidx, is, icall)
      END IF

      !
      !     2.2 Do mixing of cumulus and environmental air and check for negative
      !         buoyancy. Then set values for downdraft at lfs.
      !         ----------------------------------------------------------------------
      !
!DIR$ IVDEP
!OCL NOVREC
      DO jl=jcs,kproma
        llo3(jl)=.FALSE.
        IF(llo2(jl)) THEN
          zttest=0.5_wp*(ptu(jl,jk)+ztenwb(jl,jk))
          zqtest=0.5_wp*(pqu(jl,jk)+zqenwb(jl,jk))
          zbuo=zttest*(1._wp+vtmpc1*zqtest)-                                         &
            &            ptenh(jl,jk)*(1._wp+vtmpc1*pqenh(jl,jk))
          zcond(jl)=pqenh(jl,jk)-zqenwb(jl,jk)
          zmftop=-cmfdeps*pmfub(jl)
          IF(zbuo.LT.0._wp.AND.prfl(jl).GT.10._wp*zmftop*zcond(jl)) THEN
            llo3(jl)=.TRUE.
            kdtop(jl)=jk
            lddraf(jl)=.TRUE.
            ptd(jl,jk)=zttest
            pqd(jl,jk)=zqtest
            pmfd(jl,jk)=zmftop
            pmfds(jl,jk)=pmfd(jl,jk)*(pcpcu(jl,jk)*ptd(jl,jk)+pgeoh(jl,jk))
            pmfdq(jl,jk)=pmfd(jl,jk)*pqd(jl,jk)
            pdmfdp(jl,jk-1)=-0.5_wp*pmfd(jl,jk)*zcond(jl)
            prfl(jl)=prfl(jl)+pdmfdp(jl,jk-1)
          END IF
        END IF
      END DO
      !
      DO jt=1,ktrac
        DO jl=jcs,kproma
          IF(llo3(jl)) THEN
            pxtd(jl,jk,jt)=0.5_wp*(pxtu(jl,jk,jt)+pxtenh(jl,jk,jt))
            pmfdxt(jl,jk,jt)=pmfd(jl,jk)*pxtd(jl,jk,jt)
          ENDIF
        END DO
      END DO
      !
      IF(lmfdudv) THEN
        DO jl=jcs,kproma
          IF(pmfd(jl,jk).LT.0._wp) THEN
            pud(jl,jk)=0.5_wp*(puu(jl,jk)+puen(jl,jk-1))
            pvd(jl,jk)=0.5_wp*(pvu(jl,jk)+pven(jl,jk-1))
          END IF
        END DO
      END IF
      !
    END DO level
    !
  END SUBROUTINE cudlfs
  !>
  !!
  SUBROUTINE cuddraf(  jg,       jb,                                                 &
    &        jcs,      kproma,   kbdim,    klev,     klevp1,                         &
    &        pmref,                                                                  &
    &        ptenh,    pqenh,    puen,     pven,                                     &
    &        ktrac,                                                                  &
    &        pxtenh,   pxtd,     pmfdxt,                                             &
    &        pgeoh,    paphp1,   prfl,                                               &
    &        ptd,      pqd,      pud,      pvd,                                      &
    &        pmfd,     pmfds,    pmfdq,    pdmfdp,                                   &
    &        pcpcu,                                                                  &
    &        lddraf                                                                 )
    !
    INTEGER, INTENT (IN) :: jg, jb
    INTEGER, INTENT (IN) :: kbdim, klev, ktrac, jcs, kproma, klevp1
    !
    REAL(wp),INTENT (IN) :: pmref(kbdim,klev)
    
    REAL(wp) :: ptenh(kbdim,klev),       pqenh(kbdim,klev),                          &
      &         puen(kbdim,klev),        pven(kbdim,klev),                           &
      &         pgeoh(kbdim,klev),       paphp1(kbdim,klevp1)
    !
    REAL(wp) :: ptd(kbdim,klev),         pqd(kbdim,klev),                            &
      &         pud(kbdim,klev),         pvd(kbdim,klev),                            &
      &         pmfd(kbdim,klev),        pmfds(kbdim,klev),                          &
      &         pmfdq(kbdim,klev),       pdmfdp(kbdim,klev),                         &
      &         prfl(kbdim)
    REAL(wp) :: pcpcu(kbdim,klev)
    LOGICAL  :: lddraf(kbdim)
    !
    REAL(wp) :: zdmfen(kbdim),           zdmfde(kbdim),                              &
      &         zcond(kbdim)
    REAL(wp) :: zph(kbdim)
    LOGICAL  :: llo2(kbdim)
    INTEGER  :: loidx(kbdim)
    REAL(wp) :: pxtenh(kbdim,klev,ktrac),pxtd(kbdim,klev,ktrac),                     &
      &         pmfdxt(kbdim,klev,ktrac)
    LOGICAL  :: llo1
    INTEGER  :: jk, is, jl, itopde, jt, ik, icall
    REAL(wp) :: zentr, zseen, zqeen, zsdde, zqdde, zmfdsk, zmfdqk, zxteen            &
      &       , zxtdde, zmfdxtk, zbuo, zdmfdp, zmfduk, zmfdvk

    ! Shortcuts to components of echam_cnv_config
    !
    LOGICAL , POINTER :: lmfdudv
    REAL(wp), POINTER :: cmfcmin, entrdd
    !
    lmfdudv  => echam_cnv_config(jg)% lmfdudv
    cmfcmin  => echam_cnv_config(jg)% cmfcmin
    entrdd   => echam_cnv_config(jg)% entrdd

    !----------------------------------------------------------------------
    !     1.  Calculate moist descent for cumulus downdraft by
    !        (A) Calculating entrainment rates, assuming
    !            linear decrease of massflux in pbl
    !        (B) Doing moist descent - evaporative cooling
    !            and moistening is calculated in *cuadjtq*
    !        (C) Checking for negative buoyancy and
    !            specifying final t,q,u,v and downward fluxes
    !            -------------------------------------------------
    !
    level: DO jk = 3, klev
      is = jcs-1
      DO jl = jcs, kproma
        llo2(jl) = .FALSE.
        IF (lddraf(jl) .AND. pmfd(jl,jk-1) < 0.0_wp) THEN
          is = is+1
          loidx(is) = jl
          llo2(jl) = .TRUE.
        ENDIF
      ENDDO
      DO jl = jcs, kproma
        zph(jl) = paphp1(jl,jk)
      END DO
      DO jl=jcs,kproma
        IF(llo2(jl)) THEN
          zentr=entrdd*pmfd(jl,jk-1)*rd*ptenh(jl,jk-1)/paphp1(jl,jk-1)*pmref(jl,jk-1)
          zdmfen(jl)=zentr
          zdmfde(jl)=zentr
        END IF
      END DO
      itopde=klev-2
      IF(jk.GT.itopde) THEN
        DO jl=jcs,kproma
          IF(llo2(jl)) THEN
            zdmfen(jl)=0._wp
            zdmfde(jl)=pmfd(jl,itopde)* pmref(jl,jk-1)*grav/             &
              &                         (paphp1(jl,klevp1)-paphp1(jl,itopde))
          END IF
        END DO
      END IF
      DO jl=jcs,kproma
        IF(llo2(jl)) THEN
          pmfd(jl,jk)=pmfd(jl,jk-1)+zdmfen(jl)-zdmfde(jl)
          zseen=(pcpcu(jl,jk-1)*ptenh(jl,jk-1)+pgeoh(jl,jk-1))*zdmfen(jl)
          zqeen=pqenh(jl,jk-1)*zdmfen(jl)
          zsdde=(pcpcu(jl,jk-1)*ptd(jl,jk-1)+pgeoh(jl,jk-1))*zdmfde(jl)
          zqdde=pqd(jl,jk-1)*zdmfde(jl)
          zmfdsk=pmfds(jl,jk-1)+zseen-zsdde
          zmfdqk=pmfdq(jl,jk-1)+zqeen-zqdde
          pqd(jl,jk)=zmfdqk*(1._wp/MIN(-cmfcmin,pmfd(jl,jk)))
          ptd(jl,jk)=(zmfdsk*(1._wp/MIN(-cmfcmin,pmfd(jl,jk)))-                      &
            &                      pgeoh(jl,jk))/pcpcu(jl,jk)
          ptd(jl,jk)=MIN(400._wp,ptd(jl,jk))
          ptd(jl,jk)=MAX(100._wp,ptd(jl,jk))
          zcond(jl)=pqd(jl,jk)
        END IF
      END DO
      !
      DO jt=1,ktrac
        DO jl=jcs,kproma
          IF(llo2(jl)) THEN
            zxteen=pxtenh(jl,jk-1,jt)*zdmfen(jl)
            zxtdde=pxtd(jl,jk-1,jt)*zdmfde(jl)
            zmfdxtk=pmfdxt(jl,jk-1,jt)+zxteen-zxtdde
            pxtd(jl,jk,jt)=zmfdxtk*(1._wp/MIN(-cmfcmin,pmfd(jl,jk)))
          ENDIF
        END DO
      END DO
      !
      ik=jk
      icall=2
      IF (is .GE. jcs) THEN
        CALL cuadjtq(jb, jcs, kproma, kbdim, klev, ik, zph, ptd, pqd, loidx, is, icall) 
      END IF
      !
      DO jl=jcs,kproma
        IF(llo2(jl)) THEN
          zcond(jl)=zcond(jl)-pqd(jl,jk)
          zbuo=ptd(jl,jk)*(1._wp+vtmpc1*pqd(jl,jk))-                                &
            &          ptenh(jl,jk)*(1._wp+vtmpc1*pqenh(jl,jk))
          llo1=zbuo.LT.0._wp.AND.(prfl(jl)-pmfd(jl,jk)*zcond(jl).GT.0._wp)
          pmfd(jl,jk)=MERGE(pmfd(jl,jk),0._wp,llo1)
          pmfds(jl,jk)=(pcpcu(jl,jk)*ptd(jl,jk)+pgeoh(jl,jk))*pmfd(jl,jk)
          pmfdq(jl,jk)=pqd(jl,jk)*pmfd(jl,jk)
          zdmfdp=-pmfd(jl,jk)*zcond(jl)
          pdmfdp(jl,jk-1)=zdmfdp
          prfl(jl)=prfl(jl)+zdmfdp
        END IF
      END DO
      !
      DO jt=1,ktrac
        DO jl=jcs,kproma
          IF(llo2(jl)) THEN
            pmfdxt(jl,jk,jt)=pxtd(jl,jk,jt)*pmfd(jl,jk)
          ENDIF
        END DO
      END DO
      !
      IF(lmfdudv) THEN
        DO jl=jcs,kproma
          IF(llo2(jl).AND.pmfd(jl,jk).LT.0._wp) THEN
            zmfduk=pmfd(jl,jk-1)*pud(jl,jk-1)+zdmfen(jl)*puen(jl,jk-1)-              &
              &                               zdmfde(jl)*pud(jl,jk-1)
            zmfdvk=pmfd(jl,jk-1)*pvd(jl,jk-1)+zdmfen(jl)*pven(jl,jk-1)-              &
              &                               zdmfde(jl)*pvd(jl,jk-1)
            pud(jl,jk)=zmfduk*(1._wp/MIN(-cmfcmin,pmfd(jl,jk)))
            pvd(jl,jk)=zmfdvk*(1._wp/MIN(-cmfcmin,pmfd(jl,jk)))
          END IF
        END DO
      END IF
      !
    END DO level

  END SUBROUTINE cuddraf

END MODULE mo_cudescent
