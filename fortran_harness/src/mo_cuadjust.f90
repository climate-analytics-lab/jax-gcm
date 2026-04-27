#include "fsel.inc"
#include "consistent_fma.inc"
!>
!! @brief Module produces t, q and l values for cloud ascent
!!
!! @remarks
!!     This routine is called from subroutines:
!!         *cubase*   (t and q at condensation level)
!!         *cuasc*    (t and q at cloud levels)
!!         *cuini*    (environmental t and qs values at half levels)
!!     Input are unadjusted t and q values, it returns adjusted values of t and q
!!     Note: input parameter kcall defines calculation as
!!          kcall=0   env. t and qs in *cuini*
!!          kcall=1   condensation in updrafts  (e.g. *cubase*, *cuasc*)
!!          kcall=2   evaporation in downdrafts (e.g. *cudlfs*, *cuddraf*)
!!
!! @author M. Tiedtke, ECMWF,    Dec 1989
!!         D. Salmond, Cray(UK), Aug. 1991
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
MODULE mo_cuadjust
  USE mo_kind,               ONLY: wp
  USE mo_physical_constants, ONLY: vtmpc1
  USE mo_echam_convect_tables,     ONLY: lookup_ua_list_spline, lookup_ubc_list

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: cuadjtq

CONTAINS
  !>
  !!
  SUBROUTINE cuadjtq(  jb, jcs, kproma, kbdim, klev, kk,                            &
    &                   pp,       pt,       pq,       ldidx, ldcnt,  kcall)

    !  Scalar arguments with intent(In):
    INTEGER,  INTENT (IN) :: kcall, kk, klev, jcs, kproma, kbdim, jb

    !  Array arguments with intent(In):
    REAL(wp), INTENT (IN) :: pp(kbdim)
    INTEGER,  INTENT (IN) :: ldidx(kbdim)
    INTEGER,  INTENT (IN) :: ldcnt

    !  Array arguments with intent(InOut):
    REAL(wp), INTENT (INOUT) :: pq(kbdim,klev), pt(kbdim,klev)

    !  Local scalars:
    REAL(wp):: zcond1, zdqsdt, zqsat, zes, zcor, zlcdqsdt
    INTEGER :: jl, nl, nsum

    !  Local arrays:
    REAL(wp) :: zcond(kbdim),zppi(kbdim)
    REAL(wp) :: ua(kbdim),dua(kbdim),ub(kbdim),uc(kbdim)
    INTEGER  :: idx(kbdim),ncond(kbdim)

    !  Executable statements

    !
    !---------------------------------------------------------------------------------
    !
    !     2.           CALCULATE CONDENSATION AND ADJUST T AND Q ACCORDINGLY
    !                  -----------------------------------------------------
    !

    IF (kcall >= 0.AND. kcall <= 2 ) THEN

      CALL lookup_ubc_list(jcs,kproma,ldcnt,ldidx(1),pt(1,kk),ub(1),uc(1))
      CALL lookup_ua_list_spline('cuadjtq (1)',jcs,kproma,ldcnt,ldidx(1),pt(1,kk),ua(1), &
        &                                      dua(1),klev=klev,kblock=jb,kblock_size=kbdim)

!DIR$ IVDEP
!OCL NOVREC
!IBM* ASSERT(NODEPS)
      DO nl=jcs,ldcnt
        jl = ldidx(nl)
        zppi(jl)=1._wp/pp(jl)
      END DO

      IF (kcall.EQ.0) THEN

      ! mpuetz: 40% of cuadtjq (lookup of tlucua und tlucub !!)

!DIR$ IVDEP
!OCL NOVREC
!IBM ASSERT(NODEPS)
        DO nl=jcs,ldcnt
          jl = ldidx(nl)

          zes  = ua(nl)*zppi(jl)
          zes  = MIN(0.5_wp,zes)
          zcor = SWDIV_NOCHK(1._wp,(1._wp-vtmpc1*zes))
          zqsat= zes*zcor

          zdqsdt = zppi(jl)*zcor**2*dua(nl)
          zlcdqsdt  = FSEL(zes-0.4_wp,zqsat*zcor*ub(nl),zdqsdt*uc(nl))

          zcond(jl) = SWDIV_NOCHK((pq(jl,kk)-zqsat),(1._wp+zlcdqsdt))

          pt(jl,kk) = pt(jl,kk) + uc(nl)*zcond(jl)
          pq(jl,kk) = pq(jl,kk) - zcond(jl)

          ! mpuetz: zcond is almost always > 0
          ncond(nl) = INT(FSEL(-ABS(zcond(jl)),0._wp,1._wp))
        END DO

      ELSE IF (kcall.EQ.1) THEN

!DIR$ IVDEP
!OCL NOVREC
!IBM ASSERT(NODEPS)
        DO nl=jcs,ldcnt
          jl = ldidx(nl)

          zes  = ua(nl)*zppi(jl)
          zes  = MIN(0.5_wp,zes)
          zcor = SWDIV_NOCHK(1._wp,(1._wp-vtmpc1*zes))
          zqsat= zes*zcor

          zdqsdt = zppi(jl)*zcor**2*dua(nl)
          zlcdqsdt  = FSEL(zes-0.4_wp,zqsat*zcor*ub(nl),zdqsdt*uc(nl))

          zcond(jl) = SWDIV_NOCHK((pq(jl,kk)-zqsat),(1._wp+zlcdqsdt))
          zcond(jl) = MAX(zcond(jl),0._wp)

          pt(jl,kk) = pt(jl,kk) + uc(nl)*zcond(jl)
          pq(jl,kk) = pq(jl,kk) - zcond(jl)
          ! mpuetz: zcond is almost always > 0
          ncond(nl) = INT(FSEL(-ABS(zcond(jl)),0._wp,1._wp))
        END DO

      ELSE

!DIR$ IVDEP
!OCL NOVREC
!IBM* ASSERT(NODEPS)
        DO nl=jcs,ldcnt
          jl = ldidx(nl)

          zes  = ua(nl)*zppi(jl)
          zes  = MIN(0.5_wp,zes)
          zcor = SWDIV_NOCHK(1._wp,(1._wp-vtmpc1*zes))
          zqsat= zes*zcor

          zdqsdt = zppi(jl)*zcor**2*dua(nl)
          zlcdqsdt  = FSEL(zes-0.4_wp,zqsat*zcor*ub(nl),zdqsdt*uc(nl))
          zcond(jl) = SWDIV_NOCHK((pq(jl,kk)-zqsat),(1._wp+zlcdqsdt))

          zcond(jl) = MIN(zcond(jl),0._wp)

          pt(jl,kk) = pt(jl,kk) + uc(nl)*zcond(jl)
          pq(jl,kk) = pq(jl,kk) - zcond(jl)
          ! mpuetz: zcond is almost always > 0
          ncond(nl) = INT(FSEL(-ABS(zcond(jl)),0._wp,1._wp))
        END DO

      END IF

      nsum = jcs
      DO nl=jcs,ldcnt
        idx(nsum) = ldidx(nl)
        nsum = nsum + ncond(nl)
      END DO
      nsum = nsum - 1

#ifdef __ibmdbg__
      print *,'cuad(',kcall,')',ldcnt,nsum,ldcnt
#endif

      IF (nsum > jcs-1) THEN

        CALL lookup_ubc_list(jcs,kproma,nsum,idx(1),pt(1,kk),ub(1),uc(1))
        CALL lookup_ua_list_spline('cuadjtq (2)',jcs,kproma,nsum,idx(1),pt(1,kk),ua(1),  &
          &                                      dua(1),klev=klev,kblock=jb,kblock_size=kbdim)

!PREVENT_INCONSISTENT_IFORT_FMA
!DIR$ IVDEP
!OCL NOVREC
!IBM* ASSERT(NODEPS)
!IBM* UNROLL(3)
        DO nl=jcs,nsum
          jl = idx(nl)

          zes  = ua(nl)*zppi(jl)
          zes  = MIN(0.5_wp,zes)
          zcor = SWDIV_NOCHK(1._wp,(1._wp-vtmpc1*zes))
          zqsat= zes*zcor

          zdqsdt = zppi(jl)*zcor**2*dua(nl)
          zlcdqsdt = FSEL(zes-0.4_wp,zqsat*zcor*ub(nl),zdqsdt*uc(nl))
          zcond1   = SWDIV_NOCHK((pq(jl,kk)-zqsat),(1._wp+zlcdqsdt))

          pt(jl,kk) = pt(jl,kk) + uc(nl)*zcond1
          pq(jl,kk) = pq(jl,kk) - zcond1
        END DO

      END IF

    END IF

  END SUBROUTINE cuadjtq

END MODULE mo_cuadjust
