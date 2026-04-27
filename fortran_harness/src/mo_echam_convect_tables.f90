!! @par Copyright and License
!!
!! This code is subject to the DWD and MPI-M-Software-License-Agreement in
!! its most recent form.
!! Please see the file LICENSE in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the
!! headers of the routines.
#if defined __xlC__ && !defined NOXLFPROCESS
@PROCESS HOT
#endif
#include "fsel.inc"

MODULE mo_echam_convect_tables

  ! Lookup tables for convective adjustment code
  !
  ! D. Salmond, CRAY (UK), August 1991, original code
  ! L. Kornblueh, MPI, April 2003, cleanup and move of table setup code
  !                            from setphys.f90 in module
  ! M. Puetz, IBM, April 2009, added spline option
  ! L. Kornblueh, MPI, July 2010, corrected inconsistent use of saturation
  !                            vapour formulae
  ! L. Kornblueh, August 2010, corrected index bug causing inconsistencies
  !                            in table lookup for splines

  USE mo_kind,      ONLY: wp
  USE mo_exception, ONLY: message_text, message, finish
  USE mo_physical_constants, ONLY: alv, als, rd, rv, tmelt, cpd
  USE mo_echam_cld_config,   ONLY: echam_cld_config

  ! Debug-only diagnostics dropped in the standalone harness (mo_model_domain,
  ! mo_math_constants used to print lon/lat for out-of-table-bound errors).
  ! The error path now just calls ``lookuperror`` directly.

  IMPLICIT NONE

  SAVE

  PRIVATE

  ! Reference: Sonntag D., 1990: Important new values of the physical
  ! constants of 1986, vapour pressure formulations based on ITS-90,
  ! and psychrometer formulae. Z. Meteor. 70, pp 340-344.

  REAL(wp), PARAMETER :: cavl1 = -6096.9385_wp
  REAL(wp), PARAMETER :: cavl2 =    21.2409642_wp
  REAL(wp), PARAMETER :: cavl3 =    -2.711193_wp
  REAL(wp), PARAMETER :: cavl4 =     1.673952_wp
  REAL(wp), PARAMETER :: cavl5 =     2.433502_wp

  REAL(wp), PARAMETER :: cavi1 = -6024.5282_wp
  REAL(wp), PARAMETER :: cavi2 =    29.32707_wp
  REAL(wp), PARAMETER :: cavi3 =     1.0613868_wp
  REAL(wp), PARAMETER :: cavi4 =    -1.3198825_wp
  REAL(wp), PARAMETER :: cavi5 =    -0.49382577_wp

  ! variables public

  PUBLIC :: jptlucu1          ! lookup table lower bound
  PUBLIC :: jptlucu2          ! lookup table upper bound

  PUBLIC :: fjptlucu1         ! lookup table lower bound (real)
  PUBLIC :: fjptlucu2         ! lookup table upper bound (real)

  PUBLIC :: tlucua            ! table -- e_s*Rd/Rv
  PUBLIC :: tlucub            ! table -- derivative: d es/ d t
  PUBLIC :: tlucuc            ! table -- lv/cp

  PUBLIC :: tlucuaw           ! table -- e_s*Rd/Rv - forcing saturation wrt water
  Public :: tlucubw           ! table -- derivative: d es/ d t - forcing saturation wrt water
  PUBLIC :: tlucucw           ! table -- lv/cp - forcing saturation wrt water

  PUBLIC :: tlucu             ! fused table
  PUBLIC :: tlucuw            ! fused table

  PUBLIC :: lookupoverflow
  PUBLIC :: flookupoverflow

  PUBLIC :: prepare_ua_index_spline

  PUBLIC :: lookup_ua_spline
  PUBLIC :: lookup_uaw_spline
  PUBLIC :: lookup_ua_eor_uaw_spline

  PUBLIC :: lookup_ua_list_spline
  PUBLIC :: lookup_uaw_list_spline

  PUBLIC :: prepare_ua_index

  PUBLIC :: lookup_ua
  PUBLIC :: lookup_uaw
  PUBLIC :: lookup_ua_eor_uaw

  PUBLIC :: lookup_ua_list
  PUBLIC :: lookup_uaw_list

  PUBLIC :: lookup_ubc
  PUBLIC :: lookup_ubc_list

  ! subroutines public

  PUBLIC :: init_convect_tables ! initialize LUTs
  PUBLIC :: lookuperror         ! error handling routine

  !---------------------------------------------------------------------------
  !
  ! Configuration section:
  !
  !-- temperature evaluation bounds:
  !
  REAL(wp), PARAMETER :: tlbound =  50.0_wp  ! lower bound [K]
  REAL(wp), PARAMETER :: tubound = 400.0_wp  ! upper bound [K]
  !
  !-- full table:
  !
  REAL(wp), PARAMETER :: fdeltat  =   0.001_wp  ! distance of table knots [K]
  ! division is not sufficiently precise, have to replace 1.0_wp/fdeltat
  REAL(wp), PARAMETER :: rfdeltat = 1000.0_wp
  !
  !-- spline interpolation:
  !
  REAL(wp), PARAMETER :: sdeltat  =   0.025_wp  ! distance of table knots [K]
  ! division is not sufficiently precise, have to replace 1.0_wp/sdeltat
  REAL(wp), PARAMETER :: rsdeltat =  40.0_wp
  !
  !---------------------------------------------------------------------------
  !
  ! Derived bounds and deltas:
  !
#ifdef __SX__
  !-- full table:
  !
  INTEGER,  PARAMETER :: jptlucu0 = 273150   ! lookup table physical center
  INTEGER,  PARAMETER :: jptlucu1 =  50000   ! lookup table lower bound
  INTEGER,  PARAMETER :: jptlucu2 = 400000   ! lookup table upper bound
  !
  !-- spline interpolation:
  !
  INTEGER,  PARAMETER :: lucupctr =  10926   ! lookup table physical center
  INTEGER,  PARAMETER :: lucupmin =   2000   ! lookup table lower bound
  INTEGER,  PARAMETER :: lucupmax =  16000   ! lookup table upper bound
#else
  !-- full table:
  !
  INTEGER,  PARAMETER :: jptlucu0 = NINT(rfdeltat*tmelt)   ! lookup table physical center
  INTEGER,  PARAMETER :: jptlucu1 = NINT(rfdeltat*tlbound) ! lookup table lower bound
  INTEGER,  PARAMETER :: jptlucu2 = NINT(rfdeltat*tubound) ! lookup table upper bound
  !
  !-- spline interpolation:
  !
  INTEGER,  PARAMETER :: lucupctr = NINT(rsdeltat*tmelt)   ! lookup table physical center
  INTEGER,  PARAMETER :: lucupmin = NINT(rsdeltat*tlbound) ! lookup table lower bound
  INTEGER,  PARAMETER :: lucupmax = NINT(rsdeltat*tubound) ! lookup table upper bound
#endif
  !
  !   For the spline interpolation the melting point tmelt = 273.15
  !   is resembled twice at index = lucubctr and index = lucubctr-1
  !   to be able to interpolate values close to the melting point
  !   correctly.
  !
  INTEGER, PARAMETER :: idxctr   = lucupctr
  INTEGER, PARAMETER :: idxctrm1 = idxctr-1
  !
  !---------------------------------------------------------------------------

  ! add float point constants for accelerated boundary guard code
  ! original big table
#ifdef __SX__
  REAL(wp), PARAMETER :: fjptlucu1 =  50000.0_wp
  REAL(wp), PARAMETER :: fjptlucu2 = 400000.0_wp
  ! new basic table bounds
  REAL(wp), PARAMETER :: flucupmin =   2000.0_wp
  REAL(wp), PARAMETER :: flucupmax =  16000.0_wp
#else
  REAL(wp), PARAMETER :: fjptlucu1 = REAL(jptlucu1,wp)
  REAL(wp), PARAMETER :: fjptlucu2 = REAL(jptlucu2,wp)
  ! new basic table bounds
  REAL(wp), PARAMETER :: flucupmin = REAL(lucupmin,wp)
  REAL(wp), PARAMETER :: flucupmax = REAL(lucupmax,wp)
#endif

  ! logical is fine and logical, but fp faster
  LOGICAL  :: lookupoverflow = .FALSE.          ! preset with false
  REAL(wp) :: flookupoverflow = 0.0_wp          ! preset with false

  REAL(wp) :: tlucua(jptlucu1-1:jptlucu2+1)     ! table - Es*Rd/Rv, mixed phases
  REAL(wp) :: tlucuaw(jptlucu1-1:jptlucu2+1)    ! table - Es*Rd/Rv, water phase only
  REAL(wp) :: tlucuad(jptlucu1-1:jptlucu2+1)    ! table - dEs/dT*Rd/Rv, mixed phases
  REAL(wp) :: tlucuawd(jptlucu1-1:jptlucu2+1)   ! table - dEs/dT*Rd/Rv, water phase only
  REAL(wp) :: tlucub(jptlucu1-1:jptlucu2+1)     ! table - inner dEs/dT*L/cp, mixed phases
  REAL(wp) :: tlucubw(jptlucu1-1:jptlucu2+1)    ! table - inner dEs/dT*L/cp, water phase only
  REAL(wp) :: tlucuc(jptlucu1-1:jptlucu2+1)     ! table - L/cp, mixed phases
  REAL(wp) :: tlucucw(jptlucu1-1:jptlucu2+1)    ! table - L/cp, water phase only

  ! fused tables for splines
  REAL(wp) :: tlucu(1:2,lucupmin-2:lucupmax+1)     ! fused table
  REAL(wp) :: tlucuw(1:2,lucupmin-2:lucupmax+1)    ! fused table

  !----------------------------------------------------------------------------
CONTAINS
  !----------------------------------------------------------------------------
  SUBROUTINE init_convect_tables
    REAL(wp) :: zalvdcp, zalsdcp
    REAL(wp) :: ztt, zldcp
    REAL(wp) :: zavm1, zavm2, zavm3, zavm4, zavm5
    REAL(wp) :: zminner,zdminner,zlinner,zdlinner
    INTEGER :: it

    ! check if tmelt is hit by one of the table entries (tricky to make right!):
    IF (ABS(REAL(lucupctr,wp)-rsdeltat*tmelt) > EPSILON(1.0_wp)) THEN
      CALL finish('init_convect_tables','distance of spline knots incorrect.')
    ENDIF

    WRITE (message_text,'(a,f8.2,a)')  'Melting temperature  : ', tmelt, ' [K]'
    CALL message('',message_text)
    WRITE (message_text,'(a,f8.2,a)')  'Lower table bound    : ', tlbound, ' [K]'
    CALL message('',message_text)
    WRITE (message_text,'(a,f8.2,a)')  'Upper table bound    : ', tubound, ' [K]'
    CALL message('',message_text)
    WRITE (message_text,'(a)')         'Full table settings  : '
    CALL message('',message_text)
    WRITE (message_text,'(a,f12.3,a)') '  Temperature delta  : ', fdeltat, ' [K]'
    CALL message('',message_text)
    WRITE (message_text,'(a,f12.3,a)') '  Inverse delta      : ', rfdeltat, ' [1/K]'
    CALL message('',message_text)
    WRITE (message_text,'(a,3i8)')     '  Indices [lb,ctr,ub]: ', jptlucu1, jptlucu0, jptlucu2
    CALL message('',message_text)
    WRITE (message_text,'(a)')         'Spline table settings: '
    CALL message('',message_text)
    WRITE (message_text,'(a,f12.3,a)') '  Temperature delta  : ', sdeltat, ' [K]'
    CALL message('',message_text)
    WRITE (message_text,'(a,f12.3,a)') '  Inverse delta      : ', rsdeltat, ' [1/K]'
    CALL message('',message_text)
    WRITE (message_text,'(a,3i8)')     '  Indices [lb,ctr,ub]: ', lucupmin, lucupctr, lucupmax
    CALL message('',message_text)
    WRITE (message_text,'(a,2i8)')     '  Table split indices: ', idxctr, idxctrm1
    CALL message('',message_text)

    zalvdcp = alv/cpd
    zalsdcp = als/cpd

    ! extend the table by one to safe guard code for accessing it+1, when
    ! it is already checked

    ! This are the old tables, extending over 350000 values: kills performance

!IBM* NOVECTOR
    DO it = jptlucu1-1, jptlucu2+1

      ztt = fdeltat*it
      IF ((ztt-tmelt) > 0.0_wp) THEN
        zldcp = zalvdcp
        zavm1 = cavl1
        zavm2 = cavl2
        zavm3 = cavl3
        zavm4 = cavl4
        zavm5 = cavl5
      ELSE
        zldcp = zalsdcp
        zavm1 = cavi1
        zavm2 = cavi2
        zavm3 = cavi3
        zavm4 = cavi4
        zavm5 = cavi5
      END IF

      zminner  = (zavm1/ztt+zavm2+zavm3*0.01_wp*ztt+zavm4*ztt*ztt*1.e-5_wp+zavm5*LOG(ztt))
      zdminner = (-zavm1/(ztt*ztt) + zavm3*0.01_wp + zavm4*ztt*2.e-5_wp + zavm5/ztt)

      zlinner  = (cavl1/ztt+cavl2+cavl3*0.01_wp*ztt+cavl4*ztt*ztt*1.e-5_wp+cavl5*LOG(ztt))
      zdlinner = (-cavl1/(ztt*ztt) + cavl3*0.01_wp + cavl4*ztt*2.e-5_wp + cavl5/ztt)

      tlucua(it)  = EXP(zminner)*rd/rv
      tlucuaw(it) = EXP(zlinner)*rd/rv

      tlucuad(it)  = zdminner*EXP(zminner)*rd/rv
      tlucuawd(it) = zdlinner*EXP(zlinner)*rd/rv

      tlucub(it)  = zldcp*zdminner
      tlucubw(it) = zalvdcp*zdlinner

      tlucuc(it)  = zldcp
      tlucucw(it) = zalvdcp

    END DO

    ! This are the new tables, base for the spline interpolation

!IBM* NOVECTOR
    DO it = lucupmin-2, idxctrm1
      ztt = sdeltat*(it+1)
      zldcp = zalsdcp
      zavm1 = cavi1
      zavm2 = cavi2
      zavm3 = cavi3
      zavm4 = cavi4
      zavm5 = cavi5

      zminner  = (zavm1/ztt+zavm2+zavm3*0.01_wp*ztt+zavm4*ztt*ztt*1.e-5_wp+zavm5*LOG(ztt))
      zdminner = (-zavm1/(ztt*ztt) + zavm3*0.01_wp + zavm4*ztt*2.e-5_wp + zavm5/ztt)

      zlinner  = (cavl1/ztt+cavl2+cavl3*0.01_wp*ztt+cavl4*ztt*ztt*1.e-5_wp+cavl5*LOG(ztt))
      zdlinner = (-cavl1/(ztt*ztt) + cavl3*0.01_wp + cavl4*ztt*2.e-5_wp + cavl5/ztt)

      tlucu(1,it)  = EXP(zminner)*rd/rv
      tlucu(2,it)  = sdeltat*zdminner*EXP(zminner)*rd/rv

      tlucuw(1,it) = EXP(zlinner)*rd/rv
      tlucuw(2,it) = sdeltat*zdlinner*EXP(zlinner)*rd/rv

    END DO

!IBM* NOVECTOR
    DO it = idxctr, lucupmax+1
      ztt = sdeltat*it
      zldcp = zalvdcp
      zavm1 = cavl1
      zavm2 = cavl2
      zavm3 = cavl3
      zavm4 = cavl4
      zavm5 = cavl5

      zminner  = (zavm1/ztt+zavm2+zavm3*0.01_wp*ztt+zavm4*ztt*ztt*1.e-5_wp+zavm5*LOG(ztt))
      zdminner = (-zavm1/(ztt*ztt) + zavm3*0.01_wp + zavm4*ztt*2.e-5_wp + zavm5/ztt)

      zlinner  = (cavl1/ztt+cavl2+cavl3*0.01_wp*ztt+cavl4*ztt*ztt*1.e-5_wp+cavl5*LOG(ztt))
      zdlinner = (-cavl1/(ztt*ztt) + cavl3*0.01_wp + cavl4*ztt*2.e-5_wp + cavl5/ztt)

      tlucu(1,it)  = EXP(zminner)*rd/rv
      tlucu(2,it)  = sdeltat*zdminner*EXP(zminner)*rd/rv

      tlucuw(1,it) = EXP(zlinner)*rd/rv
      tlucuw(2,it) = sdeltat*zdlinner*EXP(zlinner)*rd/rv
    END DO

  END SUBROUTINE init_convect_tables
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ubc(jcs,size, temp, ub, uc)
    INTEGER,            INTENT(in)  :: jcs, size
    REAL(wp),           INTENT(in)  :: temp(size)
    REAL(wp),           INTENT(out) :: ub(size)
    REAL(wp), OPTIONAL, INTENT(out) :: uc(size)

    REAL(wp) :: zalvdcp, zalsdcp, zavm1, zavm3, zavm4, zavm5, zldcp
    INTEGER :: jl

    zalvdcp = alv/cpd
    zalsdcp = als/cpd

    IF (PRESENT(uc)) THEN
!IBM* NOVECTOR
      DO jl = jcs, size
        zavm1 = FSEL(tmelt-temp(jl),cavi1,cavl1)
        zavm3 = FSEL(tmelt-temp(jl),cavi3,cavl3)
        zavm4 = FSEL(tmelt-temp(jl),cavi4,cavl4)
        zavm5 = FSEL(tmelt-temp(jl),cavi5,cavl5)
        zldcp = FSEL(tmelt-temp(jl),zalsdcp,zalvdcp)
        ub(jl) = zldcp*(-zavm1/(temp(jl)*temp(jl))+zavm3*0.01_wp+zavm4*temp(jl)*2.e-5_wp+zavm5/temp(jl))
        uc(jl) = zldcp
      END DO
    ELSE
!IBM* NOVECTOR
      DO jl = jcs, size
        zavm1 = FSEL(tmelt-temp(jl),cavi1,cavl1)
        zavm3 = FSEL(tmelt-temp(jl),cavi3,cavl3)
        zavm4 = FSEL(tmelt-temp(jl),cavi4,cavl4)
        zavm5 = FSEL(tmelt-temp(jl),cavi5,cavl5)
        zldcp = FSEL(tmelt-temp(jl),zalsdcp,zalvdcp)
        ub(jl) = zldcp*(-zavm1/(temp(jl)*temp(jl))+zavm3*0.01_wp+zavm4*temp(jl)*2.e-5_wp+zavm5/temp(jl))
      END DO
    END IF

  END SUBROUTINE lookup_ubc
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ubc_list(jcs, size, kidx, list, temp, ub, uc)
    INTEGER,            INTENT(in)  :: jcs, size, kidx
    INTEGER,            INTENT(in)  :: list(kidx)
    REAL(wp),           INTENT(in)  :: temp(size)
    REAL(wp),           INTENT(out) :: ub(size)
    REAL(wp), OPTIONAL, INTENT(out) :: uc(size)

    REAL(wp) :: zalvdcp, zalsdcp, zavm1, zavm3, zavm4, zavm5, zldcp
    INTEGER :: nl, jl

    zalvdcp = alv/cpd
    zalsdcp = als/cpd

    IF (PRESENT(uc)) THEN
!IBM* NOVECTOR
!IBM* ASSERT(NODEPS)
      DO nl = jcs, kidx
        jl = list(nl)
        zavm1 = FSEL(tmelt-temp(jl),cavi1,cavl1)
        zavm3 = FSEL(tmelt-temp(jl),cavi3,cavl3)
        zavm4 = FSEL(tmelt-temp(jl),cavi4,cavl4)
        zavm5 = FSEL(tmelt-temp(jl),cavi5,cavl5)
        zldcp = FSEL(tmelt-temp(jl),zalsdcp,zalvdcp)
        ub(nl) = zldcp*(-zavm1/(temp(jl)*temp(jl))+zavm3*0.01_wp+zavm4*temp(jl)*2.e-5_wp+zavm5/temp(jl))
        uc(nl) = zldcp
      END DO
    ELSE
!IBM* NOVECTOR
!IBM* ASSERT(NODEPS)
      DO nl = jcs, kidx
        jl = list(nl)
        zavm1 = FSEL(tmelt-temp(jl),cavi1,cavl1)
        zavm3 = FSEL(tmelt-temp(jl),cavi3,cavl3)
        zavm4 = FSEL(tmelt-temp(jl),cavi4,cavl4)
        zavm5 = FSEL(tmelt-temp(jl),cavi5,cavl5)
        zldcp = FSEL(tmelt-temp(jl),zalsdcp,zalvdcp)
        ub(nl) = zldcp*(-zavm1/(temp(jl)*temp(jl))+zavm3*0.01_wp+zavm4*temp(jl)*2.e-5_wp+zavm5/temp(jl))
      END DO
    END IF

  END SUBROUTINE lookup_ubc_list
  !----------------------------------------------------------------------------
  SUBROUTINE fetch_ua_spline(jcs,size,idx,zalpha,table,ua,dua)
    INTEGER,            INTENT(in)  :: jcs, size
    INTEGER,            INTENT(in)  :: idx(size)
    REAL(wp),           INTENT(in)  :: zalpha(size)
    REAL(wp),           INTENT(in)  :: table(1:2,lucupmin-2:lucupmax+1)
    REAL(wp), OPTIONAL, INTENT(out) :: ua(size), dua(size)

    REAL(wp) :: a, b, c, d, dx, ddx, x, bxa
    INTEGER :: jl

    IF (PRESENT(ua) .AND. .NOT. PRESENT(dua)) THEN
      DO jl = jcs,size
        x = zalpha(jl)
        ! derivative and second derivative approximations (2 flops)
        dx   = table(1,idx(jl)+1) - table(1,idx(jl))
        ddx  = table(2,idx(jl)+1) + table(2,idx(jl))
        ! determine coefficients (2 fma + 1 flop)
        a = ddx - 2.0_wp*dx
        b = 3.0_wp*dx - ddx - table(2,idx(jl))
        c = table(2,idx(jl))
        d = table(1,idx(jl))
        ! Horner's scheme to compute the spline functions (3 fmas)
        bxa = b + x*a
        ua(jl) = d + x*(c + x*bxa)
      END DO
    ELSE IF (PRESENT(ua) .AND. PRESENT(dua)) THEN
      DO jl = jcs,size
        x = zalpha(jl)
        ! derivate and second derivate approximations (2 flops)
        dx   = table(1,idx(jl)+1) - table(1,idx(jl))
        ddx  = table(2,idx(jl)+1) + table(2,idx(jl))
        ! determine coefficients (2 fma + 1 flop)
        a = ddx - 2.0_wp*dx
        b = 3.0_wp*dx - ddx - table(2,idx(jl))
        c = table(2,idx(jl))
        d = table(1,idx(jl))
        ! Horner's scheme to compute the spline functions (5 fmas + 1 flop)
        bxa = b + x*a
        ua(jl)  = d + x*(c + x*bxa)
        dua(jl) = rsdeltat*(c + x*(3.0_wp*bxa - b))
      END DO
    END IF

  END SUBROUTINE fetch_ua_spline
  !----------------------------------------------------------------------------
  SUBROUTINE fetch_ua(size, idx, table, u)
    INTEGER,  INTENT(in)  :: size
    INTEGER,  INTENT(in)  :: idx(size)
    REAL(wp), INTENT(in)  :: table(jptlucu1-1:jptlucu2+1)
    REAL(wp), INTENT(out) :: u(size)

    INTEGER :: jl

    DO jl = 1, size
      u(jl) = table(idx(jl))
    END DO

  END SUBROUTINE fetch_ua
  !----------------------------------------------------------------------------
  SUBROUTINE fetch_ua_list_spline(size, kidx1, kidx2, store_idx, lookup_idx, zalpha, table, ua, dua)
    INTEGER,            INTENT(in)    :: size, kidx1, kidx2
    INTEGER,            INTENT(in)    :: store_idx(size), lookup_idx(size)
    REAL(wp),           INTENT(in)    :: zalpha(size)
    REAL(wp),           INTENT(in)    :: table(1:2,lucupmin-2:lucupmax+1)
    REAL(wp), OPTIONAL, INTENT(inout) :: ua(size), dua(size)

    REAL(wp) :: a, b, c, d, dx, ddx, x, bxa
    INTEGER  :: jl, nl

    IF (PRESENT(ua) .AND. .NOT. PRESENT(dua)) THEN
!IBM* ASSERT(NODEPS)
      DO nl = kidx1, kidx2
        jl = store_idx(nl)
        x = zalpha(jl)
        ! derivate and second derivate approximations (2 flops)
        dx   = table(1,lookup_idx(jl)+1) - table(1,lookup_idx(jl))
        ddx  = table(2,lookup_idx(jl)+1) + table(2,lookup_idx(jl))
        ! determine coefficients (2 fma + 1 flop)
        a = ddx - 2.0_wp*dx
        b = 3.0_wp*dx - ddx - table(2,lookup_idx(jl))
        c = table(2,lookup_idx(jl))
        d = table(1,lookup_idx(jl))
        ! Horner's scheme to compute the spline functions (3 fmas)
        bxa = b + x*a
        ua(jl) = d + x*(c + x*bxa)
      END DO
    ELSE IF (PRESENT(ua) .AND. PRESENT(dua)) THEN

!IBM* ASSERT(NODEPS)
      DO nl = kidx1, kidx2
        jl = store_idx(nl)
        x = zalpha(jl)
        ! derivate and second derivate approximations (2 flops)
        dx   = table(1,lookup_idx(jl)+1) - table(1,lookup_idx(jl))
        ddx  = table(2,lookup_idx(jl)+1) + table(2,lookup_idx(jl))
        ! determine coefficients (2 fma + 1 flop)
        a = ddx - 2.0_wp*dx
        b = 3.0_wp*dx - ddx - table(2,lookup_idx(jl))
        c = table(2,lookup_idx(jl))
        d = table(1,lookup_idx(jl))
        ! Horner's scheme to compute the spline functions (5 fmas + 1 flop)
        bxa = b + x*a
        ua(jl)  = d + x*(c + x*bxa)
        dua(jl) = rsdeltat*(c + x*(3.0_wp*bxa - b))
      END DO
    END IF

  END SUBROUTINE fetch_ua_list_spline
  !----------------------------------------------------------------------------
  SUBROUTINE fetch_ua_list(size, kidx, store_idx, lookup_idx, table, u)
    INTEGER,  INTENT(in)    :: size, kidx
    INTEGER,  INTENT(in)    :: store_idx(kidx), lookup_idx(size)
    REAL(wp), INTENT(in)    :: table(jptlucu1-1:jptlucu2+1)
    REAL(wp), INTENT(inout) :: u(size)

    INTEGER :: jl, nl

!IBM* ASSERT(NODEPS)
    DO nl = 1, kidx
      jl = store_idx(nl)
      u(jl) = table(lookup_idx(jl))
    END DO

  END SUBROUTINE fetch_ua_list
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ua_eor_uaw_spline(jcs, size, idx, zalpha, nphase, iphase, ua, dua)
    INTEGER,            INTENT(in)  :: jcs, size, nphase
    INTEGER,            INTENT(in)  :: idx(size), iphase(size)
    REAL(wp),           INTENT(in)  :: zalpha(size)
    REAL(wp), OPTIONAL, INTENT(out) :: ua(size), dua(size)

    INTEGER :: tmpidx(size)
    INTEGER :: jl,iw,inw

!    REAL(wp) :: ua1(size), dua1(size)

    IF (nphase == 0) THEN ! temp > 0 for all indices
      IF (PRESENT(dua)) THEN
        CALL fetch_ua_spline(jcs, size, idx, zalpha, tlucuw, ua, dua)
      ELSE
        CALL fetch_ua_spline(jcs, size, idx, zalpha, tlucuw, ua)
      END IF

    ELSE IF (nphase == size-jcs+1) THEN ! temp < 0 for all indices
      IF (PRESENT(dua)) THEN
        CALL fetch_ua_spline(jcs, size, idx, zalpha, tlucu, ua, dua)
      ELSE
        CALL fetch_ua_spline(jcs, size, idx, zalpha, tlucu, ua)
      END IF

    ELSE
      ! mixed case, must build store index
      iw = jcs    ! store indices with temp < tmelt at iw (iw = jcs:jcs+nphase-1)
      inw = size  ! store indices with temp >= tmelt at inw (inw = jcs+nphase : size)
      DO jl = jcs,size
        tmpidx(iw)  = jl  ! lower part of tmpidx() filled with cond = .true.
        tmpidx(inw) = jl  ! upper part of tmpidx() filled with cond = .false.
        iw = iw + iphase(jl)          ! iphase(jl)=1 if temp(jl) < tmelt, =0 if temp .ge. tmelt
        inw = inw - (1 - iphase(jl))
      END DO
      iw = iw - 1
      IF (PRESENT(dua)) THEN
        CALL fetch_ua_list_spline(size, jcs, iw+1, tmpidx, idx, zalpha, tlucu , ua=ua, dua=dua)
        CALL fetch_ua_list_spline(size, iw+1, size, tmpidx, idx, zalpha, tlucuw, ua=ua, dua=dua)
      ELSE
        CALL fetch_ua_list_spline(size, jcs, iw+1, tmpidx, idx, zalpha, tlucu , ua=ua)
        CALL fetch_ua_list_spline(size, iw+1, size, tmpidx, idx, zalpha, tlucuw, ua=ua)
      END IF
    END IF

  END SUBROUTINE lookup_ua_eor_uaw_spline
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ua_eor_uaw(size, idx, nphase, iphase, ua, dua)
    INTEGER,            INTENT(in)    :: size, nphase
    INTEGER,            INTENT(in)    :: idx(size), iphase(size)
    REAL(wp),           INTENT(inout) :: ua(size)
    REAL(wp), OPTIONAL, INTENT(inout) :: dua(size)

    INTEGER :: tmpidx(size)
    INTEGER :: jl, iw, inw

    IF (nphase == 0) THEN
      ! only water
      CALL fetch_ua(size,idx,tlucuaw,ua)
      IF (PRESENT(dua)) CALL fetch_ua(size,idx,tlucuawd,dua)
    ELSE IF (nphase == size) THEN
      ! only ice
      CALL fetch_ua(size,idx,tlucua,ua)
      IF (PRESENT(dua)) CALL fetch_ua(size,idx,tlucuad,dua)
    ELSE
      ! mixed case, must build store index
      iw = 1
      inw = size
      DO jl = 1,size
        tmpidx(iw)  = jl  ! lower part of tmpidx() filled with cond = .true.
        tmpidx(inw) = jl  ! upper part of tmpidx() filled with cond = .false.
        iw = iw + iphase(jl)
        inw = inw - (1 - iphase(jl))
      END DO
      iw = iw - 1
      CALL fetch_ua_list(size, iw     , tmpidx(1)   , idx, tlucua, ua)
      CALL fetch_ua_list(size, size-iw, tmpidx(iw+1), idx, tlucuaw, ua)
      IF (PRESENT(dua)) THEN
        CALL fetch_ua_list(size, iw     , tmpidx(1)   , idx, tlucuad, dua)
        CALL fetch_ua_list(size, size-iw, tmpidx(iw+1), idx, tlucuawd, dua)
      END IF
    END IF

  END SUBROUTINE lookup_ua_eor_uaw
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ua_spline(jcs, size, idx, zalpha, ua, dua)
    INTEGER,            INTENT(in)  :: jcs, size
    INTEGER,            INTENT(in)  :: idx(size)
    REAL(wp),           INTENT(in)  :: zalpha(size)
    REAL(wp), OPTIONAL, INTENT(out) :: ua(size), dua(size)

    CALL fetch_ua_spline(jcs, size, idx, zalpha, tlucu, ua, dua)

  END SUBROUTINE lookup_ua_spline
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ua(size, idx, ua, dua)
    INTEGER,            INTENT(in)    :: size
    INTEGER,            INTENT(in)    :: idx(size)
    REAL(wp),           INTENT(inout) :: ua(size)
    REAL(wp), OPTIONAL, INTENT(inout) :: dua(size)

    CALL fetch_ua(size, idx, tlucua, ua)
    IF (PRESENT(dua)) CALL fetch_ua(size, idx, tlucuad, dua)

  END SUBROUTINE lookup_ua
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_uaw_spline(jcs, size, idx, zalpha, ua, dua)
    INTEGER,            INTENT(in)  :: jcs, size
    INTEGER,            INTENT(in)  :: idx(size)
    REAL(wp),           INTENT(in)  :: zalpha(size)
    REAL(wp), OPTIONAL, INTENT(out) :: ua(size), dua(size)

    CALL fetch_ua_spline(jcs, size,idx,zalpha,tlucuw,ua,dua)

  END SUBROUTINE lookup_uaw_spline
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_uaw(size, idx, ua, dua)
    INTEGER,            INTENT(in)  :: size
    INTEGER,            INTENT(in)  :: idx(size)
    REAL(wp), OPTIONAL, INTENT(out) :: ua(size), dua(size)

    CALL fetch_ua(size, idx, tlucuaw, ua)
    IF (PRESENT(dua)) CALL fetch_ua(size, idx, tlucuawd, dua)

  END SUBROUTINE lookup_uaw
  !----------------------------------------------------------------------------
  SUBROUTINE prepare_ua_index_spline(jg, name, jcs, size, temp, idx, zalpha, &
    &                                xi, nphase, zphase, iphase,             &
    &                                klev, kblock, kblock_size)
    INTEGER,            INTENT(in)  :: jg
    CHARACTER(len=*),   INTENT(in)  :: name
    INTEGER,            INTENT(in)  :: size, jcs ! start and end index of block
    REAL(wp),           INTENT(in)  :: temp(size)
    INTEGER,            INTENT(out) :: idx(size)
    REAL(wp),           INTENT(out) :: zalpha(size)

    INTEGER,  OPTIONAL, INTENT(in)  :: klev
    INTEGER,  OPTIONAL, INTENT(in)  :: kblock
    INTEGER,  OPTIONAL, INTENT(in)  :: kblock_size

    REAL(wp), OPTIONAL, INTENT(in)  :: xi(size)
    INTEGER,  OPTIONAL, INTENT(out) :: nphase
    REAL(wp), OPTIONAL, INTENT(out) :: zphase(size)
    INTEGER,  OPTIONAL, INTENT(out) :: iphase(size)

    REAL(wp) :: ztt, ztshft, zinbounds, ztmin,ztmax,znphase,ztest
    INTEGER :: jl

    ! Shortcuts to components of echam_cld_config
    !
    REAL(wp), POINTER :: csecfrl, cthomi
    !
    csecfrl => echam_cld_config(jg)% csecfrl
    cthomi  => echam_cld_config(jg)% cthomi

    zinbounds = 1._wp
    ztmin = flucupmin
    ztmax = flucupmax
    IF (PRESENT(xi)) THEN
      znphase = 0.0_wp
      DO jl = jcs,size
        ztshft = FSEL(tmelt-temp(jl),1.0_wp,0.0_wp)
        ztt = rsdeltat*temp(jl)
        zalpha(jl) = ztt - AINT(ztt)
        idx(jl) = INT(ztt-ztshft)
        zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
        zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)

        ! check dual phase conditions:
        ! lo2 = (ptm1(jl,jk) < cthomi) .OR. (ptm1(jl,jk) < tmelt .AND. zxised > csecfrl)
        ztest = FSEL(temp(jl)-tmelt,0.0_wp,1.0_wp)
        ztest = FSEL(csecfrl-xi(jl),0.0_wp,ztest)
        ztest = FSEL(temp(jl)-cthomi,ztest,1.0_wp)
        ! normalize ztest to 0 and 1
        iphase(jl) = INT(ztest)
        zphase(jl) = ztest-0.5_wp
        znphase = znphase + ztest
      END DO
      nphase = INT(znphase)
    ELSE
      DO jl = jcs, size
        ztshft = FSEL(tmelt-temp(jl),1.0_wp,0.0_wp)
        ztt = rsdeltat*temp(jl)
        zalpha(jl) = ztt - AINT(ztt)
        idx(jl) = INT(ztt-ztshft)
        zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
        zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)
      END DO
    END IF
    ! if one index was out of bounds -> print error and exit
    ! Debug lon/lat/cell-ID printing removed for the standalone harness.
    IF (zinbounds == 0.0_wp) THEN
      CALL lookuperror(name, 'prepare_ua_index_spline')
    END IF

  END SUBROUTINE prepare_ua_index_spline
  !----------------------------------------------------------------------------
  SUBROUTINE prepare_ua_index(jg, name, size, temp, idx, xi, nphase, zphase, iphase)
    INTEGER,            INTENT(in)  :: jg
    CHARACTER(len=*),   INTENT(in)  :: name
    INTEGER,            INTENT(in)  :: size
    REAL(wp),           INTENT(in)  :: temp(size)
    INTEGER,            INTENT(out) :: idx(size)
    REAL(wp), OPTIONAL, INTENT(in)  :: xi(size)
    INTEGER,  OPTIONAL, INTENT(out) :: nphase
    REAL(wp), OPTIONAL, INTENT(out) :: zphase(size)
    INTEGER,  OPTIONAL, INTENT(out) :: iphase(size)

    REAL(wp) :: ztt, zinbounds, ztmin,ztmax,znphase,ztest
    INTEGER :: jl

    ! Shortcuts to components of echam_cld_config
    !
    REAL(wp), POINTER :: csecfrl, cthomi
    !
    csecfrl => echam_cld_config(jg)% csecfrl
    cthomi  => echam_cld_config(jg)% cthomi

    ! first compute all lookup indices and check if they are all within allowed bounds

    zinbounds = 1.0_wp
    ztmin = fjptlucu1
    ztmax = fjptlucu2
    IF (PRESENT(xi)) THEN
      znphase = 0.0_wp
      DO jl = 1, size
        ztt = ANINT(rfdeltat*temp(jl))
        idx(jl) = INT(ztt)
        zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
        zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)
        ! check dual phase conditions
        ! lo2 = (ptm1(jl,jk) < cthomi) .OR. (ptm1(jl,jk) < tmelt .AND. zxised > csecfrl)
        ztest = FSEL(temp(jl)-tmelt,0.0_wp,1.0_wp)
        ztest = FSEL(csecfrl-xi(jl),0.0_wp,ztest)
        ztest = FSEL(temp(jl)-cthomi,ztest,1.0_wp)
        ! normalize ztest to 0 and 1
        iphase(jl) = INT(ztest)
        zphase(jl) = ztest-0.5_wp
        znphase = znphase + ztest
      END DO
      nphase = INT(znphase)
    ELSE
      DO jl = 1,size
        ztt = ANINT(rfdeltat*temp(jl))
        idx(jl) = INT(ztt)
        zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
        zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)
      END DO
    END IF
    ! if one index was out of bounds -> print error and exit
    IF (zinbounds == 0.0_wp) CALL lookuperror(name, 'prepare_ua_index')

  END SUBROUTINE prepare_ua_index
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ua_list_spline(name, jcs, size, kidx, list, temp, ua, dua, &
    &                              klev, kblock, kblock_size)
    CHARACTER(len=*),   INTENT(in)  :: name
    INTEGER,            INTENT(in)  :: jcs, size, kidx
    INTEGER,            INTENT(in)  :: list(kidx)
    REAL(wp),           INTENT(in)  :: temp(size)
    REAL(wp), OPTIONAL, INTENT(out) :: ua(size), dua(size)
    INTEGER,  OPTIONAL, INTENT(in)  :: klev
    INTEGER,  OPTIONAL, INTENT(in)  :: kblock
    INTEGER,  OPTIONAL, INTENT(in)  :: kblock_size

    INTEGER :: idx(size)
    REAL(wp) :: zalpha(size)

    REAL(wp) :: ztt, ztshft, zinbounds, ztmax, ztmin
    INTEGER :: nl, jl

    zinbounds = 1.0_wp
    ztmin = flucupmin
    ztmax = flucupmax

    ! first compute all lookup indices and check if they are all within allowed bounds

!IBM* ASSERT(NODEPS)
    DO nl = jcs, kidx
      jl = list(nl)
      ztshft = FSEL(tmelt-temp(jl),1.0_wp,0.0_wp)
      ztt = rsdeltat*temp(jl)
      zalpha(nl) = ztt - AINT(ztt)
      idx(nl) = INT(ztt-ztshft)
      zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
      zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)
    END DO
    ! if one index was out of bounds -> print error and exit
    ! Debug lon/lat/cell-ID printing removed for the standalone harness.
    IF (zinbounds == 0.0_wp) THEN
      WRITE(0,*) 'lookup_ua_list_spline name=', TRIM(name)
      WRITE(0,*) '  bounds(K) = ', flucupmin/rsdeltat, ' .. ', flucupmax/rsdeltat
      WRITE(0,*) '  jcs=', jcs, ' kidx=', kidx
      DO nl = jcs, kidx
        jl = list(nl)
        WRITE(0,*) '  nl=', nl, ' col=', jl, ' T=', temp(jl)
      END DO
      CALL lookuperror(name, 'lookup_ua_list_spline')
    ENDIF
    CALL fetch_ua_spline(jcs, kidx, idx, zalpha, tlucu, ua, dua)

  END SUBROUTINE lookup_ua_list_spline
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_ua_list(name, size, kidx, list, temp, ua, dua)
    CHARACTER(len=*),   INTENT(in)  :: name
    INTEGER,            INTENT(in)  :: size, kidx
    INTEGER,            INTENT(in)  :: list(kidx)
    REAL(wp),           INTENT(in)  :: temp(size)
    REAL(wp), OPTIONAL, INTENT(out) :: ua(size), dua(size)

    INTEGER :: idx(size)

    REAL(wp) :: ztt, zinbounds, ztmax, ztmin
    INTEGER :: nl, jl

    zinbounds = 1.0_wp
    ztmin = fjptlucu1
    ztmax = fjptlucu2

    ! first compute all lookup indices and check if they are all within allowed bounds

!IBM* ASSERT(NODEPS)
    DO nl = 1,kidx
      jl = list(nl)
      ztt = ANINT(rfdeltat*temp(jl))
      idx(nl) = INT(ztt)
      zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
      zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)
    END DO
    ! if one index was out of bounds -> print error and exit
    IF (zinbounds == 0.0_wp) CALL lookuperror(name)
    CALL fetch_ua(kidx, idx, tlucua, ua)
    IF (PRESENT(dua)) CALL fetch_ua(kidx, idx, tlucuad, dua)

  END SUBROUTINE lookup_ua_list
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_uaw_list_spline(name, size, list, temp, uaw, duaw)

  ! VM: this SR is never called!

    CHARACTER(len=*),   INTENT(in)  :: name
    INTEGER,            INTENT(in)  :: size
    INTEGER,            INTENT(in)  :: list(size)
    REAL(wp),           INTENT(in)  :: temp(size)
    REAL(wp), OPTIONAL, INTENT(out) :: uaw(size), duaw(size)

    INTEGER :: idx(size)
    REAL(wp) :: zalpha(size)

    REAL(wp) :: ztt, ztshft, zinbounds, ztmax, ztmin
    INTEGER :: nl, jl

    zinbounds = 1.0_wp
    ztmin = flucupmin
    ztmax = flucupmax

    ! first compute all lookup indices and check if they are all within allowed bounds

!IBM* ASSERT(NODEPS)
    DO nl = 1, size
      jl = list(nl)
      ztshft = FSEL(tmelt-temp(jl),1.0_wp,0.0_wp)
      ztt = rsdeltat*temp(jl)
      zalpha(nl) = ztt - AINT(ztt)
      idx(nl) = INT(ztt-ztshft)
      zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
      zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)
    END DO
    ! if one index was out of bounds -> print error and exit
    IF (zinbounds == 0.0_wp) CALL lookuperror(name, 'lookup_uaw_list_spline')
    CALL fetch_ua_spline(1, size, idx, zalpha, tlucuw, uaw, duaw)

  END SUBROUTINE lookup_uaw_list_spline
  !----------------------------------------------------------------------------
  SUBROUTINE lookup_uaw_list(name, size, list, temp, uaw, duaw)
    CHARACTER(len=*),   INTENT(in)  :: name
    INTEGER,            INTENT(in)  :: size
    INTEGER,            INTENT(in)  :: list(size)
    REAL(wp),           INTENT(in)  :: temp(size)
    REAL(wp), OPTIONAL, INTENT(out) :: uaw(size), duaw(size)

    INTEGER :: idx(size)

    REAL(wp) :: ztt, zinbounds, ztmax, ztmin
    INTEGER :: nl, jl

    zinbounds = 1.0_wp
    ztmin = fjptlucu1
    ztmax = fjptlucu2

    ! first compute all lookup indices and check if they are all within allowed bounds

!IBM* ASSERT(NODEPS)
    DO nl = 1, size
      jl = list(nl)
      ztt = ANINT(rfdeltat*temp(jl))
      idx(nl) = INT(ztt)
      zinbounds = FSEL(ztmin-ztt,0.0_wp,zinbounds)
      zinbounds = FSEL(ztt-ztmax,0.0_wp,zinbounds)
    END DO
    ! if one index was out of bounds -> print error and exit
    IF (zinbounds == 0.0_wp) CALL lookuperror(name, 'lookup_uaw_list')
    CALL fetch_ua(size, idx, tlucuaw, uaw)
    IF (PRESENT(duaw)) CALL fetch_ua(size, idx, tlucuaw, duaw)

  END SUBROUTINE lookup_uaw_list
  !----------------------------------------------------------------------------
  SUBROUTINE lookuperror(name, name2)
    CHARACTER(len=*), INTENT(in) :: name
    CHARACTER(len=*), OPTIONAL, INTENT(in) :: name2

    IF (PRESENT(name2)) THEN
       WRITE(message_text, '(3a)') name, ': ', name2
    ELSE
       WRITE(message_text, '(a)') name
    END IF

    CALL finish (message_text, ' lookup table overflow (temp out of bounds)')
  END SUBROUTINE lookuperror
  !----------------------------------------------------------------------------
END MODULE mo_echam_convect_tables

