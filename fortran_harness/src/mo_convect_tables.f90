! Shim that re-exports the lookup spline functions from
! ``mo_echam_convect_tables`` (which the harness already has) and adds
! a thin ``compute_qsat`` helper. Lets ``mo_turbulence_diag`` USE
! ``mo_convect_tables`` unmodified (the real ICON port; the older
! ECHAM port called ``mo_echam_convect_tables`` directly).
MODULE mo_convect_tables
  USE mo_kind, ONLY: wp
  USE mo_echam_convect_tables, ONLY: &
    prepare_ua_index_spline, lookup_ua_spline
  USE mo_physical_constants, ONLY: rd, rv

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: prepare_ua_index_spline, lookup_ua_spline, compute_qsat

CONTAINS

  ! Saturation specific humidity at given (T, p), for a list of
  ! columns identified by ``loidx``. Mirrors ECHAM's ``compute_qsat``
  ! signature: input pressure ``ppsfc`` and temperature ``ptsfc`` per
  ! column; output ``pqsat`` per column.
  SUBROUTINE compute_qsat(kproma, kcount, loidx, ppsfc, ptsfc, pqsat)
    INTEGER,  INTENT(IN)  :: kproma, kcount
    INTEGER,  INTENT(IN)  :: loidx(kproma)
    REAL(wp), INTENT(IN)  :: ppsfc(kproma), ptsfc(kproma)
    REAL(wp), INTENT(OUT) :: pqsat(kproma)

    REAL(wp) :: za(kproma), ua(kproma)
    INTEGER  :: idx(kproma)
    INTEGER  :: jl, n
    REAL(wp), PARAMETER :: vtmpc1 = rv/rd - 1.0_wp

    ! Use the lookup spline to get saturation vapor pressure / Rd*Rv
    ! (the convention ECHAM ``tlucua`` returns).
    CALL prepare_ua_index_spline(1, 'compute_qsat', 1, kproma, ptsfc, idx, za, &
         klev=1)
    CALL lookup_ua_spline(1, kproma, idx, za, ua)

    ! Convert table value to specific humidity:
    !   q_sat = (Rd/Rv * e_sat) / (p - (1 - Rd/Rv) * e_sat)
    ! ECHAM stores ``ua = e_sat * Rd/Rv`` directly in ``tlucua``.
    DO n = 1, kcount
      jl = loidx(n)
      pqsat(jl) = ua(jl) / (ppsfc(jl) - vtmpc1 * ua(jl))
    END DO
  END SUBROUTINE compute_qsat

END MODULE mo_convect_tables
