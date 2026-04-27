! Stub of mo_math_constants for the harness. Provides the minimal
! constants used by mo_cloud.f90 (only ``pi`` from this module).
MODULE mo_math_constants
  USE mo_kind, ONLY: wp
  IMPLICIT NONE
  PUBLIC

  REAL(wp), PARAMETER :: pi = 3.14159265358979323846_wp

END MODULE mo_math_constants
