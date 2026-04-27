! Stub of `mo_echam_cld_config` for the harness. The convect_tables module
! references only ``csecfrl`` (security-cloud-water-floor for liquid) and
! ``cthomi`` (homogeneous-freezing temperature, K) per domain. Default
! values pulled from ECHAM6.3 cloud-physics configuration.
MODULE mo_echam_cld_config
  USE mo_kind, ONLY: wp
  IMPLICIT NONE
  PUBLIC

  TYPE :: t_echam_cld_config
    REAL(wp) :: csecfrl
    REAL(wp) :: cthomi
  END TYPE t_echam_cld_config

  TYPE(t_echam_cld_config), TARGET :: echam_cld_config(1)

CONTAINS
  SUBROUTINE init_cld_config_defaults()
    echam_cld_config(1)%csecfrl = 5.0E-6_wp
    echam_cld_config(1)%cthomi  = 236.15_wp
  END SUBROUTINE init_cld_config_defaults
END MODULE mo_echam_cld_config
