! Stub of ``mo_echam_cld_config`` matching the ICON-port atm_phy_echam
! cumastr/cuascent/cudescent/cuinitialize and mo_cloud usage. Default
! values pulled from ECHAM6.3 ``mo_echam_cloud_params.f90``.
MODULE mo_echam_cld_config
  USE mo_kind, ONLY: wp
  IMPLICIT NONE
  PUBLIC

  TYPE :: t_echam_cld_config
    ! Used by convect_tables (cumastr path)
    REAL(wp) :: csecfrl
    REAL(wp) :: cthomi
    ! Used by mo_cloud
    REAL(wp) :: cqtmin
    REAL(wp) :: cvtfall
    REAL(wp) :: crhosno
    REAL(wp) :: cn0s
    REAL(wp) :: cauloc
    REAL(wp) :: clmax
    REAL(wp) :: clmin
    REAL(wp) :: ccraut
    REAL(wp) :: ceffmin
    REAL(wp) :: ceffmax
    REAL(wp) :: crhoi
    REAL(wp) :: ccsaut
    REAL(wp) :: ccsacl
    REAL(wp) :: ccracl
    REAL(wp) :: ccwmin
    REAL(wp) :: clwprat
    INTEGER  :: jks
  END TYPE t_echam_cld_config

  TYPE(t_echam_cld_config), TARGET :: echam_cld_config(1)

CONTAINS
  SUBROUTINE init_cld_config_defaults()
    REAL(wp), PARAMETER :: tmelt = 273.15_wp
    echam_cld_config(1)%csecfrl = 5.0E-6_wp
    echam_cld_config(1)%cthomi  = tmelt - 35.0_wp
    echam_cld_config(1)%cqtmin  = 1.0E-12_wp
    echam_cld_config(1)%cvtfall = 3.29_wp     ! ECHAM default for T63
    echam_cld_config(1)%crhosno = 100.0_wp
    echam_cld_config(1)%cn0s    = 3.0E6_wp
    echam_cld_config(1)%cauloc  = 0.0_wp
    echam_cld_config(1)%clmax   = 0.5_wp
    echam_cld_config(1)%clmin   = 0.0_wp
    echam_cld_config(1)%ccraut  = 15.0_wp
    echam_cld_config(1)%ceffmin = 10.0_wp
    echam_cld_config(1)%ceffmax = 150.0_wp
    echam_cld_config(1)%crhoi   = 500.0_wp
    echam_cld_config(1)%ccsaut  = 95.0_wp
    echam_cld_config(1)%ccsacl  = 0.10_wp
    echam_cld_config(1)%ccracl  = 6.0_wp
    echam_cld_config(1)%ccwmin  = 1.0E-7_wp
    echam_cld_config(1)%clwprat = 4.0_wp      ! ECHAM default for T63
    echam_cld_config(1)%jks     = 1
  END SUBROUTINE init_cld_config_defaults
END MODULE mo_echam_cld_config
