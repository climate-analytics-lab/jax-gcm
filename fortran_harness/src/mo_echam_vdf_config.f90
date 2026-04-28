! Stub of mo_echam_vdf_config matching the ICON-port atm_phy_echam vdiff
! usage. Field defaults pulled from ECHAM6.3 ``mo_echam_vdiff_params.f90``
! and the ICON port's vdf-config setup. ``c_e``, ``c_f``, ``c_n``,
! ``f_tau0``, ``f_theta0`` are the TTE-TKE closure constants.
MODULE mo_echam_vdf_config
  USE mo_kind, ONLY: wp
  IMPLICIT NONE
  PUBLIC

  TYPE :: t_echam_vdf_config
    REAL(wp) :: c_e        ! TTE-TKE dissipation coefficient
    REAL(wp) :: c_f        ! TTE-TKE flux coefficient
    REAL(wp) :: c_n        ! TTE-TKE neutral coefficient
    REAL(wp) :: fbl        ! Boundary-layer factor for mixing length
    REAL(wp) :: fsl        ! Surface-layer factor for mixing length
    REAL(wp) :: f_tau0     ! Reference stress factor
    REAL(wp) :: f_theta0   ! Reference flux factor
    REAL(wp) :: pr0        ! Neutral Prandtl number
    REAL(wp) :: wmc        ! Wave-mixing coefficient
    LOGICAL  :: lsfc_heat_flux  ! whether to apply surface heat flux BC
    LOGICAL  :: lsfc_mom_flux   ! whether to apply surface momentum flux BC
  END TYPE t_echam_vdf_config

  TYPE(t_echam_vdf_config), TARGET :: echam_vdf_config(1)

CONTAINS
  SUBROUTINE init_vdf_config_defaults()
    echam_vdf_config(1)%c_e        = 0.845_wp     ! TTE-TKE: dissipation
    echam_vdf_config(1)%c_f        = 1.65_wp      ! TTE-TKE: flux ratio
    echam_vdf_config(1)%c_n        = 0.5_wp
    echam_vdf_config(1)%fbl        = 3.0_wp
    echam_vdf_config(1)%fsl        = 0.4_wp
    echam_vdf_config(1)%f_tau0     = 0.17_wp
    echam_vdf_config(1)%f_theta0   = 1.0_wp
    echam_vdf_config(1)%pr0        = 0.7_wp
    echam_vdf_config(1)%wmc        = 0.5_wp
    echam_vdf_config(1)%lsfc_heat_flux = .TRUE.
    echam_vdf_config(1)%lsfc_mom_flux  = .TRUE.
  END SUBROUTINE init_vdf_config_defaults
END MODULE mo_echam_vdf_config
