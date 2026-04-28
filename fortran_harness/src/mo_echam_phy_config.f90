! Stub of mo_echam_phy_config for the harness. The vdiff solver
! references only ``ljsb`` (whether to call JSBACH land surface).
! For the harness we keep this FALSE — JSBACH is a separate subsystem
! we don't need for column-mode vdiff comparisons.
MODULE mo_echam_phy_config
  IMPLICIT NONE
  PUBLIC

  TYPE :: t_echam_phy_config
    LOGICAL :: ljsb = .FALSE.
  END TYPE t_echam_phy_config

  TYPE(t_echam_phy_config), TARGET :: echam_phy_config(1)

CONTAINS
  SUBROUTINE init_phy_config_defaults()
    echam_phy_config(1)%ljsb = .FALSE.
  END SUBROUTINE init_phy_config_defaults
END MODULE mo_echam_phy_config
