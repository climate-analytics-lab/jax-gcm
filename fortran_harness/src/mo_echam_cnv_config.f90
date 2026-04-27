! Stub of `mo_echam_cnv_config` matching the ICON-port `atm_phy_echam`
! cumastr / cuascent / cudescent / cuinitialize usage. All scalar fields
! are populated with the ECHAM6.3 ``__ICON__`` defaults from
! ``mo_echam_conv_constants::cuparam``. ``cevapcu`` is computed at
! ``init`` time from the full-level eta values supplied by the caller.
MODULE mo_echam_cnv_config
  USE mo_kind, ONLY: wp
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: t_echam_cnv_config, echam_cnv_config, cevapcu
  PUBLIC :: init_cnv_config_defaults

  TYPE :: t_echam_cnv_config
    LOGICAL  :: lmfdd
    LOGICAL  :: lmfdudv
    LOGICAL  :: lmfmid
    REAL(wp) :: entrpen
    REAL(wp) :: entrscv
    REAL(wp) :: entrmid
    REAL(wp) :: entrdd
    REAL(wp) :: cmfdeps
    REAL(wp) :: cmftau
    REAL(wp) :: cmfctop
    REAL(wp) :: cmfcmin
    REAL(wp) :: cmfcmax
    REAL(wp) :: cprcon
    REAL(wp) :: cminbuoy
    REAL(wp) :: cmaxbuoy
    REAL(wp) :: cbfac
    REAL(wp) :: centrmax
    REAL(wp) :: dlev_land
    REAL(wp) :: dlev_ocean
    INTEGER  :: nmctop
  END TYPE t_echam_cnv_config

  TYPE(t_echam_cnv_config), TARGET :: echam_cnv_config(1)
  REAL(wp), ALLOCATABLE, TARGET   :: cevapcu(:,:)

CONTAINS
  ! Populate config + cevapcu from level structure. ``eta_full(klev)`` is
  ! the full-level eta = (a_full + b_full*p0)/p0 at standard p0=101325 Pa,
  ! consistent with ECHAM ``iniphy.f90::cevapcu(jk)`` initialisation.
  SUBROUTINE init_cnv_config_defaults(klev, eta_full)
    INTEGER,  INTENT(IN) :: klev
    REAL(wp), INTENT(IN) :: eta_full(klev)
    INTEGER :: jk
    REAL(wp), PARAMETER :: grav = 9.80665_wp

    echam_cnv_config(1)%lmfdd      = .TRUE.
    echam_cnv_config(1)%lmfdudv    = .TRUE.
    echam_cnv_config(1)%lmfmid     = .TRUE.
    echam_cnv_config(1)%entrpen    = 1.0E-4_wp
    echam_cnv_config(1)%entrscv    = 3.0E-3_wp
    echam_cnv_config(1)%entrmid    = 1.0E-4_wp
    echam_cnv_config(1)%entrdd     = 2.0E-4_wp
    echam_cnv_config(1)%cmfdeps    = 0.3_wp
    echam_cnv_config(1)%cmftau     = 7200.0_wp
    echam_cnv_config(1)%cmfctop    = 0.2_wp
    echam_cnv_config(1)%cmfcmin    = 1.0E-10_wp
    echam_cnv_config(1)%cmfcmax    = 1.0_wp
    echam_cnv_config(1)%cprcon     = 2.5E-4_wp
    echam_cnv_config(1)%cminbuoy   = 0.2_wp
    echam_cnv_config(1)%cmaxbuoy   = 1.0_wp
    echam_cnv_config(1)%cbfac      = 1.0_wp
    echam_cnv_config(1)%centrmax   = 3.0E-4_wp
    ! `dlev_land` / `dlev_ocean` are the boundary-layer depth thresholds
    ! used to decide where shallow vs deep convection is permitted in
    ! the ICON-port `cuascent`. ECHAM-default values from atm_phy_echam.
    echam_cnv_config(1)%dlev_land  = 3.0E4_wp
    echam_cnv_config(1)%dlev_ocean = 1.5E4_wp

    ! `nmctop` — highest model level where mid-level convection's cloud
    ! base is permitted. ECHAM `iniphy.f90` searches for the level whose
    ! pressure first exceeds 300 hPa.
    echam_cnv_config(1)%nmctop = 1
    DO jk = 1, klev
      echam_cnv_config(1)%nmctop = jk
      IF (eta_full(jk) * 101325.0_wp >= 30000.0_wp) EXIT
    END DO

    IF (ALLOCATED(cevapcu)) DEALLOCATE(cevapcu)
    ALLOCATE(cevapcu(klev, 1))
    DO jk = 1, klev
      cevapcu(jk, 1) = 1.93E-6_wp * 261.0_wp                          &
        & * SQRT(1.0E3_wp / (38.3_wp * 0.293_wp) * SQRT(eta_full(jk))) &
        & * 0.5_wp / grav
    END DO
  END SUBROUTINE init_cnv_config_defaults

END MODULE mo_echam_cnv_config
