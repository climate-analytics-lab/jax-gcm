! Standalone driver for the ICON-port `cumastr` Tiedtke-Nordeng convection
! scheme. Reads a single column (or a small batch of columns) of input
! state from a Fortran-unformatted file, runs cumastr, and writes the
! output tendencies + diagnostics to a second file.
!
! File format (sequential unformatted, one record per write):
!   Record 1 — INTEGER :: kproma, klev
!   Record 2 — REAL(wp):: dtime
!   Record 3 — REAL(wp):: eta_full(klev)            ! for cevapcu init
!   Record 4 — REAL(wp):: pten(kproma,klev)         ! T (K)
!   Record 5 — REAL(wp):: pqen(kproma,klev)         ! qv (kg/kg)
!   Record 6 — REAL(wp):: pxen(kproma,klev)         ! qc+qi (kg/kg)
!   Record 7 — REAL(wp):: puen(kproma,klev)         ! u (m/s)
!   Record 8 — REAL(wp):: pven(kproma,klev)         ! v (m/s)
!   Record 9 — REAL(wp):: pverv(kproma,klev)        ! omega (Pa/s)
!   Record 10 — REAL(wp):: papp1(kproma,klev)       ! p_full (Pa)
!   Record 11 — REAL(wp):: paphp1(kproma,klev+1)    ! p_half (Pa)
!   Record 12 — REAL(wp):: pgeo(kproma,klev)        ! geopotential (m^2/s^2)
!   Record 13 — REAL(wp):: pgeoh(kproma,klev+1)     ! geopotential half
!   Record 14 — REAL(wp):: pzf(kproma,klev)         ! z full (m)
!   Record 15 — REAL(wp):: pzh(kproma,klev+1)       ! z half (m)
!   Record 16 — REAL(wp):: pmref(kproma,klev)       ! ref mass per layer (kg/m^2)
!   Record 17 — REAL(wp):: pqte(kproma,klev)        ! prior dq/dt (kg/kg/s)
!   Record 18 — REAL(wp):: pqhfla(kproma)           ! sfc latent flux (kg/m^2/s)
!   Record 19 — REAL(wp):: pthvsig(kproma)          ! std dev virt pot T
!   Record 20 — LOGICAL :: ldland(kproma)           ! land mask
!
! Output (sequential unformatted):
!   Record 1 — INTEGER :: ktype(kproma)
!   Record 2 — INTEGER :: kctop(kproma)
!   Record 3 — REAL(wp):: pq_cnv(kproma,klev)        ! T tendency (J/kg/s = K/s * cp)
!   Record 4 — REAL(wp):: pqte_cnv(kproma,klev)      ! q tendency (kg/kg/s)
!   Record 5 — REAL(wp):: pvom_cnv(kproma,klev)      ! u tendency (m/s/s)
!   Record 6 — REAL(wp):: pvol_cnv(kproma,klev)      ! v tendency (m/s/s)
!   Record 7 — REAL(wp):: pxtecl(kproma,klev)        ! cloud-liq detrainment
!   Record 8 — REAL(wp):: pxteci(kproma,klev)        ! cloud-ice detrainment
!   Record 9 — REAL(wp):: prsfc(kproma)              ! rainfall rate (kg/m^2/s)
!   Record 10 — REAL(wp):: pssfc(kproma)             ! snowfall rate
!   Record 11 — REAL(wp):: ptop(kproma)              ! pressure of cloud top (Pa)
!   Record 12 — REAL(wp):: pcon_dtrl(kproma)
!   Record 13 — REAL(wp):: pcon_dtri(kproma)
!   Record 14 — REAL(wp):: pcon_iqte(kproma)
PROGRAM cumastr_driver
  USE mo_kind,                ONLY: wp
  USE mo_cumastr,             ONLY: cumastr
  USE mo_echam_convect_tables, ONLY: init_convect_tables
  USE mo_echam_cnv_config,    ONLY: init_cnv_config_defaults
  USE mo_echam_cld_config,    ONLY: init_cld_config_defaults

  IMPLICIT NONE

  CHARACTER(len=512) :: in_path, out_path
  INTEGER :: kproma, klev, klevp1, klevm1, kbdim
  INTEGER, PARAMETER :: jg = 1, jb = 1, jcs = 1
  INTEGER, PARAMETER :: ktrac = 0
  REAL(wp) :: dtime

  REAL(wp), ALLOCATABLE :: eta_full(:)
  REAL(wp), ALLOCATABLE :: pten(:,:), pqen(:,:), pxen(:,:)
  REAL(wp), ALLOCATABLE :: puen(:,:), pven(:,:), pverv(:,:)
  REAL(wp), ALLOCATABLE :: papp1(:,:), paphp1(:,:)
  REAL(wp), ALLOCATABLE :: pgeo(:,:),  pgeoh(:,:)
  REAL(wp), ALLOCATABLE :: pzf(:,:),   pzh(:,:)
  REAL(wp), ALLOCATABLE :: pmref(:,:), pqte(:,:)
  REAL(wp), ALLOCATABLE :: pqhfla(:),  pthvsig(:)
  LOGICAL,  ALLOCATABLE :: ldland(:)
  REAL(wp), ALLOCATABLE :: pxten(:,:,:)

  ! Outputs
  INTEGER,  ALLOCATABLE :: ktype(:), kctop(:)
  REAL(wp), ALLOCATABLE :: prsfc(:), pssfc(:), ptop(:)
  REAL(wp), ALLOCATABLE :: pcon_dtrl(:), pcon_dtri(:), pcon_iqte(:)
  REAL(wp), ALLOCATABLE :: pq_cnv(:,:), pvom_cnv(:,:), pvol_cnv(:,:)
  REAL(wp), ALLOCATABLE :: pqte_cnv(:,:)
  REAL(wp), ALLOCATABLE :: pxtte_cnv(:,:,:)
  REAL(wp), ALLOCATABLE :: pxtecl(:,:), pxteci(:,:)

  INTEGER :: u_in, u_out

  IF (COMMAND_ARGUMENT_COUNT() < 2) THEN
    WRITE(0,*) 'usage: cumastr_driver <input.bin> <output.bin>'
    ERROR STOP 2
  END IF
  CALL GET_COMMAND_ARGUMENT(1, in_path)
  CALL GET_COMMAND_ARGUMENT(2, out_path)

  OPEN(newunit=u_in, file=TRIM(in_path), form='unformatted', status='old', &
       access='sequential', action='read')

  READ(u_in) kproma, klev
  klevp1 = klev + 1
  klevm1 = klev - 1
  kbdim  = kproma

  ALLOCATE(eta_full(klev))
  ALLOCATE(pten(kbdim,klev), pqen(kbdim,klev), pxen(kbdim,klev))
  ALLOCATE(puen(kbdim,klev), pven(kbdim,klev), pverv(kbdim,klev))
  ALLOCATE(papp1(kbdim,klev), paphp1(kbdim,klevp1))
  ALLOCATE(pgeo(kbdim,klev),  pgeoh(kbdim,klevp1))
  ALLOCATE(pzf(kbdim,klev),   pzh(kbdim,klevp1))
  ALLOCATE(pmref(kbdim,klev), pqte(kbdim,klev))
  ALLOCATE(pqhfla(kbdim),     pthvsig(kbdim), ldland(kbdim))
  ALLOCATE(pxten(kbdim,klev,MAX(ktrac,1)))   ! at least size 1 for safe shape
  pxten = 0.0_wp

  READ(u_in) dtime
  READ(u_in) eta_full
  READ(u_in) pten
  READ(u_in) pqen
  READ(u_in) pxen
  READ(u_in) puen
  READ(u_in) pven
  READ(u_in) pverv
  READ(u_in) papp1
  READ(u_in) paphp1
  READ(u_in) pgeo
  READ(u_in) pgeoh
  READ(u_in) pzf
  READ(u_in) pzh
  READ(u_in) pmref
  READ(u_in) pqte
  READ(u_in) pqhfla
  READ(u_in) pthvsig
  READ(u_in) ldland
  CLOSE(u_in)

  ! Initialise lookup tables and configs
  CALL init_cld_config_defaults()
  CALL init_cnv_config_defaults(klev, eta_full)
  CALL init_convect_tables()

  ! Allocate outputs
  ALLOCATE(ktype(kbdim), kctop(kbdim))
  ALLOCATE(prsfc(kbdim), pssfc(kbdim), ptop(kbdim))
  ALLOCATE(pcon_dtrl(kbdim), pcon_dtri(kbdim), pcon_iqte(kbdim))
  ALLOCATE(pq_cnv(kbdim,klev), pvom_cnv(kbdim,klev), pvol_cnv(kbdim,klev))
  ALLOCATE(pqte_cnv(kbdim,klev))
  ALLOCATE(pxtte_cnv(kbdim,klev,MAX(ktrac,1)))
  ALLOCATE(pxtecl(kbdim,klev), pxteci(kbdim,klev))

  CALL cumastr(jg, jb,                                                    &
    &          jcs, kproma, kbdim,                                        &
    &          klev, klevp1, klevm1,                                      &
    &          dtime,                                                     &
    &          pzf, pzh,                                                  &
    &          pmref,                                                     &
    &          pten, pqen, pxen, puen, pven,                              &
    &          ktrac, ldland,                                             &
    &          pxten,                                                     &
    &          pverv, pqhfla,                                             &
    &          papp1, paphp1,                                             &
    &          pgeo,  pgeoh,                                              &
    &          pqte,                                                      &
    &          pthvsig,                                                   &
    &          ktype, kctop,                                              &
    &          prsfc, pssfc,                                              &
    &          pcon_dtrl, pcon_dtri, pcon_iqte,                           &
    &          pq_cnv, pvom_cnv, pvol_cnv, pqte_cnv, pxtte_cnv,           &
    &          pxtecl, pxteci,                                            &
    &          ptop)

  ! Write outputs
  OPEN(newunit=u_out, file=TRIM(out_path), form='unformatted', status='replace', &
       access='sequential', action='write')
  WRITE(u_out) ktype
  WRITE(u_out) kctop
  WRITE(u_out) pq_cnv
  WRITE(u_out) pqte_cnv
  WRITE(u_out) pvom_cnv
  WRITE(u_out) pvol_cnv
  WRITE(u_out) pxtecl
  WRITE(u_out) pxteci
  WRITE(u_out) prsfc
  WRITE(u_out) pssfc
  WRITE(u_out) ptop
  WRITE(u_out) pcon_dtrl
  WRITE(u_out) pcon_dtri
  WRITE(u_out) pcon_iqte
  CLOSE(u_out)

  WRITE(*,'(a)') 'cumastr_driver: ok'
END PROGRAM cumastr_driver
