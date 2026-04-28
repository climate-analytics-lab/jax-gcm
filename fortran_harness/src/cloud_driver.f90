! Standalone driver for the ICON-port `mo_cloud` module (Sundqvist
! cloud cover + condensation/evaporation + ECHAM-1m microphysics in
! one routine). Reads a single column of input state from a Fortran-
! unformatted binary file, runs `cloud()`, and writes the output
! tendencies + diagnostics to a second file.
!
! File format (sequential unformatted, one record per write):
!   Record 1  — INTEGER :: kproma, klev
!   Record 2  — REAL(wp):: dtime
!   Record 3  — INTEGER :: kctop(kproma)         ! convective cloud top idx
!   Record 4  — INTEGER :: ktype(kproma)         ! convection type (in/out)
!   Record 5  — REAL(wp):: papm1(kproma,klev)    ! pressure at full levels (Pa)
!   Record 6  — REAL(wp):: pdz(kproma,klev)      ! layer thickness (m)
!   Record 7  — REAL(wp):: pmref(kproma,klev)    ! mass per area (kg/m^2)
!   Record 8  — REAL(wp):: prho(kproma,klev)     ! density (kg/m^3)
!   Record 9  — REAL(wp):: pcpair(kproma,klev)   ! cp of moist air
!   Record 10 — REAL(wp):: pacdnc(kproma,klev)   ! cloud droplet number conc
!   Record 11 — REAL(wp):: ptm1(kproma,klev)     ! T at n-1 (K)
!   Record 12 — REAL(wp):: pqm1(kproma,klev)     ! q at n-1 (kg/kg)
!   Record 13 — REAL(wp):: pxlm1(kproma,klev)    ! cloud liquid water (kg/kg)
!   Record 14 — REAL(wp):: pxim1(kproma,klev)    ! cloud ice (kg/kg)
!   Record 15 — REAL(wp):: paclc(kproma,klev)    ! cloud cover (in/out)
!
! Output (sequential unformatted):
!   Record 1  — INTEGER :: ktype(kproma)
!   Record 2  — REAL(wp):: paclc(kproma,klev)
!   Record 3  — REAL(wp):: paclcov(kproma)
!   Record 4  — REAL(wp):: prsfl(kproma)
!   Record 5  — REAL(wp):: pssfl(kproma)
!   Record 6  — REAL(wp):: prelhum(kproma,klev)
!   Record 7  — REAL(wp):: pq_cld(kproma,klev)
!   Record 8  — REAL(wp):: pqte_cld(kproma,klev)
!   Record 9  — REAL(wp):: pxlte_cld(kproma,klev)
!   Record 10 — REAL(wp):: pxite_cld(kproma,klev)
PROGRAM cloud_driver
  USE mo_kind,                ONLY: wp
  USE mo_cloud,               ONLY: cloud
  USE mo_echam_convect_tables, ONLY: init_convect_tables
  USE mo_echam_cld_config,    ONLY: init_cld_config_defaults

  IMPLICIT NONE

  CHARACTER(len=512) :: in_path, out_path
  INTEGER :: kproma, klev, kbdim
  INTEGER, PARAMETER :: jg = 1, jb = 1, jcs = 1
  REAL(wp) :: dtime

  INTEGER,  ALLOCATABLE :: kctop(:), ktype(:)
  REAL(wp), ALLOCATABLE :: papm1(:,:), pdz(:,:), pmref(:,:), prho(:,:)
  REAL(wp), ALLOCATABLE :: pcpair(:,:), pacdnc(:,:)
  REAL(wp), ALLOCATABLE :: ptm1(:,:), pqm1(:,:), pxlm1(:,:), pxim1(:,:)
  REAL(wp), ALLOCATABLE :: paclc(:,:)
  REAL(wp), ALLOCATABLE :: paclcov(:), prsfl(:), pssfl(:)
  REAL(wp), ALLOCATABLE :: prelhum(:,:), pq_cld(:,:), pqte_cld(:,:)
  REAL(wp), ALLOCATABLE :: pxlte_cld(:,:), pxite_cld(:,:)

  INTEGER :: u_in, u_out

  IF (COMMAND_ARGUMENT_COUNT() < 2) THEN
    WRITE(0,*) 'usage: cloud_driver <input.bin> <output.bin>'
    ERROR STOP 2
  END IF
  CALL GET_COMMAND_ARGUMENT(1, in_path)
  CALL GET_COMMAND_ARGUMENT(2, out_path)

  OPEN(newunit=u_in, file=TRIM(in_path), form='unformatted', status='old', &
       access='sequential', action='read')

  READ(u_in) kproma, klev
  kbdim = kproma

  ALLOCATE(kctop(kbdim), ktype(kbdim))
  ALLOCATE(papm1(kbdim,klev), pdz(kbdim,klev), pmref(kbdim,klev))
  ALLOCATE(prho(kbdim,klev), pcpair(kbdim,klev), pacdnc(kbdim,klev))
  ALLOCATE(ptm1(kbdim,klev), pqm1(kbdim,klev))
  ALLOCATE(pxlm1(kbdim,klev), pxim1(kbdim,klev))
  ALLOCATE(paclc(kbdim,klev))
  ALLOCATE(paclcov(kbdim), prsfl(kbdim), pssfl(kbdim))
  ALLOCATE(prelhum(kbdim,klev), pq_cld(kbdim,klev), pqte_cld(kbdim,klev))
  ALLOCATE(pxlte_cld(kbdim,klev), pxite_cld(kbdim,klev))

  READ(u_in) dtime
  READ(u_in) kctop
  READ(u_in) ktype
  READ(u_in) papm1
  READ(u_in) pdz
  READ(u_in) pmref
  READ(u_in) prho
  READ(u_in) pcpair
  READ(u_in) pacdnc
  READ(u_in) ptm1
  READ(u_in) pqm1
  READ(u_in) pxlm1
  READ(u_in) pxim1
  READ(u_in) paclc
  CLOSE(u_in)

  ! Initialise lookup tables and configs
  CALL init_cld_config_defaults()
  CALL init_convect_tables()

  CALL cloud(jg, jb,                                                      &
    &        jcs, kproma, kbdim, klev,                                    &
    &        dtime,                                                       &
    &        kctop,                                                       &
    &        papm1, pdz, pmref, prho, pcpair, pacdnc,                     &
    &        ptm1, pqm1, pxlm1, pxim1,                                    &
    &        ktype,                                                       &
    &        paclc,                                                       &
    &        paclcov, prsfl, pssfl,                                       &
    &        prelhum, pq_cld, pqte_cld, pxlte_cld, pxite_cld)

  ! Write outputs
  OPEN(newunit=u_out, file=TRIM(out_path), form='unformatted', status='replace', &
       access='sequential', action='write')
  WRITE(u_out) ktype
  WRITE(u_out) paclc
  WRITE(u_out) paclcov
  WRITE(u_out) prsfl
  WRITE(u_out) pssfl
  WRITE(u_out) prelhum
  WRITE(u_out) pq_cld
  WRITE(u_out) pqte_cld
  WRITE(u_out) pxlte_cld
  WRITE(u_out) pxite_cld
  CLOSE(u_out)

  WRITE(*,'(a)') 'cloud_driver: ok'
END PROGRAM cloud_driver
