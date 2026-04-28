!>
!! @brief Second half of the driver routine for turbulent mixing.
!!
!! @author Hui Wan, MPI-M
!!
!! @par Revision History
!! First version by Hui Wan, MPI-M (2011-04)
!!
!! @par Copyright and License
!!
!! This code is subject to the DWD and MPI-M-Software-License-Agreement in
!! its most recent form.
!! Please see the file LICENSE in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the
!! headers of the routines.
!!
#if defined __xlC__ && !defined NOXLFPROCESS
@PROCESS HOT
#endif

MODULE mo_vdiff_upward_sweep

  USE mo_kind,               ONLY: wp
  USE mo_vdiff_solver,       ONLY: nvar_vdiff, nmatrix, rhs_bksub, vdiff_tendencies
  USE mo_echam_vdiff_params, ONLY: itop

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: vdiff_up

CONTAINS
  !>
  !!
  !!
  SUBROUTINE vdiff_up( jcs, kproma, kbdim, klev, klevm1,                 &! in
                       ktrac,      ksfc_type,   idx_wtr,                 &! in
                       pdtime, pfrc,                                     &! in
                       pcfm_tile,                                        &! in 
                       aa,         pcptgz,                               &! in
                       pum1,       pvm1,        ptm1,                    &! in
                       pmair,      pmref,                                &! in
                       pqm1,       pxlm1,       pxim1,       pxtm1,      &! in
                       pgeom1,      pztottevn,                           &! in
                       bb,                                               &! inout
                       pzthvvar,   pxvar,       pz0m_tile,               &! inout
                       pkedisp,                                          &! out
                       pute_vdf,   pvte_vdf,    pq_vdf,                  &! out
                       pqte_vdf,   pxlte_vdf,   pxite_vdf,   pxtte_vdf,  &! out
                       pz0m,                                             &! out
                       pthvvar,                                          &! out
                       ptotte,                                           &! out
                       psh_vdiff,  pqv_vdiff                             )! out

    INTEGER, INTENT(IN) :: jcs, kproma, kbdim, klev, klevm1, ktrac
    INTEGER, INTENT(IN) :: ksfc_type, idx_wtr
    REAL(wp),INTENT(IN) :: pdtime

    REAL(wp),INTENT(IN) ::           &
      & pfrc      (kbdim,ksfc_type), &!< area fraction of each surface type
      & pcfm_tile (kbdim,ksfc_type)   !< exchange coeff

    REAL(wp),INTENT(IN) :: aa    (kbdim,klev,3,nmatrix) !< for all variables


    ! The input variables below are needed only by "vdiff_tendencies"

    REAL(wp),INTENT(IN) :: pcptgz  (kbdim,klev)  !< dry static energy

    REAL(wp),INTENT(IN) :: pum1    (kbdim,klev)  !< u-wind at step t-dt
    REAL(wp),INTENT(IN) :: pvm1    (kbdim,klev)  !< q-wind at step t-dt
    REAL(wp),INTENT(IN) :: ptm1    (kbdim,klev)  !< temperature at step t-dt
    REAL(wp),INTENT(IN) :: pmair   (kbdim,klev)  !< moist air mass [kg/m2]
    REAL(wp),INTENT(IN) :: pmref   (kbdim,klev)  !< dry   air mass [kg/m2]
    REAL(wp),INTENT(IN) :: pqm1    (kbdim,klev)  !< specific humidity at step t-dt
    REAL(wp),INTENT(IN) :: pxlm1   (kbdim,klev)  !< cloud water concentration at step t-dt
    REAL(wp),INTENT(IN) :: pxim1   (kbdim,klev)  !< cloud ice   concentration at step t-dt
    REAL(wp),INTENT(IN) :: pxtm1   (kbdim,klev,ktrac) !< specific density of other tracers at step t-dt

    REAL(wp),INTENT(IN) :: pgeom1 (kbdim,klev)   !< geopotential above ground
    REAL(wp),INTENT(IN) :: pztottevn(kbdim,klev) !< intermediate value of TTE

    REAL(wp),INTENT(INOUT) :: bb    (kbdim,klev,nvar_vdiff)  !<

    REAL(wp),INTENT(IN)    :: pzthvvar (kbdim,klev) !< intermediate value of thvvar
    REAL(wp),INTENT(INOUT) :: pxvar    (kbdim,klev) !< distribution width (b-a)
                                                    !< in: step t-dt, out: modified
                                                    !< due to vertical diffusion

    ! Roughness length
    ! In: values over each surface type.
    ! Out: z0m_tile over the ocean is updated using the time average ("hat" value)
    ! of u and v and the t-dt value of some other variables.

    REAL(wp),INTENT(INOUT) :: pz0m_tile (kbdim,ksfc_type)

    ! Vertically integrated dissipation of kinetic energy [W/m2]

    REAL(wp),INTENT(OUT) :: pkedisp(kbdim)

    ! Tendencies

    REAL(wp),INTENT(OUT) :: pute_vdf (kbdim,klev)
    REAL(wp),INTENT(OUT) :: pvte_vdf (kbdim,klev)
    REAL(wp),INTENT(OUT) :: pq_vdf   (kbdim,klev)
    REAL(wp),INTENT(OUT) :: pqte_vdf (kbdim,klev)
    REAL(wp),INTENT(OUT) :: pxlte_vdf(kbdim,klev)
    REAL(wp),INTENT(OUT) :: pxite_vdf(kbdim,klev)
    REAL(wp),INTENT(OUT) :: pxtte_vdf(kbdim,klev,ktrac)

    ! Some other diagnostics

    REAL(wp),INTENT(OUT) :: pz0m      (kbdim)      !< grid-box mean roughness height
    REAL(wp),INTENT(OUT) :: pthvvar   (kbdim,klev) !< variance of virtual potential temperature
                                                   !< at the new time step t
    REAL(wp),INTENT(OUT) :: ptotte    (kbdim,klev)
    REAL(wp),INTENT(OUT) :: psh_vdiff (kbdim)      ! sens. heat flux
    REAL(wp),INTENT(OUT) :: pqv_vdiff (kbdim)      ! qv flux


    !-----------------------------------------------------------------------
    ! 6. Obtain solution of the tri-diagonal system by back-substitution.
    !    Then compute tendencies and diagnose moisture flux etc.
    !-----------------------------------------------------------------------
    CALL rhs_bksub( jcs, kproma, kbdim, itop, klev, aa, bb ) ! in,...,in, inout

    CALL vdiff_tendencies( jcs, kproma, kbdim, itop, klev, klevm1,      &! in
                         & ktrac, ksfc_type, idx_wtr,                   &! in
                         & pdtime,                                      &! in
                         & pum1, pvm1, ptm1,                            &! in
                         & pmair, pmref,                                &! in
                         & pqm1, pxlm1, pxim1, pxtm1,                   &! in
                         & pgeom1, pcptgz,                              &! in
                         & pztottevn, pzthvvar,                         &! in
                         & pcfm_tile, pfrc, bb,                         &! in
                         & pkedisp,                                     &! out
                         & pxvar, pz0m_tile,                            &! inout
                         & pute_vdf, pvte_vdf, pq_vdf,                  &! out
                         & pqte_vdf, pxlte_vdf, pxite_vdf, pxtte_vdf,   &! out
                         & pz0m, ptotte, pthvvar,                       &! out
                         & psh_vdiff, pqv_vdiff                         )! out

    ! Note: computation of additional diagnostics, e.g., surface sensible heat flux,
    !       wind stress, 10m wind, 2m temperature etc., has not been implemented yet.

  END SUBROUTINE vdiff_up
  !-------------

END MODULE mo_vdiff_upward_sweep
