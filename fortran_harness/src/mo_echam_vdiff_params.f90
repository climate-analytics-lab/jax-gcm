!>
!! @brief Parameters for the boundary layer parameterization. 
!!
!! @author Hui Wan, MPI-M
!!
!! @par Revision History
!! First version by Hui Wan, MPI-M (2010-08-31)
!! Modification by Constantin Junk, MPI-M (2011-05-05)
!! - moved echam_vdiff_ctl and setup_vdiff to namelists/mo_echam_vdiff_nml
!!
!! @par Copyright and License
!!
!! This code is subject to the DWD and MPI-M-Software-License-Agreement in
!! its most recent form.
!! Please see the file LICENSE in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the
!! headers of the routines.
!!
MODULE mo_echam_vdiff_params

  USE mo_kind,              ONLY: wp
  USE mo_physical_constants,ONLY: grav

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: ckap, cchar, cb, cc                         !< parameters
  PUBLIC :: eps_shear, eps_corio, totte_min, z0m_min    !< parameters
  PUBLIC :: z0m_ice, z0m_oce
  PUBLIC :: chneu, shn, smn, da1                        !< parameters
  PUBLIC :: cons5                                       !< parameters
  PUBLIC :: cvdifts, tpfac1, tpfac2, tpfac3, tpfac4     !< parameters
  PUBLIC :: itop, itopp1, ibl, iblm1, iblmin, iblmax    !< parameters
  
  PUBLIC :: lmix_max
  PUBLIC :: init_vdiff_params                           !< subroutine


  !-------------------
  ! Module parameters
  !-------------------

  REAL(wp),PARAMETER :: ckap    = 0.4_wp       !< karman constant.
  REAL(wp),PARAMETER :: cchar   = 0.018_wp     !< charnock constant.
  REAL(wp),PARAMETER :: cb      = 5._wp        !< stability parameter near neutrality.
  REAL(wp),PARAMETER :: cc      = 5._wp        !< stability parameter for unstable cases.

  REAL(wp),PARAMETER :: lmix_max  = 1.e3_wp    !< maximum mixing length in neutral and stable conditions
  REAL(wp),PARAMETER :: eps_shear = 1.e-5_wp   !< zepshr in sbr. vdiff of ECHAM6
  REAL(wp),PARAMETER :: eps_corio = 5.e-5_wp   !< zepcor in sbr. vdiff of ECHAM6
  REAL(wp),PARAMETER :: totte_min = 1.e-10_wp  !< minimum total turbulent energy 
  REAL(wp),PARAMETER :: z0m_min   = 1.5e-5_wp  !< zepzzo in sbr. vdiff of ECHAM5
  REAL(wp),PARAMETER :: z0m_ice   = 1e-3_wp    !< cz0ice in sbr. vdiff of ECHAM5
  REAL(wp),PARAMETER :: z0m_oce   = 5e-4_wp    !< see mo_surface_ocean.f90 of ECHAM6

  REAL(wp),PARAMETER :: chneu  = 0.3_wp

  REAL(wp),PARAMETER :: cons5  = 3._wp*cb*cc

  ! Parameters related to time step weighting in *rhs* of *vdiff* and *scv*

  REAL(wp),PARAMETER :: cvdifts = 1.5_wp
  REAL(wp),PARAMETER :: tpfac1  = cvdifts
  REAL(wp),PARAMETER :: tpfac2  = 1._wp / tpfac1
  REAL(wp),PARAMETER :: tpfac3  = 1._wp - tpfac2
  REAL(wp),PARAMETER :: tpfac4  = 1._wp + tpfac3

  REAL(wp),PARAMETER :: shn = 2.22_wp*0.22_wp*SQRT(2._wp)
  REAL(wp),PARAMETER :: smn = shn*1.24_wp*2.37_wp/3.69_wp
  REAL(wp),PARAMETER :: da1 = 1._wp/smn**3

  !-------------------
  ! Module variables
  !-------------------

  INTEGER :: itop
  INTEGER :: itopp1
  INTEGER :: ibl
  INTEGER :: iblm1
  INTEGER :: iblmin
  INTEGER :: iblmax

CONTAINS
  !-------------
  !>
  !!
  SUBROUTINE init_vdiff_params( klev,klevp1,kvclev,vct )

    INTEGER,INTENT(IN)  :: klev, klevp1, kvclev
    REAL(wp),INTENT(IN) :: vct(2*kvclev)

    INTEGER  :: jk
    REAL(wp) :: zph(klevp1), zp(klev), zh(klev)

    !------------------

    itop   = 1
    itopp1 = itop + 1

    ibl    = klev - 1    ! CO2 flux is distributed into lowest layers klev to ibl
    iblm1  = ibl - 1

    ! CO2 mixing in PBL.
    ! Compute lowest (klev-2, approx. 500m in L19/31) and highest
    !   (highest below 2km) level ktop.
    ! CO2 is ideally mixed between bottom and itop.

    iblmin = klev - 2

    ! Search for the highest model level below 2000 m:
    ! First compute the half level pressure values,
    ! assuming 101320 Pa surface pressure

    DO jk=1,klevp1
      zph(jk)=vct(jk)+vct(jk+kvclev)*101320.0_wp
    END DO

    ! Then the full level pressure

    DO jk = 1,klev
      zp(jk)=(zph(jk)+zph(jk+1))*0.5_wp
    END DO

    ! Compute the elevation with respect to the Earth's surface

    DO jk = 1,klev
      zh(jk)=(zph(klevp1)-zp(jk))/(grav*1.25_wp)
    END DO

    ! Search for highest level below 2000m

    DO jk = 1,klev
      iblmax=jk
      IF(zh(jk).LT.2000.0_wp) EXIT
    END DO

  END SUBROUTINE init_vdiff_params
  !-------------

END MODULE mo_echam_vdiff_params

