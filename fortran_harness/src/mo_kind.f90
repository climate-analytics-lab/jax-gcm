!>
!! @par Copyright
!! This code is subject to the MPI-M-Software - License - Agreement in it's most recent form.
!! Please see URL http://www.mpimet.mpg.de/en/science/models/model-distribution.html and the
!! file COPYING in the root of the source tree for this code.
!! Where software is supplied by third parties, it is indicated in the headers of the routines.
!!
MODULE mo_kind

  ! L. Kornblueh, MPI, August 2001, added working precision and comments 

  IMPLICIT NONE

  ! Number model from which the SELECTED_*_KIND are requested:
  !
  !                   4 byte REAL      8 byte REAL
  !          CRAY:        -            precision =   13
  !                                    exponent  = 2465
  !          IEEE:    precision =  6   precision =   15  
  !                   exponent  = 37   exponent  =  307 
  !
  ! Most likely this are the only possible models.

  ! Floating point section: 

  INTEGER, PARAMETER :: ps = 6
  INTEGER, PARAMETER :: rs = 37

  INTEGER, PARAMETER :: pd = 12
  INTEGER, PARAMETER :: rd = 307

  INTEGER, PARAMETER :: pi4 = 9
  INTEGER, PARAMETER :: pi8 = 14

  INTEGER, PARAMETER :: sp = SELECTED_REAL_KIND(ps,rs)  
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(pd,rd)

  ! Floating point working precision

  INTEGER, PARAMETER :: wp = dp   

  ! Integer section

  INTEGER, PARAMETER :: i4 = SELECTED_INT_KIND(pi4)
  INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(pi8)

  ! Working precision for index variables
  !
  ! predefined preprocessor macros:
  !
  ! xlf         __64BIT__   checked with P6 and AIX
  ! gfortran    __LP64__    checked with Darwin and Linux
  ! Intel, PGI  __x86_64__  checked with Linux
  ! Sun         __x86_64    checked with Linux 

#if defined (__64BIT__) || defined (__LP64__) || defined (__x86_64__) || defined (__x86_64)
  INTEGER, PARAMETER :: widx = i8
#else
  INTEGER, PARAMETER :: widx = i4
#endif

CONTAINS

  SUBROUTINE print_kinds  

    PRINT *, 'single precision : ', sp
    PRINT *, 'double precision : ', dp
    PRINT *, 'working precision: ', wp
    PRINT *, '4 byte integer   : ', i4
    PRINT *, '8 byte integer   : ', i8
    PRINT *, 'index integer    : ', widx

#if defined (__64BIT__) 
    ! xlf
    PRINT *, '__64BIT__' 
#endif
#if defined (__LP64__)
    ! gfortran
    PRINT *, '__LP64__'
#endif
#if defined (__x86_64__)
    ! Intel, PGI
    PRINT *, '__x86_64__'
#endif
#if defined (__x86_64)
    ! Sun
    PRINT *, '__x86_64'
#endif
  ! NAG has no predefined macro but speed is no issue anyhow

  END SUBROUTINE print_kinds

END MODULE mo_kind
