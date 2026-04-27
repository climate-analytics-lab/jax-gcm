! Minimal stub of ECHAM/ICON `mo_exception` for the standalone harness.
! Only `message`, `message_text`, `finish` are actually used by the modules
! we pull in. We make `message` a no-op (it would otherwise spam stdout)
! and `finish` an `error stop`.
MODULE mo_exception
  IMPLICIT NONE
  PUBLIC
  CHARACTER(LEN=512) :: message_text
  INTEGER, PARAMETER :: em_param = 1
CONTAINS
  SUBROUTINE message(name, text, level)
    CHARACTER(*), INTENT(IN) :: name, text
    INTEGER, INTENT(IN), OPTIONAL :: level
    ! Suppressed in harness — uncomment if you need the diagnostic chatter:
    ! WRITE(*,'(a,a,a)') TRIM(name), ': ', TRIM(text)
    IF (.FALSE.) PRINT *, name, text, level   ! silence unused-arg warnings
  END SUBROUTINE message

  SUBROUTINE finish(name, text)
    CHARACTER(*), INTENT(IN) :: name, text
    WRITE(0,'(a,a,a)') TRIM(name), ': ', TRIM(text)
    ERROR STOP 1
  END SUBROUTINE finish
END MODULE mo_exception
