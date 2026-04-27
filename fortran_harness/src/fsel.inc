! FSEL macro: floating-point select (preferred over MERGE because it can
! be implemented with a single conditional-move instruction on POWER and
! some x86 dialects). Definition matches ECHAM `mo_cloud.f90` etc.
#define FSEL(a,b,c) MERGE(b,c,(a).GE.0._wp)
#define SWDIV_NOCHK(a,b) ((a)/(b))
