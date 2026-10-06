# ECHAM6.3 `ssodrag`, evaluated

`ssodrag_T63L47.npz` holds the zonal/meridional wind and temperature
tendencies that ECHAM6.3-HAM2.3 (r7492) `mo_ssortns.f90::ssodrag` (Lott & Miller
sub-grid orographic drag, with `mo_ssodrag.f90::sugwd` setting its level
constants on the jcm L47 grid) returns for four real T63L47 columns.
`jcm/physics/gravity_waves/sso/lott_miller_test.py` compares jcm's
`sso_drag` against it, with `nktopg` from `echam_nktopg`.

The file contains numbers only. No ECHAM source code is stored in this
repository.

## Columns

Five-day means (days 0-5) of the spun-up `ma-t63-l47` warm state over
land with real T63 sub-grid orography descriptors
(`hf://bundles/t63/terrain.nc`): Tibet (27.0N 91.9E), the Andes (14.0S
69.4W), the Alps (49.4N 13.1E) and the Rockies (53.2N 112.5W). They span
both signs of the column stress.

## Arrays

| name | unit | meaning |
|---|---|---|
| `pressure_half`, `pressure_full` | Pa | half (48) / full (47) level pressures, top-first |
| `height_full` | m | full-level geopotential height above sea level (jcm's `height_full`); ECHAM was given `(height_full - orog)·grav` as `pgeom1` |
| `temperature`, `u_wind`, `v_wind` | K, m/s | full-level state, top-first |
| `orog`, `orostd`, `orosig`, `orogam`, `orothe`, `oropic`, `oroval` | m, m, –, –, deg, m, m | sub-grid orography descriptors (`pmea` … `pval`) |
| `lat` | deg | column latitude (ECHAM's `orolift` input; inert at `gklift = 0`) |
| `dt` | s | time step (720) |
| `echam_dudt`, `echam_dvdt`, `echam_dtdt` | m/s², m/s², K/s | ECHAM `pvom`, `pvol`, `ptte` |
| `echam_u_stress`, `echam_v_stress` | N/m² | ECHAM `pustrgw`, `pvstrgw` |
| `echam_nktopg` | – | the `nktopg` `sugwd` set on this grid (45) |

## How it was produced

`mo_ssodrag.f90` and `mo_ssortns.f90` compiled unmodified (gfortran,
`-cpp -fdefault-real-8`, neither `__ICON__` nor `HAMMOZ` defined, `nn = 63`)
against stub `mo_kind`, `mo_math_constants`, `mo_control` and
`mo_exception` modules and an `mo_physical_constants` carrying jcm's
`grav = 9.81`, `rd = 287.04`, `cpd = 1004.64`, so the comparison isolates the
algorithm. `vct` is jcm's L47 `(a, b)` table. A driver read the columns and
called `ssodrag` once with zero incoming tendencies.
