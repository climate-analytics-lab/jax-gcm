# ECHAM6.3-HAM2.3 below-cloud scavenging (bc_rain/bc_snow) reference data (#1017)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 `mo_ham_wetdep.f90::bc_rain`/`bc_snow` (`kscavBCtype=3`, the Croft aerosol size-dependent below-cloud scheme), including the `indexy1`/`indexy2` index-computing snippet that normally runs inline in `ham_wetdep` just before calling them, on a designed grid of (precip flux, wet radius) test points. Compiled standalone in double precision via a visibility overlay (`bc_rain`/`bc_snow` are `PRIVATE` in the real module) -- see `mo_bc_wrap.f90`'s header for exact sed line ranges. **Data only**: no ECHAM or HAM source is part of this repository.

They are the numerical reference for `jcm/physics/aerosol/jam/wetdep/ham_below_cloud.py::bc_rain_rate`/`bc_snow_rate`, compared by `jcm/physics/aerosol/jam/wetdep/ham_below_cloud_test.py`.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`hamwetdep_bc_provenance.json`).

## Arrays (one entry per test point)

| name | meaning |
|---|---|
| `flux` | precip flux entering the layer, `pfrain`/`pfsnow` [kg/m2/s] |
| `radius` | wet radius fed to `bc_rain`/`bc_snow` as `rwet_p` (pre-`cmedr2mmedr`, `zrad_fac=1` for both phases here) [m] |
| `phase` | 1 = number (`ktrac_phase=1`), 2 = mass (`ktrac_phase=2`) |
| `rain_rate` | `bc_rain`'s output `sfrain` [1/s] |
| `snow_rate` | `bc_snow`'s output `sfsnow` [1/s] |

## Test points

48 (flux, radius) pairs x 2 phases = 96 rows, covering: zero flux/radius, every `crainrate` node exactly and between nodes, flux above the top node (extrapolation), radii from sub-eps through every decade, exactly at and well above the 50 um `MIN(..., 50e-6)` clip (`mo_ham_wetdep.f90:272`), and the aerosol-radius bin edges.

## Regenerate

`python build_reference_hamwetdep_bc.py <jcm>/jcm/data/test/echam_cloud_reference` from this directory (`fortran_harness/echam_hamwetdep`).
