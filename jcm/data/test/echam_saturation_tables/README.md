# ECHAM6.3 saturation lookup tables, evaluated

`echam_saturation_tables.npz` holds the saturation vapour pressure that
ECHAM6.3-HAM2.3 (r7492) reads from its lookup tables, and its temperature
derivative, on 1102 temperatures. `jcm/physics/thermodynamics_test.py`
compares jcm's Sonntag (1990) functions against it.

The file contains numbers only. No ECHAM source code is stored in this
repository.

## Arrays

| name | unit | meaning |
|---|---|---|
| `temperature` | K | 1001 points `150.0123 + 0.18·k` (150.0 to 330.0 K), 100 points between 273.1003 and 273.1997 K, and 273.15 K itself |
| `es_ua` | Pa | `ua·rv/rd`: ECHAM's `ua` table (ice at and below `tmelt`, water above) |
| `dlnes_dT_ua` | 1/K | `dua/ua`: its logarithmic temperature derivative |
| `es_uaw` | Pa | `uaw·rv/rd`: ECHAM's `uaw` table (water at all temperatures) |
| `dlnes_dT_uaw` | 1/K | `duaw/uaw` |
| `echam_rd`, `echam_rv`, `echam_tmelt` | J/kg/K, J/kg/K, K | ECHAM's `rd = 287.04`, `rv = 461.51`, `tmelt = 273.15` (`mo_physical_constants.f90`), used to undo the tables' `rd/rv` factor |

The spline tables hold knots every 0.025 K. The temperatures sit between
knots, where the cubic Hermite interpolation error is largest, apart from
the one point at `tmelt`.

## How it was produced

ECHAM's own `mo_echam_convect_tables.f90` (md5
`ce2f14c2d84e7aa5c30149456671da0c`), with `mo_kind`, `mo_physical_constants`
(md5 `564906f3c4d084a935584d64a5b0d7a3`) and `mo_echam_cloud_params`, was
compiled unmodified with GNU Fortran 11.3.0
(`-O2 -ffp-contract=off -fimplicit-none`) against a small driver that calls
`init_convect_tables`, `prepare_ua_index_spline`, `lookup_ua_spline` and
`lookup_uaw_spline` for these temperatures (the lookups that convection,
the cloud cover, the 1M scheme, the vertical diffusion and the surface use),
on 2026-09-30.

## What it shows

Against the analytic Sonntag (1990) fit that the tables tabulate, evaluated
in float64, the tables differ by at most 4.0e-12 in `es` and 1.6e-9 in
`d ln es/dT` on these temperatures (4.0e-12 and 1.8e-9 on a 1e-4 K scan from
150 to 330 K). That is the interpolation error of the spline, and it bounds
how closely any evaluation of the fit can agree with ECHAM.

The 2M scheme reads the 0.001 K tables `tlucua` and `tlucuaw` at the nearest
knot instead. At their knots those tables equal the analytic fit to 1.6e-14,
except at the knot `it = 273150`, whose temperature `0.001·273150` rounds
above `tmelt` in double precision and therefore holds the water value.
