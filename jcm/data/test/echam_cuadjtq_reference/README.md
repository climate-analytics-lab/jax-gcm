# ECHAM6.3 `cuadjtq`, evaluated

`echam_cuadjtq.npz` holds the temperature and humidity that ECHAM6.3-HAM2.3
(r7492) `mo_cuadjust.f90::cuadjtq` returns for 55 designed parcels, for each
of its three modes `kcall = 0, 1, 2`.
`jcm/physics/convection/tiedtke_nordeng/cuadjtq_test.py` compares jcm's
`cuadjtq`, `cuadjtq_newton` and `cuadjtq_newton_evap` against it.

The file contains numbers only. No ECHAM source code is stored in this
repository.

## Arrays

| name | unit | meaning |
|---|---|---|
| `case` | – | a label per parcel (see below) |
| `temperature_in`, `humidity_in`, `pressure` | K, kg/kg, Pa | the parcel handed to `cuadjtq` (`pt`, `pq`, `pp`) |
| `saturation_ratio_design` | – | `S` in `humidity_in = S·qs(temperature_in)`, with `qs` the analytic `ua` form of ECHAM's constants; a design label, not a model input |
| `kcall{k}_temperature_out`, `kcall{k}_humidity_out` | K, kg/kg | `pt`, `pq` after `cuadjtq(kcall = k)`, ECHAM exactly as it runs (spline tables) |
| `analytic_kcall{k}_temperature_out`, `analytic_kcall{k}_humidity_out` | K, kg/kg | the same with the lookup tables replaced by the Sonntag (1990) fit they tabulate (see below) |
| `echam_rd`, `echam_rv`, `echam_cpd`, `echam_alv`, `echam_als`, `echam_tmelt`, `echam_vtmpc1` | SI | ECHAM's constants the routine ran with (`mo_physical_constants.f90`) |

## Parcels

- `warm`: 285, 295, 303 K at 950, 850, 1000 hPa; `cold`: 230, 250, 262 and
  195 K at 300, 500, 650 and 100 hPa. Each at `S = 0.5`, `1.0`, `1.02`, `1.3`
  and `2.0`: subsaturated, saturated, and supersaturated up to a first
  Newton step that overshoots by up to 2.7 K.
- `warms across tmelt`: 272.6 to 273.15 K, `S = 1.2` and `1.5` at 800 hPa,
  where the first step's condensation heats the parcel above the melting
  point, so the refinement reads the water table and `alv`.
- `cools across tmelt`: 273.16 to 274.0 K, `S = 0.3` and `0.7` at 800 hPa,
  where the first step's evaporation cools it below, so the refinement reads
  the ice table and `als`.
- `zes branch`: 300 K at the pressures where `es·rd/rv/p` is 0.45 and 0.7,
  ECHAM's `zes ≥ 0.4` form of the slope, uncapped and capped at 0.5.

Every parcel is one column of a `klev = 1` call at level `kk = 1`, with
every column in the `ldidx` list, as `cuini`, `cubase`, `cuasc`, `cudlfs`
and `cuddraf` pass their active columns.

## How it was produced

ECHAM's own `mo_cuadjust.f90` (md5 `ee47f6072e3d5446b70b598a23c735ba`) and
`mo_echam_convect_tables.f90` (md5 `ce2f14c2d84e7aa5c30149456671da0c`), with
`mo_kind`, `mo_math_constants`, `mo_physical_constants` (md5
`564906f3c4d084a935584d64a5b0d7a3`) and `mo_echam_cloud_params`, were
compiled unmodified with GNU Fortran 11.3.0
(`-O2 -ffp-contract=off -fimplicit-none`) against a small driver that calls
`init_convect_tables` and then `cuadjtq` on the parcels, on 2026-09-30. The
same build at `-O0` gives bit-identical output.

For the `analytic_` arrays, `mo_echam_convect_tables` was replaced by a
module that provides the lookups `cuadjtq` calls (`lookup_ua_list_spline`,
`lookup_ubc_list`) by evaluating the Sonntag fit and its derivative, ice at
and below `tmelt` and water above, instead of interpolating the splines;
`mo_cuadjust.f90` itself is unchanged.

## What it shows

Within ECHAM, the spline tables move the result from the analytic one by at
most 2.4e-11 K in temperature and 4.5e-13 of the humidity: the tables'
interpolation error (4e-12 in `es`, 1.8e-9 in its slope; see
`../echam_saturation_tables/`) carried through the two Newton steps.

Returning only the first Newton step moves the temperature by up to 2.7 K
from ECHAM's result, so the second step is resolved by every tolerance the
test uses.
