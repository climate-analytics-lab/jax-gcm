# ECHAM6.3 `vdiff` interior buoyancy and surface-layer Richardson number, evaluated

`vdiff_T63L47.npz` holds what ECHAM6.3-HAM2.3 (r7492) computes, in double
precision, for the moist, cloud-weighted buoyancy of the vertical diffusion:

* the interior half-level `zbuoy`, `zshear` and `zri` of `vdiff.f90`
  (l.658-700 and 777-799) for 384 columns, and the quantities they are built
  from;
* the bulk Richardson number of the surface layer, `zril`, `zriw` and `zrii`
  of `mo_surface_land.f90::precalc_land`, `mo_surface_ocean.f90::precalc_ocean`
  and `mo_surface_ice.f90::precalc_ice`, for 96 cells.

`jcm/physics/vertical_diffusion/tte_tke/moist_buoyancy_test.py` compares
`interior_buoyancy_terms`, `compute_richardson_number` and
`surface_bulk_richardson` against it (#962).

The file contains numbers only. No ECHAM source code is stored in this
repository. `provenance.json` records the checksums of every source file, the
verbatim line ranges, the compiler and flags, and the stubs.

## Arrays

Column arrays are `(level, column)` with the model top first, as in ECHAM
(`jk = 1` is the top). `klev = 47`; half-level arrays have 46 rows,
`[i]` is the interface between full levels `i` and `i + 1`.

| name | unit | meaning |
|---|---|---|
| `in_ptm1`, `in_pqm1`, `in_pxlm1`, `in_pxim1` | K, kg/kg | temperature, humidity, cloud water, cloud ice (`vdiff`'s `ptm1`, `pqm1`, `pxlm1`, `pxim1`) |
| `in_pum1`, `in_pvm1` | m/s | winds |
| `in_papm1`, `in_paphm1` | Pa | full-level pressure (47 rows), interface pressure (48 rows, the last is the surface) |
| `in_pgeom1` | m²/s² | full-level geopotential (the model's, orography included; only differences enter) |
| `in_paclc` | – | cloud cover (`cover`'s `paclc`, here the run's end-of-step fraction) |
| `echam_zqss`, `echam_zlteta1`, `echam_ztvir1` | kg/kg, K, K | full-level saturation humidity, liquid-water potential temperature, virtual potential temperature (47 rows) |
| `echam_zqssm`, `echam_zdus1`, `echam_zdus2` | kg/kg, –, – | interface saturation humidity and the two cloud-weighted multipliers |
| `echam_zteldif`, `echam_zqddif`, `echam_zthvirdif` | K/m, kg/kg/m, K/m | `∂θ_l/∂z`, `∂q_t/∂z`, `∂θ_v/∂z` of the interface |
| `echam_zbuoy`, `echam_zshear`, `echam_zri` | 1/s², 1/s², – | buoyancy, squared shear, Richardson number |
| `echam_analytic_zqss`, `echam_analytic_zbuoy`, `echam_analytic_zri` | | the same with the spline tables replaced by the analytic Sonntag fit (see below) |
| `sfc_in_*`, `sfc_echam_*` | | the 96 land cells: ECHAM's `precalc_land` inputs and its `zril`, `zqsl`, and the full-level `zx`, `zteta1`, `ztvir1`, `zfaxe`, `zlteta1`, `zqss` from `atm_conditions` |
| `sfc_in_ocean_ptslm1`, `sfc_in_ocean_paz0lm`, `sfc_echam_ocean_*` | | the same atmosphere over open water: skin temperature, roughness length and `precalc_ocean`'s `zriw` |
| `sfc_in_ice_ptslm1`, `sfc_in_ice_paz0lm`, `sfc_echam_ice_*` | | and over sea ice (`zrii`) |
| `column_class`, `column_origin` | | label of the stratum a column was drawn from, and the snapshot (`cap_day1.nc@t<i>`) or `designed` |
| `echam_grav`, `echam_rd`, `echam_rv`, `echam_cpd`, `echam_alv`, `echam_als`, `echam_tmelt`, `echam_vtmpc1`, `echam_vtmpc2` | SI | ECHAM's constants the routines ran with (`mo_physical_constants.f90`) |

## Columns

* 320 model columns from a 1-day `t63-echam-1m` run from a spun-up state
  (instantaneous fields every 6 h), 10 per class per snapshot: unstable
  boundary layer, stable boundary layer (an inversion in the lowest five
  layers), cloudy (cover > 0.3 and condensate), saturated (cover > 0.9), a
  level pair straddling 273.15 K, cold (below 215 K), low cloud (cover > 0.5 in
  the lowest twelve levels) and random.
* 64 designed columns on the L47 hybrid levels, for what the model states reach
  rarely: every cover from 0 to 1 in one profile, supersaturation, T equal to
  and on both sides of the melting point at adjacent levels, calm air (shear
  below ECHAM's floor `zepshr = 1e-5`), strong shear and jets, inversions and a
  super-adiabatic surface layer.
* 96 surface cells: the lowest level of columns with cover > 0 (48) and
  cover = 0 (48), with the skin temperature near the air's. `sfc_in_pgeom1` is
  the hypsometric height of the lowest full level above the surface times `g`,
  which is what ECHAM's `pgeom1(klev)` is at the surface layer; the interior
  columns carry the model's geopotential instead because only differences of it
  enter the interior.
* `time_step_len = delta_time = 720 s`, `cvdifts = 1.5`, `ckap = 0.4`,
  `cb = cc = 5` (`iniphy.f90`); the Richardson numbers do not read the first
  two.

## How it was produced

The statements of `vdiff.f90` above were copied verbatim (line ranges checked
against their first and last lines) into a wrapper that declares the variables
they use, and compiled with GNU Fortran 11.3.0 (`-O2 -ffp-contract=off
-fimplicit-none`) together with the unmodified `mo_kind`,
`mo_physical_constants`, `mo_physc2`, `mo_echam_cloud_params`,
`mo_echam_convect_tables`, `mo_surface_boundary`, `mo_surface_land`,
`mo_surface_ocean` and `mo_surface_ice`. The `-O0` build gives bit-identical
output. On 2026-10-02.

The `echam_analytic_*` arrays come from a second build in which
`mo_echam_convect_tables` is replaced by a module that evaluates the Sonntag
(1990) fit the tables tabulate (ice at and below the melting point, water above)
instead of interpolating the splines; the `vdiff.f90` statements are unchanged.
It separates the arithmetic from the tables' interpolation error.

## What it shows

* ECHAM with its tables and with the analytic fit differ in `zqss` by 6.5e-14 of
  its largest value (the tables' interpolation error) and in `zbuoy` and `zri`
  by 9e-16 and 1e-16 of theirs, so the arithmetic of the statements is
  separated from the tables; the test compares jcm's evaluation of the fit with
  the analytic build at round-off and with ECHAM as it runs to the tables'
  interpolation error.
* The cover is non-zero in 11% of the full-level cells and strictly between 0
  and 1 in 7%; the shear floor is the denominator in 79% of the interfaces; 3.9%
  of the interfaces are unstable (`zri < 0`). 278 interfaces have one level
  below the melting point and the other at or above it, so the latent-heat blend
  is exercised.
