# ECHAM6.3 / JSBACH land-tile reference data

Inputs and outputs of the **unmodified** ECHAM6.3 / JSBACH routines that couple
the land surface to the atmosphere, run in double precision on single columns
by a standalone harness. They are the numerical reference for the ECHAM land
tile in `jcm/physics/surface/echam/jsbach_land.py` (humidity factors, skin
energy balance) and its implicit coupling in
`jcm/physics/vertical_diffusion/tte_tke/` (#979).

These files hold **data only**. ECHAM6 and JSBACH are under the MPI-M Software
Licence Agreement; no ECHAM or JSBACH source is part of this repository. The
harness copies the routines from a local ECHAM6.3 tree at build time.
`provenance.json` records the revision and checksums of every source file, the
verbatim line ranges, the compiler and flags, every stub, the inputs and the
integrity checks.

## File

`land_T63L47.npz`, every array one-dimensional:

| prefix | content |
|---|---|
| `meta/names`, `meta/box` | column label `box/month/lat/lon` and its box |
| `col/*` | the control-run column the inputs came from (surface fluxes, prescribed fields, `wsmx_echam` = ECHAM's own T63 `WSMX` at that point) and the stand-in `cap`/`lam` used for the energy balance |
| `precalc_in/*`, `precalc1/*` | `atm_conditions` + `precalc_land` with the beta-form factors `cair = csat = w` |
| `factors_in/*`, `factors/*` | the three relative-humidity functions and the canopy + humidity-factor blocks of `update_soil`, `bucket/` = `calc_relative_humidity` (nsoil = 1), `upper/` = `calc_relative_humidity_upper` (nsoil = 5) |
| `precalc2_in/*`, `precalc2/*` | `precalc_land` again with the `upper/` factors |
| `seb_in/*`, `seb/*` | `richtmyer_land` (`zetnl`, `zftnl`, `zeqnl`, `zfqnl`), `update_surfacetemp` (`psnew`), the new surface `qsnew` and `update_land` (`ztklevl`, `zqklevl`) |
| `factor_scan_in/*`, `factor_scan/*` | 1134 synthetic columns through every switch: snow 0/0.4/1, glacier 0/0.5/1, vegetation 0/0.6/1, moisture across wilting and critical, `q_a/q_s` across the bare-soil hinge and the dew switch |
| `rh_scan_in/*`, `rh_scan/*` | the relative-humidity functions and the stress factor on a 401-point moisture grid for seven capacities |
| `soiltemp_scan_in/*`, `soiltemp_scan/*` | `update_soiltemp` (5 layers): the top layer's capacity `pgrndcapc` and flux `pgrndhflx` with `pgrndc(1)`, `pgrndd(1)`, for the six FAO soil rows, glacier, and five snow depths |

## Conventions

* Units are ECHAM's: dry static energy `s` in J/kg (`psold`, `psnew`,
  `ztklevl`), `zcfh`/`zcfhl` in ECHAM's pressure-scaled form
  (`zcfh·zqdp` dimensionless), humidity in kg/kg.
* Columns were taken top-first from the surface-first model output (selected by
  `pressure_full`/`pressure_half`, the lowest level is the largest pressure).
* `time_step_len = delta_time = 720 s`, jcm's T63L47 physics step: jcm has one
  two-time-level step where ECHAM's leapfrog distinguishes the two.
* `cvdifts = 1.5`, `ckap = 0.4`, `cb = cc = 5` (`iniphy.f90`).
* Saturation is ECHAM's spline table (`mo_echam_convect_tables.f90`).

## Regenerate

```bash
cd /scr/dwatsonparris/land-seb/harness
make && make OPT=-O0
python py/build_reference.py <worktree>
python py/check_opt.py <worktree>
```
