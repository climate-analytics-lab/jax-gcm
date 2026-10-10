# ECHAM-HAM `ic_scav_imp`, evaluated

`hamimpaction_T63L47.npz` holds the in-cloud impaction-scavenging fractions
that ECHAM6.3-HAM2.3 (r7492) `mo_ham_wetdep.f90::ic_scav_imp` returns for 61
cells and the four MAM4 modes, both phases (cloud droplets, ice plates) and
both moments (number, mass). `jcm/physics/aerosol/jam/wetdep/incloud_impaction_test.py`
compares `jcm.physics.aerosol.jam.wetdep.incloud_impaction` against it.

The file contains numbers only. No ECHAM or HAM source code is stored in this
repository.

## Cells

- **47 real cells** from the first step of a T63L47 `ma-t63-l47` run from the
  January warm state (`jam_fix1068_state365`, start 2001-12-31): the values
  `WetScavenging` received from the two-moment microphysics (`reffl`,
  `reffi`, the in-cloud crystal number after the microphysics) and from the
  MAM4 core (the wet count-median radius of each mode), with the air
  density. They were drawn at random within strata of droplet radius
  (5 um bins), crystal radius (0-50, 50-100, 100-150, 150 um), mixed-phase
  and clear cells, from 83°S to 87°N and physics levels 23-46 (top-first),
  206-294 K. `level`/`column` give the physics level and the lon-major
  column index (`column = ilon*96 + ilat`). The model's droplets stay below
  25 um and its crystals within [10, 150] um (`ceffmin`/`ceffmax`).
- **14 designed cells** cover what the real cells do not: droplets at and
  beyond 25-45 um (the `cdroprad(6)` node and the clamp), crystals below
  1 um, in [1, 5) and at 45-50, 50, 75, 99, 100, 149 and 700 um, a cell with
  crystal radius but no crystals, empty modes and modes above the 50 um cap.
  `cell_kind` names every cell.

## Arrays

| name | unit | meaning |
|---|---|---|
| `reffl`, `reffi` | um | in-cloud droplet / crystal effective radius (0 without the phase) |
| `icnc` | kg-1 | in-cloud crystal number; HAM's `zicnc = pxtp1c(idt_icnc)*prhop1` is `icnc*rho` |
| `rho` | kg m-3 | air density |
| `rwet` | m | (cell, mode) wet count-median radius, modes in `mode_names` order |
| `geom_std_dev` | – | mode geometric standard deviations; the driver was given `cmedr2mmedr = exp(3 ln²σ)` |
| `dt` | s | time step (720) |
| `out_mr_um` | um | (cell, mode, moment) HAM's lookup radius `mr` (number, mass) |
| `out_r7492_sfimp` | – | (cell, mode, phase, moment) `sfimp` of the unmodified routine; phase 0 water, 1 ice |
| `out_fixed_sfimp` | – | the same with the three corrections below |
| `table_*` | um, –, cm3 s-1 | the compiled `mo_ham_wetdep_data` tables (`caerorad`, `cdroprad`, `cplaterad`, `scavdropn`, `scavdropm`, `scaviceplate`) |

`sfimp` is unclipped; `get_icscavfrac` clips it to [0, 1], and so does the
test.

## How it was produced

Compiled with gfortran 11.3 at `-O0` and `-O2` (identical results), double
precision, against own stub `mo_kind`, `mo_time_control`, `mo_tracdef` and
`mo_activ` modules:

- `mo_ham_wetdep_data.f90` unmodified;
- `mo_ham_tools.f90::scavcoef_bilinterp` unmodified;
- `mo_ham_wetdep.f90::ic_scav_imp` unmodified, with the `mr`/`indexy1`/
  `indexy2` block of `ham_wetdep` (l. 262-286) wrapped in a subroutine.

A driver set `reffl`/`reffi`, `pxtp1c(idt_icnc)`, `prhop1` and
`time_step_len`, ran the `mr` block for both tracer phases of each mode and
called `ic_scav_imp` for both water phases. `kbdim = kproma` is the number of
cells and `klev = 1`.

`out_fixed_sfimp` comes from the same build with three edits to the source
copy: `cdroprad(6) = 30.0`; the `Q12`/`Q21` assignments of `ic_scav_imp`
exchanged so that `Q21` is the (x2, y1) corner and `Q12` the (x1, y2) corner
`scavcoef_bilinterp` reads them as; and the plate index above 50 um
`8 + FLOOR(reffi/25)` / `9 + FLOOR(reffi/25)`. See
`jcm/physics/aerosol/jam/wetdep/incloud_impaction.py`.
