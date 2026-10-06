# ECHAM6.3-HAM2.3 Mie-table lookup reference (#1017)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine
`mo_ham_rad.f90::ham_rad_fitplus` (nearest-neighbour branch, `loint=.FALSE.`),
compiled standalone with declaration-only stubs of `mo_ham_rad_data` (axis
metadata only, no NetCDF reading, no per-species refractive-index tables),
fed jcm's own built tables (`ham_mie_tables.build_ham_mie_tables()` -- the
authentic HAM LUT files are not available; see that module's docstring).
They are the numerical reference for
`jcm/physics/aerosol/jam/optics/ham_mie_tables.py::ham_rad_fitplus`, compared
by `ham_mie_tables_test.py`. **Data only**: no ECHAM or HAM source is part
of this repository.

This file verifies the LOOKUP and index arithmetic against the real
`ham_rad_fitplus`; the table CONTENT itself (whether jcm's Mie kernel
reproduces the physical Qext/SSA/g at a given point) is validated separately
by a direct `mie.py` spot check in the same test module.

## Arrays (one entry per case, `labels` order)

| name | meaning |
|---|---|
| `labels` | case name, see below |
| `ktable` | 1=SW fine (sigma=1.59), 2=SW coarse (sigma=2.0), 3=LW fine, 4=LW coarse |
| `x`, `nr`, `ni` | the lookup's size parameter, real RI, imaginary RI |
| `q_ext`, `ssa`, `g` | `ham_rad_fitplus`'s outputs (`ssa`/`g` are 0 for an LW table, which carries no such table) |

## Cases (56: 14 per table x 4 tables)

Per table: `x`/`nr`/`ni` axis-min and axis-max edges (6), one value just
below/above range on `x`, `nr` and `ni` respectively (4, each must give
exactly `0,0,0`), a point landing exactly on a half-integer step of each
axis's own `NINT` (3, pins round-half-AWAY-from-zero against NumPy/JAX's
round-half-to-even), and one ordinary interior point (1).

## Regenerate

From the harness copy, `fortran_harness/ham_mie_tables` (private scratch,
not pushed):

```sh
python build_tables_binary.py <jcm worktree>
gfortran -O2 -fcheck=all -ffpe-trap=invalid,zero,overflow -fbacktrace \
  -ffree-line-length-none mo_kind.f90 mo_ham_rad_data.f90 mo_ham_rad.f90 \
  driver.f90 -o harness
python gen_cases.py > cases.txt
./harness < cases.txt > fortran_out.txt
python compare_and_save.py <jcm worktree>
```
