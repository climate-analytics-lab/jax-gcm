# ECHAM6.3-HAM2.3 Mie-table lookup reference (#1017)

Two reference files, two purposes (#1017 W1):

## `hamrad_lookup.npz` — lookup arithmetic, data-free

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine
`mo_ham_rad.f90::ham_rad_fitplus` (nearest-neighbour branch, `loint=.FALSE.`),
compiled standalone with declaration-only stubs of `mo_ham_rad_data` (axis
metadata only, no NetCDF reading, no per-species refractive-index tables),
fed a SYNTHETIC test fixture
(`ham_mie_tables_test.py::synthetic_ham_mie_tables()` — jcm's own Bohren-
Huffman kernel on HAM's axes, NOT HAM's own data; see `ham_mie_tables.py`'s
module docstring for the measured differences). This file verifies the
LOOKUP and index arithmetic against the real `ham_rad_fitplus`; the table
CONTENT itself is irrelevant to what it checks (and is validated separately,
against jcm's own quadrature, by a direct `mie.py` spot check in the same
test module, and against HAM's real data by the file below). **Data only**:
no ECHAM or HAM source is part of this repository.

## `hamrad_lookup_authentic.npz` — parity against HAM's real data

The same unmodified `ham_rad_fitplus`, fed HAM's **authentic** Mie tables
(`ham_mie_tables.py::load_ham_mie_tables()`, reading
`lut_optical_properties_M7.nc` / `lut_optical_properties_lw_M7.nc`). This is
the test that would catch an axis-order or transpose mistake in
`load_ham_mie_tables` the data-free file above cannot, because that one
never touches HAM's real file layout. The parity test against this file
(`ham_mie_tables_test.py::test_authentic_table_matches_compiled_
ham_rad_fitplus`) skips when `HAM_INPUT_DIR` or this file is absent — a
data-presence gate, not an optional extra.

## Arrays (one entry per case, `labels` order) — both files share this layout

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

From a copy of the harness (`/scr/dwatsonparris/ham-m7/w2/optics/harness`,
private scratch, not pushed; commit `69aafb5`) — copy it, never edit the
original (RULES.md):

```sh
# hamrad_lookup.npz (data-free, synthetic fixture):
python build_tables_binary.py <jcm worktree>    # writes the synthetic *.bin
gfortran -O2 -fcheck=all -ffpe-trap=invalid,zero,overflow -fbacktrace \
  -ffree-line-length-none mo_kind.f90 mo_ham_rad_data.f90 mo_ham_rad.f90 \
  driver.f90 -o harness
python gen_cases.py > cases.txt
./harness < cases.txt > fortran_out.txt
python compare_and_save.py <jcm worktree>

# hamrad_lookup_authentic.npz (HAM's real tables; needs HAM_INPUT_DIR):
python build_authentic_binary.py <jcm worktree> $HAM_INPUT_DIR
./harness < cases.txt > fortran_out_authentic.txt   # run with cwd set to the
                                                     # authentic *.bin files
python compare_authentic.py <jcm worktree> $HAM_INPUT_DIR
```

The compiled `harness` binary and `cases.txt` are shared between both runs
(axis metadata and case geometry do not depend on which table is fed in);
only the four `*_qext.bin`/`*_ssa.bin`/`*_g.bin` files it reads change.
