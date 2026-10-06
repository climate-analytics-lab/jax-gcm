# ECHAM6.3-HAM2.3 ndust=5 MSG-Sahara-override reference data (jax-gcm#1017)

Output of the **unmodified** `mo_ham_dust.f90:685-747` block (the
`k_dust_easo` East-Asia soil-replacement and the `ndust = 5` MSG-SEVIRI
Saharan dust-source-activation override, both inside
`bgc_read_annual_fields`) on 7 designed soil-mixture/`mat_msg` combinations.
The intended numerical reference for
`jcm/physics/aerosol/jam/emissions/dust.py::DustEmissions._soil_weights`'s
`use_msg_source` branch. **Data only**: no ECHAM or HAM source is part of
this repository — see `hamdustmsg_provenance.json`.

Integrity: -O0 vs -O2 max abs difference 0.0.

This block is wind- and soil-moisture-**independent** (it runs once at
annual-field read time, before any time-stepping), so the 7 cells vary only
the soil mixture, the preferential-source input and `dust_msg` — not wind
speed or soil wetness. The shared saltation/threshold-velocity chain those DO
affect is ndust=5's own unchanged ndust=4 code path, already covered by
`dust_test.py`'s pre-existing (hand-derived, not compiled) reference values.

## Arrays

| name | meaning |
|---|---|
| `labels` | the 7 cells' names |
| `type2`, `type3`, `type4`, `type6`, `type15` | input soil-texture area fractions fed to `DustEmissions._soil_weights` |
| `dust_preferential_in` | input `forcing.dust_preferential` before any override |
| `dust_msg` | input `forcing.dust_msg` (HAM's `mat_msg`) |
| `fortran_mat_s1`, `..._s2`, `..._s3`, `..._s4`, `..._s6` | the compiled Fortran's soil-fraction arrays AFTER both overrides |
| `fortran_mat_psrc` | the compiled Fortran's preferential-source fraction after both overrides |
| `fortran_residual_weighted` | `(1 - mat_psrc) * (mat_s1 - mat_s2 - mat_s3 - mat_s4 - mat_s6 - east_sum)`, the exact quantity `_soil_weights`'s first return row is — compared bit-for-bit (at JAX x64) in `dust_test.py`'s `MSGSourceTest` |

## Regenerate

`cd /scr/dwatsonparris/ham-m7/w7/dust-harness && gfortran -ffree-line-length-none -c mo_kind.f90 && gfortran -ffree-line-length-none -I. -c driver.f90 && gfortran -o dust_msg_harness mo_kind.o driver.o && JAX_PLATFORMS=cpu <venv>/bin/python compare.py` (needs jcm importable; `compare.py` inserts the worktree path itself).
