# ECHAM6.3-HAM2.3 GCR-ionisation reference data (jax-gcm#1017 Kazil/GCR task, Part B)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 `mo_ham_gcrion.f90::gcr_ionization` (+ `mo_geopack.f90`'s `recalc`/`geo2mag` it calls, + `mo_physical_constants.f90`'s `grav`) on designed columns, against the REAL O'Brien solar-min/max tables staged at `HAM_INPUT_DIR`. The intended numerical reference for `jcm/physics/aerosol/jam/chemistry/gcr_ionisation.py::gcr_ion_pair_rate`. **Data only**: no ECHAM or HAM source is part of this repository.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`hamgcr_provenance.json`).

`solar_activity` is NOT in this reference: it is a closed form of the date with no table/geomagnetic dependency, and `jcm/physics/aerosol/jam/chemistry/gcr_ionisation_test.py` already compares it against the Fortran formula transcribed directly in the test (the task's own split of this requirement).

## Arrays

| name | meaning |
|---|---|
| `meta/column_lat`, `meta/column_lon` | geographic coordinates [deg] of the 28 designed columns |
| `meta/scenario_names`, `meta/scenario_psolact`, `meta/scenario_date` | per-scenario solar-activity parameter and (year,month,day,hour,minute,second) |
| `in/pressure`, `in/temperature` | per-column, per-level inputs [Pa]/[K], shape (28, 10) |
| `in/vertical_cutoff_rigidity`, `in/mass_column_density` | the O'Brien table's own axes, as read from `HAM_INPUT_DIR` |
| `in/ipr_solmin`, `in/ipr_solmax` | the O'Brien table's own ion-pair-production arrays |
| `out/pgcripr` | `gcr_ionization`'s output [cm-3 s-1], shape (8 scenarios, 28 columns, 10 levels) |

## Regenerate

`python build_reference_hamgcr.py <jcm>/jcm/data/test/echam_cloud_reference --ham-input-dir <dir with gcr_ipr_solmin.txt/gcr_ipr_solmax.txt>` from the harness copy `fortran_harness/echam_hamgcr/py` (needs jcm importable for its own `read_obrien_gcr_ipr`, e.g. run from the jcm checkout root with `PYTHONPATH` set to it).
