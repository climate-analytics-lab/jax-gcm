# ECHAM6.3-HAM2.3 mixed-phase freezing-input reference data (#953)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine `mo_ham_freezing.f90::ham_IN_setup`
(which calls `get_aerofreez_nc`) on designed single-level M7 aerosol cells, compiled
standalone in double precision with declaration-only stubs of the HAM modules it USEs.
They are the numerical reference for
`jcm/physics/aerosol/jam/ice_nucleation/ham_freezing.py::ham_freezing_aerosol`, compared by
`jcm/physics/aerosol/jam/ice_nucleation/ham_freezing_reference_test.py` on an M7-shaped
jcm population. **Data only**: no ECHAM or HAM source is part of this repository.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`hamfrz_provenance.json`).

## Arrays (one entry per cell, `meta/names` order)

| name | meaning |
|---|---|
| `in/mass/<mode>/<species>` | tracer mass mixing ratio pxtm1 [kg/kg] (M7 modes nucs aits accs coas aiti acci coai; species so4 bc oc ss du) |
| `in/number/<mode>` | mode number pxtm1 [1/kg] |
| `in/nact/<mode>` | stratiform activated number nact_strat [1/m3] |
| `in/rwet/<mode>` | wet radius [m] |
| `in/rho`, `in/cdncact` | air density [kg/m3]; activated CDNC pcdncact [1/m3] |
| `in/density/<species>` | HAM species density [kg/m3] (mo_ham_species.f90) |
| `out/fracdusol`, `out/fracbcsol` | dust / BC fraction of the activated droplets |
| `out/fracduai`, `out/fracduci`, `out/fracbcinsol` | insoluble accumulation dust / coarse dust / BC over all insoluble aerosol |
| `out/rwetki`, `out/rwetai`, `out/rwetci` | wet radii of the insoluble Aitken / accumulation / coarse modes [m] |
| `out/ascs`, `out/apnx`, `out/aprx`, `out/apsigx` | cirrus inputs (not used by jcm) |
| `out/ndusol_strat`, `out/nbcsol_strat`, `out/nduinsolai`, `out/nduinsolci`, `out/nbcinsol`, `out/naerinsol` | the stream fields get_aerofreez_nc fills [1/m3] |

## Cells

| # | name | description |
|---|---|---|
| 0 | `empty` | no aerosol: every ratio 0 (zdenom <= zeps, F 248-253), every fraction 0 |
| 1 | `dust_soluble` | dust internally mixed in the soluble accumulation and coarse modes, activated |
| 2 | `bc_soluble` | BC in the soluble Aitken (excluded, DN #295), accumulation and coarse modes |
| 3 | `insoluble` | insoluble Aitken BC/OC, insoluble accumulation and coarse dust (contact inputs) |
| 4 | `mixed_all` | every mode populated |
| 5 | `fraction_clipped` | surface-weighted dust number above the activated CDNC: fraction MIN(., 1) = 1 |
| 6 | `no_activation` | dust present but no activated droplets and cdncact 0: fracdusol 0 |
| 7 | `tiny_ratio` | a dust volume ratio below EPSILON(1d0) is zeroed (F 256-257) |
| 8 | `tiny_class` | a class whose total mass lies below EPSILON(1d0) counts as empty (F 248-253) |

## Regenerate

`python build_reference_hamfrz.py <jcm>/jcm/data/test/echam_cloud_reference`
from the harness copy `fortran_harness/echam_hamfrz/py`.
