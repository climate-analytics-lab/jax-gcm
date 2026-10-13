# ECHAM6.3-HAM2.3 ham_m7_invertlogtail reference data (#1017 follow-up A)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 `mo_ham_tools.f90::ham_m7_invertlogtail` (the closed-form inverse-erf critical-radius inversion `ic_scav_nuc` uses) on a designed grid of (count-median-radius, xie, sigma) test points spanning the small/mid/huge-tail branches and both M7 geometric standard deviations (1.59, 2.0). **Data only**: no ECHAM or HAM source is part of this repository.

They are the numerical reference for `jcm/physics/aerosol/jam/wetdep/ham_nucleation.py::ham_m7_invertlogtail`, compared by `jcm/physics/aerosol/jam/wetdep/ham_nucleation_test.py`.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`haminvertlogtail_provenance.json`).

## Arrays (one entry per test point)

| name | meaning |
|---|---|
| `cmr` | count-median radius, `pcmr` [m] |
| `xie` | the inverse-erf argument, `pxie` [-] |
| `sigma` | geometric standard deviation (sigmaln = log(sigma) is what the routine actually reads) |
| `critrad` | `ham_m7_invertlogtail`'s output, `pcritrad` [m] |

## Regenerate

`python build_reference_haminvertlogtail.py <jcm>/jcm/data/test/echam_cloud_reference` from this directory (`fortran_harness/echam_hamnuc`).
