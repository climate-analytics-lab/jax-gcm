# ECHAM6.3-HAM2.3 ic_scav -> get_icscavfrac -> {ic_scav_nuc,ic_scav_imp} reference (#1017 follow-up B)

Outputs of the **unmodified** `ic_scav -> get_icscavfrac -> {ic_scav_nuc,
ic_scav_imp}` chain (ECHAM6.3-HAM2.3 r7492's `mo_ham_wetdep.f90`, plus
everything it calls -- `ham_m7_logtail`/`ham_m7_invertlogtail`/
`scavcoef_bilinterp` from `mo_ham_tools.f90`, `m7_cumulative_normal` from
`mo_ham_m7.f90`, and the REAL `mo_ham_wetdep_data.f90` -- all compiled
verbatim) on a 27-case matrix of M7 columns (`driver.f90`). Cases 1-19
are follow-up A's nucleation matrix (now additionally exercising the REAL
`ic_scav_imp`, not the follow-up-A stub); cases 20-27 are new,
impaction-specific: the water collector-radius (`cdroprad`) bins
including the disputed node 6 (`reffl` in [25,35) um), the ice
collector-radius (`cplaterad`) three regimes (dead/`reffi<1`,
fine/`[1,50)`, coarse/`>=50`), and the `icnc<eps` gate. **Data only**: no
ECHAM or HAM source is part of this repository.

They are the numerical reference for
`jcm/physics/aerosol/jam/wetdep/ham_impaction.py` AND the COMBINATION
`get_icscavfrac` performs (`pfrac = clip(pfrac_nuc+pfrac_imp, 0, 1)`),
compared by `jcm/physics/aerosol/jam/wetdep/ham_impaction_test.py`.

Integrity: -O0 vs -O2 max abs difference 0.0 (`icscavimp_provenance.json`).

## Arrays (one row per (case, tracer phase); 27 cases x 2 tracer phases = 54 rows)

| name | meaning |
|---|---|
| `case_name` | the test case's Fortran driver label |
| `kwat_phase` | 1 = water, 2 = ice |
| `kmod` | M7 mode index, 1-based Fortran convention (1=NS,2=KS,3=AS,4=CS,5=KI,6=AI,7=CI) |
| `ncd_activ` | 2 = ARG (dry radius); anything else = Lin & Leaitch (wet radius) |
| `ktrac_phase` | 1 = number, 2 = mass |
| `cdnc`, `icnc`, `nks`, `nas`, `ncs`, `na`, `frac`, `radius` | nucleation inputs (see `icscavnuc_README.md`); `cdnc` defaults to 1e-9 here (see driver.f90's comment) so `get_icscavfrac`'s own `pxtp1c(:,:,kt)>zeps` impaction gate passes |
| `prho`, `sigma` | air density [kg/m3], geometric standard deviation |
| `wetrad` | the aerosol wet radius [m] fed to `compute_indexy_mr` (ic_scav_imp's `mr`/`indexy1`/`indexy2`, SHARED with the below-cloud pathway -- always wet, never switched by `ncd_activ`) |
| `reffl`, `reffi` | the collector radius [um] (cloud-droplet / ice-plate) fed to `ic_scav_imp` |
| `pfrac_nuc` | `get_icscavfrac`'s nucleation scavenged fraction |
| `pfrac_imp` | `get_icscavfrac`'s impaction scavenged fraction |
| `pfrac` | the COMBINED, clipped fraction `get_icscavfrac` actually returns -- `clip(pfrac_nuc + pfrac_imp, 0, 1)` |

## Regenerate

`python build_reference_icscavimp.py <jcm>/jcm/data/test/echam_cloud_reference`
from this directory (`fortran_harness/echam_icscavimp`), after rebuilding
and rerunning `driver.f90` (see that file's own header) at both `-O0` and `-O2`.
