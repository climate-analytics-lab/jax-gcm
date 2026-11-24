# ECHAM6.3-HAM2.3 ic_scav -> get_icscavfrac -> ic_scav_nuc reference data (#1017 follow-up A)

Outputs of the **unmodified** `ic_scav -> get_icscavfrac -> ic_scav_nuc` chain
(ECHAM6.3-HAM2.3 r7492's `mo_ham_wetdep.f90`, plus everything it calls --
`ham_m7_logtail`/`ham_m7_invertlogtail` from `mo_ham_tools.f90` and
`m7_cumulative_normal` from `mo_ham_m7.f90`, all compiled verbatim) on a
19-case matrix of M7 columns (`driver.f90`): liquid-only, ice-only and
mixed-phase; CDNC/ICNC above and below `zeps_mass`; `na` above and below
`zeps`; every M7 mode populated, and KS/AS/CS emptied in turn (the ice
phase's CS-then-AS-then-KS depletion order); ARG and Lin & Leaitch radius
selection; and the `xie->=1` "huge tail" clip. **Data only**: no ECHAM or
HAM source is part of this repository.

They are the numerical reference for
`jcm/physics/aerosol/jam/wetdep/ham_nucleation.py` (`water_phase_xie`,
`ice_phase_xie`, `ham_m7_invertlogtail`, `nucleation_scavenged_fraction`),
compared by `jcm/physics/aerosol/jam/wetdep/ham_nucleation_test.py`.
`ic_scav`'s own `pdxt_nuc/pxt` (`icscav_frac`) is recorded too, and equals
`sfnuc` exactly in every row (`peff=1`, `paclc=1` throughout) -- confirming
the outer wrapper the real model actually calls does not perturb the
nucleation fraction.

Integrity: -O0 vs -O2 max abs difference 0.0 (`icscavnuc_provenance.json`).

## Arrays (one row per (case, tracer phase); 19 cases x 2 tracer phases = 38 rows)

| name | meaning |
|---|---|
| `case_name` | the test case's Fortran driver label |
| `kwat_phase` | 1 = water, 2 = ice |
| `kmod` | M7 mode index, 1-based Fortran convention (1=NS,2=KS,3=AS,4=CS,5=KI,6=AI,7=CI) |
| `ncd_activ` | 2 = ARG (dry radius); anything else = Lin & Leaitch (wet radius) |
| `ktrac_phase` | 1 = number, 2 = mass |
| `cdnc`, `icnc` | in-cloud CDNC/ICNC mixing ratio fed to `pxtp1c_sav` [kg-1]; 0 when this case's phase does not read it |
| `nks`, `nas`, `ncs` | in-cloud KS/AS/CS number mixing ratio fed to `pxtp1c_sav` [kg-1]; 0 when unused |
| `na` | total activated/available aerosol number [m-3] (water phase only); 0 when unused |
| `frac` | this mode's activated fraction [-] (water phase only); 0 when unused |
| `radius` | the mode's dry or wet radius fed to `rdry`/`rwet` per `ncd_activ` [m] |
| `prho` | air density, `prhop1` [kg/m3] |
| `sigma` | the mode's geometric standard deviation (`sigmaln(kmod) = log(sigma)` is what the routine reads) |
| `rcritrad` | the mode's critical radius for this (phase, mode) [m] |
| `sfnuc` | `get_icscavfrac`'s nucleation scavenged fraction, `pfrac_nuc` [-] |
| `icscav_frac` | `ic_scav`'s own `pdxt_nuc/pxt`; equals `sfnuc` by construction (see above) |

Non-relevant modes (`kmod` not in {2,3,4}) are 0 by `ic_scav_nuc`'s own
early `RETURN` -- jcm never calls the nucleation math for them at all
(zeroed directly in `wetdep_term.py`'s per-mode loop instead), a different
code path reaching the same answer; the comparison test checks both sides
are exactly 0 for these rows rather than running them through the ported
functions.

## Regenerate

`python build_reference_icscavnuc.py <jcm>/jcm/data/test/echam_cloud_reference`
from this directory (`fortran_harness/echam_icscavnuc`), after rebuilding
and rerunning `driver.f90` (see that file's own header for the build
commands) at both `-O0` and `-O2`.
