# ECHAM6.3-HAM2.3 aqueous-sulfur-chemistry reference data (jax-gcm#1017)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine
`mo_ham_chemistry.f90::ham_wet_chemistry` (HAM_M7 branch) on designed
single-level M7 slices, compiled standalone in double precision with
declaration-only stubs of the modules it USEs. They are the intended
numerical reference for `jcm/physics/aerosol/jam/chemistry/aqueous.py`'s
HAM path (`spec.aqueous_sulfate_modes` set) -- but see "Known gap: SO2
Henry's law constant" below, which blocks the 1e-12 comparison test
this data was built for. **Data only**: no ECHAM or HAM source is part
of this repository.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`hamaqueous_provenance.json`).

## Known gap: SO2 Henry's law constant mismatch (jax-gcm#1017 task 3, STOP)

jcm's `_aqueous_so4` (`jcm/physics/aerosol/jam/chemistry/aqueous.py`,
`_H_SO2_0, _H_SO2_ACT = 1.23, 3020.0`) disagrees with this reference's
compiled `speclist(id_so2)%henry = (1.36, 4250.0)`
(`mo_ham_species.f90`'s SO2 registration, tagged `!csld(#275)` -- a later
correction jcm's port predates). This is the dominant driver of a
**-9% to -52%** relative disagreement in the produced-sulfate rate across
the 16 cells below (measured against `out/pxtte_ms4as + out/pxtte_ms4cs`,
grid-mean-weighted by `in/paclc` to match `pxtte`'s own convention) --
worse at low temperature, consistent with an activation-energy mismatch.
Patching `_H_SO2_0, _H_SO2_ACT` to `(1.36, 4250.0)` in a scratch copy drops
the disagreement to <2.3% (the residual from smaller literal-rounding
differences: HAM's `avo` 6.022e20 vs jcm's precise 6.02214179e20, HAM's
`zrgas` 0.082 vs jcm's `r_universal/101.325` ~0.082057, `zmolgair` 28.84).

`_aqueous_so4` is **shared with the MAM4 default path**, so per the house
rule this reference does NOT change it (that would move MAM4's calibrated
behaviour) -- reported to the lead instead. The
`aqueous_hamaqueous_reference_test.py` comparison test this data was
built for is therefore **not yet added**: it would either fail at 1e-12 as
measured above, or need a tolerance that doesn't test anything meaningful.
Options for the lead: (a) parametrize `_H_SO2_0`/`_H_SO2_ACT` the same way
task 1 parametrized `mw_so4`/`conv_so2_so4` (new optional args defaulting
to today's MAM4 values, M7's `AqueousSulfur` passing the HAM-correct ones),
then add the comparison test; (b) accept the MAM4-shared constant as a
known approximation and track it as a separate issue; (c) something else.

## Known gap: H2O2 remaining is NOT recorded

`ham_wet_chemistry` depletes H2O2 in a purely LOCAL variable (`zh2o2m`)
across its 5 sub-steps and discards it at the end of the loop -- it is
never written to an `INTENT(out)`/`INTENT(inout)` argument or a stream,
so there is no way to recover it from the compiled routine's own
interface without editing the Fortran, which the house rule for this
harness forbids. The produced sulfate, the SO2 consumed and the new CS
number (all derived from `pxtte`, which IS exposed) are recorded; H2O2-
remaining comparisons are out of scope for this reference.

## Arrays (one entry per cell, `meta/names` order)

| name | meaning |
|---|---|
| `in/so2`, `in/ms4as`, `in/ms4cs` | tracer mass mixing ratio pxtm1 [kg/kg] (SO2 gas; SO4 soluble accumulation/coarse) |
| `in/nas`, `in/ncs` | mode number pxtm1 [1/kg] (soluble accumulation/coarse) |
| `in/pmlwc`, `in/paclc` | in-cloud liquid water [kg/kg]; cloud fraction [-] |
| `in/tm1`, `in/app1`, `in/rhop1` | temperature [K]; pressure [Pa]; air density [kg/m3] |
| `in/o3_density`, `in/h2o2_density` | target O3/H2O2 number density [molec/cm3] (what jcm's oxidants diagnostic supplies directly) |
| `in/o3_mmr`, `in/h2o2_mmr` | the mass mixing ratios [kg/kg] that reproduce those densities through ham_wet_chemistry's own `zc` conversion -- what the harness feeds `bc_apply` |
| `in/h2o2_mw`, `in/so2_henry` | mo_ham_species.f90 H2O2 molar mass [g/mol]; SO2 Henry's law (H0 [mol/l/atm], activation [K]) |
| `out/pxtte_so2`, `out/pxtte_ms4as`, `out/pxtte_ms4cs` | post-call tendency [kg/kg/s] (SO2 consumed is `-pxtte_so2`; produced sulfate per mode is `pxtte_ms4as`/`pxtte_ms4cs`) |
| `out/pxtte_nas`, `out/pxtte_ncs` | post-call number tendency [1/kg/s] (the new CS number in the empty-fallback cells) |
| `out/pxtte_lfrac` | SO2 liquid-fraction diagnostic `plfrac` |

## Cells

| # | name | description |
|---|---|---|
| 0 | `no_cloud` | paclc=0, pmlwc=0: the LWC gate (zlwcmin, line 249) is never cleared, so no sulfate forms |
| 1 | `thin_cloud` | small in-cloud fraction and liquid water |
| 2 | `thick_cloud` | large in-cloud fraction and liquid water |
| 3 | `low_so2` | SO2 scarce relative to oxidants |
| 4 | `high_so2` | SO2 abundant relative to oxidants |
| 5 | `low_h2o2` | H2O2-limited: the SO2+H2O2 channel starved |
| 6 | `high_h2o2` | H2O2 in excess: the SO2+H2O2 channel saturated |
| 7 | `low_o3` | O3-limited: the SO2+O3 channel starved |
| 8 | `high_o3` | O3 in excess: the SO2+O3 channel saturated |
| 9 | `as_below_cs_below` | both AS/CS number < 1e-3 kg-1 but nonzero: HAM's 'neither present' branch (lines 470-477) -- all production to CS mass, new CS number from mass |
| 10 | `as_above_cs_above` | both AS/CS number >= 1e-3 kg-1: split by number fraction (lines 443-449) |
| 11 | `as_above_cs_below` | AS present, CS below threshold: all production to AS (lines 453-455) |
| 12 | `as_below_cs_above` | AS below threshold, CS present: all production to CS (lines 459-461) |
| 13 | `both_empty` | AS/CS number EXACTLY zero: the same 'neither present' branch as as_below_cs_below, at the other end of the threshold |
| 14 | `warm` | T=298 K |
| 15 | `cold` | T=250 K |

## Regenerate

`python build_reference_hamaqueous.py <jcm>/jcm/data/test/echam_cloud_reference`
from the harness copy `fortran_harness/echam_hamaqueous/py`.
