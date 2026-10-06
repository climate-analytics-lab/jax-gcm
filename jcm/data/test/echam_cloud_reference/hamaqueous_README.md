# ECHAM6.3-HAM2.3 aqueous-sulfur-chemistry reference data (jax-gcm#1017)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine
`mo_ham_chemistry.f90::ham_wet_chemistry` (HAM_M7 branch) on designed
single-level M7 slices, compiled standalone in double precision with
declaration-only stubs of the modules it USEs. They are the intended
numerical reference for `jcm/physics/aerosol/jam/chemistry/aqueous.py`'s
HAM path (`spec.aqueous_sulfate_modes` set), and
`aqueous_hamaqueous_reference_test.py` compares the full M7 path against
every array below at float64 rtol=1e-12. **Data only**: no ECHAM or HAM
source is part of this repository.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`hamaqueous_provenance.json`).

## Fixed: SO2 Henry's law constant mismatch (jax-gcm#1017 task 3; jax-gcm#1031)

jcm's `_aqueous_so4` (`jcm/physics/aerosol/jam/chemistry/aqueous.py`) originally
hardcoded `_H_SO2_0, _H_SO2_ACT = 1.23, 3020.0`, which disagreed with this
reference's compiled `speclist(id_so2)%henry = (1.36, 4250.0)`
(`mo_ham_species.f90`'s SO2 registration, tagged `!csld(#275)` -- a later
correction jcm's port predates) -- the dominant driver of a **-9% to -52%**
relative disagreement in the produced-sulfate rate across the 16 cells
below. Three smaller literal-rounding differences (`zrgas`, `avo`, the
`xtoc`/`ctox` `avo_xtoc` literal) and SO2's molar mass (`mw_so2`) made up
the <2.3% residual once the Henry pair alone was patched in a scratch copy.

`_aqueous_so4` is **shared with the MAM4 default path**, so per the house
rule none of these five literals could simply be changed (that would move
MAM4's calibrated behaviour) -- reported to the lead as a STOP, which
confirmed the Fortran line numbers and filed the MAM4-shared defect as
jax-gcm#1031. Fix (this PR): every one of the five is now an optional
`AqueousConstants` override (`aqueous_constants.py`) that `_aqueous_so4`
reads only when given one; the default (`None`, every MAM4 population)
reproduces today's values exactly (bitid-verified bit-identical), and
`M7_SPEC` sets `aqueous_constants=HAM_AQUEOUS_CONSTANTS` -- r7492's own
numbers. `aqueous_hamaqueous_reference_test.py` runs the full M7
`AqueousSulfur` term (not just the bare kernel) against every array below
and passes at float64 rtol=1e-12 with no loosening.

## Known gap: H2O2 remaining is NOT recorded (benign)

`ham_wet_chemistry` depletes H2O2 in a purely LOCAL variable (`zh2o2m`)
across its 5 sub-steps and discards it at the end of the loop -- it is
never written to an `INTENT(out)`/`INTENT(inout)` argument or a stream,
so there is no way to recover it from the compiled routine's own
interface without editing the Fortran, which the house rule for this
harness forbids. This is fine: jcm's `AqueousSulfur` likewise discards its
own depleted H2O2 at the end of each step and resets it from the
prescribed oxidant field on the next one, matching HAM's own offline-oxidant
treatment of H2O2 (there is no coupled prognostic H2O2 budget on either
side, so nothing is lost by not carrying it across steps). The produced
sulfate, the SO2 consumed and the new CS number (all derived from `pxtte`,
which IS exposed) are recorded; H2O2-remaining comparisons are out of scope
for this reference.

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
