# ECHAM6.3-HAM2.3 aqueous-sulfur-chemistry reference data (jax-gcm#1031)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine
`mo_ham_chemistry.f90::ham_wet_chemistry` (HAM_M7 branch) on designed
single-level M7 slices, compiled standalone in double precision with
declaration-only stubs of the modules it USEs. They are the intended
numerical reference for `jcm/physics/aerosol/jam/chemistry/aqueous.py`'s
`_aqueous_so4` kernel, and `aqueous_hamaqueous_reference_test.py` compares
it against every array below at float64 rtol=1e-12. **Data only**: no
ECHAM or HAM source is part of this repository.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`hamaqueous_provenance.json`).

## Fixed: six HAM literals (jax-gcm#1031)

`_aqueous_so4` hardcoded six constants that differ from r7492's own
compiled values: the SO2 Henry's-law pair (`_H_SO2_0, _H_SO2_ACT = 1.23,
3020.0`, predating a correction HAM made — `speclist(id_so2)%henry =
(1.36, 4250.0)`, `mo_ham_species.f90:181`), the gas constant `zrgas`
(`mo_ham_chemistry.f90:209`'s rounded `8.2e-2` against a value derived
from jcm's own R*), Avogadro's number and its separately-rounded
`xtoc`/`ctox` literal (`mo_physical_constants.f90:58`), and SO2's molar
mass (`mo_ham.f90:310`). The Henry pair alone drives a **-9% to -52%**
relative disagreement in the produced-sulfate rate across the 16 cells
below; the other four literals account for the remaining <2.3% once the
Henry pair alone is patched.

Fix (jax-gcm#1031, maintainer decision 2026-10-06): all five become
`_aqueous_so4`'s own module constants at r7492's values — this
**intentionally changes the `echam-jam`/MAM4 default path**; see the PR
for the measured change in SO4 production and burden. SO4's own molar
mass (`mw_so4`, defaulting to jcm's MAM4-MOM value, 115 g/mol — jcm's own
species choice, not a HAM literal) stays a function parameter, so this
fixture's M7 value (96.0631 g/mol) can still be supplied explicitly
without wiring up an M7 population on this branch.
`aqueous_hamaqueous_reference_test.py` calls the bare `_aqueous_so4`
kernel directly (`mw_so4=96.0631`), deriving the kernel's expected
in-cloud sulfate production from the recorded grid-mean tendencies
(`(pxtte_ms4as + pxtte_ms4cs)·dt/paclc` — the two modes' production sums
to the kernel's own undivided output, since HAM's number-fraction split
only redistributes it), and passes at float64 rtol=1e-12 with no
loosening.

## Known gap: H2O2 remaining is NOT recorded (benign)

`ham_wet_chemistry` depletes H2O2 in a purely LOCAL variable (`zh2o2m`)
across its 5 sub-steps and discards it at the end of the loop -- it is
never written to an `INTENT(out)`/`INTENT(inout)` argument or a stream,
so there is no way to recover it from the compiled routine's own
interface without editing the Fortran, which the house rule for this
harness forbids. This is fine: jcm's `_aqueous_so4`/`AqueousSulfur`
likewise discards its own depleted H2O2 at the end of each call and
resets it from the prescribed oxidant field on the next step, matching
HAM's own offline-oxidant treatment of H2O2 (there is no coupled
prognostic H2O2 budget on either side, so nothing is lost by not carrying
it across steps). The produced sulfate, the SO2 consumed and the new CS
number (all derived from `pxtte`, which IS exposed) are recorded;
H2O2-remaining comparisons are out of scope for this reference.

## Arrays (one entry per cell, `meta/names` order)

| name | meaning |
|---|---|
| `in/so2`, `in/ms4as`, `in/ms4cs` | tracer mass mixing ratio pxtm1 [kg/kg] (SO2 gas; SO4 soluble accumulation/coarse) |
| `in/nas`, `in/ncs` | mode number pxtm1 [1/kg] (soluble accumulation/coarse) |
| `in/pmlwc`, `in/paclc` | in-cloud liquid water [kg/kg]; cloud fraction [-] |
| `in/tm1`, `in/app1`, `in/rhop1` | temperature [K]; pressure [Pa]; air density [kg/m3] |
| `in/o3_density`, `in/h2o2_density` | target O3/H2O2 number density [molec/cm3] (what jcm's oxidants diagnostic supplies directly, and what `_aqueous_so4`'s own `o3`/`h2o2` arguments want) |
| `in/o3_mmr`, `in/h2o2_mmr` | the mass mixing ratios [kg/kg] that reproduce those densities through ham_wet_chemistry's own `zc` conversion -- what the harness feeds `bc_apply` |
| `in/h2o2_mw`, `in/so2_henry` | mo_ham_species.f90 H2O2 molar mass [g/mol]; SO2 Henry's law (H0 [mol/l/atm], activation [K]) |
| `in/time_step_len` | the time step [s] (`_aqueous_so4`'s own `dt`) |
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

The mode-split cells (9-13) are recorded for completeness (the fixture was
originally built to validate the full `AqueousSulfur` mode-split path) but
are not distinguishing for the bare-kernel comparison here: the kernel's
own output is the SAME undivided in-cloud production regardless of how
HAM's number-fraction split later distributes it across AS/CS, so
`(pxtte_ms4as + pxtte_ms4cs)` is identical across cells 9-13 — the kernel
test checks this sum, not the individual mode split.

## Regenerate

`python build_reference_hamaqueous.py <jcm>/jcm/data/test/echam_cloud_reference`
from the harness copy `fortran_harness/echam_hamaqueous/py`.
