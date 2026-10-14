# nic_cirrus=2 end-to-end reference (jax-gcm#1017 task 2 part 2b)

Runs the UNMODIFIED ECHAM6.3-HAM2.3 r7492 `mo_cloud_micro_2m.f90::cloud_micro_interface` WITH the UNMODIFIED `mo_cirrus.f90::xfrzmstr` linked in (`nic_cirrus=2`), feeding `pascs`/`papnx`/`paprx`/`papsigx` through the same `cloud_subm_1` interface `ham_IN_setup` uses. Verifies the TWO equivalence claims jcm's `cloud_microphysics_2m` wiring rests on: (a) jcm's `icnc_melt` at the `xfrzmstr` call site is ECHAM's `zicncq` there; (b) the step-start ice supersaturation is what ECHAM passes, not section 5's adjusted humidity. Numbers only; no ECHAM source is part of this repository.

Integrity: -O0 vs -O2 max abs difference, outputs 0.0e+00, diagnostics 0.0e+00.

## Columns

| name | description |
|---|---|
| `pure_cirrus_supersaturated` | 230 hPa, 210 K, RH_ice 1.8 (S_ice-1=0.8, above SCRHOM(210)~=0.563, the homogeneous threshold at this T -- see cirrus.py's _scrhom), moderate updraft (TKE 0.3), pascs 1e8 kg-1, no pre-existing ice: the base case this task's claim (b) needs (step-start supersaturation drives XFRZMSTR, not section 5's). |
| `pure_cirrus_with_preexisting_ice` | Same level/supersaturation/aerosol as pure_cirrus_supersaturated, but with pre-existing ice (pxim1 5e-6, ICNC 5e7 m-3) already present: claim (a) needs this -- the depletion zapnx=max(1e-6*(papnx-icnc),1e-6) must see THIS column's own zicncq, not zero, for the two columns to differ as jcm predicts. |
| `cirrus_low_updraft` | Same level/supersaturation/aerosol, TKE near 0 (updraft ~0): COOLR is small, so TAU is large and the growth/relaxation chain (xfrzhom's X root search) runs in a different regime. |
| `cirrus_high_updraft` | Same level/supersaturation/aerosol, TKE 3 m2/s2 (strong updraft): COOLR large, TAU small. |
| `cirrus_no_aerosol_floor_control` | Same supersaturated/updraft cell with NO aerosol (pascs=papnx=0): the nic_cirrus=2 candidate caps at 0 regardless of XFRZMSTR's own output, pinning ICNC at the icemin floor -- the #552 regression this whole task guards, as a control against the non-zero cells above. |

## Arrays

`in/*`: the driver-set cirrus/temperature/humidity fields per designed level. `out/*`: every INOUT/OUT argument of `cloud_micro_interface`. `diag/*`: the output-only diagnostic patch's full set (section-1 `icncq`/`ninucl`/`s1_picnc`, `update_in_cloud_water`'s `uicw_icnc`/`icnc_cand`, the final `icnc_7`) -- same conventions as `cloud2m_README.md` (TOP-FIRST levels, `(nlev, ncol)`).

## Regenerate

From the harness copy (`fortran_harness/echam_cloud2m/py`, this private scratch copy): `python build_cirrus2m_reference.py <jcm>/jcm/data/test/echam_cloud_reference`.
