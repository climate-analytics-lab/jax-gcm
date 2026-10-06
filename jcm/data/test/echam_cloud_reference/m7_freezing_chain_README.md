# M7 mixed-phase-freezing CHAIN reference (jax-gcm#1017 task 2 part 1)

Chains the UNMODIFIED ECHAM6.3-HAM2.3 r7492 `mo_ham_freezing.f90::ham_IN_setup` (via `hamfrz_M7.npz`'s `mixed_all` cell) to the UNMODIFIED `mo_cloud_micro_2m.f90::het_mxphase_freezing` (F 2675-2840), on ONE fully-populated M7 aerosol state, instead of each routine's existing reference data being fed its own independently-chosen inputs (`hamfrz_M7.npz`'s designed mass/number cells; `cloud2m_frz_T63L47.npz`'s hand-picked DUST/BC fraction dicts). Numbers only; no ECHAM source is part of this repository.

Integrity: -O0 vs -O2 max abs difference dt1200 0.0e+00, dt720 0.0e+00.

## Columns

| name | meaning |
|---|---|
| `frz_none` | control: no aerosol, nothing freezes heterogeneously |
| `frz_m7_mixed` | `columns_frz.py`'s `M7_MIXED`: hamfrz_M7.npz's `mixed_all` cell's ham_IN_setup output (fracdusol/fracbcsol/fracduai/fracduci/fracbcinsol, rwetki/rwetai/rwetci) fed to het_mxphase_freezing |

## Arrays

`in/<key>`: the HAM freezing inputs set per column (see `columns_frz.FRZ_KEYS`). `out/<step>/*`, `diag/<step>/*`: every INOUT/OUT argument and the freezing intermediates, as in `cloud2m_frz_README.md` -- same conventions (TOP-FIRST levels, `(nlev, ncol)`, `dt1200`/`dt720`).

## Regenerate

From the harness copy (`fortran_harness/echam_cloud2m/py`): `python build_m7_freezing_chain.py <jcm>/jcm/data/test/echam_cloud_reference`.
