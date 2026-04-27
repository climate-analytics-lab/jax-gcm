# Plan: JAX vs Fortran ECHAM/ICON physics comparison

Goal: find numerical bugs in the JAX physics ports by diffing tendencies
against the unmodified Fortran reference on the same input column.

## Where we are

- Branch: `fortran-harness-cumastr` (off `feature/composable-physics-206`).
- `fortran_harness/build/cumastr_driver` compiles and runs end-to-end.
- `test_rce_column.py` produces real Tiedtke-Nordeng tendencies on a
  tropical sounding (ktype=3, 33 active levels, peak ~6.86 J/kg/s heating).
- `compare_cumastr.py` has the netcdf → binary input plumbing, calls
  the driver, and parses outputs — but the JAX-side call is still a
  `NotImplementedError` stub.

## Phase 1 — wire up the JAX cumastr (today)

1. Implement `run_jax_cumastr_equivalent(state, dtime)` in
   `compare_cumastr.py`. Calls `tiedtke_nordeng_full(...)` from
   `jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng`. The JAX
   side wants a `PhysicsState`-like input and produces a `PhysicsTendency`
   plus diagnostics. Build a single-column PhysicsState from the dict
   we already have, run the term in eager mode (no JIT), pull
   tendencies + the convection-data sub-struct.
2. Map JAX outputs onto the Fortran field names:
   - `pq_cnv` ↔ JAX temperature tendency × cp (J/kg/s)
   - `pqte_cnv` ↔ JAX specific_humidity tendency
   - `pvom_cnv`, `pvol_cnv` ↔ JAX u, v tendencies
   - `pxtecl`, `pxteci` ↔ JAX qc, qi detrainment (in
     `physics_data.convection.qc_detrain` etc.)
   - `prsfc`, `pssfc` ↔ JAX rain, snow at surface
   - `ptop` ↔ JAX cloud-top pressure (need to dig into PhysicsData)
   - `ktype`, `kctop` ↔ JAX `convection.ktype`, `cloud_top_index`
3. Verify the comparison runs on the RCE sounding from
   `test_rce_column.py` (we know Fortran fires ktype=3 there).
4. **Diff every field**: print max-abs and mean-abs error per field, plus
   per-level deltas for the worst offender.

## Phase 2 — bisect Tiedtke-Nordeng (next session if needed)

If the totals don't match (likely), bisect by sub-routine. Tiedtke-Nordeng
is `cumastr → cuini → cubase → cuasc/cudescent → cuflx/cudtdq/cududv`.
The JAX port mirrors this:

| Stage | Fortran  | JAX module |
|---|---|---|
| cuini  | `mo_cuinitialize.f90::cuini`  | `tiedtke_nordeng.py::initialize` |
| cubase | `mo_cuinitialize.f90::cubase` | `updraft.py::find_cloud_base` |
| cuasc  | `mo_cuascent.f90::cuasc`      | `updraft.py::updraft_ascent` |
| cuadjtq | `mo_cuadjust.f90::cuadjtq`   | `updraft.py::saturation_adjustment` |
| cudlfs/cuddraf | `mo_cudescent.f90`    | `downdraft.py` |
| cuflx/cudtdq/cududv | `mo_cufluxdts.f90` | `flux_tendencies.py` |

Strategy: build smaller standalone Fortran drivers for the most
suspicious sub-routines. Already on the radar:

- **`cuasc`** — heaviest JAX touch (commits 1bed9fa, bec22b7, 363bdca,
  7dafd8d, cd7e9f7). Most likely place for a porting bug.
- **`cuadjtq`** — recently rewritten as Newton-Raphson in JAX; needs
  to match Fortran iterative solver bit-for-bit on simple inputs.
- **`cufluxdts`** — fix in cd7e9f7 fixed the deviation-flux + signed-dp
  bug; verify the JAX `calculate_tendencies` matches the Fortran
  `cudtdq` flux divergence.

Each sub-routine driver follows the same template as `cumastr_driver.f90`.
Reuse the support stubs (`mo_kind`, `mo_exception`, `mo_echam_*_config`,
`mo_echam_convect_tables`).

## Phase 3 — Sundqvist + ECHAM-1m microphysics

After convection is clean, the next high-suspicion area is the cloud
package (`mo_cloud.f90` for both Sundqvist and ECHAM-1m microphysics in
the ICON port). It's the destabiliser per the term-removal experiment
(removing convection didn't help, removing clouds dodged the cascade).

ECHAM `mo_cloud.f90` is one big module covering both stages, so the
driver shape is similar to cumastr but simpler dependencies. JAX side:
`jcm.physics.clouds.sundqvist.py` + `jcm.physics.clouds.echam_1m.py`.

## Phase 4 — TTE-TKE vertical diffusion

Lower priority — vdiff was a no-op in the term-removal experiment
(removing it gave +1 day before the day-44 cascade). But the JAX
port has had repeated TKE-runaway issues (commits b6172a7, 8250601),
so worth a sanity check.

Fortran: `vdiff.f90` (ECHAM5 convention) or the
`atm_phy_echam` equivalent if there is one. JAX:
`jcm.physics.vertical_diffusion.tte_tke.*`.

## Open questions / things to remember

- **Sigma TOA**: don't use `sigma=0` at the top — `compare_cumastr.py`
  clamps to `0.01` (1 hPa) so the dry-static lift in cubase doesn't
  push T past the 400 K lookup-table bound. Real ICON hybrid coords
  start at `a=0` but `b=0` too, so the top pressure is always > 0
  in production.
- **Unit gotcha**: dataset stores `specific_humidity` in g/kg
  (ICON-physics boundary rescaling, commit 08e3fc5). Internal
  state is kg/kg. Fortran cumastr expects kg/kg.
- **Geopotential sign**: `dgeo = R_d · T_v · d(ln p)` (positive going
  surface-ward); my first cut used `-dpln` and ECHAM blew up
  immediately because the dry-static lift inverted.
- **Diagnostic prints**: I added one in
  `mo_echam_convect_tables.f90::lookup_ua_list_spline` that fires on
  out-of-bounds T. Useful when something goes wrong; keep.
- **ktrac=0**: the driver compiles tracer arrays with `MAX(ktrac,1)`
  size for safety. Fine for cumastr unit-testing.
- **JAX tendencies → Fortran convention**: cumastr returns `pq_cnv` in
  J/kg/s (i.e. `cp * dT/dt`). JAX returns `dT/dt` in K/s. Divide one
  by `cp` (1004.64) before diffing.

## How to resume in a fresh session

1. `cd /data/dwatsonparris/jax-gcm`
2. `git checkout fortran-harness-cumastr`
3. `cd fortran_harness && make`  → confirm build still clean
4. `python test_rce_column.py`   → confirm Fortran fires ktype=3
5. Open `compare_cumastr.py::run_jax_cumastr_equivalent` and start
   Phase 1.

## Done so far

- Standalone build of `cumastr` + 6 dependent modules.
- Stubs for `mo_exception`, `mo_echam_cnv_config`, `mo_echam_cld_config`.
- Python harness reads/writes Fortran-unformatted records.
- Driver outputs ktype, kctop, pq_cnv, pqte_cnv, pvom_cnv, pvol_cnv,
  pxtecl, pxteci, prsfc, pssfc, ptop, pcon_dtrl, pcon_dtri, pcon_iqte.
- Smoke test (RCE) confirms convection fires.
- JAX side wired up in `compare_cumastr.py::run_jax_cumastr_equivalent`
  with the JAX `tiedtke_nordeng_convection`. Config knobs pinned to
  Fortran ECHAM6.3 ``__ICON__`` defaults so any diff is a port bug.

## Port bugs found by the harness (TODO: fix)

Phase 1 already turned up three bugs in `tiedtke_nordeng.py` itself
(the wrapper around the inner cumastr equivalents in updraft.py /
flux_tendencies.py — not the inner physics). Listed in increasing
order of complexity to fix:

1. **`state.ktop` reports the scan ceiling, not the actual cloud top**
   (`tiedtke_nordeng.py:664`). The diagnostic `state.ktop` is the
   `kbase - cloud_depth` value from the initial placeholder, but the
   updraft mass flux actually terminates wherever the dynamic
   termination criterion fires. Should derive from `mfu > 0` extent.
   Trivial fix.

2. **`ktype=3` (mid-level convection) is missing entirely** from the
   trigger logic (`tiedtke_nordeng.py:528-532`). The conv_type ternary
   only returns 0/1/2 (none / deep / shallow). ECHAM's mid-level
   branch (`mo_cumastr.f90:754`, `zentr=entrscv` when ktype=2 etc.)
   needs a Python equivalent. Look at ECHAM `mo_cumastr.f90` lines
   700-770 for the trigger.

3. **`convective_adjustment` runs on the *whole* column** instead of
   only the convective levels (`tiedtke_nordeng.py:632-635`). Result:
   on a moist column the post-conv saturation-adjustment fires at
   every level where (after the tendencies are added) RH > 100, even
   though those levels are nowhere near the cloud. Mask the tendency
   inputs to convective_adjustment by `conv_mask` (or limit the call
   to the (kbase, ktop) range).

After these are fixed, re-run `compare_cumastr.py --rce` and see what
discrepancies remain. The actual updraft / saturation-adjustment / flux
divergence (in updraft.py and flux_tendencies.py) are likely closer to
correct, but we won't know until the wrapper bugs are out of the way.
