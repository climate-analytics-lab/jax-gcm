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

## Port bugs found by the harness

### Wrapper bugs — FIXED in commit a299118

1. **`state.ktop` reports the scan ceiling, not the actual cloud top**
   (`tiedtke_nordeng.py:664`). Was using `kbase - cloud_depth` from
   the initial placeholder; now derives from where the updraft mass
   flux actually extends.

2. **`convective_adjustment` ran on the *whole* column** instead of
   the cloud levels. On a moist column the post-conv saturation
   adjustment fired at every level whose initial RH exceeded the JAX
   qsat cutoff, producing spurious heating ~600× larger than the
   actual flux divergence (229 K/day across levels 9-45 vs the
   0.4 K/day the divergence delivers). Now masked to (kbase, ktop).

3. **`calculate_tendencies` divided dtedt/dqdt by dt at the end**
   (`flux_tendencies.py:282-285`), making convective heating ~1500×
   too small for dt=1800 s. The divergence math gives K/s already.

### Remaining algorithm-level disagreements

After the wrapper fixes, the harness still shows substantial
divergence between JAX and Fortran on the same RCE column:

| Field | Fortran | JAX (post-wrapper-fix) |
|---|---|---|
| ktype | 3 (mid) | 1 (deep) |
| kctop | 17 (~377 hPa) | 36 (~765 hPa) |
| pq_cnv peak | 6.86 J/kg/s **at level 16 (top)** | 8.19 J/kg/s **at level 45 (base)** |

These are inner-physics disagreements:

A. **`ktype=3` (mid-level convection) is missing** from the trigger
   logic (`tiedtke_nordeng.py:528-532`). The conv_type ternary only
   returns 0/1/2. ECHAM's mid-level branch (`mo_cumastr.f90:754`,
   `zentr=entrscv` when ktype=2 etc., plus the `kctype=3`
   classification when CAPE is moderate and the trigger is in the
   free troposphere) needs a Python equivalent.

B. **The JAX updraft terminates ~18 levels too early.** Fortran
   reaches level 17 (~377 hPa); JAX terminates at level 35 (~765 hPa).
   Both use the same dynamic termination criterion (negative
   buoyancy or mfu < 1 % of base). Either:
   - the JAX buoyancy calculation differs from Fortran's
     (different latent-heat sign? different env-temperature?), or
   - the JAX entrainment is too aggressive (ECHAM's `entrpen=1e-4` is
     matched, but ICON port has organized entrainment that may have
     a different scale).

C. **The heating profile is inverted.** Fortran peaks at the cloud
   top (level 16, ~398 hPa); JAX peaks at the cloud base (level 45,
   ~981 hPa). Suspect the deviation-flux formulation in
   `flux_tendencies.py` (`(s_par − s̄)·mfu`) is producing heating
   where mfu is biggest (base) instead of where the detrainment is
   biggest (top). ECHAM uses the full DSE flux `pmfus = mfu·s_par`
   plus explicit `alv*(plude + pdmfup)` detrainment terms, which
   concentrate heating at cloud top. Recommended:
   1. Implement ECHAM's full-flux + explicit detrainment formula
      directly (mirror `mo_cufluxdts.f90:298-310`).
   2. The deviation-flux derivation in commit cd7e9f7 may have
      missed the latent-heat-of-detrainment contribution.

### Outcome so far

The harness has paid for itself: three wrapper bugs fixed, three
inner-algorithm disagreements isolated, no false alarms.

Next: implement the ECHAM-style heating formula (issue C) — it's the
biggest single contributor and we have the Fortran reference right
there in `fortran_harness/src/mo_cufluxdts.f90`.
