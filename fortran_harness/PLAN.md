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

ECHAM `mo_cloud.f90` is one big module (~1200 lines) covering cloud
cover diagnosis + condensation + microphysics + sedimentation in a
single SUBROUTINE. Dependencies include `mo_cloud_utils.f90` (helpers),
`mo_echam_cld_config.f90` (already stubbed for cumastr harness), and
`mo_echam_convect_tables.f90` (already pulled in). The JAX side has it
split into ``jcm.physics.clouds.sundqvist.py`` (cloud cover +
condensation, ~400 lines) and ``jcm.physics.clouds.echam_1m.py``
(microphysics, ~700 lines).

**Status (2026-04-28)**: harness scaffold built. ``cloud_driver`` compiles
clean; ``compare_cloud.py`` runs end-to-end on a tropical RCE column,
producing diff tables. Next session can immediately start iterating on
bugs.

**Bug E — design difference, not a bug per se**: JAX
``shallow_cloud_scheme`` diagnoses cloud cover internally via
``calculate_cloud_fraction(RH, RHc)``. ECHAM ``cloud()`` expects
``paclc`` as INPUT (cloud cover is set in a separate ``mo_cover.f90``
upstream). On a 90 % RH column with no initial cloud water:
  - Fortran ``paclc`` stays 0 (no condensation triggered)
  - JAX ``paclc`` jumps to 0.42 in the moist mid-troposphere

Fix options for fair comparison:
  (a) pull in ``mo_cover.f90`` upstream of cloud_driver (need to find
      and stub its dependencies — likely simpler than mo_cloud)
  (b) run JAX with cover-diagnosis disabled, comparing only
      condensation/microphysics
  (c) feed both Fortran and JAX a column with non-zero pxlm1/pxim1
      (e.g. post-convection state)

Recommended: (c) for the next iteration — the most physically
meaningful test case is "what happens when convection has already
detrained cloud water and then the cloud module fires?" That's the
realistic flow in the production model.

**With seeded cloud water (`compare_cloud.py --rce --with-cloud-water`)**:

The harness fires meaningful tendencies in both Fortran and JAX. Initial
findings:

| Field | Fortran | JAX |
|---|---|---|
| pq_cld peak (W/m²)    | -83.7 (cooling) | -45.2 (cooling) |
| pq_cld column sum     | -171.9          | -106.3          |
| pqte_cld peak         | +1.4e-7         | +8.6e-8         |
| pxlte_cld peak (drop) | -4.7e-9         | **-8.6e-8** (~20x larger) |
| pxite_cld peak (drop) | **-3.5e-8** (~3x larger) | -1.0e-8 |
| paclc peak            | 0.50 (preset)   | 0.42 (diagnosed) |

Discrepancies for next-session investigation:
- **Cloud-water removal too fast in JAX** (20x): suggests
  ``echam_1m`` autoconversion (Khairoutdinov-Kogan 2000) or accretion
  rate is too aggressive vs ECHAM's Beheng (1994).
- **Cloud-ice sedimentation too slow in JAX** (3x): suggests
  ``echam_1m`` ice fall speed (Heymsfield-Donner) is underestimated.
- Heating sign agrees (sublimation-driven cooling at the ice levels)
  but magnitude differs by ~2x.

These are tractable next-pass fixes once the harness is in shape.

### Beheng autoconversion: full formula, simplified bundling

The `autoconversion_kk2000` function in `jcm/physics/clouds/echam_1m.py`
(name retained for backwards-compat; body is now Beheng) faithfully
ports the ECHAM rate formula and implicit Bernoulli integration from
`mo_cloud.f90:841-863`:

- Rate prefactor `ccraut * 1.2e27 / rho` ✓
- `Nc^-3.3 * ρ^4.7 * qc^3.7` exponents ✓
- `(1 + rate*dt*3.7*qc^3.7)^(-1/3.7)` integration ✓
- Default `ccraut = 15.0` matches ECHAM ✓

What's NOT ported is ECHAM's bundled "autoconversion + accretion-by-rain
+ accretion-by-snow" combined update (`mo_cloud.f90:858-872`):

```fortran
ztmp1 = exp(-ccracl*zxrp1*dt)              ! accretion by existing rain
ztmp2 = exp(-ccracl*zauloc*rho*zraut*dt)   ! by autoconv-generated rain
zrac1 = zxlb*(1-ztmp1)
zrac2 = (zxlb-zrac1)*(1-ztmp2)
zxlb  = zxlb - zraut - zrac1 - zrac2       ! single combined update
```

The JAX side has `autoconversion_kk2000` and `accretion_rain_cloud` as
separate piecewise functions. Each is correct in isolation but they
don't share state with each other within the timestep. The `zauloc`
factor — the in-cloud rain reservoir generated by *this layer's*
autoconversion that's available for *immediate accretion of the
remaining cloud water in the same layer* — has no JAX equivalent.

This is the most likely source of any residual cloud-water tendency
discrepancy after the autoconversion-rate fix. Tracked as task #50,
"Bundle auto+accretion in echam_1m". A clean follow-up: fold
`accretion_rain_cloud` into `autoconversion_kk2000` using the nested-
exponential combined form, and add a column scan to track
`zauloc` (cloud-top-to-current-level rain accumulator).

## Phase 4 — TTE-TKE vertical diffusion

Lower priority — vdiff was a no-op in the term-removal experiment
(removing it gave +1 day before the day-44 cascade). But the JAX
port has had repeated TKE-runaway issues (commits b6172a7, 8250601),
so worth a sanity check.

Fortran: `~/atm_phy_echam/mo_vdiff_solver.f90` (939 lines) +
`mo_vdiff_downward_sweep.f90` (321) + `mo_vdiff_upward_sweep.f90`
(148) + `mo_echam_vdiff_params.f90`. JAX:
`jcm.physics.vertical_diffusion.tte_tke/` (~1700 lines across
``vertical_diffusion.py``, ``turbulence_coefficients.py``,
``matrix_solver.py``, ``tke_budget.py``).

**Status**: deferred — multi-session effort.

Fortran side ~1400 lines across 4 files plus deps (need to stub
``mo_echam_vdiff_params``, possibly ``mo_thermodyn``,
``mo_surface_tools``). JAX side splits across coefficient calc, TKE
budget, and the implicit tridiagonal solver — three separate
comparison points. TKE-runaway is path-dependent (integrates over
time), so single-column single-step harness only catches the
coefficient/solver pieces; multi-step is needed for runaway.

**Quick-win observation (not yet harness-validated)**:
``compute_exchange_coefficients`` clips ``K_m, K_h, K_q`` to
``[0.1, 1000.0] m²/s``. The ``0.1`` floor is generous vs. ECHAM's
``cmin_kh = 0.01 m²/s`` background. Worth tightening to 0.01 in a
quick test simulation before building the full harness.

## Phase 5 — Surface fluxes (ocean / sea-ice / land)

The 1-year T85×47 aquaplanet run from the harness branch (commit
493d480) failed at day 280-285 with bottom-level air-T drifting to
188 K over a prescribed 273 K SST. Surface-flux audit found six
issues (F1-F6); all fixed.

**Outcome (commit b409b1f and earlier)**: full 1-year T85×47 ICON
aquaplanet runs cleanly for 365 days from a 1 g/kg humidity
perturbation. Final state has tropical surface air at ~280 K under
~300 K SST, polar surface air at ~260 K, q max 7.3 g/kg, u max
95 m/s. No NaN, no climate drift, surface flux coupling holds
near-surface air close to the prescribed SST. Compare to the prior
"10 days stable" baseline before this branch — >36× improvement.

Per-bug detail follows.

### Bug F1 — FIXED (commit 4332fec): Businger-Dyer sign error

``compute_surface_exchange_coefficients`` had unstable-branch
multiplier as Φh = (1−16Ri)^(−0.5), which suppresses CH for
unstable conditions instead of enhancing it. Should be 1/Φh =
(1−16Ri)^(+0.5). At polar drift conditions (Ri ≈ −3.6) the
multiplier swings from 0.13 (suppressed) to 7.66 (enhanced) —
60x change in surface coupling.

### Bug F2 — TODO: vdiff uses hardcoded all-land surface_fraction

``apply_vertical_diffusion`` (icon_physics.py:747-748) sets
``surface_fraction = [0, 0, 1]`` (all land) regardless of the actual
boundary tile fractions. ``apply_surface`` uses the real fractions
(L:880-882). The vdiff field is dead in current code (not consumed
downstream — verified by grep), but this is confusing dead code that
will cause subtle bugs when the field IS consumed. Either remove or
wire up properly.

### Bug F3 — TODO: per-tile surface temperature broadcast from one

``apply_vertical_diffusion`` does ``surface_temperature =
jnp.repeat(surface_temp_col, nsfc_type, axis=1)`` — broadcasts one
SST across all 3 tiles. ECHAM has separate ``ptsfc_tile`` per tile
type, so cold sea ice and warm open water get different fluxes. With
prescribed SST and no live ice tile, this happens to be OK in our
aquaplanet, but breaks when sea ice is present.

### Bug F4 — TODO: Mixed-layer ocean is dead code

``ocean_surface_temperature_step`` computes a temperature tendency
from the surface energy balance (mixed-layer model with SW
penetration, etc.). But ``apply_surface`` (icon_physics.py:884)
RESETS ``ocean_temp = surface_temp`` from the boundary at every
physics step. So the mixed-layer evolution is never persisted. For
prescribed-SST aquaplanet this is intentional behaviour, but the
code looks like it's meant to be interactive. Either delete the
mixed-layer code or wire it through ``physics_data`` properly.

### Bug F5 — TODO: No frazil ice formation from supercooled ocean

``ice_thickness_evolution`` (sea_ice.py:209-296) only grows ice
from existing ice via bottom conduction. It cannot create new ice
when an open-ocean cell drops below the freezing point. ECHAM uses
``ctfreez = 271.38 K`` (saline-water freezing point) and creates
new ice when SST < ctfreez. Required for any non-aquaplanet run with
seasonal sea ice.

### Bug F6 — TODO: Default forcing.stl_am = 0 K

``ForcingData.zeros`` defaults ``stl_am`` (land temperature) to a
zero array. If a run is built with ``ForcingData.zeros(...)`` and
land actually exists in the terrain (fmask > 0), the apply_forcing
path sets surface_temperature = 0 K over land. Defensive: should
default to 273.15 K or to the global mean SST.

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

The harness has paid for itself: three wrapper bugs fixed, four
inner-algorithm bugs found (one of which was new — runaway downdraft),
no false alarms.

### Bug fixes from second harness pass

**Bug D — runaway downdraft mass flux (FIXED, commit ed50547)**

`downdraft.py::downdraft_step` used `entrscv * 0.5` (1.5e-3 / m) as
the entrainment rate instead of ECHAM's `entrdd` (2e-4 / m), and only
entrained without detraining. As a result `|mfd|` grew ~50x going
down a deep RCE column (peak −2.33 kg/m²/s vs ECHAM's conserved
~−0.02 kg/m²/s). This bogus flux dominated the DSE flux divergence
and produced peak heating of 704 K/day at the cloud base.

Fixes:
- Add `entrdd` to `ConvectionParameters` (default 2.0e-4 / m).
- Use `cmfdeps` (not `cmfctop`) for LFS-init mass flux: ECHAM cudlfs
  has `zmftop = -cmfdeps*pmfub`.
- Match entrainment with detrainment in the bulk (mfd conserved),
  add a surface-taper ramp in the bottom 2 layers so |mfd| drops
  to zero at the surface.

Result: JAX peak heating drops 704 → 56 K/day.

**Follow-on fixes (commit 317ab17)**

1. **Adiabatic compression in downdraft.** The temperature update was
   missing the (pgeoh(k-1)-pgeoh(k))/cp term that ECHAM gets implicitly
   from its DSE-based formulation. Without it, the downdraft cooled
   monotonically as it descended (273 → 248 K) instead of warming
   adiabatically. Fixed by decomposing into (a) explicit adiabatic
   warming td += g*dz/cp, (b) linear mixing toward env at fraction
   zentr/|mfd|, (c) evaporative cooling from rain.

2. **Allow deeper updraft scan.** `cloud_depth=15` capped deep
   convection at ~750 hPa; bumped to 35 (matches the ~28 layers
   ECHAM cumastr uses on the same RCE column).

3. **Saturation-adjustment cloud mask uses actual cloud top.** The
   mask was using the scan-ceiling `ktop`, so above-real-cloud-top
   levels with env supersaturated relative to JAX's qsat triggered
   the iterative cuadjtq, releasing massive bogus condensational
   heating (~870 K/day spike at level 21). Now derived from where
   `mfu` is actually nonzero.

### Remaining algorithm disagreements

| Field | Fortran | JAX (post-second-pass) |
|---|---|---|
| ktype | 3 (mid) | 1 (deep) |
| kctop | 17 (~377 hPa) | 26 (~571 hPa) |
| pq_cnv peak | 6.86 W/m² **at level 16 (top)** | 0.87 W/m² **at level 30 (mid)** |
| max dT/dt | ~2.7 K/day | ~75 K/day |
| Column-integrated heating | ~50 W/m² | ~−3 W/m² (small net) |
| Precipitation | ~0.5 mm/day (typical) | 0.008 mm/day (negligible) |

**Bug A — `ktype=3` (mid-level convection)** still missing. Same as
before; trigger logic needs the mid-level branch.

**Bug B — JAX cloud terminates too low.** JAX reaches level 26 (~571
hPa) vs Fortran's level 17 (~377 hPa). Likely buoyancy diverges
because JAX precip generation (cprcon-based) is inadequate, so lu
keeps building up in the parcel rather than being depleted, which
distorts the parcel temperature evolution. Will need to mirror
ECHAM's per-layer precip conversion `zlnew = plu/(1 + cprcon·dz·g)`.

**Bug C — heating profile distribution.** JAX peaks at level 30 (mid
cloud, ~75 K/day) while Fortran peaks at level 16 (top, ~2.7 K/day).
JAX column-integrated heating is near zero (lots of cancellation
between heating in the upper cloud and cooling in the lower), while
Fortran integrates to ~50 W/m². Root causes (interleaved):

  i. **JAX precip rate too low.** Per ECHAM's column-mass balance,
     total heating ≈ alv × precip rate. JAX precip is ~0.008 mm/day,
     so total heating is ~0.24 W/m². Need to fix the precip
     formulation first.
  ii. **Deviation flux vs full flux** distribute heating differently.
      ECHAM's full-flux + explicit detrainment puts more heating at
      cloud top via `alv * plude` (detrainment burst). The deviation
      flux gives a more spatially distributed profile.

Order of attack: Bug B first (precip + termination), then Bug A
(ktype=3), then Bug C (heating distribution) — Bug C likely
auto-resolves once precipitation is correctly tracked.
