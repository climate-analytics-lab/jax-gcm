# ECHAM+RRTMGP fix campaign — plan and PR sequencing

**Date:** 2026-07-02 · **Base:** `dev` @ `b41076a`. · **Status updated:**
2026-08-12 (PRs 1–2 merged; PR 3 in review; see the table below).

Companion documents — the two point-in-time review reports
(`echam_rrtmgp_physics_review.md`: scientific correctness, findings numbered
PR-1.x criticals / PR-2.x majors / PR-3.x minors, referenced below;
`echam_rrtmgp_maintainability_review.md`: comments Part A, gradient
smoothness Part B, refactoring/dead code Part C) were written against
`dev` @ `b41076a` and live on the archived review branch
(`claude/echam-rrtmgp-physics-review-zcof6u`), not in `dev` — their cited
line numbers drift with the code, so the durable tracking for still-open
findings is GitHub issues, converted per scheme as each PR below is picked
up.

## Strategy

Physics first, with a thin structural pre-pass. Rationale:

- The physics defects are leading-order (convection violates energy/water
  conservation at the precipitation rate; SW insolation scales as cos²θ;
  Twomey factor ~100×) — tuning/validation done while they persist is
  wasted and gets redone.
- But three structural items de-risk the physics campaign and go first:
  mechanical dead-code deletion (shrinks review surface in the 1400-3800
  line cloud modules), the thermodynamics module (it *is* the fix for
  three saturation bugs), and the NaN-gradient fix + parameter-gradient
  tests (a regression gate for everything after).
- Big refactors (lohmann_2m package split, vmap hoisting) and the
  smoothing work (Part B) come last: they churn the same lines the
  physics fixes touch, and smoothing changes physics by design, so the
  faithful baseline must be validated first (otherwise calibration drift
  and bug fixes are indistinguishable).

## PR sequence

| # | Branch / PR | Type | Status |
|---|---|---|---|
| 1 | `claude/pr1-dead-code-cleanup` | structural, zero behavior change | **merged** (landed via #569, `chore/dead-code`) |
| 2 | `claude/pr2-thermodynamics` | structural+physics | **merged** (branch fully contained in `dev`) |
| 3 | `claude/pr3-radiation-quick-wins` | physics | **merged** (#548, landed as `1e8808f`: insolation µ0², surface BC, halo, gas-optics water, Twomey) |
| 4 | convection ledger | physics | not started as a unit — but later merges already fixed parts of its scope; **re-scope against current `dev` first** |
| 5 | 1M/2M microphysics ledgers | physics | not started; re-scope first (#558 NaN hardening overlaps) |
| 6 | MACv2-SP fidelity + 2M param threading | physics+structural | not started; re-scope against post-JAM `macv2_sp.py` (Twomey dNovrN already landed with PR 3) |
| 7 | module splits, vmap hoisting, smoothing (Part B) | structural | partially landed via #556 (differentiable smoothing, term ordering, structural splits); re-scope the remainder |

Housekeeping: the `claude/pr3-radiation-quick-wins` and
`claude/echam-surface-faithful-wiring` branches were merged via rebase
(#548 → `1e8808f`, #532 → `748413a`) but each still carries a small
unmerged residual diff vs `dev` (pr3: ~+308 lines incl. rrtmgp/macv2
tests; surface: ~+369/−298 in tte_tke + reference regeneration) — triage
those diffs for leftover value before deleting the branches. The JAM/AIDE
convergence programme (issue #609, `docs/aide_jcm_convergence_roadmap.md`)
runs alongside this campaign.

All branches base on `dev`. Working conventions: `ruff check .`;
`JAX_PLATFORMS=cpu pytest -n 12 -m "not slow" -q` must be green; **test-pin
discipline** — a changed numeric pin must be justified against the corrected
physics (old→new in the commit body), never tolerance-loosened; PRs 3-6 add
per-scheme column conservation checks (water: Σdp/g·(dq+dqc+dqi+…) + precip
= 0; energy: cp·dT + L·dq closure) and parameter-gradient finiteness tests
as they touch each scheme.

## PR 1 — dead-code deletion (zero behavior change)

Scope implemented (from maintainability review Part C.1, each deletion
re-verified by caller grep):

- Delete: `tracer_transport.py` (+`__init__` exports, test class);
  `echam/unit_conversions.py` + `UNIT_CONVERSIONS.md` (3 radiation tests get
  local helpers); `REMAINING_ISSUES.md`; grey_two_stream
  `ICON_FULL_IMPLEMENTATION_ROADMAP.md`; `sucloud`; lohmann_2m dead symbols
  (`cloud_micro_interface`, `lookup_1d/2d_interp`, `sat_spec_hum`,
  `MicrophysicsState_2M`, `…_2m_minimal` alias, ~110-line commented-out
  struct, commented blocks, `# ...existing code...`); `moist_static_energy`;
  in-module test scaffolding (adjustment.py, planck.py, cloud_optics.py) +
  `__main__` blocks; `radiation_scheme_rrtmgp_fn`; `RadiationFluxes`,
  `SpectralBands`; test-only radiation helpers with their tests;
  jam `species_tuple`/`mass_names_for_mode`; macv2_sp dead `total_aod`.
- Doc fixes riding along: `docs/source/echam_physics.rst` factual fix (the
  column sweep IS the wired path), CLAUDE.md `icon/`→`echam/` structure
  update, accurate MACv2-SP README rewrite.
- **Deliberately deferred**: legacy `cloud_microphysics` in echam_1m (needs
  the ice-sedimentation port first — PR 5); sundqvist `shallow_cloud_scheme`
  (carries the stratospheric condensation gate — PR 5); surface test-only
  orchestrators (coverage risk); flux_tendencies `conv_levels` block (PR 4
  rewrites that function anyway).

**Status: merged** (via #569). The deliberately-deferred items above remain
live and are still owned by PRs 4–5 as noted.

## PR 2 — thermodynamics module + saturation fixes + NaN gradients

Scope implemented:

- New `jcm/physics/thermodynamics.py` (ECHAM 6.3 constants c1es=610.78,
  c3les=17.269/c4les=35.86, c3ies=21.875/c4ies=7.66): es(T, phase),
  qs(T, p, phase), (qs, dqs/dT), `mixed_phase_weight`,
  `grid_mean_to_in_cloud`. + `thermodynamics_test.py` (value pins vs
  Murphy-Koop, FD-checked derivative, grad finiteness, broadcasting).
- Physics fixes (review refs): **PR-1.3** convection ice saturation
  (A_ICE 35.86 → c3ies 21.875; convection was ~3× low at −20 °C, ~60× at
  −60 °C) via delegation to the shared module; **PR-2.20** lohmann_2m
  `es_water` used ice coefficients below 0 °C (degenerate Bergeron
  variable) → water coefficients at all T; **PR-3.1** frozen-surface
  saturation (sea_ice.py, land.py) used over-water es with sublimation
  latent heat → ice formula.
- **B.0** NaN-gradient fix: double-where on fractional-power bases in the
  1M sweep (`zxrp1`, `zxsp1`, rain-evap 0.61 power) + a parameter-gradient
  regression test (fails on dev with NaN, passes with fix).

**Status: merged.** The remaining-item sweep for other
fractional-power-at-zero sites (grep `power(jnp.maximum` / `** 0.` patterns
in echam scope) is not confirmed done — carry it into PR 5/7 review unless
it landed with the merge.

## PR 3 — radiation quick wins (small isolated diffs, biggest W/m² per line)

From the physics review §4 (RRTMGP glue):

1. **PR-1.5 µ0² double-cosine**: `rrtmgp.py:452` passes
   `radiation_flux(...)` (already ×sinα) as `irrad` while the library
   multiplies by µ again. Fix: `irrad = direct_solar_irradiance(orbital
   phase)` (normal-incidence, distance-corrected), keep `zenith`. ~110 W/m²
   global-mean insolation deficit today.
2. **PR-1.6 surface albedo/emissivity**: `sfc_alb=0.07`/`sfc_emis=0.98`
   hardcoded in `AtmosphericStateCfg` (rrtmgp.py:96-98); per-column values
   from the surface scheme only reach diagnostics. Needs a jax-rrtmgp API
   hook (like the vmr_fields/cloud-path hooks) or per-column
   dataclasses.replace. Restores ice-albedo feedback + atmosphere/surface
   energy consistency.
3. **PR-2.39 pressure halo**: `_to_3d_with_filled_halo` edge-fill halves
   the effective Δp of top/bottom layers under the library's centered
   difference → exactly 2× heating there. Fix: linear extrapolation
   (`2p[0]−p[1]`).
4. **PR-2.40 condensate-as-vapor**: `total_water = h2o + in-cloud
   condensate` (rrtmgp.py:222) survives while q_liq/q_ice/q_c are zeroed
   (:559-562) → gas optics sees condensate as vapor everywhere incl.
   clear-sky CRE. Fix: q_t = vapor only after the zeroing.
5. **PR-1.4 Twomey**: replace `get_CDNC(aod)/get_CDNC(0)` (≈137 at
   AOD 0.43) with Stevens et al. (2017) `dNovrN = ln(1000·(aod_sp+aod_bg)+1)
   / ln(1000·aod_bg+1)` (≈1.0-1.6), aod_bg from the background AOD.
   Touches macv2_sp.py:100-110 + consumers (echam_1m base_cdnc,
   cloud_optics r_eff) stay as-is.
6. Ride-alongs if cheap: **PR-2.36** liquid/ice r_eff overhaul (Martin
   1994 from LWC/Nd; Sun-Rikus-style rei(IWC,T); fix the degenerate
   `cip/max(1.0, cwp+cip)` argument and hard-wired `land_fraction=0.5`) —
   or split to its own PR if it balloons.

## PR 4 — convection ledger (the deep one)

Physics review §2 (Tiedtke-Nordeng), reference mo_cufluxdts/cuascent:

1. **PR-1.1 cudtdq form**: flux_tendencies.py:186-209 — latent-heat term
   sign (+L·Δ(lu·mfu) should be −L·(Δpmful − plude − (pdmfup+pdmfdp)));
   add the missing plude/pdmfup/pdmfdp sinks to both T and q tendencies.
   Today: water created at the precipitation rate; column heating 454 vs
   L·P=140 W/m².
2. **PR-1.2 cprcon units**: updraft.py:358-363 — ECHAM's CPRCON=1.1e-3/g
   multiplies g·dz; code multiplies per-metre 1.4e-3 by g·dz (~12×).
   Fix: cprcon·dz (or cprcon/g·g·dz), default 1.1e-3.
3. **PR-2.1 remove the grid-mean saturation adjustment**
   (`convective_adjustment` call, tiedtke_nordeng.py:922) — no ECHAM
   counterpart; double-counts stratiform condensation; ~2/3 of current
   column heating.
4. **PR-2.2 downdraft evaporation**: replace the `cevapcu`-scaled pseudo-
   evaporation with the existing faithful `cuadjtq(kcall=2)` at every
   downdraft level; restore cevapcu to its Kessler sub-cloud-rain-evap
   meaning.
5. **PR-2.3 precip budget**: sub-cloud rain evaporation (CEVAPCU1/2),
   downdraft `pdmfdp`, rain/snow partition + `alf·pdpmel` melting term.
6. **PR-2.4 magic-number qc/qi tendencies** (`lu*0.1`/`lu*0.05`,
   `flux*0.1*0.001`) → proper `plude` detrainment.
7. Smaller: **PR-2.5** deep-closure form (Nordeng zheat), **PR-2.6**
   momentum transport (cududv), **PR-2.7** cuadjtq Ls/Lv phase pairing,
   entrscv value, turbulent detrainment δ=ε, cmfctop at cloud top,
   cloud-base condensate discard, surface-layer tendency, `_DTDT_MAX`
   T/q asymmetry, `energy_conservation_check` fix.
   (#530 half-level restagger stays a separate tracked issue — do after.)
   Gate: new column energy/water conservation tests; RCE testbed (#523)
   comparison; 3D T63L47 revalidation like #529.

## PR 5 — microphysics ledgers

1M (physics review §2.9-2.16): port ice sedimentation into the column sweep
(from the legacy path, then delete legacy per Part C); replace ad-hoc
`ice_autoconversion` with ECHAM zsaut (Levkov); snow-melt cooling extra /dt;
rain-evap `zbst` missing Rv; inverted `sqrt(rho/1.3)` in zxrp1; snow-kernel
riming/aggregation constants (cn0s/crhosno/ccsacl/colleffi — un-deads those
params); warm-ice melt (`zimlt`); clear-cell force-evaporation; 1M
`thermo_run` consumption (post-convection T/q like the 2M term); KK2000
unit fix; `zrac2` dt.

2M (§2.17-2.26, after/with PR-1 ledger items): qr/qs dual accounting (add
fallout sinks or drop the tracers to ECHAM's flux form); per-level ice-flux
dumping (kk==klev gate); dqsdt Boltzmann-constant fix (`c.ak`→Tetens
coefficient — value fix beyond PR 2's scope); `_qsat` missing ε; WBF
vertical-velocity gate; melting mass/number decoupling; scalar `snow_melt`
broadcast; KK2000 in-cloud scaling + double saturation adjustment (#539
interaction); activation dead zone (`ll2` only at the floor).
Sundqvist ride-alongs: inversion off-by-one (+land/sea-ice/ktype guards),
cptop 100→10 hPa, ice-presence-conditional saturation switch.

## PR 6 — MACv2-SP fidelity + 2M parameter threading

- MACv2-SP (§2.30-2.35): dz-weighted vertical-profile normalization;
  per-feature annual cycle inside the spatial sum; AOD-weighted (not
  Gaussian-weighted) SSA/ASY; orography truncation; real
  MACv2.0-SP_v1.nc parameter loader (fix `aerosol_parameters_from_macv2`
  TypeError) + wire year_weight/ann_cycle defaults; 3000 nm cutoff;
  background-AOD handling documented as a deviation.
- `CloudParams2M` threading: replace the import-time module-global bake
  (lohmann_2m_params.py:434-499) with the threaded params struct —
  prerequisite for any 2M calibration; prune the ~40 dead parameter
  fields across all Parameters structs (Part C.2); un-sever the SPA-knob
  gradient path (`float()` casts in echam_terms.py:183-186).

## PR 7 — structure + smoothing (after answers stop changing)

- lohmann_2m package split (Part C.4), tiedtke types split, vmap hoisting
  (perf-plan Step 2), thermodynamics migration of the remaining inline
  qsat copies (lohmann_2m orchestrator, moist_air_state, runners).
- Smoothing per Part B priorities: sigmoid trigger/type blend for Tiedtke
  (w≈20-50 J/kg, annealable), tapered updraft termination, learnable
  zdnoprc, Sundqvist soft-clip + softmax inversion, KK2000 ccraut ramp,
  smooth floors/STE for positivity clips, deterministic-key McICA for
  calibration, log-parameterization of tunables.
- Comment-hygiene sweep for whatever Part A items didn't ride along with
  their code fixes.

## How to resume

PRs 1–3 are merged. Before picking up any of PRs 4–7: **re-audit that PR's
finding list against current `dev`** — substantial overlapping work has
merged since the review (e.g. #548 radiation, #556 smoothing/splits, #558
NaN hardening, the coupled surface-flux solve, JAM) — and convert the
findings that are still open into GitHub issues so tracking survives the
review branch's retirement. Each PR branches fresh from `dev`. Standard
gates throughout: `ruff check .` and
`JAX_PLATFORMS=cpu pytest -n 12 -m "not slow" -q` green; test-pin
discipline (changed pins justified old→new against the corrected physics);
per-scheme conservation + parameter-gradient tests added as each scheme is
touched (see "Working conventions" above).
