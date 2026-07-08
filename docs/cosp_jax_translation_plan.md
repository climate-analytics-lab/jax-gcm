# COSP → JAX translation plan

Author: scoping draft · Target branch: `dev`

This is a scoping and effort-estimate document for translating the **CFMIP
Observation Simulator Package (COSP v2.0)** into differentiable JAX inside
`jcm`. The concrete driver for this work is the **warm-rain-fraction**
diagnostic used by Mülmenstädt et al. (2015, 2020) to constrain aerosol–cloud
interactions (ACI); that diagnostic turns out to be a *satellite-simulator*
product, so supporting it properly means porting the radar-simulator slice of
COSP first, then growing the rest.

The recommendation, in one line: **build COSP in JAX phase-by-phase, front-load
the CloudSat radar path so the warm-rain-fraction calibration target lands in
~3–4.5 months, and treat the full multi-instrument simulator as a 7–12 month
effort for one engineer.**

---

## 1. Why this is a COSP problem, not a one-off diagnostic

The warm-rain fraction *f*<sub>warm</sub> is "the fraction of rain occurrences
that are due solely to warm-rain processes". Two papers define it:

- **Mülmenstädt et al. (2015, GRL, 10.1002/2015GL064604)** — the
  *observational target*. A CloudSat/CALIPSO/MODIS climatology. Per 2°×2° box,
  `f_i = n_i / Σ_i n_i`, `i ∈ {ice, liquid, mixed}`, where `n_i` counts raining
  columns whose precipitating-cloud top is phase `i`. "Warm rain" = liquid cloud
  top with **no ice anywhere in the precipitating column**.

- **Mülmenstädt et al. (2020, Sci. Adv., 10.1126/sciadv.aaz6433)** — the
  *model-side constraint*. It computes the model's *f*<sub>warm</sub> **through
  COSP** and tunes the autoconversion parameterization to match the 2015
  climatology. The exact model classification (Methods) is:

  > COSP splits each grid box into *n* subcolumns (n = 100) and uses the
  > **Quickbeam radar simulator** to compute a radar reflectivity in each
  > subcolumn. Model columns that produce liquid precipitation at the surface
  > are classified as **drizzling** if their maximum reflectivity *Z*<sub>e</sub>
  > exceeds **−15 dBZ<sub>e</sub>** and **raining** if it exceeds **0 dBZ<sub>e</sub>**.
  > A column is **cold** if the highest cloud layer in which the reflectivity
  > threshold is reached contains ice, and **warm** otherwise (a cloud layer =
  > vertically contiguous levels with nonzero cloud condensate).

The physically important point (and why we cannot shortcut COSP if we want to
reproduce the *published* constraint): the classification is defined on
**simulated radar reflectivity in stochastically-generated subcolumns**, not on
grid-mean fluxes. Reflectivity is `∝ ∫ N(D) D⁶ dD`, so it is dominated by the
largest drops and depends on the assumed hydrometeor particle-size
distributions (PSDs) — exactly the "scale- and definition-aware" evaluation the
2020 paper argues is necessary. A grid-mean surface-flux proxy (see §7) is a
useful *interim* but is not the same observable and will not reproduce the
paper's numbers.

> **This is the central scoping consequence: reproducing the ACI constraint as
> published requires SCOPS + PREC_SCOPS subcolumn sampling and the Quickbeam /
> CloudSat radar simulator. Those three components are the "key early steps".**

The user has confirmed this diagnostic only makes sense with the **2-moment
cloud scheme** (`Lohmann2MMicrophysics`, `jcm/physics/clouds/lohmann_2m.py`),
which carries prognostic droplet/ice number (`qnc`, `qni`) and Nd-sensitive
Khairoutdinov–Kogan autoconversion — the process COSP's radar simulator needs
PSD information for, and the process the ACI constraint acts on.

---

## 2. What the warm-rain-fraction path actually needs from COSP

COSP is much larger than the warm-rain problem. The **minimum viable subset**
for *f*<sub>warm</sub> is the CloudSat/radar branch:

| COSP component | File (v2.0) | bytes | ~lines | Needed for warm rain? |
| --- | --- | ---: | ---: | --- |
| Subcolumn cloud overlap (SCOPS) | `subsample_and_optics_example/subcol/scops.F90` | 11,168 | ~370 | **Yes** |
| Precip subcolumns (PREC_SCOPS) | `.../subcol/prec_scops.F90` | 9,734 | ~320 | **Yes** |
| Radar reflectivity core | `src/simulator/quickbeam/quickbeam.F90` | 30,081 | ~1000 | **Yes** |
| Hydrometeor → radar optics | `.../quickbeam_optics/quickbeam_optics.F90` | 65,887 | ~2200 | **Yes** |
| Mie / dielectric library | `.../quickbeam_optics/optics_lib.F90` | 41,174 | ~1370 | **Yes** (→ LUT) |
| Math helpers | `.../quickbeam_optics/math_lib.F90` | 12,834 | ~430 | Partly (→ `jnp`) |
| Array helpers | `.../quickbeam_optics/array_lib.F90` | 3,921 | ~130 | Mostly `jnp` |
| Sort (ranking) | `.../quickbeam_optics/mrgrnk.F90` | 20,188 | ~670 | Replace w/ `jnp.sort` |
| Optics glue | `.../optics/cosp_optics.F90` | 23,681 | ~790 | Radar parts only |
| Utils | `.../optics/cosp_utils.F90` | 4,356 | ~145 | Yes |
| CloudSat interface | `src/simulator/cosp_cloudsat_interface.F90` | 11,544 | ~385 | Yes |
| Config | `src/cosp_config.F90` | 31,177 | ~1040 | Subset |
| Constants | `src/cosp_constants.F90` | 3,676 | ~120 | Yes |
| Statistics / binning | `src/cosp_stats.F90` | 36,374 | ~1210 | Subset (Ze handling) |
| kinds + error handling | `model-interface/*.F90` | 4,470 | ~150 | Trivial |

Line counts are byte-derived (≈30 bytes/line for free-form F90) and
**approximate**. Of the ~9,700 lines above, a large fraction is replaceable by
JAX built-ins (`mrgrnk`, `array_lib`, much of `math_lib`) or collapses into
lookup tables (`optics_lib` Mie). The **irreducible physics to translate and
validate** for warm rain is roughly **SCOPS + PREC_SCOPS + Quickbeam +
Quickbeam-optics ≈ 4,000–5,000 lines**.

Everything else in COSP (CALIPSO lidar, ISCCP, MODIS, MISR, PARASOL, RTTOV,
joint histograms) is **not** on the warm-rain critical path and is deferred to
later phases.

---

## 3. COSP v2.0 architecture (full inventory)

For the full-port estimate, the remaining large components (RTTOV excluded — see
§8) are:

| Simulator | File | bytes | ~lines |
| --- | --- | ---: | ---: |
| CALIPSO/ATLID lidar | `src/simulator/actsim/lidar_simulator.F90` | 70,398 | ~2350 |
| MODIS | `src/simulator/MODIS_simulator/modis_simulator.F90` | 49,354 | ~1650 |
| ISCCP (ICARUS) | `src/simulator/icarus/icarus.F90` | 30,828 | ~1030 |
| MISR | `src/simulator/MISR_simulator/MISR_simulator.F90` | 13,725 | ~460 |
| PARASOL | `src/simulator/parasol/parasol.F90` | 8,075 | ~270 |
| Instrument interfaces (×7) | `src/simulator/cosp_*_interface.F90` | ~61,000 | ~2000 |
| Top-level driver / types | `src/cosp.F90` | 322,242 | (orchestration) |

`cosp.F90` is large but is mostly derived-type definitions, allocation, and
orchestration glue. In JAX we do **not** port it verbatim — we replace it with a
lean Python driver over `@tree_math.struct` state and explicit function
composition. Treat it as design work, not line-for-line translation.

Data flow (unchanged in JAX):

```
host model (jcm) ──► input mapping ──► SCOPS ─┐
  T, p, q, ρ, dz                              ├─► subcolumn hydrometeors
  qc, qi, qr, qs, qnc, qni, cloud_fraction    │       │
                              PREC_SCOPS ──────┘       ▼
                                            per-instrument OPTICS
                                                       │
                          ┌────────────┬───────────────┼───────────────┐
                          ▼            ▼               ▼               ▼
                       Quickbeam    CALIPSO         ISCCP/MODIS      MISR/…
                       (radar Ze)   (lidar β)       (passive)        …
                          │            │               │
                          └────────────┴──────►  cosp_stats  ──► gridded diagnostics
                                              (CFADs, joint hist,   (── warm-rain here)
                                               phase classification)
```

---

## 4. JAX translation strategy

Design principles, all consistent with `CLAUDE.md` and the existing
`jcm/physics/diagnostics/` conventions:

1. **Package layout.** New subtree `jcm/physics/diagnostics/cosp/` with
   `subcol/` (scops, prec_scops), `radar/` (quickbeam, optics, mie_lut),
   `config.py`, `inputs.py` (host→COSP mapping), `classify.py` (warm/cold), and
   a thin `cosp_simulator.py` driver. Later instruments add `lidar/`, `modis/`,
   etc. Co-located `*_test.py` throughout.

2. **Pure, functional, vmap-friendly.** Everything operates on
   `(nlev, ncols)` / `(nlev, ncols, nsub)` arrays with **static** shapes.
   `n_subcolumns = 100` is a compile-time constant. Fortran column loops become
   `vmap`; the subcolumn loop becomes an added static axis. No Python control
   flow on traced values — `lax.cond`/`jnp.where` throughout.

3. **State as `@tree_math.struct`.** COSP derived types (`cosp_config`,
   `cosp_optical_inputs`, `cosp_column_inputs`, `cosp_outputs`) become immutable
   tree-math structs, matching the `PhysicsState` / `CloudData` pattern.

4. **PhysicsTerm integration.** The final classifier is a
   `PhysicsTerm(category="diagnostics")` that runs downstream of
   `Lohmann2MMicrophysics`, reads `diagnostics["clouds"]` (a `CloudData`) and the
   `qr`/`qs`/`qnc`/`qni` tracers, and writes top-level flag arrays that flow to
   xarray output and time-averaging automatically (see §6).

5. **Mie via lookup tables.** Rather than run iterative Bohren–Huffman Mie
   (`optics_lib.F90`) inside the model, precompute backscatter/extinction
   efficiency tables offline (once) over (size parameter × refractive index ×
   phase) and interpolate with `jnp.interp`/`map_coordinates`. This is faster,
   trivially differentiable, and avoids porting the trickiest numerical loop.
   The offline generator can itself be the ported Mie code, validated once
   against `optics_lib`.

6. **Validation against COSP KGO.** COSP ships `driver/` with a reference input
   file and "known good output" (KGO) plus `compare_to_kgo.py`. We build a
   single-column offline rig that feeds identical inputs to (a) reference Fortran
   COSP and (b) JAX COSP, and asserts agreement on Ze profiles, subcolumn
   statistics, and *f*<sub>warm</sub> within tolerance. This is the acceptance
   test for each phase.

---

## 5. The hard problems (and how we handle them)

These are the risk-bearing design decisions; they dominate the effort more than
raw line count.

### 5.1 Differentiability of stochastic subcolumn sampling — *the* key decision
SCOPS/PREC_SCOPS draw random numbers to place cloud and precipitation in
subcolumns given an overlap assumption. This is **non-differentiable** (sampling)
and is the crux of whether *f*<sub>warm</sub> can be a **gradient** calibration
target or only a **scalar objective** for gradient-free / ensemble calibration.

Three options, in increasing ambition:

- **(A) Stochastic forward, gradient-free calibration.** Port SCOPS faithfully
  with `jax.random`; `f`<sub>warm</sub> is a forward diagnostic. Calibrate the
  autoconversion parameters with finite differences, Bayesian/ensemble methods,
  or gradient-free optimizers. Lowest risk, fully faithful to COSP. Loses the
  headline "differentiable GCM" advantage for *this* target.
- **(B) Relaxed / reparameterized sampling.** Replace hard subcolumn assignment
  with a continuous relaxation (Gumbel-softmax-style overlap, or
  straight-through estimator) so gradients flow. Medium risk; needs care that the
  relaxed statistics match SCOPS in expectation.
- **(C) Analytic expected-value simulator.** Derive the *expected* reflectivity
  distribution / warm-vs-cold probability directly from grid-mean condensate +
  overlap assumption, bypassing subcolumns entirely. Cleanest gradients, best
  performance, but a genuine research task and a departure from bit-comparability
  with COSP.

**Recommendation:** ship **(A)** in the MVP (it reproduces the published
constraint and unblocks calibration immediately), and pursue **(B)/(C)** as a
parallel research spike so the diagnostic later becomes a first-class
differentiable loss. This choice materially changes Phase-1/3 effort, so it is
the first thing to settle with the team.

### 5.2 Hard reflectivity thresholds (0 / −15 dBZ)
The step classification kills gradients even if 5.1 is solved. Provide a **smooth
sigmoid** variant (`σ((Z − Z_thr)/τ)`) with a sharpness knob for the
gradient path, alongside the exact step for reporting. Cheap; already the pattern
used elsewhere in `jcm`.

### 5.3 PSD consistency between JCM microphysics and COSP
COSP's radar optics need PSD parameters (N₀, slope, or effective size) for each
hydrometeor. These **must be consistent** with the host 2-moment scheme's
assumptions, or model and "observation-like" quantities diverge for
non-physical reasons. The 2020 paper uses the Nam & Quaas (2012) assumptions.
We must map `Lohmann2MMicrophysics` prognostics (`qc,qi,qr,qs` + `qnc,qni`) to
COSP hydrometeor inputs explicitly and document every assumption. This mapping
(`inputs.py`) is small in code but high in scientific-review weight.

### 5.4 Static shapes / performance
Subcolumns multiply cost ×100. Mitigations: keep the radar path `vmap`-ed over
columns and subcolumns, use `lax.map` batching if memory-bound, exploit existing
SPMD sharding, and run the simulator on a **diagnostic sub-cycle** (every N
steps, like radiation) rather than every step. COSP output is a climatology, so
sub-cycling is physically fine.

### 5.5 Validation data & reproducibility
Random-number streams will not match Fortran bit-for-bit. Validate on
**statistics** (subcolumn cloud fraction, Ze CFADs, *f*<sub>warm</sub>) with
tolerances, not element-wise equality. Pin `jax.random` keys for test
determinism.

---

## 6. JCM integration of the warm-rain classifier (concrete design)

This is the Phase-3 deliverable, fully scoped against the current code so it can
start the moment the radar path (Phase 2) produces Ze:

- **Term:** `WarmRainFractionDiagnostic(PhysicsTerm)` in
  `jcm/physics/diagnostics/cosp/classify.py`.
  - `category = "diagnostics"`, `requires = ("clouds", "pressure_half")`,
    `provides = ("warm_rain_flag", "cold_rain_flag", "warm_drizzle_flag",
    "cold_drizzle_flag")`.
  - Placed **after** `Lohmann2MMicrophysics` (category `"clouds"`) in the
    `echam_physics(...)` term list, behind a `warm_rain_diagnostic: bool = False`
    factory flag (2M-only). `ComposablePhysics._validate_ordering` enforces the
    `requires` dependency.
- **Inputs read** (all already available):
  - `diagnostics["clouds"]` → `CloudData.precip_rain` (surface liquid flux,
    `(ncols,)`), `CloudData.qc`, `CloudData.qi` (`(nlev, ncols)`).
  - `state.tracers["qr"], ["qs"], ["qnc"], ["qni"]` (`(nlev, ncols)`).
  - `diagnostics["pressure_half"]` for layer masses (`Δp/g`).
  - Vertical ordering is **k=0 top → k=nlev−1 surface** (per
    `MoistAirColumnState`).
- **Computation:** feed subcolumn hydrometeors (SCOPS/PREC_SCOPS) + Quickbeam
  into per-subcolumn max-Ze; classify raining/drizzling and warm/cold per the
  2020 Methods; aggregate over subcolumns to per-column
  `warm/cold × rain/drizzle` flags.
- **Outputs:** four `(ncols,)` flag arrays. `ComposablePhysics.data_struct_to_dict`
  keeps any top-level `jax.Array` diagnostic key and reshapes `(ncols,)` →
  `(lon, lat)`, so the flags surface as maps with **zero extra plumbing**. With
  `output_averages=True` they are time-averaged; the climatological
  *f*<sub>warm</sub> map is then `⟨warm⟩ / (⟨warm⟩ + ⟨cold⟩)`, directly
  comparable to the Mülmenstädt (2015) satellite target.
- **Calibration loss:** a helper `warm_rain_fraction_from_flags(warm, cold,
  area_weights)` and an obs-comparison term (regrid the 2015 climatology to the
  model grid; MSE or likelihood on *f*<sub>warm</sub>). Differentiable insofar as
  §5.1 option B/C is adopted.

An **interim, COSP-free proxy** (surface-flux thresholds + column frozen-water
path within the precipitating column) can be delivered in ~2–3 days on top of
the existing 2M outputs, to unblock plumbing and give a first (non-authoritative)
*f*<sub>warm</sub> map while the radar path is built. It must be clearly labelled
as *not* the published observable.

---

## 7. Phased roadmap & effort estimate

Assumptions: **one engineer** fluent in JAX and cloud/radar physics; effort
includes reading the Fortran, translating, unit tests, and KGO validation;
ranges reflect the §5.1 differentiability decision and Mie-LUT vs direct.

| Phase | Scope | Deliverable | Effort |
| --- | --- | --- | ---: |
| **0** | Foundations: package skeleton, `cosp_kinds/constants/config` subset, host→COSP input mapping (`inputs.py`, §5.3), KGO single-column test rig | Config + input mapping + CI harness | **2–3 wk** |
| **1** | **SCOPS + PREC_SCOPS** subcolumn generators; PRNG; static n=100; forward stochastic (option A) + relaxed variant spike (option B) | Differentiable-forward subcolumn sampler, validated on subcolumn statistics | **2–4 wk** |
| **2** | **Radar path**: Mie→LUT generator, `quickbeam_optics`, `quickbeam`, `cosp_cloudsat_interface`; `math/array/mrgrnk`→`jnp` | Per-subcolumn Ze profiles validated vs KGO | **5–8 wk** |
| **3** | **Warm/cold classification + JCM wiring** (§6); smooth-threshold variant; obs comparison to 2015 climatology; calibration loss | **Warm-rain-fraction calibration target** (MVP COSP complete) | **2–3 wk** |
| — | *Subtotal to warm-rain MVP* | | **11–18 wk (~3–4.5 mo)** |
| **4** | CALIPSO/ATLID lidar simulator (+GOCCP-style phase, complements radar) | Lidar backscatter/SR + lidar cloud/phase diagnostics | **4–6 wk** |
| **5** | Passive simulators: MODIS (re, τ — ACI-relevant), ISCCP, MISR, PARASOL + interfaces | Passive-instrument diagnostics | **6–10 wk** |
| **6** | `cosp_stats`: CFADs, joint histograms, cloud-type/phase classification, lean JAX driver replacing `cosp.F90` orchestration | Full COSP diagnostic suite | **3–5 wk** |
| **7** | Full KGO regression suite, performance (sub-cycling, sharding, `lax.map`), gradient tests, docs | Validated, documented, performant COSP-in-JAX | **3–5 wk** |
| — | **Full COSP (excl. RTTOV)** | | **27–44 wk (~7–11 mo)** |

Notes:
- Phases 0–3 are the committed critical path for the ACI constraint; 4–7 grow
  COSP to a general satellite-simulator capability and can be reprioritized.
- Add **~30–40% calendar overhead** for reviews, scientific validation
  iterations, and integration with ongoing `jcm` changes if this is not a
  full-time single-focus effort.
- RTTOV (`cosp_rttov_v13.F90`, ~77 KB, plus the external RTTOV package) is
  **out of scope** — keep the COSP STUB. It is a licensed radiative-transfer
  model, not needed for cloud/precip ACI work.

---

## 8. Risks & open decisions

**Decisions to settle before Phase 1 (they move the estimate):**
1. **Differentiability requirement (§5.1).** Must *f*<sub>warm</sub> be a
   gradient loss (adopt B/C, +2–6 wk research), or is gradient-free / ensemble
   calibration acceptable for the MVP (option A)? *Recommendation: A now, B/C as
   a research spike.*
2. **Mie: LUT vs direct.** LUT (recommended) is faster/differentiable but adds an
   offline generation + validation step; direct port is simpler to trust but
   slower and harder to differentiate.
3. **Scope confirmation.** Full COSP is the stated goal; confirm instrument
   priority order after radar (lidar next is the natural choice for phase).
4. **PSD source of truth (§5.3).** Adopt Nam & Quaas (2012) COSP assumptions, or
   derive PSDs self-consistently from the 2M scheme's own size distributions?
   The latter is more "internally honest" but departs from the published setup.

**Risks:**
- *Subcolumn differentiability* (§5.1) — highest; mitigated by phased A→B/C.
- *PSD inconsistency* (§5.3) — scientific-validity risk; mitigated by explicit,
  reviewed `inputs.py` and side-by-side with ECHAM-HAM's COSP config.
- *`cosp.F90` size* — mitigated by replacing orchestration with a lean JAX driver
  rather than translating it.
- *Performance ×100 subcolumns* — mitigated by diagnostic sub-cycling + sharding.
- *Validation fidelity* — statistics-based tolerances, pinned PRNG keys.

---

## 9. Recommendation

1. **Approve Phases 0–3** as a ~3–4.5-month milestone delivering the
   warm-rain-fraction calibration target through a faithful CloudSat-radar COSP
   slice — the observable actually used by Mülmenstädt et al. (2020).
2. **Settle the §5.1 differentiability decision first**; default to the
   stochastic-forward MVP (option A) with a parallel relaxed/analytic research
   spike.
3. **Land the interim flux-based proxy (§6) in the first week** to unblock the
   JCM plumbing and give an early — explicitly non-authoritative —
   *f*<sub>warm</sub> map.
4. **Treat Phases 4–7 as a follow-on program** to turn the radar slice into a
   general COSP-in-JAX capability (lidar, MODIS, ISCCP, MISR, PARASOL, joint
   diagnostics), ~7–11 months total, RTTOV excluded.

---

## References

- Mülmenstädt, J., et al. (2015). *Frequency of occurrence of rain from liquid-,
  mixed-, and ice-phase clouds derived from A-Train satellite retrievals.* GRL
  42, 6502–6509. doi:10.1002/2015GL064604.
- Mülmenstädt, J., et al. (2020). *Reducing the aerosol forcing uncertainty using
  observational constraints on warm rain processes.* Sci. Adv. 6, eaaz6433.
  doi:10.1126/sciadv.aaz6433.
- Swales, D. J., et al. (2018). *The Cloud Feedback Model Intercomparison Project
  Observational Simulator Package: COSP2.* GMD 11, 77–81. Code:
  https://github.com/CFMIP/COSPv2.0
- Nam, C. C. W., & Quaas, J. (2012). *Evaluation of clouds and precipitation in
  ECHAM5 using CALIPSO and CloudSat.* J. Climate 25, 4975–4992.
