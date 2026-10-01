# JAM modal aerosol microphysics — an Apache core (plan)

**Status:** PLAN, awaiting approval. Nothing here is implemented.
**Branch:** `aer/modal-microphysics`, worktree `/data/lkemp/jax-gcm-aerosol`,
based on `origin/dev` `0bd6deee`. No upstream, nothing pushed (the D21 rule
carried over from the superparameterization work).
**Venv:** `/data/lkemp/venvs/jcm-aer` (CPU, python 3.12.2, jcm editable from
this worktree, ruff pinned 0.15.17 to match).
**Baseline measured 2026-09-25:** `ruff check .` clean; `pytest
jcm/physics/aerosol/ -m "not slow"` = 428 passed, 1 skipped in 126 s.

## 1. The gap

JAM is a near-complete MAM4-style modal aerosol harness: emissions (dust,
sea salt, DMS, anthropogenic), sulfur and aqueous chemistry, ARG activation,
ice nucleation, Stokes sedimentation, Slinn dry deposition, wet scavenging,
Mie optics and a cloud-borne phase. `jam_terms.py:373` assembles it as

```
terms = [*pre_core, core, *optics_terms, *post_core]
```

where `core` is the single swap point: a `ModalMicrophysicsTerm`
(`microphysics/base.py:25`, category `aerosol_microphysics`) that publishes
the `_jam_state` diagnostic every downstream term reads.

Two cores exist. `PlaceholderMicrophysics` computes κ-Köhler equilibrium
radii and returns **zero tendency** — no nucleation, condensation,
coagulation, ageing or renaming. `Mam4JaxMicrophysics` has all five, but
wraps `mam4-jax`, which is **GPL-3.0** and an optional extra that is not
installed here and is absent from CI.

So in the Apache tree the aerosol population is transported, emitted,
scavenged, deposited and optically active, but it never *evolves*. Modes
neither grow nor exchange mass, and `jam_carbon_aging.md` records what one
of those omissions costs on its own: with no ageing, primary carbon can
never leave the hydrophobic `pcm` mode, giving a BC lifetime of ~21 days
against an observed 5–8.

**This plan writes that core in pure jcm**, so JAM runs a physically
evolving population under Apache alone.

## 2. Scope

### Phase A — the core (this branch)

Five processes, in MAM4's operator order, inside one
`ModalMicrophysicsTerm` subclass:

| # | Process | What it does |
|---|---|---|
| A1 | Binary H₂SO₄–H₂O homogeneous nucleation | New particles into `aitken`; consumes `g_h2so4` |
| A2 | Gas–aerosol condensation | `g_h2so4` and `g_soag` onto every mode by condensation sink |
| A3 | Brownian coagulation | Intra-modal (number loss, mass conserved) and inter-modal transfer |
| A4 | Carbonaceous ageing | `pcm` → `acc` once a soluble coating passes the monolayer criterion |
| A5 | Renaming / mode merging | Mass+number transfer when a mode grows past its `dgnum_hi` |

The population shape is **unchanged**: `MAM4_SPEC`
(`microphysics/mam4_data.py:89`), 4 modes, the existing `m_<sp>_<mode>` /
`n_<mode>` tracer layout (`tracer_layout.py`), the existing gas tracers
(`gas_species.py`: `dms`, `so2`, `h2so4`, `soag`). Nothing upstream or
downstream of `core` changes.

### Phase B — aerosol-aware CRM coupling (planned, not started)

Kept in this plan because it is the user's research direction, but it is
**blocked**: it needs the embedded CRM's two-moment microphysics, which is
card SP-15 in work unit WU-6 on `sp/superparameterization` and has not
started. Sketch only — see §6.

## 3. Cards

Dependencies in brackets. Each card is implement → review → full gates.

| Card | Title | Deps |
|---|---|---|
| **AER-01** | ✅ `JamModalMicrophysics` skeleton: subclass, `spec = MAM4_SPEC`, process-order scaffold, `_jam_state` via the shared κ-Köhler diagnostics (moved to `koehler.py`), zero tendency, bitwise-equal to `PlaceholderMicrophysics`. Classified in `layout_audit_test.py` — that could not wait for AER-08, since the roster assertion fails on any unclassified `PhysicsTerm`. | — |
| **AER-02** | ✅ `moments.py`: log-normal moment algebra — `moment`/`moment_factor`, the M₃ pair `volume_from_diameter`/`diameter_from_volume`, `surface_area` (π·M₂), `mode_volume`/`mode_mass`. `koehler.py` now delegates its diameter inversion here, so the core and the diagnostics cannot disagree about it. Property tests against brute-force quadrature. **Rolled in:** the float32 NaN-gradient trap at empty modes (§9). | AER-01 |
| **AER-03** | ✅ `condensation.py`: MAM4's gas-aerosol exchange from its own documentation — Fuchs–Sutugin per particle, Fuller (1966) diffusivity, two-point Gauss–Hermite modal integration on the **wet** diameter (not the supplement's B&S harmonic mean; §9). H₂SO₄ nonvolatile with the chemical production coupled into the solve (`SulfurGasChemistry` now publishes `_h2so4_production`); SOAG at 0.81·C with Raoult's law over SOA + 10 % oxygenated POA, semi-implicit sub-steps in a static scan. `pcm` condensate routed to accumulation and recorded in `coating` for AER-06. Sulfur moles and organic mass conserved; 20 tests. | AER-02 |
| **AER-04** | ✅ `new_particle_formation.py`: NPF from MAM4's own documentation — binary H₂SO₄–H₂O (Vehkamäki et al. 2002 polynomials, validity clipping as MAM does it) **plus** the PBL first-order pathway (Sihto et al. 2006, 1.0×10⁻⁶ s⁻¹, 1 nm pure-acid clusters) below max(PBL top, 100 m), the larger rate winning; Kerminen & Kulmala (2002) survival to the Aitken lower bound with MAM's condensation sink from AER-03's summed uptake coefficient and the step-mean H₂SO₄; MAM's vapour-availability clipping. PBL height taken lagged — sub-decision (a) in D-A2, adopted. **Rolled in:** AER-03's gas update rewritten so the vapour can never go negative by rounding (§9). 18 tests. | AER-03 |
| **AER-05** | ✅ `coagulation.py`: MAM4's coagulation from E3SM's `modal_aero_coag.F90` (the MAM doc-site pages are stubs; §9) — CMAQ v4.6 fast coefficients (Whitby et al. 1991 analytic moment integrals, Binkowski & Roselle 2003 correction tables in `coagulation_tables.py`, harmonic mean of the free-molecular and near-continuum limits) on the wet κ-Köhler state; semi-analytic number loss (receiver first, donors at the receivers' step-mean numbers); first-order volume transfer. **Aitken, accumulation and primary-carbon only** — MAM neglects coagulation involving coarse as much slower. Aitken mass goes to accumulation whichever partner it hit; the primary-carbon share of its sulfate and SOA is recorded in `coating` for AER-06. Checked against the monodisperse closed forms (exact) and brute-force quadrature (+3…+7 % at MAM4 geometry; §9). No tunables. 21 tests. | AER-02 |
| **AER-06** | ✅ `ageing.py`: MAM4's primary-carbon ageing from E3SM's `modal_aero_gasaerexch.F90` (and its duplicate in `modal_aero_coag.F90`) — the sulfate-plus-equivalent-SOA shell deposited this step (`coating`, written by AER-03 and AER-05), as a thickness over the mode's surface (`6 e^{-2.5 ln²σ}/D_g`), against `n_so4_monolayers` of 4.76 Å; that fraction of POA, BC, marine organics and number moves to accumulation; SOA weighted by κ_SOA/κ_SO4. Runs once, last, on the sum of both deposits (D-A1). **Default `n_so4_monolayers = 8`, not the 3 this card first said** — 3 is MAM7's (Liu et al. 2012); MAM4 uses 8 (Liu et al. 2016 §2, and both E3SM's and CESM's sources; §9). **Rolled in:** the `1 − 10ε` transfer cap is dtype-aware (it was exactly 1.0 in float32 in AER-05 too). 14 tests. | AER-03, AER-05 |
| **AER-07** | ✅ `rename.py`: MAM4's renaming from E3SM's `modal_aero_rename.F90` (doc-site page is a stub; §9) — after gas-aerosol exchange, the Aitken tail above `D_cut = √(D_v,ait · D_v,acc)` (≈55 nm) moves to accumulation: volume fraction `[F_v(D_new)V_new − F_v(D_old)V_old]/V_new`, number fraction `F_n(D_new) − F_n(D_old)` capped by it, erfc tail integrals; MAM's guards (empty, no growth, below reference diameter, relaxed number bounds ×27, guaranteed transfer past the cut). Growth measured against the new `env.dry_volume_start`. Both phases, from the combined volume (D-A5, adopted). No tunables. 16 tests. | AER-02 |
| **AER-08** | ✅ `jam_microphysics="modal"` registered in `jam_terms._MICROPHYSICS` and so reachable through `echam_physics()` and Hydra (`physics.jam_microphysics=modal`); defaults unchanged per D-A4. Wiring tests in `jam_terms_test` (both factories, cloud-borne store ordering, the store requirement dropped for `cloud_borne=False`, the x64 mixed-dtype trace) and a slow six-step run in `jam_integration_test`. **Rolled in:** the SOAG sub-step scan's carry promoted to float64 under `jax_enable_x64` — a hard error on the pySES path; outputs now cast to the state dtype. | AER-01, D-A4 |
| **AER-09** | ✅ `ledger.py`: budgets per kg of air over both phases — sulfur moles (sulfate at the booked molar mass + DMS, SO₂, H₂SO₄), organic mass (POA, SOA, MOA + SOAG), each inert species, total number — and a per-process `CONTRACT` (exchange, renaming, ageing: number equal; NPF: non-decreasing; coagulation: non-increasing; every mass budget equal everywhere). `JamModalMicrophysics(audit=True)` publishes `_modal_ledger_<process>_<budget>` relative changes; the step-level sulfur/organic/number closure is always published as `_modal_ledger_step_*` for health gates. `ledger_test` holds all five processes to the contract over warm/heavy-vapour/cold/near-empty columns with a cloud-borne phase and without, checks the processes actually acted, and shows a deliberately leaking process is caught. All green at 1e-5 in float32. | AER-03..07 |
| **AER-10** | Science doc + release note; `jam.md` cross-links; reference verification (§7). | AER-09 |
| **AER-11** | Differentiability check: finite-difference vs autodiff through each process; no `jnp.where` NaN traps on empty modes. | AER-09 |
| **AER-12** | A/B against `Mam4JaxMicrophysics` **if** the GPL extra can be installed in a throwaway venv — as a validation reference only, never a dependency. | AER-09 |

## 4. Design constraints

- **Apache only.** `mam4-jax` is GPL-3.0. This core is written from the
  published literature and the E3SM Fortran *descriptions*, not by
  translating `mam4-jax` source. AER-12 may compare against it numerically;
  it must never be imported by the new core.
- **Differentiable.** Every process is traced through `jax.grad`. Empty
  modes are the sharp edge: a mode with zero number must give a finite
  gradient, so guarded divisions use the increment form already adopted
  elsewhere in this tree (the SP contract's float32 division audit).
- **Follows jcm's working dtype** (float32 by default), matching D24 on the
  SP branch.
- **Mass and number conserved per process**, ledgered, not merely asserted
  at the end. Nucleation and condensation both consume `g_h2so4`; the order
  and the shared sink are where conservation is easiest to lose.
- **No change to the population spec or tracer layout.** If M7 modes are
  wanted later (Tegen et al. 2019), they are a *second* `ModalAerosolSpec`,
  not an edit to this one.

## 5. Verification

The AER-09 ledger is the standing conservation gate: with ``audit=True``
every process's relative change of every budget is a diagnostic, so a
failure names the process. The step closure is on in every run.


1. `ruff check .` clean, pinned 0.15.17.
2. `pytest jcm/physics/aerosol/` green — the 428-test baseline must not
   regress, since `jam_microphysics="placeholder"` stays the default until
   AER-08.
3. Per-process analytic checks: condensation against an isolated single-mode
   solution, coagulation coefficients against the monodisperse closed forms
   and brute-force quadrature over the log-normals (the self-preserving
   distribution would test the time integration, not MAM's coefficients,
   and is not what MAM itself is validated against), NPF on structure, since
   no tabulated Vehkamäki values exist outside the paper's figures (§9).
4. Budget closure (AER-09) and gradient checks (AER-11).
5. A single-column JAM run before/after, showing BC lifetime moving from the
   ~21-day no-ageing figure toward the observed 5–8 days.

## 6. Phase B sketch — aerosol-aware superparameterized convection

The scientific target. Not startable yet.

Today the SP plan keeps aerosol on the **GCM column** and does not expose it
to the embedded CRM (roadmap v3 line 40: the SP term "keeps RRTMGP, TTE-TKE
vertical diffusion, EchamSurface, … aerosol, chemistry and gravity-wave drag
on the GCM column"). So CRM convection is aerosol-blind, and the aerosol
sees only GCM-mean updraughts.

The coupling has two directions, and they are not symmetric:

- **Aerosol → CRM.** ARG activation (`activation/arg.py`) is driven by an
  updraught velocity. On the GCM column that is a sub-grid distribution; in
  an SP model the CRM *resolves* the updraughts, which is the entire reason
  superparameterization improves aerosol–cloud interaction. This means
  activation per CRM column at the resolved `w`, feeding CDNC into the CRM's
  two-moment microphysics (**SP-15**, not written).
- **CRM → aerosol.** Resolved in-cloud scavenging and cloud-borne exchange
  per CRM column, then aggregation back to the GCM column's tracers. Touches
  `wetdep/`, `cloud_borne.py` and the cloud-borne carry store.

Open questions to settle when it starts: whether aerosol tracers are carried
per CRM column or held GCM-mean and only *sampled* by the CRM (cost: the
aerosol carry is already 25 tracers × nx); how cloud-borne aerosol survives a
CRM substep; and whether activation runs per CRM substep or once per GCM
step.

**Blocked on SP-15.** Raise with the SP work unit sequence, not here.

## 7. References — VERIFIED 2026-09-26

Checked against the papers, not recalled. The authority for what MAM
actually does is the **supplement** to Liu et al. (2012) — sections S1.1.4
to S1.1.7 — because the main text of neither Liu paper describes the
microphysics schemes. Vehkamäki is cited in neither main text.

| Process | Reference |
|---|---|
| Binary nucleation | Vehkamäki, H., Kulmala, M., Napari, I., Lehtinen, K. E. J., Timmreck, C., Noppel, M., Laaksonen, A. (2002), *An improved parameterization for sulfuric acid–water nucleation rates for tropospheric and stratospheric conditions*, JGR 107, 4622, doi:10.1029/2002JD002184 |
| PBL nucleation | Sihto, S.-L., et al. (2006), ACP 6, 4079–4091 — empirical first-order-in-H₂SO₄ rate; coefficient 1.0×10⁻⁶ s⁻¹ as in Wang et al. (2009) |
| Nucleated-particle survival | Kerminen, V.-M., Kulmala, M. (2002), *Analytical formulae connecting the "real" and the "apparent" nucleation rate…*, J. Aerosol Sci. 33(4), 609–622 |
| Condensation | Seinfeld & Pandis (1998) mass-transfer expressions, integrated over each mode's size distribution per Binkowski, F. S., Shankar, U. (1995), JGR 100, 26191–26209 |
| H₂SO₄ accommodation coefficient (0.65) | Pöschl, U., et al. (1998), J. Phys. Chem. A 102(49), 10082–10089 |
| Coagulation | Binkowski, F., Roselle, S. (2003), *Models-3 CMAQ aerosol component – 1*, JGR 108(D6), 4183, doi:10.1029/2001JD001409 — MAM uses CMAQ v4.6's fast/approximate algorithms. The analytic moment forms are Whitby, E. R., McMurry, P. H., Shankar, U., Binkowski, F. S. (1991), *Modal aerosol dynamics modeling*, EPA 600/3-91/020, appendix H. Implementation reference: E3SM `components/eam/src/chemistry/modal_aero/modal_aero_coag.F90` (`modal_aero_coag_sub`, `getcoags_wrapper_f`, `getcoags`; BSD-3), read 2026-10-01 |
| Renaming (Aitken → accumulation) | Easter, R. C., et al. (2004), *MIRAGE* |
| Water uptake / κ | Ghan & Zaveri (2007) equilibrium Köhler; κ per Petters, M. D., Kreidenweis, S. M. (2007), ACP 7(8), 1961–1971 — already the basis of ``koehler.py`` |
| MAM3 / CAM5 | Liu, X., et al. (2012), GMD 5, 709–739, doi:10.5194/gmd-5-709-2012 (+ supplement) |
| MAM4 four-mode | Liu, X., et al. (2016), GMD 9, 505–522, doi:10.5194/gmd-9-505-2016 |
| M7 (Phase-B mode comparison) | Vignati, E., Wilson, J., Stier, P. (2004), JGR 109, D22202, doi:10.1029/2003JD004485 |
| Ternary nucleation (**not used** — needs NH₃) | Merikanto, J., et al. (2007), JGR 112, D15207 |

### Corrections to the first draft of this plan

Three attributions in the version written before this check were wrong,
and all three had reached the card table and the code docstrings:

1. **Fuchs–Sutugin was cited for condensation.** It appears nowhere in
   MAM's description. The transition-regime treatment enters through the
   Seinfeld & Pandis mass-transfer expressions as integrated by Binkowski
   & Shankar (1995).
2. **Binkowski & Shankar (1995) was cited for coagulation.** It is the
   *condensation* size-distribution integration. Coagulation is Binkowski
   & Roselle (2003).
3. **Renaming had no reference.** It is Easter et al. (2004), and the
   supplement is explicit that it runs *after* continuous growth — which
   independently confirms the scaffold's condensation-before-rename order.

## 8. Decisions

### D-A1 — process order: ADOPT MAM's ORDER, NO SUB-CYCLING YET

`gas_aerosol_exchange → rename → nucleation → coagulation →
carbonaceous_ageing`, one pass per physics step.

What the evidence does and does not support, stated plainly:

- **Supported.** Rename follows continuous growth (supplement S1.1.5,
  Easter et al. 2004). Ageing is driven by the mass condensation and
  coagulation deposit, so it cannot precede them.
- **Not documented.** Neither Liu paper nor the supplement states the
  `amicphys` operator sequence or any sub-stepping. The five-step order
  comes from jcm's own `Mam4JaxMicrophysics` docstring, which describes
  the core it wraps. That provenance is weaker than a paper and is
  recorded as such rather than dressed up.

**A consequence to carry into AER-03/05/06.** In MAM, ageing is *not* a
terminal step: it is computed inside condensation (from the condensed
mass on `pcm`) and again inside coagulation (Aitken→pcm transfer ages
some primary carbon). This scaffold keeps ageing as its own process, so
**AER-03 and AER-05 must each record the soluble mass they deposit on
`pcm`** for AER-06 to consume. If that bookkeeping turns out to be more
awkward than folding ageing into the two donors, revisit this decision
rather than working around it.

**Sub-cycling: not now.** Condensation and nucleation compete for the
same `g_h2so4` and the pair is genuinely stiff, but no measurement here
yet justifies the cost. Revisit with a trigger, not a hunch: if AER-04
shows a single step depleting `g_h2so4` by more than ~50 %, sub-cycle
that pair. AER-03 publishes the measurement as
`_modal_condensation_h2so4_depletion`. The driver is a fold over `self.processes`, so adding
sub-cycling is a driver change, not an architecture change.

### D-A2 — nucleation scheme: BINARY **PLUS** PBL, NO TERNARY

The question as originally posed — "binary only (MAM4's default), or also
a boundary-layer ternary/organic pathway?" — had a false premise.
**MAM's default is not binary-only.** Supplement S1.1.4, verbatim:

> "A binary parameterization (Vehkamaki et al., 2002) is used in the
> 3-mode version, which does not predict NH3, while a ternary
> parameterization (Merikanto et al., 2007) is used in the 7-mode
> version. The boundary layer parameterization, which is applied in the
> planetary boundary layer (PBL) in both versions, uses the empirical
> first order (in H2SO4) nucleation rate from Sihto et al. (2006), with a
> first order rate coefficient of 1.0×10-6 s-1 as in Wang et al. (2009)."

So:

- **Binary (Vehkamäki 2002)** — the free-troposphere pathway, since MAM4
  inherits MAM3's 3-mode chemistry and predicts no NH₃.
- **PBL first-order (Sihto 2006)**, coefficient 1.0×10⁻⁶ s⁻¹, applied
  within the boundary layer. This is what addresses the continental
  Aitken-number deficit that binary-only nucleation causes — the worry
  that prompted the original question. MAM answers it with this pathway,
  not with ternary.
- **Kerminen & Kulmala (2002)** for survival from the critical cluster to
  the Aitken lower bound. New particles go to the Aitken mode.
- **Ternary is out of scope and structurally unavailable**: it requires
  NH₃, and JAM's `gas_species.py` carries only `dms`, `so2`, `h2so4` and
  `soag`. Adding it means a new prognostic gas, its emissions and its
  chemistry — a separate piece of work, not a knob.

**Sub-decision for AER-04 — where the PBL height comes from.** The Sihto
pathway needs to know which levels are in the boundary layer. jcm has
`pbl_height` on the vertical-diffusion diagnostics struct
(`vertical_diffusion_types.py:359`), but the term order is against us:
measured on `echam_physics(aerosol_module="jam", cloud_scheme="2m")`, the
microphysics core is term **12** and `TteTkeVerticalDiffusion` is term
**21**, so any published PBL height is **one step lagged** (and absent on
the first step). Two options, to settle in AER-04:

- (a) read the lagged `pbl_height` and document the lag — defensible,
  since the PBL depth varies slowly against a 30-minute step;
- (b) diagnose the boundary layer locally inside the process from the
  temperature/humidity profile, which is self-contained but duplicates a
  diagnosis the host already makes.

**Adopted in AER-04: (a), the lagged height.** The core reads
`diagnostics["vertical_diffusion"].pbl_height` when it is there and takes
zero when it is not — the first step of a run, and any configuration
without the TTE-TKE scheme. Zero is not "no boundary layer": the pathway
applies below max(PBL top, 100 m), so a missing height degrades to a
100 m surface layer rather than switching the pathway off, which is the
behaviour MAM's own floor prescribes for a shallow PBL. One step of lag
on a quantity that varies over hours is accepted and pinned by a test
(`test_hundred_metre_floor_applies_when_pbl_height_is_absent`). Height
above the surface comes from `MoistAirColumnState`'s `height_full` minus
the surface `height_half`, with the geopotential as fallback. (b) stays
available if a run without TTE-TKE ever needs the pathway above 100 m.

### D-A5 — cloud-borne renaming: IN-CORE, MAM-FAITHFUL (decided 2026-10-01)

MAM computes the renaming fractions once from the combined interstitial
+ cloud-borne Aitken volume and number and applies them to both phases.
Three options were weighed: (a) renaming in the core on the combined
population, moving the cloud-borne half through the carry store in the
same step; (b) a separate post-core term applying fractions the core
publishes; (c) accepting the gap and relying on resuspension. **(a) is
adopted.** What settled it:

- The carry store (`cloud_borne_store.py`) is written by a *pre-core*
  term and read/updated through `tracer_view`/`apply_updates` by any
  term, so the core can move cloud-borne Aitken in the same step with no
  lag and no new term — the `AqueousSulfur` pattern, including
  `_jam_cloud_borne` in `requires` when `spec.cloud_borne`.
- The GPL `Mam4JaxMicrophysics` adapter packs `qqcw` and reads back
  cloud-borne tendencies, so the AER-12 reference renames cloud-borne;
  (c) would have been a standing A/B difference.
- MAM's second renaming call, after aqueous chemistry, is moot in jcm:
  `AqueousSulfur` puts its product in cloud-borne accumulation and
  coarse only, never Aitken, so there is no cloud-borne Aitken growth
  to rename. Only the exchange-driven call is reproduced.

Implementation: `ModalPopulation.cloud_borne` carries the mirrors
through the fold (every process now returns via `dataclasses.replace`,
so a field only one process touches survives the others); renaming
forms the combined volume and number and moves both phases by the same
fractions; the driver books the cloud-borne change to the store as a
rate via `apply_updates`, never to the tracer tendency, and only when
the store exists (the structural probe runs before the store term).
The store module is imported lazily in `modal.py`: the JAM package
imports it before the core, and a module-level import closed a cycle.
Pinned by `CloudBorneTest` (fractions against the per-cell
transcription with combined inputs, both-phase conservation, the store
booking through the driver, and the probe without a store).

### D-A4 — default switch: REGISTER ``modal``, OPT-IN (decided 2026-10-01)

Three positions were put: register ``"modal"`` and change no default;
make it the default of the Hydra JAM config (`physics/echam-jam.yaml`,
today on the GPL ``mam4_jax``); make it the default everywhere. **The
first is adopted.** `echam_physics()` keeps ``"placeholder"`` and
`echam-jam.yaml` keeps ``mam4_jax``; the in-tree core is selected with
``jam_microphysics="modal"`` or ``physics.jam_microphysics=modal``. The
trigger to revisit is concrete: AER-09's conservation ledger green and
AER-12's A/B against ``mam4_jax`` within tolerance (which needs D-A3).
Until then every existing JAM test and run keeps its behaviour, and the
new core is exercised by its own wiring tests and a slow integration run.

### D-A3 — validation reference: ``requires_extra("mam4")`` A/B, SEPARATE LOCAL VENV (decided 2026-10-01)

The repository had already settled the licensing half: ``mam4-jax`` is
the opt-in ``jcm[mam4]`` extra (GPL-3.0, never bundled, never imported
by default), the ``extras-tests`` CI job installs every extra, and a
test gates on it only through ``@pytest.mark.requires_extra("mam4")``
with the import inside the test. AER-12 follows that pattern: the A/B
against ``Mam4JaxMicrophysics`` is such a test, run by CI's extras job.
Locally it runs in a second venv (``jcm-aer-mam4``) so the working venv
stays GPL-free and ``mam4-jax``'s process-wide ``jax_enable_x64`` flip
on import cannot leak into other local runs. Two rules, both already in
§4: the comparison reads outputs only — the in-tree core is never
corrected by reading ``mam4-jax`` source, only MAM's documentation,
E3SM's BSD-3 Fortran and the papers — and a disagreement is a finding to
record, not a number to tune toward. The options declined: installing
the extra into the working venv (simplest, but the GPL package in the
dev environment and the x64 flip on every run), and no numeric A/B
(leaves D-A4's flip trigger without its second half).

### Terminology (2026-10-01)

The H₂SO₄-vapour new-particle process is called **new-particle formation
(NPF)** in prose from here on. "Condensation nuclei activation" is a
*different* process — particles becoming droplets — and is what the
existing `ArgActivation` term already implements, so using it for NPF
would mislabel the physics. The function in `modal.py` keeps the name the
MAM literature uses; AER-04 is the card that implements it.

## 9. Findings rolled in from AER-02 onward

### The float32 NaN-gradient trap at empty modes

Measured 2026-10-01, before any fix, with `jax.grad` through
`equilibrium_modal_state` at an accumulation mode with zero mass or zero
number:

| dtype | `r_dry` | `rho` | `kappa` | `r_wet` |
|---|---|---|---|---|
| float32 | NaN (∂/∂number) | NaN (∂/∂mass) | NaN (∂/∂mass) | NaN (both) |
| float64 | finite | finite | finite | finite |

Cause: a floored divisor, `x / max(y, tiny)`. Its reverse-mode cotangent
is `-x / max(y, tiny)²`, and `tiny²` (1e-60, 1e-80) underflows to zero in
float32, giving `0/0`. `jnp.where` does not rescue it — the unselected
branch's cotangent is `NaN × 0 = NaN`. The fix is the standard double
`where`: divide by a harmless `1.0` in the unselected branch and mask the
*result*. Forward values are unchanged on every populated mode.

Float32-only, and the user runs float64 — but CI and the default
configuration run float32, the same trap was in the brand-new
`diameter_from_volume`, and a NaN cotangent at one empty mode reaches
every upstream parameter. Same failure class, same PR
(`CLAUDE.md`, "related findings get rolled in"). Regression tests:
`koehler_test.py`, `moments_test.py::test_empty_mode_is_finite…`.

### AER-03: the formulation is MAM4's own, and it is not the one the supplement names

The card was planned from the Liu et al. (2012) supplement, which says
condensation is "integrated over the size distribution of each mode
(Binkowski and Shankar, 1995)" — the harmonic mean of the free-molecular
and near-continuum growth laws. MAM4's own documentation
(`mam_doc_site/gas_aer_exch/`, seven pages, read 2026-10-01) describes
the code that runs in E3SM, and it is different: the Fuchs–Sutugin
integrand `f(Kn, α)·D_p` is integrated by **two-point Gauss–Hermite
quadrature** after factoring out `D_p^β`, with `β` the integrand's local
log-slope at the mode's **wet** geometric-mean diameter. The
documentation is followed because it describes the implementation; the
supplement's citation is the earlier CAM5 scheme. Pinned in
`condensation_test.py` against brute-force integration of the same
integrand (≤3 % at every mode bound, which is MAM's own verification
tolerance for the two-point rule) and against the continuum limit
`2πD·M₁`.

Two further things the documentation settled that the supplement does
not state:

- **Chemical production is coupled into the condensation solve.** MAM4
  solves `dq/dt = P* − q·ΣC` with the step-mean production `P*` because
  sequential splitting of source and sink at climate-model steps was
  shown to produce large errors (Wan et al. 2013; Wang et al. 2019). jcm's
  additive split would commit exactly that error, so `SulfurGasChemistry`
  now publishes `_h2so4_production` and the core reads it; the tendency
  the core returns is the condensation part only, so the two sum
  correctly. Pinned: with zero initial gas, condensation of what is
  produced within the step is nonzero (the split would give zero).
- **SOAG's coefficient is 0.81 of H₂SO₄'s**, from the Fuller ratio at
  typical conditions with the SOAG diffusion volume scaled by molecular
  weight; the equilibrium is `1e-10 mol/mol` at 298 K and 1 atm with
  ΔH_vap = 156 kJ/mol, over SOA plus 10 % of POA taken as oxygenated.

### AER-03: the sub-step scan is bounded, and MAM's loop is not

MAM4 integrates the nonlinear SOA system with semi-implicit Euler and
**adaptive, unbounded** sub-steps (each may move the most-imbalanced mode
by at most α = 5 %). A data-dependent loop count is not
reverse-differentiable, so here the same scheme runs inside a
fixed-length `lax.scan`; a sub-step that finds no time remaining is a
no-op, so the result is independent of the length whenever the adaptive
count fits.

Measured 2026-10-01 on a continental-strength population (number
~1e9 kg⁻¹, dt = 1800 s): the result became **exactly** scan-length
independent between 200 and 400 sub-steps in every SOAG regime, and a
cap of 10 changed SOA masses by 10–50 %. The default is therefore 400,
and `condensation_soag_substep_shortfall` publishes the fraction of the
step the scan did not reach. **This is a known difference from MAM**: a
more polluted column scales the demand with the total uptake
coefficient, the cap will bind there, and the last sub-step is then
oversized — unconditionally stable and conserving, but first-order. The
diagnostic exists so that is visible, not silent. Pinned: shortfall is
zero at the default and the result is bitwise identical at 800; at 10
the shortfall is positive.

### AER-03: MAM's 1e-20 safeguards are not float32-safe

`numerics_SOA` floors the solvent amount and the gas/saturation scale at
ε = 1e-20 mol/mol. The reverse-mode derivative of `x / max(y, ε)` carries
`1/ε²`, and 1e-40 is below the smallest float32 subnormal — the same trap
as AER-02, found the same way (a NaN gradient in float32, finite in
float64, only when the gas tracers were absent and everything sat at the
floors). MAM runs in float64 and never meets it. The floors here are
**1e-15 mol/mol** (squares to 1e-30, a float32 normal; five orders below
any SOAG amount that matters) and every division by a floored quantity
goes through the double-`where` so the dead branch divides by 1.0.
Pinned in float32 at an empty mode with no gas.

### AER-03: the primary-carbon mode's condensate

`pcm` carries no sulfate or SOA slot (`_PCARBON_SPECIES`) yet has surface
area and takes up vapour like any mode. MAM moves that condensate to
accumulation "along with" the aged carbon; the carbon transfer is AER-06
and needs the deposited amount to compute `f_age`. So the condensate is
routed to accumulation now and recorded, by species, in
`ModalPopulation.coating`. This is also the sulfur leak the `mam4-jax`
adapter had to close separately.

### The refactor is not bitwise, and here is exactly how much

Pointing `koehler` at `moments.diameter_from_volume` was meant to be
bitwise-neutral and was measured instead of assumed. `rho`, `kappa`,
`mass`, `number`: **identical** in both dtypes. `r_dry`, `r_wet`:
**1 ulp in float32** (3.6e-15 on 1e-7), **~2.6 ulp in float64** (5.8e-16
relative). The source is the σ_g moment constant — the old code evaluated
`(π/6)·exp(4.5·ln²σ)` with `jnp` in the working dtype; `moments` evaluates
`0.5·k²·ln²σ` with `math` in float64 and casts. Different association,
different `exp` implementation, last-digit difference. No reference
rounding exists to prefer, every consumer's test tolerance is ≥1e-6, and
keeping two definitions of the M₃ inversion is the drift this card exists
to remove — so the shift is accepted and recorded rather than hidden
behind a bitwise claim it would not survive.

### AER-04: the NPF formulation is also MAM's own, not the supplement's

As with AER-03, the implemented formulation is MAM's documentation
(`mam_doc_site/aerosol_nucleation/`, read 2026-10-01), where it differs
from the Liu et al. (2012) supplement's two-line description:

- the **condensation sink** in the Kerminen–Kulmala survival is MAM's
  `CS' = C_sum / (4π D_g α)` built from the gas-aerosol exchange step's
  summed uptake coefficient, not Kerminen and Kulmala's own integral over
  the size distribution — so NPF reads AER-03's published diagnostics
  (`condensation_h2so4_uptake_sum`, `condensation_h2so4_step_mean`) and
  must run after it, which D-A1's order guarantees;
- the **growth rate** uses the step-mean H₂SO₄ of the same exchange step,
  and the critical-cluster rate is zeroed below a 4e-16 molar floor;
- **clipping** scales the rate by `available / wanted` sulfate and drops
  it to zero below 1e-12 m⁻³ s⁻¹.

**Units in the growth expressions are inferred, not quoted.** MAM's page
labels `GR` as m s⁻¹, but with its own constant `3.0e-9` and inputs the
expression is Kerminen and Kulmala's eq. 20 and comes out in nm h⁻¹;
`γ` is then in nm² m² h⁻¹ and `η = γ CS'/GR` in nm, the only reading
under which `η/D` with `D` in nm is dimensionless. That reading is
implemented, documented in the module docstring, and pinned by
`test_eta_is_of_order_nanometres` — a wrong reading is off by 3600 or
1e9, not by a tolerance.

**No numeric Vehkamäki benchmark exists outside the paper's figures.**
The binary pathway is therefore pinned on structure: validity clipping
(not extrapolation), monotonicity in vapour and humidity, a cold/warm
ratio above 1e10, and critical clusters of 1–200 molecules and 0.2–3 nm.
AER-12's A/B against `mam4-jax` is the numeric check, if D-A3 allows it.

### AER-04: the growth-rate floor is the same float32 trap again

`gamma·sink / max(GR, tiny)` has cotangent `−gamma·sink / tiny²`, and
`1e-60` underflows in float32 to a division by zero — measured as a NaN
gradient through the full core at an empty mode with no gas, exactly the
pattern of the AER-02 finding. Fixed with the same double-`where`
(divide by 1.0 in the dead branch) and a 1e-9 nm h⁻¹ floor below which
nothing survives; pinned by
`test_gradient_finite_in_float32_with_no_gas_and_an_empty_mode`.

### AER-04: AER-03's gas update could go negative by rounding

The first end-to-end probe produced a NaN in the Aitken update. Two
causes, both fixed: the condensation step updated the vapour as
`q − condensed + dt·production`, which can round to a value of order
−1e-17 when the vapour is fully depleted, and the NPF clipping then
divided by a `wanted` sulfate that is zero wherever nothing forms. The
condensation update now books the solved end-of-step vapour directly
(`gases[h2so4] += (q_new − q − dt·production)`), so it is non-negative by
construction and `condensation_test` asserts it; the clipping guards
both branches.

### AER-04: a zero Aitken increment at heavy aerosol load is physics, then float32

The probe fixture (1e6–1e9 kg⁻¹ in every mode, 1e-9 kg/kg per species)
has a condensation sink of 0.03–0.08 s⁻¹ — a polluted boundary layer —
and the Kerminen–Kulmala survival there is 1e-250 or smaller: float64
gives `J_app` of 1e-256 to 1e-42 cm⁻³ s⁻¹ and float32 underflows it to
zero. Scaling the fixture's numbers by 1e-2 (sink 4e-4–1e-3 s⁻¹) gives
`J_app` of 270–480 cm⁻³ s⁻¹ with float32 and float64 agreeing to 1e-6,
and the clipping then limits the booked sulfate to what the vapour
holds. The remaining float32 effect is the ordinary one: an increment
of tens of particles per kg is below the ulp of a 1e8–1e9 kg⁻¹ number
and is lost at the add. That is a property of accumulating tendencies in
float32, not of this process; running the aerosol core in float64
is the standing recommendation.

### AER-05: the reference is E3SM's Fortran, because MAM's coagulation pages are empty

The MAM documentation site that carried AER-03 and AER-04 has
`coagulation/coag_equations`, `coag_numerics`, `coag_verification`,
`aging/` and `amicphys_coupling/` as title-only stubs (fetched
2026-10-01). The formulation therefore comes from E3SM's
`modal_aero_coag.F90` (BSD-3 — a formulation reference, unlike the GPL
`mam4-jax`), which is also what the documentation would have described.
What it pins down that the Liu et al. (2012) supplement does not:

- `pair_option_acoag = 3`: three donor→receiver pairs, Aitken→accum,
  pcm→accum, Aitken→pcm, with accumulation the *effective* receiver of
  every species (pcm has no sulfate or SOA slot). Species map to the same
  species in the receiver; a species with no slot there would be lost
  (not the case for MAM4's membership, and the code mirrors the rule).
- Receiver first: accumulation loses number to itself only,
  `N/(1 + Δt β N)`; then pcm (self + accumulation at its step mean);
  then Aitken (self + both at their step means). No mode gains number.
- Three closed forms of the donor solution, switched at
  `|a + bN| < 0.01` and `|a| < 0.001`; the general form is the exact
  solution of `dN/dt = −(aN + bN²)` and the other two are its
  well-conditioned limits.
- Coefficients on the **wet** diameter and wet density of the
  start-of-step state; the 3rd-moment transfer is divided by the donor's
  `dg³ exp(4.5 ln²σ)` to make a first-order volume loss rate.
- The 3-hour coagulation frequency in the comments is dead: `deltat =
  deltat_main`, every step.

### AER-05: CMAQ's nine-term free-molecular form, and why it is kept

`getcoags` writes the last term of Whitby's H.7a as `2 r⁻¹ es_i(4) +
es_j(1)`; the moment expansion of `(d_i + d_j)² (d_i^{-3/2} + d_j^{-3/2})`
gives `2 r⁻¹ es_i(4) es_j(1)` — one term, not two. The tables settle which
to use: at σ_g = 1 the eight genuine terms sum to 8 against the exact
`4√2`, and `bm0 = 0.70710678 = 4√2/8` exactly; the intermodal table has
`bm0ij(1,1,1) = 0.628539 = 4√2/9`, i.e. it was **fitted with the
nine-term form** and absorbs it at that corner. At MAM4 geometry the
nine-term form is 1.6 % (Aitken→accum), 5.9 % (Aitken→pcm) and 4.3 %
(pcm→accum) above the eight-term one before the table and the harmonic
mean, so swapping the expression under the fitted table would make the
result *worse*, not better. Kept as MAM has it, pinned by
`test_monodisperse_corners_are_the_classical_constants`.

### AER-05: the fast method against quadrature, measured

With the per-pair kernel as the harmonic mean of the free-molecular and
near-continuum forms (Binkowski and Shankar 1995, A5/A6) integrated by
brute force over both log-normals, the fast coefficients at MAM4
geometry (285 K, 900 hPa) are **+3.4 to +7.4 %** high: Aitken→accum
β₀ +7.4 %, β₃ +6.9 %; Aitken→pcm +4.9/+4.3 %; pcm→accum +5.6/+4.2 %;
self-coagulation +3.4 to +4.8 %. Over a sweep of donor 10–60 nm,
receiver 60–1000 nm, σ_g 1.6–2.0 the worst case is 14.3 % (β₀) and
11.8 % (β₃). The bias is the method's — a harmonic mean of modal
integrals is not the integral of the harmonic mean — and is the price
MAM pays for closed forms; the test holds it at 10 % at MAM4 geometry.
In the monodisperse limit the forms are exact to float32 rounding.

### AER-05: the 3rd-moment limits underflow float32

Whitby's H.7b/H.10b carry the donor's `d³` (~2e-23 m³ for Aitken) in
both limits, which puts each at ~1e-37 and their product in the
harmonic mean at ~1e-75: float32 returns `β_ij3 = 0` exactly (measured
2026-10-01 at MAM4 geometry — every mass transfer silently off). `d³`
is factored out before the harmonic mean, which is algebraically
identical and keeps both limits ~1e-16. Pinned by
`test_third_moment_coefficient_survives_float32`.

### AER-05: what crosses the seam to ageing, and the operator split it implies

MAM computes the aged fraction of primary carbon twice per step, inside
gas-aerosol exchange from the condensed shell and inside coagulation
from the coagulated shell (`vol_shell`, the Aitken sulfate+SOA whose
partner was pcm, with the SOA scaled by `soa_equivso4_factor`). This
scaffold records both deposits in `population.coating` as mass by
species and lets AER-06 compute the fraction once from their sum. That
is a different operator split from MAM's — one transfer at the end
instead of two in sequence — equivalent to first order in the step; it
is the D-A1 decision applied, and AER-06 is where to revisit it if the
single-column BC lifetime comes out wrong. The non-hygroscopic Aitken
species that coagulate onto pcm (sea salt, marine organics) are routed
to accumulation with the rest and not counted as shell, as in MAM.

### AER-05: the scaffold criterion narrows again

Through AER-04 a populated population with no gas and no SOA was inert
under the full core and bitwise equal to the placeholder; coagulation
acts on any two particles, so that criterion now holds only on a clean
atmosphere (empty modes, no gas). `modal_test` pins both the surviving
criterion and the reason it narrowed.

### AER-06: MAM4 ages at eight monolayers, not three

The card said 3, from the Liu et al. (2012) supplement. That is MAM7's
value. MAM4's paper is explicit (Liu et al. 2016, §2): "A criterion of
eight monolayers of sulfate or equivalent amount of secondary organic
aerosol (SOA) with the same increase in volume-weighted hygroscopicity as
sulfate is required to convert POM/BC particles in the primary carbon
mode to the accumulation mode", with the set-1 experiments MAM4L1/L2/L4/L8
spanning 1–8 and L8 the production choice; and
`n_so4_monolayers_pcage = 8.0_r8` in both E3SM's and CESM's
`modal_aero_gasaerexch.F90`. The default is 8, as a differentiable leaf
(`AgeingParameters.n_so4_monolayers`), so the MAM7 value and ECHAM-HAM's
1 are one argument away for the single-column lifetime check in §5.
MAM4's paper also sets κ_POM = 0; `species.py` carries 0.010, which is
not this card's to change and is noted for AER-10.

### AER-06: the criterion's own float32 trap, and the transfer cap that was not one

MAM forms the fraction as `tmp1/tmp2` with `tmp1 = V_shell D_g e^{2.5
ln²σ}` and `tmp2 = 6 n δ V_core`, both ~1e-21 per kg of air. The
quotient's reverse derivative is `−tmp1/tmp2²`, and `tmp2²` ~ 1e-42 is
flushed to zero on CPU float32: measured 2026-10-01 as `−inf` for
d/d(n_so4_monolayers), the one tunable. Rewritten as `(V_shell/V_core)
· (D_g e^{2.5 ln²σ} / 6 n δ)` — the same number from two ratios of
comparably sized quantities — with a core floor of 1e-22 m³/kg (1e-19
kg/kg of carbon) below which the mode is empty for the criterion, since
`−V_shell/V_core²` itself needs a floor to stay representable. Pinned by
`check_gradients` on the criterion and a float32 d/dn test through the
process.

Separately: MAM's `xferfrac_max = 1 − 10 ε` is a float64 literal, and in
float32 it rounds to exactly 1.0, so AER-05's cap was no cap. Both
modules now take the cap in the working dtype (`_transfer_max`).

### AER-06: how the shell crosses the seam

`coating` holds mass per species; the criterion needs volume, so ageing
divides by each species' density and weights SOA by κ_SOA/κ_SO4 = 0.276
(MAM's `soa_equivso4_factor`, computed from the spec rather than
hard-coded). The deposited sulfate and SOA were already booked to
accumulation by the processes that deposited them — MAM does the same,
adding `dqdt_so4(pcm)` to the accumulation slot — so ageing moves only
the carbon species and the number, and zeroes `coating` so nothing can be
aged twice. The scaffold's one-transfer split (§9, AER-05) stands.

### AER-07: renaming from the Fortran, and what it measures growth against

`modal_aero_rename.F90` is the reference (doc-site `renaming/` and
`calcsize/` are stubs). The scheme is Easter et al. (2004) as MAM
carries it: one pair, Aitken → accumulation; the cut diameter is the
geometric mean of the two modes' volume-mean reference diameters,
`D_g e^{1.5 ln²σ}`, ≈ 55 nm for MAM4; tail fractions are
`½ erfc((ln D_cut − ln D)/(√2 ln σ))`, with `ln D + 3 ln²σ` for volume;
the number used for the diameters is relaxed into
`[V/(f D_hi³)/27, 27 V/(f D_lo³)]` (``frelax = 27``) so a number far out
of step with the volume cannot give an absurd diameter; nothing happens
below 1e-25 m³/kmol-air, below 1e-6 relative growth, or while the new
diameter is still under the reference `D_g`; past the cut, the old
diameter is taken ≤ 0.99 `D_cut` so some transfer is guaranteed. All
transcribed and pinned against a numpy copy of MAM's per-cell arithmetic
over nine cases including the clipped-number one.

MAM calls renaming with the growth process's tendency in hand
(`dqdt·Δt`), both interstitial and cloud-borne. Here the growth is the
change of the Aitken dry volume between the step start and the moment
renaming runs — `env.dry_volume_start`, a per-mode field the driver now
computes before the fold — which in D-A1's order is exactly the
gas-aerosol exchange step's deposit. Number is unchanged by exchange, so
the old and new diameters share it, as in MAM. The cloud-borne half is
D-A5, adopted in-core (§8).

### AER-07: float32 and the finite-difference check

The transfer fraction is smooth in float32 — secants along each input
agree to 1e-3 — but `check_gradients` found no usable step in a mixed
direction: its two-rung consistency test is sensitive to the ~1e-6
roughness that float32 `erfc` puts on a fraction of 0.2, which a
relative-step difference turns into 1 % disagreement between one-sided
secants. The check runs in float64 (flipped and restored, the
`physics_interface_test` precedent); the float32 property that matters
— finite reverse-mode gradients where nothing moves (empty, unchanged,
shrinking) — has its own test.

### The five processes are in; the scaffold's TEMPORARY test is gone

`test_unimplemented_processes_are_still_identities` was written to be
deleted process by process; with AER-07 its list is empty and it is
removed. What remains of the AER-01 criterion is the clean-atmosphere
equivalence to the placeholder, which still holds.

### AER-09: the ledger closes at 1e-5 in float32, and what it does not cover

With all five processes real, every mass budget (sulfur moles, organic
mass, BC, dust, sea salt) is conserved by every process to better than
1e-5 relative in float32 across the four audit columns — warm with
modest vapour, warm with heavy vapour (clipping and renaming active),
cold upper troposphere (binary NPF active) and near-empty — with and
without the cloud-borne phase; number is conserved by exchange,
renaming and ageing, only lowered by coagulation and only raised by
new-particle formation. No process needed a fix to pass: the
conservation each card asserted locally holds in composition.

The budgets are reservoir totals; they do not check the per-mode split
(which is what the processes change and the per-process tests pin), nor
water (not prognosed by the core), nor the chemistry that converts DMS →
SO₂ → H₂SO₄ (outside the core; it only has to keep the gas keys). A gate
on the per-step closure (``_modal_ledger_step_sulfur``) is cheap enough
to run in production; the per-process audit is for bisection.
