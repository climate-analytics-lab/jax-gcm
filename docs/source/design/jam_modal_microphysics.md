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
| **AER-01** | `ModalMicrophysicsCore` skeleton: subclass, `spec = MAM4_SPEC`, process-order scaffold, `_jam_state` written via the existing κ-Köhler diagnostics, zero tendency. Bitwise-equal to `PlaceholderMicrophysics`. | — |
| **AER-02** | Modal moment helpers: dry/wet diameter from mass+number, mode surface area, the M₀/M₃ conversions every later card needs. Property tests (moment self-consistency, σ_g invariance). | AER-01 |
| **AER-03** | A2 condensation: Fuchs–Sutugin transition-regime mass transfer of `g_h2so4`/`g_soag` onto all modes, partitioned by condensation sink. Conserves S and C to rounding. | AER-02 |
| **AER-04** | A1 nucleation: binary H₂SO₄–H₂O rate, plus the survival correction from the critical cluster to the Aitken lower bound. Competes with AER-03 for the same `g_h2so4`. | AER-03 |
| **AER-05** | A3 coagulation: Brownian kernel, intra-modal number sink and inter-modal mass/number transfer over the MAM4 mode pairs. | AER-02 |
| **AER-06** | A4 ageing: monolayer coating criterion on `pcm`, transfer to `acc`. `n_so4_monolayers` knob, default 3.0 (matching `Mam4JaxMicrophysics`; ECHAM-HAM's `m7_coat` uses 1.0). | AER-03, AER-05 |
| **AER-07** | A5 renaming: `aitken` → `accum` when the mode grows past `dgnum_hi`. | AER-02 |
| **AER-08** | Wire `jam_microphysics="modal"` through `echam_physics()` and the Hydra config; classify the term in `layout_audit_test.py`. | AER-01 |
| **AER-09** | Conservation gates: per-process sulfur/carbon/number budgets as a ledger, asserted in tests. | AER-03..07 |
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

1. `ruff check .` clean, pinned 0.15.17.
2. `pytest jcm/physics/aerosol/` green — the 428-test baseline must not
   regress, since `jam_microphysics="placeholder"` stays the default until
   AER-08.
3. Per-process analytic checks: condensation against an isolated single-mode
   solution, coagulation against the known self-preserving distribution,
   nucleation rate against published tabulated values at fixed (T, RH,
   [H₂SO₄]).
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

## 7. References — TO VERIFY BEFORE IMPLEMENTATION

Listed from memory and **not yet checked against the papers**. Each must be
confirmed before it enters a docstring or the science register, because a
parameterization doc that miscites its own scheme is worse than one with no
citation.

- Vehkamäki et al. (2002), JGR 107(D22), 4622 — binary H₂SO₄–H₂O nucleation.
- Kerminen & Kulmala (2002), J. Aerosol Sci. 33 — apparent-vs-real
  nucleation rate / survival to a larger size.
- Fuchs & Sutugin (1971) — transition-regime correction.
- Binkowski & Shankar (1995), JGR 100(D12) — modal coagulation moments.
- Liu et al. (2012), GMD 5, 709–739 — MAM3/CAM5; Liu et al. (2016), GMD 9,
  505–522 — MAM4 four-mode.
- Vignati et al. (2004), JGR 109, D22202 — M7, for the Phase-B mode-choice
  comparison.
- Abdul-Razzak & Ghan — already cited by `activation/arg.py`; reuse that
  citation rather than restating it.

## 8. Decisions needed before AER-01

- **D-A1 — process order.** MAM4 runs gas exchange, rename, nucleation,
  coagulation, ageing within one substep. Adopt that order as-is, or
  sub-cycle the stiff nucleation/condensation pair?
- **D-A2 — nucleation scheme.** Binary H₂SO₄–H₂O only (MAM4's default), or
  also a boundary-layer ternary/organic pathway? Binary alone understates
  continental Aitken number.
- **D-A3 — validation reference.** Is installing GPL `mam4-jax` into a
  throwaway venv for AER-12 acceptable as a *test-only* comparison, given
  it is never imported by shipped code?
- **D-A4 — default switch.** Does `jam_microphysics` stay `"placeholder"`
  after AER-08, with `"modal"` opt-in until validated?
