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
| **AER-03** | A2 condensation: Seinfeld & Pandis mass transfer integrated over each mode's size distribution (Binkowski & Shankar 1995), accommodation coefficient 0.65 for H₂SO₄ (Pöschl et al. 1998). H₂SO₄ irreversible, SOA reversible. Must also record the soluble mass deposited on `pcm` — AER-06 consumes it (D-A1). Conserves S and C to rounding. | AER-02 |
| **AER-04** | A1 nucleation: binary H₂SO₄–H₂O (Vehkamäki et al. 2002) **plus** the PBL first-order pathway (Sihto et al. 2006, 1.0×10⁻⁶ s⁻¹), with Kerminen & Kulmala (2002) survival to the Aitken lower bound. Competes with AER-03 for the same `g_h2so4`. Settles the PBL-height sub-decision in D-A2. | AER-03 |
| **AER-05** | A3 coagulation: CMAQ v4.6 fast/approximate algorithms (Binkowski & Roselle 2003). **Aitken, accumulation and primary-carbon only** — MAM neglects coagulation involving coarse and larger modes as much slower. Aitken↔pcm also ages, so this feeds AER-06. | AER-02 |
| **AER-06** | A4 ageing: 3-monolayer sulfate criterion on `pcm` with SOA included, `f_age = M_SO4,cond/M_SO4,age-all + M_SOA,cond/M_SOA,age-all`, transferring that fraction of POA/BC/number to `acc`. `n_so4_monolayers` default 3.0 (Liu et al. 2012 supplement S1.1.5; ECHAM-HAM's `m7_coat` uses 1.0). | AER-03, AER-05 |
| **AER-07** | A5 renaming: transfer the upper tail of `aitken` number and mass to `accum` after continuous growth, following Easter et al. (2004). | AER-02 |
| **AER-08** | Wire `jam_microphysics="modal"` through `echam_physics()` and the Hydra config. (Classification landed in AER-01.) | AER-01, D-A4 |
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
| Coagulation | Binkowski, F., Roselle, S. (2003), *Models-3 CMAQ aerosol component – 1*, JGR 108(D6), 4183, doi:10.1029/2001JD001409 — MAM uses CMAQ v4.6's fast/approximate algorithms |
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
that pair. The driver is a fold over `self.processes`, so adding
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

(a) is the lighter option and the likely answer, but it needs the
first-step behaviour pinned down before it is chosen.

### Terminology (2026-10-01)

The H₂SO₄-vapour new-particle process is called **new-particle formation
(NPF)** in prose from here on. "Condensation nuclei activation" is a
*different* process — particles becoming droplets — and is what the
existing `ArgActivation` term already implements, so using it for NPF
would mislabel the physics. The function in `modal.py` keeps the name the
MAM literature uses; AER-04 is the card that implements it.

## 9. Findings rolled in from AER-02

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

### D-A3, D-A4 — still open

- **D-A3 — validation reference.** Is installing GPL `mam4-jax` into a
  throwaway venv for AER-12 acceptable as a *test-only* comparison, given
  it is never imported by shipped code?
- **D-A4 — default switch.** Does `jam_microphysics` stay `"placeholder"`
  after AER-08, with `"modal"` opt-in until validated?
