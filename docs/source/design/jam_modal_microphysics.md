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

### D-A3, D-A4 — still open

- **D-A3 — validation reference.** Is installing GPL `mam4-jax` into a
  throwaway venv for AER-12 acceptable as a *test-only* comparison, given
  it is never imported by shipped code?
- **D-A4 — default switch.** Does `jam_microphysics` stay `"placeholder"`
  after AER-08, with `"modal"` opt-in until validated?
