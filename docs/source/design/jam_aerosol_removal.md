# JAM aerosol removal: settling, dry deposition and scavenging

JAM removes aerosol through three composed terms, in this order:

| term | category | what it removes |
|---|---|---|
| `StokesSedimentation` | `aerosol_sedimentation` | gravitational settling, whole column |
| `SlinnDryDeposition` | `aerosol_drydep` | turbulent + Brownian removal, lowest layer |
| `WetScavenging` | `aerosol_wetdep` | in-cloud nucleation + below-cloud impaction, net of re-evaporation |

Every formulation below is matched against CAM's MAM4 (ESCOMP/CAM
`cam_development`); where jcm deviates, the deviation is stated.

## Below-cloud impaction: the Slinn collection integral

CAM `wetdepa_v2`'s below-cloud term is

```
odds  = min(1, precab / cldv · Λ₁ · Δt)
srcs2 = sol_factb · cldv · odds · q / Δt
```

The swept precipitating volume `cldv` cancels against the in-precip-area
rain rate inside `odds`, leaving a first-order removal rate

```
Λ = sol_factb · Λ₁(D_wet) · R          [1/s]
```

with `R` the precipitation flux (kg m⁻² s⁻¹ ≡ mm s⁻¹) and `Λ₁` the
impaction scavenging coefficient in 1/mm. There is no cloud weighting
left: it acts on the whole grid-mean interstitial mixing ratio. That is
consistent because interstitial aerosol is by definition the out-of-droplet
population, and — exactly as in CAM, where `sol_facti = 0` for interstitial
aerosol — jcm's stratiform in-cloud pathway acts on the cloud-borne phase,
not on it. `sol_factb` is the only reduction applied.

`Λ₁` comes from CAM's `calc_1_impact_rate` (`aero_model.F90`), a double
sum over a raindrop spectrum and the mode's lognormal, weighted by Slinn's
collection efficiency

```
E_brown     = 4(1 + 0.4 Re^½ Sc^⅓) / (Re · Sc)
E_intercept = 4χ(χ + (1 + 2μ*χ)/(1 + μ*/Re^½)),   χ = a/r,  μ* = 60
E_impact    = ((St - S*)/(St - S* + ⅔))^{3/2}     for St > S*
E           = min(E_brown + E_intercept + E_impact, 1)
```

integrated separately against the number and volume weights, giving the
two coefficients CAM carries as `scavcoefnv(:,:,1)` and `(:,:,2)` and
applies to the number and mass mixing ratios respectively.

Two properties matter and neither survives an `r²` parameterisation:

- **the Greenfield gap** — Brownian collection falls and inertial
  collection rises with size, so `Λ₁` has a *minimum* near 0.1–0.5 µm;
- **saturation** — `E ≤ 1` bounds `Λ₁` by the rain's geometric sweep-out
  rate, so it flattens above a few micron instead of growing as `r²`.

`Λ₁` is tabulated over CAM's growth-ratio grid — 20 nodes spaced `ln(1.25)`
apart in `D_wet/D_dry`, at CAM's reference state (273.16 K, 750 hPa, the
mode's first-species material density) — and looked up at runtime by
differentiable log-linear interpolation, clamped below the grid and linearly
extrapolated above, exactly as `modal_aero_bcscavcoef_get` does. The integral
is 50×51 terms per evaluation; tabulating is what makes the scheme
affordable, and it is why CAM tabulates too.

The table is split so the knobs stay differentiable. Everything independent
of them — the drop and aerosol geometry, the Brownian efficiency, the
inertial efficiency at unit scale — is built once per mode in float64 NumPy
(`build_impaction_table`); the knob-dependent completion, Slinn's
interception term and the impaction scale, is evaluated in JAX inside the
step (`table_log_coefficients`) on 20 nodes rather than one per cell.

Two knobs on the integral are exposed as differentiable leaves on
`WetDepParameters`, at CAM's values by default: `mu_water_air` (the
water/air viscosity ratio in interception, 60) and `impact_scale` (a
multiplier on the inertial-impaction efficiency, 1), alongside `sol_factb`.
Impaction is the most uncertain part of the scheme and these are where that
uncertainty lives. `impact_scale` raises collection monotonically;
`mu_water_air` does not — the sign of ∂/∂μ in `(1 + 2μχ)/(1 + μ/√Re)` is
that of `2χ − 1/√Re`, so it *lowers* collection for particles small enough
that `χ < 1/(2√Re)`.

The port reproduces CAM's own compiled `calc_1_impact_rate` to eight
significant figures when given CAM's constants (`impaction_test.CAM_REFERENCE`).
Production uses `jcm.constants` per CLAUDE.md, whose `ak`, `r_universal` and
`m_air` differ from CAM `mo_constants` by ≤7.4e-6 relative; that shifts the
coefficient by <1e-4 and is bounded by its own test rather than folded into
the parity tolerance.

`sol_factb` is CAM's `sol_factb_interstitial`, whose namelist default is
**0.1** for interstitial aerosol; cloud-borne aerosol gets 0 (it is
in-droplet by definition, so below-cloud collection does not apply to it).
CAM's fallback when the namelist leaves it unset is the mode's
mass-weighted hygroscopicity; no supported CAM configuration uses that
path, so jcm carries the scalar as a differentiable parameter instead.

Two harmless departures from `modal_aero_bcscavcoef_get`: CAM
short-circuits a growth ratio within 1 % of unity to node 0 exactly, where
`_interp_log_table` always interpolates (numerically indistinguishable);
and CAM gates the lookup on an `isprx` precipitation mask, where jcm relies
on `Λ ∝ R` vanishing without precip — equivalent, and why there is no mask.

### The convective carrier's footprint

CAM's cancellation above holds for the stratiform carrier, whose rain rate
`wetdepa_v2` rescales to the precipitating area. For the convective carrier
jcm follows HAMMOZ instead (PR #776). `mo_hammoz_wetdep.f90::prep_wetdep_hydro`
takes the fraction of the grid box the convective precipitation falls
through to be the **updraft area**,

```
f_cu = M_u / (ρ · w_u),      w_u = 2 m/s   (zwu)
```

and `mo_ham_wetdep.f90::ham_wetdep` removes `q_ambient · f_cu · (1 − exp(−Λ·Δt))`
from each layer, with `Λ` looked up at the grid-mean rain flux (`bc_rain`).
The updraft mass flux it sees is the one ECHAM `mo_cufluxdts.f90::cuflx`
hands on: the plume profile through the cloud and, below the cloud base,
`pmfu(jk) = pmfu(kcbot)·zzp` with `zzp = (p_s − p_half(jk))/(p_s − p_half(kcbot))`
(squared for mid-level convection) — the updraft draws on the whole sub-cloud
layer, so the shaft keeps its footprint under the base and tapers to zero at
the surface.

`conv_precip_cover` reproduces both. `ConvectionData.mass_flux_up` is the
plume profile alone (the tracer transport derives the cloud-base supply from
its jump and must keep doing so), so the sub-cloud taper is rebuilt from the
layer masses: `p_s − p_half(k) = g·Σ_{j≥k} m_j`, so the pressure ratio is the
ratio of the air mass below the two interfaces, which the term already holds
as `ρ·Δz`. The cover is clipped to [0, 1] (HAMMOZ does not clip; an updraft
area above the whole box is a closure artefact, not a cover). `w_u` is the
differentiable `WetDepParameters.conv_updraft_velocity`, floored at a
physical 0.01 m/s inside the division. The density is the environment's
where HAMMOZ uses the updraft's (`zrhou = p/(rd·ptu)`,
`mo_cufluxdts.f90:406`): `ConvectionData` publishes no updraft temperature,
and the resulting under-estimate of `f_cu`, `(T_u − T_env)/T_env`, is under
2 % against an assumed `w_u` that is the estimate's dominant uncertainty.

`conv_below_cloud_rate` applies the cover to the **removed fraction**, as
HAMMOZ does, and converts it back to the equivalent first-order rate
`−log1p(−f_cu·(1 − exp(−Λ·Δt)))/Δt` so it composes with the other pathways in
the term's batched exponential update: a step can take at most `f_cu` of a
layer's aerosol however hard it rains, and for `Λ·Δt ≪ 1` the rate is
`f_cu·Λ`.

Two things to be clear about:

- **The references disagree, by the factor `f_cu`.** CAM would have the
  shaft rain at `R/f_cu` inside the exponential, which cancels the area for
  `Λ·Δt ≪ 1`; HAMMOZ evaluates `Λ` at the grid-mean `R`, so its convective
  washout is `f_cu` (a few per cent) times CAM's. jcm takes HAMMOZ's form for
  the convective carrier: the updraft area is the physical footprint of the
  shaft, and the in-plume sink (`ConvectiveTracerTransport`) already removes
  what is inside it, so applying CAM's cancellation on top would double-count
  that in-plume removal (PR #776). The CAM-consistent
  variant (`R/f_cu` in the exponential, still capped at `f_cu` per step) is a
  one-line change here if validation says otherwise — and ECHAM's own
  sub-cloud rain evaporation argues for it: under `lham`, `cuflx` uses this
  same updraft area as its evaporation footprint and evaluates the rain
  intensity inside the shaft, `sqrt(zrfl/zcucov)`. jcm's convection scheme
  matches that: with the JAM chain composed it takes the updraft area as the
  sub-cloud evaporation cover through the shared `updraft_area_cover`, so the
  evaporation and this washout share one footprint (and one `zwu`). The
  convection scheme divides by the true updraft density `p/(R_d·T_u)`, which
  it has; this washout keeps the environment-density stand-in noted above.
- **HAMMOZ's cloud-free gating is not ported.** `ham_wetdep` zeroes
  below-cloud scavenging wherever the stratiform cover exceeds `1e-10`. jcm's
  stratiform carrier carries no cover (CAM), and gating the convective carrier
  on the stratiform cover would couple the two carriers, which jcm deliberately
  avoids (PR #776).
  The ambient tracer scavenged is the grid-mean working copy, standing in for
  HAMMOZ's environment value `pxtenh` to O(`f_cu`).

## Wet particle density

Settling and Slinn deposition use the **wet** radius, so they must use the
density of that same wet particle: the mass-weighted mixture of dry
material and condensed water,

```
ρ_wet = (ρ_dry + (g³ - 1)·ρ_water) / g³
```

with `g = r_wet/r_dry` the κ-Köhler growth factor. CAM passes `wetdens`
from `modal_aero_wateruptake` for exactly this reason. Pairing the dry
density with the wet radius overstates the settled mass by 1.64× for
coarse sea salt at 80 % RH and 1.89× at 99 %. The MAM4-JAX core already
publishes `wetdens`; the κ-Köhler placeholder core now does too.

## Operator splitting across the removal chain

`ComposablePhysics` hands every term the **step-start** state and sums the
tendencies it returns. Each removal term bounds its own removal at 100 %
of what it sees — settling by the CFL cap, dry deposition and scavenging
by their implicit `1 - exp(-Λ·Δt)` update — but three independently-bounded
sinks can sum past the available mass; a raining marine surface cell
removed 164 % of its coarse sea salt.

ECHAM and CAM avoid this by operator splitting, and so does jcm: each
removal term reads the working copy the previous terms left, reconstructed
from the running tendency `ComposablePhysics` publishes as `_tendency_run`
on both its whole-grid and column-vectorized hosts
(`removal_split.split_view`). Removing a fraction of what remains can never
exceed the whole, and each term still reports exactly the mass it took.

The reconstruction folds in **every** term already run this step, not only
the removal chain: emissions, convective transport, chemistry, the
microphysics core and activation all precede sedimentation in
`jam_aerosol_physics`. That is the full sequential split, and it is what
the removal terms want — aerosol emitted or formed this step is there to be
removed, and aerosol convection has already exported is not.

Sequential splitting was chosen over a joint limiter that rescales the
three tendencies to fit: the limiter changes the answer only in the cells
where it binds and leaves the unlimited cells double-counting the aerosol
each process sees, whereas splitting is the treatment the reference models
use everywhere and needs no cross-term communication. Its one cost is
order dependence — settling, then dry deposition, then scavenging — which
is the order the terms are composed in and the order the processes act in
physically.

Cloud-borne tracers live in the physics carry, which the removal terms
already integrate sequentially through `cloud_borne_store.apply_updates`,
so they need no reconstruction.

The splitting is the *only* bound on the removal sum, by design.
`physics_interface.verify_tendencies` deliberately does **not** cap aerosol
or gas tendencies: their tendency is a sum over conservative
redistributions (tracer vertical diffusion, convective transport) and
paired transfers (sulfur chemistry, the activation exchange), and a
per-cell positivity cap clips one side of a conserved pair — clamping a
donor cell while the receiving cells keep their gain creates column mass.
Bounding the removal where it is produced is what makes the cap
unnecessary, and removing it is also what makes the split *exact*: with a
cap in place `split_view` reconstructed the working copy from tendencies
the interface would later clamp, so the reconstruction disagreed with the
state the dycore actually received wherever the cap fired.

What happens to a cell the parallel transport terms leave negative: nothing,
within physics. It persists through the dynamics step and is cleaned on the
way back in by the dycore-side `filters.MassConservingPositivity`, a
column-mass-conserving hole-filler enabled by default for JAM runs
(`diffusion.tracer_positivity: auto`) and wired into the dinosaur backend
only — under pySES, or with it switched off, the negative persists and stays
visible to the mass-budget gauge. It is a boundary guard, not
positivity-preserving transport.

## Deposition-flux ledger

`dry_<species>` is gravitational settling **plus** turbulent/Brownian
surface deposition; `wet_<species>` is scavenging net of re-evaporation,
plus the in-plume convective scavenging the transport term performs. Each
term column-integrates its own tendencies
(`flux_diagnostic.accumulate_deposition_fluxes`), so the ledger cannot
drift from the mass actually removed: with operator splitting in place,
`dry_* + wet_*` equals the chain's total mass change, up to the interface
guard below (each term records its ledger before `verify_tendencies` sees
the summed tendency).

(ham-below-cloud-scheme)=
## The HAM below-cloud scheme (`ham_below_cloud`, #1017)

`WetScavenging(scheme="ham_below_cloud")` swaps the STRATIFORM below-cloud
pathway above for ECHAM-HAM r7492's own `nwetdep=3` scheme
(`mo_ham_wetdep.f90::bc_rain`/`bc_snow`, `ham_below_cloud.py`): a bilinear
lookup against Betty Croft's size-dependent rain and snow collection tables
(`mo_ham_wetdep_data.f90`), rather than CAM's Slinn integral. Named
`"ham_below_cloud"`, not `"ham"`: the in-cloud pathways (nucleation,
impaction) are unaffected and still run as the `"jcm"` scheme does under
either setting, and the convective below-cloud pathway above is also
unaffected (it already mirrors HAMMOZ's own convective form with the
Slinn coefficient). The remaining in-cloud pathways are tracked as
jax-gcm#1017's follow-ups A (nucleation) and B (impaction).

**`pclc`.** HAM's removal acts only within the precipitating fraction of
the box — `pxtp10·pclc·(1 − exp(−Δt·(sfrain+sfsnow)))`
(`mo_ham_wetdep.f90:434-437`) — unlike CAM's form, whose swept-volume
cancellation makes the cloud weighting a no-op (see `below_cloud_rate`'s
docstring above). `pclc` traces to `mo_submodel_interface.f90`'s
`pclcpre`, ECHAM's cumulative max-overlap precipitating-area recurrence,
computed internally by the Lohmann 2M scheme but not previously published.
Rather than add a field to `CloudData` — which would change the
checkpoint pytree of every 2M composition and break restart from existing
stamped checkpoints — the 2M scheme gained a static
`configure_precip_cover_diagnostic(bool)` flag that publishes the
per-level recurrence as a plain `"precip_cover"` diagnostics key only when
set; `echam_physics` sets it exactly when `jam_wetdep_scheme=
"ham_below_cloud"`. The published value is the recurrence's POST-update
state for the current level (`mo_cloud_micro_2m.f90:1719-1742`), i.e. the
cover ECHAM itself passes into `cloud_subm_2` for that level.

Checked against a compiled number: the existing 2M Fortran reference
(`cloud2m_T63L47.npz`) already carries ECHAM's own `zclcpre` as
`diag/<step>/clcpre`, and comparing against it
(`lohmann_2m_fortran_reference_test.py::test_precip_cover_matches_echam_clcpre`)
found the two agree everywhere but three synthetic test columns designed to
stress ice-number diagnosis, mixed-phase detrainment and sedimentation —
not realistic precipitating cells. One cause is identified and benign (a
pre-existing, documented 1e-9-vs-ECHAM's-1e-12 flux floor in
`cloud_utils.gridbox_falling_hydrometeor`, kept for its reverse-mode VJP
safety); the rest is open as jax-gcm#1036. The test is `xfail(strict=True)`
referencing it; below-cloud scavenging only reads `precip_cover` well below
the first precipitating level in practice, so this does not block the
below-cloud slice (confirmed by a full smoke-tested end-to-end model run
with `jam_wetdep_scheme="ham_below_cloud"`).

**Reference-harness findings.** A compiled-Fortran harness driving
`bc_rain`/`bc_snow` end to end (wet radius → bin index → bilinear lookup →
rate) over 96 designed cases caught two latent defects before they shipped:

- The 50 µm wet-radius clip (`mo_ham_wetdep.f90:272`,
  `MIN(rwet_p·zrad_fac, 50e-6)`) was missing from the initial port. Harmless
  whenever the radius bin saturates at the table's last node, but
  `caerorad`'s actual top node is 83.23 µm — well above the clip — so an
  unclipped radius between the two extrapolated past the clamped value
  instead of reading it.
- `bc_rain`'s own data-filling loop assigns the bilinear interpolation's
  two off-diagonal corners (`Q12`, `Q21`) the OPPOSITE of what
  `scavcoef_bilinterp`'s formula needs for a textbook bilinear read (traced
  by deriving the formula's corner roles directly from its `lint4`
  branch). `bc_snow` is immune, since its X axis is always the dummy
  `X1=X2=1`, which is why it alone did not surface this. Ported exactly as
  compiled, flagged to the maintainer as a likely-unintentional upstream
  quirk rather than silently "corrected" — see `ham_below_cloud.py::_lookup`.

**`pfrain`/`pfsnow`.** `bc_rain`/`bc_snow` need the separate rain and snow
carrier fluxes `update_precip_fluxes` computes every level (the in-cloud,
cover-normalised, pre-evaporation flux) — and the 2M scheme already
computes them internally, threaded to ECHAM's `cloud_subm_2` as
`zfrain`/`zfsnow` (`mo_cloud_micro_2m.f90:1813`; `cloud_subm_2`'s own
comment at `mo_submodel_interface.f90:1676-1677`, "rain/snow flux before
evaporation", confirms the match). `configure_wetdep_hydro_diagnostics`
threads them out alongside `precip_cover`, under the same static flag;
`WetScavenging("ham_below_cloud")` reads them directly rather than
deriving an approximation — an earlier version of this wiring split the
stratiform carrier ledger by the in-cloud ice fraction as a proxy, which
review correctly flagged as unnecessary once the exact quantity was
located.

(ham-nucleation-scavenging)=
## HAM in-cloud nucleation scavenging (`ham_nuc_bc`, #1017 follow-up A)

`WetScavenging(scheme="ham_nuc_bc")` additionally replaces the STRATIFORM
in-cloud nucleation pathway above (the implicit activated-fraction
treatment) with ECHAM-HAM's own aerosol-size-dependent `ic_scav_nuc`
(`mo_ham_wetdep.f90:684-795`, `jcm.physics.aerosol.jam.wetdep.ham_nucleation`)
for the three M7 soluble activating modes (KS/AS/CS; `ic_scav_nuc` itself
zeroes every other mode, `mo_ham_wetdep.f90:707-710` — it is M7-specific by
construction, "made unuseable if an alternate aerosol microphysics scheme
is used", per the reference's own comment). Impaction (`ic_scav_imp`) is
follow-up B and still runs implicitly under every setting.

**The formula.** `ic_scav_nuc` INVERTS the mode's own lognormal tail at a
critical radius that reproduces the ACTUAL in-cloud droplet/crystal number
this step (`ham_m7_invertlogtail`, a closed-form inverse-erf
approximation — new code, since `ham_logtail`/the forward normal CDF it
also needs are already ported in `ham_activation.py`), then reads the SAME
tail forward for the tracer in question (number with `mass_factor=1`,
mass with `mass_factor=cmedr2mmedr` — the identical `ham_logtail` two-call
pattern `HamActivation`'s own ARG branch already uses to turn a number
tail into a mass one). Unlike jcm's existing blended `f_comb`
(`(1-pice)*f_wat + pice*f_ice`), this scheme keeps water and ice as two
SEPARATE removal terms, each with its OWN rate and its OWN activated
fraction (`zxtwat`/`zxtice`, mo_ham_wetdep.f90:300-320) — HAM never blends
the two phases' efficiencies, only the resulting mass changes, so blending
here would be a new approximation this scheme does not need (the ledger
already carries `f_wat`/`f_ice` separately before `WetScavenging` blends
them for the other schemes).

**Water phase.** The critical-radius inversion's target is `1 -
2·clip(cdnc_incloud·ρ·frac(kmod)/na, 0, 1)` (`mo_ham_wetdep.f90:749-755`):
`na` and `frac(kmod)` are read EXACTLY as HAM's own `ham_activ_diag_
abdulrazzak_ghan_strat` publishes them for ARG (`na` = the SAME sum over
activating modes as `activated_cdnc`; `frac(kmod)` = the SAME per-mode
fraction as `_jam_activation.number_frac[kmod]`, confirmed identical by
tracing both to the one Fortran routine that sets them,
`mo_ham_activ.f90:409-416`) — this scheme reads those two diagnostics
directly for `nucleation_activation="ham_arg"`. Lin & Leaitch publishes a
DIFFERENT `na`/`frac(kmod)` from the same routine family
(`ham_avail_activ_lin_leaitch`, `mo_ham_activ.f90:606-732`) that
`HamActivation`'s own published `number_frac` does NOT equal (that term's
Lin & Leaitch `number_frac` additionally scales by `cdncact/na` for its
OWN consumer, the cloud-borne exchange term — a different contract);
`nucleation_activation="ham_lin_leaitch"` recomputes the raw `na`/`frac`
instead, reusing `ham_logtail`/`LL_CRCUT_STRAT` (the shared primitives,
not a re-port).

**Ice phase.** No activation-term coupling at all — a pure M7 size-ordered
depletion of ICNC against the three modes' own NUMBER tracers
(`mo_ham_wetdep.f90:757-778`): CS (coarse) is assumed to use up ICNC
first, then AS the remainder, then KS what is left — three literal
`IF (kmod == ...)` branches in the reference, not a general recurrence,
ported the same way.

**Validation.** `ham_m7_invertlogtail` — the one genuinely new piece of
math — is checked against the compiled, unmodified routine on 33 designed
cases spanning its small/mid/huge-tail branches and both M7 sigmas,
exactly (float64 round-off, 1.6e-16; float32 1.8e-3 at the single most
extreme case). The water/ice `xie` formulas and the orchestration
(`get_icscavfrac`'s nucleation branch) are direct, literally-cited
transcriptions checked by unit test rather than a second compiled
harness — a deliberate scope line for this slice, not an oversight: see
the follow-up A PR description for the tradeoff.

## Known gaps

- Ice-sedimentation flux reaching the surface as snow carries no aerosol
  removal (the non-carrier stance in `wetdep_term`); CAM has no ice-phase
  aerosol scavenging either.
- Dry deposition uses a neutral log-law aerodynamic resistance; a
  Monin-Obukhov stability correction awaits a usable surface `L`.
- `ham_below_cloud` ports only the below-cloud pathway; the in-cloud
  nucleation and impaction pathways under `nwetdep=3` are jax-gcm#1017's
  follow-ups A and B.
- The 2M scheme's `precip_cover` disagrees with ECHAM's `clcpre` on three
  synthetic test columns, for a reason not fully diagnosed (jax-gcm#1036).
- `ham_nuc_bc` ports only the nucleation pathway; impaction under
  `nwetdep=3` is jax-gcm#1017's follow-up B.
- `ham_nuc_bc`'s water/ice `xie` formulas and `get_icscavfrac`'s
  orchestration are validated by careful transcription + unit test, not an
  additional compiled harness beyond `ham_m7_invertlogtail` itself (see
  the section above).
