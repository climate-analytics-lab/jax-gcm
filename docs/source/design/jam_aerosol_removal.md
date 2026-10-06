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
`"ham_below_cloud"`, not `"ham"`, DELIBERATELY: with both in-cloud
pathways now also ported (follow-ups A and B below), `"ham"` is the full
`nwetdep=3` scheme and `"ham_below_cloud"` is kept as a genuinely
narrower configuration — below-cloud only, implicit in-cloud treatment —
for an ablation run that isolates the below-cloud change on its own. The
convective below-cloud pathway above is unaffected by either selector (it
already mirrors HAMMOZ's own convective form with the Slinn coefficient).

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
`configure_wetdep_hydro_diagnostics(bool)` flag that publishes the
per-level recurrence as a plain `"precip_cover"` diagnostics key (plus
`"pfrain"`/`"pfsnow"`, and now `"reffl"`/`"reffi"` for follow-up B below)
only when set; `echam_physics` sets it exactly when `jam_wetdep_scheme`
selects a HAM pathway that needs them. The published value is the recurrence's POST-update
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
## HAM in-cloud nucleation scavenging (`ham`, #1017 follow-up A)

`WetScavenging(scheme="ham")` additionally replaces the STRATIFORM
in-cloud nucleation pathway above (the implicit activated-fraction
treatment) with ECHAM-HAM's own aerosol-size-dependent `ic_scav_nuc`
(`mo_ham_wetdep.f90:684-795`, `jcm.physics.aerosol.jam.wetdep.ham_nucleation`)
for the three M7 soluble activating modes (KS/AS/CS; `ic_scav_nuc` itself
zeroes every other mode, `mo_ham_wetdep.f90:707-710` — it is M7-specific by
construction, "made unuseable if an alternate aerosol microphysics scheme
is used", per the reference's own comment). This selector was named
`"ham_nuc_bc"` while this slice (nucleation) and follow-up B (impaction,
next section) landed separately; now that both are in, it is `"ham"` —
see the next section's combination and the below-cloud section above for
why `"ham_below_cloud"` is kept as a separate, narrower selector.

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

**Validation.** Two compiled references, not one. `ham_m7_invertlogtail` —
the one genuinely new piece of math — is checked standalone against the
compiled, unmodified routine on 33 designed cases spanning its small/mid/
huge-tail branches and both M7 sigmas, exactly (float64 round-off, 1.6e-16;
float32 1.8e-3 at the single most extreme case). The surrounding chain —
the UNMODIFIED `ic_scav -> get_icscavfrac -> ic_scav_nuc` itself, plus
`ham_m7_logtail` and the normal CDF it calls, all compiled together — is
checked separately on 19 designed M7 columns (liquid-only/ice-only/mixed-
phase; cdnc/icnc and na on both sides of their gates; KS/AS/CS emptied in
turn; ARG and Lin & Leaitch radius selection; the huge-tail clip), matching
jcm's `water_phase_xie`/`ice_phase_xie`/`nucleation_scavenged_fraction`
exactly (float64, measured max relative error 0.0 — not merely within
tolerance) in `ham_nucleation_test.py::test_full_chain_matches_compiled_
icscavnuc_reference`, against `jcm/data/test/echam_cloud_reference/
icscavnuc.npz`.

(ham-impaction-scavenging)=
## HAM in-cloud impaction scavenging (`ham`, #1017 follow-up B)

`ic_scav_imp` (`mo_ham_wetdep.f90:798-960`,
`jcm.physics.aerosol.jam.wetdep.ham_impaction`) is the other half of
`kscavICtype=3`'s in-cloud scavenging: a bilinear lookup of a Croft et
al. (2010) collection coefficient against (collector radius, aerosol
radius), where the collector is the cloud-droplet effective radius
(water) or the ice-plate effective radius (ice) — ECHAM's own
`reffl`/`reffi` 2M-microphysics streams, NOT the radiation term's
independently-formed `clouds.r_eff_*` (see
`Lohmann2MMicrophysics.configure_wetdep_hydro_diagnostics`'s docstring
for why these are deliberately different quantities for different
consumers). Water's table value is the scavenged fraction directly;
ice's is a coefficient rescaled by `1 - exp(-coef·1e-6·ICNC·Δt)` — the
two phases are NOT unified to the same form, because the reference
itself does not. The aerosol-radius axis and bin index
(`aerosol_radius_bin`) are the SAME ones the below-cloud pathway and
follow-up A's nucleation path already use (`mo_ham_wetdep.f90:262-286`,
the `mr`/`indexy1`/`indexy2` block every `kscavBCtype=3`/`kscavICtype=3`
caller shares) — reused, not re-derived. The four interpolation corners
are gathered with the SAME (row2,col1)/(row1,col2) swap the below-cloud
`bc_rain` harness found (`ham_below_cloud.py`'s `lookup_swapped_corners`
docstring) — confirmed the identical quirk, not a coincidence, by this
slice's own compiled harness, so it is reused rather than re-derived too.

**No activating-mode gate.** Unlike `ic_scav_nuc`, `ic_scav_imp` has no
`IF (kmod < 2 .OR. kmod > 4) RETURN` in the reference — every M7 mode,
soluble or not, gets an impaction term. `WetScavenging` computes it once
per mode (regardless of `can_activate`) and adds it into every branch of
the per-mode dispatch, including the insoluble (NS/KI/AI/CI) and the
explicit-cloud-borne-interstitial branches that previously got nothing
in-cloud at all.

**Cloud-borne split.** ECHAM has no cloud-borne/interstitial distinction
(M7 carries one tracer per mode); jcm's is a differentiability-motivated
addition (#602). Nucleation represents aerosol already incorporated into
a droplet — cloud-borne by this split — and impaction represents
still-interstitial particles being swept up; with an explicit cloud-borne
phase, nucleation's full rate goes to the cloud-borne tracers (unchanged)
and impaction's share goes to the interstitial partner (new). This
mapping is this port's own physical reading, not something ECHAM can
confirm directly (it has nothing to confirm against), but the two
pathways' physical pictures are unambiguous once stated this way.

**Combination.** `get_icscavfrac` sums the two in-cloud fractions and
clips the SUM, not each fraction separately first:
`pfrac = clip(pfrac_nuc + pfrac_imp, 0, 1)` (`mo_ham_wetdep.f90:673-679`
— `pfrac_nuc`/`pfrac_imp` are ALSO separately clipped on the following
two lines, but `pfrac` itself is built from the unclipped sum, confirmed
by reading the statement order). `WetScavenging` reproduces this exactly
per (water/ice phase, number/mass tracer): `jnp.clip(fn_water +
fi_water_num, 0, 1)` etc., before multiplying by `rate_water`/`rate_ice`.

**`cdroprad(6)` reads 0.0, not 30.0.** ECHAM's cloud-droplet radius axis
is `(0, 5, 10, 15, 20, 25, [0], 35, 40, 45, 50)` µm
(`mo_ham_wetdep_data.f90:299-301`) — index 6 breaks the otherwise-regular
5 µm spacing, confirmed in the COMPILED module's own printed output (not
merely the source listing), so this is a genuine upstream data value, not
a transcription slip on this side. It is a suspected upstream typo (the
regular spacing implies 30.0), and jcm's default is the corrected axis,
30 µm at index 6 (`CDROPRAD_UM_TYPO_CORRECTED`, the
`WetDepParameters.default()` value), by maintainer decision 2026-10-06:
a 0.0 node inside an otherwise monotone axis interpolates every droplet in
`[25, 35)` µm against a spurious zero radius, which no physical reading
supports. The axis is an overridable, differentiable parameter
(`WetDepParameters.cdroprad_um`); passing `CDROPRAD_UM_AS_COMPILED`
reproduces r7492 exactly for like-for-like comparisons, and the
compiled-reference tests do so. Measured effect
(`ham_impaction.measure_cdroprad_bug_6_effect`): a lookup lands on the
disputed node whenever `reffl` (the droplet effective radius) falls in
`[25, 35)` µm — a range ordinary warm-cloud droplets (effective radii
commonly 10-20 µm, occasionally larger in maritime/drizzling cloud) do
reach — and there, the AS-COMPILED-vs-typo-corrected relative difference
in the water impaction fraction peaks at 22% (`reffl=30`, exactly on the
node) and falls off to a few percent at the bin's edges; outside
`[25, 35)` the two readings are identical by construction (the node is
never interpolated against). This is a real, occasionally material
effect, which is why the default carries the corrected node and the r7492
value is kept one override away.

**Validation.** Extends follow-up A's full-chain harness so the REAL
`ic_scav_imp` runs (follow-up A's harness stubbed it to 0): the same
compiled, unmodified `ic_scav -> get_icscavfrac -> {ic_scav_nuc,
ic_scav_imp}` chain, plus `scavcoef_bilinterp`
(`mo_ham_tools.f90:424-528`) and the REAL `mo_ham_wetdep_data.f90` (not a
stub), on 27 designed M7 columns — follow-up A's 19 (now also exercising
impaction) plus 8 new ones targeting the water/ice collector-radius bins
(including the disputed node), the ice three-regime index, and the
`ICNC<ε` gate. `get_icscavfrac`'s own `pfrac_nuc`, `pfrac_imp` AND the
COMBINED `pfrac` are all captured directly (not recovered indirectly
through `ic_scav`'s wrapper). jcm's
`water_phase_xie`/`ice_phase_xie`/`nucleation_scavenged_fraction`
(follow-up A) plus `water_impaction_fraction`/`ice_impaction_fraction`
(this slice), combined exactly as above, match all 54 rows exactly
(float64, measured max relative error 0.0) in
`ham_impaction_test.py::test_full_chain_matches_compiled_icscavimp_
reference`, against `jcm/data/test/echam_cloud_reference/icscavimp.npz`.

## Known gaps

- Ice-sedimentation flux reaching the surface as snow carries no aerosol
  removal (the non-carrier stance in `wetdep_term`); CAM has no ice-phase
  aerosol scavenging either.
- Dry deposition uses a neutral log-law aerodynamic resistance; a
  Monin-Obukhov stability correction awaits a usable surface `L`.
- `ham_below_cloud` ports only the below-cloud pathway (a deliberate,
  narrower ablation configuration, not an in-progress one — see that
  section above); `"ham"` carries the full `nwetdep=3` scheme now that
  jax-gcm#1017's follow-ups A (nucleation) and B (impaction) have both
  landed.
- The 2M scheme's `precip_cover` disagrees with ECHAM's `clcpre` on three
  synthetic test columns, for a reason not fully diagnosed (jax-gcm#1036).
