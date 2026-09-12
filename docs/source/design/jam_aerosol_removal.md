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

`Λ₁` is tabulated at construction time (`impaction.build_impaction_table`)
over CAM's growth-ratio grid — 20 nodes spaced `ln(1.25)` apart in
`D_wet/D_dry`, at CAM's reference state (273.16 K, 750 hPa, the mode's
first-species material density) — and looked up at runtime by
differentiable log-linear interpolation, clamped below the grid and
linearly extrapolated above, exactly as `modal_aero_bcscavcoef_get` does.
The integral is 50×51 terms per evaluation; tabulating is what makes the
scheme affordable, and it is why CAM tabulates too. The port reproduces
CAM's own compiled `calc_1_impact_rate` to eight significant figures
(`impaction_test.CAM_REFERENCE`).

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

## Known gaps

- Ice-sedimentation flux reaching the surface as snow carries no aerosol
  removal (the non-carrier stance in `wetdep_term`); CAM has no ice-phase
  aerosol scavenging either.
- The convective below-cloud pathway still drives every sub-cloud level
  with the surface convective precip flux rather than a per-level profile.
- Dry deposition uses a neutral log-law aerodynamic resistance; a
  Monin-Obukhov stability correction awaits a usable surface `L`.
