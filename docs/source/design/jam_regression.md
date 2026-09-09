# JAM aerosol regression statistics

*Issue #762, statistics half. The spun-up-state half is deliberately not done —
see "Not yet done" below.*

A JAM release-validation member is a 365-day GPU run whose aerosol can be badly
wrong for eleven months without any existing gate noticing. The August-2026
T63 L47 year is the worked example: sulfate sat inside its climatological
anchor range all year and then grew by two orders of magnitude over the final
forty days, while temperature, surface pressure and the primary species stayed
flat. A range gate with ×3 slack first fails in the last fortnight of a run
that was already unusable by day 300.

This document says which statistics `tools/release_validation/aerosol_stats.py`
computes, why each one exists, and what tolerance each is scored against.

## The statistic set

Everything is an area-weighted global mean, computed chunk by chunk (a JAM year
is ~60 GB of output and is never opened as one array) and then reduced over the
record.

| statistic | why it exists |
|---|---|
| `burden_<sp>_mg_m2` for so4, bc, du, ss, poa, soa | the loading itself, against the AeroCom-magnitude anchors. Includes the **cloud-borne** phase, which is 20–25 % of sulfate mass and which the report omitted until #762. |
| `dlnB_dt_<sp>_per_day` over the final six months | the runaway detector — see below. |
| `lifetime_<sp>_days` for so4, bc, du, ss | burden ÷ deposition separates "source too big" from "sink too small" in one number. A burden time series alone cannot. |
| `budget_residual_<sp>` and `budget_residual_max` | mass conservation: does what came in, minus what left, equal the change in what is held? |
| `so4_frac_above_500hPa` | #658 was a *free-tropospheric* accumulation: 85 % of the growth sat above 600 hPa while the boundary layer looked fine. A column burden hides that; this does not. |
| `so4_nh_sh_ratio`, `so4_burden_60_90N_mg_m2` | anthropogenic sulfate is a Northern-Hemisphere phenomenon (observed ratio ≈ 3). A run whose sulfur has migrated to the Southern Hemisphere or the pole is wrong even at the right global mean. |
| `aod_550`, `angstrom` | the radiative quantity the aerosol exists to produce, and a size-distribution check that a mass burden cannot give. |
| `cdnc_900hPa_cm3`, `N100_900hPa_cm3` | the aerosol–cloud pathway. Activation can fail while every burden is correct. |
| `r_dry_<mode>_um` at the lowest level | the modal radii the optics and the removal rates both key off. A drifting mode radius changes AOD and lifetime together, which makes it invisible in either alone. |
| `dyn_frac_per_step_<sp>` | mass the *transport* created or destroyed, per step, as a fraction of the advected mass — from the #713 in-step gauge. See "Transport, not physics" below. |

### Where the numbers come from

* Burdens integrate `q·Δp/g` over the column, summing **interstitial**
  (`m_<sp>_<mode>`) and **cloud-borne** (`jam_cloud_borne.mc_<sp>_<mode>`) mass
  over the modes carrying each species.
* Deposition is `dry_* + wet_*`. It does **not** add `conv_scav_flux.*`: the
  wet-deposition term already folds in-plume convective scavenging into
  `wet_*`, so a third term would double-count that sink.
* The sulfate budget is the whole **sulfur family**, because `emi_so4` is only a
  few per cent of the real sulfate source — the rest arrives as SO₂ and DMS and
  passes through the gas reservoirs before it is aerosol. Sources are
  `emi_so2 + emi_dms + emi_so4`, storage is the sulfate burden plus the SO₂,
  DMS and H₂SO₄ column burdens, and every carrier is converted to **sulfate
  aerosol mass** using jcm's own `so4` species — ammonium bisulfate,
  115.0 g/mol (`jam/species.py`), *not* SO₄ at 96.06. Using the wrong one
  understates the source by 20 % and turns a modest closure error into an
  apparent leak.
* SOA is not scored for closure at all. Its source is condensation of the SOAG
  gas, which has no `emi_*` channel, so a residual computed for it measures a
  missing diagnostic rather than a mass leak.

## The two tolerance tiers

**Tier 1 — absolute physics gates.** These are not tuning targets. They are
statements about what the model must do regardless of how well calibrated it is,
so they are the same number for every member and every release.

| gate | limit |
|---|---|
| `\|d ln B/dt\|` per species, final six months | `< 0.002 /day` |
| `\|budget residual\|`, max over species | `< 5 %` |
| `dyn_frac_per_step_<sp>` | `< 0.1 % per step` |

**Tier 2 — climatological anchors.** The existing per-species ranges in
`tools/jam_burden_report.py`'s `_SPECIES` table, widened by the release gate's
×3 slack. Unchanged by #762; a "did the model produce a plausible planetary
loading" check, not a calibration.

**Tier 3 — regression tolerances**, for comparing a run against a stored
reference: `max(3σ, 15 % relative)`. σ is the standard error of the record mean,
with the effective sample size taken from the chunk series' own lag-1
autocorrelation, `N_eff = N(1−a)/(1+a)`. Neither half alone works: 5-day burden
samples are strongly autocorrelated, so an uncorrected σ is far too small
(≈ 1 % of the annual mean, which every real change would trip), while a flat
15 % would let a slow bias through on a quiet statistic.

## The drift gate is the runaway detector

An aerosol runaway is multiplicative — a fixed factor per unit time — so the
right statistic is the slope of `ln B`, not of `B`. That makes one threshold
work for every species regardless of its loading, and makes a merely noisy but
stationary burden average to zero.

The window is the **final six months**. A from-zero spin-up year ramps for its
first months by design; scoring the whole record would call that ramp a
runaway. Tropospheric burdens equilibrate in weeks (lifetimes ~4–7 days), so
six months is comfortably past the ramp while still long enough that a slope of
0.002 /day is resolvable above the noise.

`0.002 /day` is a factor e over ~500 days: a burden that genuinely has not
settled may still move that much over a validation year, but the ×60-in-40-days
growth of a #658-class runaway exceeds it by an order of magnitude, and — the
point of the gate — so does the much gentler growth those runaways show for
months *before* they become visible. On the August-2026 year the gate fails on
sulfate at day 300, on a window that ends before the burden leaves its anchor
range at all.

A species the run never carries (SOA with no SOAG production, say) has a
correctly-zero burden and no drift to measure; it is skipped rather than scored
as NaN.

## The closure gate, and its sign

The residual is `(Σ emitted − Σ deposited − ΔB) / Σ emitted` over the record.
Its **sign** is the diagnosis:

* **Negative** — more mass was deposited than ever entered the column. No
  missing ledger entry can produce that. It is mass creation, and it is a defect
  in the physics.
* **Positive** — emitted mass is unaccounted for. That is what a real leak looks
  like, but it is *also* exactly what a missing sink **diagnostic** looks like.
  On output written before the #722 removal-ledger fix, `dry_*` omits the Slinn
  turbulent/Brownian deposition entirely (it captures only ~31 % of the dust dry
  sink), so a positive residual on such a run is expected by construction. The
  gate says so in its own message instead of reporting a leak.

The drift gate reads burdens only, never the deposition ledger, so it stays
trustworthy either way. That separation is deliberate: the two gates must not
fail for the same reason.

## Transport, not physics: what the August-2026 runaway actually was

The runaway this whole gate set was built to catch turned out not to be an
aerosol-physics defect at all. It was the **semi-Lagrangian tracer transport
failing to conserve mass**, and it is fixed by #720.

The mechanism is one-signed, which is why it compounded. SL interpolation does
not conserve `∫q·dp`, and under the quasi-monotone limiter the error has a
preferred direction wherever a strong sink leaves sharp minima: clipping an
interpolation undershoot at a minimum can only *add* mass. Strong scavenging
and sedimentation produce exactly that state, so every step added a little
sulfate and dust, and a per-step error of order 1e-3 became orders of magnitude
over a year.

Two consequences for these gates:

* The **`budget_dyn_<sp>` gauge (#713) is the direct measurement**, and it is
  what `dyn_frac_per_step_<sp>` scores. The burden-drift gate sees the symptom
  months later; this sees the cause in one chunk. The limit is set at the
  honest per-step magnitude (0.1 %) rather than at the fixer's tolerance,
  because the failure is *systematic* accumulation, not the size of any single
  step's error.
* dev's remedy, `_fix_nodal_tracer_mass` (`jcm/dycore/dinosaur/dycore.py`), is
  an ECMWF-style **proportional global mass fixer**: one factor per tracer per
  step, clipped to `[2/3, 1.5]`, restoring the post-transport total to the
  pre-transport one. It stops the accumulation, but it is a *global rescale* —
  it does not repair the **distribution**, so mass wrongly moved between
  regions stays wrongly placed and only the total is corrected. **#521
  (positive-definite tracer transport) remains the faithful cure**; the fixer
  is the stopgap that makes year-long runs usable meanwhile. The gate is set
  ~500× tighter than the fixer's clip so it fires long before the fixer
  saturates.

**Follow-up.** The fixer's rescale factor is computed inside
`_fix_nodal_tracer_mass` and is **not exported**, so a run cannot be checked
for the factor reaching its clip — the condition the fixer's own docstring says
"must surface in the `budget_dyn_*` gauge". Exporting it as a per-tracer
diagnostic would let the gate distinguish "transport error the fixer absorbed"
from "transport error the fixer could not absorb"; until then the gate scores
`budget_dyn` alone, which is the post-fix residual.

The gate is scored only when the run saved a Hydra config to read its timestep
from, and only on output that carries the gauge at all — pre-#713 runs, the
August-2026 year included, have neither, so on those the block reports the gate
as unscored rather than silently passing it.

## Both output vertical conventions

Per the "Inspecting model output" rule in `CLAUDE.md`, no statistic here selects
a level by a bare index.

* A level near a target pressure (900 hPa for the number diagnostics, the
  lowest layer for the modal radii) is found by searching the file's own
  `pressure_full` for the nearest mean pressure.
* "Above 500 hPa" is a mask on the 4-D `pressure_full` field, not a slice.
* Layer thicknesses come from `jam_burden_report._layer_dp`. Post-#710 files run
  both vertical axes surface-first, which is what `jcm.analysis`
  assumes. Pre-#710 files store interfaces **TOA-first** while their `level`
  fields stay surface-first, so differencing the interfaces yields a Δp reversed
  against the tracers — pairing stratospheric layer masses with the boundary
  layer. Those files are detected from the pressure values themselves rather
  than from the axis labels (a pre-#710 `level_i` is a bare integer index with
  no `positive` attribute to read) and the Δp is reversed.

The size of that second defect is worth recording: on the August-2026 year the
uncorrected Δp gives a column-integrated water vapour of 1.1 kg/m², against
29.5 kg/m² corrected. Every burden computed from a pre-#710 file before this
fix was wrong by a comparable factor.

## Not yet done: the spun-up state (#762's other half)

Issue #762 asks for two things. This document covers the statistics; the
**published spun-up JAM state** is deliberately not delivered, and #762 stays
open for it.

The plan, unchanged:

* The artefact is a full `jcm.checkpoint.save_checkpoint` msgpack per grid
  (T63 L47 and T63 L95), produced by the release-validation matrix itself:
  year 1, then `launch.py --resume` for a year 2 once the year-1 aerosol gates
  above pass, publishing the day-730 state as
  `bundles/<grid>/init_states/echam_jam_year2.msgpack`.
* It is published **through the data engine** — a `_MANIFEST_PRODUCTS` row,
  sha256 in the registry, publication-gated — not by hand, like every other
  mirror artefact.
* Consumption is **opt-in** via new `ma-t63-l47-warm` / `ma-t63-l95-warm`
  configurations (base + `init=from_state`), never a production default: a
  default warm start would hide exactly the cold-start regressions the matrix
  exists to catch.

Two things block it, and neither is a matter of effort:

1. **The aerosol mass budget is open.** A spun-up state is a *frozen* copy of
   the model's aerosol; publishing one taken from a run whose sulfate budget
   does not close would enshrine the defect and hand every warm-started run a
   contaminated aerosol population. The gates in this document are the
   precondition — there is no year-2 state to publish until a year-1 run passes
   them.
2. **Issue #731** — a checkpoint breaks when a diagnostic struct gains a leaf —
   is a prerequisite for any state with a shelf life, and is its own PR. A
   published state that stops loading on the next physics change is worse than
   no published state.

The secondary check that state would enable is also specified and also waiting:
a 10-day T63 L47 warm-start regression test against a stored
`default_statistics.nc`, `@pytest.mark.slow` and gated by
`JCM_RUN_GPU_INTEGRATION_TESTS=1`, mirroring the existing ECHAM T63L47
integration test. It could never replace the gates above — ten days is far too
short to see a slow runaway — which is why the statistics half is the half that
was worth doing first.
