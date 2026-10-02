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
| `dlnB_dt_<sp>_per_day` over the final six months (final year for a record of a year or more) | the runaway detector — see below. |
| `yoy_burden_ratio_<sp>` | mean burden of the final 365 days over the 365 before; reported for a two-year record, not gated. |
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
* SOA is not scored for closure at all. `emi_soa` exists but carries only
  primary emission, which is zero in the supported configurations; SOA's real
  source is condensation of the SOAG gas, and no `emi_*` channel accounts for
  it. A residual computed for SOA would therefore measure an unrepresented
  source rather than a mass leak.

## The two tolerance tiers

**Tier 1 — absolute physics gates.** These are not tuning targets. They are
statements about what the model must do regardless of how well calibrated it is,
so they are the same number for every member and every release.

| gate | limit |
|---|---|
| `\|d ln B/dt\|` per species, final six months (final year for a record of a year or more) | `< 0.002 /day` |
| `\|budget residual\|`, max over species | `< 5 %` |
| `dyn_frac_per_step_<sp>` | `< 0.1 % per step` |

**Tier 2 — climatological anchors.** The existing per-species ranges in
`tools/jam_burden_report.py`'s `_SPECIES` table, widened by the release gate's
×3 slack. Unchanged by #762; a "did the model produce a plausible planetary
loading" check, not a calibration.

**Tier 3 — regression tolerances**, for comparing a run against a stored
reference: `max(3σ, 15 % relative, an absolute floor)`. The floor matters
because a legitimately-zero reference — an unused species' burden — would
otherwise give a zero tolerance that any nonzero value fails. Statistics with an
absolute gate of their own (drift, closure, dynamics) are **not** scored here:
their references are ~1e-3, so a relative tolerance would be tighter than the
gate they already have. σ is the standard error of the record mean,
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

The window depends on how much record there is. A record shorter than a year
— the 40 saves (~195 days) the release recipe scores of a cold-start year —
is fit over its **final six months**: a from-zero spin-up year ramps for its
first months by design, and scoring the whole record would call that ramp a
runaway. Tropospheric burdens equilibrate in weeks (lifetimes ~4–7 days), so
six months is comfortably past the ramp while still long enough that a slope of
0.002 /day is resolvable above the noise. A record that covers a year or more —
a warm or second year scored whole, or two years in one directory — is fit over
its **final 365 days**, which hold every season exactly once. A cold-start
year therefore has to be sliced with `--last-n` to its settled months; the
whole-year rule is for a record that no longer contains a spin-up ramp.

The reason for the longer window is the seasonal sources. Dust, biomass-burning
carbon and sulfate all have a seasonal cycle, and a least-squares slope over
part of a cycle reads the cycle as growth or decay. Measured on JAM T63 L47,
two consecutive years of the release recipe on dev `80699ac8` (the second
started from the first's end state; each scored over its last 40 saves, so the
six-month fit sits inside a 195-day record):

| species | final six months, year 1 | final six months, year 2 | whole year 2 | year-1 day 150 to year-2 end |
|---|---|---|---|---|
| SO4 | +0.0020 | +0.0025 | +0.0003 | +0.0002 |
| BC | +0.0032 | +0.0033 | +0.0004 | +0.0003 |
| POA | −0.0016 | −0.0014 | +0.0007 | −0.00004 |
| dust | −0.0021 | −0.0011 | −0.0021 | −0.0004 |
| sea salt | +0.0003 | −0.0003 | −0.0011 | −0.0001 |

(slopes of `ln B`, per day; the limit is 0.002.) The same-window slopes repeat
from year to year with the same sign and size, which a spin-up drift would not,
and BC exceeds the limit in both; the whole-year slopes of SO4, BC and POA are
below 0.001 /day, and the slope of the year-2 / year-1 ratio over the settled
days is at most 0.001 /day for all five species. On JAM T63 L95, measured the same way, the year-2
same-window slopes of BC (+0.0022) and POA (−0.0026) exceed the limit while the
five whole-year slopes are at most 0.0008 /day. The whole-year dust slope of L47 sits
at the limit because dust emission is event-driven (the same 30-day block holds
a storm in one year and none in the other); the year-over-year ratio is the
better statistic for it, which is why it is reported for a two-year record.

The longer window is less sensitive to a late runaway, and that is the price. A
×60 growth over the last 40 days of an otherwise seasonal year fits a slope of
0.018 /day over the final six months (9× the limit) and 0.0039 /day over the
final year (2×); a 0.03 /day growth over the last 120 days fits 0.027 and 0.008.
It still fails, but with a smaller margin, and the budget-closure and per-step
dynamics gates below do not depend on this window.

`0.002 /day` is a factor e over ~500 days: a burden that genuinely has not
settled may still move that much over a validation year, but the ×60-in-40-days
growth of a #658-class runaway exceeds it by an order of magnitude over the
six-month window — and so does the much gentler growth those runaways show for
months *before* they become visible. On the August-2026 year the gate fails on
sulfate at day 300, on a window that ends before the burden leaves its anchor
range at all.

A species the run never carries (SOA with no SOAG production, say) has a
correctly-zero burden and no drift to measure; it is skipped rather than scored
as NaN, and the **climatological anchor gate skips it on the same grounds** —
the two tiers must not contradict each other about the same species on adjacent
lines of one report.

The window itself has a floor. A least-squares slope carries noise
`σ_resid / (Δt·√(N(N²−1)/12))`; at the 5-day output cadence and the ~0.07
log-burden scatter of a settled species, three of those falls below the
0.002/day limit only past **90 days**. `health.py --last-n` is the documented
way to score the settled months (the slice carries the label of the chunk
before it, so the first retained window is centred where it really sits; a
record whose uniformly spaced chunks evidently start mid-run, such as a resumed
run in a fresh output directory, infers its start from the cadence), and on a
shorter window the fitted slope is
noise — so below that span the statistic is reported **UNSCORED**, with the
number of days it needs, rather than gated. The closure residual takes the same
floor: its storage term is a single endpoint difference, which over a few
chunks swamps the flux integral it is compared against.

**No gate ever passes by absence.** Anything that could not be evaluated —
a species the run does not carry, a window too short, a carried species with
no usable `budget_dyn_<sp>` gauge or `budget_mass_<sp>` denominator (checked
per species, since trimmed or mixed-version output can gauge one and not
another), no timestep to express the dynamics gate per step, a lifetime species
whose deposition ledgers are absent or holed — is printed as `UNSCORED` with
its reason. A missing row is otherwise
indistinguishable from a row that passed.

An UNSCORED gate is reported but does **not** fail the exit code, deliberately:
its commonest causes are the operator's own `--last-n` and a species the
configuration does not carry, and failing those would make the tool unusable
for the windows it is documented to support. Scoring *nothing at all* is the
exception and does exit non-zero. Statistics that carry an absolute gate are
also exempt from the tier-3 reference comparison for the same reason — being
correctly unscored there must not become a hard failure against a reference
that happens to have the number.

The flux integral runs on **chunk centres**, not the end-of-chunk day the
filenames carry: a chunk holds a time average, which belongs at the middle of
its window. Under the uniform cadence of a real run the offset cancels, but
naming it keeps the quadrature honest if the cadence ever varies.

## The closure gate, and its sign

The residual is `(Σ emitted − Σ deposited − ΔB) / Σ emitted` over the record.
Its **sign** is the diagnosis:

* **Negative** — more mass was deposited than ever entered the column. No
  missing ledger entry can produce that, so it is mass creation. It does *not*
  say where: the ledger carries emission, deposition and storage but no
  transport term, so a negative residual indicts the model, not the physics
  specifically — the dynamics gate below is what separates the two.
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

## The spun-up state (#762's other half)

Issue #762 asks for two things. This document covers the statistics; the
**published spun-up JAM state** is delivered separately, and #762 stays open for
its publication.

The artefact is a full `jcm.checkpoint.save_checkpoint` msgpack per grid (T63 L47
and T63 L95), produced by the release-validation recipe: a cold first year, which
is the aerosol spin-up, and a warm second year started from its end state (the
`tools/release_validation` README and jcm-monitor's experiments do this), whose
end state is the published one. It carries a provenance record
(`<state>.provenance.json`: the run, its length, code, environment and the state
it started from) that `generate_stats.warm_state_source` checks before a fixture
is drawn from it.

How long the spin-up takes was measured on JAM T63 L47 and L95, each run for two
consecutive years on one commit (the second started from the first's end state).
From the cold dry init the burdens of year 1 lie below those of year 2 in the same
season by up to ~110 % (dust), ~65 % (SO4) and ~50 % (sea salt) in the first two
months, and the two years agree to within a few percent from day ~150 on L47 and
day ~90-120 on L95, while the mass budget closes to within 0.15 % throughout, so
the early deficit is the water cycle and winds spinning up, not an aerosol
accumulation. Year 1's end state is already on
the equilibrated trajectory; year 2 is the first whole equilibrated year.

A state outlives code changes that add carry fields: `load_checkpoint` seeds a
field a newer model carries from its documented initial value and drops one it no
longer carries, reporting each in the log (#731), and the aerosol tracers
themselves are unchanged. It does not outlive a change of climate: after the land-surface energy balance
(#979) lengthened aerosol lifetimes (dust 1.6 to 2.0 days), the burdens of the
first year started from a pre-#979 state were up to ~+100 % above the pre-#979
year (BC, SO4) and decayed over the year, and the following year differed from
that one by under ~10 % in most 30-day blocks from day 60 on. States are therefore drawn from the final physics of a release.

Publication is **through the data engine** — a `_MANIFEST_PRODUCTS` row, sha256
in the registry, publication-gated — not by hand, like every other mirror
artefact, and consumption is **opt-in** via `ma-t63-l47-warm` / `ma-t63-l95-warm`
configurations (base + `init=from_state`), never a production default: a default
warm start would hide exactly the cold-start regressions the matrix exists to
catch.

The secondary check that state would enable is also specified and also waiting:
a 10-day T63 L47 warm-start regression test against a stored
`default_statistics.nc`, `@pytest.mark.slow` and gated by
`JCM_RUN_GPU_INTEGRATION_TESTS=1`, mirroring the existing ECHAM T63L47
integration test. It could never replace the gates above — ten days is far too
short to see a slow runaway — which is why the statistics half is the half that
was worth doing first.
