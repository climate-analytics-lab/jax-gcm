# The Lohmann 2M scheme is ECHAM's `column_processes` loop

## The design

`cloud_microphysics_2m` runs the **entire** two-moment process chain inside one
top-down `lax.scan`, one scan step per level, in the section order of ECHAM's
`mo_cloud_micro_2m.f90` `column_processes` loop. Section 8 (the tendency
ledger, `update_tendencies_and_important_vars`) has no cross-level coupling and
runs vectorized on the stacked per-level outputs afterwards.

The monolithic sweep is forced by the reference's data flow: every process at
level `jk` consumes the precipitation state the levels above produced *this
step* — the rain/snow fluxes (`prfl`/`pssfl`), the precipitation cover
(`zclcpre`), and the falling ice flux (`zxiflux`) are loop-carried. Lifting
apparently "level-independent" processes (the condensation partition,
freezing, WBF, precipitation formation) out of the sweep severs those
couplings: accretion by rain and snow from above loses its collector fluxes,
the cloud∩precipitation overlap `zclcstar = min(paclc, zclcpre)` cannot be
formed, ice created mid-step (by deposition, freezing, or WBF) never meets
its aggregation/riming sink that step, and warm rain cannot run after
condensation and activation, which is where both ECHAM and CAM place it. The
whole chain therefore lives in the scan. Wall-clock cost is unchanged at
leading order: the same arithmetic runs inside the scan body rather than in a
`(nlev,)`-vectorized context.

## What the sweep does per level

ECHAM's section 1 runs once per column before the level loop, and so do its
jcm counterparts, vectorized over the levels before the scan:

- `zrid`, the temperature-parameterised crystal radius (lines 945-956),
  from the step-start temperature;
- `lo2_2d`, the WBF criterion for the phase of detrained condensate (lines
  859-885), on the ice present before this step's detrainment and the
  incoming crystal number, with no temperature terms;
- `ll_cv` and `znidetr`, the gate and the crystal number of detrained ice
  (lines 958-983), from the whole detrained condensate of the step at
  `zrid`.

Inside the scan, in ECHAM section numbering: 3.1 melting of BOTH the falling
snow/ice-crystal FLUX entering the level and the in-cloud ice already present
(where `T_m1 > tmelt`) → 3.2/3.3 sublimation/rain-evaporation on that same,
now-melted flux (ECHAM runs all three before its own sedimentation, lines
1170-1190 < 1200-1219 < 1221 — see "What changed" below) → 4 sedimentation of
the melt-adjusted ice present before detrainment (`pxim1 + ztmst·pxite`,
lines 1227-1248, with the melt-updated `pxite`), then
`ICNC = min(ICNC + znidetr, icemax)` (lines 1251-1252) → the section-4 `lo2`
and the phase split of the detrained condensate with its fusion-heat
correction (lines 1276-1317) → in-cloud prep with clear-sky evaporation
(`zxlevap`/`zxievap`) → 5 the `zqcdif` condensation closure → 5.4
supersaturation corrections → 5.5 in-cloud update + activation / ICNC
diagnosis at `zrid` → 6.1 homogeneous freezing → 6.2 heterogeneous
freezing + WBF → 7 precipitation geometry (`zclcstar`, `zauloc`, the
Marshall–Palmer inversions `zxrp1`/`zxsp1`) → 7.1 warm rain → 7.2 cold
precipitation → 7.3 flux update.

Detrained ice therefore joins the cell after sedimentation, carrying the
crystal number `znidetr`, and does not fall at the old number in the step it
arrives. The section-4 `lo2` sees the post-sedimentation ice and the crystal
number that includes `znidetr`, and assigns the whole detrained condensate
to ice where it holds and to liquid elsewhere (lines 1310-1314). The
reassigned condensate enters `zxidt`/`zxldt` (lines 1316-1317) with the
upstream increments, so the in-cloud prep, the clear-sky evaporation and
the section-5 closure all see the re-split state.

## Deliberate deviations

Each was reviewed and kept.

This section lists what is left once #972/#1110/#1116 are fixed; none of
melt, sublimation or sedimentation's relative order is a deviation any
more — "What changed" below is history, not a standing deviation.

- **ECHAM's one `melting_snow_and_ice` routine is two calls here**
  (`melt_falling_flux` / `melt_in_cloud_ice`), back to back, both BEFORE
  sublimation and sedimentation — matching ECHAM's own order for both
  halves (lines 1170-1190, before 1200-1219 and 1221): `picnc`/`pxite` are
  melt's INOUT arguments (lines 1176-1182), and section 4's own
  `zxip1 = pxim1 + ztmst·pxite` (lines 1227-1232) reads them post-melt, so
  a warm level's (`T_m1 > tmelt`) in-cloud ice is melted before any of it
  can sediment out as a still-frozen flux. The split itself is pure
  code organisation (the two pieces act on numerically independent pools —
  the falling flux vs. the in-cloud mass — so nothing stops them running
  back to back as separate calls); `melting_snow_and_ice` remains as a
  thin wrapper for the one existing unit test built around the combined
  call. In-cloud melt receives the crystal number BEFORE `znidetr`/cirrus-
  nucleation join it (ECHAM's `picnc` at line 1176 is still the section-1
  exit value; the join is lines 1251-1258, after section 4).
- **The fusion-heat correction of the detrainment re-split uses the moist
  `cp` and is signed.** ECHAM subtracts `(als − alv)·zxtec/cpd` from `ptte`
  where `ztconv <= tmelt .AND. .NOT. lo2` (lines 1300-1308): dry `cpd`, and
  only convective ice turned liquid. jcm corrects exactly the mass the 2M
  moves against the convection scheme's own split, in both directions, with
  the per-level moist `lsdcp − lvdcp` that every other ledger term of the
  scheme uses. Convective liquid turned ice (the case where vertical
  diffusion warmed the convection's temperature across `tmelt` while the
  step-start temperature stays below it) therefore gains its fusion heat,
  where ECHAM lets it leak. The column moist-enthalpy identity closes to
  round-off with detrainment present.
- **No `icemin` floor on the crystal number** at the two places ECHAM applies
  one: before the loop in cold cloudy cells (lines 1127-1131) and after the
  section-4 additions (line 1253). A cell at or below `icemin` is re-diagnosed
  from its ice mass in `update_in_cloud_water`, and the floor would inject
  `icemin` crystals per step into cells with no ice. ECHAM's entry floor at
  `cqtmin` (lines 600-605) is kept: the strict inequality of the WBF criteria
  needs a positive crystal number to hold at zero updraft.
- **The ICNC diagnosis is capped at `icemax`.** ECHAM's diagnosis (lines
  2610-2624) is uncapped. jcm applies the bound ECHAM puts on the section-1
  additions (line 1252). The `zascs` cap in ECHAM belongs to the cirrus
  nucleation `zninucl` (lines 991-999), which jcm does not have.
- **The sedimentation input is floored at 0**, where ECHAM floors it at
  `EPSILON(1d0)` (line 1228). In float32 the machine epsilon is 1e-7 kg/kg,
  a physical ice amount, so jcm keeps the plain non-negativity floor.

Formulation choices inside the sweep, for provenance:

- The falling-ice cover and the in-cloud sedimentation ledger receive the
  signed from-level flux (lines 2255-2267): a level that gains more ice from
  above than it loses lowers the falling-ice cover and reports a negative
  in-cloud snow-formation rate. The total falling flux stays non-negative in
  exact arithmetic and is guarded against round-off only. The JAM consumers
  floor the ledger per phase, as HAM's wet deposition does
  (`mo_hammoz_wetdep.f90` lines 426-435).
- Grid-scale condensation/evaporation is the section-5 `zqcdif` closure with
  ECHAM's Newton saturation damper (`zqcon`; 0.36–0.93 through the
  troposphere), so no external saturation adjustment is composed alongside
  the scheme and no supersaturation survives a step.
- Clear-sky condensate — including cells whose cover reached zero this step —
  evaporates through `zxlevap`/`zxievap` verbatim.
- Accretion by rain and snow from above converts the loop-carried fluxes to
  local mixing ratios with the Marshall–Palmer inversions (`zxrp1`/`zxsp1`).
- The WBF threshold updraft `peta` is ECHAM's diffusional-growth ζ
  (`mo_cloud_micro_2m.f90` line 856), recomputed after freezing so the
  post-freezing crystal population sets the threshold.
- The WBF threshold updraft is compared with ECHAM's `zvervx` turbulent term
  `100·fact_tke·√TKE` (zero at the lowest level), and the threshold uses
  ECHAM's volume-mean radius `0.9·r_eff` at all four of ECHAM's decisions:
  the detrainment phase `lo2_2d`, the section-4 `lo2`, the section-5
  correction and the WBF gate.
- Diagnostic cirrus ICNC (`nic_cirrus = 1`) inverts the ice mass at ECHAM's
  temperature-parameterised radius `zrid`, in metres (passed as `prid`, line
  1511). The `cirrus_min_ice_radius` floor never binds, because `zrid` is at
  least 1 µm.
- The section-5.4 supersaturation corrections receive the ice part of the
  re-split detrained condensate as their detrainment input, as ECHAM passes
  `zxite` there (line 1490).

## What changed (#972, #1110, #1116)

Three rounds on one root cause, in the order they were found:

1. **#972/#1110**: an earlier version of this sweep ran sublimation AFTER
   this level's own sedimentation (sediment → melt → sublimate: the only
   order change made at first, to fix #1110's specific finding), so a
   level with no incoming ice — the ordinary case for the first level
   that sediments any — spuriously sublimated part of the ice it had
   itself just produced, before that ice ever reached the level below.
2. A first fix moved only the sublimation call ahead of sedimentation,
   leaving `melting_snow_and_ice` (both halves) positioned after it, as
   jcm had run it since before this PR (that position traces to #662
   finding 2, which found the melt tendency was being DISCARDED across
   routines — not that ECHAM's own melt-before-sediment order was wrong
   or that jcm needed a different one; #662 itself calls the order "fine"
   only because MG/PUMAS, `micro_pumas_v1.F90`, happens to use the same
   one, not because ECHAM requires it). That position was safe only
   incidentally, while sublimation's own order was still wrong: once
   sublimation correctly reads the flux ENTERING the level, melt (still
   after sedimentation) and sublimation both read the SAME pre-melt
   `snow_flux`, so what sublimation removed was never actually subtracted
   from what melt also claimed — reproduced in local review as a negative
   outgoing snow flux in a subsaturated level above freezing with
   incoming snow.
3. Moving the WHOLE of `melting_snow_and_ice` (both the flux and in-cloud
   halves) ahead of sublimation and sedimentation — matching ECHAM's
   literal order exactly, as above — fixed the snow double-spend AND a
   second, independent defect a local Codex review caught on that fix:
   with in-cloud melt still after sedimentation, a warm level holding
   in-cloud ice sedimented that ice (as a real outgoing flux, now that the
   self-sublimation damping was gone) before ever getting a chance to
   melt it — a physical impossibility in ECHAM, where a warm level cannot
   emit ice it never had the chance to keep frozen. This is the #1116
   gap's actual content: it was filed, in the PR before this history was
   understood, as a "narrower, dormant" leftover order difference; it
   turned out to be live, and is closed by this reorder rather than left
   open.

Confirmed exact (to float64 round-off) against the `cloud2m_T63L47.npz`
fixture's `sediment_then_detrain` and `detrainment_mixed_lo2_true` columns
at every stage: `icnc_add` (ECHAM's end of section 4) matches every level
of every fixture column, where before step 1 a level that only *received*
sedimented ice from above (no detrainment of its own) ran ~1.8% low on
both ICNC and in-cloud ice mass, compounding to ICNC residuals up to
3.6e-3 across three stacked detraining+sedimenting levels. A fixture-free
test (`lohmann_2m_test.py`) pins step 3's own fix directly: a warm level
holding in-cloud ice with an incoming frozen flux emits no ice flux of its
own, and the melt heating lands in that level.

## The state-splitting convention

ECHAM's scheme receives the previous time level (`ptm1`, `pqm1`, `pxlm1`,
`pxim1`, the number tracers' `pxtm1`) plus the tendencies accumulated since
(`ptte`, `pqte`, `pxlte`, `pxite`, `pxtte`) and the convective detrainment
apart (`pxtecl`, `pxteci`), and *adds* its own contributions to the
tendencies. jcm's terms each return their own tendency against a provisional
state, and the host sums them.

`cloud_microphysics_2m` takes ECHAM's split directly:

- the **anchor** `*_m1` = ECHAM's previous time level. Every quantity ECHAM
  evaluates at t−1 reads it: the air density (the term forms ECHAM's
  `zrho = papm1/(rd·ptvm1)`, `mo_cloud_micro_2m.f90:578`, from the anchor and
  passes it in with the layer depth `Δp/(ρ·g)` that keeps the layer mass),
  saturation and the other section-1 fields,
  `zrid`, the temperature tests of `ll_cv` and `lo2`, the moist `cp`,
  melting and falling-ice sublimation;
- the **increments** `*_increment` = `ztmst·ptte`, `ztmst·pqte`,
  `ztmst·pxlte`, `ztmst·pxite`, `ztmst·pxtte` — everything since the anchor
  except the convective detrainment. They play the role of ECHAM's
  accumulated tendencies in the condensation closure and the
  clear-sky-evaporation split, and the ice increment feeds sedimentation;
- `detrained_qc`, `detrained_qi` = `ztmst·pxtecl`, `ztmst·pxteci`, mass per
  step. The scheme uses their sum `zxtec` as ECHAM does (next section): it
  is not sedimented this step, carries its own crystal number, and enters
  the clear-sky-evaporation split after the `lo2` re-split.

The column forms the provisional state as `anchor + increment` (plus the
detrainment for the condensate) — what the returned tendencies are relative
to. The term fills these from
`jcm.physics.clouds.cloud_inputs.cloud_scheme_inputs`: the anchor is the
previous step's post-physics state carried by the model, and the increments
are the dynamics since then plus every upstream term's tendency (see
[operator-split physics](operator_split_physics.md#coupling-within-physics)).

The section-8 ledger reconstructs ECHAM's end state as
`pxlm1 + Δt·(upstream+own) ≡ qc_provisional + Δt·own`, so the negative-mass
guard bounds the true end-of-step state and the host's tendency sum
telescopes exactly as ECHAM's INOUT accumulation.

## The detrainment contract

ECHAM's two-moment scheme receives one field of detrained condensate,
`zxtec` (the boundary condition 'Detrained condensate', lines 555-570), and
re-splits it by `lo2`. The convection scheme had split the same condensate
at `tmelt` for the one-moment scheme and heated the column accordingly
(`mo_cufluxdts.f90`, `cudtdq`). jcm keeps both views:

- `CloudData` carries `conv_detrainment_qc` and `conv_detrainment_qi`
  [kg kg⁻¹ s⁻¹], the condensate tendencies the Tiedtke term adds to
  `clouds.qc`/`clouds.qi` this step, in Tiedtke's own split (liquid where its
  environment temperature exceeds `tmelt`). They are zero under every other
  convection scheme (the cover term resets them every step) and are written
  to the output as `clouds.conv_detrainment_qc` and
  `clouds.conv_detrainment_qi`. They are the one channel the cloud schemes
  receive detrainment through.
- `cloud_inputs.cloud_scheme_inputs`, which both cloud-scheme wrappers call,
  takes `Δt·conv_detrainment_qc` and `Δt·conv_detrainment_qi` out of the
  running condensate tendency (Tiedtke returns its detrainment as its qc/qi
  tendency) and passes them to the column function as `detrained_qc` and
  `detrained_qi`, beside the anchor and the other increments. The one-moment
  scheme uses Tiedtke's split as it stands, as ECHAM's `cloud` adds
  `pxtecl`/`pxteci` to its condensate increments.
- The column function takes `zxtec = detrained_qc + detrained_qi`, splits it
  by the section-4 `lo2`, and moves `move = liq_part − detrained_qc` from ice
  to liquid (negative where convective liquid becomes ice) with the
  temperature increment `−(lsdcp − lvdcp)·move`.
- The tendency ledger (section 8) runs on the re-split provisional state,
  `qc + move` and `qi − move`. The returned tendencies add `move/Δt` to the
  liquid, subtract it from the ice and add the fusion-heat increment to the
  temperature, so they stay relative to the host's provisional state and the
  host's additive sum is unchanged.

Detrainment arrives separately from the other increments because ECHAM
passes it separately (`physc.f90:1081`) and applies these rules to it alone.

## The number-tendency rule

The scheme floors the incoming number tracers at `cqtmin` for its own
arithmetic (ECHAM lines 600-605), so dycore ringing cannot drive the
activation or diagnosis steps, and applies no upper bound at entry: the
crystal number is capped at `icemax` only after the detrained number joins
it (line 1252), and the droplet number is not capped. The returned number
tendencies are taken against the raw provisional tracers (anchor plus
increment, unfloored), which the host adds them to; ECHAM passes the raw
`pxtm1` to its ledger (lines 1780-1781) and replaces its tendency with
`pxtte = (N/ρ − pxtm1)/ztmst` (lines 3625-3628), the same end state. The
end-of-step tracer is then the scheme's crystal or droplet number per
kilogram, or zero where the negative-mass repair removed the condensate
(lines 3641-3652), and an out-of-range tracer value does not persist from
step to step. `clouds.qnc_prev`/`qni_prev` record the raw number tracers of
the state the term receives.

## Stratiform activation at cloud base (#955)

ECHAM's section 1 runs this droplet-number source before the main level
loop, vectorized over the whole column — `cloud_microphysics_2m` runs it
in the same place, right after `znidetr`, on the pre-scan `cdnc0` that used
to go straight from the entry floor to the `minimum_CDNC` floor with
nothing in between.

**Stratiform activation at cloud base, copied to the cloud above it**
(`mo_cloud_micro_2m.f90` 742-782,
`cloud_utils.stratiform_activation_at_cloud_base`) needs a cloud-base
search: ECHAM's `get_cloud_bounds` (`mo_cloud_utils.f90` 136-220) flags,
for a contiguous run of cloudy levels, the run's base (the level nearest
the surface; the surface level is always its own base, since there is no
level below it to be clear) and tags every level strictly above the base
with the base's own index. `cloud_utils.cloud_base_search` ports the BASE
side only (cloud tops are not needed here) as a single `lax.scan` over the
level axis with `reverse=True`: the carry is the nearest base found so
far, reset to the current level at a base, to -1 outside any cloud, and
inherited while climbing a run above its base. Four triggers decide
whether the base activates — CDNC at or below `cdnc_min`, the liquid still
growing, the cloud cover having just grown, or water supersaturation —
matched to ECHAM's own inputs: `cdnc_min` is the SAME per-level floor
`minimum_CDNC` already computes from the pre-detrainment in-cloud liquid
(previously applied as a floor immediately, now deferred past this
source so the trigger can tell a cell that has not yet been reactivated
this step apart from one that has); "the cover just grew" needs the
*previous* step's cloud cover, which `SundqvistCloudFraction` now
publishes as the plain diagnostics key `"cloud_fraction_prev"` (captured
from the carry it receives, before overwriting it with this step's
RH-diagnosed cover) since `diagnostics["clouds"].cloud_fraction` has
already been replaced with this step's value by the time the microphysics
term runs.

This source is a soft, composition-optional read
(`optional_requires = (..., "cloud_fraction_prev")`): a stack without
`SundqvistCloudFraction` still composes and runs, just without the "cover
grew" trigger, matching how every other JAM-only optional read on this
term (`activated_cdnc`, `freezing_aerosol`, ...) already degrades.

The deferred `minimum_CDNC` floor is applied once, after this source, so
a column that never satisfies its gates reaches the scan with exactly the
value it always did.

## Deliberate omissions (tracked)

- **Two section-1 number sources are still absent**: the cirrus nucleation
  `zninucl` (lines 986-999), whose cap is the soluble-aerosol number `zascs`
  the scheme does not receive; and the droplet number of detrained liquid
  `zqlnuccv` (lines 889-941), which needs the activated number at
  convective cloud base (#955). Stratiform activation at cloud base (lines
  742-782) is implemented; see "Stratiform activation at cloud base"
  above.
- **Mixed-phase heterogeneous freezing without a prognostic aerosol** is a
  jcm closure (freezing up to the DeMott (2010) INP number), not ECHAM's
  aerosol-free `lccnclim` mode (its own implementation is separate, tracked
  work, #1149). With JAM, section
  6.2 is ECHAM-HAM's `het_mxphase_freezing` on the `ham_IN_setup` inputs the
  aerosol publishes (`freezing_aerosol`). The choice is a static switch on the
  presence of those inputs, so the aerosol-free composition traces the closure
  exactly as before. {doc}`../science/clouds_microphysics` gives the details.
  `demott2010_inp`'s background coarse-aerosol number (`n_aer_coarse`,
  default 0.5 cm⁻³) is itself a differentiable tunable, not a fixed property
  of the closure: at `n_aer_coarse = 0` the INP number — and therefore every
  freezing increment downstream of it — is exactly 0, so a composition that
  wants a literally aerosol-free closure (matching ECHAM's own aerosol-free
  build, as `lohmann_2m_fortran_reference_test.py`'s comparisons do) gets one.

## The gates

`TestColumnWaterConservation2M` and `TestColumnEnthalpyConservation2M`
(modelled on CAM's `check_energy_chng`) close the water and enthalpy budgets
against the surface fluxes for warm-liquid, WBF, cold-fallout, and
melt-in-place fixtures, and with detrained condensate of both phases
re-split by `lo2`. `lohmann_2m_ice_sources_test.py` pins each section-1 and section-4
rule (`zrid`, `znidetr`, the sedimentation input, the re-split with its
fusion heat, the raw-tracer tendencies, the closure's INP maximum), and
`lohmann_2m_fortran_reference_test.py` compares them block by block and end
to end, on designed columns, with the unmodified ECHAM routine run by a
standalone harness (fixtures under
`jcm/data/test/echam_cloud_reference/cloud2m_*`), including section 7's
`precip_formation_warm`/`_cold` (the rain/riming/snow formation mass that
feeds `update_precip_fluxes`) fed ECHAM's own section-6-exit per-level state
and chained exactly as the scan sequences them.
`lohmann_2m_freezing_reference_test.py` does the same for section 6.2 with the
HAM freezing inputs set (`cloud2m_frz_*`), and `lohmann_2m_freezing_test.py`
pins the aerosol-free path to its output before the switch existed.
`TestSaturationGate2M` pins that no supersaturation survives a step;
`TestColdChainSameStepCoupling2M` pins that a deck glaciating via WBF
exports a frozen flux the same step. Budget tests pin the
ledger's self-consistency — a defect that mis-states the in-cloud state on
both sides of a transfer needs a *state* assertion, which is why the het-INP
test bounds the per-level fusion heat from below rather than trusting closure
alone.

`clouds.cloud_fraction` is the post-microphysics cover under both the 1M and
2M schemes (documented on `CloudData`), and the 2M negative-mass repair is
exported as `clouds.negative_mass_repair` [W/m²] so its sign-definite heating
is measurable in any run.

The stratiform cloud-base activation source (#955) has its own gates.
`cloud_utils_test.py`'s `TestCloudBaseSearch` pins `cloud_base_search`
against hand-built single- and multi-run columns, including the
surface-is-always-a-base boundary case; `TestStratiformActivationAtCloudBase`
pins each of the four triggers individually, the copy-above-base rule, a
triggered base that gains no droplets not copying anything, and gradients
through the activated branch. Both also check a `(nlev,)` column and its
`(nlev, ncols)` broadcast agree. `lohmann_2m_test.py`'s
`TestCloudFractionPrevWiring955` spies on `cloud_microphysics_2m`'s
`jax.vmap` call (the same technique `TestLargeScaleVerticalVelocityWiring`
uses for #705) to pin that `"cloud_fraction_prev"` actually reaches the
function with the right value and the right fallback when its producer
is absent — not just that the pure function works in isolation.
`TestCloudBaseActivationSensitivity955` is the motivating case itself: in
a column whose CDNC already exceeds `cdnc_min` (closing the pre-#955
reactivation path), doubling the activated number now raises the
in-cloud CDNC one level above a freshly formed cloud base, and has no
effect at all without that growing base.

## The wet-scavenging interface

The scheme publishes the same process-time ledger that ECHAM-HAM's
`cloud_subm_2` receives from `column_processes`: `zmlwc`/`zmiwc` (in-cloud
condensate captured at section 7, before precipitation formation depletes
it), the in-cloud formation rates `zmratepr`/`zmrateps`/`zmsnowacl`
(`zmrateps` seeded from `sedimentation_ice`: sedimenting ice **is** a
scavenging carrier in ECHAM-HAM), the cover the processes ran under, and the
condensate-evaporation ledger (`zxlevap + zxievap`). These appear on
`CloudData` as `incloud_*`, `process_cloud_fraction`, and
`condensate_evaporation_rate` (see `ScavengingLedger` in
`lohmann_2m/types.py`), and the JAM wet-deposition and cloud-borne exchange
terms key to them — the ledger is why `aerosol_module="jam"` requires the 2M
scheme:

- **In-cloud removal** is HAMMOZ `prep_wetdep_hydro`'s
  `peffwat = (zmratepr+zmsnowacl)·Δt/zmlwc` and `peffice = zmrateps·Δt/zmiwc`
  (clipped to [0, 1]), split by the in-cloud ice mass fraction. Numerator and
  denominator are both captured at process time, so the fraction is bounded
  by construction in every cell, including near-empty ones.
- **Resuspension** of cloud-borne aerosol keys to the condensate-evaporation
  ledger: a sky cleared by evaporation releases the reservoir in one step, a
  sky cleared by rainout releases nothing (that aerosol leaves with the
  precip), and the same step's rainout claim caps the released share so the
  two sinks cannot jointly overdraw. The end-of-step cover cannot make this
  distinction — both endings read `cloud_fraction = 0` — which is why the
  ledger, not the cover, is the interface.

One documented deviation from the reference: ECHAM-HAM zeroes `zmlwc`/`zmiwc`
*after* the `paclc` write-back (mo_cloud_micro_2m.f90:3655 → 3660), so a cell
whose condensate fully converted to precipitation reaches `cloud_subm_2` with
a zero pool, `peffwat = 0`, and **no scavenging in exactly the step with the
largest removal** — the reference has the dead zone itself. jcm keeps the
faithful zeroing, because the marker is information — a zero pool with a
positive formation rate identifies the fully-converting cell — but maps the
marker to scavenged fraction **1**, not 0: the droplets became precipitation,
and everything they carried went with them.
