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

Inside the scan, in ECHAM section numbering: 4 sedimentation of the ice
present before detrainment (`pxim1 + ztmst·pxite`, lines 1227-1248), then
`ICNC = min(ICNC + znidetr, icemax)` (lines 1251-1252) → 3.1 melting →
3.2/3.3 sublimation/rain-evaporation → the section-4 `lo2` and the phase
split of the detrained condensate with its fusion-heat correction (lines
1276-1317) → in-cloud prep with clear-sky evaporation
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

- **Sedimentation runs before melting** (the MG/PUMAS order,
  `micro_pumas_v1.F90`), with the running ice tendency threaded through the
  melt routine so the two sinks cannot claim the same mass. ECHAM melts
  first; both orderings are internally consistent ledgers. Melting receives
  the crystal number after `znidetr` for both its `zicncq` and `picnc`
  arguments. ECHAM's `zicncq` already includes `znidetr` (line 982), and
  `znidetr` is zero wherever melting acts, since `ll_cv` needs a step-start
  temperature below `tmelt` and melting one above it, so the two orders
  give the same number.
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

## The state-splitting convention

ECHAM is leapfrog: the scheme receives the t−1 state (`ptm1`, `pqm1`,
`pxlm1`, `pxim1`) plus accumulated tendencies (`ptte`, `pqte`, …) and *adds*
its own contributions to the tendencies. jcm is additive operator-split: each
term returns its own tendency against a provisional post-upstream state.

The mapping used (and documented on `cloud_microphysics_2m`):

- primary inputs = the **post-upstream provisional** state (`thermo_run` T/q,
  `clouds.qc/qi` including this step's convective detrainment) — what the
  returned tendencies are relative to;
- optional `*_m1` inputs = the **step-start** state — ECHAM's t−1 anchors.
  Every quantity ECHAM evaluates at t−1 reads them: saturation and the other
  section-1 fields, `zrid`, the temperature tests of `lo2_2d`, `ll_cv` and
  `lo2`, the moist `cp`, melting and falling-ice sublimation;
- `detrained_qc` and `detrained_qi` = the convective detrainment of this
  step as mass per step (ECHAM `ztmst·pxtecl`, `ztmst·pxteci`).

The pure upstream increments are `(x − x_m1)` for T and q, and
`(qc − qc_m1) − detrained_qc`, `(qi − qi_m1) − detrained_qi` for the
condensate. They play the role of `ztmst·pqte` and `ztmst·ptte` in the
condensation closure, and of `ztmst·pxlte` and `ztmst·pxite` in the
clear-sky-evaporation split; the ice increment also feeds sedimentation.
Today the T and q increments carry this step's vertical-diffusion,
prescribed-flux and Tiedtke tendencies, because those terms advance
`thermo_run`. The condensate increments are zero in the ECHAM ordering: the
cover term snapshots `clouds.qc/qi` from `thermo_run` before vertical
diffusion runs, vertical diffusion advances only `thermo_run`, and the
convection term's addition is the detrainment the scheme receives
separately. The sedimentation input is therefore the step-start ice, and
the vertical-diffusion condensate increment reaches the state without
passing through the scheme. ECHAM's `ptte`/`pqte`/`pxlte`/`pxite` also
carry the dynamics, radiative heating and gravity-wave drag. The #940
rewiring supplies all of these.

The ledger reconstruction is identical either way:
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
  convection scheme and are written to the output as
  `clouds.conv_detrainment_qc` and `clouds.conv_detrainment_qi`. Tiedtke
  keeps advancing `clouds.qc`/`clouds.qi`, so the one-moment contract is
  unchanged.
- The `Lohmann2MMicrophysics` wrapper passes `Δt·conv_detrainment_qc` and
  `Δt·conv_detrainment_qi` to the column function as `detrained_qc` and
  `detrained_qi`. It forms the other increments as it always has.
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
passes it separately and because the input rewiring of #940, which forms
the increments from every upstream term and the dynamics, replaces only the
way the wrapper forms the other increments. The two changes compose without
touching each other.

## The number-tendency rule

The scheme floors the incoming number tracers at `cqtmin` for its own
arithmetic (ECHAM lines 600-605), so dycore ringing cannot drive the
activation or diagnosis steps, and applies no upper bound at entry: the
crystal number is capped at `icemax` only after the detrained number joins
it (line 1252), and the droplet number is not capped. The returned number
tendencies are taken against the raw tracers, as ECHAM passes the raw
`pxtm1` to its ledger (lines 1780-1781) and writes
`pxtte = (N/ρ − pxtm1)/ztmst` (lines 3625-3628). The end-of-step tracer is
then the scheme's crystal or droplet number per kilogram, or zero where the
negative-mass repair removed the condensate (lines 3641-3652), and an
out-of-range tracer value does not persist from step to step. The
previous-step stash `clouds.qnc_prev`/`qni_prev` holds the raw tracers too.

## Deliberate omissions (tracked)

- **Large-scale vertical velocity is not plumbed** (`zvervx` is TKE-only; the
  `knvb`/`lonacc` inversion gate on `zauloc` is omitted; `het_mxphase_freezing`
  likewise lacks `pvervel`) (#705).
- **`nic_cirrus = 2`** still expects the Kärcher–Lohmann `pnicex`/`zqinucl`
  source jcm does not compute (#552); its section-5 deposition branch returns
  zero, as in the reference with a missing external source.
- **Three section-1 number sources are absent**: the cirrus nucleation
  `zninucl` (lines 986-999), whose cap is the soluble-aerosol number `zascs`
  the scheme does not receive; the droplet number of detrained liquid
  `zqlnuccv` (lines 889-941), which needs the activated number at convective
  cloud base; and stratiform activation at cloud base (lines 742-782)
  (#955).
- **Mixed-phase heterogeneous freezing** is a jcm closure (freezing up to an
  INP number from DeMott (2010), or `max(ice_nuclei, DeMott)` under JAM),
  not ECHAM's `het_mxphase_freezing` or its aerosol-free `lccnclim` mode.
  {doc}`../science/clouds_microphysics` gives the reasons and the path to the
  faithful form.

## The gates

`TestColumnWaterConservation2M` and `TestColumnEnthalpyConservation2M`
(modelled on CAM's `check_energy_chng`) close the water and enthalpy budgets
against the surface fluxes for warm-liquid, WBF, cold-fallout, and
melt-in-place fixtures, and with detrained condensate of both phases
re-split by `lo2`. They are the contract the #940 input rewiring must
keep. `lohmann_2m_ice_sources_test.py` pins each section-1 and section-4
rule (`zrid`, `znidetr`, the sedimentation input, the re-split with its
fusion heat, the raw-tracer tendencies, the JAM/DeMott maximum), and
`lohmann_2m_fortran_reference_test.py` compares them block by block and end
to end, on designed columns, with the unmodified ECHAM routine run by a
standalone harness (fixtures under
`jcm/data/test/echam_cloud_reference/cloud2m_*`).
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
