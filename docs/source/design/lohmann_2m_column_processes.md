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

In ECHAM section numbering: 4 sedimentation → 3.1 melting → 3.2/3.3
sublimation/rain-evaporation → in-cloud prep with clear-sky evaporation
(`zxlevap`/`zxievap`) → 5 the `zqcdif` condensation closure → 5.4
supersaturation corrections → 5.5 in-cloud update + activation/nucleation →
6.1 homogeneous freezing → 6.2 heterogeneous freezing + WBF → 7 precipitation
geometry (`zclcstar`, `zauloc`, the Marshall–Palmer inversions
`zxrp1`/`zxsp1`) → 7.1 warm rain → 7.2 cold precipitation → 7.3 flux update.

One deliberate deviation, reviewed and kept: sedimentation runs **before**
melting (the MG/PUMAS order, `micro_pumas_v1.F90`), with the running ice
tendency threaded through the melt routine so the two sinks cannot claim the
same mass. ECHAM melts first; both orderings are internally consistent
ledgers.

Formulation choices inside the sweep, for provenance:

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
- Diagnostic cirrus ICNC (`nic_cirrus = 1`) uses the Schumann volume-mean
  radius, in metres.

## The state-splitting convention

ECHAM is leapfrog: the scheme receives the t−1 state (`ptm1`, `pqm1`,
`pxlm1`, `pxim1`) plus accumulated tendencies (`ptte`, `pqte`, …) and *adds*
its own contributions to the tendencies. jcm is additive operator-split: each
term returns its own tendency against a provisional post-upstream state.

The mapping used (and documented on `cloud_microphysics_2m`):

- primary inputs = the **post-upstream provisional** state (`thermo_run` T/q,
  `clouds.qc/qi` with convective detrainment folded in) — what the returned
  tendencies are relative to;
- optional `*_m1` inputs = the **step-start** state — ECHAM's t−1 anchors.
  Saturation and all section-1 fields evaluate there, and the differences
  `(x − x_m1)` play the role of `ztmst·pqte` in the condensation closure and
  of `ztmst·pxlte` in the clear-sky-evaporation split.

The ledger reconstruction is identical either way:
`pxlm1 + Δt·(upstream+own) ≡ qc_provisional + Δt·own`, so the negative-mass
guard bounds the true end-of-step state and the host's tendency sum
telescopes exactly as ECHAM's INOUT accumulation.

## Deliberate omissions (tracked)

- **Large-scale vertical velocity is not plumbed** (`zvervx` is TKE-only; the
  `knvb`/`lonacc` inversion gate on `zauloc` is omitted; `het_mxphase_freezing`
  likewise lacks `pvervel`). Tracked in #705.
- **`nic_cirrus = 2`** still expects the Kärcher–Lohmann `pnicex`/`zqinucl`
  source jcm does not compute (#552); its section-5 deposition branch returns
  zero, as in the reference with a missing external source.
- **Moist heat capacity**: ECHAM's `zlvdcp = Lv/(cpd + cpd·vtmpc2·q)`; jcm
  uses `Lv/cpd`. ~1 % on latent heating; kept so the column enthalpy gate has
  a single cp. Tracked in #706.

## The gates

`TestColumnWaterConservation2M` and `TestColumnEnthalpyConservation2M`
(modelled on CAM's `check_energy_chng`) close the water and enthalpy budgets
against the surface fluxes for warm-liquid, WBF, cold-fallout, and
melt-in-place fixtures. `TestSaturationGate2M` pins that no supersaturation
survives a step; `TestColdChainSameStepCoupling2M` pins that a deck
glaciating via WBF exports a frozen flux the same step. Budget tests pin the
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
