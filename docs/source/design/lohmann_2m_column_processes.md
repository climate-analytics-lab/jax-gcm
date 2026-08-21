# The Lohmann 2M scheme is ECHAM's `column_processes` loop

*(2026-08; issues #662, #667, #685, #686, #687, #688, #689; parent audit #614)*

## The decision

`cloud_microphysics_2m` runs the **entire** two-moment process chain inside one
top-down `lax.scan`, one scan step per level, in the section order of ECHAM's
`mo_cloud_micro_2m.f90` `column_processes` loop. Section 8 (the tendency
ledger, `update_tendencies_and_important_vars`) has no cross-level coupling and
runs vectorized on the stacked per-level outputs afterwards.

The previous layout split the chain: "level-independent" processes
(condensation partition, freezing, WBF, warm/cold precipitation formation) ran
vectorized outside the sweep, and only sedimentation/melting/sublimation ran
inside it. That looked like a harmless performance-friendly factorization. It
was not, because in the reference **every process at level `jk` consumes the
precipitation state the levels above produced *this step*** — the rain/snow
fluxes (`prfl`/`pssfl`), the precipitation cover (`zclcpre`), and the falling
ice flux (`zxiflux`) are loop-carried. Splitting the chain silently severed
those couplings:

- accretion by rain/snow from above read `qr`/`qs` tracers that no term
  declared any more — both identically zero, so the pathway was dead code and
  precipitation formation was understated ~3× (#662 finding 5, and through
  `rate_cb = p_form/(qc+qi)` the leading suspect for the cloud-borne aerosol
  drainage failure in #658);
- the cloud∩precipitation overlap `zclcstar = min(paclc, zclcpre)` could not
  be formed, so accretion geometry fell back to `paclc` (#685);
- ice created mid-step (deposition, freezing, WBF) never met its
  aggregation/riming sink that step (#686);
- warm rain ran *first*, before condensation and activation, matching neither
  ECHAM nor CAM (#667 finding 4).

The restructure is therefore not an optimization choice reversed — it is the
minimal faithful shape. Wall-clock cost is unchanged at leading order: the
same arithmetic moves from a `(nlev,)`-vectorized context into the scan body,
and the scan already existed.

## What the sweep does per level

In ECHAM section numbering: 4 sedimentation → 3.1 melting → 3.2/3.3
sublimation/rain-evaporation → in-cloud prep with clear-sky evaporation
(`zxlevap`/`zxievap`) → 5 the `zqcdif` condensation closure → 5.4
supersaturation corrections → 5.5 in-cloud update + activation/nucleation →
6.1 homogeneous freezing → 6.2 heterogeneous freezing + WBF → 7 precipitation
geometry (`zclcstar`, `zauloc`, the Marshall–Palmer inversions
`zxrp1`/`zxsp1`) → 7.1 warm rain → 7.2 cold precipitation → 7.3 flux update.

One deliberate deviation, reviewed and kept: sedimentation runs **before**
melting (MG/PUMAS order, `micro_pumas_v1.F90`), with the running ice tendency
threaded through the melt routine so the two sinks cannot claim the same mass
(#662 finding 2). ECHAM melts first; both orderings are internally consistent
ledgers.

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

## What this made live (previously inert or miscalibrated)

| pathway | before | after |
|---|---|---|
| internal saturation adjustment | `zqcon ≈ 1/50…1/650` (zdqsdt ×1000; #667.1) | Newton damper 0.36–0.93; 10 % supersat → 1 % in one step |
| grid-scale condensation | external Sundqvist bolt-on (double-adjustment risk) | ECHAM section-5 `zqcdif` closure in-scheme; bolt-on removed |
| clear-sky condensate sink | none (`pxlevap = pxievap = 0`) | `zxlevap`/`zxievap` verbatim; cf=0 cells evaporate in one step |
| rain/snow-from-above accretion | dead (`qr`/`qs` ≡ 0) | Marshall–Palmer inversion of carry fluxes |
| WBF threshold updraft | fed a 0–1 clip as `peta` | ECHAM's diffusional-growth ζ (line 856); recomputed post-freezing |
| diagnostic ICNC (`nic_cirrus=1`) | `prid` in µm (r³ off by 10¹⁸) → pinned at floor | Schumann volume-mean radius in metres |
| cold-chain gate | step-start `qi > ccwmin` | `ll_cc` + in-scan `zxib > cqtmin` (#686) |

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
survives a step (the property whose absence let the double adjustment hide);
`TestColdChainSameStepCoupling2M` pins that a deck glaciating via WBF exports
a frozen flux the same step. Budget tests pin the ledger's self-consistency —
a defect that mis-states the in-cloud state on both sides of a transfer needs
a *state* assertion, which is why the het-INP test bounds the per-level
fusion heat from below rather than trusting closure alone.

`clouds.cloud_fraction` now means the post-microphysics cover under both the
1M and 2M schemes (#687; documented on `CloudData`), and the 2M
negative-mass repair is exported as `clouds.negative_mass_repair` [W/m²]
(#689) so its sign-definite heating is measurable in any run.
