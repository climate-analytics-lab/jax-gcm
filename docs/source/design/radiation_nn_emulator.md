# The neural-network radiation emulator

RRTMGP is roughly three quarters of a T63L47 ECHAM timestep, so replacing it
with a network that reproduces its fluxes is the single largest speedup
available to this model. `jcm/physics/radiation/nn_emulator.py` holds the
network, `nn_emulator_scheme.py` the `PhysicsTerm` that drops into a physics
package in place of `RRTMGPRadiation`, and `tools/radiation_emulator/` the
offline pipeline that generates labels and trains against them.

This document records the decisions that are not obvious from the code, and
the traps that cost time when they were not.

## Architecture

Both bands use the same recurrent architecture, ported from Ukkonen's
`rte-rrtmgp-nn`: a forward GRU up the column, a dense layer on
`[last hidden state, surface auxiliary]`, a backward GRU over the extended
sequence, a third GRU on the concatenation, and a sigmoid dense output. The
surface auxiliary is the broadband albedo for shortwave and the emissivity for
longwave. The vertical dimension is the sequence dimension, which is what lets
one network serve any column without a fixed level count.

The port is pinned against a real upstream checkpoint
(`jcm/data/test/rrtmgp_nn_bigru16_reference.npz`, weights of
`bigru_gru_16_new.onnx` plus onnxruntime's output for fixed inputs) because
every convention involved — GRU gate order, the `reset_after=True` two-row
bias, backward-pass alignment — mismatches *silently*, costing accuracy rather
than raising. That test found a real bug: Keras's bare `go_backwards=True` GRU
does **not** re-reverse its output (only `Bidirectional` does), so level *i*'s
forward state was being concatenated against level *nlev - i*'s backward state.
`gru_backward_sequence(..., realign=False)` preserves the upstream behaviour
for checkpoint compatibility; `realign=True` is the correct pairing and is what
jax-gcm trains and runs.

## Inputs

The base features are
`[T, log p, h2o^1/4, o3^1/4, lwp, iwp, cloud fraction, mu0 or co2]`.
Aerosol enters through `band_mode`:

| `band_mode` | features | rationale |
|---|---|---|
| `none` | 8 | aerosol-blind; for testing only |
| `broadband` | 11 | one AOD/SSA/ASY triple |
| `per_band` | 50 SW / 56 LW | one triple per RRTMGP band |

### Cloud fraction is a separate feature, not folded into the paths

The water and ice paths handed to the network are GRID MEANS — in-cloud water
times cloud fraction. Passing only those makes a thin overcast layer and a
thick broken one identical inputs at equal mean path, while their shortwave
reflectance is quite different, reflectance being nonlinear in optical depth.

That degeneracy is not a small effect and it caps skill at any network size.
Shortwave TOA upward flux sat at ~18.3 W/m² RMSE across a 4x range of width
while shortwave *heating* improved 3.5x over the same range:

| units / epochs | SW TOA up | LW TOA up | SW heating | LW heating |
|---|---|---|---|---|
| 64 / 40 | 18.69 | 6.67 | 2.324 | 0.611 |
| 128 / 300 | 18.35 | 6.32 | 0.837 | 0.720 |
| 256 / 200 | 18.29 | 6.41 | 0.663 | 0.717 |

With the label-noise floor measured at 4.70 W/m² (see below), neither capacity
nor the labels explained it. Longwave, whose emission is far closer to linear
in optical depth, kept improving — which is the asymmetry the degeneracy
predicts, and which is what identified the missing variable.

Adding the feature confirms it. Same 128 units, same 300 epochs, same seed,
cloud fraction the only change:

| metric | without cf | with cf |
|---|---|---|
| SW TOA up (W/m²) | 18.35 | **7.43** |
| SW surface down | 31.41 | **12.71** |
| LW TOA up | 6.32 | **2.27** |
| LW surface down | 13.38 | **3.95** |
| SW heating (K/day) | 0.837 | **0.723** |
| LW heating | 0.720 | **0.196** |
| total heating | 1.556 | **0.920** |

Every metric improves, most by 2.5-3.7x. **One input feature was worth more
than 4x the network width and 7.5x the training budget put together.** At 256
units / 200 epochs it goes a little further — SW TOA 7.25, LW TOA 2.28, total
heating 0.796 — but width has clearly stopped being the lever.

Note this moves the shortwave close to the label-noise floor: 7.25 W/m² against
4.70 leaves ~5.5 W/m² of model error, so McICA sampling is now ~40% of the TOA
shortwave error rather than the ~6% it was before. More draws per column is now
worth considering, where earlier it was not.

Adding the fraction moves the `per_band` widths from 49/55 to 50/56.
Checkpoints trained on the old layout are rejected by the input-width
validation rather than silently misread. Upstream `rte-rrtmgp-nn` weights
predate the feature and need `n_input_features(..., n_base=7)`.

`per_band` is the default. It costs essentially nothing: the settled
end-to-end throughput at T63L47 is 4.4x over RRTMGP (see "Measured cost"
below), close to the Amdahl ceiling implied by radiation dominating the step,
so the network itself is already free and a narrower input would buy no wall
time. The same arithmetic is why
width is a free lever: a wider network changes the input tensor and the matmul
sizes but not the fact that radiation has stopped being the bottleneck.

**The band structure must come from the active radiation backend.** The
emulator is matched in `runners._band_config_for_terms` alongside
`RRTMGPRadiation` for exactly this reason. Before that, the emulator was handed
`RadiationBandConfig.broadband()` — 1 SW band, 0 LW — and the width mismatch
crashed in a matmul. That crash was luck: had the widths happened to align, the
emulator would have run silently on aerosol input unlike anything in its
training set. `_check_band_counts` now raises with the offending numbers, and
loading a weight file validates `band_mode` and input width against its
metadata.

## Outputs: four channels, and why the longwave scale is what it is

The network predicts four normalised interface profiles — all-sky down, all-sky
up, clear-sky down, clear-sky up — so cloud radiative effect is a real
diagnostic rather than a stub of zeros.

Normalisation makes the target scale-free, and the choice of scale is
load-bearing because **the output layer ends in a sigmoid, so predictions are
confined to (0, 1)**:

- **Shortwave** divides by the incoming TOA flux. Downward flux cannot exceed
  it and upward cannot exceed downward, so targets are in [0, 1] by
  construction. The TOA downward boundary is then set exactly rather than
  learned.
- **Longwave** divides by `sigma * T_max^4`, black-body emission at the
  *warmest temperature anywhere in the column, surface included*
  (`lw_flux_scale`). No interface flux can exceed that.

Scaling the longwave by the surface emission `eps * sigma * T_s^4` — the
obvious choice, and the original one — does **not** bound the target. Over a
cold surface under a warmer atmosphere (polar night, Antarctica) the outgoing
longwave exceeds what the surface emits: ~14% of RRTMGP-labelled T63L47 targets
landed above 1 that way, reaching 1.28, in a region the sigmoid cannot
represent at all. The residual was untrainable and concentrated exactly where
the emulator is hardest to check. Emissivity left the scale but still reaches
the network as the surface auxiliary input, so the dependence is learned.

## Training data

`tools/radiation_emulator/generate_training_data.py` drives the in-repo
`radiation_scheme_rrtmgp` over batches of columns and stores **raw physical
inputs** next to the flux labels — not network features. The feature layout is
still being tuned and must be changeable without regenerating the expensive
labels.

Two column sources, deliberately combined:

- **`trajectory`** samples a JCM output netCDF over (time, lon, lat). These are
  the states the model actually visits. Point it at **snapshot** output: a
  5-day mean state is not a state any radiation call ever sees, and its solar
  geometry is the chunk timestamp rather than the averaged field's.
- **`perturbation`** is a Latin-hypercube sweep over the radiatively active
  parameters. It extends coverage into aerosol and cloud states a short run
  never reaches, which is what the emulator must extrapolate over when aerosol
  forcing is perturbed.

An ERA5 source is a documented extension point; ARCO-ERA5 carries the required
fields, including the four profiles WeatherBench2 lacks.

### McICA labels are stochastic

RRTMGP samples clouds per g-point, so a single call carries several W/m² of
sampling noise. Labels are averaged over `--n-seeds` independent draws (8 in
production) to give a clean conditional mean. The draw is varied through
`model_step`, the *traced* int32 that `mcica.column_key` folds into the PRNG
key — not through `base_seed`, which is a Python static and would force a full
~40 s XLA re-trace per seed.

**The residual noise is a floor on any reported skill**, so measure it rather
than assuming it. `tools/radiation_emulator/label_noise.py` differences two
independent N-draw means of the same columns; on the 2026-08 T63L47 set with
8 draws:

| quantity | noise on an 8-draw mean | for scale: 128-unit model RMSE |
|---|---|---|
| SW TOA up | 4.70 W/m² | 18.35 |
| SW surface down | 5.16 | 31.41 |
| LW TOA up | 0.40 | 6.32 |
| LW surface down | 0.89 | 13.38 |

At that skill level sampling noise is ~6% of the shortwave error *variance*, so
the labels are **not** what limits the emulator and buying more draws would
achieve almost nothing — 4 draws give 7.74 W/m² against 8 draws' 4.70, the
expected 1/sqrt(N). Re-measure before concluding otherwise once model error
approaches ~10 W/m².

### Inputs are bounded, and labels are checked

Model-output diagnostics leave physical bounds, and the violations are not
benign. A 550 nm SSA a whisker above 1 (float32 accumulator drift in the
time-averaged diagnostic) flips the sign of the MACv2-SP per-band denominator
and becomes a ~1e21 far-infrared SSA; RRTMGP then returns **1089 W/m² of OLR**
from an unremarkable 270 K clear-sky column. Surface emissivity reaches 1.9
over polar land (jax-gcm#703, an unclipped sea-ice fraction in the radiative
surface-optics blend).

So the generator clips inputs into `INPUT_BOUNDS` — the 550 nm SSA *before* the
per-band scaling, so the bands get the correct small value rather than a
clipped-to-1 wrong one — and tallies every clip. Behind that,
`label_quality_mask` rejects any column whose fluxes are non-finite, negative,
brighter in reflected than incident shortwave, or above a black body at the
column's warmest temperature. That backstop does not depend on knowing every
failure mode, which is the point: one column emitting 1000 W/m² would dominate
a mean-squared-error loss over thousands of good ones.

## Training

`tools/radiation_emulator/train_emulator.py`. Two properties matter more than
the optimiser details.

**Feature parity.** Inputs are built by calling the same
`preprocess_sw_inputs` / `preprocess_lw_inputs` the online scheme calls, from
the same raw fields, with the same derived quantities. Re-deriving features in
the trainer would let the network drift from what it is fed at run time — the
classic silent failure of an emulator. Scaling is fitted on the training split
alone, stored in the weight file, and applied through the one shared
`apply_input_scaling` so train and inference cannot diverge.

**Input conditioning.** The scaling is affine, `(x - x_offset) / x_max`, not
the reference divide-by-max. Profiling what the network actually receives found
two features unusable under divide-by-max alone:

| feature | mean | std |
|---|---|---|
| temperature | 0.848 | 0.042 |
| cloud liquid path | 0.004 | 0.038 |
| cloud ice path | 0.0003 | 0.006 |
| co2 | 1.000 | 0.000 |

Temperature — the dominant longwave input — sits in a narrow band far from
zero, so the network must learn large weights to resolve the variation it cares
about. The cloud paths are so skewed that a few extreme columns set the scale
and the cloud signal is crushed to nothing. Centring fixes the first; the cloud
paths additionally take the same fourth root the gas features use, which is
also a reasonable compression of an optical depth that enters transmittance
exponentially. `x_offset` defaults to 0, so upstream checkpoints and any
weights fitted before this keep loading unchanged. The denominator is floored
in `fit_scaling` so a constant feature (CO2, in a fixed-concentration dataset)
maps to 0 rather than dividing rounding noise by ~0.

**Measured neutral, and kept anyway.** A controlled A/B — same 64 units, same
40 epochs, same seed, scaling the only difference — gives:

| metric | divide-by-max | affine + cloud 4th root |
|---|---|---|
| SW TOA up (W/m²) | 19.55 | 18.69 |
| SW surface down | 38.79 | 37.71 |
| LW TOA up | 7.02 | 6.67 |
| LW surface down | 14.85 | 14.56 |
| SW heating (K/day) | 2.246 | 2.324 |
| LW heating | 0.620 | 0.611 |
| **total heating** | **2.866** | **2.935** |

All four flux measures improve by 2-5%, but total heating — the metric that
matters — is 2.4% *worse*. On one seed there is no noise floor, so the honest
reading is that this is a wash. The feature statistics above are real, but
conditioning was **not** the binding constraint: a GRU's gate biases can absorb
a constant input offset, and the network had evidently already compensated.
Capacity and training budget are what limit skill (see the sweep below).

It is kept because it is backward-compatible, improves the flux metrics
consistently, and plausibly matters more at larger widths — not because it was
shown to help. Do not cite it as an improvement.

**Honest validation.** The split is by **solar-geometry group**: every column
sharing an `(orbital_phase, synodic_phase)` pair came from one model snapshot
and lands wholly in one partition. Columns from one snapshot are spatially
correlated, so a column-wise random split validates against near-copies of
training data. The synthetic sweep randomises geometry per column, so the same
rule degenerates to a plain random split there — correct, because those columns
are independent by construction. Groups are assigned largest-first to whichever
partition is furthest below its column quota; assigning in random order let a
run of small groups fill the training quota and a single large snapshot arrive
while training was still emptiest, swallowing every column.

### Offline skill is not a stability criterion

**The single most important lesson from building this.** The first emulator to
reach good offline skill — 7.43 W/m² shortwave TOA, 0.92 K/day total heating,
every metric improved — **NaN'd the GCM at 100% temperature within five days**,
while the identical 10 days under RRTMGP ran clean.

The error was one layer:

| level | pressure | SW heating error RMSE |
|---|---|---|
| **0** | **1.0 Pa** | **129.9 K/day** (422 max) |
| 1 | 4.3 Pa | 6.24 |
| 2 | 11.1 Pa | 4.51 |
| 10-40 | 0.6-785 hPa | **0.02-0.17** |

Heating is `(g/cp) dF/dp`, and the topmost layer is ~2 Pa thick, so it
amplifies flux error by **421 K/day per W/m²**. A 0.3 W/m² flux-difference
error — negligible against a 7 W/m² flux RMSE — is 130 K/day. Applied every
radiation step, that is a dead model.

The heating loss was **mass-weighted** at the time, which gave that layer
~1e-5 of the loss, and the headline score was the mass-weighted 0.72 K/day.
Mass weighting measures the *energy* error and is the right lens for asking
whether the emulator conserves energy; it is the wrong lens for asking whether
the model survives, because a level's temperature does not care how little mass
it holds.

So the trainer now:

- trains the heating term on **uniform per-level weights** (`uniform_weights`),
  not mass weights;
- reports the **worst level's** heating RMSE and its index, because a column
  mean under *any* weighting hides one catastrophic layer;
- **ranks configurations on that worst level** (`score`), because that is what
  decides whether the run survives.

Mass weighting stays available and reported as the energy-error lens.

Generalise the lesson before trusting any future metric here: an emulator can
be near-perfect in everything you chose to measure and still destroy the model
through a layer your metric discounted. Put it in the loop.

Hyperparameters are ranked on validation at a short budget (`--sweep`), and the
winner is retrained for the full budget. Test metrics are computed once, from
the third partition, and never used to choose anything.

### What the first sweep found

Seven candidates at 8 epochs each, on 160k training columns, scored by total
mass-weighted heating RMSE on validation:

| units | lr | heating weight | K/day |
|---|---|---|---|
| 16 | 3e-3 | 0 | 21.27 |
| 32 | 3e-3 | 0 | 16.18 |
| 64 | 3e-3 | 0 | 10.07 |
| 32 | 1e-3 | 0 | 22.22 |
| 32 | 3e-3 | 1e-3 | 21.56 |
| 32 | 3e-3 | 1e-2 | 19.33 |
| **64** | **3e-3** | **1e-2** | **8.18** |

Two things to carry forward. **Width dominates** — it is the only axis that
moves the number much, and since the network's run-time cost is already in the
noise (the settled 4.4x speedup over RRTMGP is near the Amdahl ceiling), width
is a free lever rather than a trade-off. And **the heating term interacts with
capacity**: at 32 units it hurts (16.18 to 19.33), at 64 it helps (10.07 to
8.18). A small network cannot satisfy the flux and heating objectives at once,
so the extra term just competes; a larger one can.

### Ukkonen's network scale does NOT transfer

His production shortwave model is 3 GRU layers x 16 units — 5,698 parameters —
and reaches 0.16 K/day; nothing published in that literature exceeds 96 units.
That is a strong argument for shrinking, and it was tested here directly.
It does not hold. Ranking on the worst level's heating RMSE, on the corrected
300k-column dataset (so the model top is properly sampled and fitting it does
not require extrapolation):

| units | worst-level heating RMSE (K/day) |
|---|---|
| 16 | 18039 |
| 24 | 9364 |
| 32 | 6630 |
| 64 | 2038 |

Monotonic, with the largest gain at the last step. The same monotonicity
appeared on the earlier, hole-y dataset, and the natural explanation there —
that capacity was being rewarded for overfitting sparse data in the coverage
gap — is ruled out by its surviving the fix.

**These four numbers are contaminated and the ranking they support is not
safe to reuse.** They were measured before `band_metrics` applied the
deployed TOA-downward boundary, so each includes a phantom shortwave error
at interface 0: the network's sigmoid cannot emit exactly 1, but the online
path (`reconstruct_sw_interface_fluxes`) overwrites that interface with the
exact incoming flux, so the shortfall never reaches the GCM. At ~421 K/day
per W/m² across the 2 Pa top layer, a 2% shortfall against a 1361 W/m²
incoming flux is ~11,000 K/day — the same order as the table itself. The
monotonicity is therefore consistent with the sweep having partly ranked
candidates on how closely each approaches an output that is discarded at
inference, which is a capacity proxy but not the one intended. The
conclusion below may well survive re-measurement — the feature-count and
output-channel arguments are independent of this metric — but it has not
been re-measured. Treat the table as a record of what was run, not as
evidence, until the sweep is repeated on the corrected score (jax-gcm#743,
which also covers re-selecting the packaged checkpoint).

The mass-weighted table in the previous section is not affected: it gives
the top layer ~1e-5 of the total weight, which is the same property that
made it blind to the layer that NaN'd the model in the first place.

The differences that plausibly explain it all make this the larger problem:
**50-56 input features against his 11**, four output channels against two,
per-band aerosol optics and McICA cloud variability he did not have (his clouds
fill the grid box and his effective radii are constants), and a model top a
decade lower in pressure. Capacity requirements scale with what the network is
asked to represent. Do not shrink this network on the strength of his
parameter count.

Every candidate's loss was still falling at its last epoch, so these numbers
rank how fast a configuration learns, not the skill it reaches. Retraining the
winner at 40 epochs took total heating RMSE to 2.87 K/day, with test-set TOA
upward RMSE of 19.6 W/m² (SW) and 7.0 W/m² (LW) at near-zero bias.

## Coupled stability: the offline/online gap

**Offline skill did not predict coupled behaviour, and the gap was entirely in
the training distribution.** Three coupled T63L47 attempts NaN'd within a day
while RRTMGP ran clean from the identical start. Driving the *live* scheme
offline on real columns, though, the emulator matched RRTMGP to 0.09-1.68
K/day through the mid-column — so the network was not the problem, and neither
was orientation, feature wiring, nor the loss.

The failure localised to longwave at the model top:

| level | p (Pa) | emulator LW rms | RRTMGP LW rms |
|---|---|---|---|
| 0 (surface) | 96417 | 33.2 | 10.8 |
| 20-30 | 23499-3743 | 1.0-1.2 | 1.3-1.6 |
| 46 (top) | **1.0** | **323.5** (max 1111) | **41.7** (max 79) |

with the model top warming 232 to 257 K in twelve hours before diverging.
Shortwave was unaffected, because shortwave heating at 1 Pa does not depend on
the local Planck function.

**The cause was a coverage hole of our own making.** At 1 Pa the training data
was bimodal: the trajectory source spanned 242-252 K (a spun-up run's narrow
band) and the synthetic sweep piled at exactly 190 K (its floor), leaving
nothing between 205 and 240 — which is where a realistically-initialised model
sits, median 234 K. Every point was inside the union of the two ranges, so a
min/max check saw nothing wrong. The emulator was extrapolating into the gap,
and the resulting error warmed the top further into it.

Two defects fed the hole, both in the sweep's base state and both fixed: its
pressure grid stopped at 100 Pa (no coverage at all for the top eight levels),
and its temperature was a single tropospheric lapse rate floored at 190 K, with
no stratosphere. The grid is now the model's own hybrid levels — inventing one
instead produced per-layer aerosol optical depths of 21 and ~9% unphysical
columns — and the temperature is piecewise-linear in log(p) through surface,
tropopause, stratopause and model top.

The other half of the fix is training on the states the coupled model actually
visits: `trajectory_era5.nc` samples an ERA5-initialised RRTMGP run in the
configuration the emulator is meant to run in. That is the intervention the
literature supports most strongly — see below.

### What the literature says about coupling one of these

- **Ukkonen (2022, JAMES; peterukk/rte-rrtmgp-nn)** never coupled the
  full-scheme emulator in that work, and his model top is **10 Pa over 60
  layers**, not 1 Pa. He applies no dp-weighting, no top-level exclusion and no
  clipping, and reports up to 20 K/day heating error at the top.
- **Bertoli et al. (2025, ICON)** needed *"a Gaussian smoothing as
  postprocessing and a simplified computation of the fluxes at the upper
  levels"* — above 25 km longwave, 40 km shortwave — *"to ensure stability of
  the ICON model top"*.
- **Hafner et al. (2025)** avoided flux prediction entirely, predicting heating
  rates plus boundary fluxes with an explicit energy-consistency loss term,
  citing exactly those stability issues.
- **CMA-GFS (operational)** had **18% of 50 coupled ten-day forecasts crash**.
  Output bounds and physical reconstruction of heating did not fix it;
  harvesting the states at which the model broke and retraining on them did,
  with **no architecture change**.
- Ukkonen's emulators that *did* run stably in the IFS were trained on the host
  model's own ecRad inputs and outputs — on-trajectory by construction.

So the field's verdict is consistent: predict fluxes and do something explicit
at the top, or predict heating rates and penalise energy imbalance — and in
either case train on the distribution the coupled model produces.

## Gases the emulator does not see

The feature vector carries ozone and CO2; **CH4 and N2O are absent**, and the
labels are generated at RRTMGP's own defaults for both. A scenario that varies
either gas would get fluxes with no trace of its forcing, so
`jcm.runners.guard_emulator_ghg_forcing` refuses that combination and points at
`physics=echam-rrtmgp-2m`. The guard covers the Hydra paths, where forcing is
concrete at build time; a direct `Model.run(forcing=...)` caller can still hand
the term traced values it cannot branch on. Adding the features (and *varying*
both gases in the labels, or they are dead weight) is jax-gcm#738 — worth doing
once, on whatever feature vector survives the sub-column question (#734).

## Measured cost (v3 weights, 2026-08-25)

Settled per-chunk wall clock (`tools/benchmark.py`, T63L47, 5-day chunks,
last-two-chunks-within-3% convergence gate, same A100):

- RRTMGP control (`t63-echam-rrtmgp-2m`): **22.7 s per sim day**
- Emulator u64 (`t63-echam-emulated-2m`, packaged weights): **5.13 s per sim
  day**, NaN-free over 45 days — an end-to-end **4.4x**

Per-term attribution (`tools/profile_terms.py`, same configs): RRTMGP is
132 ms/step amortised — 82% of device time, 1322 ms per actual call at the
10-step radiation cadence — while the u64 emulator term is 2.55 ms/step
(25.5 ms/call), a ~50x cheaper radiation term; the residual step is dominated
by convection (8.7 ms) and dynamics (5.4 ms). The wall-clock 4.4x sits
slightly under the profiler-implied ceiling because chunk boundaries (host
sync, health check, netCDF write) are outside the step.

One negative result worth keeping: the v3 **u256** checkpoint — offline
metrics indistinguishable from u64's — went 100% temperature-NaN inside five
coupled days, while u64 ran 45 days clean. v2's u256 was coupled-stable, so
this is not a width law; it is another instance of "offline skill is not a
stability criterion" (above), and the reason the packaged default is the
narrower, stability-proven network.

## Running a trained emulator

Trained weights ship with the package
(`jcm/data/emulator_weights_per_band_u64.nc`, resolved by the config default
`weights_file: auto`; provenance in the file's global attributes), so the
preset runs out of the box:

```bash
python -m jcm.main physics=echam-emulated-2m grid=echam_t63_l47_hybrid ...
```

Point `weights_file` at another checkpoint to swap networks; larger
checkpoints stay out of the repo (stage them on the HF data bundle).
`weights_file: null` initialises random weights instead. That path exists for
cost benchmarking and is paired with `zero_tendency: true`, which discards the
heating so an untrained network cannot NaN the model while its cost is
measured. **`zero_tendency` must be applied downstream of the radiation
sub-step `lax.cond`**: zeroing inside the compute branch leaves
`cached_radiation_tendency` rebuilding untrained heating from the stored rates
on the other seven steps in eight.
