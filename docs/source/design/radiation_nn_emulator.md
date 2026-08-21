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

## Inputs: per-band aerosol optics

The base features are `[T, log p, h2o^1/4, o3^1/4, lwp, iwp, mu0 or co2]`.
Aerosol enters through `band_mode`:

| `band_mode` | features | rationale |
|---|---|---|
| `none` | 7 | aerosol-blind; for testing only |
| `broadband` | 10 | one AOD/SSA/ASY triple |
| `per_band` | 49 SW / 55 LW | one triple per RRTMGP band |

`per_band` is the default. It costs essentially nothing: the measured
throughput at T63L47 is 4.13x over RRTMGP, which is exactly the Amdahl ceiling
implied by radiation being 75.8% of the step, so the network itself is already
free and a narrower input would buy no wall time.

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

**The heating-rate loss term is mass-weighted.** Heating is `(g/cp) dF/dp` and
the topmost model layers are ~1 Pa thick, so a 0.1 W/m² flux error there is
~80 K/day and would dominate an unweighted term entirely. Mass weighting turns
it into an energy error. Both weightings are reported, because the raw one is
still what the model has to integrate.

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
noise (the measured 4.13x speedup over RRTMGP is at the Amdahl ceiling), width
is a free lever rather than a trade-off. And **the heating term interacts with
capacity**: at 32 units it hurts (16.18 to 19.33), at 64 it helps (10.07 to
8.18). A small network cannot satisfy the flux and heating objectives at once,
so the extra term just competes; a larger one can.

Every candidate's loss was still falling at its last epoch, so these numbers
rank how fast a configuration learns, not the skill it reaches. Retraining the
winner at 40 epochs took total heating RMSE to 2.87 K/day, with test-set TOA
upward RMSE of 19.6 W/m² (SW) and 7.0 W/m² (LW) at near-zero bias.

## Running a trained emulator

```bash
python -m jcm.main physics=echam-emulated-2m \
    +physics.terms.nn_emulator_radiation.weights_file=/path/to/weights.nc \
    grid=echam_t63_l47_hybrid ...
```

`weights_file: null` initialises random weights instead. That path exists for
cost benchmarking and is paired with `zero_tendency: true`, which discards the
heating so an untrained network cannot NaN the model while its cost is
measured. **`zero_tendency` must be applied downstream of the radiation
sub-step `lax.cond`**: zeroing inside the compute branch leaves
`cached_radiation_tendency` rebuilding untrained heating from the stored rates
on the other seven steps in eight.
