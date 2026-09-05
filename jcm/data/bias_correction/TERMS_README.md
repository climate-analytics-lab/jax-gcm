# Bias-correction terms (T31 SPEEDY)

What ships here, what it scored, and where the rest is. **One term ships in
this directory**, `online_term_t31_big_insol_vt.npz`. A "term" is a trained
`NNBiasCorrection` saved with `term.save(path)`; load one with
`load_bias_correction(path)`. The file is self-contained: the weights, the
input normalisation, the output scale, the context feature and the taper are
all inside it, so nothing else is needed to run it.

**The method, the results and the limitations are documented in
`docs/source/bias_correction.rst`. That page is the source of truth.** This
file is the artifact index. The scripts that produced and scored every term,
with every argument, are in `tools/bias_correction/` (see its README). If this
file and the .rst ever disagree, the .rst is right and this file needs fixing.

## Scores

Cos-latitude area-weighted RMS error against ERA5 over **2016-2022**, seven
years no term was trained on, from a free run with no nudging and no reanalysis
input. Lower is better. Temperatures in K, humidity in g/kg. Every number comes
from a saved field file scored by `tools/bias_correction/holdout_table.py`,
not typed by hand; the table was regenerated on 4 September 2026. Term names
drop the `online_term_t31_` prefix. All of it was produced on the SPEEDY
physics as of commit 748413a (July 2026); the surface flux and vertical
diffusion schemes changed on `dev` afterwards and the scores have not been
recomputed against the current plain SPEEDY (see the docs page).

| Term | net | seed | surf ann | surf DJF | surf JJA | 500 hPa | humidity | beats plain |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| plain SPEEDY, no term | - | - | 3.64 | 3.60 | 4.17 | 4.73 | 1.46 | - |
| `clim_vt` | 8k | 0 | 3.15 | 3.69 | 3.28 | 4.20 | 0.84 | 4 of 5 |
| `20yr_clim_vt` | 8k | 0 | 3.14 | 3.89 | 3.63 | 5.02 | 0.74 | 3 of 5 |
| `big_clim_vt` | 148k | 0 | 3.44 | 3.55 | 3.78 | 2.25 | 0.80 | 5 of 5 |
| `big_s1_clim_vt` | 148k | 1 | 2.55 | 3.64 | 2.95 | 5.05 | 1.25 | 3 of 5 |
| `big_s7_clim_vt` | 148k | 7 | 3.14 | 3.52 | 3.65 | 3.37 | 0.91 | 5 of 5 |
| `big_fmask_vt` | 148k | 0 | 3.15 | 3.39 | 3.89 | 2.67 | 0.96 | 5 of 5 |
| **`big_insol_vt`** (ships) | 148k | 0 | 3.29 | **3.32** | 3.87 | 2.53 | 0.79 | **5 of 5** |
| `big_s7_insol_vt` | 148k | 7 | 3.23 | 3.38 | 3.85 | 3.70 | 1.00 | 5 of 5 |
| `big_sice_vt` | 148k | 0 | 3.39 | 3.44 | 3.76 | 2.47 | 0.82 | 5 of 5 |
| `big_insol_notaper` (control) | 148k | 0 | 4.49 | 5.33 | 5.19 | 2.42 | 0.82 | 2 of 5 |

Floors, measured on the same holdout, in the same column order: weight
perturbation of the shipped term 0.07 / 0.07 / 0.12 / 0.06 / 0.06; sampling
noise, from consecutive 7-year blocks of one 21-year run, 0.02 / 0.06 / 0.05 /
0.03 / 0.02; reseeding the no-context recipe (three seeds) 0.89 / 0.12 / 0.83 /
2.79 / 0.45; reseeding the shipped recipe (two seeds) 0.06 / 0.06 / 0.02 /
1.17 / 0.21.

## What the table shows

The first two rows are the same 8,352-parameter network, trained on one year
and on twenty-one. More data moved surface annual and humidity but left winter
worse than doing nothing (3.69 and 3.89 against plain's 3.60), and the 21-year
term is worse aloft than plain as well.

The 148k rows are the same recipe at 256 hidden units. Every one of them beats
plain on all five metrics in its first seed, with or without a context feature,
and the shipped term does so in both seeds trained. That is the finding: the
winter failure was network capacity, not a missing input. It had been diagnosed
for weeks as a limit of correcting each column in isolation, and that diagnosis
was wrong.

What the table does not support is a ranking of the 148k terms against each
other. The gaps between them (winter 3.32 to 3.55, 500 hPa 2.25 to 3.70) are
the size of the seed spread, and the no-context recipe reseeded once landed at
3 of 5. The shipped term was chosen because it is the only one that scored 5 of
5 in two seeds and on both the in-sample and the held-out years. That is a
selection, not a proof that insolation is the right input.

Removing the taper from the shipped weights (the `notaper` control) takes
winter from 3.32 to 5.33 and 5 of 5 to 2 of 5. Only 500 hPa is better without
it.

## What is in this directory

- `online_term_t31_big_insol_vt.npz`: a 148,512-parameter network (32 profile
  inputs plus daily-mean top-of-atmosphere insolation, three hidden layers of
  256, tanh, linear head) that corrects temperature and specific humidity, with
  a post-hoc surface taper over sigma 0.7 to 1.0. **The reference term**, and
  the one every number in `docs/source/bias_correction.rst` refers to.

Nothing else. The comparison terms in the table, the seed repeats, the offline
warm starts and the wider experiment set (lambda sweeps, level-weighted losses,
the T63 transfer, seasonal windows, ensembles) are kept in the author's
research repository. Nothing in this repository loads them, they can be shared
on request, and the chain in `tools/bias_correction/README.md` regenerates any
of them from scratch. Stage 1 of that chain writes a `_stats.nc` beside the
warm start it produces; it is a record of the normalisation, which the term
also carries, and is not needed to run anything.

The `_vt` suffix means a post-hoc surface taper over sigma 0.7 to 1.0: the
near-surface temperature correction is faded to zero while the correction
aloft and the humidity correction are kept. It is applied by
`tools/bias_correction/add_surface_taper.py` after training. Without it the
near-surface temperature correction makes the surface worse. Training with the
taper active works less well than applying it afterwards: the network re-routes
the polar warming through the levels the taper leaves alone.

## Why the winter diagnosis changed

Before the wide network, the record here argued that winter was a structural
limit of correcting a column in isolation. The reasoning was that in DJF plain
SPEEDY is too cold globally but too warm over northern high-latitude land, so
that region's bias has the opposite sign to the one the correction was trained
to remove, and a single column's profile cannot carry the sign of its own local
bias. Two different added inputs, a land mask and an insolation signal, both
failed the same way at 8k parameters, which looked like confirmation.

The wide network beats plain in winter with no context feature at all, so that
explanation was wrong. What the small network lacked was capacity to represent
a correction that changes sign by region, not information about which region
it was in. The earlier argument is kept here because the evidence for it was
real and the way it failed is the useful part.
