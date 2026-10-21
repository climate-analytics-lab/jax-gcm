# SPEEDY sensitivity harness: tangent growth, GFMIP FD atlas, jvp chains

Scripts to reproduce, for SPEEDY physics, three experiment classes from the
jcm differentiable-sensitivity campaign:

1. **Tangent growth** (`growth.py`, `plot_growth.py`): how fast forward-mode
   (jvp) tangents of SST-forced runs grow, for any truncation / level count /
   dycore / hyperdiffusion strength, optionally with a tangent decay; plus a
   climatology check (`climatology.py`, `plot_climatology.py`) and a combined
   table (`matrix_table.py`), so every setting is judged on both.
2. **GFMIP finite-difference atlas** (`make_patches.py`, `gfmip_fd.py`,
   `gfmip_analysis.py`): the 109-patch Green's-function protocol of
   Bloch-Johnson et al. (2024), de-overlapped and compared with the GFMIP models.
3. **Ensemble jvp chains** (`chains.py`): Lea, Allen & Haine (2000) style
   iteration of the ensemble-averaged tangent to a linear-response equilibrium.

The branch is `origin/main` (a1b79026) plus small jcm changes (below) plus
this directory. It is a working branch for sharing, not a PR: the repository's
conventions (CLAUDE.md) keep bespoke experiment drivers out of the package and
route configurations through Hydra, so upstreaming would mean moving the
toggles into config, wiring the hybrid dycore option through Hydra, and adding
tests.

## jcm changes on this branch

Everything else these experiments need is already on `main`:

* **Vertical levels**: `get_speedy_coords(layers=N)` takes any `N >= 2`
  (SPEEDY's hand-tuned tables for 7/8 levels, a Frierson-stretched sigma grid
  otherwise; see `docs/source/design/speedy_variable_levels.md`). SPEEDY's
  stable time step shrinks automatically for thin bottom layers.
* **Dycore**: `DinosaurDycore(..., advection="eulerian" | "semi_lagrangian")`.
  Left to itself, SPEEDY picks Eulerian. The semi-Lagrangian option is
  dinosaur's native SL transport.

The branch adds three things, all opt-in and off by default (the two
environment variables are read at trace time, so set them before the script
starts):

| Change | Files | Effect |
|---|---|---|
| `JCM_PERPETUAL_EQUINOX=1` | `jcm/forcing.py`, `jcm/physics/radiation/speedy_shortwave.py` | Freezes the solar year position at the spring equinox and zeroes SPEEDY's obliquity zenith correction: no seasonal cycle in insolation. |
| `JCM_TANGENT_DECAY=<gamma>` (1/day) | `jcm/dycore/dinosaur/dycore.py` | At the start of every step, multiplies the forward-mode tangent of the whole dycore state by `exp(-gamma dt)`. The primal is unchanged (a `jax.custom_jvp` identity; verified bit-identical), and tangents injected during the step, e.g. by the SST perturbation through the physics, are not scaled. |
| `DinosaurDycore(..., advection="eulerian", eulerian_vertical_sl=True)` (harness: `--advection hybrid`) | `jcm/dycore/dinosaur/dycore.py` | The campaign's hybrid dycore: Eulerian spectral horizontal dynamics with their vertical advection replaced by a semi-Lagrangian vertical step (see "Semi-Lagrangian" below). Sigma coordinates only. |

Notes on the tangent decay:

* **It cannot make growth worse.** The homogeneous tangent propagator becomes
  exactly `exp(-gamma dt)` times the true one, so every tangent growth rate
  drops by exactly `gamma` (checked: the tangent of an initial-state
  perturbation after 1 day is scaled by `exp(-gamma)` to 6 digits). It is
  parameterised as a rate so a given `gamma` means the same thing at any time
  step. It biases sensitivities, like any tangent regularisation: responses
  that take longer than `1/gamma` to build up are suppressed. Check against
  FD. The physics' cross-step carry (SPEEDY keeps little in it) is not
  decayed.
* **Why not attenuate the physics alone.** An earlier version of this branch
  scaled only the tangent of the state handed to the physics by a factor
  `a` (the variant that controlled ECHAM's growth in the campaign: 29/day to
  0.22/day at `a = 0.1`). For SPEEDY it made growth *faster* (T31-L8
  Eulerian, OLR tangent, days 5-40: `a = 1`: 0.18/day; `0.5`: 0.20-0.22;
  `0.1`: 0.25-0.27; `0`: 0.00). The growth needs the physics in the loop
  (with `a = 0` the dycore alone is neutral), but the physics tangent also
  contains damping, and scaling it down removed more damping than
  amplification. The whole-state decay replaces it.

## Setup

Python >= 3.11 environment with this branch installed (`pip install -e .` at
the repository root; jax 0.10.x, dinosaur >= 1.5). Run the scripts from this
directory. All outputs go under `--out` (default `runs/`). `gfmip_analysis.py`
and `plot_growth.py` import only numpy/scipy/matplotlib/netCDF4 and run in any
environment that has those.

Model configuration flags, shared by every script:

| Flag | Default | Meaning |
|---|---|---|
| `--truncation` | 31 | spectral truncation (T21: 64x32 grid, T31: 96x48) |
| `--nlev` | 8 | sigma levels |
| `--advection` | eulerian | `eulerian`, `semi_lagrangian` (dinosaur's native SL), or `hybrid` (Eulerian horizontal + campaign SL vertical step) |
| `--sl_vertical_interpolation` | cubic | `semi_lagrangian` only: `cubic` or `linear` vertical departure-point interpolation |
| `--diffusion_scale` | 1 | hyperdiffusion strength multiplier (all three timescales divided by it) |
| `--time_step` | auto | minutes; default is SPEEDY's stable step for the grid |

Runs use realistic boundary conditions: the packaged T30 SPEEDY climatology
forcing and T63 terrain interpolated to the model grid.

**Background state.** Tangent work (growth, chains) needs a stationary
background, so those scripts freeze every seasonal boundary field at its
annual mean, and you should also export `JCM_PERPETUAL_EQUINOX=1`, both when
spinning the state library and when running. The campaign found that states
spun with different forcing than the tangent runs pollute the ensemble
tangent with spin-adjustment transients. The GFMIP FD atlas follows the GFMIP
protocol instead: seasonal BCs, no equinox toggle.

## 1. Tangent growth

```bash
export JCM_PERPETUAL_EQUINOX=1
python growth.py --truncation 31 --nlev 8  --advection eulerian
python growth.py --truncation 31 --nlev 16 --advection semi_lagrangian
python growth.py --truncation 31 --nlev 8  --diffusion_scale 4
JCM_TANGENT_DECAY=0.1 python growth.py --truncation 31 --nlev 8    # growth minus 0.1/day
python plot_growth.py runs/growth-*.npz --obs olr --fit 5 40 --out growth.png
```

For a tropical (0N 150E) and an extratropical (50N 200E) Gaussian SST bump
(18 deg width), `growth.py` spins `--nseed` states for `--spin_days` (or takes
them from `--lib`), takes the jvp of a `--days`-long run with daily-mean
output, and records the area-weighted **spatial variance of each observable's
tangent** per day, averaged over seeds. Observables: OLR, absorbed SW at TOA,
convective and large-scale precipitation, cloud cover, stratiform cloud, and
SPEEDY's dry static energy gradient `gse`.

`plot_growth.py` fits the growth rate as half the slope of log(variance) over
a day window: a natural-log rate per day, so `e^(lambda t)` growth and an
e-folding time of `1/lambda`. Pick the window inside the exponential regime by
looking at the plot.

Campaign reference (pre-merge stack, so expect quantitative differences on
current `main`): Eulerian T31-L8 grew fastest at about 0.19/day (e-fold ~5 d);
more levels, lower truncation and stronger damping slowed growth; the
campaign's semi-Lagrangian dycore was nearly non-growing (T31-L16 e-fold
~100 d). That SL dycore was a campaign-specific implementation; `main` uses
dinosaur's native SL, so SL numbers in particular need re-establishing.
Measured on this branch: T31-L8 Eulerian grows at 0.18/day (e-fold 5.5 d),
matching the campaign. The effect of every knob on both growth and
climatology is in the matrix below.

### Semi-Lagrangian: this branch vs the campaign's formulation

The campaign's "SL" growth numbers (T31-L16 nearly non-growing) came from a
different semi-Lagrangian formulation than the one on `main`. That formulation
is ported to this branch as the `hybrid` option, so both can be run:

| | Campaign hybrid (`--advection hybrid`) | `main` SL (`--advection semi_lagrangian`) |
|---|---|---|
| Horizontal | Eulerian spectral, IMEX SIL3 (unchanged) | semi-Lagrangian (3-D departure points), Crank-Nicolson RK2 |
| Vertical | separate first-order, operator-split SL step after each dynamics step, built in jcm from dinosaur's helper functions; moves u, v, T, tracers along departure levels | part of the 3-D SL transport |
| Vertical interpolation | **linear** (dinosaur's `_vertical_interp`) | cubic Lagrange, linear in the top/bottom cell; `--sl_vertical_interpolation linear` available |
| Mass fixers | none | global fixers for humidity and tracers |
| Hyperdiffusion | campaign stack: needed >= 1.625x default for 16-member ensembles to survive 60 d at T31-L16 (non-monotone: 1.25x and 1.5x were worse than 1x); science ran at 2x. See the matrix below for this branch | runs at default |

Why the campaign built its own: dinosaur exports a vertical SL step, but it
interpolates vorticity and divergence as if they were advected scalars, which
is dynamically wrong and blew up over orography within 2-5 days at
T21-T31. The campaign's velocity-space variant fixed that. It has not been
compared against `main`'s SL over orography.

What the hybrid formulation buys and costs (campaign measurements unless
noted; the matrix below has this branch's numbers):

* **Upside: slow tangent growth.** Single-window tangent growth was about
  0.035-0.06/day at T31-L16 across the stable damping range (smallest at
  2-2.5x damping), vs ~0.14-0.20/day for the Eulerian core. Linear
  interpolation is a plausible reason: its weights are convex, so the
  transport part of the linearisation cannot amplify a perturbation, while
  cubic Lagrange weights can be negative. Measured on this branch with `main`'s SL
  (T31-L16, OLR tangent, days 5-40, tropical / extratropical bump):

  | Configuration | Growth rate (1/day) |
  |---|---|
  | Eulerian, default damping | 0.14 / 0.20 |
  | SL (cubic), default damping | 0.11 / 0.14 |
  | Eulerian, 2x damping | 0.077 / 0.103 |
  | SL (cubic), 2x damping | 0.066 / 0.098 |
  | SL (linear vertical), 2x damping | 0.049 / 0.069 |

  At matched damping `main`'s SL is within ~15% of Eulerian; doubling the
  hyperdiffusion halves the rate and linear vertical interpolation removes
  another ~30%, which together reproduce most of the campaign variant's slow
  growth. The remainder is plausibly its vertical-only operator splitting
  (untested).
* **Short-term downsides.** It needs roughly 2x hyperdiffusion just to stay
  stable, and it is first-order in time. Linear vertical interpolation is not
  conservative for convex profiles: `main`'s own measurement for linear
  interpolation in its SL is ~0.24 mm/day of spurious water and ~11 W/m2 of
  energy drained at T63-L47 (the campaign variant, with no mass fixer, was
  not measured).
* **Long-term downsides.** At matched 2x damping vs the Eulerian core
  (16 seeds x 60 d, T31-L16): OLR -7%, absorbed SW -2.7%, convective precip
  +9%, large-scale precip -13%, cloud +4%. Its warm-pool response lacked the
  Eulerian core's upper-tropospheric heating band and anvil moistening, so
  local cloud and OLR responded with the opposite sign to observations and
  GCM Green's-function studies (campaign verdict: Eulerian more realistic).
  And its ensemble-averaged propagator has a coherent growing mode, so jvp
  chains diverged (x1.9-2.4 per iteration) for every window and ensemble size
  tested: single-window transients are fine, iterated constructions are not.
  On this branch's port (matrix below): at L8 the hybrid is stable at default
  damping and grows at only 0.01-0.04/day, but its climate is shifted (OLR
  -20 W/m2, net TOA +14 W/m2, jets 10-11 m/s stronger, the NH jet ~7 deg
  further poleward).
  At L16 and L24 its ensembles blow up at 1x and 2x damping; only T31-L16 at
  4x survives (growth ~0.005/day).

## Knob matrix: growth and climatology together

Each row is one configuration: `growth.py` (4 seeds, 60 d, OLR tangent,
fit days 5-40, tropical / extratropical bump) and `climatology.py` (4 seeds,
200 d spin-up, 120-d mean; frozen BCs, perpetual equinox). Climatology
columns give global means with the difference from the classic T31-L8
Eulerian SPEEDY in parentheses; "sd" is the across-seed standard deviation of
net TOA (the noise scale). Suffixes: `-d<k>` hyperdiffusion x k, `-vlinear`
linear vertical SL interpolation, `hyb` the campaign hybrid dycore.
Regenerate with `python matrix_table.py <dir> --ref T31-L8-eul`.

| config | growth trop / extra (1/day) | OLR | abs. SW | net TOA (sd) | precip (mm/d) | cloud | NH jet m/s @lat | SH jet m/s @lat | finite |
|---|---|---|---|---|---|---|---|---|---|
| T21-L8-eul | 0.103 / 0.097 | 226 (-4) | 237 (-2) | 10.7 (+1.7) (0.0) | 2.55 (-0.13) | 0.62 (+0.03) | 29 @36 | 34 @-30 | 4/4 |
| T21-L8-eul-d2 | 0.046 / 0.057 | 222 (-8) | 234 (-5) | 12.4 (+3.3) (0.1) | 2.42 (-0.26) | 0.65 (+0.06) | 29 @42 | 35 @-30 | 4/4 |
| T21-L8-hyb | 0.014 / 0.009 | 210 (-20) | 233 (-6) | 22.7 (+13.6) (0.0) | 2.36 (-0.33) | 0.66 (+0.08) | 37 @47 | 38 @-36 | 4/4 |
| T21-L8-hyb-d2 | 0.011 / 0.004 | 210 (-20) | 233 (-6) | 23.2 (+14.1) (0.0) | 2.33 (-0.35) | 0.67 (+0.08) | 36 @47 | 37 @-36 | 4/4 |
| T21-L8-sl | 0.069 / 0.078 | 222 (-8) | 238 (-2) | 15.4 (+6.3) (0.1) | 2.57 (-0.11) | 0.61 (+0.02) | 29 @36 | 34 @-30 | 4/4 |
| T21-L8-sl-d2 | 0.039 / 0.041 | 219 (-11) | 235 (-4) | 15.9 (+6.8) (0.0) | 2.47 (-0.21) | 0.64 (+0.06) | 28 @36 | 35 @-30 | 4/4 |
| T21-L8-sl-d2-vlinear | 0.025 / 0.022 | 212 (-18) | 234 (-5) | 22.3 (+13.2) (0.1) | 2.62 (-0.06) | 0.65 (+0.07) | 27 @42 | 31 @-30 | 4/4 |
| T21-L8-sl-vlinear | 0.039 / 0.058 | 214 (-16) | 236 (-3) | 22.4 (+13.3) (0.1) | 2.76 (+0.08) | 0.63 (+0.05) | 27 @42 | 30 @-30 | 4/4 |
| T21-L16-eul | 0.055 / 0.084 | 238 (+8) | 264 (+24) | 25.3 (+16.2) (0.1) | 1.59 (-1.10) | 0.42 (-0.17) | 23 @42 | 26 @-36 | 4/4 |
| T21-L16-eul-d2 | 0.023 / 0.044 | 237 (+7) | 263 (+24) | 26.8 (+17.8) (0.1) | 1.55 (-1.13) | 0.43 (-0.16) | 22 @42 | 26 @-36 | 4/4 |
| T21-L16-hyb | 0.331 / 0.229 | 271 (+41) | 260 (+21) | -11.2 (-20.3) (11.0) | 1.72 (-0.97) | 0.32 (-0.27) | 25 @47 | 26 @-58 | 3/4 |
| T21-L16-hyb-d2 | blew up | blew up | | | | | | | 0/4 |
| T21-L16-sl | 0.036 / 0.063 | 239 (+9) | 264 (+25) | 25.4 (+16.3) (0.1) | 1.61 (-1.08) | 0.41 (-0.17) | 19 @42 | 24 @-36 | 4/4 |
| T21-L16-sl-d2 | 0.020 / 0.027 | 238 (+8) | 265 (+26) | 26.8 (+17.7) (0.1) | 1.57 (-1.11) | 0.41 (-0.17) | 19 @47 | 25 @-36 | 4/4 |
| T21-L16-sl-d2-vlinear | 0.022 / 0.021 | 235 (+5) | 260 (+20) | 24.6 (+15.5) (0.0) | 1.58 (-1.10) | 0.45 (-0.13) | 19 @47 | 24 @-36 | 4/4 |
| T21-L16-sl-vlinear | 0.027 / 0.043 | 237 (+7) | 260 (+21) | 23.3 (+14.2) (0.1) | 1.64 (-1.05) | 0.44 (-0.14) | 18 @42 | 22 @-36 | 4/4 |
| T31-L8-eul | 0.182 / 0.184 | 230 (+0) | 239 (+0) | 9.1 (+0.0) (0.2) | 2.68 (+0.00) | 0.59 (+0.00) | 28 @39 | 31 @-32 | 4/4 |
| T31-L8-eul-d16 | 0.026 / 0.008 | 215 (-15) | 230 (-9) | 15.3 (+6.2) (0.0) | 2.33 (-0.36) | 0.69 (+0.10) | 32 @28 | 39 @-28 | 4/4 |
| T31-L8-eul-d2 | 0.119 / 0.134 | 227 (-3) | 237 (-2) | 10.0 (+0.9) (0.1) | 2.60 (-0.08) | 0.61 (+0.03) | 29 @43 | 33 @-32 | 4/4 |
| T31-L8-eul-d4 | 0.088 / 0.095 | 224 (-6) | 235 (-5) | 10.6 (+1.5) (0.1) | 2.51 (-0.17) | 0.65 (+0.06) | 29 @39 | 35 @-32 | 4/4 |
| T31-L8-hyb | 0.043 / 0.034 | 210 (-20) | 233 (-6) | 22.8 (+13.7) (0.0) | 2.42 (-0.27) | 0.67 (+0.09) | 39 @46 | 41 @-35 | 4/4 |
| T31-L8-hyb-d2 | 0.040 / 0.016 | 209 (-21) | 232 (-7) | 23.7 (+14.6) (0.0) | 2.37 (-0.31) | 0.68 (+0.09) | 39 @46 | 39 @-35 | 4/4 |
| T31-L8-hyb-d4 | 0.018 / 0.012 | 208 (-22) | 232 (-7) | 24.1 (+15.0) (0.0) | 2.39 (-0.29) | 0.68 (+0.09) | 37 @50 | 37 @-39 | 4/4 |
| T31-L8-sl | 0.143 / 0.136 | 224 (-6) | 238 (-1) | 14.6 (+5.5) (0.1) | 2.73 (+0.05) | 0.59 (+0.00) | 29 @35 | 34 @-28 | 4/4 |
| T31-L8-sl-d2 | 0.091 / 0.120 | 222 (-8) | 237 (-2) | 15.1 (+6.0) (0.0) | 2.66 (-0.02) | 0.61 (+0.03) | 30 @39 | 35 @-28 | 4/4 |
| T31-L8-sl-d2-vlinear | 0.058 / 0.071 | 213 (-17) | 235 (-4) | 22.4 (+13.3) (0.0) | 2.86 (+0.17) | 0.64 (+0.05) | 28 @43 | 29 @-32 | 4/4 |
| T31-L8-sl-d4 | 0.073 / 0.086 | 220 (-10) | 235 (-4) | 14.9 (+5.8) (0.0) | 2.57 (-0.12) | 0.64 (+0.05) | 31 @43 | 36 @-28 | 4/4 |
| T31-L8-sl-d4-vlinear | 0.042 / 0.042 | 211 (-19) | 233 (-6) | 21.9 (+12.8) (0.0) | 2.72 (+0.04) | 0.66 (+0.08) | 29 @43 | 31 @-32 | 4/4 |
| T31-L8-sl-vlinear | 0.091 / 0.095 | 214 (-16) | 237 (-3) | 22.8 (+13.7) (0.2) | 2.97 (+0.28) | 0.62 (+0.03) | 26 @43 | 27 @-32 | 4/4 |
| T31-L16-eul | 0.144 / 0.197 | 239 (+9) | 262 (+23) | 23.3 (+14.2) (0.1) | 1.63 (-1.05) | 0.42 (-0.16) | 25 @46 | 28 @-46 | 4/4 |
| T31-L16-eul-d2 | 0.077 / 0.103 | 238 (+8) | 262 (+23) | 24.3 (+15.3) (0.0) | 1.59 (-1.09) | 0.43 (-0.16) | 23 @46 | 26 @-50 | 4/4 |
| T31-L16-eul-d4 | 0.043 / 0.071 | 237 (+7) | 262 (+23) | 25.3 (+16.2) (0.1) | 1.57 (-1.11) | 0.44 (-0.15) | 22 @43 | 26 @-39 | 4/4 |
| T31-L16-hyb | blew up | blew up | | | | | | | 0/4 |
| T31-L16-hyb-d2 | 0.278 / 0.167 | 288 (+58) | 260 (+21) | -28.0 (-37.0) (0.0) | 2.37 (-0.31) | 0.28 (-0.31) | 23 @46 | 29 @-65 | 1/4 |
| T31-L16-hyb-d4 | 0.006 / 0.004 | 232 (+2) | 258 (+19) | 25.7 (+16.6) (0.0) | 1.53 (-1.15) | 0.43 (-0.16) | 31 @50 | 21 @-32 | 4/4 |
| T31-L16-sl | 0.110 / 0.136 | 239 (+9) | 262 (+23) | 23.7 (+14.6) (0.0) | 1.64 (-1.04) | 0.42 (-0.17) | 25 @43 | 25 @-39 | 4/4 |
| T31-L16-sl-d2 | 0.066 / 0.098 | 238 (+8) | 263 (+24) | 24.8 (+15.8) (0.1) | 1.62 (-1.07) | 0.43 (-0.16) | 21 @43 | 24 @-35 | 4/4 |
| T31-L16-sl-d2-vlinear | 0.049 / 0.069 | 236 (+6) | 258 (+19) | 22.0 (+12.9) (0.0) | 1.65 (-1.03) | 0.46 (-0.12) | 21 @46 | 22 @-39 | 4/4 |
| T31-L16-sl-d4 | 0.031 / 0.053 | 238 (+8) | 263 (+24) | 25.6 (+16.5) (0.1) | 1.59 (-1.10) | 0.43 (-0.16) | 20 @43 | 24 @-39 | 4/4 |
| T31-L16-sl-d4-vlinear | 0.028 / 0.062 | 235 (+5) | 258 (+19) | 22.8 (+13.7) (0.1) | 1.62 (-1.06) | 0.47 (-0.11) | 19 @46 | 22 @-39 | 4/4 |
| T31-L16-sl-vlinear | 0.089 / 0.124 | 237 (+7) | 259 (+20) | 21.5 (+12.5) (0.1) | 1.69 (-0.99) | 0.45 (-0.13) | 22 @43 | 22 @-43 | 4/4 |
| T31-L24-eul | 0.154 / 0.189 | 242 (+12) | 263 (+24) | 20.7 (+11.6) (0.1) | 1.40 (-1.28) | 0.41 (-0.17) | 22 @50 | 29 @-50 | 4/4 |
| T31-L24-eul-d2 | 0.070 / 0.127 | 242 (+12) | 263 (+24) | 21.0 (+12.0) (0.2) | 1.37 (-1.32) | 0.42 (-0.17) | 20 @46 | 27 @-54 | 4/4 |
| T31-L24-eul-d4 | 0.044 / 0.063 | 241 (+11) | 261 (+22) | 19.9 (+10.8) (0.2) | 1.35 (-1.33) | 0.44 (-0.15) | 19 @46 | 25 @-54 | 4/4 |
| T31-L24-hyb | blew up | blew up | | | | | | | 0/4 |
| T31-L24-hyb-d2 | 0.641 / 0.699 | blew up | | | | | | | 0/4 |
| T31-L24-hyb-d4 | blew up | blew up | | | | | | | 0/4 |
| T31-L24-sl | 0.106 / 0.163 | 243 (+13) | 264 (+25) | 21.2 (+12.1) (0.1) | 1.42 (-1.27) | 0.40 (-0.18) | 21 @46 | 24 @-46 | 4/4 |
| T31-L24-sl-d2 | 0.064 / 0.099 | 243 (+13) | 264 (+25) | 21.5 (+12.4) (0.1) | 1.39 (-1.29) | 0.41 (-0.17) | 20 @46 | 23 @-50 | 4/4 |
| T31-L24-sl-d2-vlinear | 0.044 / 0.080 | 244 (+14) | 262 (+23) | 17.9 (+8.9) (0.1) | 1.42 (-1.26) | 0.43 (-0.16) | 19 @46 | 22 @-50 | 4/4 |
| T31-L24-sl-d4 | 0.044 / 0.062 | 242 (+12) | 263 (+24) | 20.9 (+11.8) (0.1) | 1.37 (-1.31) | 0.42 (-0.16) | 19 @46 | 23 @-43 | 4/4 |
| T31-L24-sl-d4-vlinear | 0.026 / 0.041 | 243 (+13) | 261 (+22) | 17.6 (+8.5) (0.1) | 1.41 (-1.27) | 0.44 (-0.14) | 17 @46 | 22 @-43 | 4/4 |
| T31-L24-sl-vlinear | 0.092 / 0.119 | 244 (+14) | 262 (+23) | 18.2 (+9.1) (0.1) | 1.45 (-1.23) | 0.42 (-0.17) | 20 @46 | 23 @-50 | 4/4 |

What it shows:

* **Hyperdiffusion is the cheap lever.** 2x roughly halves growth at small
  climate cost (T31-L8: 0.12/day; OLR -3, cloud +0.03). 4x gives 0.09/day
  (OLR -6, cloud +0.06). 16x reaches 0.01-0.03/day but is a different climate
  (OLR -15 W/m2, cloud +0.10, large-scale precipitation -40%, jets 4-8 m/s
  stronger, the NH jet ~10 deg equatorward).
* **`main`'s SL alone helps only a little** (T31-L8: 0.14 vs 0.18/day), with
  a near-identical climate apart from net TOA (+5 W/m2).
* **Linear vertical interpolation slows growth further, but at L8 it costs
  ~16-18 W/m2 of OLR**; at L16 and L24 its OLR cost is only ~2 W/m2. A
  coarse-level interaction.
* **The hybrid dycore is the slowest-growing at L8 and unusable above it**
  without heavy damping (see the SL section).
* **L16 and L24 SPEEDY are untuned with every dycore**: against T31-L8, cloud
  ~0.16 lower, absorbed SW ~23 W/m2 higher, precipitation ~1.1 mm/day lower.
  The parameters were tuned at L8; recalibrate before using L16+ for science
  (an earlier ensemble Kalman calibration at T31-L16 restored cloud cover to
  ~0.68).
* Vertical resolution changes how the other knobs behave: the hybrid dycore
  is stable only at L8, linear interpolation's OLR cost only shows up at L8,
  and the damping response is similar at every level count.

## 2. GFMIP finite-difference atlas

```bash
python make_patches.py --truncation 31 --out runs
python spinup.py --truncation 31 --nlev 16 --seasonal --n 16 --days 365 --out runs
python gfmip_fd.py --truncation 31 --nlev 16 --members 16 --out runs        # resumable
python gfmip_analysis.py --truncation 31 --nlev 16 --out runs \
    --gfmip_dir /path/to/preliminary_spatial_feedbacks/data
```

* `make_patches.py` builds the 109 cos^2 patches (20 deg x 80 deg boxes, wider
  poleward of 30 deg, 50% overlap), stored at unit peak amplitude with their
  ice-free-ocean weights. At T31 it reproduces the campaign's lattice exactly.
* `gfmip_fd.py` runs paired `+A`/`-A` (default 4 K) perturbations from the same
  library states, discards `--spin_days` (60) and averages `--avg_days` (360),
  and saves per-member symmetric sensitivities `(warm - cool) / 2A` of
  time-mean OLR, absorbed SW and precipitation maps per patch. Existing patch
  files are skipped, so reruns resume; `--lane i --nlanes n` splits patches
  across GPUs; `touch runs/gfmip-fd-<tag>/STOP` stops it between groups.
* `gfmip_analysis.py` converts each patch to an ocean-mean response per K of
  patch-mean SST, de-overlaps (`dN/dSST*(x) = sum_p d_p dSST_p(x) / sum_p
  dSST_p(x)`), and reports pattern correlations against each GFMIP model and
  their mean, over ice-free ocean. The GFMIP reference maps are at
  https://github.com/GFMIP/preliminary_spatial_feedbacks.

Campaign reference: SPEEDY T31-L16 with 16 members and 14 months (first 2
discarded) reached r = 0.74 against the GFMIP multi-model mean (T31-L8:
~0.67).

## 3. Ensemble jvp chains

```bash
export JCM_PERPETUAL_EQUINOX=1
python make_patches.py --truncation 31 --out runs
python spinup.py --truncation 31 --nlev 16 --n 256 --days 200 --out runs
python chains.py --truncation 31 --nlev 16 --window 1 --niter 60 --nseed 32 --out runs
```

Each iteration draws `--nseed` states from the library, propagates the current
tangent `v` plus the patch's SST tangent through a `--window`-day run, and
averages the resulting state tangents into the next `v`:
`v_{k+1} = mean_s jvp(F, (x_s, sst0), (v_k, dsst_p))`. It records the ensemble-mean
tangent of the window-mean ocean-mean observables per K of patch-mean SST:
iteration 0 is the transient sensitivity, and a late plateau is the chain's
equilibrium. A patch stops if `|v|` exceeds `--blowup` times its first value.
Output: `runs/chains-<tag>-w<window>d.npz` with `obs_traj (patch, iteration,
observable)` and `vnorms`; `--save_tangents` adds each patch's final state
tangent. `touch runs/STOP_CHAINS` stops between patches.

Campaign reference (T31-L16, 32 members, windows 0.5/1/2 d over a 60-day
horizon): all 327 chains converged within ~20-30 d, the equilibrium did not
depend on the window and its amplitude matched FD, but its pattern correlated
only ~0.43 with the FD equilibrium (linear vs finite response around
convective thresholds, not sampling noise).

## Costs and memory

On one shared A100 (jcm-v3 environment): a T31-L8 or L16 `growth.py` run
(4 seeds, 200-d spin-up + 60-d jvp) takes ~2-4 min, a `climatology.py` run
~2-3 min, L24 and hybrid configurations ~1.5-2x that. The 104-run matrix took
~2.5 h on two shared GPUs. For the FD atlas, the campaign's T31-L16 run (all
109 patches, 16 members, 14 months) took ~6 GPU-hours. A chain iteration costs
`nseed` jvps of one window (a jvp is ~2.1x a forward run).
