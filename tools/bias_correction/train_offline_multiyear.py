"""Offline pretrain of the bias-correction term across many cached years.

Stage 1 of the training chain. Reads the per-year cache built by
build_era5_cache.py, so the expensive nudged runs happen once and every retrain
is cheap. The result is a warm start only: an offline term is unstable in a free
run by construction, and the rollout stage (train_online_multiyear.py) is what
fixes that.

    # smoke: two cached years, few steps, confirm the pipeline end to end
    python tools/bias_correction/train_offline_multiyear.py --years 2003-2004 --steps 200 \
        --out /tmp/offline_smoke.npz

    # the real warm start, 21 years
    nohup env CUDA_VISIBLE_DEVICES=1 python tools/bias_correction/train_offline_multiyear.py \
        --years 1995-2015 --out jcm/data/bias_correction/offline_term_t31_20yr.npz \
        > train_offline_20yr.log 2>&1 &

--frame-stride subsamples the saved frames (default 4, i.e. one state every
five days per year). Frames 1.25 days apart are strongly correlated, so this
buys sample diversity across seasons and years far more cheaply than keeping
every frame of a single year.
"""
# Imports sit below the environment setup on purpose: the XLA setting below has
# to be decided before jax is imported, and jax arrives with the first jcm
# import. Ruff cannot see that constraint, so E402 is waived for this file.
# ruff: noqa: E402
import argparse
import os
import sys
import time

# Repo root, two directories up; JCM_REPO overrides. See build_era5_cache.py.
REPO = os.environ.get("JCM_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

# Shared GPUs: never let XLA preallocate ~75% of the card (see
# build_era5_cache.py). Must be set before jax is imported below.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import xarray as xr

from jcm.physics.bias_correction.era5_cache import cache_meta
from jcm.physics.bias_correction.multiyear import (
    offline_dataset, parse_year_span)
from jcm.physics.bias_correction.nn_bias_correction import (
    ACTIVATIONS, init_mlp)
from jcm.physics.bias_correction.offline_training import (
    build_term, compute_norm_stats, normalized_mse, train)
from jcm.physics.speedy.speedy_coords import get_speedy_coords

TRUNC = int(os.environ.get("M4_TRUNC", "31"))
DT_MIN = float(os.environ.get("M4_DT_MIN", "30"))
CORRECT = ("temperature", "specific_humidity")
# The default 32-64-64-32 net has 8,352 parameters. The shipped term was trained
# with --hidden 256,256,256 (148,512 parameters); width is what separated the
# winter result from the small-net one, see docs/source/bias_correction.rst.
DEFAULT_HIDDEN = "64,64"


ap = argparse.ArgumentParser()
ap.add_argument("--years", required=True)
ap.add_argument("--val-years", default=None,
                help="held-out years to score after fitting. Training MSE "
                     "falls with fewer, more homogeneous samples, so it "
                     "cannot say whether more years generalise better; this "
                     "can. Scored with the TRAINING normalisation, which is "
                     "what the live term applies.")
ap.add_argument("--root", default=None, help="cache dir")
ap.add_argument("--out", required=True, help="output term .npz")
ap.add_argument("--stats-out", default=None,
                help="offline stats netCDF (default alongside --out)")
ap.add_argument("--hidden", default=DEFAULT_HIDDEN,
                help="hidden layer widths, comma separated (default 64,64 = "
                     "the shipped 8352-parameter net)")
ap.add_argument("--frame-stride", type=int, default=4)
ap.add_argument("--train-spinup-days", type=float, default=None,
                help="leading days discarded per year (default: the value the "
                     "cache recorded). Each year is its own nudged run, so "
                     "each has a convergence transient at its start.")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch-size", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--activation", default="tanh",
                help="hidden-layer nonlinearity. tanh is the shipped default "
                     "and was never varied; at 512 wide it scores held-out "
                     "0.3422 against gelu's 0.3087 on the same data and "
                     "training code. Stored in the saved term, so the later "
                     "stages inherit it.")
ap.add_argument("--seed", type=int, default=0)
# Recorded in the cache metadata; must match what the cache was built with or
# the labels would not be the nudge that was actually applied.
ap.add_argument("--cadence-hours", type=int, default=6)
ap.add_argument("--save-days", type=float, default=1.25)
ap.add_argument("--tau-hours", type=float, default=6.0)
ap.add_argument("--spinup-days", type=float, default=10.0)
a = ap.parse_args()

years = parse_year_span(a.years)
root = a.root or os.path.expanduser(f"~/era5_cache_t{TRUNC}")
coords = get_speedy_coords(layers=8, spectral_truncation=TRUNC)
meta = cache_meta(coords, cadence_hours=a.cadence_hours,
                  save_days=a.save_days, tau_seconds=a.tau_hours * 3600.0,
                  time_step_minutes=DT_MIN, spinup_days=a.spinup_days)

print("T%d grid %s" % (TRUNC, coords.horizontal.nodal_shape), flush=True)
print("cache: %s" % root, flush=True)
print("years: %d (%d..%d), frame stride %d"
      % (len(years), years[0], years[-1], a.frame_stride), flush=True)

t0 = time.time()
feats, targets = offline_dataset(root, years, meta=meta,
                                 frame_stride=a.frame_stride,
                                 spinup_days=a.train_spinup_days)
print("dataset: %s rows x %d features in %.0fs"
      % (f"{feats.shape[0]:,}", feats.shape[1], time.time() - t0), flush=True)

in_mean, in_std, out_scale = compute_norm_stats(feats, targets, CORRECT)
print("out_scale: min %.3e  max %.3e (NO variance floor, by design)"
      % (float(np.min(out_scale)), float(np.max(out_scale))), flush=True)

n_io = feats.shape[1]
hidden = tuple(int(x) for x in a.hidden.split(",") if x.strip() != "")
sizes = (n_io, *hidden, n_io)
n_params = sum(a_ * b + b for a_, b in zip(sizes[:-1], sizes[1:], strict=True))
print("network: %s  (%s parameters)"
      % ("-".join(str(s) for s in sizes), f"{n_params:,}"), flush=True)

# Train from a NON-zero output layer. With a zero last
# layer the hidden layers see no gradient at first, so the fit degenerates
# toward a linear map on random features and the resulting term is unstable in
# a long free run (it went non-finite at 450 days).
start = init_mlp(jax.random.key(a.seed), sizes, zero_last_layer=False)
# The zero-output network is the no-op baseline the MSE ratio is measured
# against, not the training start.
zero = init_mlp(jax.random.key(a.seed), sizes, zero_last_layer=True)

base_mse = float(normalized_mse(zero, feats, targets, in_mean, in_std,
                                out_scale, CORRECT,
                                ACTIVATIONS[a.activation]))
if a.activation not in ACTIVATIONS:
    sys.exit(f"unknown activation {a.activation!r}; "
             f"choose from {sorted(ACTIVATIONS)}")
t1 = time.time()
trained, history = train(start, feats, targets, in_mean, in_std, out_scale,
                         CORRECT, steps=a.steps, batch_size=a.batch_size,
                         lr=a.lr, key=jax.random.key(a.seed + 1),
                         activation=ACTIVATIONS[a.activation])
fit_mse = float(normalized_mse(trained, feats, targets, in_mean, in_std,
                               out_scale, CORRECT,
                                ACTIVATIONS[a.activation]))
print("trained %d steps in %.0fs" % (a.steps, time.time() - t1), flush=True)
print("MSE train: %.4f -> %.4f  (ratio %.3f; lower is more of the nudge "
      "explained)" % (base_mse, fit_mse, fit_mse / base_mse), flush=True)
if not np.isfinite(fit_mse):
    sys.exit("training produced a non-finite loss; term NOT written")

if a.val_years:
    # Scored with the TRAINING stats: in_mean/in_std/out_scale are part of the
    # fitted term and are what it applies at run time, so recomputing them on
    # the validation years would score a different term than the one saved.
    val_years = parse_year_span(a.val_years)
    print("validating on %d held-out years (%d..%d)"
          % (len(val_years), val_years[0], val_years[-1]), flush=True)
    v_feats, v_targets = offline_dataset(root, val_years, meta=meta,
                                         frame_stride=a.frame_stride,
                                         spinup_days=a.train_spinup_days)
    v_base = float(normalized_mse(zero, v_feats, v_targets, in_mean, in_std,
                                  out_scale, CORRECT,
                                ACTIVATIONS[a.activation]))
    v_fit = float(normalized_mse(trained, v_feats, v_targets, in_mean, in_std,
                                 out_scale, CORRECT,
                                ACTIVATIONS[a.activation]))
    print("MSE   val: %.4f -> %.4f  (ratio %.3f)  [%s rows]"
          % (v_base, v_fit, v_fit / v_base, f"{v_feats.shape[0]:,}"),
          flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
term = build_term(trained, in_mean, in_std, out_scale, CORRECT,
                  activation=a.activation)
term.save(a.out)

stats_out = a.stats_out or os.path.join(
    os.path.dirname(os.path.abspath(a.out)),
    os.path.basename(a.out).replace(".npz", "_stats.nc"))
xr.Dataset({
    "in_mean": ("feature", np.asarray(in_mean)),
    "in_std": ("feature", np.asarray(in_std)),
    "out_scale": ("feature", np.asarray(out_scale)),
}).to_netcdf(stats_out)

print("wrote %s" % a.out, flush=True)
print("wrote %s" % stats_out, flush=True)
print("total %.0fs" % (time.time() - t0), flush=True)
