"""Online rollout curriculum across many cached years (stage 2 of the chain).

The stage the offline warm start cannot skip. An offline term is fitted to
instantaneous tendencies only; integrate it free for weeks and it goes
unstable, which is why warm-starting the climatology fine-tune directly from
an offline term returns NaN. This stage backpropagates through short free
rollouts (6 h -> 12 h -> 24 h leads) so the term stays stable when it runs.

Starts come from the per-year cache and step through the SEASONAL CYCLE as
well as through the years, so consecutive starts differ in both. That coverage
is load-bearing: a term trained only on mid-January states has never seen July
and goes non-finite in the 450-day evaluation. (The climatology stage is the
opposite case and keeps a fixed seasonal window; see
multiyear.climatology_starts.)

    # de-risk: shortest stage only, few updates. One entry per stage.
    CUDA_VISIBLE_DEVICES=1 python tools/bias_correction/train_online_multiyear.py \
        jcm/data/bias_correction/offline_term_t31_20yr.npz /tmp/online_smoke.npz \
        --years 1995-2015 --start-pool 21 --updates 5,0,0,0

    # the real run
    nohup env CUDA_VISIBLE_DEVICES=1 python tools/bias_correction/train_online_multiyear.py \
        jcm/data/bias_correction/offline_term_t31_20yr.npz \
        jcm/data/bias_correction/online_term_t31_20yr.npz \
        --years 1995-2015 --start-pool 21 > train_online_20yr.log 2>&1 &

Its output is the warm start for finetune_climatology.py --years, which is the
stage that produced clim_vt.
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

# Shared GPUs: never preallocate (see build_era5_cache.py). Before jax loads.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from importlib import resources

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_datetime as jdt

from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.physics.bias_correction import (
    CurriculumStage, lat_weights, load_bias_correction, stds_from_stats,
    train_online)
from jcm.physics.bias_correction.era5_cache import cache_meta
from jcm.physics.bias_correction.multiyear import (
    parse_year_span, rollout_starts)
from jcm.physics.bias_correction.offline_training import build_term
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.terrain import TerrainData

TRUNC = int(os.environ.get("M4_TRUNC", "31"))
DT_MIN = float(os.environ.get("M4_DT_MIN", "30"))
STEPS_PER_DAY = int(round(24 * 60 / DT_MIN))
# Model steps per 6-hourly ERA5 slot. Everything converting a rollout length
# into a target slot must go through this, or a shorter model step silently
# fetches the target from the wrong time.
STEPS_PER_SLOT = STEPS_PER_DAY // 4

ap = argparse.ArgumentParser()
ap.add_argument("in_npz")
ap.add_argument("out_npz")
ap.add_argument("--years", required=True)
ap.add_argument("--cache-root", default=None)
ap.add_argument("--start-pool", type=int, default=21)
# The 96-step stage is load-bearing: it is the only one that sees a 48 h lead,
# the cheapest push toward constraining drift. Without it the term
# is excellent at 24 h and badly conditioned over a 90-day free run, which is
# what the climatology fine-tune integrates.
ap.add_argument("--stages", default="12,24,48,96",
                help="rollout lengths in MODEL STEPS, one per stage "
                     "(6 h/12 h/24 h/48 h at a 30-minute step). Each must be "
                     "a multiple of the steps-per-ERA5-slot.")
# TOTAL updates is what matters, not passes per start. The single-year recipe
# ran 2 epochs over 237 starts, about 1660 updates overall; scaling epochs to a
# 21-start pool gave 147 and left the term still unstable in a 450-day free
# run. An offline term is unstable by construction, and this stage is what
# fixes it, so it needs the update count rather than the epoch count.
ap.add_argument("--updates", default="500,500,500,250",
                help="Adam updates per stage, about 1750 in total; fewer "
                     "leaves the term non-finite when free-run.")
ap.add_argument("--epochs", default=None,
                help="passes over the start pool per stage, an alternative to "
                     "--updates (updates = epochs x pool size)")
ap.add_argument("--lrs", default="1e-4,5e-5,5e-5,2.5e-5",
                help="learning rate per stage, same length as --stages")
ap.add_argument("--clip-norm", type=float, default=1.0)
ap.add_argument("--pole-floor", type=float, default=0.3)
ap.add_argument("--spinup-days", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()


def _ints(spec, name):
    try:
        return [int(x) for x in str(spec).split(",") if x.strip() != ""]
    except ValueError as err:
        raise SystemExit(
            f"--{name} must be comma-separated integers: {spec!r}") from err


stage_steps = _ints(a.stages, "stages")
stage_lrs = [float(x) for x in a.lrs.split(",") if x.strip() != ""]
if a.updates:
    stage_updates = _ints(a.updates, "updates")
    stage_epochs = [None] * len(stage_updates)
else:
    stage_epochs = [float(x) for x in a.epochs.split(",") if x.strip() != ""]
    stage_updates = [max(1, int(round(e * a.start_pool))) for e in stage_epochs]
if not (len(stage_steps) == len(stage_updates) == len(stage_lrs)):
    raise SystemExit("--stages, --lrs and --epochs/--updates must have "
                     "equal length")
for n in stage_steps:
    if n % STEPS_PER_SLOT:
        raise SystemExit(
            f"stage {n} steps is not a multiple of {STEPS_PER_SLOT}; ERA5 "
            f"targets are 6-hourly so a rollout must land on a slot")

coords = get_speedy_coords(layers=8, spectral_truncation=TRUNC)
nlev = int(coords.vertical.layers)
bc = resources.files("jcm.data.bc.t30.clim")
terrain = TerrainData.from_file(bc / "terrain.nc", coords=coords)
forcing = ForcingData.from_file(bc / "forcing.nc", coords=coords)

base = load_bias_correction(a.in_npz)
# get_value(), not .value: the latter is deprecated on nnx Variables and
# matches how finetune_climatology.py reads a warm start.
in_mean = base.in_mean.get_value()
in_std = base.in_std.get_value()
out_scale = base.out_scale.get_value()
correct = base.correct
start_layers = base.weights.get_value()
print(f"[online] warm start {a.in_npz}: correct={correct} "
      f"cap={base.output_cap} context={base.context_features}", flush=True)

# The calendar year of a start never reaches the model (climatological ocean,
# insolation depends only on fraction-of-year), so any start_date works; the
# day of year rides in through each sample's sim_time.
start_date = jdt.to_datetime("2001-01-01")


def build_model(layers):
    term = build_term(layers, in_mean, in_std, out_scale, correct,
                      output_cap=base.output_cap,
                      polar_taper=base.polar_taper,
                      surface_taper=base.surface_taper,
                      context_features=base.context_features,
                      activation=base.activation)
    return Model(coords=coords, terrain=terrain,
                 physics=speedy_physics() + term,
                 start_date=start_date, calendar="365_day",
                 time_step=DT_MIN)


probe = build_model(start_layers)

span = parse_year_span(a.years)
meta = cache_meta(coords, cadence_hours=6, save_days=1.25,
                  tau_seconds=21600.0, time_step_minutes=DT_MIN,
                  spinup_days=a.spinup_days)
leads = [n // STEPS_PER_SLOT for n in stage_steps]
t0 = time.time()
starts = rollout_starts(a.cache_root or os.path.expanduser(f"~/era5_cache_t{TRUNC}"),
                        span, meta=meta, n_starts=a.start_pool,
                        lead_slots=leads, spinup_days=a.spinup_days)
print(f"[online] {len(starts)} starts from {len(span)} years "
      f"({span[0]}..{span[-1]}) in {time.time() - t0:.0f}s; "
      f"stage leads (slots) {leads}", flush=True)


def stack(trees):
    return jtu.tree_map(lambda *xs: jnp.stack(xs), *trees)


inits = stack([probe.dycore.initial_state(
    r["state"], sim_time=r["sim_time"],
    tracer_specs=probe.dycore.tracer_specs) for r in starts])

samples = {}
for n_steps, lead in zip(stage_steps, leads, strict=True):
    tT = jnp.stack([r["targets"][lead][0] for r in starts])
    tq = jnp.stack([r["targets"][lead][1] for r in starts])
    samples[n_steps] = (inits, tT, tq)

T_std, q_std = stds_from_stats(in_std, nlev)
weights = lat_weights(coords, pole_weight_floor=a.pole_floor)
stages = tuple(CurriculumStage(n_steps=n, updates=u, lr=lr,
                               clip_norm=a.clip_norm)
               for n, u, lr in zip(stage_steps, stage_updates, stage_lrs,
                                   strict=True))
for s, e in zip(stages, stage_epochs, strict=True):
    ep = f"  ({e:g} epochs over {len(starts)} starts)" if e else ""
    print(f"[online] stage {s.n_steps} steps "
          f"({s.n_steps / STEPS_PER_DAY * 24:.0f} h lead)  "
          f"updates={s.updates}  lr={s.lr:g}{ep}", flush=True)


def stage_end(idx, stage, _layers):
    print(f"[online] stage {idx} ({stage.n_steps} steps) done "
          f"(+{time.time() - t0:.0f}s)", flush=True)


# train_online returns (layers, history), not just layers.
layers, history = train_online(start_layers, build_model, forcing, samples,
                               stages, T_std=T_std, q_std=q_std,
                               weights=weights, time_step_minutes=DT_MIN,
                               key=jax.random.key(a.seed),
                               stage_end_hook=stage_end)

if history:
    n_bad = sum(1 for h in history if not h.get("ok", True))
    print("[online] %d updates, %d rejected; loss %.5f -> %.5f"
          % (len(history), n_bad, history[0]["loss"], history[-1]["loss"]),
          flush=True)
    for s in stages:
        rows = [h for h in history if h["n_steps"] == s.n_steps]
        if rows:
            print("[online]   stage %3d steps: loss %.5f -> %.5f  "
                  "grad_norm %.3e -> %.3e"
                  % (s.n_steps, rows[0]["loss"], rows[-1]["loss"],
                     rows[0]["grad_norm"], rows[-1]["grad_norm"]), flush=True)

finite = all(bool(jnp.isfinite(x).all()) for x in jtu.tree_leaves(layers))
if not finite:
    sys.exit("trained layers are non-finite; term NOT written")

term = build_term(layers, in_mean, in_std, out_scale, correct,
                  output_cap=base.output_cap,
                  polar_taper=base.polar_taper,
                  surface_taper=base.surface_taper,
                  context_features=base.context_features,
                  activation=base.activation)
term.save(a.out_npz)
print(f"[online] wrote {a.out_npz}  (total {time.time() - t0:.0f}s)", flush=True)
