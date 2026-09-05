"""Fine-tune a trained bias-correction term on a long-horizon CLIMATOLOGY loss.

Warm-starts from a saved term, runs a FREE rollout for a multi-week window,
takes its differentiable time-mean, and penalises the area-weighted, pole-
weighted bias of that mean vs an ERA5 time-mean over the same window -- the
long-run objective that short rollouts are blind to (see
online_training.make_climatology_loss). Preserves the warm-start term's full
config (surface/polar taper, context features, output cap) through the rebuild
and re-saves. Style mirrors add_surface_taper.py / run_nb.py.

    # de-risk: one fwd+bwd, no optimizer -- confirm the AD path is finite + time it
    python tools/bias_correction/finetune_climatology.py <in.npz> <out.npz> --n-steps 1440 --probe-only

    # micro-run (30-day window), then scale to --n-steps 4320 (90 days) in the background
    python tools/bias_correction/finetune_climatology.py <in.npz> <out.npz> \
        --n-steps 1440 --updates 30 --lr 5e-5 --year 2001 \
        --spinup-days 10 --pole-floor 0.3 [--start-pool 4]

Pass --rollout-steps > 0 to switch to the COMBINED loss
(online_training.make_combined_loss): rollout(short-lead forecast) + lam*clim.
The pure climatology loss fixes the surface but only by warming the column, so
it gives back the aloft/humidity skill; the rollout term penalises that 500 hPa
drift so the surface is fixed WITHOUT the overshoot. Warm-start from the
UNTAPERED rollout-trained term so the surface correction is trainable and the
rollout term keeps it honest:

    python tools/bias_correction/finetune_climatology.py online_term_t31.npz online_term_t31_comb.npz \
        --n-steps 4320 --rollout-steps 48 --lam 1.0 --updates 30 --lr 2.5e-5 \
        --year 2001 --spinup-days 10 --pole-floor 0.3 --start-pool 3
"""
import argparse
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_datetime as jdt
from importlib import resources

from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData, make_time_series, BY_DATE
from jcm.nudging import NudgingConfig, NudgingTarget, with_nudging
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics.bias_correction import load_bias_correction
from jcm.physics.bias_correction.nn_bias_correction import (
    CONTEXT_FEATURES, FIELD_ORDER, DenseWeights, remap_first_layer)
from jcm.physics.bias_correction.era5_cache import cache_meta
from jcm.physics.bias_correction.era5_data import load_era5
from jcm.physics.bias_correction.multiyear import (
    climatology_starts, parse_year_span)
from jcm.physics.bias_correction.online_training import (
    make_climatology_loss, make_combined_loss, make_train_step,
    stds_from_stats, lat_weights, level_weights)
from jcm.physics.bias_correction.optim import adam_init

# The chdir is for the relative data paths below, not for the imports, so it
# sits after them. The repo root is two directories up from this script;
# JCM_REPO overrides it for an out-of-tree checkout.
REPO = os.environ.get("JCM_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

# Resolution and model step are env-configurable so one script serves T31 (the
# default every reported result used) and a higher-resolution run. T63 needs a
# shorter step for stability; 12 minutes is what the repo's production T63
# config (jcm/config/run/longrun.yaml) settled on. Defaults reproduce T31
# exactly, so leaving these unset changes nothing.
TRUNC = int(os.environ.get("M4_TRUNC", "31"))
DT_MIN = float(os.environ.get("M4_DT_MIN", "30"))
STEPS_PER_DAY = int(round(24 * 60 / DT_MIN))
# Model steps per 6-hourly ERA5 slot: 12 at the 30-minute T31 step, 30 at the
# 12-minute T63 step. Everything that converts a rollout length into a target
# slot index must go through this, or a shorter step silently fetches the
# target from the wrong time (a 120-step T63 rollout is a 1-day lead, but
# dividing by a hardcoded 12 would fetch the +60 h ERA5 snapshot instead).
STEPS_PER_SLOT = STEPS_PER_DAY // 4
SAVE = 1.25                 # nudged-run save cadence (days), matches the cache

ap = argparse.ArgumentParser()
ap.add_argument("in_npz")
ap.add_argument("out_npz")
ap.add_argument("--n-steps", type=int, default=1440)      # 1440 = 30-day free run
ap.add_argument("--updates", type=int, default=30)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--clip-norm", type=float, default=1.0)
ap.add_argument("--year", type=int, default=2001)
ap.add_argument("--years", default=None,
                help="multi-year span, e.g. 1995-2015. Draws start states "
                     "from the cache built by build_era5_cache.py instead of "
                     "running a nudged year inline, and cycles YEARS first so "
                     "a small pool spans many years at one day of year. "
                     "Mutually exclusive with --year.")
ap.add_argument("--cache-root", default=None,
                help="cache dir for --years (default ~/era5_cache_t<TRUNC>)")
ap.add_argument("--spinup-days", type=float, default=10.0)
ap.add_argument("--pole-floor", type=float, default=0.3)
ap.add_argument("--start-pool", type=int, default=1)
ap.add_argument("--rollout-steps", type=int, default=0,   # >0 -> combined loss
                help="short-lead forecast length (model steps, multiple of 12) "
                     "for the combined rollout+clim loss; 0 = pure climatology")
ap.add_argument("--lam", type=float, default=1.0,
                help="weight on the climatology term in the combined loss")
ap.add_argument("--polar-taper", default=None,          # "lat0,lat1" in degrees
                help="fade the correction to zero poleward of lat1 (full "
                     "equatorward of lat0); caps the polar surface overshoot the "
                     "rollout term can't police. Overrides the warm-start term's.")
ap.add_argument("--surface-taper", default=None,        # "sigma0,sigma1"
                help="fade the near-surface T correction to zero (sigma >= "
                     "sigma1). Overrides the warm-start term's.")
ap.add_argument("--context", default=None,              # e.g. "fmask" or "fmask,orog"
                help="per-column context features to give the net, comma "
                     f"separated, from {CONTEXT_FEATURES}. Features the "
                     "warm-start term lacks are added by widening the first "
                     "layer with ZERO rows, so the term starts identical to "
                     "the warm start and learns to use them. Unlike a taper "
                     "this ADDS information rather than removing correction.")
ap.add_argument("--level-weight", default=None,   # e.g. "1,1,2,3,3,2,1,1"
                help="comma-separated per-level loss weights, one per model "
                     "level, normalized to mean 1. Level 0 is the model TOP "
                     "and the last level is the surface. Default (unset) "
                     "weights every level equally, which is what the loss did "
                     "before this flag existed. Weight mid-levels up to keep "
                     "the 500 hPa skill that the surface-focused runs give up.")
ap.add_argument("--clim-days", default=None,
                help="comma-separated days of year to draw climatology "
                     "windows from, e.g. 11.25,101.25,191.25,281.25 for one "
                     "window per season. Default (unset) keeps the shipped "
                     "single-window behaviour, which is mid-January only: the "
                     "pool advances the day of year by one save (1.25 d) per "
                     "completed year-cycle, so it CANNOT reach another season "
                     "however large --start-pool gets. Repeat a day to weight "
                     "it; weighting is by sampling frequency because Adam "
                     "makes a per-update loss multiplier nearly a no-op.")
ap.add_argument("--new-row-lr", type=float, default=None,
                help="separate learning rate for context rows this run ADDS. "
                     "Adam's step is bounded by lr regardless of gradient, so "
                     "at the shipped budget (210 updates, lr 2.5e-5) a zero "
                     "row cannot exceed 5.3e-3 against a Glorot scale of "
                     "8.3e-2; the two shipped context terms reached 0.4%% of "
                     "Glorot and are numerically inert. Unset = old behaviour.")
ap.add_argument("--report-new-rows", action="store_true",
                help="print the RMS of the added rows against the trained "
                     "profile rows, so a working feature is distinguishable "
                     "from a row that never left zero")
ap.add_argument("--max-bad-steps", type=int, default=5)
ap.add_argument("--probe-only", action="store_true")
a = ap.parse_args()

# --year runs a nudged year inline; --years reads pre-built years from the
# cache. Silently preferring one would make the log ambiguous about where the
# start states actually came from. Match both "--year 2001" and "--year=2001".
if a.years and any(arg == "--year" or arg.startswith("--year=")
                   for arg in sys.argv):
    raise SystemExit("pass --year OR --years, not both")

win_days = a.n_steps / STEPS_PER_DAY
print(f"[clim] n_steps={a.n_steps} ({win_days:.1f}-day free run)  updates={a.updates}  "
      f"lr={a.lr}  pole_floor={a.pole_floor}  start_pool={a.start_pool}", flush=True)

# --- coords / terrain / FREE forcing ---------------------------------------
coords = get_speedy_coords(layers=8, spectral_truncation=TRUNC)
nlev = coords.vertical.layers
bc = resources.files("jcm.data.bc.t30.clim")
terrain = TerrainData.from_file(bc / "terrain.nc", coords=coords)
base_forcing = ForcingData.from_file(bc / "forcing.nc", coords=coords)
start = jdt.to_datetime(f"{a.year}-01-01")

# --- warm start: load the term, keep ALL of its config ---------------------
base = load_bias_correction(a.in_npz)
in_mean = base.in_mean.get_value()
in_std = base.in_std.get_value()
out_scale = base.out_scale.get_value()
correct = base.correct
layers = base.weights.get_value()

# Taper CLI args OVERRIDE the warm-start term's own tapers (None = keep the
# term's). Baked into both the training model and the saved term so the network
# adapts to the taper during fine-tuning rather than having it slapped on after.
def _pair(s):
    return tuple(float(x) for x in s.split(",")) if s else None

polar_taper = _pair(a.polar_taper) if a.polar_taper else base.polar_taper
surface_taper = _pair(a.surface_taper) if a.surface_taper else base.surface_taper

# Context features the warm start doesn't have get zero rows, so the net starts
# numerically identical to the warm start and only improves if the new input
# earns it. Context is normalised inside build_context, so in_mean/in_std
# (profile block only) are untouched.
#
# The rows are placed BY NAME. Order is re-derived by filtering
# CONTEXT_FEATURES, so adding a feature that sorts earlier in the registry than
# one already present is a REORDER, not an append: adding `fmask` to a term
# trained with `insol` gives ("fmask", "insol"), and appending a zero row at
# the bottom would leave the trained insol weights feeding fmask. Every shipped
# term went () -> (one feature), so this never fired, but it silently corrupts
# any multi-feature warm start.
context_features = base.context_features
new_row_idx = ()
if a.context:
    want = tuple(s.strip() for s in a.context.split(",") if s.strip())
    unknown = [f for f in want if f not in CONTEXT_FEATURES]
    if unknown:
        raise SystemExit(f"unknown context feature(s) {unknown}; "
                         f"choose from {CONTEXT_FEATURES}")
    context_features = tuple(f for f in CONTEXT_FEATURES
                             if f in want or f in base.context_features)
    if context_features != base.context_features:
        layers = remap_first_layer(layers, len(FIELD_ORDER) * nlev,
                                   base.context_features, context_features)
        added = tuple(f for f in context_features
                      if f not in base.context_features)
        new_row_idx = tuple(context_features.index(f) for f in added)
        print(f"[clim] remapped first layer: {base.context_features} -> "
              f"{context_features}; new zero row(s) for {added}", flush=True)

print(f"[clim] warm start {a.in_npz}: correct={correct} cap={base.output_cap} "
      f"polar={polar_taper} surface={surface_taper} "
      f"context={context_features}", flush=True)


def build_model(layers):
    # rebuild() carries every static field from the warm start, so only the
    # ones this stage actually changes are named. Hand-listing them is how
    # `activation` and both tapers got silently dropped elsewhere.
    term = base.rebuild(layers, polar_taper=polar_taper,
                        surface_taper=surface_taper,
                        context_features=context_features)
    return Model(coords=coords, terrain=terrain,
                 physics=speedy_physics() + term,
                 start_date=start, calendar="365_day", time_step=DT_MIN)


probe = build_model(layers)

# --- ERA5 window means + near-ERA5 start states ----------------------------
win_slots = int(round(win_days * 4))          # 6-hourly ERA5 slots in the window
lead_slots = a.rollout_steps // STEPS_PER_SLOT   # model steps -> ERA5 slots
inits, targT, targq = [], [], []

if a.years:
    # Multi-year: read pre-built start states and targets from the cache. The
    # nudged runs already happened once in build_era5_cache.py, which is what
    # makes a 21-year pool affordable -- running 21 nudged years per fine-tune
    # would dominate the job.
    cache_root = a.cache_root or os.path.expanduser(f"~/era5_cache_t{TRUNC}")
    span = parse_year_span(a.years)
    meta = cache_meta(coords, cadence_hours=6, save_days=SAVE,
                      tau_seconds=21600.0, time_step_minutes=DT_MIN,
                      spinup_days=a.spinup_days)
    clim_days = ([float(x) for x in a.clim_days.split(",") if x.strip()]
                 if a.clim_days else None)
    starts = climatology_starts(
        cache_root, span, meta=meta, n_starts=a.start_pool,
        win_slots=win_slots,
        lead_slots=lead_slots if a.rollout_steps > 0 else None,
        days=clim_days)
    print(f"[clim] {len(starts)} starts from {len(span)} cached years "
          f"({span[0]}..{span[-1]}): "
          f"{[(r['year'], round(r['sim_time'] / 86400.0, 2)) for r in starts]}",
          flush=True)
    for r in starts:
        inits.append(probe.dycore.initial_state(
            r["state"], sim_time=r["sim_time"],
            tracer_specs=probe.dycore.tracer_specs))
        if a.rollout_steps > 0:
            targT.append((r["leadT"], r["climT"]))
            targq.append((r["leadq"], r["climq"]))
        else:
            targT.append(r["climT"])
            targq.append(r["climq"])
else:
    # Single-year path, unchanged: run one nudged year inline. This is what
    # produced every reported term, so it stays byte-identical.
    first = int(round(a.spinup_days / SAVE))
    start_idxs = [first + k for k in range(a.start_pool)]
    last_start_day = (start_idxs[-1] + 1) * SAVE
    nudge_days = last_start_day + SAVE                          # reach the last start frame
    era5_days = int(math.ceil(last_start_day + win_days)) + 3   # cover every target window

    fields, time_seconds = load_era5(coords, a.year, era5_days, cadence_hours=6)
    target_run = NudgingTarget(**{
        k: make_time_series(jnp.asarray(v), jnp.asarray(time_seconds), align_mode=BY_DATE)
        for k, v in fields.items()})
    config = NudgingConfig.temp_humidity(nlev, tau_seconds=21600.0)
    nudged = Model(coords=coords, terrain=terrain,
                   physics=with_nudging(speedy_physics(), config),
                   start_date=start, calendar="365_day", time_step=DT_MIN)
    states = nudged.run(forcing=base_forcing.replace(nudging_target=target_run),
                        save_interval=SAVE, total_time=float(nudge_days)).dynamics
    print(f"[clim] nudged frames: {states.temperature.shape[0]}; starts at days "
          f"{[round((i + 1) * SAVE, 2) for i in start_idxs]}", flush=True)

    for i in start_idxs:
        ps = jtu.tree_map(lambda x: x[i], states)
        inits.append(probe.dycore.initial_state(
            ps, sim_time=(i + 1) * SAVE * 86400.0,
            tracer_specs=probe.dycore.tracer_specs))
        s0 = int(round((i + 1) * SAVE * 4))       # window start in 6-hourly slots
        climT = jnp.asarray(fields["temperature"][s0:s0 + win_slots].mean(0))
        climq = jnp.asarray(fields["specific_humidity"][s0:s0 + win_slots].mean(0))
        if a.rollout_steps > 0:
            # Combined loss: pack the short-lead ERA5 snapshot (rollout target) with
            # the window mean (clim target). make_combined_loss / make_train_step
            # thread the tuple through unchanged.
            rT = jnp.asarray(fields["temperature"][s0 + lead_slots])
            rq = jnp.asarray(fields["specific_humidity"][s0 + lead_slots])
            targT.append((rT, climT))
            targq.append((rq, climq))
        else:
            targT.append(climT)
            targq.append(climq)

# --- loss ingredients ------------------------------------------------------
T_std, q_std = stds_from_stats(in_std, nlev)
weights = lat_weights(coords, pole_weight_floor=a.pole_floor)
level_w = level_weights(
    [x for x in a.level_weight.split(",")] if a.level_weight else None, nlev)
if a.level_weight:
    # Print sigma beside each weight: it is the only way to see at a glance
    # whether the intended level actually got the boost (level 0 is the top).
    sigma = coords.vertical.centers
    print("[clim] level weights (index, sigma, weight):", flush=True)
    for k in range(nlev):
        print(f"         {k}  sigma={float(sigma[k]):.3f}  w={float(level_w[k]):.3f}",
              flush=True)
else:
    level_w = None          # keep the byte-identical unweighted path
if a.rollout_steps > 0:
    # ERA5 targets are 6-hourly, so a rollout must land on one of those slots.
    assert a.rollout_steps % STEPS_PER_SLOT == 0, \
        (f"--rollout-steps must be a multiple of {STEPS_PER_SLOT} "
         f"(6-hourly ERA5 cadence at a {DT_MIN:g}-minute step)")
    loss_fn = make_combined_loss(
        build_model, base_forcing, n_rollout_steps=a.rollout_steps,
        n_clim_steps=a.n_steps, lam=a.lam,
        T_std=T_std, q_std=q_std, weights=weights, level_w=level_w,
        time_step_minutes=DT_MIN)
    print(f"[clim] COMBINED loss: rollout {a.rollout_steps} steps "
          f"({a.rollout_steps / STEPS_PER_DAY:.2f}-day lead) + {a.lam} * "
          f"clim {a.n_steps} steps", flush=True)
else:
    loss_fn = make_climatology_loss(build_model, base_forcing, n_steps=a.n_steps,
                                    T_std=T_std, q_std=q_std, weights=weights,
                                    level_w=level_w, time_step_minutes=DT_MIN)

# --- de-risk probe: one fwd+bwd, finiteness + timing -----------------------
if a.probe_only:
    t0 = time.time()
    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(
        layers, inits[0], targT[0], targq[0])
    gn = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jtu.tree_leaves(grads))))
    finite = bool(jnp.isfinite(loss)) and math.isfinite(gn)
    print(f"[clim probe] loss={float(loss):.5f}  grad_norm={gn:.4e}  finite={finite}  "
          f"({time.time() - t0:.0f}s for one compile+fwd+bwd)", flush=True)
    raise SystemExit(0)

# --- fine-tune loop (reuses the guarded step / Adam / NaN guard) -----------
n_profile = len(FIELD_ORDER) * nlev

if a.new_row_lr is not None and new_row_idx:
    # Per-parameter rate: the converged weights keep --lr while the rows added
    # this run get their own. Without this a new input cannot reach a usable
    # magnitude inside the climatology budget, which is why every shipped
    # context term is numerically inert.
    lr_arg = jtu.tree_map(lambda w: jnp.full_like(w, a.lr), layers)
    kernel0 = lr_arg[0].kernel
    for i in new_row_idx:
        kernel0 = kernel0.at[n_profile + i].set(a.new_row_lr)
    lr_arg = (DenseWeights(kernel0, lr_arg[0].bias),) + tuple(lr_arg[1:])
    print(f"[clim] new-row lr {a.new_row_lr:g} on context slot(s) "
          f"{new_row_idx}; {a.lr:g} everywhere else", flush=True)
else:
    lr_arg = a.lr


def _new_row_report(layers_now, tag):
    """RMS of the added rows against the trained profile rows."""
    k = layers_now[0].kernel
    prof = float(jnp.sqrt(jnp.mean(k[:n_profile] ** 2)))
    rows = jnp.stack([k[n_profile + i] for i in new_row_idx])
    new = float(jnp.sqrt(jnp.mean(rows ** 2)))
    print(f"[clim] {tag}: new-row RMS {new:.3e}  profile RMS {prof:.3e}  "
          f"ratio {new / prof:.5f}", flush=True)


if a.report_new_rows and new_row_idx:
    _new_row_report(layers, "before")

step = make_train_step(loss_fn, lr=lr_arg, clip_norm=a.clip_norm)
m, v = adam_init(layers)
t = 0
bad = 0
for u in range(a.updates):
    k = u % len(inits)
    t += 1
    layers, m, v, loss, gn, ok = step(layers, m, v, t, inits[k], targT[k], targq[k])
    okb = bool(ok)
    print(f"[clim] update {u:3d}  loss {float(loss):.5f}  grad_norm {float(gn):.3e}  "
          f"ok={okb}", flush=True)
    bad = 0 if okb else bad + 1
    if bad >= a.max_bad_steps:
        print(f"[clim] aborting: {bad} consecutive non-finite steps", flush=True)
        break

# --- save with the SAME config ---------------------------------------------
term = base.rebuild(layers, polar_taper=polar_taper,
                    surface_taper=surface_taper,
                    context_features=context_features)
if a.report_new_rows and new_row_idx:
    _new_row_report(layers, "after ")

term.save(a.out_npz)
print(f"[clim] wrote {a.out_npz}", flush=True)
