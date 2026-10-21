"""Tangent-growth experiment (the "AD tangents blow up" figure).

For one tropical (0N 150E) and one extratropical (50N 200E) Gaussian SST bump,
jvp a ``--days``-long forward run with daily-mean diagnostics from ``--nseed``
spun-up states, and record the area-weighted SPATIAL VARIANCE of each
observable's tangent per day, averaged over seeds. In the exponential regime
the log-slope of that variance is 2x the tangent growth rate (plot_growth.py
fits it). No chaining, no ensemble averaging of tangents: each seed separately.

  JCM_PERPETUAL_EQUINOX=1 python growth.py --truncation 31 --nlev 8 --advection eulerian
  JCM_PERPETUAL_EQUINOX=1 python growth.py --truncation 31 --nlev 16 --advection semi_lagrangian
  JCM_TANGENT_DECAY=0.1 ... python growth.py ...     # with tangent decay (1/day)

Writes <out>/growth-<tag>[-<suffix>].npz.
"""
import argparse
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import common

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
common.add_config_args(ap)
ap.add_argument("--nseed", type=int, default=4)
ap.add_argument("--days", type=float, default=60.0)
ap.add_argument("--spin_days", type=float, default=200.0, help="spin-up per seed if no --lib")
ap.add_argument("--lib", default=None, help="use the first nseed states of this state library instead of spinning")
ap.add_argument("--suffix", default=None, help="extra output tag (default: g<gamma> if JCM_TANGENT_DECAY is set)")
ap.add_argument("--out", default="runs")
args = ap.parse_args()

tag = common.config_tag(args)
gamma = float(os.environ.get("JCM_TANGENT_DECAY", "0"))
suffix = args.suffix or (f"g{gamma:g}" if gamma != 0 else None)
name = f"growth-{tag}" + (f"-{suffix}" if suffix else "")
model, coords, terrain = common.build_model_from_args(args)
forcing = common.load_forcing(coords, frozen_bcs=True)
common.log(name, f"{common.env_summary()} days={args.days:g} nseed={args.nseed}")

lat = np.rad2deg(np.asarray(coords.horizontal.latitudes))
lon = np.rad2deg(np.asarray(coords.horizontal.longitudes))
w = np.cos(np.deg2rad(lat))
W = jnp.asarray(np.broadcast_to(w[None, :] / w.sum() / len(lon), (len(lon), len(lat))).astype(np.float32))
LG, AG = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat), indexing="ij")


def bump(lat_c, lon_c, sig=18.0):
    s, la, lc = np.deg2rad(sig), np.deg2rad(lat_c), np.deg2rad(lon_c)
    dlon = (LG - lc + np.pi) % (2 * np.pi) - np.pi
    return jnp.asarray(np.exp(-((AG - la) ** 2 + (dlon * np.cos(la)) ** 2) / (2 * s ** 2)).astype(np.float32))


BUMPS = {"tropical": bump(0.0, 150.0), "extratropical": bump(50.0, 200.0)}


def daily_fields(state, delta):
    _, preds = model.run_from_state(initial_state=state, forcing=common.with_sst_delta(forcing, delta),
                                    save_interval=1.0, total_time=args.days, output_averages=True)
    ph = preds.physics
    return jnp.stack([ph["_longwave_rad"].ftop, ph["_shortwave_rad"].ftop,
                      86.4 * ph["_convection"].precnv, 86.4 * ph["_condensation"].precls,
                      ph["_shortwave_rad"].cloudc, ph["_shortwave_rad"].cloudstr,
                      ph["_shortwave_rad"].gse])          # (nobs, day, lon, lat)


@jax.jit
def tangent_variance(state, bmp):
    _, tan = jax.jvp(lambda a: daily_fields(state, a * bmp), (0.0,), (1.0,))
    mean = jnp.sum(W * tan, axis=(-2, -1), keepdims=True)
    return jnp.sum(W * (tan - mean) ** 2, axis=(-2, -1))  # (nobs, day)


if args.lib:
    states = common.load_library(pathlib.Path(args.lib), args.nseed)
else:
    states = jax.tree_util.tree_map(jnp.asarray, common.spin_library(model, forcing, args.nseed, args.spin_days, args.nseed))
out = {}
for pname, bmp in BUMPS.items():
    per_seed = np.asarray(jax.vmap(lambda s, bmp=bmp: tangent_variance(s, bmp))(states))   # (seed, nobs, day)
    out[pname] = np.nanmean(np.where(np.isfinite(per_seed), per_seed, np.nan), axis=0)
    common.log(name, f"{pname}: finite seeds {int(np.isfinite(per_seed).all(axis=(1, 2)).sum())}/{args.nseed}")
pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
np.savez(pathlib.Path(args.out) / f"{name}.npz", obs_names=np.array(common.OBS_NAMES),
         tropical=out["tropical"], extratropical=out["extratropical"], config=tag,
         days=args.days, nseed=args.nseed, env=common.env_summary())
common.log(name, f"saved {args.out}/{name}.npz")
