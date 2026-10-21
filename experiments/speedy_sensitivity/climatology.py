"""Climatology of one configuration, for checking that a damping / dycore choice stays physical.

Spins ``--nseed`` states, then averages ``--avg_days`` and saves global and
zonal means of the observables plus zonal-mean zonal wind and temperature
(same frozen-BC / equinox background as the tangent experiments unless
``--seasonal``). Compare configurations with plot_climatology.py.

  JCM_PERPETUAL_EQUINOX=1 python climatology.py --truncation 31 --nlev 8 --out runs
  JCM_PERPETUAL_EQUINOX=1 python climatology.py --truncation 31 --nlev 8 --diffusion_scale 4 --out runs
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import common

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
common.add_config_args(ap)
ap.add_argument("--nseed", type=int, default=4)
ap.add_argument("--spin_days", type=float, default=200.0)
ap.add_argument("--avg_days", type=float, default=120.0)
ap.add_argument("--seasonal", action="store_true")
ap.add_argument("--out", default="runs")
args = ap.parse_args()

tag = common.config_tag(args)
model, coords, terrain = common.build_model_from_args(args)
forcing = common.load_forcing(coords, frozen_bcs=not args.seasonal)
lat = np.rad2deg(np.asarray(coords.horizontal.latitudes))
sigma = np.asarray(coords.vertical.centers)
common.log(f"clim-{tag}", f"{args.nseed} seeds, spin {args.spin_days:g} + avg {args.avg_days:g} d, {common.env_summary()}")


def one(state):
    _, preds = model.run_from_state(initial_state=state, forcing=forcing, save_interval=args.avg_days,
                                    total_time=args.avg_days, output_averages=True)
    dyn = preds.dynamics
    return common.obs_fields(preds), dyn.u_wind[0].mean(-2), dyn.temperature[0].mean(-2)   # zonal means (lev, lat)


states = jax.tree_util.tree_map(jnp.asarray, common.spin_library(model, forcing, args.nseed, args.spin_days, args.nseed))
obs, u, t = (np.asarray(x) for x in jax.vmap(one)(states))
finite = np.isfinite(obs).all(axis=(1, 2, 3))
w = np.cos(np.deg2rad(lat))
w = w / w.sum()
glob = (obs[finite].mean(-2) * w).sum(-1)          # (seed, obs): area-weighted global means
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"clim-{tag}.npz", obs_names=np.array(common.OBS_NAMES), global_mean=glob,
         zonal=obs[finite].mean(-2), u_zonal=u[finite], t_zonal=t[finite], lat=lat, sigma=sigma,
         finite=finite, env=common.env_summary())
common.log(f"clim-{tag}", f"finite {finite.sum()}/{args.nseed}; global means: " + ", ".join(
    f"{n} {glob[:, i].mean():.3g}" for i, n in enumerate(common.OBS_NAMES)))
