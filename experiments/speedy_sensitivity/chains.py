"""Ensemble jvp chains (Lea, Allen & Haine 2000) for GFMIP SST patches.

Iterates the ensemble-averaged tangent of a ``--window``-day propagator F
without ever forming a Jacobian:

    v_{k+1} = mean_s  d/de F(x_s + e v_k, sst0 + e dsst_p)          (jvp)

where x_s are ``--nseed`` states drawn at random from the state library each
iteration (frozen BCs; export JCM_PERPETUAL_EQUINOX=1 when spinning the
library AND running chains). Each iteration also records the ensemble-mean
tangent of the window-mean ocean-mean observables (per K of patch-mean SST):
iteration 0 is the plain transient sensitivity, and a late-iteration plateau
is the chain's linear-response equilibrium. A patch is abandoned if |v| grows
past ``--blowup`` x |v_1|.

  JCM_PERPETUAL_EQUINOX=1 python spinup.py --truncation 31 --nlev 16 --n 256 --out runs
  JCM_PERPETUAL_EQUINOX=1 python chains.py --truncation 31 --nlev 16 --window 1 --niter 60 --nseed 32 --out runs

Writes <out>/chains-<tag>-w<window>d.npz (obs_traj (patch, iter, obs), vnorms,
final tangent per patch); saved every 10 patches, so it can be inspected
while running. Stop cleanly: touch <out>/STOP_CHAINS.
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import common

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
common.add_config_args(ap)
ap.add_argument("--window", type=float, required=True, help="days per chain iteration")
ap.add_argument("--niter", type=int, required=True)
ap.add_argument("--nseed", type=int, default=32, help="states averaged per iteration")
ap.add_argument("--patches", default="all", help='"all", "a-b" or "i,j,k"')
ap.add_argument("--blowup", type=float, default=1e4)
ap.add_argument("--lib", default=None, help="state library (default: <out>/statelib-<tag>-frozen)")
ap.add_argument("--save_tangents", action="store_true", help="also save each patch's final state tangent")
ap.add_argument("--out", default="runs")
args = ap.parse_args()

tag = common.config_tag(args)
name = f"chains-{tag}-w{args.window:g}d"
model, coords, terrain = common.build_model_from_args(args)
forcing = common.load_forcing(coords, frozen_bcs=True)
out = pathlib.Path(args.out)
P = np.load(out / f"gfmip-patches-T{args.truncation}.npz")
dsst_unit, mdu = P["dsst_unit"], P["mean_dsst_unit"]
W = jnp.asarray(P["ocean_weight"])
pool = common.load_library(pathlib.Path(args.lib) if args.lib else common.library_dir(args.out, tag, frozen=True))
npool = jax.tree_util.tree_leaves(pool)[0].shape[0]
if args.patches == "all":
    pids = list(range(dsst_unit.shape[0]))
elif "-" in args.patches:
    a, b = args.patches.split("-")
    pids = list(range(int(a), int(b) + 1))
else:
    pids = [int(x) for x in args.patches.split(",")]
common.log(name, f"pool {npool} states, {args.nseed}/iter, {len(pids)} patches, "
           f"{args.niter} x {args.window:g} d, {common.env_summary()}")


def F(state, delta):
    rs, preds = model.run_from_state(initial_state=state, forcing=common.with_sst_delta(forcing, delta),
                                     save_interval=args.window, total_time=args.window, output_averages=True)
    return rs.dynamics, jnp.sum(W * common.obs_fields(preds), axis=(-2, -1))


def tangent_zeros(state):
    """Zero tangent matching ``state``: float leaves get zeros, integer leaves float0."""
    return jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)
        else np.zeros(np.shape(x), jax.dtypes.float0), state)


@jax.jit
def step(batch, v, dsst):
    def one(state):
        _, (x_dot, obs_dot) = jax.jvp(F, (state, jnp.zeros_like(dsst)), (v, dsst))
        return x_dot, obs_dot
    x_dots, obs_dots = jax.vmap(one)(batch)
    v_next = jax.tree_util.tree_map(
        lambda t, v0: t.mean(0) if jnp.issubdtype(t.dtype, jnp.inexact) else v0, x_dots, v)
    return v_next, obs_dots.mean(0)


def norm(v):
    return float(np.sqrt(sum(float(jnp.sum(jnp.asarray(x, jnp.float32) ** 2))
                             for x in jax.tree_util.tree_leaves(v) if jnp.issubdtype(x.dtype, jnp.inexact))))


rng = np.random.default_rng(0)
v0 = tangent_zeros(common.take(pool, 0))
obs_traj = np.full((dsst_unit.shape[0], args.niter, len(common.OBS_NAMES)), np.nan, np.float32)
vnorms = np.full((dsst_unit.shape[0], args.niter), np.nan, np.float32)
finals = {}
for n_done, p in enumerate(pids):
    if (out / "STOP_CHAINS").exists():
        common.log(name, "STOP_CHAINS found")
        break
    v, dsst = v0, jnp.asarray(dsst_unit[p])
    for k in range(args.niter):
        batch = common.take(pool, rng.choice(npool, args.nseed, replace=False))
        v, obs = step(batch, v, dsst)
        obs_traj[p, k] = np.asarray(obs) / mdu[p]
        vnorms[p, k] = norm(v)
        if not np.isfinite(vnorms[p, k]) or vnorms[p, k] > args.blowup * vnorms[p, 0]:
            common.log(name, f"patch {p} diverged at iteration {k}")
            break
    if args.save_tangents:
        finals[f"v_p{p}"] = np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(v)
                                            if jnp.issubdtype(x.dtype, jnp.inexact)])
    if n_done % 10 == 0 or p == pids[-1]:
        np.savez(out / f"{name}.npz", obs_traj=obs_traj, vnorms=vnorms, obs_names=np.array(common.OBS_NAMES),
                 window=args.window, nseed=args.nseed, env=common.env_summary(), **finals)
        common.log(name, f"patch {p}: olr sens iter0 {obs_traj[p, 0, 0]:+.3f}, last {obs_traj[p, k, 0]:+.3f}")
np.savez(out / f"{name}.npz", obs_traj=obs_traj, vnorms=vnorms, obs_names=np.array(common.OBS_NAMES),
         window=args.window, nseed=args.nseed, env=common.env_summary(), **finals)
common.log(name, f"saved {out}/{name}.npz")
