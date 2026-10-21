"""GFMIP-protocol finite-difference Green's functions for SPEEDY.

For each patch p, run paired +A / -A SST perturbations from the same library
states (seasonal BCs), discard ``--spin_days``, average ``--avg_days``, and
save per-member symmetric sensitivities (warm - cool) / (2A) of the
time-mean OLR, absorbed SW, convective and large-scale precipitation maps,
per K of patch PEAK. Patches are processed in groups of ``--group``, all
(group x 2 signs x members) runs vmapped in one call.

  python spinup.py --truncation 31 --nlev 16 --seasonal --n 16 --out runs
  python make_patches.py --truncation 31 --out runs
  python gfmip_fd.py --truncation 31 --nlev 16 --members 16 --out runs [--patches 0-20] [--lane 0 --nlanes 2]

Outputs <out>/gfmip-fd-<tag>/patch_###.npz (existing files are skipped, so
reruns resume; use --lane/--nlanes to split patches across GPUs).
Stop cleanly: touch <out>/gfmip-fd-<tag>/STOP.
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import common

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
common.add_config_args(ap)
ap.add_argument("--members", type=int, default=16)
ap.add_argument("--amplitude", type=float, default=4.0, help="patch peak |A| in K")
ap.add_argument("--spin_days", type=float, default=60.0)
ap.add_argument("--avg_days", type=float, default=360.0)
ap.add_argument("--save_days", type=float, default=30.0,
                help="output averaging interval; spin_days and avg_days must be multiples of it")
ap.add_argument("--group", type=int, default=2, help="patches per vmapped call")
ap.add_argument("--patches", default="all", help='"all", "a-b" or "i,j,k"')
ap.add_argument("--lane", type=int, default=0)
ap.add_argument("--nlanes", type=int, default=1)
ap.add_argument("--lib", default=None, help="state library (default: <out>/statelib-<tag>-seasonal)")
ap.add_argument("--out", default="runs")
args = ap.parse_args()

tag = common.config_tag(args)
for _name in ("spin_days", "avg_days"):
    if abs(getattr(args, _name) / args.save_days - round(getattr(args, _name) / args.save_days)) > 1e-9:
        raise SystemExit(f"--{_name} must be a multiple of --save_days")
n_avg = round(args.avg_days / args.save_days)
model, coords, terrain = common.build_model_from_args(args)
forcing = common.load_forcing(coords, frozen_bcs=False)
P = np.load(pathlib.Path(args.out) / f"gfmip-patches-T{args.truncation}.npz")
dsst_unit = P["dsst_unit"]
NP = dsst_unit.shape[0]
lib = pathlib.Path(args.lib) if args.lib else common.library_dir(args.out, tag, frozen=False)
states = common.load_library(lib, args.members)
outdir = pathlib.Path(args.out) / f"gfmip-fd-{tag}"
outdir.mkdir(parents=True, exist_ok=True)
FIELDS = ("olr", "swtoa", "precnv", "precls")

if args.patches == "all":
    pids = list(range(NP))
elif "-" in args.patches:
    a, b = args.patches.split("-")
    pids = list(range(int(a), int(b) + 1))
else:
    pids = [int(x) for x in args.patches.split(",")]
pids = [p for i, p in enumerate(pids) if i % args.nlanes == args.lane]
common.log(f"gfmip-{tag}", f"{len(pids)} patches, {args.members} members, A={args.amplitude} K, "
           f"spin {args.spin_days:g} + avg {args.avg_days:g} d, {common.env_summary()}")


def one(state, delta):
    # One continuous run: every run_from_state call restarts the model clock
    # (and so the seasonal forcing) at start_time, so spin-up and averaging
    # must not be split across calls. Output is averaged over --save_days
    # frames; the last avg_days / save_days frames form the time mean.
    _, preds = model.run_from_state(initial_state=state, forcing=common.with_sst_delta(forcing, delta),
                                    save_interval=args.save_days, total_time=args.spin_days + args.avg_days,
                                    output_averages=True)
    ph = preds.physics
    fields = jnp.stack([ph["_longwave_rad"].ftop, ph["_shortwave_rad"].ftop,
                        86.4 * ph["_convection"].precnv, 86.4 * ph["_condensation"].precls])  # (4, frame, lon, lat)
    return fields[:, -n_avg:].mean(axis=1)


run = jax.jit(jax.vmap(one))
for g in range(0, len(pids), args.group):
    if (outdir / "STOP").exists():
        common.log(f"gfmip-{tag}", "STOP file found")
        break
    grp = [p for p in pids[g:g + args.group] if not (outdir / f"patch_{p:03d}.npz").exists()]
    if not grp:
        continue
    deltas = np.stack([s * args.amplitude * dsst_unit[p] for p in grp for s in (1.0, -1.0)
                       for _ in range(args.members)]).astype(np.float32)
    reps = deltas.shape[0] // args.members
    sb = jax.tree_util.tree_map(lambda x, reps=reps: jnp.concatenate([x] * reps), states)
    y = np.asarray(run(sb, jnp.asarray(deltas))).reshape(len(grp), 2, args.members, len(FIELDS), *dsst_unit.shape[1:])
    for i, p in enumerate(grp):
        np.savez(outdir / f"patch_{p:03d}.npz", sens=((y[i, 0] - y[i, 1]) / (2 * args.amplitude)).astype(np.float32),
                 fields=np.array(FIELDS), warm_mean=y[i, 0].mean(0), cool_mean=y[i, 1].mean(0),
                 amplitude=args.amplitude, members=args.members, center=P["centers"][p])
    common.log(f"gfmip-{tag}", f"saved patches {grp}")
common.log(f"gfmip-{tag}", "done")
