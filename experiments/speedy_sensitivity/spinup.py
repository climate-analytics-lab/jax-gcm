"""Spin up a library of SPEEDY states for one configuration.

  python spinup.py --truncation 31 --nlev 8 --advection eulerian --n 64 --days 200 --out runs
  python spinup.py ... --seasonal      # seasonal BCs (for the GFMIP FD atlas)

Frozen (annual-mean) BCs are the default: growth.py and chains.py need a
stationary background, so also export JCM_PERPETUAL_EQUINOX=1 for those
libraries. Writes <out>/statelib-<tag>-{frozen,seasonal}/chunk_*.pkl.
"""
import argparse

import common

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
common.add_config_args(ap)
ap.add_argument("--n", type=int, default=64, help="number of states")
ap.add_argument("--days", type=float, default=200.0, help="spin-up length from random initial states")
ap.add_argument("--batch", type=int, default=16, help="states per vmapped batch")
ap.add_argument("--chunk", type=int, default=32, help="states per saved chunk file")
ap.add_argument("--seasonal", action="store_true", help="seasonal BCs instead of frozen annual means")
ap.add_argument("--out", default="runs")
args = ap.parse_args()

tag = common.config_tag(args)
if args.seasonal and args.days % 365:
    # Every later run_from_state call starts the clock (and the seasonal
    # forcing) at start_time, Jan 1: a state spun a whole number of years
    # ends on Jan 1 too, so it continues without a jump in the forcing.
    common.log("spinup", f"WARNING: --days {args.days:g} is not a whole number of years; runs started "
               "from these states will see a jump in the seasonal forcing")
model, coords, terrain = common.build_model_from_args(args)
forcing = common.load_forcing(coords, frozen_bcs=not args.seasonal)
common.log("spinup", f"{tag} n={args.n} days={args.days:g} seasonal={args.seasonal} {common.env_summary()}")
states = common.spin_library(model, forcing, args.n, args.days, args.batch)
d = common.library_dir(args.out, tag, frozen=not args.seasonal)
common.save_library(states, d, args.chunk)
common.log("spinup", f"saved {args.n} states to {d}")
