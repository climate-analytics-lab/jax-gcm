"""Build the per-year ERA5 + nudged-state cache the multi-year training reads.

Streams one calendar year of ERA5, runs a nudged model across it for near-ERA5
start states, and writes both to local disk. Resumable: a year already cached
with matching settings is skipped, so an interrupted multi-hour build is
restarted by re-running the same command. Style mirrors finetune_climatology.py.

    # smoke test first -- two years spanning a leap boundary, ~10 min
    python tools/bias_correction/build_era5_cache.py --years 2003-2004

    # the training span, in the background
    nohup env CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python tools/bias_correction/build_era5_cache.py --years 1995-2015 \
        > build_cache_train.log 2>&1 &

    # the held-out span
    python tools/bias_correction/build_era5_cache.py --years 2016-2023

    # what is already on disk, without building anything
    python tools/bias_correction/build_era5_cache.py --years 1995-2023 --list

The cache directory defaults to ~/era5_cache_t<TRUNC>, deliberately OUTSIDE the
repo: it reaches tens of GB and must not land in the working tree. Put it on
home, not /data, which is ~98% full.
"""
# The jcm imports deliberately sit below the environment setup rather than at
# the top of the file, because the XLA settings below have to be decided before
# jax is imported and jax arrives with the first jcm import. Ruff cannot see
# that ordering constraint, so E402 is waived for this file only.
# ruff: noqa: E402
import argparse
import os
import sys
import time

# Repo root: two directories up from tools/bias_correction/. JCM_REPO overrides
# it when the script runs from a copy outside the checkout.
REPO = os.environ.get("JCM_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

# Both of these must be decided BEFORE jax is imported, which happens via the
# jcm imports below and claims a device immediately.
#
# The GPUs here are shared. Left to itself XLA preallocates ~75% of the card at
# startup, so a second job on the same GPU dies in a wall of shrinking
# CUDA_ERROR_OUT_OF_MEMORY retries. Never let that be the default.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# --list only reads manifests off disk. Asking for a GPU to do that emits the
# same OOM noise whenever another user happens to be filling card 0.
if "--list" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from importlib import resources

from jcm.forcing import ForcingData
from jcm.physics.bias_correction.era5_cache import (
    build_year, cache_meta, is_cached)
from jcm.physics.bias_correction.multiyear import parse_year_span
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

# Same env knobs as finetune_climatology.py / evaluate_term.py so one setting drives
# the whole chain; defaults reproduce the T31 configuration every reported
# number used.
TRUNC = int(os.environ.get("M4_TRUNC", "31"))
DT_MIN = float(os.environ.get("M4_DT_MIN", "30"))


ap = argparse.ArgumentParser()
ap.add_argument("--years", required=True,
                help="e.g. 1995-2015 or 2001,2002")
ap.add_argument("--root", default=None,
                help="cache dir (default ~/era5_cache_t<TRUNC>)")
ap.add_argument("--cadence-hours", type=int, default=6)
ap.add_argument("--save-days", type=float, default=1.25,
                help="spacing of saved nudged start states")
ap.add_argument("--tau-hours", type=float, default=6.0,
                help="nudging relaxation time; 6 h is what every shipped term used")
ap.add_argument("--spinup-days", type=float, default=10.0,
                help="recorded only; the training side discards these frames")
ap.add_argument("--list", action="store_true",
                help="report what is cached and exit")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

years = parse_year_span(a.years)
root = a.root or os.path.expanduser(f"~/era5_cache_t{TRUNC}")

coords = get_speedy_coords(layers=8, spectral_truncation=TRUNC)
meta = cache_meta(coords, cadence_hours=a.cadence_hours,
                  save_days=a.save_days, tau_seconds=a.tau_hours * 3600.0,
                  time_step_minutes=DT_MIN, spinup_days=a.spinup_days)

print("T%d grid %s  dt %g min" % (TRUNC, coords.horizontal.nodal_shape, DT_MIN),
      flush=True)
print("cache root: %s" % root, flush=True)
print("years: %d (%d..%d)" % (len(years), years[0], years[-1]), flush=True)

done = [y for y in years if is_cached(root, y, meta)]
todo = [y for y in years if y not in set(done)]
print("cached: %d   to build: %d" % (len(done), len(todo)), flush=True)

if a.list:
    cached = set(done)
    try:
        for y in years:
            print("  %d  %s" % (y, "cached" if y in cached else "MISSING"),
                  flush=True)
    except BrokenPipeError:
        # Piping into `head` closes stdout early. That is a normal way to use
        # a listing, not an error worth a traceback.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    raise SystemExit(0)

bc = resources.files("jcm.data.bc.t30.clim")
terrain = TerrainData.from_file(bc / "terrain.nc", coords=coords)
base_forcing = ForcingData.from_file(bc / "forcing.nc", coords=coords)

t0 = time.time()
for i, year in enumerate(years, 1):
    t_year = time.time()
    build_year(year, root=root, coords=coords, terrain=terrain,
               base_forcing=base_forcing, cadence_hours=a.cadence_hours,
               save_days=a.save_days, tau_seconds=a.tau_hours * 3600.0,
               time_step_minutes=DT_MIN, spinup_days=a.spinup_days,
               overwrite=a.overwrite)
    print("  [%d/%d] %d done in %.0fs (total %.0fs)"
          % (i, len(years), year, time.time() - t_year, time.time() - t0),
          flush=True)

print("\ncache build complete: %d years in %.0fs" % (len(years), time.time() - t0),
      flush=True)
print("disk: run  du -sh %s" % root, flush=True)
