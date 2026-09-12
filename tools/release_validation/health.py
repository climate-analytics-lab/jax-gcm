"""Climatological health check for a finished jcm run directory.

Usage:
    python check_run_health.py <run_dir with *_dayNNN.nc chunks> [--log FILE]

Computes annual, area-weighted global means from the saved chunks and
checks loose climatological ranges (spin-up tolerant — this is a
"did the model produce a climate" gate, not a tuning target):

    TOA net  = radiation.toa_sw_down - toa_sw_up - toa_lw_up   |net| <= 10 W/m2
    precip   = clouds.precip_rain + precip_snow + convection.precip_conv
               (kg/m2/s -> mm/day)                              2 - 4 mm/day
    cloud    = column max of clouds.cloud_fraction              0.4 - 0.8
    near-sfc T = temperature at the lowest level                278 - 295 K

Also scans every saved variable for NaN/Inf and, with --log, reports the
settled sim-days/hr (last chunk wall) for runtime-regression tracking.
Exit code 0 = all checks pass.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Source-checkout bootstrap: put the repo root (for ``jcm``) AND tools/ (for
# the sibling ``jam_burden_report`` module) on sys.path *before* the imports
# that need them, so the documented ``python tools/release_validation/health.py``
# works without a pip-installed jcm.
_TOOLS = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shared weighting/column-integration machinery lives in jcm.analysis (#640);
# the species table and the mode-summing burden() are tool domain and stay in
# tools/jam_burden_report.py (it includes cloud-borne tracers and the
# pressure_half level-orientation handling).
from jcm.analysis import area_weights, global_mean  # noqa: E402
from aerosol_stats import (  # noqa: E402
    _AOD_KEYS, BURDEN_RANGES, anchor_gates, collect, format_table, is_jam_run,
    missing_jam_diagnostics, physics_gates, run_files, summarize,
    timestep_seconds, unscored_gates)

RANGES = {
    "toa_net_wm2": (-10.0, 10.0),
    "precip_mm_day": (2.0, 4.0),
    "cloud_cover": (0.4, 0.8),
    "near_surface_T": (278.0, 295.0),
    # Global-mean 550 nm AOD: JAM's jam_optics.aod_550 or MACv2-SP's
    # macsp.od550aer. Wide gate — a from-zero JAM spin-up year sits low,
    # so use --last-n to score the settled months.
    "aod_550": (0.02, 0.35),
}

def wmean(da, weights):
    """Time-mean, area-weighted global mean (over the horizontal dims)."""
    if "time" in da.dims:
        da = da.mean("time")
    return float(global_mean(da, weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--log")
    ap.add_argument("--last-n", type=int, default=None,
                    help="use only the last N chunks (default: all)")
    a = ap.parse_args()

    # Same discovery as aerosol_stats.run_files, so the two cannot disagree
    # about which files are chunks (``run.snapshot_interval`` writes a
    # ``_day<N>_snapshots.nc`` stream that a ``*_day*.nc`` glob also matches).
    files = run_files(a.run_dir)
    if not files:
        print(f"FAIL  no chunk files in {a.run_dir}")
        return 1
    if a.last_n:
        files = files[-a.last_n:]
    ds = xr.open_mfdataset(files, combine="by_coords")
    weights = area_weights(ds)

    ok = True

    def check(name, value, lo, hi):
        nonlocal ok
        good = lo <= value <= hi
        print(f"{'PASS' if good else 'FAIL'}  {name} = {value:.2f} "
              f"(expected [{lo:g}, {hi:g}])")
        ok = ok and good

    # NaN scan over everything saved, across the WHOLE opened window —
    # a run that NaN'd mid-year and was restarted can end on a finite
    # chunk, so the last time step alone is not evidence of health.
    bad = []
    for v in ds.data_vars:
        if not bool(np.isfinite(ds[v].values).all()):
            bad.append(v)
    print(f"{'PASS' if not bad else 'FAIL'}  NaN scan: "
          f"{len(bad)}/{len(ds.data_vars)} variables non-finite "
          f"{bad[:5] if bad else ''}")
    ok = ok and not bad

    speedy = "longwave_rad.ftop" in ds       # SPEEDY field dialect
    if speedy:
        # shortwave_rad.ftop is the net downward SW at TOA and
        # longwave_rad.ftop the OUTGOING LW (see speedy_longwave.py), so
        # net TOA = SW_net_down − OLR.
        toa = ds["shortwave_rad.ftop"] - ds["longwave_rad.ftop"]
    else:
        toa = (ds["radiation.toa_sw_down"] - ds["radiation.toa_sw_up"]
               - ds["radiation.toa_lw_up"])
    check("toa_net_wm2", wmean(toa, weights), *RANGES["toa_net_wm2"])

    if speedy:
        # SPEEDY precls/precnv are g/m²/s → ×86.4 for mm/day.
        precip = (ds.get("condensation.precls", 0)
                  + ds.get("convection.precnv", 0)) * 86.4
    else:
        precip = (ds.get("clouds.precip_rain", 0)
                  + ds.get("clouds.precip_snow", 0)
                  + ds.get("convection.precip_conv", 0)) * 86400.0
    check("precip_mm_day", wmean(precip, weights), *RANGES["precip_mm_day"])

    if speedy:
        cf = ds["shortwave_rad.cloudc"]
    else:
        cf = ds["clouds.cloud_fraction"].max("level")
    check("cloud_cover", wmean(cf, weights), *RANGES["cloud_cover"])

    # Lowest model level (level index orientation: take the max-pressure end;
    # jcm output has level index 0 = lowest layer).
    t_low = ds["temperature"].isel(level=0)
    check("near_surface_T", wmean(t_low, weights), *RANGES["near_surface_T"])

    # 550 nm AOD — JAM publishes jam_optics.aod_550 (jam_band_optics.aod_550
    # before #640), MACv2-SP runs publish macsp.od550aer; whichever is present
    # is the scheme's AOD. The JAM keys are shared with aerosol_stats so a run
    # the aerosol block can score is never skipped here.
    _aod_names = "/".join(_AOD_KEYS + ("macsp.od550aer",))
    aod = None
    for key in _AOD_KEYS + ("macsp.od550aer",):
        if key in ds:
            aod = ds[key]
            break
    if aod is not None:
        check("aod_550", wmean(aod, weights), *RANGES["aod_550"])
    else:
        print(f"NOTE  no AOD field found ({_aod_names}); skipping")

    # JAM aerosol block (#762). The statistics are built chunk by chunk
    # (a year of JAM output does not fit in memory), so this takes the file
    # list rather than the opened Dataset. Two tiers, per
    # ``docs/source/design/jam_regression.md``: the climatological anchor
    # ranges, and the absolute physics gates on burden drift and mass-budget
    # closure — the latter are what catch a slow runaway that stays inside a
    # x3-slack range gate until its final fortnight.
    if is_jam_run(ds):
        for namespace in missing_jam_diagnostics(ds):
            print(f"NOTE  no {namespace}.* diagnostics saved; the aerosol "
                  "statistics that read them are absent from the report")
        days, series = collect(files)
        # The dynamics-conservation gate is per STEP, so it needs the run's
        # timestep; ``unscored_gates`` reports it when either is missing.
        dt = timestep_seconds(a.run_dir)
        stats = summarize(days, series, dt)
        for sp in BURDEN_RANGES:
            if f"burden_{sp}_mg_m2" not in stats:
                print(f"NOTE  no {sp} mass tracers; skipping burden")
        # Anchors and physics gates share one implementation with the
        # standalone scorer, so the two commands cannot disagree about a run.
        for name, value, limit, good in (anchor_gates(stats)
                                         + physics_gates(stats)):
            print(f"{'PASS' if good else 'FAIL'}  {name} = {value:.4g} "
                  f"(expected {limit})")
            ok = ok and good
        unscored = unscored_gates(days, series, dt)
        for name, reason in unscored:
            print(f"UNSCORED  {name}: {reason}")
        if unscored:
            print(f"NOTE  {len(unscored)} aerosol gate(s) could not be "
                  "evaluated; they have passed nothing")
        print()
        print(format_table(stats, []))
    else:
        print("NOTE  no m_<species>_<mode> aerosol tracers in the output — "
              "not a JAM run; skipping the aerosol statistics")

    if a.log:
        walls = re.findall(r"Wall: ([0-9.]+)s this chunk", open(a.log).read())
        # Chunk length from the day-numbered filenames themselves, not a
        # constant that can drift from the run's actual chunk_days.
        day_nums = sorted(int(re.search(r"day(\d+)", f).group(1))
                          for f in files)
        spacings = {b - a_ for a_, b in zip(day_nums, day_nums[1:])}
        if len(walls) >= 2 and len(spacings) == 1:
            w = float(walls[-1])
            chunk_days = spacings.pop()
            print(f"INFO  settled rate ~ {chunk_days * 3600 / w:.0f} "
                  f"sim-days/hr (last chunk {w:.0f}s, {chunk_days}-day "
                  "chunks; compare vs the recorded baselines)")

    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
