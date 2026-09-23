"""Climatological health check for a finished jcm run directory.

Usage:
    python tools/release_validation/health.py <run_dir with *_dayNNN.nc chunks>
        [--last-n N] [--log FILE]

Computes annual, area-weighted global means from the saved chunks and
checks loose climatological ranges (spin-up tolerant — this is a
"did the model produce a climate" gate, not a tuning target):

    TOA net  = radiation.toa_sw_down - toa_sw_up - toa_lw_up   |net| <= 10 W/m2
    precip   = clouds.precip_rain + precip_snow + convection.precip_conv
               (kg/m2/s -> mm/day)                              2 - 4 mm/day
    cloud    = max-random total cover of clouds.cloud_fraction  0.5 - 0.9
               (SPEEDY: its own shortwave_rad.cloudc)           0.4 - 0.8
    near-sfc T = temperature at the lowest level                278 - 295 K

Also scans every saved variable for NaN/Inf and, with --log, reports the
settled sim-days/hr (last chunk wall) for runtime-regression tracking.
Exit code 0 = all checks pass.

Cloud cover
-----------
``cloud_cover`` scores ECHAM's own total cover ``aclcov``
(:func:`jcm.analysis.total_cloud_cover`, ``mo_cloud.f90`` section 10.2):
maximum overlap within a vertically contiguous cloud, random overlap between
clouds separated by clear air. That is the construction the reference model
uses, and a *total* cover is the basis the satellite climatologies are quoted
on; it is deterministic, and it needs nothing but ``clouds.cloud_fraction`` —
so every saved output, at any radiation scheme, can be scored the same way.
The gated number is still not identical to what either reports, because the
saved cloud fraction is already a time mean (see below).

A SPEEDY run instead scores its own ``shortwave_rad.cloudc``, which is already
a column cover and has no profile to overlap — so it is a **different
quantity**, gated on its own band (``RANGES["cloud_cover_speedy"]``). Nothing
about it changed with the ECHAM definition, and shifting its band with the
ECHAM one would tighten a floor it sits closest to for a reason that does not
apply to it.

Two further covers are **printed and not gated**, because they answer
different questions and have no agreed band of their own:
``cloud_cover_colmax`` is the previous gate quantity (a column maximum, hence
only a lower bound on cover) and is kept so the earlier release-validation
tables stay readable across this change; ``cloud_cover_radiation`` is the
McICA sub-column cover the RRTMGP flux solve integrates, under the configured
overlap and decorrelation length.

``cloud_cover_radiation`` is a **different measurement, not a consistency
check** on the gate, and the two are not expected to agree: it is a time mean
of an *instantaneous* cover built from ``effective_cloud_fraction`` (which
independently zeroes cells with ``cloud_fraction <= 2*cld_frac_min``), whereas
``cloud_cover``/``cloud_cover_colmax`` are overlaps of the ``cloud_fraction``
the file holds, which under ``run.output_averages`` is already a time mean
over the output interval. On a 90-day T63 L47 2M arm the two read 0.53 and
0.78. Treat a large gap as expected, not as a defect.

**Cover numbers from before #707 are not comparable with these.** #707 gave
the 1M scheme ECHAM's post-microphysics cover write-back (``mo_cloud.f90``:
``paclc = FSEL(-(zxlp1_d*zxip1_d), paclc, 0)``), which clears a cell's cover
when its end-of-step condensate is below ``ccwmin`` in *both* phases. That
redefined what ``clouds.cloud_fraction`` counts, so every overlap of it
shifts. The size of the shift is known only for the column max, where the
#782 bisect measured that merge at -0.066 of low cloud for +0.15 W/m2 (the
TOA column says the redefinition is bookkeeping rather than cloud); it was not
measured on the max-random cover, and ``cloud_cover_radiation`` reduces a
differently-preprocessed field, so that figure must not be carried across to
either. Rationale, measured magnitudes and the #782 decomposition:
``docs/source/design/cloud_cover_gate.md``.
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
from jcm.analysis import (  # noqa: E402
    area_weights, global_mean, total_cloud_cover)
from aerosol_stats import (  # noqa: E402
    _AOD_KEYS, BURDEN_RANGES, anchor_gates, chunk_day, collect, format_table,
    is_jam_run, positive_int,
    missing_jam_diagnostics, physics_gates, run_files, summarize,
    timestep_seconds, unscored_gates)

RANGES = {
    "toa_net_wm2": (-10.0, 10.0),
    "precip_mm_day": (2.0, 4.0),
    # ECHAM total cloud cover under maximum-random overlap (see the module
    # docstring). Deliberately wide: this is a "did the model produce a
    # climate" gate, not a tuning target. It is also calibrated on THIS
    # definition, which matters because max-random reads 0.11-0.15 above a
    # column maximum of the same field — a band taken from column-max
    # experience sits ~0.1 low here and fails correct members on the ceiling.
    #
    # Both anchors sit inside it. Observations: a global cloud amount of
    # 0.68 +/- 0.03 for clouds of optical depth > 0.1, itself running from
    # 0.56 (COD > 2) to 0.74 (COD > 0.01) with the detection threshold (GEWEX
    # Cloud Assessment, Stubenrauch et al. 2013, BAMS 94, 1031-1049,
    # doi:10.1175/BAMS-D-12-00117.1). Model: every ECHAM member recorded in
    # the #638/#782 matrix maps into 0.59-0.83 on this definition.
    # Derivation and the measured table: docs/source/design/cloud_cover_gate.md.
    "cloud_cover": (0.5, 0.9),
    # SPEEDY scores a different quantity and so gets its own band. Its
    # ``shortwave_rad.cloudc`` is the scheme's own RH-based column cover
    # (speedy_shortwave.py), not an overlap of a profile, and no part of this
    # definition work touched it — so the max-random offset that places the
    # band above does not apply, and applying it anyway would tighten a floor
    # the member is already closest to. The recorded values are 0.57 (#638)
    # and 0.58 (#782), both comfortably inside.
    "cloud_cover_speedy": (0.4, 0.8),
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


def cloud_cover_fields(ds, speedy):
    """Cloud-cover diagnostics for one opened run window, by field dialect.

    Returns ``(fields, radiation_note)``. ``fields`` is ``{name: DataArray}``
    with ``cloud_cover`` always present and the only entry the exit code
    depends on; the others are reported for context. Each is still a full
    field — the overlap product is non-linear, so the time and area means are
    taken downstream of it by :func:`wmean`, never of the cloud fraction.
    ``radiation_note`` says *why* ``cloud_cover_radiation`` is not among the
    fields, in the two cases where a reader would otherwise wonder — the
    diagnostic is absent from the window, or it is present and identically
    zero. Those are different states of the run and printing one for the other
    misleads. It is ``None`` both when the cover *is* reported and on the
    SPEEDY path, where no radiation-view cover is expected in the first place
    and a note would be noise.

    ECHAM dialect: ``cloud_cover`` is the max-random total cover,
    ``cloud_cover_colmax`` the column maximum kept for continuity with the
    earlier tables, and ``cloud_cover_radiation`` the McICA sub-column cover
    when the run saved a non-zero one. It is absent from output written before
    the diagnostic existed (``b772ffec``, 2026-07-31) and identically zero
    under grey two-stream radiation, which samples no sub-columns; both cases
    drop the key rather than report a zero as if it were a measurement. The NN
    emulator does publish it — the analytic expectation of the same McICA
    draw.

    SPEEDY dialect: ``shortwave_rad.cloudc`` is already the scheme's own
    column cover, so there is nothing to overlap and nothing to cross-check.
    """
    if speedy:
        return {"cloud_cover": ds["shortwave_rad.cloudc"]}, None

    cloud_fraction = ds["clouds.cloud_fraction"]
    fields = {
        "cloud_cover": total_cloud_cover(cloud_fraction),
        "cloud_cover_colmax": cloud_fraction.max("level"),
    }
    radiation_cover = ds.get("radiation.total_cloud_cover")
    if radiation_cover is None:
        return fields, ("the window saves no radiation.total_cloud_cover "
                        "(output written before b772ffec carries none)")
    # Lazily: the window is an open_mfdataset, and even this 2-D field is a
    # year of it. ``.values`` here would load the lot to answer a yes/no.
    if not bool((radiation_cover != 0.0).any()):
        return fields, ("radiation.total_cloud_cover is identically zero "
                        "(grey two-stream output written before #678 "
                        "carries zeros, or the window is cloudless)")
    fields["cloud_cover_radiation"] = radiation_cover
    return fields, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--log")
    ap.add_argument("--last-n", type=positive_int, default=None,
                    help="use only the last N chunks (default: all)")
    a = ap.parse_args()

    # Same discovery as aerosol_stats.run_files, so the two cannot disagree
    # about which files are chunks (``run.snapshot_interval`` writes a
    # ``_day<N>_snapshots.nc`` stream that a ``*_day*.nc`` glob also matches).
    files = run_files(a.run_dir)
    if not files:
        print(f"FAIL  no chunk files in {a.run_dir}")
        return 1
    # The label of the chunk before a ``--last-n`` slice is the retained
    # window's start; ``chunk_centres`` needs it to place the first node.
    window_start = None
    if a.last_n:
        if a.last_n < len(files):
            window_start = chunk_day(files[-a.last_n - 1])
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
    # ``np.isfinite(ds[v])`` rather than ``np.isfinite(ds[v].values)``: on the
    # dask-backed window the former reduces chunk by chunk, the latter pulls
    # every variable of the whole window into memory one at a time.
    bad = []
    for v in ds.data_vars:
        if not bool(np.isfinite(ds[v]).all()):
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

    cover, radiation_note = cloud_cover_fields(ds, speedy)
    # The band follows the dialect, because the quantity does: see RANGES.
    check("cloud_cover", wmean(cover["cloud_cover"], weights),
          *RANGES["cloud_cover_speedy" if speedy else "cloud_cover"])
    # Reported, never gated: the column max carries the earlier
    # release-validation tables forward, the McICA cover says what the flux
    # solve saw (a different measurement, not a check on the gate — see the
    # module docstring).
    for name in ("cloud_cover_colmax", "cloud_cover_radiation"):
        if name in cover:
            print(f"INFO  {name} = {wmean(cover[name], weights):.2f} "
                  "(reported, not gated)")
    if radiation_note:
        print(f"NOTE  {radiation_note}; no radiation-view cover to report")

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
        stats = summarize(days, series, dt, window_start)
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
        unscored = unscored_gates(days, series, dt, window_start)
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
