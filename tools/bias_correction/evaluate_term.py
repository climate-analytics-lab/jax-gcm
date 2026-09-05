"""Free-run evaluation: run a configuration and save its mean fields.

Runs SPEEDY (plain and/or with a bias-correction term) for 450 days
(Jan 1 start, first 90 days dropped so the seasonal windows are spin-up
free), computes the time-mean fields, and saves them to small NetCDFs under
OUT_DIR so figures can be restyled without re-running:

  eval_fields_<tag>.nc   per config: t_surf_annual/djf/jja, t_mid_annual
                         (~500 hPa), q_surf_annual
  eval_fields_era5.nc    the regridded ERA5 references (written once)

Paths resolve from the repo root, two directories above this script, so it
runs from any cwd; JCM_REPO overrides that. The run uses the GPU pinned via
CUDA_VISIBLE_DEVICES and does not force CPU. Run with:

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python tools/bias_correction/evaluate_term.py

Environment knobs. The M4_ prefix is the name of the milestone this evaluator
was written in; it is kept because every recorded run and the shell drivers
use it.

  M4_TAG      tag for the term config's output file (default "term")
  M4_TERM     .npz artifact to evaluate (default the shipped reference term).
              A COMMA-SEPARATED list evaluates the ensemble mean of those
              terms: each member's out_scale is divided by the member count and
              all of them are added to the physics, and since the terms sum
              their tendencies that is exactly the mean correction. Tested and
              found not to beat the best member; see the comment at MEMBERS.
  M4_TRUNC    spectral truncation (default 31)
  M4_DT_MIN   model step in minutes (default 30)
  M4_OUT_DIR  where the .nc files are read and written (default <repo>/eval_out)
  M4_YEAR     calendar year the run starts on, 1 January (default 2001)
  M4_DAYS     run length in days including the 90-day spin-up (default 450)
  M4_CHUNK_DAYS  integrate in segments of this many days, reducing each to
                 season sums before the next (default 0, one segment). A long
                 run needs this: 21 years saved every 5 days is 1533 snapshots
                 held at once. Segments are one continuous trajectory, not
                 restarts, so the answer does not depend on the segment length.
  M4_BLOCK_YEARS  block length the scored span is also cut into, in years
                 (default 7). Blocks are written as extra `_blkN` files and
                 share a trained network, so their spread across a metric is
                 sampling noise alone. Set 0 to skip them.

A longer run scores a longer MODEL climatology against the same observed one.
That is legitimate here because the free run is forced by a climatology, not by
the boundary conditions of the scored years: the calendar year picks which ERA5
years the reference averages and nothing else. Lengthening the run therefore
buys down the model side of the sampling noise and leaves the observed side
alone, and the observed side is common to every config so it cannot change a
ranking.

The cached plain and ERA5 references are tied to the MODEL GRID, because the
ERA5 fields are interpolated onto it. A run at a different M4_TRUNC therefore
needs its own M4_OUT_DIR: reusing the T31 directory would score a T63 term
against T31-grid references, which xarray aligns to an empty intersection and
reports as NaN rather than as an error. The grid check below refuses that.

Plain SPEEDY is evaluated only if eval_fields_plain.nc does not exist yet.

Stats print BOTH unweighted grid means and cos(lat) area-weighted ones. The
weighted pair is the area-true statistic and the one every reported number uses.
"""
import glob
import os
import shutil
import time

import numpy as np
import xarray as xr
import jax_datetime as jdt
from importlib import resources

from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics.bias_correction import load_bias_correction
from jcm.physics.bias_correction.eval_protocol import (
    BLOCK_YEARS, LAST_FULL_YEAR, SEASONS, SeasonMeans, chunk_spans,
    period_suffix, reference_label, resolve_reference, scoring_blocks,
    too_short_to_score)

# The chdir is for the relative data paths below, not for the imports, so it
# sits after them. This script uses the GPU pinned via CUDA_VISIBLE_DEVICES and
# deliberately does NOT force CPU. The repo root is two directories up from
# this script; JCM_REPO overrides it for an out-of-tree checkout.
REPO = os.environ.get("JCM_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

OUT_DIR = os.environ.get("M4_OUT_DIR") or os.path.join(REPO, "eval_out")
os.makedirs(OUT_DIR, exist_ok=True)

# The evaluated period. Defaults reproduce the canonical single-year protocol
# (450 days from 1 January 2001, first 90 discarded), which every reported
# number used. M4_YEAR/M4_DAYS move it, e.g. a 2016-2022 holdout is
# M4_YEAR=2016 M4_DAYS=2645 (90 spin-up + 7 x 365).
YEAR = int(os.environ.get("M4_YEAR", "2001"))
TOTAL_DAYS = float(os.environ.get("M4_DAYS", "450"))
SAVE_DAYS = 5.0
DROP_DAYS = 90.0          # spin-up excluded from every window
CHUNK_DAYS = float(os.environ.get("M4_CHUNK_DAYS", "0"))
BLOCK_YEARS_ENV = int(os.environ.get("M4_BLOCK_YEARS", str(BLOCK_YEARS)))
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")

TAG = os.environ.get("M4_TAG", "term")
# Defaults to the shipped reference term, so running this script with no
# arguments scores the same thing docs/source/bias_correction.rst reports. It
# used to default to online_term_t31.npz, which is an early single-year term.
TERM_PATH = os.environ.get(
    "M4_TERM", "jcm/data/bias_correction/online_term_t31_big_insol_vt.npz")

t0 = time.time()
# Resolution/step follow the same env knobs as finetune_climatology.py so an
# evaluation always matches the run it is scoring. Defaults are T31 at 30 min,
# which is what every reported number used.
TRUNC = int(os.environ.get("M4_TRUNC", "31"))
DT_MIN = float(os.environ.get("M4_DT_MIN", "30"))
coords = get_speedy_coords(layers=8, spectral_truncation=TRUNC)
print("T%d grid %s  dt %g min  out %s"
      % (TRUNC, coords.horizontal.nodal_shape, DT_MIN, OUT_DIR), flush=True)
bc = resources.files("jcm.data.bc.t30.clim")
terrain = TerrainData.from_file(bc / "terrain.nc", coords=coords)
forcing = ForcingData.from_file(bc / "forcing.nc", coords=coords)
start = jdt.to_datetime(f"{YEAR}-01-01")

_scored_days = TOTAL_DAYS - DROP_DAYS
# Which observed years to score against, and whether this is the canonical
# protocol every reported figure used. See eval_protocol for why the canonical
# reference is pinned rather than derived from the run length.
REF_YEARS, REF2_YEAR, IS_CANONICAL = resolve_reference(
    YEAR, TOTAL_DAYS, drop_days=DROP_DAYS,
    ref_years_env=os.environ.get("M4_REF_YEARS"),
    ref2_year_env=os.environ.get("M4_REF2_YEAR"),
    last_full_year=int(os.environ.get("M4_LAST_FULL_YEAR",
                                      str(LAST_FULL_YEAR))))
_ref_label = reference_label(REF_YEARS)
print("scoring %d days from %d-01-01 (%.0f scored); ERA5 reference %s, "
      "spot check %s%s" % (TOTAL_DAYS, YEAR, _scored_days, _ref_label,
                           REF2_YEAR or "none",
                           "  [canonical protocol]" if IS_CANONICAL else ""),
      flush=True)

# Cached plain/ERA5 references and the term's own output are tied to the
# PERIOD, not just the grid, so a holdout run launched without M4_OUT_DIR
# cannot overwrite a canonical file with fields from a different period.
PERIOD_SUFFIX = period_suffix(YEAR, TOTAL_DAYS)

# Every window this run is scored over: the whole scored span first, then the
# equal blocks it is long enough to be cut into. The blocks share a trained
# network, so the spread of a metric across them is sampling noise from the
# model's own internal variability alone. Nothing measured that before:
# reseeding training moves the scores, but it changes the network at the same
# time, so a seed spread cannot say how much of itself is sampling.
#
# Blocks are anchored at the spin-up drop and are a whole number of years long,
# so block 0 of a long run covers exactly the days the 2645-day holdout scored.
# It will not reproduce that number to the decimal -- segmenting the rollout
# perturbs the trajectory at roundoff and chaos grows it -- but it should land
# within the spread the blocks themselves measure. If it does not, the long run
# and the published one disagree about something other than sampling.
_too_short = too_short_to_score(TOTAL_DAYS, drop_days=DROP_DAYS)
if _too_short:
    raise SystemExit("Nothing to score: " + _too_short)

BLOCKS = (scoring_blocks(TOTAL_DAYS, drop_days=DROP_DAYS,
                         block_years=BLOCK_YEARS_ENV)
          if BLOCK_YEARS_ENV else ())
WINDOWS = ((("", DROP_DAYS, TOTAL_DAYS),)
           + tuple((f"_blk{i}", lo, hi) for i, (lo, hi) in enumerate(BLOCKS)))
if BLOCKS:
    print("scored in %d blocks of %d years: %s"
          % (len(BLOCKS), BLOCK_YEARS_ENV,
             ", ".join("%.0f-%.0f" % b for b in BLOCKS)), flush=True)


def run_config(label, physics):
    """Integrate one config, reducing each segment to season sums as it goes."""
    spans = chunk_spans(TOTAL_DAYS, CHUNK_DAYS, SAVE_DAYS)
    print("running: %s (%d segment%s) ..."
          % (label, len(spans), "" if len(spans) == 1 else "s"), flush=True)
    model = Model(coords=coords, terrain=terrain, physics=physics,
                  start_date=start, calendar="365_day", time_step=DT_MIN)
    acc = SeasonMeans(WINDOWS)
    for i, span in enumerate(spans):
        # `resume` rather than a fresh `run`: it threads the cross-step physics
        # carry (radiation cache, prior-step TKE), so the segments are one
        # continuous trajectory instead of a sequence of restarts. The seam
        # still perturbs the trajectory at roundoff and chaos grows that into a
        # different realisation, which is not a defect: it is the same sampling
        # noise the blocks measure, and it is why the metrics are read off a
        # multi-year mean rather than off the trajectory itself.
        preds = (model.run(forcing=forcing, save_interval=SAVE_DAYS,
                           total_time=span) if i == 0 else
                 model.resume(forcing=forcing, save_interval=SAVE_DAYS,
                              total_time=span))
        ds = preds.to_xarray()
        ds = ds.assign_coords({c: np.asarray(ds[c].values) for c in ds.coords})
        if not bool(np.isfinite(np.asarray(ds["temperature"].values)).all()):
            print("  segment %d went non-finite (+%.0fs)"
                  % (i + 1, time.time() - t0), flush=True)
            return None
        acc.add(ds, sim_days(ds))
        print("  segment %d/%d, %.0f days (+%.0fs)"
              % (i + 1, len(spans), span, time.time() - t0), flush=True)
    return acc


def sim_days(ds):
    """Elapsed run days from the time coordinate.

    ModelPredictions.to_xarray writes ABSOLUTE datetime64 dates. A float cast
    of those yields nanoseconds since 1970, so `days >= DROP_DAYS` would drop
    nothing and `days % 365` would scramble the season masks into near-annual
    noise. Anchor on the run start date instead.
    """
    t = np.asarray(ds.time.values)
    if np.issubdtype(t.dtype, np.datetime64):
        # Anchor on the run's own start, not a literal year: a hardcoded 2001
        # made every day negative for any other period and silently disabled
        # both the spin-up drop and the season masks.
        return ((t - np.datetime64(f"{YEAR}-01-01")) / np.timedelta64(1, "D")
                ).astype(float)
    return t.astype(float)


def level_indices(acc, window=""):
    """Pick the (surface, ~500 hPa) level indices to read the metrics off.

    Chosen once from the full window and reused for every block. The surface
    index is the warmest level in the annual mean, which is the bottom one in
    any sane run, but deriving it per block would let one block score a
    different level from the others and call the difference a result.
    """
    t_annual = acc.mean(window, "annual", "temperature")
    surf = int(t_annual.mean(["lon", "lat"]).argmax("level"))
    # ~500 hPa: sigma nearest 0.5 in coords.vertical.centers (top-down),
    # flipped because the xarray level axis is surface-first.
    sigmas = np.asarray(coords.vertical.centers)
    mid = len(sigmas) - 1 - int(np.argmin(np.abs(sigmas - 0.5)))
    return surf, mid


def config_fields(acc, window="", levels=None):
    """Return the plotted fields for one config and window, all (lon, lat)."""
    t = {season: acc.mean(window, season, "temperature") for season in SEASONS}
    surf, mid = levels if levels is not None else level_indices(acc, window)

    out = {f"t_surf_{season}": t[season].isel(level=surf) for season in SEASONS}
    out["t_mid_annual"] = t["annual"].isel(level=mid)
    out["q_surf_annual"] = acc.mean(window, "annual",
                                    "specific_humidity").isel(level=surf)
    return out


def save_fields(fields, path, attrs=None):
    ds = xr.Dataset({k: v.reset_coords(drop=True) for k, v in fields.items()})
    if attrs:
        ds.attrs.update(attrs)
    ds.to_netcdf(path)
    print("wrote", path, flush=True)


def save_config(acc, tag):
    """Write the full-window fields and each block's; return both."""
    levels = level_indices(acc)
    fields = config_fields(acc, levels=levels)
    save_fields(fields, os.path.join(OUT_DIR,
                                     f"eval_fields_{tag}{PERIOD_SUFFIX}.nc"))
    blocks = {}
    for name, lo, hi in WINDOWS[1:]:
        blocks[name] = config_fields(acc, name, levels=levels)
        save_fields(blocks[name],
                    os.path.join(OUT_DIR,
                                 f"eval_fields_{tag}{name}{PERIOD_SUFFIX}.nc"),
                    attrs={"block_days": "%.0f-%.0f" % (lo, hi),
                           "block_years": str(BLOCK_YEARS_ENV)})
    return fields, blocks


def load_blocks(tag):
    """Block fields already on disk for a config whose run was cached."""
    out = {}
    for name, _, _ in WINDOWS[1:]:
        path = os.path.join(OUT_DIR, f"eval_fields_{tag}{name}{PERIOD_SUFFIX}.nc")
        if os.path.exists(path):
            out[name] = {k: v for k, v in xr.open_dataset(path).items()}
    return out


# --- ERA5 references (written once) -----------------------------------------
def era5_climatology(var, year, months=None, level=None):
    """Observed climatology for a year, or a ``(first, last)`` span."""
    era5 = xr.open_zarr(WB2, consolidated=True,
                        storage_options={"token": "anon"})
    era5 = era5.rename({"latitude": "lat", "longitude": "lon"})
    if isinstance(year, tuple):
        lo, hi = year
        da = era5[var].sel(time=slice(str(lo), str(hi)))
    else:
        da = era5[var].sel(time=str(year))
    da = da.isel(time=slice(0, None, 4))
    if months is not None:
        da = da.sel(time=da["time"].dt.month.isin(months))
    if level is not None:
        da = da.sel(level=level)
    da = da.mean("time")
    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    return da


def on_model_grid(da, like):
    tgt_lon = like.lon % 360
    out = da.interp(lat=like.lat, lon=tgt_lon,
                    kwargs={"fill_value": "extrapolate"})
    return out.assign_coords(lon=like.lon)


def stats(bias):
    """(unweighted mean, unweighted RMS, cos-lat-weighted mean, weighted RMS).

    The weighted pair is the area-true statistic and the reported one: an
    equal-angle lat-lon grid over-weights the poles about 2.4x, which inflates
    the polar band's RMS contribution and lets it spuriously cancel the
    tropical mean bias. The unweighted pair is printed for comparison.
    """
    b = np.asarray(bias)
    w = np.broadcast_to(np.cos(np.deg2rad(np.asarray(bias.lat))), b.shape)
    return (float(b.mean()), float(np.sqrt((b ** 2).mean())),
            float((w * b).sum() / w.sum()),
            float(np.sqrt((w * b * b).sum() / w.sum())))


# --- run the configs ---------------------------------------------------------
plain_path = os.path.join(OUT_DIR, f"eval_fields_plain{PERIOD_SUFFIX}.nc")
era5_path = os.path.join(OUT_DIR, f"eval_fields_era5{PERIOD_SUFFIX}.nc")

if not os.path.exists(plain_path):
    acc = run_config("plain SPEEDY", speedy_physics())
    if acc is None:
        raise SystemExit("plain SPEEDY went non-finite")
    plain_fields, plain_blocks = save_config(acc, "plain")
else:
    plain_fields = {k: v for k, v in xr.open_dataset(plain_path).items()}
    cached = plain_fields["t_surf_annual"]
    got = (cached.sizes["lon"], cached.sizes["lat"])
    want = tuple(coords.horizontal.nodal_shape)
    if got != want:
        raise SystemExit(
            "%s is on a %s grid but this run is %s (M4_TRUNC=%d). Point "
            "M4_OUT_DIR at a directory for this resolution -- reusing the "
            "other one scores against mismatched references, which aligns to "
            "an empty intersection and prints NaN instead of failing."
            % (plain_path, got, want, TRUNC))
    # The blocks are separate files; a cached plain run may or may not have
    # them, depending on whether it predates blocking.
    plain_blocks = load_blocks("plain")
    print("plain fields loaded from cache%s"
          % (" (%d blocks)" % len(plain_blocks) if plain_blocks else ""),
          flush=True)

# The ensemble mean is free to evaluate: the terms sum, so scaling each member's
# out_scale by 1/N makes the sum the mean. It does not help. Averaging seed pairs,
# context-feature sets and all seven wide terms (2026-08-25, 2016-2022 holdout)
# landed near the member average every time rather than beating the best member,
# because each member is a different biased correction and not independent noise
# around the truth. Kept as a mechanism, not as a recommendation.
MEMBERS = [t.strip() for t in TERM_PATH.split(",") if t.strip()]
terms = [load_bias_correction(m) for m in MEMBERS]
if len(terms) > 1:
    print("evaluating a %d-member ensemble mean:" % len(terms), flush=True)
    for m in MEMBERS:
        print("   ", m, flush=True)
    # Scaling out_scale is exact whether or not a member has an output_cap:
    # the cap acts on the dimensionless network output, before this scale.
    terms = [t.rebuild(t.weights.get_value(),
                       out_scale=t.out_scale.get_value() / len(terms))
             for t in terms]
else:
    print("evaluating term:", MEMBERS[0], "| tag:", TAG,
          "| output_cap:", terms[0].output_cap, flush=True)

physics = speedy_physics()
for t in terms:
    physics = physics + t
acc = run_config("+ term (%s)" % TAG, physics)
if acc is None:
    raise SystemExit("term run went non-finite")
term_fields, term_blocks = save_config(acc, TAG)

def adopt_matching_era5(path, label):
    """Copy in a reference file built for the same observed span, if one exists.

    The reference depends on the ERA5 years and the model grid, not on how long
    the model ran, but its filename is keyed to the run period. Without this a
    21-year run re-downloads a file identical to the one the 7-year run already
    built, and needs network access on a box that may not have it.
    """
    want_grid = tuple(coords.horizontal.nodal_shape)
    for cand in sorted(glob.glob(os.path.join(OUT_DIR, "eval_fields_era5*.nc"))):
        if os.path.abspath(cand) == os.path.abspath(path):
            continue
        with xr.open_dataset(cand) as ds:
            if ds.attrs.get("ref_years") != label:
                continue
            ref = ds.get("era5_t2m_annual")
            # Grid, not just span: an out-dir shared across resolutions would
            # otherwise hand a T63 run the T31 reference, which xarray aligns
            # to an empty intersection and reports as NaN.
            if ref is None or (ref.sizes["lon"], ref.sizes["lat"]) != want_grid:
                continue
        shutil.copyfile(cand, path)
        print("era5 refs copied from %s (same reference span %s)"
              % (os.path.basename(cand), label), flush=True)
        return True
    return False


if not os.path.exists(era5_path):
    adopt_matching_era5(era5_path, _ref_label)

if not os.path.exists(era5_path):
    like = plain_fields["t_surf_annual"]
    months = {"annual": None, "djf": [12, 1, 2], "jja": [6, 7, 8]}
    refs = {}
    for season, mm in months.items():
        refs[f"era5_t2m_{season}"] = on_model_grid(
            era5_climatology("2m_temperature", REF_YEARS, months=mm), like)
    if REF2_YEAR is not None:
        refs["era5_t2m_2002_annual"] = on_model_grid(
            era5_climatology("2m_temperature", REF2_YEAR), like)
    refs["era5_t500"] = on_model_grid(
        era5_climatology("temperature", REF_YEARS, level=500), like)
    refs["era5_q925"] = on_model_grid(
        era5_climatology("specific_humidity", REF_YEARS, level=925),
        like) * 1000.0
    save_fields(refs, era5_path,
                attrs={"ref_years": _ref_label,
                       "ref2_year": str(REF2_YEAR or "none")})
    # Re-open what was just written. The dict above holds lazy handles on the
    # WeatherBench2 store, so every stats() line below would re-read seven
    # years of it over the network, once per metric per config. That was merely
    # slow when the table was ten lines; with a block table per config it is
    # dozens of round trips for data already sitting on disk.
    refs = {k: v for k, v in xr.open_dataset(era5_path).items()}
else:
    # Report the years the CACHED file was built from, not the ones this run
    # would have chosen. Printing the intended span while silently scoring
    # against a differently-built cache is how a log ends up describing an
    # evaluation that never happened.
    _cached = xr.open_dataset(era5_path)
    refs = {k: v for k, v in _cached.items()}
    _cached_label = _cached.attrs.get("ref_years")
    if _cached_label is None:
        # Written before this attribute existed. The filename still identifies
        # the span: `period_suffix` returns "" only for the canonical 2001/450
        # protocol, so an unsuffixed file is necessarily the 2001 reference,
        # while a suffixed one was built by a version that already honoured
        # M4_YEAR/M4_DAYS and therefore holds the span its name states.
        # Claiming "canonical 2001" for a suffixed file -- as this branch used
        # to -- sends the reader chasing a contamination that is not there.
        if PERIOD_SUFFIX == "":
            print("era5 refs loaded from cache (built before the reference "
                  "span was recorded; unsuffixed, so the canonical 2001 refs)",
                  flush=True)
        else:
            print("era5 refs loaded from cache (span not recorded in the file; "
                  "the %s suffix means it was built for %s)"
                  % (PERIOD_SUFFIX, _ref_label), flush=True)
    else:
        print("era5 refs loaded from cache: reference %s, spot check %s"
              % (_cached_label, _cached.attrs.get("ref2_year", "unrecorded")),
              flush=True)
        if _cached_label != _ref_label:
            print("  WARNING: this run asked for %s but the cache holds %s; "
                  "delete %s to rebuild."
                  % (_ref_label, _cached_label, era5_path), flush=True)

# --- stats table --------------------------------------------------------------
# Two pairs per line: unweighted, for comparison, and cos(lat)-weighted, the
# area-true statistic every reported number uses.
for label, fields in (("plain SPEEDY", plain_fields), (TAG, term_fields)):
    for season in ("annual", "djf", "jja"):
        gm, rm, wgm, wrm = stats(
            fields[f"t_surf_{season}"] - refs[f"era5_t2m_{season}"])
        print("%-14s T %-7s mean %+.2f K  RMS %.2f K | wgt mean %+.2f K  RMS %.2f K"
              % (label, season, gm, rm, wgm, wrm), flush=True)
    gm, rm, wgm, wrm = stats(fields["t_mid_annual"] - refs["era5_t500"])
    print("%-14s T 500hPa mean %+.2f K  RMS %.2f K | wgt mean %+.2f K  RMS %.2f K"
          % (label, gm, rm, wgm, wrm), flush=True)
    gm, rm, wgm, wrm = stats(fields["q_surf_annual"] - refs["era5_q925"])
    print("%-14s q annual mean %+.2f    RMS %.2f g/kg | wgt mean %+.2f  RMS %.2f"
          % (label, gm, rm, wgm, wrm), flush=True)
    if "era5_t2m_2002_annual" in refs:
        gm, rm, wgm, wrm = stats(
            fields["t_surf_annual"] - refs["era5_t2m_2002_annual"])
        print("%-14s T vs%s mean %+.2f K  RMS %.2f K | wgt mean %+.2f K  "
              "RMS %.2f K" % (label, REF2_YEAR, gm, rm, wgm, wrm), flush=True)

# --- the sampling floor this run measures -----------------------------------
# Every block holds the same trained network and scores a different slice of
# the same trajectory, so their spread is the model's internal variability and
# nothing else. That is the number the five metrics have been missing: a reseed
# moves them too, but a reseed also changes the network, so it cannot say how
# much of its own spread is just sampling. Read the spread below as the
# smallest gap between two terms that could mean anything at this window
# length, and compare it against the seed spreads in holdout_table.py.
METRIC_REFS = (("T ann", "t_surf_annual", "era5_t2m_annual"),
               ("T DJF", "t_surf_djf", "era5_t2m_djf"),
               ("T JJA", "t_surf_jja", "era5_t2m_jja"),
               ("T 500", "t_mid_annual", "era5_t500"),
               ("q", "q_surf_annual", "era5_q925"))


def block_table(label, full, blocks):
    """Per-block weighted RMS for one config, with the spread and full window."""
    if not blocks:
        return
    hdr = "".join(f"{lbl:>8s}" for lbl, _, _ in METRIC_REFS)
    print()
    print(f"{label} per block ({BLOCK_YEARS_ENV}-year windows)")
    print(f"{'window':16s}{hdr}")
    rows = {}
    # Numeric order, not lexicographic: with ten or more blocks a plain sort
    # puts _blk10 between _blk1 and _blk2.
    for name in sorted(blocks, key=lambda n: int(n.rsplit("blk", 1)[1])):
        vals = [stats(blocks[name][k] - refs[r])[3] for _, k, r in METRIC_REFS]
        rows[name] = vals
        print(f"{name.lstrip('_'):16s}" + "".join(f"{v:8.2f}" for v in vals))
    spread = [max(c) - min(c) for c in zip(*rows.values())] if len(rows) > 1 \
        else [float("nan")] * len(METRIC_REFS)
    print(f"{'SAMPLING SPREAD':16s}" + "".join(f"{v:8.2f}" for v in spread))
    print(f"{'full window':16s}"
          + "".join(f"{stats(full[k] - refs[r])[3]:8.2f}"
                    for _, k, r in METRIC_REFS))


block_table("plain SPEEDY", plain_fields, plain_blocks)
block_table(TAG, term_fields, term_blocks)

print()
print("M4 EVAL DONE (+%.0fs)" % (time.time() - t0), flush=True)
