r"""WeatherBench2 cloud ERA5 → model-grid nudging targets and initial states.

WeatherBench2 hosts public, pre-regridded ERA5 on GCS
(``gs://weatherbench2/datasets/era5``, anonymous access) — 6-hourly,
1959–2023, on 13 pressure levels (50–1000 hPa) at several resolutions.
This module pulls a run's window from the smallest store that still
oversamples the model grid, interpolates to the model's horizontal grid
and hybrid/sigma levels, caches the result locally, and hands back
either a :class:`jcm.nudging.NudgingTarget` or an initial
:class:`jcm.physics_interface.PhysicsState` (issue #610).

Vertical extent: the 13-level stores stop at 50 hPa, so interpolated
fields **clamp to the 50 hPa value above it**. Initial states spin up a
real stratosphere within days; nudging must not relax toward the
clamped values, which is why the runner masks nudging off above
``nudging.min_pressure_hpa`` (default 60).

Caching: regridded windows are written once to
``$JCM_ERA5_CACHE`` (else ``$SCRATCH/jcm-era5-cache``, else
``~/.cache/jcm/era5``) as netCDF. Derecho compute nodes have no
internet — prefetch on a login node first::

    python -m jcm.data.era5 --grid echam_t63_l47_hybrid \\
        --start 1979-01-01 --end 1980-01-01

Memory: the regrid (the prefetch, and any cache miss) runs over the time
axis in chunks of ``time_chunk`` samples (default: a 4 GiB working-set
budget, :data:`DEFAULT_CHUNK_BUDGET_BYTES`) and writes each chunk into the
cache file as it is produced, so its peak host memory scales with the
chunk, not the window. A *run* that attaches the target holds all of it in
host memory for the whole run — ``3 vars × nlev × nlon × nlat × times``.
At T63L47 that is ~15 GB per 6-hourly year; use ``freq="1d"``
(~2.5 GB/year) or shorter windows where that bites.

Requires the ``era5`` extra (``pip install jcm[era5]`` → gcsfs, zarr).
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: (nlon, store URL) — 6-hourly, 13 pressure levels, ascending preference.
#: Selection picks the smallest store whose nlon >= the model's nlon.
WB2_ERA5_STORES = (
    (64, "gs://weatherbench2/datasets/era5/"
         "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"),
    (240, "gs://weatherbench2/datasets/era5/"
          "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative"
          ".zarr"),
    (360, "gs://weatherbench2/datasets/era5/"
          "1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative"
          ".zarr"),
)

_RENAME = {
    "latitude": "lat",
    "longitude": "lon",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "T",
    "specific_humidity": "q",
    "geopotential": "z",
    "surface_pressure": "sp",
}

#: Hours between samples in the WB2 6-hourly stores.
_STORE_HOURS = 6
_FREQ_STEPS = {"6h": 1, "12h": 2, "1d": 4}

#: Host-memory budget for one time chunk of the regrid (see
#: :func:`_default_time_chunk`). 4 GiB keeps the prefetch comfortable on a
#: login node while still amortising the per-chunk store read; the working
#: set is linear in the chunk, so the budget is the knob, not the window.
DEFAULT_CHUNK_BUDGET_BYTES = 4 * 2**30


def select_store(model_nlon: int) -> str:
    """URL of the smallest WB2 store that oversamples ``model_nlon``."""
    for nlon, url in WB2_ERA5_STORES:
        if nlon >= model_nlon:
            return url
    # Higher-resolution model than any 6-hourly store: take the finest.
    return WB2_ERA5_STORES[-1][1]


def cache_dir() -> Path:
    """Local cache directory for regridded ERA5 windows."""
    env = os.environ.get("JCM_ERA5_CACHE")
    if env:
        return Path(env)
    scratch = os.environ.get("SCRATCH")
    if scratch and Path(scratch).is_dir():
        return Path(scratch) / "jcm-era5-cache"
    return Path.home() / ".cache" / "jcm" / "era5"


def _open_store(model_nlon: int) -> xr.Dataset:
    url = select_store(model_nlon)
    logger.info("era5: opening %s", url)
    ds = xr.open_zarr(url, consolidated=True,
                      storage_options={"token": "anon"})
    return ds.rename({k: v for k, v in _RENAME.items()
                      if k in ds or k in ds.coords})


def _model_latlon_deg(coords) -> tuple[np.ndarray, np.ndarray]:
    lat = np.rad2deg(np.asarray(coords.horizontal.latitudes))
    lon = np.rad2deg(np.asarray(coords.horizontal.longitudes))
    return lat, lon


def _model_level_pressures(vertical, ps: np.ndarray) -> np.ndarray:
    """Per-column model level-center pressures [Pa], ``(..., K, X, Y)``.

    ``ps`` has shape ``(..., X, Y)``; hybrid coordinates use
    ``a + b·ps``, sigma coordinates ``σ·ps``.
    """
    if hasattr(vertical, "a_centers"):
        a = np.asarray(vertical.a_centers)
        b = np.asarray(vertical.b_centers)
    else:
        a = np.zeros_like(np.asarray(vertical.centers))
        b = np.asarray(vertical.centers)
    shape = (1,) * (ps.ndim - 2) + (-1, 1, 1)
    return a.reshape(shape) + b.reshape(shape) * ps[..., None, :, :]


def _interp_log_p(field: np.ndarray, p_src_pa: np.ndarray,
                  p_tgt_pa: np.ndarray) -> np.ndarray:
    """Linear-in-log-p interpolation, vectorized over columns.

    ``field``: ``(..., L, X, Y)`` on fixed ascending ``p_src_pa (L,)``;
    ``p_tgt_pa``: ``(..., K, X, Y)`` per-column targets. Values outside
    the source range clamp to the end levels (the 50 hPa cap — see
    module docstring).
    """
    log_src = np.log(p_src_pa)
    log_tgt = np.log(p_tgt_pa)
    axis = field.ndim - 3
    idx = np.clip(np.searchsorted(log_src, log_tgt) - 1,
                  0, log_src.size - 2)
    f_lo = np.take_along_axis(field, idx, axis=axis)
    f_hi = np.take_along_axis(field, idx + 1, axis=axis)
    w = (log_tgt - log_src[idx]) / (log_src[idx + 1] - log_src[idx])
    w = np.clip(w, 0.0, 1.0)
    return f_lo + w * (f_hi - f_lo)


def _to_model_grid(ds: xr.Dataset, coords,
                   variables: tuple[str, ...]) -> xr.Dataset:
    """Regrid an ERA5 slice to the model grid: bilinear lat/lon, log-p levels.

    Returns ``(time, lev, lon, lat)`` fields plus ``sp (time, lon, lat)``
    — the layout ``NudgingTarget.from_dataset`` and ``PhysicsState``
    expect (``nodal_shape`` is lon-major).
    """
    lat, lon = _model_latlon_deg(coords)
    # Clamp the request into the store's latitude range: the pole-free
    # stores stop at ±87°, and nearest-edge is the right physical
    # fallback at the poles. Longitude is periodic — pad one wrapped
    # column so model points past the store's last longitude interpolate
    # across the seam instead of going NaN.
    lat_req = np.clip(lat, float(ds.lat[0]), float(ds.lat[-1]))
    seam = ds.isel(lon=0).assign_coords(lon=float(ds.lon[0]) + 360.0)
    padded = xr.concat([ds, seam], dim="lon")
    horiz = padded.interp(lat=xr.DataArray(lat_req, dims="lat"),
                          lon=xr.DataArray(lon, dims="lon")) \
        .assign_coords(lat=lat, lon=lon).load()

    ps = np.asarray(horiz["sp"].transpose("time", "lon", "lat").values)
    p_src = np.asarray(horiz["level"].values, dtype=float) * 100.0
    p_tgt = _model_level_pressures(coords.vertical, ps)

    out = xr.Dataset(coords={"time": horiz.time.values,
                             "lev": np.arange(p_tgt.shape[-3]),
                             "lon": lon, "lat": lat})
    for name in variables:
        src = np.asarray(
            horiz[name].transpose("time", "level", "lon", "lat").values)
        out[name] = (("time", "lev", "lon", "lat"),
                     _interp_log_p(src, p_src, p_tgt).astype(np.float32))
    out["sp"] = (("time", "lon", "lat"), ps.astype(np.float32),
                 {"units": "Pa"})
    return out


def _default_time_chunk(coords, n_variables: int) -> int:
    """Time samples per regrid chunk that fit :data:`DEFAULT_CHUNK_BUDGET_BYTES`.

    Per sample and horizontal model point the regrid holds, in float64, the
    bilinearly-interpolated source column (``n_variables`` fields on the 13
    store levels), the model level pressures and their logs, and the
    interpolation weights/neighbours (about six model-level fields), plus the
    float32 output (``n_variables`` model-level fields). The store-side read
    and the longitude-seam pad add roughly half again on top. At T63L47 with
    three variables that is ~90 MB per sample, i.e. ~45 samples (11 days of
    6-hourly data) per 4 GiB.
    """
    nlon, nlat = coords.horizontal.nodal_shape
    nlev = (np.asarray(coords.vertical.a_centers).size
            if hasattr(coords.vertical, "a_centers")
            else np.asarray(coords.vertical.centers).size)
    n_src_levels = 13
    per_point = (8 * (n_variables * n_src_levels + 6 * nlev)
                 + 4 * n_variables * nlev)
    per_sample = int(1.5 * per_point * int(nlon) * int(nlat))
    return max(1, DEFAULT_CHUNK_BUDGET_BYTES // per_sample)


def _regrid_chunks(window: xr.Dataset, coords,
                   variables: tuple[str, ...], chunk: int):
    """Yield ``(start_index, regridded_chunk)`` over the time axis.

    The regrid is separable in time: the bilinear lat/lon interpolation and
    the per-column log-p interpolation act on one time sample at a time, so
    a chunk's values are bit-identical to the same samples of a single-piece
    regrid. ``chunk`` bounds the samples handed to :func:`_to_model_grid`,
    which is what bounds the host memory.
    """
    for i0 in range(0, window.time.size, chunk):
        yield i0, _to_model_grid(window.isel(time=slice(i0, i0 + chunk)),
                                 coords, variables)


def _output_template(first: xr.Dataset, times: np.ndarray, *, lazy: bool,
                     attrs: dict) -> xr.Dataset:
    """Full-window dataset shaped like the chunk ``first`` but spanning ``times``.

    On the lazy path the data variables are dask zeros, one time sample per
    dask chunk, which only fix the on-disk layout and are never held at full
    size. ``lazy=False`` allocates them (the ``cache=False`` return value has
    to exist in memory anyway).
    """
    import dask.array as da

    other = {k: v for k, v in first.coords.items() if k != "time"}
    out = xr.Dataset(coords={"time": times, **other})
    for name, var in first.data_vars.items():
        shape = (times.size,) + var.shape[1:]
        if lazy:
            data = da.zeros(shape, dtype=var.dtype,
                            chunks=(1,) + var.shape[1:])
        else:
            data = np.empty(shape, dtype=var.dtype)
        out[name] = (var.dims, data, dict(var.attrs))
    out.attrs.update(attrs)
    return out


def _regrid_window(window: xr.Dataset, coords, variables: tuple[str, ...],
                   chunk: int, *, attrs: dict,
                   path: Path | None = None) -> xr.Dataset | None:
    """Regrid ``window`` in time chunks, into memory or into the file ``path``.

    With ``path`` each chunk is written into the netCDF file as soon as it is
    produced and then dropped, so the peak is one chunk; returns ``None``.
    Without it the chunks fill a preallocated result, so the peak is that
    result plus one chunk. The file's layout comes from writing the
    full-window template with xarray, so it is the file a single
    ``to_netcdf`` of the whole result would write (netCDF cannot grow a fixed
    time dimension by appending).
    """
    import itertools

    chunks = _regrid_chunks(window, coords, variables, chunk)
    first_start, first = next(chunks)
    all_chunks = itertools.chain([(first_start, first)], chunks)
    template = _output_template(first, window.time.values,
                                lazy=path is not None, attrs=attrs)
    if path is None:
        for start, part in all_chunks:
            for name in part.data_vars:
                template[name].values[start:start + part.time.size] = \
                    part[name].values
        return template

    import netCDF4

    template.to_netcdf(path, engine="netcdf4", compute=False) \
        .compute(scheduler="synchronous")
    with netCDF4.Dataset(path, "r+") as nc:
        for start, part in all_chunks:
            for name in part.data_vars:
                nc[name][start:start + part.time.size] = part[name].values
    return None


def _window_key(coords, start: str, end: str, freq: str,
                variables: tuple[str, ...]) -> str:
    import hashlib

    nlon, nlat = coords.horizontal.nodal_shape
    # Hash the actual coordinate values, not just the dimensions: two
    # grids with the same shape (e.g. different 8-level sigma
    # definitions) must not share a cache entry.
    if hasattr(coords.vertical, "a_centers"):
        vert = (np.asarray(coords.vertical.a_centers),
                np.asarray(coords.vertical.b_centers))
    else:
        vert = (np.asarray(coords.vertical.centers),)
    nlev = vert[0].size
    h = hashlib.sha256()
    for arr in (np.asarray(coords.horizontal.latitudes),
                np.asarray(coords.horizontal.longitudes), *vert):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return (f"wb2_{nlon}x{nlat}_l{nlev}_{h.hexdigest()[:8]}_"
            f"{start}_{end}_{freq}_{'-'.join(variables)}.nc")


def dataset_on_model_grid(coords, start: str, end: str, *,
                          freq: str = "6h",
                          variables: tuple[str, ...] = ("u", "v", "T"),
                          cache: bool = True,
                          time_chunk: int | None = None) -> xr.Dataset:
    """Return the regridded ERA5 window ``[start, end]``, cached locally.

    ``freq`` subsamples the 6-hourly store (``"6h"``, ``"12h"``,
    ``"1d"``). The cache key covers grid, levels, window, frequency and
    variables, so different runs share files only when identical.

    The regrid runs over the window in chunks of ``time_chunk`` time
    samples (default: as many as fit :data:`DEFAULT_CHUNK_BUDGET_BYTES`,
    see :func:`_default_time_chunk`), so host memory is bounded by the chunk
    rather than the window. ``time_chunk`` changes the peak only: the
    result, and the cache file, are the same for every value. With
    ``cache=True`` each chunk is written into the cache file as it is
    produced and the returned dataset is the lazily-opened file (as on a
    cache hit); with ``cache=False`` it is the in-memory result.
    """
    if freq not in _FREQ_STEPS:
        raise ValueError(f"freq {freq!r} not in {sorted(_FREQ_STEPS)}")
    path = cache_dir() / _window_key(coords, start, end, freq, variables)
    if cache and path.exists():
        logger.info("era5: cache hit %s", path)
        return xr.open_dataset(path)

    nlon, _ = coords.horizontal.nodal_shape
    ds = _open_store(int(nlon))
    window = ds.sel(time=slice(start, end)) \
        .isel(time=slice(None, None, _FREQ_STEPS[freq]))
    if window.time.size == 0:
        raise ValueError(
            f"ERA5 window [{start}, {end}] is empty — the WB2 stores "
            f"cover {str(ds.time.values[0])[:10]} to "
            f"{str(ds.time.values[-1])[:10]}")
    if time_chunk is None:
        time_chunk = _default_time_chunk(coords, len(variables))
    if time_chunk < 1:
        raise ValueError(f"time_chunk must be >= 1, got {time_chunk}")
    window = window[list(variables) + ["sp"]]
    attrs = {"source": select_store(int(nlon))}
    if not cache:
        return _regrid_window(window, coords, variables, time_chunk,
                              attrs=attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Globally-unique temp name + atomic replace: concurrent runs may
    # request the same uncached window, and a shared tmp path would let
    # their writes interleave (or the second rename fail).
    #
    # NOT os.getpid(): each container gets its own PID namespace, so two
    # pods sharing a cache PVC routinely land on the SAME pid and collide
    # on the temp file. That is not hypothetical — two jcm Jobs launched
    # together both got pid 382 and one died with
    # "PermissionError: .../wb2_..._6h_u-v-T.tmp382.nc". uuid4 is unique
    # across hosts, which pid is not.
    tmp = path.with_suffix(f".tmp{uuid.uuid4().hex}.nc")
    try:
        _regrid_window(window, coords, variables, time_chunk,
                       attrs=attrs, path=tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("era5: cached %s", path)
    return xr.open_dataset(path)


def nudging_target(coords, start: str, end: str, *, freq: str = "6h",
                   cache: bool = True):
    """Build a time-varying :class:`~jcm.nudging.NudgingTarget` for the window."""
    from jcm.nudging import NudgingTarget
    ds = dataset_on_model_grid(coords, start, end, freq=freq, cache=cache)
    return NudgingTarget.from_dataset(ds)


def initial_state(coords, date: str, *, cache: bool = True):
    """Build a :class:`PhysicsState` initial condition from ERA5 at ``date``.

    The nearest at-or-before 6-hourly sample is used. Above 50 hPa the
    thermodynamic profile clamps (see module docstring) — acceptable for
    an initial condition, which the model re-equilibrates within days.
    """
    import jax.numpy as jnp

    import jcm.constants as c
    from jcm.physics_interface import PhysicsState

    day = str(date)[:10]
    ds = dataset_on_model_grid(coords, day, day, freq="6h",
                               variables=("u", "v", "T", "q", "z"),
                               cache=cache)
    at = ds.sel(time=str(date), method="pad").load()

    def grab(name):
        return jnp.asarray(at[name].values)

    return PhysicsState(
        u_wind=grab("u"),
        v_wind=grab("v"),
        temperature=grab("T"),
        specific_humidity=grab("q"),
        geopotential=grab("z"),
        normalized_surface_pressure=jnp.asarray(at["sp"].values) / c.p0,
    )


def _main(argv=None) -> int:
    """Prefetch a window into the local cache (login node, then run)."""
    import argparse

    ap = argparse.ArgumentParser(description=_main.__doc__)
    ap.add_argument("--grid", required=True,
                    help="jcm grid preset, e.g. echam_t63_l47_hybrid")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--freq", default="6h", choices=sorted(_FREQ_STEPS))
    ap.add_argument("--time-chunk", type=int, default=None,
                    help="time samples regridded at once (default: sized "
                         "to a 4 GiB working set); lower it on a "
                         "memory-limited node, the cache is unchanged")
    ap.add_argument("--init", action="store_true",
                    help="also prefetch the initial-state variables at "
                         "--start")
    args = ap.parse_args(argv)

    import pathlib as _pathlib

    from hydra import compose, initialize_config_dir

    from jcm.runners import build_coords

    # initialize_config_dir, not initialize_config_module: jcm/config is a
    # directory of YAML shipped as package data, not an importable module, so
    # the module form fails with "Primary config module 'jcm.config' not
    # found" and made this documented prefetch command unusable.
    config_dir = _pathlib.Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base=None,
                               config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[f"grid={args.grid}"])
    coords = build_coords(cfg)
    ds = dataset_on_model_grid(coords, args.start, args.end, freq=args.freq,
                               time_chunk=args.time_chunk)
    print(f"nudging window cached: {ds.time.size} steps")
    if args.init:
        initial_state(coords, args.start)
        print("initial-state slice cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
