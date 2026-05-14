"""Ozone forcing for the ECHAM-RRTMGP pipeline.

Reads a *pre-interpolated* ozone netCDF (``(time, level=nlev, lat, lon)``
mole/mole — the format produced by ``jcm.data.bc.interpolate_ozone``)
and exposes a per-column profile that ``EchamBoundaryConditions`` hands
straight to RRTMGP. The vertical interpolation from the source file's
plev grid onto the model's hybrid-level centers happens **offline** in
the prep script so the online code is just an array slice — no per-step
``vmap`` of ``jnp.interp``.

Two routing modes, auto-detected from the file's time-axis length:

* ``ntime == 12`` — climatology, ``align_mode=WRAP_YEAR``. Year wraps
  to itself; the same January slice gets returned every January
  regardless of year. Matches the prep-script default for files like
  ``T63L47_ozone_picontrol.nc``.
* ``ntime > 12``  — transient, ``align_mode=BY_DATE``. The file's
  ``time`` coordinate is decoded to absolute seconds since
  ``1970-01-01`` and ``ForcingData.select(date)`` looks up the
  date-aligned slice (e.g. an SSP / historical multi-year run gets the
  right monthly value for *that* year).

In both cases ``ForcingData.select(date)`` descends into
``OzoneClimatology`` (it is a ``tree_math.struct``, i.e. a pytree) and
collapses the ``TimeSeries`` leaf to that step's slice, so downstream
consumers always read a plain ``(nlev, ncols)`` array.

Expected file layout (output of ``jcm/data/bc/interpolate_ozone.py``):
- ``O3``: ``(time, level, lat, lon)`` in mole/mole.
- ``level_pressure_pa``: ``(level,)`` reference pressure in Pa
  (informational; not used at run time).
- ``lat``: ``(nlat,)`` degrees north; values must match the model grid.
- ``lon``: ``(nlon,)`` degrees east; values must match the model grid.
- ``level``: ``(nlev,)`` indices; ``nlev`` must match the model.

Source files in their native plev grid (CMIP6 piControl, SSP forcing,
etc.) should be passed through ``jcm.data.bc.interpolate_ozone`` first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import tree_math


# Coordinate-match tolerance (degrees). T63 grid spacing is ~1.875°, so
# 1e-3° = ~110 m on the surface — well below any sane regridding error.
_LATLON_TOL_DEG = 1e-3


@tree_math.struct
class OzoneClimatology:
    """Per-column ozone profiles in ppmv on the model's vertical grid.

    Carried on :class:`~jcm.forcing.ForcingData` so the seasonal /
    scenario evolution rides through the same ``select(date)`` slicer
    that already drives SST, sea ice, CO2, etc. Pre-select,
    :attr:`o3_ppmv` is a :class:`~jcm.forcing.TimeSeries` of shape
    ``(ntime, nlev, ncols)`` (12 for a climatology, ``> 12`` for a
    transient file); post-select it is a plain ``jnp.ndarray`` of
    shape ``(nlev, ncols)`` for the current step.

    The empty sentinel (no climatology file loaded) stays a plain
    zero-size ``jnp.ndarray`` — never a ``TimeSeries`` — so
    :meth:`is_loaded` can structurally distinguish "no data" from
    "loaded but a tiny grid" (SCM column).
    """

    # Either a ``jnp.ndarray`` (post-select slice OR empty sentinel) or a
    # ``TimeSeries`` (pre-select monthly climatology / transient). The
    # annotation is intentionally generic — the same field name is reused
    # on both sides of ``ForcingData.select(date)`` to match how
    # ``sea_surface_temperature`` and friends behave.
    o3_ppmv: Any

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        nlon: int,
        nlat: int,
        nlev: int,
        var_name: str = "O3",
        lat_deg: np.ndarray | None = None,
        lon_deg: np.ndarray | None = None,
    ) -> "OzoneClimatology":
        """Load a pre-interpolated ozone file as a ``TimeSeries`` leaf.

        Auto-routes ``ntime`` to either ``WRAP_YEAR`` (climatology, 12
        months) or ``BY_DATE`` (transient, anything else) so an SSP /
        historical multi-year file lands on the correct year, not the
        same fraction-of-year every loop.

        Args:
            path: Path to the netCDF file produced by
                ``jcm.data.bc.interpolate_ozone``.
            nlon: Expected number of longitude points (must match file).
            nlat: Expected number of latitude points (must match file).
            nlev: Expected number of vertical levels (must match the
                model's hybrid grid; the prep script writes this many).
            var_name: Source variable name (default ``"O3"``).
            lat_deg: Optional 1-D ``(nlat,)`` model latitudes in
                degrees. When provided, the loader checks the file's
                ``lat`` values match within ``1e-3°``. Catches files
                with the right shape but flipped or shifted grids
                (descending vs ascending latitude, ``[0,360)`` vs
                ``[-180,180)`` longitude, etc.) before they silently
                wire ozone into the wrong columns.
            lon_deg: Optional 1-D ``(nlon,)`` model longitudes in
                degrees, same role as ``lat_deg``.

        Returns:
            ``OzoneClimatology`` whose ``o3_ppmv`` is a
            ``TimeSeries`` shaped ``(ntime, nlev, nlon * nlat)`` with
            longitude as the slower index, in ppmv. ``ntime == 12``
            triggers ``WRAP_YEAR`` alignment; anything else triggers
            ``BY_DATE``.

        """
        import xarray as xr
        # Local import: ``jcm.forcing`` already imports this module via
        # ``ForcingData``, so importing it at module top would cycle.
        from jcm.forcing import BY_DATE, WRAP_YEAR, make_time_series

        path = Path(path)
        # ``decode_times=False`` to read the raw values + units; we
        # decode below only if we end up in the BY_DATE branch.
        ds = xr.open_dataset(path, decode_times=False)
        if var_name not in ds.data_vars:
            raise ValueError(
                f"Ozone file {path} missing '{var_name}' variable; have "
                f"{list(ds.data_vars)}"
            )
        o3 = ds[var_name].values
        if o3.ndim != 4:
            raise ValueError(
                f"Expected '{var_name}' shape (time, level, lat, lon); "
                f"got {o3.shape}"
            )
        ntime, nlev_file, nlat_file, nlon_file = o3.shape
        if (nlev_file, nlon_file, nlat_file) != (nlev, nlon, nlat):
            raise ValueError(
                f"Ozone file grid ({nlev_file}×{nlat_file}×{nlon_file} "
                f"= level×lat×lon) does not match model "
                f"({nlev}×{nlat}×{nlon}). Re-run "
                f"``jcm.data.bc.interpolate_ozone`` against the right "
                f"vertical resolution / horizontal grid."
            )

        # Coord-value validation — shape match alone won't catch a file
        # with the same N points but flipped/shifted lat or lon (e.g.
        # descending latitude or 0..360 vs -180..180 longitude). Optional
        # because the test fixtures and SCM cases sometimes don't have
        # the model lat/lon to compare against.
        if lat_deg is not None or lon_deg is not None:
            if "lat" not in ds.coords or "lon" not in ds.coords:
                raise ValueError(
                    f"Ozone file {path} missing lat/lon coordinates; "
                    f"cannot validate grid. Either provide a file with "
                    f"both ``lat`` and ``lon`` coords or skip the check "
                    f"by omitting the model coord arrays."
                )
        if lat_deg is not None:
            file_lat = np.asarray(ds["lat"].values, dtype=float)
            if not np.allclose(file_lat, lat_deg, atol=_LATLON_TOL_DEG):
                raise ValueError(
                    f"Ozone file {path} latitudes don't match model "
                    f"grid (atol={_LATLON_TOL_DEG}°). File: "
                    f"[{file_lat[0]:.3f}..{file_lat[-1]:.3f}], "
                    f"model: [{lat_deg[0]:.3f}..{lat_deg[-1]:.3f}]. "
                    f"Re-interpolate the source file onto the model "
                    f"grid via ``jcm.data.bc.interpolate_ozone``."
                )
        if lon_deg is not None:
            file_lon = np.asarray(ds["lon"].values, dtype=float)
            if not np.allclose(file_lon, lon_deg, atol=_LATLON_TOL_DEG):
                raise ValueError(
                    f"Ozone file {path} longitudes don't match model "
                    f"grid (atol={_LATLON_TOL_DEG}°). File: "
                    f"[{file_lon[0]:.3f}..{file_lon[-1]:.3f}], "
                    f"model: [{lon_deg[0]:.3f}..{lon_deg[-1]:.3f}]. "
                    f"Re-interpolate the source file onto the model "
                    f"grid via ``jcm.data.bc.interpolate_ozone``."
                )

        # mole/mole → ppmv (consumed as ppmv by ``RRTMGPRadiation``).
        o3_ppmv_raw = o3 * 1e6
        # Reorder each timestep to (nlev, nlon, nlat) then flatten to
        # ncols matching ``ComposablePhysics._reshape_state_to_columns``
        # (3-D ``(nlev, nlon, nlat) → reshape(nlev, ncols)`` ⇒
        # lon-major, lat-minor in memory).
        o3_t = np.transpose(o3_ppmv_raw, (0, 1, 3, 2))  # (T, lev, lon, lat)
        o3_cols = o3_t.reshape(ntime, nlev, nlon * nlat)

        # Length-based routing. Anything but 12 is treated as transient
        # — ``WRAP_YEAR`` would silently sample the wrong absolute date
        # every loop for a multi-year SSP / historical file.
        if ntime == 12:
            seconds_per_month = 30.4375 * 86400.0  # 365.25/12 days
            time_seconds = jnp.asarray(
                (np.arange(ntime) + 0.5) * seconds_per_month,
                dtype=jnp.float32,
            )
            align = WRAP_YEAR
        else:
            time_seconds = _decode_time_axis_seconds(ds, path)
            align = BY_DATE

        ts = make_time_series(
            jnp.asarray(o3_cols, dtype=jnp.float32),
            time_seconds,
            align_mode=align,
        )
        return cls(o3_ppmv=ts)

    @classmethod
    def empty(cls) -> "OzoneClimatology":
        """Sentinel value used when no climatology file is provided.

        Uses a zero-size ``jnp.ndarray`` (not a ``TimeSeries``) so
        :meth:`is_loaded` can distinguish the sentinel from a
        legitimately-loaded single-column climatology (e.g. an SCM run
        with ``nlon == nlat == 1``). Callers can check
        :meth:`is_loaded` to decide whether to use this forcing or fall
        back to an analytical profile.
        """
        return cls(o3_ppmv=jnp.zeros((0, 0), dtype=jnp.float32))

    def is_loaded(self) -> bool:
        """Cheap Python-side check that the climatology has real data.

        Works at both stages of the forcing pipeline:
        - Pre-select, ``o3_ppmv`` is a ``TimeSeries`` whose ``.values``
          carries the data.
        - Post-select (and for the empty sentinel), ``o3_ppmv`` is a
          plain ``jnp.ndarray``.

        Both expose ``.size``.
        """
        arr = getattr(self.o3_ppmv, "values", self.o3_ppmv)
        return bool(arr.size > 0)


def _decode_time_axis_seconds(ds, path: Path) -> jnp.ndarray:
    """Decode a transient ozone file's time axis to seconds since 1970.

    Mirrors ``jcm.forcing._time_axis_seconds_from_ds`` but works on the
    raw ``(values, units)`` pair (we opened with ``decode_times=False``
    above so the climatology branch could keep month indices as plain
    integers).
    """
    import pandas as pd
    import xarray as xr

    # Re-decode just the time coord. ``xr.decode_cf`` on the whole
    # dataset would also try to re-encode masked O3 values etc., which
    # is unnecessary here.
    if "time" not in ds.coords:
        raise ValueError(
            f"Transient ozone file {path} has ntime>{12} but no decodable "
            f"``time`` coordinate."
        )
    time_da = xr.decode_cf(ds[["time"]])["time"]
    times = pd.DatetimeIndex(np.asarray(time_da.values))
    epoch = pd.Timestamp("1970-01-01")
    delta_s = (times - epoch).total_seconds().to_numpy()
    return jnp.asarray(delta_s, dtype=jnp.float32)
