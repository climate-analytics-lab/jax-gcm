"""Ozone climatology forcing for the ECHAM-RRTMGP pipeline.

Reads a *pre-interpolated* ozone netCDF (``(time=12, level=nlev,
lat, lon)`` mole/mole — the format produced by
``jcm.data.bc.interpolate_ozone``) and exposes a per-column profile
that ``EchamBoundaryConditions`` hands straight to RRTMGP. The vertical
interpolation from the source file's plev grid onto the model's
hybrid-level centers happens **offline** in the prep script so the
online code is just an array slice — no per-step ``vmap`` of
``jnp.interp``.

The 12-month seasonal cycle is preserved by wrapping ``o3_ppmv`` in a
:class:`~jcm.forcing.TimeSeries` with ``align_mode=WRAP_YEAR``. The
existing ``ForcingData.select(date)`` walker descends into this struct
(it is a ``tree_math.struct``, i.e. a pytree) and replaces the
``TimeSeries`` leaf with that step's monthly slice. Downstream consumers
(``EchamBoundaryConditions``) therefore always read a single
``(nlev, ncols)`` array — they don't need to know about the time axis.

Expected file layout (output of ``jcm/data/bc/interpolate_ozone.py``):
- ``O3``: ``(time, level, lat, lon)`` in mole/mole.
- ``level_pressure_pa``: ``(level,)`` reference pressure in Pa
  (informational; not used at run time).
- ``lat``: ``(nlat,)`` degrees north; must match the model grid.
- ``lon``: ``(nlon,)`` degrees east; must match the model grid.
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


@tree_math.struct
class OzoneClimatology:
    """Per-column ozone profiles in ppmv on the model's vertical grid.

    Carried on :class:`~jcm.forcing.ForcingData` so the seasonal /
    scenario evolution rides through the same ``select(date)`` slicer
    that already drives SST, sea ice, CO2, etc. Pre-select,
    :attr:`o3_ppmv` is a :class:`~jcm.forcing.TimeSeries` of shape
    ``(12, nlev, ncols)`` (monthly climatology, ``WRAP_YEAR`` align
    mode); post-select it is a plain ``jnp.ndarray`` of shape
    ``(nlev, ncols)`` for the current step.

    The empty sentinel (no climatology file loaded) stays a plain
    zero-size ``jnp.ndarray`` — never a ``TimeSeries`` — so
    :meth:`is_loaded` can structurally distinguish "no data" from
    "loaded but a tiny grid" (SCM column).
    """

    # Either a ``jnp.ndarray`` (post-select slice OR empty sentinel) or a
    # ``TimeSeries`` (pre-select monthly climatology). The annotation is
    # intentionally generic — the same field name is reused on both
    # sides of ``ForcingData.select(date)`` to match how
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
    ) -> "OzoneClimatology":
        """Load a pre-interpolated 12-month ozone climatology.

        Wraps the full ``(12, nlev, ncols)`` array in a
        :class:`~jcm.forcing.TimeSeries` so the seasonal cycle is
        preserved through ``ForcingData.select(date)``.

        Args:
            path: Path to the netCDF file produced by
                ``jcm.data.bc.interpolate_ozone``.
            nlon: Expected number of longitude points (must match file).
            nlat: Expected number of latitude points (must match file).
            nlev: Expected number of vertical levels (must match the
                model's hybrid grid; the prep script writes this many).
            var_name: Source variable name (default ``"O3"``).

        Returns:
            ``OzoneClimatology`` whose ``o3_ppmv`` is a 12-month
            ``TimeSeries`` shaped ``(12, nlev, nlon * nlat)`` with
            longitude as the slower index, in ppmv.

        """
        import xarray as xr
        # Local import: ``jcm.forcing`` already imports this module via
        # ``ForcingData``, so importing it at module top would cycle.
        from jcm.forcing import WRAP_YEAR, make_time_series

        path = Path(path)
        # ``decode_times=False`` — the source CMIP6 file uses
        # ``months since 1850-1-1`` units the prep script preserves.
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

        # mole/mole → ppmv (consumed as ppmv by ``RRTMGPRadiation``).
        o3_ppmv_raw = o3 * 1e6
        # Reorder each month to (nlev, nlon, nlat) then flatten to ncols
        # matching ``ComposablePhysics._reshape_state_to_columns`` (3-D
        # ``(nlev, nlon, nlat) → reshape(nlev, ncols)`` ⇒ lon-major,
        # lat-minor in memory).
        o3_per_month = np.transpose(o3_ppmv_raw, (0, 1, 3, 2))  # (T, lev, lon, lat)
        o3_per_month_cols = o3_per_month.reshape(ntime, nlev, nlon * nlat)

        # ``time_seconds`` is structurally required by ``TimeSeries`` but
        # ignored under ``WRAP_YEAR`` selection. Use month-center seconds
        # within a reference year so the array is well-formed for any
        # ``by_date`` consumer that might appear later.
        seconds_per_month = 30.4375 * 86400.0  # 365.25/12 days
        time_seconds = jnp.asarray(
            (np.arange(ntime) + 0.5) * seconds_per_month, dtype=jnp.float32,
        )

        ts = make_time_series(
            jnp.asarray(o3_per_month_cols, dtype=jnp.float32),
            time_seconds,
            align_mode=WRAP_YEAR,
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
