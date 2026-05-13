"""Ozone climatology forcing for the ECHAM-RRTMGP pipeline.

Reads a *pre-interpolated* ozone netCDF (``(time=12, level=nlev,
lat, lon)`` mole/mole — the format produced by
``jcm.data.bc.interpolate_ozone``) and exposes a per-column profile
that ``EchamBoundaryConditions`` hands straight to RRTMGP. The vertical
interpolation from the source file's plev grid onto the model's
hybrid-level centers happens **offline** in the prep script so the
online code is just an array slice — no per-step ``vmap`` of
``jnp.interp``.

Phase 1 returns the **annual mean** so the field is static across the
integration. The class still loads and stores all 12 months internally,
so adding a date-aware monthly selector later — for the seasonal cycle,
then for transient SSP scenarios where the profile changes year over
year — is a one-line change in the public accessor.

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

import jax.numpy as jnp
import numpy as np
import tree_math


@tree_math.struct
class OzoneClimatology:
    """Per-column ozone profiles in ppmv on the model's vertical grid.

    Carried on :class:`~jcm.forcing.ForcingData` so the seasonal /
    scenario evolution can ride through the same ``select(date)`` slicer
    that already drives SST, sea ice, CO2, etc. (Phase 1 holds the
    annual mean directly; the slicing hook lands when the seasonal
    cycle is wired in.)
    """

    # Annual-mean O3 (ppmv) per column on the model's hybrid-level grid.
    # Shape: ``(nlev, ncols)`` matching the column convention
    # (lon-major / lat-minor; see
    # :func:`jcm.physics.composable_physics._reshape_state_to_columns`).
    o3_ppmv: jnp.ndarray

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        nlon: int,
        nlat: int,
        nlev: int,
        var_name: str = "O3",
    ) -> "OzoneClimatology":
        """Load a pre-interpolated ozone climatology and annual-mean it.

        Args:
            path: Path to the netCDF file produced by
                ``jcm.data.bc.interpolate_ozone``.
            nlon: Expected number of longitude points (must match file).
            nlat: Expected number of latitude points (must match file).
            nlev: Expected number of vertical levels (must match the
                model's hybrid grid; the prep script writes this many).
            var_name: Source variable name (default ``"O3"``).

        Returns:
            ``OzoneClimatology`` whose ``o3_ppmv`` is the annual mean,
            flattened to the model's column ordering ``(nlev, nlon * nlat)``
            with longitude as the slower index.
        """
        import xarray as xr

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
        # Annual mean across the 12 months (Phase 1; future date-aware
        # slicing keeps the time axis).
        o3_annual = o3_ppmv_raw.mean(axis=0)  # (nlev, nlat, nlon)
        # Reorder to (nlev, nlon, nlat) then flatten to ncols, matching
        # ``ComposablePhysics._reshape_state_to_columns`` (3-D
        # ``(nlev, nlon, nlat) → reshape(nlev, ncols)`` ⇒ lon-major,
        # lat-minor in memory).
        o3_annual = np.transpose(o3_annual, (0, 2, 1))
        o3_cols = o3_annual.reshape(nlev, nlon * nlat)

        return cls(o3_ppmv=jnp.asarray(o3_cols, dtype=jnp.float32))

    @classmethod
    def empty(cls) -> "OzoneClimatology":
        """Sentinel value used when no climatology file is provided.

        Callers can check ``is_loaded()`` to decide whether to use this
        forcing or fall back to an analytical profile.
        """
        return cls(o3_ppmv=jnp.zeros((1, 1), dtype=jnp.float32))

    def is_loaded(self) -> bool:
        """Cheap Python-side check that the climatology has real data."""
        return bool(self.o3_ppmv.shape[1] > 1)
