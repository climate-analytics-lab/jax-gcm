"""General, mass-conserving regridding of emission fields onto the model grid.

Emissions reach jax-gcm on whatever grid their source happens to use — a regular
0.5° lat/lon CEDS product, or an unstructured ``ncol`` mesh like the CESM ne30
files in ``inputdata/atm/cam/chem/emis/cmip7``. This module remaps any such
source onto the model's spectral Gaussian nodal grid **conservatively**, so the
global emitted mass is (to first order) preserved — the property that matters for
a flux boundary condition.

Design (kept deliberately light — no ESMF/xesmf dependency):

* The regridder is **first-order conservative / area-weighted**. Each source cell
  is assigned, by nearest cell centre, to exactly one target cell; the target
  value is the source-area-weighted mean of the cells that land in it
  (``Σ fₛ Aₛ / Σ Aₛ``). For *coarsening* (fine source → coarse model grid, the
  usual case here) many source cells fall in each target cell and this converges
  to the true cell-mean flux; the area-weighting makes the column-integrated mass
  conservative to the binning accuracy. It does **not** smooth or do higher-order
  reconstruction, and it is not intended for *refinement* (coarse → fine), which
  would alias.
* It is a pure host-side (numpy/scipy) tool used offline by
  :mod:`jcm.data.emissions.prepare`, not inside the JIT'd model.

The source grid is described purely by per-cell ``(lon, lat, area)`` triples, so
structured and unstructured sources go through the same path — only the adapter
that produces those triples differs.

Note — this is intentionally *not* the same remapper as
``jcm.data.bc.interpolate`` (``upsample_forcings_ds``): that path **bilinearly
interpolates** smooth structured lat/lon boundary fields (SST, soil moisture, …)
to a higher spectral resolution and is *not* mass-conserving, whereas emissions
are **flux** fields where conserving the column-integrated source is the property
that matters, and their sources are often unstructured (``ncol``). Different
problems (non-conservative bilinear upsampling vs conservative,
unstructured-capable flux remap), hence kept separate.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Regridder:
    """A conservative source→target remap operator, reusable across fields.

    Built once from the source/target geometry, then applied to any field on the
    same source grid (e.g. every month/level of an emission time series) via
    :meth:`__call__`. Internally a sparse ``(n_target, n_source)`` matrix whose
    rows are the source-area weights of the cells assigned to each target cell.
    """

    def __init__(self, matrix: sp.csr_matrix, target_shape: tuple[int, int],
                 covered_area: np.ndarray):
        """Hold the prebuilt remap matrix, target shape, and area normaliser."""
        self._matrix = matrix              # (n_target, n_source), data = src area
        self._target_shape = target_shape  # (nlon, nlat)
        # Σ source-area landing in each target cell; the normaliser that turns
        # accumulated mass back into an (area-weighted mean) flux.
        self._covered_area = covered_area  # (n_target,)

    @property
    def target_shape(self) -> tuple[int, int]:
        return self._target_shape

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Regrid ``values`` shaped ``(..., n_source)`` → ``(..., nlon, nlat)``.

        Leading axes (time, level, …) are preserved. Target cells that received
        no source cell (only possible when *refining*) come back as zero.
        """
        values = np.asarray(values, dtype=np.float64)
        lead = values.shape[:-1]
        flat = values.reshape(-1, values.shape[-1])          # (K, n_source)
        mass = flat @ self._matrix.T                          # (K, n_target)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self._covered_area > 0.0,
                            mass / self._covered_area, 0.0)
        nlon, nlat = self._target_shape
        return mean.reshape(*lead, nlon, nlat)


def _nearest_lon_index(src_lon: np.ndarray, dst_lon: np.ndarray) -> np.ndarray:
    """Nearest target longitude index under periodic (wrap-around) distance."""
    # (n_src, nlon) circular separation in [0, π]; argmin over targets.
    d = np.abs(src_lon[:, None] - dst_lon[None, :])
    d = np.minimum(d, 2.0 * np.pi - d)
    return np.argmin(d, axis=1)


def _nearest_lat_index(src_lat: np.ndarray, dst_lat: np.ndarray) -> np.ndarray:
    """Nearest target latitude index (latitudes are not periodic)."""
    return np.argmin(np.abs(src_lat[:, None] - dst_lat[None, :]), axis=1)


def build_regridder(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    src_area: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
    *,
    src_in_degrees: bool = True,
    dst_in_degrees: bool = False,
) -> Regridder:
    """Build a conservative regridder from source points to a target lon×lat grid.

    Args:
        src_lon, src_lat: 1-D source cell-centre coordinates, length ``n_source``
            (e.g. the flattened lat/lon mesh, or the ``ncol`` arrays of an
            unstructured file).
        src_area: 1-D per-source-cell area weight (any consistent units — only
            ratios matter), length ``n_source``.
        dst_lon, dst_lat: 1-D target grid coordinates (the model's
            ``horizontal.longitudes`` / ``.latitudes``), lengths ``nlon`` /
            ``nlat``. The target is the tensor-product grid ``(nlon, nlat)``.
        src_in_degrees, dst_in_degrees: unit flags; coordinates are converted to
            radians internally (netCDF lon/lat are degrees; the dinosaur grid is
            radians).

    Returns:
        A :class:`Regridder` mapping ``(..., n_source)`` arrays to
        ``(..., nlon, nlat)``.

    """
    to_rad = np.deg2rad
    sl = to_rad(src_lon) if src_in_degrees else np.asarray(src_lon, float)
    sb = to_rad(src_lat) if src_in_degrees else np.asarray(src_lat, float)
    dl = to_rad(dst_lon) if dst_in_degrees else np.asarray(dst_lon, float)
    db = to_rad(dst_lat) if dst_in_degrees else np.asarray(dst_lat, float)
    sl = np.mod(sl, 2.0 * np.pi)
    dl = np.mod(dl, 2.0 * np.pi)

    area = np.asarray(src_area, dtype=np.float64).ravel()
    n_src = area.size
    nlon, nlat = dl.size, db.size

    i_lon = _nearest_lon_index(sl, dl)
    i_lat = _nearest_lat_index(sb, db)
    # Row-major (lon, lat) flattening — matches numpy reshape((nlon, nlat)).
    target_idx = i_lon * nlat + i_lat

    matrix = sp.coo_matrix(
        (area, (target_idx, np.arange(n_src))),
        shape=(nlon * nlat, n_src),
    ).tocsr()
    covered_area = np.asarray(matrix.sum(axis=1)).ravel()
    return Regridder(matrix, (nlon, nlat), covered_area)


def model_grid(coords) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target ``(lon_rad, lat_rad, cell_solid_angle)`` for a model CoordinateSystem.

    ``cell_solid_angle`` is the per-cell quadrature weight (steradians, summing to
    4π) — handy as the target area for conservation diagnostics.
    """
    h = coords.horizontal
    lon = np.asarray(h.longitudes)
    lat = np.asarray(h.latitudes)
    area = np.asarray(h.quadrature_weights)  # (nlon, nlat)
    return lon, lat, area
