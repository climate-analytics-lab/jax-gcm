"""Subgrid-orography (SSO) statistics from the GMTED2010 30″ DEM.

Computes, per target grid cell, the seven ECHAM terrain fields:

* ``orog``   — mean elevation [m] (ocean pixels enter as 0)
* ``orostd`` — standard deviation of elevation [m]
* ``orosig`` — mean-slope parameter σ (sqrt of the largest eigenvalue of
  the Lott & Miller 1997 gradient tensor)
* ``orogam`` — anisotropy γ ∈ [0, 1]
* ``orothe`` — principal-axis angle θ [degrees, ECHAM convention].
  The DEM is north-up, so the strip differences give the *southward*
  height gradient and M = ⟨hx·hy⟩ carries the opposite sign to the
  textbook northward-y Lott-Miller tensor — deliberately: ECHAM's own
  boundary files (validated against T127GR15_jan_surf) use this sign,
  and the ported ssodrag consumer was tuned against them. Do not
  "fix" the sign of hy.
* ``oropic`` — peak (max) elevation [m]
* ``oroval`` — valley (min) elevation [m]

``lsm`` here is a placeholder from DEM validity — GMTED stores oceans
as elevation 0, so the real land fraction comes from the ERA5
invariant land-sea mask in the bundle assembly, which also provides
the ocean mask applied to the SSO fields there.

The DEM is streamed in latitude strips; per strip the pixels (and the
spherical-metric gradients computed inside the strip) are assigned to
target cells — regular (lat, lon) bins for Gaussian grids, nearest
column via a unit-sphere KDTree for unstructured grids (ne30pg3) — and
the sufficient statistics (Σh, Σh², Σhx², Σhy², Σhx·hy, max, min,
counts) accumulate into flat per-cell arrays. This keeps memory at one
strip and makes the pass exact rather than a two-level coarsening.
"""

from __future__ import annotations

import numpy as np

R_EARTH = 6.371e6
_STRIP = 240                      # DEM rows per strip (2° at 30")

_SUM_KEYS = ("land", "sh", "sh2", "shx2", "shy2", "shxy")


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over f×f blocks (array dims must be divisible by f)."""
    ny, nx = a.shape
    return a.reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def _accumulate(dem_path: str, assign_strip, ncells: int,
                grad_coarsen: int = 20):
    """One pass over the DEM accumulating flat per-cell sufficient statistics.

    ``assign_strip(lats, lons) -> (nrows, nlon) int`` maps each pixel to a
    flat target-cell index; negative means "not on the target grid".

    Elevation statistics (mean/std/peak/valley) use the native 30″ pixels.
    The gradient tensor is computed on ``grad_coarsen``-pixel block means
    (default 20 → 10′ ≈ 18 km): SSO drag was tuned against ECHAM boundary
    files whose slopes derive from ~10′ topography, and slope magnitude
    scales strongly with the differencing resolution — native 30″
    gradients come out ~5× larger than the reference (validated against
    T127GR15_jan_surf).
    """
    import rasterio          # lazy: only the Glade-side builder needs GDAL

    acc = {k: np.zeros(ncells) for k in ("n", "ng") + _SUM_KEYS}
    acc["pic"] = np.full(ncells, -np.inf)
    acc["val"] = np.full(ncells, np.inf)
    f = grad_coarsen

    with rasterio.open(dem_path) as src:
        nodata = src.nodata if src.nodata is not None else -32768
        t = src.transform
        dlam = np.deg2rad(t.a)                      # pixel size, radians
        assert _STRIP % f == 0 and src.width % f == 0
        for row0 in range(0, src.height, _STRIP):
            nrows = min(_STRIP, src.height - row0)
            # one extra coarse block below for the meridional gradient
            read_rows = min(nrows + f, src.height - row0)
            h = src.read(1, window=((row0, row0 + read_rows),
                                    (0, src.width))).astype(np.float64)
            land = h != nodata
            h = np.where(land, h, 0.0)

            lats = t.f + t.e * (row0 + 0.5 + np.arange(read_rows))
            lons = t.c + t.a * (0.5 + np.arange(src.width))

            # -- gradient tensor on block-mean topography ----------------
            nc_rows = nrows // f
            full = (read_rows // f) * f
            hc = _block_mean(h[:full], f)
            c_lats = lats[:full].reshape(-1, f).mean(1)
            coslat_c = np.cos(np.deg2rad(c_lats[:nc_rows]))[:, None]
            dlam_c = dlam * f
            hx = np.empty((nc_rows, hc.shape[1]))
            hx[:, :-1] = hc[:nc_rows, 1:] - hc[:nc_rows, :-1]
            hx[:, -1] = hc[:nc_rows, 0] - hc[:nc_rows, -1]   # periodic
            hx /= (R_EARTH * np.maximum(coslat_c, 0.05) * dlam_c)
            hy = np.empty_like(hx)
            if hc.shape[0] > nc_rows:
                hy[:] = hc[1:nc_rows + 1] - hc[:nc_rows]
            else:
                hy[:-1] = hc[1:nc_rows] - hc[:nc_rows - 1]
                hy[-1] = hy[-2] if nc_rows > 1 else 0.0
            hy /= (R_EARTH * dlam_c)

            idx_c = assign_strip(c_lats[:nc_rows], lons.reshape(-1, f).mean(1))
            valid_c = idx_c >= 0
            flat_c = idx_c[valid_c]
            acc["ng"] += np.bincount(flat_c, minlength=ncells)
            for key, arr in (("shx2", hx ** 2), ("shy2", hy ** 2),
                             ("shxy", hx * hy)):
                acc[key] += np.bincount(flat_c, weights=arr[valid_c],
                                        minlength=ncells)

            # -- elevation statistics on native pixels -------------------
            h = h[:nrows]
            land = land[:nrows]
            idx = assign_strip(lats[:nrows], lons)
            valid = idx >= 0
            flat = idx[valid]

            acc["n"] += np.bincount(flat, minlength=ncells)
            for key, arr in (("land", land.astype(float)), ("sh", h),
                             ("sh2", h ** 2)):
                acc[key] += np.bincount(flat, weights=arr[valid],
                                        minlength=ncells)
            np.maximum.at(acc["pic"], flat, h[valid])
            np.minimum.at(acc["val"], flat, h[valid])
    return acc


def finalize(acc) -> dict[str, np.ndarray]:
    """Sufficient statistics -> the seven SSO fields + land fraction."""
    n = np.maximum(acc["n"], 1.0)
    ng = np.maximum(acc.get("ng", acc["n"]), 1.0)
    mean = acc["sh"] / n
    var = np.maximum(acc["sh2"] / n - mean ** 2, 0.0)
    K = 0.5 * (acc["shx2"] + acc["shy2"]) / ng
    L = 0.5 * (acc["shx2"] - acc["shy2"]) / ng
    M = acc["shxy"] / ng
    lm = np.sqrt(L ** 2 + M ** 2)
    lsm = acc["land"] / n
    ocean = lsm < 0.5
    out = {
        "orog": np.where(ocean, 0.0, mean),
        "orostd": np.where(ocean, 0.0, np.sqrt(var)),
        "orosig": np.where(ocean, 0.0, np.sqrt(np.maximum(K + lm, 0.0))),
        "orogam": np.where(
            ocean, 0.0,
            np.sqrt(np.maximum(K - lm, 0.0) / np.maximum(K + lm, 1e-30))),
        "orothe": np.where(ocean, 0.0,
                           np.rad2deg(0.5 * np.arctan2(M, L))),
        "oropic": np.where(ocean, 0.0, np.where(np.isfinite(acc["pic"]),
                                                acc["pic"], 0.0)),
        "oroval": np.where(ocean, 0.0, np.where(np.isfinite(acc["val"]),
                                                acc["val"], 0.0)),
        "lsm": lsm,
    }
    return out


def gaussian_grid_sso(dem_path: str, lats: np.ndarray,
                      lons: np.ndarray,
                      grad_coarsen: int = 20) -> dict[str, np.ndarray]:
    """SSO fields on a Gaussian (lat, lon) grid (lats ascending, degrees)."""
    lat_edges = np.concatenate([[-90.0],
                                0.5 * (lats[1:] + lats[:-1]), [90.0]])
    nlat, nlon = lats.size, lons.size
    dlon = 360.0 / nlon
    lon0 = lons[0] - dlon / 2.0

    def assign_strip(strip_lats, strip_lons):
        lat_bin = np.searchsorted(lat_edges, strip_lats) - 1
        lat_bin = np.where((lat_bin >= 0) & (lat_bin < nlat), lat_bin, -1)
        lon_bin = ((strip_lons - lon0) // dlon).astype(int) % nlon
        return np.where(lat_bin[:, None] >= 0,
                        lat_bin[:, None] * nlon + lon_bin[None, :], -1)

    acc = _accumulate(dem_path, assign_strip, nlat * nlon,
                      grad_coarsen=grad_coarsen)
    return {k: v.reshape(nlat, nlon) for k, v in finalize(acc).items()}


def column_grid_sso(dem_path: str, col_lats: np.ndarray,
                    col_lons: np.ndarray,
                    grad_coarsen: int = 20) -> dict[str, np.ndarray]:
    """SSO fields on an unstructured column grid (e.g. ne30pg3).

    Each DEM pixel goes to the nearest column center by great-circle
    distance (KDTree on unit-sphere vectors) — the Voronoi partition of
    the columns, which is exact and gap-free without needing cell bounds.
    """
    from scipy.spatial import cKDTree

    from jcm.data.regridding import unit_sphere_vectors
    tree = cKDTree(unit_sphere_vectors(col_lats, col_lons))

    def assign_strip(strip_lats, strip_lons):
        pts = unit_sphere_vectors(
            np.repeat(strip_lats, strip_lons.size),
            np.tile(strip_lons, strip_lats.size))
        _, idx = tree.query(pts, workers=-1)
        return idx.reshape(strip_lats.size, strip_lons.size)

    acc = _accumulate(dem_path, assign_strip, col_lats.size,
                      grad_coarsen=grad_coarsen)
    return finalize(acc)
