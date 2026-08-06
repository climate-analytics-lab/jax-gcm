"""Subgrid-orography (SSO) statistics from the GMTED2010 30″ DEM.

Computes, per target grid cell, the seven ECHAM terrain fields:

* ``orog``   — mean elevation [m] (ocean pixels enter as 0)
* ``orostd`` — standard deviation of elevation [m]
* ``orosig`` — mean-slope parameter σ (sqrt of the largest eigenvalue of
  the Lott & Miller 1997 gradient tensor)
* ``orogam`` — anisotropy γ ∈ [0, 1]
* ``orothe`` — principal-axis angle θ [degrees, ECHAM convention]
* ``oropic`` — peak (max) elevation [m]
* ``oroval`` — valley (min) elevation [m]

``lsm`` here is a placeholder from DEM validity — GMTED stores oceans
as elevation 0, so the real land fraction comes from the ERA5
invariant land-sea mask in the bundle assembly, which also provides
the ocean mask applied to the SSO fields there.

The DEM is streamed in latitude strips; per strip the pixels (and the
spherical-metric gradients computed inside the strip) are binned onto the
target cells, accumulating the sufficient statistics (Σh, Σh², Σhx²,
Σhy², Σhx·hy, max, min, counts). This keeps memory at one strip and makes
the pass exact rather than a two-level coarsening.
"""

from __future__ import annotations

import numpy as np
import rasterio

R_EARTH = 6.371e6
_STRIP = 240                      # DEM rows per strip (2° at 30")


def _accumulate(dem_path: str, lat_edges: np.ndarray,
                assign_lon, nlon_bins: int):
    """One pass over the DEM accumulating per-cell sufficient statistics.

    ``lat_edges`` are target latitude bin edges (ascending, degrees).
    ``assign_lon(lon_deg) -> int bin`` vectorised longitude binning.
    """
    nlat_bins = lat_edges.size - 1
    shape = (nlat_bins, nlon_bins)
    acc = {k: np.zeros(shape) for k in
           ("n", "land", "sh", "sh2", "shx2", "shy2", "shxy")}
    acc["pic"] = np.full(shape, -np.inf)
    acc["val"] = np.full(shape, np.inf)

    with rasterio.open(dem_path) as src:
        nodata = src.nodata if src.nodata is not None else -32768
        t = src.transform
        dlam = np.deg2rad(t.a)                      # pixel size, radians
        for row0 in range(0, src.height, _STRIP):
            nrows = min(_STRIP, src.height - row0)
            # one extra row below for the meridional gradient
            read_rows = min(nrows + 1, src.height - row0)
            h = src.read(1, window=((row0, row0 + read_rows),
                                    (0, src.width))).astype(np.float64)
            land = h != nodata
            h = np.where(land, h, 0.0)

            lats = t.f + t.e * (row0 + 0.5 + np.arange(read_rows))
            lons = t.c + t.a * (0.5 + np.arange(src.width))

            # gradients on the strip (spherical metric); last strip-row
            # gradient reuses the previous row's to keep shapes aligned.
            coslat = np.cos(np.deg2rad(lats))[:nrows, None]
            hx = np.empty((nrows, src.width))
            hx[:, :-1] = (h[:nrows, 1:] - h[:nrows, :-1])
            hx[:, -1] = h[:nrows, 0] - h[:nrows, -1]      # periodic in lon
            hx /= (R_EARTH * np.maximum(coslat, 0.05) * dlam)
            hy = np.empty((nrows, src.width))
            if read_rows > nrows:
                hy[:] = (h[1:nrows + 1] - h[:nrows])
            else:
                hy[:-1] = h[1:nrows] - h[:nrows - 1]
                hy[-1] = hy[-2] if nrows > 1 else 0.0
            hy /= (R_EARTH * dlam)

            h = h[:nrows]
            land = land[:nrows]
            lat_bin = np.searchsorted(lat_edges, lats[:nrows]) - 1
            ok_rows = (lat_bin >= 0) & (lat_bin < nlat_bins)
            lon_bin = assign_lon(lons)

            for r in np.nonzero(ok_rows)[0]:
                li = lat_bin[r]
                flat = li * nlon_bins + lon_bin
                n = np.bincount(flat, minlength=nlat_bins * nlon_bins)
                acc["n"] += n.reshape(shape)
                for key, arr in (("land", land[r].astype(float)),
                                 ("sh", h[r]), ("sh2", h[r] ** 2),
                                 ("shx2", hx[r] ** 2), ("shy2", hy[r] ** 2),
                                 ("shxy", hx[r] * hy[r])):
                    acc[key] += np.bincount(
                        flat, weights=arr,
                        minlength=nlat_bins * nlon_bins).reshape(shape)
                np.maximum.at(acc["pic"].ravel(), flat, h[r])
                np.minimum.at(acc["val"].ravel(), flat, h[r])
    return acc


def finalize(acc) -> dict[str, np.ndarray]:
    """Sufficient statistics -> the seven SSO fields + land fraction."""
    n = np.maximum(acc["n"], 1.0)
    mean = acc["sh"] / n
    var = np.maximum(acc["sh2"] / n - mean ** 2, 0.0)
    K = 0.5 * (acc["shx2"] + acc["shy2"]) / n
    L = 0.5 * (acc["shx2"] - acc["shy2"]) / n
    M = acc["shxy"] / n
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
                      lons: np.ndarray) -> dict[str, np.ndarray]:
    """SSO fields on a Gaussian (lat, lon) grid (lats ascending, degrees)."""
    lat_edges = np.concatenate([[-90.0],
                                0.5 * (lats[1:] + lats[:-1]), [90.0]])
    dlon = 360.0 / lons.size
    lon0 = lons[0] - dlon / 2.0

    def assign_lon(lon_deg):
        return ((lon_deg - lon0) // dlon).astype(int) % lons.size

    acc = _accumulate(dem_path, lat_edges, assign_lon, lons.size)
    return finalize(acc)
