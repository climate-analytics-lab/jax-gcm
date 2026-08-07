"""Grid helpers shared by the mirror builders (pure math, no Glade I/O)."""

from __future__ import annotations

import numpy as np
import xarray as xr


def gaussian_latlon(nlat: int):
    lats = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))
    return lats, np.arange(2 * nlat) * 360.0 / (2 * nlat)


def interp_to(da: xr.DataArray, lats, lons) -> xr.DataArray:
    """Bilinear regrid with periodic longitude wrap; lat clamped at ends."""
    latn, lonn = da.dims[-2], da.dims[-1]
    if float(da[latn][0]) > float(da[latn][-1]):
        da = da.isel({latn: slice(None, None, -1)})
    dlon = float(da[lonn][1] - da[lonn][0])
    wrapped = xr.concat(
        [da.isel({lonn: -1}).assign_coords(
            {lonn: float(da[lonn][0]) - dlon}),
         da,
         da.isel({lonn: 0}).assign_coords(
             {lonn: float(da[lonn][-1]) + dlon})], dim=lonn)
    # constant extension to the poles so Gaussian lats beyond the source's
    # first/last row interpolate instead of going NaN
    if float(wrapped[latn][0]) > -90.0:
        wrapped = xr.concat(
            [wrapped.isel({latn: 0}).assign_coords({latn: -90.0}), wrapped],
            dim=latn)
    if float(wrapped[latn][-1]) < 90.0:
        wrapped = xr.concat(
            [wrapped, wrapped.isel({latn: -1}).assign_coords({latn: 90.0})],
            dim=latn)
    out = wrapped.interp({latn: lats, lonn: lons}, method="linear")
    return out.rename({latn: "lat", lonn: "lon"})


def fill_nearest(field: np.ndarray, lats, lons) -> np.ndarray:
    """Fill NaNs (land in ocean products) with the nearest valid value."""
    from scipy.spatial import cKDTree

    glon, glat = np.meshgrid(np.deg2rad(lons), np.deg2rad(lats))
    xyz = np.stack([np.cos(glat) * np.cos(glon),
                    np.cos(glat) * np.sin(glon), np.sin(glat)], -1)
    out = field.copy()
    for t in range(field.shape[0]):
        bad = ~np.isfinite(field[t])
        if not bad.any():
            continue
        tree = cKDTree(xyz[~bad])
        _, idx = tree.query(xyz[bad], workers=-1)
        out[t][bad] = field[t][~bad][idx]
    return out


def conservative_to_gaussian(field: np.ndarray, src_lats, src_lons,
                             lats, lons) -> np.ndarray:
    """Area-weighted binning of a regular-grid flux onto Gaussian cells.

    Each source cell contributes its cos(lat)-weighted value to the target
    cell containing its center — conservative in the flux-density sense,
    which is what per-m² emissions need (bilinear would smear point
    sources and lose mass).
    """
    lat_edges = np.concatenate([[-90.0], 0.5 * (lats[1:] + lats[:-1]),
                                [90.0]])
    nlat, nlon = lats.size, lons.size
    dlon = 360.0 / nlon
    lat_bin = np.clip(np.searchsorted(lat_edges, src_lats) - 1, 0, nlat - 1)
    lon_bin = ((np.asarray(src_lons) - (lons[0] - dlon / 2)) // dlon
               ).astype(int) % nlon
    flat = (lat_bin[:, None] * nlon + lon_bin[None, :]).ravel()
    w = np.cos(np.deg2rad(src_lats))[:, None].repeat(len(src_lons), 1).ravel()
    wsum = np.bincount(flat, weights=w, minlength=nlat * nlon)
    lead = field.shape[:-2]
    out = np.empty(lead + (nlat, nlon))
    for idx in np.ndindex(lead):
        num = np.bincount(flat, weights=w * field[idx].ravel(),
                          minlength=nlat * nlon)
        out[idx] = (num / np.maximum(wsum, 1e-30)).reshape(nlat, nlon)
    return out
