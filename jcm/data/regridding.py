"""Host-side regridding shared by the data preparation and mirror pipelines.

One module for every offline remap in jcm — nothing here runs inside the
JIT'd model:

* **Conservative flux remap** (:class:`Regridder` / :func:`build_regridder`
  / :func:`conservative_to_gaussian`): first-order area-weighted binning by
  nearest cell centre, ``Σ fₛ Aₛ / Σ Aₛ`` per target cell. Mass-conserving
  to binning accuracy — what emission fluxes need. Handles structured and
  unstructured (``ncol``) sources via per-cell ``(lon, lat, area)`` triples.
  Coarsening only; refinement would alias.
* **Bilinear sampling** (:func:`interp_to`): periodic-longitude wrap and
  constant pole extension for smooth climatology fields (SST, soil, ozone).
  Not conservative — do not use it for fluxes.
* **Sphere geometry** (:func:`unit_sphere_vectors`, :func:`nearest_index`,
  :func:`fill_nearest`, :func:`gaussian_latlon`): the shared unit-vector /
  KDTree machinery for nearest-neighbour matching on the sphere.

Note — the *runtime* boundary-condition upsampler
(``jcm.data.bc.interpolate.upsample_forcings_ds``) stays separate: it
bilinearly refines packaged forcing files to higher spectral resolutions at
model start with its own pole-averaging conventions, and changing it would
change existing runs.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import xarray as xr


class Regridder:
    """A conservative source→target remap operator, reusable across fields.

    Built once from the source/target geometry, then applied to any field on the
    same source grid (e.g. every month/level of an emission time series) via
    :meth:`__call__`. Internally a sparse ``(n_target, n_source)`` matrix whose
    rows are the source-area weights of the cells assigned to each target cell.
    """

    def __init__(self, matrix: sp.csr_matrix, target_shape: tuple[int, int],
                 covered_area: np.ndarray,
                 source_grid: tuple[int, int] | None = None,
                 source_latlon: bool = False):
        """Hold the prebuilt remap matrix, target shape, and area normaliser."""
        self._matrix = matrix              # (n_target, n_source), data = src area
        self._target_shape = target_shape  # (nlon, nlat)
        # Σ source-area landing in each target cell; the normaliser that turns
        # accumulated mass back into an (area-weighted mean) flux.
        self._covered_area = covered_area  # (n_target,)
        # Rectilinear source (#533): fields arrive with the two spatial axes
        # unflattened. ``source_grid`` is (nlon_src, nlat_src); the matrix
        # columns are lon-major, so lat-major fields transpose on the way in.
        # ``source_latlon`` records the layout the src_area was given in — the
        # tie-breaker for square grids, where shape alone cannot distinguish.
        self._source_grid = source_grid
        self._source_latlon = source_latlon

    @property
    def target_shape(self) -> tuple[int, int]:
        return self._target_shape

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Regrid ``values`` shaped ``(..., n_source)`` → ``(..., nlon, nlat)``.

        For a rectilinear source, ``values`` instead carries the two spatial
        axes unflattened — ``(..., nlat_src, nlon_src)`` (the common netCDF
        layout) or ``(..., nlon_src, nlat_src)``; they are flattened here in
        the matrix's ordering.

        Leading axes (time, level, …) are preserved. Target cells that received
        no source cell (only possible when *refining*) come back as zero.
        """
        values = np.asarray(values, dtype=np.float64)
        if self._source_grid is not None:
            nlon_s, nlat_s = self._source_grid
            trailing = values.shape[-2:] if values.ndim >= 2 else None
            lat_major = trailing == (nlat_s, nlon_s)
            if nlat_s == nlon_s:
                lat_major = self._source_latlon    # shape cannot distinguish
            if trailing not in ((nlon_s, nlat_s), (nlat_s, nlon_s)):
                raise ValueError(
                    f"rectilinear-source regridder expects trailing spatial "
                    f"axes ({nlat_s}, {nlon_s}) or ({nlon_s}, {nlat_s}), "
                    f"got {values.shape}")
            if lat_major:
                values = np.swapaxes(values, -1, -2)
            values = values.reshape(*values.shape[:-2], nlon_s * nlat_s)
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

    area = np.asarray(src_area, dtype=np.float64)
    source_grid = None
    source_latlon = False
    if sl.size != area.size:
        # Rectilinear source: 1-D lon/lat axes with a 2-D area, the common
        # native layout of input4MIPs products (#533). Expand to the
        # per-cell mesh this operator is defined on; the returned Regridder
        # remembers the layout so fields can be applied unflattened.
        source_grid = (sl.size, sb.size)
        if area.shape == (sl.size, sb.size):
            pass
        elif area.shape == (sb.size, sl.size):
            source_latlon = True
            area = area.T
        else:
            raise ValueError(
                f"src_area shape {area.shape} matches neither the flattened "
                f"source ({sl.size} cells) nor a (lon, lat)/(lat, lon) "
                f"rectilinear mesh of the 1-D axes ({sl.size}x{sb.size})")
        sl, sb = (m.ravel() for m in np.meshgrid(sl, sb, indexing="ij"))
    area = area.ravel()
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
    return Regridder(matrix, (nlon, nlat), covered_area,
                     source_grid=source_grid, source_latlon=source_latlon)


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


def gaussian_latlon(nlat: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian latitudes (ascending, degrees) and lons for a 2:1 grid."""
    lats = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))
    return lats, np.arange(2 * nlat) * 360.0 / (2 * nlat)


def unit_sphere_vectors(lat_deg, lon_deg) -> np.ndarray:
    """(..., 3) unit vectors for lat/lon in degrees."""
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def nearest_index(src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """Index of the nearest source point for each destination point.

    Great-circle nearest neighbour via a KDTree on unit-sphere vectors;
    all coordinates in degrees.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(unit_sphere_vectors(src_lat, src_lon))
    return tree.query(unit_sphere_vectors(dst_lat, dst_lon), workers=-1)[1]


def interp_to(da: xr.DataArray, lats, lons) -> xr.DataArray:
    """Bilinear regrid of a (..., lat, lon) DataArray to new coordinates.

    Periodic longitude wrap on both ends and constant extension to the
    poles, so Gaussian targets outside the source's first/last row
    interpolate instead of going NaN. Renames the trailing dims to
    ``lat``/``lon``. Not conservative — use the Regridder for fluxes.
    """
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
    """Fill NaNs (e.g. land in ocean products) with the nearest valid value.

    ``field`` is ``(time, lat, lon)``; the mask may differ per time step.
    """
    from scipy.spatial import cKDTree

    glon, glat = np.meshgrid(lons, lats)
    xyz = unit_sphere_vectors(glat, glon)
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
    """Conservatively remap a regular-grid flux onto a Gaussian grid.

    Thin adapter over :func:`build_regridder` for ``(..., lat, lon)``
    structured sources: the source mesh is flattened to ``(lon, lat,
    cos-lat-area)`` triples and the result reshaped back to
    ``(..., nlat, nlon)``.
    """
    src_lats = np.asarray(src_lats, float)
    src_lons = np.asarray(src_lons, float)
    glat, glon = np.meshgrid(src_lats, src_lons, indexing="ij")
    area = np.cos(np.deg2rad(glat)).ravel()
    rg = build_regridder(glon.ravel(), glat.ravel(), area,
                         np.asarray(lons, float), np.asarray(lats, float),
                         dst_in_degrees=True)
    lead = field.shape[:-2]
    flat = field.reshape(*lead, -1)
    out = rg(flat)                                   # (..., nlon, nlat)
    return np.swapaxes(out, -2, -1)                  # (..., nlat, nlon)


def latitude_bounds(lats) -> np.ndarray:
    """Cell edges ``(nlat + 1,)`` in degrees for ascending cell-centre latitudes.

    A Gaussian grid gets its exact quadrature cells: the edges in ``μ = sin φ``
    are the cumulative Gauss–Legendre weights, so each cell's area is exactly
    its quadrature weight — the cell definition ECHAM and CDO's ``remapcon``
    use for Gaussian grids. Any other grid gets centre midpoints closed at ±90°.
    """
    lats = np.asarray(lats, float)
    n = lats.size
    gauss = gaussian_latlon(n)[0]
    if np.allclose(lats, gauss, atol=1e-6):
        weights = np.polynomial.legendre.leggauss(n)[1]
        mu = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
        return np.rad2deg(np.arcsin(np.clip(mu, -1.0, 1.0)))
    mid = 0.5 * (lats[1:] + lats[:-1])
    return np.concatenate([[-90.0], mid, [90.0]])


def _latitude_overlap(src_lats, dst_lats) -> np.ndarray:
    """``(ndst, nsrc)`` overlap of latitude bands in ``μ = sin φ`` (∝ area)."""
    s = np.sin(np.deg2rad(latitude_bounds(src_lats)))
    d = np.sin(np.deg2rad(latitude_bounds(dst_lats)))
    lo = np.maximum(d[:-1, None], s[None, :-1])
    hi = np.minimum(d[1:, None], s[None, 1:])
    return np.clip(hi - lo, 0.0, None)


def _longitude_overlap(src_lons, dst_lons) -> np.ndarray:
    """``(ndst, nsrc)`` overlap of periodic longitude intervals, in degrees.

    Both grids are regular in longitude; each cell spans ± half a spacing about
    its centre. The source intervals are tried at −360/0/+360 so an interval
    straddling the date line overlaps correctly.
    """
    def edges(lons):
        lons = np.asarray(lons, float)
        half = 0.5 * 360.0 / lons.size
        return lons - half, lons + half
    s_lo, s_hi = edges(src_lons)
    d_lo, d_hi = edges(dst_lons)
    total = np.zeros((d_lo.size, s_lo.size))
    for shift in (-360.0, 0.0, 360.0):
        lo = np.maximum(d_lo[:, None], s_lo[None, :] + shift)
        hi = np.minimum(d_hi[:, None], s_hi[None, :] + shift)
        total += np.clip(hi - lo, 0.0, None)
    return total


def conservative_overlap(field: np.ndarray, src_lats, src_lons,
                         lats, lons) -> np.ndarray:
    """Exact first-order conservative remap between two rectilinear grids.

    Every target cell receives the area-weighted mean of the source cells it
    overlaps, weighted by the exact overlap area — CDO ``remapcon``'s scheme,
    and the one the HAMMOZ boundary files were themselves produced with. Unlike
    :func:`conservative_to_gaussian` (nearest-centre binning, built for fine
    0.25° sources onto much coarser model grids) it stays accurate when source
    and target resolutions are close (e.g. T127 → T106, ratio 1.2) and gives the
    correct piecewise-constant answer when *refining*.

    Rectilinear lat/lon grids make the overlap separable — a latitude-band
    overlap in ``sin φ`` times a longitude-interval overlap — so the operator is
    two small dense matrices, not a sparse polygon intersection.

    ``field`` is ``(..., nlat_src, nlon_src)`` with ascending latitudes; the
    result is ``(..., nlat, nlon)``. NaN marks missing source data and is
    excluded with the normalisation renormalised over the valid overlap (CDO's
    ``fracarea``); a target cell with no valid overlap is NaN.
    """
    field = np.asarray(field, dtype=np.float64)
    w_lat = _latitude_overlap(src_lats, lats)            # (nlat, nlat_src)
    w_lon = _longitude_overlap(src_lons, lons)           # (nlon, nlon_src)
    valid = np.isfinite(field)
    num = w_lat @ np.where(valid, field, 0.0) @ w_lon.T
    den = w_lat @ valid.astype(np.float64) @ w_lon.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.0, num / den, np.nan)
