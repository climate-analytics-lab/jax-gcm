"""MACv2 natural aerosol background, regridded and labelled per model grid.

MACv2 (Kinne 2019, Tellus B 71, doi:10.1080/16000889.2019.1623639) is the
monthly 1°×1° aerosol climatology the MACv2-SP simple plumes were fitted to
(Stevens et al. 2017, doi:10.5194/gmd-10-433-2017). The plumes describe only
the anthropogenic increment; the background they sit on is the climatology's
coarse mode (dust, sea salt) plus its pre-industrial fine mode, which is what
this module publishes — the pairing ICON's ``irad_aero = 18`` makes
(``mo_echam_phy_bcs.f90``: the Kinne climatology read at a fixed 1850, plus the
plumes).

The source is the ``for_ICON`` tree MPI-M serves at
:data:`SOURCE_URL` (data statement: Kinne 2019, ACP 19, 10919,
doi:10.5194/acp-19-10919-2019). Its files carry no usable coordinates beyond
latitude and longitude, so the bundle states them:

* **Bands.** The optics files stack one file per band along an unnamed
  ``record`` dimension. Their order is read from each file's own ``history``
  attribute (the ``ncecat`` command that built it lists the per-wavelength
  files) and checked against :data:`SW_WAVELENGTH_NM` /
  :data:`LW_WAVELENGTH_NM`. It runs in *ascending wavelength*: the reverse of
  a wavenumber-ordered band list. The band limits are RRTMG's, assigned by
  position as in the ECHAM6 user guide's table of band index against
  wavelength range (appendix "Kinne aerosol optical properties").
* **Months.** Twelve monthly means, ``month = 1..12``; the two source time
  axes disagree and neither is a calendar.
* **Layers.** The altitude files hold the fraction of column optical depth in
  each of 20 uneven layers above sea level (not the 40 even extinction levels
  ``mo_aero_kinne.f90`` reads), so the layer bounds are coordinates and a
  reader distributes by overlap.

Regridding is exact-overlap first-order conservative
(:func:`jcm.data.regridding.conservative_overlap`) on every grid, so the area
mean of every extensive quantity is the source's to round-off. Optical depth
is regridded directly; single-scattering albedo, asymmetry and the layer
fractions are intensive and are regridded weighted by the quantity they
partition — the same optical-depth weights the reference uses when it
combines the two modes (``set_aop_kinne``: sums of τ, τω, τωg).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.mirror import sites
from jcm.data.regridding import conservative_overlap, gaussian_latlon, latitude_bounds

#: Bundle file stem: ``bundles/{grid}/macv2_natural_aerosol.nc``.
PRODUCT = "macv2_natural_aerosol"

#: Where MPI-M serves the source tree (``JCM_MACV2_DIR`` is a local copy of it).
SOURCE_URL = ("ftp://ftp-projects.mpimet.mpg.de/aerocom/climatology/"
              "MACv2_2018/for_ICON")

#: Source file per output group, relative to the ``for_ICON`` tree, with the
#: one-letter component tag its per-wavelength files carry (``gt_<tag>_*nm``).
OPTICS_FILES = {
    "sw_coarse": ("spec_2005/aeropt_kinne_sw_b14_coa.nc", "c"),
    "sw_fine": ("spec_2005/aeropt_kinne_sw_b14_fin_pre.nc", "p"),
    "lw_coarse": ("spec_2005/aeropt_kinne_lw_b16_coa.nc", "i"),
}
#: Altitude file and variable per mode.
ALTITUDE_FILES = {
    "coarse": ("altitude/aeropt_coa_alt_km20.nc", "zkm_aer_coarse_mo"),
    "fine": ("altitude/aeropt_fin_alt_km20.nc", "zkm_aer_fine_mo"),
}

#: Nominal wavelength [nm] of each band, in file order (ascending). These are
#: the labels in the source's per-band file names.
SW_WAVELENGTH_NM = (230, 300, 400, 550, 700, 1000, 1270, 1460, 1780, 2050,
                    2320, 2790, 3470, 8000)
LW_WAVELENGTH_NM = (3600, 4000, 4200, 4600, 5200, 6150, 7000, 7850, 8850,
                    9750, 11200, 13200, 15100, 18000, 30000, 90000)

#: RRTMG band limits [cm⁻¹] in the same (ascending-wavelength) order. Assigned
#: by position; the nominal 4200 nm and 30000 nm labels sit marginally outside
#: their band, which is why a reader maps on these limits and not the labels.
SW_WAVENUMBER_BOUNDS = (
    (38000, 50000), (29000, 38000), (22650, 29000), (16000, 22650),
    (12850, 16000), (8050, 12850), (7700, 8050), (6150, 7700), (5150, 6150),
    (4650, 5150), (4000, 4650), (3250, 4000), (2600, 3250), (820, 2600))
LW_WAVENUMBER_BOUNDS = (
    (2600, 3250), (2380, 2600), (2250, 2380), (2080, 2250), (1800, 2080),
    (1480, 1800), (1390, 1480), (1180, 1390), (1080, 1180), (980, 1080),
    (820, 980), (700, 820), (630, 700), (500, 630), (350, 500), (10, 350))

#: Index of the 550 nm band, whose optical depth weights the layer fractions.
_I550 = SW_WAVELENGTH_NM.index(550)

_REFERENCES = (
    "Kinne (2019), Tellus B 71, 1623639, doi:10.1080/16000889.2019.1623639; "
    "Kinne (2019), ACP 19, 10919-10959, doi:10.5194/acp-19-10919-2019")


def source_files() -> list[str]:
    """Every source-relative file :func:`build_macv2_natural` reads."""
    return sorted([rel for rel, _ in OPTICS_FILES.values()]
                  + [rel for rel, _ in ALTITUDE_FILES.values()])


def _history_wavelengths(history: str, tag: str) -> tuple[int, ...]:
    """Band wavelengths [nm] in the order ``ncecat`` stacked them.

    The optics files record the command that built them, e.g. ``ncecat
    gt_c_00230nm.nc gt_c_00300nm.nc ... out.nc``; the ``gt_<tag>_<nm>nm.nc``
    arguments are the bands in ``record`` order.
    """
    return tuple(int(nm) for nm in
                 re.findall(rf"gt_{re.escape(tag)}_(\d+)nm\.nc", history))


def _ascending(lats: np.ndarray, field: np.ndarray):
    """Return ``lats`` ascending and ``field`` flipped to match (lat on axis -2)."""
    if lats[0] > lats[-1]:
        return lats[::-1], field[..., ::-1, :]
    return lats, field


def _read_optics(root: Path, key: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """``({aod, ssa, asy}, lats, lons)`` as ``(band, month, lat, lon)`` float64."""
    rel, tag = OPTICS_FILES[key]
    expected = SW_WAVELENGTH_NM if key.startswith("sw") else LW_WAVELENGTH_NM
    with xr.open_dataset(root / rel, decode_times=False) as ds:
        found = _history_wavelengths(str(ds.attrs.get("history", "")), tag)
        if found != expected:
            raise ValueError(
                f"{rel}: the file history lists the bands as {found}, not the "
                f"{expected} this builder labels them with")
        lats = np.asarray(ds["lat"].values, dtype=np.float64)
        lons = np.asarray(ds["lon"].values, dtype=np.float64)
        out = {}
        for name in ("aod", "ssa", "asy"):
            da = ds[name].transpose("record", "time", "lat", "lon")
            if da.shape[:2] != (len(expected), 12):
                raise ValueError(f"{rel}: {name} is {da.shape[0]} bands × "
                                 f"{da.shape[1]} months, expected "
                                 f"{len(expected)} × 12")
            field = np.asarray(da.values, dtype=np.float64)
            if not np.isfinite(field).all():
                raise ValueError(f"{rel}: {name} has non-finite values")
            _, out[name] = _ascending(lats, field)
    return out, np.sort(lats), lons


def _read_altitude(root: Path, mode: str):
    """``(fraction (month, layer, lat, lon), bottom, top, surface, lats, lons)``."""
    rel, var = ALTITUDE_FILES[mode]
    with xr.open_dataset(root / rel, decode_times=False) as ds:
        lats = np.asarray(ds["lat"].values, dtype=np.float64)
        lons = np.asarray(ds["lon"].values, dtype=np.float64)
        frac = np.asarray(ds[var].transpose("time", "lay", "lat", "lon").values,
                          dtype=np.float64)
        bottom = np.asarray(ds["zbot"].values, dtype=np.float64)
        top = np.asarray(ds["ztop"].values, dtype=np.float64)
        surface = np.asarray(ds["asl"].transpose("lat", "lon").values,
                             dtype=np.float64)
    if frac.shape[0] != 12:
        raise ValueError(f"{rel}: {var} has {frac.shape[0]} months, expected 12")
    if not (np.all(top > bottom) and np.allclose(top[:-1], bottom[1:])):
        raise ValueError(f"{rel}: the layer bounds are not contiguous")
    total = frac.sum(axis=1)
    if not np.isfinite(frac).all() or np.abs(total - 1.0).max() > 1e-4:
        raise ValueError(f"{rel}: {var} does not sum to one over the layers "
                         f"(span [{total.min():.5f}, {total.max():.5f}])")
    _, frac = _ascending(lats, frac)
    _, surface = _ascending(lats, surface)
    return frac, bottom, top, surface, np.sort(lats), lons


def _area_mean(field: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Area mean over the trailing ``(lat, lon)`` axes of a rectilinear grid."""
    w = np.diff(np.sin(np.deg2rad(latitude_bounds(lats))))
    return (field.mean(axis=-1) * w).sum(axis=-1) / w.sum()


def _regrid_optics(src: dict, src_lats, src_lons, lats, lons) -> dict:
    """Regrid ``{aod, ssa, asy}`` with the reference's optical-depth weights.

    ``aod`` is extensive in the column and is remapped directly. ``ssa`` is
    the scattering share of ``aod`` and ``asy`` the asymmetry of the scattered
    part, so they are the ratios of the remapped ``aod·ssa`` to ``aod`` and of
    ``aod·ssa·asy`` to ``aod·ssa`` — what a column holding the source cells
    side by side would show. Where the denominator vanishes the limits are the
    reference's (``set_aop_kinne``): ``ssa → 1``, ``asy → 0``.
    """
    def remap(field):
        return conservative_overlap(field, src_lats, src_lons, lats, lons)

    aod = remap(src["aod"])
    scat = remap(src["aod"] * src["ssa"])
    forward = remap(src["aod"] * src["ssa"] * src["asy"])
    has_aod, has_scat = aod > 0.0, scat > 0.0
    ssa = np.where(has_aod, scat / np.where(has_aod, aod, 1.0), 1.0)
    asy = np.where(has_scat, forward / np.where(has_scat, scat, 1.0), 0.0)
    return {"aod": aod, "ssa": ssa, "asy": asy}


def _regrid_fraction(frac, weight, src_lats, src_lons, lats, lons):
    """Regrid layer fractions weighted by the optical depth they partition.

    ``frac`` is ``(month, layer, lat, lon)`` and ``weight`` the mode's 550 nm
    optical depth ``(month, lat, lon)``; the result still sums to one over the
    layers, and ``weight·frac`` (the layer optical depth) is conserved.
    """
    num = conservative_overlap(frac * weight[:, None], src_lats, src_lons,
                               lats, lons)
    den = conservative_overlap(weight, src_lats, src_lons, lats, lons)[:, None]
    plain = conservative_overlap(frac, src_lats, src_lons, lats, lons)
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), plain)


def build_macv2_natural(nlat: int, out_path, source_dir=None) -> None:
    """Write the MACv2 natural-aerosol bundle for a Gaussian grid of ``nlat``.

    ``source_dir`` is a copy of the ``for_ICON`` tree (:data:`SOURCE_URL`); by
    default the active site's (``JCM_MACV2_DIR``). The build refuses a source
    whose band order, month count or layer fractions are not what the labels
    it writes claim, and checks that the remap conserved the global-mean
    optical depth of every band.
    """
    if source_dir is None:
        site = sites.current()
        source_dir = site.macv2
        if not source_dir:
            raise FileNotFoundError(
                f"the MACv2 for_ICON tree is not declared for site "
                f"{site.name!r}: download {SOURCE_URL} and set JCM_MACV2_DIR "
                "to the copy.")
    root = Path(source_dir)
    lats, lons = gaussian_latlon(nlat)
    out_vars, means = {}, {}
    grids = {}

    for key in OPTICS_FILES:
        src, src_lats, src_lons = _read_optics(root, key)
        grids[key] = (src_lats, src_lons)
        new = _regrid_optics(src, src_lats, src_lons, lats, lons)
        before = _area_mean(src["aod"].mean(axis=1), src_lats)
        after = _area_mean(new["aod"].mean(axis=1), lats)
        if not np.allclose(after, before, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"{key} at {nlat} latitudes: the remap changed the global-mean "
                f"optical depth (worst band {np.abs(after / before - 1).max():.2e})")
        means[key] = (src, new, after)
        spectrum, mode = key.split("_")
        band = f"band_{spectrum}"
        mode_name = ("coarse mode" if mode == "coarse"
                     else "pre-industrial fine mode")
        for name, long_name in (
                ("aod", "column aerosol optical depth"),
                ("ssa", "single-scattering albedo"),
                ("asy", "asymmetry parameter")):
            out_vars[f"{name}_{key}"] = (
                ("month", band, "lat", "lon"),
                np.moveaxis(new[name], 0, 1).astype(np.float32),
                {"units": "1", "long_name": f"{long_name}, {mode_name}"})

    for mode, optics_key in (("coarse", "sw_coarse"), ("fine", "sw_fine")):
        frac, bottom, top, surface, a_lats, a_lons = _read_altitude(root, mode)
        src_lats, src_lons = grids[optics_key]
        if not (np.array_equal(a_lats, src_lats)
                and np.array_equal(a_lons, src_lons)):
            raise ValueError(f"{ALTITUDE_FILES[mode][0]} is not on the grid of "
                             f"{OPTICS_FILES[optics_key][0]}")
        weight = means[optics_key][0]["aod"][_I550]
        new = _regrid_fraction(frac, weight, src_lats, src_lons, lats, lons)
        mode_name = ("coarse mode" if mode == "coarse"
                     else "pre-industrial fine mode")
        out_vars[f"layer_fraction_{mode}"] = (
            ("month", "layer", "lat", "lon"), new.astype(np.float32),
            {"units": "1",
             "long_name": ("fraction of column aerosol optical depth in the "
                           f"layer, {mode_name}"),
             "comment": "sums to one over layer; spectrally uniform"})
    out_vars["source_surface_altitude"] = (
        ("lat", "lon"),
        conservative_overlap(surface, src_lats, src_lons, lats, lons
                             ).astype(np.float32),
        {"units": "m", "long_name": ("surface altitude above sea level the "
                                     "climatology's layers were built on"),
         "comment": "no optical depth lies in a layer wholly below it"})

    def band_coords(spectrum, wavelengths, bounds):
        dim = f"band_{spectrum}"
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        return {
            dim: (dim, np.arange(1, len(wavelengths) + 1),
                  {"long_name": f"{spectrum} band index in source file order "
                                "(ascending wavelength)"}),
            f"wavelength_{spectrum}": (
                dim, np.array(wavelengths, dtype=np.float64),
                {"units": "nm", "long_name": "nominal band wavelength (the "
                                             "source's per-band file label)"}),
            f"wavenumber_lower_{spectrum}": (
                dim, lo, {"units": "cm-1",
                          "long_name": "RRTMG band lower limit"}),
            f"wavenumber_upper_{spectrum}": (
                dim, hi, {"units": "cm-1",
                          "long_name": "RRTMG band upper limit"}),
        }

    coords = {
        "month": ("month", np.arange(1, 13),
                  {"long_name": "calendar month of the monthly mean"}),
        "layer": ("layer", np.arange(1, bottom.size + 1),
                  {"long_name": "layer index, surface upward",
                   "positive": "up"}),
        "layer_bottom": ("layer", bottom, {
            "units": "m", "long_name": "layer bottom altitude above sea level"}),
        "layer_top": ("layer", top, {
            "units": "m", "long_name": "layer top altitude above sea level"}),
        "lat": ("lat", lats, {"units": "degrees_north",
                              "standard_name": "latitude"}),
        "lon": ("lon", lons, {"units": "degrees_east",
                              "standard_name": "longitude"}),
        **band_coords("sw", SW_WAVELENGTH_NM, SW_WAVENUMBER_BOUNDS),
        **band_coords("lw", LW_WAVELENGTH_NM, LW_WAVENUMBER_BOUNDS),
    }
    src_nlat = grids["sw_coarse"][0].size
    attrs = {
        "product": PRODUCT,
        "title": ("MACv2 natural aerosol background: coarse mode and "
                  "pre-industrial fine mode, monthly climatology"),
        "source": f"{SOURCE_URL} (" + ", ".join(source_files()) + ")",
        "references": _REFERENCES,
        "history": (
            "jcm.data.mirror.macv2: latitude flipped to ascending, longitude "
            f"to 0-360; {src_nlat}-latitude regular -> {nlat}-latitude "
            "Gaussian, exact-overlap first-order conservative (ssa, asy and "
            "layer fractions weighted by the optical depth they partition)"),
        "band_note": (
            "bands are in the source's order, ascending wavelength, which is "
            "the reverse of a wavenumber-ordered band list; limits are "
            "RRTMG's, assigned by position"),
        "time_note": ("monthly means; the reference interpolates linearly "
                      "between neighbouring months"),
        "usage_note": (
            "background for the MACv2-SP anthropogenic plumes; shortwave "
            "optics combine the two modes as sums of aod, aod*ssa and "
            "aod*ssa*asy; longwave uses coarse-mode absorption, "
            "aod*(1-ssa)"),
        "global_mean_aod_550nm_coarse": float(
            means["sw_coarse"][2][_I550]),
        "global_mean_aod_550nm_fine": float(means["sw_fine"][2][_I550]),
    }
    if nlat > src_nlat:
        attrs["regrid_approximation"] = (
            f"this grid is finer than the {src_nlat}-latitude source, so the "
            "fields carry the source's detail (piecewise constant).")

    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for name in out_vars}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(out_vars, coords=coords, attrs=attrs).to_netcdf(
        out_path, encoding=encoding)
