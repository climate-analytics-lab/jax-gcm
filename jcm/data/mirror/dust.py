"""Mirror products for the Tegen/HAMMOZ dust emission scheme (issue #802).

Five bundles per grid: the monthly potential-source (effective-LAI)
climatology, the static preferential-source (paleolake) fraction, the nine
soil-texture area fractions, the categorical 1-8 tuning-region mask and the
monthly satellite roughness map.

Sources — the ECHAM-HAMMOZ input pool (``/pool/data/ECHAM6-HAMMOZ`` on DKRZ
Levante, see :mod:`jcm.data.mirror.sites`), at every resolution HAMMOZ ships:

* **T63** — ``v0007/hammoz/T63``: the set HAMMOZ runs with today.
* **T127** — ``v0003/hammoz/T127``: the same 2016 processing as the T63 set,
  same file names (its T63 siblings are identical to ``v0007``'s, and every
  field has the same global mean at T63 and T127).
* **T255** — only the older ``v01_001/hammoz/T255`` lineage exists. Its files
  are the *same products* under older names, verified at T63 and T127, where
  both lineages exist: ``ndvi_lai_eff.12m`` ``laieff`` is ``pot_source``
  exactly, ``pot_sources`` ``source`` is the preferential-source fraction (to
  3e-7), and ``soil_typeN`` ``type`` is ``soil_type_all`` ``typeN`` exactly.
  The one product with no T255 file is the roughness map: the old lineage's
  ``surface_rough.12m`` is a different field (no NaN mask, zeros where the new
  map has its 0.001 floor), so it is not used.

Per target grid each product comes from, in order of preference:

1. the **native** file for that truncation, copied through with only a latitude
   flip;
2. otherwise the **finest** native file, remapped with the exact-overlap
   first-order conservative scheme (:func:`jcm.data.regridding.
   conservative_overlap` — CDO ``remapcon``, the scheme the HAMMOZ files were
   made with). Coarsening (e.g. T106 from T255) is its intended use; refining
   (T255 roughness from T127, the only case) returns the piecewise-constant
   source field and is stamped as an approximation in the file attributes.

The **region mask** is never regridded: it is regenerated on every grid from
the eight ``cdo setclonlatbox`` longitude/latitude boxes recorded in the HAMMOZ
file's own history (Huneeus et al. 2011 regions). The recipe reproduces the
native T63 and T127 masks cell for cell, which the builder re-checks against
any native file it can see — so it is exact, categorical, and needs no
nearest-neighbour choice.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.mirror import sites
from jcm.data.regridding import conservative_overlap, gaussian_latlon

#: Truncation -> Gaussian latitude count, for the resolutions HAMMOZ ships.
NATIVE_NLAT = {63: 96, 127: 192, 255: 384}

_SOIL_TYPES = (2, 3, 4, 6, 13, 14, 15, 16, 17)


def _new_lineage(trunc: int, version: str) -> dict:
    t = f"T{trunc}"
    base = f"{version}/hammoz/{t}"
    return {
        "dust_potential_sources": {
            "pot_source": (f"{base}/dust_potential_sources_{t}.nc",
                           "pot_source")},
        "dust_preferential_sources": {
            "source": (f"{base}/dust_preferential_sources_{t}.nc", "source")},
        "dust_soil_types": {
            f"type{i}": (f"{base}/soil_type_all_{t}.nc", f"type{i}")
            for i in _SOIL_TYPES},
        "dust_regions": {
            "regions": (f"{base}/dust_regions_{t}.nc", "regions")},
        "dust_surface_roughness": {
            "surfrough": (f"{base}/surface_rough_12m_{t}.nc", "surfrough")},
    }


#: ``truncation -> product -> {output variable: (pool-relative file, source
#: variable)}`` — every native HAMMOZ file the builder may read. ``soilpHfrac``
#: and ``xtsurf_v2`` of the same input sets are deliberately absent:
#: ``mo_ham_dust.f90`` never references them, so they are not emission inputs.
NATIVE_SOURCES: dict[int, dict[str, dict[str, tuple[str, str]]]] = {
    63: _new_lineage(63, "v0007"),
    127: _new_lineage(127, "v0003"),
    255: {
        "dust_potential_sources": {
            "pot_source": ("v01_001/hammoz/T255/ndvi_lai_eff.12m.T255.nc",
                           "laieff")},
        "dust_preferential_sources": {
            "source": ("v01_001/hammoz/T255/pot_sources.T255.nc", "source")},
        "dust_soil_types": {
            f"type{i}": (f"v01_001/hammoz/T255/soil_type{i}.T255.nc", "type")
            for i in _SOIL_TYPES},
    },
}

DUST_PRODUCTS = ("dust_potential_sources", "dust_preferential_sources",
                 "dust_soil_types", "dust_regions", "dust_surface_roughness")

#: Products whose time axis is a real 12-month climatology; the rest carry a
#: degenerate length-1 axis in the source that is dropped on the way out.
MONTHLY = ("dust_potential_sources", "dust_surface_roughness")

#: The tuning-region recipe from the ``history`` attribute of HAMMOZ's
#: ``dust_regions_T63.nc`` (C. Siegenthaler, C2SM/ETHZ, after Huneeus et al.
#: 2011): ``cdo setclonlatbox,<region>,<lon1>,<lon2>,<lat1>,<lat2>`` applied in
#: this order, each later box overwriting the earlier ones. A cell belongs to a
#: box when its centre lies inside it (edges inclusive), longitudes in
#: [-180, 180). Region 1 is the global background.
REGION_BOXES: tuple[tuple[int, float, float, float, float], ...] = (
    (1, -180.0, 180.0, -90.0, 90.0),
    (2, -170.0, -50.0, 15.0, 75.0),     # North America
    (3, -90.0, -30.0, -60.0, 15.0),     # South America
    (4, -20.0, 35.0, 0.0, 30.0),        # North Africa
    (5, 10.0, 40.0, -45.0, 0.0),        # South Africa
    (6, 35.0, 60.0, 10.0, 35.0),        # Middle East
    (7, 60.0, 135.0, 10.0, 80.0),       # Asia
    (8, 110.0, 155.0, -40.0, -10.0),    # Australia
)

#: Canonical month-start axis of the HAMMOZ monthly files (``v0007``/``v0003``);
#: written on every monthly product so all grids decode identically whatever
#: time encoding the source lineage used.
_MONTH_STARTS = np.array(
    [(np.datetime64(f"2000-{m:02d}-01") - np.datetime64("2000-01-01"))
     .astype(int) for m in range(1, 13)], dtype=np.float64)
_TIME_ATTRS = {"standard_name": "time",
               "units": "days since 2000-01-01 00:00:00",
               "calendar": "proleptic_gregorian"}

_LONG_NAMES = {
    "pot_source": "effective-LAI erodible bare-soil fraction",
    "source": "preferential (paleolake) dust source area fraction",
    "regions": "regional dust-tuning index (1-8, Huneeus et al. 2011)",
    "surfrough": "Prigent et al. (2005) satellite surface roughness length",
}


def region_mask(lats, lons) -> np.ndarray:
    """Return the 1-8 tuning-region mask on a ``(lat, lon)`` grid (``REGION_BOXES``)."""
    lons = np.asarray(lons, float)
    wrapped = np.where(lons >= 180.0, lons - 360.0, lons)
    la, lo = np.meshgrid(np.asarray(lats, float), wrapped, indexing="ij")
    mask = np.zeros(la.shape)
    for region, lon1, lon2, lat1, lat2 in REGION_BOXES:
        inside = ((lo >= lon1) & (lo <= lon2) & (la >= lat1) & (la <= lat2))
        mask[inside] = region
    return mask


def _truncation_for(nlat: int) -> int | None:
    return next((t for t, n in NATIVE_NLAT.items() if n == nlat), None)


def _read(root: Path, rel: str, var: str, monthly: bool):
    """Return ``(field (…, lat, lon) ascending, lats, lons)`` from one file."""
    with xr.open_dataset(root / rel, decode_times=False) as ds:
        field = np.asarray(ds[var].values, dtype=np.float64)
        lats, lons = ds["lat"].values, ds["lon"].values
    if not monthly and field.ndim == 3:
        field = field[0]
    if monthly and field.shape[0] != 12:
        raise ValueError(f"{rel}:{var} has {field.shape[0]} records, not 12")
    if lats[0] > lats[-1]:
        field, lats = field[..., ::-1, :], lats[::-1]
    return field, lats, lons


def _check_on_grid(rel, src_lats, src_lons, lats, lons):
    if not (np.allclose(src_lats, lats, atol=1e-6)
            and np.allclose(src_lons, lons, atol=1e-6)):
        raise ValueError(f"{rel}: native file is not on the model's Gaussian "
                         f"grid ({src_lats.size} latitudes)")


def build_dust_product(name: str, nlat: int, out_path, source_dir=None) -> None:
    """Write one dust bundle for a Gaussian grid of ``nlat`` latitudes.

    ``source_dir`` is the HAMMOZ pool root (default: the active site's).
    Validates what the readers will later refuse anyway, but at build time where
    it is cheap to fix: the region mask must be integral in [1, 8] and match any
    native HAMMOZ mask exactly, and the global Zobler soil textures must remain a
    partition (the type13-17 group is a separate, overlapping China-only
    partition and is deliberately not summed with them).
    """
    if name not in DUST_PRODUCTS:
        raise KeyError(f"unknown dust product {name!r}")
    if os.environ.get("JCM_HAMMOZ_DUST_DIR"):
        # The retired variable pointed at a flat T63-only directory; the pool
        # layout differs, so silently ignoring it would read other files.
        raise ValueError("JCM_HAMMOZ_DUST_DIR is retired: set JCM_HAMMOZ_DIR to "
                         "a directory laid out like /pool/data/ECHAM6-HAMMOZ.")
    root = Path(source_dir or sites.current().hammoz or "")
    lats, lons = gaussian_latlon(nlat)
    trunc = _truncation_for(nlat)
    monthly = name in MONTHLY
    out_vars, attrs = {}, {"product": name}

    if name == "dust_regions":
        values = region_mask(lats, lons)
        native = NATIVE_SOURCES.get(trunc, {}).get(name)
        checked = []
        for t, table in NATIVE_SOURCES.items():
            if name not in table:
                continue
            rel, var = table[name]["regions"]
            if not (root / rel).exists():
                continue
            ref, rlats, rlons = _read(root, rel, var, monthly=False)
            if not np.array_equal(region_mask(rlats, rlons), ref):
                raise ValueError(f"{rel}: the region recipe no longer "
                                 "reproduces the native HAMMOZ mask")
            checked.append(rel)
        if native is not None and native["regions"][0] not in checked:
            raise FileNotFoundError(
                f"native region mask {root / native['regions'][0]} is missing")
        out_vars["regions"] = (("lat", "lon"), values,
                               {"units": "1", "long_name": _LONG_NAMES["regions"]})
        attrs["source"] = ("HAMMOZ dust_regions setclonlatbox recipe "
                           "(Huneeus et al. 2011), regenerated on this grid")
        attrs["history"] = ("jcm.data.mirror.dust.region_mask; verified "
                            "cell-for-cell against " + (", ".join(checked)
                                                         or "no native file"))
    else:
        available = [t for t, table in NATIVE_SOURCES.items() if name in table]
        if trunc in available:
            src_trunc, method = trunc, "native"
        else:
            src_trunc = max(available)
            method = ("conservative" if NATIVE_NLAT[src_trunc] > nlat
                      else "refined")
        table = NATIVE_SOURCES[src_trunc][name]
        files = sorted({rel for rel, _ in table.values()})
        for out_name, (rel, var) in table.items():
            field, src_lats, src_lons = _read(root, rel, var, monthly)
            if method == "native":
                _check_on_grid(rel, src_lats, src_lons, lats, lons)
            else:
                field = conservative_overlap(field, src_lats, src_lons,
                                             lats, lons)
            dims = ("time", "lat", "lon") if monthly else ("lat", "lon")
            out_vars[out_name] = (dims, field, {
                "units": "cm" if out_name == "surfrough" else "1",
                "long_name": _LONG_NAMES.get(
                    out_name, f"soil texture fraction {out_name}")})
        attrs["source"] = "HAMMOZ dust input set (" + ", ".join(files) + ")"
        attrs["history"] = "jcm.data.mirror.dust: latitude flipped to ascending"
        if method != "native":
            attrs["history"] += (f"; T{src_trunc} -> {nlat}-latitude Gaussian, "
                                 "exact-overlap first-order conservative")
        if method == "refined":
            attrs["regrid_approximation"] = (
                f"no native HAMMOZ file exists at this resolution; refined from "
                f"the finest native one (T{src_trunc}), so the field carries "
                f"T{src_trunc} detail on this grid (piecewise constant).")

    if name == "dust_regions":
        values = out_vars["regions"][1]
        if not np.allclose(values, np.round(values)) or values.min() < 1 \
                or values.max() > 8:
            raise ValueError(
                f"{name} at {nlat} latitudes: the region mask lost integrality "
                f"(span [{values.min()}, {values.max()}]).")
    if name == "dust_soil_types":
        total = sum(out_vars[f"type{i}"][1] for i in (2, 3, 4, 6))
        if total.max() > 1.0 + 1e-5:
            raise ValueError(
                f"{name} at {nlat} latitudes: the global Zobler textures sum to "
                f"{total.max():.4f} > 1, so the type-1 residual would go negative.")

    coords = {"lat": lats, "lon": lons}
    if monthly:
        coords["time"] = ("time", _MONTH_STARTS, dict(_TIME_ATTRS))
        attrs["time_note"] = (
            "month starts; the scheme steps this climatology by month "
            "(WRAP_YEAR), it is never interpolated")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(out_vars, coords=coords, attrs=attrs).to_netcdf(out_path)
