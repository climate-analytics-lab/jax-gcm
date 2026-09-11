"""Mirror products for the Tegen/HAMMOZ dust emission scheme (issue #802).

Five bundles per grid, built from the HAMMOZ T63 input set: the monthly
potential-source (effective-LAI) climatology, the static preferential-source
(paleolake) fraction, the nine soil-texture area fractions, the categorical
1-8 tuning-region mask and the monthly satellite roughness map.

T63 is the native grid and is copied through with only a latitude flip.
T106 is a **refinement** (96 -> 160 latitudes), which the conservative path in
:mod:`jcm.data.regridding` explicitly cannot do, so every field is regridded
**nearest-neighbour** — for a refinement that is also the first-order
conservative answer, it preserves the T63 values exactly, and it is the only
choice that keeps the region mask categorical. Each T106 file records the
approximation in its attributes; native-resolution HAMMOZ files replace it when
they become available (issue #802, O7).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.regridding import gaussian_latlon, nearest_index

#: Where the maintainer's HAMMOZ input set lives on Glade. The files carry no
#: licence restriction (unlike the ``mo_ham_dust.f90`` source they drive).
SOURCE_DIR = Path(os.environ.get("JCM_HAMMOZ_DUST_DIR",
                                 "/glade/u/home/duncanwp"))

#: ``product name -> (source file, {output variable: source variable})``.
#: ``soilpHfrac`` and ``xtsurf_v2`` of the same input set are deliberately absent
#: — ``mo_ham_dust.f90`` never references them, so they are not emission inputs.
DUST_PRODUCTS: dict[str, tuple[str, dict[str, str]]] = {
    "dust_potential_sources": ("dust_potential_sources_T63.nc",
                               {"pot_source": "pot_source"}),
    "dust_preferential_sources": ("dust_preferential_sources_T63.nc",
                                  {"source": "source"}),
    "dust_soil_types": ("soil_type_all_T63.nc",
                        {f"type{i}": f"type{i}"
                         for i in (2, 3, 4, 6, 13, 14, 15, 16, 17)}),
    "dust_regions": ("dust_regions_T63.nc", {"regions": "regions"}),
    "dust_surface_roughness": ("surface_rough_12m_T63.nc",
                               {"surfrough": "surfrough"}),
}

#: Products whose time axis is a real 12-month climatology; the rest carry a
#: degenerate length-1 axis in the source that is dropped on the way out.
MONTHLY = ("dust_potential_sources", "dust_surface_roughness")

_LONG_NAMES = {
    "pot_source": "effective-LAI erodible bare-soil fraction",
    "source": "preferential (paleolake) dust source area fraction",
    "regions": "regional dust-tuning index (1-8, Huneeus et al. 2011)",
    "surfrough": "Prigent et al. (2005) satellite surface roughness length",
}


def _ascending(field: np.ndarray, lats: np.ndarray):
    """Flip a (..., lat, lon) field and its latitudes to ascending order."""
    if lats[0] > lats[-1]:
        return field[..., ::-1, :], lats[::-1]
    return field, lats


def _nearest_refine(field: np.ndarray, src_lat, src_lon, dst_lat, dst_lon):
    """Nearest-neighbour (..., lat, lon) -> (..., dst_lat, dst_lon)."""
    src_mesh_lat, src_mesh_lon = np.meshgrid(src_lat, src_lon, indexing="ij")
    dst_mesh_lat, dst_mesh_lon = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    idx = nearest_index(src_mesh_lat.ravel(), src_mesh_lon.ravel(),
                        dst_mesh_lat.ravel(), dst_mesh_lon.ravel())
    lead = field.shape[:-2]
    flat = field.reshape(*lead, -1)[..., idx]
    return flat.reshape(*lead, dst_lat.size, dst_lon.size)


def build_dust_product(name: str, nlat: int, out_path, source_dir=None) -> None:
    """Write one dust bundle for a Gaussian grid of ``nlat`` latitudes.

    Validates what the readers will later refuse anyway, but at build time where
    it is cheap to fix: the region mask must stay integral in [1, 8] (so the
    nearest-neighbour refinement is the only admissible one) and the global
    Zobler soil textures must remain a partition (the type13-17 group is a
    separate, overlapping China-only partition and is deliberately not summed
    with them).
    """
    filename, variables = DUST_PRODUCTS[name]
    src = Path(source_dir or SOURCE_DIR) / filename
    lats, lons = gaussian_latlon(nlat)
    out_vars, attrs = {}, {}
    with xr.open_dataset(src, decode_times=False) as ds:
        src_lat, src_lon = ds["lat"].values, ds["lon"].values
        native = src_lat.size == nlat
        for out_name, in_name in variables.items():
            field = np.asarray(ds[in_name].values, dtype=np.float64)
            if name not in MONTHLY:
                field = field[0] if field.ndim == 3 else field
            field, flipped = _ascending(field, src_lat)
            if not native:
                field = _nearest_refine(field, flipped, src_lon, lats, lons)
            elif not (np.allclose(flipped, lats, atol=1e-6)
                      and np.allclose(src_lon, lons, atol=1e-6)):
                raise ValueError(
                    f"{src}: {nlat}-latitude source is not the model's Gaussian "
                    "grid; only an exact native copy or a refinement is supported.")
            dims = (("time", "lat", "lon") if name in MONTHLY
                    else ("lat", "lon"))
            out_vars[out_name] = (dims, field,
                                  {"units": "cm" if out_name == "surfrough" else "1",
                                   "long_name": _LONG_NAMES.get(
                                       out_name, f"soil texture fraction {out_name}")})
        coords = {"lat": lats, "lon": lons}
        if name in MONTHLY:
            coords["time"] = ds["time"].values
            attrs["time_note"] = (
                "month starts; the scheme steps this climatology by month "
                "(WRAP_YEAR), it is never interpolated")
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
    attrs.update({
        "source": f"HAMMOZ dust input set ({filename})",
        "product": name,
        "history": ("jcm.data.mirror.dust: latitude flipped to ascending"
                    + ("" if native else
                       "; regridded T63 -> Gaussian nearest-neighbour")),
    })
    if not native:
        attrs["regrid_approximation"] = (
            "derived from the native T63 file by nearest neighbour: the "
            "conservative path cannot refine a grid, and the region mask must "
            "stay categorical. Replace with native-resolution HAMMOZ files when "
            "they become available (jax-gcm issue #802, O7).")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(out_vars, coords=coords, attrs=attrs).to_netcdf(out_path)
