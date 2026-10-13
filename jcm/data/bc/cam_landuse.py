"""Prepare CAM's eleven dry-deposition surface classes from its PFT inventory.

The PFT-to-Wesely mapping is mo_drydep.F90::interp_map. Ocean cells are
water; residual uncovered land is water. Fractions are normalized on the target grid as in CAM.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import xarray as xr

SOURCE_URL = "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/cam/chem/trop_mozart/dvel/regrid_vegetation.nc"
SOURCE_SHA256 = "63436565807d15c76d2a443735fadbdeb04df311541e7296a78ed08ad953f0fe"


def prepare(source: str | Path) -> xr.Dataset:
    """Read the official, hash-pinned inventory and reduce it to eleven classes."""
    source = Path(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("CAM land-use inventory SHA256 mismatch")
    with xr.open_dataset(source) as ds:
        pft = np.asarray(ds.PCT_PFT, dtype=np.float64) / 100.0
        extra = [np.asarray(ds[k], dtype=np.float64) / 100.0
                 for k in ("PCT_LAKE", "PCT_WETLAND", "PCT_URBAN")]
        values = np.concatenate([pft, np.stack(extra)])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid CAM land-use fractions")
        values[17] += np.maximum(1.0 - values.sum(axis=0), 0.0)
        ocean = np.asarray(ds.LANDMASK) < 0.5
        values[:, ocean] = 0.0
        values[17, ocean] = 1.0
        # Fortran one-based PFT ranges: 20;16:17;13:15;5:9;2:4;19;18;1;
        # unused mixed/wet classes 9/10;10:12. Preserve that exact ordering.
        groups = [(19,), (15,16), (12,13,14), (4,5,6,7,8),
                  (1,2,3), (18,), (17,), (0,), (), (), (9,10,11)]
        classes = np.stack([values[list(g)].sum(axis=0) for g in groups])
        return xr.Dataset({"fraction_landuse": (("class", "lat", "lon"), classes)},
                          coords={"class": np.arange(1,12), "lat": ds.lat, "lon": ds.lon},
                          attrs={"source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                                 "mapping": "CAM mo_drydep.F90::interp_map (Wesely 11 classes)"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    data = prepare(args.source)
    data.fraction_landuse.attrs["units"] = "1"
    data.to_netcdf(args.output, encoding={"fraction_landuse":
                   {"dtype": "float32", "zlib": True, "complevel": 4}})
