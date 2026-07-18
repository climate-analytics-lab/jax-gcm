"""Column-burden sanity report for pySES JAM runs.

Reads one or more chunk netCDFs from ``tools/run_pyses_climatology.py``
(regular lat/lon output, hybrid tables attached by
``PysesCamSEDycore.to_xarray``), sums interstitial + cloud-borne mass over
all modes per species, integrates ``q·dp/g`` over the column, and prints
time-mean global burdens against the HAMMOZ/CESM climatological anchors —
the §8 validation gate for the online-aerosol wiring (BC/SO4/dust must come
alive, sea salt must stay in range).

Usage:
    python tools/jam_burden_report.py out_day*.nc [--png burdens.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GRAV = 9.80665

# species -> (modes carrying it, (lo, hi) mg/m² global-mean anchor range).
# Anchors are the handoff's HAMMOZ/CESM climatology magnitudes.
_SPECIES = {
    "so4": (("acc", "ait", "cor"), (2.0, 4.0)),
    "bc": (("acc", "cor", "pcm"), (0.1, 0.3)),
    "du": (("acc", "cor"), (5.0, 20.0)),
    "ss": (("acc", "ait", "cor"), (10.0, 20.0)),
    "poa": (("acc", "cor", "pcm"), (1.0, 3.0)),
}


def burden(ds: xr.Dataset, species: str, modes) -> xr.DataArray | None:
    """Time-mean column burden [mg/m²] of a species summed over modes."""
    from jcm.dycore.pyses.coords import full_echam_hybrid

    names = [f"{p}_{species}_{m}" for m in modes for p in ("m", "mc")]
    present = [n for n in names if n in ds]
    if not present:
        return None
    q = sum(ds[n] for n in present)              # (time, level, lon, lat) kg/kg
    a_b, b_b = full_echam_hybrid(ds.sizes["level"])
    # Output level axis is surface-first; boundaries are top-first.
    da = np.diff(np.asarray(a_b))[::-1]
    db = np.diff(np.asarray(b_b))[::-1]
    ps = ds["surface_pressure"]                  # (time, lon, lat) Pa
    dp = (xr.DataArray(da, dims="level") +
          xr.DataArray(db, dims="level") * ps)
    return (q * dp / GRAV).sum("level").mean("time") * 1e6   # kg/m² -> mg/m²


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--png", default=None, help="optional burden-map figure")
    args = ap.parse_args()

    ds = xr.open_mfdataset(args.files, combine="by_coords")
    w = np.cos(np.deg2rad(ds["lat"]))            # regridded regular lat/lon

    print(f"{'species':>8} {'global mean':>12} {'max':>10}   anchor [mg/m²]")
    maps = {}
    for sp, (modes, (lo, hi)) in _SPECIES.items():
        b = burden(ds, sp, modes)
        if b is None:
            print(f"{sp:>8} {'— no tracers in file —':>24}")
            continue
        b = b.compute()
        gmean = float(b.weighted(w).mean())
        flag = "OK" if lo <= gmean <= hi else ("LOW" if gmean < lo else "HIGH")
        print(f"{sp:>8} {gmean:12.3f} {float(b.max()):10.2f}   "
              f"[{lo:g}–{hi:g}] {flag}")
        maps[sp] = b

    if args.png and maps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(maps)
        fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 3 * ((n + 1) // 2)),
                                 constrained_layout=True)
        for ax, (sp, b) in zip(np.ravel(axes), maps.items()):
            # (lon, lat) -> plot as (lat, lon)
            pm = ax.pcolormesh(b["lon"], b["lat"], b.transpose("lat", "lon"),
                               shading="auto")
            ax.set_title(f"{sp} burden [mg/m²]")
            fig.colorbar(pm, ax=ax)
        for ax in np.ravel(axes)[n:]:
            ax.axis("off")
        fig.savefig(args.png, dpi=120)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
