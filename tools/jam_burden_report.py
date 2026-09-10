"""Column-burden report for JAM aerosol runs — any dycore, any grid.

Reads jcm output netCDF(s), sums interstitial + cloud-borne mass over the
modes carrying each species, integrates ``q·dp/g`` over the column with the
file's own ``pressure_thickness`` diagnostic (falling back to differencing
``pressure_half``), and prints time-mean global burdens against
climatological anchor ranges. With ``--emissions-file`` it also prints each
primarily-emitted species' inferred lifetime
(burden / global-mean primary emission rate).

Usage:
    python tools/jam_burden_report.py out_day*.nc
        [--emissions-file emis.nc] [--png burdens.png]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import xarray as xr

# Source-checkout bootstrap: allow ``python tools/jam_burden_report.py`` when
# jcm is not pip-installed by putting the repo root on sys.path before the jcm
# import below (mirrors the idiom in tools/radiation_emulator/*).
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from jcm.analysis import (  # noqa: E402
    area_weights,
    column_integral,
    global_mean,
    layer_pressure_thickness,
)

# species -> (modes carrying it, (lo, hi) mg/m² global-mean anchor range;
# HAMMOZ/CESM climatology magnitudes).
_SPECIES = {
    "so4": (("acc", "ait", "cor"), (2.0, 4.0)),
    "bc": (("acc", "cor", "pcm"), (0.1, 0.3)),
    "du": (("acc", "cor"), (5.0, 20.0)),
    "ss": (("acc", "ait", "cor"), (10.0, 20.0)),
    "poa": (("acc", "cor", "pcm"), (1.0, 3.0)),
    "soa": (("acc", "ait", "cor"), (0.5, 3.0)),
}

# Emission channels are speciated as SO2/BC/OC; map each burden species to
# its primary channel. so4's sulfur arrives as SO2 — its source is reported
# as potential sulfate (× 96/64 by molar mass).
_EMIS_SPECIES = {"so4": ("so2", 96.0 / 64.0), "bc": ("bc", 1.0),
                 "poa": ("oc", 1.0)}


# Weighting / layer-Δp / column integration are the shared post-processing
# machinery in ``jcm.analysis`` (#640) — kept as thin wrappers here so the
# report and the release gates (which import them from this module) call one
# implementation. ``_area_weights`` preserves the tool's "no lat -> None"
# contract; ``jcm.analysis.area_weights`` returns Gauss-Legendre-exact weights
# for dinosaur output grids (cos(lat) only for non-Gaussian grids).
_layer_dp = layer_pressure_thickness


def _area_weights(ds: xr.Dataset):
    if "lat" in getattr(ds, "coords", {}):
        return area_weights(ds)
    return None


def _wmean(da: xr.DataArray, weights) -> float:
    return float(global_mean(da, weights))


def burden(ds: xr.Dataset, species: str, modes) -> xr.DataArray | None:
    """Time-mean column burden [mg/m²] of a species summed over modes."""
    names = [f"{p}_{species}_{m}" for m in modes for p in ("m", "mc")]
    present = [n for n in names if n in ds]
    if not present:
        return None
    q = sum(ds[n] for n in present)
    col = column_integral(q, _layer_dp(ds)) * 1e6   # kg/m² -> mg/m²
    return col.mean("time") if "time" in col.dims else col


def emission_rate(emis: xr.Dataset, channel: str, weights) -> float | None:
    """Global-mean emission of one channel [mg/m²/day], summed over sectors."""
    fields = [v for v in emis.data_vars
              if str(v).startswith("emis_") and str(v).endswith(f"_{channel}")]
    if not fields:
        return None
    total = sum(emis[v] for v in fields)          # kg/m²/s
    if "time" in total.dims:
        total = total.mean("time")
    return _wmean(total, weights) * 86400.0 * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--emissions-file", default=None,
                    help="jcm emissions netCDF; adds an inferred-lifetime "
                         "column (burden / primary emission rate)")
    ap.add_argument("--png", default=None, help="optional burden-map figure")
    args = ap.parse_args()

    ds = xr.open_mfdataset(args.files, combine="nested", concat_dim="time")
    weights = _area_weights(ds)

    sources: dict[str, float] = {}
    if args.emissions_file:
        with xr.open_dataset(args.emissions_file) as emis:
            ew = _area_weights(emis)
            for sp, (channel, scale) in _EMIS_SPECIES.items():
                rate = emission_rate(emis, channel, ew)
                if rate:
                    sources[sp] = scale * rate

    header = f"{'species':>8} {'global mean':>12} {'max':>10}   anchor [mg/m²]"
    if sources:
        header += "   lifetime [d]"
    print(header)
    maps = {}
    for sp, (modes, (lo, hi)) in _SPECIES.items():
        col = burden(ds, sp, modes)
        if col is None:
            print(f"{sp:>8} {'— no tracers in file —':>24}")
            continue
        col = col.compute()
        gmean = _wmean(col, weights)
        flag = "OK" if lo <= gmean <= hi else ("LOW" if gmean < lo else "HIGH")
        line = (f"{sp:>8} {gmean:12.3f} {float(col.max()):10.2f}   "
                f"[{lo:g}–{hi:g}] {flag}")
        if sp in sources:
            line += f"   {gmean / sources[sp]:8.1f}"
        print(line)
        maps[sp] = col

    if args.png and maps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(maps)
        fig, axes = plt.subplots((n + 1) // 2, 2,
                                 figsize=(12, 3 * ((n + 1) // 2)),
                                 constrained_layout=True)
        for ax, (sp, b) in zip(np.ravel(axes), maps.items()):
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
