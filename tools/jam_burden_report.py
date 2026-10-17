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
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES  # noqa: E402
from jcm.physics.aerosol.jam.microphysics import M7_SPEC, MAM4_SPEC  # noqa: E402

# species -> (modes carrying it, (lo, hi) mg/m² global-mean anchor range;
# HAMMOZ/CESM climatology magnitudes). The modes are MAM4's, then M7's (#1017);
# a run carries one population, so the sum picks up its own. M7's ``oc`` is
# its whole organic aerosol (primary plus biogenic SOA emitted as primary), so
# its range is the sum of MAM4's ``poa`` and ``soa`` ranges.
_SPECIES = {
    "so4": (("acc", "ait", "cor", "ns", "ks", "as", "cs"), (2.0, 4.0)),
    "bc": (("acc", "cor", "pcm", "ks", "as", "cs", "ki"), (0.1, 0.3)),
    "du": (("acc", "cor", "as", "cs", "ai", "ci"), (5.0, 20.0)),
    "ss": (("acc", "ait", "cor", "as", "cs"), (10.0, 20.0)),
    "poa": (("acc", "cor", "pcm"), (1.0, 3.0)),
    "soa": (("acc", "ait", "cor"), (0.5, 3.0)),
    "oc": (("ks", "as", "cs", "ki"), (1.5, 6.0)),
}

# Emission channels are speciated as SO2/BC/OC; map each burden species to
# its primary channel and the kilograms of that species one kilogram of the
# channel becomes. so4's sulfur arrives as SO2, so its factor is the potential
# sulfate per SO2 by molar mass — which depends on the population that wrote
# the file (:func:`sulfate_molar_mass`), so it is filled in from the file
# rather than fixed here.
_EMIS_SPECIES = {"so4": ("so2", None), "bc": ("bc", 1.0),
                 "poa": ("oc", 1.0), "oc": ("oc", 1.0)}

#: The modal populations a JAM run can carry. Both call their sulfate tracer
#: ``so4`` but carry different compounds under that name: MAM4's is ammonium
#: bisulfate (115 g/mol, ``jam/species.py``), M7's is SO4 itself
#: (96.0631 g/mol, HAM's ``mo_ham.f90::mw_so4``). The model converts sulfur
#: into each with the population's own molar mass, so any sulfur budget over
#: the output must use the same one. The modes carrying ``so4`` do not overlap
#: between the two (MAM4's acc/ait/cor, M7's ns/ks/as/cs), which is what lets a
#: file's tracers say which population wrote it.
_POPULATIONS = (MAM4_SPEC, M7_SPEC)


def sulfate_molar_mass(ds: xr.Dataset) -> float | None:
    """Molar mass [kg/mol] of the compound the file's ``so4`` tracers carry.

    Read from the population whose ``so4``-carrying modes the file's
    interstitial ``m_so4_<mode>`` tracers are on. ``None`` when the file
    carries no sulfate tracer, or tracers of more than one population: no
    single conversion is then right, and a budget built on a guessed one is a
    wrong number, not a missing one.
    """
    matched = [spec for spec in _POPULATIONS
               if any(f"m_so4_{mode.short}" in ds
                      for mode in spec.classes_for("so4"))]
    if len(matched) != 1:
        return None
    return matched[0].species_props("so4").molar_mass


def sulfate_per_so2(ds: xr.Dataset) -> float | None:
    """Kilograms of the file's sulfate one kilogram of SO2 becomes, or ``None``."""
    m_sulfate = sulfate_molar_mass(ds)
    if m_sulfate is None:
        return None
    return m_sulfate / GAS_SPECIES["so2"].molar_mass


# Weighting / layer-Δp / column integration are the shared post-processing
# machinery in ``jcm.analysis`` (#640) — kept as thin wrappers here so the
# report and the release gates (which import them from this module) call one
# implementation. ``_area_weights`` preserves the tool's "no lat -> None"
# contract; ``jcm.analysis.area_weights`` returns Gauss-Legendre-exact weights
# for dinosaur output grids (cos(lat) only for non-Gaussian grids).
def _interfaces_are_toa_first(ds: xr.Dataset) -> bool:
    """Report whether ``pressure_half`` runs top-down (pre-#710 files).

    Read from the pressure values themselves rather than from the axis
    labels, so a file whose ``level_i`` is a bare integer index is still
    oriented correctly. See ``docs/source/design/jam_regression.md``.
    """
    ph = ds["pressure_half"]
    other = [d for d in ph.dims if d != "level_i"]
    prof = np.asarray(ph.mean(other).values) if other else np.asarray(ph.values)
    return bool(prof[0] < prof[-1])


def _layer_dp(ds: xr.Dataset) -> xr.DataArray:
    """Per-layer Δp [Pa] aligned with the ``level`` axis of the 3-D fields.

    Post-#710 files run both vertical axes surface-first, which is what
    :func:`jcm.analysis.layer_pressure_thickness` assumes. Pre-#710 files
    store interfaces TOA-first while ``level`` fields stay surface-first, so
    the differenced Δp is reversed to align; without that, ``q·Δp`` pairs the
    thinnest stratospheric layers with the boundary layer and every burden is
    wrong. The model's own ``pressure_thickness`` diagnostic is already on the
    ``level`` axis and never needs the flip.
    """
    dp = layer_pressure_thickness(ds)
    if "pressure_thickness" not in ds and _interfaces_are_toa_first(ds):
        return dp.isel(level=slice(None, None, -1))
    return dp


def _area_weights(ds: xr.Dataset):
    if "lat" in getattr(ds, "coords", {}):
        return area_weights(ds)
    return None


def _wmean(da: xr.DataArray, weights) -> float:
    return float(global_mean(da, weights))


def burden(ds: xr.Dataset, species: str, modes) -> xr.DataArray | None:
    """Time-mean column burden [mg/m²] of a species summed over modes."""
    # Interstitial ``m_<sp>_<mode>`` are top-level tracers; the cloud-borne
    # phase is a carry diagnostic and is written under the flattened
    # ``jam_cloud_borne.`` namespace (bare ``mc_`` kept for hand-built files).
    names = [f"{p}_{species}_{m}"
             for m in modes
             for p in ("m", "mc", "jam_cloud_borne.mc")]
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
                if scale is None:
                    # so4: the run's own sulfate, identified from its tracers.
                    scale = sulfate_per_so2(ds)
                    if scale is None:
                        continue
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
