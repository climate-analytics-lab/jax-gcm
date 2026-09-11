"""Aerosol regression statistics for a finished JAM run (#762).

Reduces a year-long JAM run directory to a small set of scalars that a
release gate can score, and applies the two tolerance tiers described in
``docs/source/design/jam_regression.md``:

* **absolute physics gates** — the exponential drift ``|d ln B/dt|`` of every
  species' burden over the final six months, the mass-budget residual, and the
  per-step dynamics residual from the #713 in-step gauge. These are the runaway
  detector: an aerosol runaway grows multiplicatively with the meteorology
  entirely normal, so a climatological range gate does not notice it until the
  final fortnight of a year that was unusable months earlier.
* **climatological anchors** — the shared ``jam_burden_report._SPECIES``
  ranges widened by the release gate's slack, unchanged.
* **regression tolerances** — ``max(3 sigma, 15 %)`` against a reference run
  (``--reference``), with ``sigma`` the standard error of the annual mean
  estimated from the chunk series' own lag-1 autocorrelation. No reference is
  vendored: the reference has to be a run that passes the absolute gates
  above, and none does yet (see the design doc's "Not yet done").

Everything is computed chunk by chunk: a year of T63 L47 JAM output is ~60 GB
and must never be held open as one array. :func:`collect` reduces each chunk
file to scalars and returns the 5-day series; :func:`summarize` turns that
series into the statistic set.

Both output vertical conventions are supported. Levels are chosen by their
*pressure*, read from the file's own ``pressure_full``, never by a bare index
(see the "Inspecting model output" rule in ``CLAUDE.md``), and the layer
thicknesses come from :func:`jam_burden_report._layer_dp`, which orients
pre-#710 TOA-first interfaces onto the surface-first ``level`` axis.

Usage:
    python tools/release_validation/aerosol_stats.py <run_dir>
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Source-checkout bootstrap: repo root (for ``jcm``) and tools/ (for the
# sibling ``jam_burden_report``) before the imports that need them.
_TOOLS = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jcm.analysis import area_weights, column_integral, global_mean  # noqa: E402
from jam_burden_report import _SPECIES, _layer_dp, burden  # noqa: E402

#: Species carrying a lifetime statistic — those whose removal is dominated by
#: the two published deposition ledgers (``dry_*`` sedimentation and ``wet_*``
#: scavenging, which already includes in-plume convective removal).
LIFETIME_SPECIES = ("so4", "bc", "du", "ss")

#: Species whose emitted mass is a complete source inventory, so
#: ``emi - dep - dB`` is expected to close. ``soa`` is excluded: its source is
#: condensation of the SOAG gas, which has no ``emi_*`` channel, so a residual
#: computed for it would measure a missing diagnostic, not a mass leak.
#: ``moa`` is excluded for a different reason — it has emission and deposition
#: ledgers but no entry in the shared ``_SPECIES`` anchor table, so no burden
#: is computed for it and there is no storage term to close against. Adding it
#: needs a climatological anchor range, which is a calibration decision.
PRIMARY_BUDGET_SPECIES = ("bc", "du", "ss", "poa")

#: Kilograms of sulfate AEROSOL per kilogram of each sulfur carrier, so the
#: whole family can be totalled in one mass unit. Read from jcm's own species
#: table rather than restated: the ``so4`` tracer is ammonium bisulfate, not
#: SO4, and converting SO2/DMS emissions with the SO4 molar mass instead
#: understates the sulfate source by 20 %.
from jcm.physics.aerosol.jam.species import SPECIES as _JAM_SPECIES  # noqa: E402

_M_SO4_AEROSOL = _JAM_SPECIES["so4"].molar_mass
_SULFUR_CARRIERS = {"so2": _M_SO4_AEROSOL / 0.0640648,     # 1.795
                    "dms": _M_SO4_AEROSOL / 0.0621324,     # 1.851
                    "h2so4": _M_SO4_AEROSOL / 0.0980784,   # 1.172
                    "so4": 1.0}
#: Gas reservoirs and emission channels of that family. The gases are stored
#: as raw column burdens by :func:`chunk_reduction`; the conversion above is
#: applied here, in the ledger, so the reduction stays unit-neutral.
_S_FAMILY_GASES = ("so2", "dms", "h2so4")
_S_FAMILY_EMIS = ("so2", "dms", "so4")

#: Dynamics-conservation gate. ``budget_dyn_<sp>`` (#713) is the mass the
#: transport created or destroyed per step, so ``|dyn|*dt/mass`` is the
#: fractional error semi-Lagrangian advection commits each step. Set AT that
#: error's honest magnitude, so it trips on a transport error that has become
#: systematic rather than on ordinary interpolation noise, and ~500x tighter
#: than the [2/3, 1.5] clip on the proportional mass fixer.
DYN_RESIDUAL_PER_STEP = 1.0e-3

#: Absolute physics gates (see the design doc). The drift limit admits a
#: factor e over ~500 days — slower than any runaway, faster than a burden that
#: has genuinely settled. It reads burdens only, so it stays trustworthy when
#: the deposition ledger is incomplete.
DRIFT_LIMIT_PER_DAY = 0.002
BUDGET_RESIDUAL_LIMIT = 0.05

#: The closure gate's sign carries the diagnosis. NEGATIVE means more mass was
#: deposited than entered the column, which no missing ledger entry can produce
#: — it is mass creation. POSITIVE means emitted mass is unaccounted for, which
#: a missing sink *diagnostic* looks exactly like: before the #722 ledger fix
#: ``dry_*`` omits Slinn deposition, so a positive residual on such output is
#: expected rather than evidence of a leak.
_POSITIVE_RESIDUAL_CAVEAT = (
    "unaccounted emission; on output written before the #722 removal-ledger "
    "fix, dry_* omits Slinn dry deposition and reads positive by construction")

#: Shortest window on which the drift and closure statistics are scored. A
#: least-squares slope has noise ``sigma_resid / (dt * sqrt(N(N^2-1)/12))``; at
#: the 5-day output cadence and the ~0.07 log-burden scatter of a settled
#: species, 3 sigma of that falls below ``DRIFT_LIMIT_PER_DAY`` only past ~90
#: days. On a shorter fit the statistic is noise, so it is reported UNSCORED
#: rather than gated — the same "not measurable is not scored" rule the drift
#: statistic already applies to a species the run does not carry. The closure
#: residual shares the limit: its storage term is one endpoint difference, which
#: over a few chunks swamps the flux integral it is compared against.
MIN_WINDOW_DAYS = 90.0

#: Regression tolerance: whichever of a 3-sigma excursion and a 15 % relative
#: change is larger. 3 sigma alone is far too tight for a well-sampled mean
#: (sigma is ~1 % of the annual mean); 15 % alone would let a slow bias
#: through on a noisy statistic.
_N_SIGMA = 3.0
_REL_TOLERANCE = 0.15

#: Statistics the regression tier does NOT score: each has an absolute physics
#: gate of its own, and their references are ~1e-3, so a 15 % relative
#: tolerance would be tighter than the gate and fail every real run.
_GATED_PREFIXES = ("dlnB_dt_", "budget_residual", "dyn_frac_per_step_")

#: Absolute tolerance floors [statistic units], so a reference that is legibly
#: zero (an unused species' burden) does not collapse the tolerance to zero and
#: fail on any nonzero value. Matched by prefix, longest first.
_TOLERANCE_FLOORS = {
    "burden_": 0.05,            # mg/m2
    "so4_burden_": 0.05,        # mg/m2
    "lifetime_": 0.1,           # days
    "aod_550": 0.002,
    "angstrom": 0.02,
    "cdnc_": 1.0,               # cm-3
    "N100_": 1.0,               # cm-3
    "r_dry_": 0.001,            # um
    "so4_frac_above_500hPa": 0.01,
    "so4_nh_sh_ratio": 0.05,
}

#: Pressure [Pa] used for the near-surface aerosol-number diagnostics, and the
#: divide separating the free troposphere reservoir from the boundary layer.
_CDNC_PRESSURE = 90000.0
_UPPER_LEVEL_PRESSURE = 50000.0

_AOD_KEYS = ("jam_optics.aod_550", "jam_band_optics.aod_550")
_ANGSTROM_KEYS = ("jam_optics.angstrom", "jam_band_optics.angstrom",
                  "ang4487aer")


def is_jam_run(ds: xr.Dataset) -> bool:
    """Report whether ``ds`` carries JAM's prognostic modal aerosol tracers.

    Detected from the variables present rather than from a config file, so a
    run directory alone is enough. The test is the interstitial mass tracers
    alone: they are what every statistic here is built from, and a MACv2-SP run
    (which publishes AOD but no ``m_<sp>_<mode>``) has none. Requiring a
    ``jam_*`` diagnostic namespace as well would let a JAM run with a trimmed
    output set skip the aerosol gates silently, which is the one outcome this
    block exists to prevent — a trimmed run is a misconfiguration to name (see
    :func:`missing_jam_diagnostics`), not a reason to skip.
    """
    return any(re.fullmatch(r"m_[a-z0-9]+_[a-z]+", str(v))
               for v in ds.data_vars)


def missing_jam_diagnostics(ds: xr.Dataset) -> list[str]:
    """JAM diagnostic namespaces absent from a run that carries JAM tracers.

    Their absence does not stop the burdens being computed, but it does silently
    drop the cloud-borne phase, the optics and the modal radii from the report,
    so it is named rather than left to be inferred from missing rows.
    """
    present = {str(v).split(".", 1)[0] for v in ds.data_vars if "." in str(v)}
    wanted = ("jam_cloud_borne", "jam_state")
    optics = ("jam_optics", "jam_band_optics")
    missing = [n for n in wanted if n not in present]
    if not any(n in present for n in optics):
        missing.append("/".join(optics))
    return missing


def _band_indices(ds: xr.Dataset, lo: float, hi: float) -> np.ndarray:
    lat = np.asarray(ds["lat"].values, dtype=float)
    return np.nonzero((lat >= lo) & (lat <= hi))[0]


def _band_mean(da: xr.DataArray, weights: xr.DataArray,
               ds: xr.Dataset, lo: float, hi: float) -> float:
    """Area-weighted mean of ``da`` over the latitude band [lo, hi]."""
    idx = _band_indices(ds, lo, hi)
    if idx.size == 0:
        return float("nan")
    return float(global_mean(da.isel(lat=idx), weights.isel(lat=idx)))


def nearest_pressure_level(ds: xr.Dataset, pressure: float) -> int:
    """Index on the ``level`` axis whose mean pressure is nearest ``pressure``.

    The index is *derived* from the file's own ``pressure_full`` rather than
    assumed, which is what makes the selection valid under either vertical
    convention.
    """
    pf = ds["pressure_full"]
    prof = pf.mean([d for d in pf.dims if d != "level"])
    return int(np.argmin(np.abs(np.asarray(prof.values) - pressure)))


def _optional_key(ds: xr.Dataset, keys) -> str | None:
    for key in keys:
        if key in ds:
            return key
    return None


def chunk_reduction(ds: xr.Dataset) -> dict[str, float]:
    """Reduce one output chunk to the scalars the statistics are built from.

    Returns global (and, for sulfate, hemispheric/polar/upper-level) means of
    burdens, the emission and deposition fluxes, and the optics/number
    diagnostics. Time is averaged out here: a chunk is one 5-day sample.
    """
    weights = area_weights(ds)
    dp = _layer_dp(ds)
    out: dict[str, float] = {}

    def tmean(da):
        return da.mean("time") if "time" in da.dims else da

    burdens = {}
    for species, (modes, _range) in _SPECIES.items():
        col = burden(ds, species, modes)
        if col is None:
            continue
        col = col.compute()
        burdens[species] = col
        out[f"burden_{species}"] = float(global_mean(col, weights))

    if "so4" in burdens:
        col = burdens["so4"]
        out["so4_nh"] = _band_mean(col, weights, ds, 0.0, 90.0)
        out["so4_sh"] = _band_mean(col, weights, ds, -90.0, 0.0)
        out["so4_60_90N"] = _band_mean(col, weights, ds, 60.0, 90.0)
        # Upper-level share: mask on the file's own pressure field, so no
        # level ordering is assumed.
        names = [f"{p}_so4_{m}"
                 for m in _SPECIES["so4"][0]
                 for p in ("m", "mc", "jam_cloud_borne.mc")]
        q = sum(ds[n] for n in names if n in ds)
        aloft = q.where(ds["pressure_full"] < _UPPER_LEVEL_PRESSURE, 0.0)
        col_aloft = tmean(column_integral(aloft, dp)) * 1e6
        out["so4_above_500hPa"] = float(global_mean(col_aloft, weights))

    # Sulfur-family gas burdens [mg/m2] as carried, for the closure test; the
    # conversion to sulfate-aerosol mass belongs to the ledger.
    for gas in _S_FAMILY_GASES:
        name = f"g_{gas}"
        if name in ds:
            col = tmean(column_integral(ds[name], dp)) * 1e6
            out[f"gas_{gas}"] = float(global_mean(col, weights))

    # #713 in-step budget gauges: column mass of the ADVECTED tracers and the
    # dynamics residual. Interstitial only (the host never transports the
    # cloud-borne carry), so the two share a denominator.
    for prefix in ("budget_mass", "budget_dyn"):
        for var in ds.data_vars:
            name = str(var)
            if name.startswith(f"{prefix}_") and ds[var].ndim <= 3:
                out[name] = float(global_mean(tmean(ds[var]), weights))

    for prefix in ("emi", "dry", "wet"):
        for var in ds.data_vars:
            name = str(var)
            # ``emi_bb_*`` comes along as a separate key; it is a *split* of
            # ``emi_*``, never added to it.
            if name.startswith(f"{prefix}_") and ds[var].ndim <= 3:
                out[name] = float(global_mean(tmean(ds[var]), weights))

    aod_key = _optional_key(ds, _AOD_KEYS)
    if aod_key:
        out["aod_550"] = float(global_mean(tmean(ds[aod_key]), weights))
    ang_key = _optional_key(ds, _ANGSTROM_KEYS)
    if ang_key:
        out["angstrom"] = float(global_mean(tmean(ds[ang_key]), weights))

    k_cdnc = (nearest_pressure_level(ds, _CDNC_PRESSURE)
              if "pressure_full" in ds else None)
    if k_cdnc is not None and "activated_cdnc" in ds:
        # m^-3 -> cm^-3.
        cdnc = tmean(ds["activated_cdnc"]).isel(level=k_cdnc)
        out["cdnc_900hPa_cm3"] = float(global_mean(cdnc, weights)) * 1e-6
    if k_cdnc is not None and "aerocom_N100" in ds:
        n100 = tmean(ds["aerocom_N100"]).isel(level=k_cdnc)
        out["N100_900hPa_cm3"] = float(global_mean(n100, weights)) * 1e-6

    if "jam_state.r_dry" in ds and "pressure_full" in ds:
        k_sfc = int(np.argmax(np.asarray(
            ds["pressure_full"].mean([d for d in ds["pressure_full"].dims
                                      if d != "level"]).values)))
        r_dry = tmean(ds["jam_state.r_dry"]).isel(level=k_sfc)
        for i, mode in enumerate(np.asarray(ds["mode"].values)):
            out[f"r_dry_{str(mode)}_um"] = float(
                global_mean(r_dry.isel(mode=i), weights)) * 1e6

    return out


def collect(files) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reduce every chunk file to scalars; return ``(days, series)``.

    Files are opened and closed one at a time — a year of JAM output does not
    fit in memory, and the statistics only ever need per-chunk reductions.
    """
    days: list[float] = []
    rows: list[dict[str, float]] = []
    for path in files:
        match = re.search(r"day(\d+)", str(path))
        with xr.open_dataset(path) as ds:
            rows.append(chunk_reduction(ds))
        days.append(float(match.group(1)) if match else float(len(days)))
    keys = sorted({k for row in rows for k in row})
    series = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float)
              for k in keys}
    return np.asarray(days, dtype=float), series


def log_drift(days: np.ndarray, values: np.ndarray,
              window_days: float = 182.5) -> float:
    """``d ln B / dt`` [1/day] from a least-squares fit over the last window.

    A logarithmic slope is the right statistic for an aerosol burden because
    the failure mode is multiplicative: a runaway grows by a fixed factor per
    unit time, so it shows up as a slope that is large regardless of the
    species' absolute loading, while a merely noisy but stationary burden
    averages to zero.
    """
    good = np.isfinite(values) & (values > 0)
    if good.sum() < 3:
        return float("nan")
    d, v = days[good], values[good]
    sel = d >= (d[-1] - window_days)
    if sel.sum() < 3:
        sel = np.ones_like(d, dtype=bool)
    return float(np.polyfit(d[sel], np.log(v[sel]), 1)[0])


def standard_error(values: np.ndarray) -> float:
    """Estimate the standard error of the mean of an autocorrelated series.

    ``N_eff = N (1 - a) / (1 + a)`` with ``a`` the lag-1 autocorrelation of the
    chunk series — the usual first-order correction, which is what makes the
    3-sigma tier meaningful for a burden whose 5-day samples are anything but
    independent. Negative ``a`` is clipped to zero (never claim *more*
    independent samples than there are chunks).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4 or np.allclose(v, v[0]):
        return 0.0 if n else float("nan")
    a = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    a = min(max(a, 0.0), 0.99)
    n_eff = max(n * (1.0 - a) / (1.0 + a), 1.0)
    return float(np.std(v, ddof=1) / np.sqrt(n_eff))


def tolerance_floor(name: str) -> float:
    """Absolute tolerance floor for a statistic, by longest matching prefix."""
    matches = [v for k, v in _TOLERANCE_FLOORS.items() if name.startswith(k)]
    return max(matches) if matches else 0.0


def regression_tolerance(reference: float, sigma: float,
                         floor: float = 0.0) -> float:
    """``max(3 sigma, 15 % of the reference, an absolute floor)``.

    The floor is what keeps a legitimately-zero reference — an unused species'
    burden — from collapsing the tolerance to zero and failing on any nonzero
    value. See the design doc.
    """
    return max(_N_SIGMA * sigma, _REL_TOLERANCE * abs(reference), floor)


def _budget_residual(days, series, species) -> float | None:
    """Compute ``(emitted - deposited - dB) / emitted`` over the record.

    The fluxes are chunk means of ``kg/m2/s``, so their record mean times the
    record length is the time integral; ``dB`` is the change in the chunk-mean
    burden between the first and last chunk. For sulfate the ledger is the
    whole sulfur family in SO4-equivalent mass (SO2 and DMS emissions are the
    real sulfate source; ``emi_so4`` alone is a few per cent of it), and the
    stored mass includes the gas-phase reservoirs.
    """
    span = float(days[-1] - days[0])
    if span <= 0 or days.size < 2:
        return None

    def integral(key):
        # The storage term differences the first and last chunk MEANS, i.e.
        # chunk centres, so the flux integral spans the same interval: the
        # chunks after the first, times the centre-to-centre span. Averaging
        # all N chunks over an (N-1)-chunk span understates it by 1/N.
        v = series.get(key)
        return None if v is None else float(np.nanmean(v[1:])) * span * 86400e6

    if species == "so4":
        # Every carrier converted to sulfate-AEROSOL mass, the unit the
        # burden and the deposition ledger are already in.
        contributions = [integral(f"emi_{c}") for c in _S_FAMILY_EMIS]
        if any(e is None for e in contributions):
            return None
        emitted = sum(e * _SULFUR_CARRIERS[c]
                      for e, c in zip(contributions, _S_FAMILY_EMIS))
        stored = series.get("burden_so4")
        if stored is None:
            return None
        for gas in _S_FAMILY_GASES:
            reservoir = series.get(f"gas_{gas}")
            if reservoir is not None:
                stored = stored + reservoir * _SULFUR_CARRIERS[gas]
    else:
        emitted = integral(f"emi_{species}")
        stored = series.get(f"burden_{species}")
        if emitted is None or stored is None:
            return None

    deposited = 0.0
    for prefix in ("dry", "wet"):
        contribution = integral(f"{prefix}_{species}")
        if contribution is not None:
            deposited += contribution
    if not np.isfinite(emitted) or emitted <= 0:
        return None
    residual = float((emitted - deposited - (stored[-1] - stored[0])) / emitted)
    # The storage endpoints are as capable of being NaN as the emission was: a
    # chunk missing a burden or a gas reservoir poisons the difference. A NaN
    # here would score as a gate FAIL for something never measured, and would
    # win ``max(..., key=abs)`` too, since every comparison against NaN is
    # False.
    return residual if np.isfinite(residual) else None


def summarize(days: np.ndarray, series: dict[str, np.ndarray],
              timestep_seconds: float | None = None) -> dict[str, float]:
    """Build the statistic set from the series returned by :func:`collect`.

    Every entry is a single number a gate or a stored reference can be
    compared against; the rationale for each is in the design doc.
    """
    stats: dict[str, float] = {}
    span_days = float(days[-1] - days[0]) if days.size > 1 else 0.0

    for species in _SPECIES:
        b = series.get(f"burden_{species}")
        if b is None:
            continue
        stats[f"burden_{species}_mg_m2"] = float(np.nanmean(b))
        # Emit the drift only where it is measurable: a species the run does
        # not carry yields NaN, and a window shorter than MIN_WINDOW_DAYS
        # yields a slope dominated by its own noise. Both are reported
        # unscored (see :func:`unscored_gates`) rather than gated, because a
        # gate FAIL for something never measured is worse than no number.
        drift = log_drift(days, b)
        if np.isfinite(drift) and span_days >= MIN_WINDOW_DAYS:
            stats[f"dlnB_dt_{species}_per_day"] = drift

    for species in LIFETIME_SPECIES:
        b = series.get(f"burden_{species}")
        if b is None:
            continue
        # ``wet_*`` already includes in-plume convective scavenging (the
        # transport term's flux is folded into it in wetdep_term.py), so
        # adding ``conv_scav_flux.*`` here would double-count that sink.
        sink = 0.0
        for prefix in ("dry", "wet"):
            v = series.get(f"{prefix}_{species}")
            if v is not None:
                sink += float(np.nanmean(v)) * 86400e6   # kg/m2/s -> mg/m2/day
        if sink > 0:
            stats[f"lifetime_{species}_days"] = float(np.nanmean(b)) / sink

    residuals = {}
    for species in ("so4",) + PRIMARY_BUDGET_SPECIES:
        r = (_budget_residual(days, series, species)
             if span_days >= MIN_WINDOW_DAYS else None)
        if r is not None:
            residuals[species] = r
            stats[f"budget_residual_{species}"] = r
    if residuals:
        stats["budget_residual_max"] = max(residuals.values(), key=abs)

    if "so4_above_500hPa" in series and "burden_so4" in series:
        aloft = float(np.nanmean(series["so4_above_500hPa"]))
        total = float(np.nanmean(series["burden_so4"]))
        if total > 0:
            stats["so4_frac_above_500hPa"] = aloft / total
    if "so4_nh" in series and "so4_sh" in series:
        sh = float(np.nanmean(series["so4_sh"]))
        if sh > 0:
            stats["so4_nh_sh_ratio"] = float(np.nanmean(series["so4_nh"])) / sh
    if "so4_60_90N" in series:
        stats["so4_burden_60_90N_mg_m2"] = float(np.nanmean(
            series["so4_60_90N"]))

    for key in ("aod_550", "angstrom", "cdnc_900hPa_cm3", "N100_900hPa_cm3"):
        if key in series:
            stats[key] = float(np.nanmean(series[key]))
    for key in series:
        if key.startswith("r_dry_"):
            stats[key] = float(np.nanmean(series[key]))

    # Dynamics conservation: the transport residual as a fraction of the
    # advected mass, per step. Needs the run's timestep; without it the
    # statistic is not emitted (and its gate is therefore not scored).
    if timestep_seconds:
        for key in series:
            if not key.startswith("budget_dyn_"):
                continue
            species = key[len("budget_dyn_"):]
            mass = series.get(f"budget_mass_{species}")
            if mass is None:
                continue
            mean_mass = float(np.nanmean(mass))
            if mean_mass > 0:
                stats[f"dyn_frac_per_step_{species}"] = abs(
                    float(np.nanmean(series[key])) * timestep_seconds
                    / mean_mass)

    stats["record_days"] = span_days
    return stats


def timestep_seconds(run_dir: str) -> float | None:
    """Read the run.s timestep [s] from the Hydra config saved beside it.

    Returned as ``None`` when the run directory has no ``.hydra/config.yaml``
    (a hand-assembled directory, or output copied without it), which makes the
    dynamics-conservation gate unscored rather than wrong.
    """
    cfg = Path(run_dir) / ".hydra" / "config.yaml"
    if not cfg.is_file():
        return None
    import yaml
    loaded = yaml.safe_load(cfg.read_text()) or {}
    minutes = (loaded.get("run") or {}).get("time_step")
    return float(minutes) * 60.0 if minutes else None


def compare_to_reference(stats: dict[str, float],
                         reference: dict[str, float],
                         series: dict[str, np.ndarray] | None = None
                         ) -> list[tuple[str, float, str, bool]]:
    """Score this run's statistics against a stored reference (tier 3).

    The tolerance is ``max(3 sigma, 15 %)``. ``sigma`` comes from the *current*
    run's own chunk series where one is available, so a statistic that is
    genuinely noisy in this configuration is not held to the precision of a
    quiet one; without a series the 15 % floor decides. Returns the same
    ``(name, value, limit, ok)`` rows the gate functions use.
    """
    rows = []
    for key in sorted(reference):
        if key not in stats or key == "record_days":
            continue
        # Statistics with an absolute physics gate are not scored here too:
        # their references are ~1e-3, so a relative tolerance would be tighter
        # than their own gate.
        if key.startswith(_GATED_PREFIXES):
            continue
        value, ref = stats[key], reference[key]
        sigma = 0.0
        if series is not None:
            for candidate in (key, key.removesuffix("_mg_m2"),
                              f"burden_{key.removeprefix('burden_')}"):
                if candidate in series:
                    sigma = standard_error(series[candidate])
                    break
        tol = regression_tolerance(ref, sigma, tolerance_floor(key))
        ok = abs(value - ref) <= tol
        rows.append((key, value, f"{ref:.5g} +- {tol:.3g}", bool(ok)))
    return rows


def unscored_gates(days: np.ndarray, series: dict[str, np.ndarray],
                   timestep_seconds: float | None = None
                   ) -> list[tuple[str, str]]:
    """Gates that could NOT be evaluated, each with the reason.

    A gate that is absent from :func:`physics_gates` has passed nothing — it
    was never run. Reporting the absence is what stops a run with no dynamics
    gauge, or a window too short to fit a slope, from reading as clean.
    Returns ``(gate_name, reason)`` pairs for the caller to print.
    """
    span = float(days[-1] - days[0]) if days.size > 1 else 0.0
    short_slope = (f"window spans {span:.0f} days; a slope needs "
                   f"{MIN_WINDOW_DAYS:.0f} to rise above its own fit noise")
    short_budget = (f"window spans {span:.0f} days; the storage endpoints "
                    f"swamp the flux integral below {MIN_WINDOW_DAYS:.0f}")
    rows: list[tuple[str, str]] = []

    for species in _SPECIES:
        b = series.get(f"burden_{species}")
        if b is None:
            continue
        name = f"dlnB_dt_{species}_per_day"
        if not np.any(np.isfinite(b) & (b > 0)):
            rows.append((name, "the run carries no burden of this species"))
        elif span < MIN_WINDOW_DAYS:
            rows.append((name, short_slope))

    if span < MIN_WINDOW_DAYS:
        rows.append(("budget_residual_max", short_budget))
    else:
        # A species missing from ``budget_residual_max`` is a species whose
        # mass was never checked, and the maximum over an incomplete subset
        # cannot see a leak confined to the species it omits. Report each
        # omission, and the aggregate when nothing at all could be closed.
        closed = []
        for species in ("so4",) + PRIMARY_BUDGET_SPECIES:
            if f"burden_{species}" not in series:
                continue        # the run does not carry it; nothing to close
            if _budget_residual(days, series, species) is None:
                rows.append((
                    f"budget_residual_{species}",
                    "the mass ledger is incomplete — a missing emi_*/dry_*/"
                    "wet_* diagnostic, or a non-finite burden endpoint"))
            else:
                closed.append(species)
        if not closed:
            rows.append(("budget_residual_max",
                         "no species had a complete mass ledger to close "
                         "against; no closure was checked"))

    # The dynamics gate needs BOTH the #713 in-step gauge and a timestep to
    # express it per step; say which is missing rather than omitting the row.
    has_gauge = any(k.startswith("budget_dyn_") for k in series)
    if not has_gauge:
        rows.append(("dyn_frac_per_step", "the run publishes no budget_dyn_* "
                                          "gauge (output predates #713)"))
    elif not timestep_seconds:
        rows.append(("dyn_frac_per_step", "no run timestep is available "
                     "(.hydra/config.yaml absent, or run.time_step is null as "
                     "on the pySES backend), and the gate is per step"))
    return rows


def physics_gates(stats: dict[str, float]) -> list[tuple[str, float, str, bool]]:
    """Score the absolute physics gates; returns ``(name, value, gate, ok)``.

    These are not tuning targets: a burden that grows exponentially, or a
    species whose mass ledger does not close, is a defect at any loading.
    """
    rows = []
    for key, value in sorted(stats.items()):
        if key.startswith("dlnB_dt_"):
            ok = np.isfinite(value) and abs(value) < DRIFT_LIMIT_PER_DAY
            rows.append((key, value, f"|x| < {DRIFT_LIMIT_PER_DAY:g} /day",
                         bool(ok)))
    for key, value in sorted(stats.items()):
        if key.startswith("dyn_frac_per_step_"):
            ok = np.isfinite(value) and value < DYN_RESIDUAL_PER_STEP
            rows.append((key, value,
                         f"x < {DYN_RESIDUAL_PER_STEP:.1%}/step", bool(ok)))
    if "budget_residual_max" in stats:
        value = stats["budget_residual_max"]
        ok = np.isfinite(value) and abs(value) < BUDGET_RESIDUAL_LIMIT
        limit = f"|x| < {BUDGET_RESIDUAL_LIMIT:.0%}"
        if not ok and np.isfinite(value) and value > 0:
            limit += f" ({_POSITIVE_RESIDUAL_CAVEAT})"
        rows.append(("budget_residual_max", value, limit, bool(ok)))
    return rows


def format_table(stats: dict[str, float],
                 gates: list[tuple[str, float, str, bool]]) -> str:
    """Human-readable report of the statistic set and the physics gates."""
    lines = [f"{'statistic':<34}{'value':>14}", "-" * 48]
    for key, value in sorted(stats.items()):
        if isinstance(value, str):
            lines.append(f"{key:<34}{value:>14}")
        else:
            lines.append(f"{key:<34}{value:>14.5g}")
    if not gates:
        return "\n".join(lines)
    lines += ["", f"{'physics gate':<34}{'value':>14}  {'limit':<20}result",
              "-" * 76]
    notes = []
    for name, value, limit, ok in gates:
        head, _, note = limit.partition(" (")
        lines.append(f"{name:<34}{value:>14.5g}  {head:<20}"
                     f"{'PASS' if ok else 'FAIL'}")
        if note:
            notes.append(f"NOTE  {name}: {note.rstrip(')')}")
    lines += notes
    return "\n".join(lines)


#: A saved chunk is ``<prefix>_day<N>.nc``. ``run.snapshot_interval`` writes
#: ``<prefix>_day<N>_snapshots.nc`` beside it, which a ``*_day*.nc`` glob also
#: matches — those are 2-D snapshot streams with no ``pressure_half``, so they
#: would break the column integral, and any that did parse would double-count
#: the day. Anchored on the extension, they are excluded.
CHUNK_FILE = re.compile(r"_day(\d+)\.nc$")


def run_files(run_dir: str) -> list[str]:
    """Chunk files of a run directory, in day order (snapshots excluded)."""
    matched = ((m, f) for f, m in
               ((f, CHUNK_FILE.search(f)) for f in glob.glob(f"{run_dir}/*.nc"))
               if m)
    return [f for _m, f in sorted(matched, key=lambda mf: int(mf[0].group(1)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--last-n", type=int, default=None,
                    help="use only the last N chunks (default: all)")
    ap.add_argument("--series-out", default=None,
                    help="save the per-chunk reductions to this .npz")
    ap.add_argument("--series-in", default=None,
                    help="re-score a saved .npz instead of reading the run "
                         "(reducing a year of JAM output takes ~40 min)")
    ap.add_argument("--reference", default=None,
                    help="a .npz of a reference run's series; adds the "
                         "regression comparison (max(3 sigma, 15 %%))")
    ap.add_argument("--timestep-minutes", type=float, default=None,
                    help="override the run timestep the dynamics gate needs "
                         "(default: the run's Hydra config, or the one saved "
                         "with --series-out)")
    args = ap.parse_args()

    dt = None
    if args.series_in:
        loaded = np.load(args.series_in)
        days = loaded["_days"]
        # Underscore keys are the reduction's own metadata, not statistics.
        series = {k: loaded[k] for k in loaded.files if not k.startswith("_")}
        # The timestep travels WITH the reduction: re-scoring a saved series
        # has no run directory to read it from, and without it every
        # dynamics-conservation gate would silently drop out of a re-score
        # that the original scored.
        if "_timestep_seconds" in loaded.files:
            stored = float(loaded["_timestep_seconds"])
            dt = stored if np.isfinite(stored) and stored > 0 else None
        if args.last_n:
            # Applied here too: silently ignoring it would score a different
            # window than the one asked for.
            days = days[-args.last_n:]
            series = {k: v[-args.last_n:] for k, v in series.items()}
    else:
        files = run_files(args.run_dir)
        if not files:
            print(f"FAIL  no chunk files in {args.run_dir}")
            return 1
        if args.last_n:
            files = files[-args.last_n:]
        days, series = collect(files)
        dt = timestep_seconds(args.run_dir)
    if args.timestep_minutes:
        dt = args.timestep_minutes * 60.0
    if args.series_out:
        np.savez(args.series_out, _days=days,
                 _timestep_seconds=np.array(dt if dt else np.nan), **series)
    stats = summarize(days, series, dt)
    gates = physics_gates(stats)
    print(format_table(stats, gates))
    unscored = unscored_gates(days, series, dt)
    for name, reason in unscored:
        print(f"UNSCORED  {name}: {reason}")
    ok = all(row[3] for row in gates)

    if args.reference:
        loaded = np.load(args.reference)
        ref_series = {k: loaded[k] for k in loaded.files if k != "_days"}
        ref = summarize(loaded["_days"], ref_series)
        rows = compare_to_reference(stats, ref, series)
        print(f"\n{'regression vs reference':<34}{'value':>14}  "
              f"{'expected':<24}result")
        print("-" * 80)
        for name, value, limit, good in rows:
            print(f"{name:<34}{value:>14.5g}  {limit:<24}"
                  f"{'PASS' if good else 'FAIL'}")
        ok = ok and all(row[3] for row in rows)

    # Say how much was actually measured: a PASS over zero scored gates is
    # not the same result as a PASS over all of them.
    print(f"\nAEROSOL: {'PASS' if ok else 'FAIL'} "
          f"({len(gates)} gate(s) scored, {len(unscored)} unscored)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
