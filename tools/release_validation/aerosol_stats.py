"""Aerosol regression statistics for a finished JAM run (#762).

Reduces a year-long JAM run directory to a small set of scalars that a
release gate can score, and applies the two tolerance tiers described in
``docs/source/design/jam_regression.md``:

* **absolute physics gates** — the exponential drift ``|d ln B/dt|`` of every
  species' burden over the final six months, the mass-budget residual, and the
  per-step dynamics residual from the #713 in-step gauge.
  These are the runaway detector: the August-2026 T63 L47 year grew sulfate
  13 -> 771 mg/m2 in its last forty days while temperature and surface
  pressure stayed flat, and a climatological range gate with x3 slack only
  notices that in the final fortnight.
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
#: whole family can be totalled in one mass unit. jcm's ``so4`` tracer is
#: ammonium bisulfate NH4HSO4 at 115.0 g/mol (``jam/species.py``) — not SO4 at
#: 96.06 — so converting SO2 (64.06) and DMS (62.13) emissions with the sulfate
#: molar mass instead understates the source by 20 % and turns a modest closure
#: error into an apparent leak.
_M_SO4_AEROSOL = 0.115
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
#: transport machinery created or destroyed per step, so ``|dyn|*dt/mass`` is
#: the fractional error the semi-Lagrangian advection commits each step. Its
#: honest magnitude is O(1e-3) per step and one-signed under a strong sink,
#: which is how it compounded into the August-2026 runaway; 0.1 %/step is
#: therefore set AT that magnitude, to trip on a transport error that has
#: become systematic rather than on ordinary interpolation noise. It is also
#: ~500x tighter than the [2/3, 1.5] clip on dev's proportional mass fixer,
#: so the gate fires long before the fixer saturates.
DYN_RESIDUAL_PER_STEP = 1.0e-3

#: Absolute physics gates (see the design doc). The drift limit is set so a
#: burden may change by a factor e over the ~500 days it takes to matter, but
#: not by the factor 60 in 40 days a #658-class runaway produces. The drift
#: statistic reads burdens only, so it is independent of the deposition ledger
#: and is the gate to trust when the ledger is incomplete.
DRIFT_LIMIT_PER_DAY = 0.002
BUDGET_RESIDUAL_LIMIT = 0.05

#: The closure gate's sign carries information, and the two directions have
#: different diagnoses. A NEGATIVE residual means more mass was deposited than
#: entered the column: no missing ledger entry can produce that, so it is mass
#: creation. A POSITIVE one means emitted mass is unaccounted for, which a
#: *missing sink diagnostic* looks exactly like — and on output written before
#: the #722 removal-ledger fix, ``dry_*`` omits the Slinn turbulent/Brownian
#: deposition entirely (it captures only ~31 % of the dust dry sink), so a
#: positive residual on such a run is expected and is not evidence of a leak.
_POSITIVE_RESIDUAL_CAVEAT = (
    "unaccounted emission; on output written before the #722 removal-ledger "
    "fix, dry_* omits Slinn dry deposition and reads positive by construction")

#: Regression tolerance: whichever of a 3-sigma excursion and a 15 % relative
#: change is larger. 3 sigma alone is far too tight for a well-sampled mean
#: (sigma is ~1 % of the annual mean); 15 % alone would let a slow bias
#: through on a noisy statistic.
_N_SIGMA = 3.0
_REL_TOLERANCE = 0.15

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
    run directory alone is enough. Requires an interstitial mass tracer *and*
    a JAM-specific diagnostic namespace, so a MACv2-SP run (which publishes
    AOD but no ``m_<sp>_<mode>``) is not mistaken for one.
    """
    has_mass = any(re.fullmatch(r"m_[a-z0-9]+_[a-z]+", str(v))
                   for v in ds.data_vars)
    has_jam = any(str(v).startswith(("jam_state.", "jam_cloud_borne.",
                                     "jam_optics.", "jam_band_optics."))
                  for v in ds.data_vars)
    return has_mass and has_jam


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


def regression_tolerance(reference: float, sigma: float) -> float:
    """``max(3 sigma, 15 % of the reference)`` — see the design doc."""
    return max(_N_SIGMA * sigma, _REL_TOLERANCE * abs(reference))


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
        # The storage term is a difference between the FIRST and LAST chunk
        # means, i.e. between chunk centres, so the flux integral must cover
        # that same interval: the chunks after the first, times the span
        # between the centres. Averaging all N chunks over an (N-1)-chunk span
        # understates the integral by 1/N — 1.4 % on a 73-chunk year, but 25 %
        # on a four-chunk window, straight into a residual gated at 5 %.
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
    return float((emitted - deposited - (stored[-1] - stored[0])) / emitted)


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
        # Emit the drift only when it could actually be fitted. A species the
        # run never carries (soa with no SOAG production) and a record too
        # short for a slope both yield NaN, and a NaN scored against the gate
        # reads as a FAIL for something that was never measured — which is how
        # ``health.py --last-n 2`` on a healthy run reported an aerosol
        # failure. Not measurable is "not scored", not "failed".
        drift = log_drift(days, b)
        if np.isfinite(drift):
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
        r = _budget_residual(days, series, species)
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
        value, ref = stats[key], reference[key]
        sigma = 0.0
        if series is not None:
            for candidate in (key, key.removesuffix("_mg_m2"),
                              f"burden_{key.removeprefix('burden_')}"):
                if candidate in series:
                    sigma = standard_error(series[candidate])
                    break
        tol = regression_tolerance(ref, sigma)
        ok = abs(value - ref) <= tol
        rows.append((key, value, f"{ref:.5g} +- {tol:.3g}", bool(ok)))
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


def run_files(run_dir: str) -> list[str]:
    """Chunk files of a run directory, in day order."""
    return sorted(glob.glob(f"{run_dir}/*_day*.nc"),
                  key=lambda f: int(re.search(r"day(\d+)", f).group(1)))


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
    args = ap.parse_args()

    if args.series_in:
        loaded = np.load(args.series_in)
        days = loaded["_days"]
        series = {k: loaded[k] for k in loaded.files if k != "_days"}
    else:
        files = run_files(args.run_dir)
        if not files:
            print(f"FAIL  no chunk files in {args.run_dir}")
            return 1
        if args.last_n:
            files = files[-args.last_n:]
        days, series = collect(files)
    if args.series_out:
        np.savez(args.series_out, _days=days, **series)
    dt = timestep_seconds(args.run_dir) if args.run_dir else None
    stats = summarize(days, series, dt)
    if dt is None:
        print("NOTE  no .hydra/config.yaml timestep — the dynamics-"
              "conservation gate is not scored\n")
    gates = physics_gates(stats)
    print(format_table(stats, gates))
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

    print("\nAEROSOL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
