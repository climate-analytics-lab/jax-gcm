"""Atmosphere health checks for long-running integrations.

``check_health`` and ``print_report`` lift the chunked-run watchdog out of
``utils/run_icon_longrun.py`` so any harness (or notebook, or test) can
inspect the ``ModelPredictions.to_xarray()`` output for the usual NaN /
extreme-value failure modes.
"""

from __future__ import annotations

import numpy as np


def check_health(ds, chunk_idx: int, elapsed_days: float) -> tuple[bool, dict]:
    """Inspect ``ds`` for atmosphere-blowup signatures.

    ``ds`` is the xarray Dataset returned by
    ``ModelPredictions.to_xarray()``. The function looks at the *last*
    timestep in the dataset.

    Returns
    -------
    (ok, report)
        ``ok`` is ``False`` once any failure threshold is breached. The
        report dict can be passed straight to ``print_report``.

    """
    report: dict = {"chunk": chunk_idx, "elapsed_days": elapsed_days}

    nan_vars = [v for v in ds.data_vars if ds[v].isnull().any()]
    report["n_nan_vars"] = len(nan_vars)
    report["n_total_vars"] = len(ds.data_vars)

    if "temperature" in ds:
        T_vals = ds["temperature"].isel(time=-1).values
        report["T_min"] = float(np.nanmin(T_vals))
        report["T_max"] = float(np.nanmax(T_vals))
        report["T_mean"] = float(np.nanmean(T_vals))
        report["T_nan_frac"] = float(np.isnan(T_vals).mean())

    if "specific_humidity" in ds:
        q_vals = ds["specific_humidity"].isel(time=-1).values
        report["q_max_gkg"] = float(np.nanmax(q_vals)) * 1000
        report["q_mean_gkg"] = float(np.nanmean(q_vals)) * 1000
        report["q_nan_frac"] = float(np.isnan(q_vals).mean())

    for key in (
        "radiation.toa_lw_up", "radiation.surface_lw_down",
        "radiation.toa_sw_down", "radiation.toa_sw_up",
    ):
        if key in ds:
            vals = ds[key].isel(time=-1).values
            short = key.split(".")[-1]
            report[f"{short}_mean"] = float(np.nanmean(vals))

    sfc_t_key = next(
        (v for v in ds.data_vars
         if "surface_temperature" in v and "tendency" not in v),
        None,
    )
    if sfc_t_key:
        sfc = ds[sfc_t_key].isel(time=-1).values
        report["sfc_T_min"] = float(np.nanmin(sfc))
        report["sfc_T_max"] = float(np.nanmax(sfc))
        report["sfc_T_mean"] = float(np.nanmean(sfc))

    if "convection.precip_conv" in ds:
        precip = ds["convection.precip_conv"].isel(time=-1).values
        report["precip_conv_mean_mmday"] = float(np.nanmean(precip)) * 86400
        report["precip_conv_max_mmday"] = float(np.nanmax(precip)) * 86400

    ok = True
    reasons: list[str] = []
    # Any NaN in temperature is a failure — the integration has either
    # blown up locally or the dycore has produced an out-of-domain state.
    if report.get("T_nan_frac", 0) > 0:
        ok = False
        reasons.append(f"T NaN fraction {report['T_nan_frac']:.1%}")
    if report.get("T_min", 200) < 100:
        ok = False
        reasons.append(f"T_min={report['T_min']:.0f}K (< 100K)")
    if report.get("T_max", 300) > 500:
        ok = False
        reasons.append(f"T_max={report['T_max']:.0f}K (> 500K)")
    if report.get("q_max_gkg", 0) > 100:
        ok = False
        reasons.append(f"q_max={report['q_max_gkg']:.1f} g/kg (> 100)")

    report["ok"] = ok
    report["reasons"] = reasons
    return ok, report


def print_report(report: dict) -> None:
    """Pretty-print a single ``check_health`` report to stdout."""
    days = report["elapsed_days"]
    years = days / 365.25
    status = "OK" if report["ok"] else "FAILED: " + "; ".join(report["reasons"])
    print(f"\n{'='*60}")
    print(f"  Chunk {report['chunk']} | Day {days:.0f} ({years:.2f} yr) | {status}")
    print(f"{'='*60}")
    print(f"  NaN vars:    {report['n_nan_vars']}/{report['n_total_vars']}")
    if "T_min" in report:
        print(
            f"  Temperature: {report['T_min']:.1f} - {report['T_max']:.1f} K "
            f"(mean {report['T_mean']:.1f} K, NaN {report.get('T_nan_frac', 0):.1%})"
        )
    if "q_max_gkg" in report:
        print(
            f"  Humidity:    max {report['q_max_gkg']:.2f} g/kg, "
            f"mean {report['q_mean_gkg']:.4f} g/kg "
            f"(NaN {report.get('q_nan_frac', 0):.1%})"
        )
    if "sfc_T_mean" in report:
        print(
            f"  Surface T:   {report['sfc_T_min']:.1f} - "
            f"{report['sfc_T_max']:.1f} K (mean {report['sfc_T_mean']:.1f} K)"
        )
    for key, label in (
        ("toa_lw_up_mean", "TOA LW up"),
        ("surface_lw_down_mean", "SFC LW down"),
        ("toa_sw_up_mean", "TOA SW up"),
    ):
        if key in report:
            print(f"  {label}:   {report[key]:.1f} W/m²")
    if "precip_conv_mean_mmday" in report:
        print(
            f"  Precip conv: mean {report['precip_conv_mean_mmday']:.4f} mm/day, "
            f"max {report['precip_conv_max_mmday']:.2f} mm/day"
        )
    print()
