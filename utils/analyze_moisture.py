"""Analyse a moisture-diagnostic probe netcdf.

Prints daily global-mean E vs P balance, column moisture, convection,
and vdiff diagnostics to help identify why the model drifts to a
catastrophic NaN 30-40 days in.

Usage:
    python utils/analyze_moisture.py moisture_probe_hybrid_q5.nc
"""

import sys
import numpy as np
import xarray as xr


def main(path):
    ds = xr.open_dataset(path)
    print(f"File: {path}")
    print(f"Shape: times={ds.sizes.get('time')}, levels={ds.sizes.get('level')}, "
          f"lon={ds.sizes.get('lon')}, lat={ds.sizes.get('lat')}")
    print(f"Variables ({len(ds.data_vars)}): {sorted(ds.data_vars)}")

    # Global-mean moisture budget over time.
    def gmean(field):
        return ds[field].mean(dim=["lon", "lat"]) if field in ds else None

    t = ds.time.values
    print("\n=== Moisture / energy balance ===")
    hdr = (f"{'day':>5} {'qmean':>9} {'qmax':>9} {'evap':>10} {'pR':>10} "
           f"{'pS':>10} {'pC':>10} {'E-P':>10}  {'LH':>7} {'SH':>7}  "
           f"{'TKE':>6} {'PBL':>7}")
    print(hdr)
    for i in range(len(t)):
        day = i
        q = ds["specific_humidity"].isel(time=i).mean().item() * 1000
        qmax = ds["specific_humidity"].isel(time=i).max().item() * 1000
        e = gmean("surface.evaporation").isel(time=i).item() if gmean("surface.evaporation") is not None else np.nan
        pr = gmean("clouds.precip_rain").isel(time=i).item() if gmean("clouds.precip_rain") is not None else 0
        ps_ = gmean("clouds.precip_snow").isel(time=i).item() if gmean("clouds.precip_snow") is not None else 0
        pc = gmean("convection.precip_conv").isel(time=i).item() if gmean("convection.precip_conv") is not None else 0
        ep = e - pr - ps_ - pc
        lh = gmean("surface.latent_heat_flux").isel(time=i).item() if gmean("surface.latent_heat_flux") is not None else np.nan
        sh = gmean("surface.sensible_heat_flux").isel(time=i).item() if gmean("surface.sensible_heat_flux") is not None else np.nan
        tke = gmean("vdiff.tke").isel(time=i).mean().item() if gmean("vdiff.tke") is not None else np.nan
        pbl = gmean("vdiff.pbl_height").isel(time=i).item() if gmean("vdiff.pbl_height") is not None else np.nan

        print(f"{day:>5} {q:>9.3f} {qmax:>9.1f} {e:>10.2e} {pr:>10.2e} "
              f"{ps_:>10.2e} {pc:>10.2e} {ep:>10.2e}  {lh:>7.1f} {sh:>7.1f}  "
              f"{tke:>6.3f} {pbl:>7.0f}")

    # Surface-latent-heat cumulative vs precip cumulative
    if all(v in ds for v in ["surface.evaporation", "clouds.precip_rain", "convection.precip_conv"]):
        print("\n=== Integrated mass (column kg/m²) over 1 day ===")
        dt_s = 86400.0
        e_cum = gmean("surface.evaporation").values * dt_s  # kg/m²/day
        pr_cum = gmean("clouds.precip_rain").values * dt_s
        ps_cum = gmean("clouds.precip_snow").values * dt_s if "clouds.precip_snow" in ds else np.zeros_like(pr_cum)
        pc_cum = gmean("convection.precip_conv").values * dt_s
        print(f"Day 0 - {int(t[-1] - t[0])//86400} totals:")
        print(f"  Evap integrated: {e_cum.sum():.2f} mm water/m²")
        print(f"  Large-scale rain: {pr_cum.sum():.2f}")
        print(f"  Large-scale snow: {ps_cum.sum():.2f}")
        print(f"  Convective:       {pc_cum.sum():.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "moisture_probe_hybrid_q5.nc")
