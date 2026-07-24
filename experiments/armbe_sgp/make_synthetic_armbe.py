"""Write a synthetic ARMBE-shaped netCDF for testing the pipeline offline.

This is a *fixture*, not science: it lets us build and exercise the loader, the
SCM run, and the evaluation before real ARM credentials work. It mimics ARMBE's
structure — hourly time axis, 25-mb pressure levels, dewpoint rather than
specific humidity, missing (NaN) levels below the surface — with plausible
SGP-in-June magnitudes.

When real data arrives, nothing downstream should need to change: the loader
resolves variable names through ``armbe_io.CANDIDATES`` either way.

Usage::

    python make_synthetic_armbe.py --days 7 --output data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# SGP central facility sits at ~315 m, so surface pressure is ~975 hPa, not 1013.
PS_MEAN_HPA = 975.0


def build(days: int = 7, seed: int = 0) -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(seed)
    nt = days * 24
    time = np.datetime64("2018-06-01T00:00:00") + np.arange(nt) * np.timedelta64(1, "h")
    hour = np.arange(nt) % 24

    # ARMBE standard vertical spacing: 25 mb.
    lev = np.arange(1000.0, 99.0, -25.0)          # 1000 .. 100 hPa
    nlev = lev.size

    # --- surface ---------------------------------------------------------
    diurnal = np.cos((hour - 15) / 24 * 2 * np.pi)          # peak ~15 LT
    ps = PS_MEAN_HPA + 2.0 * np.sin(np.arange(nt) / 24.0) + rng.normal(0, 0.3, nt)
    t_sfc = 299.0 + 6.0 * diurnal + rng.normal(0, 0.5, nt)  # ~293-305 K

    # --- profiles (time, lev) -------------------------------------------
    temp = np.full((nt, nlev), np.nan)
    dewp = np.full((nt, nlev), np.nan)
    uwind = np.full((nt, nlev), np.nan)
    vwind = np.full((nt, nlev), np.nan)

    for i in range(nt):
        # Levels below the surface are missing — exactly what ARMBE does.
        valid = lev <= ps[i]
        p = lev[valid]
        # Hydrostatic-ish height from pressure, then a 6.5 K/km lapse to the
        # tropopause, isothermal above.
        z = 7500.0 * np.log(ps[i] / p)
        t = t_sfc[i] - 6.5e-3 * z
        t = np.maximum(t, 215.0)
        temp[i, valid] = t
        # Dewpoint depression grows with height: moist boundary layer, dry aloft.
        depression = 3.0 + 18.0 * (1.0 - np.exp(-z / 4000.0))
        dewp[i, valid] = t - depression
        # Southerly low-level jet, becoming westerly aloft.
        uwind[i, valid] = 2.0 + 18.0 * (1 - np.exp(-z / 9000.0)) + rng.normal(0, .6, p.size)
        vwind[i, valid] = 7.0 * np.exp(-z / 2500.0) + rng.normal(0, .6, p.size)

    # --- ARMBEATM ---------------------------------------------------------
    atm = xr.Dataset(
        {
            "temp_p": (("time", "lev"), temp, {"units": "K",
                       "long_name": "Dry bulb temperature (pressure grid)"}),
            "dp_temp_p": (("time", "lev"), dewp, {"units": "K",
                          "long_name": "Dewpoint temperature (pressure grid)"}),
            "u_p": (("time", "lev"), uwind, {"units": "m/s",
                    "long_name": "Eastward wind component (pressure grid)"}),
            "v_p": (("time", "lev"), vwind, {"units": "m/s",
                    "long_name": "Northward wind component (pressure grid)"}),
            "pressure_sfc": (("time",), ps, {"units": "hPa",
                             "long_name": "Barometric pressure"}),
            "temp_sfc": (("time",), t_sfc, {"units": "K",
                         "long_name": "Surface air temperature"}),
            "precip_rate_sfc": (("time",), _precip(rng, nt, hour),
                                {"units": "mm/hr", "long_name": "Precipitation rate"}),
            "sensible_heat_flux": (("time",), 90.0 * np.clip(diurnal, 0, None)
                                   + rng.normal(0, 4, nt),
                                   {"units": "W/m2"}),
            "latent_heat_flux": (("time",), 170.0 * np.clip(diurnal, 0, None)
                                 + rng.normal(0, 6, nt),
                                 {"units": "W/m2"}),
        },
        coords={"time": time, "lev": ("lev", lev, {"units": "hPa"})},
        attrs={"comment": "SYNTHETIC ARMBE-shaped fixture — not real observations",
               "site_id": "sgp", "facility_id": "C1"},
    )

    # --- ARMBECLDRAD ------------------------------------------------------
    # Clear-sky-ish diurnal SW with cloud attenuation; LW down from a warm,
    # moist summer atmosphere.
    sw_clear = 950.0 * np.clip(np.cos((hour - 12) / 24 * 2 * np.pi), 0, None) ** 1.2
    cld = np.clip(0.35 + 0.3 * np.sin(np.arange(nt) / 17.0) + rng.normal(0, .08, nt), 0, 1)
    cldrad = xr.Dataset(
        {
            "sw_dn_sfc": (("time",), sw_clear * (1 - 0.65 * cld),
                          {"units": "W/m2",
                           "long_name": "Surface downwelling shortwave irradiance"}),
            "lw_dn_sfc": (("time",), 330.0 + 25.0 * cld + 8.0 * diurnal
                          + rng.normal(0, 2, nt),
                          {"units": "W/m2",
                           "long_name": "Surface downwelling longwave irradiance"}),
            "cld_frac": (("time",), cld, {"units": "fraction",
                          "long_name": "Cloud fraction"}),
            "qc_tot_cld": (("time",), np.zeros(nt, dtype=np.int32),
                           {"long_name": "Cloud fraction quality-control flag"}),
            "lwp": (("time",), 90.0 * cld + rng.normal(0, 8, nt).clip(0),
                    {"units": "g/m2", "long_name": "Liquid water path"}),
        },
        coords={"time": time},
        attrs={"comment": "SYNTHETIC ARMBE-shaped fixture — not real observations"},
    )
    return atm, cldrad


def _precip(rng, nt, hour):
    """Sparse afternoon convective showers, mm/hr."""
    p = np.zeros(nt)
    for start in rng.choice(nt, size=max(1, nt // 40), replace=False):
        dur = rng.integers(1, 4)
        amp = rng.gamma(2.0, 1.5)
        # Bias toward late-day convection, SGP summer style.
        if 12 <= hour[start] <= 22:
            p[start:start + dur] += amp
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "data" / "synthetic")
    args = ap.parse_args(argv)

    atm, cldrad = build(args.days, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    a = args.output / "sgparmbeatmC1.c1.synthetic.nc"
    c = args.output / "sgparmbecldradC1.c1.synthetic.nc"
    atm.to_netcdf(a)
    cldrad.to_netcdf(c)
    print(f"wrote {a}  dims={dict(atm.sizes)}")
    print(f"wrote {c}  dims={dict(cldrad.sizes)}")
    print(f"\nsurface pressure: {float(atm.pressure_sfc.min()):.1f}"
          f"..{float(atm.pressure_sfc.max()):.1f} hPa")
    print(f"levels: {atm.sizes['lev']} @ 25 mb "
          f"({float(atm.lev.max()):.0f}..{float(atm.lev.min()):.0f} hPa)")
    frac_nan = float(np.isnan(atm.temp_p.values).mean())
    print(f"underground/missing profile points: {frac_nan:.1%} (expected — "
          "levels below surface)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
