"""Score the SPEEDY column against ARMBE observations.

    source env.sh
    python run_scm.py && python evaluate.py

Comparison is at **daily mean** resolution, not hourly, and that is deliberate.
SPEEDY's shortwave reads only fraction-of-year (``speedy_shortwave.py`` ->
``forcing.solar.tyear``) and computes daily-mean, zonally-averaged insolation.
It has no diurnal cycle and cannot have one, so scoring it against ARMBE's
hourly irradiance would be comparing a flat line to a day/night swing. Daily
means are the honest resolution for this model. (ECHAM's radiation uses a real
cos(zenith) and would support an hourly comparison.)

Units and indexing:
* ``jcm/physics/speedy/units_table.csv`` gives precip and evaporation in g/m²/s
  and the radiative/heat fluxes in W/m².
* Surface-flux fields carry a trailing tile axis of length 3. **Tile 0 is land,
  tile 1 is sea** — from ``speedy_surface_flux.py:264``, which blends them as
  ``var[:,:,1] + fmask*(var[:,:,0] - var[:,:,1])``, so ``fmask`` is the *land*
  fraction and selects tile 0 as fmask -> 1.

  Getting this wrong is quiet and expensive. I originally read tile 1 as land and
  "confirmed" it because ``rlus[1]`` = 443.8 matched sigma*T^4 for the surface
  temperature. That check proves nothing — Stefan-Boltzmann holds for *any*
  surface, so it can't distinguish tiles. The real tell is that tile 1 was frozen
  (std 1e-4) while the land tile tracks the surface temperature (std 18).

  Related trap in ``run_scm.py``: ``TerrainData.single_column`` defaults to
  ``lfluxland=False``, and the whole land-flux branch sits behind
  ``jax.lax.cond(lfluxland, land_fluxes, pass_fn)`` (line 228). With
  ``fmask=1`` and ``lfluxland=False`` you select the land tile *and* never
  compute it — the atmosphere silently sees zero surface fluxes.

OPEN QUESTION: PRECIPITATION
----------------------------
Treat the precip row below as unexplained, not as a score. On the synthetic
fixture the model rains ~12 mm/hr continuously against observations that are wet
4% of the time. Facts, as measured:

* Convection fired on 168 of 168 steps (100%).
* The rate barely moved: 10.3-12.8 mm/hr, std 0.73.
* The fixture's own convective instability (surface theta_e minus saturated
  theta_e at 500 hPa) varied a lot over the same period: -1.3 to +26.9 K,
  unstable in 94% of hours.

A rate that is flat while instability swings by ~20x is not the scheme simply
tracking CAPE. Candidate explanations, none of them verified: the mass-flux
closure saturating against the fixture's very moist surface layer (~19 g/kg at
all times); the static surface forcing (see run_scm.py); diagnostic mode
discarding convection's stabilizing tendency each step so CAPE is never consumed;
or the fixture itself being unphysical. Note the fixture is unstable in 94% of
hours by construction, which real ARMBE soundings will not be — so some of this
gap is certainly an artifact of the fixture and not of the method.

Do not resolve this on synthetic data; it cannot settle it. Revisit when real
ARMBE profiles are in hand, and check whether precip becomes intermittent and
whether the rate starts tracking instability.

If precip does need the convective feedback to be scored fairly, the SCM supports
running the column prognostically with nudging:
``SingleColumnModel(..., relaxation_timescales={"temperature": tau,
"specific_humidity": tau})`` — the column evolves under its own physics (so
convection can stabilize it) while staying tied to the observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LAND_TILE = 0           # speedy_surface_flux.py:264 -- index 0 is land, 1 is sea
LATENT_HEAT_VAPORIZATION = 2.5e6      # J/kg
G_PER_M2_PER_S_TO_MM_PER_HR = 3.6     # 1 g/m²/s = 1e-3 kg/m²/s = 1e-3 mm/s

# canonical name -> (how to get the model series, how to get the obs series)
PAIRS = {
    "precip [mm/hr]": ("precip", "obs.precip"),
    "sfc SW down [W/m2]": ("sw_down", "obs.sw_down_sfc"),
    "sfc SW net [W/m2]": ("sw_net", "obs.sw_net_sfc"),
    "sfc LW down [W/m2]": ("lw_down", "obs.lw_down_sfc"),
    # ARMBE's `lwup` is measured at 10 m while SPEEDY emits at the surface.
    "sfc LW up [W/m2]": ("lw_up", "obs.lw_up_sfc"),
    "TOA SW down [W/m2]": ("sw_down_toa", "obs.sw_down_toa"),
    "TOA SW net [W/m2]": ("sw_net_toa", "obs.sw_net_toa"),
    "cloud fraction [1]": ("cloud_fraction", "obs.cloud_fraction"),
    "sensible heat [W/m2]": ("shf", "obs.sensible_heat_flux"),
    "latent heat [W/m2]": ("lhf", "obs.latent_heat_flux"),
}


def _tile(v: np.ndarray) -> np.ndarray:
    """Collapse (nt,1,1[,3]) to (nt,), taking the land tile when present."""
    v = np.asarray(v)
    if v.ndim == 4:
        v = v[..., LAND_TILE]
    return v.reshape(v.shape[0], -1)[:, 0]


def model_series(d) -> dict[str, np.ndarray]:
    """Model diagnostics in ARMBE-comparable units."""
    precls = _tile(d["model.condensation.precls"])
    precnv = _tile(d["model.convection.precnv"])
    return {
        "precip": (precls + precnv) * G_PER_M2_PER_S_TO_MM_PER_HR,
        "sw_down": _tile(d["model.shortwave_rad.rsds"]),
        "sw_net": _tile(d["model.shortwave_rad.rsns"]),
        "lw_down": _tile(d["model.surface_flux.rlds"]),
        "lw_up": _tile(d["model.surface_flux.rlus"]),
        "sw_down_toa": _tile(d["model.shortwave_rad.fsol"]),
        "sw_net_toa": _tile(d["model.shortwave_rad.ftop"]),
        "cloud_fraction": _tile(d["model.shortwave_rad.cloudc"]),
        "shf": _tile(d["model.surface_flux.shf"]),
        "lhf": _tile(d["model.surface_flux.evap"]) * 1e-3 * LATENT_HEAT_VAPORIZATION,
    }


def to_daily(x: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over each calendar day. Returns (day, daily_mean)."""
    days = times.astype("datetime64[D]")
    uniq = np.unique(days)
    out = np.array([np.nanmean(x[days == u]) for u in uniq])
    return uniq, out


def metrics(m: np.ndarray, o: np.ndarray) -> dict[str, float]:
    good = np.isfinite(m) & np.isfinite(o)
    if good.sum() < 2:
        return {"n": int(good.sum()), "bias": np.nan, "rmse": np.nan, "corr": np.nan}
    m, o = m[good], o[good]
    corr = np.nan
    if m.std() > 1e-12 and o.std() > 1e-12:
        corr = float(np.corrcoef(m, o)[0, 1])
    return {
        "n": int(good.sum()),
        "obs_mean": float(o.mean()),
        "mod_mean": float(m.mean()),
        "bias": float((m - o).mean()),
        "rmse": float(np.sqrt(((m - o) ** 2).mean())),
        "corr": corr,
    }


def main(argv=None) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=here / "outputs" / "scm_run.npz")
    ap.add_argument("--plot", action="store_true", help="write outputs/compare.png")
    args = ap.parse_args(argv)

    d = np.load(args.run)
    times = d["times"].astype("datetime64[s]")
    mod = model_series(d)

    rows = []
    daily = {}
    for label, (mkey, okey) in PAIRS.items():
        if okey not in d:
            print(f"  (skip {label}: no {okey} in run file)")
            continue
        m_h, o_h = mod[mkey], np.asarray(d[okey], dtype=float)
        n = min(len(m_h), len(o_h))
        day, m_d = to_daily(m_h[:n], times[:n])
        _, o_d = to_daily(o_h[:n], times[:n])
        daily[label] = (day, m_d, o_d)
        rows.append((label, metrics(m_d, o_d)))

    print(f"\nDaily-mean comparison, {len(times)} input steps "
          f"-> {len(next(iter(daily.values()))[0])} days\n")
    hdr = f"{'field':24s}{'obs':>10s}{'model':>10s}{'bias':>10s}{'rmse':>10s}{'corr':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for label, s in rows:
        corr = f"{s['corr']:8.2f}" if np.isfinite(s["corr"]) else f"{'n/a':>8s}"
        flag = "  <- diagnostic; see docstring" if "precip" in label else ""
        print(f"{label:24s}{s['obs_mean']:10.3f}{s['mod_mean']:10.3f}"
              f"{s['bias']:10.3f}{s['rmse']:10.3f}{corr}{flag}")

    print("\nnote: precipitation is diagnostic rather than a headline skill score. "
          "In prescribed-state mode,\n      the model cannot retain convective "
          "thermodynamic feedback between observations. Assess it on\n      a "
          "longer contiguous real-data window and against a relaxation sensitivity "
          "run; see the module docstring. Surface LW-up compares SPEEDY's surface "
          "emission\n      against ARMBE's 10 m measurement and is therefore a "
          "near-surface diagnostic, not an exact collocation.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(daily)
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True)
        for ax, (label, (day, m_d, o_d)) in zip(np.atleast_1d(axes), daily.items()):
            ax.plot(day, o_d, "o-", label="ARMBE", color="#222")
            ax.plot(day, m_d, "s--", label="SPEEDY SCM", color="#c33")
            ax.set_ylabel(label, fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
        fig.suptitle("SPEEDY single column vs ARMBE, SGP (daily means)")
        fig.tight_layout()
        out = args.run.parent / "compare.png"
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
