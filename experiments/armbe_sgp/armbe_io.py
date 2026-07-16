"""Turn ARMBE observations at SGP into SPEEDY single-column model inputs.

Three jobs:

1. ``load_armbe``       open ARMBEATM (+ optional ARMBECLDRAD) and slice a window.
2. ``to_state_series``  ARMBE pressure-level profiles -> a list of 1-D
                        ``PhysicsState`` on SPEEDY's sigma levels.
3. ``to_obs_targets``   ARMBECLDRAD/ATM observations to score the run against.

Two things this module has to get right, both verified against jcm:

* **Vertical grid.** ARMBE profiles live on *pressure* levels. SPEEDY physics
  works on *sigma* = p/p_sfc and rebuilds its own sigma distribution from the
  level count alone (``SpeedyCoords.from_coordinate_system`` uses
  ``kx = coords.nodal_shape[0]``), so we must interpolate onto
  ``compute_speedy_vertical_coords(nlev)`` -> ``fsg``, the *full-level sigma*
  (nlev=8: 0.025 .. 0.95, ordered top->bottom).
  ``sigl`` from that same call is ``log(fsg)`` (checked: ``exp(sigl) == fsg``) —
  interpolating onto it would silently yield negative levels. Use ``fsg``.

* **Moisture.** ARMBE ships *dewpoint temperature*, not specific humidity, so we
  derive q. Preference order: an explicit specific-humidity variable if the file
  has one, else dewpoint, else relative humidity.

Variable names: ARM's public docs describe ARMBE's contents in prose ("Dry Bulb
Temperature", "Eastward Wind Component") but do not publish the netCDF variable
names, and they differ across sites/versions. So every field is resolved through
``CANDIDATES`` — an ordered list of plausible names — and :func:`describe_vars`
reports what was matched. When a real file lands, run ``python armbe_io.py
<file.nc>`` to print its variables and, if needed, extend ``CANDIDATES``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

from jcm.physics.speedy.speedy_coords import compute_speedy_vertical_coords
from jcm.constants import p0, rd
from jcm.physics_interface import PhysicsState

# SGP Central Facility (Lamont, OK) — exact coords from the ARM datastream
# metadata for sgparmbeatmC1.c1 / sgparmbecldradC1.c1.
SGP_LAT_DEG = 36.607322
SGP_LON_DEG = -97.487643
SGP_OROG_M = 315.0  # approx. elevation; land point (fmask=1)

# Ordered candidate netCDF names per canonical field. First match wins.
CANDIDATES: dict[str, tuple[str, ...]] = {
    # coordinates
    "time": ("time",),
    "level": ("lev", "level", "pressure", "plev", "p", "pres"),
    # profiles (pressure grid). ARMBE also carries height-grid variants
    # (e.g. *_h); we deliberately prefer the pressure-grid names (*_p).
    "temperature": ("temp_p", "temperature_p", "T_p", "t_p",
                    "temp", "temperature", "T"),
    "dewpoint": ("dp_temp_p", "dewpoint_p", "dp_p", "dewpt_p",
                 "dp_temp", "dewpoint", "dewpt"),
    "relative_humidity": ("rh_p", "relative_humidity_p", "rh",
                          "relative_humidity"),
    "specific_humidity": ("q_p", "spec_hum_p", "sphum_p", "qv_p",
                          "q", "spec_hum", "specific_humidity"),
    "u_wind": ("u_wind_p", "u_p", "u_wind", "u", "eastward_wind"),
    "v_wind": ("v_wind_p", "v_p", "v_wind", "v", "northward_wind"),
    # surface
    "surface_pressure": ("pressure_sfc", "p_sfc", "psfc", "surface_pressure",
                         "bar_pres", "barometric_pressure"),
    "surface_temperature": ("temp_sfc", "T_sfc", "tsfc", "surface_temperature",
                            "temp_mean"),
    # evaluation targets
    "precip": ("precip_rate_sfc", "precip_rate", "prec_sfc", "precip",
               "precip_sfc"),
    "sensible_heat_flux": ("sensible_heat_flux", "sh_flux", "shf", "SH"),
    "latent_heat_flux": ("latent_heat_flux", "lh_flux", "lhf", "LH"),
    "sw_down_sfc": ("sw_dn_sfc", "surface_downwelling_shortwave",
                    "rsds", "swdn_sfc", "sw_down"),
    "lw_down_sfc": ("lw_dn_sfc", "surface_downwelling_longwave",
                    "rlds", "lwdn_sfc", "lw_down"),
    "cloud_fraction": ("cld_frac", "cloud_fraction", "cldfrac"),
    "lwp": ("lwp", "liquid_water_path"),
}


class VariableNotFound(KeyError):
    """No candidate name for a required field exists in the dataset."""


def pick(ds: xr.Dataset, field: str, required: bool = True) -> str | None:
    """Resolve a canonical ``field`` to an actual variable name in ``ds``."""
    for name in CANDIDATES[field]:
        if name in ds.variables:
            return name
    if required:
        raise VariableNotFound(
            f"none of {CANDIDATES[field]!r} found for {field!r}. "
            f"Available: {sorted(map(str, ds.variables))}. "
            f"Extend CANDIDATES[{field!r}] in armbe_io.py."
        )
    return None


def describe_vars(ds: xr.Dataset) -> dict[str, str | None]:
    """Report which real variable each canonical field resolved to."""
    return {f: pick(ds, f, required=False) for f in CANDIDATES}


# --------------------------------------------------------------------------
# moisture
# --------------------------------------------------------------------------

def _saturation_vapor_pressure_hpa(t_kelvin: np.ndarray) -> np.ndarray:
    """Magnus/Tetens saturation vapour pressure over water [hPa]."""
    t_c = t_kelvin - 273.15
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def specific_humidity_from_dewpoint(dewpoint_k: np.ndarray,
                                    pressure_hpa: np.ndarray) -> np.ndarray:
    """q [kg/kg] from dewpoint [K] and pressure [hPa].

    Vapour pressure at the dewpoint is the actual vapour pressure, so
    ``e = e_sat(Td)`` and ``q = eps*e / (p - (1-eps)*e)`` with eps = 0.622.
    """
    e = _saturation_vapor_pressure_hpa(dewpoint_k)
    eps = 0.622
    return eps * e / np.maximum(pressure_hpa - (1.0 - eps) * e, 1e-6)


def specific_humidity_from_rh(rh_percent: np.ndarray, t_kelvin: np.ndarray,
                              pressure_hpa: np.ndarray) -> np.ndarray:
    """q [kg/kg] from RH [%], temperature [K], pressure [hPa]."""
    e = np.clip(rh_percent, 0.0, 100.0) / 100.0 * _saturation_vapor_pressure_hpa(t_kelvin)
    eps = 0.622
    return eps * e / np.maximum(pressure_hpa - (1.0 - eps) * e, 1e-6)


# --------------------------------------------------------------------------
# vertical grid
# --------------------------------------------------------------------------

def speedy_sigma_levels(nlev: int = 8) -> np.ndarray:
    """SPEEDY full-level sigma (``fsg``), ordered top->bottom.

    NB ``sigl`` from the same call is ``log(fsg)``; ``fsg`` is the sigma we want.
    """
    _hsg, fsg, _dhs, _sigl, _gs, _gc, _wvi = compute_speedy_vertical_coords(nlev)
    return np.asarray(fsg).ravel()


def interp_to_sigma(profile: np.ndarray, p_hpa: np.ndarray, ps_hpa: float,
                    sigma_target: np.ndarray) -> np.ndarray:
    """Interpolate one pressure-level profile onto SPEEDY sigma levels.

    ``profile`` and ``p_hpa`` are 1-D over ARMBE's pressure levels (any order,
    NaNs allowed). Returns values on ``sigma_target`` (top->bottom).

    Missing levels are dropped before interpolation; ``np.interp`` clamps at the
    profile's own edges, so sigma levels outside ARMBE's pressure span take the
    nearest valid value rather than extrapolating wildly.
    """
    good = np.isfinite(profile) & np.isfinite(p_hpa)
    if good.sum() < 2:
        return np.full(sigma_target.shape, np.nan)
    sigma_obs = p_hpa[good] / ps_hpa           # sigma increases downward
    vals = profile[good]
    order = np.argsort(sigma_obs)              # np.interp needs increasing x
    return np.interp(sigma_target, sigma_obs[order], vals[order])


def geopotential_on_sigma(temperature_k: np.ndarray,
                          sigma: np.ndarray) -> np.ndarray:
    """Hydrostatic geopotential on the same top-to-bottom sigma grid as ``T``.

    ARMBE profiles have already been interpolated to SPEEDY's ``fsg`` levels.
    The generic ``create_single_column_state`` helper constructs geopotential on
    its own linear pressure grid instead, so it must not be used here: radiation,
    convection, and the surface scheme require every vertical field to share one
    coordinate.
    """
    temperature_k = np.asarray(temperature_k, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if temperature_k.shape != sigma.shape:
        raise ValueError("temperature and sigma must have the same shape")
    # Same column-mean-temperature hydrostatic approximation as the generic
    # helper, evaluated at SPEEDY's actual full-level sigma values.
    return -rd * np.mean(temperature_k) * np.log(sigma)


def speedy_column_state(temperature_k: np.ndarray, specific_humidity: np.ndarray,
                        u_wind: np.ndarray, v_wind: np.ndarray,
                        surface_pressure_pa: float,
                        sigma: np.ndarray) -> PhysicsState:
    """Build a SPEEDY-column state whose every profile uses ``sigma``."""
    return PhysicsState(
        temperature=np.asarray(temperature_k),
        specific_humidity=np.maximum(np.asarray(specific_humidity), 0.0),
        u_wind=np.nan_to_num(np.asarray(u_wind)),
        v_wind=np.nan_to_num(np.asarray(v_wind)),
        geopotential=geopotential_on_sigma(temperature_k, sigma),
        normalized_surface_pressure=np.asarray(surface_pressure_pa / p0),
        tracers={},
    )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_armbe(atm: str | Path | Iterable[str | Path],
               cldrad: str | Path | Iterable[str | Path] | None = None,
               t0: str | None = None, t1: str | None = None) -> xr.Dataset:
    """Open ARMBEATM (+ optional ARMBECLDRAD), merge, slice to [t0, t1]."""
    def _open(src):
        if src is None:
            return None
        if isinstance(src, (str, Path)):
            paths = sorted(Path(src).glob("*.nc")) if Path(src).is_dir() else [Path(src)]
        else:
            paths = [Path(p) for p in src]
        if not paths:
            raise FileNotFoundError(f"no netCDF files found for {src!r}")
        if len(paths) == 1:
            return xr.open_dataset(paths[0])
        return xr.open_mfdataset(paths, combine="by_coords")

    ds = _open(atm)
    ds_c = _open(cldrad)
    if ds_c is not None:
        # Targets may be on the same hourly axis; merge non-conflicting vars.
        ds = xr.merge([ds, ds_c], compat="override", join="inner")
    if t0 is not None or t1 is not None:
        ds = ds.sel(time=slice(t0, t1))
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"no timesteps in window {t0}..{t1}")
    return ds


def _level_pressure_hpa(ds: xr.Dataset) -> np.ndarray:
    """ARMBE level coordinate as pressure in hPa (handles Pa-valued files)."""
    lev_name = pick(ds, "level")
    lev = np.asarray(ds[lev_name].values, dtype=float)
    units = str(ds[lev_name].attrs.get("units", "")).lower()
    if "pa" in units and "hpa" not in units and "mb" not in units:
        lev = lev / 100.0          # Pa -> hPa
    elif lev.max() > 2000.0:       # unit-less but clearly Pa
        lev = lev / 100.0
    return lev


def _surface_pressure_hpa(ds: xr.Dataset) -> np.ndarray:
    ps_name = pick(ds, "surface_pressure")
    ps = np.asarray(ds[ps_name].values, dtype=float)
    units = str(ds[ps_name].attrs.get("units", "")).lower()
    if "pa" in units and "hpa" not in units and "mb" not in units:
        ps = ps / 100.0
    elif np.nanmax(ps) > 2000.0:
        ps = ps / 100.0
    return ps


def _moisture_profiles(ds: xr.Dataset, p_hpa: np.ndarray,
                       temp: np.ndarray) -> np.ndarray:
    """Specific humidity [kg/kg] with shape (ntime, nlev_arm)."""
    if (q_name := pick(ds, "specific_humidity", required=False)) is not None:
        q = np.asarray(ds[q_name].values, dtype=float)
        units = str(ds[q_name].attrs.get("units", "")).lower()
        if "g/kg" in units or "g kg" in units:
            q = q / 1000.0
        return q
    if (dp_name := pick(ds, "dewpoint", required=False)) is not None:
        dp = np.asarray(ds[dp_name].values, dtype=float)
        if np.nanmax(dp) < 100.0:      # Celsius -> Kelvin
            dp = dp + 273.15
        return specific_humidity_from_dewpoint(dp, p_hpa[None, :])
    if (rh_name := pick(ds, "relative_humidity", required=False)) is not None:
        rh = np.asarray(ds[rh_name].values, dtype=float)
        return specific_humidity_from_rh(rh, temp, p_hpa[None, :])
    raise VariableNotFound(
        "no moisture variable found (specific humidity, dewpoint, or RH). "
        f"Available: {sorted(map(str, ds.variables))}"
    )


def to_state_series(ds: xr.Dataset, nlev: int = 8):
    """Build the prescribed ``PhysicsState`` series for the SCM.

    Returns ``(states, times, meta)`` where ``states`` is a list of 1-D
    ``PhysicsState`` (one per timestep) on SPEEDY sigma levels.
    """
    sigma = speedy_sigma_levels(nlev)
    p_hpa = _level_pressure_hpa(ds)
    ps_hpa = _surface_pressure_hpa(ds)

    temp = np.asarray(ds[pick(ds, "temperature")].values, dtype=float)
    if np.nanmax(temp) < 100.0:       # Celsius -> Kelvin
        temp = temp + 273.15
    q = _moisture_profiles(ds, p_hpa, temp)
    u = np.asarray(ds[pick(ds, "u_wind")].values, dtype=float)
    v = np.asarray(ds[pick(ds, "v_wind")].values, dtype=float)

    ntime = temp.shape[0]
    states = []
    retained_indices = []
    n_bad = 0
    for i in range(ntime):
        psi = float(ps_hpa[i])
        if not np.isfinite(psi):
            n_bad += 1
            continue
        ti = interp_to_sigma(temp[i], p_hpa, psi, sigma)
        qi = interp_to_sigma(q[i], p_hpa, psi, sigma)
        ui = interp_to_sigma(u[i], p_hpa, psi, sigma)
        vi = interp_to_sigma(v[i], p_hpa, psi, sigma)
        if not (np.all(np.isfinite(ti)) and np.all(np.isfinite(qi))):
            n_bad += 1
            continue
        states.append(speedy_column_state(
            temperature_k=ti,
            specific_humidity=qi,
            u_wind=ui,
            v_wind=vi,
            surface_pressure_pa=psi * 100.0,   # hPa -> Pa
            sigma=sigma,
        ))
        retained_indices.append(i)
    times = np.asarray(ds[pick(ds, "time")].values)
    meta = {
        "n_input_times": int(ntime),
        "n_states": len(states),
        "n_dropped": int(n_bad),
        "retained_indices": np.asarray(retained_indices, dtype=int),
        "sigma_levels": sigma,
        "resolved": describe_vars(ds),
    }
    return states, times[meta["retained_indices"]], meta


def to_obs_targets(ds: xr.Dataset,
                   indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Observed series to score the SCM against (missing fields are skipped)."""
    out: dict[str, np.ndarray] = {}
    for field in ("precip", "sw_down_sfc", "lw_down_sfc",
                  "sensible_heat_flux", "latent_heat_flux",
                  "cloud_fraction", "lwp"):
        name = pick(ds, field, required=False)
        if name is not None:
            values = np.asarray(ds[name].values, dtype=float)
            out[field] = values if indices is None else values[indices]
    return out


def _main(argv: Sequence[str]) -> int:
    """Inspect a real ARMBE file: print variables and what resolves."""
    if not argv:
        print(__doc__)
        return 1
    ds = load_armbe(argv[0])
    print(f"dims: {dict(ds.sizes)}\n")
    print("variables in file:")
    for v in sorted(map(str, ds.variables)):
        va = ds[v]
        print(f"  {v:28s} {str(tuple(va.dims)):32s} {va.attrs.get('units','')}")
    print("\ncanonical field -> resolved variable:")
    for f, name in describe_vars(ds).items():
        print(f"  {f:22s} -> {name}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
