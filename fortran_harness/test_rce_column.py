"""Quick smoke test: feed a tropical RCE sounding (305 K, 90 % RH) into
the Fortran cumastr driver. ECHAM's convection should trigger and
produce nonzero ktype, mass flux, and precipitation. If not, something
is wrong in our harness even before we ask the JAX side anything.

Mirrors the conditions used in jcm.physics.icon.convection.rce_integration_test.
"""
import subprocess
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_cumastr import (
    DRIVER, write_input_file, read_output_file,
)


def tropical_rce_sounding(klev: int = 47):
    """Build a tropical RCE-like sounding on equidistant sigma levels."""
    p0 = 101325.0
    sigma_bnds = np.linspace(0.0, 1.0, klev + 1)
    # Use 1 hPa (1000 Pa) as TOA — keeps top-layer thickness bounded so
    # the dry-static lift in cubase doesn't push T above the lookup table
    # bound at 400 K.
    sigma_bnds = np.maximum(sigma_bnds, 0.01)
    sigma_bnds[-1] = 1.0
    sigma_full = 0.5 * (sigma_bnds[:-1] + sigma_bnds[1:])
    p_full = sigma_full * p0
    p_half = sigma_bnds * p0
    sigma_full = 0.5 * (sigma_bnds[:-1] + sigma_bnds[1:])
    p_full = sigma_full * p0
    p_half = sigma_bnds * p0

    # Lapse-rate profile up to 12 km, isothermal above.
    z = -7000.0 * np.log(np.maximum(sigma_full, 1e-4))   # rough scale-height
    T_sfc, gamma, T_top = 305.0, 6.5e-3, 200.0
    T = np.maximum(T_sfc - gamma * z, T_top)

    # Saturation vapor (Tetens), 90% RH up to 200 hPa, 0 above.
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    qsat = 0.622 * es / np.maximum(p_full - es, 1.0)
    rh = np.where(p_full > 20000.0, 0.9, 0.0)
    q = np.clip(rh * qsat, 1e-8, 0.03)

    grav = 9.80665
    Rd = 287.04
    Tv = T * (1 + 0.608 * q)
    pmref = np.diff(p_half) / grav

    # Geopotential / heights via hypsometric. ``dpln`` increases going
    # surface-ward (p_half[k+1] > p_half[k]); ``dgeo`` is the geopotential
    # *thickness* of each layer (always positive).
    dpln = np.diff(np.log(np.maximum(p_half, 1.0)))
    dgeo = Rd * Tv * dpln
    geo_full = np.zeros(klev)
    geo_full[-1] = 0.5 * dgeo[-1]
    for k in range(klev - 2, -1, -1):
        geo_full[k] = geo_full[k+1] + 0.5 * (dgeo[k+1] + dgeo[k])
    geo_half = np.zeros(klev + 1)
    for k in range(klev - 1, -1, -1):
        geo_half[k] = geo_half[k+1] + dgeo[k]
    z_full = geo_full / grav
    z_half = geo_half / grav

    # Light large-scale ascent (omega < 0 means upward) — needed for the
    # cubase trigger; without it Tiedtke-Nordeng often returns ktype=0.
    omega = np.where(p_full > 30000.0, -0.05, 0.0)  # Pa/s

    state = {
        "pten":   T.reshape(1, -1),
        "pqen":   q.reshape(1, -1),
        "pxen":   np.zeros((1, klev)),
        "puen":   np.zeros((1, klev)),
        "pven":   np.zeros((1, klev)),
        "pverv":  omega.reshape(1, -1),
        "papp1":  p_full.reshape(1, -1),
        "paphp1": p_half.reshape(1, -1),
        "pgeo":   geo_full.reshape(1, -1),
        "pgeoh":  geo_half.reshape(1, -1),
        "pzf":    z_full.reshape(1, -1),
        "pzh":    z_half.reshape(1, -1),
        "pmref":  pmref.reshape(1, -1),
        "pqte":   np.zeros((1, klev)),
        "pqhfla": np.array([1.5e-4]),     # ~375 W/m² LHF — strong tropical sea
        "pthvsig": np.array([1.0]),
        "ldland":  np.array([False]),
    }
    return state, sigma_full


if __name__ == "__main__":
    KLEV = 47
    state, eta_full = tropical_rce_sounding(klev=KLEV)
    print(f"Tropical RCE sounding ({KLEV} levels)")
    print(f"  surface T = {state['pten'][0, -1]:.1f} K, "
          f"q = {state['pqen'][0, -1]*1000:.2f} g/kg")
    print(f"  TOA   T = {state['pten'][0, 0]:.1f} K, "
          f"q = {state['pqen'][0, 0]*1000:.4f} g/kg")
    print(f"  pqhfla = {state['pqhfla'][0]:.3e} kg/m^2/s")

    inp = Path("/tmp/cumastr_rce_in.bin")
    outp = Path("/tmp/cumastr_rce_out.bin")
    write_input_file(inp, kproma=1, klev=KLEV, dtime=600.0,
                     eta_full=eta_full, state=state)
    res = subprocess.run([str(DRIVER), str(inp), str(outp)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("driver stderr:", res.stderr)
        raise SystemExit(res.returncode)
    print(res.stdout.strip())

    out = read_output_file(outp, kproma=1, klev=KLEV)
    print(f"\nFortran result:")
    print(f"  ktype = {out['ktype'][0]}  (1=deep, 2=shallow, 3=mid, 0=none)")
    print(f"  kctop = {out['kctop'][0]}")
    print(f"  prsfc = {out['prsfc'][0]:.6e} kg/m²/s "
          f"({out['prsfc'][0]*86400:.4f} mm/day)")
    print(f"  pssfc = {out['pssfc'][0]:.6e} kg/m²/s")
    print(f"  ptop  = {out['ptop'][0]:.1f} Pa")
    print(f"  max|pq_cnv|  = {np.max(np.abs(out['pq_cnv'])):.3e} J/kg/s")
    print(f"  max|pqte_cnv| = {np.max(np.abs(out['pqte_cnv'])):.3e} kg/kg/s")
    nonzero_levels = np.nonzero(out['pq_cnv'][0])[0]
    if nonzero_levels.size > 0:
        print(f"  active levels: {nonzero_levels.tolist()}")
        cp = 1004.64
        heating_K_day = out['pq_cnv'][0] / cp * 86400
        print("  per-level heating (K/day):")
        for k in nonzero_levels:
            print(f"    k={k:>3d}  T={state['pten'][0,k]:6.1f}  "
                  f"p={state['papp1'][0,k]/100:7.1f} hPa  "
                  f"dT/dt={heating_K_day[k]:+.3f}")
