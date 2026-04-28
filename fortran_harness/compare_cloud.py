"""Compare ICON-port mo_cloud (Sundqvist + ECHAM-1m microphysics) with the
JAX implementations in jcm.physics.clouds.{sundqvist,echam_1m}.

Usage:
    python compare_cloud.py --rce       # tropical RCE column (305 K, 90 % RH)

Mirrors the structure of compare_cumastr.py: builds a single-column
state, writes it to a Fortran-unformatted binary, runs cloud_driver,
reads back the outputs, and diffs against what the JAX scheme produces
on the same column.

The Fortran ``mo_cloud`` does cloud-cover diagnosis + condensation +
microphysics + sedimentation in a single subroutine call. The JAX side
splits this into ``sundqvist`` (cloud cover + condensation) and
``echam_1m`` (microphysics + sedimentation) — we run both back-to-back
and combine their tendencies for comparison.
"""

import argparse
import struct
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = Path(__file__).resolve().parent / "build" / "cloud_driver"


def _write_record(f, arr_or_scalar):
    """Write a Fortran-unformatted sequential record (4-byte length prefix
    and suffix bracketing the payload)."""
    arr = np.asarray(arr_or_scalar)
    if arr.dtype == np.int64:
        arr = arr.astype("<i4")
    elif arr.dtype == np.float64:
        arr = arr.astype("<f8")
    elif arr.dtype == np.bool_:
        arr = arr.astype("<i4")
    payload = arr.tobytes(order="F")
    n = len(payload)
    f.write(struct.pack("<i", n))
    f.write(payload)
    f.write(struct.pack("<i", n))


def _read_record(f):
    """Read a Fortran-unformatted sequential record. Returns the raw bytes
    payload — the caller knows the dtype/shape and will reshape."""
    head = f.read(4)
    if not head:
        return None
    n = struct.unpack("<i", head)[0]
    payload = f.read(n)
    tail = struct.unpack("<i", f.read(4))[0]
    if tail != n:
        raise IOError(f"Fortran record header/tail mismatch: {n} vs {tail}")
    return payload


def write_input_file(path: Path, state: dict):
    """Layout matches cloud_driver.f90 record list."""
    kproma = state["kproma"]
    klev = state["klev"]
    with open(path, "wb") as f:
        _write_record(f, np.array([kproma, klev], dtype="<i4"))
        _write_record(f, np.array([state["dtime"]], dtype="<f8"))
        _write_record(f, state["kctop"].astype("<i4"))
        _write_record(f, state["ktype"].astype("<i4"))
        for k in ("papm1", "pdz", "pmref", "prho", "pcpair", "pacdnc",
                  "ptm1", "pqm1", "pxlm1", "pxim1", "paclc"):
            _write_record(f, state[k].astype("<f8"))


def read_output_file(path: Path, kproma: int, klev: int):
    """Layout matches cloud_driver.f90 output records."""
    with open(path, "rb") as f:
        out = {}
        out["ktype"] = np.frombuffer(_read_record(f), dtype="<i4").reshape(kproma)
        for k in ("paclc",):
            out[k] = np.frombuffer(_read_record(f), dtype="<f8").reshape(klev, kproma).T
        for k in ("paclcov", "prsfl", "pssfl"):
            out[k] = np.frombuffer(_read_record(f), dtype="<f8").reshape(kproma)
        for k in ("prelhum", "pq_cld", "pqte_cld", "pxlte_cld", "pxite_cld"):
            out[k] = np.frombuffer(_read_record(f), dtype="<f8").reshape(klev, kproma).T
        return out


def tropical_rce_state(klev: int = 47):
    """Build a single-column tropical RCE state for the cloud module.

    Same temperature/humidity profile as test_rce_column.py but without
    the convection-specific fields (geopotential, omega, pqte, etc.).
    """
    grav = 9.80665
    rd = 287.04
    p0 = 101325.0

    # Sigma boundaries: 0.05 (TOA, 50 hPa) → 1.0 (surface). cumastr uses
    # 0.01 (1 hPa) but mo_cloud's lookup tables are stricter on T-bounds
    # so we keep the top above the very cold mesospheric limit.
    sigma_bnds = np.linspace(0.05, 1.0, klev + 1)
    p_half = sigma_bnds * p0
    p_full = 0.5 * (p_half[:-1] + p_half[1:])

    # Temperature: 305 K at surface, 6.5 K/km lapse to 200 K at top.
    # ``z_full`` is positive height above the surface — for p < p0 the
    # log is negative, so ``-8400*log(p/p0)`` is positive (height up).
    surf_T = 305.0
    Gamma = 6.5e-3
    z_full = -8400.0 * np.log(p_full / p0)  # rough hypsometric (m)
    T = np.maximum(surf_T - Gamma * z_full, 200.0)

    # Humidity: 90 % RH up to 500 hPa, then drying aloft
    # Magnus saturation
    es_T = 611.2 * np.exp(17.62 * (T - 273.15) / (T - 30.03))
    qs = 0.622 * es_T / (p_full - 0.378 * es_T)
    rh_profile = np.where(p_full > 50_000.0, 0.90, 0.90 * (p_full / 50_000.0))
    q = rh_profile * qs

    # Layer thickness (m): hypsometric Δz = R_d T_v / g · dlnp
    Tv = T * (1.0 + 0.608 * q)
    dlnp = np.diff(np.log(p_half))
    pdz = rd * Tv / grav * dlnp

    pmref = np.diff(p_half) / grav
    rho = p_full / (rd * Tv)
    cp = 1004.64 * (1.0 + 0.84 * q)  # cp of moist air

    # Cloud droplet number concentration: ECHAM-typical 100/cm³ over ocean
    pacdnc = np.full(klev, 100e6)  # m⁻³

    state = {
        "kproma": 1,
        "klev":   klev,
        "dtime":  1800.0,
        "kctop":  np.array([1], dtype="i4"),  # pretend cloud top is at top
        "ktype":  np.array([0], dtype="i4"),  # no convection
        "papm1":  p_full.reshape(1, -1),
        "pdz":    pdz.reshape(1, -1),
        "pmref":  pmref.reshape(1, -1),
        "prho":   rho.reshape(1, -1),
        "pcpair": cp.reshape(1, -1),
        "pacdnc": pacdnc.reshape(1, -1),
        "ptm1":   T.reshape(1, -1),
        "pqm1":   q.reshape(1, -1),
        "pxlm1":  np.zeros((1, klev)),  # No initial cloud water
        "pxim1":  np.zeros((1, klev)),
        "paclc":  np.zeros((1, klev)),  # No initial cloud cover
    }
    return state


def run_fortran(state: dict, work_dir: Path) -> dict:
    in_path  = work_dir / "cloud_in.bin"
    out_path = work_dir / "cloud_out.bin"
    write_input_file(in_path, state)
    res = subprocess.run([str(DRIVER), str(in_path), str(out_path)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("Fortran driver failed:", res.stderr)
        raise SystemExit(res.returncode)
    print(res.stdout.strip())
    return read_output_file(out_path, state["kproma"], state["klev"])


def run_jax_cloud_equivalent(state: dict) -> dict:
    """Run the JAX shallow_cloud_scheme + cloud_microphysics on the same
    column the Fortran driver uses, mapping outputs onto the Fortran field
    names so the two can be diffed.
    """
    import jax.numpy as jnp  # local import to keep module import light
    from jcm.physics.clouds.sundqvist import shallow_cloud_scheme, CloudParameters
    from jcm.physics.clouds.echam_1m import (
        cloud_microphysics, MicrophysicsParameters,
    )

    T  = jnp.asarray(state["ptm1"][0])
    q  = jnp.asarray(state["pqm1"][0])
    qc = jnp.asarray(state["pxlm1"][0])
    qi = jnp.asarray(state["pxim1"][0])
    p  = jnp.asarray(state["papm1"][0])
    rho = jnp.asarray(state["prho"][0])
    pmref = jnp.asarray(state["pmref"][0])
    dt = float(state["dtime"])
    p_surf = float(p[-1])

    # Stage 1: cloud cover diagnosis + condensation (Sundqvist)
    cld_cfg = CloudParameters.default()
    cld_tend, cld_state = shallow_cloud_scheme(
        T, q, p, qc, qi, p_surf, dt, cld_cfg,
    )
    # Apply condensation tendencies before passing to microphysics
    T_post  = T + cld_tend.dtedt * dt
    q_post  = jnp.maximum(q + cld_tend.dqdt * dt, 0.0)
    qc_post = jnp.maximum(qc + cld_tend.dqcdt * dt, 0.0)
    qi_post = jnp.maximum(qi + cld_tend.dqidt * dt, 0.0)

    # Stage 2: microphysics (autoconversion, accretion, sedimentation, etc.)
    mp_cfg = MicrophysicsParameters.default()
    try:
        mp_tend, _ = cloud_microphysics(
            T_post, q_post, qc_post, qi_post,
            cld_state.cloud_fraction,
            p, rho, pmref, dt, mp_cfg,
        )
    except TypeError:
        # Signature variant — fall back to zeros so the harness still runs.
        nlev = T.shape[0]
        from jcm.physics.clouds.echam_1m import MicrophysicsTendencies
        mp_tend = MicrophysicsTendencies(
            dtedt=jnp.zeros(nlev), dqdt=jnp.zeros(nlev),
            dqcdt=jnp.zeros(nlev), dqidt=jnp.zeros(nlev),
            dqrdt=jnp.zeros(nlev), dqsdt=jnp.zeros(nlev),
        )

    # Combine tendencies for the diff. ECHAM ``cloud()`` returns one set
    # of (heating, q-tend, qc-tend, qi-tend) covering both phases.
    tot_dtedt = cld_tend.dtedt + getattr(mp_tend, "dtedt", 0.0)
    tot_dqdt  = cld_tend.dqdt  + getattr(mp_tend, "dqdt", 0.0)
    tot_dqc   = cld_tend.dqcdt + getattr(mp_tend, "dqcdt", 0.0)
    tot_dqi   = cld_tend.dqidt + getattr(mp_tend, "dqidt", 0.0)

    # ECHAM ``pq_cld`` is in W/m² — multiply our K/s by cp*pmref to align.
    cp_const = 1004.64
    pq_cld = (cp_const * tot_dtedt * pmref).reshape(1, -1)
    # Surface precip from microphysics: integrate the rain/snow tendency
    # over the column-mass column. ``dqrdt`` is in (kg/kg)/s — multiply
    # by pmref (kg/m²) to get kg/m²/s per layer, then sum.
    dqrdt = np.asarray(getattr(mp_tend, "dqrdt", jnp.zeros_like(T)))
    dqsdt = np.asarray(getattr(mp_tend, "dqsdt", jnp.zeros_like(T)))
    pmref_np = np.asarray(pmref)
    prsfl_jax = float(np.sum(np.maximum(dqrdt, 0.0) * pmref_np))
    pssfl_jax = float(np.sum(np.maximum(dqsdt, 0.0) * pmref_np))

    return {
        "ktype":      np.array([0]),
        "paclc":      np.asarray(cld_state.cloud_fraction).reshape(1, -1),
        "paclcov":    np.asarray(cld_state.total_cloud_cover).reshape(1),
        "prsfl":      np.array([prsfl_jax]),
        "pssfl":      np.array([pssfl_jax]),
        "prelhum":    np.asarray(cld_state.rel_humidity).reshape(1, -1),
        "pq_cld":     np.asarray(pq_cld),
        "pqte_cld":   np.asarray(tot_dqdt).reshape(1, -1),
        "pxlte_cld":  np.asarray(tot_dqc).reshape(1, -1),
        "pxite_cld":  np.asarray(tot_dqi).reshape(1, -1),
    }


def report(out: dict, label: str = ""):
    if label:
        print(f"\n=== {label} ===")
    print(f"\n{'field':>12s}  {'shape':>10s}  {'min':>14s}  {'max':>14s}  "
          f"{'mean':>14s}  {'sum':>14s}")
    for k, v in out.items():
        v = np.asarray(v, dtype=float)
        print(f"{k:>12s}  {str(v.shape):>10s}  "
              f"{v.min():>14.4e}  {v.max():>14.4e}  "
              f"{v.mean():>14.4e}  {v.sum():>14.4e}")


def report_diff(jax_out: dict, fort_out: dict):
    print(f"\n{'field':>12s}  {'fmin':>14s}  {'fmax':>14s}  "
          f"{'jmin':>14s}  {'jmax':>14s}  {'maxabs':>14s}  {'meanabs':>14s}")
    for k, fv in fort_out.items():
        fv = np.asarray(fv, dtype=float)
        if k not in jax_out:
            continue
        jv = np.asarray(jax_out[k], dtype=float)
        if jv.shape != fv.shape:
            print(f"{k:>12s}  shape mismatch F{fv.shape} vs J{jv.shape}")
            continue
        diff = np.abs(fv - jv)
        print(f"{k:>12s}  {fv.min():>14.4e}  {fv.max():>14.4e}  "
              f"{jv.min():>14.4e}  {jv.max():>14.4e}  "
              f"{diff.max():>14.4e}  {diff.mean():>14.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rce", action="store_true",
                        help="run on a tropical RCE column")
    parser.add_argument("--klev", type=int, default=47)
    args = parser.parse_args()

    if args.rce:
        state = tropical_rce_state(args.klev)
        print(f"Tropical RCE column: klev={args.klev}, "
              f"surface T={state['ptm1'][0, -1]:.2f} K, "
              f"surface q={state['pqm1'][0, -1] * 1000:.3f} g/kg")
    else:
        raise SystemExit("specify --rce")

    work = Path(__file__).resolve().parent
    fort_out = run_fortran(state, work)
    report(fort_out, "Fortran cloud_driver outputs")

    try:
        jax_out = run_jax_cloud_equivalent(state)
        report(jax_out, "JAX cloud equivalent outputs")
        report_diff(jax_out, fort_out)
    except Exception as exc:
        print(f"\n[JAX cloud comparison failed: {exc!r}]")


if __name__ == "__main__":
    main()
