"""Verify the JAM performance levers against the compiled XLA graph.

Builds the ne30 2m-jam model with each lever set from the CLI, then:

1. times individual model steps (17 single-step ``resume`` calls; the first
   is compile and is reported separately) so the radiation-cadence pattern is
   visible — with the optics gate ON, non-radiation steps must be measurably
   cheaper than radiation steps, and vice versa contrastable with the gate
   OFF;
2. optionally wraps three steps in a ``jax.profiler`` trace (``--trace-dir``);
3. if ``XLA_FLAGS=--xla_dump_to=...`` is set by the caller, the compiled
   HLO of the step executable lands there — count ``f64[`` occurrences to
   confirm the float32 MAM4 core actually removed the double-precision ops
   (the dynamics' f64 remains by design).

Usage (inside a GPU job; see runs/derecho_verify_levers.pbs):

    python tools/verify_perf_levers.py --core-dtype float32 --gate on --cre on
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYSES_BACKEND", "jax")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--cre", default="on", choices=["on", "off"])
    ap.add_argument("--gate", default="on", choices=["on", "off"])
    ap.add_argument("--nx", type=int, default=30)
    ap.add_argument("--steps", type=int, default=17)
    ap.add_argument("--trace-dir", default=None,
                    help="write a jax.profiler trace of 3 extra steps here")
    ap.add_argument("--emissions-file", default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    import jcm
    from jcm.model import Model
    from jcm.physics.echam.echam_terms import echam_physics

    bc = Path(jcm.__file__).resolve().parent / "data" / "bc" / "t63"

    # Pre-mesh phase (see run_pyses_climatology.build for the rationale).
    jax.config.update("jax_enable_x64", True)
    import mam4_jax.coag  # noqa: F401
    from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
        Mam4JaxMicrophysics,
    )
    core = Mam4JaxMicrophysics(core_dtype=args.core_dtype)
    physics = echam_physics(
        radiation_scheme="rrtmgp",
        radiation_compute_cre=(args.cre == "on"),
        cloud_scheme="2m", aerosol_module="jam", jam_microphysics=core,
    )
    if args.gate == "off":
        for t in physics.terms:
            if hasattr(t, "configure_radiation_gate"):
                t.configure_radiation_gate(0.0)
    from jcm.physics.aerosol.jam.optics.mie_lut import default_mie_lut
    default_mie_lut()
    from jcm.physics.radiation.rrtmgp import _ensure_rrtmgp
    _ensure_rrtmgp()

    from jcm.dycore.pyses import PysesCamSEDycore, build_forcing
    dycore = PysesCamSEDycore(
        nx=args.nx, npt=4, dt_seconds=900.0, physics_dtype=jnp.float32,
        terrain_file=str(bc / "terrain.nc"),
        coupling="hybrid", hypervis="quasi_uniform",
        nu_top=2.5e5, tracer_substeps=5,
    )
    model = Model(dycore=dycore, time_step=15.0, physics=physics)
    forcing = build_forcing(str(bc / "forcing.nc"), dycore,
                            emissions_file=args.emissions_file)
    model.bootstrap_state(None)

    step_days = 900.0 / 86400.0
    label = f"core={args.core_dtype} cre={args.cre} gate={args.gate}"
    print(f"[cfg] {label} ne{args.nx}")

    times = []
    for i in range(args.steps):
        t0 = time.time()
        model.resume(forcing=forcing, save_interval=step_days,
                     total_time=step_days, output_averages=False)
        jax.block_until_ready(model._final_dycore_state)
        dt_wall = time.time() - t0
        times.append(dt_wall)
        tag = "compile" if i == 0 else ("RAD" if i % 8 == 0 else "   ")
        print(f"[step {i:3d}] {dt_wall*1000.0:9.1f} ms {tag}")

    steady = times[1:]
    rad = [t for i, t in enumerate(steady, start=1) if i % 8 == 0]
    non = [t for i, t in enumerate(steady, start=1) if i % 8 != 0]
    print(f"[summary] {label}")
    print(f"  compile: {times[0]:.1f} s")
    if rad:
        print(f"  radiation steps: {1000*sum(rad)/len(rad):9.1f} ms mean (n={len(rad)})")
    print(f"  other steps:     {1000*sum(non)/len(non):9.1f} ms mean (n={len(non)})")
    per_day = 96.0 * (sum(steady) / len(steady))
    print(f"  implied: {3600.0/per_day:.1f} sim days/hr")

    if args.trace_dir:
        with jax.profiler.trace(args.trace_dir):
            for _ in range(3):
                model.resume(forcing=forcing, save_interval=step_days,
                             total_time=step_days, output_averages=False)
                jax.block_until_ready(model._final_dycore_state)
        print(f"[trace] wrote {args.trace_dir}")


if __name__ == "__main__":
    main()
