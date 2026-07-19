"""Probe: does the jcm pySES 2m-jam step run under multi-device sharding?

pyses's JAX backend auto-shards the element axis whenever it sees more than
one device (explicit mesh + shard_map DSS). The jcm physics bridge is
element-major on the column axis, so the same block partitioning should
carry the whole physics round trip with no resharding — but the combination
has never been traced (design doc: "multi-GPU runs need these two schemes
reconciled"). This probe builds the smallest real 2m-jam model and steps it
a few times, printing the sharding of the dycore state and physics fields.

Run on a login node with virtual CPU devices (no GPU needed):

    XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    PYSES_BACKEND=jax PYSES_USE_CPU=1 PYSES_SHARD_CPU_COUNT=4 \
    JAX_PLATFORMS=cpu python tools/probe_pyses_sharding.py --nx 4 --steps 2

or on a multi-GPU node (real scaling): omit the CPU vars, request ngpus=4.
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
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--config", default="2m-jam", choices=["2m", "2m-jam"])
    args = ap.parse_args()

    import jax

    print(f"devices: {jax.devices()}")

    import jax.numpy as jnp

    import jcm
    from jcm.dycore.pyses import PysesCamSEDycore, build_forcing
    from jcm.model import Model
    from jcm.physics.echam.echam_terms import echam_physics

    bc = Path(jcm.__file__).resolve().parent / "data" / "bc" / "t63"
    t0 = time.time()
    # --- pre-mesh phase -------------------------------------------------
    # Everything that creates module-level / memoised jnp arrays must run
    # BEFORE the pyses backend installs the explicit device mesh, or those
    # arrays carry explicit-mesh typing that clashes as closure constants
    # inside the auto-mode physics region. x64 is flipped FIRST so the
    # physics' construction-time dtypes match the proven single-GPU ordering
    # (there the dycore build flipped it before physics was built).
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
    kwargs = dict(radiation_scheme="rrtmgp", cloud_scheme="2m")
    if args.config == "2m-jam":
        kwargs.update(aerosol_module="jam", jam_microphysics="mam4_jax")
        import mam4_jax.coag  # noqa: F401  (import-time lookup tables)
    physics = echam_physics(**kwargs)
    if args.config == "2m-jam":
        from jcm.physics.aerosol.jam.optics.mie_lut import default_mie_lut
        default_mie_lut()        # memoised Mie table (~4 s), pre-mesh
    from jcm.physics.radiation.rrtmgp import _ensure_rrtmgp
    _ensure_rrtmgp()             # memoised RRTMGP lookup tables, pre-mesh
    # --- mesh phase -----------------------------------------------------
    dycore = PysesCamSEDycore(
        nx=args.nx, npt=4, dt_seconds=900.0,
        physics_dtype=jnp.float32,
        terrain_file=str(bc / "terrain.nc"),
    )
    model = Model(dycore=dycore, time_step=15.0, physics=physics)
    forcing = build_forcing(str(bc / "forcing.nc"), dycore)
    print(f"[build] ne{args.nx} ncols={dycore.colmap.num_cols} "
          f"({time.time() - t0:.1f}s)")

    def show(label, arr):
        try:
            sh = arr.sharding
            kind = type(sh).__name__
            spec = getattr(sh, "spec", None)
            n_dev = len(getattr(sh, "device_set", [])) or 1
        except Exception as e:  # noqa: BLE001
            kind, spec, n_dev = f"? ({e})", None, "?"
        print(f"  {label}: shape={tuple(arr.shape)} sharding={kind} "
              f"spec={spec} devices={n_dev}")

    def show_state(label, state):
        dyn = state["model_state"]["dynamics"]
        print(f"[sharding] {label}:")
        show("horizontal_wind", dyn["horizontal_wind"])

    model.bootstrap_state(None)
    show_state("initial dycore state", model._final_dycore_state)

    t1 = time.time()
    steps_days = args.steps * 900.0 / 86400.0
    model.resume(forcing=forcing, save_interval=steps_days,
                 total_time=steps_days, output_averages=False)
    jax.block_until_ready(model._final_dycore_state)
    print(f"[run] {args.steps} steps in {time.time() - t1:.1f}s "
          "(includes compile)")

    show_state("final dycore state", model._final_dycore_state)

    t2 = time.time()
    preds2 = model.resume(forcing=forcing, save_interval=steps_days,
                          total_time=steps_days, output_averages=False)
    jax.block_until_ready(model._final_dycore_state)
    print(f"[run] {args.steps} more steps in {time.time() - t2:.1f}s "
          "(compiled)")

    import numpy as np
    ds = preds2.to_xarray()
    nan_vars = sum(bool(np.isnan(v.values).any()) for v in ds.data_vars.values())
    print(f"[health] NaN vars: {nan_vars}/{len(ds.data_vars)}")
    print("PROBE OK")


if __name__ == "__main__":
    main()
