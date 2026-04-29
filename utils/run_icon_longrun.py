r"""Long ICON simulation with periodic health checks.

Builds a model from ``jcm.runners.build_model`` (the same builder used by
``python -m jcm.main``) and runs it in chunks, calling
``jcm.diagnostics.check_health`` between each chunk and stopping early on
the first NaN/extreme-value alarm.

Usage::

    CUDA_VISIBLE_DEVICES=6 python utils/run_icon_longrun.py \
        --years 3 --radiation grey --sigma --output icon_3yr
"""

from __future__ import annotations

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)


def _build_cfg(args):
    """Compose a Hydra ``DictConfig`` for the requested ICON setup."""
    from hydra import compose, initialize_config_module

    overrides = [
        "physics=icon",
        f"physics.radiation={args.radiation}",
        "grid=" + ("icon_t42_l8_sigma" if args.sigma else "icon_t85_l47_hybrid"),
        f"run.time_step={args.dt_min}",
        f"run.save_interval={args.save_interval}",
        f"run.total_time={args.years * 365.25}",
        "run.log_level=INFO",
        f"diffusion.scale={args.diffusion_scale}",
    ]
    if args.sponge_levels > 0:
        overrides += [
            f"run.sponge.levels={args.sponge_levels}",
            f"run.sponge.timescale_h={args.sponge_timescale_h}",
            f"run.sponge.enspodi={args.sponge_enspodi}",
        ]
    if args.jw_init:
        overrides.append("init=jw")

    with initialize_config_module(version_base=None, config_module="jcm.config"):
        return compose(config_name="config", overrides=overrides)


def main():
    parser = argparse.ArgumentParser(description="Long ICON run with health monitoring")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--radiation", default="grey",
                        choices=("grey", "emulated", "rrtmgp"))
    parser.add_argument("--sigma", action="store_true",
                        help="Use equidistant sigma levels (T42/L8) instead of "
                             "T85/L47 hybrid coords")
    parser.add_argument("--output", default="icon_longrun")
    parser.add_argument("--chunk_days", type=float, default=90.0)
    parser.add_argument("--save_interval", type=float, default=30.0)
    parser.add_argument("--dt_min", type=float, default=30.0)
    parser.add_argument("--jw_init", action="store_true",
                        help="Inject a JW lapse-rate initial condition")
    parser.add_argument("--diffusion_scale", type=float, default=1.0)
    parser.add_argument("--sponge_levels", type=int, default=0)
    parser.add_argument("--sponge_timescale_h", type=float, default=3.0)
    parser.add_argument("--sponge_enspodi", type=float, default=2.0)
    args = parser.parse_args()

    import jax
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    from jcm.runners import build_model, run_chunked
    cfg = _build_cfg(args)
    model = build_model(cfg)
    print(f"Timestep: {args.dt_min:.1f} min ({args.dt_min*60:.0f} s)")
    print(f"\n=== Starting {args.years:.0f}-year run, {args.chunk_days:.0f}-day chunks ===")

    reports = run_chunked(
        cfg, chunk_days=args.chunk_days,
        output_prefix=args.output, model=model,
    )

    report_path = f"{args.output}_reports.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"Saved health reports to {report_path}")


if __name__ == "__main__":
    main()
