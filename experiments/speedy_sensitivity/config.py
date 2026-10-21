"""CLI arguments, run tags and logging shared by the harness scripts (no jcm/JAX import).

Kept separate from common.py so the analysis/plot scripts run in any Python
environment with numpy/scipy/matplotlib/netCDF4.
"""
import time


def log(tag: str, msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


def add_config_args(ap) -> None:
    """Model-configuration CLI arguments shared by every script."""
    ap.add_argument("--truncation", type=int, default=31)
    ap.add_argument("--nlev", type=int, default=8)
    ap.add_argument("--advection", choices=("eulerian", "semi_lagrangian", "hybrid"), default="eulerian",
                    help="hybrid = Eulerian horizontal + campaign semi-Lagrangian vertical step")
    ap.add_argument("--diffusion_scale", type=float, default=1.0,
                    help="hyperdiffusion strength multiplier (timescales divided by this)")
    ap.add_argument("--sl_vertical_interpolation", choices=("cubic", "linear"), default=None,
                    help="semi-Lagrangian vertical interpolation order (default: dycore default, cubic)")
    ap.add_argument("--time_step", type=float, default=None,
                    help="minutes; default = SPEEDY's stable step for this grid")


def config_tag(args) -> str:
    adv = {"eulerian": "eul", "semi_lagrangian": "sl", "hybrid": "hyb"}[args.advection]
    tag = f"T{args.truncation}-L{args.nlev}-{adv}"
    if args.diffusion_scale != 1.0:
        tag += f"-d{args.diffusion_scale:g}"
    if getattr(args, "sl_vertical_interpolation", None) and args.advection == "semi_lagrangian":
        tag += f"-v{args.sl_vertical_interpolation}"
    return tag
