"""Background-friendly Held-Suarez 90-day run on hybrid coords.

Usage:
    CUDA_VISIBLE_DEVICES=1 python utils/run_hs_90d.py
"""

import time
import logging

logging.basicConfig(level=logging.INFO)


def main(total_days=90.0, save_interval=5.0, output="hs_t85_47hybrid_90d"):
    import jax
    import jax.numpy as jnp
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    from jcm.model import Model
    from jcm.utils import get_coords
    from jcm.physics.icon.icon_levels import get_icon_levels
    from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics

    vertical = get_icon_levels(47)
    coords = get_coords(vertical, spectral_truncation=85)
    model = Model(
        coords=coords,
        physics=held_suarez_physics(),
        time_step=3.0,
        log_level=logging.CRITICAL,
    )
    print(f"Grid {coords.horizontal.nodal_shape}, 47 hybrid levels, dt=3 min")

    t0 = time.perf_counter()
    preds = model.run(save_interval=save_interval, total_time=total_days)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        preds._predictions,
    )
    wall = time.perf_counter() - t0
    print(f"Wall time: {wall:.0f}s ({total_days/wall*3600:.0f} sim days/hr)")

    T = preds.dynamics.temperature
    u = preds.dynamics.u_wind
    for i in range(T.shape[0]):
        Ti, ui = T[i], u[i]
        nan = float(jnp.isnan(Ti).mean())
        print(f"  day {i*save_interval:.1f}: T [{float(jnp.nanmin(Ti)):.1f},"
              f"{float(jnp.nanmax(Ti)):.1f}]  "
              f"|u|max={float(jnp.nanmax(jnp.abs(ui))):.1f}  "
              f"nan={nan:.1%}")

    # Save dataset for later inspection
    ds = preds.to_xarray()
    ds.to_netcdf(f"{output}.nc")
    print(f"Saved {output}.nc")


if __name__ == "__main__":
    main()
