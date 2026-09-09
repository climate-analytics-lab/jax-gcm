"""The 10 m wind the surface-flux emission schemes are calibrated to (#723).

Gong sea salt (``u10**3.41``) and Nightingale DMS (``k_w ~ u10**2``) are both
fitted to the 10 m wind — HAMMOZ passes ``vphysc%velo10m`` — so reading the
lowest model level instead biases them high by the surface-layer shear over
10 m … z₁ (~33 m at L47: +37-46 % sea salt, +20-25 % DMS).
"""

from __future__ import annotations

import jax.numpy as jnp


def wind_10m(state, diagnostics) -> jnp.ndarray:
    """Diagnosed 10 m wind speed [m/s], shaped ``(ncols,)``.

    Reads the ECHAM ``nsurf_diag`` reduction published by the vertical-diffusion
    term (which owns the surface-layer profile). Emission terms run before
    vdiff in the ECHAM ordering, so this is the previous step's 10 m wind —
    the same one-step lag the dust term's ``u*`` already carries.

    The lowest model level is the fallback wherever no reduction has been
    diagnosed: with no vertical-diffusion term composed there is no surface
    layer to reduce through, and on the first step of a run its carry is still
    zero (a zero 10 m wind under a moving lowest level can only mean "not
    diagnosed yet", and emitting nothing for a step is worse than emitting
    unreduced).
    """
    lowest = jnp.sqrt(
        jnp.maximum(state.u_wind[-1] ** 2 + state.v_wind[-1] ** 2, 1.0e-30)
    )
    vdiff = diagnostics.get("vertical_diffusion")
    speed_10m = getattr(vdiff, "wind_10m", None) if vdiff is not None else None
    if speed_10m is None:
        return lowest
    speed_10m = jnp.ravel(speed_10m)
    return jnp.where(speed_10m > 0.0, speed_10m, lowest)
