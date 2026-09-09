"""The 10 m wind the surface-flux emission schemes are calibrated to (#723).

Gong sea salt (``u10**3.41``) and Nightingale DMS (``k_w ~ u10**2``) are both
fitted to the 10 m wind — HAMMOZ passes ``vphysc%velo10m`` — so reading the
lowest model level instead biases them high by the surface-layer shear over
10 m … z₁ (~33 m at L47: +37-46 % sea salt, +20-25 % DMS).
"""

from __future__ import annotations

import jax.numpy as jnp

#: Per-column flag published by the emission terms: 1 where the emission wind
#: fell back to the lowest model level, 0 where the diagnosed 10 m wind was
#: used. Zeroed every step with the other emission diagnostics, so a run in
#: which the fallback persists is visible instead of quietly emitting 37-46 %
#: too much sea salt. Deliberately NOT an ``emis_``/``emi_`` name: those
#: prefixes select emission *fluxes* in the forcing reader and the burden
#: report.
MODEL_LEVEL_WIND_KEY = "wind_10m_model_level"


def wind_10m(state, diagnostics):
    """Emission wind [m/s] and its provenance, both shaped ``(ncols,)``.

    Returns ``(speed, from_model_level)`` — the ECHAM ``nsurf_diag`` 10 m
    reduction published by the vertical-diffusion term, and a 0/1 flag per
    column saying where that was unavailable and the lowest model level was
    used instead.

    Emission terms run before vdiff in the ECHAM ordering, so the value read
    here is the previous step's. ``vertical_diffusion`` is a **declared**
    cross-step slot (``Physics.initial_carry_state``), so this is the
    deliberate carry #673 distinguishes from an undeclared stale read, and the
    same one-step lag the dust term's ``u*`` already carries.

    The fallback is therefore reachable only where no 10 m wind can exist: on
    step 1 of a cold start, whose carry slot is still zero-filled because no
    step has diagnosed a surface layer yet, or with no vdiff term composed at
    all. A resumed run carries a real 10 m wind and never takes it.
    """
    lowest = jnp.sqrt(
        jnp.maximum(state.u_wind[-1] ** 2 + state.v_wind[-1] ** 2, 1.0e-30)
    )
    vdiff = diagnostics.get("vertical_diffusion")
    speed_10m = getattr(vdiff, "wind_10m", None) if vdiff is not None else None
    if speed_10m is None:
        return lowest, jnp.ones_like(lowest)
    speed_10m = jnp.ravel(speed_10m)
    # A zero 10 m wind under a moving lowest level can only mean "not
    # diagnosed yet": the reduction is strictly positive wherever it has run.
    unset = speed_10m <= 0.0
    return (jnp.where(unset, lowest, speed_10m),
            unset.astype(lowest.dtype))
