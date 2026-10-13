"""SPA-style cloud-droplet activation floor.

Implements the prescribed-aerosol activation rule from Lin et al. (2025,
*Atmos. Chem. Phys.*, https://acp.copernicus.org/articles/25/15105/2025/),
adapted to take its CCN input from the MACv2-SP plumes already computed in
`jcm.physics.aerosol.macv2_sp`. Two-moment microphysics (#341) calls
`spa_activated_cdnc` once per step to obtain the per-grid-cell droplet
floor `Nc_min`; the microphysics then evolves `Nc` subject to that floor:

    Nc <- max(Nc, Nc_min)

The SPA paper found that a *linear* CCN→Nc activation overestimates the
indirect aerosol effect; their analysis of E3SMv3 climatology yields a
sublinear power-law with exponent ~0.55:

    Nc_min [cm^-3] = prefactor * (Nccn * cloud_fraction) ** exponent

The fit values from Lin (2025) are ``prefactor = 2000`` and
``exponent = 0.55`` — those live as defaults on
:class:`jcm.physics.aerosol.macv2_sp_params.AerosolParameters` so they are
differentiable through ``jax.grad`` (for calibration / sensitivity work).
"""

import jax.numpy as jnp


# Conversion factor from cm^-3 (the units of `Nccn` and the SPA fit output)
# to m^-3 (the convention used inside the two-moment microphysics).
_CM3_TO_M3: float = 1.0e6


def spa_activated_cdnc(Nccn: jnp.ndarray, cloud_fraction: jnp.ndarray,
                       prefactor: jnp.ndarray, exponent: jnp.ndarray,
                       cap_smoothing: jnp.ndarray | float = 0.0) -> jnp.ndarray:
    """Per-cell SPA-style cloud-droplet floor `Nc_min`, in m^-3.

    Args:
        Nccn: Cloud condensation nuclei concentration [cm^-3]. Typically
            broadcast from the column-mean MACv2-SP CCN value to every
            level (so vertical aerosol structure is not resolved here);
            shape can be either ``(..., ncols)`` or ``(..., nlev, ncols)``.
        cloud_fraction: Cloud fraction in [0, 1], same shape as the
            broadcast target.
        prefactor: SPA fit coefficient. Lin (2025) gives 2000 — typically
            sourced from ``AerosolParameters.spa_prefactor`` so that it
            is differentiable.
        exponent: SPA fit exponent. Lin (2025) gives 0.55 — typically
            sourced from ``AerosolParameters.spa_exponent``.
        cap_smoothing: Half-width [m^-3] of the smooth replacement for the
            hard ``min(Nc_min, arg)`` physical cap; 0 (the default) keeps
            the hard cap. Typically sourced from
            ``AerosolParameters.spa_cap_smoothing``. See the note on that
            field: the hard cap makes the loss piecewise in ``prefactor``
            and the exact local gradient unrepresentative of the
            large-scale response, which is what blocks gradient
            calibration of the SPA fit.

    Returns:
        `Nc_min` in m^-3, ready to be passed as `activated_cdnc` to
        `cloud_microphysics_2m`. Zero where ``cloud_fraction == 0``.

    """
    # Lin (2025)'s fit (prefactor 2000, exponent 0.55) is calibrated with
    # N_ccn in SI **m^-3**, not cm^-3 — applying it to cm^-3 values
    # overestimates N_c by ~(1e6)^0.45 ≈ 4e2, giving >1e4 cm^-3 (more droplets
    # than CCN). MACv2-SP supplies N_ccn in cm^-3, so convert to m^-3 first; the
    # result is then already in m^-3 (no further unit conversion).
    nccn_m3 = jnp.maximum(Nccn, 0.0) * _CM3_TO_M3
    arg = jnp.maximum(nccn_m3 * cloud_fraction, 0.0)
    # Double-where guard: ``arg`` is 0 in a clear cell (cloud_fraction == 0,
    # the common case) and the fractional ``arg ** exponent`` has an infinite
    # derivative at 0, poisoning the reverse pass while the forward is 0
    # (issue #558). Keep the forward exactly 0 there; differentiate the power
    # only on a strictly-positive floored argument.
    nc_min_m3 = prefactor * jnp.where(
        arg > 0.0, jnp.maximum(arg, 1.0e-30) ** exponent, 0.0,
    )
    # Physical bound: the (grid-mean) droplet floor cannot exceed the cloudy CCN
    # that actually entered the activation fit — i.e. ``arg = Nccn·cloud_fraction``,
    # NOT the full-column ``Nccn``. The power-law only overshoots at very low
    # aerosol; capping at ``arg`` keeps partial-cloud clean cells from
    # activating more droplets than the CCN in their cloudy area (Cf<1). With
    # Cf=1 this reduces to the previous ``min(·, Nccn)``.
    #
    # The cap is a smooth (hyperbolic) minimum when ``cap_smoothing > 0``:
    # ``0.5*(a + b - sqrt((a-b)^2 + w^2))`` undershoots ``min(a, b)`` by at
    # most ``w/2`` at the corner and is exact away from it. The cap boundary
    # (``prefactor = arg**(1-exponent)``) crosses the middle of the cloudy-cell
    # distribution, so with the hard min every prefactor perturbation flips
    # cells between branches and the exact local gradient is a poor (even
    # wrong-signed) estimate of the response; the smooth cap makes
    # d/d(prefactor) reflect the trend at the width scale. Double-where on the
    # width so ``cap_smoothing == 0`` recovers the hard min exactly without the
    # unselected branch differentiating ``sqrt`` at zero. The final clamp keeps
    # clear cells (both operands 0, smooth min -w/2) at exactly zero.
    smoothing_on = cap_smoothing > 0.0
    w_safe = jnp.where(smoothing_on, cap_smoothing, 1.0)
    gap = nc_min_m3 - arg
    smooth_cap = 0.5 * (nc_min_m3 + arg - jnp.sqrt(gap * gap + w_safe * w_safe))
    capped = jnp.where(smoothing_on, smooth_cap, jnp.minimum(nc_min_m3, arg))
    return jnp.maximum(capped, 0.0)
