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

`spa_activation_potential` (jax-gcm#1160) is a second entry point onto the
SAME power law, WITHOUT the ``cloud_fraction`` factor: ECHAM-HAM's own
activation diagnostic, ``zcdncact``/``cdncact_cv`` (``cloud_subm_1``,
``mo_ham_activ.f90``/``mo_activ.f90``), is never cloud-fraction-weighted —
it is a per-cell activation *potential* ("the number ARG would activate
here"), and every Fortran site that reads it
(``mo_cloud_micro_2m.f90``:749, 905-917, 2594) uses it exactly that way.
`spa_activated_cdnc`'s ``cloud_fraction`` multiply/cap is a deliberate SPA
design choice for ITS OWN use as a grid-mean `Nc_min` floor (the physical
bound discussed on its own docstring), not a port of anything in the
Fortran — correct for the floor, but wrong as a stand-in for ECHAM's
unmasked ``zcdncact`` at a *different* cell than the one the floor was
masked for (jax-gcm#1160: gathering `activated_cdnc` at a convective cloud
base that sits in a clear-sky stratiform cell reads an exact zero, because
the floor was masked to THAT stratiform cell's cover, not the convective
column's aerosol availability). `spa_activation_potential` is what
`Lohmann2MMicrophysics` now feeds the two section-1 droplet-number sources
that need a true activation potential, leaving `spa_activated_cdnc` itself,
and every existing caller of it, unchanged.
"""

import jax.numpy as jnp


# Conversion factor from cm^-3 (the units of `Nccn` and the SPA fit output)
# to m^-3 (the convention used inside the two-moment microphysics).
_CM3_TO_M3: float = 1.0e6


def _power_law_capped(arg: jnp.ndarray, prefactor: jnp.ndarray,
                      exponent: jnp.ndarray,
                      cap_smoothing: jnp.ndarray | float) -> jnp.ndarray:
    """Lin (2025)'s sublinear power law, capped at its own argument.

    Shared by `spa_activated_cdnc` (``arg = Nccn·cloud_fraction``) and
    `spa_activation_potential` (``arg = Nccn``) — the fit and its physical
    cap are identical; only what `arg` represents differs between the two
    callers.
    """
    # Double-where guard: ``arg`` is 0 in a clear cell (the common case for
    # the cover-weighted caller) and the fractional ``arg ** exponent`` has
    # an infinite derivative at 0, poisoning the reverse pass while the
    # forward is 0 (issue #558). Keep the forward exactly 0 there;
    # differentiate the power only on a strictly-positive floored argument.
    fit = prefactor * jnp.where(
        arg > 0.0, jnp.maximum(arg, 1.0e-30) ** exponent, 0.0,
    )
    # Physical bound: the result cannot exceed the CCN that actually entered
    # the fit (``arg``). The power-law only overshoots at very low aerosol;
    # capping at ``arg`` keeps that overshoot from activating more droplets
    # than the CCN available to `arg`'s own area/column. With `arg = Nccn`
    # (no cloud-fraction factor) this is "no more droplets than CCN"; with
    # `arg = Nccn·cloud_fraction` it is `spa_activated_cdnc`'s own partial-
    # cloud bound (Cf=1 reduces both to the same ``min(·, Nccn)``).
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
    gap = fit - arg
    smooth_cap = 0.5 * (fit + arg - jnp.sqrt(gap * gap + w_safe * w_safe))
    capped = jnp.where(smoothing_on, smooth_cap, jnp.minimum(fit, arg))
    return jnp.maximum(capped, 0.0)


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
        `cloud_microphysics_2m`. Zero where ``cloud_fraction == 0`` — a
        deliberate, GRID-MEAN floor (see the module docstring); callers
        that need the per-cell activation POTENTIAL instead (unmasked by
        cloud cover, matching ECHAM's own `zcdncact`/`cdncact_cv`) want
        `spa_activation_potential`, not this function.

    """
    # Lin (2025)'s fit (prefactor 2000, exponent 0.55) is calibrated with
    # N_ccn in SI **m^-3**, not cm^-3 — applying it to cm^-3 values
    # overestimates N_c by ~(1e6)^0.45 ≈ 4e2, giving >1e4 cm^-3 (more droplets
    # than CCN). MACv2-SP supplies N_ccn in cm^-3, so convert to m^-3 first; the
    # result is then already in m^-3 (no further unit conversion).
    nccn_m3 = jnp.maximum(Nccn, 0.0) * _CM3_TO_M3
    arg = jnp.maximum(nccn_m3 * cloud_fraction, 0.0)
    return _power_law_capped(arg, prefactor, exponent, cap_smoothing)


def spa_activation_potential(Nccn: jnp.ndarray, prefactor: jnp.ndarray,
                             exponent: jnp.ndarray,
                             cap_smoothing: jnp.ndarray | float = 0.0,
                             ) -> jnp.ndarray:
    """Per-cell SPA-style activation POTENTIAL, in m^-3 (jax-gcm#1160).

    The SAME Lin (2025) power law as `spa_activated_cdnc`, but evaluated on
    the full-column ``Nccn`` with no ``cloud_fraction`` factor — matching
    what ECHAM-HAM's own `zcdncact`/`cdncact_cv` carry (see the module
    docstring): an activation diagnostic that exists at every cell
    regardless of whether THAT cell happens to be cloudy, because the
    cloud-base/convective-detrainment sources that consume it apply their
    OWN "is this cell applicable" gate (a cloud-base search, a convective-
    detrainment flag) rather than relying on the value itself going to
    zero in clear air.

    Args:
        Nccn: Cloud condensation nuclei concentration [cm^-3]. Same
            broadcasting convention as `spa_activated_cdnc`.
        prefactor, exponent, cap_smoothing: Same as `spa_activated_cdnc`.

    Returns:
        The activation potential in m^-3. Zero only where ``Nccn <= 0``
        (never cloud-fraction-masked) — never less than
        `spa_activated_cdnc`'s floor at the same cell, since that floor's
        extra ``cloud_fraction <= 1`` factor can only shrink the argument
        to the (monotonic, sub-``arg``-capped) power law.

    """
    nccn_m3 = jnp.maximum(Nccn, 0.0) * _CM3_TO_M3
    return _power_law_capped(nccn_m3, prefactor, exponent, cap_smoothing)
