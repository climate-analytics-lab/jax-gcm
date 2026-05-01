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
                       prefactor: jnp.ndarray, exponent: jnp.ndarray) -> jnp.ndarray:
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

    Returns:
        `Nc_min` in m^-3, ready to be passed as `activated_cdnc` to
        `cloud_microphysics_2m`. Zero where ``cloud_fraction == 0``.

    """
    arg = jnp.maximum(Nccn * cloud_fraction, 0.0)
    nc_min_cm3 = prefactor * arg ** exponent
    return nc_min_cm3 * _CM3_TO_M3
