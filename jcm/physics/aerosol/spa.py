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

    Nc_min [cm^-3] = SPA_PREFACTOR * (Nccn * cloud_fraction) ** SPA_EXPONENT

with SPA_PREFACTOR = 2000 and SPA_EXPONENT = 0.55. The slope falls in the
0.3 ≤ d ln Nc / d ln Nccn ≤ 0.8 range constrained by observations.
"""

import jax.numpy as jnp


# Lin et al. (2025) sublinear fit to E3SMv3 climatology.
SPA_PREFACTOR: float = 2000.0
SPA_EXPONENT: float = 0.55

# Conversion factor from cm^-3 (the units of `Nccn` and the SPA fit output)
# to m^-3 (the convention used inside the two-moment microphysics).
_CM3_TO_M3: float = 1.0e6


def spa_activated_cdnc(Nccn: jnp.ndarray, cloud_fraction: jnp.ndarray,
                       prefactor: float = SPA_PREFACTOR,
                       exponent: float = SPA_EXPONENT) -> jnp.ndarray:
    """Per-cell SPA-style cloud-droplet floor `Nc_min`, in m^-3.

    Args:
        Nccn: Cloud condensation nuclei concentration [cm^-3]. Typically
            broadcast from the column-mean MACv2-SP CCN value to every
            level (so vertical aerosol structure is not resolved here);
            shape can be either ``(..., ncols)`` or ``(..., nlev, ncols)``.
        cloud_fraction: Cloud fraction in [0, 1], same shape as the
            broadcast target.
        prefactor: SPA fit coefficient (default 2000, matches Lin 2025).
        exponent: SPA fit exponent (default 0.55).

    Returns:
        `Nc_min` in m^-3, ready to be passed as `activated_cdnc` to
        `cloud_microphysics_2m`. Zero where ``cloud_fraction == 0``.

    """
    arg = jnp.maximum(Nccn * cloud_fraction, 0.0)
    nc_min_cm3 = prefactor * arg ** exponent
    return nc_min_cm3 * _CM3_TO_M3
