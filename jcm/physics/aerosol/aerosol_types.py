"""Aerosol diagnostic sub-struct.

Currently used by the MACv2-SP simple-plume scheme; lives next to it so
new aerosol schemes can either reuse or extend ``AerosolData`` without
reaching into the ECHAM tree.
"""

from __future__ import annotations

import jax.numpy as jnp
import tree_math


@tree_math.struct
class AerosolData:
    """Aerosol optical properties and CCN concentration.

    The Twomey effect on cloud droplet number concentration
    (``cdnc_factor``) and the activated CCN field (``Nccn``) are the
    cross-scheme outputs consumed by the cloud microphysics path.
    """

    # Aerosol optical properties by level
    aod_profile: jnp.ndarray         # AOD profile [1] (nlev, ncols)
    ssa_profile: jnp.ndarray         # SSA profile [1] (nlev, ncols)
    asy_profile: jnp.ndarray         # Asymmetry parameter profile [1] (nlev, ncols)

    # Column-integrated properties
    aod_total: jnp.ndarray           # Total column AOD [1] (ncols,)
    aod_anthropogenic: jnp.ndarray   # Anthropogenic AOD [1] (ncols,)
    aod_background: jnp.ndarray      # Background AOD [1] (ncols,)

    # For Twomey effect (cloud-aerosol interactions)
    cdnc_factor: jnp.ndarray         # CDNC modification factor [1] (ncols,)

    # Cloud condensation nuclei number concentration [cm^-3] (ncols,).
    # Derived from MACv2-SP plumes (anthropogenic + background AOD via the
    # AEROCOM-P1 Twomey relation) and consumed by the SPA-style activation
    # in the 2M microphysics path. See ``jcm.physics.aerosol.spa``.
    Nccn: jnp.ndarray

    # Spectral scaling
    angstrom: jnp.ndarray            # Angstrom exponent [1] (ncols,)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            aod_profile=jnp.zeros((nlev,) + nodal_shape),
            ssa_profile=jnp.zeros((nlev,) + nodal_shape),
            asy_profile=jnp.zeros((nlev,) + nodal_shape),
            aod_total=jnp.zeros(nodal_shape),
            aod_anthropogenic=jnp.zeros(nodal_shape),
            aod_background=jnp.zeros(nodal_shape),
            cdnc_factor=jnp.ones(nodal_shape),  # Start with factor of 1.0
            Nccn=jnp.zeros(nodal_shape),
            angstrom=jnp.ones(nodal_shape) * 1.5,  # Typical fine-mode aerosol
        )

    def copy(self, **kwargs):
        new_data = {
            'aod_profile': self.aod_profile,
            'ssa_profile': self.ssa_profile,
            'asy_profile': self.asy_profile,
            'aod_total': self.aod_total,
            'aod_anthropogenic': self.aod_anthropogenic,
            'aod_background': self.aod_background,
            'cdnc_factor': self.cdnc_factor,
            'Nccn': self.Nccn,
            'angstrom': self.angstrom,
        }
        new_data.update(kwargs)
        return AerosolData(**new_data)
