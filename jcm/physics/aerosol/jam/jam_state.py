"""Inter-term aerosol diagnostic sub-struct (the ``_jam_state`` key).

The active microphysics core writes per-step physical state needed by the
downstream harness terms (activation, dry/wet deposition, sedimentation):
per-class dry/wet radius, particle density and hygroscopicity, plus the
gathered per-class mass and number. Stored under the underscored diagnostics
key ``"_jam_state"`` (inter-term plumbing, auto-flattened to ``jam_state.*``
in xarray output, mirroring the ``_radiation`` / ``_humidity`` pattern).

The leading axis ``n_aer`` is an aerosol-*class* index — modes for a modal
scheme, bins for a sectional scheme — so this struct is agnostic to the
microphysics representation. Every field is shaped ``(n_aer, nlev, ncols)``
with the class axis first, so harness terms ``vmap``/broadcast over classes
uniformly.
"""

from __future__ import annotations

import jax.numpy as jnp
import tree_math


@tree_math.struct
class JamAerosolState:
    """Per-step aerosol physical state (per class: mode or bin)."""

    r_dry: jnp.ndarray   # dry radius [m]              (n_aer, nlev, ncols)
    r_wet: jnp.ndarray   # wet (ambient) radius [m]    (n_aer, nlev, ncols)
    rho: jnp.ndarray     # particle density [kg/m³]    (n_aer, nlev, ncols)
    kappa: jnp.ndarray   # hygroscopicity κ [-]        (n_aer, nlev, ncols)
    mass: jnp.ndarray    # total dry mass [kg/kg]      (n_aer, nlev, ncols)
    number: jnp.ndarray  # number [kg^-1]              (n_aer, nlev, ncols)

    @classmethod
    def zeros(cls, nodal_shape, nlev, n_aer) -> "JamAerosolState":
        shape = (n_aer, nlev) + tuple(nodal_shape)
        return cls(
            r_dry=jnp.zeros(shape),
            r_wet=jnp.zeros(shape),
            rho=jnp.zeros(shape),
            kappa=jnp.zeros(shape),
            mass=jnp.zeros(shape),
            number=jnp.zeros(shape),
        )

    def copy(self, **kwargs) -> "JamAerosolState":
        data = {
            "r_dry": self.r_dry,
            "r_wet": self.r_wet,
            "rho": self.rho,
            "kappa": self.kappa,
            "mass": self.mass,
            "number": self.number,
        }
        data.update(kwargs)
        return JamAerosolState(**data)
