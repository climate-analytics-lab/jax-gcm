"""Shared radiation band configuration.

A small frozen-dataclass container so wavelength-dependent terms
(aerosols, prescribed gas profiles, etc.) don't have to hardcode band
centers from whichever radiation backend the user happens to be running.
Owned by :class:`~jcm.physics.composable_physics.ComposablePhysics` and
injected into ``diagnostics["_band_config"]`` at the top of every
``compute_tendencies`` call (same pattern as ``_dt_seconds``).

Stored as ``tuple[float, ...]`` rather than ``jnp.ndarray`` so the
config is a pure Python static value: it lives on the ``nnx.Module``
without being treated as a traced pytree leaf, hashes as part of the
JIT cache key without re-tracing on every step, and consumers convert
to ``jnp.asarray`` at the point of use.

The band centers come from the live radiation backend via
:meth:`from_rrtmgp` (queries the gas-optics ``bnd_lims_wn`` arrays).
For grey two-stream / SPEEDY radiation there's no band concept, so
:meth:`broadband` returns a single 550 nm "band" — wavelength-dependent
terms that consume this fall back to using their 550 nm references.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np


@dataclasses.dataclass(frozen=True)
class RadiationBandConfig:
    """Band-center wavelengths (nm) for the active radiation backend.

    Both fields store the *band-center* wavelength in nanometres,
    computed from the band wavenumber limits as
    ``λ = 1e7 / (0.5 * (wn_lo + wn_hi))``.
    """

    # Empty tuple when the backend has no LW bands (e.g., a SW-only
    # proxy or grey radiation).
    lw_band_centers_nm: tuple[float, ...]
    # Length-1 ``(550.0,)`` for grey/SPEEDY radiation.
    sw_band_centers_nm: tuple[float, ...]

    @classmethod
    def from_rrtmgp(cls, rrtmgp_instance) -> "RadiationBandConfig":
        """Build from a live ``rrtmgp.RRTMGP`` instance.

        Reads ``optics_lib.gas_optics_{lw,sw}.bnd_lims_wn`` (cm⁻¹) and
        converts to nm centers.
        """
        gl = rrtmgp_instance.optics_lib.gas_optics_lw
        gs = rrtmgp_instance.optics_lib.gas_optics_sw
        lw_lims = np.asarray(gl.bnd_lims_wn)
        sw_lims = np.asarray(gs.bnd_lims_wn)
        return cls(
            lw_band_centers_nm=tuple(
                float(x) for x in 1e7 / (0.5 * (lw_lims[:, 0] + lw_lims[:, 1]))
            ),
            sw_band_centers_nm=tuple(
                float(x) for x in 1e7 / (0.5 * (sw_lims[:, 0] + sw_lims[:, 1]))
            ),
        )

    @classmethod
    def broadband(cls) -> "RadiationBandConfig":
        """Single-band fallback (550 nm SW, no LW bands).

        Use for grey two-stream and SPEEDY radiation where there is no
        meaningful band structure — wavelength-dependent terms that
        consume this fall back to their 550 nm reference values.
        """
        return cls(lw_band_centers_nm=(), sw_band_centers_nm=(550.0,))


# Register as an *opaque static* pytree node so JAX leaves the config
# alone when flattening containers it lives in (the diagnostics dict,
# the ``ComposablePhysics`` module). All data is carried in the aux
# slot — frozen dataclass + tuple fields make it hashable, so it can
# back JIT cache keys without re-tracing on every step.
jax.tree_util.register_pytree_node(
    RadiationBandConfig,
    flatten_func=lambda cfg: ((), cfg),
    unflatten_func=lambda aux, _children: aux,
)
