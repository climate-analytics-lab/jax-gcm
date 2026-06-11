"""Differentiable parameters for heterogeneous ice nucleation (#494)."""

from __future__ import annotations

import jax.numpy as jnp
import tree_math


@tree_math.struct
class IceNucleationParameters:
    """Calibratable knobs shared by the freezing schemes.

    The load-bearing uncertainties — the dust soluble/insoluble split, the BC
    IN efficiency, and the deposition strength — are exposed so they can be
    tuned by gradient through the model.
    """

    frac_du_soluble: jnp.ndarray      # dust immersion(soluble) vs deposition split
    bc_efficiency: jnp.ndarray        # BC IN efficiency relative to dust [-]
    deposition_scale: jnp.ndarray     # deposition active-site / number scale [-]
    scale: jnp.ndarray                # overall INP scale

    @classmethod
    def default(cls) -> "IceNucleationParameters":
        return cls(
            frac_du_soluble=jnp.asarray(0.9),
            bc_efficiency=jnp.asarray(0.01),
            deposition_scale=jnp.asarray(1.0),
            scale=jnp.asarray(1.0),
        )
