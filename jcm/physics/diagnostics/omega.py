"""Model-agnostic pressure-vertical-velocity output diagnostic.

The dycore computes omega = Dp/Dt [Pa/s] (see
``DinosaurDycore(compute_omega=True)``, jax-gcm#409) and injects it as the
``"omega"`` dycore field; this term simply republishes that field as a
top-level ``"omega"`` diagnostic so it reaches saved output under any
physics package (SPEEDY, Held-Suarez, ECHAM, ...). It is deliberately not
an AeroCom-ism: the AeroCom ``plev`` group separately derives its
wap/w500/w700 submission fields from the same dycore field.

Composing this term makes the provider a hard requirement (Model
construction fails with a pointed error if the dycore flag is off), which
is the honest failure mode for an explicitly requested diagnostic. The
``jcm run`` CLI enables the provider automatically when the composed
physics requires it.
"""

from typing import ClassVar

import jax.numpy as jnp

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency


class OmegaDiagnostic(PhysicsTerm):
    """Republish the dycore-supplied omega as an output diagnostic."""

    name: ClassVar[str] = "omega_diagnostic"
    category: ClassVar[str] = "diagnostics"
    provides: ClassVar[tuple[str, ...]] = ("omega",)
    requires_dycore_fields: ClassVar[tuple[str, ...]] = ("omega",)

    def __call__(self, state, diagnostics, forcing, terrain):
        # ``get`` fallback: the ``get_empty_data`` structural probe runs
        # terms without dycore-field injection, and consumers must
        # tolerate absence with an identically-structured result. Omega
        # shares temperature's (nlev, *horiz) shape and dtype.
        fields = diagnostics.get("_dycore_fields", {})
        omega = fields.get("omega", jnp.zeros_like(state.temperature))
        tend = PhysicsTendency.zeros(state.temperature.shape)
        return tend, {**diagnostics, "omega": omega}
