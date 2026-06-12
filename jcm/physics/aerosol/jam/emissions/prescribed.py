"""``PreSpeciatedEmissions`` — CAM6/MAM4-faithful prescribed modal emissions (#498).

The companion to :class:`AnthropogenicEmissions`. Where that term takes *bulk*
SO₂/BC/OC and does the speciation + smooth injection in-model (so injection
height and the primary-SO₄ fraction are differentiable), this term mirrors how
**CAM6 actually applies emissions**: it reads **already-speciated** per-tracer
emission fields and injects them directly, with no in-model speciation or
injection-profile parameters. The split into modes/species and the vertical
placement were done offline when the emission files were built (CAM does the same
— ``mo_srf_emissions`` for surface fields, ``mo_extfrc`` for altitude-resolved
"external forcing" fields).

This is deliberately **representation-agnostic**: each ``ForcingData`` field is
keyed by the *tracer name it feeds* (e.g. ``m_so4_acc``, ``m_bc_pcm``, ``n_pcm``,
``g_so2``) and simply added to that tracer's tendency — the term knows nothing
about MAM4 specifically, so any aerosol layout works. A field is applied as:

* **surface** — shape ``(ncols,)`` (a 2-D ``(lon, lat)`` field after
  ``select``): a surface mass/number flux [X/m²/s] added to the **lowest layer**.
* **volume** — shape ``(nlev, ncols)`` (a 3-D ``(lev, lon, lat)`` field): a
  per-model-layer flux [X/m²/s per layer] added across levels (the ``mo_extfrc``
  analogue for elevated emissions pre-distributed onto model levels).

Both are mass-conserving (``Σ_k ρ_k Δz_k · dq_k = Σ_k flux_k``). There is no
injection-height/SO₄ parameter here — but because the emission fields are
differentiable ``ForcingData`` leaves entering the tracer tendencies linearly,
``∂(aerosol mmr)/∂(emission field)`` is still well-defined and finite, so the
emission magnitudes remain calibratable by gradient.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm


class PreSpeciatedEmissions(PhysicsTerm):
    """Inject prescribed, already-speciated per-tracer emission fields directly."""

    name: ClassVar[str] = "jam_prescribed_aerosol_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, scale: float = 1.0):
        """Hold an overall (differentiable) emission scale."""
        # A single multiplicative knob, handy for sensitivity runs / calibration
        # of the whole prescribed source without per-field plumbing.
        self.scale = nnx.Param(jnp.asarray(scale))

    def __call__(self, state, diagnostics, forcing, terrain):
        rho = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape
        scale = self.scale.get_value()

        emis = (getattr(forcing, "prescribed_aerosol_emissions", None)
                if forcing is not None else None)

        tends: dict[str, jnp.ndarray] = {}
        if emis:
            inv = 1.0 / (rho * dz)               # (nlev, ncols) mmr per kg/m²
            for tracer, field in emis.items():
                f = scale * jnp.asarray(field)
                if jnp.size(f) == ncols:
                    # Surface flux → lowest layer only.
                    dq = jnp.zeros((nlev, ncols)).at[-1].set(
                        f.ravel() * inv[-1])
                else:
                    # Per-layer volume flux already on model levels.
                    dq = f.reshape(nlev, ncols) * inv
                tends[tracer] = tends.get(tracer, jnp.zeros((nlev, ncols))) + dq

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tends,
        )
        return tendency, diagnostics
