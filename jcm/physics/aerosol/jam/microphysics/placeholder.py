"""κ-Köhler equilibrium placeholder microphysics core.

Computes diagnostic per-mode dry/wet radius, density and hygroscopicity from
the prognostic mass + number tracers, with **zero** tendency. It is
mass-conserving and differentiable, and exposes the real MAM4 4-mode
population shape, so it exercises the whole harness (emissions, activation,
deposition, sedimentation) end-to-end while a real microphysics core is
wired in later. No coagulation/condensation/nucleation/ageing happens here —
those belong to a real core (MAM4-JAX, #490; or the in-tree
:mod:`~jcm.physics.aerosol.jam.microphysics.modal`).

The κ-Köhler diagnostics themselves now live in
:mod:`~jcm.physics.aerosol.jam.microphysics.koehler`, because every core
needs them and none owns them. They are re-exported here unchanged.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.koehler import (
    equilibrium_modal_state,
    saturation_ratio,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics_interface import PhysicsTendency

#: Re-exported so the existing importers of these names keep working.
__all__ = [
    "PlaceholderMicrophysics",
    "equilibrium_modal_state",
    "saturation_ratio",
]

class PlaceholderMicrophysics(ModalMicrophysicsTerm):
    """Zero-tendency κ-Köhler equilibrium core on the MAM4 4-mode population."""

    name: ClassVar[str] = "jam_placeholder_microphysics"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    spec: ClassVar[ModalAerosolSpec] = MAM4_SPEC

    def __init__(self, spec: ModalAerosolSpec | None = None):
        """Optionally override the population (defaults to MAM4 4-mode)."""
        # Allow injecting an alternative population (e.g. a reduced test
        # spec); defaults to the full MAM4 4-mode layout.
        if spec is not None:
            self.spec = spec
        # No tunable parameters; keep an nnx-friendly marker attribute.
        self._initialized = True

    def __call__(self, state, diagnostics, forcing, terrain):
        from jcm.physics.aerosol.jam.removal_split import split_view

        spec = self.spec
        zeros = jnp.zeros_like(state.temperature)
        view = split_view(spec, state, diagnostics)
        masses = {
            mass_name(sp, mode.short): view.get(
                mass_name(sp, mode.short), zeros
            )
            for mode in spec.modes
            for sp in mode.species
        }
        numbers = {
            number_name(mode.short): view.get(
                number_name(mode.short), zeros
            )
            for mode in spec.modes
        }

        sat = saturation_ratio(
            state.temperature, state.specific_humidity,
            diagnostics["pressure_full"],
        )
        jam_state = equilibrium_modal_state(masses, numbers, spec, sat)

        # Zero tendency — equilibrium diagnostics only.
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "_jam_state": jam_state}
