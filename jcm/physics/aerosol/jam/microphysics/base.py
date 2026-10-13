"""Microphysics-core contract for the JAM aerosol harness.

The microphysics term is the single swap point between the harness and an
interchangeable aerosol core. It publishes a static ``ModalAerosolSpec``
(the population shape), declares the prognostic tracers that population
needs, and each step writes the ``_jam_state`` diagnostic the downstream
harness terms read.

Concrete cores:
  * :class:`PlaceholderMicrophysics` — κ-Köhler equilibrium radii, zero
    tendency. Exercises the harness end-to-end.
  * (later) a wrapper around MAM4-JAX's ``amicphys`` — issue #490.
"""

from __future__ import annotations

from typing import ClassVar

from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import tracer_specs
from jcm.physics.physics_term import PhysicsTerm, TracerSpec


class ModalMicrophysicsTerm(PhysicsTerm):
    """Base class for an interchangeable modal aerosol microphysics core.

    Subclasses set :attr:`spec` (a :class:`ModalAerosolSpec`) and implement
    :meth:`__call__` to write the ``_jam_state`` diagnostic. The tracer set
    is derived from the population, so the harness seeds exactly the tracers
    the active core needs.
    """

    category: ClassVar[str] = "aerosol_microphysics"
    provides: ClassVar[tuple[str, ...]] = ("_jam_state",)

    #: Set by subclasses — the population this core advances.
    spec: ClassVar[ModalAerosolSpec]

    def required_tracers(self) -> tuple[TracerSpec, ...]:  # type: ignore[override]
        """Interstitial + cloud-borne tracers for this population."""
        return tracer_specs(self.spec)
