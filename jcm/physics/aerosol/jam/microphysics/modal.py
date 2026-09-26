"""In-tree modal aerosol microphysics core for the JAM harness.

The third :class:`ModalMicrophysicsTerm` core, alongside
:class:`~...microphysics.placeholder.PlaceholderMicrophysics` (κ-Köhler
diagnostics, zero tendency) and
:class:`~...microphysics.mam4_jax.Mam4JaxMicrophysics` (the real processes,
but behind the **GPL-3.0** ``mam4-jax`` optional dependency).

Why a third one: with only those two, an Apache-licensed jcm has a
population that is emitted, transported, scavenged, deposited and optically
active but which never *evolves*. Modes neither grow nor exchange mass, and
``jam_carbon_aging.md`` prices one of the missing processes on its own —
with no ageing, primary carbon can never leave the hydrophobic ``pcm`` mode,
giving a black-carbon lifetime of ~21 days against an observed 5-8.

**Licensing.** This core is written from the published literature and from
prose descriptions of the E3SM Fortran, never by translating ``mam4-jax``.
It must not import that package; a numerical A/B against it (AER-12) is a
test-only comparison run in a throwaway environment.

Status: AER-01 — the scaffold. Every process below is an identity, so this
core is **behaviourally identical to the placeholder**: it publishes the
same ``_jam_state`` and returns the same zero tendency. AER-03 onward fill
the processes in one at a time; the plan is
``docs/source/design/jam_modal_microphysics.md``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, ClassVar

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


@dataclasses.dataclass(frozen=True)
class MicrophysicsEnvironment:
    """Per-step ambient fields the processes share.

    Held in one object so adding a field a later process needs (air density
    for the coagulation kernel, for instance) touches one signature rather
    than five. Fields are ``(nlev, ncols)`` arrays except ``dt``.
    """

    temperature: jnp.ndarray   # [K]
    pressure: jnp.ndarray      # [Pa]
    saturation: jnp.ndarray    # water activity a_w, clipped below 1 [-]
    dt: float                  # [s]


@dataclasses.dataclass(frozen=True)
class ModalPopulation:
    """The prognostic population a process reads and returns.

    Flat tracer-keyed dicts, matching ``tracer_layout``: ``m_<sp>_<mode>``
    dry mass [kg/kg] and ``n_<mode>`` number [kg^-1]. A plain dataclass
    rather than a ``tree_math.struct`` because it never crosses a jit
    boundary on its own — it is built, threaded through the processes and
    consumed inside a single traced ``__call__``.
    """

    masses: dict[str, jnp.ndarray]
    numbers: dict[str, jnp.ndarray]


#: A microphysical process: pure, population-in/population-out.
#:
#: Every process has this one signature so the driver below is a plain fold
#: and the order is data rather than control flow. ``spec`` is static
#: configuration (mode geometry, species properties); ``env`` is the
#: per-step ambient state.
Process = Callable[
    [ModalPopulation, MicrophysicsEnvironment, ModalAerosolSpec],
    ModalPopulation,
]


def gas_aerosol_exchange(population, env, spec):
    """Condensation of H2SO4 and SOAG onto existing modes. **AER-03.**

    Seinfeld & Pandis (1998) mass-transfer expressions integrated over each
    mode's size distribution (Binkowski and Shankar, 1995), with an
    accommodation coefficient of 0.65 for H2SO4 (Poschl et al., 1998).
    H2SO4 condensation is irreversible; SOA is reversible.

    Competes with nucleation for the same ``g_h2so4``, which is why it runs
    first and why the two share a budget. It must also record the soluble
    mass it deposits on the primary-carbon mode: MAM computes ageing inside
    this step, and this scaffold keeps ageing separate, so the bookkeeping
    has to cross the seam instead (D-A1).
    """
    return population


def rename(population, env, spec):
    """Transfer across a mode boundary once a mode grows past it. **AER-07.**

    Aitken to accumulation, following Easter et al. (2004): after the
    continuous-growth processes, the particles on the upper tail of the
    Aitken distribution are accumulation-mode by definition, and their
    number and mass transfer across. Without it, condensation inflates
    Aitken sizes without bound. The supplement to Liu et al. (2012) is
    explicit that this runs *after* continuous growth, which is what fixes
    its position behind ``gas_aerosol_exchange`` here.
    """
    return population


def nucleation(population, env, spec):
    """New-particle formation from H2SO4 vapour. **AER-04.**

    Two pathways, because MAM's default is not binary alone (D-A2):

    * binary H2SO4-H2O homogeneous nucleation (Vehkamaki et al., 2002),
      the free-troposphere term - MAM4 predicts no NH3, so the ternary
      parameterization used by the 7-mode version is unavailable;
    * an empirical first-order-in-H2SO4 rate within the planetary boundary
      layer (Sihto et al., 2006) with a coefficient of 1.0e-6 s^-1. This
      is what addresses the continental Aitken-number deficit that
      binary-only nucleation leaves.

    Kerminen and Kulmala (2002) then carries the critical cluster up to the
    Aitken mode's lower bound, where only a fraction survives coagulation
    on the way. New particles enter the Aitken mode. This is the sole
    source of new particle *number* in the population.
    """
    return population


def coagulation(population, env, spec):
    """Brownian coagulation, intra- and inter-modal. **AER-05.**

    Intra-modal coagulation removes number at constant mass; inter-modal
    moves both mass and number to the larger mode. This is what bounds the
    number that nucleation creates.

    Rates follow the CMAQ v4.6 fast/approximate algorithms (Binkowski and
    Roselle, 2003), over the Aitken, accumulation and primary-carbon modes
    ONLY: MAM neglects coagulation involving the coarse and larger modes
    as much slower. The Aitken-to-primary-carbon path also ages carbon, so
    like condensation this step owes AER-06 a record of what it deposited.
    """
    return population


def carbonaceous_ageing(population, env, spec):
    """Primary carbon to accumulation once coated. **AER-06.**

    ``pcm`` is ``can_activate=False``: structurally excluded from
    activation, in-cloud scavenging and the cloud-borne phase. Right for
    fresh hydrophobic carbon, wrong once a soluble coating has grown on it.
    This transfer is the only pathway that ends the exclusion, so its
    absence is what sets the ~21-day BC lifetime.

    The criterion is three monolayers of sulfate with SOA included, and the
    aged fraction is ``M_SO4,cond / M_SO4,age-all + M_SOA,cond /
    M_SOA,age-all`` (Liu et al., 2012, supplement S1.1.5); that fraction of
    the mode's POA, BC and number moves to accumulation. In MAM this is
    computed inside condensation and coagulation rather than as its own
    step - see D-A1 for why it is separate here and what that costs.
    """
    return population


#: Process order, one entry per step of MAM4's ``amicphys``: gas-aerosol
#: exchange, rename, nucleation, coagulation, ageing.
#:
#: The order is deliberate and load-bearing, not incidental. Condensation
#: runs before nucleation because both consume ``g_h2so4`` and condensation
#: onto existing surface is the larger sink whenever the population is not
#: nearly clean; running nucleation first would systematically over-produce
#: new particles. Ageing runs last because it consumes the coating the two
#: earlier mass processes deposited.
#:
#: **D-A1 DECIDED 2026-09-26**: adopt this order, one pass per physics
#: step, no sub-cycling yet. Two caveats worth knowing before changing it:
#: the published MAM description does NOT state the amicphys sequence (it
#: comes from jcm's own ``Mam4JaxMicrophysics`` docstring), and MAM folds
#: ageing into condensation and coagulation rather than running it last.
#: Sub-cycling gets revisited on a measurement, not a hunch - if AER-04
#: shows one step depleting ``g_h2so4`` by more than ~50 %.
PROCESS_ORDER: tuple[tuple[str, Process], ...] = (
    ("gas_aerosol_exchange", gas_aerosol_exchange),
    ("rename", rename),
    ("nucleation", nucleation),
    ("coagulation", coagulation),
    ("carbonaceous_ageing", carbonaceous_ageing),
)


class JamModalMicrophysics(ModalMicrophysicsTerm):
    """Modal aerosol microphysics on the MAM4 4-mode population, in pure jcm."""

    name: ClassVar[str] = "jam_modal_microphysics"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    spec: ClassVar[ModalAerosolSpec] = MAM4_SPEC

    def __init__(
        self,
        spec: ModalAerosolSpec | None = None,
        *,
        processes: tuple[tuple[str, Process], ...] | None = None,
    ):
        """Build the core.

        Args:
            spec: Population override; defaults to the MAM4 4-mode layout.
            processes: Process sequence override. Present so a test can run
                one process in isolation, and so AER-09's conservation gates
                can bisect a budget failure to a single process rather than
                to the whole step.

        """
        if spec is not None:
            self.spec = spec
        self.processes = PROCESS_ORDER if processes is None else tuple(processes)
        # nnx-friendly marker, mirroring PlaceholderMicrophysics.
        self._initialized = True

    def _gather(self, state) -> ModalPopulation:
        """Read the population out of ``state.tracers``.

        Absent tracers fall back to zeros: ``state.tracers`` is empty during
        ``Model.get_empty_data``'s structural probe, and the key set has to
        come from the static ``spec`` either way so the output structure is
        the same in the probe and in a real step.
        """
        zeros = jnp.zeros_like(state.temperature)
        masses = {
            mass_name(sp, mode.short): state.tracers.get(
                mass_name(sp, mode.short), zeros
            )
            for mode in self.spec.modes
            for sp in mode.species
        }
        numbers = {
            number_name(mode.short): state.tracers.get(
                number_name(mode.short), zeros
            )
            for mode in self.spec.modes
        }
        return ModalPopulation(masses=masses, numbers=numbers)

    def __call__(self, state, diagnostics, forcing, terrain):
        population = self._gather(state)

        saturation = saturation_ratio(
            state.temperature,
            state.specific_humidity,
            diagnostics["pressure_full"],
        )
        env = MicrophysicsEnvironment(
            temperature=state.temperature,
            pressure=diagnostics["pressure_full"],
            saturation=saturation,
            # Injected by ComposablePhysics; the fallback mirrors the other
            # JAM terms (cloud_borne.py, wetdep_term.py, drydep_term.py).
            dt=diagnostics.get("_dt_seconds", 1800.0),
        )

        for _, process in self.processes:
            population = process(population, env, self.spec)

        jam_state = equilibrium_modal_state(
            population.masses, population.numbers, self.spec, saturation,
        )

        # Zero tendency while every process is an identity (AER-01), which
        # is what makes this core bitwise-equal to the placeholder.
        #
        # AER-03 turns the net population change into tracer tendencies,
        # ``(after - before) / dt``. When it does, the emitted key set must
        # be derived from ``spec`` and be the SAME in the structural probe
        # as in a real step — a dict that is empty in the probe and full at
        # run time would change the output tree between the two.
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "_jam_state": jam_state}
