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
from typing import Any, Callable, ClassVar

import jax.numpy as jnp
from flax import nnx

import jcm.constants as c

from jcm.physics.aerosol.jam.gas_species import SULFUR_GASES
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.microphysics import condensation
from jcm.physics.aerosol.jam.microphysics import ageing
from jcm.physics.aerosol.jam.microphysics import ledger
from jcm.physics.aerosol.jam.microphysics import coagulation as coag
from jcm.physics.aerosol.jam.microphysics import moments
from jcm.physics.aerosol.jam.microphysics import new_particle_formation as npf
from jcm.physics.aerosol.jam.microphysics import rename as renaming
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.koehler import (
    equilibrium_modal_state,
    saturation_ratio,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name,
    mass_name,
    number_name,
)
from jcm.physics_interface import PhysicsTendency


def _store():
    """Import the cloud-borne store lazily.

    ``cloud_borne_store`` imports ``microphysics.mam4_data``, which runs
    this package's ``__init__``, which imports this module: a module-level
    import of the store here closes that cycle and fails at package import
    with a partially initialised ``cloud_borne_store``.
    """
    from jcm.physics.aerosol.jam import cloud_borne_store
    return cloud_borne_store


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
    air_density: jnp.ndarray   # dry air [kg/m^3], turns kg^-1 into m^-3
    # κ-Köhler state of the population at the START of the step. The
    # uptake coefficients use the wet geometric-mean diameter, and MAM
    # freezes them at their start-of-step values (numerics_SO4), so this is
    # computed once before the processes run rather than inside them.
    state_start: JamAerosolState
    # Step-mean H2SO4 chemical production [kg/kg/s], published by the
    # chemistry term; zero when no chemistry is composed.
    h2so4_production: jnp.ndarray
    # Height of each level above the surface [m], ``(nlev, ncols)``, and
    # the boundary-layer height [m], ``(ncols,)``. The latter is published
    # by TTE-TKE vertical diffusion, which runs AFTER this core in the
    # ECHAM order (term 21 against 12), so it is one step lagged and
    # absent on the first step; the D-A2 sub-decision records why the lag
    # is accepted. Absent, it is zero and the empirical new-particle
    # pathway falls back to MAM's 100 m minimum.
    height_above_surface: jnp.ndarray
    pbl_height: jnp.ndarray
    dt: float                  # [s]
    # Dry volume [m³/kg] of each mode (keyed by ``mode.short``) at the
    # START of the step. Renaming measures continuous growth against it;
    # ``None`` only in tests that exercise other processes alone.
    dry_volume_start: dict[str, jnp.ndarray] | None = None


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
    #: Gas-phase precursors ``g_<species>`` [kg/kg]; condensation and
    #: new-particle formation consume them.
    gases: dict[str, jnp.ndarray] = dataclasses.field(default_factory=dict)
    #: Cloud-borne mirrors ``mc_<sp>_<mode>`` / ``nc_<mode>`` read from the
    #: carry store (D-A5); empty for a population without the explicit
    #: phase. Only renaming writes them — the other processes act on the
    #: interstitial phase, as in MAM — and the driver books the change to
    #: the store, not to the tracer tendency.
    cloud_borne: dict[str, jnp.ndarray] = dataclasses.field(default_factory=dict)
    #: Soluble mass [kg/kg] deposited on the primary-carbon mode this step,
    #: by species. Written by condensation and coagulation, read by ageing
    #: (D-A1): MAM computes the aged fraction inside those two processes,
    #: and keeping ageing separate means the deposit has to cross the seam.
    coating: dict[str, jnp.ndarray] = dataclasses.field(default_factory=dict)
    #: Per-process scalars worth publishing (the D-A1 depletion trigger,
    #: later the AER-09 ledger). Keyed by name, ``(nlev, ncols)`` each.
    diagnostics: dict[str, jnp.ndarray] = dataclasses.field(default_factory=dict)


#: A microphysical process: pure, population-in/population-out.
#:
#: Every process has this one signature so the driver below is a plain fold
#: and the order is data rather than control flow. ``spec`` is static
#: configuration (mode geometry, species properties); ``env`` is the
#: per-step ambient state; ``params`` is the process's own differentiable
#: parameter struct, or ``None`` for a process that has none.
Process = Callable[
    [ModalPopulation, MicrophysicsEnvironment, ModalAerosolSpec, Any],
    ModalPopulation,
]


def gas_aerosol_exchange(population, env, spec, params):
    """Condensation of H2SO4 and SOAG onto existing modes. **AER-03.**

    Seinfeld & Pandis (1998) mass-transfer expressions integrated over each
    mode's size distribution (Binkowski and Shankar, 1995), with an
    accommodation coefficient of 0.65 for H2SO4 (Poschl et al., 1998).
    H2SO4 condensation is irreversible; SOA is reversible.

    Competes with nucleation for the same ``g_h2so4``, which is why it runs
    first and why the two share a budget. It also records the soluble
    mass it deposits on the primary-carbon mode: MAM computes ageing inside
    this step, and this scaffold keeps ageing separate, so the bookkeeping
    crosses the seam in ``population.coating`` (D-A1).

    Implemented in :mod:`~jcm.physics.aerosol.jam.microphysics.condensation`.
    """
    return condensation.condense(population, env, spec, params)


def rename(population, env, spec, params):
    """Transfer across a mode boundary once a mode grows past it. **AER-07.**

    Aitken to accumulation, following Easter et al. (2004): after the
    continuous-growth processes, the particles on the upper tail of the
    Aitken distribution are accumulation-mode by definition, and their
    number and mass transfer across. Without it, condensation inflates
    Aitken sizes without bound. The supplement to Liu et al. (2012) is
    explicit that this runs *after* continuous growth, which is what fixes
    its position behind ``gas_aerosol_exchange`` here; the growth is
    measured against ``env.dry_volume_start``. The fractions come from the
    combined interstitial + cloud-borne volume and move both phases, as in
    MAM (D-A5). No tunables; ``params`` is ``None``.

    Implemented in :mod:`~jcm.physics.aerosol.jam.microphysics.rename`.
    """
    return renaming.rename_aitken(population, env, spec, params)


def nucleation(population, env, spec, params):
    """New-particle formation from H2SO4 vapour. **AER-04.**

    The function keeps the name the MAM literature uses for the process;
    in prose it is new-particle formation (NPF). Implemented in
    :mod:`~jcm.physics.aerosol.jam.microphysics.new_particle_formation`.

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
    return npf.form_new_particles(population, env, spec, params)


def coagulation(population, env, spec, params):
    """Brownian coagulation, intra- and inter-modal. **AER-05.**

    Intra-modal coagulation removes number at constant mass; inter-modal
    moves both mass and number to the larger mode. This is what bounds the
    number that new-particle formation creates.

    Rates follow the CMAQ v4.6 fast algorithms (Binkowski and Roselle,
    2003) as MAM carries them, over the Aitken, accumulation and
    primary-carbon modes ONLY: MAM neglects coagulation involving the
    coarse and larger modes as much slower. The Aitken-to-primary-carbon
    path also ages carbon, so like condensation this step records the
    soluble mass it deposits on ``pcm`` in ``population.coating`` for
    AER-06. No tunables; ``params`` is ``None``.

    Implemented in :mod:`~jcm.physics.aerosol.jam.microphysics.coagulation`.
    """
    return coag.coagulate(population, env, spec, params)


def carbonaceous_ageing(population, env, spec, params):
    """Primary carbon to accumulation once coated. **AER-06.**

    ``pcm`` is ``can_activate=False``: structurally excluded from
    activation, in-cloud scavenging and the cloud-borne phase. Right for
    fresh hydrophobic carbon, wrong once a soluble coating has grown on it.
    This transfer is the only pathway that ends the exclusion, so its
    absence is what sets the ~21-day BC lifetime.

    The criterion is MAM4's: the shell of sulfate plus hygroscopicity-
    equivalent SOA deposited this step, as a thickness over the mode's
    surface, against ``n_so4_monolayers`` (8 in MAM4's sources; 3 was
    MAM7's) of sulfate; that fraction of the mode's POA, BC, marine
    organics and number moves to accumulation. It runs last and consumes
    ``population.coating``, which condensation and coagulation wrote
    (D-A1).

    Implemented in :mod:`~jcm.physics.aerosol.jam.microphysics.ageing`.
    """
    return ageing.age(population, env, spec, params)


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


#: The mode ROLES the five processes read off the population — never
#: MAM4's mode names. A population with other names runs by pointing these
#: at its own modes (``population.ModalAerosolSpec``); one that cannot
#: supply a role is refused at construction, not with a ``KeyError``
#: mid-step.
_ROLES: tuple[str, ...] = ("accumulation_mode", "aitken_mode", "primary_carbon_mode")


def _check_roles(spec: ModalAerosolSpec) -> None:
    """Refuse a population that cannot play every role the core needs."""
    missing = []
    for role in _ROLES:
        name = getattr(spec, role)
        try:
            if name is None:
                raise KeyError(name)
            spec.mode(name)
        except KeyError:
            missing.append(f"{role}={name!r}")
    if missing:
        raise ValueError(
            "JamModalMicrophysics reads MAM4's mode roles off the population "
            f"and cannot resolve {', '.join(missing)} among the modes "
            f"{list(spec.mode_names)}. Point the role fields of "
            "ModalAerosolSpec at this population's own modes; the core "
            "implements MAM4's process chain (Aitken-to-accumulation "
            "renaming, primary-carbon ageing), so a population without "
            "those classes needs a different core.")


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
        condensation_params: condensation.CondensationParameters | None = None,
        npf_params: npf.NewParticleFormationParameters | None = None,
        ageing_params: ageing.AgeingParameters | None = None,
        audit: bool = False,
    ):
        """Build the core.

        Args:
            spec: Population override; defaults to the MAM4 4-mode layout.
            processes: Process sequence override. Present so a test can run
                one process in isolation, and so AER-09's conservation gates
                can bisect a budget failure to a single process rather than
                to the whole step.
            condensation_params: Gas-aerosol exchange tunables; MAM4's
                defaults when omitted.
            npf_params: New-particle formation tunables; MAM4's defaults
                when omitted.
            ageing_params: Primary-carbon ageing tunables; MAM4's defaults
                when omitted.
            audit: publish the conservation ledger per process (AER-09):
                ``_modal_ledger_<process>_<budget>`` relative changes for
                every budget in :mod:`.ledger`. Static, off by default —
                the per-step sulfur, organic and number changes are always
                published as ``_modal_ledger_step_<budget>``; this adds the
                per-process bisection a failing gate needs.

        """
        if spec is not None:
            self.spec = spec
        _check_roles(self.spec)
        # Renaming moves the cloud-borne Aitken particles too (D-A5), read
        # from and written to the carry store the way ``AqueousSulfur`` does.
        if self.spec.cloud_borne:
            self.requires = (*type(self).requires, _store().CARRY_KEY)
        self.processes = PROCESS_ORDER if processes is None else tuple(processes)
        self.audit = bool(audit)
        # Held in an nnx.Param so every numeric tunable is a gradient-visible
        # leaf (CLAUDE.md: physics parameters are differentiable like all
        # the others); ``max_substeps`` is static aux inside the struct.
        self.condensation = nnx.Param(
            condensation.CondensationParameters()
            if condensation_params is None else condensation_params)
        self.new_particle_formation = nnx.Param(
            npf.NewParticleFormationParameters()
            if npf_params is None else npf_params)
        self.ageing = nnx.Param(
            ageing.AgeingParameters() if ageing_params is None else ageing_params)
        # nnx-friendly marker, mirroring PlaceholderMicrophysics.
        self._initialized = True

    def _process_params(self) -> dict[str, Any]:
        """Per-process parameter structs, keyed by process name."""
        return {
            "gas_aerosol_exchange": self.condensation.get_value(),
            "nucleation": self.new_particle_formation.get_value(),
            "carbonaceous_ageing": self.ageing.get_value(),
        }

    def _gather(self, state, diagnostics) -> ModalPopulation:
        """Read the population out of ``state.tracers`` and the carry store.

        Absent tracers fall back to zeros: ``state.tracers`` is empty during
        ``Model.get_empty_data``'s structural probe, and the key set has to
        come from the static ``spec`` either way so the output structure is
        the same in the probe and in a real step.
        """
        zeros = jnp.zeros_like(state.temperature)

        # Negative tracers are real input — spectral transport rings
        # around sharp sources — and every JAM consumer treats them as
        # empty rather than pumping them (the storage A/B measured pumping
        # as net-negative). So does this core: it reads the population
        # floored at zero and books ``after − floored_before``, which leaves
        # a negative excursion exactly where transport put it. Unfloored, a
        # negative sulfate mass gives the κ-Köhler state a negative wet
        # density and the coagulation kernel ``sqrt(3kT/ρ)`` a NaN that the
        # forward ``where`` masks and reverse mode does not (AER-11).
        def read(key):
            return jnp.maximum(state.tracers.get(key, zeros), 0.0)

        masses = {
            mass_name(sp, mode.short): read(mass_name(sp, mode.short))
            for mode in self.spec.modes
            for sp in mode.species
        }
        numbers = {
            number_name(mode.short): read(number_name(mode.short))
            for mode in self.spec.modes
        }
        # The gas list is static too (``SULFUR_GASES``), for the same
        # reason: a tendency key set that depended on which gases happen
        # to be present would differ between the structural probe and a
        # real step. Keys the host does not carry are dropped at the merge
        # (``composable_physics`` folds a term's tracer tendencies into
        # the state's own key set), so emitting them is harmless.
        gases = {gas_name(gas): read(gas_name(gas)) for gas in SULFUR_GASES}
        # Cloud-borne mirrors from the carry store (D-A5); the static key
        # set again, zeros before the store term has run (the probe).
        store = _store()
        view = store.tracer_view(self.spec, state, diagnostics)
        cloud_borne = {
            name: jnp.maximum(view.get(name, zeros), 0.0)
            for name in (store.mirror_names(self.spec) if self.spec.cloud_borne else ())
        }
        return ModalPopulation(masses=masses, numbers=numbers, gases=gases,
                               cloud_borne=cloud_borne)

    @staticmethod
    def _height_above_surface(state, diagnostics):
        """Level height above the surface [m], from the moist-air term.

        ``MoistAirColumnState`` runs first in the ECHAM order and publishes
        ``height_full``/``height_half`` as geopotential height, so the
        surface is the bottom interface (physics frame: top first, index
        -1). Falls back to the state's geopotential when the moist-air
        term is not composed, as in the unit tests.
        """
        if "height_full" in diagnostics and "height_half" in diagnostics:
            return diagnostics["height_full"] - diagnostics["height_half"][-1]
        height = state.geopotential / c.grav
        return height - height[-1]

    @staticmethod
    def _pbl_height(state, diagnostics):
        """Lagged boundary-layer height [m] ``(ncols,)``, zero if absent.

        A Python-level branch on the key, like ``tracer_diffusion.py``:
        presence is a property of the composition, not of the data, so it
        is static per trace.
        """
        vdiff = diagnostics.get("vertical_diffusion")
        if vdiff is None:
            return jnp.zeros(state.temperature.shape[1:],
                             dtype=state.temperature.dtype)
        return vdiff.pbl_height

    def __call__(self, state, diagnostics, forcing, terrain):
        before = self._gather(state, diagnostics)
        pressure = diagnostics["pressure_full"]
        # Injected by ComposablePhysics; the fallback mirrors the other JAM
        # terms (cloud_borne.py, wetdep_term.py, drydep_term.py).
        dt = diagnostics.get("_dt_seconds", 1800.0)
        zeros = jnp.zeros_like(state.temperature)

        saturation = saturation_ratio(
            state.temperature, state.specific_humidity, pressure,
        )
        state_start = equilibrium_modal_state(
            before.masses, before.numbers, self.spec, saturation,
        )
        env = MicrophysicsEnvironment(
            temperature=state.temperature,
            pressure=pressure,
            saturation=saturation,
            # Dry-air density; the moisture correction is well below the
            # uncertainty of any uptake coefficient it multiplies.
            air_density=pressure / (c.rd * state.temperature),
            state_start=state_start,
            h2so4_production=diagnostics.get("_h2so4_production", zeros),
            height_above_surface=self._height_above_surface(state, diagnostics),
            pbl_height=self._pbl_height(state, diagnostics),
            dt=dt,
            dry_volume_start={
                mode.short: moments.mode_volume(before.masses, mode, self.spec)
                for mode in self.spec.modes},
        )

        params = self._process_params()
        after = before
        ledger_diagnostics = {}
        budgets_before = ledger.budgets(before, self.spec)
        running = budgets_before
        for name, process in self.processes:
            after = process(after, env, self.spec, params.get(name))
            if self.audit:
                budgets_after = ledger.budgets(after, self.spec)
                for budget, value in ledger.relative_change(
                        running, budgets_after).items():
                    ledger_diagnostics[f"ledger_{name}_{budget}"] = value
                running = budgets_after
        # The step-level closure is cheap and always on: a health gate can
        # read it without paying for the per-process ledger.
        budgets_after = ledger.budgets(after, self.spec)
        for budget, value in ledger.relative_change(
                budgets_before, budgets_after).items():
            if budget in ("sulfur", "organic", "number"):
                ledger_diagnostics[f"ledger_step_{budget}"] = value

        jam_state = equilibrium_modal_state(
            after.masses, after.numbers, self.spec, saturation,
        )

        # The population change over the step becomes the tracer tendency,
        # ``(after - before) / dt`` -- the convention every other JAM term
        # and the mam4-jax adapter use. The key set is derived from ``spec``
        # and ``SULFUR_GASES``, so it is the SAME in the structural probe as
        # in a real step; a dict that was empty in the probe and full at run
        # time would change the output tree between the two.
        tracers = {}
        for group in ("masses", "numbers", "gases"):
            old, new = getattr(before, group), getattr(after, group)
            for key in old:
                tracers[key] = (new[key] - old[key]) / dt
        tendency = PhysicsTendency.zeros(state.temperature.shape,
                                         tracers=tracers)
        # The cloud-borne change goes to the carry store as a rate, the
        # way every other cloud-borne writer books it (D-A5) — only when
        # the store is there: ``apply_updates`` would otherwise invent one,
        # and in the structural probe (no store term has run) the gathered
        # mirrors are zeros and there is nothing to book anyway.
        if before.cloud_borne and _store().CARRY_KEY in diagnostics:
            diagnostics, _ = _store().apply_updates(
                self.spec, diagnostics,
                {key: (after.cloud_borne[key] - before.cloud_borne[key]) / dt
                 for key in before.cloud_borne},
                dt)
        published = {
            f"_modal_{key}": value
            for key, value in {**after.diagnostics, **ledger_diagnostics}.items()
        }
        return tendency, {**diagnostics, **published, "_jam_state": jam_state}
