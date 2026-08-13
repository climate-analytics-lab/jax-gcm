"""Microphysics-neutral aerosol population description.

These are pure-Python frozen dataclasses that describe the *shape* of an
aerosol population — how many aerosol classes, which species each carries,
and the per-class/per-species physical metadata a process harness needs.
``ModalAerosolSpec`` is the modal realisation (each class is a log-normal
mode); a sectional realisation (each class a size bin) would be a sibling
spec with the same role (#491). They are the swap point between the
microphysics-agnostic harness (emissions, dry/wet deposition, sedimentation,
ARG activation) and an interchangeable microphysics core (placeholder today;
MAM4-JAX later).

Design notes
------------
* They are *static* config, never JAX pytree leaves — the harness caches a
  ``ModalAerosolSpec`` at compose time so the class count, σ, density and κ
  are known to Python before tracing. No data-dependent shapes result.
* Field names deliberately echo the AMBRS modal structs
  (``AerosolModeState`` / ``AerosolSpecies``: number, geometric mean
  diameter, log-normal width, mass fractions; species molar_mass / density /
  hygroscopicity) so a future part2pop/AMBRS interop adapter is a thin
  mapping rather than a rewrite. See issue #461.
* ``family`` admits "modal" | "sectional" | "bulk"; only "modal" is built
  out today (sectional/bulk tracked in #491).
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class AerosolSpecies:
    """Physical properties of one aerosol species.

    Attributes:
        name: short canonical token used in tracer keys (e.g. ``"so4"``).
        molar_mass: molar mass [kg/mol] (informational for the placeholder;
            used by gas–aerosol exchange in the real core).
        density: particle-material density [kg/m³].
        hygroscopicity: κ (kappa-Köhler hygroscopicity parameter) [-].
        long_name: descriptive name (e.g. ``"sulfate"``), for readability /
            output labelling (the short ``name`` is what tracer keys and the
            microphysics-core adapters key off).
        refractive_index: complex refractive index, or ``None`` (optics
            are out of scope here — see #495).

    """

    name: str
    molar_mass: float
    density: float
    hygroscopicity: float
    long_name: str = ""
    refractive_index: complex | None = None


@dataclasses.dataclass(frozen=True)
class AerosolMode:
    """One log-normal mode of a modal aerosol population.

    Attributes:
        name: mode name (e.g. ``"accum"``).
        short: short token used in tracer keys (e.g. ``"acc"``).
        geom_std_dev: geometric standard deviation σ_g of the log-normal
            size distribution [-] (AMBRS: ``10**log10_geom_std_dev``).
        dgnum: reference geometric-mean dry *diameter* [m].
        dgnum_lo: lower bound on dry diameter [m].
        dgnum_hi: upper bound on dry diameter [m].
        species: tuple of species ``name`` tokens this mode may carry.
        soluble: whether the mode is treated as soluble (CCN-relevant).
        can_activate: whether ARG activation may draw droplets from it.
        sediments: whether gravitational settling acts on it.

    """

    name: str
    short: str
    geom_std_dev: float
    dgnum: float
    dgnum_lo: float
    dgnum_hi: float
    species: tuple[str, ...]
    soluble: bool
    can_activate: bool
    sediments: bool

    # ------------------------------------------------------------------
    # Geometry: per-class conversions from mass to number / surface area.
    #
    # These are the family-agnostic interface a process harness uses to turn
    # a per-class dry-aerosol *volume* (mass / material-density) into a
    # number- or area-concentration without knowing whether the class is a
    # log-normal mode or a size bin. The modal realisation below uses the
    # log-normal moments; a sectional class (#491) exposes the same two
    # properties from its bin geometry, so consumers (e.g. the ice-nucleation
    # IN populations) stay invariant to the aerosol family.
    # ------------------------------------------------------------------

    @property
    def number_factor(self) -> float:
        """Particle number per unit dry-aerosol volume [1/m³].

        ``number = (mass / ρ_material) · number_factor``. Modal: the
        reciprocal mean single-particle volume of the log-normal,
        ``v_p = (π/6)·dgnum³·exp(4.5·ln²σ_g)``.
        """
        ln_sigma = math.log(self.geom_std_dev)
        v_p = (math.pi / 6.0) * self.dgnum ** 3 * math.exp(4.5 * ln_sigma ** 2)
        return 1.0 / v_p

    @property
    def area_factor(self) -> float:
        """Particle surface area per unit dry-aerosol volume [1/m].

        ``area = (mass / ρ_material) · area_factor``. Modal: the surface-area
        moment of the log-normal, ``6/(dgnum·exp(2.5·ln²σ_g))``.
        """
        ln_sigma = math.log(self.geom_std_dev)
        return 6.0 / (self.dgnum * math.exp(2.5 * ln_sigma ** 2))


@dataclasses.dataclass(frozen=True)
class ModalAerosolSpec:
    """A full modal aerosol population description.

    This is the contract a microphysics core publishes and the harness
    consumes. It is intentionally a plain data object: helpers are pure
    Python and run at compose time only.
    """

    modes: tuple[AerosolMode, ...]
    species: tuple[AerosolSpecies, ...]
    family: str = "modal"
    #: Whether the population prognoses an explicit cloud-borne phase per
    #: class (MAM-style ``mc_*``/``nc_*`` fields alongside the interstitial
    #: tracers). This is a property of the population *representation*, not
    #: of the process harness: MAM keeps in-droplet aerosol as separate
    #: constituents, while M7 (ECHAM-HAM) and sectional schemes like TOMAS
    #: represent it implicitly, scavenging the interstitial tracers by
    #: their activated fraction. The explicit phase lives in the cross-step
    #: physics carry (CAM's ``qqcw``-in-pbuf pattern), never in dycore
    #: tracers — the measured #602 decision, see ``cloud_borne_store``.
    #: ``False`` falls back to the implicit treatment; both settings are
    #: complete, comparable physics.
    cloud_borne: bool = True
    #: Population policy for where freshly-emitted **primary** mass of a species
    #: goes: ``{species: ((mode_short, mass_fraction), ...)}`` with fractions
    #: summing to 1. This centralises the modal (or sectional) assumption with
    #: the population — HAMMOZ keeps the same table in its aerosol module — so
    #: emission terms ask :meth:`primary_split` instead of hardcoding modes. A
    #: sectional spec supplies its own bin split here. ``ss``/``du`` size-mapped
    #: source schemes (Gong/Tegen) instead use :meth:`classes_for` + the class
    #: size ranges, so they need no entry.
    primary_emission: dict[str, tuple[tuple[str, float], ...]] = (
        dataclasses.field(default_factory=dict)
    )

    def __post_init__(self) -> None:
        """Validate the family tag and species references."""
        if self.family not in ("modal", "sectional", "bulk"):
            raise ValueError(f"Unknown aerosol family {self.family!r}.")
        known = {s.name for s in self.species}
        for mode in self.modes:
            missing = set(mode.species) - known
            if missing:
                raise ValueError(
                    f"Mode {mode.name!r} references unknown species "
                    f"{sorted(missing)}; declare them in ``species``."
                )

    # ------------------------------------------------------------------
    # Compose-time helpers (pure Python — never called under jit)
    # ------------------------------------------------------------------

    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def mode_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.modes)

    @property
    def mode_shorts(self) -> tuple[str, ...]:
        return tuple(m.short for m in self.modes)

    def mode_index(self, name: str) -> int:
        for i, m in enumerate(self.modes):
            if m.name == name or m.short == name:
                return i
        raise KeyError(f"No mode named {name!r} in {self.mode_names}.")

    def mode(self, name: str) -> AerosolMode:
        return self.modes[self.mode_index(name)]

    def species_props(self, name: str) -> AerosolSpecies:
        for s in self.species:
            if s.name == name:
                return s
        raise KeyError(f"No species named {name!r}.")

    def classes_for(self, species: str) -> tuple[AerosolMode, ...]:
        """Classes (modes/bins) that may carry ``species``, in spec order.

        The family-agnostic way for an emission term to discover which classes a
        species lives in without naming modes literally — a size-resolved source
        scheme (Gong sea salt, Tegen dust) partitions its size-distributed flux
        across these using each class's ``dgnum_lo``/``dgnum_hi`` range.
        """
        return tuple(m for m in self.modes if species in m.species)

    def primary_split(
        self, species: str
    ) -> tuple[tuple[AerosolMode, float], ...]:
        """Default ``((class, mass_fraction), ...)`` for primary ``species``.

        Resolves the population's :attr:`primary_emission` policy table to the
        actual classes. Terms with a *tunable* split (e.g. dust's accumulation
        fraction) take the class set from here and substitute their own
        fractions. Raises ``KeyError`` for a species with no primary policy.
        """
        table = self.primary_emission.get(species)
        if table is None:
            raise KeyError(
                f"No primary-emission split defined for {species!r}; add it to "
                "the spec's ``primary_emission`` table."
            )
        return tuple((self.mode(short), frac) for short, frac in table)
