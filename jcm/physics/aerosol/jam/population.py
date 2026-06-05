"""Microphysics-neutral aerosol population description.

These are pure-Python frozen dataclasses that describe the *shape* of an
aerosol population — how many modes, which species each carries, and the
per-mode/per-species physical metadata a process harness needs. They are
the swap point between the microphysics-agnostic harness (emissions, dry/
wet deposition, sedimentation, ARG activation) and an interchangeable
microphysics core (placeholder today; MAM4-JAX later).

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


@dataclasses.dataclass(frozen=True)
class AerosolSpecies:
    """Physical properties of one aerosol species.

    Attributes:
        name: short canonical token used in tracer keys (e.g. ``"so4"``).
        molar_mass: molar mass [kg/mol] (informational for the placeholder;
            used by gas–aerosol exchange in the real core).
        density: particle-material density [kg/m³].
        hygroscopicity: κ (kappa-Köhler hygroscopicity parameter) [-].
        long_name: descriptive name (e.g. ``"sulfate"``); also the bridge
            to MAM4's ``SPECNAME_AMODE`` tokens.
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
