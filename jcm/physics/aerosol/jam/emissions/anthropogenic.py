"""``AnthropogenicEmissions`` — prescribed primary super-sector emissions (#498).

Emits prescribed SO₂/BC/OC surface fluxes (per super-sector, read from
``ForcingData``) over a **smooth, differentiable vertical profile** rather than
HAMMOZ's discrete injection levels, so injection height is calibratable by
gradient (:mod:`injection`). The super-sectors (:mod:`sectors`) cover both CEDS
anthropogenic activity (surface / elevated-industrial / shipping) and open
**biomass burning** (GFED), which differs only in its deeper FIRE injection
profile — all four run through the identical speciation + injection path and are
independently gated by which ``emis_<sector>_<species>`` forcing channels exist.
Each super-sector's flux is split following HAMMOZ (:mod:`sectors`):

* SO₂ → a primary-SO₄ fraction (default 2.5 %, differentiable) into Aitken+accum
  modal sulfate, the remainder into the ``g_so2`` gas tracer (oxidised by the
  gas-phase sulfur chemistry, #496);
* BC → primary-carbon-mode black carbon;
* OC → primary-carbon-mode POA (mass scaled by OM:OC = 1.4).

The injection height/thickness and the primary-SO₄ fraction are differentiable
``EmissionParameters`` (per super-sector). With no CEDS forcing supplied the
flux fields default to zero, so the term is inert until the data pipeline
(Phases B–D) is wired.

When ``spec.sector_emission`` is set (M7, jax-gcm#1017), the species->mode
split above is replaced by HAM's own per-sector-CLASS mode/size targets
(:mod:`ham_sectors`): the organic species' token comes from the policy (M7's
``"oc"``, never hard-coded), and each target carries its own emitted size, so
the implied number comes from HAM's ``cmr``-based factor rather than the
mode's equilibrium geometry. ``surface_combustion``/``elevated_industrial``
each carry an optional subset channel (``emis_residential_*``/
``emis_energy_*``) splitting out their RCO/ENE share, which HAM sizes
differently from the rest of the super-sector; absent, the whole super-sector
takes its parent class's size (a declared, logged approximation — jax-gcm#1017
tracked gap F6). MAM4 (``spec.sector_emission is None``) is unaffected.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.emissions.distributors import (
    emit_over_profile,
    particle_mean_mass,
)
from jcm.physics.aerosol.jam.emissions.ham_sectors import (
    SECTOR_ROUTING,
    cmr_to_emission_diameter,
)
from jcm.physics.aerosol.jam.emissions.injection import (
    gaussian_injection_weights,
)
from jcm.physics.aerosol.jam.emissions.sectors import (
    OM_OC_RATIO,
    SECTOR_DEFAULTS,
    SO4_PRIMARY_FRACTION,
    SUPER_SECTORS,
)
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name,
    mass_name,
    number_name,
)
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm
from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
    accumulate_emission_fluxes, emission_flux_keys)

logger = logging.getLogger(__name__)


@tree_math.struct
class EmissionParameters:
    """Differentiable per-super-sector emission knobs (HAMMOZ defaults).

    The arrays are indexed by :data:`SUPER_SECTORS`. ``injection_height`` and
    ``injection_thickness`` set the smooth Gaussian vertical profile (the
    load-bearing calibratable uncertainty); ``so4_primary_fraction`` is the
    SO₂→primary-SO₄ split.
    """

    injection_height: jnp.ndarray       # (n_sector,) [m]
    injection_thickness: jnp.ndarray    # (n_sector,) [m]
    so4_primary_fraction: jnp.ndarray   # (n_sector,) [-]
    scale: jnp.ndarray                  # overall emission scale

    @classmethod
    def default(cls) -> "EmissionParameters":
        return cls(
            injection_height=jnp.asarray(
                [SECTOR_DEFAULTS[s].injection_height for s in SUPER_SECTORS]
            ),
            injection_thickness=jnp.asarray(
                [SECTOR_DEFAULTS[s].injection_thickness for s in SUPER_SECTORS]
            ),
            so4_primary_fraction=jnp.full(
                len(SUPER_SECTORS), SO4_PRIMARY_FRACTION
            ),
            scale=jnp.asarray(1.0),
        )


class AnthropogenicEmissions(PhysicsTerm):
    """Prescribed CEDS anthropogenic SO₂/BC/OC emission over super-sectors."""

    name: ClassVar[str] = "jam_anthropogenic_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = (
        "air_density", "layer_thickness", "height_full",
    )
    provides: ClassVar[tuple[str, ...]] = emission_flux_keys()

    def __init__(
        self,
        params: EmissionParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold the (differentiable) emission params and the population."""
        self.params = nnx.Param(params or EmissionParameters.default())
        self._spec = spec or MAM4_SPEC
        # SO2->SO4 mass factor from the POPULATION's own so4 (same expression
        # as sectors.SO2_TO_SO4_MASS, so bit-identical for MAM4; diverges only
        # where the population's so4 molar mass differs, e.g. M7's 96.0631
        # g/mol vs MAM4-MOM's 115 g/mol ammonium bisulfate).
        self._so2_to_so4_mass = (
            self._spec.species_props("so4").molar_mass
            / GAS_SPECIES["so2"].molar_mass
        )
        # HAM's per-sector-class mode/size targets (jax-gcm#1017): None for
        # every MAM4-family population, which keeps the primary_split-based
        # path below untouched.
        self._sector_policy = self._spec.sector_emission

    @staticmethod
    def _flux(forcing, name, ncols):
        """Per-super-sector species surface flux [kg/m²/s]; 0 if not forced.

        Reads from the ``anthropogenic_emissions`` mapping on ``ForcingData``
        (keyed ``emis_<sector>_<species>``; see the emissions-file contract in
        ``.claude/aerosol_emissions_plan.md``). A single dict-valued field —
        rather than one struct field per (sector, species) — keeps the forcing
        general: new sectors/species need no ``ForcingData`` change, and
        ``select(date)`` slices the per-channel ``TimeSeries`` leaves
        automatically. Absent forcing, mapping, or channel ⇒ zero, so the term
        is inert until the matching channel is supplied.
        """
        emis = getattr(forcing, "anthropogenic_emissions", None) if forcing is not None else None
        v = emis.get(name) if emis is not None else None
        if v is not None and jnp.size(v) == ncols:
            return jnp.ravel(v)
        return jnp.zeros((ncols,))

    @staticmethod
    def _has_channel(forcing, name, ncols):
        """Check a channel's presence — a compose-time Python bool, never traced."""
        emis = getattr(forcing, "anthropogenic_emissions", None) if forcing is not None else None
        v = emis.get(name) if emis is not None else None
        return v is not None and jnp.size(v) == ncols

    def _emit_sector_species(self, add_mass, weights, species, flux, ham_class):
        """Emit ``species``'s flux into HAM class ``ham_class``'s own targets.

        Mass goes to ``mass_name(species, mode.short)``; number uses HAM's
        own ``cmr``-based mass->number factor (:func:`cmr_to_emission_diameter`
        through :func:`particle_mean_mass`'s monodisperse-diameter path),
        NOT the mode's own equilibrium ``number_factor`` -- HAM assumes the
        freshly emitted particles sit at its own prescribed size, which
        differs from the mode's steady-state geometry (see
        ``emissions/ham_sectors.py``).
        """
        density = self._spec.species_props(species).density
        for target in self._sector_policy.targets[ham_class].get(species, ()):
            mode = self._spec.mode(target.mode)
            mass_flux = flux * target.mass_fraction
            diameter = cmr_to_emission_diameter(target.cmr_m, mode.geom_std_dev)
            m_p = particle_mean_mass(mode, density, emission_diameter=diameter)
            add_mass(mass_name(species, mode.short), mass_flux, weights)
            add_mass(number_name(mode.short), mass_flux / m_p, weights)

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        rho = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        height_full = diagnostics["height_full"]
        nlev, ncols = state.temperature.shape
        zeros3 = jnp.zeros((nlev, ncols))

        tends: dict[str, jnp.ndarray] = {}

        def add_mass(name, flux2d, weights):
            tends[name] = tends.get(name, zeros3) + emit_over_profile(
                flux2d, weights, rho, dz
            )

        def add_aerosol(species, mode, flux2d, weights):
            # Mass into (species, class); implied number from the class's
            # ``number_factor`` (the family-agnostic mass→number conversion, so
            # this is unchanged for a sectional bin), both over the same profile.
            add_mass(mass_name(species, mode.short), flux2d, weights)
            density = self._spec.species_props(species).density
            add_mass(number_name(mode.short),
                     flux2d * mode.number_factor / density, weights)

        emi_bb: dict[str, jnp.ndarray] = {}
        for i, sector in enumerate(SUPER_SECTORS):
            weights = gaussian_injection_weights(
                height_full, dz,
                p.injection_height[i], p.injection_thickness[i],
            )
            so2 = p.scale * self._flux(forcing, f"emis_{sector}_so2", ncols)
            bc = p.scale * self._flux(forcing, f"emis_{sector}_bc", ncols)
            oc = p.scale * self._flux(forcing, f"emis_{sector}_oc", ncols)

            # SO2 → primary SO4 + g_so2 gas remainder (S-conserving); the gas
            # remainder is independent of mode targets, so this is unchanged
            # by the sector-class policy below.
            frac = p.so4_primary_fraction[i]
            so4_mass = frac * so2 * self._so2_to_so4_mass
            add_mass(gas_name("so2"), (1.0 - frac) * so2, weights)

            if sector == "biomass_burning":
                # MMPPE emi_bb_*: the open-burning fluxes as emitted
                # (SO2 as SO2, OC as OC), before speciation/OM scaling.
                emi_bb = {"so2": so2, "bc": bc, "oc": oc}

            if self._sector_policy is None:
                # MAM4 default, unchanged: the population owns which classes
                # receive primary sulfate/BC/POA and in what proportion
                # (``primary_split``) — no Aitken/accum assumption here.
                for mode, mode_frac in self._spec.primary_split("so4"):
                    add_aerosol("so4", mode, so4_mass * mode_frac, weights)
                for mode, mode_frac in self._spec.primary_split("bc"):
                    add_aerosol("bc", mode, bc * mode_frac, weights)
                for mode, mode_frac in self._spec.primary_split("poa"):
                    add_aerosol("poa", mode, oc * OM_OC_RATIO * mode_frac, weights)
                continue

            # HAM's own per-sector-class mode/size targets (jax-gcm#1017).
            # jcm's four super-sectors each resolve to one HAM class, with
            # two of them (surface_combustion, elevated_industrial) carrying
            # an optional SUBSET channel (residential/energy) whose flux is
            # INCLUDED in the super-sector total, not additional to it.
            main_class, subset = SECTOR_ROUTING[sector]
            om_oc = self._sector_policy.om_oc
            for species, flux, scale in (
                ("so4", so4_mass, 1.0), ("bc", bc, 1.0), ("oc", oc, om_oc),
            ):
                main_flux, subset_flux, subset_class = flux * scale, None, None
                if subset is not None:
                    subset_name, subset_class = subset
                    channel = f"emis_{subset_name}_{species if species != 'so4' else 'so2'}"
                    if self._has_channel(forcing, channel, ncols):
                        # The subset channel carries the pre-speciation
                        # quantity (SO2 for so4, matching the main channels'
                        # own convention); scale it the same way as the
                        # parent flux before splitting mass.
                        raw = p.scale * self._flux(forcing, channel, ncols)
                        subset_raw = raw if species != "so4" else frac * raw * self._so2_to_so4_mass
                        subset_flux = subset_raw * scale
                        main_flux = main_flux - subset_flux
                    else:
                        logger.warning(
                            "AnthropogenicEmissions: forcing.anthropogenic_emissions "
                            "has no %r channel, so the whole %r super-sector's %s "
                            "is sized as %r (jax-gcm#1017, tracked gap F6) rather "
                            "than split out its %r share at %r's own HAM size.",
                            channel, sector, species, main_class, subset_name,
                            subset_class)
                self._emit_sector_species(add_mass, weights, species, main_flux, main_class)
                if subset_flux is not None:
                    self._emit_sector_species(
                        add_mass, weights, species, subset_flux, subset_class)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tends,
        )
        # Publish this term's contribution to the AeroCom per-species
        # emission fluxes (accumulated across all emitting terms).
        diagnostics = accumulate_emission_fluxes(
            diagnostics, tends,
            diagnostics["air_density"],
            diagnostics["layer_thickness"])
        # Biomass-burning splits ride the same reset-per-step keys.
        for spc, flux in emi_bb.items():
            key = f"emi_bb_{spc}"
            diagnostics = {**diagnostics,
                           key: diagnostics.get(key, 0.0) + flux}

        return tendency, diagnostics
