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
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.emissions.distributors import (
    emit_over_profile,
    particle_mean_mass,
)
from jcm.physics.aerosol.jam.emissions.injection import (
    gaussian_injection_weights,
)
from jcm.physics.aerosol.jam.emissions.sectors import (
    CARBON_MODE,
    OM_OC_RATIO,
    SECTOR_DEFAULTS,
    SO2_TO_SO4_MASS,
    SO4_AITKEN_FRACTION,
    SO4_MODES,
    SO4_PRIMARY_FRACTION,
    SUPER_SECTORS,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name,
    mass_name,
    number_name,
)
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm


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
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: EmissionParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold the (differentiable) emission params and the population."""
        self.params = nnx.Param(params or EmissionParameters.default())
        self._spec = spec or MAM4_SPEC

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

        def add_aerosol(species, mode_short, flux2d, weights):
            # Mass into (species, mode); implied number into the mode, both over
            # the same vertical profile.
            add_mass(mass_name(species, mode_short), flux2d, weights)
            density = self._spec.species_props(species).density
            m_p = particle_mean_mass(self._spec.mode(mode_short), density)
            add_mass(number_name(mode_short), flux2d / m_p, weights)

        for i, sector in enumerate(SUPER_SECTORS):
            weights = gaussian_injection_weights(
                height_full, dz,
                p.injection_height[i], p.injection_thickness[i],
            )
            so2 = p.scale * self._flux(forcing, f"emis_{sector}_so2", ncols)
            bc = p.scale * self._flux(forcing, f"emis_{sector}_bc", ncols)
            oc = p.scale * self._flux(forcing, f"emis_{sector}_oc", ncols)

            # SO2 → primary SO4 (Aitken/accum) + g_so2 gas (S-conserving).
            frac = p.so4_primary_fraction[i]
            so4_mass = frac * so2 * SO2_TO_SO4_MASS
            ait, acc = SO4_MODES
            add_aerosol("so4", ait, so4_mass * SO4_AITKEN_FRACTION, weights)
            add_aerosol("so4", acc, so4_mass * (1.0 - SO4_AITKEN_FRACTION),
                        weights)
            add_mass(gas_name("so2"), (1.0 - frac) * so2, weights)

            # Primary carbonaceous mass → primary_carbon mode.
            add_aerosol("bc", CARBON_MODE, bc, weights)
            add_aerosol("poa", CARBON_MODE, oc * OM_OC_RATIO, weights)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tends,
        )
        return tendency, diagnostics
