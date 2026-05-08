"""``EchamBoundaryConditions`` — ECHAM time-varying boundary condition term.

Sets the surface and chemistry inputs that the rest of the ECHAM physics
expects to find in the typed ``RadiationData`` / ``SurfaceData`` /
``ChemistryData`` sub-structs:

- Surface albedo (visible + NIR) and emissivity, computed as the
  fmask/ice/ocean weighted average of fixed per-type values.
- Surface temperature (land = ``forcing.stl_am``; ocean = ``forcing.sea_surface_temperature``).
- Roughness length (1 cm over land, 0.1 mm over ocean).
- CO2 (from ``forcing.co2_vmr``); CH4 (1900 ppbv) and O3 (300 ppbv) hardcoded
  for now (forcing-field equivalents are a follow-up; see
  ``echam/forcing.py`` history).

The numerical implementation matches what was previously in
``apply_forcing_data`` (echam/forcing.py); this term is the ECHAM-specific
home for that routine. The ``Echam`` prefix is intentional — the weighted
albedo defaults and the hardcoded CH4/O3 are ECHAM choices, not generic
boundary conditions.

Typed sub-structs are written into the diagnostics dict under the legacy
``_radiation`` / ``_surface`` / ``_chemistry`` keys so that the legacy
``apply_*`` consumer terms (which still build a full ``PhysicsData`` via
``_data_from_diagnostics``) see the same shape they always have. As those
consumer terms migrate to scheme-named terms in later phases, this term
will move to writing scheme-public keys directly.

Date: 2026-05-07
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from jcm.forcing import ForcingData
from jcm.physics.chemistry.simple_chemistry import ChemistryData
from jcm.physics.radiation.radiation_types import RadiationData
from jcm.physics.surface.echam.surface_types import SurfaceData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


def _surface_optical_properties(
    land_fraction: jnp.ndarray,
    sea_ice_fraction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Weighted-average ECHAM albedo/emissivity per grid box."""
    # Per-type values match the legacy ``_compute_surface_properties`` in
    # echam/forcing.py. They are tunable defaults, not Parameters.
    land_albedo_vis, land_albedo_nir, land_emissivity = 0.15, 0.25, 0.95
    ocean_albedo_vis, ocean_albedo_nir, ocean_emissivity = 0.05, 0.05, 0.98
    seaice_albedo_vis, seaice_albedo_nir, seaice_emissivity = 0.80, 0.70, 0.95

    ocean_fraction = jnp.maximum(
        1.0 - land_fraction - sea_ice_fraction, 0.0,
    )
    albedo_vis = (
        land_fraction * land_albedo_vis
        + ocean_fraction * ocean_albedo_vis
        + sea_ice_fraction * seaice_albedo_vis
    )
    albedo_nir = (
        land_fraction * land_albedo_nir
        + ocean_fraction * ocean_albedo_nir
        + sea_ice_fraction * seaice_albedo_nir
    )
    emissivity = (
        land_fraction * land_emissivity
        + ocean_fraction * ocean_emissivity
        + sea_ice_fraction * seaice_emissivity
    )
    return albedo_vis, albedo_nir, emissivity


class EchamBoundaryConditions(PhysicsTerm):
    """Apply ECHAM time-varying boundary conditions to the diagnostics dict.

    Operates on column-vectorized state ``(nlev, ncols)``. Writes ECHAM
    radiation/surface/chemistry typed sub-structs under the legacy
    ``_radiation`` / ``_surface`` / ``_chemistry`` keys for downstream
    legacy ``apply_*`` consumers.
    """

    name: ClassVar[str] = "echam_boundary_conditions"
    category: ClassVar[str] = "forcing"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = (
        "radiation", "surface", "chemistry",
    )

    def __init__(self):
        """No tunables; nothing to initialise."""

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Populate radiation, surface, and chemistry inputs."""
        nlev, ncols = state.temperature.shape

        albedo_vis, albedo_nir, emissivity = _surface_optical_properties(
            terrain.fmask, forcing.sice_am,
        )
        surface_temperature = jnp.where(
            terrain.fmask > 0.5,
            forcing.stl_am,
            forcing.sea_surface_temperature,
        )
        roughness_length = jnp.where(
            terrain.fmask > 0.5,
            0.01,    # 1 cm over land
            0.0001,  # 0.1 mm over ocean
        )

        # Reshape from grid (nlon, nlat) to column (ncols,) format.
        albedo_vis = albedo_vis.reshape(ncols)
        albedo_nir = albedo_nir.reshape(ncols)
        emissivity = emissivity.reshape(ncols)
        surface_temperature = surface_temperature.reshape(ncols)
        roughness_length = roughness_length.reshape(ncols)

        # CH4 and O3 are still ECHAM-hardcoded; the forcing-field versions
        # can land in a follow-up. CO2 already comes from ``forcing.co2_vmr``.
        co2_vmr_value = forcing.co2_vmr
        ch4_vmr_value = 1900.0e-3  # ppbv → ppmv: 1.9 ppmv
        o3_vmr_value = 300.0e-3    # ppbv → ppmv: 0.3 ppmv

        # Start from whatever the previous step (or upstream term) left us
        # so we don't clobber radiation cache or other sub-struct fields.
        radiation = diagnostics.get(
            "radiation", RadiationData.zeros((ncols,), nlev),
        ).copy(
            surface_albedo_vis=albedo_vis,
            surface_albedo_nir=albedo_nir,
            surface_emissivity=emissivity,
        )
        surface = diagnostics.get(
            "surface", SurfaceData.zeros((ncols,), nlev),
        ).copy(
            surface_temperature=surface_temperature,
            skin_temperature=surface_temperature,
            roughness_length=roughness_length,
        )
        chemistry_zero = diagnostics.get(
            "chemistry", ChemistryData.zeros((ncols,), nlev),
        )
        chemistry = chemistry_zero.copy(
            co2_vmr=jnp.ones_like(chemistry_zero.co2_vmr) * co2_vmr_value,
            methane_vmr=jnp.ones_like(chemistry_zero.methane_vmr)
            * ch4_vmr_value,
            ozone_vmr=jnp.ones_like(chemistry_zero.ozone_vmr)
            * o3_vmr_value,
        )

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {
            **diagnostics,
            "radiation": radiation,
            "surface": surface,
            "chemistry": chemistry,
        }
