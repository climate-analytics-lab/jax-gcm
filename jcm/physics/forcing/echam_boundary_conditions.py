"""``EchamBoundaryConditions`` — ECHAM time-varying boundary condition term.

Sets the surface and chemistry inputs that the rest of the ECHAM physics
expects to find in the typed ``RadiationData`` / ``SurfaceData`` /
``ChemistryData`` sub-structs:

- Surface albedo (visible + NIR) and emissivity, computed as the
  fmask/ice/ocean weighted average of per-type values held in a
  differentiable :class:`SurfaceOpticsParameters` (#347).
- Surface temperature (land = ``forcing.stl_am``; ocean = ``forcing.sea_surface_temperature``).
- Roughness length (1 cm over land, 0.1 mm over ocean).
- CO2 and CH4 from ``forcing.co2_vmr`` / ``forcing.ch4_vmr``; O3 from the
  ``forcing.ozone_climatology`` when loaded, else this term's analytical
  profile.

The numerical implementation matches what was previously in
``apply_forcing_data`` (echam/forcing.py); this term is the ECHAM-specific
home for that routine. The ``Echam`` prefix is intentional — the weighted
albedo defaults and the analytic-ozone fallback are ECHAM choices, not
generic boundary conditions.

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
from flax import nnx, struct

from jcm.forcing import ForcingData
from jcm.physics.chemistry.simple_chemistry import ChemistryData
from jcm.physics.radiation.radiation_types import RadiationData
from jcm.physics.surface.echam.surface_types import SurfaceData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


@struct.dataclass
class SurfaceOpticsParameters:
    """Per-surface-type albedo/emissivity, ECHAM defaults (#347).

    All nine values are differentiable pytree leaves, like every other
    physics parameter.
    """

    land_albedo_vis: jnp.ndarray = 0.15
    land_albedo_nir: jnp.ndarray = 0.25
    land_emissivity: jnp.ndarray = 0.95
    ocean_albedo_vis: jnp.ndarray = 0.05
    ocean_albedo_nir: jnp.ndarray = 0.05
    ocean_emissivity: jnp.ndarray = 0.98
    seaice_albedo_vis: jnp.ndarray = 0.80
    seaice_albedo_nir: jnp.ndarray = 0.70
    seaice_emissivity: jnp.ndarray = 0.95


def _surface_optical_properties(
    land_fraction: jnp.ndarray,
    sea_ice_fraction: jnp.ndarray,
    p: SurfaceOpticsParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Weighted-average ECHAM albedo/emissivity per grid box."""
    ocean_fraction = jnp.maximum(
        1.0 - land_fraction - sea_ice_fraction, 0.0,
    )
    albedo_vis = (
        land_fraction * p.land_albedo_vis
        + ocean_fraction * p.ocean_albedo_vis
        + sea_ice_fraction * p.seaice_albedo_vis
    )
    albedo_nir = (
        land_fraction * p.land_albedo_nir
        + ocean_fraction * p.ocean_albedo_nir
        + sea_ice_fraction * p.seaice_albedo_nir
    )
    emissivity = (
        land_fraction * p.land_emissivity
        + ocean_fraction * p.ocean_emissivity
        + sea_ice_fraction * p.seaice_emissivity
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
    requires: ClassVar[tuple[str, ...]] = (
        # Needed by the analytical ozone profile (Fortuin & Kelder-style)
        # that seeds ``chemistry.ozone_vmr`` each step. ``MoistAirColumnState``
        # populates both diagnostics; this term must run after it.
        "pressure_full", "surface_pressure",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "radiation", "surface", "chemistry",
    )
    # Carry seeded as zeros by the base class. The first
    # ``compute_tendencies`` call overwrites every boundary field from
    # ``ForcingData`` at the top of the term loop, so the zero seed
    # never leaks into downstream physics.
    carry_slots: ClassVar[dict[str, type]] = {
        "radiation": RadiationData,
        "surface": SurfaceData,
        "chemistry": ChemistryData,
    }

    def __init__(
        self,
        ozone_peak_ppmv: float = 8.0,
        ozone_peak_height_m: float = 20_000.0,
        ozone_scale_height_m: float = 7_000.0,
        surface_optics: SurfaceOpticsParameters | None = None,
    ):
        """Hold the analytical-ozone profile parameters.

        Defaults broadly track Fortuin & Kelder 1998 zonal-mean
        climatology near the equator: ~8 ppmv peak at ~20 km height with
        a 7 km e-folding scale above (and a linear ramp from surface to
        peak below). Override per-Hydra to drive a different vertical
        profile.

        Args:
            ozone_peak_ppmv: Stratospheric peak ozone volume mixing
                ratio (ppmv).
            ozone_peak_height_m: Altitude of the ozone maximum (m).
            ozone_scale_height_m: e-folding height for decay above the
                peak (m).
            surface_optics: Per-surface-type albedo/emissivity; ECHAM
                defaults when ``None``. Held in an ``nnx.Param`` so all
                nine values are differentiable leaves (#347).

        """
        # Store as plain Python floats; the ``ChemistryParameters``
        # struct (a ``tree_math.struct`` of JAX arrays) is built fresh
        # inside ``__call__`` so it stays a JIT-friendly closure value
        # rather than a stored pytree on this flax ``nnx`` term.
        self._ozone_peak_ppmv = float(ozone_peak_ppmv)
        self._ozone_peak_height_m = float(ozone_peak_height_m)
        self._ozone_scale_height_m = float(ozone_scale_height_m)
        self.surface_optics = nnx.Param(
            surface_optics or SurfaceOpticsParameters())

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
            terrain.fmask, forcing.sice_am, self.surface_optics.get_value(),
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

        # CH4 comes from ``ForcingData`` (#347) and seeds the chemistry
        # diagnostic (where the methane-loss scheme evolves it). CO2 is *not*
        # seeded here: it is a prescribed forcing read straight from
        # ``forcing.co2_vmr`` by radiation, so the chemistry diagnostic never
        # carries it.
        ch4_vmr_value = forcing.ch4_vmr

        # O3: prefer the realistic CMIP6/ECHAM-style climatology carried
        # on ``forcing.ozone_climatology`` (loaded from a netCDF in
        # ``build_forcing``). Without that file, fall back to the
        # analytical Fortuin & Kelder-style surrogate driven by this
        # term's ``ozone_peak_*`` constructor kwargs. The analytical
        # profile is known to be a poor match for the real climatology
        # (peak in the wrong place, troposphere overestimated by ~50x,
        # mesopause underestimated by ~30x — see validation against
        # T63_ozone_picontrol.nc), so it should be treated as a
        # placeholder for unit tests / SCM where no climatology is
        # available.
        #
        # ``chemistry.ozone_vmr`` is consumed by RRTMGP as ppmv (a
        # ``* 1e-6`` converts to mole fraction inside that term).
        if forcing.ozone_climatology.is_loaded():
            # Pre-interpolated to the model's hybrid grid offline (see
            # ``jcm.data.bc.interpolate_ozone``) — straight slice, no
            # online vertical interp.
            ozone_vmr_ppmv = forcing.ozone_climatology.o3_ppmv
            # Enforce the OzoneClimatology contract here, at trace time,
            # rather than letting a violation travel. Unlike the oxidant
            # fields -- whose consumer reshapes to ``temperature.shape`` --
            # this array is handed to RRTMGP's ``lev_to_col`` as a plain
            # transpose, so an extra horizontal axis does not fail here: it
            # fails much later inside the radiation halo padder with a shape
            # error that names neither ozone nor the loader that produced it.
            # ``forcing`` reaches terms UNFLATTENED, so every loader owes the
            # flattened ``(nlev, ncols)`` layout the state already has.
            if ozone_vmr_ppmv.ndim != state.temperature.ndim:
                raise ValueError(
                    "forcing.ozone_climatology.o3_ppmv has "
                    f"{ozone_vmr_ppmv.ndim} dims {ozone_vmr_ppmv.shape}, but "
                    f"the physics view is {state.temperature.ndim}-D "
                    f"{state.temperature.shape}. OzoneClimatology must be "
                    "stored with the horizontal already flattened to the "
                    "term view -- (ntime, nlev, ncols) pre-select. Fix the "
                    "loader that built this forcing."
                )
        else:
            from jcm.physics.chemistry.simple_chemistry import (
                ChemistryParameters,
                fixed_ozone_distribution,
            )
            defaults = ChemistryParameters.default()
            ozone_params = ChemistryParameters(
                ozone_scale_height=jnp.asarray(self._ozone_scale_height_m),
                ozone_max_vmr=jnp.asarray(self._ozone_peak_ppmv * 1000.0),
                ozone_tropopause_height=jnp.asarray(self._ozone_peak_height_m),
                ozone_stratosphere_coeff=defaults.ozone_stratosphere_coeff,
                methane_surface_vmr=defaults.methane_surface_vmr,
                methane_lifetime=defaults.methane_lifetime,
                methane_oh_scaling=defaults.methane_oh_scaling,
            )
            ozone_vmr_ppmv = fixed_ozone_distribution(
                pressure=diagnostics["pressure_full"],
                surface_pressure=diagnostics["surface_pressure"],
                temperature=state.temperature,
                config=ozone_params,
            ) * 1e-3

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
            methane_vmr=jnp.ones_like(chemistry_zero.methane_vmr)
            * ch4_vmr_value,
            ozone_vmr=ozone_vmr_ppmv,
        )

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {
            **diagnostics,
            "radiation": radiation,
            "surface": surface,
            "chemistry": chemistry,
        }
