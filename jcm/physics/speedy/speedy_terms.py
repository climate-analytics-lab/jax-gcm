"""PhysicsTerm wrappers for existing SPEEDY physics functions.

Each wrapper delegates to the original SPEEDY function, translating between
the composable ``diagnostics`` dict and the legacy typed ``PhysicsData``
struct. The numerical implementation is untouched.

Date: 2026-04-12
"""

from __future__ import annotations

from typing import ClassVar

from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.speedy.physics_data import (
    PhysicsData,
)
from jcm.physics.speedy.speedy_coords import SpeedyCoords
from jcm.physics.speedy.params import (
    Parameters,
    ConvectionParameters,
    CondensationParameters,
    ShortwaveRadiationParameters,
    ModRadConParameters,
    SurfaceFluxParameters,
    VerticalDiffusionParameters,
)

import jax.numpy as jnp
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData


def set_physics_flags(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData = None,
    terrain: TerrainData = None,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Set per-step compute flags for SPEEDY parameterizations.

    Currently only toggles the shortwave-radiation flag every ``nstrad`` steps
    so that the costly clouds + shortwave fluxes only recompute on radiation
    sub-steps. The step counter is the radiation carry slot's own
    :attr:`SWRadiationData.step` — incremented each call so the gate
    advances without any model-wide step plumbing.
    """
    from jcm.physics.speedy.physical_constants import nstrad
    step = physics_data.shortwave_rad.step
    compute_shortwave = (jnp.mod(step, nstrad) == 0)
    shortwave_data = physics_data.shortwave_rad.copy(
        compute_shortwave=compute_shortwave,
        step=step + 1,
    )
    physics_data = physics_data.copy(shortwave_rad=shortwave_data)
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return physics_tendencies, physics_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_with(**overrides) -> Parameters:
    """Build a Parameters from defaults, overriding specific sub-structs."""
    p = Parameters.default()
    return Parameters(
        convection=overrides.get("convection", p.convection),
        condensation=overrides.get("condensation", p.condensation),
        shortwave_radiation=overrides.get("shortwave_radiation", p.shortwave_radiation),
        mod_radcon=overrides.get("mod_radcon", p.mod_radcon),
        surface_flux=overrides.get("surface_flux", p.surface_flux),
        vertical_diffusion=overrides.get("vertical_diffusion", p.vertical_diffusion),
    )


# ---------------------------------------------------------------------------
# Helpers for diagnostics ↔ PhysicsData translation
# ---------------------------------------------------------------------------

def _data_from_diagnostics(
    diagnostics: dict, coords: SpeedyCoords,
    nodal_shape: tuple, num_levels: int,
) -> PhysicsData:
    """Reconstruct a PhysicsData from the diagnostics dict.

    Keys that haven't been populated yet will get their default zero values.
    ``nodal_shape`` and ``num_levels`` are passed explicitly (not from the
    diagnostics dict) so they remain static Python values under JIT.

    ``dt_seconds`` is sourced from the ``"_dt_seconds"`` plumbing slot
    that ``ComposablePhysics`` injects at the top of every
    ``compute_tendencies`` call. The shortwave sub-stepping counter
    lives on the radiation carry (see :func:`set_physics_flags`), so
    no date / model-wide step is threaded into PhysicsData any more.
    """
    dt_seconds = diagnostics.get("_dt_seconds", 1800.0)

    data = PhysicsData.zeros(
        nodal_shape, num_levels,
        dt_seconds=dt_seconds,
        speedy_coords=coords,
    )

    # Restore any previously populated sub-structs from the diagnostics
    if "_shortwave_rad" in diagnostics:
        data = data.copy(shortwave_rad=diagnostics["_shortwave_rad"])
    if "_longwave_rad" in diagnostics:
        data = data.copy(longwave_rad=diagnostics["_longwave_rad"])
    if "_convection" in diagnostics:
        data = data.copy(convection=diagnostics["_convection"])
    if "_mod_radcon" in diagnostics:
        data = data.copy(mod_radcon=diagnostics["_mod_radcon"])
    if "_humidity" in diagnostics:
        data = data.copy(humidity=diagnostics["_humidity"])
    if "_condensation" in diagnostics:
        data = data.copy(condensation=diagnostics["_condensation"])
    if "_surface_flux" in diagnostics:
        data = data.copy(surface_flux=diagnostics["_surface_flux"])
    if "_land_model" in diagnostics:
        data = data.copy(land_model=diagnostics["_land_model"])

    return data


def _diagnostics_from_data(diagnostics: dict, data: PhysicsData) -> dict:
    """Store all PhysicsData sub-structs into the diagnostics dict."""
    return {
        **diagnostics,
        "_shortwave_rad": data.shortwave_rad,
        "_longwave_rad": data.longwave_rad,
        "_convection": data.convection,
        "_mod_radcon": data.mod_radcon,
        "_humidity": data.humidity,
        "_condensation": data.condensation,
        "_surface_flux": data.surface_flux,
        "_land_model": data.land_model,
    }


# ---------------------------------------------------------------------------
# Base class for SPEEDY term wrappers
# ---------------------------------------------------------------------------

class SpeedyTermBase(PhysicsTerm):
    """Base for SPEEDY term wrappers.

    Handles SpeedyCoords caching as nnx.Variable and provides the
    translation helpers. Subclasses hold their own parameter sub-struct
    as nnx.Param and implement __call__.
    """

    def __init__(self):
        """Initialize SpeedyTermBase."""
        # Placeholder — populated by cache_coords
        self._coords_cached = False

    def cache_coords(self, coords):
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)
        self._speedy_coords = nnx.Variable(speedy_coords)
        self._nodal_shape = coords.horizontal.nodal_shape
        self._num_levels = coords.nodal_shape[0]
        self._coords_cached = True

    def _build_data(self, diagnostics: dict) -> PhysicsData:
        """Reconstruct PhysicsData from diagnostics with cached shape info."""
        return _data_from_diagnostics(
            diagnostics, self._speedy_coords.get_value(),
            self._nodal_shape, self._num_levels,
        )


# ---------------------------------------------------------------------------
# Concrete SPEEDY term wrappers
# ---------------------------------------------------------------------------

class SpeedyFlags(SpeedyTermBase):
    """Sets physics flags (e.g. whether to compute shortwave radiation this step)."""

    name: ClassVar[str] = "speedy_flags"
    category: ClassVar[str] = "flags"

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = Parameters.default()  # flags don't use tunable params

        tend, data = set_physics_flags(state, data, params, forcing, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyForcing(SpeedyTermBase):
    """Sets time-varying boundary conditions (albedo, CO2, ozone, etc.)."""

    name: ClassVar[str] = "speedy_forcing"
    category: ClassVar[str] = "forcing"

    def __init__(
        self, mod_radcon_params: ModRadConParameters | None = None,
    ):
        """Initialize SpeedyForcing."""
        super().__init__()
        self.mod_radcon_params = nnx.Param(
            mod_radcon_params or ModRadConParameters.default()
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(
            mod_radcon=self.mod_radcon_params.get_value(),
        )

        from jcm.physics.forcing.speedy_forcing import set_forcing
        tend, data = set_forcing(state, data, params, forcing, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        # Downstream terms read the current-step forcing slice off this
        # diagnostic key.
        diagnostics["_forcing_2d"] = forcing
        return tend, diagnostics


class SpeedyHumidity(SpeedyTermBase):
    """Converts specific humidity to relative humidity."""

    name: ClassVar[str] = "speedy_humidity"
    category: ClassVar[str] = "humidity"
    provides: ClassVar[tuple[str, ...]] = ("_humidity",)

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = Parameters.default()

        from jcm.physics.clouds.speedy_humidity import spec_hum_to_rel_hum
        tend, data = spec_hum_to_rel_hum(state, data, params, forcing, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyConvection(SpeedyTermBase):
    """Mass-flux convection scheme."""

    name: ClassVar[str] = "speedy_convection"
    category: ClassVar[str] = "convection"

    def __init__(
        self, convection_params: ConvectionParameters | None = None,
    ):
        """Initialize SpeedyConvection."""
        super().__init__()
        self.params = nnx.Param(convection_params or ConvectionParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(convection=self.params.get_value())

        from jcm.physics.convection.speedy_convection import get_convection_tendencies
        tend, data = get_convection_tendencies(state, data, params, forcing, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyLargeScaleCondensation(SpeedyTermBase):
    """Large-scale condensation and precipitation."""

    name: ClassVar[str] = "speedy_large_scale_condensation"
    category: ClassVar[str] = "condensation"

    def __init__(
        self, condensation_params: CondensationParameters | None = None,
    ):
        """Initialize SpeedyLargeScaleCondensation."""
        super().__init__()
        self.params = nnx.Param(condensation_params or CondensationParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(condensation=self.params.get_value())

        from jcm.physics.clouds.speedy_condensation import (
            get_large_scale_condensation_tendencies,
        )
        tend, data = get_large_scale_condensation_tendencies(
            state, data, params, forcing, terrain,
        )

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyClouds(SpeedyTermBase):
    """Cloud diagnostics for radiation."""

    name: ClassVar[str] = "speedy_clouds"
    category: ClassVar[str] = "clouds"

    def __init__(
        self, sw_params: ShortwaveRadiationParameters | None = None,
    ):
        """Initialize SpeedyClouds."""
        super().__init__()
        self.params = nnx.Param(sw_params or ShortwaveRadiationParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(shortwave_radiation=self.params.get_value())

        from jcm.physics.radiation.speedy_shortwave import get_clouds
        tend, data = get_clouds(state, data, params, forcing, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyShortwaveRadiation(SpeedyTermBase):
    """Shortwave radiation fluxes and heating rates."""

    name: ClassVar[str] = "speedy_shortwave_radiation"
    category: ClassVar[str] = "radiation_sw"

    def __init__(
        self,
        sw_params: ShortwaveRadiationParameters | None = None,
        mod_radcon_params: ModRadConParameters | None = None,
    ):
        """Initialize SpeedyShortwaveRadiation."""
        super().__init__()
        self.sw_params = nnx.Param(
            sw_params or ShortwaveRadiationParameters.default()
        )
        self.mod_radcon_params = nnx.Param(
            mod_radcon_params or ModRadConParameters.default()
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(
            shortwave_radiation=self.sw_params.get_value(),
            mod_radcon=self.mod_radcon_params.get_value(),
        )

        from jcm.physics.radiation.speedy_shortwave import (
            get_shortwave_rad_fluxes,
        )
        tend, data = get_shortwave_rad_fluxes(state, data, params, forcing, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyDownwardLongwaveRadiation(SpeedyTermBase):
    """Downward longwave radiation fluxes."""

    name: ClassVar[str] = "speedy_downward_longwave"
    category: ClassVar[str] = "radiation_lw_down"

    def __init__(
        self, mod_radcon_params: ModRadConParameters | None = None,
    ):
        """Initialize SpeedyDownwardLongwaveRadiation."""
        super().__init__()
        self.mod_radcon_params = nnx.Param(
            mod_radcon_params or ModRadConParameters.default()
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(
            mod_radcon=self.mod_radcon_params.get_value(),
        )

        from jcm.physics.radiation.speedy_longwave import (
            get_downward_longwave_rad_fluxes,
        )
        tend, data = get_downward_longwave_rad_fluxes(
            state, data, params, forcing, terrain,
        )

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedySurfaceFlux(SpeedyTermBase):
    """Surface exchange of momentum, heat, and moisture."""

    name: ClassVar[str] = "speedy_surface_flux"
    category: ClassVar[str] = "surface"

    def __init__(
        self,
        surface_params: SurfaceFluxParameters | None = None,
        mod_radcon_params: ModRadConParameters | None = None,
    ):
        """Initialize SpeedySurfaceFlux."""
        super().__init__()
        # SurfaceFluxParameters contains bools — use Variable
        # for non-differentiable parts.
        self.surface_params = nnx.Variable(
            surface_params or SurfaceFluxParameters.default()
        )
        self.mod_radcon_params = nnx.Param(
            mod_radcon_params or ModRadConParameters.default()
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(
            surface_flux=self.surface_params.get_value(),
            mod_radcon=self.mod_radcon_params.get_value(),
        )

        # Use the day-sliced forcing computed by SpeedyForcing
        forcing_2d = diagnostics.get("_forcing_2d", forcing)

        from jcm.physics.surface.speedy_surface_flux import get_surface_fluxes
        tend, data = get_surface_fluxes(state, data, params, forcing_2d, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyUpwardLongwaveRadiation(SpeedyTermBase):
    """Upward longwave radiation and final radiative heating."""

    name: ClassVar[str] = "speedy_upward_longwave"
    category: ClassVar[str] = "radiation_lw_up"

    def __init__(
        self, mod_radcon_params: ModRadConParameters | None = None,
    ):
        """Initialize SpeedyUpwardLongwaveRadiation."""
        super().__init__()
        self.mod_radcon_params = nnx.Param(
            mod_radcon_params or ModRadConParameters.default()
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(
            mod_radcon=self.mod_radcon_params.get_value(),
        )

        from jcm.physics.radiation.speedy_longwave import (
            get_upward_longwave_rad_fluxes,
        )
        tend, data = get_upward_longwave_rad_fluxes(
            state, data, params, forcing, terrain,
        )

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedyVerticalDiffusion(SpeedyTermBase):
    """Vertical diffusion, shallow convection, and super-adiabatic damping."""

    name: ClassVar[str] = "speedy_vertical_diffusion"
    category: ClassVar[str] = "vertical_diffusion"

    def __init__(
        self, vdiff_params: VerticalDiffusionParameters | None = None,
    ):
        """Initialize SpeedyVerticalDiffusion."""
        super().__init__()
        self.params = nnx.Param(vdiff_params or VerticalDiffusionParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(vertical_diffusion=self.params.get_value())

        from jcm.physics.vertical_diffusion.speedy_vdiff import get_vertical_diffusion_tend
        tend, data = get_vertical_diffusion_tend(state, data, params, forcing, terrain)

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def speedy_physics(parameters: Parameters | None = None, checkpoint_terms: bool = True):
    """Create a ComposablePhysics with the standard SPEEDY term ordering.

    Args:
        parameters: Optional Parameters struct. Uses defaults if None.
        checkpoint_terms: Whether to checkpoint terms for memory efficiency.

    Returns:
        A ComposablePhysics instance with all SPEEDY terms.

    """
    from jcm.physics.composable_physics import ComposablePhysics

    p = parameters or Parameters.default()

    return ComposablePhysics(
        terms=[
            SpeedyFlags(),
            SpeedyForcing(
                mod_radcon_params=p.mod_radcon,
            ),
            SpeedyHumidity(),
            SpeedyConvection(convection_params=p.convection),
            SpeedyLargeScaleCondensation(condensation_params=p.condensation),
            SpeedyClouds(sw_params=p.shortwave_radiation),
            SpeedyShortwaveRadiation(
                sw_params=p.shortwave_radiation,
                mod_radcon_params=p.mod_radcon,
            ),
            SpeedyDownwardLongwaveRadiation(mod_radcon_params=p.mod_radcon),
            SpeedySurfaceFlux(
                surface_params=p.surface_flux,
                mod_radcon_params=p.mod_radcon,
            ),
            SpeedyUpwardLongwaveRadiation(mod_radcon_params=p.mod_radcon),
            SpeedyVerticalDiffusion(vdiff_params=p.vertical_diffusion),
        ],
        checkpoint_terms=checkpoint_terms,
    )
