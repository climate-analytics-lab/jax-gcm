"""PhysicsTerm wrappers for existing SPEEDY physics functions.

Each wrapper delegates to the original SPEEDY function, translating between
the composable ``diagnostics`` dict and the legacy typed ``PhysicsData``
struct. The wrappers also own SPEEDY's humidity-unit boundary: public
``PhysicsState`` values are kg/kg, while the translated routines retain their
native g/kg arithmetic. The numerical implementation is untouched.

"""

from __future__ import annotations

import functools
from importlib import resources
from typing import ClassVar

from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.surface.surface_exchange import (
    SURFACE_EXCHANGE_KEY,
    SURFACE_EXCHANGE_OUTPUT_ATTRS,
    SurfaceExchange,
)
from jcm.physics.surface.prescribed_flux import (
    PRESCRIBED_FLUX_FORCING_FIELDS,
    check_prescribed_flux_forcing,
)
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

#: Units and descriptions of every diagnostic the SPEEDY terms publish.
SPEEDY_UNITS_TABLE_CSV_PATH = resources.files('jcm.physics.speedy') / 'units_table.csv'

_KG_PER_KG_TO_G_PER_KG = 1000.0


def _state_in_legacy_speedy_units(state: PhysicsState) -> PhysicsState:
    """Return ``state`` with only specific humidity converted to g/kg.

    SPEEDY's thermodynamic constants, thresholds, and precipitation budgets
    were translated in their original g/kg convention. Keeping the conversion
    here makes the public physics contract unambiguous without perturbing those
    validated calculations. Additional tracers are deliberately untouched;
    their contracts are declared independently by ``TracerSpec``.
    """
    return state.copy(
        specific_humidity=(
            state.specific_humidity * _KG_PER_KG_TO_G_PER_KG
        ),
    )


def _tendency_from_legacy_speedy_units(
    tendency: PhysicsTendency,
) -> PhysicsTendency:
    """Convert only SPEEDY's humidity tendency from g/kg/s to kg/kg/s."""
    return tendency.copy(
        specific_humidity=(
            tendency.specific_humidity / _KG_PER_KG_TO_G_PER_KG
        ),
    )


def _call_legacy_speedy(routine, state: PhysicsState, *args):
    """Call one translated SPEEDY routine across the humidity-unit boundary."""
    tendency, data = routine(_state_in_legacy_speedy_units(state), *args)
    return _tendency_from_legacy_speedy_units(tendency), data


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

    UNITS_TABLE_CSV_PATH: ClassVar = SPEEDY_UNITS_TABLE_CSV_PATH

    def __init__(self):
        """Initialize SpeedyTermBase."""
        # Placeholder — populated by cache_coords
        self._coords_cached = False

    def preferred_advection(self) -> str:
        """SPEEDY runs on the Eulerian spectral core.

        SPEEDY was formulated and tuned on an Eulerian spectral dycore and
        carries no extra tracers (``specific_humidity`` is modal under either
        scheme), so semi-Lagrangian transport buys it nothing — while on CPU
        it costs ~4x the whole step at T31L8. A SPEEDY composition that adds
        tracers still resolves to semi-Lagrangian (#521).
        See docs/source/design/dinosaur_transport_selection.md.
        """
        return "eulerian"

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

        tend, data = _call_legacy_speedy(
            set_physics_flags, state, data, params, forcing, terrain,
        )

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
        tend, data = _call_legacy_speedy(
            set_forcing, state, data, params, forcing, terrain,
        )

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
        tend, data = _call_legacy_speedy(
            spec_hum_to_rel_hum, state, data, params, forcing, terrain,
        )

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
        tend, data = _call_legacy_speedy(
            get_convection_tendencies, state, data, params, forcing, terrain,
        )

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
        tend, data = _call_legacy_speedy(
            get_large_scale_condensation_tendencies,
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
        tend, data = _call_legacy_speedy(
            get_clouds, state, data, params, forcing, terrain,
        )

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
        tend, data = _call_legacy_speedy(
            get_shortwave_rad_fluxes, state, data, params, forcing, terrain,
        )

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
        tend, data = _call_legacy_speedy(
            get_downward_longwave_rad_fluxes,
            state, data, params, forcing, terrain,
        )

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


class SpeedySurfaceFlux(SpeedyTermBase):
    """Surface exchange of momentum, heat, and moisture.

    Besides the SPEEDY-internal ``_surface_flux`` diagnostics, this term
    publishes the package-independent :class:`~jcm.physics.surface.
    surface_exchange.SurfaceExchange` coupling struct (#754). It sits here —
    not in a separate publisher term — because at this point in SPEEDY's
    fixed ordering everything the contract needs is already on the
    diagnostics dict (convective + large-scale precipitation, downward
    radiation) and the fluxes are published in the very step they are
    delivered, from the same ``merged`` values the tendencies use.

    With ``prescribed_fluxes=True`` the term runs in forced mode (#301):
    the turbulent fluxes are read from the ``prescribed_*`` fields of
    :class:`~jcm.forcing.ForcingData` instead of the bulk formulae — see
    :func:`jcm.physics.surface.speedy_surface_flux.get_surface_fluxes`.
    """

    name: ClassVar[str] = "speedy_surface_flux"
    category: ClassVar[str] = "surface"
    # Literal string (== SURFACE_EXCHANGE_KEY) so the requires-audit's AST
    # walk can evaluate the tuple.
    provides: ClassVar[tuple[str, ...]] = ("surface_exchange",)
    output_attrs: ClassVar = SURFACE_EXCHANGE_OUTPUT_ATTRS

    def __init__(
        self,
        surface_params: SurfaceFluxParameters | None = None,
        mod_radcon_params: ModRadConParameters | None = None,
        prescribed_fluxes: bool = False,
    ):
        """Initialize SpeedySurfaceFlux.

        Args:
            surface_params: SPEEDY surface-flux parameters.
            mod_radcon_params: SPEEDY radiation/convection shared parameters.
            prescribed_fluxes: Static flag selecting forced mode (#301):
                replace the bulk-formula turbulent fluxes with the
                ``prescribed_*`` fields of the run's ``ForcingData``.

        """
        super().__init__()
        # SurfaceFluxParameters contains bools — use Variable
        # for non-differentiable parts.
        self.surface_params = nnx.Variable(
            surface_params or SurfaceFluxParameters.default()
        )
        self.mod_radcon_params = nnx.Param(
            mod_radcon_params or ModRadConParameters.default()
        )
        self.prescribed_fluxes = prescribed_fluxes

    def stable_time_step_minutes(self, coords) -> float | None:
        """Forward-Euler stability limit of the explicit surface drag.

        This term applies the surface stress as an explicit tendency
        ``du/dt = ustr * g / (dsigma_bot * p0)`` in the lowest layer, so its
        stability limit shrinks with the bottom sigma-layer thickness (and
        with truncation, through the resolved near-surface wind) — see
        :func:`jcm.physics.speedy.physical_constants.stable_time_step_from_geometry`
        and docs/source/design/speedy_variable_levels.md. The thickness is read off the *actual*
        coordinate system, so custom sigma spacings are handled, not just the
        standard tables. Returns ``None`` (no constraint) when the grid does
        not expose sigma boundaries / a spectral truncation.
        """
        from jcm.physics.speedy.physical_constants import (
            stable_time_step_from_geometry,
        )
        try:
            boundaries = coords.vertical.boundaries
            truncation = int(coords.horizontal.longitude_wavenumbers) - 1
        except AttributeError:
            return None
        dsigma_bottom = float(boundaries[-1]) - float(boundaries[-2])
        if dsigma_bottom <= 0.0:
            return None
        return stable_time_step_from_geometry(dsigma_bottom, truncation)

    def augment_probe_forcing(self, forcing):
        """Seed the shape probe's ``prescribed_*`` fields with zeros.

        In forced mode the probe (``get_empty_data``) must trace the
        prescribed-flux code path, not the ``None`` fallback. Fill the four
        fields with zero maps sized off an always-present forcing leaf so
        the abstract trace matches a live step; the real run's validation
        lives in :meth:`validate_forcing`.
        """
        if not self.prescribed_fluxes:
            return forcing
        zeros = jnp.zeros_like(forcing.stl_am)
        return forcing.copy(
            prescribed_sensible_heat_flux=zeros,
            prescribed_evaporation=zeros,
            prescribed_stress_u=zeros,
            prescribed_stress_v=zeros,
        )

    def consumed_forcing_fields(self):
        """Return the ``prescribed_*`` fields in forced mode, nothing interactively."""
        return PRESCRIBED_FLUX_FORCING_FIELDS if self.prescribed_fluxes else ()

    def validate_forcing(self, forcing, run_window=None):
        """Fail loudly at run start if forced mode lacks its forcing fields
        or a date-aligned archive does not cover the run window.
        """
        if not self.prescribed_fluxes:
            return
        check_prescribed_flux_forcing(
            forcing, "SpeedySurfaceFlux(prescribed_fluxes=True)", run_window)

    def __call__(self, state, diagnostics, forcing, terrain):
        data = self._build_data(diagnostics)
        params = _params_with(
            surface_flux=self.surface_params.get_value(),
            mod_radcon=self.mod_radcon_params.get_value(),
        )

        # Use the day-sliced forcing computed by SpeedyForcing
        forcing_2d = diagnostics.get("_forcing_2d", forcing)

        from jcm.physics.surface.speedy_surface_flux import (
            PrescribedFluxes,
            get_surface_fluxes,
        )
        routine = get_surface_fluxes
        if self.prescribed_fluxes:
            # A missing field is caught before the run by ``validate_forcing``;
            # here (which also runs under the abstract shape probe) fall back
            # to a zero map so the trace stays well-defined.
            zeros = jnp.zeros_like(state.temperature[-1])

            def _prescribed(name):
                value = getattr(forcing_2d, name)
                return zeros if value is None else value

            prescribed = PrescribedFluxes(
                sensible_heat_flux=_prescribed("prescribed_sensible_heat_flux"),
                evaporation=_prescribed("prescribed_evaporation"),
                stress_u=_prescribed("prescribed_stress_u"),
                stress_v=_prescribed("prescribed_stress_v"),
            )
            routine = functools.partial(
                get_surface_fluxes, prescribed=prescribed)
        tend, data = _call_legacy_speedy(
            routine, state, data, params, forcing_2d, terrain,
        )

        diagnostics = _diagnostics_from_data(diagnostics, data)
        diagnostics[SURFACE_EXCHANGE_KEY] = self._publish_surface_exchange(
            state, data)
        return tend, diagnostics

    def _publish_surface_exchange(self, state, data) -> SurfaceExchange:
        """Fill the #754 coupling contract from SPEEDY's delivered fluxes.

        Everything comes from the same ``surface_flux`` grid means the
        bottom-level tendencies were just computed from (so published ==
        delivered, in interactive and forced mode alike), plus the
        convective/large-scale precipitation already on the diagnostics.
        Unit and sign normalisation to the contract happens here and only
        here: evaporation/precipitation g -> kg/m2/s, latent heat via
        SPEEDY's own ``alhc`` [J/g], stress negated from "on the
        atmosphere" to "into the surface". SPEEDY has no faithful
        rain/snow split and no per-tile delivered fluxes (the sea tile
        blends the ice *temperature*, not fluxes), so the optional
        contract fields stay ``None`` — see
        docs/source/design/surface_exchange.md.
        """
        import jcm.constants as c
        from jcm.physics.speedy.physical_constants import alhc

        sf = data.surface_flux
        g_to_kg = 1.0 / _KG_PER_KG_TO_G_PER_KG
        # Lowest-model-level thermodynamics for the external bulk-flux
        # algorithms (#301 discussion): moist density and potential
        # temperature at the full-level pressure sigma_bot * p0 * psa.
        fsg_bot = data.speedy_coords.fsg[-1]
        psa = state.normalized_surface_pressure
        p_bot = fsg_bot * c.p0 * psa
        t_bot = state.temperature[-1]
        q_bot = state.specific_humidity[-1]  # kg/kg on the public state
        # Gradient-safe near-surface wind speed: at exactly zero wind (the
        # SPEEDY default initial state is at rest) ``d/dx sqrt(x)`` is
        # infinite, so a bare ``sqrt(u0**2+v0**2)`` poisons reverse-mode with
        # a 0*inf -> NaN the moment ``surface_exchange`` is in the
        # differentiated output. The double-``where`` keeps the value exact
        # and the derivative finite (zero) at the origin — the standard
        # sqrt-at-zero JAX idiom (see JAX_gotchas.md / gradient_nan_hardening).
        wind_sq = sf.u0 ** 2 + sf.v0 ** 2
        wind_speed = jnp.where(
            wind_sq > 0.0, jnp.sqrt(jnp.where(wind_sq > 0.0, wind_sq, 1.0)), 0.0)
        return SurfaceExchange(
            net_heat_flux=sf.hfluxn,
            sensible_heat_flux=sf.shf,
            latent_heat_flux=alhc * sf.evap,
            evaporation=sf.evap * g_to_kg,
            precipitation=(data.convection.precnv
                           + data.condensation.precls) * g_to_kg,
            stress_u=-sf.ustr,
            stress_v=-sf.vstr,
            wind_speed=wind_speed,
            air_density=p_bot / (c.rd * t_bot * (1.0 + c.vtmpc1 * q_bot)),
            air_potential_temperature=t_bot * (1.0 / (fsg_bot * psa)) ** c.akap,
        )


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
        tend, data = _call_legacy_speedy(
            get_upward_longwave_rad_fluxes,
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
        tend, data = _call_legacy_speedy(
            get_vertical_diffusion_tend,
            state, data, params, forcing, terrain,
        )

        diagnostics = _diagnostics_from_data(diagnostics, data)
        return tend, diagnostics


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def speedy_physics(parameters: Parameters | None = None, checkpoint_terms: bool = True,
                   diagnose_omega: bool = False):
    """Create a ComposablePhysics with the standard SPEEDY term ordering.

    Args:
        parameters: Optional Parameters struct. Uses defaults if None.
        checkpoint_terms: Whether to checkpoint terms for memory efficiency.
        diagnose_omega: Append the :class:`~jcm.physics.diagnostics.omega.
            OmegaDiagnostic` term, publishing the dycore's pressure
            vertical velocity [Pa/s] as an ``omega`` output field. Needs
            ``DinosaurDycore(compute_omega=True)`` (Model construction
            fails with a pointed error otherwise; the CLI enables the
            provider automatically).

    Returns:
        A ComposablePhysics instance with all SPEEDY terms.

    """
    from jcm.physics.composable_physics import ComposablePhysics

    p = parameters or Parameters.default()

    omega_terms = []
    if diagnose_omega:
        from jcm.physics.diagnostics.omega import OmegaDiagnostic
        omega_terms = [OmegaDiagnostic()]

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
            *omega_terms,
        ],
        checkpoint_terms=checkpoint_terms,
    )
