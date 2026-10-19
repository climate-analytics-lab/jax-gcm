"""Main surface physics interface for ECHAM surface scheme.

This module provides the main interface for surface physics calculations,
coordinating different surface types (ocean, sea ice, land) and computing
grid-box mean fluxes.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c

from .surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing,
    SurfaceFluxes, SurfaceTendencies, SurfaceDiagnostics
)
from .turbulent_fluxes import (
    compute_bulk_richardson_number,
    compute_surface_resistances, compute_surface_diagnostics
)
from .ocean import ocean_physics_step
from .sea_ice import sea_ice_physics_step
from .land import land_surface_physics_step
from .albedo import CTFREEZ


def initialize_surface_state(
    ncol: int,
    surface_fractions: jnp.ndarray,
    ocean_temp: jnp.ndarray,
    ice_temp: jnp.ndarray,
    soil_temp: jnp.ndarray,
    params: SurfaceParameters | None = None,
    soil_moisture: jnp.ndarray | None = None,
) -> SurfaceState:
    """Initialize surface state from basic inputs.

    Args:
        ncol: Number of columns
        surface_fractions: Surface type fractions [water, ice, land] (ncol, 3)
        ocean_temp: Ocean temperature [K] (ncol,)
        ice_temp: Ice temperature [K] (ncol, nice_layers)
        soil_temp: Soil temperature [K] (ncol, nsoil_layers)
        params: Surface parameters
        soil_moisture: Soil moisture [-] (ncol, nsoil_layers). ``None``
            (the default, every pre-existing caller) fills a flat 0.3 —
            a synthetic placeholder, since this module has no prognostic
            soil-moisture state of its own. ``EchamSurface`` passes a
            live-derived value so its published bulk-flux estimate reads
            this step's actual land wetness rather than the placeholder
            (jax-gcm#1118 Codex review).

    Returns:
        Initialized surface state

    """
    if params is None:  # not a def default: it would build jax arrays at import (#859)
        params = SurfaceParameters.default()
    # Use fixed value for nsfc_type since it needs to be concrete for array creation
    nsfc_type = 3  # Always 3: water, ice, land
    nice_layers = 2  # Default ice layers
    nsoil_layers = 4  # Default soil layers
    
    # Surface temperatures (combine different surface types)
    surface_temp = jnp.zeros((ncol, nsfc_type))
    surface_temp = surface_temp.at[:, params.iwtr].set(ocean_temp)
    surface_temp = surface_temp.at[:, params.iice].set(ice_temp[:, 0])  # Top ice layer
    surface_temp = surface_temp.at[:, params.ilnd].set(soil_temp[:, 0])  # Top soil layer
    
    # Radiative temperature (area-weighted mean)
    temp_rad = jnp.sum(surface_fractions * surface_temp, axis=1)
    
    # Ocean variables
    ocean_u = jnp.zeros(ncol)
    ocean_v = jnp.zeros(ncol)
    
    # Ice variables
    ice_thickness = jnp.full((ncol, nice_layers), 2.0)  # 2m default thickness
    snow_depth = jnp.zeros(ncol)
    
    # Soil variables
    if soil_moisture is None:
        soil_moisture = jnp.full((ncol, nsoil_layers), 0.3)  # 30% default moisture
    vegetation_temp = soil_temp[:, 0]  # Same as surface
    
    # Default surface properties
    roughness_momentum = jnp.zeros((ncol, nsfc_type))
    roughness_momentum = roughness_momentum.at[:, params.iwtr].set(params.z0_water)
    roughness_momentum = roughness_momentum.at[:, params.iice].set(params.z0_ice)
    roughness_momentum = roughness_momentum.at[:, params.ilnd].set(params.z0_land)
    
    roughness_heat = roughness_momentum * 0.1  # Typical ratio
    
    # Default albedos
    albedo_vis_direct = jnp.zeros((ncol, nsfc_type))
    albedo_vis_diffuse = jnp.zeros((ncol, nsfc_type))
    albedo_nir_direct = jnp.zeros((ncol, nsfc_type))
    albedo_nir_diffuse = jnp.zeros((ncol, nsfc_type))
    
    # Ocean albedo
    albedo_vis_direct = albedo_vis_direct.at[:, params.iwtr].set(0.06)
    albedo_vis_diffuse = albedo_vis_diffuse.at[:, params.iwtr].set(0.06)
    albedo_nir_direct = albedo_nir_direct.at[:, params.iwtr].set(0.06)
    albedo_nir_diffuse = albedo_nir_diffuse.at[:, params.iwtr].set(0.06)
    
    # Ice albedo
    albedo_vis_direct = albedo_vis_direct.at[:, params.iice].set(0.75)
    albedo_vis_diffuse = albedo_vis_diffuse.at[:, params.iice].set(0.75)
    albedo_nir_direct = albedo_nir_direct.at[:, params.iice].set(0.65)
    albedo_nir_diffuse = albedo_nir_diffuse.at[:, params.iice].set(0.65)
    
    # Land albedo
    albedo_vis_direct = albedo_vis_direct.at[:, params.ilnd].set(0.15)
    albedo_vis_diffuse = albedo_vis_diffuse.at[:, params.ilnd].set(0.15)
    albedo_nir_direct = albedo_nir_direct.at[:, params.ilnd].set(0.30)
    albedo_nir_diffuse = albedo_nir_diffuse.at[:, params.ilnd].set(0.30)
    
    return SurfaceState(
        temperature=surface_temp,
        temperature_rad=temp_rad,
        fraction=surface_fractions,
        ocean_temp=ocean_temp,
        ocean_u=ocean_u,
        ocean_v=ocean_v,
        ice_thickness=ice_thickness,
        ice_temp=ice_temp,
        snow_depth=snow_depth,
        soil_temp=soil_temp,
        soil_moisture=soil_moisture,
        vegetation_temp=vegetation_temp,
        roughness_momentum=roughness_momentum,
        roughness_heat=roughness_heat,
        albedo_visible_direct=albedo_vis_direct,
        albedo_visible_diffuse=albedo_vis_diffuse,
        albedo_nir_direct=albedo_nir_direct,
        albedo_nir_diffuse=albedo_nir_diffuse
    )


@jax.jit
def surface_physics_step(
    atmospheric_state: AtmosphericForcing,
    surface_state: SurfaceState,
    dt: float,
    params: SurfaceParameters | None = None,
    vegetation_fraction: jnp.ndarray | None = None,
    land_cair: jnp.ndarray | None = None,
    land_csat: jnp.ndarray | None = None,
) -> Tuple[SurfaceFluxes, SurfaceTendencies, SurfaceDiagnostics]:
    """Complete surface physics step for all surface types.

    Args:
        atmospheric_state: Atmospheric forcing
        surface_state: Surface state
        dt: Time step [s]
        params: Surface parameters
        vegetation_fraction: Vegetated land fraction [-] (ncol,). ``None``
            (the default, every pre-existing caller) fills a flat 0.5 — a
            synthetic placeholder, since this module reads no vegetation
            boundary condition of its own. ``EchamSurface`` passes the live
            ``forcing.forest_fraction`` so its published bulk-flux estimate
            reads this step's actual vegetated fraction rather than the
            placeholder (jax-gcm#1118 Codex review).
        land_cair: JSBACH land-tile humidity factor of the air [1] (ncol,),
            forwarded to :func:`compute_surface_diagnostics`. ``None`` (the
            default) is a fully-saturated boundary; ``EchamSurface`` passes
            the live, same-step ``SurfaceData.cair`` (jax-gcm#1140 review).
        land_csat: JSBACH land-tile humidity factor of the surface
            saturation [1] (ncol,), same convention as ``land_cair``.

    Returns:
        Tuple of (surface_fluxes, tendencies, diagnostics)

    """
    if params is None:  # not a def default: it would build jax arrays at import (#859)
        params = SurfaceParameters.default()
    ncol, nsfc_type = surface_state.temperature.shape
    
    # Compute bulk Richardson number
    surface_humidity = jnp.full_like(surface_state.temperature, 0.01)  # Simplified
    wind_speed = jnp.sqrt(jnp.maximum(atmospheric_state.u_wind**2 + atmospheric_state.v_wind**2, 1.0e-30))
    
    ri_bulk = compute_bulk_richardson_number(
        atmospheric_state.temperature, surface_state.temperature,
        atmospheric_state.humidity, surface_humidity, wind_speed
    )
    
    # Surface exchange velocities (CH·|U|, CE·|U|, CM·|U|, all m/s) come from
    # the configured surface-layer scheme upstream (TteTkeVerticalDiffusion,
    # ECHAM-Louis by default), threaded in via ``atmospheric_state``. Using
    # those here — rather than recomputing a separate bulk-Richardson scheme —
    # makes the flux and the vdiff-side coefficient one and the same, mirroring
    # ECHAM's single ``sfc_exchange_coeff``. The standalone bulk-Richardson
    # routines in this module (``compute_bulk_richardson_number`` /
    # ``compute_stability_functions`` / ``compute_exchange_coefficients``)
    # remain available and tested as a surface-side reference scheme, but are
    # no longer in the default flux path. ``ri_bulk`` above is still used for
    # the aerodynamic-resistance diagnostics below.
    exchange_coeff_momentum = atmospheric_state.exchange_coeff_momentum
    exchange_coeff_heat = atmospheric_state.exchange_coeff_heat
    exchange_coeff_moisture = atmospheric_state.exchange_coeff_moisture
    
    # Initialize output arrays
    all_fluxes = []
    all_tendencies = []
    all_roughness = []
    
    # Process each surface type using explicit unrolled loop with JAX-compatible conditionals
    # This avoids issues with jax.lax.switch requiring identical return types
    
    # Ocean surface (isfc = 0)
    fluxes_ocean, tendencies_ocean, roughness_ocean = ocean_physics_step(
        atmospheric_state,
        surface_state.ocean_temp,
        surface_state.ocean_u,
        surface_state.ocean_v,
        exchange_coeff_heat[:, params.iwtr],  # iwtr = 0
        exchange_coeff_moisture[:, params.iwtr],
        exchange_coeff_momentum[:, params.iwtr],
        jnp.zeros(ncol),  # Solar zenith angle (simplified)
        dt, params
    )
    all_fluxes.append(fluxes_ocean)
    all_tendencies.append(tendencies_ocean)
    all_roughness.append(roughness_ocean)
    
    # Ice surface (isfc = 1)
    fluxes_ice, tendencies_ice, roughness_ice = sea_ice_physics_step(
        atmospheric_state,
        surface_state.ice_temp,
        surface_state.ice_thickness,
        surface_state.snow_depth,
        surface_state.ocean_temp,
        exchange_coeff_heat[:, params.iice],  # iice = 1
        exchange_coeff_moisture[:, params.iice],
        exchange_coeff_momentum[:, params.iice],
        dt, params
    )
    all_fluxes.append(fluxes_ice)
    all_tendencies.append(tendencies_ice)
    all_roughness.append(roughness_ice)
    
    # Land surface (isfc = 2)
    if vegetation_fraction is None:
        vegetation_fraction = jnp.full(ncol, 0.5)  # 50% vegetation (default)
    soil_depths = jnp.array([0.1, 0.3, 0.6, 1.0])  # Soil layer depths
    
    fluxes_land, tendencies_land, roughness_land = land_surface_physics_step(
        atmospheric_state,
        surface_state.soil_temp,
        surface_state.soil_moisture,
        surface_state.vegetation_temp,
        surface_state.snow_depth,
        exchange_coeff_heat[:, params.ilnd],  # ilnd = 2
        exchange_coeff_moisture[:, params.ilnd],
        exchange_coeff_momentum[:, params.ilnd],
        vegetation_fraction,
        soil_depths,
        dt, params
    )
    all_fluxes.append(fluxes_land)
    all_tendencies.append(tendencies_land)
    all_roughness.append(roughness_land)
    
    # Combine fluxes from all surface types
    combined_fluxes = combine_surface_fluxes(all_fluxes, surface_state.fraction)
    
    # Combine tendencies
    combined_tendencies = combine_surface_tendencies(all_tendencies, surface_state.fraction)
    
    # Compute surface resistances
    resistances = compute_surface_resistances(
        atmospheric_state, surface_state, ri_bulk, params
    )
    
    # Compute diagnostics
    diagnostics = compute_surface_diagnostics(
        atmospheric_state, surface_state, combined_fluxes, resistances,
        params, land_cair=land_cair, land_csat=land_csat,
    )
    
    return combined_fluxes, combined_tendencies, diagnostics


@jax.jit
def combine_surface_fluxes(
    flux_list: list,
    fractions: jnp.ndarray
) -> SurfaceFluxes:
    """Combine fluxes from different surface types into grid-box means.
    
    Args:
        flux_list: List of SurfaceFluxes for each surface type
        fractions: Surface type fractions (ncol, nsfc_type)
        
    Returns:
        Combined surface fluxes

    """
    ncol, nsfc_type = fractions.shape
    
    # Stack fluxes from all surface types
    sensible_heat = jnp.stack([flux.sensible_heat[:, 0] for flux in flux_list], axis=1)
    latent_heat = jnp.stack([flux.latent_heat[:, 0] for flux in flux_list], axis=1)
    longwave_net = jnp.stack([flux.longwave_net[:, 0] for flux in flux_list], axis=1)
    shortwave_net = jnp.stack([flux.shortwave_net[:, 0] for flux in flux_list], axis=1)
    ground_heat = jnp.stack([flux.ground_heat[:, 0] for flux in flux_list], axis=1)
    momentum_u = jnp.stack([flux.momentum_u[:, 0] for flux in flux_list], axis=1)
    momentum_v = jnp.stack([flux.momentum_v[:, 0] for flux in flux_list], axis=1)
    evaporation = jnp.stack([flux.evaporation[:, 0] for flux in flux_list], axis=1)
    transpiration = jnp.stack([flux.transpiration[:, 0] for flux in flux_list], axis=1)
    
    # Compute grid-box means
    sensible_mean = jnp.sum(fractions * sensible_heat, axis=1)
    latent_mean = jnp.sum(fractions * latent_heat, axis=1)
    momentum_u_mean = jnp.sum(fractions * momentum_u, axis=1)
    momentum_v_mean = jnp.sum(fractions * momentum_v, axis=1)
    evaporation_mean = jnp.sum(fractions * (evaporation + transpiration), axis=1)
    
    return SurfaceFluxes(
        sensible_heat=sensible_heat,
        latent_heat=latent_heat,
        longwave_net=longwave_net,
        shortwave_net=shortwave_net,
        ground_heat=ground_heat,
        momentum_u=momentum_u,
        momentum_v=momentum_v,
        evaporation=evaporation,
        transpiration=transpiration,
        sensible_heat_mean=sensible_mean,
        latent_heat_mean=latent_mean,
        momentum_u_mean=momentum_u_mean,
        momentum_v_mean=momentum_v_mean,
        evaporation_mean=evaporation_mean
    )


@jax.jit
def combine_surface_tendencies(
    tendency_list: list,
    fractions: jnp.ndarray
) -> SurfaceTendencies:
    """Combine tendencies from different surface types.
    
    Args:
        tendency_list: List of SurfaceTendencies for each surface type
        fractions: Surface type fractions (ncol, nsfc_type)
        
    Returns:
        Combined surface tendencies

    """
    ncol, nsfc_type = fractions.shape
    
    # Extract tendencies (simplified - just use first valid tendency)
    ocean_temp_tendency = tendency_list[0].ocean_temp_tendency  # From ocean
    ice_temp_tendency = tendency_list[1].ice_temp_tendency if len(tendency_list) > 1 else jnp.zeros((ncol, 2))
    soil_temp_tendency = tendency_list[2].soil_temp_tendency if len(tendency_list) > 2 else jnp.zeros((ncol, 4))
    
    # Surface temperature tendency (area-weighted)
    surface_temp_tendency = jnp.stack([
        tendency.surface_temp_tendency[:, 0] for tendency in tendency_list
    ], axis=1)
    
    # Other tendencies
    ice_thickness_tendency = tendency_list[1].ice_thickness_tendency if len(tendency_list) > 1 else jnp.zeros((ncol, 2))
    snow_depth_tendency = jnp.zeros(ncol)  # Simplified
    soil_moisture_tendency = tendency_list[2].soil_moisture_tendency if len(tendency_list) > 2 else jnp.zeros((ncol, 4))
    
    return SurfaceTendencies(
        surface_temp_tendency=surface_temp_tendency,
        ocean_temp_tendency=ocean_temp_tendency,
        ice_temp_tendency=ice_temp_tendency,
        soil_temp_tendency=soil_temp_tendency,
        ice_thickness_tendency=ice_thickness_tendency,
        snow_depth_tendency=snow_depth_tendency,
        soil_moisture_tendency=soil_moisture_tendency
    )


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.diagnostics.moist_air_state import advance_thermo_run  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics.radiation.radiation_types import RadiationData  # noqa: E402
from jcm.physics.surface.echam.surface_types import (  # noqa: E402
    BULK_OUTPUT_ATTRS,
    DIAG_OUTPUT_ATTRS,
    LAND_OUTPUT_ATTRS,
    AtmosphericForcing,
    SurfaceData,
    SurfaceParameters,
)


def correct_surface_longwave(radiation: RadiationData, t_old, t_new, pressure_half):
    """Re-emit the cached surface longwave at the new surface temperature.

    ECHAM holds the absorbed downward longwave between radiation calls and
    emits from the current surface temperature every step, the change heating
    the lowest layer with the top of atmosphere held (``radheat.f90:404-410``),
    linearised about the old temperature as ``land_rad`` forms it
    (``zteffl4 = T_old³·(4·T_new − 3·T_old)``, ``mo_surface_land.f90:612``)::

        Δ = ε·σ·T_old³·(4·T_new − 3·T_old) − ε·σ·T_old⁴

    with ``ε`` the emissivity the radiation solved with. ``Δ`` is added to the
    surface upward flux (all-sky, and clear-sky where the scheme solved it: the
    surface emits the same in both) and the lowest-level heating, and written
    back into the radiation carry: the cached longwave then always describes
    the current surface, so successive steps telescope exactly as the
    shortwave zenith rescale does, and a radiation solve starts from the
    temperature the last correction left.

    Returns:
        ``(radiation, heating)``: the corrected carry and the lowest-level
        heating ``Δ·g/(c_pd·Δp_K)`` [K/s] to apply this step, ``(ncols,)``.

    """
    dtype = radiation.surface_lw_up.dtype
    eps = radiation.surface_emissivity.reshape(t_old.shape)
    d_up = (4.0 * eps * c.sbc * t_old ** 3 * (t_new - t_old)).astype(dtype)
    dp_k = pressure_half[-1] - pressure_half[-2]
    heating = (d_up * c.grav / (c.cpd * dp_k.reshape(d_up.shape))).astype(dtype)
    shape = radiation.surface_lw_up.shape
    # A scheme without a clear-sky solve leaves the clear-sky profile at zero.
    clear_solved = radiation.lw_flux_up_clear[-1] != 0.0
    radiation = radiation.copy(
        surface_lw_up=radiation.surface_lw_up + d_up.reshape(shape),
        lw_flux_up=radiation.lw_flux_up.at[-1].add(d_up.reshape(shape)),
        lw_flux_up_clear=radiation.lw_flux_up_clear.at[-1].add(
            jnp.where(clear_solved, d_up.reshape(shape), 0.0).astype(dtype)),
        lw_heating_rate=radiation.lw_heating_rate.at[-1].add(
            heating.reshape(radiation.lw_heating_rate.shape[1:])),
    )
    return radiation, heating
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class EchamSurface(PhysicsTerm):
    """ECHAM multi-tile (water/ice/land) surface energy-balance/diagnostics term.

    The turbulent surface fluxes are DELIVERED by the vertical-diffusion
    term: :class:`~jcm.physics.vertical_diffusion.tte_tke.TteTkeVerticalDiffusion`
    carries the surface exchange as the bottom-row Robin boundary condition
    of its implicit column solve (the ECHAM ``vdiff``/``mo_surface`` scheme,
    where one tridiagonal spans the column plus the surface exchange) and
    diagnoses the fluxes from the implicit solution, so reported flux ==
    flux received by the column, by construction. This term therefore
    returns ZERO u/v/T/qv tendencies and republishes the vdiff-delivered
    fluxes (read from ``diagnostics["vertical_diffusion"]``) as the public
    ``"surface"`` fields — ``evaporation == effective_evaporation`` now.

    It still reads exchange coefficients per tile from
    ``"vertical_diffusion"``, surface SW / LW downward fluxes from
    ``"radiation"``, sea-ice / land surface temp from forcing and ``fmask``
    from terrain, and runs :func:`surface_physics_step` for the
    albedo/radiative/tile bookkeeping; the explicit bulk fluxes that step
    computes are no longer the delivery path, but (jax-gcm#1118) are
    published as the grid-mean reference diagnostics
    ``surface.bulk_sensible_heat_flux`` / ``bulk_latent_heat_flux`` /
    ``bulk_evaporation`` / ``bulk_momentum_flux_u/v`` — an always-on check
    of how closely an explicit one-shot bulk-aerodynamic estimate tracks the
    implicit solution's delivered flux, rather than the computed-and-
    discarded value it used to be. The SAME step's ``SurfaceDiagnostics``
    bundle (jax-gcm#1140) is likewise published as ``surface.temperature_2m``
    / ``humidity_2m`` / ``dewpoint_2m`` / ``richardson_number`` /
    ``energy_balance_residual`` and the ``_tile`` pair — see
    :data:`~jcm.physics.surface.echam.surface_types.DIAG_OUTPUT_ATTRS` for
    what each means and :data:`~jcm.physics.surface.echam.surface_types.
    SurfaceData`'s field comments for the three fields of that bundle
    deliberately left unpublished (a second friction velocity, a hardcoded
    "100 m" PBL height, and a net-radiation pair that duplicates fields
    already public).

    After the vdiff's land energy balance it publishes the new grid surface
    temperature and re-emits the cached surface longwave at it
    (:func:`correct_surface_longwave`, ECHAM ``radheat``) — the one non-zero
    tendency of this term, a heating of the lowest level, which advances the
    running thermodynamic state (``thermo_run``) as every other temperature
    tendency does.
    """

    name: ClassVar[str] = "echam_surface"
    category: ClassVar[str] = "surface"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half", "height_full", "height_half",
        "vertical_diffusion", "radiation", "surface",
    )
    provides: ClassVar[tuple[str, ...]] = ("surface", "radiation")
    output_attrs: ClassVar = {
        **LAND_OUTPUT_ATTRS, **BULK_OUTPUT_ATTRS, **DIAG_OUTPUT_ATTRS,
    }

    def __init__(self, params: SurfaceParameters | None = None):
        """Hold the scheme-native :class:`SurfaceParameters`."""
        # ``SurfaceParameters`` includes a few non-floats; mirror SPEEDY's
        # SurfaceFlux convention of using ``Variable`` (not ``Param``)
        # since boolean flags don't need to be differentiated through.
        self.params = nnx.Variable(
            params if params is not None
            else SurfaceParameters.default(),
        )

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute lowest-level surface-flux tendencies and surface diagnostics."""
        # Lazy: ``echam_boundary_conditions`` imports ``jcm.physics.surface.
        # echam`` (for the albedo schemes), so a module-level import here
        # would cycle back into this package's own ``__init__`` mid-import.
        # Deferred to call time, this module is always fully loaded by then.
        from jcm.physics.forcing.echam_boundary_conditions import (
            tile_weighted_radiative_temperature,
        )

        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        height_full = diagnostics["height_full"]
        height_half = diagnostics["height_half"]

        # Read EARLY: ``TteTkeVerticalDiffusion`` runs before this term
        # (``requires`` includes ``"vertical_diffusion"``) and has already
        # written THIS step's land state onto ``diagnostics["surface"]`` —
        # the live skin temperature and JSBACH humidity factors the
        # implicit solve itself used. Needed below, before the bulk-flux
        # ``surface_state``/``surface_physics_step`` inputs are built.
        prev_surface = diagnostics.get(
            "surface", SurfaceData.zeros((ncols,), nlev),
        )

        nsfc_type = 3
        land_fraction = terrain.fmask.reshape((ncols,))
        sea_ice_fraction = jnp.clip(
            forcing.sice_am.reshape((ncols,)),
            0.0, 1.0 - land_fraction,
        )
        water_fraction = 1.0 - land_fraction - sea_ice_fraction
        surface_fractions = jnp.zeros((ncols, nsfc_type))
        surface_fractions = surface_fractions.at[:, 0].set(water_fraction)
        surface_fractions = surface_fractions.at[:, 1].set(sea_ice_fraction)
        surface_fractions = surface_fractions.at[:, 2].set(land_fraction)

        # Per-tile surface temperatures: SST for ocean, min(SST, ctfreez)
        # for ice (saline freezing point). Read straight from forcing rather
        # than ``_surface.surface_temperature`` — that field is the
        # tile-WEIGHTED T**4 mean EchamBoundaryConditions forms for radiation
        # (:func:`~jcm.physics.forcing.echam_boundary_conditions.tile_weighted_radiative_temperature`,
        # #988), a single blended grid value that is never the right
        # temperature for any one tile on its own. Neither ocean nor sea
        # ice is prognostic in this model (both are themselves derived from
        # forcing/SST every step, exactly as the implicit solve derives
        # them — see ``TteTkeVerticalDiffusion.__call__``'s ``sst_col``/
        # ``ice_temp_col``), so forcing-direct values here already ARE the
        # live state; only the land tile has a genuinely prognostic skin.
        ocean_temp = forcing.sea_surface_temperature.reshape(ncols)
        # Live land skin temperature, not a synthetic ``forcing.stl_am``
        # fallback: vdiff (same-step fresh — this term's ``requires``
        # includes ``"vertical_diffusion"``) has ALREADY written THIS
        # step's ``land_surface_temperature`` here — ``forcing.stl_am`` in
        # prescribed mode, the skin energy balance's just-solved value in
        # prognostic mode (``TteTkeVerticalDiffusion.__call__``'s
        # ``land_temp_col``) — so reading it back is correct in BOTH modes
        # without this term needing to know which is active. Before this
        # fix, the published ``surface.bulk_*`` land contribution read the
        # forcing value unconditionally, which silently diverges from the
        # implicit solve's own land state once the skin temperature evolves
        # away from ``stl_am`` (jax-gcm#1118 Codex review).
        land_temp = prev_surface.land_surface_temperature.reshape(ncols)
        ice_surface_temp = jnp.where(
            sea_ice_fraction > 0.0,
            jnp.minimum(ocean_temp, CTFREEZ),
            ocean_temp,
        )
        ice_temp = jnp.repeat(ice_surface_temp[:, jnp.newaxis], 2, axis=1)
        soil_temp = jnp.repeat(land_temp[:, jnp.newaxis], 4, axis=1)

        # Live land wetness/vegetation for the bulk-flux estimate below
        # (same Codex finding). ``land_surface_physics_step`` has no
        # cair/csat concept of its own — its evaporation efficiency is a
        # single ``soil_beta = min(soil_moisture[:, 0]/0.75, 1)`` — so the
        # closest faithful proxy within that existing formula is to drive
        # it from the SAME live JSBACH ``csat`` the implicit solve published
        # this step (``prev_surface.csat``) rather than the hardcoded 0.3
        # ``initialize_surface_state`` otherwise defaults to. This does not
        # reproduce JSBACH's ``cair`` term (the atmospheric-humidity-side
        # reduction) — ``land_surface_physics_step``'s formula has no slot
        # for it — so the land bulk estimate remains an approximation of
        # the delivered flux even with this fix, just one driven by this
        # step's actual wetness rather than an unrelated constant. Layers
        # 1-3 keep the 0.3 default: only ``soil_moisture[:, 0]`` is read
        # (``land.py::land_surface_physics_step``'s ``soil_beta``).
        csat_live = jnp.clip(prev_surface.csat.reshape(ncols), 0.0, 1.0)
        soil_moisture_land = jnp.full((ncols, 4), 0.3).at[:, 0].set(
            0.75 * csat_live)
        # The matching JSBACH ``cair`` factor (the atmospheric-humidity-side
        # reduction), threaded into ``compute_surface_diagnostics`` below so
        # the published land ``humidity_2m_tile``/grid ``humidity_2m`` use
        # the SAME boundary humidity (``csat·q_sat + (1-cair)·q_air``) the
        # live implicit surface exchange saw this step, rather than
        # anchoring the land tile at saturation like water/ice (jax-gcm#1140
        # review).
        cair_live = jnp.clip(prev_surface.cair.reshape(ncols), 0.0, 1.0)
        # Same ``veg_col`` derivation ``TteTkeVerticalDiffusion`` uses for
        # JSBACH (vegetated fraction = the forest fraction).
        vegetation_fraction_live = (
            jnp.zeros(ncols) if forcing.forest_fraction is None
            else jnp.clip(forcing.forest_fraction.reshape(ncols), 0.0, 1.0)
        )

        surface_state = initialize_surface_state(
            ncols, surface_fractions, ocean_temp, ice_temp, soil_temp,
            params, soil_moisture=soil_moisture_land,
        )

        # Lowest-model-level atmospheric inputs.
        atm_temp = state.temperature[-1, :]
        atm_qv = state.specific_humidity[-1, :]
        atm_u = state.u_wind[-1, :]
        atm_v = state.v_wind[-1, :]
        atm_p = pressure_full[-1, :]

        # Height of the lowest model level ABOVE THE LOCAL SURFACE [m]
        # (ncols,) — ``height_full``/``height_half``'s own lowest-level/
        # bottom-interface pair, exactly as ``surface_layer.py``'s
        # ``z_ref = height_full[:,-1] - height_half[:,-1]`` derives it (same
        # quantity, transposed axis order: this module's level axis is
        # first, index -1 = lowest/surface). This replaces a dead,
        # never-consumed placeholder (``height_full[-1,:] -
        # height_full[-1,:].min()``, a GLOBAL minimum across every column —
        # not this column's own surface height) that #1140's review caught
        # once ``AtmosphericForcing.reference_height`` below gave this
        # value a real consumer: the 2 m diagnostics' interpolation
        # fraction and bulk Richardson number, both of which previously
        # assumed (wrongly) that ``atm_temp`` etc. were sampled at 10 m.
        ref_height = jnp.maximum(height_full[-1, :] - height_half[-1, :], 1.0)

        vdiff = diagnostics["vertical_diffusion"]
        exchange_coeff_heat = vdiff.surface_exchange_heat.reshape(
            ncols, nsfc_type,
        )
        exchange_coeff_moisture = vdiff.surface_exchange_moisture.reshape(
            ncols, nsfc_type,
        )
        exchange_coeff_momentum = vdiff.surface_exchange_momentum.reshape(
            ncols, nsfc_type,
        )

        radiation = diagnostics["radiation"]
        atm_forcing = AtmosphericForcing(
            temperature=atm_temp,
            humidity=atm_qv,
            u_wind=atm_u,
            v_wind=atm_v,
            pressure=atm_p,
            surface_pressure=state.normalized_surface_pressure * c.p0,
            sw_downward=radiation.surface_sw_down,
            lw_downward=radiation.surface_lw_down,
            rain_rate=jnp.zeros(ncols),
            snow_rate=jnp.zeros(ncols),
            exchange_coeff_heat=exchange_coeff_heat,
            exchange_coeff_moisture=exchange_coeff_moisture,
            exchange_coeff_momentum=exchange_coeff_momentum,
            reference_height=ref_height,
        )

        # Albedo/radiative/tile bookkeeping, PLUS the explicit ECHAM bulk-
        # aerodynamic flux estimate (reference diagnostic only — the fluxes
        # the column actually receives are delivered by the vdiff implicit
        # solve and read back below; ``bulk_fluxes`` is published alongside
        # them as ``surface.bulk_*``, jax-gcm#1118, not fed back into the
        # column, which would double-count the delivered flux). This
        # includes the ocean tile's own Charnock roughness
        # (``ocean.py::compute_ocean_roughness``), computed here from this
        # step's bulk estimate of u* for the reference flux alone — a
        # SEPARATE evaluation from the live water-tile roughness
        # (``TteTkeVerticalDiffusion``, ``VDiffParameters.cchar``), which
        # uses the previous step's u* from the implicit solve itself (see
        # that term's docstring); the two are not required to agree, since
        # one is diagnostic and the other is the live state.
        # ``_tendencies`` is discarded deliberately, not silently, though
        # for two different reasons depending on the tile: the ocean/ice
        # fields are PROVABLY zero by construction (``ocean_temp`` and
        # sea-ice temperature are prescribed boundary conditions here, not
        # prognostic — see ``ocean.py``'s module docstring), while the land
        # fields (``soil_temp_tendency``/``soil_moisture_tendency``,
        # ``land.py::land_surface_physics_step``) are genuinely NONZERO —
        # that function integrates its own synthetic soil-column bucket —
        # but have no consumer anywhere in jcm's actual state (the live
        # land prognostic is ``TteTkeVerticalDiffusion``'s own skin-energy
        # balance, entirely separate from this module's soil column), so
        # discarding them is correct either way, just not "nothing to
        # discard" for land as it is for ocean/ice. ``surface_diag`` (2 m
        # T/q/dewpoint, bulk Richardson number, the bulk estimate's own
        # energy-balance residual) is now published too (jax-gcm#1140,
        # ``surface.temperature_2m`` etc. below) — the 2 m dew point's
        # placeholder formula is replaced with the exact inverse of jcm's
        # Tetens saturation law (``turbulent_fluxes.py``), and three of its
        # fields stay unpublished because a better (or the only real)
        # version already exists elsewhere; see the ``SurfaceData`` field
        # comments for exactly which and why.
        bulk_fluxes, _tendencies, surface_diag = surface_physics_step(
            atm_forcing, surface_state, dt, params,
            vegetation_fraction=vegetation_fraction_live,
            land_cair=cair_live, land_csat=csat_live,
        )

        # No turbulent-flux tendencies here: the vdiff term's implicit solve
        # already carried the surface exchange into the column, replacing the
        # old operator-split single-layer delivery (imp_* × bulk flux into
        # the lowest level), which silently discarded ~half the flux at T63L47
        # (delivered = imp·bulk, imp ≈ 0.5). The only tendency is the
        # longwave re-emission at the new surface temperature, below.
        temperature_tendency = jnp.zeros_like(state.temperature)

        ch = atm_forcing.exchange_coeff_heat[:, 0]
        cm = atm_forcing.exchange_coeff_momentum[:, 0]

        # Publish the DELIVERED fluxes diagnosed from the vdiff implicit
        # solution. By the ECHAM ``pev_vdiff`` identity these equal the
        # column-integrated vdiff tendencies exactly, so ``evaporation ==
        # effective_evaporation`` — the raw-vs-damped distinction (and the
        # imp_moist factor) no longer exists. The field is kept for API
        # stability (the Tiedtke moisture-budget closure reads it).
        evaporation = vdiff.surface_evaporation.reshape(ncols)
        # The grid surface temperature after the land balance, weighted the
        # same way EchamBoundaryConditions forms it for the radiation
        # (#988): the re-emission below must mirror the temperature the
        # cached longwave was SOLVED with, or the correction and the cached
        # flux describe two different surfaces. ``t_grid_old`` is last
        # step's published (already tile-weighted) value; ``t_grid_new``
        # re-applies the identical weights to this step's post-balance
        # tile temperatures, using the land tile's NEW prognostic skin
        # (``prev_surface.land_surface_temperature``, just updated by the
        # vdiff's implicit solve) with ocean/sea-ice held at their
        # prescribed values (unchanged within a step).
        t_grid_old = prev_surface.surface_temperature.reshape(ncols)
        t_grid_new = tile_weighted_radiative_temperature(
            land_fraction, sea_ice_fraction,
            prev_surface.land_surface_temperature.reshape(ncols),
            ice_surface_temp, ocean_temp,
        )
        out = {}
        if isinstance(radiation, RadiationData):
            radiation_out, heating = correct_surface_longwave(
                radiation, t_grid_old, t_grid_new, diagnostics["pressure_half"])
            temperature_tendency = temperature_tendency.at[-1].set(
                heating.reshape(temperature_tendency.shape[1:]).astype(
                    temperature_tendency.dtype))
            out["radiation"] = radiation_out
        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=temperature_tendency,
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers={},
        )
        # The re-emission heating advances the running thermodynamic view like
        # every other temperature tendency, so convection and the cloud scheme
        # downstream see it. ECHAM's ``physc`` runs ``radheat`` after ``vdiff``
        # and before ``cucall``; ``radheat`` adds the lowest layer's heating at
        # the new surface temperature to ``tte``, and ``mo_cumastr::cumastr``
        # forms its environment as ``ptm1 + ptte * dt``.
        diagnostics = advance_thermo_run(
            diagnostics, dt, d_temperature=temperature_tendency)
        surface_out = prev_surface.copy(
            surface_temperature=t_grid_new,
            skin_temperature=t_grid_new,
            sensible_heat_flux=vdiff.surface_sensible_heat.reshape(ncols),
            latent_heat_flux=vdiff.surface_latent_heat.reshape(ncols),
            momentum_flux_u=vdiff.surface_stress_u.reshape(ncols),
            momentum_flux_v=vdiff.surface_stress_v.reshape(ncols),
            evaporation=evaporation,
            effective_evaporation=evaporation,
            ch=ch,
            cm=cm,
            bulk_sensible_heat_flux=bulk_fluxes.sensible_heat_mean,
            bulk_latent_heat_flux=bulk_fluxes.latent_heat_mean,
            bulk_evaporation=bulk_fluxes.evaporation_mean,
            bulk_momentum_flux_u=bulk_fluxes.momentum_u_mean,
            bulk_momentum_flux_v=bulk_fluxes.momentum_v_mean,
            temperature_2m=surface_diag.temperature_2m,
            humidity_2m=surface_diag.humidity_2m,
            dewpoint_2m=surface_diag.dewpoint_2m,
            richardson_number=surface_diag.richardson_number,
            energy_balance_residual=surface_diag.energy_balance_residual,
            temperature_2m_tile=surface_diag.temperature_2m_tile,
            humidity_2m_tile=surface_diag.humidity_2m_tile,
        )

        return tendency, {**diagnostics, **out, "surface": surface_out}