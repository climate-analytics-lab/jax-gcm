"""ICON physics term functions.

Standalone functions implementing the individual ICON parameterizations
(``apply_radiation``, ``apply_convection``, etc.). These are wrapped by
``ComposablePhysics`` term classes in ``icon_terms.py``; there is no
monolithic orchestrator class — use ``icon_physics()`` from
``icon_terms`` to build a composable ICON physics package.
"""

import jax
from jax import jit
import jax.numpy as jnp
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.physics.icon.constants import physical_constants

# Import physics modules (will be implemented progressively)
from jcm.physics.radiation.grey_two_stream.radiation_scheme import radiation_scheme
from jcm.physics.icon.icon_physics_data import RadiationData
from jcm.physics.convection.tiedtke_nordeng import tiedtke_nordeng_convection
from jcm.physics.clouds.sundqvist import shallow_cloud_scheme
from jcm.physics.clouds.echam_1m import cloud_microphysics
from jcm.physics.icon.parameters import Parameters
from jcm.physics.surface.icon import surface_physics_step, initialize_surface_state
from jcm.physics.surface.icon.surface_types import AtmosphericForcing
from jcm.physics.gravity_waves.hines import gravity_wave_drag
from jcm.physics.chemistry import simple_chemistry
from jcm.physics.icon.icon_physics_data import PhysicsData

@jit
def _prepare_common_physics_state(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Prepare common physics variables that are used by multiple physics modules.
    
    This reduces code duplication by computing pressure levels, heights, air density,
    and other commonly needed variables once for all physics modules.
    
    Args:
        state: Physics state variables (already in 2D format [nlev, ncols])
        boundaries: Boundary conditions (already updated with time-varying conditions)
        geometry: Model geometry
        
    Returns:
        Dictionary with common physics variables

    """
    p0 = physical_constants.p0
    
    # Calculate pressure levels from surface pressure and hybrid (a, b) coefficients.
    # Works for pure sigma (a=0, b=sigma) and ICON hybrid (a + b*P_s).
    surface_pressure = state.normalized_surface_pressure * p0  # Convert to Pa
    pressure_levels = physics_data.icon_coords.calculate_pressure_full(surface_pressure)
    pressure_half = physics_data.icon_coords.calculate_pressure_half(surface_pressure)
    
    # Convert geopotential to height
    height_levels = state.geopotential / physical_constants.grav

    # Calculate height at interfaces (half levels)
    # Internal interfaces are midpoints between full levels
    height_half_internal = (height_levels[1:] + height_levels[:-1]) / 2

    # Top interface: extrapolate using the same spacing as the top layer
    # This maintains consistent layer thickness at the top
    top_layer_thickness = height_levels[0] - height_half_internal[0]
    height_top = height_levels[0] + top_layer_thickness

    # Surface interface: use actual surface height (from geopotential at lowest level)
    # For sigma coordinates, assume surface is at orography height
    # A reasonable approximation is half the lowest layer below the lowest full level
    bottom_layer_thickness = height_half_internal[-1] - height_levels[-1]
    height_surface = height_levels[-1] - bottom_layer_thickness

    height_half = jnp.concatenate((
        height_top[jnp.newaxis],
        height_half_internal,
        height_surface[jnp.newaxis]), axis=0)

    # Calculate air density
    rho = pressure_levels / (physical_constants.rd * state.temperature)
    
    # Calculate layer thickness (clamp to minimum 10m for numerical stability
    # with thin uniform sigma layers)
    dp = jnp.diff(pressure_half, axis=0)
    dz_full = jnp.maximum(dp / (rho * physical_constants.grav), 10.0)
    
    # Calculate relative humidity (Tetens formula; clip T only enough to avoid
    # divide-by-zero at T=29.65K and exp overflow)
    # Wide math-safety clip; NOT a physical-range bound
    T_clip = jnp.clip(state.temperature, 50.0, 500.0)
    q_clip = jnp.maximum(state.specific_humidity, 0.0)
    es = 611.2 * jnp.exp(17.67 * (T_clip - 273.15) / (T_clip - 29.65))
    e = q_clip * pressure_levels / (0.622 + 0.378 * q_clip)
    rel_humidity = e / jnp.maximum(es, 1e-3)

    diagnostic_data = physics_data.diagnostics.copy(
        pressure_full=pressure_levels,
        pressure_half=pressure_half,
        height_full=height_levels,
        height_half=height_half,
        relative_humidity=rel_humidity,
        surface_pressure=surface_pressure,
        air_density=rho,
        layer_thickness=dz_full,
    )

    # Note: chemistry is intentionally not initialized here. ``apply_forcing_data``
    # (the next term in the physics sequence) unconditionally overwrites
    # ``physics_data.chemistry`` with constant GHG concentrations every step,
    # so any initialization work done here would be immediately discarded.
    updated_physics_data = physics_data.copy(diagnostics=diagnostic_data)

    zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return zero_tendencies, updated_physics_data

# Physics term methods


def _radiation_with_caching(
    radiation_fn,
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Wrap a radiation term with sub-stepping via ``jax.lax.cond``.

    On radiation steps the full scheme runs; on other steps the cached
    heating rates from ``physics_data.radiation`` are reused.
    """
    nlev, ncols = state.temperature.shape
    interval = parameters.radiation.radiation_interval
    dt = physics_data.date.dt_seconds
    step = physics_data.date.model_step

    # interval <= 0 ⇒ compute every step (default)
    steps_per_call = jnp.where(
        interval > 0, jnp.int32(jnp.round(interval / dt)), jnp.int32(1)
    )
    should_compute = jnp.mod(step, steps_per_call) == 0

    def _compute():
        return radiation_fn(state, physics_data, parameters, forcing, terrain)

    def _use_cached():
        cached_tend = PhysicsTendency(
            u_wind=jnp.zeros((nlev, ncols)),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=(
                physics_data.radiation.sw_heating_rate
                + physics_data.radiation.lw_heating_rate
            ),
            specific_humidity=jnp.zeros((nlev, ncols)),
            tracers={},
        )
        return cached_tend, physics_data

    return jax.lax.cond(should_compute, _compute, _use_cached)


def apply_radiation(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Grey radiation with sub-stepping."""
    return _radiation_with_caching(
        _apply_radiation_inner, state, physics_data, parameters, forcing, terrain
    )


def apply_radiation_rrtmgp(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """RRTMGP radiation with sub-stepping."""
    return _radiation_with_caching(
        _apply_radiation_rrtmgp_inner,
        state, physics_data, parameters, forcing, terrain,
    )


def apply_radiation_emulated(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Emulated (neural network) radiation with sub-stepping."""
    return _radiation_with_caching(
        _apply_radiation_emulated_inner,
        state, physics_data, parameters, forcing, terrain,
    )


@jit
def _apply_radiation_inner(state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply grey radiation heating rates."""
    # Note: state is already in 2D format [nlev, ncols] from compute_tendencies
    nlev, ncols = state.temperature.shape
    
    # Get lat/lon from cached coordinates
    lat, lon = jax.numpy.meshgrid(
        physics_data.icon_coords.lat * 180.0 / jnp.pi,  # Convert to degrees
        physics_data.icon_coords.lon * 180.0 / jnp.pi,  # degrees
    )
    # Then reshape to (ncols,) to match column format
    latitudes, longitudes = lat.reshape(ncols), lon.reshape(ncols)

    # Get date information for solar calculations
    date = physics_data.date.dt
    
    # Get cloud properties from tracers and previous physics
    cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    cloud_fraction = physics_data.clouds.cloud_fraction

    # Get ozone from chemistry data
    ozone_vmr = physics_data.chemistry.ozone_vmr * 1e-6  # Convert ppmv to VMR
    # CO2 is well-mixed, so use a scalar mean value (radiation scheme expects scalar)
    co2_vmr = jnp.mean(physics_data.chemistry.co2_vmr) * 1e-6  # Convert ppmv to VMR, scalar
    
    # Reshape surface properties to (ncols,) for vmap
    surface_temperature_col = physics_data.surface.surface_temperature.reshape(ncols)  # (ncols,)
    surface_albedo_vis_col = physics_data.radiation.surface_albedo_vis.reshape(ncols)  # (ncols,)
    surface_albedo_nir_col = physics_data.radiation.surface_albedo_nir.reshape(ncols)  # (ncols,)
    surface_emissivity_col = physics_data.radiation.surface_emissivity.reshape(ncols)  # (ncols,)

    # Prepare aerosol data for vmap - reshape to have column as the mapped dimension
    aerosol_data_for_vmap = physics_data.aerosol.copy(
        aod_profile=physics_data.aerosol.aod_profile.reshape(nlev, ncols).T,  # (ncols, nlev)
        ssa_profile=physics_data.aerosol.ssa_profile.reshape(nlev, ncols).T,  # (ncols, nlev)
        asy_profile=physics_data.aerosol.asy_profile.reshape(nlev, ncols).T,  # (ncols, nlev)
        cdnc_factor=physics_data.aerosol.cdnc_factor.reshape(ncols),  # (ncols,)
        aod_total=physics_data.aerosol.aod_total.reshape(ncols),  # (ncols,)
        aod_anthropogenic=physics_data.aerosol.aod_anthropogenic.reshape(ncols),  # (ncols,)
        aod_background=physics_data.aerosol.aod_background.reshape(ncols),  # (ncols,)
        angstrom=physics_data.aerosol.angstrom.reshape(ncols),  # (ncols,)
    )
    
    radiation_results = jax.vmap(
        radiation_scheme,
        in_axes=(1, 1, 1, 1, 1,    # temperature, specific_humidity, pressure_full, pressure_half, layer_thickness (nlev/nlev+1, ncols)
                 1, 1, 1, 1,       # air_density, cloud_water, cloud_ice, cloud_fraction (nlev, ncols)
                 0, 0, 0, 0,       # surface_temperature, surface_albedo_vis, surface_albedo_nir, surface_emissivity (ncols,)
                 None, 0, 0,       # date (scalar), latitudes (ncols,), longitudes (ncols,)
                 None, 0, 1, None),  # parameters (scalar), aerosol_data (per column), ozone_vmr (nlev, ncols), co2_vmr (scalar)
        out_axes=(0, 0),  # Returns (RadiationTendencies, RadiationData) per column
        axis_size=ncols
    )(state.temperature, state.specific_humidity, physics_data.diagnostics.pressure_full, physics_data.diagnostics.pressure_half, physics_data.diagnostics.layer_thickness,
      physics_data.diagnostics.air_density, cloud_water, cloud_ice, cloud_fraction,
      surface_temperature_col, surface_albedo_vis_col,
      surface_albedo_nir_col, surface_emissivity_col,
      date, latitudes, longitudes,
      parameters.radiation, aerosol_data_for_vmap, ozone_vmr, co2_vmr)
    
    # Unpack structured results directly
    tendencies_vmapped, diagnostics_vmapped = radiation_results
    
    # Extract temperature tendencies and transpose to [nlev, ncols]
    temperature_tendency = tendencies_vmapped.temperature_tendency.T
    
    # Create physics tendencies
    # Note: All tendencies should be in [nlev, ncols] format to match the reshaped state
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros((nlev, ncols)),  # No wind tendencies from radiation
        v_wind=jnp.zeros((nlev, ncols)),
        temperature=temperature_tendency,
        specific_humidity=jnp.zeros((nlev, ncols)),  # Match the expected shape
        tracers={}
    )
    
    # Reconstruct RadiationData from vmapped diagnostics
    # Most fields need to be transposed from [ncols, ...] to [..., ncols]
    rad_out = RadiationData(
        cos_zenith=diagnostics_vmapped.cos_zenith.squeeze(),  # [ncols, 1] -> [ncols]
        surface_albedo_vis=diagnostics_vmapped.surface_albedo_vis,
        surface_albedo_nir=diagnostics_vmapped.surface_albedo_nir,
        surface_emissivity=diagnostics_vmapped.surface_emissivity,
        sw_flux_up=diagnostics_vmapped.sw_flux_up.transpose(1, 0, 2).sum(axis=-1),  # [nlev+1, ncols] (summed over bands)
        sw_flux_down=diagnostics_vmapped.sw_flux_down.transpose(1, 0, 2).sum(axis=-1),
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,  # [ncols, nlev] -> [nlev, ncols]
        lw_flux_up=diagnostics_vmapped.lw_flux_up.transpose(1, 0, 2).sum(axis=-1),
        lw_flux_down=diagnostics_vmapped.lw_flux_down.transpose(1, 0, 2).sum(axis=-1),
        lw_heating_rate=tendencies_vmapped.longwave_heating.T,  # [ncols, nlev] -> [nlev, ncols]
        surface_sw_down=diagnostics_vmapped.surface_sw_down,  # Already [ncols]
        surface_lw_down=diagnostics_vmapped.surface_lw_down,
        surface_sw_up=diagnostics_vmapped.surface_sw_up,
        surface_lw_up=diagnostics_vmapped.surface_lw_up,
        toa_sw_up=diagnostics_vmapped.toa_sw_up,
        toa_lw_up=diagnostics_vmapped.toa_lw_up,
        toa_sw_down=diagnostics_vmapped.toa_sw_down
    )
    
    updated_physics_data = physics_data.copy(radiation=rad_out)

    return physics_tendencies, updated_physics_data


@jit
def _apply_radiation_rrtmgp_inner(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply RRTMGP radiation heating rates (inner, always-compute version)."""
    from jcm.physics.radiation.rrtmgp import (
        radiation_scheme_rrtmgp,
    )

    nlev, ncols = state.temperature.shape

    # Get lat/lon from cached coordinates
    lat, lon = jax.numpy.meshgrid(
        physics_data.icon_coords.lat * 180.0 / jnp.pi,
        physics_data.icon_coords.lon * 180.0 / jnp.pi,
    )
    latitudes, longitudes = lat.reshape(ncols), lon.reshape(ncols)

    date = physics_data.date.dt

    cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    cloud_fraction = physics_data.clouds.cloud_fraction

    ozone_vmr = physics_data.chemistry.ozone_vmr * 1e-6
    co2_vmr = jnp.mean(physics_data.chemistry.co2_vmr) * 1e-6

    surface_temperature_col = physics_data.surface.surface_temperature.reshape(ncols)
    surface_albedo_vis_col = physics_data.radiation.surface_albedo_vis.reshape(ncols)
    surface_albedo_nir_col = physics_data.radiation.surface_albedo_nir.reshape(ncols)
    surface_emissivity_col = physics_data.radiation.surface_emissivity.reshape(ncols)

    aerosol_data_for_vmap = physics_data.aerosol.copy(
        aod_profile=physics_data.aerosol.aod_profile.reshape(nlev, ncols).T,
        ssa_profile=physics_data.aerosol.ssa_profile.reshape(nlev, ncols).T,
        asy_profile=physics_data.aerosol.asy_profile.reshape(nlev, ncols).T,
        cdnc_factor=physics_data.aerosol.cdnc_factor.reshape(ncols),
        aod_total=physics_data.aerosol.aod_total.reshape(ncols),
        aod_anthropogenic=physics_data.aerosol.aod_anthropogenic.reshape(ncols),
        aod_background=physics_data.aerosol.aod_background.reshape(ncols),
        angstrom=physics_data.aerosol.angstrom.reshape(ncols),
    )

    radiation_results = jax.vmap(
        radiation_scheme_rrtmgp,
        in_axes=(
            1, 1, 1, 1, 1,     # temperature..layer_thickness
            1, 1, 1, 1,        # air_density..cloud_fraction
            0, 0, 0, 0,        # surface scalars
            None, 0, 0,        # date, lat, lon
            None, 0, 1, None,  # parameters, aerosol, ozone, co2
        ),
        out_axes=(0, 0),
        axis_size=ncols,
    )(
        state.temperature, state.specific_humidity,
        physics_data.diagnostics.pressure_full,
        physics_data.diagnostics.pressure_half,
        physics_data.diagnostics.layer_thickness,
        physics_data.diagnostics.air_density,
        cloud_water, cloud_ice, cloud_fraction,
        surface_temperature_col, surface_albedo_vis_col,
        surface_albedo_nir_col, surface_emissivity_col,
        date, latitudes, longitudes,
        parameters.radiation, aerosol_data_for_vmap, ozone_vmr, co2_vmr,
    )

    tendencies_vmapped, diagnostics_vmapped = radiation_results
    temperature_tendency = tendencies_vmapped.temperature_tendency.T

    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros((nlev, ncols)),
        v_wind=jnp.zeros((nlev, ncols)),
        temperature=temperature_tendency,
        specific_humidity=jnp.zeros((nlev, ncols)),
        tracers={},
    )

    rad_out = RadiationData(
        cos_zenith=diagnostics_vmapped.cos_zenith.squeeze(),
        surface_albedo_vis=diagnostics_vmapped.surface_albedo_vis,
        surface_albedo_nir=diagnostics_vmapped.surface_albedo_nir,
        surface_emissivity=diagnostics_vmapped.surface_emissivity,
        sw_flux_up=diagnostics_vmapped.sw_flux_up.transpose(1, 0, 2).sum(axis=-1),
        sw_flux_down=diagnostics_vmapped.sw_flux_down.transpose(1, 0, 2).sum(axis=-1),
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
        lw_flux_up=diagnostics_vmapped.lw_flux_up.transpose(1, 0, 2).sum(axis=-1),
        lw_flux_down=diagnostics_vmapped.lw_flux_down.transpose(1, 0, 2).sum(axis=-1),
        lw_heating_rate=tendencies_vmapped.longwave_heating.T,
        surface_sw_down=diagnostics_vmapped.surface_sw_down,
        surface_lw_down=diagnostics_vmapped.surface_lw_down,
        surface_sw_up=diagnostics_vmapped.surface_sw_up,
        surface_lw_up=diagnostics_vmapped.surface_lw_up,
        toa_sw_up=diagnostics_vmapped.toa_sw_up,
        toa_lw_up=diagnostics_vmapped.toa_lw_up,
        toa_sw_down=diagnostics_vmapped.toa_sw_down,
    )

    updated_physics_data = physics_data.copy(radiation=rad_out)
    return physics_tendencies, updated_physics_data


@jit
def _apply_radiation_emulated_inner(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply emulated (neural-network) radiation heating rates.

    Uses bidirectional GRU networks to predict SW and LW fluxes for each
    atmospheric column, then derives heating rates from flux divergence.
    """
    from jcm.physics.radiation.nn_emulator_scheme import (
        radiation_scheme_emulated,
    )

    nlev, ncols = state.temperature.shape

    # Lat/lon from cached coordinates
    lat, lon = jax.numpy.meshgrid(
        physics_data.icon_coords.lat * 180.0 / jnp.pi,
        physics_data.icon_coords.lon * 180.0 / jnp.pi,
    )
    latitudes, longitudes = lat.reshape(ncols), lon.reshape(ncols)

    date = physics_data.date.dt

    cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    cloud_fraction = physics_data.clouds.cloud_fraction

    ozone_vmr = physics_data.chemistry.ozone_vmr * 1e-6
    co2_vmr = jnp.mean(physics_data.chemistry.co2_vmr) * 1e-6

    surface_temperature_col = physics_data.surface.surface_temperature.reshape(ncols)
    surface_albedo_vis_col = physics_data.radiation.surface_albedo_vis.reshape(ncols)
    surface_albedo_nir_col = physics_data.radiation.surface_albedo_nir.reshape(ncols)
    surface_emissivity_col = physics_data.radiation.surface_emissivity.reshape(ncols)

    aerosol_data_for_vmap = physics_data.aerosol.copy(
        aod_profile=physics_data.aerosol.aod_profile.reshape(nlev, ncols).T,
        ssa_profile=physics_data.aerosol.ssa_profile.reshape(nlev, ncols).T,
        asy_profile=physics_data.aerosol.asy_profile.reshape(nlev, ncols).T,
        cdnc_factor=physics_data.aerosol.cdnc_factor.reshape(ncols),
        aod_total=physics_data.aerosol.aod_total.reshape(ncols),
        aod_anthropogenic=physics_data.aerosol.aod_anthropogenic.reshape(ncols),
        aod_background=physics_data.aerosol.aod_background.reshape(ncols),
        angstrom=physics_data.aerosol.angstrom.reshape(ncols),
    )

    # Extract emulator weights and scaling from radiation parameters
    emulator_weights = parameters.radiation.emulator_weights
    sw_scaling = parameters.radiation.sw_scaling
    lw_scaling = parameters.radiation.lw_scaling

    radiation_results = jax.vmap(
        radiation_scheme_emulated,
        in_axes=(
            1, 1, 1, 1, 1,     # temperature..layer_thickness
            1, 1, 1, 1,        # air_density..cloud_fraction
            0, 0, 0, 0,        # surface scalars
            None, 0, 0,        # date, lat, lon
            None, 0, 1, None,  # parameters, aerosol, ozone, co2
            None, None, None,  # emulator_weights, sw_scaling, lw_scaling
        ),
        out_axes=(0, 0),
        axis_size=ncols,
    )(
        state.temperature, state.specific_humidity,
        physics_data.diagnostics.pressure_full,
        physics_data.diagnostics.pressure_half,
        physics_data.diagnostics.layer_thickness,
        physics_data.diagnostics.air_density,
        cloud_water, cloud_ice, cloud_fraction,
        surface_temperature_col, surface_albedo_vis_col,
        surface_albedo_nir_col, surface_emissivity_col,
        date, latitudes, longitudes,
        parameters.radiation, aerosol_data_for_vmap, ozone_vmr, co2_vmr,
        emulator_weights, sw_scaling, lw_scaling,
    )

    tendencies_vmapped, diagnostics_vmapped = radiation_results
    temperature_tendency = tendencies_vmapped.temperature_tendency.T

    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros((nlev, ncols)),
        v_wind=jnp.zeros((nlev, ncols)),
        temperature=temperature_tendency,
        specific_humidity=jnp.zeros((nlev, ncols)),
        tracers={},
    )

    # Emulated fluxes are 1D per column (no spectral bands), so the
    # vmapped output shapes are [ncols, nlev+1] for fluxes and
    # [ncols, nlev] for heating rates.
    rad_out = RadiationData(
        cos_zenith=diagnostics_vmapped.cos_zenith.squeeze(),
        surface_albedo_vis=diagnostics_vmapped.surface_albedo_vis.squeeze(),
        surface_albedo_nir=diagnostics_vmapped.surface_albedo_nir.squeeze(),
        surface_emissivity=diagnostics_vmapped.surface_emissivity.squeeze(),
        sw_flux_up=diagnostics_vmapped.sw_flux_up.T,        # [ncols, nlev+1] -> [nlev+1, ncols]
        sw_flux_down=diagnostics_vmapped.sw_flux_down.T,
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
        lw_flux_up=diagnostics_vmapped.lw_flux_up.T,
        lw_flux_down=diagnostics_vmapped.lw_flux_down.T,
        lw_heating_rate=tendencies_vmapped.longwave_heating.T,
        surface_sw_down=diagnostics_vmapped.surface_sw_down.squeeze(),
        surface_lw_down=diagnostics_vmapped.surface_lw_down.squeeze(),
        surface_sw_up=diagnostics_vmapped.surface_sw_up.squeeze(),
        surface_lw_up=diagnostics_vmapped.surface_lw_up.squeeze(),
        toa_sw_up=diagnostics_vmapped.toa_sw_up.squeeze(),
        toa_lw_up=diagnostics_vmapped.toa_lw_up.squeeze(),
        toa_sw_down=diagnostics_vmapped.toa_sw_down.squeeze(),
    )

    updated_physics_data = physics_data.copy(radiation=rad_out)
    return physics_tendencies, updated_physics_data


@jit
def apply_convection(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply Tiedtke-Nordeng convection scheme with fixed qc/qi transport"""
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    layer_thickness = physics_data.diagnostics.layer_thickness
    air_density = physics_data.diagnostics.air_density

    # Extract fixed qc/qi tracers (with defaults if not present)
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    conv_results = jax.vmap(
        tiedtke_nordeng_convection,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (ConvectionTendencies, ConvectionState) per column
    )(state.temperature, state.specific_humidity, pressure_levels, layer_thickness, 
      air_density, state.u_wind, state.v_wind, qc, qi, dt, parameters.convection)
    
    # Unpack structured results directly (no tuple unpacking needed)
    conv_tendencies_all, conv_states_all = conv_results
    
    physics_tendencies = PhysicsTendency(
        u_wind=conv_tendencies_all.dudt.T,
        v_wind=conv_tendencies_all.dvdt.T,
        temperature=conv_tendencies_all.dtedt.T,
        specific_humidity=conv_tendencies_all.dqdt.T,
        tracers={
            'qc': conv_tendencies_all.dqc_dt.T,
            'qi': conv_tendencies_all.dqi_dt.T
        }
    )
    
    # Update physics data with convection diagnostics (transpose scalars)
    convection_data = physics_data.convection.copy(
        qc_conv=conv_tendencies_all.qc_conv.T,
        qi_conv=conv_tendencies_all.qi_conv.T,
        precip_conv=conv_tendencies_all.precip_conv,  # Already 1D per column
    )
    updated_physics_data = physics_data.copy(convection=convection_data)
    
    return physics_tendencies, updated_physics_data

def _cloud_and_microphysics_column(
    temperature, specific_humidity, pressure, qc, qi,
    surface_pressure, air_density, layer_thickness, droplet_number,
    dt, cloud_config, micro_config
):
    """Compute cloud and microphysics for a single column.

    Following ECHAM mo_cloud.f90: condensation, cloud fraction, autoconversion,
    accretion, and precipitation are all computed in a single column sweep.
    This avoids the coupling issues of splitting them into separate calls.

    Tendency accounting (no double counting):
        The cloud scheme computes condensation and applies it within the
        timestep to produce updated cloud water (cloud_state.cloud_water).
        Microphysics then acts on this updated cloud water.

        Both schemes return SEPARATE tendencies that are additive:
        - Cloud:  dqcdt = +condensation,  dqdt = -condensation,  dtedt = +L*condensation/cp
        - Micro:  dqcdt = -autoconversion, dqdt = +evaporation,  dtedt = micro heating/cooling

        The integrator applies: qc_new = qc_old + (cloud_dqcdt + micro_dqcdt) * dt
        This gives: qc_new = 0 + (condensation - autoconversion) * dt

        Moisture is conserved: dq + dqc + precip = 0
        (-condensation + evap) + (condensation - autoconv) + (autoconv - evap) = 0

        The within-timestep cloud water update is used ONLY to provide
        microphysics with a physically meaningful input — it does not
        affect the tendencies returned to the integrator.
    """
    # 1. Cloud fraction and condensation
    cloud_tendencies, cloud_state = shallow_cloud_scheme(
        temperature, specific_humidity, pressure,
        qc, qi, surface_pressure, dt, cloud_config
    )

    # 2. Microphysics acts on the condensation-updated cloud water/ice
    micro_tendencies, micro_state = cloud_microphysics(
        temperature, specific_humidity, pressure,
        cloud_state.cloud_water, cloud_state.cloud_ice,
        cloud_state.cloud_fraction, air_density, layer_thickness,
        droplet_number, dt, micro_config
    )

    return cloud_tendencies, cloud_state, micro_tendencies, micro_state


@jit
def apply_clouds_and_microphysics(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply cloud scheme and microphysics in a single coupled step.

    Combines condensation → cloud fraction → autoconversion → precipitation
    in one vmapped column call, following ECHAM mo_cloud.f90.
    """
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    surface_pressure = physics_data.diagnostics.surface_pressure
    air_density = physics_data.diagnostics.air_density
    dz = physics_data.diagnostics.layer_thickness
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))

    # Droplet number concentration from aerosol scheme
    base_cdnc = parameters.microphysics.base_cdnc  # Clean-air baseline CDNC (1/m³)
    cdnc_factor = physics_data.aerosol.cdnc_factor  # (ncols,)
    cdnc_m3 = jnp.ones_like(state.temperature) * base_cdnc * cdnc_factor[jnp.newaxis, :]
    droplet_number_per_kg = cdnc_m3 / air_density  # 1/m³ → 1/kg (for microphysics)

    cloud_config = parameters.clouds
    micro_config = parameters.microphysics

    # Single vmap over columns: cloud + microphysics together
    cloud_tend_all, cloud_state_all, micro_tend_all, micro_state_all = jax.vmap(
        _cloud_and_microphysics_column,
        in_axes=(1, 1, 1, 1, 1, 0, 1, 1, 1, None, None, None),
        out_axes=(0, 0, 0, 0)
    )(state.temperature, state.specific_humidity, pressure_levels,
      qc, qi, surface_pressure, air_density, dz, droplet_number_per_kg,
      dt, cloud_config, micro_config)

    # Combine tendencies: cloud (condensation) + microphysics (autoconversion etc.)
    # These are separate physical processes — see _cloud_and_microphysics_column
    # docstring for the full accounting showing no double counting.
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=cloud_tend_all.dtedt.T + micro_tend_all.dtedt.T,
        specific_humidity=cloud_tend_all.dqdt.T + micro_tend_all.dqdt.T,
        tracers={
            'qc': cloud_tend_all.dqcdt.T + micro_tend_all.dqcdt.T,
            'qi': cloud_tend_all.dqidt.T + micro_tend_all.dqidt.T
        }
    )

    # Update physics data with cloud and microphysics diagnostics
    cloud_data = physics_data.clouds.copy(
        cloud_fraction=cloud_state_all.cloud_fraction.T,
        qc=cloud_state_all.cloud_water.T,
        qi=cloud_state_all.cloud_ice.T,
        precip_rain=micro_state_all.precip_rain,
        precip_snow=micro_state_all.precip_snow,
        droplet_number=cdnc_m3  # Store in 1/m³ for diagnostics/radiation
    )

    diagnostics = physics_data.diagnostics.copy(
        relative_humidity=cloud_state_all.rel_humidity.T,
    )

    updated_physics_data = physics_data.copy(clouds=cloud_data,
                                             diagnostics=diagnostics)

    return physics_tendencies, updated_physics_data

@jit
def apply_vertical_diffusion(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply vertical diffusion and boundary layer physics.

    The underlying ``vertical_diffusion_column`` routine already accepts
    batched ``(ncols, nlev)`` arrays, so we call it once directly instead of
    wrapping a fake single-column vmap around it. Inputs to physics terms are
    ``(nlev, ncols)``; we transpose to ``(ncols, nlev)`` at the boundary.
    """
    from jcm.physics.vertical_diffusion.tte_tke import (
        prepare_vertical_diffusion_state,
        vertical_diffusion_column,
    )

    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_full = physics_data.diagnostics.pressure_full
    pressure_half = physics_data.diagnostics.pressure_half
    height_full = physics_data.diagnostics.height_full
    height_half = physics_data.diagnostics.height_half

    # Prognostic TKE (reshape grid format to column format if needed).
    # ``thv_variance`` is not a stored diagnostic, so just zero it out each call.
    tke = physics_data.vertical_diffusion.tke
    if tke.ndim == 3:
        tke = tke.reshape(nlev, ncols)
    thv_variance = jnp.zeros((nlev, ncols))

    # Surface properties (simplified - should come from boundaries)
    nsfc_type = 3  # water, ice, land
    surface_fraction = jnp.zeros((ncols, nsfc_type))
    surface_fraction = surface_fraction.at[:, 2].set(1.0)  # All land for now

    # Get surface properties from boundaries
    surface_temp_col = physics_data.surface.surface_temperature.reshape(ncols)
    roughness_length_col = physics_data.surface.roughness_length.reshape(ncols)
    surface_temperature = jnp.repeat(surface_temp_col[:, jnp.newaxis], nsfc_type, axis=1)
    roughness = jnp.repeat(roughness_length_col[:, jnp.newaxis], nsfc_type, axis=1)

    # Ocean currents (zero for now)
    ocean_u = jnp.zeros(ncols)
    ocean_v = jnp.zeros(ncols)

    # Extract fixed qc/qi tracers
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))

    # Transpose column-first fields from (nlev, ncols) to (ncols, nlev) for the
    # batched vertical diffusion routines.
    vdiff_state = prepare_vertical_diffusion_state(
        u=state.u_wind.T,
        v=state.v_wind.T,
        temperature=state.temperature.T,
        qv=state.specific_humidity.T,
        qc=qc.T,
        qi=qi.T,
        pressure_full=pressure_full.T,
        pressure_half=pressure_half.T,
        geopotential=state.geopotential.T,
        height_full=height_full.T,
        height_half=height_half.T,
        surface_temperature=surface_temperature,
        surface_fraction=surface_fraction,
        roughness_length=roughness,
        ocean_u=ocean_u,
        ocean_v=ocean_v,
        tke=tke.T,
        thv_variance=thv_variance.T,
    )

    vdiff_tendencies, vdiff_diagnostics = vertical_diffusion_column(
        vdiff_state, parameters.vertical_diffusion, dt
    )
    
    # Extract tendencies (already in correct shape [ncols, nlev] from vmap)
    u_tend = vdiff_tendencies.u_tendency.T  # Transpose to [nlev, ncols]
    v_tend = vdiff_tendencies.v_tendency.T
    temp_tend = vdiff_tendencies.temperature_tendency.T
    qv_tend = vdiff_tendencies.qv_tendency.T
    qc_tend = vdiff_tendencies.qc_tendency.T
    qi_tend = vdiff_tendencies.qi_tendency.T
    tke_tend = vdiff_tendencies.tke_tendency.T
    
    # Extract diagnostics (already in correct shape from vmap)
    km = vdiff_diagnostics.exchange_coeff_momentum.T  # Transpose to [nlev, ncols]
    kh = vdiff_diagnostics.exchange_coeff_heat.T
    pbl_height = vdiff_diagnostics.boundary_layer_height  # Shape [ncols]
    u_star = vdiff_diagnostics.friction_velocity  # Shape [ncols]

    # Extract surface exchange coefficients (per surface type)
    surface_exchange_heat = vdiff_diagnostics.surface_exchange_heat  # (ncols, nsfc_type)
    surface_exchange_moisture = vdiff_diagnostics.surface_exchange_moisture  # (ncols, nsfc_type)
    # Momentum: use lowest-level profile coefficient, broadcast across surface types
    surface_exchange_momentum = jnp.repeat(
        vdiff_diagnostics.exchange_coeff_momentum[:, -1:], nsfc_type, axis=1
    )  # (ncols, nsfc_type)
    
    # Update TKE — clip to a physically defensible range. Strong
    # convective storms produce updraft-core TKE of ~50–100 m²/s²;
    # values above ~250 are non-physical and indicate the implicit
    # vdiff solver has gone unstable (which it does in the moist run
    # within ~10 timesteps if left unconstrained — TKE then cascades
    # through the exchange-coefficient back-reaction and NaNs the
    # whole column on the next step). A hard upper cap is the cheapest
    # safeguard that lets the rest of the physics step finish; the
    # underlying TKE budget should be retuned separately.
    new_tke = tke + dt * tke_tend
    new_tke = jnp.clip(new_tke, 0.01, 250.0)

    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=u_tend,
        v_wind=v_tend,
        temperature=temp_tend,
        specific_humidity=qv_tend,
        tracers={
            'qc': qc_tend,
            'qi': qi_tend
        }
    )
    
    # Update physics data with vertical diffusion diagnostics
    # Only update fields that actually exist in VerticalDiffusionData
    vdiff_data = physics_data.vertical_diffusion.copy(
        tke=new_tke,
        km=km,
        kh=kh,
        surface_exchange_heat=surface_exchange_heat,
        surface_exchange_moisture=surface_exchange_moisture,
        surface_exchange_momentum=surface_exchange_momentum,
        pbl_height=pbl_height,
        surface_friction_velocity=u_star,
    )
    
    updated_physics_data = physics_data.copy(vertical_diffusion=vdiff_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_surface(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply surface physics and calculate surface fluxes"""
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    # Get surface properties from boundaries (now guaranteed to be present)
    # Reshape boundary fields to column format
    surface_temp = physics_data.surface.surface_temperature.reshape(ncols)

    # Surface tile fractions: water (0), sea ice (1), land (2).
    # Sea ice fraction is taken from prescribed boundary conditions and
    # constrained to the non-land area so that fractions sum to exactly 1.
    nsfc_type = 3
    surface_fractions = jnp.zeros((ncols, nsfc_type))
    land_fraction = terrain.fmask.reshape((ncols,))
    raw_ice = forcing.sice_am[..., 0] if forcing.sice_am.ndim == 3 else forcing.sice_am
    sea_ice_fraction = jnp.clip(raw_ice.reshape((ncols,)), 0.0, 1.0 - land_fraction)
    water_fraction = 1.0 - land_fraction - sea_ice_fraction
    surface_fractions = surface_fractions.at[:, 0].set(water_fraction)
    surface_fractions = surface_fractions.at[:, 1].set(sea_ice_fraction)
    surface_fractions = surface_fractions.at[:, 2].set(land_fraction)

    ocean_temp = surface_temp
    ice_temp = jnp.repeat(surface_temp[:, jnp.newaxis], 2, axis=1)  # 2 ice layers
    soil_temp = jnp.repeat(surface_temp[:, jnp.newaxis], 4, axis=1)  # 4 soil layers
    
    surface_state = initialize_surface_state(
        ncols, surface_fractions, ocean_temp, ice_temp, soil_temp, parameters.surface
    )
    
    # Prepare atmospheric forcing
    # Use lowest model level for surface conditions
    atm_temp = state.temperature[-1, :]  # Lowest model level
    atm_qv = state.specific_humidity[-1, :]
    atm_u = state.u_wind[-1, :]
    atm_v = state.v_wind[-1, :]
    atm_p = pressure_levels[-1, :]
    
    # Height of lowest model level above surface
    ref_height = physics_data.diagnostics.height_full[-1, :] - physics_data.diagnostics.height_full[-1, :].min()
    ref_height = jnp.maximum(ref_height, 10.0)  # At least 10m
    
    # Get exchange coefficients from vertical diffusion diagnostics
    nsfc_type = 3
    exchange_coeff_heat = physics_data.vertical_diffusion.surface_exchange_heat.reshape(ncols, nsfc_type)
    exchange_coeff_moisture = physics_data.vertical_diffusion.surface_exchange_moisture.reshape(ncols, nsfc_type)
    exchange_coeff_momentum = physics_data.vertical_diffusion.surface_exchange_momentum.reshape(ncols, nsfc_type)

    atm_forcing = AtmosphericForcing(
        temperature=atm_temp,
        humidity=atm_qv,
        u_wind=atm_u,
        v_wind=atm_v,
        pressure=atm_p,
        sw_downward=physics_data.radiation.surface_sw_down,
        lw_downward=physics_data.radiation.surface_lw_down,
        rain_rate=jnp.zeros(ncols),  # No rain for now
        snow_rate=jnp.zeros(ncols),  # No snow for now
        exchange_coeff_heat=exchange_coeff_heat,
        exchange_coeff_moisture=exchange_coeff_moisture,
        exchange_coeff_momentum=exchange_coeff_momentum
    )
    
    # Apply surface physics to all columns
    fluxes, tendencies, diagnostics = surface_physics_step(
        atm_forcing, surface_state, dt, parameters.surface
    )
    
    # Extract grid-box mean fluxes
    sensible_heat = fluxes.sensible_heat_mean
    latent_heat = fluxes.latent_heat_mean
    tau_u = fluxes.momentum_u_mean
    tau_v = fluxes.momentum_v_mean
    evaporation = fluxes.evaporation_mean
    
    # Convert fluxes to atmospheric tendencies
    # Only the lowest model level is directly affected by surface fluxes
    
    # Air density at surface
    rho_sfc = pressure_levels[-1, :] / (physical_constants.rd * state.temperature[-1, :])
    
    # Layer thickness at surface (approximate, clamp to minimum 50m to avoid
    # enormous tendencies from thin uniform sigma layers)
    dp_sfc = pressure_levels[-1, :] - pressure_levels[-2, :]
    dz_sfc = jnp.maximum(dp_sfc / (rho_sfc * physical_constants.grav), 50.0)
    
    # Surface flux tendencies (applied to lowest level only)
    temp_tend_sfc = sensible_heat / (rho_sfc * physical_constants.cp * dz_sfc)
    qv_tend_sfc = evaporation / (rho_sfc * dz_sfc)
    u_tend_sfc = -tau_u / (rho_sfc * dz_sfc)
    v_tend_sfc = -tau_v / (rho_sfc * dz_sfc)
    
    # Initialize tendencies (only surface level affected)
    temp_tend = jnp.zeros_like(state.temperature)
    qv_tend = jnp.zeros_like(state.specific_humidity)
    u_tend = jnp.zeros_like(state.u_wind)
    v_tend = jnp.zeros_like(state.v_wind)
    
    # Apply surface tendencies to lowest level
    temp_tend = temp_tend.at[-1, :].set(temp_tend_sfc)
    qv_tend = qv_tend.at[-1, :].set(qv_tend_sfc)
    u_tend = u_tend.at[-1, :].set(u_tend_sfc)
    v_tend = v_tend.at[-1, :].set(v_tend_sfc)
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=u_tend,
        v_wind=v_tend,
        temperature=temp_tend,
        specific_humidity=qv_tend,
        tracers={}
    )
    
    # Update physics data with surface diagnostics
    # Extract exchange coefficients from atmospheric forcing
    ch = atm_forcing.exchange_coeff_heat[:, 0]  # Heat exchange coefficient
    cm = atm_forcing.exchange_coeff_momentum[:, 0]  # Momentum exchange coefficient
    
    surface_data = physics_data.surface.copy(
        sensible_heat_flux=sensible_heat,
        latent_heat_flux=latent_heat,
        momentum_flux_u=tau_u,
        momentum_flux_v=tau_v,
        evaporation=evaporation,  # Use 'evaporation' not 'evaporation_flux'
        ch=ch,
        cm=cm,
    )
    
    updated_physics_data = physics_data.copy(surface=surface_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_gravity_waves(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply gravity wave drag"""
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    height_levels = physics_data.diagnostics.height_full
    air_density = physics_data.diagnostics.air_density
    
    # Subgrid orography standard deviation drives the launch amplitude
    # for orographic GWs (flux ∝ h_std²). Without it the scheme
    # defaulted to a hard-coded 200 m global field, which on an
    # aquaplanet (no orography) sprays gravity-wave momentum into the
    # stratosphere everywhere and overshoots the column wind in the
    # 30-200 hPa range — we observed u_wind growing past ±70 m/s at
    # levels 2-4 by day 0.4 of the moist run, which then NaN'd the
    # dynamics. The proper subgrid std dev needs a terrain preprocessing
    # step we don't have yet; for now scale ~10% of the local orography
    # height as a rough proxy. Aquaplanet (orog ≡ 0) ⇒ h_std = 0 ⇒ GWD
    # is silent, which is what we want for flat ocean.
    orog_flat = terrain.orog.reshape(-1)
    h_std = jnp.abs(orog_flat) * 0.1
    
    gwd_results = jax.vmap(
        gravity_wave_drag,
        in_axes=(1, 1, 1,
                 1, 1, 1,
                 0, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (GWDTendencies, GWDState) per column
    )(state.u_wind, state.v_wind, state.temperature,
        pressure_levels, height_levels, air_density,
        h_std, dt, parameters.gravity_waves)
    
    # Unpack structured results directly
    gwd_tendencies_all, gwd_states_all = gwd_results
    
    physics_tendencies = PhysicsTendency(
        u_wind=gwd_tendencies_all.dudt.T,
        v_wind=gwd_tendencies_all.dvdt.T,
        temperature=gwd_tendencies_all.dtedt.T,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={}
    )
    
    # Update physics data
    # Note: PhysicsData doesn't have a gravity_waves field, so no diagnostics storage for now
    updated_physics_data = physics_data
    
    return physics_tendencies, updated_physics_data

@jit
def apply_chemistry(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply chemistry tendencies
    
    Computes tendencies from simple chemistry including:
    - Fixed ozone distribution with relaxation
    - Methane chemistry with simple decay
    - CO2 tracking (no chemistry)
    """
    # Extract state variables
    nlev, ncols = state.temperature.shape
    temperature = state.temperature.T  # (ncols, nlev)
    pressure = physics_data.diagnostics.pressure_full.T  # (ncols, nlev)
    surface_pressure = physics_data.diagnostics.surface_pressure
    
    # Get current chemistry tracers from physics data
    current_ozone = physics_data.chemistry.ozone_vmr.T  # (ncols, nlev)
    current_methane = physics_data.chemistry.methane_vmr.T  # (ncols, nlev)
    
    dt = parameters.convection.dt_conv  # Time step (from convection for now)
    
    # Call chemistry scheme
    chemistry_tend, chemistry_state = simple_chemistry(
        pressure=pressure.T,  # Back to (nlev, ncols)
        surface_pressure=surface_pressure,
        temperature=temperature.T,  # Back to (nlev, ncols)
        current_ozone=current_ozone.T,  # Back to (nlev, ncols)
        current_methane=current_methane.T,  # Back to (nlev, ncols)
        dt=dt,
        config=None  # Use default chemistry parameters
    )
    
    # Update physics data with chemistry diagnostics
    updated_chemistry_data = physics_data.chemistry.copy(
        ozone_vmr=chemistry_state.ozone_vmr,
        methane_vmr=chemistry_state.methane_vmr,
        co2_vmr=chemistry_state.co2_vmr,
        ozone_production=chemistry_state.ozone_production,
        ozone_loss=chemistry_state.ozone_loss,
        methane_loss=chemistry_state.methane_loss
    )
    
    updated_physics_data = physics_data.copy(chemistry=updated_chemistry_data)
    
    # Currently chemistry doesn't directly affect temperature or dynamics
    # In future could add:
    # - Ozone heating rates in radiation
    # - Methane oxidation heating
    # For now, return zero tendencies
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    
    return physics_tendencies, updated_physics_data