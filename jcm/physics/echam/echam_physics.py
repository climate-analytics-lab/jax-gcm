"""ECHAM physics term functions.

Standalone functions implementing the individual ECHAM parameterizations
(``apply_radiation``, ``apply_convection``, etc.). These are wrapped by
``ComposablePhysics`` term classes in ``echam_terms.py``; there is no
monolithic orchestrator class — use ``echam_physics()`` from
``echam_terms`` to build a composable ECHAM physics package.
"""

import logging

import jax
from jax import jit
import jax.numpy as jnp
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm import constants as physical_constants

# Import physics modules (will be implemented progressively)
from jcm.physics.radiation.grey_two_stream.radiation_scheme import radiation_scheme
from jcm.physics.echam.echam_physics_data import RadiationData
from jcm.physics.convection.tiedtke_nordeng import tiedtke_nordeng_convection
from jcm.physics.clouds.sundqvist import shallow_cloud_scheme
from jcm.physics.clouds.echam_1m import cloud_microphysics
from jcm.physics.echam.parameters import Parameters
from jcm.physics.surface.echam import surface_physics_step, initialize_surface_state
from jcm.physics.surface.echam.surface_types import AtmosphericForcing
from jcm.physics.gravity_waves.hines import hines_gwd
from jcm.physics.gravity_waves.simple import simple_gwd
from jcm.physics.gravity_waves.sso import sso_drag
from jcm.physics.chemistry import simple_chemistry
from jcm.physics.echam.echam_physics_data import PhysicsData
from jcm.physics.aerosol.spa import spa_activated_cdnc

logger = logging.getLogger(__name__)


def _column_vector(value: jnp.ndarray, ncols: int) -> jnp.ndarray:
    """Return a vmapped scalar diagnostic as one value per column."""
    return jnp.reshape(value, (ncols,))


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
    pressure_levels = physics_data.echam_coords.calculate_pressure_full(surface_pressure)
    pressure_half = physics_data.echam_coords.calculate_pressure_half(surface_pressure)
    
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
    dt = physics_data.dt_seconds
    step = physics_data.model_step

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
        physics_data.echam_coords.lat * 180.0 / jnp.pi,  # Convert to degrees
        physics_data.echam_coords.lon * 180.0 / jnp.pi,  # degrees
    )
    # Then reshape to (ncols,) to match column format
    latitudes, longitudes = lat.reshape(ncols), lon.reshape(ncols)

    # Get date information for solar calculations
    # Solar geometry comes pre-baked on `forcing.solar` (populated by
    # `Model._get_step_fn_factory` ↔ `ForcingData.select(date)`). The
    # radiation scheme stays date-free.
    solar = forcing.solar
    
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
      solar, latitudes, longitudes,
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
    
    # Reconstruct RadiationData from vmapped diagnostics. The grey scheme
    # keeps a per-band axis on the flux profiles (shape
    # ``(ncols, nlev+1, n_bands)`` after vmap), so they need
    # ``transpose(1, 0, 2).sum(axis=-1)`` here to collapse to
    # ``(nlev+1, ncols)``. Scalar diagnostics are flattened to
    # ``[ncols]`` via ``_column_vector(...)`` so the compute and cache
    # branches preserve identical shapes for single-column and global
    # runs (otherwise the radiation ``lax.cond`` at ``ncols=1`` errors
    # on a shape mismatch).
    rad_out = RadiationData(
        cos_zenith=_column_vector(diagnostics_vmapped.cos_zenith, ncols),  # [ncols, 1] -> [ncols]
        surface_albedo_vis=_column_vector(diagnostics_vmapped.surface_albedo_vis, ncols),
        surface_albedo_nir=_column_vector(diagnostics_vmapped.surface_albedo_nir, ncols),
        surface_emissivity=_column_vector(diagnostics_vmapped.surface_emissivity, ncols),
        sw_flux_up=diagnostics_vmapped.sw_flux_up.transpose(1, 0, 2).sum(axis=-1),  # [nlev+1, ncols] (summed over bands)
        sw_flux_down=diagnostics_vmapped.sw_flux_down.transpose(1, 0, 2).sum(axis=-1),
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,  # [ncols, nlev] -> [nlev, ncols]
        lw_flux_up=diagnostics_vmapped.lw_flux_up.transpose(1, 0, 2).sum(axis=-1),
        lw_flux_down=diagnostics_vmapped.lw_flux_down.transpose(1, 0, 2).sum(axis=-1),
        lw_heating_rate=tendencies_vmapped.longwave_heating.T,  # [ncols, nlev] -> [nlev, ncols]
        surface_sw_down=_column_vector(diagnostics_vmapped.surface_sw_down, ncols),  # Already [ncols]
        surface_lw_down=_column_vector(diagnostics_vmapped.surface_lw_down, ncols),
        surface_sw_up=_column_vector(diagnostics_vmapped.surface_sw_up, ncols),
        surface_lw_up=_column_vector(diagnostics_vmapped.surface_lw_up, ncols),
        toa_sw_up=_column_vector(diagnostics_vmapped.toa_sw_up, ncols),
        toa_lw_up=_column_vector(diagnostics_vmapped.toa_lw_up, ncols),
        toa_sw_down=_column_vector(diagnostics_vmapped.toa_sw_down, ncols)
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
        physics_data.echam_coords.lat * 180.0 / jnp.pi,
        physics_data.echam_coords.lon * 180.0 / jnp.pi,
    )
    latitudes, longitudes = lat.reshape(ncols), lon.reshape(ncols)

    # Solar geometry comes pre-baked on `forcing.solar` (populated by
    # `Model._get_step_fn_factory` ↔ `ForcingData.select(date)`). The
    # radiation scheme stays date-free.
    solar = forcing.solar

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

    # Chunked vmap: RRTMGP allocates ~150 intermediate arrays of shape
    # (ngpt, nlev) per column inside ``compute_heating_rate`` (gas
    # optics interpolation tables, planck source functions, optical
    # depth, working memory for the tridiagonal flux solver, etc.).
    # At T63L47 with ngpt=128 (LW), nlev=47, ncols=18432, vmapping
    # all columns at once costs ~67 GiB of peak memory and OOMs on
    # an 80 GiB A100. Splitting the vmap into ``n_chunks`` smaller
    # batches via ``lax.map`` linearises the work over chunks while
    # keeping vmap parallelism inside each chunk; peak memory drops
    # by roughly a factor of ``n_chunks``.
    #
    # Chunk-size sweep at T63L47 on a single 80 GiB A100:
    #   chunk=18432 (1 chunk):  OOM at 67 GiB peak
    #   chunk= 9216 (2 chunks): ~8.7 s/step (avg, 7200 s rad cache)
    #   chunk= 4608 (4 chunks): ~15.2 s/step
    # 2 chunks is ~74% faster than 4 because XLA needs less
    # rematerialization at lower memory pressure.
    #
    # The chunk size is auto-detected from the device's HBM by default
    # (``parameters.radiation.rrtmgp_chunk_size = 0``). The empirically-
    # measured per-cell cost at the current g128/g112 config is ~3.6 MB
    # at nlev=47 (linear in nlev for other resolutions). We use 50 % of
    # the device memory budget so XLA has working room — the rest is
    # rematerialisation buffers and other kernels' allocations. Override
    # via ``RadiationParameters(rrtmgp_chunk_size=N)`` if the auto pick
    # OOMs (e.g. shared GPUs) or if you want a fixed chunk count for
    # reproducible kernel launches.
    from jcm.physics.radiation.rrtmgp import chunk_budget as _rrtmgp_chunk_budget
    chunk_budget = _rrtmgp_chunk_budget(nlev)
    if ncols <= chunk_budget:
        chunk_size = ncols
    else:
        # Pick the smallest n_chunks ≥ ncols / chunk_budget such that
        # n_chunks divides ncols (so all chunks are equal size).
        n_chunks = -(-ncols // chunk_budget)  # ceil-div
        while ncols % n_chunks != 0:
            n_chunks += 1
        chunk_size = ncols // n_chunks
    n_chunks = ncols // chunk_size

    def _per_column_inputs():
        """Pack all vmap inputs as (n_chunks, chunk_size, ...)."""
        # ``(nz, ncols)`` arrays (where ``nz`` may be ``nlev`` or
        # ``nlev+1`` depending on full vs half levels) → reshape to
        # ``(nz, n_chunks, chunk_size)`` then transpose to
        # ``(n_chunks, chunk_size, nz)`` so the leading axis is the
        # chunk axis lax.map iterates over.
        def split_lev_first(a):
            nz = a.shape[0]
            return a.reshape(nz, n_chunks, chunk_size).transpose(1, 2, 0)

        def split_col(a):
            return a.reshape(n_chunks, chunk_size, *a.shape[1:])

        return dict(
            temperature=split_lev_first(state.temperature),
            specific_humidity=split_lev_first(state.specific_humidity),
            pressure_full=split_lev_first(physics_data.diagnostics.pressure_full),
            pressure_half=split_lev_first(physics_data.diagnostics.pressure_half),
            layer_thickness=split_lev_first(physics_data.diagnostics.layer_thickness),
            air_density=split_lev_first(physics_data.diagnostics.air_density),
            cloud_water=split_lev_first(cloud_water),
            cloud_ice=split_lev_first(cloud_ice),
            cloud_fraction=split_lev_first(cloud_fraction),
            surface_temperature=split_col(surface_temperature_col),
            surface_albedo_vis=split_col(surface_albedo_vis_col),
            surface_albedo_nir=split_col(surface_albedo_nir_col),
            surface_emissivity=split_col(surface_emissivity_col),
            latitudes=split_col(latitudes),
            longitudes=split_col(longitudes),
            ozone_vmr=split_lev_first(ozone_vmr),
            aerosol=physics_data.aerosol.copy(
                aod_profile=split_col(aerosol_data_for_vmap.aod_profile),
                ssa_profile=split_col(aerosol_data_for_vmap.ssa_profile),
                asy_profile=split_col(aerosol_data_for_vmap.asy_profile),
                cdnc_factor=split_col(aerosol_data_for_vmap.cdnc_factor),
                aod_total=split_col(aerosol_data_for_vmap.aod_total),
                aod_anthropogenic=split_col(aerosol_data_for_vmap.aod_anthropogenic),
                aod_background=split_col(aerosol_data_for_vmap.aod_background),
                Nccn=split_col(physics_data.aerosol.Nccn.reshape(ncols)),
                angstrom=split_col(aerosol_data_for_vmap.angstrom),
            ),
        )

    chunked_inputs = _per_column_inputs()

    def _vmap_one_chunk(chunk_inputs):
        """Vmap radiation_scheme_rrtmgp across one chunk of `chunk_size` columns."""
        return jax.vmap(
            radiation_scheme_rrtmgp,
            in_axes=(
                # temperature..layer_thickness, air_density..cloud_fraction:
                # all stored as (chunk_size, nlev) here, so column axis is 0
                # and the per-column profile is handled inside the function.
                0, 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,           # surface scalars
                None, 0, 0,           # solar, lat, lon
                None, 0, 0, None,     # parameters, aerosol, ozone, co2
            ),
            out_axes=(0, 0),
            axis_size=chunk_size,
        )(
            chunk_inputs['temperature'], chunk_inputs['specific_humidity'],
            chunk_inputs['pressure_full'], chunk_inputs['pressure_half'],
            chunk_inputs['layer_thickness'], chunk_inputs['air_density'],
            chunk_inputs['cloud_water'], chunk_inputs['cloud_ice'],
            chunk_inputs['cloud_fraction'],
            chunk_inputs['surface_temperature'], chunk_inputs['surface_albedo_vis'],
            chunk_inputs['surface_albedo_nir'], chunk_inputs['surface_emissivity'],
            solar, chunk_inputs['latitudes'], chunk_inputs['longitudes'],
            parameters.radiation, chunk_inputs['aerosol'],
            chunk_inputs['ozone_vmr'], co2_vmr,
        )

    # ``lax.map`` is sequential over the leading axis but JIT-compatible.
    # Outputs come back stacked on the leading axis as (n_chunks, chunk_size, ...).
    chunked_results = jax.lax.map(_vmap_one_chunk, chunked_inputs)
    tendencies_chunked, diagnostics_chunked = chunked_results

    # Re-merge chunk axis: (n_chunks, chunk_size, ...) → (ncols, ...).
    def merge(a):
        return a.reshape(ncols, *a.shape[2:])
    tendencies_vmapped = jax.tree_util.tree_map(merge, tendencies_chunked)
    diagnostics_vmapped = jax.tree_util.tree_map(merge, diagnostics_chunked)
    radiation_results = (tendencies_vmapped, diagnostics_vmapped)

    tendencies_vmapped, diagnostics_vmapped = radiation_results
    temperature_tendency = tendencies_vmapped.temperature_tendency.T

    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros((nlev, ncols)),
        v_wind=jnp.zeros((nlev, ncols)),
        temperature=temperature_tendency,
        specific_humidity=jnp.zeros((nlev, ncols)),
        tracers={},
    )

    # Per-gpoint flux profiles are summed over g-points inside the
    # vmapped per-column compute (rrtmgp.py:268), so the diagnostic
    # arrays here are shape (ncols, nlev+1) and need only a transpose
    # to (nlev+1, ncols).
    rad_out = RadiationData(
        # Use ``_column_vector(...)`` for scalars (handles single-column
        # vs global shape consistency). Fluxes are already
        # ``(ncols, nlev+1)`` here because the chunked-vmap RRTMGP path
        # sums per-gpoint inside the per-column compute (rrtmgp.py:268),
        # so a plain ``.T`` is enough — DO NOT add the ``transpose+sum``
        # pattern from the grey path, the per-band axis no longer exists
        # at this point.
        cos_zenith=_column_vector(diagnostics_vmapped.cos_zenith, ncols),
        surface_albedo_vis=_column_vector(diagnostics_vmapped.surface_albedo_vis, ncols),
        surface_albedo_nir=_column_vector(diagnostics_vmapped.surface_albedo_nir, ncols),
        surface_emissivity=_column_vector(diagnostics_vmapped.surface_emissivity, ncols),
        sw_flux_up=diagnostics_vmapped.sw_flux_up.T,
        sw_flux_down=diagnostics_vmapped.sw_flux_down.T,
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
        lw_flux_up=diagnostics_vmapped.lw_flux_up.T,
        lw_flux_down=diagnostics_vmapped.lw_flux_down.T,
        lw_heating_rate=tendencies_vmapped.longwave_heating.T,
        surface_sw_down=_column_vector(diagnostics_vmapped.surface_sw_down, ncols),
        surface_lw_down=_column_vector(diagnostics_vmapped.surface_lw_down, ncols),
        surface_sw_up=_column_vector(diagnostics_vmapped.surface_sw_up, ncols),
        surface_lw_up=_column_vector(diagnostics_vmapped.surface_lw_up, ncols),
        toa_sw_up=_column_vector(diagnostics_vmapped.toa_sw_up, ncols),
        toa_lw_up=_column_vector(diagnostics_vmapped.toa_lw_up, ncols),
        toa_sw_down=_column_vector(diagnostics_vmapped.toa_sw_down, ncols),
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
        physics_data.echam_coords.lat * 180.0 / jnp.pi,
        physics_data.echam_coords.lon * 180.0 / jnp.pi,
    )
    latitudes, longitudes = lat.reshape(ncols), lon.reshape(ncols)

    # Solar geometry comes pre-baked on `forcing.solar` (populated by
    # `Model._get_step_fn_factory` ↔ `ForcingData.select(date)`). The
    # radiation scheme stays date-free.
    solar = forcing.solar

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
        solar, latitudes, longitudes,
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
        cos_zenith=_column_vector(diagnostics_vmapped.cos_zenith, ncols),
        surface_albedo_vis=_column_vector(diagnostics_vmapped.surface_albedo_vis, ncols),
        surface_albedo_nir=_column_vector(diagnostics_vmapped.surface_albedo_nir, ncols),
        surface_emissivity=_column_vector(diagnostics_vmapped.surface_emissivity, ncols),
        sw_flux_up=diagnostics_vmapped.sw_flux_up.T,        # [ncols, nlev+1] -> [nlev+1, ncols]
        sw_flux_down=diagnostics_vmapped.sw_flux_down.T,
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
        lw_flux_up=diagnostics_vmapped.lw_flux_up.T,
        lw_flux_down=diagnostics_vmapped.lw_flux_down.T,
        lw_heating_rate=tendencies_vmapped.longwave_heating.T,
        surface_sw_down=_column_vector(diagnostics_vmapped.surface_sw_down, ncols),
        surface_lw_down=_column_vector(diagnostics_vmapped.surface_lw_down, ncols),
        surface_sw_up=_column_vector(diagnostics_vmapped.surface_sw_up, ncols),
        surface_lw_up=_column_vector(diagnostics_vmapped.surface_lw_up, ncols),
        toa_sw_up=_column_vector(diagnostics_vmapped.toa_sw_up, ncols),
        toa_lw_up=_column_vector(diagnostics_vmapped.toa_lw_up, ncols),
        toa_sw_down=_column_vector(diagnostics_vmapped.toa_sw_down, ncols),
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
    ncols = state.temperature.shape[1]

    # Extract fixed qc/qi tracers (with defaults if not present)
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))

    # Per-column land fraction selects between ECHAM's ocean and land
    # ``zdnoprc`` precip-zone thresholds inside the updraft.
    land_fraction = terrain.fmask.reshape(ncols)

    conv_results = jax.vmap(
        tiedtke_nordeng_convection,
        # dt and config are scalar (None); land_fraction is per-column (axis 0)
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None, 0),
        out_axes=(0, 0)  # Returns (ConvectionTendencies, ConvectionState) per column
    )(state.temperature, state.specific_humidity, pressure_levels, layer_thickness,
      air_density, state.u_wind, state.v_wind, qc, qi, dt, parameters.convection,
      land_fraction)
    
    # Unpack structured results directly (no tuple unpacking needed)
    conv_tendencies_all, conv_states_all = conv_results

    # Hard limit on the convective T tendency: 5 K / hr, applied symmetrically.
    # Healthy deep convection over the warmest tropical SSTs gives ~1 K/hr at
    # the most active level; the cap only fires when the column's parcel-vs-
    # environment energy balance has gone pathological. The companion
    # cloud-base mass-flux CFL cap in ``tiedtke_nordeng_convection`` (added
    # in the same branch) bounds the column-integrated mass flux but does
    # not contain per-level latent-heat spikes inside the updraft loop —
    # ECHAM bounds those via the per-level moist-adjustment limits in
    # ``mo_cuadjust.f90`` which we have not yet ported. Until that lands
    # this cap is the safety net.
    _DTDT_MAX = 5.0 / 3600.0  # K/s
    dt_conv_capped = jnp.clip(
        conv_tendencies_all.dtedt, -_DTDT_MAX, _DTDT_MAX,
    )

    physics_tendencies = PhysicsTendency(
        u_wind=conv_tendencies_all.dudt.T,
        v_wind=conv_tendencies_all.dvdt.T,
        temperature=dt_conv_capped.T,
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
def apply_cloud_fraction(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Run the ECHAM shallow cloud / condensation scheme.

    Emits the condensation tendencies (dtedt, dqdt, dqcdt, dqidt) and
    publishes post-condensation ``cloud_fraction``, ``qc``, ``qi`` and
    ``relative_humidity`` on ``physics_data`` for downstream microphysics
    terms to consume. Split from ``apply_clouds_and_microphysics`` so that
    the microphysics scheme (1M or 2M) can be swapped independently via
    ComposablePhysics.replace("clouds", ...).
    """
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    surface_pressure = physics_data.diagnostics.surface_pressure
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    cloud_config = parameters.clouds

    cloud_tend_all, cloud_state_all = jax.vmap(
        shallow_cloud_scheme,
        in_axes=(1, 1, 1, 1, 1, 0, None, None),
        out_axes=(0, 0),
    )(state.temperature, state.specific_humidity, pressure_levels,
      qc, qi, surface_pressure, dt, cloud_config)

    tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=cloud_tend_all.dtedt.T,
        specific_humidity=cloud_tend_all.dqdt.T,
        tracers={
            'qc': cloud_tend_all.dqcdt.T,
            'qi': cloud_tend_all.dqidt.T,
        },
    )

    cloud_data = physics_data.clouds.copy(
        cloud_fraction=cloud_state_all.cloud_fraction.T,
        qc=cloud_state_all.cloud_water.T,
        qi=cloud_state_all.cloud_ice.T,
    )
    diagnostics = physics_data.diagnostics.copy(
        relative_humidity=cloud_state_all.rel_humidity.T,
    )
    return tendencies, physics_data.copy(clouds=cloud_data, diagnostics=diagnostics)


@jit
def apply_microphysics_1m(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Run ECHAM 1-moment cloud microphysics.

    Consumes the post-condensation ``qc``, ``qi``, ``cloud_fraction`` that
    :func:`apply_cloud_fraction` wrote to ``physics_data.clouds`` — so this
    term must be composed after it.
    """
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    air_density = physics_data.diagnostics.air_density
    dz = physics_data.diagnostics.layer_thickness
    micro_config = parameters.microphysics

    qc_interim = physics_data.clouds.qc
    qi_interim = physics_data.clouds.qi
    cloud_fraction = physics_data.clouds.cloud_fraction

    base_cdnc = parameters.microphysics.base_cdnc
    cdnc_factor = physics_data.aerosol.cdnc_factor
    cdnc_m3 = jnp.ones_like(state.temperature) * base_cdnc * cdnc_factor[jnp.newaxis, :]
    droplet_number_per_kg = cdnc_m3 / air_density

    # Reverted from cloud_microphysics_column_sweep — that scheme's
    # Rotstayn rain evaporation creates a positive-feedback loop
    # (rain evaporates → moistens dry layer → Sundqvist condenses →
    # latent heat release → drives convection → more rain) that the
    # surface bisect identified as the dominant amplifier of the
    # day-7 NaN on T63L47 + real terrain. The per-level
    # `cloud_microphysics` discards rain each step (no propagating
    # flux, no inter-level evap coupling), which breaks the feedback
    # at the cost of microphysics fidelity. Tracked as follow-up
    # work — needs either ICON's RH-hysteresis bound on the evap
    # source or a tighter Newton solve so the evap stays bounded
    # under coupling with Sundqvist.
    micro_tend_all, micro_state_all = jax.vmap(
        cloud_microphysics,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),
        out_axes=(0, 0),
    )(state.temperature, state.specific_humidity, pressure_levels,
      qc_interim, qi_interim, cloud_fraction, air_density, dz,
      droplet_number_per_kg, dt, micro_config)

    tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=micro_tend_all.dtedt.T,
        specific_humidity=micro_tend_all.dqdt.T,
        tracers={
            'qc': micro_tend_all.dqcdt.T,
            'qi': micro_tend_all.dqidt.T,
        },
    )

    cloud_data = physics_data.clouds.copy(
        precip_rain=micro_state_all.precip_rain,
        precip_snow=micro_state_all.precip_snow,
        droplet_number=cdnc_m3,
    )
    return tendencies, physics_data.copy(clouds=cloud_data)


@jit
def apply_microphysics_2m(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Run ECHAM 2-moment cloud microphysics.

    Consumes the post-condensation ``qc``/``qi``/``cloud_fraction`` emitted
    by :func:`apply_cloud_fraction` and returns tendencies for the full 2M
    tracer set ``{qc, qi, qnc, qni, qr, qs}``. The orchestrator
    :func:`jcm.physics.clouds.lohmann_2m.cloud_microphysics_2m` chains the
    full ECHAM6 process list: warm precip (KK2000) + mixed-phase
    deposition/condensation + freezing-below-238K + DeMott(2010) INP
    mixed-phase freezing (placeholder for ECHAM's ``het_mxphase_freezing``
    which would need HAM aerosol modes — see #436) + WBF + cold precip +
    a top-down ``lax.scan`` over levels for sedimentation / melting /
    sublimation+evap / precip-flux accumulation, then ECHAM's
    ``update_tendencies_and_important_vars`` for the final tendency
    bookkeeping. Heterogeneous freezing is intentionally simplified to
    DeMott(2010) since JCM does not yet ingest a real IN field.
    """
    from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m

    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    air_density = physics_data.diagnostics.air_density
    layer_thickness = physics_data.diagnostics.layer_thickness
    tke = physics_data.vertical_diffusion.tke
    params_2m = parameters.microphysics_2m

    qc_interim = physics_data.clouds.qc
    qi_interim = physics_data.clouds.qi
    cloud_fraction = physics_data.clouds.cloud_fraction

    # Default any declared-but-missing tracers to zero.
    zeros = jnp.zeros_like(state.temperature)
    qnc = state.tracers.get('qnc', zeros)
    qni = state.tracers.get('qni', zeros)
    qr = state.tracers.get('qr', zeros)
    qs = state.tracers.get('qs', zeros)

    # Aerosol-activated CDNC floor from the MACv2-SP plume CCN
    # concentration via the SPA sublinear power-law (Lin et al. 2025;
    # #374). Output is per-level `(nlev, ncols)` in m^-3 — the column-
    # mean Nccn is broadcast to every level (vertical aerosol structure
    # is not resolved by the simple-plumes scheme). The fit's prefactor
    # and exponent come from `parameters.aerosol` so they remain
    # differentiable for calibration work.
    Nccn = physics_data.aerosol.Nccn  # (ncols,), units cm^-3
    activated_cdnc = spa_activated_cdnc(
        Nccn=Nccn[jnp.newaxis, :],
        cloud_fraction=cloud_fraction,
        prefactor=parameters.aerosol.spa_prefactor,
        exponent=parameters.aerosol.spa_exponent,
    )

    tend_all, surface_rain_flux, surface_snow_flux = jax.vmap(
        cloud_microphysics_2m,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),
        out_axes=(0, 0, 0),
    )(state.temperature, state.specific_humidity, pressure_levels,
      qc_interim, qi_interim, qnc, qni, qr, qs,
      cloud_fraction, air_density, layer_thickness, tke,
      activated_cdnc, dt, params_2m)

    tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=tend_all.dtedt.T,
        specific_humidity=tend_all.dqdt.T,
        tracers={
            'qc': tend_all.dqcdt.T,
            'qi': tend_all.dqidt.T,
            'qnc': tend_all.dqncdt.T,
            'qni': tend_all.dqnidt.T,
            'qr': tend_all.dqrdt.T,
            'qs': tend_all.dqsdt.T,
        },
    )
    # Stash the current-step qnc/qni as the tm1 state so the next call of
    # this term (or downstream update_tendencies_and_important_vars) can
    # read previous-step number concentrations. PhysicsData.clouds is
    # carried forward across timesteps in ComposableEchamPhysics.__call__.
    # Also expose the large-scale surface precip from the column scan as
    # diagnostic ``precip_rain``/``precip_snow`` (kg/m^2/s) — these are the
    # gravitational-fall flux at the bottom of the orchestrator's
    # top-down ``lax.scan``, summing autoconv + accretion + melt - evap
    # contributions across all levels.
    clouds_next = physics_data.clouds.copy(
        qnc_prev=qnc, qni_prev=qni,
        precip_rain=surface_rain_flux,
        precip_snow=surface_snow_flux,
    )
    return tendencies, physics_data.copy(clouds=clouds_next)


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

    # Surface tile fractions: 0=water, 1=sea-ice, 2=land. Derived from
    # boundary forcing the same way ``apply_surface`` does so the vdiff
    # path sees consistent fractions.
    nsfc_type = 3  # water, ice, land
    land_fraction = terrain.fmask.reshape(ncols)
    sea_ice_fraction = jnp.clip(forcing.sice_am.reshape(ncols), 0.0, 1.0 - land_fraction)
    water_fraction = 1.0 - land_fraction - sea_ice_fraction
    surface_fraction = jnp.zeros((ncols, nsfc_type))
    surface_fraction = surface_fraction.at[:, 0].set(water_fraction)
    surface_fraction = surface_fraction.at[:, 1].set(sea_ice_fraction)
    surface_fraction = surface_fraction.at[:, 2].set(land_fraction)

    # Per-tile surface temperature: boundary SST for water, the saline
    # freezing point ``ctfreez = 271.38 K`` (ECHAM ``iniphy.f90:71``)
    # capped by SST for ice, and ``forcing.stl_am`` for land.
    sst_col = physics_data.surface.surface_temperature.reshape(ncols)
    land_temp_col = forcing.stl_am.reshape(ncols)
    ctfreez = 271.38  # K, ECHAM ``iniphy.f90:71``
    ice_temp_col = jnp.where(sea_ice_fraction > 0.0,
                             jnp.minimum(sst_col, ctfreez),
                             sst_col)
    surface_temperature = jnp.stack([sst_col, ice_temp_col, land_temp_col], axis=1)
    # Roughness: same per-tile structure (water and ice ~1e-4 m, land 1e-2).
    roughness_length_col = physics_data.surface.roughness_length.reshape(ncols)
    roughness = jnp.stack([
        jnp.full(ncols, 1e-4),  # water
        jnp.full(ncols, 1e-3),  # sea ice (rougher than water)
        roughness_length_col,   # land (from boundary)
    ], axis=1)

    # Per-tile heat roughness z0h. ECHAM uses tile-specific forms:
    # open water gets ``exp(2 - 86·z0^0.375)`` (Charnock-derived), sea
    # ice keeps ``z0`` (rough = smooth in heat sense), and land uses
    # the JSBACH ``paz0lh`` from the boundary forcing — for which we
    # currently fall back to the same ``z0`` since no JSBACH coupling
    # is wired in yet.
    z0_water = jnp.exp(2.0 - 86.0 * roughness[:, 0] ** 0.375)
    z0_ice = roughness[:, 1]
    z0_land = roughness[:, 2]
    roughness_heat = jnp.stack([z0_water, z0_ice, z0_land], axis=1)

    # Per-tile surface wetness — fraction of saturation specific
    # humidity available at the surface for evaporation. Open water
    # and ice are fully saturated (1.0). Land wetness is taken from
    # the boundary soil-moisture field ``forcing.soilw_am``, which is
    # already a 0–1 fraction (1 = saturated soil).
    soilw_col = forcing.soilw_am.reshape(ncols)
    soilw_col = jnp.clip(soilw_col, 0.0, 1.0)
    surface_wetness = jnp.stack([
        jnp.ones(ncols),         # water — fully saturated
        jnp.ones(ncols),         # ice — saturated wrt ice
        soilw_col,               # land — soil-moisture fraction
    ], axis=1)

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
        roughness_heat=roughness_heat,
        surface_wetness=surface_wetness,
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
    
    # Update TKE
    new_tke = tke + dt * tke_tend
    new_tke = jnp.maximum(new_tke, 0.01)  # Minimum TKE

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
    # Per-tile surface temperatures are read directly from forcing below
    # (``ocean_temp``, ``land_temp``, ``ice_surface_temp``); the upstream-
    # blended ``physics_data.surface.surface_temperature`` is no longer
    # consulted here — see comment by ``ocean_temp`` for the reason.

    # Surface tile fractions: water (0), sea ice (1), land (2).
    # Sea ice fraction is taken from prescribed boundary conditions and
    # constrained to the non-land area so that fractions sum to exactly 1.
    nsfc_type = 3
    surface_fractions = jnp.zeros((ncols, nsfc_type))
    land_fraction = terrain.fmask.reshape((ncols,))
    sea_ice_fraction = jnp.clip(forcing.sice_am.reshape((ncols,)), 0.0, 1.0 - land_fraction)
    water_fraction = 1.0 - land_fraction - sea_ice_fraction
    surface_fractions = surface_fractions.at[:, 0].set(water_fraction)
    surface_fractions = surface_fractions.at[:, 1].set(sea_ice_fraction)
    surface_fractions = surface_fractions.at[:, 2].set(land_fraction)

    # Per-tile surface temperatures: boundary SST for ocean, the saline
    # freezing point (``ctfreez = 271.38 K``, ECHAM ``iniphy.f90:71``)
    # capped by SST for sea ice, and ``forcing.stl_am`` for land. Sea
    # ice uses min(SST, ctfreez) because the underlying ocean caps the
    # ice surface temperature physically.
    #
    # Read ``ocean_temp`` and ``land_temp`` straight from the forcing rather
    # than the upstream-blended ``physics_data.surface.surface_temperature``,
    # which is snapped to one-or-the-other via ``where(fmask>0.5)`` in
    # ``EchamForcing`` and would feed the wrong T into the minority tile
    # (e.g. the 40% ocean fraction of a fmask=0.6 cell would otherwise
    # use ``stl_am`` instead of ``sst``).
    ocean_temp = forcing.sea_surface_temperature.reshape(ncols)
    ctfreez = 271.38  # K, ECHAM ``iniphy.f90:71`` saline-water freezing
    # ``stl_am`` is the JSBACH land surface temperature climatology
    # (``surf_temp`` from ``ic_land_soil_T63GR15_*.nc``), already at the
    # model's orography — no lapse correction needed. (An earlier workaround
    # subtracted 6.5 K/km · orog because the bundled BCs used ``stl ≈ sst``
    # extrapolated over land — see ``utils/convert_echam_bc.py`` for the
    # path that picks the right field.)
    land_temp = forcing.stl_am.reshape(ncols)
    ice_surface_temp = jnp.where(sea_ice_fraction > 0.0,
                                 jnp.minimum(ocean_temp, ctfreez),
                                 ocean_temp)
    ice_temp = jnp.repeat(ice_surface_temp[:, jnp.newaxis], 2, axis=1)  # 2 ice layers
    soil_temp = jnp.repeat(land_temp[:, jnp.newaxis], 4, axis=1)         # 4 soil layers
    
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

    # The surface-flux divergence at the bottom level is a linear relaxation
    # toward the surface value with timescale ``dz_sfc / K`` (K = exchange
    # velocity in m/s). An *explicit* time step of size ``dt`` is unstable
    # whenever ``K * dt / dz_sfc > 2`` — and over rough terrain at the
    # ECHAM-tuned exchange coefficients this CFL is easily violated, with
    # the wind flipping sign each step until the column blows up. ECHAM
    # itself avoids this by handling the surface as an implicit BC of the
    # vdiff tridiagonal solve. JCM's explicit pipeline can't do that
    # directly, but we can damp each explicit tendency by the same factor
    # an implicit Euler step would — ``1 / (1 + K*dt/dz_sfc)``. This is
    # exact for the simple linear-relaxation form, recovers the explicit
    # tendency in the small-K*dt limit, and is unconditionally stable.
    #
    # ``surface_fractions`` and ``exchange_coeff_*`` are per-tile (ocean,
    # ice, land); the grid-box-mean exchange velocity is the area-weighted
    # sum.
    ch_grid = jnp.sum(surface_fractions * exchange_coeff_heat, axis=1)
    cm_grid = jnp.sum(surface_fractions * exchange_coeff_momentum, axis=1)
    ce_grid = jnp.sum(surface_fractions * exchange_coeff_moisture, axis=1)
    imp_heat = 1.0 / (1.0 + ch_grid * dt / dz_sfc)
    imp_mom = 1.0 / (1.0 + cm_grid * dt / dz_sfc)
    imp_moist = 1.0 / (1.0 + ce_grid * dt / dz_sfc)

    # Surface flux tendencies (applied to lowest level only).
    temp_tend_sfc = imp_heat * sensible_heat / (rho_sfc * physical_constants.cp * dz_sfc)
    qv_tend_sfc = imp_moist * evaporation / (rho_sfc * dz_sfc)
    u_tend_sfc = imp_mom * (-tau_u) / (rho_sfc * dz_sfc)
    v_tend_sfc = imp_mom * (-tau_v) / (rho_sfc * dz_sfc)
    
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
def apply_simple_gwd(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply the simple monochromatic GWD scheme (cheap fallback)."""
    diag = physics_data.diagnostics
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv

    # Placeholder std-dev of sub-grid orography (200 m). The simple
    # scheme keeps a single fixed value; the real Lott-Miller SSO
    # scheme uses the per-column ``terrain.orostd`` instead.
    h_std = jnp.ones(ncols) * 200.0

    tend, _state = jax.vmap(
        simple_gwd,
        in_axes=(1, 1, 1, 1, 1, 1, 0, None, None),
        out_axes=(0, 0),
    )(state.u_wind, state.v_wind, state.temperature,
      diag.pressure_full, diag.height_full, diag.air_density,
      h_std, dt, parameters.simple_gwd)

    physics_tendencies = PhysicsTendency(
        u_wind=tend.dudt.T,
        v_wind=tend.dvdt.T,
        temperature=tend.dtedt.T,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={},
    )
    return physics_tendencies, physics_data


@jit
def apply_hines(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply the Hines (1997) doppler-spread spectral non-orographic GWD."""
    diag = physics_data.diagnostics
    # Layer mass per unit area Δp / g, full-level array.
    layer_mass = (diag.pressure_half[1:, :]
                  - diag.pressure_half[:-1, :]) / physical_constants.grav

    tend, _state = jax.vmap(
        lambda *a: hines_gwd(*a, parameters.hines),
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1),
        out_axes=(0, 0),
    )(diag.pressure_half, diag.pressure_full, diag.height_half,
      diag.air_density, layer_mass,
      state.temperature, state.u_wind, state.v_wind)

    # Convert energy dissipation (W/kg) to a temperature tendency.
    dt_temperature = tend.dissip / physical_constants.cpd

    physics_tendencies = PhysicsTendency(
        u_wind=tend.dudt.T,
        v_wind=tend.dvdt.T,
        temperature=dt_temperature.T,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={},
    )
    return physics_tendencies, physics_data


@jit
def apply_sso(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply the Lott-Miller (1997) sub-grid orographic GW drag."""
    diag = physics_data.diagnostics
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    layer_mass = (diag.pressure_half[1:, :]
                  - diag.pressure_half[:-1, :]) / physical_constants.grav

    # Coriolis is only read by the (unported) mountain-lift branch.
    coriolis = jnp.zeros((ncols,))

    def _sso_one_col(pressure_full, pressure_half, layer_mass_col,
                     temperature, u_wind, v_wind, height_full,
                     surface_height, mean_orography, orography_std,
                     orography_slope, orography_anisotropy,
                     orography_orientation, peak_elevation,
                     valley_elevation, coriolis_col, land_fraction):
        return sso_drag(
            jnp.asarray(dt), coriolis_col, height_full, surface_height,
            pressure_half, pressure_full, layer_mass_col,
            temperature, u_wind, v_wind,
            mean_orography, orography_std, orography_slope,
            orography_anisotropy, orography_orientation,
            peak_elevation, valley_elevation,
            land_fraction, parameters.sso,
            nktopg=1, ntop=1,
        )

    tend, _state = jax.vmap(
        _sso_one_col,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )(diag.pressure_full, diag.pressure_half, layer_mass,
      state.temperature, state.u_wind, state.v_wind, diag.height_full,
      terrain.orog.reshape(-1), terrain.orog.reshape(-1),
      terrain.orostd.reshape(-1), terrain.orosig.reshape(-1),
      terrain.orogam.reshape(-1), terrain.orothe.reshape(-1),
      terrain.oropic.reshape(-1), terrain.oroval.reshape(-1),
      coriolis, terrain.fmask.reshape(-1))

    dt_temperature = tend.dissip / physical_constants.cpd

    physics_tendencies = PhysicsTendency(
        u_wind=tend.dudt.T,
        v_wind=tend.dvdt.T,
        temperature=dt_temperature.T,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={},
    )
    return physics_tendencies, physics_data

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