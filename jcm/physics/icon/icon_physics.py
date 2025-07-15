"""
Main ICON Physics class for JAX-GCM

This module contains the main IconPhysics class that orchestrates the 
ICON atmospheric physics parameterizations. It follows the same pattern
as SpeedyPhysics but implements the ICON physics suite.

Date: 2025-01-09
"""

import jax
from jax import jit
import jax.numpy as jnp
import tree_math
from collections import abc
from typing import Callable, Tuple, Optional
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.date import DateData
from jcm.physics.icon.constants import physical_constants

# Import physics modules (will be implemented progressively)
from jcm.physics.icon.radiation import radiation_scheme
from jcm.physics.icon.icon_physics_data import RadiationData
from jcm.physics.icon.convection import tiedtke_nordeng_convection
from jcm.physics.icon.clouds import shallow_cloud_scheme, cloud_microphysics
from jcm.physics.icon.parameters import Parameters
from jcm.physics.icon.vertical_diffusion import vertical_diffusion_scheme
from jcm.physics.icon.surface import surface_physics_step, initialize_surface_state
from jcm.physics.icon.surface.surface_types import SurfaceState, AtmosphericForcing
from jcm.physics.icon.gravity_waves import gravity_wave_drag
# from jcm.physics.icon.chemistry import chemistry_tendencies
from jcm.physics.icon.aerosol.simple_aerosol import get_simple_aerosol
from jcm.physics.icon.icon_physics_data import PhysicsData

class IconPhysics(Physics):
    """
    ICON atmospheric physics implementation for JAX-GCM
    
    This class implements the ICON physics suite including:
    - Radiation (shortwave and longwave)
    - Convection (Tiedtke-Nordeng scheme)
    - Large-scale cloud microphysics
    - Vertical diffusion and boundary layer
    - Surface fluxes and land model
    - Gravity wave drag
    - Simple chemistry schemes
    """
    
    def __init__(self, 
                 write_output: bool = True,
                 checkpoint_terms: bool = True,
                 parameters: Optional[Parameters] = None):
        """
        Initialize the ICON physics.
        
        Args:
            write_output: Whether to write physics output to predictions
            checkpoint_terms: Whether to checkpoint physics terms
            parameters: Optional physics parameters (uses defaults if None)
        """
        self.write_output = write_output
        self.checkpoint_terms = checkpoint_terms
        
        # Store parameters
        self.parameters = parameters or Parameters.default()
        
        # Build list of physics terms
        self.terms = [
            _prepare_common_physics_state,
            get_simple_aerosol,  # Aerosol before radiation
            apply_radiation,     # Radiation early for surface fluxes
            apply_convection, 
            apply_clouds,
            apply_microphysics,
            apply_surface,       # Surface after radiation
            apply_vertical_diffusion, 
            apply_gravity_waves
        ]
    
    def compute_tendencies(
        self,
        state: PhysicsState,
        boundaries: BoundaryData,
        geometry: Geometry,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
        """
        Compute the physical tendencies given the current state and data structs. Loops through the Speedy physics terms, accumulating the tendencies.

        Args:
            state: Current state variables
            parameters: Parameters object
            boundaries: Boundary data
            geometry: Geometry data
            date: Date data

        Returns:
            Physical tendencies in PhysicsTendency format
            Object containing physics data (PhysicsData format)
        """
        physics_data = PhysicsData.zeros(
            geometry.nodal_shape[1:],
            geometry.nodal_shape[0],
            date=date
        )

        # Initialize zero tendencies with tracer tendencies
        tracer_tends = {name: jnp.zeros_like(tracer) for name, tracer in state.tracers.items()}
        tendencies = PhysicsTendency.zeros(state.temperature.shape, tracers=tracer_tends)

        # Get array dimensions for vectorization
        nlev, nlat, nlon = state.temperature.shape
        ncols = nlat * nlon
        
        # OPTIMIZATION: Single reshape operation using tree_map for TPU efficiency
        vectorized_state = self._reshape_state_to_columns(state, nlev, ncols)
        
        # OPTIMIZATION: Accumulate tendencies in column format for TPU efficiency
        # Initialize column-format accumulators
        accumulated_tendencies = self._initialize_column_tendencies(nlev, ncols, state.tracers)
        
        # Apply physics terms with column-based accumulation
        for term in self.terms:
            if self.checkpoint_terms:
                term = jax.checkpoint(term)
            
            # Apply term to vectorized state (returns column format)
            term_tendency, physics_data = term(
                vectorized_state, physics_data, self.parameters, boundaries, geometry
            )
            
            # OPTIMIZATION: Direct accumulation in column format (no reshape)
            accumulated_tendencies = self._accumulate_column_tendencies(
                accumulated_tendencies, term_tendency
            )
        
        # OPTIMIZATION: Single reshape to 3D only at the very end
        tendencies = self._reshape_tendencies_to_3d(accumulated_tendencies, nlev, nlat, nlon)
        
        return tendencies, physics_data
    
    def _reshape_state_to_columns(self, state: PhysicsState, nlev: int, ncols: int) -> PhysicsState:
        """
        TPU-optimized reshape using single tree_map operation
        
        This creates one XLA operation instead of multiple reshapes, 
        which is crucial for TPUv4 performance at T85 resolution.
        """
        def reshape_field(field):
            if field.ndim == 3:  # [nlev, nlat, nlon] → [nlev, ncols]
                return field.reshape(nlev, ncols)
            elif field.ndim == 2:  # [nlat, nlon] → [ncols]
                return field.reshape(ncols)
            else:
                return field  # Leave scalars unchanged
        
        # Single tree_map operation handles all main fields efficiently
        reshaped_fields = jax.tree_map(reshape_field, {
            'u_wind': state.u_wind,
            'v_wind': state.v_wind,
            'temperature': state.temperature,
            'specific_humidity': state.specific_humidity,
            'geopotential': state.geopotential,
            'surface_pressure': state.surface_pressure,
        })
        
        # Handle tracers separately to maintain dict structure
        vectorized_tracers = {
            name: tracer.reshape(nlev, ncols) 
            for name, tracer in state.tracers.items()
        }
        
        return PhysicsState(
            **reshaped_fields,
            tracers=vectorized_tracers
        )
    
    def _initialize_column_tendencies(self, nlev: int, ncols: int, tracers: dict) -> dict:
        """
        Initialize tendency accumulators in column format
        
        Using dict avoids repeated PhysicsTendency object creation during accumulation
        """
        return {
            'u_wind': jnp.zeros((nlev, ncols)),
            'v_wind': jnp.zeros((nlev, ncols)),
            'temperature': jnp.zeros((nlev, ncols)),
            'specific_humidity': jnp.zeros((nlev, ncols)),
            'tracers': {name: jnp.zeros((nlev, ncols)) for name in tracers.keys()}
        }
    
    def _accumulate_column_tendencies(self, accumulated: dict, new_tendency: PhysicsTendency) -> dict:
        """
        Efficiently accumulate tendencies in column format
        
        Avoids object creation and intermediate arrays for optimal TPU performance
        """
        return {
            'u_wind': accumulated['u_wind'] + new_tendency.u_wind,
            'v_wind': accumulated['v_wind'] + new_tendency.v_wind,
            'temperature': accumulated['temperature'] + new_tendency.temperature,
            'specific_humidity': accumulated['specific_humidity'] + new_tendency.specific_humidity,
            'tracers': {
                name: accumulated['tracers'][name] + new_tendency.tracers.get(name, 0.0)
                for name in accumulated['tracers'].keys()
            }
        }
    
    def _reshape_tendencies_to_3d(self, tendencies: dict, nlev: int, nlat: int, nlon: int) -> PhysicsTendency:
        """
        Final reshape to 3D format - single operation at the end
        
        This is the only reshape back to 3D, done once at the very end
        """
        def reshape_to_3d(field):
            if field.ndim == 2:  # [nlev, ncols] → [nlev, nlat, nlon]
                return field.reshape(nlev, nlat, nlon)
            else:
                return field
        
        # Single tree_map for all main fields
        reshaped_main = jax.tree_map(reshape_to_3d, {
            'u_wind': tendencies['u_wind'],
            'v_wind': tendencies['v_wind'],
            'temperature': tendencies['temperature'],
            'specific_humidity': tendencies['specific_humidity'],
        })
        
        # Reshape tracers
        reshaped_tracers = {
            name: field.reshape(nlev, nlat, nlon)
            for name, field in tendencies['tracers'].items()
        }
        
        return PhysicsTendency(
            **reshaped_main,
            tracers=reshaped_tracers
        )

@jit
def _prepare_common_physics_state(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:

    """
    Prepare common physics variables that are used by multiple physics modules.
    
    This reduces code duplication by computing pressure levels, heights, air density,
    and other commonly needed variables once for all physics modules.
    
    Args:
        state: Physics state variables (already in 2D format [nlev, ncols])
        boundaries: Boundary conditions
        geometry: Model geometry
        
    Returns:
        Dictionary with common physics variables
    """
    p0 = physical_constants.p0
    
    # Calculate pressure levels from surface pressure and sigma coordinates
    surface_pressure = state.surface_pressure * p0  # Convert to Pa
    sigma_levels = geometry.fsg  # sigma coordinates at level centers
    pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
    
    # Calculate pressure at interfaces (half levels)
    sigma_half = geometry.hsg
    pressure_half = sigma_half[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
    
    # Convert geopotential to height
    height_levels = state.geopotential / physical_constants.grav
    
    # Calculate height at interfaces (approximate using hydrostatic)
    dz = jnp.diff(height_levels, axis=0, prepend=height_levels[0:1, :] + 1000.0)
    height_half = jnp.cumsum(dz, axis=0) - dz
    
    # Calculate air density
    rho = pressure_levels / (physical_constants.rd * state.temperature)
    
    # Calculate layer thickness
    dp = jnp.diff(pressure_half, axis=0)
    dz_full = -dp / (rho * physical_constants.grav)
    
    # Calculate relative humidity
    es = 611.2 * jnp.exp(17.67 * (state.temperature - 273.15) / (state.temperature - 29.65))
    e = state.specific_humidity * pressure_levels / (0.622 + 0.378 * state.specific_humidity)
    rel_humidity = e / es

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

    zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return zero_tendencies, physics_data.copy(diagnostics=diagnostic_data)

# Physics term methods
@jit
def apply_radiation(state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:

    """Apply radiation heating rates"""
    p0 = physical_constants.p0
    
    # Note: state is already in 2D format [nlev, ncols] from compute_tendencies
    nlev, ncols = state.temperature.shape
    
    # Get date information for solar calculations
    date = physics_data.date
    day_of_year = date.day_of_year if hasattr(date, 'day_of_year') else 172.0  # Default to summer
    seconds_since_midnight = date.seconds_since_midnight if hasattr(date, 'seconds_since_midnight') else 43200.0  # Default to noon
    
    # Get lat/lon from geometry - for now use dummy values
    # In real implementation, these would come from geometry for each column
    latitudes = jnp.zeros(ncols)  # Equator for now
    longitudes = jnp.linspace(-180, 180, ncols)  # Span globe
    
    # Get cloud properties from tracers and previous physics
    cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    # Cloud fraction needs to be reshaped to 2D if it's 3D
    cloud_fraction = physics_data.clouds.cloud_fraction
    if cloud_fraction.ndim == 3:
        nlev_cf, nlat, nlon = cloud_fraction.shape
        cloud_fraction = cloud_fraction.reshape(nlev_cf, nlat * nlon)

    # Define single column radiation function
    def apply_radiation_to_column(temp_col, humid_col, ps_col, geopot_col,
                                qc_col, qi_col, cf_col, lat, lon):
        """Apply radiation to single column"""
        
        tendencies, diagnostics = radiation_scheme(
            temperature=temp_col,
            specific_humidity=humid_col,
            surface_pressure=ps_col,  # Already normalized
            geopotential=geopot_col,
            cloud_water=qc_col,
            cloud_ice=qi_col,
            cloud_fraction=cf_col,
            day_of_year=day_of_year,
            seconds_since_midnight=seconds_since_midnight,
            latitude=lat,
            longitude=lon,
            parameters=parameters.radiation
        )
        
        # Return temperature tendency and key diagnostics
        return tendencies, diagnostics
    
    # Apply radiation using vmap
    results = jax.vmap(
        apply_radiation_to_column,
        in_axes=(1, 1, 0, 1, 1, 1, 1, 0, 0),  # ps, lat, lon are 1D
        out_axes=0  # Map over the first (column) dimension of outputs
    )(state.temperature, state.specific_humidity, state.surface_pressure,
        state.geopotential, cloud_water, cloud_ice, cloud_fraction,
        latitudes, longitudes)
    
    # Unpack the vmapped results
    # results is a tuple of (RadiationTendencies, RadiationData) with column dimension first
    tendencies_vmapped, diagnostics_vmapped = results
    
    # Extract temperature tendencies and transpose to [nlev, ncols]
    temperature_tendency = tendencies_vmapped.temperature_tendency.T
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),  # No wind tendencies from radiation
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=temperature_tendency,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={}
    )
    
    # Reconstruct RadiationData from vmapped diagnostics
    # Most fields need to be transposed from [ncols, ...] to [..., ncols]
    rad_out = RadiationData(
        cos_zenith=diagnostics_vmapped.cos_zenith.squeeze(),  # [ncols, 1] -> [ncols]
        sw_flux_up=diagnostics_vmapped.sw_flux_up.transpose(1, 0, 2),  # [ncols, nlev+1, nbands] -> [nlev+1, ncols, nbands]
        sw_flux_down=diagnostics_vmapped.sw_flux_down.transpose(1, 0, 2),
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,  # [ncols, nlev] -> [nlev, ncols]
        lw_flux_up=diagnostics_vmapped.lw_flux_up.transpose(1, 0, 2),
        lw_flux_down=diagnostics_vmapped.lw_flux_down.transpose(1, 0, 2),
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
def apply_convection(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply Tiedtke-Nordeng convection scheme with fixed qc/qi transport"""
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    height_levels = physics_data.diagnostics.height_full
    
    # Extract fixed qc/qi tracers (with defaults if not present)
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    # SIMPLIFIED: Direct vmap of tiedtke_nordeng_convection (no wrapper needed)
    conv_results = jax.vmap(
        tiedtke_nordeng_convection,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (ConvectionTendencies, ConvectionState) per column
    )(state.temperature, state.specific_humidity, pressure_levels, height_levels, 
        state.u_wind, state.v_wind, qc, qi, dt, parameters.convection)
    
    # Unpack structured results directly (no tuple unpacking needed)
    conv_tendencies_all, conv_states_all = conv_results
    
    # SIMPLIFIED: Extract tendencies from structured result (transpose to [nlev, ncols])
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

@jit
def apply_clouds(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply shallow cloud scheme with microphysics"""
    nlev, ncols = state.temperature.shape
    
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    surface_pressure = physics_data.diagnostics.surface_pressure
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    # Get cloud configuration from parameters
    cloud_config = parameters.clouds
    
    # SIMPLIFIED: Direct vmap of shallow_cloud_scheme (no wrapper needed)
    cloud_results = jax.vmap(
        shallow_cloud_scheme,
        in_axes=(1, 1, 1, 1, 1, 0, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (CloudTendencies, CloudState) per column
    )(state.temperature, state.specific_humidity, pressure_levels,
        qc, qi, surface_pressure, dt, cloud_config)
    
    # Unpack structured results directly
    cloud_tendencies_all, cloud_states_all = cloud_results
    
    # SIMPLIFIED: Extract tendencies from structured result (transpose to [nlev, ncols])
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),  # No wind tendencies from clouds
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=cloud_tendencies_all.dtedt.T,
        specific_humidity=cloud_tendencies_all.dqdt.T,
        tracers={
            'qc': cloud_tendencies_all.dqcdt.T,
            'qi': cloud_tendencies_all.dqidt.T
        }
    )
    
    # Update physics data with cloud diagnostics
    cloud_data = physics_data.clouds.copy(
        cloud_fraction=cloud_states_all.cloud_fraction.T,
        precip_rain=cloud_tendencies_all.rain_flux,  # 1D per column
        precip_snow=cloud_tendencies_all.snow_flux   # 1D per column
    )
    
    diagnostics = physics_data.diagnostics.copy(
        relative_humidity=cloud_states_all.rel_humidity.T,
    )

    updated_physics_data = physics_data.copy(clouds=cloud_data, 
                                             diagnostics=diagnostics)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_microphysics(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply cloud microphysics scheme"""
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    cloud_fraction = physics_data.clouds.cloud_fraction
    air_density = physics_data.diagnostics.air_density
    dz = physics_data.diagnostics.layer_thickness

    # Extract fixed cloud water and ice tracers only
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    # Droplet number concentration (simple profile)
    droplet_number = jnp.ones_like(state.temperature) * 100e6  # 100 per cm³
    
    # Get microphysics configuration
    micro_config = parameters.microphysics
    
    # SIMPLIFIED: Direct vmap of cloud_microphysics (no wrapper needed)
    micro_results = jax.vmap(
        cloud_microphysics,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (MicrophysicsTendencies, MicrophysicsState) per column
    )(state.temperature, state.specific_humidity, pressure_levels,
        qc, qi, cloud_fraction, air_density, dz, droplet_number, dt, micro_config)
    
    # Unpack structured results directly
    micro_tendencies_all, micro_states_all = micro_results
    
    # SIMPLIFIED: Extract tendencies from structured result (transpose to [nlev, ncols])
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=micro_tendencies_all.dtedt.T,
        specific_humidity=micro_tendencies_all.dqdt.T,
        tracers={
            'qc': micro_tendencies_all.dqcdt.T,
            'qi': micro_tendencies_all.dqidt.T
        }
    )
    
    # Update physics data
    micro_data = physics_data.clouds.copy(
        precip_rain=micro_states_all.precip_rain,  # 1D per column
        precip_snow=micro_states_all.precip_snow,  # 1D per column
        droplet_number=droplet_number
    )
    
    updated_physics_data = physics_data.copy(clouds=micro_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_vertical_diffusion(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply vertical diffusion and boundary layer physics"""
    from jcm.physics.icon.vertical_diffusion import prepare_vertical_diffusion_state, vertical_diffusion_column
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    pressure_half = physics_data.diagnostics.pressure_half
    height_levels = physics_data.diagnostics.height_full
    height_half = physics_data.diagnostics.height_half
    
    # Initialize prognostic variables if not present
    tke = physics_data.vertical_diffusion.tke if hasattr(physics_data.vertical_diffusion, 'tke') else jnp.ones((nlev, ncols)) * 0.1
    thv_variance = physics_data.vertical_diffusion.thv_variance if hasattr(physics_data.vertical_diffusion, 'thv_variance') else jnp.zeros((nlev, ncols))
    
    # Surface properties (simplified - should come from boundaries)
    nsfc_type = 3  # water, ice, land
    surface_fraction = jnp.zeros((ncols, nsfc_type))
    surface_fraction = surface_fraction.at[:, 2].set(1.0)  # All land for now
    
    # Get surface properties from boundaries
    surface_temp = boundaries.surface_temperature if hasattr(boundaries, 'surface_temperature') else state.temperature[-1, :]
    roughness_length = boundaries.roughness_length if hasattr(boundaries, 'roughness_length') else jnp.full(ncols, 0.001)

    surface_temperature = jnp.repeat(surface_temp[:, jnp.newaxis], nsfc_type, axis=1)
    roughness = jnp.repeat(roughness_length[:, jnp.newaxis], nsfc_type, axis=1)
    
    # Ocean currents (zero for now)
    ocean_u = jnp.zeros(ncols)
    ocean_v = jnp.zeros(ncols)
    
    # Extract fixed qc/qi tracers for vertical diffusion
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    # Define single column vertical diffusion function
    def apply_vdiff_to_column(col_idx):
        """Apply vertical diffusion to a single column"""        
        # Prepare state for this column
        vdiff_state = prepare_vertical_diffusion_state(
            u=state.u_wind[:, col_idx],
            v=state.v_wind[:, col_idx],
            temperature=state.temperature[:, col_idx],
            qv=state.specific_humidity[:, col_idx],
            qc=qc[:, col_idx],
            qi=qi[:, col_idx],
            pressure_full=pressure_levels[:, col_idx],
            pressure_half=pressure_half[:, col_idx],
            geopotential=state.geopotential[:, col_idx],
            height_full=height_levels[:, col_idx],
            height_half=height_half[:, col_idx],
            surface_temperature=surface_temperature[col_idx, :],
            surface_fraction=surface_fraction[col_idx, :],
            roughness_length=roughness[col_idx, :],
            ocean_u=ocean_u[col_idx],
            ocean_v=ocean_v[col_idx],
            tke=tke[:, col_idx],
            thv_variance=thv_variance[:, col_idx],
        )
        
        # Compute vertical diffusion
        tendencies, diagnostics = vertical_diffusion_column(
            vdiff_state, parameters.vertical_diffusion, dt
        )
        
        return (
            tendencies.u_tend,
            tendencies.v_tend,
            tendencies.temp_tend,
            tendencies.qv_tend,
            tendencies.qc_tend,
            tendencies.qi_tend,
            tendencies.tke_tend,
            diagnostics.exchange_coeff_momentum,
            diagnostics.exchange_coeff_heat,
            diagnostics.pbl_height,
            diagnostics.surface_friction_velocity,
            diagnostics.surface_buoyancy_flux
        )
    
    # Apply vertical diffusion using vmap
    results = jax.vmap(apply_vdiff_to_column)(jnp.arange(ncols))
    
    # Unpack results
    (u_tend, v_tend, temp_tend, qv_tend, qc_tend, qi_tend, tke_tend,
        km, kh, pbl_height, u_star, b_flux) = results
    
    # Transpose results back to [nlev, ncols]
    u_tend = u_tend.T
    v_tend = v_tend.T
    temp_tend = temp_tend.T
    qv_tend = qv_tend.T
    qc_tend = qc_tend.T
    qi_tend = qi_tend.T
    tke_tend = tke_tend.T
    km = km.T
    kh = kh.T
    
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
    vdiff_data = physics_data.vertical_diffusion.copy(
        tke=new_tke,
        thv_variance=thv_variance,  # Not updated yet
        exchange_coeff_momentum=km,
        exchange_coeff_heat=kh,
        pbl_height=pbl_height,
        surface_friction_velocity=u_star,
        surface_buoyancy_flux=b_flux,
    )
    
    updated_physics_data = physics_data.copy(vertical_diffusion=vdiff_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_surface(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply surface physics and calculate surface fluxes"""
    from jcm.physics.icon.surface.surface_types import AtmosphericForcing
    from jcm.physics.icon.surface import initialize_surface_state
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    # Get surface properties from boundaries
    surface_temp = boundaries.surface_temperature if hasattr(boundaries, 'surface_temperature') else state.temperature[-1, :]

    # Initialize surface state (simplified)
    # In reality, this should come from the model's surface state
    nsfc_type = 3  # Fixed value: water, ice, land
    surface_fractions = jnp.zeros((ncols, nsfc_type))
    surface_fractions = surface_fractions.at[:, 2].set(1.0)  # All land for now (index 2)
    
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
    
    # Create atmospheric forcing for all columns
    # Initialize exchange coefficients with dummy values for now
    nsfc_type = 3
    dummy_exchange = jnp.ones((ncols, nsfc_type)) * 0.001  # Small exchange coefficient
    
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
        exchange_coeff_heat=dummy_exchange,
        exchange_coeff_moisture=dummy_exchange,
        exchange_coeff_momentum=dummy_exchange
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
    
    # Layer thickness at surface (approximate)
    dp_sfc = pressure_levels[-1, :] - pressure_levels[-2, :]
    dz_sfc = dp_sfc / (rho_sfc * physical_constants.grav)
    
    # Surface flux tendencies (applied to lowest level only)
    temp_tend_sfc = sensible_heat / (rho_sfc * physical_constants.cp * dz_sfc)
    qv_tend_sfc = (evaporation / 2.45e6) / (rho_sfc * dz_sfc)  # Latent heat = 2.45 MJ/kg
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
    surface_data = physics_data.surface.copy(
        sensible_heat_flux=sensible_heat,
        latent_heat_flux=latent_heat,
        momentum_flux_u=tau_u,
        momentum_flux_v=tau_v,
        evaporation_flux=evaporation,
        surface_temp_tendency=surf_temp_tend,
        exchange_coeff_heat=ch,
        exchange_coeff_momentum=cm,
        surface_resistance=resistance,
    )
    
    updated_physics_data = physics_data.copy(surface=surface_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_gravity_waves(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply gravity wave drag"""
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    height_levels = physics_data.diagnostics.height_full
    
    # Need orography standard deviation - use a placeholder for now
    # In a real implementation, this would come from boundary data
    h_std = jnp.ones(ncols) * 200.0  # 200m standard deviation
    
    # SIMPLIFIED: Direct vmap of gravity_wave_drag (no wrapper needed)
    gwd_results = jax.vmap(
        gravity_wave_drag,
        in_axes=(1, 1, 1, 1, 1, 0, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (GWDTendencies, GWDState) per column
    )(state.u_wind, state.v_wind, state.temperature,
        pressure_levels, height_levels, h_std, dt, parameters.gravity_waves)
    
    # Unpack structured results directly
    gwd_tendencies_all, gwd_states_all = gwd_results
    
    # SIMPLIFIED: Extract tendencies from structured result (transpose to [nlev, ncols])
    physics_tendencies = PhysicsTendency(
        u_wind=gwd_tendencies_all.dudt.T,
        v_wind=gwd_tendencies_all.dvdt.T,
        temperature=gwd_tendencies_all.dtedt.T,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={}
    )
    
    # Update physics data
    gravity_wave_data = physics_data.gravity_waves.copy(
        surface_stress=gwd_states_all.wave_stress[:, -1],  # Surface stress (last level)
    )
    
    updated_physics_data = physics_data.copy(gravity_waves=gravity_wave_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_chemistry(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply chemistry tendencies"""
    # Placeholder - will implement chemistry_tendencies function
    return PhysicsTendency.zeros(state.temperature.shape), physics_data