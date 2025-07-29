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
from collections import abc
from typing import Tuple, Optional
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
from jcm.physics.icon.chemistry import simple_chemistry, initialize_chemistry_tracers
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
            get_simple_aerosol,            # Aerosol before radiation FIXME: get_CDNC issue
            apply_chemistry,               # Chemistry for ozone, methane etc.
            apply_radiation,               # Radiation early for surface fluxes. FIXME: revisit two-stream coefficients--top of atmosphere is emitting shortwave up while receiving none from below, causing cooling. downward shortwave flux is constant and not heating the atmosphere.
            apply_convection,
            apply_clouds,
            apply_microphysics,
            apply_surface,                 # Surface after radiation
            apply_vertical_diffusion,      # FIXME: check for vertical ordering issues # FIXME: takes a really long time (~6m for one step not jitted, ~1m15s jitted. For comparison, the rest of the physics runs in ~10s jitted)
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
        Compute the physical tendencies given the current state and data structs. Loops through the ICON physics terms, accumulating the tendencies.

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
        nlev, nlon, nlat = state.temperature.shape  # Note: geometry uses (nlev, nlon, nlat) convention
        ncols = nlat * nlon
        
        # Update boundaries with time-varying conditions before applying physics
        from jcm.boundaries import compute_time_varying_boundaries
        # Convert DateData to day_of_year and time_of_day format
        day_of_year = date.tyear * 365.25  # Convert fractional year to day of year
        time_of_day = (day_of_year % 1.0) * 24.0  # Extract time of day from fractional part
        day_of_year = jnp.floor(day_of_year)  # Get integer day of year
        year = date.model_year
        
        updated_boundaries = compute_time_varying_boundaries(
            boundaries, 
            geometry,
            day_of_year=day_of_year,
            time_of_day=time_of_day,
            year=year
        )
        
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
                vectorized_state, physics_data, self.parameters, updated_boundaries, geometry
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
        reshaped_fields = jax.tree_util.tree_map(reshape_field, {
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
            if field.ndim == 2:  # [nlev, ncols] → [nlev, nlon, nlat]
                return field.reshape(nlev, nlon, nlat)
            else:
                return field
        
        # Single tree_map for all main fields
        reshaped_main = jax.tree_util.tree_map(reshape_to_3d, {
            'u_wind': tendencies['u_wind'],
            'v_wind': tendencies['v_wind'],
            'temperature': tendencies['temperature'],
            'specific_humidity': tendencies['specific_humidity'],
        })
        
        # Reshape tracers
        reshaped_tracers = {
            name: field.reshape(nlev, nlon, nlat)
            for name, field in tendencies['tracers'].items()
        }
        
        return PhysicsTendency(
            **reshaped_main,
            tracers=reshaped_tracers
        )
    
    def data_struct_to_dict(self, struct, geometry, sep="."):
        """
        Convert physics data struct to dictionary, reshaping column data to 3D.
        
        This overrides the base class method to handle ICON physics data which
        contains fields in column format that need reshaping for xarray output.
        """
        if struct is None:
            return {}
        
        # Get the base struct conversion or handle manually if needed
        result = self._get_base_struct_dict(struct, geometry, sep)
        
        # Reshape all arrays to add time dimension and convert column format to spatial grid
        result = self._reshape_arrays_for_xarray(result, geometry)
        
        # Handle multi-channel arrays and filter problematic fields
        result = self._process_multi_channel_arrays(result, geometry)
        
        # Filter out non-array and scalar fields
        return self._filter_xarray_compatible_fields(result)
    
    def _get_base_struct_dict(self, struct, geometry, sep):
        """Get the base dictionary conversion, handling AttributeError gracefully."""
        try:
            return super().data_struct_to_dict(struct, geometry, sep)
        except AttributeError:
            # Handle case where some fields are not arrays
            return self._manual_struct_to_dict(struct, geometry, sep)
    
    def _manual_struct_to_dict(self, struct, geometry, sep):
        """Manual conversion when base class fails."""
        def _to_dict_recursive(obj, parent_key=""):
            items = {}
            for key, val in obj.__dict__.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                if hasattr(val, "__dict__") and val.__dict__:
                    items.update(_to_dict_recursive(val, parent_key=new_key))
                else:
                    items[new_key] = val
            return items
        
        result = _to_dict_recursive(struct)
        
        # Process array fields for multi-channel splitting (from base class)
        _original_keys = list(result.keys())
        for k in _original_keys:
            val = result[k]
            if hasattr(val, 'shape'):
                s = val.shape
                if len(s) == 5 and s[1:-1] == geometry.nodal_shape or len(s) == 4 and s[1:-1] == geometry.nodal_shape[1:]:
                    result.update({f"{k}.{i}": result[k][..., i] for i in range(s[-1])})
                    del result[k]
        
        return result
    
    def _reshape_arrays_for_xarray(self, result, geometry):
        """Reshape arrays to add time dimension and handle column format."""
        import numpy as np
        
        nlev = geometry.nlevels
        nlon = geometry.nlon
        nlat = geometry.nlat
        ncols = nlon * nlat
        
        for key, value in list(result.items()):
            if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 1:
                reshaped = self._reshape_single_array(value, nlev, nlon, nlat, ncols)
                if reshaped is not None:
                    result[key] = reshaped
        
        return result
    
    def _reshape_single_array(self, value, nlev, nlon, nlat, ncols):
        """Reshape a single array based on its dimensions."""
        # 1D arrays
        if value.ndim == 1:
            if value.shape[0] == ncols:
                return value.reshape(1, nlon, nlat)
            elif value.shape[0] != 1:
                return value.reshape(1, *value.shape)
        
        # 2D arrays
        elif value.ndim == 2:
            return self._reshape_2d_array(value, nlev, nlon, nlat, ncols)
        
        # 3D arrays
        elif value.ndim == 3:
            return self._reshape_3d_array(value, nlev, nlon, nlat, ncols)
        
        # 4D arrays
        elif value.ndim == 4:
            return self._reshape_4d_array(value, nlev, nlon, nlat, ncols)
        
        return None
    
    def _reshape_2d_array(self, value, nlev, nlon, nlat, ncols):
        """Reshape 2D arrays with various patterns."""
        if value.shape[1] == 1 and value.shape[0] == ncols:
            # [ncols, 1] -> [1, nlon, nlat]
            return value.reshape(1, nlon, nlat)
        elif value.shape == (nlev, ncols):
            # [nlev, ncols] -> [1, nlev, nlon, nlat]
            return value.reshape(1, nlev, nlon, nlat)
        elif value.shape == (nlon, nlat):
            # [nlon, nlat] -> [1, nlon, nlat]
            return value.reshape(1, nlon, nlat)
        elif value.shape[0] == nlev + 1 and value.shape[1] == ncols:
            # [nlev+1, ncols] -> [1, nlev+1, nlon, nlat] (interfaces)
            return value.reshape(1, nlev + 1, nlon, nlat)
        elif value.shape[1] == ncols:
            # [time, ncols] -> [time, nlon, nlat]
            ntime = value.shape[0]
            return value.reshape(ntime, nlon, nlat)
        
        return None
    
    def _reshape_3d_array(self, value, nlev, nlon, nlat, ncols):
        """Reshape 3D arrays with various patterns."""
        if value.shape == (nlev, ncols, value.shape[2]):
            # [nlev, ncols, channels] -> [1, nlev, nlon, nlat, channels]
            return value.reshape(1, nlev, nlon, nlat, value.shape[2])
        elif value.shape == (nlev + 1, ncols, value.shape[2]):
            # [nlev+1, ncols, channels] -> [1, nlev+1, nlon, nlat, channels]
            return value.reshape(1, nlev + 1, nlon, nlat, value.shape[2])
        elif value.shape[0] == nlev and value.shape[1] == nlon and value.shape[2] == nlat:
            # [nlev, nlon, nlat] -> [1, nlev, nlon, nlat]
            return value.reshape(1, nlev, nlon, nlat)
        elif value.shape[1] == nlev and value.shape[2] == ncols:
            # [time, nlev, ncols] -> [time, nlev, nlon, nlat]
            ntime = value.shape[0]
            return value.reshape(ntime, nlev, nlon, nlat)
        elif value.shape[1] == nlev + 1 and value.shape[2] == ncols:
            # [time, nlev+1, ncols] -> [time, nlev+1, nlon, nlat] (interfaces)
            ntime = value.shape[0]
            return value.reshape(ntime, nlev + 1, nlon, nlat)
        
        return None
    
    def _reshape_4d_array(self, value, nlev, nlon, nlat, ncols):
        """Reshape 4D arrays with various patterns."""
        if value.shape[1] == nlev and value.shape[2] == ncols:
            # [time, nlev, ncols, channels] -> [time, nlev, nlon, nlat, channels]
            ntime = value.shape[0]
            nchannels = value.shape[3]
            return value.reshape(ntime, nlev, nlon, nlat, nchannels)
        elif value.shape[1] == nlev + 1 and value.shape[2] == ncols:
            # [time, nlev+1, ncols, channels] -> [time, nlev+1, nlon, nlat, channels]
            ntime = value.shape[0]
            nchannels = value.shape[3]
            return value.reshape(ntime, nlev + 1, nlon, nlat, nchannels)
        
        return None
    
    def _process_multi_channel_arrays(self, result, geometry):
        """Handle multi-channel arrays and filter problematic fields."""
        nlev = geometry.nlevels
        nlon = geometry.nlon
        nlat = geometry.nlat
        
        _original_keys = list(result.keys())
        for k in _original_keys:
            val = result[k]
            if not hasattr(val, 'shape'):
                continue
                
            s = val.shape
            if len(s) == 5 and s[1:4] == (nlev, nlon, nlat):
                # [time, nlev, nlon, nlat, channels] -> split into separate fields
                result.update({f"{k}.{i}": result[k][..., i] for i in range(s[-1])})
                del result[k]
            elif len(s) == 5 and s[1:4] == (nlev + 1, nlon, nlat):
                # Skip interface-level multi-channel arrays
                print(f"Skipping interface-level multi-channel array: {k} with shape {s}")
                del result[k]
            elif len(s) == 4 and s[1:4] == (nlev + 1, nlon, nlat):
                # Skip interface-level data
                print(f"Skipping interface-level data: {k} with shape {s}")
                del result[k]
        
        return result
    
    def _filter_xarray_compatible_fields(self, result):
        """Filter out non-array and scalar fields that xarray doesn't handle well."""
        filtered_result = {}
        for key, value in result.items():
            if not hasattr(value, 'shape'):
                # Skip non-array fields
                continue
            if hasattr(value, 'shape') and value.shape == ():
                # Skip scalar fields - they're not needed for xarray conversion
                continue
            filtered_result[key] = value
        
        return filtered_result

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
        boundaries: Boundary conditions (already updated with time-varying conditions)
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
    height_half = jnp.concatenate((
        height_levels[:1] + 1000.0, # FIXME: choice of offset
        (height_levels[1:] + height_levels[:-1]) / 2,
        0 * height_levels[-1:]), axis=0)

    # Calculate air density
    rho = pressure_levels / (physical_constants.rd * state.temperature)
    
    # Calculate layer thickness
    dp = jnp.diff(pressure_half, axis=0)
    dz_full = dp / (rho * physical_constants.grav)
    
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

    # Initialize chemistry tracers if not already done
    # Check if chemistry is initialized by checking if ozone maximum is reasonable
    ozone_max = jnp.max(physics_data.chemistry.ozone_vmr)
    should_initialize = ozone_max < 100.0  # If max ozone < 100 ppbv, initialize
    
    # Initialize chemistry tracers with reasonable distributions
    chemistry_state = initialize_chemistry_tracers(
        pressure_levels,
        surface_pressure,
        state.temperature,
        config=None  # Use defaults
    )
    
    # Use JAX where to conditionally update chemistry
    chemistry_data = physics_data.chemistry.copy(
        ozone_vmr=jnp.where(should_initialize, chemistry_state.ozone_vmr, physics_data.chemistry.ozone_vmr),
        methane_vmr=jnp.where(should_initialize, chemistry_state.methane_vmr, physics_data.chemistry.methane_vmr),
        co2_vmr=jnp.where(should_initialize, chemistry_state.co2_vmr, physics_data.chemistry.co2_vmr),
        ozone_production=jnp.where(should_initialize, chemistry_state.ozone_production, physics_data.chemistry.ozone_production),
        ozone_loss=jnp.where(should_initialize, chemistry_state.ozone_loss, physics_data.chemistry.ozone_loss),
        methane_loss=jnp.where(should_initialize, chemistry_state.methane_loss, physics_data.chemistry.methane_loss)
    )
    
    updated_physics_data = physics_data.copy(
        diagnostics=diagnostic_data,
        chemistry=chemistry_data
    )

    zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return zero_tendencies, updated_physics_data

# Physics term methods
@jit
def apply_radiation(state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply radiation heating rates"""
    
    # Note: state is already in 2D format [nlev, ncols] from compute_tendencies
    nlev, ncols = state.temperature.shape
    
    # Get date information for solar calculations
    date = physics_data.date
    day_of_year = date.day_of_year if hasattr(date, 'day_of_year') else 172.0  # Default to summer
    seconds_since_midnight = date.seconds_since_midnight if hasattr(date, 'seconds_since_midnight') else 43200.0  # Default to noon
    
    # Get lat/lon from geometry - reshape to column format
    nlon, nlat = boundaries.surface_temperature.shape
    latitudes = jnp.tile(geometry.radang, nlon)  # Repeat lats for each lon
    longitudes_1d = jnp.linspace(-jnp.pi, jnp.pi, nlon, endpoint=False)  # radians
    longitudes = jnp.repeat(longitudes_1d, nlat)  # Repeat lons for each lat
    
    # Get cloud properties from tracers and previous physics
    cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    # Cloud fraction needs to be reshaped to 2D if it's 3D
    cloud_fraction = physics_data.clouds.cloud_fraction
    if cloud_fraction.ndim == 3:
        nlev_cf, nlat, nlon = cloud_fraction.shape
        cloud_fraction = cloud_fraction.reshape(nlev_cf, nlat * nlon)

    # Get ozone from chemistry data (reshape to column format)
    ozone_vmr = physics_data.chemistry.ozone_vmr * 1e-9  # Convert ppbv to VMR
    
    # Get CO2 from boundaries (reshape to column format)
    co2_vmr = boundaries.co2_concentration.reshape(ncols) * 1e-6  # Convert ppmv to VMR
    
    # Prepare aerosol data for vmap - reshape to have column as the mapped dimension
    aerosol_data_for_vmap = physics_data.aerosol.copy(
        aod_profile=physics_data.aerosol.aod_profile.reshape(nlev, ncols).T,  # (ncols, nlev)
        ssa_profile=physics_data.aerosol.ssa_profile.reshape(nlev, ncols).T,  # (ncols, nlev)
        asy_profile=physics_data.aerosol.asy_profile.reshape(nlev, ncols).T,  # (ncols, nlev)
        cdnc_factor=physics_data.aerosol.cdnc_factor.reshape(ncols),  # (ncols,)
        aod_total=physics_data.aerosol.aod_total.reshape(ncols),  # (ncols,)
        aod_anthropogenic=physics_data.aerosol.aod_anthropogenic.reshape(ncols),  # (ncols,)
        aod_background=physics_data.aerosol.aod_background.reshape(ncols),  # (ncols,)
    )
    
    radiation_results = jax.vmap(
        radiation_scheme,
        in_axes=(1, 1, 1, 1,
                 1, 1, 1, 1,
                 None, None, 0, 0,
                 None, 0, 1, 0),  # day_of_year, seconds_since_midnight, parameters are scalars, aerosol_data is per column
        out_axes=(0, 0)  # Returns (RadiationTendencies, RadiationData) per column
    )(state.temperature, state.specific_humidity, physics_data.diagnostics.pressure_full, physics_data.diagnostics.layer_thickness,
      physics_data.diagnostics.air_density, cloud_water, cloud_ice, cloud_fraction,
      day_of_year, seconds_since_midnight, latitudes, longitudes,
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

@jit
def apply_clouds(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply shallow cloud scheme """
    
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    surface_pressure = physics_data.diagnostics.surface_pressure
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    # Get cloud configuration from parameters
    cloud_config = parameters.clouds
    
    cloud_results = jax.vmap(
        shallow_cloud_scheme,
        in_axes=(1, 1, 1, 1, 1, 0, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (CloudTendencies, CloudState) per column
    )(state.temperature, state.specific_humidity, pressure_levels,
        qc, qi, surface_pressure, dt, cloud_config)
    
    # Unpack structured results directly
    cloud_tendencies_all, cloud_states_all = cloud_results

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
    
    micro_results = jax.vmap(
        cloud_microphysics,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),  # dt and config are scalars
        out_axes=(0, 0)  # Returns (MicrophysicsTendencies, MicrophysicsState) per column
    )(state.temperature, state.specific_humidity, pressure_levels,
        qc, qi, cloud_fraction, air_density, dz, droplet_number, dt, micro_config)
    
    # Unpack structured results directly
    micro_tendencies_all, micro_states_all = micro_results
    
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
    
    # Initialize prognostic variables if not present (ensure proper column format)
    if hasattr(physics_data.vertical_diffusion, 'tke'):
        tke = physics_data.vertical_diffusion.tke
        # Reshape TKE if it's in grid format [nlev, nlat, nlon] -> [nlev, ncols]
        if tke.ndim == 3:
            tke = tke.reshape(nlev, ncols)
    else:
        tke = jnp.ones((nlev, ncols)) * 0.1
        
    if hasattr(physics_data.vertical_diffusion, 'thv_variance'):
        thv_variance = physics_data.vertical_diffusion.thv_variance
        # Reshape if needed
        if thv_variance.ndim == 3:
            thv_variance = thv_variance.reshape(nlev, ncols)
    else:
        thv_variance = jnp.zeros((nlev, ncols))
    
    # Surface properties (simplified - should come from boundaries)
    nsfc_type = 3  # water, ice, land
    surface_fraction = jnp.zeros((ncols, nsfc_type))
    surface_fraction = surface_fraction.at[:, 2].set(1.0)  # All land for now
    
    # Get surface properties from boundaries (now guaranteed to be present)
    # Reshape boundary fields to column format
    surface_temp = boundaries.surface_temperature.reshape(ncols)
    roughness_length = boundaries.roughness_length.reshape(ncols)

    surface_temperature = jnp.repeat(surface_temp[:, jnp.newaxis], nsfc_type, axis=1)
    roughness = jnp.repeat(roughness_length[:, jnp.newaxis], nsfc_type, axis=1)
    
    # Ocean currents (zero for now)
    ocean_u = jnp.zeros(ncols)
    ocean_v = jnp.zeros(ncols)
    
    # Extract fixed qc/qi tracers for vertical diffusion
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    
    def apply_vdiff_to_column(u_col, v_col, temp_col, qv_col, qc_col, qi_col,
                             pressure_full_col, pressure_half_col, geopot_col,
                             height_full_col, height_half_col, surface_temp_col,
                             surface_frac_col, roughness_col, ocean_u_scalar, ocean_v_scalar,
                             tke_col, thv_var_col):
        """Apply vertical diffusion to a single column with structured output"""
        
        # Prepare state for this column - reshape to expected 2D format (1, nlev) or (1, nlev+1)
        vdiff_state = prepare_vertical_diffusion_state(
            u=u_col[None, :], v=v_col[None, :], temperature=temp_col[None, :], 
            qv=qv_col[None, :], qc=qc_col[None, :], qi=qi_col[None, :],
            pressure_full=pressure_full_col[None, :], pressure_half=pressure_half_col[None, :],
            geopotential=geopot_col[None, :], height_full=height_full_col[None, :], 
            height_half=height_half_col[None, :],
            surface_temperature=surface_temp_col[None, :], surface_fraction=surface_frac_col[None, :],
            roughness_length=roughness_col[None, :], ocean_u=ocean_u_scalar[None], ocean_v=ocean_v_scalar[None],
            tke=tke_col[None, :], thv_variance=thv_var_col[None, :]
        )
        
        # Compute vertical diffusion
        tendencies, diagnostics = vertical_diffusion_column(
            vdiff_state, parameters.vertical_diffusion, dt
        )
        
        # Squeeze outputs to remove the dummy column dimension (1, nlev) -> (nlev)
        def squeeze_first_dim(x):
            return jnp.squeeze(x, axis=0) if x.ndim > 1 else x
        
        tendencies = jax.tree_util.tree_map(squeeze_first_dim, tendencies)
        diagnostics = jax.tree_util.tree_map(squeeze_first_dim, diagnostics)
        
        # Return structured data directly
        return tendencies, diagnostics
    
    vdiff_results = jax.vmap(
        apply_vdiff_to_column,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1),  # Fix surface properties to axis 0 (column mapped)
        out_axes=(0, 0)  # Returns (VDiffTendencies, VDiffDiagnostics) per column
    )(state.u_wind, state.v_wind, state.temperature, state.specific_humidity, qc, qi,
      pressure_levels, pressure_half, state.geopotential, height_levels, height_half,
      surface_temperature, surface_fraction, roughness, ocean_u, ocean_v, tke, thv_variance)
    
    # Unpack structured results from vmap
    vdiff_tendencies, vdiff_diagnostics = vdiff_results
    
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
    b_flux = vdiff_diagnostics.surface_heat_flux  # Shape [ncols] (using heat flux as buoyancy proxy)
    
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
        km=km,  # Use correct field name
        kh=kh,  # Use correct field name
        pbl_height=pbl_height,
        surface_friction_velocity=u_star,
        # Note: thv_variance and surface_buoyancy_flux don't exist in VerticalDiffusionData
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
    # Get surface properties from boundaries (now guaranteed to be present)
    # Reshape boundary fields to column format
    surface_temp = boundaries.surface_temperature.reshape(ncols)
    surface_albedo_vis = boundaries.surface_albedo_vis.reshape(ncols)
    surface_albedo_nir = boundaries.surface_albedo_nir.reshape(ncols)
    surface_emissivity = boundaries.surface_emissivity.reshape(ncols)

    # Initialize surface state (simplified)
    # In reality, this should come from the model's surface state
    nsfc_type = 3  # Fixed value: water, ice, land
    surface_fractions = jnp.zeros((ncols, nsfc_type))
    land_fraction = boundaries.fmask.reshape((ncols,))
    surface_fractions = surface_fractions.at[:, 0].set(1.0 - land_fraction)
    surface_fractions = surface_fractions.at[:, 2].set(land_fraction)  # FIXME

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
    
    # Surface properties are now extracted earlier in the function
    
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
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply gravity wave drag"""
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    height_levels = physics_data.diagnostics.height_full
    air_density = physics_data.diagnostics.air_density
    
    # Need orography standard deviation - use a placeholder for now
    # In a real implementation, this would come from boundary data
    h_std = jnp.ones(ncols) * 200.0  # 200m standard deviation
    
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
    boundaries: BoundaryData,
    geometry: Geometry
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