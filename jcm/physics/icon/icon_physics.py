"""Main ICON Physics class for JAX-GCM

This module contains the main IconPhysics class that orchestrates the 
ICON atmospheric physics parameterizations. It follows the same pattern
as SpeedyPhysics but implements the ICON physics suite.

Date: 2025-01-09
"""

import jax
from jax import jit
import jax.numpy as jnp
from typing import Tuple, Optional
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData
from jcm.physics.icon.constants import physical_constants
from jcm.physics.icon.icon_coords import IconCoords

# Import physics modules (will be implemented progressively)
from jcm.physics.icon.radiation.radiation_scheme import radiation_scheme
from jcm.physics.icon.icon_physics_data import RadiationData
from jcm.physics.icon.convection import tiedtke_nordeng_convection
from jcm.physics.icon.clouds import shallow_cloud_scheme, cloud_microphysics
from jcm.physics.icon.parameters import Parameters
from jcm.physics.icon.surface import surface_physics_step, initialize_surface_state
from jcm.physics.icon.surface.surface_types import AtmosphericForcing
from jcm.physics.icon.gravity_waves import gravity_wave_drag
from jcm.physics.icon.chemistry import simple_chemistry, initialize_chemistry_tracers
from jcm.physics.icon.aerosol.simple_aerosol import get_simple_aerosol
from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.physics.icon.forcing import apply_forcing_data

class IconPhysics(Physics):
    """ICON atmospheric physics implementation for JAX-GCM
    
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
                 parameters: Optional[Parameters] = None,
                 dt_physics: Optional[float] = None,
                 radiation_scheme: str = "grey"):
        """Initialize the ICON physics.

        Args:
            write_output: Whether to write physics output to predictions
            checkpoint_terms: Whether to checkpoint physics terms
            parameters: Optional physics parameters (uses defaults if None)
            dt_physics: Physics timestep in seconds. If provided, overrides
                all internal physics timesteps (dt_conv, dt_rad, etc.) to match
                the model integration timestep. This is important since we don't
                have sub-timestepping - all physics must use the same timestep.
            radiation_scheme: Which radiation scheme to use. Either "grey"
                (default simplified multi-band scheme) or "rrtmgp" (requires
                jax-rrtmgp package).

        """
        self.write_output = write_output
        self.checkpoint_terms = checkpoint_terms

        # Store parameters, optionally updating timesteps
        params = parameters or Parameters.default()
        if dt_physics is not None:
            params = params.with_timestep(dt_physics)
        self.parameters = params

        # Cached coordinate data (populated by cache_coords)
        self.cached_coords = None

        # Select radiation term based on scheme choice
        if radiation_scheme == "rrtmgp":
            rad_term = apply_radiation_rrtmgp
        elif radiation_scheme == "grey":
            rad_term = apply_radiation
        else:
            raise ValueError(
                f"Unknown radiation_scheme={radiation_scheme!r}. "
                "Choose 'grey' or 'rrtmgp'."
            )

        # Build list of physics terms
        self.terms = [
            _prepare_common_physics_state,
            apply_forcing_data,             # Time-varying boundary conditions
            get_simple_aerosol,            # Aerosol before radiation
            apply_chemistry,               # Chemistry for ozone, methane etc.
            rad_term,
            apply_convection,
            apply_clouds_and_microphysics,
            apply_vertical_diffusion,
            apply_surface,                 # Surface after radiation and vertical diffusion
            apply_gravity_waves
        ]
    
    def cache_coords(self, coords: CoordinateSystem):
        """Cache coordinate system data needed by ICON physics."""
        self.cached_coords = IconCoords.from_coordinate_system(coords)

    def compute_tendencies(
        self,
        state: PhysicsState,
        forcing: ForcingData,
        terrain: TerrainData,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
        """Compute the physical tendencies given the current state and data structs. Loops through the ICON physics terms, accumulating the tendencies.

        Args:
            state: Current state variables
            forcing: Forcing data
            terrain: Terrain data (boundary conditions)
            date: Date data

        Returns:
            Physical tendencies in PhysicsTendency format
            Object containing physics data (PhysicsData format)

        """
        nodal_shape = self.cached_coords.nodal_shape

        physics_data = PhysicsData.zeros(
            (nodal_shape[1]*nodal_shape[2], ),
            nodal_shape[0],
            icon_coords=self.cached_coords,
            date=date
        )

        # Initialize zero tendencies with tracer tendencies
        tracer_tends = {name: jnp.zeros_like(tracer) for name, tracer in state.tracers.items()}
        tendencies = PhysicsTendency.zeros(state.temperature.shape, tracers=tracer_tends)

        # Get array dimensions for vectorization
        nlev, nlon, nlat = state.temperature.shape  # Note: geometry uses (nlev, nlon, nlat) convention
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
                vectorized_state, physics_data, self.parameters, forcing, terrain
            )
            
            # OPTIMIZATION: Direct accumulation in column format (no reshape)
            accumulated_tendencies = self._accumulate_column_tendencies(
                accumulated_tendencies, term_tendency
            )
        
        # OPTIMIZATION: Single reshape to 3D only at the very end
        tendencies = self._reshape_tendencies_to_3d(accumulated_tendencies, nlev, nlat, nlon)

        return tendencies, physics_data
    
    def _reshape_state_to_columns(self, state: PhysicsState, nlev: int, ncols: int) -> PhysicsState:
        """TPU-optimized reshape using single tree_map operation
        
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
            'normalized_surface_pressure': state.normalized_surface_pressure,
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
        """Initialize tendency accumulators in column format
        
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
        """Efficiently accumulate tendencies in column format
        
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
        """Reshape tendencies to 3D format as a single operation at the end.

        This is the only reshape back to 3D, done once at the very end.
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
    
    def data_struct_to_dict(self, struct, nodal_shape=None, sep="."):
        """Convert physics data struct to dictionary, reshaping column data to 3D.

        This overrides the base class method to handle ICON physics data which
        contains fields in column format that need reshaping for xarray output.
        """
        if struct is None:
            return {}

        nodal_shape = nodal_shape or self.cached_coords.nodal_shape

        # Use a simple namespace to pass nodal_shape to helper methods
        class _ShapeInfo:
            pass
        shape_info = _ShapeInfo()
        shape_info.nodal_shape = nodal_shape

        # Get the base struct conversion or handle manually if needed
        result = self._get_base_struct_dict(struct, shape_info, sep)

        # Reshape all arrays to add time dimension and convert column format to spatial grid
        result = self._reshape_arrays_for_xarray(result, shape_info)

        # Handle multi-channel arrays and filter problematic fields
        result = self._process_multi_channel_arrays(result, shape_info)

        # Filter out non-array and scalar fields
        return self._filter_xarray_compatible_fields(result)
    
    def _get_base_struct_dict(self, struct, shape_info, sep):
        """Get the base dictionary conversion, handling non-array fields gracefully."""
        try:
            return super().data_struct_to_dict(struct, nodal_shape=shape_info.nodal_shape, sep=sep)
        except (AttributeError, ValueError):
            # Handle case where some fields are not arrays (e.g. IconCoords.nodal_shape is a tuple)
            return self._manual_struct_to_dict(struct, shape_info, sep)
    
    def _manual_struct_to_dict(self, struct, shape_info, sep):
        """Manual conversion when base class fails."""
        _skip_keys = {'icon_coords'}

        def _to_dict_recursive(obj, parent_key=""):
            items = {}
            for key, val in obj.__dict__.items():
                if key in _skip_keys:
                    continue
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                if hasattr(val, "__dict__") and val.__dict__:
                    items.update(_to_dict_recursive(val, parent_key=new_key))
                else:
                    items[new_key] = val
            return items

        result = _to_dict_recursive(struct)
        nodal_shape = shape_info.nodal_shape

        # Process array fields for multi-channel splitting (from base class)
        _original_keys = list(result.keys())
        for k in _original_keys:
            val = result[k]
            if hasattr(val, 'shape'):
                s = val.shape
                if len(s) == 5 and s[1:-1] == nodal_shape or len(s) == 4 and s[1:-1] == nodal_shape[1:]:
                    result.update({f"{k}.{i}": result[k][..., i] for i in range(s[-1])})
                    del result[k]

        return result
    
    def _reshape_arrays_for_xarray(self, result, shape_info):
        """Reshape arrays to add time dimension and handle column format."""
        import numpy as np

        nlev, nlon, nlat = shape_info.nodal_shape
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
    
    def _process_multi_channel_arrays(self, result, shape_info):
        """Handle multi-channel arrays and filter problematic fields."""
        nlev, nlon, nlat = shape_info.nodal_shape
                
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
            elif len(s) == 2 and s[0] == 1:
                # [1, ncols] -> skip single time dimension arrays
                print(f"Skipping single time dimension array: {k} with shape {s}")
                del result[k]
            elif len(s) == 3 and s[1] == nlat*nlon:
                # [time, ncols, channels] -> skip for now
                print(f"Skipping [time, ncols, channels] array: {k} with shape {s}")
                del result[k]
        
        return result
    
    def _filter_xarray_compatible_fields(self, result):
        """Filter out fields that xarray/data_to_xarray doesn't handle well."""
        nlev = self.cached_coords.nodal_shape[0]
        filtered_result = {}
        for key, value in result.items():
            if not hasattr(value, 'shape'):
                continue
            if value.shape == ():
                continue
            if nlev + 1 in value.shape:
                # Skip interface-level (half-level) fields
                continue
            filtered_result[key] = value

        return filtered_result

    def reshape_physics_data_to_3d(self, physics_data: PhysicsData, nodal_shape=None) -> PhysicsData:
        """Reshape PhysicsData from column format to 3D nodal format for output.

        This converts arrays from (ncols,) or (nlev, ncols) to (nlon, nlat) or (nlev, nlon, nlat)
        respectively, making them suitable for plotting and analysis.

        Args:
            physics_data: PhysicsData in column format (ncols,) or (nlev, ncols)
            nodal_shape: Tuple (nlev, nlon, nlat). If None, uses cached nodal_shape.

        Returns:
            PhysicsData with arrays reshaped to 3D nodal format

        """
        nodal_shape = nodal_shape or self.cached_coords.nodal_shape
        nlev, nlon, nlat = nodal_shape
        ncols = nlon * nlat

        def reshape_array(arr):
            """Reshape a single array based on its dimensions."""
            if not isinstance(arr, jnp.ndarray):
                return arr  # Return non-arrays unchanged

            if arr.ndim == 1 and arr.shape[0] == ncols:
                # (ncols,) -> (nlon, nlat)
                return arr.reshape(nlon, nlat)
            elif arr.ndim == 2 and arr.shape[1] == ncols:
                # (nlev, ncols) -> (nlev, nlon, nlat)
                return arr.reshape(arr.shape[0], nlon, nlat)
            elif arr.ndim == 2 and arr.shape[0] == nlev and arr.shape[1] == ncols:
                # Explicitly handle (nlev, ncols) -> (nlev, nlon, nlat)
                return arr.reshape(nlev, nlon, nlat)
            elif arr.ndim == 2 and arr.shape[0] == nlev + 1 and arr.shape[1] == ncols:
                # Handle interface levels (nlev+1, ncols) -> (nlev+1, nlon, nlat)
                return arr.reshape(nlev + 1, nlon, nlat)
            else:
                # Return unchanged if shape doesn't match expected patterns
                return arr

        # Use tree_map to reshape all arrays in the nested PhysicsData structure
        reshaped_data = jax.tree_util.tree_map(reshape_array, physics_data)

        return reshaped_data

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
    
    # Calculate pressure levels from surface pressure and sigma coordinates
    surface_pressure = state.normalized_surface_pressure * p0  # Convert to Pa
    sigma_levels = physics_data.icon_coords.fsg  # sigma coordinates at level centers
    pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]

    # Calculate pressure at interfaces (half levels)
    sigma_half = physics_data.icon_coords.hsg
    pressure_half = sigma_half[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
    
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
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply radiation heating rates"""
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
def apply_radiation_rrtmgp(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply RRTMGP radiation heating rates.

    Drop-in replacement for ``apply_radiation`` that calls the RRTMGP
    radiation scheme instead of the grey/simplified scheme.  The input
    preparation and output post-processing are identical.
    """
    from jcm.physics.icon.radiation.radiation_scheme_rrtmgp import (
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
        sw_flux_up=diagnostics_vmapped.sw_flux_up.transpose(1, 0, 2),
        sw_flux_down=diagnostics_vmapped.sw_flux_down.transpose(1, 0, 2),
        sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
        lw_flux_up=diagnostics_vmapped.lw_flux_up.transpose(1, 0, 2),
        lw_flux_down=diagnostics_vmapped.lw_flux_down.transpose(1, 0, 2),
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
    
    # Get surface properties from forcing
    # Reshape boundary fields to column format
    surface_temp = physics_data.surface.surface_temperature.reshape(ncols)
    roughness_length = physics_data.surface.roughness_length.reshape(ncols)

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
        apply_vdiff_to_column, # FIXME: this should call vertical_diffusion_scheme
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