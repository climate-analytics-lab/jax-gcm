"""
RRTMGP-based radiation scheme for ICON physics.

This module integrates jax-rrtmgp with ICON's radiation interface, handling:
- Location-specific solar geometry via ICON's calculate_solar_radiation_gcm
- ICON vertical ordering (TOA→surface) vs RRTMGP (surface→TOA) conversion
- Halo management (temperature NaN-padded for RRTMGP fill; others edge-filled)
- Stretched grid mapping for non-uniform vertical coordinates
- Unit conversions and cloud effective radii from ICON functions
- Output conversion to ICON's RadiationTendencies and RadiationData formats

Key entry point: `radiation_scheme_rrtmgp` - drop-in replacement for ICON's radiation_scheme.
"""

from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import jax.numpy as jnp
from jax import lax

from jcm.physics.icon.radiation import calculate_solar_radiation_gcm
from jcm.physics.icon.radiation.radiation_types import RadiationParameters, RadiationTendencies
from jcm.physics.icon.icon_physics_data import RadiationData
from jcm.physics.icon.radiation.radiation_scheme import prepare_radiation_state
from jcm.physics.icon.radiation.cloud_optics import effective_radius_liquid, effective_radius_ice
from jcm.physics.icon.constants.physical_constants import PhysicalConstants

import rrtmgp
from rrtmgp.config import radiative_transfer
from rrtmgp import stretched_grid_util
from rrtmgp.rrtmgp import RRTMGP


# Load RRTMGP data paths
rrtmgp_root = Path(rrtmgp.__path__[0])
rrtmgp_optics_path = rrtmgp_root / "optics"
rrtmgp_data_path = rrtmgp_optics_path / "rrtmgp_data"
test_data_path = rrtmgp_optics_path / "test_data"

# Base RRTMGP optics configuration
_BASE_RRTMGP_OPTICS = radiative_transfer.OpticsParameters(
    optics=radiative_transfer.RRTMOptics(
        longwave_nc_filepath=str(rrtmgp_data_path / "rrtmgp-gas-lw-g128.nc"),
        shortwave_nc_filepath=str(rrtmgp_data_path / "rrtmgp-gas-sw-g112.nc"),
        cloud_longwave_nc_filepath=str(rrtmgp_data_path / "cloudysky_lw.nc"),
        cloud_shortwave_nc_filepath=str(rrtmgp_data_path / "cloudysky_sw.nc"),
    )
)

# Volume mixing ratio data file
_VMR_FILEPATH = str(test_data_path / "vmr_global_means.json")

# Get ICON's default radiation parameters for representative values
_ICON_DEFAULTS = RadiationParameters.default()

# Global RRTMGP instance (created once at module load to avoid JAX tracer issues)
# Uses representative values for configuration; actual solar geometry calculated per gridcell
_GLOBAL_RRTMGP_INSTANCE = RRTMGP(
    radiative_transfer_cfg=radiative_transfer.RadiativeTransfer(
        optics=_BASE_RRTMGP_OPTICS,
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_emis=_ICON_DEFAULTS.surface_emissivity,
            sfc_alb=_ICON_DEFAULTS.surface_albedo_vis,
            zenith=1.0,                                   # updated per grid cell
            irrad=_ICON_DEFAULTS.solar_constant,          # updated per grid cell
            toa_flux_lw=0.0,                              # Longwave TOA flux (not used in our setup)
            vmr_global_mean_filepath=_VMR_FILEPATH
        ),
        save_lw_sw_heating_rates=True
    ),
    dz=1.0,  # Placeholder layer thickness (actual value handled by stretched grid)
    diagnostic_fields=(
        'surf_lw_flux_down_2d_xy',
        'surf_lw_flux_up_2d_xy', 
        'surf_sw_flux_down_2d_xy',
        'surf_sw_flux_up_2d_xy',
        'toa_sw_flux_incoming_2d_xy',
        'toa_sw_flux_outgoing_2d_xy',
        'toa_lw_flux_outgoing_2d_xy'
    )
)

def _to_3d_with_nan_halo(arr_1d: jnp.ndarray, nlev: int, halo: int = 1) -> jnp.ndarray:
    """Convert 1D array to 3D with NaN-filled halos (for temperature - let RRTMGP extrapolate)."""
    nzh = nlev + 2 * halo
    arr_3d = jnp.full((1, 1, nzh), jnp.nan)
    arr_3d = arr_3d.at[0, 0, halo:halo + nlev].set(arr_1d)
    return arr_3d


def _to_3d_with_filled_halo(arr_1d: jnp.ndarray, nlev: int, halo: int = 1) -> jnp.ndarray:
    """Convert 1D array to 3D with edge-filled halos (for non-temperature fields)."""
    nzh = nlev + 2 * halo
    arr_3d = jnp.empty((1, 1, nzh), dtype=arr_1d.dtype)
    arr_3d = arr_3d.at[0, 0, halo:halo + nlev].set(arr_1d)
    arr_3d = arr_3d.at[0, 0, 0].set(arr_1d[0])        # Bottom halo = bottom value
    arr_3d = arr_3d.at[0, 0, -1].set(arr_1d[-1])      # Top halo = top value
    return arr_3d


def _reverse_if_needed(pressure: jnp.ndarray) -> jnp.ndarray:
    """Return JAX boolean: True if pressure order is TOA→surface (increasing with index)."""
    return pressure[0] < pressure[-1]


def prepare_rrtmgp_data(
    icon_data,
    layer_thickness,
    cdnc_factor: jnp.ndarray,
    land_fraction: float = 0.5
) -> dict:
    """Convert ICON RadiationState to RRTMGP inputs.

    Args:
        icon_data: ICON RadiationState with atmospheric profiles
        layer_thickness: Layer thickness (m) [nlev]
        cdnc_factor: Cloud droplet number concentration factor
        land_fraction: Land fraction for effective radius calculation

    Returns:
        Dictionary of RRTMGP inputs with proper shapes, ordering, and units

    Handles:
        - Vertical order conversion (ICON TOA→surface → RRTMGP surface→TOA)
        - Halo padding (temperature NaN, others edge-filled)
        - Stretched grid mapping for non-uniform vertical coordinates
        - Water variable conversions (VMR→mass mixing, paths→mixing ratios)
        - Cloud effective radii calculation and unit conversion (μm→m)
    """
    nlev = icon_data.temperature.shape[0]
    halo = 1
    
    # Helper functions for 3D conversion
    to3d_nan = lambda a: _to_3d_with_nan_halo(a, nlev, halo)
    to3d_fill = lambda a: _to_3d_with_filled_halo(a, nlev, halo)

    # Calculate air density using ICON's gas constant
    phys_const = PhysicalConstants()
    rho = icon_data.pressure / (phys_const.rgas * icon_data.temperature)

    # Check if vertical order needs reversal (ICON TOA→surface vs RRTMGP surface→TOA)
    needs_reversal = _reverse_if_needed(icon_data.pressure)
    flip = lambda a: a[::-1]
    identity = lambda a: a
    
    # Conditionally reverse all vertical profiles
    layer_thickness = lax.cond(needs_reversal, flip, identity, layer_thickness)
    rho = lax.cond(needs_reversal, flip, identity, rho)
    temperature_1d = lax.cond(needs_reversal, flip, identity, icon_data.temperature)
    pressure_1d = lax.cond(needs_reversal, flip, identity, icon_data.pressure)
    cloud_water_path_1d = lax.cond(needs_reversal, flip, identity, icon_data.cloud_water_path)
    cloud_ice_path_1d = lax.cond(needs_reversal, flip, identity, icon_data.cloud_ice_path)

    # Create stretched grid mapping for non-uniform vertical coordinates
    layer_thickness_3d = to3d_fill(layer_thickness)
    sg_map = {
        stretched_grid_util.hc_key(2): layer_thickness_3d,  # Node-centered thickness
        stretched_grid_util.hf_key(2): layer_thickness_3d,  # Face-centered thickness
    }

    # Convert cloud paths to mixing ratios
    cloud_water_mixing = cloud_water_path_1d / (rho * layer_thickness)
    cloud_ice_mixing = cloud_ice_path_1d / (rho * layer_thickness)
    total_condensate = cloud_water_mixing + cloud_ice_mixing
    
    # Convert water vapor VMR to mass mixing ratio: q = VMR * (M_h2o / M_dry) = VMR * eps
    h2o_mass_mixing = icon_data.h2o_vmr * phys_const.eps
    total_water = h2o_mass_mixing + total_condensate
    
    # Calculate cloud effective radii using ICON's parameterizations
    r_eff_liq = effective_radius_liquid(cdnc_factor, land_fraction)
    r_eff_ice = effective_radius_ice(
        temperature_1d,
        cloud_ice_path_1d / jnp.maximum(1.0, cloud_water_path_1d + cloud_ice_path_1d)
    )
    
    # Convert effective radii from microns to meters and ensure proper shape
    if jnp.asarray(r_eff_liq).ndim == 0:
        cloud_r_eff_liq = jnp.full((nlev,), r_eff_liq) * 1e-6
    else:
        r_liq_1d = jnp.asarray(r_eff_liq).reshape(-1)
        cloud_r_eff_liq = (jnp.full((nlev,), r_liq_1d[0]) if r_liq_1d.shape[0] != nlev else r_liq_1d) * 1e-6
    cloud_r_eff_ice = jnp.asarray(r_eff_ice).reshape(-1) * 1e-6
    
    # Return RRTMGP inputs
    return {
        'rho_xxc': to3d_fill(rho),                          # Air density [kg/m³]
        'q_t': to3d_fill(total_water),                      # Total water mixing ratio
        'q_liq': to3d_fill(cloud_water_mixing),             # Liquid water mixing ratio
        'q_ice': to3d_fill(cloud_ice_mixing),               # Ice water mixing ratio
        'q_c': to3d_fill(total_condensate),                 # Total condensate mixing ratio
        'cloud_r_eff_liq': to3d_fill(cloud_r_eff_liq),     # Liquid droplet effective radius [m]
        'cloud_r_eff_ice': to3d_fill(cloud_r_eff_ice),     # Ice crystal effective radius [m]
        'temperature': to3d_nan(temperature_1d),            # Temperature [K] (NaN halos)
        'sfc_temperature': icon_data.surface_temperature.reshape(1, 1),  # Surface temperature
        'p_ref_xxc': to3d_fill(pressure_1d),               # Pressure [Pa]
        'sg_map': sg_map,                                   # Stretched grid mapping
        'use_scan': True                                    # Use scan for efficiency
    }


def prepare_icon_data(
    rrtmgp_data: dict,
    icon_data
) -> Tuple[RadiationTendencies, RadiationData]:
    """Convert RRTMGP output to ICON RadiationTendencies and RadiationData.

    Args:
        rrtmgp_data: Raw RRTMGP diagnostic output dictionary
        icon_data: ICON RadiationState

    Returns:
        Tuple of (RadiationTendencies, RadiationData)

    Note:
        Currently builds simple flux profiles from surface/TOA diagnostics.
        Full 3D flux profiles require modification of jax-rrtmgp in terms of expanding the diagnostic fields.
    """
    # Extract information from available data
    halo = 1
    nlev = icon_data.temperature.shape[0]
    cos_zenith = icon_data.cos_zenith[0]  # Extract scalar from 1-element array
    
    # Extract heating rates (remove halos)
    total_heating = rrtmgp_data['rad_heat_src'][0, 0, halo:halo+nlev]
    lw_heating = rrtmgp_data['rad_heat_lw_3d'][0, 0, halo:halo+nlev]
    sw_heating = rrtmgp_data['rad_heat_sw_3d'][0, 0, halo:halo+nlev]
    
    # Create radiation tendencies
    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating,
        shortwave_heating=sw_heating
    )
    
    # Extract surface and TOA fluxes from diagnostics
    surf_sw_down = rrtmgp_data['surf_sw_flux_down_2d_xy'][0, 0]
    surf_sw_up = rrtmgp_data['surf_sw_flux_up_2d_xy'][0, 0]
    surf_lw_down = rrtmgp_data['surf_lw_flux_down_2d_xy'][0, 0]
    surf_lw_up = rrtmgp_data['surf_lw_flux_up_2d_xy'][0, 0]
    toa_sw_down = rrtmgp_data['toa_sw_flux_incoming_2d_xy'][0, 0]
    toa_sw_up = rrtmgp_data['toa_sw_flux_outgoing_2d_xy'][0, 0]
    toa_lw_up = rrtmgp_data['toa_lw_flux_outgoing_2d_xy'][0, 0]
    
    # Build flux profiles with singleton band dimension for IconPhysics transpose
    # Shape: [nlev+1, 1] per column → [ncols, nlev+1, 1] after vmap → [nlev+1, ncols, 1] after transpose
    sw_flux_up_profile = jnp.zeros((nlev + 1, 1))
    sw_flux_down_profile = jnp.zeros((nlev + 1, 1))
    lw_flux_up_profile = jnp.zeros((nlev + 1, 1))
    lw_flux_down_profile = jnp.zeros((nlev + 1, 1))

    # Set surface (index 0) and TOA (index -1) boundary values
    sw_flux_down_profile = sw_flux_down_profile.at[0, 0].set(surf_sw_down)
    sw_flux_up_profile = sw_flux_up_profile.at[0, 0].set(surf_sw_up)
    sw_flux_down_profile = sw_flux_down_profile.at[-1, 0].set(toa_sw_down)
    sw_flux_up_profile = sw_flux_up_profile.at[-1, 0].set(toa_sw_up)
    lw_flux_down_profile = lw_flux_down_profile.at[0, 0].set(surf_lw_down)
    lw_flux_up_profile = lw_flux_up_profile.at[0, 0].set(surf_lw_up)
    lw_flux_up_profile = lw_flux_up_profile.at[-1, 0].set(toa_lw_up)
    
    # Create radiation diagnostics
    diagnostics = RadiationData(
        cos_zenith=cos_zenith,
        sw_flux_up=sw_flux_up_profile,
        sw_flux_down=sw_flux_down_profile,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up_profile,
        lw_flux_down=lw_flux_down_profile,
        lw_heating_rate=lw_heating,
        surface_sw_down=surf_sw_down,
        surface_lw_down=surf_lw_down,
        surface_sw_up=surf_sw_up,
        surface_lw_up=surf_lw_up,
        toa_sw_up=toa_sw_up,
        toa_lw_up=toa_lw_up,
        toa_sw_down=toa_sw_down
    )
    
    return tendencies, diagnostics


def radiation_scheme_rrtmgp_fn(
    rrtmgp_data: dict, 
    toa_flux: jnp.ndarray,
    cos_zenith: jnp.ndarray
) -> dict:
    """Compute heating rates using RRTMGP with dynamic solar parameters.
    
    Temporarily updates the atmospheric state with per-gridcell solar parameters
    by monkey-patching the frozen dataclass. 
    
    Args:
        rrtmgp_data: Dictionary of RRTMGP inputs (from prepare_rrtmgp_data)
        toa_flux: Top-of-atmosphere solar flux [W/m²] (from ICON calculation)
        cos_zenith: Cosine of solar zenith angle (from ICON calculation)
        
    Returns:
        Dictionary of RRTMGP outputs (heating rates and diagnostics)
    """
    # Store original values
    original_zenith = _GLOBAL_RRTMGP_INSTANCE.atmospheric_state.zenith
    original_irrad = _GLOBAL_RRTMGP_INSTANCE.atmospheric_state.irrad
    
    # Compute dynamic solar parameters
    zenith_angle = jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0))
    
    # Monkey-patch the atmospheric state temporarily with JAX arrays
    # This now works because we fixed RRTMGP to handle JAX arrays
    object.__setattr__(_GLOBAL_RRTMGP_INSTANCE.atmospheric_state, 'zenith', zenith_angle)
    object.__setattr__(_GLOBAL_RRTMGP_INSTANCE.atmospheric_state, 'irrad', toa_flux)
    
    try:
        # Use the standard RRTMGP compute_heating_rate method with updated state
        rrtmgp_output = _GLOBAL_RRTMGP_INSTANCE.compute_heating_rate(**rrtmgp_data)
        return rrtmgp_output
    finally:
        # Always restore original values
        object.__setattr__(_GLOBAL_RRTMGP_INSTANCE.atmospheric_state, 'zenith', original_zenith)
        object.__setattr__(_GLOBAL_RRTMGP_INSTANCE.atmospheric_state, 'irrad', original_irrad)


def radiation_scheme_rrtmgp(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    day_of_year: float,
    seconds_since_midnight: float,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,  # AerosolData from physics_data
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6
) -> Tuple[RadiationTendencies, RadiationData]:
    """RRTMGP-based radiation scheme compatible with ICON physics interface.
    
    Args:
        temperature: Temperature profile [K] (nlev,)
        specific_humidity: Specific humidity [kg/kg] (nlev,)
        pressure_levels: Pressure at layer centers [Pa] (nlev,)
        layer_thickness: Layer thickness [m] (nlev,)
        air_density: Air density [kg/m³] (nlev,)
        cloud_water: Cloud liquid water mixing ratio [kg/kg] (nlev,)
        cloud_ice: Cloud ice mixing ratio [kg/kg] (nlev,)
        cloud_fraction: Cloud fraction [0-1] (nlev,)
        day_of_year: Day of year [1-365]
        seconds_since_midnight: Seconds since midnight UTC
        latitude: Latitude [degrees]
        longitude: Longitude [degrees]
        parameters: Radiation parameters
        aerosol_data: Aerosol optical properties
        ozone_vmr: Ozone volume mixing ratio [nlev] (optional)
        co2_vmr: CO2 volume mixing ratio
        
    Returns:
        Tuple of (RadiationTendencies, RadiationData)

    This function:
        1. Calculates location-specific solar geometry
        2. Prepares ICON radiation state
        3. Converts inputs to RRTMGP format
        4. Runs RRTMGP radiative transfer
        5. Converts outputs back to ICON format
    """
    # Extract cloud droplet number concentration factor
    if aerosol_data.cdnc_factor.ndim == 0:
        cdnc_factor = jnp.array(aerosol_data.cdnc_factor)  # Scalar from vmap
    else:
        cdnc_factor = aerosol_data.cdnc_factor  # Take first element if array
    
    # Calculate location-specific solar geometry using ICON's function
    # This handles JAX tracers correctly and avoids float(tracer) issues
    toa_flux, cos_zenith = calculate_solar_radiation_gcm(
        day_of_year=day_of_year,
        seconds_since_midnight=seconds_since_midnight,
        longitude=longitude,
        latitude=latitude,
        solar_constant=parameters.solar_constant
    )
    
    # Prepare ICON radiation state
    icon_state = prepare_radiation_state(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        cos_zenith=cos_zenith,
        ozone_vmr=ozone_vmr
    )
    
    # Convert to RRTMGP input format
    rrtmgp_input = prepare_rrtmgp_data(
        icon_state, 
        layer_thickness,
        cdnc_factor
    )
    
    # Run RRTMGP radiative transfer with dynamic solar parameters
    rrtmgp_output = radiation_scheme_rrtmgp_fn(rrtmgp_input, toa_flux, cos_zenith)
    
    # Convert outputs back to ICON format
    tendencies, diagnostics = prepare_icon_data(rrtmgp_output, icon_state)
    
    return tendencies, diagnostics