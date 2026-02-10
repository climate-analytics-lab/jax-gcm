"""
RRTMGP-based radiation scheme for ICON physics.

This module integrates jax-rrtmgp with ICON's radiation interface, handling:
- Location-specific solar geometry via jax_solar (OrbitalTime, radiation_flux, get_solar_sin_altitude)
- ICON vertical ordering (TOA→surface) vs RRTMGP (surface→TOA) conversion
- Halo management (temperature NaN-padded for RRTMGP fill; others edge-filled)
- Stretched grid mapping for non-uniform vertical coordinates
- Unit conversions and cloud effective radii from ICON functions
- Output conversion to ICON's RadiationTendencies and RadiationData formats

Key entry point: `radiation_scheme_rrtmgp` - ICON-signature (date, surface_*, pressure_interfaces) drop-in for ICON's radiation_scheme.
"""

from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import jax.numpy as jnp
from jax import lax

from jax_solar import OrbitalTime, radiation_flux, get_solar_sin_altitude
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

# Representative surface/solar defaults for RRTMGP instance (actual values set per grid cell)
_SFC_EMIS_DEFAULT = 0.98
_SFC_ALB_DEFAULT = 0.07
_SOLAR_CONSTANT_DEFAULT = 1361.0

# Global RRTMGP instance (created once at module load to avoid JAX tracer issues)
# Uses representative values for configuration; actual solar geometry calculated per gridcell
_GLOBAL_RRTMGP_INSTANCE = RRTMGP(
    radiative_transfer_cfg=radiative_transfer.RadiativeTransfer(
        optics=_BASE_RRTMGP_OPTICS,
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_emis=_SFC_EMIS_DEFAULT,
            sfc_alb=_SFC_ALB_DEFAULT,
            zenith=1.0,                                   # updated per grid cell
            irrad=_SOLAR_CONSTANT_DEFAULT,                # updated per grid cell
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
    surface_temperature: jnp.ndarray,
    land_fraction: float = 0.5
) -> dict:
    """Convert ICON RadiationState to RRTMGP inputs.

    Args:
        icon_data: ICON RadiationState with atmospheric profiles
        layer_thickness: Layer thickness (m) [nlev]
        cdnc_factor: Cloud droplet number concentration factor
        surface_temperature: Surface temperature (K); scalar or 0-d array
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
        'sfc_temperature': jnp.reshape(surface_temperature, (1, 1)),  # Surface temperature [K]
        'p_ref_xxc': to3d_fill(pressure_1d),               # Pressure [Pa]
        'sg_map': sg_map,                                   # Stretched grid mapping
        'use_scan': True                                    # Use scan for efficiency
    }


def prepare_icon_data(
    rrtmgp_data: dict,
    icon_data,
    surface_albedo_vis: jnp.ndarray,
    surface_albedo_nir: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Convert RRTMGP output to ICON RadiationTendencies and RadiationData.

    Args:
        rrtmgp_data: Raw RRTMGP diagnostic output dictionary
        icon_data: ICON RadiationState
        surface_albedo_vis: Surface visible albedo (scalar or array)
        surface_albedo_nir: Surface near-IR albedo (scalar or array)
        surface_emissivity: Surface emissivity (scalar or array)

    Returns:
        Tuple of (RadiationTendencies, RadiationData)
    """
    # Extract information from available data
    halo = 1
    nlev = icon_data.temperature.shape[0]
    cos_zenith = icon_data.cos_zenith[0]  # Extract scalar from 1-element array
    
    # Extract heating rates (remove halos)
    total_heating = rrtmgp_data['rad_heat_src'][0, 0, halo:halo+nlev]
    lw_heating = rrtmgp_data['rad_heat_lw_3d'][0, 0, halo:halo+nlev]
    sw_heating = rrtmgp_data['rad_heat_sw_3d'][0, 0, halo:halo+nlev]
    
    # Check if we need to reverse vertical order back to ICON convention
    # RRTMGP outputs in surface→TOA order, ICON expects TOA→surface order
    needs_reversal = _reverse_if_needed(icon_data.pressure)
    flip = lambda a: a[::-1]
    identity = lambda a: a

    # Conditionally reverse heating rates back to ICON order
    total_heating = lax.cond(needs_reversal, flip, identity, total_heating)
    lw_heating = lax.cond(needs_reversal, flip, identity, lw_heating)
    sw_heating = lax.cond(needs_reversal, flip, identity, sw_heating)

    # Create radiation tendencies (now in ICON's TOA→surface order)
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
    
    # Extract full flux profiles from RRTMGP output
    # RRTMGP already sums across spectral bands, so ngpts=1 by default
    sw_flux_up = rrtmgp_data['sw_flux_up_full'][0, :, :].transpose(1, 0)  # (nlev+1, ngpts)
    sw_flux_down = rrtmgp_data['sw_flux_down_full'][0, :, :].transpose(1, 0)  # (nlev+1, ngpts)
    lw_flux_up = rrtmgp_data['lw_flux_up_full'][0, :, :].transpose(1, 0)  # (nlev+1, ngpts)
    lw_flux_down = rrtmgp_data['lw_flux_down_full'][0, :, :].transpose(1, 0)  # (nlev+1, ngpts)

    # Reverse flux profiles if needed (ICON order: index 0 = TOA, index -1 = surface)
    sw_flux_up = lax.cond(needs_reversal, flip, identity, sw_flux_up)
    sw_flux_down = lax.cond(needs_reversal, flip, identity, sw_flux_down)
    lw_flux_up = lax.cond(needs_reversal, flip, identity, lw_flux_up)
    lw_flux_down = lax.cond(needs_reversal, flip, identity, lw_flux_down)

    # Create radiation diagnostics using ICON's RadiationData structure
    diagnostics = RadiationData(
        cos_zenith=cos_zenith,
        surface_albedo_vis=jnp.atleast_1d(surface_albedo_vis),
        surface_albedo_nir=jnp.atleast_1d(surface_albedo_nir),
        surface_emissivity=jnp.atleast_1d(surface_emissivity),
        sw_flux_up=sw_flux_up,
        sw_flux_down=sw_flux_down,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up,
        lw_flux_down=lw_flux_down,
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
    """Compute heating rates using RRTMGP with per-column solar parameters.

    Passes zenith and irrad into RRTMGP's compute_heating_rate so the call is
    pure and vmap over columns can be compiled and run in parallel.

    Args:
        rrtmgp_data: Dictionary of RRTMGP inputs (from prepare_rrtmgp_data)
        toa_flux: Top-of-atmosphere solar flux [W/m²] (from ICON calculation)
        cos_zenith: Cosine of solar zenith angle (from ICON calculation)

    Returns:
        Dictionary of RRTMGP outputs (heating rates and diagnostics)
    """
    zenith_angle = jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0))
    rrtmgp_output = _GLOBAL_RRTMGP_INSTANCE.compute_heating_rate(
        zenith=zenith_angle, irrad=toa_flux, **rrtmgp_data
    )
    return rrtmgp_output


def radiation_scheme_rrtmgp(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_albedo_vis: jnp.ndarray,
    surface_albedo_nir: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    date,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6,
) -> Tuple[RadiationTendencies, RadiationData]:
    """RRTMGP radiation scheme with ICON-compatible signature (date, surface_*, pressure_interfaces).

    Drop-in replacement for ICON's radiation_scheme. date must be a jax_datetime or have .dt
    (e.g. DateData); it is passed to jax_solar's OrbitalTime.from_datetime.
    """
    # Extract cloud droplet number concentration factor
    if aerosol_data.cdnc_factor.ndim == 0:
        cdnc_factor = jnp.array(aerosol_data.cdnc_factor)  # Scalar from vmap
    else:
        cdnc_factor = aerosol_data.cdnc_factor  # Take first element if array
    # Solar geometry via jax_solar; date must be jax_datetime or have .dt (e.g. DateData)
    actual_date = getattr(date, "dt", date)
    orbital_time = OrbitalTime.from_datetime(actual_date)
    toa_flux = radiation_flux(orbital_time, longitude, latitude, parameters.solar_constant)
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    cos_zenith = sin_altitude  # cos(zenith) = sin(altitude)

    # Prepare ICON radiation state
    icon_state = prepare_radiation_state(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        pressure_interfaces=pressure_interfaces,
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
        cdnc_factor,
        surface_temperature,
    )
    
    # Run RRTMGP radiative transfer with dynamic solar parameters
    rrtmgp_output = radiation_scheme_rrtmgp_fn(rrtmgp_input, toa_flux, cos_zenith)
    
    # Convert outputs back to ICON format
    tendencies, diagnostics = prepare_icon_data(
        rrtmgp_output,
        icon_state,
        surface_albedo_vis,
        surface_albedo_nir,
        surface_emissivity,
    )
    return tendencies, diagnostics