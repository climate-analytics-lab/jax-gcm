"""Main radiation scheme interface for ECHAM physics

This module provides the main entry point for radiation calculations,
coordinating shortwave and longwave radiation computations.

Date: 2025-01-10
"""

import jax.numpy as jnp
from typing import Tuple, Optional

from ..radiation_types import (
    RadiationParameters,
    RadiationState,
    RadiationTendencies,
    OpticalProperties,
    RadiationData,
)

from jax_solar import radiation_flux, get_solar_sin_altitude, OrbitalTime
from jcm.forcing import SolarGeometry

from .gas_optics import gas_optical_depth_lw, gas_optical_depth_sw
from ..cloud_optics import cloud_optics
from ..mcica import column_total_cover, in_cloud_path
from .planck import planck_bands_lw
from .two_stream import longwave_fluxes, shortwave_fluxes, flux_to_heating_rate



def combine_optical_properties(
    gas_optical_depth: jnp.ndarray,
    cloud_optics: OpticalProperties,
    aerosol_optical_depth: Optional[jnp.ndarray] = None,
    aerosol_ssa: Optional[jnp.ndarray] = None,
    aerosol_asymmetry: Optional[jnp.ndarray] = None
) -> OpticalProperties:
    """Combine gas, cloud, and aerosol optical properties.
    
    Args:
        gas_optical_depth: Gas optical depth [nlev, nbands]
        cloud_optics: Cloud optical properties
        aerosol_optical_depth: Aerosol optical depth [nlev, nbands] 
        aerosol_ssa: Aerosol single scatter albedo [nlev, nbands]
        aerosol_asymmetry: Aerosol asymmetry factor [nlev, nbands]
        
    Returns:
        Combined optical properties

    """
    # Start with gas + cloud
    total_tau = gas_optical_depth + cloud_optics.optical_depth
    
    # If no aerosols, return gas + cloud
    if aerosol_optical_depth is None:
        return OpticalProperties(
            optical_depth=total_tau,
            single_scatter_albedo=cloud_optics.single_scatter_albedo,
            asymmetry_factor=cloud_optics.asymmetry_factor
        )
    
    # Ensure aerosol properties have the right shape for the current band structure
    nlev, nbands = total_tau.shape
    if aerosol_optical_depth.shape != (nlev, nbands):
        # If aerosol data doesn't match band structure, skip aerosol effects
        return OpticalProperties(
            optical_depth=total_tau,
            single_scatter_albedo=cloud_optics.single_scatter_albedo,
            asymmetry_factor=cloud_optics.asymmetry_factor
        )
    
    # Add aerosol optical depth
    total_tau_with_aerosol = total_tau + aerosol_optical_depth
    
    # Combine single scattering albedo (weighted by scattering optical depth)
    cloud_scattering = cloud_optics.optical_depth * cloud_optics.single_scatter_albedo
    aerosol_scattering = aerosol_optical_depth * aerosol_ssa
    total_scattering = cloud_scattering + aerosol_scattering
    
    combined_ssa = jnp.where(
        total_tau_with_aerosol > 0,
        total_scattering / total_tau_with_aerosol,
        0.0
    )
    
    # Combine asymmetry factor (weighted by scattering optical depth)
    cloud_g_weighted = cloud_scattering * cloud_optics.asymmetry_factor
    aerosol_g_weighted = aerosol_scattering * aerosol_asymmetry
    
    combined_g = jnp.where(
        total_scattering > 0,
        (cloud_g_weighted + aerosol_g_weighted) / total_scattering,
        0.0
    )
    
    return OpticalProperties(
        optical_depth=total_tau_with_aerosol,
        single_scatter_albedo=combined_ssa,
        asymmetry_factor=combined_g
    )


def prepare_radiation_state(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    cos_zenith: float,
    ozone_vmr: Optional[jnp.ndarray] = None,
    aerosol_optical_depth: Optional[jnp.ndarray] = None,
    aerosol_ssa: Optional[jnp.ndarray] = None,
    aerosol_asymmetry: Optional[jnp.ndarray] = None
) -> RadiationState:
    """Prepare radiation state from physics state variables.

    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        pressure_levels: Pressure at full levels (Pa) [nlev]
        pressure_interfaces: Pressure at half levels (Pa) [nlev+1]
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        cloud_water: Cloud water content (kg/kg) [nlev]
        cloud_ice: Cloud ice content (kg/kg) [nlev]
        cloud_fraction: Cloud fraction (0-1) [nlev]
        cos_zenith: Cosine of solar zenith angle
        ozone_vmr: Ozone volume mixing ratio [nlev]
        aerosol_optical_depth: Aerosol optical depth [nlev, nbands]
        aerosol_ssa: Aerosol single scatter albedo [nlev, nbands]
        aerosol_asymmetry: Aerosol asymmetry factor [nlev, nbands]

    Returns:
        RadiationState ready for radiation calculations

    """
    # Convert specific humidity to volume mixing ratio
    # q/(1-q) * Md/Mv where Md/Mv = 29/18 = 1.608
    # Clip to physical range — dynamics can produce q > 1 transiently
    q_clipped = jnp.clip(specific_humidity, 0.0, 0.99)
    h2o_vmr = q_clipped / (1 - q_clipped) * 1.608

    # Default ozone profile if not provided (simplified)
    if ozone_vmr is None:
        # Simple ozone profile peaking in stratosphere
        p_mb = jnp.maximum(pressure_levels / 100.0, 1e-3)  # Convert to mb, guard log
        ozone_vmr = jnp.where(
            p_mb < 100,  # Stratosphere
            5e-6 * jnp.exp(-((jnp.log(p_mb) - jnp.log(30)) ** 2) / 2),
            1e-6  # Troposphere
        )

    # Convert cloud water/ice from kg/kg to kg/m²
    # cloud_path = mixing_ratio * air_density * layer_thickness
    # Clip cloud fields to non-negative (dynamics can produce small negatives)
    cloud_water_path = jnp.maximum(cloud_water, 0.0) * air_density * layer_thickness
    cloud_ice_path = jnp.maximum(cloud_ice, 0.0) * air_density * layer_thickness

    # Use the model's pressure interfaces directly (already computed from sigma/hybrid levels)
    # pressure_interfaces should be [nlev+1] with TOA at index 0 and surface at index -1

    return RadiationState(
        cos_zenith=cos_zenith[jnp.newaxis],
        daylight_fraction=jnp.where(cos_zenith > 0, 1.0, 0.0)[jnp.newaxis],
        temperature=temperature,
        pressure=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        h2o_vmr=h2o_vmr,
        o3_vmr=ozone_vmr,
        cloud_fraction=cloud_fraction,
        cloud_water_path=cloud_water_path,
        cloud_ice_path=cloud_ice_path,
        aerosol_optical_depth=aerosol_optical_depth,
        aerosol_ssa=aerosol_ssa,
        aerosol_asymmetry=aerosol_asymmetry
    )



def radiation_scheme(
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
    solar: SolarGeometry,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,  # AerosolData from physics_data
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6
) -> Tuple[RadiationTendencies, RadiationData]:
    """Radiation scheme wrapper that extracts aerosol data and includes aerosol effects.

    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        pressure_levels: Pressure at full levels (Pa) [nlev]
        pressure_interfaces: Pressure at half levels (Pa) [nlev+1]
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        cloud_water: Cloud water content (kg/kg) [nlev]
        cloud_ice: Cloud ice content (kg/kg) [nlev]
        cloud_fraction: Cloud fraction (0-1) [nlev]
        surface_temperature: Surface temperature (K)
        surface_albedo_vis: Surface visible albedo
        surface_albedo_nir: Surface near-infrared albedo
        surface_emissivity: Surface emissivity
        solar: Precomputed solar/orbital geometry — see jcm.forcing.SolarGeometry.
            Replaces the legacy `date` argument; the radiation scheme no longer
            needs to know what calendar date it is.
        latitude: Latitude (degrees)
        longitude: Longitude (degrees)
        parameters: Radiation parameters
        aerosol_data: AerosolData containing optical properties
        ozone_vmr: Ozone volume mixing ratio [nlev]
        co2_vmr: CO2 volume mixing ratio

    Returns:
        Tuple of (radiation tendencies, radiation diagnostics)

    """
    nlev = temperature.shape[0]

    # Minimal math-safety guards only: positivity for fields used in
    # divisions, logs, or sqrt, and q < 1 for the `q / (1 - q)` conversion.
    # We intentionally do NOT clip T or q to physical ranges here — that
    # masks bugs in convection / cloud physics instead of surfacing them.
    pressure_levels = jnp.maximum(pressure_levels, 1.0)
    pressure_interfaces = jnp.maximum(pressure_interfaces, 1.0)
    layer_thickness = jnp.maximum(layer_thickness, 1.0)
    air_density = jnp.maximum(air_density, 1e-6)
    specific_humidity = jnp.clip(specific_humidity, 0.0, 0.99)
    cloud_water = jnp.maximum(cloud_water, 0.0)
    cloud_ice = jnp.maximum(cloud_ice, 0.0)
    cloud_fraction = jnp.clip(cloud_fraction, 0.0, 1.0)

    # Expand aerosol profiles to radiation bands with Angstrom spectral scaling
    # AOD(λ) = AOD(550nm) * (λ/0.55)^(-α)
    # Handle both 1D (single column from vmap) and 2D (full grid) aerosol data
    if aerosol_data.aod_profile.ndim == 1:
        # 1D case: single column from vmap
        aerosol_aod_col = aerosol_data.aod_profile  # [nlev]
        aerosol_ssa_col = aerosol_data.ssa_profile
        aerosol_asy_col = aerosol_data.asy_profile
        angstrom = aerosol_data.angstrom  # scalar from vmap
    else:
        # 2D case: take first column
        aerosol_aod_col = aerosol_data.aod_profile[:, 0]
        aerosol_ssa_col = aerosol_data.ssa_profile[:, 0]
        aerosol_asy_col = aerosol_data.asy_profile[:, 0]
        angstrom = aerosol_data.angstrom[0]

    # SW bands - use fixed default values (2 SW bands, 3 LW bands)
    default_n_sw_bands = 2
    default_n_lw_bands = 3

    # Compute representative wavelengths (μm) from SW band limits (wavenumbers cm⁻¹)
    # λ = 1e4 / ν_mid, where ν_mid is the midpoint wavenumber of the band
    sw_band_limits = parameters.sw_band_limits  # [[4000, 14500], [14500, 50000]]
    sw_wavelengths = 1e4 / ((sw_band_limits[:, 0] + sw_band_limits[:, 1]) / 2.0)

    # Apply Angstrom scaling: AOD(λ) = AOD(550nm) * (λ/0.55)^(-α)
    ref_wavelength = 0.55  # μm (550 nm reference)
    sw_scaling = (sw_wavelengths / ref_wavelength) ** (-angstrom)  # [n_sw_bands]

    aerosol_tau_sw = aerosol_aod_col[:, None] * sw_scaling[None, :]  # [nlev, n_sw_bands]
    aerosol_ssa_sw = jnp.tile(aerosol_ssa_col[:, None], (1, default_n_sw_bands))
    aerosol_asy_sw = jnp.tile(aerosol_asy_col[:, None], (1, default_n_sw_bands))

    # LW bands: apply Angstrom scaling (gives very small AOD at long wavelengths)
    lw_band_limits = parameters.lw_band_limits  # [[10, 350], [350, 500], [500, 2500]]
    lw_wavelengths = 1e4 / ((lw_band_limits[:, 0] + lw_band_limits[:, 1]) / 2.0)
    lw_scaling = (lw_wavelengths / ref_wavelength) ** (-angstrom)

    aerosol_tau_lw = aerosol_aod_col[:, None] * lw_scaling[None, :]  # [nlev, n_lw_bands]
    aerosol_ssa_lw = jnp.zeros((nlev, default_n_lw_bands))  # Pure absorption in LW
    aerosol_asy_lw = jnp.zeros((nlev, default_n_lw_bands))

    # Concatenate SW and LW
    aerosol_optical_depth = jnp.concatenate([aerosol_tau_sw, aerosol_tau_lw], axis=1)
    aerosol_ssa = jnp.concatenate([aerosol_ssa_sw, aerosol_ssa_lw], axis=1)
    aerosol_asymmetry = jnp.concatenate([aerosol_asy_sw, aerosol_asy_lw], axis=1)

    # Cloud droplet number concentration factor
    # Handle both 1D (single column from vmap) and 2D (full grid) cases
    if aerosol_data.cdnc_factor.ndim == 0:
        # 0D case: scalar from vmap
        cdnc_factor = aerosol_data.cdnc_factor
    else:
        # 1D case: take first element
        cdnc_factor = aerosol_data.cdnc_factor[0]
    
    # Now perform the actual radiation calculation
    
    # Solar radiation calculations. `solar` is precomputed by
    # `Model._get_step_fn_factory` ↔ `ForcingData.select(date)`, so the
    # radiation scheme stays date-free.
    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    toa_flux = radiation_flux(orbital_time, longitude, latitude, parameters.solar_constant)
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    cos_zenith = sin_altitude  # cos(zenith) = sin(altitude) since they are complementary

    # Clip pressure to positive so downstream log / divisions don't produce NaN
    pressure_levels = jnp.maximum(pressure_levels, 1.0)
    pressure_interfaces = jnp.maximum(pressure_interfaces, 1.0)

    # Prepare radiation state
    rad_state = prepare_radiation_state(
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
        ozone_vmr=ozone_vmr,
        aerosol_optical_depth=aerosol_optical_depth,
        aerosol_ssa=aerosol_ssa,
        aerosol_asymmetry=aerosol_asymmetry
    )
        
    # Calculate gas optical depths
    gas_tau_lw = gas_optical_depth_lw(
        temperature=temperature,
        pressure=pressure_levels,
        h2o_vmr=rad_state.h2o_vmr,
        o3_vmr=rad_state.o3_vmr,
        co2_vmr=co2_vmr,
        layer_thickness=layer_thickness,
        air_density=air_density,
    )
    
    gas_tau_sw = gas_optical_depth_sw(
        temperature=temperature,
        pressure=pressure_levels,
        h2o_vmr=rad_state.h2o_vmr,
        o3_vmr=rad_state.o3_vmr,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cos_zenith=cos_zenith
    )
    
    # Beam-split partial-cloud treatment: run radiative transfer twice,
    # once through a clear-sky column and once through a fully-cloudy
    # column with in-cloud LWP/IWP scaled up by 1/cf, then combine the
    # fluxes with the column-total cloud cover c_col. This captures the
    # plane-parallel inhomogeneity bias the homogeneous-cloud
    # approximation misses, at twice the radiative-transfer cost. For
    # canonical McICA see the RRTMGP path (rrtmgp.py) — there the
    # gpoint count makes per-gpoint sub-columns effectively free.
    in_cloud_lwp = in_cloud_path(
        rad_state.cloud_water_path, rad_state.cloud_fraction,
    )
    in_cloud_ipath = in_cloud_path(
        rad_state.cloud_ice_path, rad_state.cloud_fraction,
    )

    cloud_sw_optics_cloudy, cloud_lw_optics_cloudy = cloud_optics(
        cloud_water_path=in_cloud_lwp,
        cloud_ice_path=in_cloud_ipath,
        temperature=temperature,
        cdnc_factor=cdnc_factor,
    )
    zero_optics_sw = OpticalProperties(
        optical_depth=jnp.zeros_like(cloud_sw_optics_cloudy.optical_depth),
        single_scatter_albedo=jnp.zeros_like(
            cloud_sw_optics_cloudy.single_scatter_albedo,
        ),
        asymmetry_factor=jnp.zeros_like(
            cloud_sw_optics_cloudy.asymmetry_factor,
        ),
    )
    zero_optics_lw = OpticalProperties(
        optical_depth=jnp.zeros_like(cloud_lw_optics_cloudy.optical_depth),
        single_scatter_albedo=jnp.zeros_like(
            cloud_lw_optics_cloudy.single_scatter_albedo,
        ),
        asymmetry_factor=jnp.zeros_like(
            cloud_lw_optics_cloudy.asymmetry_factor,
        ),
    )

    sw_optics_clear = combine_optical_properties(
        gas_tau_sw, zero_optics_sw,
        aerosol_optical_depth[:, :default_n_sw_bands],
        aerosol_ssa[:, :default_n_sw_bands],
        aerosol_asymmetry[:, :default_n_sw_bands],
    )
    sw_optics_cloudy = combine_optical_properties(
        gas_tau_sw, cloud_sw_optics_cloudy,
        aerosol_optical_depth[:, :default_n_sw_bands],
        aerosol_ssa[:, :default_n_sw_bands],
        aerosol_asymmetry[:, :default_n_sw_bands],
    )

    lw_optics_clear = combine_optical_properties(
        gas_tau_lw, zero_optics_lw,
        aerosol_optical_depth[:, default_n_sw_bands:],
        aerosol_ssa[:, default_n_sw_bands:],
        aerosol_asymmetry[:, default_n_sw_bands:],
    )
    lw_optics_cloudy = combine_optical_properties(
        gas_tau_lw, cloud_lw_optics_cloudy,
        aerosol_optical_depth[:, default_n_sw_bands:],
        aerosol_ssa[:, default_n_sw_bands:],
        aerosol_asymmetry[:, default_n_sw_bands:],
    )

    # Calculate Planck functions for longwave
    lw_band_limits = parameters.lw_band_limits
    planck_layers = planck_bands_lw(temperature, lw_band_limits)
    planck_interfaces = planck_bands_lw(
        jnp.linspace(temperature[0], temperature[-1], nlev + 1),
        lw_band_limits
    )

    # Surface properties
    # Note: When vmapped, surface_temperature is a scalar; otherwise it's an array
    surface_temp_for_planck = surface_temperature if surface_temperature.ndim == 0 else surface_temperature
    surface_planck = planck_bands_lw(surface_temp_for_planck, lw_band_limits)
    if surface_planck.ndim > 1:
        surface_planck = surface_planck[0]

    # Note: When vmapped, surface properties are scalars; otherwise extract first element
    emissivity_val = surface_emissivity if surface_emissivity.ndim == 0 else surface_emissivity[0]

    # Two longwave RT calls, one per beam.
    flux_up_lw_clear, flux_down_lw_clear = longwave_fluxes(
        lw_optics_clear, planck_layers, planck_interfaces,
        emissivity_val, surface_planck, default_n_lw_bands,
    )
    flux_up_lw_cloudy, flux_down_lw_cloudy = longwave_fluxes(
        lw_optics_cloudy, planck_layers, planck_interfaces,
        emissivity_val, surface_planck, default_n_lw_bands,
    )

    # Calculate shortwave fluxes.  ``default_n_sw_bands`` is a Python int, so
    # this allocation has a statically-known shape and does not need a
    # ``max_bands`` buffer + mask.
    toa_flux_bands = jnp.full(
        (default_n_sw_bands,), toa_flux / max(default_n_sw_bands, 1)
    )

    # Note: When vmapped, albedos are scalars; otherwise extract first element
    albedo_vis_val = surface_albedo_vis if surface_albedo_vis.ndim == 0 else surface_albedo_vis[0]
    albedo_nir_val = surface_albedo_nir if surface_albedo_nir.ndim == 0 else surface_albedo_nir[0]
    surface_albedo_arr = jnp.array([albedo_vis_val, albedo_nir_val])

    # Two shortwave RT calls, one per beam. ``shortwave_fluxes`` also
    # returns direct/diffuse split components which we don't propagate
    # to ``RadiationData`` today (legacy behaviour).
    flux_up_sw_clear, flux_down_sw_clear, _, _ = shortwave_fluxes(
        sw_optics_clear, cos_zenith, toa_flux_bands,
        surface_albedo_arr, default_n_sw_bands,
    )
    flux_up_sw_cloudy, flux_down_sw_cloudy, _, _ = shortwave_fluxes(
        sw_optics_cloudy, cos_zenith, toa_flux_bands,
        surface_albedo_arr, default_n_sw_bands,
    )

    # Column-total cloud cover under the configured overlap rule, used
    # only as the scalar weight between the clear and cloudy beams.
    c_col = column_total_cover(
        rad_state.cloud_fraction, parameters.cloud_overlap,
    )
    flux_up_lw = (1.0 - c_col) * flux_up_lw_clear + c_col * flux_up_lw_cloudy
    flux_down_lw = (
        (1.0 - c_col) * flux_down_lw_clear + c_col * flux_down_lw_cloudy
    )
    flux_up_sw = (1.0 - c_col) * flux_up_sw_clear + c_col * flux_up_sw_cloudy
    flux_down_sw = (
        (1.0 - c_col) * flux_down_sw_clear + c_col * flux_down_sw_cloudy
    )
    
    # Zero out fluxes if sun is not up
    is_daylight = cos_zenith > 0
    flux_up_sw = jnp.where(is_daylight, flux_up_sw, 0.0)
    flux_down_sw = jnp.where(is_daylight, flux_down_sw, 0.0)
    
    # Convert fluxes to heating rates
    lw_heating_rate = flux_to_heating_rate(
        jnp.sum(flux_up_lw, axis=1), jnp.sum(flux_down_lw, axis=1),
        rad_state.pressure_interfaces
    )
    
    sw_heating_rate = flux_to_heating_rate(
        jnp.sum(flux_up_sw, axis=1), jnp.sum(flux_down_sw, axis=1),
        rad_state.pressure_interfaces
    )
    
    # Ensure SW heating is zero when no sunlight
    sw_heating_rate = jnp.where(is_daylight, sw_heating_rate, 0.0)
    
    total_heating = lw_heating_rate + sw_heating_rate
    
    # Extract diagnostic fluxes
    olr = jnp.sum(flux_up_lw[0, :])
    toa_sw_down = jnp.sum(flux_down_sw[0, :])
    toa_sw_up = jnp.sum(flux_up_sw[0, :])
    surface_sw_down = jnp.sum(flux_down_sw[-1, :])
    surface_sw_up = jnp.sum(flux_up_sw[-1, :])
    surface_lw_down = jnp.sum(flux_down_lw[-1, :])
    surface_lw_up = jnp.sum(flux_up_lw[-1, :])

    # Clear-sky TOA fluxes from the beam-split's clear branch — exposed
    # for cloud-radiative-effect diagnostics. SW: zero out at night to
    # match the all-sky convention used by ``flux_up_sw`` above.
    toa_sw_up_clear = jnp.where(
        is_daylight, jnp.sum(flux_up_sw_clear[0, :]), 0.0,
    )
    toa_lw_up_clear = jnp.sum(flux_up_lw_clear[0, :])
    
    # Create output structures
    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating_rate,
        shortwave_heating=sw_heating_rate
    )
    
    diagnostics = RadiationData(
        cos_zenith=cos_zenith[jnp.newaxis],
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=surface_emissivity,
        sw_flux_up=flux_up_sw,
        sw_flux_down=flux_down_sw,
        lw_flux_up=flux_up_lw,
        lw_flux_down=flux_down_lw,
        sw_heating_rate=sw_heating_rate,
        lw_heating_rate=lw_heating_rate,
        toa_sw_down=toa_sw_down,
        toa_sw_up=toa_sw_up,
        toa_lw_up=olr,
        surface_sw_down=surface_sw_down,
        surface_sw_up=surface_sw_up,
        surface_lw_down=surface_lw_down,
        surface_lw_up=surface_lw_up,
        toa_sw_up_clear=toa_sw_up_clear,
        toa_lw_up_clear=toa_lw_up_clear,
    )

    return tendencies, diagnostics


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.radiation.radiation_types import RadiationData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics.radiation import (  # noqa: E402
    cached_radiation_tendency,
    radiation_should_compute,
)
from jcm.physics.radiation.radiation_types import RadiationParameters  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


def _column_vector(value: jnp.ndarray, ncols: int) -> jnp.ndarray:
    """Return a vmapped scalar diagnostic as one value per column."""
    return jnp.reshape(value, (ncols,))


class GreyTwoStreamRadiation(PhysicsTerm):
    """Grey two-stream radiation as a composable PhysicsTerm.

    Wraps :func:`radiation_scheme`. Reads pressure / height / density
    from the moist-air diagnostics dict; reads cloud water / ice from
    state tracers and cloud fraction from the public ``"clouds"`` key;
    reads ozone / CO2 from ``"chemistry"``; reads aerosol optical
    properties from ``"aerosol"``; reads surface temperature from
    ``"surface"`` (still legacy until the surface migration) and
    surface albedos / emissivity from ``"radiation"`` (set by
    :class:`~jcm.physics.forcing.echam_boundary_conditions.EchamBoundaryConditions`).

    Caches its own heating-rate output across radiation sub-steps
    (when ``parameters.radiation_interval > 0``) by reading the
    previous-step ``RadiationData`` from ``diagnostics["radiation"]``
    and re-emitting it instead of re-running the scheme.
    """

    name: ClassVar[str] = "grey_two_stream_radiation"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half", "layer_thickness",
        "air_density", "chemistry", "aerosol",
        "radiation", "surface", "clouds",
    )
    provides: ClassVar[tuple[str, ...]] = ("radiation", "clouds")

    def __init__(self, params: RadiationParameters | None = None):
        """Hold the scheme-native :class:`RadiationParameters`."""
        self.params = nnx.Param(params or RadiationParameters.default())
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (deg) for the radiation scheme."""
        lat_deg = jnp.asarray(coords.horizontal.latitudes) * 180.0 / jnp.pi
        lon_deg = jnp.asarray(coords.horizontal.longitudes) * 180.0 / jnp.pi
        lat_2d, lon_2d = jnp.meshgrid(lat_deg, lon_deg)
        self._lats = nnx.Variable(lat_2d.reshape(-1))
        self._lons = nnx.Variable(lon_2d.reshape(-1))
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute or reuse cached radiative heating rates."""
        nlev, ncols = state.temperature.shape
        params = self.params.get_value()
        radiation = diagnostics["radiation"]

        def _compute():
            return self._compute_full(state, diagnostics, forcing, params)

        def _use_cached():
            return cached_radiation_tendency(
                radiation, state.temperature.shape,
            ), radiation

        tendency, new_radiation = jax.lax.cond(
            radiation_should_compute(diagnostics, params),
            _compute, _use_cached,
        )
        # Mirror the all-sky and clear-sky TOA fluxes onto the
        # ``"clouds"`` sub-struct so users can read everything CRE-
        # related (= toa_*_clear − toa_*_all) from a single diagnostic.
        clouds = diagnostics["clouds"].copy(
            toa_sw_up_all=new_radiation.toa_sw_up,
            toa_sw_up_clear=new_radiation.toa_sw_up_clear,
            toa_lw_up_all=new_radiation.toa_lw_up,
            toa_lw_up_clear=new_radiation.toa_lw_up_clear,
        )
        return tendency, {
            **diagnostics, "radiation": new_radiation, "clouds": clouds,
        }

    def _compute_full(
        self, state, diagnostics, forcing, params,
    ):
        """Run the full grey two-stream scheme, return (tendency, RadiationData)."""
        nlev, ncols = state.temperature.shape

        latitudes = self._lats.get_value()
        longitudes = self._lons.get_value()
        solar = forcing.solar

        cloud_water = state.tracers.get(
            "qc", jnp.zeros_like(state.temperature),
        )
        cloud_ice = state.tracers.get(
            "qi", jnp.zeros_like(state.temperature),
        )
        cloud_fraction = diagnostics["clouds"].cloud_fraction

        chemistry = diagnostics["chemistry"]
        # Convert ppmv → VMR. CO2 is well-mixed; pass as scalar.
        ozone_vmr = chemistry.ozone_vmr * 1e-6
        co2_vmr = jnp.mean(chemistry.co2_vmr) * 1e-6

        # Surface temperature still lives in the legacy "surface" key
        # (until the EchamSurface migration); the radiation surface
        # albedo / emissivity is on the "radiation" sub-struct.
        surface_temperature = diagnostics["surface"].surface_temperature.reshape(ncols)
        radiation = diagnostics["radiation"]
        surface_albedo_vis = radiation.surface_albedo_vis.reshape(ncols)
        surface_albedo_nir = radiation.surface_albedo_nir.reshape(ncols)
        surface_emissivity = radiation.surface_emissivity.reshape(ncols)

        # Reshape aerosol fields so column is the leading (mapped) axis.
        aerosol_in = diagnostics["aerosol"]
        aerosol_for_vmap = aerosol_in.copy(
            aod_profile=aerosol_in.aod_profile.reshape(nlev, ncols).T,
            ssa_profile=aerosol_in.ssa_profile.reshape(nlev, ncols).T,
            asy_profile=aerosol_in.asy_profile.reshape(nlev, ncols).T,
            cdnc_factor=aerosol_in.cdnc_factor.reshape(ncols),
            aod_total=aerosol_in.aod_total.reshape(ncols),
            aod_anthropogenic=aerosol_in.aod_anthropogenic.reshape(ncols),
            aod_background=aerosol_in.aod_background.reshape(ncols),
            angstrom=aerosol_in.angstrom.reshape(ncols),
        )

        tendencies_vmapped, diagnostics_vmapped = jax.vmap(
            radiation_scheme,
            # T, q, p_full, p_half, dz on axis 1
            in_axes=(
                1, 1, 1, 1, 1,
                1, 1, 1, 1,
                # surface scalars on axis 0
                0, 0, 0, 0,
                # solar None, lat/lon on axis 0
                None, 0, 0,
                # parameters None, aerosol per-col, ozone on axis 1, co2 None
                None, 0, 1, None,
            ),
            out_axes=(0, 0),
            axis_size=ncols,
        )(
            state.temperature, state.specific_humidity,
            diagnostics["pressure_full"], diagnostics["pressure_half"],
            diagnostics["layer_thickness"], diagnostics["air_density"],
            cloud_water, cloud_ice, cloud_fraction,
            surface_temperature, surface_albedo_vis,
            surface_albedo_nir, surface_emissivity,
            solar, latitudes, longitudes,
            params, aerosol_for_vmap, ozone_vmr, co2_vmr,
        )

        # Grey scheme keeps a per-band axis on the flux profiles; sum it
        # out and transpose back to (nlev+1, ncols).
        rad_out = RadiationData(
            cos_zenith=_column_vector(diagnostics_vmapped.cos_zenith, ncols),
            surface_albedo_vis=_column_vector(
                diagnostics_vmapped.surface_albedo_vis, ncols,
            ),
            surface_albedo_nir=_column_vector(
                diagnostics_vmapped.surface_albedo_nir, ncols,
            ),
            surface_emissivity=_column_vector(
                diagnostics_vmapped.surface_emissivity, ncols,
            ),
            sw_flux_up=diagnostics_vmapped.sw_flux_up.transpose(1, 0, 2).sum(axis=-1),
            sw_flux_down=diagnostics_vmapped.sw_flux_down.transpose(1, 0, 2).sum(axis=-1),
            sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
            lw_flux_up=diagnostics_vmapped.lw_flux_up.transpose(1, 0, 2).sum(axis=-1),
            lw_flux_down=diagnostics_vmapped.lw_flux_down.transpose(1, 0, 2).sum(axis=-1),
            lw_heating_rate=tendencies_vmapped.longwave_heating.T,
            surface_sw_down=_column_vector(
                diagnostics_vmapped.surface_sw_down, ncols,
            ),
            surface_lw_down=_column_vector(
                diagnostics_vmapped.surface_lw_down, ncols,
            ),
            surface_sw_up=_column_vector(
                diagnostics_vmapped.surface_sw_up, ncols,
            ),
            surface_lw_up=_column_vector(
                diagnostics_vmapped.surface_lw_up, ncols,
            ),
            toa_sw_up=_column_vector(diagnostics_vmapped.toa_sw_up, ncols),
            toa_lw_up=_column_vector(diagnostics_vmapped.toa_lw_up, ncols),
            toa_sw_down=_column_vector(
                diagnostics_vmapped.toa_sw_down, ncols,
            ),
            toa_sw_up_clear=_column_vector(
                diagnostics_vmapped.toa_sw_up_clear, ncols,
            ),
            toa_lw_up_clear=_column_vector(
                diagnostics_vmapped.toa_lw_up_clear, ncols,
            ),
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros((nlev, ncols)),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=tendencies_vmapped.temperature_tendency.T,
            specific_humidity=jnp.zeros((nlev, ncols)),
            tracers={},
        )
        return tendency, rad_out