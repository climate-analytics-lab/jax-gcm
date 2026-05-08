"""Type definitions and parameters for radiation calculations

This module defines the data structures and configuration parameters
used throughout the radiation scheme.

Date: 2025-01-10
"""

import jax.numpy as jnp
from typing import NamedTuple, Optional
import tree_math

from .constants import (
    N_SW_BANDS, N_LW_BANDS, SW_BAND_LIMITS, LW_BAND_LIMITS,
)


@tree_math.struct
class RadiationParameters:
    """Configuration parameters for radiation scheme"""

    # Seconds between full radiation calls. Heating rates from the most
    # recent call are cached and reused by ``_radiation_with_caching`` on
    # intermediate dynamics steps. Default 7200 s (2 hours) matches the
    # standard ECHAM/ICON convention — radiation is the most expensive
    # physics component and varies on hour timescales, so calling it
    # every dynamics step (``radiation_interval=0``) wastes ~10× the
    # runtime of the dynamics for negligible accuracy gain. Set to 0 if
    # you want every-step computation (e.g. for diagnostic comparisons
    # against the dynamics-step heating rate).
    radiation_interval: float

    # Solar parameters
    solar_constant: float    # Solar constant (W/m²)

    # Spectral bands
    n_sw_bands: int          # Number of shortwave bands
    n_lw_bands: int          # Number of longwave bands

    # Band limits (wavenumber in cm⁻¹)
    lw_band_limits: tuple    # LW bands
    sw_band_limits: tuple    # SW bands

    # Gas concentrations (volume mixing ratios)
    co2_vmr: float           # CO2 volume mixing ratio
    ch4_vmr: float           # CH4 volume mixing ratio
    n2o_vmr: float           # N2O volume mixing ratio

    # Numerical parameters
    min_cos_zenith: float    # Minimum cosine solar zenith angle (~88 deg)
    flux_epsilon: float      # Small value for flux calculations

    # Cloud optics parameters
    cld_tau_min: float       # Minimum cloud optical depth
    cld_frac_min: float      # Minimum cloud fraction

    # Neural-network emulator (only used when radiation_scheme="emulated")
    emulator_weights: Optional[object] = None  # EmulatorWeights pytree
    sw_scaling: Optional[object] = None        # InputScaling for SW network
    lw_scaling: Optional[object] = None        # InputScaling for LW network

    @classmethod
    def default(cls, radiation_interval=7200.0,
                 solar_constant=1361.0,
                 n_sw_bands=N_SW_BANDS, n_lw_bands=N_LW_BANDS,
                 lw_band_limits=LW_BAND_LIMITS,
                 sw_band_limits=SW_BAND_LIMITS,
                 co2_vmr=400e-6, ch4_vmr=1.8e-6, n2o_vmr=0.32e-6,
                 min_cos_zenith=0.035, flux_epsilon=1e-6,
                 cld_tau_min=1e-6, cld_frac_min=1e-3,
                 emulator_weights=None, sw_scaling=None,
                 lw_scaling=None) -> 'RadiationParameters':
        """Return default radiation parameters"""
        return cls(
            radiation_interval=jnp.array(radiation_interval),
            solar_constant=jnp.array(solar_constant),
            n_sw_bands=jnp.asarray(n_sw_bands),
            n_lw_bands=jnp.asarray(n_lw_bands),
            lw_band_limits=jnp.asarray(lw_band_limits),
            sw_band_limits=jnp.asarray(sw_band_limits),
            co2_vmr=jnp.array(co2_vmr),
            ch4_vmr=jnp.array(ch4_vmr),
            n2o_vmr=jnp.array(n2o_vmr),
            min_cos_zenith=jnp.array(min_cos_zenith),
            flux_epsilon=jnp.array(flux_epsilon),
            cld_tau_min=jnp.array(cld_tau_min),
            cld_frac_min=jnp.array(cld_frac_min),
            emulator_weights=emulator_weights,
            sw_scaling=sw_scaling,
            lw_scaling=lw_scaling,
        )


@tree_math.struct
class RadiationData:
    """Radiation diagnostics shared by every radiation scheme.

    Written by the radiation term on its compute step (and re-used,
    unchanged, on cached steps); seeded by ``EchamBoundaryConditions``
    which fills the surface optical properties and zeros the flux
    fields. Lives next to :class:`RadiationParameters` so the radiation
    schemes (grey two-stream, RRTMGP, NN emulator) share one home.
    """

    # Solar/geometric variables
    cos_zenith: jnp.ndarray           # Cosine solar zenith angle [1] (ncols,)

    # Surface properties
    surface_albedo_vis: jnp.ndarray    # Surface albedo visible [1] (ncols,)
    surface_albedo_nir: jnp.ndarray    # Surface albedo near-infrared [1] (ncols,)
    surface_emissivity: jnp.ndarray    # Surface emissivity [1] (ncols,)

    # Shortwave fluxes
    sw_flux_up: jnp.ndarray          # Upward SW flux [W/m²] (nlev+1, ncols)
    sw_flux_down: jnp.ndarray        # Downward SW flux [W/m²] (nlev+1, ncols)
    sw_heating_rate: jnp.ndarray     # SW heating rate [K/s] (nlev, ncols)

    # Longwave fluxes
    lw_flux_up: jnp.ndarray          # Upward LW flux [W/m²] (nlev+1, ncols)
    lw_flux_down: jnp.ndarray        # Downward LW flux [W/m²] (nlev+1, ncols)
    lw_heating_rate: jnp.ndarray     # LW heating rate [K/s] (nlev, ncols)

    # Surface fluxes
    surface_sw_down: jnp.ndarray     # Surface downward SW [W/m²] (ncols,)
    surface_lw_down: jnp.ndarray     # Surface downward LW [W/m²] (ncols,)
    surface_sw_up: jnp.ndarray       # Surface upward SW [W/m²] (ncols,)
    surface_lw_up: jnp.ndarray       # Surface upward LW [W/m²] (ncols,)

    # TOA fluxes
    toa_sw_up: jnp.ndarray           # TOA upward SW [W/m²] (ncols,)
    toa_lw_up: jnp.ndarray           # TOA upward LW (OLR) [W/m²] (ncols,)
    toa_sw_down: jnp.ndarray         # TOA downward SW [W/m²] (ncols,)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            cos_zenith=jnp.zeros(nodal_shape),
            surface_albedo_vis=jnp.zeros(nodal_shape),
            surface_albedo_nir=jnp.zeros(nodal_shape),
            surface_emissivity=jnp.zeros(nodal_shape),
            sw_flux_up=jnp.zeros((nlev + 1,) + nodal_shape),
            sw_flux_down=jnp.zeros((nlev + 1,) + nodal_shape),
            sw_heating_rate=jnp.zeros((nlev,) + nodal_shape),
            lw_flux_up=jnp.zeros((nlev + 1,) + nodal_shape),
            lw_flux_down=jnp.zeros((nlev + 1,) + nodal_shape),
            lw_heating_rate=jnp.zeros((nlev,) + nodal_shape),
            surface_sw_down=jnp.zeros(nodal_shape),
            surface_lw_down=jnp.zeros(nodal_shape),
            surface_sw_up=jnp.zeros(nodal_shape),
            surface_lw_up=jnp.zeros(nodal_shape),
            toa_sw_up=jnp.zeros(nodal_shape),
            toa_lw_up=jnp.zeros(nodal_shape),
            toa_sw_down=jnp.zeros(nodal_shape),
        )

    def copy(self, **kwargs):
        new_data = {
            'cos_zenith': self.cos_zenith,
            'surface_albedo_vis': self.surface_albedo_vis,
            'surface_albedo_nir': self.surface_albedo_nir,
            'surface_emissivity': self.surface_emissivity,
            'sw_flux_up': self.sw_flux_up,
            'sw_flux_down': self.sw_flux_down,
            'sw_heating_rate': self.sw_heating_rate,
            'lw_flux_up': self.lw_flux_up,
            'lw_flux_down': self.lw_flux_down,
            'lw_heating_rate': self.lw_heating_rate,
            'surface_sw_down': self.surface_sw_down,
            'surface_lw_down': self.surface_lw_down,
            'surface_sw_up': self.surface_sw_up,
            'surface_lw_up': self.surface_lw_up,
            'toa_sw_up': self.toa_sw_up,
            'toa_lw_up': self.toa_lw_up,
            'toa_sw_down': self.toa_sw_down,
        }
        new_data.update(kwargs)
        return RadiationData(**new_data)


class RadiationState(NamedTuple):
    """State variables for radiation calculations"""
    
    # Solar geometry
    cos_zenith: jnp.ndarray          # Cosine of solar zenith angle
    daylight_fraction: jnp.ndarray   # Fraction of grid cell in daylight
    
    # Atmospheric profiles
    temperature: jnp.ndarray         # Temperature (K) [nlev]
    pressure: jnp.ndarray            # Pressure (Pa) [nlev]
    pressure_interfaces: jnp.ndarray # Pressure at interfaces (Pa) [nlev+1]
    
    # Gas mixing ratios
    h2o_vmr: jnp.ndarray            # Water vapor volume mixing ratio [nlev]
    o3_vmr: jnp.ndarray             # Ozone volume mixing ratio [nlev]
    
    # Cloud properties
    cloud_fraction: jnp.ndarray      # Cloud fraction [nlev]
    cloud_water_path: jnp.ndarray    # Cloud water path (kg/m²) [nlev]
    cloud_ice_path: jnp.ndarray      # Cloud ice path (kg/m²) [nlev]
    
    # Aerosol properties (optional)
    aerosol_optical_depth: jnp.ndarray     # Aerosol optical depth [nlev, nbands]
    aerosol_ssa: jnp.ndarray               # Aerosol single scatter albedo [nlev, nbands]
    aerosol_asymmetry: jnp.ndarray         # Aerosol asymmetry factor [nlev, nbands]


class RadiationFluxes(NamedTuple):
    """Radiation fluxes at interfaces"""
    
    # Shortwave fluxes (W/m²) [nlev+1]
    sw_down: jnp.ndarray             # Downward shortwave flux
    sw_up: jnp.ndarray               # Upward shortwave flux
    sw_down_clear: jnp.ndarray       # Clear-sky downward shortwave
    sw_up_clear: jnp.ndarray         # Clear-sky upward shortwave
    
    # Longwave fluxes (W/m²) [nlev+1]
    lw_down: jnp.ndarray             # Downward longwave flux
    lw_up: jnp.ndarray               # Upward longwave flux
    lw_down_clear: jnp.ndarray       # Clear-sky downward longwave
    lw_up_clear: jnp.ndarray         # Clear-sky upward longwave
    
    # Surface components (W/m²)
    sw_down_vis_dir: jnp.ndarray     # Direct visible at surface
    sw_down_vis_dif: jnp.ndarray     # Diffuse visible at surface
    sw_down_nir_dir: jnp.ndarray     # Direct near-IR at surface
    sw_down_nir_dif: jnp.ndarray     # Diffuse near-IR at surface


class RadiationTendencies(NamedTuple):
    """Tendencies from radiation"""
    
    temperature_tendency: jnp.ndarray # Temperature tendency (K/s) [nlev]
    longwave_heating: jnp.ndarray     # Longwave heating rate (K/s) [nlev]
    shortwave_heating: jnp.ndarray    # Shortwave heating rate (K/s) [nlev]


class OpticalProperties(NamedTuple):
    """Optical properties for radiative transfer"""
    
    optical_depth: jnp.ndarray       # Layer optical depth [nlev, nbands]
    single_scatter_albedo: jnp.ndarray  # Single scattering albedo [nlev, nbands]
    asymmetry_factor: jnp.ndarray    # Asymmetry factor [nlev, nbands]


@tree_math.struct
class SpectralBands:
    """Definition of spectral bands"""
    
    # Shortwave bands (wavelength in micrometers)
    sw_band_limits: tuple = (
        (0.2, 0.7),    # Visible
        (0.7, 4.0),    # Near-IR
    )
    
    # Longwave bands (wavenumber in cm⁻¹)
    lw_band_limits: tuple = (
        (10, 350),     # Window region
        (350, 500),    # CO2 band
        (500, 2500),   # H2O bands
    )
    
    # Band weights for solar spectrum
    sw_solar_fraction: tuple = (0.5, 0.5)  # Simplified equal split