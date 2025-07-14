"""
Physics data structures for ICON atmospheric physics

This module defines the data structures that hold state and diagnostics
for ICON physics parameterizations, following the SpeedyPhysics pattern.

Date: 2025-01-11
"""

import jax.numpy as jnp
import tree_math
from jcm.date import DateData
from jax import tree_util
from typing import Optional


@tree_math.struct
class RadiationData:
    """Data for radiation calculations"""
    
    # Solar/geometric variables
    cos_zenith: jnp.ndarray           # Cosine solar zenith angle [1] (ncols,)
    
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
            sw_flux_up=jnp.zeros((nlev+1,) + nodal_shape),
            sw_flux_down=jnp.zeros((nlev+1,) + nodal_shape),
            sw_heating_rate=jnp.zeros((nlev,) + nodal_shape),
            lw_flux_up=jnp.zeros((nlev+1,) + nodal_shape),
            lw_flux_down=jnp.zeros((nlev+1,) + nodal_shape),
            lw_heating_rate=jnp.zeros((nlev,) + nodal_shape),
            surface_sw_down=jnp.zeros(nodal_shape),
            surface_lw_down=jnp.zeros(nodal_shape),
            surface_sw_up=jnp.zeros(nodal_shape),
            surface_lw_up=jnp.zeros(nodal_shape),
            toa_sw_up=jnp.zeros(nodal_shape),
            toa_lw_up=jnp.zeros(nodal_shape),
            toa_sw_down=jnp.zeros(nodal_shape),
            toa_lw_down=jnp.zeros(nodal_shape)
        )
    
    def copy(self, **kwargs):
        new_data = {
            'cos_zenith': self.cos_zenith,
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


@tree_math.struct
class ConvectionData:
    """Data for convection calculations"""
    
    # Mass fluxes
    mass_flux_up: jnp.ndarray        # Updraft mass flux [kg/m²/s] (nlev, ncols)
    mass_flux_down: jnp.ndarray      # Downdraft mass flux [kg/m²/s] (nlev, ncols)
    
    # Convective properties
    cloud_base: jnp.ndarray          # Cloud base level index [1] (ncols,)
    cloud_top: jnp.ndarray           # Cloud top level index [1] (ncols,)
    cape: jnp.ndarray                # CAPE [J/kg] (ncols,)
    
    # Precipitation
    precip_conv: jnp.ndarray         # Convective precipitation [kg/m²/s] (ncols,)
    
    # Cloud water/ice
    qc_conv: jnp.ndarray             # Convective cloud water [kg/kg] (nlev, ncols)
    qi_conv: jnp.ndarray             # Convective cloud ice [kg/kg] (nlev, ncols)
    
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            mass_flux_up=jnp.zeros((nlev,) + nodal_shape),
            mass_flux_down=jnp.zeros((nlev,) + nodal_shape),
            cloud_base=jnp.zeros(nodal_shape, dtype=int),
            cloud_top=jnp.zeros(nodal_shape, dtype=int),
            cape=jnp.zeros(nodal_shape),
            precip_conv=jnp.zeros(nodal_shape),
            qc_conv=jnp.zeros((nlev,) + nodal_shape),
            qi_conv=jnp.zeros((nlev,) + nodal_shape),
        )
    
    def copy(self, **kwargs):
        new_data = {
            'mass_flux_up': self.mass_flux_up,
            'mass_flux_down': self.mass_flux_down,
            'cloud_base': self.cloud_base,
            'cloud_top': self.cloud_top,
            'cape': self.cape,
            'precip_conv': self.precip_conv,
            'qc_conv': self.qc_conv,
            'qi_conv': self.qi_conv,
        }
        new_data.update(kwargs)
        return ConvectionData(**new_data)


@tree_math.struct
class CloudData:
    """Data for cloud physics"""
    
    # Cloud fraction
    cloud_fraction: jnp.ndarray      # Cloud fraction [1] (nlev, ncols)
    
    # Cloud condensate
    qc: jnp.ndarray                  # Cloud water [kg/kg] (nlev, ncols)
    qi: jnp.ndarray                  # Cloud ice [kg/kg] (nlev, ncols)
    qr: jnp.ndarray                  # Rain water [kg/kg] (nlev, ncols)
    qs: jnp.ndarray                  # Snow [kg/kg] (nlev, ncols)
    
    # Precipitation
    precip_rain: jnp.ndarray         # Rain precipitation [kg/m²/s] (ncols,)
    precip_snow: jnp.ndarray         # Snow precipitation [kg/m²/s] (ncols,)

    # Cloud properties
    # These can be used for diagnostics or further calculations
    droplet_number: jnp.ndarray  # Droplet number concentration [1/m³] (nlev, ncols)
        
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            cloud_fraction=jnp.zeros((nlev,) + nodal_shape),
            qc=jnp.zeros((nlev,) + nodal_shape),
            qi=jnp.zeros((nlev,) + nodal_shape),
            qr=jnp.zeros((nlev,) + nodal_shape),
            qs=jnp.zeros((nlev,) + nodal_shape),
            precip_rain=jnp.zeros(nodal_shape),
            precip_snow=jnp.zeros(nodal_shape),
            droplet_number=jnp.zeros((nlev,) + nodal_shape),
        )

    def copy(self, **kwargs):
        new_data = {
            'cloud_fraction': self.cloud_fraction,
            'qc': self.qc,
            'qi': self.qi,
            'qr': self.qr,
            'qs': self.qs,
            'precip_rain': self.precip_rain,
            'precip_snow': self.precip_snow,
            'droplet_number': self.droplet_number,
        }
        new_data.update(kwargs)
        return CloudData(**new_data)


@tree_math.struct
class VerticalDiffusionData:
    """Data for vertical diffusion and boundary layer"""
    
    # Exchange coefficients
    km: jnp.ndarray                  # Momentum exchange coeff [m²/s] (nlev+1, ncols)
    kh: jnp.ndarray                  # Heat exchange coeff [m²/s] (nlev+1, ncols)
    
    # Turbulent kinetic energy
    tke: jnp.ndarray                 # TKE [m²/s²] (nlev, ncols)
    
    # Boundary layer diagnostics
    pbl_height: jnp.ndarray          # PBL height [m] (ncols,)
    surface_friction_velocity: jnp.ndarray  # u* [m/s] (ncols,)
    monin_obukhov_length: jnp.ndarray       # L [m] (ncols,)
    
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            km=jnp.zeros((nlev+1,) + nodal_shape),
            kh=jnp.zeros((nlev+1,) + nodal_shape),
            tke=jnp.zeros((nlev,) + nodal_shape),
            pbl_height=jnp.zeros(nodal_shape),
            surface_friction_velocity=jnp.zeros(nodal_shape),
            monin_obukhov_length=jnp.zeros(nodal_shape),
        )
    
    def copy(self, **kwargs):
        new_data = {
            'km': self.km,
            'kh': self.kh,
            'tke': self.tke,
            'pbl_height': self.pbl_height,
            'surface_friction_velocity': self.surface_friction_velocity,
            'monin_obukhov_length': self.monin_obukhov_length,
        }
        new_data.update(kwargs)
        return VerticalDiffusionData(**new_data)


@tree_math.struct
class SurfaceData:
    """Data for surface physics"""
    
    # Surface fluxes
    sensible_heat_flux: jnp.ndarray  # Sensible heat flux [W/m²] (ncols,)
    latent_heat_flux: jnp.ndarray    # Latent heat flux [W/m²] (ncols,)
    momentum_flux_u: jnp.ndarray     # U momentum flux [N/m²] (ncols,)
    momentum_flux_v: jnp.ndarray     # V momentum flux [N/m²] (ncols,)
    
    # Surface temperatures
    surface_temperature: jnp.ndarray # Surface temperature [K] (ncols,)
    skin_temperature: jnp.ndarray    # Skin temperature [K] (ncols,)
    
    # Evaporation
    evaporation: jnp.ndarray         # Evaporation [kg/m²/s] (ncols,)
    
    # Exchange coefficients
    ch: jnp.ndarray                  # Heat exchange coefficient [1] (ncols,)
    cm: jnp.ndarray                  # Momentum exchange coefficient [1] (ncols,)
    
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            sensible_heat_flux=jnp.zeros(nodal_shape),
            latent_heat_flux=jnp.zeros(nodal_shape),
            momentum_flux_u=jnp.zeros(nodal_shape),
            momentum_flux_v=jnp.zeros(nodal_shape),
            surface_temperature=jnp.zeros(nodal_shape),
            skin_temperature=jnp.zeros(nodal_shape),
            evaporation=jnp.zeros(nodal_shape),
            ch=jnp.zeros(nodal_shape),
            cm=jnp.zeros(nodal_shape),
        )
    
    def copy(self, **kwargs):
        new_data = {
            'sensible_heat_flux': self.sensible_heat_flux,
            'latent_heat_flux': self.latent_heat_flux,
            'momentum_flux_u': self.momentum_flux_u,
            'momentum_flux_v': self.momentum_flux_v,
            'surface_temperature': self.surface_temperature,
            'skin_temperature': self.skin_temperature,
            'evaporation': self.evaporation,
            'ch': self.ch,
            'cm': self.cm,
        }
        new_data.update(kwargs)
        return CloudData(**new_data)


@tree_math.struct
class DiagnosticData:
    """Diagnostic data computed from state"""
    
    # Pressure and height
    pressure_full: jnp.ndarray       # Pressure at full levels [Pa] (nlev, ncols)
    pressure_half: jnp.ndarray       # Pressure at half levels [Pa] (nlev+1, ncols)
    height_full: jnp.ndarray         # Height at full levels [m] (nlev, ncols)
    height_half: jnp.ndarray         # Height at half levels [m] (nlev+1, ncols)

    relative_humidity: jnp.ndarray  # Relative humidity [1] (nlev, ncols)
    surface_pressure: jnp.ndarray  # Surface pressure [Pa] (ncols,)
    
    # Air density and layer thickness
    air_density: jnp.ndarray         # Air density [kg/m³] (nlev, ncols)
    layer_thickness: jnp.ndarray     # Layer thickness [m] (nlev, ncols)
    
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            pressure_full=jnp.zeros((nlev,) + nodal_shape),
            pressure_half=jnp.zeros((nlev+1,) + nodal_shape),
            height_full=jnp.zeros((nlev,) + nodal_shape),
            height_half=jnp.zeros((nlev+1,) + nodal_shape),
            relative_humidity=jnp.zeros((nlev,) + nodal_shape),
            surface_pressure=jnp.zeros(nodal_shape),
            air_density=jnp.zeros((nlev,) + nodal_shape),
            layer_thickness=jnp.zeros((nlev,) + nodal_shape),
        )
    
    def copy(self, **kwargs):
        new_data = {
            'pressure_full': self.pressure_full,
            'pressure_half': self.pressure_half,
            'height_full': self.height_full,
            'height_half': self.height_half,
            'relative_humidity': self.relative_humidity,
            'surface_pressure': self.surface_pressure,
            'air_density': self.air_density,
            'layer_thickness': self.layer_thickness,
        }
        new_data.update(kwargs)
        return DiagnosticData(**new_data)


@tree_math.struct
class AerosolData:
    """Data for aerosol calculations"""
    
    # Aerosol optical properties by level
    aod_profile: jnp.ndarray         # AOD profile [1] (nlev, ncols)
    ssa_profile: jnp.ndarray         # SSA profile [1] (nlev, ncols)
    asy_profile: jnp.ndarray         # Asymmetry parameter profile [1] (nlev, ncols)
    
    # Column-integrated properties
    aod_total: jnp.ndarray           # Total column AOD [1] (ncols,)
    aod_anthropogenic: jnp.ndarray   # Anthropogenic AOD [1] (ncols,)
    aod_background: jnp.ndarray      # Background AOD [1] (ncols,)
    
    # For Twomey effect (cloud-aerosol interactions)
    cdnc_factor: jnp.ndarray         # CDNC modification factor [1] (ncols,)
    
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            aod_profile=jnp.zeros((nlev,) + nodal_shape),
            ssa_profile=jnp.zeros((nlev,) + nodal_shape),
            asy_profile=jnp.zeros((nlev,) + nodal_shape),
            aod_total=jnp.zeros(nodal_shape),
            aod_anthropogenic=jnp.zeros(nodal_shape),
            aod_background=jnp.zeros(nodal_shape),
            cdnc_factor=jnp.ones(nodal_shape),  # Start with factor of 1.0
        )
    
    def copy(self, **kwargs):
        new_data = {
            'aod_profile': self.aod_profile,
            'ssa_profile': self.ssa_profile,
            'asy_profile': self.asy_profile,
            'aod_total': self.aod_total,
            'aod_anthropogenic': self.aod_anthropogenic,
            'aod_background': self.aod_background,
            'cdnc_factor': self.cdnc_factor,
        }
        new_data.update(kwargs)
        return AerosolData(**new_data)


@tree_math.struct
class PhysicsData:
    """Main physics data container for ICON physics"""
    
    date: DateData
    diagnostics: DiagnosticData
    radiation: RadiationData
    convection: ConvectionData
    clouds: CloudData
    vertical_diffusion: VerticalDiffusionData
    surface: SurfaceData
    aerosol: AerosolData
    
    @classmethod
    def zeros(cls, nodal_shape, nlev, date=None):
        return cls(
            date=date if date is not None else DateData.zeros(),
            diagnostics=DiagnosticData.zeros(nodal_shape, nlev),
            radiation=RadiationData.zeros(nodal_shape, nlev),
            convection=ConvectionData.zeros(nodal_shape, nlev),
            clouds=CloudData.zeros(nodal_shape, nlev),
            vertical_diffusion=VerticalDiffusionData.zeros(nodal_shape, nlev),
            surface=SurfaceData.zeros(nodal_shape, nlev),
            aerosol=AerosolData.zeros(nodal_shape, nlev),
        )
    
    def copy(self, **kwargs):
        new_data = {
            'date': self.date,
            'diagnostics': self.diagnostics,
            'radiation': self.radiation,
            'convection': self.convection,
            'clouds': self.clouds,
            'vertical_diffusion': self.vertical_diffusion,
            'surface': self.surface,
            'aerosol': self.aerosol,
        }
        new_data.update(kwargs)
        return PhysicsData(**new_data)