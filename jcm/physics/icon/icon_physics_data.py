"""Physics data structures for ICON atmospheric physics

This module defines the data structures that hold state and diagnostics
for ICON physics parameterizations, following the SpeedyPhysics pattern.

Date: 2025-01-11
"""

import jax.numpy as jnp
import tree_math
from jcm.physics.icon.icon_coords import IconCoords


@tree_math.struct
class RadiationData:
    """Data for radiation calculations"""
    
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
    """Data for cloud physics."""

    # Cloud fraction
    cloud_fraction: jnp.ndarray      # Cloud fraction [1] (nlev, ncols)

    # Cloud condensate (updated by condensation within the cloud scheme)
    qc: jnp.ndarray                  # Cloud water [kg/kg] (nlev, ncols)
    qi: jnp.ndarray                  # Cloud ice [kg/kg] (nlev, ncols)

    # Surface precipitation (from microphysics autoconversion)
    precip_rain: jnp.ndarray         # Rain precipitation [kg/m²/s] (ncols,)
    precip_snow: jnp.ndarray         # Snow precipitation [kg/m²/s] (ncols,)

    # Cloud properties
    droplet_number: jnp.ndarray  # Droplet number concentration [1/m³] (nlev, ncols)

    # Previous-timestep (t-dt) 2M number concentrations, carried across steps
    # by the physics-data pass-through in ComposableIconPhysics so the 2M
    # update_tendencies_and_important_vars step has the tm1 state it needs.
    # Stored per kg of air (matching the qnc/qni tracer convention).
    qnc_prev: jnp.ndarray            # Previous-step cloud droplet number [1/kg] (nlev, ncols)
    qni_prev: jnp.ndarray            # Previous-step ice crystal number    [1/kg] (nlev, ncols)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            cloud_fraction=jnp.zeros((nlev,) + nodal_shape),
            qc=jnp.zeros((nlev,) + nodal_shape),
            qi=jnp.zeros((nlev,) + nodal_shape),
            precip_rain=jnp.zeros(nodal_shape),
            precip_snow=jnp.zeros(nodal_shape),
            droplet_number=jnp.zeros((nlev,) + nodal_shape),
            qnc_prev=jnp.zeros((nlev,) + nodal_shape),
            qni_prev=jnp.zeros((nlev,) + nodal_shape),
        )

    def copy(self, **kwargs):
        new_data = {
            'cloud_fraction': self.cloud_fraction,
            'qc': self.qc,
            'qi': self.qi,
            'precip_rain': self.precip_rain,
            'precip_snow': self.precip_snow,
            'droplet_number': self.droplet_number,
            'qnc_prev': self.qnc_prev,
            'qni_prev': self.qni_prev,
        }
        new_data.update(kwargs)
        return CloudData(**new_data)


@tree_math.struct
class VerticalDiffusionData:
    """Data for vertical diffusion and boundary layer"""
    
    # Exchange coefficients
    km: jnp.ndarray                  # Momentum exchange coeff [m²/s] (nlev+1, ncols)
    kh: jnp.ndarray                  # Heat exchange coeff [m²/s] (nlev+1, ncols)

    # Surface exchange coefficients (per surface type)
    surface_exchange_heat: jnp.ndarray      # Surface heat exchange [m²/s] (ncols, nsfc_type)
    surface_exchange_moisture: jnp.ndarray  # Surface moisture exchange [m²/s] (ncols, nsfc_type)
    surface_exchange_momentum: jnp.ndarray  # Surface momentum exchange [m²/s] (ncols, nsfc_type)

    # Turbulent kinetic energy
    tke: jnp.ndarray                 # TKE [m²/s²] (nlev, ncols)

    # Boundary layer diagnostics
    pbl_height: jnp.ndarray          # PBL height [m] (ncols,)
    surface_friction_velocity: jnp.ndarray  # u* [m/s] (ncols,)
    monin_obukhov_length: jnp.ndarray       # L [m] (ncols,)
    
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        nsfc_type = 3  # water, ice, land
        return cls(
            km=jnp.zeros((nlev+1,) + nodal_shape),
            kh=jnp.zeros((nlev+1,) + nodal_shape),
            surface_exchange_heat=jnp.zeros(nodal_shape + (nsfc_type,)),
            surface_exchange_moisture=jnp.zeros(nodal_shape + (nsfc_type,)),
            surface_exchange_momentum=jnp.zeros(nodal_shape + (nsfc_type,)),
            tke=jnp.zeros((nlev,) + nodal_shape),
            pbl_height=jnp.zeros(nodal_shape),
            surface_friction_velocity=jnp.zeros(nodal_shape),
            monin_obukhov_length=jnp.zeros(nodal_shape),
        )
    
    def copy(self, **kwargs):
        new_data = {
            'km': self.km,
            'kh': self.kh,
            'surface_exchange_heat': self.surface_exchange_heat,
            'surface_exchange_moisture': self.surface_exchange_moisture,
            'surface_exchange_momentum': self.surface_exchange_momentum,
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

    # Surface properties
    roughness_length: jnp.ndarray    # Surface roughness length [m] (ncols,)
    
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
            roughness_length=jnp.zeros(nodal_shape),
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
            'roughness_length': self.roughness_length,
            'evaporation': self.evaporation,
            'ch': self.ch,
            'cm': self.cm,
        }
        new_data.update(kwargs)
        return SurfaceData(**new_data)


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

    # Cloud condensation nuclei number concentration [cm^-3] (ncols,).
    # Derived from MACv2-SP plumes (anthropogenic + background AOD via the
    # AEROCOM-P1 Twomey relation) and consumed by the SPA-style activation
    # in the 2M microphysics path. See `jcm.physics.aerosol.spa`.
    Nccn: jnp.ndarray

    # Spectral scaling
    angstrom: jnp.ndarray            # Angstrom exponent [1] (ncols,)

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
            Nccn=jnp.zeros(nodal_shape),
            angstrom=jnp.ones(nodal_shape) * 1.5,  # Typical fine-mode aerosol
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
            'Nccn': self.Nccn,
            'angstrom': self.angstrom,
        }
        new_data.update(kwargs)
        return AerosolData(**new_data)


@tree_math.struct
class ChemistryData:
    """Data for chemistry calculations"""
    
    # Gas concentrations (volume mixing ratios)
    ozone_vmr: jnp.ndarray           # Ozone VMR [ppbv] (nlev, ncols)
    methane_vmr: jnp.ndarray         # Methane VMR [ppbv] (nlev, ncols)
    co2_vmr: jnp.ndarray            # CO2 VMR [ppmv] (nlev, ncols)
    
    # Production/loss rates
    ozone_production: jnp.ndarray    # Ozone production rate [ppbv/s] (nlev, ncols)
    ozone_loss: jnp.ndarray         # Ozone loss rate [ppbv/s] (nlev, ncols)
    methane_loss: jnp.ndarray       # Methane loss rate [ppbv/s] (nlev, ncols)
    
    # Surface emissions/deposition
    methane_surface_flux: jnp.ndarray  # Surface methane flux [ppbv m/s] (ncols,)
    ozone_dry_deposition: jnp.ndarray  # Ozone dry deposition velocity [m/s] (ncols,)
    
    @classmethod
    def zeros(cls, nodal_shape, nlev):
        # Chemistry fields should be in column format (nlev, ncols)
        # where ncols = nlat * nlon
        if len(nodal_shape) == 2:
            nlat, nlon = nodal_shape
            ncols = nlat * nlon
            column_shape = (nlev, ncols)
            surface_shape = (ncols,)
        else:
            # If already in column format
            column_shape = (nlev,) + nodal_shape
            surface_shape = nodal_shape
            
        return cls(
            ozone_vmr=jnp.zeros(column_shape),
            methane_vmr=jnp.zeros(column_shape),
            co2_vmr=jnp.zeros(column_shape),
            ozone_production=jnp.zeros(column_shape),
            ozone_loss=jnp.zeros(column_shape),
            methane_loss=jnp.zeros(column_shape),
            methane_surface_flux=jnp.zeros(surface_shape),
            ozone_dry_deposition=jnp.zeros(surface_shape),
        )
    
    def copy(self, **kwargs):
        new_data = {
            'ozone_vmr': self.ozone_vmr,
            'methane_vmr': self.methane_vmr,
            'co2_vmr': self.co2_vmr,
            'ozone_production': self.ozone_production,
            'ozone_loss': self.ozone_loss,
            'methane_loss': self.methane_loss,
            'methane_surface_flux': self.methane_surface_flux,
            'ozone_dry_deposition': self.ozone_dry_deposition,
        }
        new_data.update(kwargs)
        return ChemistryData(**new_data)


@tree_math.struct
class PhysicsData:
    """Main physics data container for ICON physics"""

    # `model_step` is the integer step counter used by the radiation toggle;
    # `dt_seconds` is the model timestep in seconds. The wall-clock side of
    # the date is consumed by physics through `forcing.solar` (populated by
    # `ForcingData.select(date)`).
    model_step: jnp.int32
    dt_seconds: float
    icon_coords: IconCoords
    diagnostics: DiagnosticData
    radiation: RadiationData
    convection: ConvectionData
    clouds: CloudData
    vertical_diffusion: VerticalDiffusionData
    surface: SurfaceData
    aerosol: AerosolData
    chemistry: ChemistryData

    @classmethod
    def zeros(cls, nodal_shape, nlev, icon_coords=None, model_step=None, dt_seconds=None):
        return cls(
            model_step=model_step if model_step is not None else jnp.int32(0),
            dt_seconds=dt_seconds if dt_seconds is not None else 1800.0,
            icon_coords=icon_coords,
            diagnostics=DiagnosticData.zeros(nodal_shape, nlev),
            radiation=RadiationData.zeros(nodal_shape, nlev),
            convection=ConvectionData.zeros(nodal_shape, nlev),
            clouds=CloudData.zeros(nodal_shape, nlev),
            vertical_diffusion=VerticalDiffusionData.zeros(nodal_shape, nlev),
            surface=SurfaceData.zeros(nodal_shape, nlev),
            aerosol=AerosolData.zeros(nodal_shape, nlev),
            chemistry=ChemistryData.zeros(nodal_shape, nlev),
        )

    def copy(self, **kwargs):
        new_data = {
            'model_step': self.model_step,
            'dt_seconds': self.dt_seconds,
            'icon_coords': self.icon_coords,
            'diagnostics': self.diagnostics,
            'radiation': self.radiation,
            'convection': self.convection,
            'clouds': self.clouds,
            'vertical_diffusion': self.vertical_diffusion,
            'surface': self.surface,
            'aerosol': self.aerosol,
            'chemistry': self.chemistry,
        }
        new_data.update(kwargs)
        return PhysicsData(**new_data)