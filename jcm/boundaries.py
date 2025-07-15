import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.scales import units
from dinosaur.coordinate_systems import HorizontalGridTypes
from jcm.physics.speedy.params import Parameters

@tree_math.struct
class BoundaryData:
    fmask: jnp.ndarray # fractional land-sea mask (ix,il)
    forog: jnp.ndarray # orographic factor for land surface drag
    orog: jnp.ndarray # orography in meters
    phi0: jnp.ndarray  # surface geopotential (ix, il)
    phis0: jnp.ndarray # spectrally-filtered surface geopotential
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    fmask: jnp.ndarray
    sice_am: jnp.ndarray
    fmask_l: jnp.ndarray # land mask - set by land_model_init()
    rhcapl: jnp.ndarray # 1/heat capacity (land)
    cdland: jnp.ndarray # 1/dissipation time (land)
    stlcl_ob: jnp.ndarray # climatology for land temperature - might not need this and stl_lm
    snowd_am: jnp.ndarray # used to be snowcl_ob in fortran - but one day of that was snowd_am
    soilw_am: jnp.ndarray # used to be soilwcl_ob in fortran - but one day of that was soilw_am
    lfluxland: jnp.bool # flag to compute land skin temperature and latent fluxes
    land_coupling_flag: jnp.bool # 0 or 1
    tsea: jnp.ndarray # SST, should come from sea_model.py or some default value

    fmask_s: jnp.ndarray # sea mask - set by sea_model_init() once we have a model (instead of fixed ssts)
    
    # Additional boundary conditions for ICON physics
    surface_temperature: jnp.ndarray  # Surface temperature [K] (nlat, nlon)
    roughness_length: jnp.ndarray     # Surface roughness length [m] (nlat, nlon)
    solar_zenith_angle: jnp.ndarray   # Solar zenith angle [radians] (nlat, nlon)
    solar_irradiance: jnp.ndarray     # Top-of-atmosphere solar irradiance [W/m²] (nlat, nlon)
    
    # Greenhouse gas concentrations
    co2_concentration: jnp.ndarray    # CO2 concentration [ppmv] (nlat, nlon)
    ch4_concentration: jnp.ndarray    # CH4 concentration [ppbv] (nlat, nlon)
    n2o_concentration: jnp.ndarray    # N2O concentration [ppbv] (nlat, nlon)
    
    # Surface optical properties
    surface_albedo_vis: jnp.ndarray   # Surface albedo visible [1] (nlat, nlon)
    surface_albedo_nir: jnp.ndarray   # Surface albedo near-infrared [1] (nlat, nlon)
    surface_emissivity: jnp.ndarray   # Surface emissivity [1] (nlat, nlon)
    
    # Sea ice properties
    sea_ice_fraction: jnp.ndarray     # Sea ice fraction [0-1] (nlat, nlon)
    sea_ice_thickness: jnp.ndarray    # Sea ice thickness [m] (nlat, nlon)


    @classmethod
    def zeros(cls,nodal_shape,fmask=None,forog=None,orog=None,phi0=None,phis0=None,
              alb0=None,sice_am=None,fmask_l=None,rhcapl=None,cdland=None,
              stlcl_ob=None,snowd_am=None,soilw_am=None,tsea=None,
              fmask_s=None,lfluxland=None, land_coupling_flag=None,
              surface_temperature=None, roughness_length=None, solar_zenith_angle=None,
              solar_irradiance=None, co2_concentration=None, ch4_concentration=None,
              n2o_concentration=None, surface_albedo_vis=None, surface_albedo_nir=None,
              surface_emissivity=None, sea_ice_fraction=None, sea_ice_thickness=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.zeros((nodal_shape)),
            forog=forog if forog is not None else jnp.zeros((nodal_shape)),
            orog=orog if orog is not None else jnp.zeros((nodal_shape)),
            phi0=phi0 if phi0 is not None else jnp.zeros((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.zeros((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.zeros((nodal_shape)+(365,)),
            fmask_l=fmask_l if fmask_l is not None else jnp.zeros((nodal_shape)),
            rhcapl=rhcapl if rhcapl is not None else jnp.zeros((nodal_shape)),
            cdland=cdland if cdland is not None else jnp.zeros((nodal_shape)),
            stlcl_ob=stlcl_ob if stlcl_ob is not None else jnp.zeros((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.zeros((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)+(365,)),
            land_coupling_flag=land_coupling_flag if land_coupling_flag is not None else False,
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.zeros((nodal_shape)),
            fmask_s=fmask_s if fmask_s is not None else jnp.zeros((nodal_shape)),
            # New ICON physics boundary conditions
            surface_temperature=surface_temperature if surface_temperature is not None else jnp.zeros(nodal_shape) + 288.0,
            roughness_length=roughness_length if roughness_length is not None else jnp.zeros(nodal_shape) + 0.001,
            solar_zenith_angle=solar_zenith_angle if solar_zenith_angle is not None else jnp.zeros(nodal_shape) + 0.5,
            solar_irradiance=solar_irradiance if solar_irradiance is not None else jnp.zeros(nodal_shape) + 1361.0,
            co2_concentration=co2_concentration if co2_concentration is not None else jnp.zeros(nodal_shape) + 420.0,
            ch4_concentration=ch4_concentration if ch4_concentration is not None else jnp.zeros(nodal_shape) + 1900.0,
            n2o_concentration=n2o_concentration if n2o_concentration is not None else jnp.zeros(nodal_shape) + 335.0,
            surface_albedo_vis=surface_albedo_vis if surface_albedo_vis is not None else jnp.zeros(nodal_shape) + 0.15,
            surface_albedo_nir=surface_albedo_nir if surface_albedo_nir is not None else jnp.zeros(nodal_shape) + 0.25,
            surface_emissivity=surface_emissivity if surface_emissivity is not None else jnp.zeros(nodal_shape) + 0.95,
            sea_ice_fraction=sea_ice_fraction if sea_ice_fraction is not None else jnp.zeros(nodal_shape),
            sea_ice_thickness=sea_ice_thickness if sea_ice_thickness is not None else jnp.zeros(nodal_shape),
        )

    @classmethod
    def ones(cls,nodal_shape,fmask=None,forog=None,orog=None,phi0=None,phis0=None,
             alb0=None,sice_am=None,fmask_l=None,rhcapl=None,cdland=None,
             stlcl_ob=None,snowd_am=None,soilw_am=None,tsea=None,
             fmask_s=None,lfluxland=None, land_coupling_flag=None,
             surface_temperature=None, roughness_length=None, solar_zenith_angle=None,
             solar_irradiance=None, co2_concentration=None, ch4_concentration=None,
             n2o_concentration=None, surface_albedo_vis=None, surface_albedo_nir=None,
             surface_emissivity=None, sea_ice_fraction=None, sea_ice_thickness=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.ones((nodal_shape)),
            forog=forog if forog is not None else jnp.ones((nodal_shape)),
            orog=orog if orog is not None else jnp.ones((nodal_shape)),
            phi0=phi0 if phi0 is not None else jnp.ones((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.ones((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)+(365,)),
            fmask_l=fmask_l if fmask_l is not None else jnp.ones((nodal_shape)),
            rhcapl=rhcapl if rhcapl is not None else jnp.ones((nodal_shape)),
            cdland=cdland if cdland is not None else jnp.ones((nodal_shape)),
            stlcl_ob=stlcl_ob if stlcl_ob is not None else jnp.ones((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.ones((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)+(365,)),
            land_coupling_flag=land_coupling_flag if land_coupling_flag is not None else False,
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.ones((nodal_shape)),
            fmask_s=fmask_s if fmask_s is not None else jnp.ones((nodal_shape)),
            # New ICON physics boundary conditions
            surface_temperature=surface_temperature if surface_temperature is not None else jnp.ones(nodal_shape) * 288.0,
            roughness_length=roughness_length if roughness_length is not None else jnp.ones(nodal_shape) * 0.001,
            solar_zenith_angle=solar_zenith_angle if solar_zenith_angle is not None else jnp.ones(nodal_shape) * 0.5,
            solar_irradiance=solar_irradiance if solar_irradiance is not None else jnp.ones(nodal_shape) * 1361.0,
            co2_concentration=co2_concentration if co2_concentration is not None else jnp.ones(nodal_shape) * 420.0,
            ch4_concentration=ch4_concentration if ch4_concentration is not None else jnp.ones(nodal_shape) * 1900.0,
            n2o_concentration=n2o_concentration if n2o_concentration is not None else jnp.ones(nodal_shape) * 335.0,
            surface_albedo_vis=surface_albedo_vis if surface_albedo_vis is not None else jnp.ones(nodal_shape) * 0.15,
            surface_albedo_nir=surface_albedo_nir if surface_albedo_nir is not None else jnp.ones(nodal_shape) * 0.25,
            surface_emissivity=surface_emissivity if surface_emissivity is not None else jnp.ones(nodal_shape) * 0.95,
            sea_ice_fraction=sea_ice_fraction if sea_ice_fraction is not None else jnp.zeros(nodal_shape),
            sea_ice_thickness=sea_ice_thickness if sea_ice_thickness is not None else jnp.zeros(nodal_shape),
        )

    def copy(self,fmask=None,phi0=None,forog=None,orog=None,phis0=None,alb0=None,
             sice_am=None,fmask_l=None,rhcapl=None,cdland=None,stlcl_ob=None,
             snowd_am=None,soilw_am=None,tsea=None,fmask_s=None,lfluxland=None,
             land_coupling_flag=None, surface_temperature=None, roughness_length=None, 
             solar_zenith_angle=None, solar_irradiance=None, co2_concentration=None, 
             ch4_concentration=None, n2o_concentration=None, surface_albedo_vis=None, 
             surface_albedo_nir=None, surface_emissivity=None, sea_ice_fraction=None, 
             sea_ice_thickness=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else self.fmask,
            forog=forog if forog is not None else self.forog,
            orog=orog if orog is not None else self.orog,
            phi0=phi0 if phi0 is not None else self.phi0,
            phis0=phis0 if phis0 is not None else self.phis0,
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            fmask_l=fmask_l if fmask_l is not None else self.fmask_l,
            rhcapl=rhcapl if rhcapl is not None else self.rhcapl,
            cdland=cdland if cdland is not None else self.cdland,
            stlcl_ob=stlcl_ob if stlcl_ob is not None else self.stlcl_ob,
            snowd_am=snowd_am if snowd_am is not None else self.snowd_am,
            land_coupling_flag=land_coupling_flag if land_coupling_flag is not None else self.land_coupling_flag,
            lfluxland=lfluxland if lfluxland is not None else self.lfluxland,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            tsea=tsea if tsea is not None else self.tsea,
            fmask_s=fmask_s if fmask_s is not None else self.fmask_s,
            # New ICON physics boundary conditions
            surface_temperature=surface_temperature if surface_temperature is not None else self.surface_temperature,
            roughness_length=roughness_length if roughness_length is not None else self.roughness_length,
            solar_zenith_angle=solar_zenith_angle if solar_zenith_angle is not None else self.solar_zenith_angle,
            solar_irradiance=solar_irradiance if solar_irradiance is not None else self.solar_irradiance,
            co2_concentration=co2_concentration if co2_concentration is not None else self.co2_concentration,
            ch4_concentration=ch4_concentration if ch4_concentration is not None else self.ch4_concentration,
            n2o_concentration=n2o_concentration if n2o_concentration is not None else self.n2o_concentration,
            surface_albedo_vis=surface_albedo_vis if surface_albedo_vis is not None else self.surface_albedo_vis,
            surface_albedo_nir=surface_albedo_nir if surface_albedo_nir is not None else self.surface_albedo_nir,
            surface_emissivity=surface_emissivity if surface_emissivity is not None else self.surface_emissivity,
            sea_ice_fraction=sea_ice_fraction if sea_ice_fraction is not None else self.sea_ice_fraction,
            sea_ice_thickness=sea_ice_thickness if sea_ice_thickness is not None else self.sea_ice_thickness
        )

    def isnan(self):
        self.lfluxland = 0
        self.land_coupling_flag = 0
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))


def _fixed_ssts(grid: HorizontalGridTypes) -> jnp.ndarray:
    """
    Returns an array of SSTs with simple cos^2 profile from 300K at the equator to 273K at 60 degrees latitude.
    Obtained from Neale, R.B. and Hoskins, B.J. (2000),
    "A standard test for AGCMs including their physical parametrizations: I: the proposal."
    Atmosph. Sci. Lett., 1: 101-107. https://doi.org/10.1006/asle.2000.0022
    """
    radang = grid.latitudes
    sst_profile = jnp.where(jnp.abs(radang) < jnp.pi/3, 27*jnp.cos(3*radang/2)**2, 0) + 273.15
    return jnp.tile(sst_profile[jnp.newaxis], (grid.nodal_shape[0], 1))

def default_boundaries(
    grid: HorizontalGridTypes,
    orography,
    parameters: Parameters=None,
    truncation_number=None
) -> BoundaryData:
    """
    Initialize the boundary conditions
    """
    from jcm.physics.speedy.surface_flux import set_orog_land_sfc_drag
    from jcm.utils import spectral_truncation
    from jcm.physics.speedy.physical_constants import grav

    parameters = parameters or Parameters.default()

    phi0 = grav * orography
    phis0 = spectral_truncation(grid, phi0, truncation_number=truncation_number)
    forog = set_orog_land_sfc_drag(phis0, parameters)

    # land-sea mask
    fmask = jnp.zeros_like(orography)
    alb0 = jnp.zeros_like(orography)
    tsea = _fixed_ssts(grid)
    
    # No land_model_init, but should be fine because fmask = 0
    return BoundaryData.zeros(
        nodal_shape=orography.shape,
        orog=orography, fmask=fmask, forog=forog, phi0=phi0, phis0=phis0, tsea=tsea, alb0=alb0)


#this function calls land_model_init and eventually will call init for sea and ice models
def initialize_boundaries(
    filename: str,
    grid: HorizontalGridTypes,
    parameters: Parameters=None,
    truncation_number=None
) -> BoundaryData:
    """
    Initialize the boundary conditions
    """
    from jcm.physics.speedy.physical_constants import grav
    from jcm.utils import spectral_truncation
    from jcm.physics.speedy.land_model import land_model_init
    from jcm.physics.speedy.surface_flux import set_orog_land_sfc_drag
    import xarray as xr

    parameters = parameters or Parameters.default()
    
    ds = xr.open_dataset(filename)

    orog = jnp.asarray(ds["orog"])
    # Read surface geopotential (i.e. orography)
    phi0 = grav * orog
    # Also store spectrally truncated surface geopotential for the land drag term
    phis0 = spectral_truncation(grid, phi0, truncation_number=truncation_number)
    forog = set_orog_land_sfc_drag(phis0, parameters)

    # Read land-sea mask
    fmask = jnp.asarray(ds["lsm"])
    # Annual-mean surface albedo
    alb0 = jnp.asarray(ds["alb"])
    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"

    tsea = _fixed_ssts(grid)
    boundaries = BoundaryData.zeros(
        nodal_shape=fmask.shape,
        fmask=fmask, forog=forog, orog=orog, phi0=phi0, phis0=phis0, tsea=tsea, alb0=alb0)
    
    boundaries = land_model_init(filename, parameters, boundaries)

    # call sea model init
    # call ice model init

    return boundaries

def update_boundaries_with_timestep(
        boundaries: BoundaryData,
        parameters: Parameters=None,
        time_step=30*units.minute
) -> BoundaryData:
    """
    Update the boundary conditions with the new time step
    """
    parameters = parameters or Parameters.default()
    # Update the land heat capacity and dissipation time
    if boundaries.land_coupling_flag:
        rhcapl = jnp.where(boundaries.alb0 < 0.4, 1./parameters.land_model.hcapl, 1./parameters.land_model.hcapli) * time_step.to(units.second).m
        return boundaries.copy(rhcapl=rhcapl)
    else:
        return boundaries


def compute_time_varying_boundaries(
    boundaries: BoundaryData,
    geometry,
    day_of_year: float = 180.0,
    time_of_day: float = 12.0,
    year: float = 2020.0
) -> BoundaryData:
    """
    Compute time-varying boundary conditions for ICON physics
    
    This function updates the boundary conditions with time-varying values
    for solar forcing, greenhouse gases, and surface properties.
    
    Args:
        boundaries: Current boundary conditions
        geometry: Geometry object containing latitude information
        day_of_year: Day of year (1-365)
        time_of_day: Time of day (hours, 0-24)
        year: Year (for solar variability)
        
    Returns:
        Updated boundary conditions
    """
    # Get latitudes from geometry (radang is in radians)
    latitudes = geometry.radang  # Shape: (nlat,)
    
    # The boundary surface_temperature shape follows the same convention as geometry
    # geometry nodal_shape is (nlev, nlon, nlat) so boundaries should be (nlon, nlat)
    nlon, nlat = boundaries.surface_temperature.shape
    longitudes = jnp.linspace(-jnp.pi, jnp.pi, nlon, endpoint=False)  # Shape: (nlon,)
    
    # Create 2D grids - use 'xy' indexing to match boundary shape (nlon, nlat)
    lon_2d, lat_2d = jnp.meshgrid(longitudes, latitudes, indexing='xy')
    
    # Compute solar zenith angle
    solar_zenith_angle = _compute_solar_zenith_angle(
        lat_2d, lon_2d, day_of_year, time_of_day
    )
    
    # Compute solar irradiance
    solar_irradiance = _compute_solar_irradiance(
        solar_zenith_angle, day_of_year, year
    )
    
    # Compute surface properties based on existing masks
    surface_albedo_vis, surface_albedo_nir, surface_emissivity = _compute_surface_properties(
        boundaries.fmask,  # Land fraction
        boundaries.sice_am[..., 0] if boundaries.sice_am.ndim == 3 else boundaries.sice_am,  # Sea ice
    )
    
    # Surface temperature (use existing SST for ocean, land temperature for land)
    surface_temperature = jnp.where(
        boundaries.fmask > 0.5,  # Land
        boundaries.stlcl_ob[..., 0] if boundaries.stlcl_ob.ndim == 3 else boundaries.stlcl_ob,  # Land temp
        boundaries.tsea  # SST
    )
    
    # Roughness length (higher over land)
    roughness_length = jnp.where(
        boundaries.fmask > 0.5,  # Land
        0.01,  # 1 cm over land
        0.0001  # 0.1 mm over ocean
    )
    
    # Greenhouse gas concentrations (uniform for now)
    co2_concentration = jnp.full_like(lat_2d, 420.0)  # ppmv
    ch4_concentration = jnp.full_like(lat_2d, 1900.0)  # ppbv
    n2o_concentration = jnp.full_like(lat_2d, 335.0)  # ppbv
    
    # Sea ice fraction (from existing data)
    sea_ice_fraction = boundaries.sice_am[..., 0] if boundaries.sice_am.ndim == 3 else boundaries.sice_am
    sea_ice_thickness = jnp.where(sea_ice_fraction > 0.1, 1.0, 0.0)  # 1m where ice exists
    
    return boundaries.copy(
        surface_temperature=surface_temperature,
        roughness_length=roughness_length,
        solar_zenith_angle=solar_zenith_angle,
        solar_irradiance=solar_irradiance,
        co2_concentration=co2_concentration,
        ch4_concentration=ch4_concentration,
        n2o_concentration=n2o_concentration,
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=surface_emissivity,
        sea_ice_fraction=sea_ice_fraction,
        sea_ice_thickness=sea_ice_thickness
    )


def _compute_solar_zenith_angle(
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    day_of_year: float,
    time_of_day: float
) -> jnp.ndarray:
    """Compute solar zenith angle"""
    # Solar declination (simplified)
    declination = 23.45 * jnp.pi / 180.0 * jnp.sin(2.0 * jnp.pi * (day_of_year - 81.0) / 365.0)
    
    # Hour angle
    hour_angle = (time_of_day - 12.0) * jnp.pi / 12.0
    
    # Solar zenith angle
    cos_zenith = (jnp.sin(latitude) * jnp.sin(declination) + 
                  jnp.cos(latitude) * jnp.cos(declination) * jnp.cos(hour_angle))
    
    zenith_angle = jnp.arccos(jnp.clip(cos_zenith, -1.0, 1.0))
    
    return zenith_angle


def _compute_solar_irradiance(
    solar_zenith_angle: jnp.ndarray,
    day_of_year: float,
    year: float
) -> jnp.ndarray:
    """Compute top-of-atmosphere solar irradiance"""
    # Constants
    solar_constant = 1361.0  # W/m²
    solar_variability = 0.001  # 0.1% variation
    solar_cycle_period = 11.0  # 11 years
    
    # Earth-Sun distance variation
    earth_sun_distance = 1.0 - 0.0167 * jnp.cos(2.0 * jnp.pi * (day_of_year - 3.0) / 365.0)
    
    # Solar variability (simplified 11-year cycle)
    solar_variation = 1.0 + solar_variability * jnp.sin(2.0 * jnp.pi * year / solar_cycle_period)
    
    # Solar irradiance
    solar_irradiance = (solar_constant * solar_variation / earth_sun_distance**2 * 
                       jnp.maximum(jnp.cos(solar_zenith_angle), 0.0))
    
    return solar_irradiance


def _compute_surface_properties(
    land_fraction: jnp.ndarray,
    sea_ice_fraction: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute surface optical properties"""
    # Default values
    land_albedo_vis = 0.15
    land_albedo_nir = 0.25
    land_emissivity = 0.95
    
    ocean_albedo_vis = 0.05
    ocean_albedo_nir = 0.05
    ocean_emissivity = 0.98
    
    seaice_albedo_vis = 0.80
    seaice_albedo_nir = 0.70
    seaice_emissivity = 0.95
    
    # Ocean fraction
    ocean_fraction = 1.0 - land_fraction - sea_ice_fraction
    ocean_fraction = jnp.maximum(ocean_fraction, 0.0)
    
    # Weighted average of surface properties
    albedo_vis = (land_fraction * land_albedo_vis +
                  ocean_fraction * ocean_albedo_vis +
                  sea_ice_fraction * seaice_albedo_vis)
    
    albedo_nir = (land_fraction * land_albedo_nir +
                  ocean_fraction * ocean_albedo_nir +
                  sea_ice_fraction * seaice_albedo_nir)
    
    emissivity = (land_fraction * land_emissivity +
                  ocean_fraction * ocean_emissivity +
                  sea_ice_fraction * seaice_emissivity)
    
    return albedo_vis, albedo_nir, emissivity