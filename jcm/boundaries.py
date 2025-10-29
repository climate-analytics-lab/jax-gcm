import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes

@tree_math.struct
class BoundaryData:
    fmask: jnp.ndarray # fractional land-sea mask (ix,il)
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    sice_am: jnp.ndarray # FIXME: need to set this
    snowd_am: jnp.ndarray # used to be snowcl_ob in fortran - but one day of that was snowd_am
    soilw_am: jnp.ndarray # used to be soilwcl_ob in fortran - but one day of that was soilw_am
    lfluxland: jnp.bool # flag to compute land skin temperature and latent fluxes
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
    def zeros(cls,nodal_shape,fmask=None,
              alb0=None,sice_am=None,snowd_am=None,
              soilw_am=None,tsea=None,lfluxland=None, land_coupling_flag=None,
              surface_temperature=None, roughness_length=None, solar_zenith_angle=None,
              solar_irradiance=None, co2_concentration=None, ch4_concentration=None,
              n2o_concentration=None, surface_albedo_vis=None, surface_albedo_nir=None,
              surface_emissivity=None, sea_ice_fraction=None, sea_ice_thickness=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.zeros((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.zeros((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.zeros((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)+(365,)),
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.zeros((nodal_shape)+(365,)),
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
    def ones(cls,nodal_shape,fmask=None,
             alb0=None,sice_am=None,snowd_am=None,
             soilw_am=None,tsea=None,lfluxland=None, land_coupling_flag=None,
             surface_temperature=None, roughness_length=None, solar_zenith_angle=None,
             solar_irradiance=None, co2_concentration=None, ch4_concentration=None,
             n2o_concentration=None, surface_albedo_vis=None, surface_albedo_nir=None,
             surface_emissivity=None, sea_ice_fraction=None, sea_ice_thickness=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.ones((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.ones((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)+(365,)),
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.ones((nodal_shape)+(365,)),
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

    def copy(self,fmask=None,alb0=None,
             sice_am=None,snowd_am=None,soilw_am=None,
             tsea=None,lfluxland=None,
             land_coupling_flag=None, surface_temperature=None, roughness_length=None, 
             solar_zenith_angle=None, solar_irradiance=None, co2_concentration=None, 
             ch4_concentration=None, n2o_concentration=None, surface_albedo_vis=None, 
             surface_albedo_nir=None, surface_emissivity=None, sea_ice_fraction=None, 
             sea_ice_thickness=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else self.fmask,
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            snowd_am=snowd_am if snowd_am is not None else self.snowd_am,
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
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(jnp.any, self))


def _fixed_ssts(grid: HorizontalGridTypes) -> jnp.ndarray:
    """
    Returns an array of SSTs with simple cos^2 profile from 300K at the equator to 273K at 60 degrees latitude.
    Obtained from Neale, R.B. and Hoskins, B.J. (2000),
    "A standard test for AGCMs including their physical parametrizations: I: the proposal."
    Atmosph. Sci. Lett., 1: 101-107. https://doi.org/10.1006/asle.2000.0022
    """
    radang = grid.latitudes
    sst_profile = jnp.where(jnp.abs(radang) < jnp.pi/3, 27*jnp.cos(3*radang/2)**2, 0) + 273.15
    return jnp.tile(sst_profile, (grid.nodal_shape[0], 1))

def default_boundaries(
    grid: HorizontalGridTypes,
) -> BoundaryData:
    """
    Initialize the boundary conditions
    """
    # land-sea mask
    fmask = jnp.zeros(grid.nodal_shape)
    alb0 = jnp.zeros(grid.nodal_shape)
    tsea = jnp.tile(_fixed_ssts(grid)[:, :, jnp.newaxis], (1, 1, 365))
        
    return BoundaryData.zeros(
        nodal_shape=grid.nodal_shape,
        fmask=fmask, tsea=tsea, alb0=alb0
    )

def boundaries_from_file(
    filename: str,
    fmask_threshold=0.1,
) -> BoundaryData:
    """
    Initialize the boundary conditions
    """
    import xarray as xr

    # Read boundaries from file
    ds = xr.open_dataset(filename)

    # land-sea mask
    fmask = jnp.asarray(ds["lsm"])
    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"
    # Set values close to 0 or 1 to exactly 0 or 1
    fmask = jnp.where(fmask <= fmask_threshold, 0.0, jnp.where(fmask >= 1.0 - fmask_threshold, 1.0, fmask))

    # annual-mean surface albedo
    alb0 = jnp.asarray(ds["alb"])

    # sea ice concentration
    sice_am = jnp.asarray(ds["icec"])

    # snow depth
    snowd_am = jnp.asarray(ds["snowd"])
    snowd_valid = (0.0 <= snowd_am) & (snowd_am <= 20000.0)
    # assert jnp.all(snowd_valid | (fmask[:,:,jnp.newaxis] == 0.0)) # FIXME: need to change the boundaries.nc file so this passes
    snowd_am = jnp.where(snowd_valid, snowd_am, 0.0)

    # soil moisture
    soilw_am = jnp.asarray(ds["soilw_am"])
    soilw_valid = (0.0 <= soilw_am) & (soilw_am <= 1.0)
    assert jnp.all(soilw_valid | (fmask[:,:,jnp.newaxis] == 0.0))

    # Prescribe SSTs
    tsea = jnp.asarray(ds["sst"])

    return BoundaryData.zeros(
        nodal_shape=fmask.shape, fmask=fmask,
        alb0=alb0, sice_am=sice_am, snowd_am=snowd_am,
        soilw_am=soilw_am, tsea=tsea
    )

    tsea = _fixed_ssts(grid)
    boundaries = BoundaryData.zeros(
        nodal_shape=fmask.shape,
        fmask=fmask, forog=forog, orog=orog, phi0=phi0, phis0=phis0, tsea=tsea, alb0=alb0)
    
    boundaries = land_model_init(filename, parameters, boundaries)

    # call sea model init
    # call ice model init

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