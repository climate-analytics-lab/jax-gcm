import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.scales import units
from dinosaur.coordinate_systems import HorizontalGridTypes
from jcm.physics.speedy.params import Parameters

@tree_math.struct
class BoundaryData:
    """This class defines the lower boundary of the atmosphere, such as orography, 
    land-sea masks, and sea surface temperatures. 

    Attributes:
        fmask (jnp.ndarray): Fractional land-sea mask. Values range from 0 (sea)
            to 1 (land). Shape: (ix, il).
        forog (jnp.ndarray): Orographic factor for land surface drag.
        orog (jnp.ndarray): Orography (surface height) in meters.
        phi0 (jnp.ndarray): Surface geopotential (g * orog). Shape: (ix, il)
        phis0 (jnp.ndarray): Spectrally-filtered surface geopotential.
        alb0 (jnp.ndarray): Bare-land annual mean albedo. Shape: (ix, il).
        sice_am (jnp.ndarray): Annual mean sea ice concentration climatology.
        fmask_l (jnp.ndarray): Binary land mask (1 for land, 0 for sea) - set by land_model_init()
        rhcapl (jnp.ndarray): Inverse of land heat capacity, scaled by the time step.
        cdland (jnp.ndarray): Inverse of dissipation time for land.
        stlcl_ob (jnp.ndarray): Climatology for land surface temperature - might not need this and stl_lm
        snowd_am (jnp.ndarray): Annual mean snow depth climatology. (One day of snowcl_ob in fortran)
        soilw_am (jnp.ndarray): Annual mean soil water climatology. (One day of soilwcl_ob in fortran)
        lfluxland (jnp.bool): Flag to enable computation of land skin temperature
            and latent heat fluxes.
        land_coupling_flag (jnp.bool): Flag to enable/disable coupling with the
            land model.
        tsea (jnp.ndarray): Sea Surface Temperature (SST).
        fmask_s (jnp.ndarray): Binary sea mask (1 for sea, 0 for land) - set by sea_model_init() once we have a model (instead of fixed ssts)
    """
    fmask: jnp.ndarray 
    forog: jnp.ndarray 
    orog: jnp.ndarray 
    phi0: jnp.ndarray  
    phis0: jnp.ndarray 
    alb0: jnp.ndarray 

    fmask: jnp.ndarray
    sice_am: jnp.ndarray
    fmask_l: jnp.ndarray 
    rhcapl: jnp.ndarray 
    cdland: jnp.ndarray 
    stlcl_ob: jnp.ndarray 
    snowd_am: jnp.ndarray 
    soilw_am: jnp.ndarray 
    lfluxland: jnp.bool 
    land_coupling_flag: jnp.bool 
    tsea: jnp.ndarray 

    fmask_s: jnp.ndarray 


    @classmethod
    def zeros(cls,nodal_shape,fmask=None,forog=None,orog=None,phi0=None,phis0=None,
              alb0=None,sice_am=None,fmask_l=None,rhcapl=None,cdland=None,
              stlcl_ob=None,snowd_am=None,soilw_am=None,tsea=None,
              fmask_s=None,lfluxland=None, land_coupling_flag=None):
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
        )

    @classmethod
    def ones(cls,nodal_shape,fmask=None,forog=None,orog=None,phi0=None,phis0=None,
             alb0=None,sice_am=None,fmask_l=None,rhcapl=None,cdland=None,
             stlcl_ob=None,snowd_am=None,soilw_am=None,tsea=None,
             fmask_s=None,lfluxland=None, land_coupling_flag=None):
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
        )

    def copy(self,fmask=None,phi0=None,forog=None,orog=None,phis0=None,alb0=None,
             sice_am=None,fmask_l=None,rhcapl=None,cdland=None,stlcl_ob=None,
             snowd_am=None,soilw_am=None,tsea=None,fmask_s=None,lfluxland=None,
             land_coupling_flag=None):
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
            fmask_s=fmask_s if fmask_s is not None else self.fmask_s
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
   
    Args:
        grid: A `HorizontalGrid` object containing grid information, including
            latitudes.

    Returns:
        A 2D JAX array of SSTs with shape (grid.nodal_shape).

    References:
        Neale, R.B. and Hoskins, B.J. (2000), "A standard test for AGCMs
        including their physical parametrizations: I: the proposal."
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
    Initializes default boundary conditions for an aqua-planet scenario.

    Args:
        grid: The model's horizontal grid.
        orography: A 2D array of surface orography in meters.
        parameters: An optional `Parameters` object. If not provided, defaults
            are used.
        truncation_number: The spectral truncation number for filtering.

    Returns:
        A `BoundaryData` object configured for an aqua-planet simulation.
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
    Initializes realistic boundary conditions from a data file.

    Args:
        filename: Path to the input NetCDF file containing boundary data.
        grid: The model's horizontal grid.
        parameters: An optional `Parameters` object.
        truncation_number: The spectral truncation number for filtering.

    Returns:
        A `BoundaryData` object with realistic, file-based boundary conditions.
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

    Args:
        boundaries: The current `BoundaryData` object.
        parameters: An optional `Parameters` object.
        time_step: The model time step as a `dinosaur.scales.units.Quantity`.

    Returns:
        A new `BoundaryData` object with updated time-step dependent fields.
    """
    parameters = parameters or Parameters.default()
    # Update the land heat capacity and dissipation time
    if boundaries.land_coupling_flag:
        rhcapl = jnp.where(boundaries.alb0 < 0.4, 1./parameters.land_model.hcapl, 1./parameters.land_model.hcapli) * time_step.to(units.second).m
        return boundaries.copy(rhcapl=rhcapl)
    else:
        return boundaries