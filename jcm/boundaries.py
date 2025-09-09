import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes
from jcm.physics.speedy.params import Parameters

@tree_math.struct
class BoundaryData:
    fmask: jnp.ndarray # fractional land-sea mask (ix,il)
    orog: jnp.ndarray # orography in meters
    phis0: jnp.ndarray # spectrally-filtered surface geopotential
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    sice_am: jnp.ndarray
    fmask_l: jnp.ndarray # land mask - set by land_model_init()
    stlcl_ob: jnp.ndarray # climatology for land temperature - might not need this and stl_lm
    snowd_am: jnp.ndarray # used to be snowcl_ob in fortran - but one day of that was snowd_am
    soilw_am: jnp.ndarray # used to be soilwcl_ob in fortran - but one day of that was soilw_am
    lfluxland: jnp.bool # flag to compute land skin temperature and latent fluxes
    tsea: jnp.ndarray # SST, should come from sea_model.py or some default value

    fmask_s: jnp.ndarray # sea mask - set by sea_model_init() once we have a model (instead of fixed ssts)


    @classmethod
    def zeros(cls,nodal_shape,fmask=None,orog=None,phis0=None,
              alb0=None,sice_am=None,fmask_l=None,stlcl_ob=None,snowd_am=None,
              soilw_am=None,tsea=None,fmask_s=None,lfluxland=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.zeros((nodal_shape)),
            orog=orog if orog is not None else jnp.zeros((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.zeros((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.zeros((nodal_shape)+(365,)),
            fmask_l=fmask_l if fmask_l is not None else jnp.zeros((nodal_shape)),
            stlcl_ob=stlcl_ob if stlcl_ob is not None else jnp.zeros((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.zeros((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)+(365,)),
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.zeros((nodal_shape)),
            fmask_s=fmask_s if fmask_s is not None else jnp.zeros((nodal_shape)),
        )

    @classmethod
    def ones(cls,nodal_shape,fmask=None,orog=None,phis0=None,
             alb0=None,sice_am=None,fmask_l=None,stlcl_ob=None,snowd_am=None,
             soilw_am=None,tsea=None,fmask_s=None,lfluxland=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.ones((nodal_shape)),
            orog=orog if orog is not None else jnp.ones((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.ones((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)+(365,)),
            fmask_l=fmask_l if fmask_l is not None else jnp.ones((nodal_shape)),
            stlcl_ob=stlcl_ob if stlcl_ob is not None else jnp.ones((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.ones((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)+(365,)),
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.ones((nodal_shape)),
            fmask_s=fmask_s if fmask_s is not None else jnp.ones((nodal_shape)),
        )

    def copy(self,fmask=None,orog=None,phis0=None,alb0=None,
             sice_am=None,fmask_l=None,stlcl_ob=None,snowd_am=None,soilw_am=None,
             tsea=None,fmask_s=None,lfluxland=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else self.fmask,
            orog=orog if orog is not None else self.orog,
            phis0=phis0 if phis0 is not None else self.phis0,
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            fmask_l=fmask_l if fmask_l is not None else self.fmask_l,
            stlcl_ob=stlcl_ob if stlcl_ob is not None else self.stlcl_ob,
            snowd_am=snowd_am if snowd_am is not None else self.snowd_am,
            lfluxland=lfluxland if lfluxland is not None else self.lfluxland,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            tsea=tsea if tsea is not None else self.tsea,
            fmask_s=fmask_s if fmask_s is not None else self.fmask_s
        )

    def isnan(self):
        self.lfluxland = 0
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
    truncation_number=None
) -> BoundaryData:
    """
    Initialize the boundary conditions
    """
    from jcm.utils import spectral_truncation
    from jcm.physics.speedy.physical_constants import grav

    phi0 = grav * orography
    phis0 = spectral_truncation(grid, phi0, truncation_number=truncation_number)

    # land-sea mask
    fmask = jnp.zeros_like(orography)
    alb0 = jnp.zeros_like(orography)
    tsea = _fixed_ssts(grid)
    
    # Default to all sea when no land-sea mask provided
    fmask_l = jnp.zeros_like(orography)  # No land
    fmask_s = jnp.ones_like(orography)   # All sea
    
    return BoundaryData.zeros(
        nodal_shape=orography.shape,
        orog=orography, fmask=fmask, phis0=phis0, tsea=tsea, alb0=alb0,
        fmask_l=fmask_l, fmask_s=fmask_s)


#this function calls land_model_init and eventually will call init for sea and ice models
def initialize_boundaries(
    filename: str,
    grid: HorizontalGridTypes,
    parameters: Parameters=Parameters.default(),
    truncation_number=None
) -> BoundaryData:
    """
    Initialize the boundary conditions
    """
    from jcm.physics.speedy.physical_constants import grav
    from jcm.utils import spectral_truncation
    from jcm.physics.speedy.land_model import land_model_init
    import xarray as xr
    
    ds = xr.open_dataset(filename)

    orog = jnp.asarray(ds["orog"])
    # Read surface geopotential (i.e. orography)
    phi0 = grav * orog
    # Also store spectrally truncated surface geopotential for the land drag term
    phis0 = spectral_truncation(grid, phi0, truncation_number=truncation_number)

    # Read land-sea mask
    fmask = jnp.asarray(ds["lsm"])
    # Annual-mean surface albedo
    alb0 = jnp.asarray(ds["alb"])
    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"

    tsea = _fixed_ssts(grid)
    boundaries = BoundaryData.zeros(
        nodal_shape=fmask.shape,
        fmask=fmask, orog=orog, phis0=phis0, tsea=tsea, alb0=alb0)
    
    boundaries = land_model_init(filename, parameters, boundaries)

    # call sea model init
    # call ice model init

    return boundaries