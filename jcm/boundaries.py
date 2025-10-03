import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes

@tree_math.struct
class BoundaryData:
    fmask: jnp.ndarray # fractional land-sea mask (ix,il)
    orog: jnp.ndarray # orography in meters
    phis0: jnp.ndarray # spectrally-filtered surface geopotential
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    sice_am: jnp.ndarray # FIXME: need to set this
    snowd_am: jnp.ndarray # used to be snowcl_ob in fortran - but one day of that was snowd_am
    soilw_am: jnp.ndarray # used to be soilwcl_ob in fortran - but one day of that was soilw_am
    lfluxland: jnp.bool # flag to compute land skin temperature and latent fluxes
    tsea: jnp.ndarray # SST, should come from sea_model.py or some default value

    @classmethod
    def zeros(cls,nodal_shape,fmask=None,orog=None,phis0=None,
              alb0=None,sice_am=None,snowd_am=None,
              soilw_am=None,tsea=None,lfluxland=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.zeros((nodal_shape)),
            orog=orog if orog is not None else jnp.zeros((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.zeros((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.zeros((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.zeros((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)+(365,)),
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.zeros((nodal_shape)+(365,)),
        )

    @classmethod
    def ones(cls,nodal_shape,fmask=None,orog=None,phis0=None,
             alb0=None,sice_am=None,snowd_am=None,
             soilw_am=None,tsea=None,lfluxland=None):
        return cls(
            fmask=fmask if fmask is not None else jnp.ones((nodal_shape)),
            orog=orog if orog is not None else jnp.ones((nodal_shape)),
            phis0=phis0 if phis0 is not None else jnp.ones((nodal_shape)),
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.ones((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)+(365,)),
            lfluxland=lfluxland if lfluxland is not None else True,
            tsea=tsea if tsea is not None else jnp.ones((nodal_shape)+(365,)),
        )

    def copy(self,fmask=None,orog=None,phis0=None,alb0=None,
             sice_am=None,snowd_am=None,soilw_am=None,
             tsea=None,lfluxland=None):
        return BoundaryData(
            fmask=fmask if fmask is not None else self.fmask,
            orog=orog if orog is not None else self.orog,
            phis0=phis0 if phis0 is not None else self.phis0,
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            snowd_am=snowd_am if snowd_am is not None else self.snowd_am,
            lfluxland=lfluxland if lfluxland is not None else self.lfluxland,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            tsea=tsea if tsea is not None else self.tsea,
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
    tsea = jnp.tile(_fixed_ssts(grid)[:, :, jnp.newaxis], (1, 1, 365))
        
    return BoundaryData.zeros(
        nodal_shape=orography.shape,
        orog=orography, fmask=fmask, phis0=phis0, tsea=tsea, alb0=alb0
    )

def boundaries_from_file(
    filename: str,
    grid: HorizontalGridTypes,
    truncation_number=None,
    fmask_threshold=0.1,
) -> BoundaryData:
    """
    Initialize the boundary conditions
    """
    from jcm.physics.speedy.physical_constants import grav
    from jcm.utils import spectral_truncation
    import xarray as xr

    # Read boundaries from file
    ds = xr.open_dataset(filename)

    # land-sea mask
    fmask = jnp.asarray(ds["lsm"])
    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"
    # Set values close to 0 or 1 to exactly 0 or 1
    fmask = jnp.where(fmask <= fmask_threshold, 0.0, jnp.where(fmask >= 1.0 - fmask_threshold, 1.0, fmask))

    # orography
    orog = jnp.asarray(ds["orog"])
    # Also store spectrally truncated surface geopotential for the land drag term
    phi0 = grav * orog
    phis0 = spectral_truncation(grid, phi0, truncation_number=truncation_number)

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
        orog=orog, phis0=phis0, alb0=alb0, sice_am=sice_am,
        snowd_am=snowd_am, soilw_am=soilw_am, tsea=tsea
    )