import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes

@tree_math.struct
class ForcingData:
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    sice_am: jnp.ndarray # sea ice concentration
    snowd_am: jnp.ndarray # snow depth (used to be snowcl_ob in fortran - but one day of that was snowd_am)
    soilw_am: jnp.ndarray # soil moisture (used to be soilwcl_ob in fortran - but one day of that was soilw_am)
    tsea: jnp.ndarray # SST, should come from sea_model.py or some default value

    @classmethod
    def zeros(cls,nodal_shape,
              alb0=None,sice_am=None,snowd_am=None,
              soilw_am=None,tsea=None):
        return cls(
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.zeros((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.zeros((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)+(365,)),
            tsea=tsea if tsea is not None else jnp.zeros((nodal_shape)+(365,)),
        )

    @classmethod
    def ones(cls,nodal_shape,
             alb0=None,sice_am=None,snowd_am=None,
             soilw_am=None,tsea=None):
        return cls(
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)+(365,)),
            snowd_am=snowd_am if snowd_am is not None else jnp.ones((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)+(365,)),
            tsea=tsea if tsea is not None else jnp.ones((nodal_shape)+(365,)),
        )

    def copy(self,alb0=None,
             sice_am=None,snowd_am=None,soilw_am=None,
             tsea=None,lfluxland=None):
        return ForcingData(
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            snowd_am=snowd_am if snowd_am is not None else self.snowd_am,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            tsea=tsea if tsea is not None else self.tsea,
        )

    def isnan(self):
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

def default_forcing(
    grid: HorizontalGridTypes,
) -> ForcingData:
    """
    Initialize the default forcing data with prescribed SSTs
    """
    alb0 = jnp.zeros(grid.nodal_shape)
    tsea = jnp.tile(_fixed_ssts(grid)[:, :, jnp.newaxis], (1, 1, 365))

    return ForcingData.zeros(
        nodal_shape=grid.nodal_shape,
        tsea=tsea, alb0=alb0
    )

def forcing_from_file(
    filename: str,
    fmask_threshold=0.1,
) -> tuple[ForcingData, jnp.ndarray]:
    """
    Initialize the forcing data from a file.

    Returns:
        ForcingData: Time-varying forcing data
        jnp.ndarray: Land-sea mask (fmask) to be added to Geometry
    """
    import xarray as xr

    # Read forcing data from file
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

    forcing_data = ForcingData.zeros(
        nodal_shape=fmask.shape,
        alb0=alb0, sice_am=sice_am, snowd_am=snowd_am,
        soilw_am=soilw_am, tsea=tsea
    )

    return forcing_data, fmask
