import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes

@tree_math.struct
class ForcingData:
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    sice_am: jnp.ndarray # sea ice concentration
    snowc_am: jnp.ndarray # snow cover (used to be snowcl_ob in fortran - but one day of that was snowc_am)
    soilw_am: jnp.ndarray # soil moisture (used to be soilwcl_ob in fortran - but one day of that was soilw_am)
    tsea: jnp.ndarray # SST, should come from sea_model.py or some default value

    @classmethod
    def zeros(cls,nodal_shape,
              alb0=None,sice_am=None,snowc_am=None,
              soilw_am=None,tsea=None):
        return cls(
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.zeros((nodal_shape)+(365,)),
            snowc_am=snowc_am if snowc_am is not None else jnp.zeros((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)+(365,)),
            tsea=tsea if tsea is not None else jnp.zeros((nodal_shape)+(365,)),
        )

    @classmethod
    def ones(cls,nodal_shape,
             alb0=None,sice_am=None,snowc_am=None,
             soilw_am=None,tsea=None):
        return cls(
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)+(365,)),
            snowc_am=snowc_am if snowc_am is not None else jnp.ones((nodal_shape)+(365,)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)+(365,)),
            tsea=tsea if tsea is not None else jnp.ones((nodal_shape)+(365,)),
        )

    def copy(self,alb0=None,
             sice_am=None,snowc_am=None,soilw_am=None,
             tsea=None):
        return ForcingData(
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            snowc_am=snowc_am if snowc_am is not None else self.snowc_am,
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

def forcing_from_file(filename: str,) -> ForcingData:
    """
    Initialize the forcing data from a file.

    Returns:
        ForcingData: Time-varying forcing data
    """
    import xarray as xr

    # Read forcing data from file
    ds = xr.open_dataset(filename)

    # annual-mean surface albedo
    alb0 = jnp.asarray(ds["alb"])

    # sea ice concentration
    sice_am = jnp.asarray(ds["icec"])

    # snow depth
    snowc_am = jnp.asarray(ds["snowc"])
    snowc_valid = (0.0 <= snowc_am) & (snowc_am <= 20000.0)
    # assert jnp.all(snowc_valid | (fmask[:,:,jnp.newaxis] == 0.0)) # FIXME: need to change the forcing.nc file so this passes
    snowc_am = jnp.where(snowc_valid, snowc_am, 0.0)

    # soil moisture
    soilw_am = jnp.asarray(ds["soilw_am"])

    # Prescribe SSTs
    tsea = jnp.asarray(ds["sst"])

    forcing_data = ForcingData.zeros(
        nodal_shape=alb0.shape,
        alb0=alb0, sice_am=sice_am, snowc_am=snowc_am,
        soilw_am=soilw_am, tsea=tsea
    )

    return forcing_data
