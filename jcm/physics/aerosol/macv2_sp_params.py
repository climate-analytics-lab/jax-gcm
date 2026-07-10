"""JAX-compatible aerosol parameters for MACv2-SP (Simple Plumes) scheme

This module defines the aerosol parameters following the SpeedyPhysics pattern
using tree_math.struct for JAX compatibility. Based on the ICON implementation
in mo_bc_aeropt_splumes.f90.

Date: 2025-01-11
"""

import jax.numpy as jnp
import tree_math
from jax import tree_util


@tree_math.struct
class AerosolParameters:
    """Parameters for MACv2-SP (Simple Plumes) aerosol scheme
    
    This implements the simple plume aerosol parametrization based on
    Kinne et al. climatology with 9 anthropogenic plumes and natural
    background aerosol.
    """
    
    # Number of plumes and features
    nplumes: int
    nfeatures: int
    
    # Plume center locations [degrees]
    plume_lat: jnp.ndarray        # (nplumes,) latitude of plume centers
    plume_lon: jnp.ndarray        # (nplumes,) longitude of plume centers
    
    # Vertical distribution parameters (beta function)
    beta_a: jnp.ndarray           # (nplumes,) beta function parameter a
    beta_b: jnp.ndarray           # (nplumes,) beta function parameter b
    
    # Aerosol optical properties at 550nm
    aod_spmx: jnp.ndarray         # (nplumes,) AOD at 550nm for simple plume (maximum)
    aod_fmbg: jnp.ndarray         # (nplumes,) AOD at 550nm for fine mode background
    asy550: jnp.ndarray           # (nplumes,) asymmetry parameter at 550nm
    ssa550: jnp.ndarray           # (nplumes,) single scattering albedo at 550nm
    angstrom: jnp.ndarray         # (nplumes,) Angstrom parameter
    
    # Spatial extent parameters [degrees]
    sig_lon_E: jnp.ndarray        # (nfeatures, nplumes) Eastward longitude extent
    sig_lon_W: jnp.ndarray        # (nfeatures, nplumes) Westward longitude extent
    sig_lat_E: jnp.ndarray        # (nfeatures, nplumes) Southward latitude extent  
    sig_lat_W: jnp.ndarray        # (nfeatures, nplumes) Northward latitude extent
    
    # Feature weights and rotation
    theta: jnp.ndarray            # (nfeatures, nplumes) Rotation angle [radians]
    ftr_weight: jnp.ndarray       # (nfeatures, nplumes) Feature weights
    
    # Natural background AOD
    background_aod: jnp.ndarray   # Background AOD at 550nm (scalar)

    # SPA-style activation fit (Lin et al. 2025) used by the 2M micro-
    # physics: ``Nc_min[cm^-3] = spa_prefactor * (Nccn * Cf)^spa_exponent``.
    # Stored on parameters (not a module-level constant) so they're
    # differentiable through `jax.grad` for sensitivity / calibration.
    spa_prefactor: jnp.ndarray    # default 2000.0 (Lin 2025, fit to E3SMv3)
    spa_exponent: jnp.ndarray     # default 0.55  (sublinear; observational band 0.3–0.8)
    # Half-width [m^-3] of the smooth transition that replaces the hard
    # ``min(Nc_min, arg)`` physical cap in ``spa_activated_cdnc``. The hard min
    # makes the loss piecewise in ``spa_prefactor`` (the cap boundary sits in
    # the middle of the cloudy-cell distribution, so cells flip branches as the
    # prefactor moves and the exact local gradient can even have the wrong sign
    # relative to the large-scale response — measured through a 2-step T21L47
    # rollout). Default 1e6 m^-3 (1 cm^-3), a few percent of typical activated
    # CDNC, so the forward changes by at most half that at the cap corner.
    # Set to 0.0 to recover the hard cap exactly.
    spa_cap_smoothing: jnp.ndarray

    @classmethod
    def default(cls, background_aod=0.02,
                spa_prefactor=2000.0, spa_exponent=0.55,
                spa_cap_smoothing=1.0e6) -> 'AerosolParameters':
        """Create the MACv2-SP v1 reference parameters.

        Static plume geometry/optics transcribed verbatim from
        ``MACv2.0-SP_v1.nc`` (Stevens et al. 2017 GMD supplement). Plume
        order follows the file: 1 Europe, 2 North America, 3 East Asia,
        4 South Asia, 5 North Africa (biomass), 6 South America (biomass),
        7 Maritime Continent (biomass), 8 South-Central Africa (biomass),
        9 Australia. The order is load-bearing: the 260-degree
        longitudinal-wrap special case in the spatial kernel is keyed to
        plume index 0 (Europe, whose trans-Atlantic tail crosses 0 E).
        Longitudes use the file's 0-360 convention, matching the
        dinosaur-derived column longitudes cached by the term.
        """
        nplumes = 9
        nfeatures = 2

        # Plume centers [degrees N / degrees E, 0-360].
        plume_lat = jnp.array(
            [49.4, 40.1, 30.0, 23.3, 3.5, -10.3, -1.0, -3.5, -20.0])
        plume_lon = jnp.array(
            [20.6, 277.5, 114.0, 88.0, 22.5, 298.0, 106.0, 16.0, 135.0])

        # Beta-function vertical-profile shape parameters.
        beta_a = jnp.array([1.5, 1.7, 1.3, 1.3, 7.0, 1.2, 2.3, 2.4, 1.4])
        beta_b = jnp.array([17.0, 17.0, 13.0, 8.0, 35.0, 9.0, 23.0, 14.0, 11.0])

        # 550 nm AOD at the plume source (anthropogenic max and fine-mode
        # background), single-scattering albedo, asymmetry, Angstrom.
        aod_spmx = jnp.array(
            [0.148, 0.094, 0.636, 0.259, 0.211, 0.351, 0.257, 0.372, 0.075])
        aod_fmbg = jnp.array([0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.1])
        ssa550 = jnp.array(
            [0.93, 0.93, 0.93, 0.93, 0.87, 0.87, 0.87, 0.87, 0.93])
        asy550 = jnp.full(9, 0.63)
        angstrom = jnp.full(9, 2.0)

        # Anisotropic Gaussian extents [degrees], (nplumes, 2 features)
        # in the file; transposed to the struct's (nfeatures, nplumes).
        sig_lat_W = jnp.array([
            [6., 10.], [7., 25.], [6., 13.], [9., 15.], [6., 1.],
            [6., 6.], [8., 4.], [9., 5.], [6., 12.],
        ]).T
        sig_lat_E = jnp.array([
            [6., 10.], [7., 8.], [6., 13.], [8., 17.], [6., 1.],
            [6., 6.], [8., 4.], [9., 5.], [6., 12.],
        ]).T
        sig_lon_W = jnp.array([
            [7., 35.], [20., 8.], [9., 15.], [15., 40.], [32., 3.],
            [10., 8.], [12., 4.], [23., 7.], [10., 20.],
        ]).T
        sig_lon_E = jnp.array([
            [13., 80.], [35., 11.], [8., 40.], [10., 15.], [6., 3.],
            [10., 8.], [10., 6.], [14., 6.], [4., 20.],
        ]).T

        # Feature rotation angles [radians, clockwise] and weights.
        theta = jnp.array([
            [0.0, 0.174533],
            [0.261799, 2.268928],
            [0.698132, 0.261799],
            [0.0, 0.261799],
            [0.0, 0.0],
            [-0.523599, -0.523599],
            [0.174533, 0.0],
            [-0.261799, -0.261799],
            [0.0, -0.523599],
        ]).T
        ftr_weight = jnp.array([
            [0.4, 0.6],
            [0.6, 0.4],
            [0.857143, 0.142857],
            [0.6, 0.4],
            [0.8, 0.2],
            [0.125, 0.875],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.8, 0.2],
        ]).T

        return cls(
            nplumes=nplumes,
            nfeatures=nfeatures,
            plume_lat=plume_lat,
            plume_lon=plume_lon,
            beta_a=beta_a,
            beta_b=beta_b,
            aod_spmx=aod_spmx,
            aod_fmbg=aod_fmbg,
            asy550=asy550,
            ssa550=ssa550,
            angstrom=angstrom,
            sig_lon_E=sig_lon_E,
            sig_lon_W=sig_lon_W,
            sig_lat_E=sig_lat_E,
            sig_lat_W=sig_lat_W,
            theta=theta,
            ftr_weight=ftr_weight,
            background_aod=jnp.array(background_aod),
            spa_prefactor=jnp.array(spa_prefactor),
            spa_exponent=jnp.array(spa_exponent),
            spa_cap_smoothing=jnp.array(spa_cap_smoothing),
        )

    @classmethod
    def from_dataset(cls, ds, background_aod=0.02,
                     spa_prefactor=2000.0, spa_exponent=0.55,
                     spa_cap_smoothing=1.0e6) -> 'AerosolParameters':
        """Build parameters from an opened ``MACv2.0-SP_v1.nc`` dataset.

        Owns the (plume, feature) -> (feature, plume) transposes and the
        jcm-specific extension fields, so callers (notebook 06) cannot
        hit the missing-field TypeError again. The time-varying
        ``year_weight`` / ``ann_cycle`` are forcing data, not parameters
        — load those with the TimeSeries recipe in notebook 06 (mind the
        _FillValue masking of year_weight beyond 2016).
        """
        as_arr = lambda name: jnp.asarray(ds[name].values)
        as_arr_T = lambda name: jnp.asarray(ds[name].values.T)
        return cls(
            nplumes=int(ds.sizes["plume_number"]),
            nfeatures=int(ds.sizes["plume_feature"]),
            plume_lat=as_arr("plume_lat"),
            plume_lon=as_arr("plume_lon"),
            beta_a=as_arr("beta_a"),
            beta_b=as_arr("beta_b"),
            aod_spmx=as_arr("aod_spmx"),
            aod_fmbg=as_arr("aod_fmbg"),
            asy550=as_arr("asy550"),
            ssa550=as_arr("ssa550"),
            angstrom=as_arr("angstrom"),
            sig_lon_E=as_arr_T("sig_lon_E"),
            sig_lon_W=as_arr_T("sig_lon_W"),
            sig_lat_E=as_arr_T("sig_lat_E"),
            sig_lat_W=as_arr_T("sig_lat_W"),
            theta=as_arr_T("theta"),
            ftr_weight=as_arr_T("ftr_weight"),
            background_aod=jnp.array(background_aod),
            spa_prefactor=jnp.array(spa_prefactor),
            spa_exponent=jnp.array(spa_exponent),
            spa_cap_smoothing=jnp.array(spa_cap_smoothing),
        )

    def isnan(self):
        """Check for NaN values in parameters"""
        return tree_util.tree_map(jnp.isnan, self)