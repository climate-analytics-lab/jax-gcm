import jax.numpy as jnp
from jcm.physics_interface import PhysicsTendency
from jcm.forcing import ForcingData
from .macv2_sp_params import AerosolParameters


def get_simple_aerosol(
    height_full: jnp.ndarray,
    lats_deg: jnp.ndarray,
    lons_deg: jnp.ndarray,
    aerosol_data,
    parameters: AerosolParameters,
    forcing: ForcingData,
):
    """Apply MACv2-SP (Simple Plumes) aerosol scheme.

    Implements the simplified aerosol parametrisation based on the Kinne
    et al. climatology with 9 anthropogenic plumes plus a natural
    background. Computes AOD, single scattering albedo, asymmetry
    parameter profiles, column-integrated properties, and the Twomey
    effect on cloud droplet number concentration.

    Args:
        height_full: Layer-centre height (m), shape ``(nlev, ncols)``.
        lats_deg: Per-column latitude in degrees, shape ``(ncols,)``.
        lons_deg: Per-column longitude in degrees, shape ``(ncols,)``.
        aerosol_data: Existing :class:`AerosolData` to update via
            ``.copy(...)``. Lets the caller decide whether to seed it
            from zeros or from the previous step.
        parameters: Aerosol parameters wrapper exposing ``.aerosol``.
        forcing: Forcing data — uses ``aerosol_year_weight`` and
            ``aerosol_ann_cycle`` for time-varying emissions.

    Returns:
        Updated ``AerosolData`` with AOD profile, optical properties,
        column AOD, anthropogenic CDNC factor, and CCN concentration set.

    """
    nlev, ncols = height_full.shape
    aerosol_params = parameters

    # Get temporal weights from forcing data (allows time-varying emissions)
    year_weight = forcing.aerosol_year_weight
    ann_cycle = forcing.aerosol_ann_cycle

    # Compute the plume spatial distribution once; ``get_anthropogenic_aod`` /
    # ``get_background_aod`` used to recompute it internally so the Gaussian
    # evaluation ran three times per step. Pass it through instead.
    spatial_dist = get_plume_spatial_distribution(
        lats_deg, lons_deg, aerosol_params,
    )

    # Calculate anthropogenic and background AOD using vectorized operations
    aod_anthropogenic = get_anthropogenic_aod(
        aerosol_params, year_weight, ann_cycle, spatial_dist,
    )
    aod_background = get_background_aod(
        aerosol_params, ann_cycle, spatial_dist,
    )

    # Calculate vertical profiles for each plume using vectorized operations
    plume_profiles = get_vertical_profiles(height_full, aerosol_params)

    # Combine plume contributions using vectorized operations
    # plume_profiles: (nplumes, nlev, ncols)
    # spatial_dist: (nplumes, ncols)
    # aod_anthropogenic: (ncols,)
    # Need to broadcast properly for multiplication
    plume_contribution = jnp.sum(
        plume_profiles
        * (aod_anthropogenic[jnp.newaxis, jnp.newaxis, :]
           * spatial_dist[:, jnp.newaxis, :]),
        axis=0,
    )

    # Add background contribution with uniform vertical distribution
    bg_profile = get_background_vertical_profile(height_full)
    bg_contribution = (
        bg_profile[:, jnp.newaxis] * aod_background[jnp.newaxis, :]
    )

    # Combine anthropogenic and background contributions
    aod_profile = plume_contribution + bg_contribution

    # Calculate optical properties using weighted averages
    ssa_profile, asy_profile, angstrom = get_optical_properties(
        aod_profile, spatial_dist, aerosol_params,
    )

    # Calculate total column AOD
    aod_total = jnp.sum(aod_profile, axis=0)

    # Calculate Twomey effect using proper CDNC relationship
    cdnc_factor = (
        get_CDNC(aod_anthropogenic)
        / get_CDNC(jnp.zeros_like(aod_anthropogenic))
    )

    # CCN concentration [cm^-3] for the SPA-style activation floor used by
    # the two-moment microphysics. Both anthropogenic and background plume
    # contributions feed in — i.e. the column AOD is the source. See
    # ``jcm.physics.aerosol.spa.spa_activated_cdnc`` for the consumer.
    Nccn = get_CDNC(aod_anthropogenic + aod_background)

    return aerosol_data.copy(
        aod_profile=aod_profile,
        ssa_profile=ssa_profile,
        asy_profile=asy_profile,
        aod_total=aod_total,
        aod_anthropogenic=aod_anthropogenic,
        aod_background=aod_background,
        cdnc_factor=cdnc_factor,
        Nccn=Nccn,
        angstrom=angstrom,
    )

def _per_feature_plume_gaussians(lats, lons, parameters):
    """Per-feature, per-plume Gaussian shapes — `(nfeatures, nplumes, ncols)`.

    Internal helper, used both by `get_plume_spatial_distribution` (which
    sums features with `ftr_weight`) and by the date-aware AOD path which
    needs to multiply per-feature time weights by the per-feature gaussian
    before reducing — exactly mirroring the Fortran `mo_simple_plumes_v1`
    behavior that the JAX port previously collapsed by treating
    `ann_cycle` as 1-D.
    """
    delta_lat = lats[jnp.newaxis, :] - parameters.plume_lat[:, jnp.newaxis]  # (nplumes, ncols)
    delta_lon = lons[jnp.newaxis, :] - parameters.plume_lon[:, jnp.newaxis]

    delta_lon_t = jnp.ones_like(parameters.plume_lon) * 180
    delta_lon_t = delta_lon_t.at[0].set(260)  # First plume is different

    delta_lon = jnp.where(
        jnp.abs(delta_lon) > delta_lon_t[:, jnp.newaxis],
        jnp.where(delta_lon >= 0, delta_lon - 360, delta_lon + 360),
        delta_lon,
    )

    sig_lon = jnp.where(
        delta_lon[jnp.newaxis, :, :] > 0.0,
        parameters.sig_lon_E[:, :, jnp.newaxis],
        parameters.sig_lon_W[:, :, jnp.newaxis],
    )
    sig_lat = jnp.where(
        delta_lon[jnp.newaxis, :, :] > 0.0,
        parameters.sig_lat_E[:, :, jnp.newaxis],
        parameters.sig_lat_W[:, :, jnp.newaxis],
    )
    a_plume = 0.5 / (sig_lon ** 2)
    b_plume = 0.5 / (sig_lat ** 2)

    cos_theta = jnp.cos(parameters.theta)[:, :, jnp.newaxis]
    sin_theta = jnp.sin(parameters.theta)[:, :, jnp.newaxis]
    lon_rot = (cos_theta * delta_lon[jnp.newaxis, :, :]
               + sin_theta * delta_lat[jnp.newaxis, :, :])
    lat_rot = (-sin_theta * delta_lon[jnp.newaxis, :, :]
               + cos_theta * delta_lat[jnp.newaxis, :, :])
    return jnp.exp(-1.0 * (a_plume * lon_rot ** 2 + b_plume * lat_rot ** 2))


def get_plume_spatial_distribution(lats, lons, parameters):
    """Calculate spatial distribution of aerosol plumes using Gaussian functions

    Args:
        lats: Array of latitudes [degrees]
        lons: Array of longitudes [degrees]
        parameters: AerosolParameters object

    Returns:
        Spatial distribution array of shape (nplumes, ncols), with the
        feature axis already collapsed via `ftr_weight`. For the
        date-aware path that needs the per-feature gaussians, use
        `_per_feature_plume_gaussians` directly.

    """
    gaussian = _per_feature_plume_gaussians(lats, lons, parameters)
    weighted_gaussian = parameters.ftr_weight[:, :, jnp.newaxis] * gaussian
    return jnp.sum(weighted_gaussian, axis=0)  # (nplumes, ncols)


def _effective_ann_weight(ann_cycle, parameters):
    """Reduce a possibly per-feature `ann_cycle` to a per-plume weight.

    Accepts either:
      * 1-D `(nplumes,)` — the legacy placeholder shape; passed through.
      * 2-D `(nfeatures, nplumes)` — the proper MACv2-SP shape, reduced
        with the parameter `ftr_weight` so the resulting per-plume weight
        is a faithful average of the per-feature annual cycle (the
        Fortran multiplies feature-by-feature inside the spatial sum;
        because the feature axis is already collapsed in `spatial_dist`
        via `ftr_weight`, the equivalent per-plume time weight here is
        the `ftr_weight`-weighted feature sum).
    """
    if ann_cycle.ndim == 1:
        return ann_cycle
    if ann_cycle.ndim == 2:
        ftr_w = parameters.ftr_weight  # (nfeatures, nplumes)
        # ftr_weights typically sum to 1 over features; if not, normalize so
        # the legacy "all-ones placeholder" still maps to all-ones.
        norm = jnp.sum(ftr_w, axis=0)
        norm = jnp.where(norm > 0.0, norm, 1.0)
        return jnp.sum(ftr_w * ann_cycle, axis=0) / norm
    raise ValueError(
        f"ann_cycle must be 1-D (nplumes,) or 2-D (nfeatures, nplumes); got shape {ann_cycle.shape}"
    )


def get_background_aod(parameters, ann_cycle, spatial_dist, constant_background=0.02):
    """Calculate background (pre-industrial) aerosol optical depth.

    Args:
        parameters: AerosolParameters object
        ann_cycle: Annual cycle weights — either (nplumes,) or
            (nfeatures, nplumes) from forcing data
        spatial_dist: Precomputed plume Gaussian distribution (nplumes, ncols)
        constant_background: Constant background AOD value

    Returns:
        Background AOD array of shape (ncols,)

    """
    eff_ann = _effective_ann_weight(ann_cycle, parameters)
    cw_bg = eff_ann[:, jnp.newaxis] * parameters.aod_fmbg[:, jnp.newaxis] * spatial_dist
    aod_PI = jnp.sum(cw_bg, axis=0) + constant_background
    return aod_PI


def get_anthropogenic_aod(parameters, year_weight, ann_cycle, spatial_dist):
    """Calculate anthropogenic aerosol optical depth.

    Args:
        parameters: AerosolParameters object
        year_weight: Year-specific emission weights (nplumes,) from forcing data
        ann_cycle: Annual cycle weights (nplumes,) or (nfeatures, nplumes)
        spatial_dist: Precomputed plume Gaussian distribution (nplumes, ncols)

    Returns:
        Anthropogenic AOD array of shape (ncols,)

    """
    eff_ann = _effective_ann_weight(ann_cycle, parameters)
    time_weight = year_weight * eff_ann
    cw_an = time_weight[:, jnp.newaxis] * parameters.aod_spmx[:, jnp.newaxis] * spatial_dist
    aod_anth = jnp.sum(cw_an, axis=0)
    return aod_anth


def get_vertical_profiles(height_full, parameters):
    """Calculate vertical profiles for all plumes using beta function distribution
    
    Args:
        height_full: Height coordinate array of shape (nlev, ncols)
        parameters: AerosolParameters object
        
    Returns:
        Vertical profiles array of shape (nplumes, nlev, ncols)

    """
    # Normalize height to 0-1 range (0 at surface, 1 at 15km)
    height_norm = jnp.clip(height_full / 15000.0, 0.0, 1.0)
    
    # Calculate beta function profiles for each plume
    # height_norm: (nlev, ncols)
    # parameters.beta_a, parameters.beta_b: (nplumes,)
    
    # Expand dimensions for vectorized calculation
    # height_norm: (1, nlev, ncols)
    # beta_a, beta_b: (nplumes, 1, 1)
    height_expanded = height_norm[jnp.newaxis, :, :]
    beta_a_expanded = parameters.beta_a[:, jnp.newaxis, jnp.newaxis]
    beta_b_expanded = parameters.beta_b[:, jnp.newaxis, jnp.newaxis]
    
    # Calculate beta function: x^(a-1) * (1-x)^(b-1)
    # Avoid issues at boundaries by adding small epsilon
    eps = 1e-10
    x = jnp.clip(height_expanded, eps, 1.0 - eps)
    
    beta_profile = (x**(beta_a_expanded - 1)) * ((1 - x)**(beta_b_expanded - 1))
    
    # Normalize profiles to integrate to 1 over height
    profile_sum = jnp.sum(beta_profile, axis=1, keepdims=True)
    profile_sum = jnp.where(profile_sum > 0, profile_sum, 1.0)  # Avoid division by zero
    
    normalized_profiles = beta_profile / profile_sum
    
    return normalized_profiles  # (nplumes, nlev, ncols)


def get_background_vertical_profile(height_full):
    """Calculate vertical profile for background aerosol
    
    Args:
        height_full: Height coordinate array of shape (nlev, ncols)
        
    Returns:
        Background vertical profile array of shape (nlev,)

    """
    # Simple exponential decay for background aerosol
    # Use mean height profile across columns
    height_mean = jnp.mean(height_full, axis=1)
    
    # Exponential decay with 2km scale height
    scale_height = 2000.0  # meters
    profile = jnp.exp(-height_mean / scale_height)
    
    # Normalize to integrate to 1
    profile = profile / jnp.sum(profile)
    
    return profile


def get_optical_properties(aod_profile, spatial_dist, parameters):
    """Calculate single scattering albedo, asymmetry parameter, and Angstrom exponent

    Args:
        aod_profile: AOD profile array of shape (nlev, ncols)
        spatial_dist: Spatial distribution array of shape (nplumes, ncols)
        parameters: AerosolParameters object

    Returns:
        Tuple of (ssa_profile, asy_profile, angstrom) where profiles are
        (nlev, ncols) and angstrom is (ncols,)

    """
    # Weight optical properties by AOD contribution from each plume
    # aod_profile: (nlev, ncols)
    # spatial_dist: (nplumes, ncols)
    # parameters.ssa550, parameters.asy550, parameters.angstrom: (nplumes,)

    # Calculate plume contributions to total AOD
    total_aod = jnp.sum(aod_profile, axis=0, keepdims=True)  # (1, ncols)
    total_aod = jnp.where(total_aod > 0, total_aod, 1.0)  # Avoid division by zero

    # Weight by spatial distribution
    plume_weights = spatial_dist / jnp.sum(spatial_dist, axis=0, keepdims=True)

    # Calculate weighted optical properties
    ssa_weighted = jnp.sum(
        plume_weights * parameters.ssa550[:, jnp.newaxis],
        axis=0
    )
    asy_weighted = jnp.sum(
        plume_weights * parameters.asy550[:, jnp.newaxis],
        axis=0
    )
    angstrom_weighted = jnp.sum(
        plume_weights * parameters.angstrom[:, jnp.newaxis],
        axis=0
    )

    # Expand to full vertical profile
    ssa_profile = jnp.ones_like(aod_profile) * ssa_weighted[jnp.newaxis, :]
    asy_profile = jnp.ones_like(aod_profile) * asy_weighted[jnp.newaxis, :]

    return ssa_profile, asy_profile, angstrom_weighted


def get_CDNC(AOD, A=60, B=20):
    """Derive CDNC from AOD using a relationship of the form: CDNC = A * ln(B*AOD + 1)
    Ross' amazon work: A=410 B=5
    MODIS original: A=16 B=1000
    AEROCOM P1 original: A=60, B=20
    """
    return 1 + A * jnp.log(B * AOD + 1)


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.physics.aerosol.aerosol_types import AerosolData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class Macv2SpAerosol(PhysicsTerm):
    """MACv2-SP simple-plumes aerosol scheme as a composable PhysicsTerm.

    Caches per-column latitude/longitude in degrees from the dinosaur
    coordinate system at ``cache_coords`` time. Each step reads
    ``height_full`` from the moist-air diagnostics dict, calls
    :func:`get_simple_aerosol` with the previous step's
    :class:`AerosolData` (or zeros on the first step), and writes the
    updated AOD/SSA/asymmetry/CDNC fields back under the public
    ``"aerosol"`` key. Returns zero atmospheric tendency — aerosol
    enters the dynamics indirectly through the radiation term and
    through the cloud-microphysics activation.
    """

    name: ClassVar[str] = "macv2_sp_aerosol"
    category: ClassVar[str] = "aerosol"
    requires: ClassVar[tuple[str, ...]] = ("height_full",)
    provides: ClassVar[tuple[str, ...]] = ("aerosol",)
    # Carry seeded as zeros; ``get_simple_aerosol`` rebuilds
    # AOD/SSA/asymmetry from the plume parameterisation every step
    # using the slot only as a shape source.
    carry_slots: ClassVar[dict[str, type]] = {"aerosol": AerosolData}

    def __init__(self, params: AerosolParameters | None = None):
        """Hold the scheme-native :class:`AerosolParameters`."""
        self.params = nnx.Param(params or AerosolParameters.default())
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (degrees) from the coordinate system.

        Uses the same lat/lon meshgrid → ``ncols`` reshape that the
        legacy ECHAM wrapper performed inline; doing it once here at
        construction time avoids repeating the ``meshgrid`` inside the
        jitted compute_tendencies loop.
        """
        lat_deg = jnp.asarray(coords.horizontal.latitudes) * 180.0 / jnp.pi
        lon_deg = jnp.asarray(coords.horizontal.longitudes) * 180.0 / jnp.pi
        # Match get_simple_aerosol's previous meshgrid convention:
        # ``meshgrid(lat, lon)`` returned (lat[None,:].repeat(nlon, 0),
        # lon[:,None].repeat(nlat, 1)) reshaped to (nlon*nlat,) ==
        # (ncols,) with longitude varying fastest.
        lat_2d, lon_2d = jnp.meshgrid(lat_deg, lon_deg)
        self._lats = nnx.Variable(lat_2d.reshape(-1))
        self._lons = nnx.Variable(lon_2d.reshape(-1))
        self._coords_cached = True

    def __call__(
        self,
        state,
        diagnostics: dict,
        forcing,
        terrain: TerrainData,
    ):
        """Update the aerosol diagnostics for the current step."""
        nlev, ncols = state.temperature.shape
        params = self.params.get_value()

        prev = diagnostics.get(
            "aerosol", AerosolData.zeros((ncols,), nlev),
        )
        new_aerosol = get_simple_aerosol(
            height_full=diagnostics["height_full"],
            lats_deg=self._lats.get_value(),
            lons_deg=self._lons.get_value(),
            aerosol_data=prev,
            parameters=params,
            forcing=forcing,
        )

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {**diagnostics, "aerosol": new_aerosol}
