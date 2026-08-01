import jax.numpy as jnp

from jcm.physics.coords_util import column_lat_lon
from jcm.physics_interface import PhysicsTendency
from jcm.forcing import ForcingData
from .macv2_sp_params import AerosolParameters


def get_simple_aerosol(
    height_full: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    orography: jnp.ndarray,
    lats_deg: jnp.ndarray,
    lons_deg: jnp.ndarray,
    aerosol_data,
    parameters: AerosolParameters,
    forcing: ForcingData,
    sw_band_centers_nm: jnp.ndarray,
):
    """Apply MACv2-SP (Simple Plumes) aerosol scheme.

    Faithful port of ``sp_aop_profile`` (mo_simple_plumes_v1.f90, Stevens
    et al. 2017 GMD supplement): 9 anthropogenic plumes with per-feature
    rotated Gaussians and time weights, dz-weighted beta vertical
    profiles truncated at the orography, per-plume anthropogenic-AOD-
    weighted optics per SW band, and the Twomey factor ``dNovrN`` from
    the column plume AOD against the natural background. The natural
    background (0.02 + fine-mode plume background) is a column scalar
    feeding only ``dNovrN`` / ``Nccn`` — it never enters the radiative
    AOD/SSA/ASY profiles (the reference has no vertically distributed
    background).

    Args:
        height_full: Layer-centre height above sea level (m),
            shape ``(nlev, ncols)``, level 0 = model top.
        layer_thickness: Layer geometric depth dz (m), ``(nlev, ncols)``.
        orography: Surface height above sea level (m), ``(ncols,)``.
        lats_deg: Per-column latitude in degrees, shape ``(ncols,)``.
        lons_deg: Per-column longitude in degrees (0-360), ``(ncols,)``.
        aerosol_data: Existing :class:`AerosolData` to update via
            ``.copy(...)``.
        parameters: :class:`AerosolParameters` (MACv2-SP v1 values).
        forcing: Forcing data — ``aerosol_year_weight`` ``(nplumes,)``
            and ``aerosol_ann_cycle`` ``(nfeatures, nplumes)`` sampled
            for the current date (piecewise-constant per calendar year /
            per 1/52-year bin in the reference; all-ones defaults mean
            "perpetual 2005 amplitude, no seasonal cycle").
        sw_band_centers_nm: 1-D array of SW band-center wavelengths (nm)
            from the active radiation backend's ``RadiationBandConfig``;
            ``[550.0]`` for grey/broadband.

    Returns:
        Updated ``AerosolData``.

    """
    year_weight = forcing.aerosol_year_weight     # (nplumes,)
    ann_cycle = forcing.aerosol_ann_cycle         # (nfeatures, nplumes)

    # Per-feature plume Gaussians; the feature axis must survive until
    # the time weights have multiplied in (Fortran f1/f2: each feature
    # carries its own ann_cycle — features genuinely differ for the
    # South-America and South-Central-Africa biomass plumes).
    gauss = _per_feature_plume_gaussians(lats_deg, lons_deg, parameters)

    cw_an, cw_bg = get_plume_column_weights(
        parameters, year_weight, ann_cycle, gauss,
    )                                              # each (nplumes, ncols)

    # dz-weighted, orography-truncated beta profiles (sum over levels
    # <= 1; < 1 over elevated terrain — the below-ground AOD is removed,
    # not redistributed, exactly as the reference).
    prof = get_vertical_profiles(
        height_full, layer_thickness, orography, parameters,
    )                                              # (nplumes, nlev, ncols)

    # 550 nm anthropogenic AOD, per plume per level (Fortran aod_550 =
    # prof * cw_an). This is the weight for every optics accumulation.
    aod_550 = prof * cw_an[:, jnp.newaxis, :]      # (nplumes, nlev, ncols)
    aod_profile = jnp.sum(aod_550, axis=0)         # (nlev, ncols)
    caod_sp = jnp.sum(aod_550, axis=(0, 1))        # (ncols,)
    caod_bg = parameters.background_aod + jnp.sum(
        prof * cw_bg[:, jnp.newaxis, :], axis=(0, 1),
    )                                              # (ncols,)

    # Zero-AOD threshold for the completing divisions. The Fortran uses
    # TINY(1.) in float64; in float32 the division VJP forms num/den**2,
    # so any threshold whose SQUARE underflows (den < ~1e-19) still
    # produces inf/NaN gradients through far-plume-tail cells. 1e-15 AOD
    # is radiatively indistinguishable from zero and keeps den**2 normal.
    tiny = jnp.asarray(1e-15, dtype=aod_profile.dtype)

    # 550 nm diagnostic SSA/ASY profiles: anthropogenic-AOD-weighted
    # plume means with the reference AOD->0 limits (ssa -> 1, asy -> 0).
    ssa_sum_550 = jnp.einsum('p,pkc->kc', parameters.ssa550, aod_550)
    asy_sum_550 = jnp.einsum(
        'p,pkc->kc', parameters.ssa550 * parameters.asy550, aod_550,
    )
    # Double-where gradient guard (the #547 pattern): the inactive branch
    # of a bare where() is still differentiated, and even a
    # maximum(den, tiny) guard NaNs in reverse mode because the division
    # VJP forms num/den**2 and tiny**2 underflows to zero in float32. The
    # denominator must be replaced by a benign constant where inactive.
    ssa_has = ssa_sum_550 > tiny
    aod_has = aod_profile > tiny
    asy_profile = jnp.where(
        ssa_has, asy_sum_550 / jnp.where(ssa_has, ssa_sum_550, 1.0), 0.0,
    )
    ssa_profile = jnp.where(
        aod_has, ssa_sum_550 / jnp.where(aod_has, aod_profile, 1.0), 1.0,
    )

    # Per-SW-band optics: the per-plume wavelength kernels (Stevens 2017
    # closed forms) enter the plume sum as (nb, nplumes) factors — an
    # einsum over the plume axis avoids materializing the 4-D
    # (nb, nplumes, nlev, ncols) product.
    lam = sw_band_centers_nm[:, jnp.newaxis]                   # (nb, 1)
    lfactor = jnp.minimum(1.0, 700.0 / lam)                    # (nb, np)-bcast
    ssa_num = parameters.ssa550 * lfactor ** 4
    ssa_b = ssa_num / (ssa_num + (1.0 - parameters.ssa550) * lfactor)
    asy_b = parameters.asy550 * jnp.sqrt(lfactor)              # (nb, np)
    lfac_aod = jnp.exp(
        -parameters.angstrom * jnp.log(lam / 550.0)
    )                                                          # (nb, np)

    aod_sw_per_band = jnp.einsum('bp,pkc->bkc', lfac_aod, aod_550)
    ssa_sum_b = jnp.einsum('bp,pkc->bkc', lfac_aod * ssa_b, aod_550)
    asy_sum_b = jnp.einsum('bp,pkc->bkc', lfac_aod * ssa_b * asy_b, aod_550)
    ssa_b_has = ssa_sum_b > tiny
    aod_b_has = aod_sw_per_band > tiny
    asy_sw_per_band = jnp.where(
        ssa_b_has, asy_sum_b / jnp.where(ssa_b_has, ssa_sum_b, 1.0), 0.0,
    )
    ssa_sw_per_band = jnp.where(
        aod_b_has,
        ssa_sum_b / jnp.where(aod_b_has, aod_sw_per_band, 1.0), 1.0,
    )

    # Column Angstrom diagnostic for hosts that band-scale a column AOD
    # themselves (grey two-stream): plume-AOD-weighted mean, arbitrary
    # (zero) where there is no plume AOD to scale.
    ang_sum = jnp.einsum('p,pkc->c', parameters.angstrom, aod_550)
    caod_has = caod_sp > tiny
    angstrom = jnp.where(
        caod_has, ang_sum / jnp.where(caod_has, caod_sp, 1.0), 0.0,
    )

    # Twomey factor (Stevens et al. 2017 dNovrN) and the absolute-CCN
    # activation floor for the SPA path.
    cdnc_factor = get_dNovrN(caod_sp, caod_bg)
    Nccn = get_CDNC(caod_sp + caod_bg)

    return aerosol_data.copy(
        aod_profile=aod_profile,
        ssa_profile=ssa_profile,
        asy_profile=asy_profile,
        # Column AOD is the anthropogenic caod_sp: the natural background
        # is not vertically distributed in the reference and is excluded
        # from all radiative fields by construction.
        aod_total=caod_sp,
        aod_anthropogenic=caod_sp,
        aod_background=caod_bg,
        cdnc_factor=cdnc_factor,
        Nccn=Nccn,
        angstrom=angstrom,
        aod_sw_per_band=aod_sw_per_band,
        ssa_sw_per_band=ssa_sw_per_band,
        asy_sw_per_band=asy_sw_per_band,
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


def get_plume_column_weights(parameters, year_weight, ann_cycle, gauss):
    """Per-plume anthropogenic / background column weights (Fortran cw_an, cw_bg).

    Implements mo_simple_plumes_v1.f90 lines 330-336: each feature's own
    time weight multiplies that feature's own Gaussian BEFORE the feature
    sum — collapsing either axis first is exact only when the features
    share an annual cycle, which the biomass plumes deliberately don't.

    Args:
        parameters: AerosolParameters.
        year_weight: ``(nplumes,)`` CEDS amplitude relative to 2005 for
            the current calendar year (anthropogenic only — the fine-mode
            background gets the annual cycle without the year weight).
        ann_cycle: ``(nfeatures, nplumes)`` weekly-cycle weights for the
            current date.
        gauss: ``(nfeatures, nplumes, ncols)`` per-feature Gaussians from
            :func:`_per_feature_plume_gaussians`.

    Returns:
        Tuple ``(cw_an, cw_bg)``, each ``(nplumes, ncols)`` — already
        scaled by ``aod_spmx`` / ``aod_fmbg``.

    """
    fw = parameters.ftr_weight                            # (nf, np)
    tw_an = year_weight[jnp.newaxis, :] * ann_cycle       # (nf, np)
    feat_sum_an = jnp.einsum('fp,fpc->pc', tw_an * fw, gauss)
    feat_sum_bg = jnp.einsum('fp,fpc->pc', ann_cycle * fw, gauss)
    cw_an = parameters.aod_spmx[:, jnp.newaxis] * feat_sum_an
    cw_bg = parameters.aod_fmbg[:, jnp.newaxis] * feat_sum_bg
    return cw_an, cw_bg


def get_vertical_profiles(height_full, layer_thickness, orography, parameters):
    """dz-weighted, orography-truncated beta vertical profiles per plume.

    Implements mo_simple_plumes_v1.f90 lines 273-274 and 289-302: the
    beta *density* is multiplied by the layer depth dz before the level
    normalization (so the per-layer AOD split is resolution-independent),
    and levels below the surface are zeroed AFTER the division — over
    elevated terrain the column sum is < 1 and the below-ground AOD is
    removed, not redistributed.

    Args:
        height_full: Layer-centre height above sea level (m),
            ``(nlev, ncols)``.
        layer_thickness: Layer depth dz (m), ``(nlev, ncols)``.
        orography: Surface height above sea level (m), ``(ncols,)``.
        parameters: AerosolParameters.

    Returns:
        Profiles ``(nplumes, nlev, ncols)`` with level sums in [0, 1].

    """
    # eta = z/15km clipped to [0,1] (Fortran line 274); the interior clip
    # keeps the beta density finite for shape parameters < 1 at eta=0/1.
    height_norm = jnp.clip(height_full / 15000.0, 0.0, 1.0)

    eps = 1e-10
    x = jnp.clip(height_norm[jnp.newaxis, :, :], eps, 1.0 - eps)
    beta_a = parameters.beta_a[:, jnp.newaxis, jnp.newaxis]
    beta_b = parameters.beta_b[:, jnp.newaxis, jnp.newaxis]

    beta_profile = (
        x ** (beta_a - 1) * (1 - x) ** (beta_b - 1)
        * layer_thickness[jnp.newaxis, :, :]
    )

    profile_sum = jnp.sum(beta_profile, axis=1, keepdims=True)
    profile_sum = jnp.where(profile_sum > 0, profile_sum, 1.0)

    # Orography mask applied after normalization (Fortran line 300).
    z_beta = jnp.where(
        height_full >= orography[jnp.newaxis, :], 1.0, 0.0,
    )                                                  # (nlev, ncols)

    return (beta_profile / profile_sum) * z_beta[jnp.newaxis, :, :]


def per_band_optical_properties(
    aod_550: jnp.ndarray,
    ssa550: jnp.ndarray,
    asy550: jnp.ndarray,
    angstrom: jnp.ndarray,
    band_centers_nm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Wavelength-dependent aerosol optical properties (Stevens 2017).

    Implements the closed-form scaling from MACv2-SP
    (``mo_bc_aeropt_splumes.f90:378-397``):

        lfactor   = min(1, 700/λ_nm)
        ssa(λ)    = ssa550·lfactor⁴ / [ssa550·lfactor⁴ + (1−ssa550)·lfactor]
        asy(λ)    = asy550·√lfactor
        aod(λ)    = aod550·exp(−angstrom·ln(λ/550))

    The ``ssa`` formula is one minus the absorption-to-extinction ratio
    derived from a Mie-style λ⁻¹ absorption and λ⁻⁴ scattering scaling
    of the 550 nm reference; ``asy`` rescales the asymmetry parameter
    by Hänel-style refractive-index dependence; ``aod`` uses the per-
    plume Angstrom exponent. All three are independent per band.

    Args:
        aod_550: 550 nm AOD profile, shape ``(..., nlev, ncols)`` or
            anything broadcastable against the band axis.
        ssa550: 550 nm single-scattering albedo, broadcastable shape.
        asy550: 550 nm asymmetry parameter, broadcastable shape.
        angstrom: Angstrom exponent, broadcastable shape.
        band_centers_nm: Band-center wavelengths in nm, shape ``(n_bnd,)``.

    Returns:
        Tuple ``(aod_per_band, ssa_per_band, asy_per_band)`` each shaped
        ``(n_bnd, ...)`` with the band axis prepended.

    """
    # Cast to a safe shape for broadcasting against (n_bnd, ...).
    lam = band_centers_nm[:, None, None]   # (n_bnd, 1, 1)
    lfactor = jnp.minimum(1.0, 700.0 / lam)

    ssa_num = ssa550 * lfactor ** 4
    ssa_den = ssa_num + (1.0 - ssa550) * lfactor
    ssa_per_band = ssa_num / jnp.maximum(ssa_den, 1e-30)

    asy_per_band = asy550 * jnp.sqrt(lfactor)

    aod_per_band = aod_550 * jnp.exp(-angstrom * jnp.log(lam / 550.0))

    return aod_per_band, ssa_per_band, asy_per_band


def get_CDNC(AOD, A=60, B=20):
    """Derive CDNC from AOD using a relationship of the form: CDNC = A * ln(B*AOD + 1)
    Ross' amazon work: A=410 B=5
    MODIS original: A=16 B=1000
    AEROCOM P1 original: A=60, B=20
    """
    return 1 + A * jnp.log(B * AOD + 1)


def get_dNovrN(aod_anthropogenic, aod_background):
    """Stevens et al. (2017) relative CDNC enhancement (Twomey factor).

    ``sp_aop_dNovrN`` from mo_simple_plumes.f90:

        dNovrN = ln(1000·(caod_sp + caod_bg) + 1) / ln(1000·caod_bg + 1)

    with ``caod_sp`` the anthropogenic plume AOD at 550 nm and ``caod_bg``
    the natural background AOD (the background plume contribution plus the
    0.02 fine-mode constant, exactly the Fortran's caod_bg accumulation).
    Dimensionless, 1.0 with no anthropogenic aerosol, and typically
    1.0–1.6 (≈2 at the East-Asia plume maximum) — it multiplies a baseline
    droplet number, unlike the ABSOLUTE AEROCOM CDNC from
    :func:`get_CDNC`.
    """
    return (
        jnp.log(1000.0 * (aod_anthropogenic + aod_background) + 1.0)
        / jnp.log(1000.0 * aod_background + 1.0)
    )


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
    requires: ClassVar[tuple[str, ...]] = ("height_full", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = ("aerosol",)
    # Carry seeded as zeros; ``get_simple_aerosol`` rebuilds
    # AOD/SSA/asymmetry from the plume parameterisation every step
    # using the slot only as a shape source.
    carry_slots: ClassVar[dict[str, type]] = {"aerosol": AerosolData}

    def __init__(self, params: AerosolParameters | None = None):
        """Hold the scheme-native :class:`AerosolParameters`."""
        self.params = nnx.Param(params or AerosolParameters.default())
        self._coords_cached = False
        # SW band count — overridden by ``cache_band_config`` once
        # ``ComposablePhysics`` knows the active radiation backend.
        # Default 14 covers the standard RRTMGP SW gas-optics file so
        # standalone construction (no ComposablePhysics) still works.
        self._n_bnd_sw: int = 14
        # LW band count for the carry slot's LW optics (zero for MACv2-SP,
        # populated by the JAM optics term, #495). 16 covers the standard
        # RRTMGP LW gas-optics file for standalone construction.
        self._n_bnd_lw: int = 16

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (degrees) from the coordinate system.

        Uses the same lat/lon meshgrid → ``ncols`` reshape that the
        legacy ECHAM wrapper performed inline; doing it once here at
        construction time avoids repeating the ``meshgrid`` inside the
        jitted compute_tendencies loop.
        """
        # column_lat_lon reproduces get_simple_aerosol's legacy
        # meshgrid(lat, lon) -> (ncols,) convention on separable grids
        # (longitude varying fastest) and returns true per-column pairs on
        # scattered-column grids (pySES SE).
        lat, lon = column_lat_lon(coords.horizontal)
        self._lats = nnx.Variable(lat * 180.0 / jnp.pi)
        self._lons = nnx.Variable(lon * 180.0 / jnp.pi)
        self._coords_cached = True

    def cache_band_config(self, band_config) -> None:
        """Capture SW band count so the carry slot has the right shape.

        Sized to match the active radiation backend (e.g. 14 for the
        standard RRTMGP SW gas-optics file, 1 for grey/SPEEDY) so the
        per-band ``aod/ssa/asy_sw_per_band`` arrays in the cross-step
        ``aerosol`` carry agree with what ``__call__`` writes back.
        """
        self._n_bnd_sw = len(band_config.sw_band_centers_nm)
        self._n_bnd_lw = len(band_config.lw_band_centers_nm)

    def initial_carry_state(self, coords):
        """Zero-fill the aerosol carry slot at the active SW/LW band counts."""
        ncols = (
            coords.horizontal.nodal_shape[0]
            * coords.horizontal.nodal_shape[1]
        )
        nlev = coords.nodal_shape[0]
        return {
            "aerosol": AerosolData.zeros(
                (ncols,), nlev,
                n_bnd_sw=self._n_bnd_sw, n_bnd_lw=self._n_bnd_lw,
            )
        }

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

        # ``ComposablePhysics`` injects ``_band_config`` each step (a
        # ``RadiationBandConfig`` matching the active radiation backend).
        # Falls back to grey-radiation broadband defaults if a caller
        # constructs the term outside ``ComposablePhysics``.
        band_config = diagnostics.get("_band_config")
        if band_config is None:
            from jcm.physics.radiation.band_config import RadiationBandConfig
            band_config = RadiationBandConfig.broadband()
        sw_band_centers_nm = jnp.asarray(
            band_config.sw_band_centers_nm, dtype=jnp.float32,
        )
        n_bnd_sw = sw_band_centers_nm.shape[0]
        n_bnd_lw = len(band_config.lw_band_centers_nm)

        prev = diagnostics.get(
            "aerosol",
            AerosolData.zeros(
                (ncols,), nlev, n_bnd_sw=n_bnd_sw, n_bnd_lw=n_bnd_lw,
            ),
        )
        new_aerosol = get_simple_aerosol(
            height_full=diagnostics["height_full"],
            layer_thickness=diagnostics["layer_thickness"],
            # Mean orography (m above sea level) truncates the plume
            # profiles; zero on aquaplanets, where the mask is a no-op.
            orography=terrain.orog.reshape(-1),
            lats_deg=self._lats.get_value(),
            lons_deg=self._lons.get_value(),
            aerosol_data=prev,
            parameters=params,
            forcing=forcing,
            sw_band_centers_nm=sw_band_centers_nm,
        )

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {**diagnostics, "aerosol": new_aerosol}
