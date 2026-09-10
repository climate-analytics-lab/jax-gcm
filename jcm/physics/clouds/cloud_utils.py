"""Contains utility routines and constants related to the 2-m cloud microphysics scheme. Based on mo_cloud_utils from ECHAM6/ICON.

Date: 2025-12-15
"""

import jax.numpy as jnp
from math import pi

import jcm.constants as c
from .lohmann_2m_params import CloudParams2M

def eff_ice_crystal_radius(
    pxice: jnp.ndarray, picnc: jnp.ndarray, params: CloudParams2M,
) -> jnp.ndarray:
    """Effective ice crystal radius following Lohmann et al. (2008, ERL), expression (1),
    using the Pruppacher & Klett (1997) mass–size relation parameters.

    Parameters
    ----------
    pxice : jnp.ndarray
        In-cloud ice mass concentration [g/m^3].
    picnc : jnp.ndarray
        Ice crystal number concentration (ICNC) [1/m^3].

    Returns
    -------
    prieff : jnp.ndarray
        Effective ice crystal radius [micron].

    """
    eps = params.eps
    # Double-where guard: the base is 0 in an ice-free cell (pxice == 0, the
    # common case) and ``0 ** (1/pow_PK)`` (a fractional power) has an infinite
    # derivative, poisoning the reverse pass while the forward is 0 (issue
    # #558). Keep the forward exactly 0 where there is no ice; differentiate
    # the power only on the strictly-positive floored base.
    base = pxice / jnp.maximum(params.fact_PK * jnp.maximum(picnc, eps), eps)
    # The ``where`` already returns 0 for ice-free cells; the base floor only
    # needs to keep the differentiated branch strictly positive, so it uses the
    # negligible ``d_epsilon`` (NOT ``eps`` ≈ 1e-7, which would inflate the
    # effective radius of small-but-nonzero ice — see the ``eps``/``d_epsilon``
    # note on CloudParams2M). Issue #558.
    return 0.5e4 * jnp.where(
        pxice > 0.0,
        jnp.maximum(base, params.d_epsilon) ** (1.0 / params.pow_PK),
        0.0,
    )

def ice_volume_mean_radius(
    ice_in_cloud_gm3: jnp.ndarray, icnc: jnp.ndarray, params: CloudParams2M,
) -> jnp.ndarray:
    """Volume-mean ice crystal radius (Fortran ``prid``/``zrice``/``zris``) in METRES.

    Chains the Lohmann (2008) effective radius, the ``[ceffmin, ceffmax]`` clip,
    and the Schumann (2011) effective -> volume-mean conversion
    ``zrih = -2261 + sqrt(5113188 + 2809 r_eff^3)``, ``r_vol = 1e-6 zrih^(1/3)``.

    Metres is load-bearing: callers invert this as
    ``N = rho q_i / ((4/3) pi r_vol^3 rho_ice)``, so returning the microns that
    ``eff_ice_crystal_radius`` produces understates crystal number by ~1e18 and
    pins ICNC at ``icemin``, saturating the ice effective radius at ``ceffmax``
    (#725).

    Parameters
    ----------
    ice_in_cloud_gm3 : jnp.ndarray
        IN-CLOUD ice mass concentration [g/m^3] — grid-mean divided by cover.
    icnc : jnp.ndarray
        Ice crystal number concentration [1/m^3].

    """
    r_eff_um = jnp.clip(
        eff_ice_crystal_radius(ice_in_cloud_gm3, icnc, params),
        params.ceffmin,
        params.ceffmax,
    )
    zrih = -2261.0 + jnp.sqrt(5113188.0 + 2809.0 * r_eff_um**3)
    # Floor guards the cube root, whose derivative is infinite at 0. The clip
    # above keeps r_eff >= ceffmin, so zrih >= ~550 and the floor never binds
    # in the forward pass.
    return 1.0e-6 * jnp.maximum(zrih, params.eps) ** (1.0 / 3.0)

def minimum_CDNC(pxwat, params: CloudParams2M):
    """Set the minimum cloud droplet number concentration, either statically or dynamically.

    Parameters
    ----------
        pxwat (array): In-cloud water mixing ratio [kg/m^3].
        params (CloudParams2M): Threaded scheme parameters; the
            ``ldyn_cdnc_min`` static switch selects the dynamic branch.

    Returns
    -------
        pcdnc_min (array): Minimum cloud droplet number concentration [m^-3].

    """
    if params.ldyn_cdnc_min:
        # Dynamic value for minimum CDNC
        pcdnc_min = params.rcd_vol_max**(-3.0) * (3.0 / (4.0 * pi * c.rhow)) * pxwat
        pcdnc_min = jnp.clip(pcdnc_min, params.cdnc_min_lower, params.cdnc_min_upper)
    else:
        # Static minimum CDNC
        pcdnc_min = params.cdnc_min_fixed * 1.0e6  # Convert from cm^-3 to m^-3
        pcdnc_min = jnp.broadcast_to(pcdnc_min, pxwat.shape).astype(pxwat.dtype)

    return pcdnc_min

def gridbox_frac_falling_hydrometeor(
    precip_flux_from_above: jnp.ndarray,
    precip_frac_from_above: jnp.ndarray,
    precip_flux_from_level: jnp.ndarray,
    precip_frac_from_level: jnp.ndarray,
    params: CloudParams2M,
) -> jnp.ndarray:
    """Compute the grid box fraction covered by falling hydrometeor (e.g., rain+snow, sedimenting ice).

    Parameters
    ----------
    precip_flux_from_above : jnp.ndarray
        Flux of falling hydrometeor from above.
    precip_frac_from_above : jnp.ndarray
        Fraction of gridbox covered by falling hydrometeor from above.
    precip_flux_from_level : jnp.ndarray
        Flux of falling hydrometeor from the current level.
    precip_frac_from_level : jnp.ndarray
        Fraction of gridbox covered by falling hydrometeor from the current level.
    min_precip_flux : float
        Minimum threshold for total flux.

    Returns
    -------
    jnp.ndarray
        Total fraction of gridbox covered by falling hydrometeor.

    """
    # Determine where flux from above is greater than flux from the current level
    ll1 = precip_flux_from_above > precip_flux_from_level

    # Update fraction from above based on condition
    updated_precip_frac_from_above = jnp.where(
        ll1, precip_frac_from_above, precip_frac_from_level
    )

    # Compute total flux
    total_precip_flux = precip_flux_from_above + precip_flux_from_level

    # Determine where total flux is greater than the minimum threshold
    # The guard threshold is a PHYSICAL minimum flux (1e-9 kg/m2/s is well
    # under a mm per year), not cqtmin = 1e-12: the division VJP forms
    # -g*x/(flux*flux), and for fluxes between 1e-12 and ~1e-6 the selected
    # branch's derivative reaches 1e12-1e24 per call. Those cotangent spikes
    # compound through the precip-fraction carry and are one of the
    # ice-regime adjoint amplifiers that break long-window reverse mode.
    _min_flux = 1.0e-9
    ll1 = total_precip_flux > _min_flux

    # Compute weighted average fraction
    weighted_precip_frac = (
        (precip_frac_from_level * precip_flux_from_level + updated_precip_frac_from_above * precip_flux_from_above)
        / jnp.maximum(total_precip_flux, _min_flux)
    )
    weighted_precip_frac = jnp.clip(weighted_precip_frac, 0.0, 1.0)

    # Compute total fraction
    total_precip_frac = jnp.where(ll1, weighted_precip_frac, 0.0)

    return total_precip_frac

def effective_2_volmean_radius_param_Schuman_2011(
    prieff: jnp.ndarray, params: CloudParams2M,
) -> jnp.ndarray:
    """Convert effective radius to volume-mean radius using Schumann et al. (2011) parametrisation.

    Parameters
    ----------
    prieff : jnp.ndarray
        Effective ice crystal radius (Fortran: prieff) given in units of 1.e-6 m (i.e. microns).

    Returns
    -------
    prvolmean : jnp.ndarray
        Volume-mean ice crystal radius (Fortran: prvolmean) in metres.

    Notes
    -----
    Fortran implementation:
        prvolmean = MAX(1.e-6_dp, conv_effr2mvr*1.e-6_dp*prieff)
    where conv_effr2mvr (imported) is the scheme constant converting effective -> vol-mean radius.

    """
    # Multiply prieff (1e-6 m units) by 1e-6 to get metres, apply conv_effr2mvr and enforce minimum 1e-6 m.
    return jnp.maximum(1e-6, params.conv_effr2mvr * 1e-6 * prieff)

def breadth_factor(pcdnc: jnp.ndarray) -> jnp.ndarray:
    """Breadth factor as a function of cloud droplet number concentration (CDNC).

    Parameters
    ----------
    pcdnc : jnp.ndarray
        Cloud droplet number concentration (Fortran: pcdnc) [1/m^3].

    Returns
    -------
    pkap : jnp.ndarray
        Breadth factor (Fortran: pkap). Parametrisation from Peng & Lohmann (2003), eq. 6:
            pkap = 0.00045e-6 * pcdnc + 1.18
        The constant 0.00045e-6 is equal to 4.5e-10.

    """
    return 4.5e-10 * pcdnc + 1.18

def eff_liquid_droplet_radius(
    liquid_in_cloud: jnp.ndarray,
    air_density: jnp.ndarray,
    cdnc: jnp.ndarray,
    eps: float | jnp.ndarray,
    liquid_cloud_flag: jnp.ndarray | bool = True,
) -> jnp.ndarray:
    """Effective cloud droplet radius (ECHAM ``preffl``), shared by the 1M and 2M schemes.

    ``r_eff = 1e6 * kappa * (3 * rho * q_l,in-cloud / (4 pi rho_w N))^(1/3)``
    with the Peng & Lohmann (2003) breadth factor ``kappa(N)``.

    Parameters
    ----------
    liquid_in_cloud : jnp.ndarray
        In-cloud liquid water mixing ratio (Fortran: pxlb) [kg/kg].
    air_density : jnp.ndarray
        Air density (Fortran: prho) [kg/m^3].
    cdnc : jnp.ndarray
        Cloud droplet number concentration (Fortran: pcdnc) [1/m^3].
    eps : float or jnp.ndarray
        Floor on the CDNC denominator.
    liquid_cloud_flag : jnp.ndarray or bool
        Additional liquid-cloud mask (Fortran: ld_liqcl); ``True`` applies none.

    Returns
    -------
    jnp.ndarray
        Effective droplet radius [micron], EXACTLY 0 where there is no liquid —
        radiation (``cloud_optics.resolve_effective_radii``) selects on
        ``r_eff > 0``, so the zero is what routes a cell to the fallback radius.

    """
    breadth = breadth_factor(cdnc)
    # Double-where guard on the cube root, whose derivative is infinite when the
    # base is 0. The mask must be "there is liquid to speak of", NOT
    # ``liquid_cloud_flag`` alone: in the 2M scheme that flag is
    # ``temperature > tmelt``, so it is True in every warm cell, including the
    # cloud-free majority where ``liquid_in_cloud == 0`` puts a 0 on the
    # *differentiated* branch. The forward is unchanged either way (the radius is
    # masked to 0 there), but the reverse pass multiplies that infinite local
    # derivative by the incoming cotangent, and a zero cotangent gives
    # 0 * inf = NaN. That NaN reaches the gradient only once radiation consumes
    # these radii from the cloud carry, i.e. from the second step of a rollout
    # onwards.
    has_liquid = jnp.logical_and(liquid_cloud_flag, liquid_in_cloud > 0.0)
    radius_base = jnp.where(
        has_liquid,
        (3.0 / (4.0 * pi * c.rhow)) * liquid_in_cloud * air_density / jnp.maximum(cdnc, eps),
        1.0,
    )
    liq_eff_radius = 1.0e6 * breadth * radius_base ** (1.0 / 3.0)
    return jnp.where(has_liquid, liq_eff_radius, 0.0)

def threshold_vert_vel(
    sat_vap_pres_water: jnp.ndarray,  # pesw [Pa]
    sat_vap_pres_ice: jnp.ndarray,    # pesi [Pa]
    icnc: jnp.ndarray,                # picnc [1/m^3]
    ice_radius: jnp.ndarray,          # price [m] volume-mean ice crystal radius
    eta: jnp.ndarray,                 # peta [-]
    params: CloudParams2M,
) -> jnp.ndarray:
    """Threshold vertical velocity for the Wegener-Bergeron-Findeisen (WBF) criterion.

    JAX port of Fortran function `threshold_vert_vel_1d` (mo_cloud_microphysics_2m).

    The WBF process (ice growth at the expense of supercooled liquid) is active when
    the actual updraft velocity is below this threshold. The threshold is proportional
    to the supersaturation of water vapour over ice, the ice crystal number concentration,
    the crystal size, and a diffusivity-related factor `eta`.

    Parameters
    ----------
    sat_vap_pres_water : array
        Saturation vapour pressure w.r.t. liquid water `pesw` [Pa].
    sat_vap_pres_ice : array
        Saturation vapour pressure w.r.t. ice `pesi` [Pa].
    icnc : array
        Ice crystal number concentration `picnc` [1/m^3].
    ice_radius : array
        Volume-mean ice crystal radius `price` [m].
    eta : array
        Diffusivity-related variable for the WBF criterion `peta` [-].

    Returns
    -------
    pvervmax : array
        Threshold vertical velocity [m/s] (same units as `pvervx` in the calling routine,
        which is compared after scaling by 0.01 from cm/s).

    """
    return (
        (sat_vap_pres_water - sat_vap_pres_ice)
        / jnp.maximum(sat_vap_pres_ice, params.eps)
        * icnc
        * ice_radius
        * eta
    )

def consistency_number_to_mass(
    pthreshold: float | jnp.ndarray,
    pmass: jnp.ndarray,
    pnumber: jnp.ndarray,
    ) -> jnp.ndarray:
    """Return a "physical" number concentration/flux: whenever the corresponding mass
    is below `pthreshold`, the number is reset to 0.

    Parameters
    ----------
    pthreshold : float or jnp.ndarray
        Threshold below which `pnumber` is forced to zero.
    pmass : jnp.ndarray
        Mass-like quantity (e.g. ice flux mass) [units arbitrary].
    pnumber : jnp.ndarray
        Number-like quantity associated with `pmass`.

    Returns
    -------
    jnp.ndarray
        `pnumber` with entries zeroed where `pmass < pthreshold`.

    """
    return jnp.where(pmass < pthreshold, 0.0, pnumber)
