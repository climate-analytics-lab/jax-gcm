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
    return 0.5e4 * jnp.where(
        pxice > 0.0,
        jnp.maximum(base, eps) ** (1.0 / params.pow_PK),
        0.0,
    )

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
    ll1 = total_precip_flux > params.cqtmin

    # Compute weighted average fraction
    weighted_precip_frac = (
        (precip_frac_from_level * precip_flux_from_level + updated_precip_frac_from_above * precip_flux_from_above)
        / jnp.maximum(total_precip_flux, params.cqtmin)
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
