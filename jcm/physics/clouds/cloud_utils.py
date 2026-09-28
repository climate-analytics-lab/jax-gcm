"""Utility routines and constants for the 2-m cloud microphysics scheme (based on mo_cloud_utils from ECHAM6/ICON)."""

import jax.numpy as jnp
from math import pi

import jcm.constants as c
from .lohmann_2m_params import CloudParams2M


def moist_isobaric_heat_capacity(specific_humidity: jnp.ndarray) -> jnp.ndarray:
    """Isobaric specific heat of moist air ``cp = cpd·(1 + vtmpc2·q)`` [J/kg/K].

    Expanded to ``cpd + (cpv - cpd)·max(q, 0)`` because ``cpd·vtmpc2 = cpv - cpd``
    (``vtmpc2 = cpv/cpd - 1``). This is ECHAM's ``pcair`` /
    ``zcair = cpd + cpd·vtmpc2·max(qm1, 0)`` (physc.f90:289, with
    ``zcons1 = cpd·vtmpc2``, mo_cloud_micro_2m.f90:534) — the humidity-weighted
    heat capacity the cloud latent-heat conversions divide by, evaluated at the
    step-start humidity. The negative-humidity clamp mirrors the Fortran
    ``MAX(pqm1, 0)`` and keeps ``cp`` physical against spectral-ringing
    undershoots.
    """
    return c.cpd + (c.cpv - c.cpd) * jnp.maximum(specific_humidity, 0.0)


def latent_heat_over_cp(
    specific_humidity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Moist ``(Lv/cp, Ls/cp)`` latent-heat-to-heat-capacity ratios [K/(kg/kg)].

    The ECHAM cloud ledgers convert a condensed/evaporated mixing-ratio
    increment to a temperature increment with ``L / cp`` where ``cp`` is the
    MOIST heat capacity, not dry ``cpd`` — ``zlvdcp = alv/pcair``,
    ``zlsdcp = als/pcair`` (mo_cloud.f90:412-414,
    mo_cloud_micro_2m.f90:844-848). Building them from dry ``cpd`` over-heats
    every condensation event by ``vtmpc2·q`` (~1.3 % in the moist tropics);
    both cloud ports now share this construction (#706).
    """
    inv_cp = 1.0 / moist_isobaric_heat_capacity(specific_humidity)
    return c.alhc * inv_cp, c.alhs * inv_cp


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
    """Plate volume-mean ice crystal radius (Fortran ``zris``) in METRES.

    Chains the Lohmann (2008) effective radius, the ``[ceffmin, ceffmax]`` clip,
    and the plate relation ``zrih = -2261 + sqrt(5113188 + 2809 r_eff^3)``,
    ``r_vol = 1e-6 zrih^(1/3)`` that ECHAM uses for the aggregation timescale
    in ``precip_formation_cold`` (``mo_cloud_micro_2m.f90:3160-3166``; the 1M
    Levkov aggregation in ``mo_cloud.f90:1031-1036`` uses the same relation).
    ECHAM's Wegener-Bergeron-Findeisen threshold uses a different conversion,
    :func:`ice_volume_mean_radius_schumann`. jcm also passes this radius to
    ``update_in_cloud_water`` as ``prid`` for the ICNC diagnosis, where ECHAM
    passes its temperature-parameterised ``zrid`` (lines 945-956; #941).

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

def ice_volume_mean_radius_schumann(
    ice_in_cloud_gm3: jnp.ndarray, icnc: jnp.ndarray, params: CloudParams2M,
) -> jnp.ndarray:
    """Volume-mean ice crystal radius for the WBF threshold (ECHAM ``zrice``) in METRES.

    Chains the Lohmann (2008) effective radius, the ``[ceffmin, ceffmax]`` clip
    and ECHAM's ``effective_2_volmean_radius_param_Schuman_2011``,
    ``r_vol = max(1e-6, conv_effr2mvr·1e-6·r_eff)`` with ``conv_effr2mvr = 0.9``
    (``mo_cloud_micro_2m.f90:4059-4085``, a simple fit to the Schumann et al.
    2011 r/r_eff data). This is the radius ECHAM hands to
    ``threshold_vert_vel`` at every Wegener-Bergeron-Findeisen decision. jcm
    uses it at the three it ports: the section-4 phase choice ``lo2``
    (line 1288), the section-5 supersaturation correction
    (``mixed_phase_deposition_and_corrections``, line 2374) and the WBF gate
    (line 1582). ECHAM's fourth, the phase split of convective detrainment
    ``lo2_2d`` (lines 872-885), has no counterpart: jcm's Tiedtke scheme
    splits detrained condensate at ``tmelt`` (#941). The plate relation of
    :func:`ice_volume_mean_radius` is ECHAM's for aggregation only.

    Parameters
    ----------
    ice_in_cloud_gm3 : jnp.ndarray
        IN-CLOUD ice mass concentration [g/m^3].
    icnc : jnp.ndarray
        Ice crystal number concentration [1/m^3].

    """
    r_eff_um = jnp.clip(
        eff_ice_crystal_radius(ice_in_cloud_gm3, icnc, params),
        params.ceffmin,
        params.ceffmax,
    )
    return effective_2_volmean_radius_param_Schuman_2011(r_eff_um, params)

def turbulent_updraft_velocity(
    tke: jnp.ndarray, params: CloudParams2M,
) -> jnp.ndarray:
    """Turbulent part of ECHAM's cloud-scheme updraft ``zvervx`` [cm/s].

    ``100·fact_tke·sqrt(TKE)`` with ``fact_tke = 0.7``, set to zero at the
    lowest model level (``mo_cloud_micro_2m.f90:814-815``). ECHAM's
    ``zvervx`` (line 816) adds the large-scale term ``−100·ω/(g·ρ)``; that
    one is the caller's to add (it needs the pressure velocity).

    Parameters
    ----------
    tke : jnp.ndarray
        Turbulent kinetic energy [m²/s²], vertical on axis 0 (top first, so
        the last index is the lowest level) and any horizontal axes after it.

    """
    nlev = tke.shape[0]
    is_lowest_level = (jnp.arange(nlev) == nlev - 1).reshape(
        (nlev,) + (1,) * (tke.ndim - 1))
    # Double-where on the root: at TKE = 0 (laminar layers, a cold start)
    # sqrt has an infinite derivative.
    positive_tke = tke > 0.0
    turbulent = 100.0 * params.fact_tke * jnp.where(
        positive_tke, jnp.sqrt(jnp.where(positive_tke, tke, 1.0)), 0.0)
    return jnp.where(is_lowest_level, 0.0, turbulent)

def air_dynamic_viscosity(temperature: jnp.ndarray) -> jnp.ndarray:
    """Dynamic viscosity of air [kg m^-1 s^-1] (ECHAM ``pviscos``).

    ``pviscos = (1.512 + 0.0052·(T − 233.15))·1e-5``, a linear fit in
    temperature (``mo_cloud_utils.f90::get_util_var``, line 132), evaluated at
    the step-start temperature ``ptm1``. The 2M scheme uses it only in the snow
    Reynolds number of riming (``precip_formation_cold``,
    ``mo_cloud_micro_2m.f90:3216``). Not to be confused with the thermal
    conductivity of air ``zkair = 4.1867e-3·(5.69 + 0.017·(T − tmelt))``
    (line 715), which enters the diffusional-growth factors instead.
    """
    return (1.512 + 0.0052 * (temperature - 233.15)) * 1.0e-5

def ice_fall_speed_air_density_factor(
    pressure: jnp.ndarray, temperature: jnp.ndarray,
) -> jnp.ndarray:
    """Air-density correction of the cloud-ice fall speed (ECHAM ``paaa``), dimensionless.

    ``paaa = (p/30000)^(-0.178)·(T/233)^(-0.394)``
    (``mo_cloud_utils.f90::get_util_var``, line 129), the Heymsfield & Iaquinta
    (2000, *J. Atmos. Sci.* 57, 916-938) pressure and temperature correction of
    the crystal fall speed, equal to 1 at 300 hPa and 233 K. Evaluated at the
    full-level pressure and step-start temperature. The 2M scheme uses it
    only in the ice sedimentation fall speed ``zxifallmc = fall·α·m^β·paaa``
    (``mo_cloud_micro_2m.f90:2224``), which moves ice mass and number alike.

    Parameters
    ----------
    pressure : jnp.ndarray
        Full-level pressure ``papm1`` [Pa].
    temperature : jnp.ndarray
        Temperature ``ptm1`` [K].

    """
    return (pressure / 30000.0) ** (-0.178) * (temperature / 233.0) ** (-0.394)

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
    radius_base = (
        (3.0 / (4.0 * pi * c.rhow)) * liquid_in_cloud * air_density
        / jnp.maximum(cdnc, eps)
    )
    # Positive liquid can still underflow to a zero base in the arithmetic
    # above. Guard the computed base too: preserve its zero forward radius,
    # but never differentiate the cube root at zero. Use != 0 rather than
    # > 0 so invalid negative/NaN bases remain visible, not silently masked.
    has_liquid = jnp.logical_and(has_liquid, radius_base != 0.0)
    radius_base = jnp.where(has_liquid, radius_base, 1.0)
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
