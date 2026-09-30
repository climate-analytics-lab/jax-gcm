"""Utility routines and constants for the 2-m cloud microphysics scheme (based on mo_cloud_utils from ECHAM6/ICON)."""

import math
from math import pi

import jax.numpy as jnp

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
    :func:`ice_volume_mean_radius_schumann`, and the ICNC diagnosis in
    ``update_in_cloud_water`` inverts the temperature-parameterised radius of
    :func:`ice_volume_mean_radius_from_temperature` (ECHAM ``zrid``).

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
    ``threshold_vert_vel`` at every Wegener-Bergeron-Findeisen decision, and
    jcm uses it at all four: the section-1 criterion ``lo2_2d`` that gates the
    crystal number of detrained ice (lines 872-885), the section-4 phase choice
    ``lo2`` that also re-splits the detrained condensate (line 1288), the
    section-5 supersaturation correction
    (``mixed_phase_deposition_and_corrections``, line 2374) and the WBF gate
    (line 1582). The plate relation of :func:`ice_volume_mean_radius` is
    ECHAM's for aggregation only.

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

def ice_volume_mean_radius_from_temperature(
    temperature: jnp.ndarray, params: CloudParams2M,
) -> jnp.ndarray:
    """Temperature-parameterised volume-mean ice crystal radius (ECHAM ``zrid``) in METRES.

    ``r_eff = max(23.2·exp(0.015·min(T − tmelt, 0)), 1)`` micrometres, turned
    into a volume-mean radius by ECHAM's
    ``effective_2_volmean_radius_param_Schuman_2011``,
    ``max(1e-6, conv_effr2mvr·1e-6·r_eff)`` (``mo_cloud_micro_2m.f90:945-956``).
    ECHAM evaluates it at the step-start temperature ``ptm1`` and uses it for
    two things: the crystal number of detrained ice ``znidetr`` (line 970,
    :func:`detrained_ice_crystal_number`) and the radius ``prid`` that
    ``update_in_cloud_water`` inverts to diagnose ICNC from ice mass
    (passed at line 1511, used at 2616).

    The Schumann helper takes micrometres and returns metres ("beware of
    units", lines 4066-4067): every consumer of this radius expects metres.

    Parameters
    ----------
    temperature : jnp.ndarray
        Step-start temperature [K] (ECHAM ``ptm1``).

    """
    delta_t = jnp.minimum(temperature - params.tmelt, 0.0)
    r_eff_um = jnp.maximum(23.2 * jnp.exp(0.015 * delta_t), 1.0)
    return effective_2_volmean_radius_param_Schuman_2011(r_eff_um, params)

def detrained_ice_crystal_number(
    detrained_condensate: jnp.ndarray,
    temperature: jnp.ndarray,
    detrainment_is_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    ice_radius: jnp.ndarray,
    params: CloudParams2M,
) -> jnp.ndarray:
    """In-cloud crystal number of convectively detrained ice (ECHAM ``znidetr``) [1/m^3].

    Transliterates ``mo_cloud_micro_2m.f90:958-982``: the detrained condensate
    enters as crystals of the temperature-parameterised radius ``zrid``
    (:func:`ice_volume_mean_radius_from_temperature`), through the plate
    mass-size relation the cloud microphysics shares with radiation (Lohmann
    et al. 2008, ERL, eq. 1):

        znidetr = conv_effr2mvr·(0.5e-2)^pow_PK·1000/fact_PK · ρ·zxtec
                  / (max(paclc, clc_min)·zrid^pow_PK)

    with ``zrid`` in metres (lines 970-972, verbatim). Since
    ``(0.5e-2/zrid[m])^pow_PK = D[cm]^-pow_PK`` with ``D = 2·zrid``, this is
    the detrained mass concentration divided by the plate crystal mass
    ``m[g] = fact_PK·D[cm]^pow_PK`` at that diameter, times
    ``conv_effr2mvr``. The whole detrained condensate ``zxtec`` (both
    phases) enters, gated by ECHAM's ``ll_cv``: condensate is detrained, and
    the step-start temperature is below ``cthomi``, or below ``tmelt`` with
    the section-1 Wegener-Bergeron-Findeisen criterion ``lo2_2d`` true
    (lines 958-963). The number is zero where the cover is at or below
    ``clc_min`` (line 974) and floored at ``cqtmin`` everywhere (line 978),
    exactly as ECHAM does, so a cell with no detrainment gains ``cqtmin``
    (1e-12 /m^3) crystals.

    Parameters
    ----------
    detrained_condensate : jnp.ndarray
        Detrained condensate this step [kg/kg] (ECHAM ``ztmst·zxtec``).
    temperature : jnp.ndarray
        Step-start temperature [K] (ECHAM ``ptm1``).
    detrainment_is_ice : jnp.ndarray
        Section-1 criterion ``lo2_2d`` (boolean): the updraft is below the
        Korolev/Mazin threshold of the pre-detrainment ice.
    cloud_fraction : jnp.ndarray
        Cloud cover ``paclc`` [0..1].
    air_density : jnp.ndarray
        Air density [kg/m^3].
    ice_radius : jnp.ndarray
        ``zrid`` [m], at least 1e-6 m.

    """
    ll_cv = (detrained_condensate > 0.0) & (
        (temperature < params.cthomi)
        | ((temperature < params.tmelt) & detrainment_is_ice)
    )
    has_cover = cloud_fraction > params.clc_min
    # ``ice_radius`` is floored at 1e-6 m by the Schumann helper, so the
    # fractional power never sees a zero base and needs no gradient guard.
    number = (
        params.conv_effr2mvr * (0.5e-2) ** params.pow_PK * 1000.0
        / params.fact_PK
        * air_density * detrained_condensate
        / (jnp.maximum(cloud_fraction, params.clc_min)
           * ice_radius ** params.pow_PK)
    )
    number = jnp.where(has_cover, jnp.maximum(number, 0.0), 0.0)
    number = jnp.where(ll_cv, number, 0.0)
    return jnp.maximum(number, params.cqtmin)

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

#: Martin et al. (1994) droplet-spectrum breadth parameters ECHAM's radiation
#: uses when droplet number is prescribed (``mo_cloud_optics.f90``:
#: ``zkap_cont = 1.143``, ``zkap_mrtm = 1.077``; ``k^(-1/3)`` for the measured
#: continental / maritime ``k = 0.67 / 0.80``).
BREADTH_CONTINENTAL = 1.143
BREADTH_MARITIME = 1.077


def prescribed_cdnc_profile(
    pressure: jnp.ndarray, continental: jnp.ndarray | bool,
) -> jnp.ndarray:
    """ECHAM's prescribed cloud droplet number concentration [1/m^3].

    The 1-moment ECHAM cloud scheme has no prognostic droplet number; the
    number its radiation and cloud microphysics see is a fixed profile,
    ``acdnc`` (``physc.f90`` section 3.12, identical in ICON-A
    ``mo_echam_phy_diag.f90::droplet_number``)::

        zprat = MIN(8, 80000/p)**2
        zn1, zn2 = 20, 180 [cm-3] continental;  20, 80 [cm-3] maritime
        zcdnc = 1e6*(zn1 + (zn2 - zn1)*EXP(1 - zprat))   for p < 80000 Pa
        zcdnc = 1e6*zn2                                  for p >= 80000 Pa

    so the surface-layer value (80 / 180 cm-3) holds up to 800 hPa and decays
    to 20 cm-3 aloft. The profile is continuous at 800 hPa (``zprat = 1``
    there). ECHAM6 evaluates it once at the initial step from that step's
    pressure; ICON-A re-evaluates it every step. Here it is evaluated from
    the pressure it is given, i.e. every radiation call (ICON-A's form); the
    two differ only through surface-pressure changes.

    Parameters
    ----------
    pressure : jnp.ndarray
        Full-level pressure [Pa], any shape.
    continental : jnp.ndarray or bool
        True for continental columns (ECHAM: land that is not glacier, or a
        lake), broadcast against ``pressure``.

    Returns
    -------
    jnp.ndarray
        Droplet number concentration [1/m^3], shaped like ``pressure``.

    """
    zn1 = 20.0
    zn2 = jnp.where(continental, 180.0, 80.0)
    zprat = jnp.minimum(8.0, 80000.0 / pressure) ** 2
    aloft = 1.0e6 * (zn1 + (zn2 - zn1) * jnp.exp(1.0 - zprat))
    return jnp.where(pressure < 80000.0, aloft, 1.0e6 * zn2)


def continental_columns(terrain, forcing) -> jnp.ndarray:
    """ECHAM's continental mask for cloud droplets: land that is not glacier.

    ``physc.f90`` section 3.12 gives the prescribed droplet number its
    continental profile for ``loland .AND. .NOT. loglac`` (or a lake), and
    ``mo_cloud_optics.f90`` takes the continental breadth constant
    ``WHERE (laland .AND. .NOT. laglac)``. ECHAM6's default
    (``lfractional_mask = .FALSE.``) reads ``loland`` from the binary
    land-sea mask; jcm's land fraction ``fmask`` is fractional, so land is
    ``fmask >= 0.5`` (the same split ``SundqvistCloudFraction`` uses).
    ``loglac`` is any glacier cover on land (``glac > 0``), from
    ``forcing.glacier_fraction`` where the forcing carries it. jcm carries no
    lake map, so no ocean-classified cell is treated as a lake.

    Returns a per-column boolean, shaped like ``terrain.fmask``. A term driven
    without terrain (``terrain=None``: unit tests and bare column drivers)
    has no land to classify, so every column is maritime.
    """
    if terrain is None:
        return jnp.asarray(False)
    land = terrain.fmask >= 0.5
    glacier = getattr(forcing, "glacier_fraction", None)
    if glacier is None:
        return land
    return jnp.logical_and(
        land, ~(jnp.reshape(jnp.asarray(glacier), land.shape) > 0.0))


def per_column(x, horizontal_shape) -> jnp.ndarray:
    """Lay a per-column field out in ``horizontal_shape``.

    Per-column fields arrive in their producer's horizontal layout (the
    terrain or aerosol grid, a flattened column vector, or a scalar); a
    column-physics term needs them in the state's own layout so they
    broadcast against ``(nlev, *horizontal_shape)`` fields.
    """
    x = jnp.asarray(x)
    if x.size == math.prod(horizontal_shape):
        return x.reshape(horizontal_shape)
    return jnp.broadcast_to(x, horizontal_shape)


def prescribed_droplet_number(
    pressure: jnp.ndarray, terrain, forcing, cdnc_factor,
) -> jnp.ndarray:
    """Droplet number [1/m^3] of the 1-moment ECHAM configuration.

    ECHAM's ``acdnc`` (:func:`prescribed_cdnc_profile` on the
    :func:`continental_columns` mask) times the MACv2-SP Twomey factor
    ``cdnc_factor``. ECHAM passes one ``acdnc`` to both its radiation
    (``mo_cloud_optics.f90``) and its 1M cloud scheme (``mo_cloud.f90``,
    ``pacdnc``), so this is the one call both jcm consumers make:
    ``Echam1MMicrophysics`` and the radiation's
    ``cloud_optics.radiation_effective_radii``.

    The Twomey factor reaches both. In MPI-ESM1.2 it scales the radiation's
    droplet number only and leaves the cloud microphysics' unperturbed
    (Mauritsen et al. 2019, JAMES, section 2.2); the extra path through the
    1M autoconversion is jcm's existing aerosol-cloud formulation, recorded
    in #932 and kept as it is for v3.0.

    Args:
        pressure: full-level pressure [Pa], ``(nlev, *horiz)``.
        terrain / forcing: for the continental mask; ``terrain=None`` is
            all-maritime.
        cdnc_factor: per-column Twomey factor, any layout with one value per
            column (or a scalar).

    """
    horiz = jnp.shape(pressure)[1:]
    continental = per_column(continental_columns(terrain, forcing), horiz)
    return (prescribed_cdnc_profile(pressure, continental)
            * per_column(cdnc_factor, horiz))


def eff_liquid_droplet_radius(
    liquid_in_cloud: jnp.ndarray,
    air_density: jnp.ndarray,
    cdnc: jnp.ndarray,
    eps: float | jnp.ndarray,
    liquid_cloud_flag: jnp.ndarray | bool = True,
    breadth: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    """Effective cloud droplet radius (ECHAM ``preffl`` / ``re_droplets``).

    ``r_eff = 1e6 * kappa * (3 * rho * q_l,in-cloud / (4 pi rho_w N))^(1/3)``

    This is the Martin et al. (1994) law in the form both ECHAM routines use:
    the 2-moment microphysics' diagnostic ``preffl``
    (``mo_cloud_micro_2m.f90``) and the radiation's
    ``mo_cloud_optics.f90::cloud_optics``
    (``zfact*zkap*(zlwc/zcdnc)**(1/3)`` with
    ``zfact = 1e6*(3e-9/(4 pi rhoh2o))**(1/3)``, ``zlwc`` in g/m3 and ``zcdnc``
    in cm-3, which is the same expression in other units). It is the single
    implementation of the law: the Lohmann 2M scheme calls it for its own
    ``preffl``, and the radiation (``cloud_optics.echam_cloud_effective_radii``)
    calls it for the radius it radiates with.

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
    breadth : jnp.ndarray, float or None
        Spectral breadth factor ``kappa``. ``None`` (the default) uses the
        Peng & Lohmann (2003) ``breadth_factor(cdnc)`` of the 2-moment scheme;
        the prescribed-number radiation passes the Martin et al. continental /
        maritime constant (``BREADTH_CONTINENTAL`` / ``BREADTH_MARITIME``).

    Returns
    -------
    jnp.ndarray
        Effective droplet radius [micron]; exactly 0 where there is no liquid
        (ECHAM's ``re_droplets2d = 0`` for a cloud-free layer).

    """
    if breadth is None:
        breadth = breadth_factor(cdnc)
    # Double-where guard on the cube root, whose derivative is infinite when the
    # base is 0. The mask must be "there is liquid to speak of", NOT
    # ``liquid_cloud_flag`` alone: in the 2M scheme that flag is
    # ``temperature > tmelt``, so it is True in every warm cell, including the
    # cloud-free majority where ``liquid_in_cloud == 0`` puts a 0 on the
    # *differentiated* branch. The forward is unchanged either way (the radius is
    # masked to 0 there), but the reverse pass multiplies that infinite local
    # derivative by the incoming cotangent, and a zero cotangent gives
    # 0 * inf = NaN. The radiation differentiates through this radius in every
    # cell of every column, clear ones included, so the guard is load-bearing.
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
