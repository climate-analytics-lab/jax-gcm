"""Utility routines and constants for the 2-m cloud microphysics scheme (based on mo_cloud_utils from ECHAM6/ICON)."""

import math
from math import pi

import jax
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


def sundqvist_condensation(
    humidity_increment, saturation_increment, cloud_fraction,
    ice_in_cloud, liquid_in_cloud, provisional_humidity, ice_weight,
    qsec, epsec,
):
    """ECHAM's section-5 condensation in the cloudy part, split by phase.

    Both ECHAM cloud schemes close the cloudy part of the cell at saturation
    and condense the part of the step's humidity increment that the
    saturation humidity does not absorb (``mo_cloud.f90`` lines 726-750 for
    the 1M scheme, ``mo_cloud_micro_2m.f90`` lines 1431-1466 for the 2M):

        zqcdif = (ztmst·pqte − zdqsat)·paclc
        zqcdif = min(max(zqcdif, −(zxib + zxlb)·paclc), qsec·zqp1)

    The two schemes differ only in how ``zdqsat`` is built from the
    temperature increment, which each computes itself and passes in. A
    negative ``zqcdif`` dissipates both phases in proportion to the in-cloud
    ice fraction ``zifrac = clip(zxib/max(epsec, zxib + zxlb), 0, 1)``; a
    positive one deposits (``lo2``) or condenses (not ``lo2``).

    ``ice_weight`` is ``lo2`` as a number, 1 for ice and 0 for liquid. The
    growth branch is ``ice_weight·zqcdif`` and ``(1 − ice_weight)·zqcdif``,
    which for a weight of exactly 0 or 1 is ECHAM's selection bit for bit and
    lets a caller give the weight a surrogate derivative.

    Args:
        humidity_increment: ``ztmst·pqte`` [kg/kg].
        saturation_increment: ``zdqsat`` [kg/kg].
        cloud_fraction: ``paclc`` / ``zclcaux``.
        ice_in_cloud, liquid_in_cloud: ``zxib``, ``zxlb`` (non-negative).
        provisional_humidity: ``zqp1`` [kg/kg].
        ice_weight: ``lo2`` as 0/1 (or a float weight equal to it).
        qsec: ECHAM ``qsec = 1 − cqtmin``.
        epsec: ECHAM ``epsec`` (1e-12).

    Returns:
        ``(zqcdif, zcnd, zdep)`` [kg/kg].

    """
    total_in_cloud = ice_in_cloud + liquid_in_cloud
    zqcdif = (humidity_increment - saturation_increment) * cloud_fraction
    zqcdif = jnp.clip(zqcdif, -total_in_cloud * cloud_fraction,
                      qsec * provisional_humidity)
    dissipation = zqcdif < 0.0
    zifrac = jnp.clip(ice_in_cloud / jnp.maximum(total_in_cloud, epsec),
                      0.0, 1.0)
    zcnd = jnp.where(dissipation, zqcdif * (1.0 - zifrac),
                     (1.0 - ice_weight) * zqcdif)
    zdep = jnp.where(dissipation, zqcdif * zifrac, ice_weight * zqcdif)
    return zqcdif, zcnd, zdep


# ECHAM's own literals for this formula (mo_cloud_utils.f90:57-60), not
# jcm's higher-precision ``c.ak``/molar-mass constants: ECHAM rounds the
# Boltzmann constant and the water-molecule mass to 3 significant figures
# here, and reproducing that rounding (not the CODATA value) is what makes
# ``karcher_lohmann_deposition_rate`` match the compiled reference to
# float64 round-off (#1017 w6).
_ZQINUCL_KB = 1.38e-23          # Boltzmann constant [J/K], mo_cloud_utils.f90:57
_ZQINUCL_XMW = 2.992e-26        # mass of one H2O molecule [kg], mo_cloud_utils.f90:59
_ZQINUCL_ALPHA = 0.5            # deposition coefficient, mo_cloud_utils.f90:58


def karcher_lohmann_deposition_rate(
    nucleation_radius_raw: jnp.ndarray,   # zri (xfrzmstr's own, unclamped) [m]
    radius_fallback: jnp.ndarray,         # zrid, the section-1 temperature-
                                           # parameterised radius [m]
    icnc_before_floor: jnp.ndarray,       # zicncq + zninucl, pre-icemin [1/m3]
    ice_mmr_previous: jnp.ndarray,        # pxim1 [kg/kg]
    cloud_fraction: jnp.ndarray,          # paclc [0..1]
    air_density: jnp.ndarray,             # zrho [kg/m3]
    air_density_correction: jnp.ndarray,  # zaaa, ice fall-speed density factor
    dynamic_viscosity: jnp.ndarray,       # zviscos [Pa s]
    ice_supersaturation: jnp.ndarray,     # zsusatix = sice = max(q/qsi-1,0) [1]
    temperature_previous: jnp.ndarray,    # ptm1 [K]
    pressure: jnp.ndarray,                # papm1 [Pa]
    sat_vap_pres_ice: jnp.ndarray,        # zesi [Pa]
    humidity_previous: jnp.ndarray,       # pqm1 [kg/kg]
    qsat_ice: jnp.ndarray,                # zqsi [kg/kg]
    dt: jnp.ndarray,
    params: CloudParams2M,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kaercher-Lohmann deposition rate ``zqinucl`` (``nic_cirrus == 2``).

    ECHAM ``mo_cloud_micro_2m.f90:1046-1102``: the vapour deposited onto the
    newly homogeneously-frozen crystals (``xfrzmstr``'s own output, section
    1) this step, with ventilation. This runs entirely on step-start
    (``*m1``) quantities, before section 5's own humidity/temperature
    adjustment -- the Fortran computes it in section 1's loop, well before
    ``column_processes`` even starts, and ``scheme.py`` calls this
    immediately after ``xfrzmstr`` for the same reason.

    Three pieces, in the Fortran's own order:

    1. ``zri`` clamp (1046-1047): fall back to ``radius_fallback`` when
       ``xfrzmstr``'s own radius is below ``epsec``, then floor at 1 um.
    2. The ICNC floor (1062-1066) and mean-crystal-mass fall speed
       (1068-1085) gated on ``ll_ice = (pxim1 > 0) & (paclc > clc_min)`` --
       the SAME mask the Fortran reuses for the ventilation factor below
       (1097), reused here for the same reason.
    3. The ventilation-corrected deposition rate itself (1089-1102):
       Fuchs-corrected diffusional growth (``zgtp``/``zvth``/``zb2``/
       ``zfuchs``) times a Reynolds-number ventilation factor (``zre``/
       ``zfre``), clamped to the available vapour above/ice below.

    Returns
    -------
    (icnc_floored, zqinucl) : the ICNC used by the deposition formula (zicncq
        after its own icemin floor, the SAME quantity the caller's
        ``icnc_sedi`` addition reads as ``zicncq_early_k + zninucl_k`` --
        this function does not re-derive that sum, only floors it) and the
        deposition rate itself [kg/kg].

    """
    epsec = params.epsec

    # 1. zri clamp (1046-1047).
    zri = jnp.where(nucleation_radius_raw >= epsec,
                     nucleation_radius_raw, radius_fallback)
    zri = jnp.maximum(zri, 1.0e-6)

    # 2. ICNC floor + mean-crystal-mass fall speed (1062-1085), gated on the
    # SAME mask the ventilation factor reuses at step 3.
    ll_ice = (ice_mmr_previous > 0.0) & (cloud_fraction > params.clc_min)
    icnc_floored = jnp.where(ll_ice,
                              jnp.maximum(icnc_before_floor, params.icemin),
                              icnc_before_floor)

    # ``icnc_floored`` is only guaranteed >= icemin on the ll_ice branch
    # (its own ``jnp.where`` leaves it at the caller's raw, possibly-zero
    # ``icnc_before_floor`` otherwise); ``jnp.where`` still evaluates this
    # division on every element, so an extra floor here (a no-op on the
    # ll_ice branch, where it is already >= icemin) keeps a gated-out
    # zero-ICNC cell from poisoning the gradient with 0/0 -> nan·0 (the
    # double-``where`` convention, JAX_gotchas.md / #558).
    zmmean = jnp.where(
        ll_ice,
        jnp.maximum(
            air_density * ice_mmr_previous
            / (jnp.maximum(icnc_floored, params.eps)
               * jnp.maximum(cloud_fraction, params.clc_min)),
            params.mi),
        params.mi)
    ll_small = zmmean < params.ri_vol_mean_1
    ll_mid = (~ll_small) & (zmmean < params.ri_vol_mean_2)
    zalfased = jnp.where(ll_small, params.alfased_1, params.alfased_2)
    zalfased = jnp.where(ll_mid, params.alfased_3, zalfased)
    zbetased = jnp.where(ll_small, params.betased_1, params.betased_2)
    zbetased = jnp.where(ll_mid, params.betased_3, zbetased)
    # NOT clamped to [0.001, 2.0] m/s: that clip belongs to the OTHER
    # ``sedimentation_ice`` subroutine (mo_cloud_micro_2m.f90:2227-2228,
    # ``sedimentation_melt.py``'s own fall speed); this formula's own copy
    # (1084-1085) has no such clip.
    zxifallmc = (params.fall * zalfased * zmmean ** zbetased
                 * air_density_correction)

    # 3. Ventilation-corrected deposition rate (1089-1102). ``zastbsti`` is
    # ECHAM's own thermodynamic term for ICE deposition (``zast``/``zbst``
    # for ice, mo_cloud_micro_2m.f90:717-721) -- the water-phase analogue
    # already lives in ``scheme.py`` as ``thermo_term_water``; ``zkair``/
    # ``zdv`` are recomputed here from the same literals ``scheme.py``'s
    # ``thermo_term_water``/``bergeron_eta`` use, since neither reaches the
    # scan as a standalone per-level array.
    zkair = 4.1867e-3 * (5.69 + 0.017 * (temperature_previous - c.tmelt))
    zdv = 2.21 / jnp.maximum(pressure, epsec)
    zastbsti = (
        c.alhs * (c.alhs / (c.rv * temperature_previous) - 1.0)
        / (zkair * temperature_previous)
        + c.rv * temperature_previous
        / jnp.maximum(zdv * sat_vap_pres_ice, epsec))

    zgtp = 1.0 / (air_density * zastbsti)
    zvth = jnp.sqrt(8.0 * _ZQINUCL_KB * temperature_previous
                    / (pi * _ZQINUCL_XMW))
    zb2 = 0.25 * _ZQINUCL_ALPHA * zvth / zdv
    zfuchs = 1.0 / (1.0 + zb2 * zri)
    zre = 2.0 * air_density * zri * zxifallmc / dynamic_viscosity
    zfre = jnp.where(ll_ice, 1.0 + 0.229 * jnp.sqrt(zre), 1.0)

    zqinucl = (4.0 * pi * zri * ice_supersaturation * icnc_floored
               * zfre * zgtp * zfuchs * _ZQINUCL_ALPHA * dt)
    zqinucl = jnp.minimum(zqinucl, humidity_previous - qsat_ice)
    zqinucl = jnp.maximum(zqinucl, -ice_mmr_previous)
    return icnc_floored, zqinucl


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


def cloud_base_search(
    cloud_fraction: jnp.ndarray, params: CloudParams2M,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-level cloud-base flags, ECHAM's ``get_cloud_bounds`` (``mo_cloud_utils.f90``:136-220).

    Ported for the BASE side only: cloud tops (``kcl_minustop``) are not
    needed by the two #955 activation sources below. On the
    physics-internal TOP-FIRST level axis (axis 0; level 0 = the model top,
    level -1 = the surface), a contiguous run of cloudy levels
    (``cloud_fraction >= epsec``) has its base at the run's bottom-most
    (highest-index, nearest-surface) level: ``get_cloud_bounds``'s
    ``llbas`` test is "cloudy here AND the level below (towards the
    surface) is clear", with the surface level always passing it —
    ``zaclcp`` at ``klev`` is never written by that routine's own loop
    (``ktdia:klev-1``), so it keeps its zero init and satisfies
    ``< epsec`` unconditionally (lines 160-183).

    ``get_cloud_bounds`` tags every level STRICTLY ABOVE its run's base
    (``kcl_minusbas``) with the base's own level index, so a per-level
    field can be gathered there and broadcast up through the run
    (``mo_cloud_micro_2m.f90``:763-772). This port returns that gather
    index directly (``base_level_index``) rather than the Fortran's
    top/base/``kcl_minus*`` four-array encoding — the one thing both
    activation sources below need.

    Implemented as a single ``lax.scan`` over the level axis with
    ``reverse=True`` (the scan body therefore runs bottom level first,
    the direction a run's contiguity needs, while the output still comes
    back indexed top-first): the carry is the nearest base index found so
    far, reset to the current level at a base, to -1 outside any cloud,
    and inherited unchanged while climbing through a run above its base.
    Runs over axis 0 only; any trailing horizontal axes broadcast (the
    broadcasting-native convention — no vmap, no reshape).

    Parameters
    ----------
    cloud_fraction : (nlev, *horiz) cloud cover ``paclc`` [0, 1].
    params : threaded scheme parameters (``epsec``).

    Returns
    -------
    is_base : (nlev, *horiz) bool. This level is its cloud run's base.
    base_level_index : (nlev, *horiz) int32. The run's base level index
        for every cloudy level (itself, at the base); -1 outside any
        cloud.
    above_base_mask : (nlev, *horiz) bool. ``cloud_fraction`` is nonzero
        AND this level is not its own base (ECHAM's ``kcl_minusbas > 0``).

    """
    is_cloudy = cloud_fraction >= params.epsec
    # "Clear below": the level nearer the surface (index + 1) is not
    # cloudy. The last (surface) level has no level below it and always
    # qualifies, matching get_cloud_bounds' unwritten zaclcp(:,klev) == 0.
    clear_below = jnp.concatenate(
        [~is_cloudy[1:], jnp.ones_like(is_cloudy[:1])], axis=0)
    is_base = is_cloudy & clear_below

    nlev = cloud_fraction.shape[0]
    levels = jnp.arange(nlev, dtype=jnp.int32).reshape(
        (nlev,) + (1,) * (cloud_fraction.ndim - 1))
    levels = jnp.broadcast_to(levels, cloud_fraction.shape)

    def _step(carry, level_inputs):
        cloudy_k, base_k, level_k = level_inputs
        found = jnp.where(base_k, level_k, carry)
        new_carry = jnp.where(cloudy_k, found, -1)
        return new_carry, new_carry

    init_carry = -jnp.ones(cloud_fraction.shape[1:], dtype=jnp.int32)
    _, base_level_index = jax.lax.scan(
        _step, init_carry, (is_cloudy, is_base, levels), reverse=True)

    above_base_mask = is_cloudy & ~is_base
    return is_base, base_level_index, above_base_mask


def stratiform_activation_at_cloud_base(
    droplet_number: jnp.ndarray,         # Original: zcdnc, entry value
    cdnc_min: jnp.ndarray,                # Original: zcdnc_min
    activated_cdnc: jnp.ndarray,          # Original: zcdncact
    temperature_m1: jnp.ndarray,          # Original: ptm1
    liquid_tendency_mass: jnp.ndarray,    # Original: ztmst·pxlte (sign only)
    cloud_fraction: jnp.ndarray,          # Original: paclc
    cloud_fraction_prev: jnp.ndarray,     # Original: zaclc_tm1
    water_supersaturation: jnp.ndarray,   # Original: zsusatw (signed)
    params: CloudParams2M,
) -> jnp.ndarray:
    """Stratiform cloud-droplet activation at cloud base, copied to the cloud above it.

    Transliterates ``mo_cloud_micro_2m.f90``:742-782, ECHAM-HAM's
    section-1 activation (``cloud_subm_1``'s ``zcdncact`` converted into
    newly-formed droplets, ``zqlnuc``). This is a SEPARATE replenishment
    path from the deeper, later ``update_in_cloud_water`` re-activation
    (``mo_cloud_micro_2m.f90``:2594-2608,
    :func:`~jcm.physics.clouds.lohmann_2m.assembly.update_in_cloud_water`'s
    own ``ll2``): that one only fires once CDNC has already decayed all
    the way to the floor ``cdnc_min``. This one fires at the cloud base on
    ANY of four parcel-theory triggers — a fresh updraft, growing liquid, a
    growing cloud, or supersaturated air — so it replenishes CDNC long
    before a column has been stripped bare (jax-gcm#955; the gap this
    closes is why a large aerosol perturbation barely moved CDNC in a
    marine-cloud-brightening sensitivity test, since without it the only
    reactivation available needed CDNC to have fallen to 40 cm⁻³ first).

    At a cloud-base level (:func:`cloud_base_search`'s ``is_base``, and
    warmer than ``cthomi``), the number of newly-activated droplets is
    ``max(activated_cdnc - droplet_number, 0)`` wherever any of

      - ``droplet_number <= cdnc_min`` (no previous nucleation this step),
      - ``liquid_tendency_mass > 0`` (the liquid is still growing),
      - ``cloud_fraction > cloud_fraction_prev`` (the cover just grew),
      - ``water_supersaturation > eps`` (the air is supersaturated),

    holds (lines 742-752). The base's updated CDNC is then copied,
    verbatim, to every level strictly above it in the SAME cloud run,
    provided the base itself nucleated a strictly positive number there
    (lines 763-782) — a base that satisfied the trigger but gained zero
    droplets (``activated_cdnc <= droplet_number`` there) leaves the
    levels above it untouched, exactly as the Fortran's own
    ``zqlnuc_bas > 0`` gate does.

    Levels outside any cloud, or at/below ``cthomi``, pass through
    unchanged. This is purely an entry-level replenishment: ECHAM's own
    phase-consistency correction that follows it in the Fortran (lines
    784-797, crediting supercooled CDNC to ICNC) is a different, later
    concern handled by this scheme's own freezing terms and is out of
    scope here (jax-gcm#955 cites only 742-782 for this item). The caller
    still applies the ``minimum_CDNC`` floor afterwards, as before this
    function existed.

    Parameters
    ----------
    droplet_number : (nlev, *horiz) cqtmin-floored grid-mean CDNC [1/m^3],
        BEFORE the ``minimum_CDNC`` floor — ECHAM floors CDNC to
        ``cdnc_min`` only later, in ``update_in_cloud_water``, not here.
    cdnc_min : (nlev, *horiz) the per-level minimum CDNC [1/m^3]
        (:func:`minimum_CDNC` of the in-cloud liquid BEFORE this step's
        convective detrainment).
    activated_cdnc : (nlev, *horiz) aerosol-activated CDNC [1/m^3].
    temperature_m1 : (nlev, *horiz) step-start temperature [K].
    liquid_tendency_mass : (nlev, *horiz) this step's liquid increment
        EXCLUDING convective detrainment [kg/kg] (``ztmst·pxlte`` has the
        same sign as ``pxlte`` since ``ztmst`` > 0).
    cloud_fraction, cloud_fraction_prev : (nlev, *horiz) this step's and
        the PREVIOUS step's cloud cover [0, 1].
    water_supersaturation : (nlev, *horiz) signed ``q/qsat_water - 1`` at
        the step-start state (NOT the clipped-negative evaporation form
        the rain-evaporation chain uses elsewhere in this scheme).
    params : threaded scheme parameters.

    Returns
    -------
    (nlev, *horiz) CDNC [1/m^3] after this activation step.

    """
    is_base, base_level_index, above_base_mask = cloud_base_search(
        cloud_fraction, params)

    trigger = is_base & (temperature_m1 > params.cthomi) & (
        (droplet_number <= cdnc_min)
        | (liquid_tendency_mass > 0.0)
        | (cloud_fraction > cloud_fraction_prev)
        | (water_supersaturation > params.eps)
    )
    newly_activated = jnp.where(
        trigger, jnp.maximum(activated_cdnc - droplet_number, 0.0), 0.0)
    droplet_number_at_base = jnp.where(
        trigger, droplet_number + newly_activated, droplet_number)

    # Gather the (possibly just-updated) base CDNC, and its own newly-
    # activated number, up through the run (lines 763-772). The index is
    # clipped to a valid position so ``take_along_axis`` never reads out
    # of bounds where there is no cloud (index -1); ``above_base_mask``
    # gates its use to genuinely cloudy, non-base levels only, so the
    # clip never leaks a spurious value.
    safe_index = jnp.clip(base_level_index, 0, droplet_number.shape[0] - 1)
    gathered_cdnc = jnp.take_along_axis(
        droplet_number_at_base, safe_index, axis=0)
    gathered_newly_activated = jnp.take_along_axis(
        newly_activated, safe_index, axis=0)

    copy_mask = above_base_mask & (gathered_newly_activated > 0.0)
    return jnp.where(copy_mask, gathered_cdnc, droplet_number_at_base)


def convective_detrainment_droplet_number(
    droplet_number: jnp.ndarray,               # Original: zcdnc (post item 1)
    activated_cdnc: jnp.ndarray,                # Original: cdncact_cv (this level)
    activated_cdnc_at_convective_base: jnp.ndarray,  # Original: cdncact_cv (iclbas)
    has_convective_cloud_base: jnp.ndarray,     # Original: iclbas > 0
    temperature_m1: jnp.ndarray,                # Original: ptm1
    detrainment_is_ice: jnp.ndarray,             # Original: lo2_2d (section 1)
    detrained_condensate: jnp.ndarray,          # Original: ztmst·zxtec (mass)
    cloud_fraction: jnp.ndarray,                # Original: paclc
    stratiform_liquid: jnp.ndarray,             # Original: pxlm1+ztmst·pxlte
    params: CloudParams2M,
) -> jnp.ndarray:
    """Droplet number of detrained convective liquid, LWC-weighted into CDNC.

    Transliterates ``mo_cloud_micro_2m.f90``:889-941, the ``zqlnuccv``
    droplet source jax-gcm#955 cites as its second item. ECHAM-HAM's
    convection scheme runs its own aerosol activation call at the
    convective updraft and publishes the result as ``cdncact_cv`` — a
    per-level profile of "the number ARG would activate if cloud base
    were here" (``mo_ham_activ.f90``/``mo_activ.f90``). jcm has no second,
    convective-updraft activation call: the SAME stratiform
    ``activated_cdnc`` diagnostic that feeds
    :func:`stratiform_activation_at_cloud_base` stands in for
    ``cdncact_cv`` here too, simply read at a different level (the
    convective base instead of the stratiform one) — the deliberate
    simplification ``docs/source/design/lohmann_2m_column_processes.md``
    records, and what jax-gcm#955 means by "build [the convective-base
    bookkeeping] once ... and consume it in both" items.

    The detrained liquid's droplet number is the lesser of the activated
    number AT THIS LEVEL and AT THE CONVECTIVE BASE (further activation
    above the base is capped by what activated below it; lines 902-914),
    gated by ``ll3_2d`` (lines 892-899): a convective base exists, this
    level received detrained condensate this step, and the condensate
    counts as liquid — warmer than ``tmelt``, or in the mixed-phase window
    with the section-1 WBF criterion ``lo2_2d`` false. The result is then
    blended into CDNC weighted by the stratiform and detrained in-cloud
    liquid water contents (lines 920-937), and the blend is never allowed
    to lower CDNC (line 939): a cell with no detrained liquid, or no
    convective base, is returned exactly as received.

    Parameters
    ----------
    droplet_number : (nlev, *horiz) grid-mean CDNC [1/m^3] after
        :func:`stratiform_activation_at_cloud_base`.
    activated_cdnc : (nlev, *horiz) the per-level activation diagnostic,
        standing in for ``cdncact_cv`` at this level (see above).
    activated_cdnc_at_convective_base : (*horiz,) the SAME diagnostic
        gathered at the column's convective cloud-base level (ECHAM's
        ``cdncact_cv`` read at ``iclbas``); 0 where there is none.
    has_convective_cloud_base : (*horiz,) bool, ECHAM's ``iclbas > 0``:
        this column has an active convective updraft this step.
    temperature_m1 : (nlev, *horiz) step-start temperature [K].
    detrainment_is_ice : (nlev, *horiz) bool, section 1's ``lo2_2d`` (the
        same array :func:`detrained_ice_crystal_number` reads).
    detrained_condensate : (nlev, *horiz) this step's total detrained
        condensate mass [kg/kg] (both phases; ECHAM's ``zxtec`` is a rate,
        this scheme's own ``zxtec`` is already the per-step mass, so no
        extra ``dt`` factor is applied here).
    cloud_fraction : (nlev, *horiz) cloud cover ``paclc`` [0, 1].
    stratiform_liquid : (nlev, *horiz) grid-mean in-cloud liquid mixing
        ratio EXCLUDING this step's convective detrainment [kg/kg]
        (``qc_m1 + qc_increment``).
    params : threaded scheme parameters.

    Returns
    -------
    (nlev, *horiz) CDNC [1/m^3] after this blend.

    """
    ll3 = (
        has_convective_cloud_base
        & (detrained_condensate > 0.0)
        & (
            (temperature_m1 > params.tmelt)
            | ((temperature_m1 > params.cthomi)
               & (temperature_m1 < params.tmelt)
               & ~detrainment_is_ice)
        )
    )
    candidate = jnp.maximum(
        jnp.minimum(activated_cdnc, activated_cdnc_at_convective_base), 0.0)
    detrained_droplet_number = jnp.where(ll3, candidate, 0.0)
    detrained_lwc = jnp.where(ll3, detrained_condensate, 0.0)

    has_stratiform = ((cloud_fraction >= params.epsec)
                       & (stratiform_liquid >= params.epsec))
    stratiform_lwc = jnp.where(
        has_stratiform,
        stratiform_liquid / jnp.maximum(cloud_fraction, params.epsec),
        0.0,
    )

    total_lwc = stratiform_lwc + detrained_lwc
    blended = (
        (droplet_number * stratiform_lwc
         + detrained_droplet_number * detrained_lwc)
        / jnp.maximum(total_lwc, params.epsec)
    )
    return jnp.maximum(jnp.maximum(droplet_number, blended), params.cqtmin)


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

def rain_mixing_ratio_from_flux(
    rain_flux: jnp.ndarray,             # prfl entering the level [kg/m^2/s]
    precip_cover: jnp.ndarray,          # pclcpre entering the level [0..1]
    air_density_13: jnp.ndarray,        # pqrho = 1.3/prho [m^3/kg]
    params: CloudParams2M,
) -> jnp.ndarray:
    """Marshall-Palmer inversion of the rain flux entering a level (ECHAM ``zxrp1``).

    ``mo_cloud_micro_2m.f90`` section 7 (eq. 10.70 of Roeckner et al. 2003, MPI
    report 349): the rain mixing ratio "from above" that ``precip_formation_warm``
    needs for the accretion term ``zrac1``, recovered from the column's carried
    rain flux and precip cover (ECHAM carries no rain tracer at all). ``12.45``
    comes from the Marshall-Palmer intercept ``a_10*(n_0r)**(-1/8) =
    90.8*(8e6)**(-1/8)``. The two guards are ECHAM's own: the ``eps`` gate on
    ``precip_cover`` (``> params.epsec`` here) and the ``cqtmin`` gate on the
    flux itself, both ``MERGE``d to zero outside them. Double-where on the
    fractional power base: ``rain_flux`` or ``precip_cover`` can be exactly 0
    where the mask is False, and ``x**(8/9)`` has an infinite derivative there.
    """
    rain_present = jnp.logical_and(precip_cover > params.epsec, rain_flux > params.cqtmin)
    clcpre_safe = jnp.maximum(precip_cover, params.epsec)
    base = jnp.where(
        rain_present,
        jnp.maximum(rain_flux, params.cqtmin) / (12.45 * clcpre_safe * jnp.sqrt(air_density_13)),
        1.0,
    )
    return jnp.where(rain_present, base ** (8.0 / 9.0), 0.0)

def snow_mixing_ratio_from_flux(
    snow_flux: jnp.ndarray,             # pssfl entering the level [kg/m^2/s]
    precip_cover: jnp.ndarray,          # pclcpre entering the level [0..1]
    params: CloudParams2M,
) -> jnp.ndarray:
    """Marshall-Palmer inversion of the snow flux entering a level (ECHAM ``zxsp1``).

    ``mo_cloud_micro_2m.f90`` section 7 (eq. 10.74 of Roeckner et al. 2003, MPI
    report 349): the snow mixing ratio "from above" that ``precip_formation_cold``
    needs for the riming/accretion terms, recovered from the carried snow flux
    and precip cover the same way :func:`rain_mixing_ratio_from_flux` recovers
    the rain mixing ratio. Same ``eps``/``cqtmin`` gates and double-where guard.
    """
    snow_present = jnp.logical_and(precip_cover > params.epsec, snow_flux > params.cqtmin)
    clcpre_safe = jnp.maximum(precip_cover, params.epsec)
    base = jnp.where(
        snow_present,
        jnp.maximum(snow_flux, params.cqtmin) / (params.cvtfall * clcpre_safe),
        1.0,
    )
    return jnp.where(snow_present, base ** (1.0 / 1.16), 0.0)

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
    ``pacdnc``). The radiation's ``cloud_optics.radiation_effective_radii``
    calls this with the MACv2-SP factor. ``Echam1MMicrophysics`` calls it
    twice: with a factor of 1 for ECHAM's ``pacdnc``, which its Bigg and
    contact freezing (section 6.2) read, and with the MACv2-SP factor for
    the autoconversion (``autoconversion_twomey``, on by default) and the
    published ``clouds.droplet_number``.

    In MPI-ESM1.2 the Twomey factor scales the radiation's droplet number
    only and leaves the cloud microphysics' unperturbed (Mauritsen et al.
    2019, JAMES, section 2.2). Its reaching the 1M autoconversion is jcm's
    aerosol-cloud formulation, the maintainer's decision recorded in #932;
    the freezing keeps ECHAM's unscaled number.

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


def safe_cbrt_quotient(
    numerator: jnp.ndarray, denominator: jnp.ndarray, valid: jnp.ndarray,
) -> jnp.ndarray:
    """``numerator**(1/3) / denominator**(1/3)``, reverse-mode safe (#1130).

    The two cube roots are taken SEPARATELY, never of their quotient: the
    combined form's float32 reverse pass overflows when a tiny-but-positive
    numerator sits over a denominator a caller has floored near zero (the
    original bug, #1130) -- its derivative wrt the denominator is
    ``~numerator**(-2/3) * denominator**(-4/3)``, and that product clears
    float32's 3.4e38 ceiling well before either factor alone would. Split,
    each factor's own derivative (``numerator**(-2/3)``,
    ``denominator**(-4/3)``) stays in range on its own; the primal differs
    from the unsplit quotient by rounding only (<= 2 ulp f32).

    ``valid`` must already fold in every condition that keeps a cube root's
    LOCAL derivative finite at the particular ``numerator``/``denominator``
    pair, in particular ``numerator != 0`` -- a numerator that underflows to
    exactly 0.0 in float32 (reachable for liquid/ice content near the
    smallest normal, ~1.2e-38, even though the physical mask that let it
    through is True) still has ``0**(1/3)``'s infinite local derivative,
    which a zero incoming cotangent turns into ``0 * inf = NaN`` (jcm #866;
    confirmed for two of #1130's three sites by probing this exact
    expression with ``numerator`` at float32's smallest normal value). This
    helper substitutes the off-``valid`` branch's numerator/denominator with
    1.0 (cube root 1, zero gradient), exactly as a caller's own outer
    ``jnp.where(valid, ..., <forward value off valid>)`` already does for
    the result -- the double-where convention in ``JAX_gotchas.md``.
    Callers choose their own floor on ``denominator`` and their own
    ``numerator != 0`` re-check (the right floor and underflow threshold are
    scheme-specific), fold both into ``valid``, and apply the forward-value
    ``jnp.where`` themselves.
    """
    safe_numerator = jnp.where(valid, numerator, 1.0)
    safe_denominator = jnp.where(valid, denominator, 1.0)
    return safe_numerator ** (1.0 / 3.0) / safe_denominator ** (1.0 / 3.0)


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
    # #1130: was ``(q*rho/max(N,eps))**(1/3)``, the cube root of the
    # QUOTIENT -- its derivative wrt N is
    # ``~(q*rho)**(-2/3) * N**(-4/3)`` once N clears the floor, and for a
    # trace of liquid (q as small as ~1e-22 kg/kg) that product exceeds
    # float32's 3.4e38 ceiling even at N orders of magnitude above the
    # 1e-12 m^-3 floor (d/dN NaN at N = 0, -inf just above it). See
    # :func:`safe_cbrt_quotient` for why splitting the cube root fixes
    # this and ``test_matches_unsplit_quotient_form_to_rounding`` for the
    # pinned <= 2 ulp f32 primal bound.
    numerator = (3.0 / (4.0 * pi * c.rhow)) * liquid_in_cloud * air_density
    denominator = jnp.maximum(cdnc, eps)
    # Positive liquid can still underflow to a zero numerator in the
    # arithmetic above (jcm #866) -- fold that into ``has_liquid`` so
    # :func:`safe_cbrt_quotient` never differentiates the cube root there.
    # Use != 0 rather than > 0 so invalid negative/NaN values remain
    # visible, not silently masked.
    has_liquid = jnp.logical_and(has_liquid, numerator != 0.0)
    liq_eff_radius = 1.0e6 * breadth * safe_cbrt_quotient(
        numerator, denominator, has_liquid)
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
