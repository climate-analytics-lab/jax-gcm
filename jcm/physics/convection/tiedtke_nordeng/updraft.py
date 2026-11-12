"""Updraft calculations for Tiedtke-Nordeng convection scheme

This module implements the updraft calculations including:
- The cloud-base plume seed (ECHAM ``cubase`` / ``cubasmc``)
- Entrainment and detrainment
- Moist ascent with condensation
- Buoyancy calculations

Based on ECHAM ``mo_cuascent.f90`` (``cuasc``).

Every profile this module returns lives on HALF levels, exactly as ECHAM
indexes them: entry ``k`` is the value at the TOP interface of layer ``k``
in the physics-internal top-first frame (see
:mod:`~jcm.physics.convection.tiedtke_nordeng.half_levels`). The ascent
through layer ``k`` carries the plume from interface ``k + 1`` (the layer's
bottom) to interface ``k`` (its top); the per-layer ledgers (``pdmfup``,
``plude``, ``dmfen``) belong to the layer the plume just crossed.
"""

import jax.numpy as jnp
from jax import lax
from typing import NamedTuple

import jcm.constants as c
from .tiedtke_nordeng import ConvectionParameters
from .half_levels import (
    HalfLevelEnvironment,
    cubase_seed_static_energy,
    half_level_environment,
    reconstruct_pressure_half,
)
from .switches import ascent_test, threshold_switch
# ECHAM's condensing saturation adjustment (``cuadjtq``, kcall = 1). It lives
# in the leaf module ``cuadjtq`` so that ``calculate_cape_cin`` (in
# tiedtke_nordeng.py, which this module imports from) can call it too;
# re-exported here under its historical name for callers and tests.
from jcm.physics.convection.tiedtke_nordeng.cuadjtq import (
    cuadjtq_newton as saturation_adjustment,
)
from jcm.physics.thermodynamics import moist_isobaric_heat_capacity



from jcm.physics.thermodynamics import (
    es_ua,
    saturation_specific_humidity_and_derivative,
)


#: ECHAM ``cuentr``'s shallow-convection entrainment band: a shallow plume
#: entrains within this pressure depth above cloud base [Pa]
#: (mo_cuascent.f90:727, ``0.2e5``).
_SHALLOW_ENTRAINMENT_DEPTH = 2.0e4
#: ``cuentr``'s moisture floor for the mid-level ``zentest`` enhancement:
#: only where the half-level humidity below the layer exceeds it [kg/kg]
#: (mo_cuascent.f90:728, ``1.e-5``).
_ZENTEST_MIN_Q = 1.0e-5
#: ``cumastr``'s scale depth of the ``khmin`` search's height weighting
#: ``sqrt(1 + depth/(zb·g))`` [m] (mo_cumastr.f90, ``zb = 25``).
_KHMIN_ZB = 25.0
#: ``cuasc``'s first-pass cloud-top bound for a column without a surface
#: plume: the plume may not rise above the lowest interface above this
#: pressure [Pa] (mo_cuascent.f90:191, ``4.e4``).
_NO_CUBASE_TOP_PA = 4.0e4
#: ``cuentr``'s value of π in the organized-detrainment ``tan`` profile
#: (mo_cuascent.f90:781-782 writes ``3.1415``). Keeping ECHAM's truncated
#: constant is what keeps ``tan`` finite at the cloud top, where the
#: fractional height reaches one.
_ECHAM_PI = 3.1415


class UpdatedraftState(NamedTuple):
    """Half-level updraft profiles (entry ``k`` = top interface of layer ``k``)."""

    tu: jnp.ndarray      # Updraft temperature (K) — ECHAM ``ptu``
    qu: jnp.ndarray      # Updraft specific humidity (kg/kg) — ``pqu``
    lu: jnp.ndarray      # Updraft condensate (kg/kg) — ``plu``: after the
                         # layer's precipitation for the continuing plume and
                         # before it for the cloud-top overshoot, weighted by
                         # the two parts' fluxes
    mfu: jnp.ndarray     # Updraft mass flux (kg/m²/s) — ``pmfu``; the plume
                         # profile from cloud base up, including the cloud-top
                         # overshoot (zero below cloud base: the sub-cloud
                         # taper is cuflx's, applied by the ledger)
    entr: jnp.ndarray    # Fractional entrainment rate (1/m) of layer k
    detr: jnp.ndarray    # Fractional detrainment rate (1/m) of layer k
    buoy: jnp.ndarray    # Plume buoyancy ``zbuoyz`` (m/s²), condensate-loaded
    pdmfup: jnp.ndarray  # Precip generated in layer k (kg/m²/s) — ``pdmfup``
    plude: jnp.ndarray   # Condensate DETRAINED in layer k (kg/m²/s) — ECHAM
                         # ``plude``. Feeds the stratiform cloud tracers (ECHAM
                         # pxtecl/pxteci via zxtec = g/Δp·plude) and the cudtdq
                         # latent-heat ledger; includes the cloud-top
                         # overshoot's detrainment.
    uu: jnp.ndarray      # Updraft zonal wind (m/s) — ECHAM ``puu`` (cuasc).
                         # Consumed by cududv's momentum-transport deviation
                         # flux ``mfu·(uu − ū)``.
    vu: jnp.ndarray      # Updraft meridional wind (m/s) — ECHAM ``pvu``.
    dmfen: jnp.ndarray | None = None  # Absolute entrainment into the plume
                         # in layer k (kg/m²/s) — ECHAM ``zdmfen + zoentr``,
                         # including the layer the cloud-top overshoot
                         # crosses; the ledger the convective tracer
                         # transport reads.
    kctop: jnp.ndarray | None = None  # Highest interface where the plume
                         # passed ECHAM's ascent test (``kctop``); ``nlev − 2``
                         # (``klevm1``) when it passed none.
    ldcum_weight: jnp.ndarray | None = None  # ECHAM's ``ldcum`` after the
                         # ascent, ``kctop /= klevm1`` (mo_cuascent.f90:541),
                         # as 1.0/0.0. Where the first ascent test decides it
                         # (a ``cubase`` plume based at ``klevm1``) it is that
                         # test's value and carries its surrogate derivative
                         # (``switches.ascent_test``), so a column's
                         # convection can be differentiated across its own
                         # onset; elsewhere it is a constant.
    peff: jnp.ndarray | None = None  # Fraction of the plume condensate
                         # converted to precipitation in layer k — HAMMOZ's
                         # convective ``peffwat``/``peffice`` combined over
                         # the phases (see ``ham_precip_efficiency``); the
                         # in-plume aerosol scavenging reads it.

#: ``cuflx``'s liquid/ice partition of the plume condensate for the
#: scavenging interface (mo_cufluxdts.f90:154-155, 281-287):
#: ``zalpha = zcaa + (1 − zcaa)·exp(−zcab·Tc²)`` below 0 °C, 1 above.
_ZCAA = 0.0059
_ZCAB = 0.003102
#: ``prep_wetdep_hydro``'s floor on a phase's condensate (``zmin``,
#: mo_hammoz_wetdep.f90:409): a phase below it converts nothing.
_HAM_ZMIN = 1.0e-10


def ham_precip_efficiency(lu_pre, lu_post, tu):
    """HAMMOZ's convective precipitation efficiency of one plume level.

    ``cuasc`` stores the condensate before and after the layer's
    conversion, ``pmwc = plu`` and ``pmrateprecip = plu − zlnew``
    (mo_cuascent.f90:459-460); ``cuflx`` splits both by the updraft
    temperature ``ptu`` with ``zalpha`` (mo_cufluxdts.f90:281-287); and
    ``prep_wetdep_hydro`` forms ``peffwat = zmratepr/zmlwc`` and
    ``peffice = zmrateps/zmiwc``, each zero where its phase holds no more
    than ``zmin`` and clipped to [0, 1] (mo_hammoz_wetdep.f90:426-435).
    ``ham_wetdep`` removes ``csr_conv·peff`` of the liquid share
    ``1 − pice`` and of the ice share ``pice`` (``ic_scav``), so the
    fraction of the level's in-plume aerosol that meets a converting phase
    is the phase-weighted sum returned here. Where both phases exceed
    ``zmin`` it is exactly ``(plu − zlnew)/plu``.
    """
    tc = tu - c.tmelt
    cold = tc < 0.0
    zalpha = jnp.where(
        cold, _ZCAA + (1.0 - _ZCAA) * jnp.exp(-_ZCAB * tc * tc), 1.0)
    has_cond = lu_pre > _HAM_ZMIN
    ratio = jnp.clip(
        (lu_pre - lu_post) / jnp.where(has_cond, lu_pre, 1.0), 0.0, 1.0)
    liquid = zalpha * lu_pre > _HAM_ZMIN
    ice = (1.0 - zalpha) * lu_pre > _HAM_ZMIN
    weight = (jnp.where(liquid, zalpha, 0.0)
              + jnp.where(ice, 1.0 - zalpha, 0.0))
    return jnp.where(has_cond, ratio * weight, 0.0)


def column_environment(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cp_moist: jnp.ndarray | None = None,
    pressure_half: jnp.ndarray | None = None,
    layer_mass: jnp.ndarray | None = None,
    condensate: jnp.ndarray | None = None,
) -> HalfLevelEnvironment:
    """Build the ``cuini`` environment for a (top-first) column.

    ``pressure_half`` is the host's interface pressure (the model path always
    supplies it). Standalone column callers may instead give the per-layer
    air mass ``layer_mass`` (``Δp/g``), or nothing, in which case the
    interfaces are reconstructed from the full-level pressures
    (:func:`~.half_levels.reconstruct_pressure_half`).
    """
    if cp_moist is None:
        cp_moist = moist_isobaric_heat_capacity(humidity)
    if pressure_half is None:
        pressure_half = reconstruct_pressure_half(pressure, layer_mass)
    return half_level_environment(
        temperature, humidity, pressure, pressure_half, cp_moist, condensate,
    )


def _pick_level(arr: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Select ``arr[idx]`` along axis 0 via masked reduction (no dynamic gather).

    Under ``jax.vmap`` across ``ncols`` columns, ``arr[idx]`` lowers to a
    per-column dynamic slice whose ``(ncols, 1, 1)`` gather tile bloats by up
    to 1024x on TPU ``(8, 128)`` registers; a masked sum along axis 0 stays on
    the dense ``(nlev, ncols)`` grid while matching ``arr[idx]`` exactly.
    """
    n = arr.shape[0]
    i = jnp.asarray(idx, dtype=jnp.int32)
    i = jnp.where(i < 0, i + n, i)
    i = jnp.clip(i, 0, n - 1)
    levels = jnp.arange(n, dtype=jnp.int32).reshape(
        (n,) + (1,) * (arr.ndim - 1)
    )
    mask = levels == jnp.expand_dims(i, axis=0)
    z = jnp.zeros((), dtype=arr.dtype)
    return jnp.sum(jnp.where(mask, arr, z), axis=0)


def cubase_parcel(env: HalfLevelEnvironment, kbase: jnp.ndarray):
    """ECHAM ``cubase`` parcel at the cloud-base interface ``kbase``.

    The parcel leaves the lowest interface (the top of the bottom layer)
    with that interface's environment (``ptu = ptenh(klev)``,
    ``pqu = pqenh(klev)``, cuini) and is walked up the interfaces
    conserving ``pcpcu·T + pgeoh`` (mo_cuinitialize.f90:294-295), which
    telescopes to a single lift of the seed's static energy
    ``pcpcu(klev)·ptenh(klev) + pgeoh(klev)``
    (:func:`~.half_levels.cubase_seed_static_energy`). At the base it is
    saturation adjusted with the damped condensation-only Newton step at the
    interface pressure (``cuadjtq`` kcall = 1, lines 296-314); the condensate
    stays in the plume, so total water is conserved wherever the step
    condensed. Within ~1 mK of ``tmelt``, where the adjustment can return the
    parcel with more vapour and no condensate (see ``cuadjtq_newton``), ECHAM
    carries that adjusted parcel on to the next interface; the telescoped
    lift here does not, a difference of ~5e-4 K and ~2e-7 kg/kg.

    Returns:
        ``(tu, qu, lu)`` at interface ``kbase``.

    """
    geoh_kb = _pick_level(env.geoh, kbase)
    cpcu_kb = _pick_level(env.cpcu, kbase)
    paph_kb = _pick_level(env.paph, kbase)
    t_dry = (cubase_seed_static_energy(env) - geoh_kb) / cpcu_kb
    return saturation_adjustment(t_dry, env.qenh[-1], paph_kb)


def cubasmc_seed_temperature(cp_moist, env, kk):
    """ECHAM ``cubasmc`` seed at the bottom interface of layer ``kk``.

    ``ptu(kk+1) = (pcpen(kk)·pten(kk) + pgeo(kk) − pgeoh(kk+1))/pcpen(kk)``
    (mo_cuascent.f90:641-642): the full-level environment brought
    dry-adiabatically down to the layer's bottom interface.
    """
    return (
        _pick_level(env.dse, kk) - _pick_level(env.geoh, kk + 1)
    ) / _pick_level(cp_moist, kk)



def _first_index_from_bottom(mask: jnp.ndarray, default) -> jnp.ndarray:
    """Largest (lowest, top-first) index where ``mask`` holds, else ``default``."""
    nlev = mask.shape[0]
    levels = jnp.arange(nlev)
    return jnp.where(jnp.any(mask), jnp.max(jnp.where(mask, levels, -1)),
                     default)


def _first_index_from_top(mask: jnp.ndarray, default) -> jnp.ndarray:
    """Smallest (highest, top-first) index where ``mask`` holds, else ``default``."""
    nlev = mask.shape[0]
    levels = jnp.arange(nlev)
    return jnp.where(jnp.any(mask), jnp.min(jnp.where(mask, levels, nlev)),
                     default)


def max_ascent_level(omega: jnp.ndarray) -> jnp.ndarray:
    """ECHAM ``cuini``'s ``klwmin``: the level of maximum resolved ascent.

    ``cuini`` (mo_cuinitialize.f90:189-196) walks the full levels from the
    lowest up to the third, keeping the level whose ``pverv`` is the most
    negative (a strict ``<``, so the lowest of equal minima wins); with no
    ascent anywhere it is the lowest level. ``cuentr`` compares it with
    interface indices (``kk ≥ klwmin``): turbulent entrainment of deep and
    mid-level plumes acts at and below the level of maximum ascent.

    Args:
        omega: Full-level pressure velocity [Pa/s], top-first.

    Returns:
        The 0-based level index.

    """
    nlev = omega.shape[0]
    levels = jnp.arange(nlev)
    valid = levels >= 2
    om = jnp.where(valid, omega, jnp.inf)
    om_min = jnp.min(om)
    return _first_index_from_bottom(
        valid & (om == om_min) & (om_min < 0.0), nlev - 1)


def no_cubase_cloud_top_bound(pressure_half: jnp.ndarray) -> jnp.ndarray:
    """``cuasc``'s cloud-top bound ``kctop0`` for a column without a surface plume.

    mo_cuascent.f90:191 sets ``kctop0`` to the lowest interface above
    400 hPa for every column ``cubase`` did not make convective — the
    columns where a mid-level (``cubasmc``) plume may start — so such a
    plume cannot rise above it in the first ascent.
    """
    return _first_index_from_bottom(
        pressure_half[:-1] < _NO_CUBASE_TOP_PA, 0)


def saturated_mse_hat(env: HalfLevelEnvironment) -> jnp.ndarray:
    """ECHAM ``cumastr``'s ``zhhatt``: the reduced saturation MSE of each interface.

    mo_cumastr.f90:617-643: the half-level saturation moist static energy
    ``pcpcu·ptenh + pgeoh + L·pqsenh``, less the part a parcel would lose
    evaporating into the environment's saturation deficit,
    ``(z + γ·z)/(1 + γ·z/L)·max(pqsenh − pqenh, 0)`` with
    ``z = pcpcu·ptenh·vtmpc1`` and ``γ = (L/pcpcu)·∂qs/∂T``. ``L`` is keyed to
    ``ptenh``. It sets the first-pass cloud-top estimate ``ictop0`` and the
    organized-detrainment limit ``zodmax``.
    """
    zalvs = jnp.where(env.tenh > c.tmelt, c.alhc, c.alhs)
    zhsat = env.cpcu * env.tenh + env.geoh + zalvs * env.qsenh
    p = jnp.maximum(env.paph[:-1], 1.0)
    _, dqsdt = saturation_specific_humidity_and_derivative(env.tenh, p)
    # ECHAM forms γ as ``zalvdcp·zdqsdt`` (L/pcpcu) where ``zes < 0.4`` and as
    # ``zqsat·zcor·ub`` above (mo_cumastr.f90 l.609-618), with lookup_ubc's
    # ``ub = (L/cpd)·d ln es/dT``: the same dqs/dT over cpd instead of pcpcu.
    # ``zes >= 0.4`` only in the top few levels (below ~150 Pa).
    zes = es_ua(env.tenh) * (c.rd / c.rv) / p
    zgam = zalvs / jnp.where(zes >= 0.4, c.cpd, env.cpcu) * dqsdt
    zzz = env.cpcu * env.tenh * c.vtmpc1
    return zhsat - (
        (zzz + zgam * zzz) / (1.0 + zgam * zzz / zalvs)
        * jnp.maximum(env.qsenh - env.qenh, 0.0)
    )


def cloud_base_mse(env: HalfLevelEnvironment, kcbot, tu, qu) -> jnp.ndarray:
    """ECHAM ``zhcbase``: the moist static energy of the cloud-base parcel.

    ``pcpcu·ptu + pgeoh + L·pqu`` at ``kcbot``, with ``L`` the sublimation
    heat at or below the melting point (mo_cumastr.f90:600-603).
    """
    zalvs = jnp.where(tu > c.tmelt, c.alhc, c.alhs)
    return _pick_level(env.cpcu, kcbot) * tu + _pick_level(env.geoh, kcbot) + zalvs * qu


def estimate_cloud_top(hhatt: jnp.ndarray, hcbase, kcbot) -> jnp.ndarray:
    """ECHAM ``cumastr``'s first-pass cloud-top estimate ``ictop0``.

    mo_cumastr.f90:606-645: starting from ``kcbot − 1`` and walking the
    interfaces ``3 … klevm1`` upward, ``ictop0`` moves to every interface
    above it where ``zhhatt < zhcbase`` — the parcel's moist static energy
    exceeds the environment's reduced saturation value there. The result is
    the highest such interface at least two above cloud base, else
    ``kcbot − 1``. The first ascent may not succeed above it.
    """
    nlev = hhatt.shape[0]
    levels = jnp.arange(nlev)
    cand = ((levels >= 2) & (levels <= nlev - 2) & (levels <= kcbot - 2)
            & (hhatt < hcbase))
    return _first_index_from_top(cand, kcbot - 1)


def mse_minimum_level(
    env: HalfLevelEnvironment,
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    cp_moist: jnp.ndarray,
    kcbot,
    ictop0,
) -> jnp.ndarray:
    """ECHAM ``cumastr``'s ``khmin``: where organized detrainment may start.

    mo_cumastr.f90:653-713, for deep plumes: walking up from ``kcbot − 1``
    to ``ictop0``, integrate the environment's moist-static-energy lapse,
    height-weighted by ``sqrt(1 + depth/(25·g))`` with ``depth`` the
    geopotential above cloud base,

        zhmin = Σ sqrt(1 + depth/(25 g))·Δz·dh/dz,
        dh/dz = g·Δ(pcpen·pten + L·pqen + pgeo)/Δpgeo  (adjacent full levels),
        Δz    = Δp·R_d·T_v(½)/(g·p(½))                 (the layer above),

    and take the first interface where the weighted saturation deficit
    ``−L·(pqsenh − pqenh)·sqrt(…)`` falls below it; without one it is
    ``kcbot``. It is then raised to at least ``ictop0``.
    """
    nlev = temperature.shape[0]
    levels = jnp.arange(nlev)
    dp_up = jnp.concatenate([env.dp[:1], env.dp[:-1]])
    geo_up = jnp.concatenate([env.geo[:1], env.geo[:-1]])
    cp_up = jnp.concatenate([cp_moist[:1], cp_moist[:-1]])
    temp_up = jnp.concatenate([temperature[:1], temperature[:-1]])
    hum_up = jnp.concatenate([humidity[:1], humidity[:-1]])
    zalvs = jnp.where(env.tenh > c.tmelt, c.alhc, c.alhs)
    zroi = (c.rd * env.tenh * (1.0 + c.vtmpc1 * env.qenh)
            / jnp.maximum(env.paph[:-1], 1.0))
    zdz = dp_up * zroi / c.grav
    za2 = geo_up - env.geo
    safe_za2 = jnp.where(levels >= 1, za2, 1.0)
    za1 = (cp_up * temp_up - cp_moist * temperature
           + zalvs * (hum_up - humidity) + za2) * c.grav
    zdhdz = za1 / safe_za2
    zdepth = env.geoh - _pick_level(env.geoh, kcbot)
    active = (levels < kcbot) & (levels >= ictop0) & (levels >= 1)
    # Above cloud base the depth is positive; the substitution keeps the
    # square root off zero on the unused sub-cloud lanes.
    zfac = jnp.sqrt(jnp.where(active, 1.0 + zdepth / (_KHMIN_ZB * c.grav),
                              1.0))
    contrib = jnp.where(active, zfac * zdz * zdhdz, 0.0)
    zhmin = jnp.flip(jnp.cumsum(jnp.flip(contrib)))
    zrh = -zalvs * (env.qsenh - env.qenh) * zfac
    found = active & (zrh < zhmin)
    khmin = _first_index_from_bottom(found, kcbot)
    return jnp.maximum(khmin, ictop0)



def calculate_updraft(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    rho: jnp.ndarray,
    kbase: int,
    ktop: int,
    ktype: int,
    mass_flux_base: float,
    config: ConvectionParameters,
    land_fraction: jnp.ndarray | None = None,
    type_weights: jnp.ndarray | None = None,
    lift: jnp.ndarray | None = None,
    u_wind: jnp.ndarray | None = None,
    v_wind: jnp.ndarray | None = None,
    cp_moist: jnp.ndarray | None = None,
    pressure_half: jnp.ndarray | None = None,
    env: HalfLevelEnvironment | None = None,
    dt: float | None = None,
    moisture_tendency: jnp.ndarray | None = None,
    klwmin: jnp.ndarray | None = None,
    khmin: jnp.ndarray | None = None,
) -> UpdatedraftState:
    """Calculate the updraft on half levels (ECHAM ``cuasc`` + ``cuentr``).

    Args:
        temperature: Environmental temperature (K) [nlev], top-first.
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Full-level pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]; accepted for call
            compatibility (layer geometry comes from the half-level
            environment).
        rho: Air density (kg/m³) [nlev]; accepted for call compatibility.
        kbase: Cloud-base interface ``kcbot``: the top interface of layer
            ``kbase``. For a ``cubase`` plume the seed sits here; for a
            mid-level (``ktype == 3``) plume it sits one interface lower, at
            the bottom of layer ``kbase`` (``cubasmc``).
        ktop: The cloud-top bound ``kctop0``: the ascent test fails at every
            interface above it (mo_cuascent.f90:450-451). The driver passes
            ``cumastr``'s first-pass estimate ``ictop0`` (or, for a column
            without a surface plume, the 400 hPa bound) to the first ascent
            and the first ascent's realized top to the second.
        ktype: Convection type (1 deep, 2 shallow, 3 mid-level).
        mass_flux_base: Cloud base mass flux ``pmfub`` (kg/m²/s)
        config: Convection configuration
        land_fraction: Fraction of column underlying land surface (0=open
            ocean, 1=land). At 0.5 or more the column takes ECHAM's land
            ``zdnoprc`` (``config.cu_dnoprc_land``), else the sea one
            (``config.cu_dnoprc_ocean``).
        type_weights: (deep, shallow, mid) weights: one-hot in the value,
            with the derivative of the deep/shallow test's surrogate
            (``switches``); ``None`` means the one-hot of ``ktype``.
        lift: ECHAM ``zlift`` [K], the sub-grid buoyancy excess, applied to
            the buoyancy test at the first ascent step of a mid-level plume
            (the only step whose lower interface is ``klab == 1``).
        u_wind, v_wind: Full-level environmental winds (m/s) for the
            prognostic plume wind; ``None`` means calm.
        cp_moist: Full-level moist heat capacity ``cpd·(1 + vtmpc2·q)``
            [J/kg/K] (ECHAM ``pcpen``). ``None`` builds it from ``humidity``.
        pressure_half: Interface pressures [Pa] (nlev+1), used to build the
            half-level environment when ``env`` is not given.
        env: Precomputed :class:`~.half_levels.HalfLevelEnvironment`.
        dt: Time step [s] for ECHAM's per-interface mass-flux limiter
            (``zmfmax``, the air mass of the layer above per step). ``None``
            disables the limiter (standalone callers without a step).
        moisture_tendency: The pre-convection moisture tendency ``pqte``
            [kg/kg/s] [nlev] that drives the mid-level ``zentest``
            entrainment; ``None`` means none.
        klwmin: Level of maximum resolved ascent (:func:`max_ascent_level`);
            ``None`` means no ascent is known (the lowest level).
        khmin: Organized-detrainment onset (:func:`mse_minimum_level`);
            ``None`` computes it with ``ktop`` as ``ictop0``.

    Returns:
        :class:`UpdatedraftState` of half-level profiles.

    """
    # Built at call time, not as a default argument: a jax array in a
    # ``def`` default is created at import and initialises the JAX backend
    # on ``import`` (#859).
    if land_fraction is None:
        land_fraction = jnp.array(0.0)
    if lift is None:
        lift = jnp.array(0.0)
    nlev = temperature.shape[0]
    if cp_moist is None:
        cp_moist = moist_isobaric_heat_capacity(humidity)
    del layer_thickness, rho
    if env is None:
        env = column_environment(
            temperature, humidity, pressure, cp_moist, pressure_half,
        )
    # Environmental winds for the prognostic plume wind (cududv). Column
    # callers that do not transport momentum pass none; a zero wind then
    # makes the plume wind identically zero, which is inert.
    if u_wind is None:
        u_wind = jnp.zeros(nlev)
    if v_wind is None:
        v_wind = jnp.zeros(nlev)
    dtype = temperature.dtype
    if moisture_tendency is None:
        moisture_tendency = jnp.zeros(nlev, dtype)
    if klwmin is None:
        klwmin = jnp.asarray(nlev - 1)
    kctop0 = jnp.asarray(ktop)
    # ECHAM's precipitation-onset depth ``zdnoprc = MERGE(zdlev, 1.5e4,
    # ldland)`` (mo_cuascent.f90:454). ``physc`` sets ``loland`` from the
    # binary land-sea mask under ECHAM6's default ``lfractional_mask =
    # .FALSE.`` (physc.f90:383-386, mo_control.f90:86), which on jcm's
    # fractional land is ``land_fraction >= 0.5``, the split every ECHAM term
    # of jcm uses (``clouds.cloud_utils.continental_columns``).
    zdnoprc_col = jnp.where(land_fraction >= 0.5, config.cu_dnoprc_land,
                            config.cu_dnoprc_ocean)
    if type_weights is None:
        type_weights = jnp.stack([
            jnp.asarray(ktype == 1, dtype=dtype),
            jnp.asarray(ktype == 2, dtype=dtype),
            jnp.asarray(ktype == 3, dtype=dtype),
        ])
    # A JAX bool even for a Python-int ``ktype``, so ``~is_midlevel`` is a
    # logical, not a bitwise integer, negation.
    is_midlevel = jnp.asarray(ktype) == 3
    w_deep, w_shallow, w_mid = type_weights[0], type_weights[1], type_weights[2]
    levels = jnp.arange(nlev)

    # --- Cloud-base seed ---------------------------------------------------
    # ``cubase`` (surface parcel) seeds the plume AT the cloud-base interface
    # kcbot; ``cubasmc`` seeds a mid-level plume at the bottom interface of
    # layer kcbot (kk+1), with the full-level environment of layer kk brought
    # down to it and ``plu = 0`` (mo_cuascent.f90:640-652), and its first
    # ascent step then crosses layer kcbot itself.
    tu_cb, qu_cb, lu_cb = cubase_parcel(env, kbase)
    kseed = jnp.where(is_midlevel, kbase + 1, kbase)
    kseed_safe = jnp.minimum(kseed, nlev - 1)
    t_mid = cubasmc_seed_temperature(
        cp_moist, env, jnp.minimum(kbase, nlev - 2))
    tu_seed = jnp.asarray(jnp.where(is_midlevel, t_mid, tu_cb), dtype)
    qu_seed = jnp.asarray(
        jnp.where(is_midlevel, _pick_level(humidity, kbase), qu_cb), dtype)
    lu_seed = jnp.asarray(jnp.where(is_midlevel, 0.0, lu_cb), dtype)
    mfu_seed = jnp.asarray(mass_flux_base, dtype)
    # The heat capacity the seed's static-energy flux is formed with: the
    # half-level ``pcpcu`` for a cubase plume, whose seed is the adjusted
    # parcel AT kcbot; for the cubasmc seed, ``pcpen(kk+1)`` of the level
    # below the seeding layer, ``pmfus(kk+1) = pmfub·(pcpen(kk+1)·ptu(kk+1)
    # + pgeoh(kk+1))`` (mo_cuascent.f90:649; see
    # :func:`~.half_levels.cubasmc_seed_static_energy`).
    cp_seed = jnp.asarray(
        jnp.where(
            is_midlevel,
            _pick_level(cp_moist, kseed_safe),
            _pick_level(env.cpcu, kseed_safe),
        ),
        dtype,
    )

    # Plume wind at the seed. cubase gives the plume the pressure-weighted
    # mean environmental wind of the sub-cloud layers from kcbot to the
    # surface, ``Σ puen·Δp / (p_s − p_kcbot)`` (mo_cuinitialize.f90:318-352);
    # cubasmc the wind of the seeding level (line 657). ECHAM's cubase loop
    # only accumulates the levels it visits AFTER kcbot has been set (plus
    # the lowest two), so for a base above the lowest two interfaces its
    # weights do not sum to one; the complete sub-cloud mean is used here —
    # the average the division by the full sub-cloud depth states.
    sub_cloud = levels >= kbase
    w_sub = jnp.where(sub_cloud, env.dp, 0.0)
    depth = jnp.maximum(jnp.sum(w_sub), 1e-6)
    u_sub = jnp.sum(w_sub * u_wind) / depth
    v_sub = jnp.sum(w_sub * v_wind) / depth
    uu_seed = jnp.asarray(
        jnp.where(is_midlevel, _pick_level(u_wind, kbase), u_sub), dtype)
    vu_seed = jnp.asarray(
        jnp.where(is_midlevel, _pick_level(v_wind, kbase), v_sub), dtype)

    # ``zbuoyz`` at the seed (cuasc evaluates it at kcbot after the
    # adjustment there) for the first step's organized entrainment.
    tenh_seed = _pick_level(env.tenh, kseed_safe)
    qenh_seed = _pick_level(env.qenh, kseed_safe)
    buoy_seed = (
        c.grav * (tu_seed - tenh_seed) / tenh_seed
        + c.grav * c.vtmpc1 * (qu_seed - qenh_seed)
    )
    buoy_seed_loaded = jnp.asarray(buoy_seed - c.grav * lu_seed, dtype)
    # The Nordeng integrated buoyancy ``zbuoy`` as cuasc has it when the
    # plume leaves cloud base: initialised to the condensate-free cloud-base
    # buoyancy (mo_cuascent.f90:250-251), then — because cuasc's level loop
    # also visits the sub-cloud interfaces, where the plume is the dry
    # cubase parcel carrying the seed's static energy and the lowest
    # interface's humidity — incremented by ``max(zbuoyz, 0)·zdz`` at every interface
    # from ``klevm1`` down to ``kcbot + 1`` (lines 516-523), with
    # ``zdz = (pgeo(jk−1) − pgeo(jk))/g``. The cloud-base interface's own
    # term is added by the first ascent step below. ECHAM forms it for deep
    # (cubase) plumes; the deep weight scales its use.
    geo_above = jnp.concatenate([env.geo[:1], env.geo[:-1]])
    zdz_above = (geo_above - env.geo) / c.grav
    t_sub = (cubase_seed_static_energy(env) - env.geoh) / env.cpcu
    zbuoyz_sub = jnp.maximum(
        c.grav * (t_sub - env.tenh) / env.tenh
        + c.grav * c.vtmpc1 * (env.qenh[-1] - env.qenh),
        0.0,
    )
    below_base = (levels > kbase) & (levels <= nlev - 2)
    zbuoy_sub = jnp.sum(jnp.where(below_base, zbuoyz_sub * zdz_above, 0.0))
    zbuoy_init = jnp.where(is_midlevel, 0.0, buoy_seed + zbuoy_sub)

    # Per-layer geometry, all from the half-level environment:
    #   dz_p  — the layer thickness cuentr's turbulent and organized
    #           detrainment use, Δp·(1/ρ at the layer's bottom interface)/g
    #           (``zdprho·zrrho``, mo_cuascent.f90:723-724, 783);
    #   dz_g  — the geometric thickness ``(pgeoh(k) − pgeoh(k+1))/g`` of the
    #           organized entrainment and the precipitation conversion.
    # Interface k+1 of the bottom layer is the surface, which no ascent step
    # starts from, so its lookups use a clamped index.
    tenh_kp1 = jnp.concatenate([env.tenh[1:], env.tenh[-1:]])
    qenh_kp1 = jnp.concatenate([env.qenh[1:], env.qenh[-1:]])
    qsenh_kp1 = jnp.concatenate([env.qsenh[1:], env.qsenh[-1:]])
    geoh_kp1 = jnp.concatenate([env.geoh[1:], env.geoh[-1:]])
    cpcu_kp1 = jnp.concatenate([env.cpcu[1:], env.cpcu[-1:]])
    zrrho = c.rd * tenh_kp1 * (1.0 + c.vtmpc1 * qenh_kp1) / env.paph[1:]
    dz_p = env.dp * zrrho / c.grav
    geoh_below = jnp.concatenate([env.geoh[1:], jnp.zeros_like(env.geoh[:1])])
    dz_g = (env.geoh - geoh_below) / c.grav
    # Nordeng ``zdrodz`` for the layer (mo_cuascent.f90:519-522), formed at
    # the layer's bottom interface from the two full levels that bracket it:
    # zdz = (pgeo(k) − pgeo(k+1))/g, the environment density-scale-height
    # gradient −ln(T(k)/T(k+1))/zdz − g/(R_d·T_v(k+1/2)).
    geo_below = jnp.concatenate([env.geo[1:], env.geo[-1:]])
    t_below = jnp.concatenate([temperature[1:], temperature[-1:]])
    zdz_full = jnp.maximum((env.geo - geo_below) / c.grav, 1.0)
    zdrodz = (
        -jnp.log(temperature / t_below) / zdz_full
        - c.grav / (c.rd * tenh_kp1 * (1.0 + c.vtmpc1 * qenh_kp1))
    )
    # Air mass of the layer ABOVE each interface per step (``zmfmax``): the
    # cap on the plume flux leaving that interface.
    dp_above = jnp.concatenate([env.dp[:1], env.dp[:-1]])
    if dt is None:
        mfmax = jnp.full(nlev, jnp.inf, dtype)
    else:
        mfmax = dp_above / (c.grav * dt)

    # --- cuentr: where turbulent entrainment acts (mo_cuascent.f90:719-765)
    # Turbulent DETRAINMENT ``zdmfde = pentr·pmfu·Δz`` acts at every layer
    # above cloud base; turbulent ENTRAINMENT at the same rate only
    #   * deep     — at and below the level of maximum ascent
    #                (``kk ≥ max(klwmin, kctop0 + 2)``) or in the lower half
    #                of the cloud (``icond1``: p above the pressure midway
    #                between cloud base and ``kctop0``);
    #   * shallow  — within 200 hPa of cloud base (``icond2``) or in the
    #                lower half;
    #   * mid-level — at and below the level of maximum ascent, where it is
    #                enhanced by ``zentest`` (below).
    # The type weights blend the three gates.
    p_base = _pick_level(env.paph, kbase)
    zpmid = 0.5 * (p_base + _pick_level(env.paph, kctop0))
    paph_k = env.paph[:-1]
    icond1 = paph_k > zpmid
    icond2 = (p_base - paph_k) <= _SHALLOW_ENTRAINMENT_DEPTH
    iklwmin = jnp.maximum(klwmin, kctop0 + 2)
    below_klw = levels >= iklwmin
    icond3 = qenh_kp1 > _ZENTEST_MIN_Q
    entr_gate = (w_deep * (below_klw | icond1)
                 + w_shallow * (icond2 | icond1)
                 + w_mid * below_klw)
    zentest_on = w_mid * (below_klw & icond3)
    # ``zentest`` (lines 756-760): the mid-level plume entrains the
    # pre-convection moisture convergence of the layer, as a fractional rate
    # ``max(pqte, 0)/pqenh / (pmfu·zrrho)`` capped at ``centrmax``.
    q_below = jnp.where(icond3, qenh_kp1, 1.0)
    zentest_rate = jnp.maximum(moisture_tendency, 0.0) / q_below

    # --- cuentr: organized detrainment (mo_cuascent.f90:767-788) ----------
    # Deep plumes only, from ``khmin`` up to ``kctop0``: the ``tan`` profile
    # in height above ``khmin``, capped at ``centrmax``, applied over
    # ``dz_p``.
    hhatt = saturated_mse_hat(env)
    hhatt_kp1 = jnp.concatenate([hhatt[1:], hhatt[-1:]])
    hcbase = cloud_base_mse(env, kbase, tu_cb, qu_cb)
    if khmin is None:
        khmin = mse_minimum_level(env, temperature, humidity, cp_moist,
                                  kbase, kctop0)
    geoh_kctop0 = _pick_level(env.geoh, kctop0)
    geoh_khmin = _pick_level(env.geoh, khmin)
    ztmzk = (geoh_kctop0 - geoh_khmin) / c.grav
    org_on = (levels <= khmin) & (levels >= kctop0) & (khmin > kctop0)
    safe_ztmzk = jnp.where(org_on, ztmzk, 1.0)
    zzmzk = (env.geoh - geoh_khmin) / c.grav
    zarg = _ECHAM_PI * jnp.clip(zzmzk / safe_ztmzk, 0.0, 1.0) * 0.5
    zorgde = jnp.tan(zarg) * _ECHAM_PI * 0.5 / safe_ztmzk
    org_detr_rate = jnp.where(
        org_on, jnp.minimum(zorgde, config.cu_centrmax), 0.0)
    # ``zodmax`` limits it below ``khmin`` (lines 371-383).
    zodmax_on = levels <= khmin

    level_inputs = (
        levels, env.tenh, env.qenh, env.geoh, env.cpcu, paph_k,
        tenh_kp1, qenh_kp1, qsenh_kp1, geoh_kp1, cpcu_kp1, hhatt_kp1,
        dz_p, dz_g, zdrodz, zdz_full, mfmax,
        u_wind, v_wind, entr_gate, zentest_on, zentest_rate, zrrho,
        org_detr_rate, zodmax_on,
    )
    entr_base_blend = jnp.maximum(
        w_deep * config.entrpen + w_shallow * config.entrscv
        + w_mid * config.entrmid, 0.0)
    cmfctop = config.cu_cmfctop

    # Scalar carry for the bottom-to-top scan: the continuing plume state at
    # the interface below (initialised to the cloud-base seed so it stays
    # ready until ``k`` crosses above ``kseed``), the Nordeng integrated
    # buoyancy ``zbuoy``, the running plume momentum fluxes, the condensate
    # left by a cloud-top overshoot for the layer above it, the realized
    # cloud top ``kctop``, whether the plume is still alive at the interface
    # below (ECHAM's ``klab(jk+1) > 0``), and the outcome of the first ascent
    # test. Per-level published profiles are emitted via ``ys`` rather than
    # mutated inside array carries.
    kctop_init = jnp.where(is_midlevel, nlev - 2, kbase).astype(jnp.int32)
    z0 = jnp.zeros((), dtype)
    initial = (
        mfu_seed, tu_seed, qu_seed, lu_seed,
        uu_seed, vu_seed, buoy_seed_loaded, cp_seed,
        jnp.asarray(zbuoy_init, dtype),
        jnp.asarray(mfu_seed * uu_seed, dtype),
        jnp.asarray(mfu_seed * vu_seed, dtype),
        z0,
        kctop_init,
        jnp.asarray(True),
        z0,
    )

    def updraft_step(carry_tuple, inputs):
        (mfu_b, tu_b, qu_b, lu_b, uu_b, vu_b, buoy_b, cp_b,
         zbuoy_accum, zmfuu, zmfuv, ov_cond, kctop_c, alive,
         first_test) = carry_tuple
        (k, tenh_k, qenh_k, geoh_k, cpcu_k, paph_k_,
         tenh_b, qenh_b, qsenh_b, geoh_b, cpcu_b, hhatt_b,
         dzp, dzg, zdrodz_k, zdz_k, mfmax_k,
         env_u, env_v, gate_k, zentest_on_k, zentest_rate_k, zrrho_k,
         org_rate_k, zodmax_on_k) = inputs

        # ``cuasc`` carries on only from an interface whose test passed
        # (``IF (klab(jk+1) == 0) klab(jk) = 0``): once the plume has failed,
        # no layer above it is visited and the plume there keeps ``cuini``'s
        # environment and no flux.
        should_compute = (k >= 1) & (k < kseed) & alive

        def compute_updraft():
            s_b = cp_b * tu_b + geoh_b
            s_e = cpcu_b * tenh_b + geoh_b
            q_e = qenh_b
            # cuentr acts only above cloud base (``kk < kcbot``): the first
            # step of a mid-level plume crosses layer kcbot unmixed.
            mixes = k < kbase

            # Turbulent entrainment and detrainment (cuentr lines 746-764).
            zentr = entr_base_blend * mfu_b * dzp
            zdmfde = jnp.where(mixes, zentr, 0.0)
            zmfb = jnp.maximum(mfu_b, config.cmfcmin)
            zentest = jnp.minimum(
                config.cu_centrmax, zentest_rate_k / (zmfb * zrrho_k))
            zdmfen = jnp.where(
                mixes,
                zentr * gate_k + zentest_on_k * zentest * zmfb * dzp,
                0.0,
            )
            # ``zmfmax`` limiter on the turbulent entrainment and the 0.75
            # cap on the detrainment (cuasc lines 355-361).
            zmftest = mfu_b + zdmfen - zdmfde
            zdmfen = jnp.maximum(
                zdmfen - jnp.maximum(zmftest - mfmax_k, 0.0), 0.0)
            zdmfde = jnp.minimum(zdmfde, 0.75 * mfu_b)
            mfu_turb = mfu_b + zdmfen - zdmfde

            # Nordeng (1994) organized entrainment of deep plumes (lines
            # 362-370, 516-526): the positive plume buoyancy at the interface
            # below over one plus the integrated buoyancy, plus the
            # density-scale-height gradient, clamped to [0, centrmax], over
            # the geometric depth and limited by ``zmfmax`` in turn.
            zbuoyz = jnp.maximum(buoy_b, 0.0)
            zbuoy_new = zbuoy_accum + zbuoyz * zdz_k
            zoentr_frac = jnp.clip(
                zbuoyz * 0.5 / (1.0 + zbuoy_new) + zdrodz_k,
                0.0, config.cu_centrmax,
            )
            zoentr = jnp.where(mixes, w_deep * zoentr_frac * dzg * mfu_b, 0.0)
            zodetr_raw = jnp.where(mixes, org_rate_k * mfu_b * dzp, 0.0)
            zmftest = mfu_turb + zoentr - w_deep * zodetr_raw
            zoentr = jnp.maximum(
                zoentr - jnp.maximum(zmftest - mfmax_k, 0.0), 0.0)
            # ``zodmax`` (lines 371-383): below ``khmin`` the organized
            # detrainment may not exceed what leaves the plume's moist static
            # energy at the cloud-base value by cloud top.
            zalvs_u = jnp.where(tu_b > c.tmelt, c.alhc, c.alhs)
            zmse = cpcu_b * tu_b + zalvs_u * qu_b + geoh_b
            znevn = ((geoh_kctop0 - geoh_b)
                     * (zmse - hhatt_b) / c.grav)
            znevn = jnp.where(znevn <= 0.0, 1.0, znevn)
            zodmax = jnp.maximum(
                (hcbase - zmse) / znevn * dzg * mfu_b, 0.0)
            zodetr = jnp.where(
                mixes & zodmax_on_k,
                jnp.minimum(zodetr_raw, zodmax), zodetr_raw)
            zodetr = w_deep * jnp.minimum(zodetr, 0.75 * mfu_turb)
            mfu_new = mfu_turb + zoentr - zodetr
            mfu_div = jnp.maximum(mfu_new, config.cmfcmin)

            # Flux-form mixing (lines 386-413): the flux arriving from below
            # plus the entrained half-level environment, minus the detrained
            # air. Turbulent detrainment leaves at the properties of the
            # plume entering the layer; organized detrainment at the static
            # energy and humidity that are NEUTRAL against the environment of
            # the layer's bottom interface — the environment carrying the
            # plume's condensate load as a virtual-temperature deficit,
            # ``zdt = (plu − vtmpc1·(qsenh − qenh))/(1/tenh + vtmpc1·γ)``.
            # Both detrain the plume's condensate. Dry static energy
            # ``pcpcu·T + pgeoh`` is what is conserved; vapour and condensate
            # are carried separately, exactly as ``pmfuq``/``pmful``.
            zalvs_e = jnp.where(tenh_b > c.tmelt, c.alhc, c.alhs)
            zga = zalvs_e * qsenh_b / (c.rv * tenh_b ** 2)
            zdt = ((lu_b - c.vtmpc1 * (qsenh_b - qenh_b))
                   / (1.0 / tenh_b + c.vtmpc1 * zga))
            zscod = jnp.maximum(
                cpcu_b * tenh_b + geoh_b + cpcu_b * zdt, 0.0)
            zqcod = jnp.maximum(qsenh_b + zga * zdt, 0.0)
            ent = zdmfen + zoentr
            s_flux = mfu_b * s_b + ent * s_e - zdmfde * s_b - zodetr * zscod
            q_flux = mfu_b * qu_b + ent * q_e - zdmfde * qu_b - zodetr * zqcod
            l_flux = mfu_b * lu_b - (zdmfde + zodetr) * lu_b
            plude_mix = lu_b * (zdmfde + zodetr)
            t_mix = jnp.clip(
                (s_flux / mfu_div - geoh_k) / cpcu_k, 100.0, 400.0)
            q_mix = q_flux / mfu_div
            l_mix = l_flux / mfu_div

            # Condensation-only saturation adjustment at the interface
            # pressure (cuadjtq kcall = 1); the condensed vapour joins the
            # condensate the plume already carries.
            tu_new, qu_new, cond = saturation_adjustment(t_mix, q_mix, paph_k_)
            lu_new = l_mix + cond

            # The ascent test (lines 442-465).
            first_mid_step = is_midlevel & (k == kbase)
            zbuo = (tu_new * (1.0 + c.vtmpc1 * qu_new - lu_new)
                    - tenh_k * (1.0 + c.vtmpc1 * qenh_k)
                    + jnp.where(first_mid_step, lift, 0.0))
            in_bound = (k >= kctop0) & (k >= 2)
            s = jnp.where(
                in_bound,
                ascent_test(cond, zbuo, mfu_new, mass_flux_base,
                            config.ascent_condensate_width,
                            config.ascent_buoyancy_width,
                            config.ascent_mass_flux_width),
                0.0)

            # Precipitation of the continuing plume (lines 452-457).
            precip_zone_w = threshold_switch(
                (p_base - paph_k_) - zdnoprc_col, config.precip_onset_width,
                inclusive=True)
            lu_after = lu_new / (
                1.0 + config.cprcon * precip_zone_w * c.grav * dzg)
            pdmfup = s * jnp.maximum((lu_new - lu_after) * mfu_new, 0.0)

            # The cloud-top overshoot (lines 540-565).
            ov = (1.0 - s) * cmfctop * mfu_b
            mfa_k = s * mfu_new
            mfu_k = mfa_k + ov
            mfu_k_div = jnp.maximum(mfu_k, config.cmfcmin)
            lu_pub = (mfa_k * lu_after + ov * lu_new) / mfu_k_div
            peff_k = jnp.where(
                mfu_k > 0.0,
                ham_precip_efficiency(lu_new, lu_after, tu_new)
                * mfa_k / mfu_k_div,
                0.0,
            )
            plude_k = (s * plude_mix
                       + (1.0 - s) * (1.0 - cmfctop) * mfu_b * lu_b)

            # Prognostic plume wind (cuasc lines 486-506).
            zdmfde_m = zdmfde + zodetr
            no_entr = ent <= 0.0
            zz = ((w_deep + w_mid) * jnp.where(no_entr, 3.0, 2.0)
                  + w_shallow * jnp.where(no_entr, 1.0, 0.0))
            zdmfeu = ent + zz * zdmfde_m
            zdmfdu = jnp.minimum(zdmfde_m + zz * zdmfde_m, 0.75 * mfu_b)
            zmfuu_new = zmfuu + zdmfeu * env_u - zdmfdu * uu_b
            zmfuv_new = zmfuv + zdmfeu * env_v - zdmfdu * vu_b
            uu_a = jnp.where(mfu_new > 0.0, zmfuu_new / mfu_div, uu_b)
            vu_a = jnp.where(mfu_new > 0.0, zmfuv_new / mfu_div, vu_b)
            uu_pub = jnp.where(
                mfu_k > 0.0, (mfa_k * uu_a + ov * uu_b) / mfu_k_div, uu_a)
            vu_pub = jnp.where(
                mfu_k > 0.0, (mfa_k * vu_a + ov * vu_b) / mfu_k_div, vu_a)

            # ``zbuoyz`` of the continuing plume for the next layer up.
            zbuoyz_here = (
                c.grav * (tu_new - tenh_k) / tenh_k
                + c.grav * c.vtmpc1 * (qu_new - qenh_k)
                - c.grav * lu_after
            )
            has_flux = mfu_b * dzp > 0.0
            den_p = jnp.where(has_flux, mfu_b * dzp, 1.0)
            den_g = jnp.where(has_flux & (dzg > 0.0), mfu_b * dzg, 1.0)
            entr_rate = jnp.where(has_flux, zdmfen / den_p + zoentr / den_g,
                                  0.0)
            detr_rate = jnp.where(has_flux, zdmfde_m / den_p, 0.0)

            passed = s > 0.5
            kctop_new = jnp.where(passed, k, kctop_c).astype(jnp.int32)
            first_new = jnp.where(k == kseed - 1, s, first_test)
            next_carry = (
                mfa_k, tu_new, qu_new, lu_after, uu_a, vu_a,
                zbuoyz_here, cpcu_k,
                zbuoy_new, zmfuu_new * s, zmfuv_new * s, ov * lu_new,
                kctop_new, passed, first_new,
            )
            # The condensate of an overshoot that ended at the interface below
            # is detrained in this layer (``plude(jk−1) = pmful(jk)``,
            # mo_cuascent.f90:557-559), added to this step's own detrainment.
            y_out = (
                tu_new, qu_new, lu_pub, mfu_k, entr_rate, detr_rate,
                zbuoyz_here, pdmfup, ov_cond + plude_k, uu_pub, vu_pub,
                ent, peff_k,
            )
            return next_carry, y_out

        def skip_updraft():
            is_seed = k == kseed_safe
            next_carry = (
                mfu_b, tu_b, qu_b, lu_b, uu_b, vu_b, buoy_b, cp_b,
                zbuoy_accum, zmfuu, zmfuv, z0, kctop_c, alive, first_test,
            )
            y_out = (
                jnp.where(is_seed, tu_seed, tenh_k),
                jnp.where(is_seed, qu_seed, qenh_k),
                jnp.where(is_seed, lu_seed, z0),
                jnp.where(is_seed, mfu_seed, z0),
                z0,
                z0,
                jnp.where(is_seed, buoy_seed_loaded, z0),
                z0,
                ov_cond,
                jnp.where(is_seed, uu_seed, z0),
                jnp.where(is_seed, vu_seed, z0),
                z0,
                z0,
            )
            return next_carry, y_out

        return lax.cond(should_compute, compute_updraft, skip_updraft)

    # Scan bottom to top.
    (*_rest, kctop, _alive, first_test), ys = lax.scan(
        updraft_step, initial, level_inputs, reverse=True,
    )
    (tu_p, qu_p, lu_p, mfu_p, entr_p, detr_p,
     buoy_p, pdmfup_p, plude_p, uu_p, vu_p, dmfen_p, peff_p) = ys
    ldcum_exact = (kctop != nlev - 2).astype(dtype)
    decided_by_first_test = (~is_midlevel) & (kbase == nlev - 2)
    ldcum_weight = jnp.where(decided_by_first_test, first_test, ldcum_exact)
    mid_seed = is_midlevel & (levels == kseed)
    return UpdatedraftState(
        tu=tu_p,
        qu=qu_p,
        lu=jnp.where(mid_seed, 0.0, lu_p),
        mfu=jnp.where(mid_seed, 0.0, mfu_p),
        entr=entr_p,
        detr=detr_p,
        buoy=buoy_p,
        pdmfup=pdmfup_p,
        plude=plude_p,
        uu=uu_p,
        vu=vu_p,
        dmfen=dmfen_p,
        peff=peff_p,
        kctop=kctop,
        ldcum_weight=ldcum_weight,
    )
