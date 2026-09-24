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

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple

import jcm.constants as c
from .tiedtke_nordeng import ConvectionParameters
from .half_levels import (
    HalfLevelEnvironment,
    half_level_environment,
    reconstruct_pressure_half,
)
# The ECHAM cuadjtq-style damped Newton adjustment. It lives in
# jcm.physics.convection.saturation so that ``calculate_cape_cin`` (in
# tiedtke_nordeng.py, which this module imports from) can call it too;
# re-exported here under its historical name for callers and tests.
from jcm.physics.convection.saturation import (
    cuadjtq_newton as saturation_adjustment,
)
from jcm.physics.thermodynamics import moist_isobaric_heat_capacity



from jcm.physics.thermodynamics import saturation_specific_humidity_and_derivative


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


def cubase_parcel(env: HalfLevelEnvironment, kbase: jnp.ndarray):
    """ECHAM ``cubase`` parcel at the cloud-base interface ``kbase``.

    The parcel leaves the lowest interface (the top of the bottom layer)
    with that interface's environment (``ptu = ptenh(klev)``,
    ``pqu = pqenh(klev)``, cuini) and is walked up the interfaces
    conserving ``pcpcu·T + pgeoh`` (mo_cuinitialize.f90:294), which
    telescopes to a single DSE lift. The static energy it carries is the
    one ``ptenh(klev)`` was defined with, the bottom full level's
    ``pcpen·pten + pgeo``: ECHAM re-forms it as ``pcpcu(klev)·ptenh(klev)``
    with the half-level heat capacity, which does not conserve the seed's
    static energy (by ``Δcp/cp`` of the lowest two levels, ~0.1 K); jcm
    carries the defined energy. At the base it is saturation adjusted
    with the damped condensation-only Newton step at the interface pressure
    (``cuadjtq`` kcall = 1, lines 296-314); the condensate stays in the
    plume, so total water is conserved.

    Returns:
        ``(tu, qu, lu)`` at interface ``kbase``.

    """
    t_dry = (env.dse[-1] - env.geoh[kbase]) / env.cpcu[kbase]
    return saturation_adjustment(t_dry, env.qenh[-1], env.paph[kbase])


def cubasmc_seed_temperature(cp_moist, env, kk):
    """ECHAM ``cubasmc`` seed at the bottom interface of layer ``kk``.

    ``ptu(kk+1) = (pcpen(kk)·pten(kk) + pgeo(kk) − pgeoh(kk+1))/pcpen(kk)``
    (mo_cuascent.f90:641-642): the full-level environment brought
    dry-adiabatically down to the layer's bottom interface.
    """
    return (env.dse[kk] - env.geoh[kk + 1]) / cp_moist[kk]



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
    zgam = zalvs / env.cpcu * dqsdt
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
    return env.cpcu[kcbot] * tu + env.geoh[kcbot] + zalvs * qu


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
    up = jnp.maximum(levels - 1, 0)
    zalvs = jnp.where(env.tenh > c.tmelt, c.alhc, c.alhs)
    zroi = (c.rd * env.tenh * (1.0 + c.vtmpc1 * env.qenh)
            / jnp.maximum(env.paph[:-1], 1.0))
    zdz = env.dp[up] * zroi / c.grav
    za2 = env.geo[up] - env.geo
    safe_za2 = jnp.where(levels >= 1, za2, 1.0)
    za1 = (cp_moist[up] * temperature[up] - cp_moist * temperature
           + zalvs * (humidity[up] - humidity) + za2) * c.grav
    zdhdz = za1 / safe_za2
    zdepth = env.geoh - env.geoh[kcbot]
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



def _rescaled_sigmoid(x, width):
    """Gate ``x`` smoothly: exactly 0 for ``x ≤ 0``, → 1 above.

    ``(σ((x − w)/w) − σ(−1))/(1 − σ(−1))``, clipped at zero: 0 at and below
    zero, 0.32 at one width, 0.9998 at ten; width → 0 recovers ``x > 0``.
    """
    s0 = jax.nn.sigmoid(-1.0)
    return jnp.maximum(
        (jax.nn.sigmoid((x - width) / width) - s0) / (1.0 - s0), 0.0)


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
    land_fraction: jnp.ndarray = jnp.array(0.0),
    type_weights: jnp.ndarray | None = None,
    lift: jnp.ndarray = jnp.array(0.0),
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
            ocean, 1=land). Selects ECHAM's per-surface ``zdnoprc``
            threshold via ``config.cu_dnoprc_ocean`` and
            ``config.cu_dnoprc_land``.
        type_weights: Smooth (deep, shallow, mid) weights; ``None`` falls
            back to the one-hot of ``ktype``.
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
    # Linear blend of ocean/land precip-zone threshold by land fraction —
    # smooth in land_fraction so the column gradient is well-defined.
    zdnoprc_col = (
        (1.0 - land_fraction) * config.cu_dnoprc_ocean
        + land_fraction * config.cu_dnoprc_land
    )
    if type_weights is None:
        type_weights = jnp.stack([
            jnp.asarray(ktype == 1, dtype=dtype),
            jnp.asarray(ktype == 2, dtype=dtype),
            jnp.asarray(ktype == 3, dtype=dtype),
        ])
    is_midlevel = (ktype == 3)
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
    tu_seed = jnp.where(is_midlevel, t_mid, tu_cb)
    qu_seed = jnp.where(is_midlevel, humidity[kbase], qu_cb)
    lu_seed = jnp.where(is_midlevel, 0.0, lu_cb)
    # The heat capacity the seed's static-energy flux is formed with: the
    # half-level ``pcpcu`` for a cubase plume, whose seed is the adjusted
    # parcel AT kcbot; for the cubasmc seed, the ``pcpen(kk)`` of the level
    # it was taken from, so the flux carries exactly the static energy
    # ``pcpen(kk)·pten(kk) + pgeo(kk)`` that defines it. (ECHAM forms that
    # flux with ``pcpen(kk+1)``, the level BELOW, mo_cuascent.f90:648 —
    # which does not conserve the seed's static energy and shifts its first
    # step by ``Δcp/cp`` of the two levels, over a kelvin across a humidity
    # jump; jcm carries the defined energy.)
    cp_seed = jnp.where(is_midlevel, cp_moist[kbase], env.cpcu[kseed_safe])
    cp_plume = env.cpcu.at[kseed_safe].set(cp_seed)

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
    uu_seed = jnp.where(is_midlevel, u_wind[kbase], u_sub)
    vu_seed = jnp.where(is_midlevel, v_wind[kbase], v_sub)

    # cuini initialises every plume property to the half-level environment.
    tu_init = env.tenh.at[kseed_safe].set(tu_seed)
    qu_init = env.qenh.at[kseed_safe].set(qu_seed)
    lu_init = jnp.zeros(nlev, dtype).at[kseed_safe].set(lu_seed)
    mfu_init = jnp.zeros(nlev, dtype).at[kseed_safe].set(mass_flux_base)
    uu_init = jnp.zeros(nlev, dtype).at[kseed_safe].set(uu_seed)
    vu_init = jnp.zeros(nlev, dtype).at[kseed_safe].set(vu_seed)
    # ``zbuoyz`` at the seed (cuasc evaluates it at kcbot after the
    # adjustment there) for the first step's organized entrainment.
    buoy_seed = (
        c.grav * (tu_seed - env.tenh[kseed_safe]) / env.tenh[kseed_safe]
        + c.grav * c.vtmpc1 * (qu_seed - env.qenh[kseed_safe])
    )
    buoy_init = jnp.zeros(nlev, dtype).at[kseed_safe].set(
        buoy_seed - c.grav * lu_seed)
    # The Nordeng integrated buoyancy ``zbuoy`` as cuasc has it when the
    # plume leaves cloud base: initialised to the condensate-free cloud-base
    # buoyancy (mo_cuascent.f90:250-251), then — because cuasc's level loop
    # also visits the sub-cloud interfaces, where the plume is the dry
    # cubase parcel carrying the lowest interface's static energy and
    # humidity — incremented by ``max(zbuoyz, 0)·zdz`` at every interface
    # from ``klevm1`` down to ``kcbot + 1`` (lines 516-523), with
    # ``zdz = (pgeo(jk−1) − pgeo(jk))/g``. The cloud-base interface's own
    # term is added by the first ascent step below. ECHAM forms it for deep
    # (cubase) plumes; the smooth deep weight scales its use.
    geo_above = jnp.concatenate([env.geo[:1], env.geo[:-1]])
    zdz_above = (geo_above - env.geo) / c.grav
    t_sub = (env.dse[-1] - env.geoh) / env.cpcu
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
    kp1 = jnp.minimum(levels + 1, nlev - 1)
    zrrho = c.rd * env.tenh[kp1] * (1.0 + c.vtmpc1 * env.qenh[kp1]) / env.paph[
        levels + 1]
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
        - c.grav / (c.rd * env.tenh[kp1] * (1.0 + c.vtmpc1 * env.qenh[kp1]))
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
    p_base = env.paph[kbase]
    zpmid = 0.5 * (p_base + env.paph[kctop0])
    paph_k = env.paph[:-1]
    icond1 = paph_k > zpmid
    icond2 = (p_base - paph_k) <= _SHALLOW_ENTRAINMENT_DEPTH
    iklwmin = jnp.maximum(klwmin, kctop0 + 2)
    below_klw = levels >= iklwmin
    icond3 = env.qenh[kp1] > _ZENTEST_MIN_Q
    entr_gate = (w_deep * (below_klw | icond1)
                 + w_shallow * (icond2 | icond1)
                 + w_mid * below_klw)
    zentest_on = w_mid * (below_klw & icond3)
    # ``zentest`` (lines 756-760): the mid-level plume entrains the
    # pre-convection moisture convergence of the layer, as a fractional rate
    # ``max(pqte, 0)/pqenh / (pmfu·zrrho)`` capped at ``centrmax``.
    q_below = jnp.where(icond3, env.qenh[kp1], 1.0)
    zentest_rate = jnp.maximum(moisture_tendency, 0.0) / q_below

    # --- cuentr: organized detrainment (mo_cuascent.f90:767-788) ----------
    # Deep plumes only, from ``khmin`` up to ``kctop0``: the ``tan`` profile
    # in height above ``khmin``, capped at ``centrmax``, applied over
    # ``dz_p``.
    hhatt = saturated_mse_hat(env)
    hcbase = cloud_base_mse(env, kbase, tu_cb, qu_cb)
    if khmin is None:
        khmin = mse_minimum_level(env, temperature, humidity, cp_moist,
                                  kbase, kctop0)
    ztmzk = (env.geoh[kctop0] - env.geoh[khmin]) / c.grav
    org_on = (levels <= khmin) & (levels >= kctop0) & (khmin > kctop0)
    safe_ztmzk = jnp.where(org_on, ztmzk, 1.0)
    zzmzk = (env.geoh - env.geoh[khmin]) / c.grav
    zarg = _ECHAM_PI * jnp.clip(zzmzk / safe_ztmzk, 0.0, 1.0) * 0.5
    zorgde = jnp.tan(zarg) * _ECHAM_PI * 0.5 / safe_ztmzk
    org_detr_rate = jnp.where(
        org_on, jnp.minimum(zorgde, config.cu_centrmax), 0.0)
    # ``zodmax`` limits it below ``khmin`` (lines 371-383).
    zodmax_on = levels <= khmin

    level_inputs = (
        levels, env.tenh, env.qenh, paph_k,
        dz_p, dz_g, zdrodz, zdz_full, mfmax,
        u_wind, v_wind, entr_gate, zentest_on, zentest_rate, zrrho,
        org_detr_rate, zodmax_on,
    )
    entr_base_blend = jnp.maximum(
        w_deep * config.entrpen + w_shallow * config.entrscv
        + w_mid * config.entrmid, 0.0)
    cmfctop = config.cu_cmfctop

    zero = jnp.zeros(nlev, dtype)
    updraft_init = UpdatedraftState(
        tu=tu_init, qu=qu_init, lu=lu_init,
        mfu=mfu_init, entr=zero, detr=zero,
        buoy=buoy_init, pdmfup=zero, plude=zero,
        uu=uu_init, vu=vu_init, dmfen=zero,
    )
    # Carry: the published profiles, the CONTINUING plume's flux, condensate
    # and winds (which differ from the published ones where part of the plume
    # overshoots), the Nordeng integrated buoyancy ``zbuoy``, the running
    # plume momentum fluxes, the condensate the last overshoot leaves for
    # the layer above it, and the realized cloud top ``kctop``: the highest
    # interface where the plume passed the ascent test, which starts at
    # ECHAM's ``klevm1`` — or, for a cubase plume, at its cloud base, whose
    # test in cuasc repeats cubase's.
    kctop_init = jnp.where(is_midlevel, nlev - 2, kbase).astype(jnp.int32)
    initial = (
        updraft_init, mfu_init, lu_init, uu_init, vu_init,
        jnp.asarray(zbuoy_init, dtype),
        jnp.asarray(mass_flux_base * uu_seed, dtype),
        jnp.asarray(mass_flux_base * vu_seed, dtype),
        jnp.zeros((), dtype),
        kctop_init,
    )

    def updraft_step(carry_tuple, inputs):
        (st, mfa, lua, uua, vua, zbuoy_accum, zmfuu, zmfuv,
         ov_cond, kctop_c) = carry_tuple
        (k, tenh_k, qenh_k, paph_k_, dzp, dzg, zdrodz_k, zdz_k, mfmax_k,
         env_u, env_v, gate_k, zentest_on_k, zentest_rate_k, zrrho_k,
         org_rate_k, zodmax_on_k) = inputs

        # The condensate of an overshoot that ended at the interface below
        # is detrained in this layer (``plude(jk−1) = pmful(jk)``,
        # mo_cuascent.f90:557-559).
        st_dep = st._replace(plude=st.plude.at[k].add(ov_cond))
        should_compute = (k >= 1) & (k < kseed)

        def compute_updraft():
            b = jnp.minimum(k + 1, nlev - 1)          # interface below (k+1)
            mfu_b = mfa[b]
            tu_b, qu_b, lu_b = st.tu[b], st.qu[b], lua[b]
            uu_b, vu_b = uua[b], vua[b]
            s_b = cp_plume[b] * tu_b + env.geoh[b]
            s_e = env.cpcu[b] * env.tenh[b] + env.geoh[b]
            q_e = env.qenh[b]
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
            zbuoyz = jnp.maximum(st.buoy[b], 0.0)
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
            zmse = env.cpcu[b] * tu_b + zalvs_u * qu_b + env.geoh[b]
            znevn = ((env.geoh[kctop0] - env.geoh[b])
                     * (zmse - hhatt[b]) / c.grav)
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
            tenh_b, qsenh_b, qenh_b = env.tenh[b], env.qsenh[b], env.qenh[b]
            zalvs_e = jnp.where(tenh_b > c.tmelt, c.alhc, c.alhs)
            zga = zalvs_e * qsenh_b / (c.rv * tenh_b ** 2)
            zdt = ((lu_b - c.vtmpc1 * (qsenh_b - qenh_b))
                   / (1.0 / tenh_b + c.vtmpc1 * zga))
            zscod = jnp.maximum(
                env.cpcu[b] * tenh_b + env.geoh[b] + env.cpcu[b] * zdt, 0.0)
            zqcod = jnp.maximum(qsenh_b + zga * zdt, 0.0)
            ent = zdmfen + zoentr
            s_flux = mfu_b * s_b + ent * s_e - zdmfde * s_b - zodetr * zscod
            q_flux = mfu_b * qu_b + ent * q_e - zdmfde * qu_b - zodetr * zqcod
            l_flux = mfu_b * lu_b - (zdmfde + zodetr) * lu_b
            plude_mix = lu_b * (zdmfde + zodetr)
            t_mix = jnp.clip(
                (s_flux / mfu_div - env.geoh[k]) / env.cpcu[k], 100.0, 400.0)
            q_mix = q_flux / mfu_div
            l_mix = l_flux / mfu_div

            # Condensation-only saturation adjustment at the interface
            # pressure (cuadjtq kcall = 1); the condensed vapour joins the
            # condensate the plume already carries.
            tu_new, qu_new, cond = saturation_adjustment(t_mix, q_mix, paph_k_)
            lu_new = l_mix + cond

            # The ascent test (lines 442-466). The plume continues through
            # this interface only if it CONDENSED here (``pqu < zqold``) and
            # is then buoyant — virtual temperature of the condensate-loaded
            # plume against the half-level environment, with ECHAM's sub-grid
            # ``zlift`` where the interface below is still ``klab == 1``
            # (the first step of a mid-level plume) — carries at least 1 % of
            # the cloud-base flux, and lies at or below the cloud-top bound
            # ``kctop0``. Each continuous test is a smooth gate whose width
            # is a parameter (width → 0 recovers the hard test); the product
            # is the continuing fraction ``s``. ``buoy_test`` is an
            # acceleration, ``lift`` a temperature, so the bonus converts
            # with g/Tv_env.
            tv_env = tenh_k * (1.0 + c.vtmpc1 * qenh_k)
            buoy_test = c.grav * (
                tu_new * (1.0 + c.vtmpc1 * qu_new - lu_new) - tv_env
            ) / tv_env
            first_mid_step = is_midlevel & (k == kbase)
            lift_accel = jnp.where(first_mid_step, c.grav * lift / tv_env, 0.0)
            surv_cond = _rescaled_sigmoid(cond, config.smooth_term_cond)
            surv_buoy = jax.nn.sigmoid(
                (buoy_test + lift_accel) / config.smooth_term_buoy)
            surv_mf = jax.nn.sigmoid(
                (mfu_new / jnp.maximum(mass_flux_base, 1e-10) - 0.01)
                / config.smooth_term_mf
            )
            in_bound = (k >= kctop0) & (k >= 2)
            s = jnp.where(in_bound, surv_cond * surv_buoy * surv_mf, 0.0)

            # Precipitation of the continuing plume (lines 454-457):
            #   zlnew  = plu / (1 + cprcon·(pgeoh(jk) − pgeoh(jk+1)))
            #   pdmfup = max(0, (plu − zlnew)·pmfu)
            # only where the interface is more than ``zdnoprc`` above cloud
            # base (ECHAM ``zpbase − paphp1(jk) ≥ zdnoprc``, ocean/land
            # thresholds blended by land fraction). The gate is a sigmoid over
            # the depth excess so the thresholds keep gradients; width → 0
            # recovers the hard gate.
            precip_zone_w = jax.nn.sigmoid(
                ((p_base - paph_k_) - zdnoprc_col) / config.smooth_precip_pa
            )
            lu_after = lu_new / (
                1.0 + config.cprcon * precip_zone_w * c.grav * dzg)
            pdmfup = s * jnp.maximum((lu_new - lu_after) * mfu_new, 0.0)

            # The cloud-top overshoot (lines 540-565): where the ascent test
            # fails, a fraction ``cmfctop`` of the flux reaching the interface
            # below carries on to this interface with its properties here
            # (condensed but not precipitated), the rest detrains in this
            # layer with the condensate of the plume that entered it, and the
            # overshooting condensate detrains in the layer above.
            ov = (1.0 - s) * cmfctop * mfu_b
            mfa_k = s * mfu_new
            mfu_k = mfa_k + ov
            mfu_k_div = jnp.maximum(mfu_k, config.cmfcmin)
            lu_pub = (mfa_k * lu_after + ov * lu_new) / mfu_k_div
            plude_k = (s * plude_mix
                       + (1.0 - s) * (1.0 - cmfctop) * mfu_b * lu_b)

            # Prognostic plume wind (cuasc lines 486-506): a running momentum
            # flux that entrains environmental momentum and detrains plume
            # momentum, both enhanced by ``zz·zdmfde`` — zz = 2 (3 when the
            # layer does not entrain) for deep and mid-level plumes, 0 (1) for
            # shallow — with the detrained part capped at 0.75 of the
            # entering flux; the wind is that flux over the plume flux. The
            # overshoot keeps the wind of the interface below (line 580).
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

            # ``zbuoyz`` of the continuing plume for the organized entrainment
            # of the next layer up (lines 516-518).
            zbuoyz_here = (
                c.grav * (tu_new - tenh_k) / tenh_k
                + c.grav * c.vtmpc1 * (qu_new - qenh_k)
                - c.grav * lu_after
            )
            # Diagnostic per-metre rates, each part over the depth ECHAM
            # forms it with: the turbulent and organized detrainment and the
            # turbulent entrainment over ``dz_p``, the Nordeng organized
            # entrainment over the geometric ``dz_g`` — so each is bounded
            # by its own rate parameter.
            has_flux = mfu_b * dzp > 0.0
            den_p = jnp.where(has_flux, mfu_b * dzp, 1.0)
            den_g = jnp.where(has_flux & (dzg > 0.0), mfu_b * dzg, 1.0)
            entr_rate = jnp.where(has_flux, zdmfen / den_p + zoentr / den_g,
                                  0.0)
            detr_rate = jnp.where(has_flux, zdmfde_m / den_p, 0.0)

            new_st = st_dep._replace(
                tu=st.tu.at[k].set(tu_new),
                qu=st.qu.at[k].set(qu_new),
                lu=st.lu.at[k].set(lu_pub),
                mfu=st.mfu.at[k].set(mfu_k),
                entr=st.entr.at[k].set(entr_rate),
                detr=st.detr.at[k].set(detr_rate),
                buoy=st.buoy.at[k].set(zbuoyz_here),
                pdmfup=st.pdmfup.at[k].set(pdmfup),
                plude=st_dep.plude.at[k].add(plude_k),
                uu=st.uu.at[k].set(uu_pub),
                vu=st.vu.at[k].set(vu_pub),
                # The whole entrainment of the layer, also where the plume
                # terminates: the overshoot carries the mixed air (cuasc
                # forms ``pxtu`` before the ascent test and section 5 moves
                # it up), so the tracer ledger must see the environmental
                # air that went into it.
                dmfen=st.dmfen.at[k].set(ent),
            )
            # The continuing plume carries on; its momentum flux scales with
            # the continuing fraction.
            # The interface passed where the majority of the plume continues
            # (exactly the hard test as the gate widths → 0).
            kctop_new = jnp.where(s > 0.5, k, kctop_c).astype(jnp.int32)
            return (new_st, mfa.at[k].set(mfa_k), lua.at[k].set(lu_after),
                    uua.at[k].set(uu_a), vua.at[k].set(vu_a),
                    zbuoy_new, zmfuu_new * s, zmfuv_new * s, ov * lu_new,
                    kctop_new)

        updated = lax.cond(
            should_compute,
            compute_updraft,
            lambda: (st_dep, mfa, lua, uua, vua, zbuoy_accum, zmfuu, zmfuv,
                     jnp.zeros((), dtype), kctop_c),
        )
        return updated, None

    # Scan bottom to top.
    (final_state, *_rest, kctop), _ = lax.scan(
        updraft_step, initial, level_inputs, reverse=True,
    )
    # Without a passing interface ``kctop`` stays ECHAM's ``klevm1``, which
    # marks the column non-convective.
    # The mid-level seed interface lies below the cloud base; its flux is
    # the cuflx sub-cloud taper's to set, so the returned plume profile —
    # like a cubase plume's — starts at kcbot.
    mid_seed = is_midlevel & (levels == kseed)
    return final_state._replace(
        mfu=jnp.where(mid_seed, 0.0, final_state.mfu),
        lu=jnp.where(mid_seed, 0.0, final_state.lu),
        kctop=kctop,
    )
