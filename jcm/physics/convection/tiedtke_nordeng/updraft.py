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


#: Ceiling on the organized-detrainment tan-profile fractional height
#: ``zzmzk/ztmzk``. ECHAM's ``tan(π·frac/2)`` diverges at ``frac == 1``
#: (cloud top); the ``centrmax`` cap below already saturates the top layers
#: for any physical cloud depth, so clipping the argument strictly below
#: π/2 changes nothing physically while keeping ``tan`` — and its VJP —
#: finite (no ``0·inf`` gradient poison, jax-gcm#558/#559).
_ORG_DETR_FRAC_MAX = 0.98

#: Plume mass flux [kg/m²/s] below which a level carries no plume: its
#: properties are reported as the environment's instead of a ratio of
#: vanishing fluxes.
_MFU_NEGLIGIBLE = 1e-6


class UpdatedraftState(NamedTuple):
    """Half-level updraft profiles (entry ``k`` = top interface of layer ``k``)."""

    tu: jnp.ndarray      # Updraft temperature (K) — ECHAM ``ptu``
    qu: jnp.ndarray      # Updraft specific humidity (kg/kg) — ``pqu``
    lu: jnp.ndarray      # Updraft condensate (kg/kg) after per-layer precip
                         # removal — ``plu``
    mfu: jnp.ndarray     # Updraft mass flux (kg/m²/s) — ``pmfu``; the plume
                         # profile from cloud base up (zero below it: the
                         # sub-cloud taper is cuflx's, applied by the ledger)
    entr: jnp.ndarray    # Fractional entrainment rate (1/m) of layer k
    detr: jnp.ndarray    # Fractional detrainment rate (1/m) of layer k
    buoy: jnp.ndarray    # Plume buoyancy ``zbuoyz`` (m/s²), condensate-loaded
    pdmfup: jnp.ndarray  # Precip generated in layer k (kg/m²/s) — ``pdmfup``
    plude: jnp.ndarray   # Condensate DETRAINED in layer k (kg/m²/s) — ECHAM
                         # ``plude``. Feeds the stratiform cloud tracers (ECHAM
                         # pxtecl/pxteci via zxtec = g/Δp·plude) and the cudtdq
                         # latent-heat ledger; includes the cloud-top dump of
                         # the remaining plume condensate when the updraft
                         # terminates.
    uu: jnp.ndarray      # Updraft zonal wind (m/s) — ECHAM ``puu`` (cuasc).
                         # Consumed by cududv's momentum-transport deviation
                         # flux ``mfu·(uu − ū)``.
    vu: jnp.ndarray      # Updraft meridional wind (m/s) — ECHAM ``pvu``.
    dmfen: jnp.ndarray | None = None  # Absolute entrainment into the plume
                         # in layer k (kg/m²/s) — ECHAM ``zdmfen + zoentr``;
                         # the ledger the convective tracer transport reads.


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
) -> UpdatedraftState:
    """Calculate the updraft on half levels (ECHAM ``cuasc``).

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
        ktop: Scan ceiling (interface index); the realized top is set by
            the dynamic termination below.
        ktype: Convection type (1 deep, 2 shallow, 3 mid-level).
        mass_flux_base: Cloud base mass flux (kg/m²/s)
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
    levels = jnp.arange(nlev)
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
    # from ``klevm1`` down to ``kcbot + 1`` (lines 516-520), with
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

    updraft_init = UpdatedraftState(
        tu=tu_init, qu=qu_init, lu=lu_init,
        mfu=mfu_init, entr=jnp.zeros(nlev, dtype),
        detr=jnp.zeros(nlev, dtype),
        buoy=buoy_init,
        pdmfup=jnp.zeros(nlev, dtype),
        plude=jnp.zeros(nlev, dtype),
        uu=uu_init, vu=vu_init,
        dmfen=jnp.zeros(nlev, dtype),
    )
    # Carry = (updraft_state, Nordeng integrated buoyancy ``zbuoy``, and the
    # running plume momentum fluxes ``zmfuu``/``zmfuv``).
    initial_state = (
        updraft_init,
        jnp.asarray(zbuoy_init, dtype),
        jnp.asarray(mass_flux_base * uu_seed, dtype),
        jnp.asarray(mass_flux_base * vu_seed, dtype),
    )

    # Type-blended base entrainment and deep-convection weight. With the
    # smooth type selection (tiedtke_nordeng.py) the per-type entrainment
    # rates combine by the softmax weights instead of a hard ktype
    # select, so entrpen/entrscv/entrmid keep gradients across the type
    # thresholds; w_deep likewise gates the Nordeng organized
    # entrainment/detrainment smoothly.
    entr_base_blend = (type_weights[0] * config.entrpen
                       + type_weights[1] * config.entrscv
                       + type_weights[2] * config.entrmid)
    w_deep = type_weights[0]
    w_deep_or_mid = type_weights[0] + type_weights[2]

    # Per-layer geometry, all from the half-level environment:
    #   dz_p  — the layer thickness ECHAM's turbulent entrainment/detrainment
    #           uses, Δp·(1/ρ at the layer's bottom interface)/g (cuentr
    #           ``zdprho·zrrho``, mo_cuascent.f90:709-711);
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
    heights = env.geoh / c.grav

    level_inputs = (
        levels, env.tenh, env.qenh, env.paph[:-1],
        dz_p, dz_g, zdrodz, zdz_full, mfmax, heights,
        u_wind, v_wind,
    )
    z_base = heights[kbase]
    z_top = heights[ktop]
    p_base = env.paph[kbase]

    def updraft_step(carry_tuple, inputs):
        carry, zbuoy_accum, zmfuu, zmfuv = carry_tuple
        (k, tenh_k, qenh_k, paph_k, dzp, dzg, zdrodz_k, zdz_k, mfmax_k,
         z_k, env_u, env_v) = inputs

        # Levels the ascent computes: from just above the seed up to the scan
        # ceiling.
        should_compute = (k >= ktop) & (k < kseed)

        def compute_updraft():
            b = jnp.minimum(k + 1, nlev - 1)          # interface below (k+1)
            mfu_b = carry.mfu[b]
            tu_b, qu_b, lu_b = carry.tu[b], carry.qu[b], carry.lu[b]
            s_b = cp_plume[b] * tu_b + env.geoh[b]
            s_e = env.cpcu[b] * env.tenh[b] + env.geoh[b]
            q_e = env.qenh[b]
            # cuentr only mixes above cloud base (``kk < kcbot``): the first
            # step of a mid-level plume crosses layer kcbot unmixed.
            mixes = k < kbase

            # Turbulent entrainment is ECHAM cuentr's fractional rate
            # (mo_cuascent.f90:746) and turbulent detrainment equals it
            # (δ = ε). jcm applies both over the whole cloud: cuentr's
            # vertical gating of the ENTRAINMENT (deep: below the
            # maximum-ascent level or in the lower half of the cloud; shallow:
            # within 200 hPa of the base or in the lower half) and its
            # mid-level moisture-convergence enhancement ``zentest``
            # (lines 756-760) are not ported — documented in the science
            # description as open deviations.
            entr_turb = jnp.clip(entr_base_blend, 0.0, 0.01)
            # Nordeng (1994) organized entrainment for deep convection
            # (mo_cuascent.f90:517-526): the positive plume buoyancy at the
            # interface below over one plus the integrated buoyancy, plus the
            # density-scale-height gradient, clamped to [0, centrmax] in
            # ECHAM's order. The accumulator adds this interface's buoyancy
            # over the full-level spacing above it before the rate is formed.
            zbuoyz = jnp.maximum(carry.buoy[b], 0.0)
            zbuoy_new = zbuoy_accum + zbuoyz * zdz_k
            zoentr = jnp.clip(
                zbuoyz * 0.5 / (1.0 + zbuoy_new) + zdrodz_k,
                0.0, config.cu_centrmax,
            )
            entr_org = w_deep * zoentr
            # Organized detrainment (mo_cuascent.f90:769-784): a tan profile in
            # height whose prefactor scales as 1/(cloud depth), capped at
            # centrmax, with heights of the half levels. ECHAM's lower bound
            # is ``khmin`` (the MSE-minimum onset level); jcm's single-pass
            # ascent uses cloud base, where tan(0) = 0 keeps near-base
            # detrainment negligible.
            ztmzk = jnp.maximum(z_top - z_base, 1.0)
            zzmzk = jnp.clip(z_k - z_base, 0.0, ztmzk)
            frac_height = jnp.minimum(zzmzk / ztmzk, _ORG_DETR_FRAC_MAX)
            zorgde = jnp.tan(jnp.pi * frac_height * 0.5) * jnp.pi * 0.5 / ztmzk
            detr_org = w_deep * jnp.minimum(zorgde, config.cu_centrmax)

            dmf_entr = jnp.where(
                mixes, entr_turb * mfu_b * dzp + entr_org * mfu_b * dzg, 0.0)
            # cuasc caps the detrained mass at 0.75 of the plume entering the
            # layer (line 360), so plude can never exceed the plume.
            dmf_detr = jnp.where(
                mixes,
                jnp.minimum((entr_turb + detr_org) * mfu_b * dzp,
                            0.75 * mfu_b),
                0.0,
            )
            # ``zmfmax`` limiter (lines 354-358): entrainment is cut so the
            # flux leaving the interface never exceeds the air mass of the
            # layer above per step.
            mfu_test = mfu_b + dmf_entr - dmf_detr
            excess = jnp.maximum(mfu_test - jnp.minimum(mfu_test, mfmax_k), 0.0)
            dmf_entr = jnp.maximum(dmf_entr - excess, 0.0)
            mfu_new = jnp.maximum(mfu_b + dmf_entr - dmf_detr, 0.0)
            has_plume = mfu_new > _MFU_NEGLIGIBLE
            mfu_div = jnp.maximum(mfu_new, config.cmfcmin)

            # Flux-form mixing (mo_cuascent.f90:398-411): the flux arriving
            # from below, plus the entrained half-level environment, minus
            # the detrained plume air — detrainment leaves at the properties
            # of the plume that ENTERED the layer (``zscde`` uses ptu(jk+1)).
            # Dry static energy ``pcpcu·T + pgeoh`` is what is conserved; the
            # plume temperature at this interface follows from it with this
            # interface's ``pcpcu`` and ``pgeoh``. Vapour and condensate are
            # carried separately, exactly as ``pmfuq``/``pmful``: condensate
            # is never evaporated back inside the plume.
            s_mix = (mfu_b * s_b + dmf_entr * s_e - dmf_detr * s_b) / mfu_div
            q_mix = (mfu_b * qu_b + dmf_entr * q_e - dmf_detr * qu_b) / mfu_div
            l_mix = (mfu_b * lu_b - dmf_detr * lu_b) / mfu_div
            t_mix = jnp.clip((s_mix - env.geoh[k]) / env.cpcu[k], 100.0, 400.0)
            t_mix = jnp.where(has_plume, t_mix, tenh_k)
            q_mix = jnp.where(has_plume, q_mix, qenh_k)
            l_mix = jnp.where(has_plume, l_mix, 0.0)

            # Condensation-only saturation adjustment at the interface
            # pressure (cuadjtq kcall = 1); the condensed vapour joins the
            # condensate the plume already carries.
            tu_new, qu_new, cond = saturation_adjustment(t_mix, q_mix, paph_k)
            lu_new = l_mix + cond

            # Buoyancy test of the ascent (line 446): virtual temperature of
            # the condensate-loaded plume against the half-level environment,
            # BEFORE this layer's precipitation is removed.
            tv_env = tenh_k * (1.0 + c.vtmpc1 * qenh_k)
            buoy_test = c.grav * (
                tu_new * (1.0 + c.vtmpc1 * qu_new - lu_new) - tv_env
            ) / tv_env

            # Per-layer precipitation (lines 454-457):
            #   zlnew  = plu / (1 + cprcon·(pgeoh(jk) − pgeoh(jk+1)))
            #   pdmfup = max(0, (plu − zlnew)·pmfu)
            # only where the interface is more than ``zdnoprc`` above cloud
            # base (ECHAM ``zpbase − paphp1(jk) ≥ zdnoprc``, ocean/land
            # thresholds blended by land fraction). The gate is a sigmoid over
            # the depth excess so the thresholds keep gradients; width → 0
            # recovers the hard gate.
            precip_zone_w = jax.nn.sigmoid(
                ((p_base - paph_k) - zdnoprc_col) / config.smooth_precip_pa
            )
            cprcon_eff = jnp.where(has_plume, config.cprcon * precip_zone_w, 0.0)
            lu_after = lu_new / (1.0 + cprcon_eff * c.grav * dzg)
            pdmfup = jnp.maximum((lu_new - lu_after) * mfu_new, 0.0)
            lu_new = lu_after

            # Detrained condensate (``plude = plu(jk+1)·zdmfde``): the mass
            # detrained in this layer carries the condensate of the plume
            # that entered it.
            plude_layer = lu_b * dmf_detr

            # Dynamic termination (review B.2.3, ECHAM lines 446-463): the
            # plume survives a level while buoyant and while its flux is at
            # least 1 % of the cloud-base value. The survival fraction is the
            # product of two sigmoids (widths → 0 recover the hard test); the
            # non-surviving fraction detrains its condensate here. ECHAM's
            # sub-grid bonus ``zlift`` applies where the interface below is
            # still ``klab == 1`` — only the first step of a mid-level plume,
            # whose seed ``cubasmc`` labels so (a cubase plume's seed is
            # ``klab == 2``). ``buoy_test`` is an acceleration, ``lift`` a
            # temperature, so the bonus converts with g/Tv_env.
            first_mid_step = is_midlevel & (k == kbase)
            lift_accel = jnp.where(first_mid_step, c.grav * lift / tv_env, 0.0)
            surv_buoy = jax.nn.sigmoid(
                (buoy_test + lift_accel) / config.smooth_term_buoy)
            surv_mf = jax.nn.sigmoid(
                (mfu_new / jnp.maximum(mass_flux_base, 1e-10) - 0.01)
                / config.smooth_term_mf
            )
            survival = surv_buoy * surv_mf
            # Forced total detrainment at the scan ceiling: a still-buoyant
            # plume reaching the supplied ``ktop`` dumps its remaining
            # condensate flux there (ECHAM's cloud-top ``plude = pmful``,
            # mo_cuascent.f90:540-563) rather than carrying it out through the
            # interface.
            survival = jnp.where(k == ktop, 0.0, survival)
            plude_layer = plude_layer + lu_new * mfu_new * (1.0 - survival)
            mfu_final = mfu_new * survival

            # Prognostic plume wind (cuasc lines 486-506): a running momentum
            # flux that entrains environmental momentum and detrains plume
            # momentum, both enhanced by ``zz·zdmfde`` — zz = 2 (3 when the
            # layer does not entrain) for deep and mid-level plumes, 0 (1) for
            # shallow — with the detrained part capped at 0.75 of the
            # entering flux; the wind is that flux over the plume flux.
            no_entr = dmf_entr <= 0.0
            zz = (w_deep_or_mid * jnp.where(no_entr, 3.0, 2.0)
                  + type_weights[1] * jnp.where(no_entr, 1.0, 0.0))
            zdmfeu = dmf_entr + zz * dmf_detr
            zdmfdu = jnp.minimum(dmf_detr + zz * dmf_detr, 0.75 * mfu_b)
            zmfuu_new = zmfuu + zdmfeu * env_u - zdmfdu * carry.uu[b]
            zmfuv_new = zmfuv + zdmfeu * env_v - zdmfdu * carry.vu[b]
            uu_new = jnp.where(mfu_final > 0.0, zmfuu_new / mfu_div, carry.uu[b])
            vu_new = jnp.where(mfu_final > 0.0, zmfuv_new / mfu_div, carry.vu[b])

            # ``zbuoyz`` for the organized entrainment of the next layer up,
            # after the precipitation (mo_cuascent.f90:517-518).
            zbuoyz_here = (
                c.grav * (tu_new - tenh_k) / tenh_k
                + c.grav * c.vtmpc1 * (qu_new - qenh_k)
                - c.grav * lu_new
            )

            new_state = carry._replace(
                tu=carry.tu.at[k].set(tu_new),
                qu=carry.qu.at[k].set(qu_new),
                lu=carry.lu.at[k].set(lu_new),
                mfu=carry.mfu.at[k].set(mfu_final),
                entr=carry.entr.at[k].set(
                    jnp.where(mixes, entr_turb + entr_org, 0.0)),
                detr=carry.detr.at[k].set(
                    jnp.where(mixes, entr_turb + detr_org, 0.0)),
                buoy=carry.buoy.at[k].set(zbuoyz_here),
                pdmfup=carry.pdmfup.at[k].set(pdmfup),
                plude=carry.plude.at[k].set(plude_layer),
                uu=carry.uu.at[k].set(uu_new),
                vu=carry.vu.at[k].set(vu_new),
                dmfen=carry.dmfen.at[k].set(dmf_entr),
            )
            # The momentum flux carried to the next layer is that of the
            # surviving plume (the wind is intensive; the flux scales with it).
            return (new_state, zbuoy_new,
                    zmfuu_new * survival, zmfuv_new * survival)

        updated = lax.cond(
            should_compute,
            compute_updraft,
            lambda: (carry, zbuoy_accum, zmfuu, zmfuv),
        )
        return updated, None

    # Scan bottom to top.
    (final_state, _zbuoy, _zmfuu, _zmfuv), _ = lax.scan(
        updraft_step, initial_state, level_inputs, reverse=True,
    )
    # The mid-level seed interface lies below the cloud base; its flux is
    # the cuflx sub-cloud taper's to set, so the returned plume profile —
    # like a cubase plume's — starts at kcbot.
    mid_seed = is_midlevel & (levels == kseed)
    return final_state._replace(
        mfu=jnp.where(mid_seed, 0.0, final_state.mfu),
        lu=jnp.where(mid_seed, 0.0, final_state.lu),
    )
