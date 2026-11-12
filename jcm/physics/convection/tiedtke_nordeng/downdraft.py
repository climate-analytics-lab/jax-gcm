"""Downdraft calculations for Tiedtke-Nordeng convection scheme

This module implements the downdraft calculations including:
- Level of free sinking (LFS) determination
- Downdraft entrainment and detrainment
- Evaporative cooling
- Moist descent

Based on ECHAM ``mo_cudescent.f90`` (``cudlfs``, ``cuddraf``).

Like the updraft, every profile lives on HALF levels: entry ``j`` is the
value at the TOP interface of layer ``j`` (physics-internal top-first frame,
see :mod:`~jcm.physics.convection.tiedtke_nordeng.half_levels`). The descent
to interface ``j`` crosses layer ``j - 1``, so the per-layer ledgers
(``pdmfdp``, ``dmfen``) of that step belong to layer ``j - 1``.
"""

import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.cuadjtq import (
    cuadjtq,
    cuadjtq_newton_evap,
)
from jcm.physics.thermodynamics import moist_isobaric_heat_capacity
from .tiedtke_nordeng import (
    ConvectionParameters
)
from .half_levels import HalfLevelEnvironment


class DowndraftState(NamedTuple):
    """Half-level downdraft profiles (entry ``j`` = top interface of layer ``j``)."""

    td: jnp.ndarray      # Downdraft temperature (K) — ECHAM ``ptd``
    qd: jnp.ndarray      # Downdraft specific humidity (kg/kg) — ``pqd``
    mfd: jnp.ndarray     # Downdraft mass flux (kg/m²/s), ≤ 0 — ``pmfd``
    pdmfdp: jnp.ndarray  # Downdraft precip sink of layer j (kg/m²/s, ≤ 0) —
                         # ECHAM ``pdmfdp = −pmfd·zcond``: rain evaporated
                         # into the descending parcel, debited from the rain
                         # flux and credited back to the environment through
                         # the cudtdq ledger.
    ud: jnp.ndarray      # Downdraft zonal wind (m/s) — ECHAM ``pud`` (cuddraf),
                         # mixed toward the entrained environment as the parcel
                         # descends. Consumed by cududv's SEPARATE downdraft
                         # momentum flux ``mfd·(ud − ū)``.
    vd: jnp.ndarray      # Downdraft meridional wind (m/s) — ECHAM ``pvd``.
    lfs: int             # Level of free sinking (interface index, ``kdtop``)
    active: bool         # Whether a downdraft was initiated (``lddraf``)
    dmfen: jnp.ndarray | None = None  # Magnitude of the environmental air
                         # entrained into the downdraft in layer j (kg/m²/s)
                         # — ECHAM ``|zdmfen|``; the tracer-transport ledger.


def wetbulb_temperature(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate wet-bulb temperature and humidity.

    ECHAM ``cuadjtq(kcall=2)`` — the evaporation-only damped Newton
    adjustment (see
    :func:`~jcm.physics.convection.tiedtke_nordeng.cuadjtq.cuadjtq_newton_evap`).
    Conserves moist static energy exactly (``cp·ΔT + L·Δq = 0``), and
    already-saturated air comes back unchanged because the evaporation-only
    clip zeroes the step.

    Args:
        temperature: Environmental temperature (K)
        humidity: Environmental humidity (kg/kg)
        pressure: Pressure (Pa)

    Returns:
        Tuple of (wetbulb_temp, wetbulb_humidity)

    """
    twb, qwb = cuadjtq_newton_evap(temperature, humidity, pressure)
    return twb.astype(temperature.dtype), qwb.astype(humidity.dtype)


def _env_or_build(env, temperature, humidity, pressure, layer_mass=None,
                  cp_moist=None, pressure_half=None):
    if env is not None:
        return env
    from .updraft import column_environment
    return column_environment(
        temperature, humidity, pressure, cp_moist, pressure_half,
        layer_mass=layer_mass,
    )


def find_lfs(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    updraft_temp: jnp.ndarray,
    updraft_humid: jnp.ndarray,
    updraft_mf: jnp.ndarray,
    precip_rate: jnp.ndarray,
    kbase: int,
    ktop: int,
    config: ConvectionParameters,
    env: HalfLevelEnvironment | None = None,
    pressure_half: jnp.ndarray | None = None,
    mass_flux_base: jnp.ndarray | None = None,
) -> Tuple[int, bool]:
    """Find the level of free sinking (ECHAM ``cudlfs``).

    Faithful port of mo_cudescent.f90:62-160 on half levels. At each
    interface ``j`` strictly inside the realized cloud (``kctop < j < kcbot``)
    and within ``3 ≤ jk ≤ klev − 3`` (0-based ``2 ≤ j ≤ nlev − 4``), the
    half-level environment is brought to its wet bulb (``cuadjtq`` kcall = 2
    at the interface pressure) and mixed 50/50 with the plume there. The
    highest interface where that mixture is negatively buoyant against the
    half-level environment, and where

        ``prfl > 10·zmftop·zcond``,  ``zmftop = −cmfdeps·pmfub``,

    holds, is the LFS. ``zmftop`` is negative and ``zcond = pqenh − q_wb``
    non-negative, so in the reference this test is met by any column that is
    raining at all; it is kept literally so the downdraft switches on under
    exactly the reference conditions.

    Args:
        temperature, humidity, pressure: Full-level environment.
        updraft_temp, updraft_humid: Half-level plume ``ptu``/``pqu``.
        updraft_mf: Half-level plume mass flux; its value at ``kbase``
            stands in for ``pmfub`` when ``mass_flux_base`` is not given.
        precip_rate: Column precipitation generated by the updraft
            (``zrfl``) [kg/m²/s].
        kbase: Cloud-base interface ``kcbot``.
        ktop: Realized cloud-top interface ``kctop``.
        config: Convection configuration.
        env: Precomputed half-level environment; built when ``None``.
        pressure_half: Interface pressures used to build ``env``.
        mass_flux_base: The cloud-base (seed) mass flux ``pmfub``.

    Returns:
        ``(lfs_interface, found)``.

    """
    from .updraft import _pick_level
    nlev = temperature.shape[0]
    env = _env_or_build(env, temperature, humidity, pressure,
                        pressure_half=pressure_half)
    twb, qwb = wetbulb_temperature(env.tenh, env.qenh, env.paph[:-1])
    t_mix = 0.5 * (updraft_temp + twb)
    q_mix = 0.5 * (updraft_humid + qwb)
    zbuo = (t_mix * (1.0 + c.vtmpc1 * q_mix)
            - env.tenh * (1.0 + c.vtmpc1 * env.qenh))
    zcond = env.qenh - qwb
    pmfub = (
        _pick_level(updraft_mf, kbase)
        if mass_flux_base is None
        else mass_flux_base
    )
    zmftop = -config.cmfdeps * pmfub
    levels = jnp.arange(nlev)
    in_range = ((levels >= 2) & (levels <= nlev - 4)
                & (levels > ktop) & (levels < kbase) & (precip_rate > 0.0))
    is_lfs = in_range & (zbuo < 0.0) & (precip_rate > 10.0 * zmftop * zcond)
    found = jnp.any(is_lfs)
    lfs = jnp.where(found, jnp.argmax(is_lfs), ktop)
    return lfs, found


def downdraft_entrainment_ledger(
    mfd: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    entrdd: float,
) -> jnp.ndarray:
    """Per-layer downdraft entrainment magnitude [kg/m²/s] from a flux profile.

    Rebuilds ECHAM ``cuddraf``'s ``|zdmfen| = entrdd·|pmfd(top of layer)|·
    Δz`` for callers that hold only a half-level downdraft profile (``mfd[j]``
    at the TOP interface of layer ``j``): layer ``j`` entrains while the
    downdraft enters it from above (``mfd[j] < 0``) and leaves it through
    its bottom interface (``mfd[j + 1] < 0``); where the descent died inside
    the layer the ledger is zero, so plume continuity dumps the arriving flux
    as pure detrainment, matching the Fortran's buoyancy shut-off. The three
    lowest layers are the surface taper (``itopde = klev − 2``), where
    ECHAM does not entrain. The scheme itself returns its exact ledger
    (:attr:`DowndraftState.dmfen`); this helper serves column callers.

    Vertical on axis 0; trailing axes broadcast (a ``(nlev,)`` column and
    a ``(nlev, ncols)`` block agree per column).
    """
    nlev = mfd.shape[0]
    mfd_out = jnp.concatenate([mfd[1:], jnp.zeros_like(mfd[:1])], axis=0)
    levels = jnp.arange(nlev).reshape((nlev,) + (1,) * (mfd.ndim - 1))
    in_bulk = (mfd < 0.0) & (mfd_out < 0.0) & (levels < nlev - 3)
    return jnp.where(in_bulk, entrdd * jnp.abs(mfd) * layer_thickness, 0.0)


def calculate_downdraft(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    rho: jnp.ndarray,
    updraft_state,  # UpdatedraftState from updraft.py
    precip_rate: jnp.ndarray,
    kbase: int,
    ktop: int,
    config: ConvectionParameters,
    u_wind: jnp.ndarray | None = None,
    v_wind: jnp.ndarray | None = None,
    cp_moist: jnp.ndarray | None = None,
    pressure_half: jnp.ndarray | None = None,
    env: HalfLevelEnvironment | None = None,
    mass_flux_base: jnp.ndarray | None = None,
) -> DowndraftState:
    """Calculate the downdraft on half levels (ECHAM ``cudlfs`` + ``cuddraf``).

    Args:
        temperature: Environmental temperature (K) [nlev], top-first.
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Full-level pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]; accepted for call
            compatibility (layer geometry comes from the half-level
            environment).
        rho: Air density (kg/m³) [nlev]; accepted for call compatibility.
        updraft_state: Half-level updraft (:class:`~.updraft.UpdatedraftState`).
        precip_rate: Column precipitation generated by the updraft (kg/m²/s)
        kbase: Cloud-base interface ``kcbot``.
        ktop: Realized cloud-top interface ``kctop``.
        config: Convection configuration
        u_wind, v_wind: Full-level environmental winds (m/s).
        cp_moist: Full-level moist heat capacity (``pcpen``); only used to
            build the environment when ``env`` is not given.
        pressure_half: Interface pressures [Pa] for building ``env``.
        env: Precomputed :class:`~.half_levels.HalfLevelEnvironment`.
        mass_flux_base: The cloud-base (seed) mass flux ``pmfub`` that sets
            the downdraft's initial flux ``−cmfdeps·pmfub``; ``None`` uses the
            updraft flux at ``kbase`` (the seed of a cubase plume; a mid-level
            plume's seed sits one interface lower).

    Returns:
        :class:`DowndraftState` of half-level profiles.

    """
    nlev = temperature.shape[0]
    dtype = temperature.dtype
    if cp_moist is None:
        cp_moist = moist_isobaric_heat_capacity(humidity)
    del layer_thickness, rho
    env = _env_or_build(env, temperature, humidity, pressure,
                        cp_moist=cp_moist, pressure_half=pressure_half)
    if u_wind is None:
        u_wind = jnp.zeros(nlev, dtype)
    if v_wind is None:
        v_wind = jnp.zeros(nlev, dtype)

    lfs, has_lfs = find_lfs(
        temperature, humidity, pressure,
        updraft_state.tu, updraft_state.qu, updraft_state.mfu,
        precip_rate, kbase, ktop, config, env=env,
        mass_flux_base=mass_flux_base,
    )

    # --- cudlfs seed at the LFS interface (mo_cudescent.f90:121-150) -------
    # 50/50 mixture of plume air and the wet-bulb half-level environment,
    # the flux ``zmftop = −cmfdeps·pmfub``, and the rain evaporated to reach
    # that wet bulb charged to the layer ABOVE the interface
    # (``pdmfdp(jk−1) = −0.5·pmfd·zcond``). The seed wind mixes the plume
    # wind with the environment of that layer above.
    from .updraft import _pick_level
    tenh_lfs = _pick_level(env.tenh, lfs)
    qenh_lfs = _pick_level(env.qenh, lfs)
    paph_lfs = _pick_level(env.paph, lfs)
    twb, qwb = wetbulb_temperature(tenh_lfs, qenh_lfs, paph_lfs)
    zcond0 = qenh_lfs - qwb
    pmfub = (
        _pick_level(updraft_state.mfu, kbase)
        if mass_flux_base is None
        else mass_flux_base
    )
    mftop = -config.cmfdeps * pmfub
    mfd_seed = jnp.asarray(jnp.where(has_lfs, mftop, 0.0), dtype)
    pdmfdp_seed = jnp.asarray(-0.5 * mfd_seed * zcond0, dtype)
    above = jnp.maximum(lfs - 1, 0)
    td_seed = jnp.asarray(
        jnp.where(
            has_lfs,
            0.5 * (_pick_level(updraft_state.tu, lfs) + twb),
            tenh_lfs,
        ),
        dtype,
    )
    qd_seed = jnp.asarray(
        jnp.where(
            has_lfs,
            0.5 * (_pick_level(updraft_state.qu, lfs) + qwb),
            qenh_lfs,
        ),
        dtype,
    )
    ud_seed = jnp.asarray(
        0.5 * (_pick_level(updraft_state.uu, lfs) + _pick_level(u_wind, above)),
        dtype,
    )
    vd_seed = jnp.asarray(
        0.5 * (_pick_level(updraft_state.vu, lfs) + _pick_level(v_wind, above)),
        dtype,
    )
    rain0 = jnp.asarray(precip_rate + pdmfdp_seed, dtype)

    # Surface taper (``itopde = klev − 2``, lines 206-216): below interface
    # itopde the downdraft stops entraining and detrains linearly in pressure,
    # ``zdmfde = pmfd(itopde)·Δp/(p_s − p(itopde))``, so its flux reaches zero
    # at the surface.
    itopde = nlev - 3
    ps = env.paph[-1]
    taper_depth = jnp.maximum(ps - env.paph[itopde], 1e-6)

    levels = jnp.arange(nlev)
    tenh_up = jnp.concatenate([env.tenh[:1], env.tenh[:-1]])
    qenh_up = jnp.concatenate([env.qenh[:1], env.qenh[:-1]])
    geoh_up = jnp.concatenate([env.geoh[:1], env.geoh[:-1]])
    cpcu_up = jnp.concatenate([env.cpcu[:1], env.cpcu[:-1]])
    paph_up = jnp.concatenate([env.paph[:1], env.paph[:nlev - 1]])
    dp_up = jnp.concatenate([env.dp[:1], env.dp[:-1]])
    u_up = jnp.concatenate([u_wind[:1], u_wind[:-1]])
    v_up = jnp.concatenate([v_wind[:1], v_wind[:-1]])
    level_inputs = (
        levels,
        tenh_up, qenh_up, geoh_up, cpcu_up,
        paph_up, dp_up,
        env.tenh, env.qenh, env.geoh, env.cpcu, env.paph[:-1],
        u_up, v_up,
    )

    entrdd_ = config.entrdd
    cmfcmin_ = config.cmfcmin
    z0 = jnp.zeros((), dtype)
    # Scalar carry for the top-to-bottom scan: the downdraft properties at
    # interface ``j - 1``, the saved mass flux at ``itopde`` for the surface
    # taper, and the remaining rain flux. Per-level outputs are emitted via
    # ``ys`` rather than written with dynamic updates into ``(nlev,)`` arrays.
    initial_carry = (z0, z0, z0, z0, z0, z0, rain0)

    def downdraft_step(carry_tuple, inputs):
        mfd_a, td_a, qd_a, ud_a, vd_a, mfd_itopde, rain_flux = carry_tuple
        (j, tenh_a, qenh_a, geoh_a, cpcu_a, paph_a, dp_a,
         tenh_j, qenh_j, geoh_j, cpcu_j, paph_j, u_a, v_a) = inputs
        active = has_lfs & (j > lfs) & (j >= 2) & (mfd_a < 0.0)

        def compute():
            zentr = (entrdd_ * mfd_a * c.rd * tenh_a
                     / (c.grav * jnp.maximum(paph_a, 1.0)) * dp_a)
            in_taper = j > itopde
            zdmfen = jnp.where(in_taper, 0.0, zentr)
            zdmfde = jnp.where(
                in_taper, mfd_itopde * dp_a / taper_depth, zentr)
            mfd_new = mfd_a + zdmfen - zdmfde
            s_a = cpcu_a * td_a + geoh_a
            s_e = cpcu_a * tenh_a + geoh_a
            div = jnp.minimum(-cmfcmin_, mfd_new)
            s_new = (mfd_a * s_a + zdmfen * s_e - zdmfde * s_a) / div
            q_new = (mfd_a * qd_a + zdmfen * qenh_a - zdmfde * qd_a) / div
            td_new = jnp.clip((s_new - geoh_j) / cpcu_j, 100.0, 400.0)
            td_adj, qd_adj, zcond = cuadjtq(td_new, q_new, paph_j, kcall=2)
            zbuo = (td_adj * (1.0 + c.vtmpc1 * qd_adj)
                    - tenh_j * (1.0 + c.vtmpc1 * qenh_j))
            keep = (zbuo < 0.0) & (rain_flux - mfd_new * zcond > 0.0)
            mfd_fin = jnp.where(keep, mfd_new, 0.0)
            zdmfdp = -mfd_fin * zcond
            ud_new = jnp.where(
                keep,
                (mfd_a * ud_a + zdmfen * u_a - zdmfde * ud_a) / div,
                z0,
            )
            vd_new = jnp.where(
                keep,
                (mfd_a * vd_a + zdmfen * v_a - zdmfde * vd_a) / div,
                z0,
            )
            dmfen_jm1 = jnp.where(keep, -zdmfen, z0)
            mfd_itopde_next = jnp.where(j == itopde, mfd_fin, mfd_itopde)
            next_carry = (
                mfd_fin, td_adj, qd_adj, ud_new, vd_new,
                mfd_itopde_next, rain_flux + zdmfdp,
            )
            y_out = (td_adj, qd_adj, mfd_fin, ud_new, vd_new, zdmfdp, dmfen_jm1)
            return next_carry, y_out

        def skip():
            is_lfs = j == lfs
            td_j = jnp.where(is_lfs, td_seed, tenh_j)
            qd_j = jnp.where(is_lfs, qd_seed, qenh_j)
            mfd_j = jnp.where(is_lfs, mfd_seed, z0)
            ud_j = jnp.where(is_lfs, ud_seed, z0)
            vd_j = jnp.where(is_lfs, vd_seed, z0)
            mfd_itopde_next = jnp.where(j == itopde, mfd_j, mfd_itopde)
            next_carry = (
                mfd_j, td_j, qd_j, ud_j, vd_j, mfd_itopde_next, rain_flux,
            )
            y_out = (td_j, qd_j, mfd_j, ud_j, vd_j, z0, z0)
            return next_carry, y_out

        return lax.cond(active, compute, skip)

    _, (td_p, qd_p, mfd_p, ud_p, vd_p, pdmfdp_sh, dmfen_sh) = lax.scan(
        downdraft_step, initial_carry, level_inputs,
    )
    # ``pdmfdp`` and ``dmfen`` emitted at interface ``j`` belong to layer
    # ``a = j - 1`` above it; ``above = max(lfs - 1, 0)`` receives the LFS
    # wet-bulb evaporation seed and never overlaps an active ``j - 1`` step.
    pdmfdp_scan = jnp.concatenate([pdmfdp_sh[1:], jnp.zeros((1,), dtype)])
    pdmfdp = jnp.where(levels == above, pdmfdp_seed, pdmfdp_scan)
    dmfen = jnp.concatenate([dmfen_sh[1:], jnp.zeros((1,), dtype)])
    return DowndraftState(
        td=td_p,
        qd=qd_p,
        mfd=mfd_p,
        pdmfdp=pdmfdp,
        ud=ud_p,
        vd=vd_p,
        lfs=lfs,
        active=has_lfs,
        dmfen=dmfen,
    )
