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
from jcm.physics.convection.saturation import cuadjtq_newton_evap
from jcm.physics.thermodynamics import moist_isobaric_heat_capacity
from .tiedtke_nordeng import (
    ConvectionParameters
)
from .half_levels import HalfLevelEnvironment
from .adjustment import cuadjtq


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
    adjustment (see :func:`~jcm.physics.convection.saturation.
    cuadjtq_newton_evap`). Conserves moist static energy exactly
    (``cp·ΔT + L·Δq = 0``), and already-saturated air comes back unchanged
    because the evaporation-only clip zeroes the step.

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
        updraft_mf: Half-level plume mass flux (``pmfub`` is its value at
            ``kbase``).
        precip_rate: Column precipitation generated by the updraft
            (``zrfl``) [kg/m²/s].
        kbase: Cloud-base interface ``kcbot``.
        ktop: Realized cloud-top interface ``kctop``.
        config: Convection configuration.
        env: Precomputed half-level environment; built when ``None``.
        pressure_half: Interface pressures used to build ``env``.

    Returns:
        ``(lfs_interface, found)``.

    """
    nlev = temperature.shape[0]
    env = _env_or_build(env, temperature, humidity, pressure,
                        pressure_half=pressure_half)
    twb, qwb = wetbulb_temperature(env.tenh, env.qenh, env.paph[:-1])
    t_mix = 0.5 * (updraft_temp + twb)
    q_mix = 0.5 * (updraft_humid + qwb)
    zbuo = (t_mix * (1.0 + c.vtmpc1 * q_mix)
            - env.tenh * (1.0 + c.vtmpc1 * env.qenh))
    zcond = env.qenh - qwb
    zmftop = -config.cmfdeps * updraft_mf[kbase]
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
    )

    # --- cudlfs seed at the LFS interface (mo_cudescent.f90:121-150) -------
    # 50/50 mixture of plume air and the wet-bulb half-level environment,
    # the flux ``zmftop = −cmfdeps·pmfub``, and the rain evaporated to reach
    # that wet bulb charged to the layer ABOVE the interface
    # (``pdmfdp(jk−1) = −0.5·pmfd·zcond``). The seed wind mixes the plume
    # wind with the environment of that layer above.
    twb, qwb = wetbulb_temperature(
        env.tenh[lfs], env.qenh[lfs], env.paph[lfs])
    zcond0 = env.qenh[lfs] - qwb
    mftop = -config.cmfdeps * updraft_state.mfu[kbase]
    mfd_seed = jnp.where(has_lfs, mftop, 0.0)
    pdmfdp_seed = -0.5 * mfd_seed * zcond0
    above = jnp.maximum(lfs - 1, 0)
    td_init = env.tenh.at[lfs].set(
        jnp.where(has_lfs, 0.5 * (updraft_state.tu[lfs] + twb), env.tenh[lfs]))
    qd_init = env.qenh.at[lfs].set(
        jnp.where(has_lfs, 0.5 * (updraft_state.qu[lfs] + qwb), env.qenh[lfs]))
    ud_init = jnp.zeros(nlev, dtype).at[lfs].set(
        0.5 * (updraft_state.uu[lfs] + u_wind[above]))
    vd_init = jnp.zeros(nlev, dtype).at[lfs].set(
        0.5 * (updraft_state.vu[lfs] + v_wind[above]))
    initial_state = DowndraftState(
        td=td_init,
        qd=qd_init,
        mfd=jnp.zeros(nlev, dtype).at[lfs].set(mfd_seed),
        pdmfdp=jnp.zeros(nlev, dtype).at[above].set(pdmfdp_seed),
        ud=ud_init,
        vd=vd_init,
        lfs=lfs,
        active=has_lfs,
        dmfen=jnp.zeros(nlev, dtype),
    )
    rain0 = precip_rate + pdmfdp_seed

    # Surface taper (``itopde = klev − 2``, lines 206-216): below interface
    # itopde the downdraft stops entraining and detrains linearly in pressure,
    # ``zdmfde = pmfd(itopde)·Δp/(p_s − p(itopde))``, so its flux reaches zero
    # at the surface.
    itopde = nlev - 3
    ps = env.paph[-1]
    taper_depth = jnp.maximum(ps - env.paph[itopde], 1e-6)

    levels = jnp.arange(nlev)
    up = jnp.maximum(levels - 1, 0)
    level_inputs = (
        levels,
        env.tenh[up], env.qenh[up], env.geoh[up], env.cpcu[up],
        env.paph[up], env.dp[up],
        env.tenh, env.qenh, env.geoh, env.cpcu, env.paph[:-1],
        u_wind[up], v_wind[up],
    )

    def downdraft_step(carry_and_rain, inputs):
        carry, rain_flux = carry_and_rain
        (j, tenh_a, qenh_a, geoh_a, cpcu_a, paph_a, dp_a,
         tenh_j, qenh_j, geoh_j, cpcu_j, paph_j, u_a, v_a) = inputs
        a = jnp.maximum(j - 1, 0)
        mfd_a = carry.mfd[a]
        active = carry.active & (j > carry.lfs) & (j >= 2) & (mfd_a < 0.0)

        def compute():
            # Entrainment into the descent through layer j−1 (line 199):
            # entrdd·pmfd·(R_d·T/p at the layer's top interface)·Δp/g — the
            # fractional rate over the layer's geometric thickness. In the
            # bulk it is matched by an equal detrainment.
            # (The model-top interface, at zero pressure, is never a descent
            # origin; the floor only keeps that traced-but-unused lane finite.)
            zentr = (entrdd_ * mfd_a * c.rd * tenh_a
                     / (c.grav * jnp.maximum(paph_a, 1.0)) * dp_a)
            in_taper = j > itopde
            zdmfen = jnp.where(in_taper, 0.0, zentr)
            zdmfde = jnp.where(
                in_taper, carry.mfd[itopde] * dp_a / taper_depth, zentr)
            mfd_new = mfd_a + zdmfen - zdmfde
            # Flux-form mixing (lines 222-233): static energy and moisture of
            # the arriving flux plus the entrained environment of the layer's
            # top interface minus the detrained downdraft air.
            s_a = cpcu_a * carry.td[a] + geoh_a
            s_e = cpcu_a * tenh_a + geoh_a
            div = jnp.minimum(-cmfcmin_, mfd_new)
            s_new = (mfd_a * s_a + zdmfen * s_e - zdmfde * s_a) / div
            q_new = (mfd_a * carry.qd[a] + zdmfen * qenh_a
                     - zdmfde * carry.qd[a]) / div
            td_new = jnp.clip((s_new - geoh_j) / cpcu_j, 100.0, 400.0)
            # Evaporate rain into the descending air toward saturation
            # (cuadjtq kcall = 2 at the interface pressure, line 272).
            td_adj, qd_adj, zcond = cuadjtq(td_new, q_new, paph_j, kcall=2)
            # ``zcond`` ≤ 0 is the vapour taken up; the rain it consumes is
            # ``zdmfdp = −pmfd·zcond`` ≤ 0.
            zbuo = (td_adj * (1.0 + c.vtmpc1 * qd_adj)
                    - tenh_j * (1.0 + c.vtmpc1 * qenh_j))
            keep = (zbuo < 0.0) & (rain_flux - mfd_new * zcond > 0.0)
            mfd_fin = jnp.where(keep, mfd_new, 0.0)
            zdmfdp = -mfd_fin * zcond
            ud_new = jnp.where(
                keep,
                (mfd_a * carry.ud[a] + zdmfen * u_a - zdmfde * carry.ud[a])
                / div,
                carry.ud[j])
            vd_new = jnp.where(
                keep,
                (mfd_a * carry.vd[a] + zdmfen * v_a - zdmfde * carry.vd[a])
                / div,
                carry.vd[j])
            new = carry._replace(
                td=carry.td.at[j].set(td_adj),
                qd=carry.qd.at[j].set(qd_adj),
                mfd=carry.mfd.at[j].set(mfd_fin),
                pdmfdp=carry.pdmfdp.at[a].set(zdmfdp),
                ud=carry.ud.at[j].set(ud_new),
                vd=carry.vd.at[j].set(vd_new),
                # Layer j−1 entrains only while the descent continues through
                # it; where it died the arriving flux detrains there entirely.
                dmfen=carry.dmfen.at[a].set(jnp.where(keep, -zdmfen, 0.0)),
            )
            return new, rain_flux + zdmfdp

        new_carry, new_rain = lax.cond(
            active, compute, lambda: (carry, rain_flux))
        return (new_carry, new_rain), None

    entrdd_ = config.entrdd
    cmfcmin_ = config.cmfcmin
    (final_state, _rain_left), _ = lax.scan(
        downdraft_step, (initial_state, rain0), level_inputs,
    )
    return final_state
