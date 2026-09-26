"""Flux calculations and tendency updates for Tiedtke-Nordeng convection

This module implements:
- Final mass flux adjustments
- Temperature and moisture tendency calculations
- Momentum transport
- Precipitation and cloud water/ice partitioning

Based on ICON mo_cufluxdts.f90

"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple

import jcm.constants as c
from jcm.physics.thermodynamics import moist_isobaric_heat_capacity
from .tiedtke_nordeng import ConvectionParameters, ConvectionTendencies
from .updraft import UpdatedraftState, column_environment
from .downdraft import DowndraftState
from .half_levels import HalfLevelEnvironment

#: Floor on the assumed updraft velocity inside :func:`updraft_area_cover`'s
#: division [m/s]: a slower "updraft" is not one, and a tiny epsilon there
#: would put the guarded-division VJP in the float32 squared-underflow window
#: (the double-where NaN class, jax-gcm#558/#559). Matches the wet-deposition
#: port's floor so the shared cover is identical on both sides.
_UPDRAFT_VELOCITY_FLOOR = 0.01
#: Floor guarding the sub-cloud-taper mass-ratio denominator [Pa or kg/m²].
_MASS_EPS = 1.0e-30
#: A rain shaft only exists where the updraft area is positive; below this the
#: column carries no plume there and the sub-cloud evaporation is switched off
#: (its safe-cover substitution keeps the discarded lane's VJP finite).
_COVER_FLOOR = 1.0e-12


def updraft_area_cover(
    mass_flux_up: jnp.ndarray,       # (nlev, *horiz) updraft flux, top-first [kg/m²/s]
    density: jnp.ndarray,            # (nlev, *horiz) density in the shaft [kg/m³]
    ktype: jnp.ndarray,             # (*horiz) convection type (3 = mid-level)
    layer_weight: jnp.ndarray,       # (nlev, *horiz) per-layer mass ∝ Δp [kg/m² or Pa]
    updraft_velocity: jnp.ndarray,   # assumed in-cloud updraft speed ``zwu`` [m/s]
) -> jnp.ndarray:
    """Fraction of the grid box occupied by the convective updraft/rain shaft.

    ECHAM ``cuflx`` (``mo_cufluxdts.f90:417``) estimates the updraft area
    from the mass flux and a prescribed updraft speed,
    ``zcucov = pmfu / (zwu·zrhou)`` with ``zwu`` the assumed in-cloud
    velocity (line 166, 2 m/s) and ``zrhou`` the density in the shaft.
    ECHAM reuses this as the footprint of the sub-cloud rain evaporation
    under the HAM submodel, and HAMMOZ ``prep_wetdep_hydro`` reuses the
    identical quantity for the convective wet deposition — so the two
    share one implementation here.

    The mass flux ECHAM feeds this is the plume's own profile through the
    cloud and, below cloud base, a taper that decreases linearly in
    pressure from the cloud-base value to zero at the surface
    (``pmfu(jk) = pmfu(kcbot)·zzp``, ``zzp = (p_s − p_half(jk)) /
    (p_s − p_half(kcbot))``, squared for mid-level convection;
    ``mo_cufluxdts.f90:233-239``) — the updraft draws its air from the
    whole sub-cloud layer, so the shaft below the base keeps its footprint
    and tapers to the surface. The supplied ``mass_flux_up`` carries only
    the plume profile (zero below cloud base), so that taper is rebuilt
    here from the layer masses: ``p_s − p_half(k) = g·Σ_{j≥k} m_j``, so the
    pressure ratio equals the ratio of the air mass below the two
    interfaces. ``layer_weight`` must therefore be proportional to the TRUE
    half-level thickness ``Δp = p_half(k+1) − p_half(k)`` of each layer —
    the moist-air ``pressure_thickness`` diagnostic (unfloored ``Δp``), the
    same layer mass the cudtdq ledger divides by — and not
    ``ρ·layer_thickness`` where that thickness carries the moist-air 10 m
    floor: the cumulative sum here turns any per-layer thickness error into
    a systematic bias in ``p_s − p_half`` on every stretched (hybrid) grid.
    Levels are top-first and ``mass_flux_up[k]`` is the flux through the TOP
    interface of layer ``k``; the cloud base is the lowest interface (largest
    index) with a non-zero flux.

    ``density`` is left to the caller: the sub-cloud evaporation inside the
    convection scheme has the updraft temperature available and passes the
    true updraft density ``zrhou = p/(rd·ptu)``, whereas the wet-deposition
    port has only the environment density and passes that as a stand-in
    (a few-per-cent difference over the ~2 % the assumed ``zwu`` already
    dominates).

    Not clipped: ``cuflx`` applies no clamp, and the evaporation uses the
    raw area. Callers that need a cover *fraction* (the wet deposition)
    clip to ``[0, 1]`` themselves. Broadcasting-native (vertical on axis 0).
    """
    # A negative updraft mass flux is not a plume; the Tiedtke ledger is
    # non-negative by construction, so this only pins the contract.
    mass_flux_up = jnp.maximum(mass_flux_up, 0.0)
    nlev = mass_flux_up.shape[0]
    idx = jnp.arange(nlev).reshape((nlev,) + (1,) * (mass_flux_up.ndim - 1))
    active = mass_flux_up > 0.0
    # Cloud base = lowest active level (largest index, top-first order);
    # -1 where the column carries no plume, which leaves every level with a
    # zero flux and hence zero cover.
    kbase = jnp.max(jnp.where(active, idx, -1), axis=0)
    take = jnp.maximum(kbase, 0)[jnp.newaxis]
    mfu_base = jnp.take_along_axis(mass_flux_up, take, axis=0)[0]
    # Air mass below each layer's TOP interface (the layer itself included).
    mass_below = jnp.cumsum(layer_weight[::-1], axis=0)[::-1]
    mass_below_base = jnp.take_along_axis(mass_below, take, axis=0)[0]
    zzp = mass_below / jnp.maximum(mass_below_base, _MASS_EPS)
    zzp = jnp.where(ktype == 3, zzp * zzp, zzp)
    sub_cloud = (idx > kbase) & (kbase >= 0)
    mfu_eff = jnp.where(sub_cloud, mfu_base * zzp, mass_flux_up)
    w_u = jnp.maximum(updraft_velocity, _UPDRAFT_VELOCITY_FLOOR)
    return mfu_eff / (w_u * density)


def subcloud_taper(
    flux: jnp.ndarray,           # (nlev, *horiz) half-level flux, top-first
    kbase: jnp.ndarray,          # (*horiz) cloud-base interface kcbot
    ktype: jnp.ndarray,          # (*horiz) convection type (3 = mid-level)
    pressure_half: jnp.ndarray,  # (nlev + 1, *horiz) interface pressures
) -> jnp.ndarray:
    """ECHAM ``cuflx``'s sub-cloud taper of an updraft flux.

    Below the cloud-base interface the flux is its cloud-base value times
    ``zzp = (p_s − p_half(k))/(p_s − p_half(kcbot))``, squared for mid-level
    convection (mo_cufluxdts.f90:237-248): the plume draws its air from the
    whole sub-cloud layer, reaching zero at the surface. Entry ``k`` of the
    flux is its value at the TOP interface of layer ``k``; everything at and
    above ``kbase`` is returned unchanged. Broadcasting-native (vertical on
    axis 0), so the scheme's single columns and the term's ``(nlev, ncols)``
    blocks share it.
    """
    nlev = flux.shape[0]
    shape = (nlev,) + (1,) * (flux.ndim - 1)
    levels = jnp.arange(nlev).reshape(shape)
    kb = jnp.asarray(kbase)[jnp.newaxis]
    ps = pressure_half[-1:]
    p_base = jnp.take_along_axis(pressure_half, kb, axis=0)
    zzp = (ps - pressure_half[:-1]) / jnp.maximum(ps - p_base, _MASS_EPS)
    zzp = jnp.where(jnp.asarray(ktype)[jnp.newaxis] == 3, zzp * zzp, zzp)
    flux_base = jnp.take_along_axis(flux, kb, axis=0)
    return jnp.where(levels > kb, flux_base * zzp, flux)


def calculate_precipitation_rate(
    updraft_state: UpdatedraftState,
    kbase: int,
    dt: float,
    config: ConvectionParameters
) -> jnp.ndarray:
    """Calculate surface precipitation rate from convection.

    Sums the per-layer ``pdmfup`` (precipitation generated in the
    updraft, kg/m²/s) computed inside ``calculate_updraft``. Each layer
    converts a fraction ``cprcon * g * dz / (1 + cprcon * g * dz)`` of
    its liquid water content to precip, mirroring ECHAM
    ``mo_cuascent.f90`` lines 454-457. The column integral of those
    per-layer rates is the surface rain mass flux.

    Args:
        updraft_state: Updraft calculation results (with per-layer
            ``pdmfup`` precip generation already computed).
        kbase: Cloud base level (unused — kept for backwards-compat).
        dt: Time step (s) (unused — kept for backwards-compat).
        config: Convection configuration (unused — kept for
            backwards-compat).

    Returns:
        Surface precipitation rate (kg/m²/s).

    """
    return jnp.sum(updraft_state.pdmfup)


def convective_precip_fluxes(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    dp_lev: jnp.ndarray,
    kbase: int,
    pdmfup: jnp.ndarray,
    pdmfdp: jnp.ndarray,
    dt: float,
    updraft_temperature: jnp.ndarray | None = None,
    updraft_mass_flux: jnp.ndarray | None = None,
    ktype: jnp.ndarray | None = None,
    updraft_velocity: jnp.ndarray = 2.0,
    use_updraft_cover: bool = False,
    updraft_layer_mass: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray]:
    """ECHAM ``cuflx`` precipitation budget (mo_cufluxdts.f90:265-491).

    Walks the column top→bottom three times, exactly as the Fortran:

    1. Partition newly generated precip (``pdmfup + pdmfdp``) into rain or
       snow by the full-level environment temperature; melt falling snow
       where ``T > tmelt + 2`` at the rate the layer's heat content allows
       (``zcons1`` form), recording the per-layer melt ``pdpmel`` whose
       ``−alf·pdpmel`` heat sink cudtdq applies.
    2. Below cloud base, evaporate the total precip flux with ECHAM's
       Kessler chain: the square root of the per-cover rain intensity is
       depleted linearly by ``cevapcu(k)·Δp·(qs−q)`` and squared back,
       capped so the layer is not moistened beyond 80 % of saturation in
       one step. The evaporated amount is charged (negative) into
       ``pdmfup`` so the same cudtdq ledger cools/moistens the layer.
       ``cevapcu`` is ECHAM's level-dependent profile (iniphy.f90:87-89),
       NOT a linear rate coefficient.
    3. Deplete the surface rain/snow fluxes proportionally by the total
       sub-cloud evaporation.

    Args:
        temperature: Full-level environment temperature [K] (TOA-first).
        humidity: Environment specific humidity [kg/kg].
        pressure: Full-level pressure [Pa].
        dp_lev: True per-layer pressure thickness ``Δp = p_half(k+1) −
            p_half(k)`` [Pa] (ECHAM ``paphp1(jk+1) − paphp1(jk)``).
        kbase: Cloud-base interface index ``kcbot`` (the evaporation acts in
            the layers at and below it).
        pdmfup: Per-layer updraft precip generation [kg/m²/s] (≥ 0).
        pdmfdp: Per-layer downdraft precip sink [kg/m²/s] (≤ 0).
        dt: Time step [s].
        updraft_temperature: Updraft temperature ``ptu`` [K] (TOA-first),
            used to build the updraft density for the sub-cloud
            evaporation cover; only read when ``use_updraft_cover``.
        updraft_mass_flux: Updraft mass flux ``pmfu`` [kg/m²/s]
            (TOA-first), the plume profile from which the updraft-area
            cover is built; only read when ``use_updraft_cover``.
        ktype: Convection type (1=deep, 2=shallow, 3=mid); selects the
            mid-level sub-cloud taper squaring in the cover.
        updraft_velocity: Assumed in-cloud updraft speed ``zwu`` [m/s].
        use_updraft_cover: When ``True`` the sub-cloud rain-evaporation
            footprint is the updraft area ``pmfu/(zwu·zrhou)`` — ECHAM's
            ``lham`` branch (``mo_cufluxdts.f90:416-417``); when ``False``
            it is ECHAM's non-HAM constant ``zcucov = 0.05`` (line 419).
        updraft_layer_mass: Per-layer air mass ∝ the true half-level
            ``Δp`` (``Δp/g`` [kg/m²]), the taper weight for the cover's
            sub-cloud ``p_s − p_half`` reconstruction. Required when
            ``use_updraft_cover``; the ledger passes ``dp_lev / g``.

    Returns:
        ``(rain_sfc, snow_sfc, prain, pdpmel, pdmfup_adj, precip_flux,
        floor_source)`` — surface rain and snow fluxes, the production-only
        diagnostic ``prain``, the per-layer snow melt, ``pdmfup`` including
        the (negative) sub-cloud evaporation increments, the total
        (rain + snow) precipitation flux ENTERING each layer from above, and
        the column water [kg/m²/s] that ``cuflx``'s non-negative floor on
        the rain and snow fluxes creates when the downdraft takes up more
        rain than the plume generates.

    """
    nlev = len(temperature)
    # ECHAM ``zcons1 = cpd/(alf·grav·dt)`` (mo_cufluxdts.f90:146); the moist
    # factor ``(1 + vtmpc2·pqen)`` is applied per layer in ``zfac`` below
    # (line 303), with the PROVISIONAL humidity ``pqen`` exactly as there.
    zcons1 = c.cpd / (c.alhf * c.grav * dt)
    zcons2 = 1.0 / (c.grav * dt)
    ztmelp2 = c.tmelt + 2.0
    # Fractional precip cover for the sub-cloud Kessler evaporation
    # (mo_cufluxdts.f90:414-420). ECHAM keys it on the HAM submodel:
    #   * plain ECHAM (``.NOT.lham``): the constant ``zcucov = 0.05``;
    #   * with HAM active (``lham``): the updraft AREA
    #     ``pmfu/(zwu·zrhou)``, ``zrhou = p/(rd·ptu)`` the updraft density
    #     (lines 406-407), ``zwu`` the assumed in-cloud updraft speed
    #     (line 166) — the same footprint the JAM convective wet
    #     deposition uses (jax-gcm#812). ``use_updraft_cover`` mirrors that
    #     compile-time ``lham`` switch (set on ``TiedtkeConvection`` when the
    #     JAM chain is composed). Below cloud base the plume profile carries
    #     cuini's initial ``ptu = ptenh`` (ECHAM's own sub-cloud ``ptu`` is the
    #     dry-lifted parcel, within a few tenths of a kelvin of it — a
    #     negligible difference in a density); the full-level temperature is
    #     the fallback for callers passing an unset (zero) profile. The cover
    #     is a per-level array here.
    if use_updraft_cover:
        if updraft_layer_mass is None:
            # The ledger's dp_lev is NOT a valid taper weight (see the
            # docstring); silently substituting it would reintroduce the
            # stretched-grid bias this argument exists to prevent.
            raise ValueError(
                "use_updraft_cover=True requires updraft_layer_mass — the "
                "true per-layer air mass (∝ half-level Δp) for the "
                "sub-cloud taper."
            )
        t_rho = jnp.where(updraft_temperature > 1.0, updraft_temperature,
                          temperature)
        rho_updraft = pressure / (c.rd * t_rho)
        zcucov_lev = updraft_area_cover(
            updraft_mass_flux, rho_updraft, ktype, updraft_layer_mass,
            updraft_velocity,
        )
    else:
        zcucov_lev = jnp.full_like(dp_lev, 0.05)

    from .tiedtke_nordeng import saturation_mixing_ratio
    qs_env = jax.vmap(saturation_mixing_ratio)(pressure, temperature)

    # ECHAM cevapcu(jk) profile (iniphy.f90:87-89) with eta ≈ p/p_surface.
    eta = pressure / jnp.maximum(pressure[-1], 1.0)
    cevapcu = (
        1.93e-6 * 261.0
        * jnp.sqrt(1.0e3 / (38.3 * 0.293) * jnp.sqrt(jnp.clip(eta, 1e-4, 1.0)))
        * 0.5 / c.grav
    )

    # --- pass 1: rain/snow partition + melting (cuflx 296-313) ------------
    def partition_step(carry, xs):
        zrfl, zsfl = carry
        gen, T_k, q_k, dp_k = xs
        warm = T_k > c.tmelt
        zrfl = jnp.where(warm, zrfl + gen, zrfl)
        zsfl = jnp.where(warm, zsfl, zsfl + gen)
        zfac = zcons1 * (1.0 + c.vtmpc2 * q_k) * dp_k
        zsnmlt = jnp.where(
            warm & (zsfl > 0.0),
            jnp.minimum(zsfl, zfac * jnp.maximum(T_k - ztmelp2, 0.0)),
            0.0,
        )
        zsfl = zsfl - zsnmlt
        zrfl = zrfl + zsnmlt
        return (zrfl, zsfl), (zsnmlt, zrfl, zsfl)

    gen = pdmfup + pdmfdp
    (prfl, psfl), (pdpmel, rain_lev, snow_lev) = lax.scan(
        partition_step, (jnp.zeros(()), jnp.zeros(())),
        (gen, temperature, humidity, dp_lev),
    )
    # ``cuflx`` floors each phase's flux at zero (mo_cufluxdts.f90:324-325).
    # The ledger's vapour and heat sources are ``pdmfup + pdmfdp`` unfloored,
    # so wherever the downdraft's rain uptake exceeds what the plume
    # generates the floor creates the difference as water: the surface
    # precipitation no longer equals the column's water loss. ECHAM has the
    # same floor; ``cumastr`` scales the first ascent's downdraft to the
    # closed flux and re-runs ``cuasc`` (mo_cumastr.f90:944-986), and a
    # second ascent that rains less than the scaled downdraft takes up (a
    # deep-to-shallow demotion with its larger entrainment) drives the flux
    # negative. The created water is returned so the column budget can
    # account for it; removing it is a deviation from ECHAM tracked in
    # #912.
    floor_source = jnp.maximum(-prfl, 0.0) + jnp.maximum(-psfl, 0.0)
    prfl = jnp.maximum(prfl, 0.0)
    psfl = jnp.maximum(psfl, 0.0)
    prain = jnp.sum(jnp.maximum(pdmfup, 0.0))

    # --- pass 2: sub-cloud Kessler evaporation (cuflx 411-440) ------------
    k_idx = jnp.arange(nlev)

    def evap_step(zpsubcl, xs):
        k, qs_k, q_k, dp_k, cevap_k, cov_k = xs
        # A rain shaft needs a positive cover; where the column carries no
        # plume the updraft-area cover is exactly zero and the constant
        # cover is 0.05, so this only ever masks the empty updraft-cover
        # lanes (below/outside the plume).
        has_shaft = cov_k > _COVER_FLOOR
        active = (k >= kbase) & (zpsubcl > 1e-20) & has_shaft
        # Substitute a safe unit cover on the discarded lanes so the
        # sqrt-division never sees a zero denominator; those lanes are
        # masked out of ``zdrfl``/``zpsubcl`` below, so the substitution is
        # invisible except that it keeps the VJP finite (the double-where
        # NaN-gradient guard, jax-gcm#558/#559).
        cov_safe = jnp.where(has_shaft, cov_k, 1.0)
        zrfl = zpsubcl
        zrnew = (
            jnp.maximum(
                0.0,
                jnp.sqrt(jnp.maximum(jnp.maximum(zrfl, 0.0) / cov_safe, 1.0e-30))
                - cevap_k * dp_k * jnp.maximum(qs_k - q_k, 0.0),
            ) ** 2
        ) * cov_safe
        zrmin = zrfl - cov_safe * jnp.maximum(0.8 * qs_k - q_k, 0.0) * zcons2 * dp_k
        zrnew = jnp.maximum(zrnew, zrmin)
        zrfln = jnp.maximum(zrnew, 0.0)
        zdrfl = jnp.where(active, jnp.minimum(0.0, zrfln - zrfl), 0.0)
        zpsubcl_new = jnp.where(active, zrfln, zpsubcl)
        return zpsubcl_new, zdrfl

    zpsubcl_final, zdrfl_per_level = lax.scan(
        evap_step, prfl + psfl,
        (k_idx, qs_env, humidity, dp_lev, cevapcu, zcucov_lev),
    )
    pdmfup_adj = pdmfup + zdrfl_per_level  # negative increments (cuflx 437)

    # --- pass 3: proportional depletion (cuflx 486-491) -------------------
    zrsum = prfl + psfl
    zdpevap_tot = zpsubcl_final - zrsum  # ≤ 0
    # ``1/MAX(1e-20, zrsum)``. At or below that floor the depletion terms
    # scale with a precipitation flux of at most 1e-20 (exactly zero without
    # precipitation), so the factor is zeroed there instead: the
    # safe-denominator form keeps the reverse pass off ``0·inf``.
    has_rain = zrsum > 1e-20
    inv = jnp.where(has_rain, 1.0 / jnp.where(has_rain, zrsum, 1.0), 0.0)
    rain_sfc = jnp.maximum(prfl + zdpevap_tot * prfl * inv, 0.0)
    snow_sfc = jnp.maximum(psfl + zdpevap_tot * psfl * inv, 0.0)

    # Total precip flux crossing each layer's TOP interface. Built from the
    # PHASE-RESOLVED legs with the same per-leg floor cuflx applies at the
    # surface (lines above): a downdraft sink can drive one leg negative
    # while the other is still falling, and a floored total would let that
    # negative cancel the surviving phase and understate the carrier. Each
    # leg is floored, then the sub-cloud evaporation already charged above
    # is applied to their sum, which reproduces ``rain_sfc + snow_sfc``
    # exactly at the surface. Shift by one layer for the top interface
    # (zero at the model top).
    flux_bottom = jnp.maximum(
        jnp.maximum(rain_lev, 0.0) + jnp.maximum(snow_lev, 0.0)
        + jnp.cumsum(zdrfl_per_level),
        0.0,
    )
    precip_flux = jnp.concatenate(
        [jnp.zeros_like(flux_bottom[:1]), flux_bottom[:-1]]
    )

    return (rain_sfc, snow_sfc, prain, pdpmel, pdmfup_adj, precip_flux,
            floor_source)


def calculate_tendencies(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    pressure: jnp.ndarray,
    rho: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    updraft_state: UpdatedraftState,
    downdraft_state: DowndraftState,
    kbase: int,
    ktop: int,
    dt: float,
    config: ConvectionParameters,
    ktype: jnp.ndarray | None = None,
    use_updraft_cover: bool = False,
    layer_mass: jnp.ndarray | None = None,
    cp_moist: jnp.ndarray | None = None,
    pressure_half: jnp.ndarray | None = None,
    env: HalfLevelEnvironment | None = None,
) -> ConvectionTendencies:
    """Compute the final convective fluxes and tendencies (``cuflx``/``cudtdq``/``cududv``).

    A finite-volume ledger on the model's own layers. The updraft and
    downdraft profiles are half-level: entry ``k`` is the value at the TOP
    interface of layer ``k`` (top-first; the surface interface carries no
    flux). ``cuflx`` turns them into deviation fluxes against the half-level
    environment and tapers the updraft fluxes through the sub-cloud layer;
    ``cudtdq`` then gives each layer the difference of the fluxes through its
    two bounding interfaces plus its own per-layer sources, divided by the
    layer's true air mass ``Δp/g``. The column integral of every flux
    difference telescopes to zero, so the column budgets are exactly the
    per-layer sources: total water changes by minus the surface
    precipitation, on the same layer mass the host integrates with.

    Args:
        temperature: Environmental temperature (K) [nlev], top-first.
        humidity: Environmental humidity (kg/kg) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        pressure: Full-level pressure (Pa) [nlev]
        rho: Air density (kg/m³) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        updraft_state: Half-level updraft state
        downdraft_state: Half-level downdraft state
        kbase: Cloud-base interface ``kcbot``
        ktop: Cloud-top interface (unused: the fluxes vanish above the
            plume, and ECHAM's ledger runs over the whole column)
        dt: Time step (s)
        config: Convection configuration
        ktype: Convection type (1=deep, 2=shallow, 3=mid) — mid-level
            plumes use the SQUARED sub-cloud taper (cuflx line 244).
        use_updraft_cover: Route the sub-cloud rain evaporation through the
            updraft-area cover instead of ECHAM's non-HAM ``zcucov = 0.05``
            (jax-gcm#812).
        layer_mass: Per-layer air mass ``Δp/g`` [kg/m²]; only used to
            rebuild the interfaces when neither ``env`` nor
            ``pressure_half`` is given (full-level midpoints otherwise).
        cp_moist: Moist heat capacity ``cpd·(1 + vtmpc2·q)`` [J/kg/K]
            [nlev] — ECHAM's ``pcpen`` in the ``zrcpm = 1/pcpen`` tendency
            conversion (mo_cufluxdts.f90:654, 718). ``None`` builds it from
            ``humidity``.
        pressure_half: Interface pressures [Pa] (nlev+1).
        env: Precomputed :class:`~.half_levels.HalfLevelEnvironment`.

    Returns:
        ConvectionTendencies with all tendency terms

    """
    del ktop
    nlev = temperature.shape[0]
    if cp_moist is None:
        cp_moist = moist_isobaric_heat_capacity(humidity)
    del rho, layer_thickness
    if env is None:
        env = column_environment(
            temperature, humidity, pressure, cp_moist, pressure_half,
            layer_mass=layer_mass,
        )
    ktype_eff = jnp.asarray(0) if ktype is None else ktype

    # True layer air mass per unit area — the ``g/(paphp1(jk+1)−paphp1(jk))``
    # of every cudtdq/cududv tendency — and the same mass the host applies
    # the tendencies with.
    mass = env.dp / c.grav

    # --- cuflx 1: deviation fluxes against the half-level environment -----
    # (mo_cufluxdts.f90:199-236)
    #   pmfus −= pmfu·(pcpcu·ptenh + pgeoh)   → pmfu·pcpcu·(ptu − ptenh)
    #   pmfuq −= pmfu·pqenh                     → pmfu·(pqu − pqenh)
    # and the same for the downdraft. The compensating subsidence of the
    # environment is what the subtraction represents: without it the
    # absolute plume static energy (~3·10⁵ J/kg) would dominate.
    mfu = updraft_state.mfu
    mfd = downdraft_state.mfd
    pmfus = mfu * env.cpcu * (updraft_state.tu - env.tenh)
    pmfuq = mfu * (updraft_state.qu - env.qenh)
    pmful = mfu * updraft_state.lu
    pmfds = mfd * env.cpcu * (downdraft_state.td - env.tenh)
    pmfdq = mfd * (downdraft_state.qd - env.qenh)

    # --- cuflx 1b: sub-cloud taper (lines 237-250) -------------------------
    # Below the cloud-base interface the updraft fluxes are the cloud-base
    # values scaled by the air mass below each interface relative to that
    # below cloud base, ``zzp = (p_s − p_half(k))/(p_s − p_half(kcbot))``
    # (squared for mid-level convection): the plume draws its air from the
    # whole sub-cloud layer, so the cloud-base flux divergence is spread
    # through it down to zero at the surface instead of landing on one layer.
    def _taper(flux):
        return subcloud_taper(flux, kbase, ktype_eff, env.paph)

    pmfus = _taper(pmfus)
    pmfuq = _taper(pmfuq)
    pmful = _taper(pmful)

    # --- flux divergence across each layer --------------------------------
    # ``F(k+1) − F(k)``: the flux entering through the bottom interface minus
    # the flux leaving through the top, with zero flux through the surface —
    # cudtdq's explicit ``jk == klev`` branch (lines 713-740) is exactly this
    # with the absent below-surface flux.
    def _div(flux):
        below = jnp.concatenate([flux[1:], jnp.zeros_like(flux[:1])], axis=0)
        return below - flux

    # cuflx 2: the precipitation budget (rain/snow partition, melting,
    # sub-cloud Kessler evaporation charged back into pdmfup), on the true
    # layer thickness.
    (rain_sfc, snow_sfc, prain, pdpmel, pdmfup_adj,
     precip_flux, floor_source) = convective_precip_fluxes(
        temperature, humidity, pressure, env.dp, kbase,
        updraft_state.pdmfup, downdraft_state.pdmfdp, dt,
        updraft_temperature=updraft_state.tu,
        updraft_mass_flux=mfu,
        ktype=ktype,
        updraft_velocity=config.cu_updraft_velocity,
        use_updraft_cover=use_updraft_cover,
        updraft_layer_mass=mass,
    )
    plude = updraft_state.plude

    # --- cudtdq (lines 647-740) -------------------------------------------
    #   dT/dt = g/Δp · (1/pcpen) · [Δpmfus + Δpmfds − alf·pdpmel
    #                               − Δ(palvsh·pmful)
    #                               + zalv·(plude + pdmfup + pdmfdp)]
    #   dq/dt = g/Δp · [Δpmfuq + Δpmfdq + Δpmful − (plude + pdmfup + pdmfdp)]
    # The condensate flux carries the latent heat of the HALF-level phase
    # (``palvsh``, keyed to ``ptenh``); the per-layer sources carry that of
    # the FULL-level environment (``zalv``, keyed to ``pten``). Negative
    # pdmfup increments (sub-cloud evaporation) and pdmfdp (downdraft
    # evaporation) flip the source signs locally: cooling + re-moistening.
    zalv = jnp.where(temperature > c.tmelt, c.alhc, c.alhs)
    ledger_src = plude + pdmfup_adj + downdraft_state.pdmfdp
    heat = (
        _div(pmfus) + _div(pmfds) - c.alhf * pdpmel
        - _div(env.alvsh * pmful) + zalv * ledger_src
    )
    dtedt = heat / (cp_moist * mass)
    dqdt = (_div(pmfuq) + _div(pmfdq) + _div(pmful) - ledger_src) / mass

    # Detrained condensate feeds the stratiform cloud tracers
    # (``zxtec = g/Δp·plude`` split into pxtecl/pxteci by the full-level
    # temperature), NOT the vapour budget.
    liquid_frac = jnp.where(temperature > c.tmelt, 1.0, 0.0)
    dqc_dt = liquid_frac * plude / mass
    dqi_dt = (1.0 - liquid_frac) * plude / mass

    def calculate_momentum_transport():
        # ECHAM cududv (mo_cufluxdts.f90:874-960): the u/v tendency is the
        # divergence of the half-level deviation MOMENTUM fluxes of the
        # updraft and downdraft, each with its own plume wind
        # (``puu``/``pud``), against the environmental wind of the full level
        # ABOVE the interface (``ik = jk − 1``); the top interface copies the
        # one below it. Below cloud base both fluxes are the cloud-base values
        # under the same ``zzp`` taper as cuflx, so cumulus friction reaches
        # the surface.
        u_up = jnp.concatenate([u_wind[:1], u_wind[:-1]])
        v_up = jnp.concatenate([v_wind[:1], v_wind[:-1]])
        zmfuu = mfu * (updraft_state.uu - u_up)
        zmfuv = mfu * (updraft_state.vu - v_up)
        zmfdu = mfd * (downdraft_state.ud - u_up)
        zmfdv = mfd * (downdraft_state.vd - v_up)

        def _top_copy(flux):
            return flux.at[0].set(flux[1])

        fluxes = [_taper(_top_copy(f)) for f in (zmfuu, zmfuv, zmfdu, zmfdv)]
        zmfuu, zmfuv, zmfdu, zmfdv = fluxes
        return (_div(zmfuu + zmfdu) / mass, _div(zmfuv + zmfdv) / mass)

    dudt, dvdt = lax.cond(
        config.lmfdudv,
        calculate_momentum_transport,
        lambda: (jnp.zeros(nlev, temperature.dtype),
                 jnp.zeros(nlev, temperature.dtype)),
    )

    # Surface precipitation = rain + snow after the full cuflx budget
    # (generation − downdraft consumption − sub-cloud evaporation, with the
    # snow phase carried through melting).
    precip_rate = rain_sfc + snow_sfc

    # In-plume condensate of each layer (the half-level ``plu`` at its top
    # interface, where the updraft is active), phase-split by the UPDRAFT
    # temperature: ECHAM keys in-plume latent heat to ``ptu``
    # (mo_cuascent.f90:370) and only environment quantities to ``ptenh``, so
    # the plume's own freezing level is the plume's, not the environment's.
    # Consumers read the SUM; the split makes each half meaningful on its own.
    lu_in_plume = jnp.where(mfu > 0, updraft_state.lu, 0.0)
    plume_liquid = jnp.where(updraft_state.tu > c.tmelt, 1.0, 0.0)
    qc_conv = plume_liquid * lu_in_plume
    qi_conv = (1.0 - plume_liquid) * lu_in_plume

    return ConvectionTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dudt=dudt,
        dvdt=dvdt,
        qc_conv=qc_conv,
        qi_conv=qi_conv,
        precip_formation=jnp.maximum(updraft_state.pdmfup, 0.0),
        precip_conv=precip_rate,
        precip_flux=precip_flux,
        precip_floor_source=floor_source,
        dqc_dt=dqc_dt,
        dqi_dt=dqi_dt
    )

#: ECHAM's constant cloud-base first-guess mass flux, used when the PBL
#: moisture-budget closure is invalid (``mo_cumastr.f90:567``:
#: ``zmfub = FSEL(-zlo1, 0.01_wp, zdqpbl/(grav·MAX(zqumqe,zdqmin)))`` — the
#: literal ``0.01 kg m⁻² s⁻¹`` fallback). It is a genuine mass flux; the CAPE
#: dependence of DEEP convection is supplied afterwards by the Nordeng
#: ``zmfub1`` rescale, and shallow/mid keep this bounded constant.
ECHAM_MFUB_FALLBACK = 0.01


def mass_flux_closure(
    cape: jnp.ndarray,
    cin: jnp.ndarray,
    moisture_conv: jnp.ndarray,
    ktype: int,
    config: ConvectionParameters,
) -> jnp.ndarray:
    """ECHAM constant cloud-base first-guess mass flux (fallback path).

    Returns ECHAM's constant fallback ``zmfub = 0.01 kg m⁻² s⁻¹``
    (``mo_cumastr.f90:567``), floored/capped to ``[cmfcmin, cmfcmax]``. This
    is the value ECHAM uses for the cloud-base mass flux when the boundary-
    layer moisture-budget closure ``zdqpbl/(g·Δq)`` is not applicable; the
    live scheme (``tiedtke_nordeng_convection``) applies that moisture
    closure directly and the Nordeng ``zmfub1`` rescale for deep columns, so
    this function supplies only the constant fallback.

    It replaces the former ``cape/(g·τ)`` "closure", which had units of
    m s⁻¹ — dimensionally a velocity, not a mass flux (review finding 2.5).
    ``cape``, ``cin``, ``moisture_conv`` and ``ktype`` are accepted for
    backward compatibility with existing call sites but are not used: ECHAM's
    fallback is a single constant independent of type.

    Args:
        cape: CAPE (J/kg) — unused.
        cin: CIN (J/kg) — unused.
        moisture_conv: Low-level moisture convergence (kg/m²/s) — unused.
        ktype: Convection type (1=deep, 2=shallow, 3=mid) — unused.
        config: Convection configuration (supplies the clip bounds).

    Returns:
        Cloud base mass flux (kg/m²/s).

    """
    del cape, cin, moisture_conv, ktype
    return jnp.clip(
        jnp.asarray(ECHAM_MFUB_FALLBACK), config.cmfcmin, config.cmfcmax,
    )


def mass_flux_closure_blend(
    cape: jnp.ndarray,
    cin: jnp.ndarray,
    moisture_conv: jnp.ndarray,
    type_weights: jnp.ndarray,
    config: ConvectionParameters,
) -> jnp.ndarray:
    """Type-weighted cloud-base mass-flux fallback.

    Type-weighted counterpart of :func:`mass_flux_closure`. Because ECHAM's
    fallback (``mo_cumastr.f90:567``) is the SAME constant ``0.01 kg m⁻² s⁻¹``
    for all convection types, the type weighting is degenerate and this
    returns the same faithful constant as :func:`mass_flux_closure`. Kept as
    a separate entry point for the (differentiable) live call site.

    Args:
        cape: CAPE (J/kg) — unused.
        cin: CIN (J/kg) — unused.
        moisture_conv: Low-level moisture convergence (kg/m2/s) — unused.
        type_weights: ``(3,)`` (deep, shallow, mid) weights — unused (the
            fallback constant is type-independent).
        config: Convection configuration (supplies the clip bounds).

    Returns:
        Cloud base mass flux (kg/m2/s).

    """
    del cape, cin, moisture_conv, type_weights
    return jnp.clip(
        jnp.asarray(ECHAM_MFUB_FALLBACK), config.cmfcmin, config.cmfcmax,
    )