"""Flux calculations and tendency updates for Tiedtke-Nordeng convection

This module implements:
- Final mass flux adjustments
- Temperature and moisture tendency calculations
- Momentum transport
- Precipitation and cloud water/ice partitioning

Based on ICON mo_cufluxdts.f90

Date: 2025-01-09
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple

import jcm.constants as c
from .tiedtke_nordeng import ConvectionParameters, ConvectionTendencies
from .updraft import UpdatedraftState
from .downdraft import DowndraftState


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

    The previous implementation returned ``sum(mfu*lu) * cprcon`` —
    i.e. it ignored the per-layer precip removal step entirely. As a
    result the surface precip estimate was ~60x too small (typical
    0.008 mm/day on a tropical RCE column vs. ECHAM's ~0.5 mm/day),
    AND the liquid water built up unphysically inside the parcel as it
    rose — distorting the buoyancy and terminating the updraft early.

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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        dp_lev: Per-layer pressure thickness [Pa] (positive).
        kbase: Cloud-base level index.
        pdmfup: Per-layer updraft precip generation [kg/m²/s] (≥ 0).
        pdmfdp: Per-layer downdraft precip sink [kg/m²/s] (≤ 0).
        dt: Time step [s].

    Returns:
        ``(rain_sfc, snow_sfc, prain, pdpmel, pdmfup_adj)`` — surface rain
        and snow fluxes, the production-only diagnostic ``prain``, the
        per-layer snow melt, and ``pdmfup`` including the (negative)
        sub-cloud evaporation increments.

    """
    nlev = len(temperature)
    zcons1 = c.cpd / (c.alhf * c.grav * dt)
    zcons2 = 1.0 / (c.grav * dt)
    ztmelp2 = c.tmelt + 2.0
    zcucov = 0.05  # fractional precip cover (cuflx line 419)

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
        return (zrfl, zsfl), zsnmlt

    gen = pdmfup + pdmfdp
    (prfl, psfl), pdpmel = lax.scan(
        partition_step, (jnp.zeros(()), jnp.zeros(())),
        (gen, temperature, humidity, dp_lev),
    )
    prfl = jnp.maximum(prfl, 0.0)
    psfl = jnp.maximum(psfl, 0.0)
    prain = jnp.sum(jnp.maximum(pdmfup, 0.0))

    # --- pass 2: sub-cloud Kessler evaporation (cuflx 411-440) ------------
    k_idx = jnp.arange(nlev)

    def evap_step(zpsubcl, xs):
        k, qs_k, q_k, dp_k, cevap_k = xs
        active = (k >= kbase) & (zpsubcl > 1e-20)
        zrfl = zpsubcl
        zrnew = (
            jnp.maximum(
                0.0,
                jnp.sqrt(jnp.maximum(zrfl, 0.0) / zcucov)
                - cevap_k * dp_k * jnp.maximum(qs_k - q_k, 0.0),
            ) ** 2
        ) * zcucov
        zrmin = zrfl - zcucov * jnp.maximum(0.8 * qs_k - q_k, 0.0) * zcons2 * dp_k
        zrnew = jnp.maximum(zrnew, zrmin)
        zrfln = jnp.maximum(zrnew, 0.0)
        zdrfl = jnp.where(active, jnp.minimum(0.0, zrfln - zrfl), 0.0)
        zpsubcl_new = jnp.where(active, zrfln, zpsubcl)
        return zpsubcl_new, zdrfl

    zpsubcl_final, zdrfl_per_level = lax.scan(
        evap_step, prfl + psfl, (k_idx, qs_env, humidity, dp_lev, cevapcu),
    )
    pdmfup_adj = pdmfup + zdrfl_per_level  # negative increments (cuflx 437)

    # --- pass 3: proportional depletion (cuflx 486-491) -------------------
    zrsum = prfl + psfl
    zdpevap_tot = zpsubcl_final - zrsum  # ≤ 0
    inv = 1.0 / jnp.maximum(zrsum, 1e-20)
    rain_sfc = jnp.maximum(prfl + zdpevap_tot * prfl * inv, 0.0)
    snow_sfc = jnp.maximum(psfl + zdpevap_tot * psfl * inv, 0.0)

    return rain_sfc, snow_sfc, prain, pdpmel, pdmfup_adj


def calculate_cloud_water_ice(
    temperature: jnp.ndarray,
    updraft_lw: jnp.ndarray,
    updraft_mf: jnp.ndarray,
    downdraft_mf: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Partition cloud condensate into liquid and ice
    
    Args:
        temperature: Temperature profile (K)
        updraft_lw: Updraft liquid water (kg/kg)
        updraft_mf: Updraft mass flux (kg/m²/s)
        downdraft_mf: Downdraft mass flux (kg/m²/s)
        
    Returns:
        Tuple of (cloud_water, cloud_ice) in kg/kg

    """
    # Temperature thresholds for ice formation
    t_ice = c.tmelt - 40.0  # All ice below this
    t_water = c.tmelt       # All water above this
    
    # Linear transition between water and ice
    ice_frac = jnp.clip((t_water - temperature) / (t_water - t_ice), 0.0, 1.0)
    water_frac = 1.0 - ice_frac
    
    # Net vertical mass flux
    net_mf = updraft_mf + downdraft_mf  # downdraft is negative
    
    # Cloud fraction estimate (simplified)
    cloud_frac = jnp.clip(net_mf / 0.1, 0.0, 1.0)  # 0.1 kg/m²/s for full cloud
    
    # In-cloud condensate
    in_cloud_lw = updraft_lw * updraft_mf / jnp.maximum(net_mf, 1e-10)
    
    # Grid-mean cloud water and ice
    cloud_water = cloud_frac * in_cloud_lw * water_frac
    cloud_ice = cloud_frac * in_cloud_lw * ice_frac
    
    return cloud_water, cloud_ice


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
    config: ConvectionParameters
) -> ConvectionTendencies:
    """Calculate final tendencies from convective fluxes

    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        pressure: Pressure (Pa) [nlev]
        rho: Air density (kg/m³) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        updraft_state: Computed updraft state
        downdraft_state: Computed downdraft state
        kbase: Cloud base level
        ktop: Cloud top level
        dt: Time step (s)
        config: Convection configuration

    Returns:
        ConvectionTendencies with all tendency terms

    """
    nlev = len(temperature)

    # Calculate mass flux divergence at each level using JAX-compatible operations

    # CRITICAL FIX: Use DRY STATIC ENERGY flux, not temperature flux!
    # ICON Fortran: pmfus = pmfu * (cp*T + geopotential)
    # This prevents the temperature blowup that was occurring

    # Compute geopotential at each level from layer thickness
    # Starting from surface (highest index), integrate upward
    # geopotential[k] = sum of layer_thickness[k:] * g
    heights_from_surface = jnp.cumsum(layer_thickness[::-1])[::-1]  # Reverse, cumsum, reverse back
    geopotential = c.grav * heights_from_surface

    # Dry static energy = cp*T + geopotential
    # The latent heat is handled separately through lh_source.
    # Per Tiedtke (1989) eq. 3.8 and ECHAM ``mo_cuflx``, the convective
    # tendency in the environment is the divergence of the *deviation*
    # flux M·(s_par − s̄), NOT M·s_par. The deviation flux carries the
    # implicit compensating-subsidence contribution: as the updraft
    # transports parcel DSE upward, an equal mass of environmental air
    # subsides and warms adiabatically. Without the s̄ subtraction,
    # the absolute s_par (~3·10⁵ J/kg) dominates and any small dmfu/dz
    # from entrainment produces unphysical heating of ~10³–10⁴ K/day.
    dse_env = c.cpd * temperature + geopotential
    dse_up = c.cpd * updraft_state.tu + geopotential
    dse_down = c.cpd * downdraft_state.td + geopotential

    # Deviation fluxes of dry static energy (W/m²)
    dse_flux_up = (dse_up - dse_env) * updraft_state.mfu
    dse_flux_down = (dse_down - dse_env) * downdraft_state.mfd

    # ECHAM ``mo_cufluxdts.f90`` writes the convective tendency as
    #     dT/dt = (g / Δp) · (F(k+1) − F(k)) / cp
    # where Δp = p_half(k+1) − p_half(k) (signed) and F is the deviation
    # flux M·(s_par − s̄). Both ``Δp`` and ``F(k+1) − F(k)`` flip sign
    # together with vertical ordering, so leaving them signed keeps the
    # formula ordering-agnostic. The previous implementation used
    # ``-diff(F)`` together with ``abs(Δp)`` — correct for surface-first
    # inputs but inverted for the TOA-first columns the running ICON
    # physics actually feeds in, leading to convective *cooling* of the
    # cloud layer in production runs.
    dp_signed = jnp.diff(pressure, axis=0)
    dse_flux_div = jnp.diff(dse_flux_up + dse_flux_down, axis=0)
    # Moisture deviation flux: same logic — env q is what gets displaced
    # by the updraft, so the convective drying tendency is governed by
    # the difference between parcel and env q.
    q_flux_div = jnp.diff(
        (updraft_state.qu - humidity) * updraft_state.mfu
        + (downdraft_state.qd - humidity) * downdraft_state.mfd,
        axis=0,
    )
    # Condensate flux divergence (ECHAM pmful = mfu·lu). In cudtdq the
    # T-ledger carries −Δ(L·pmful): moving condensate UP through a layer
    # boundary removes the latent heat that was released making it — the
    # previous code had this term with the OPPOSITE sign (+L·Δ(lu·mfu)),
    # cooling the layers where condensate was produced and heating where
    # it evaporated (review finding 0.1/PR-1.1).
    pmful_div = jnp.diff(updraft_state.lu * updraft_state.mfu, axis=0)

    # Layer mass per unit area for the flux divergence. NOTE on staggering: the
    # convective fluxes above are evaluated at FULL levels, so ``diff(flux)``
    # lives on the dual grid (between full-level centres) and its consistent mass
    # is the centre-to-centre spacing ``diff(pressure)`` — NOT the model layer
    # mass ρ·Δz. ECHAM ``cudtdq`` is a genuine finite-volume scheme whose fluxes
    # live at HALF levels and which divides by the model layer mass; matching it
    # would require reworking the updraft (``cuasc``) to carry half-level mass
    # fluxes. Swapping in ρ·Δz here *without* that rework mixes a dual-grid flux
    # with a model-grid mass (inconsistent staggering) and empirically worsens
    # the cloud-base noise, so we keep the self-consistent dual-grid form. The
    # sign is carried so the heating comes out positive regardless of ordering.
    layer_mass_per_area = dp_signed / c.grav  # kg/m² (signed), shape (nlev-1)

    # Flux-divergence parts of the cudtdq ledger (mo_cufluxdts.f90:649-662):
    #   zdtdt ∝ Δpmfus + Δpmfds − Δ(L·pmful)  (+ per-level sources below)
    #   zdqdt ∝ Δpmfuq + Δpmfdq + Δpmful      (− per-level sinks below)
    dtedt_k_levels = (dse_flux_div - c.alhc * pmful_div) / (
        c.cpd * layer_mass_per_area
    )
    dqdt_k_levels = (q_flux_div + pmful_div) / layer_mass_per_area

    # Per-level layer mass for the source/sink terms (positive, kg/m²).
    # Centered full-level spacing as the layer-thickness proxy — self-
    # consistent with the dual-grid flux divergence above (the half-level
    # restagger is tracked separately, #530). What matters for conservation
    # is that the SAME mass converts each per-level flux to a tendency and
    # back — the column budget then closes identically.
    dp_abs = jnp.abs(dp_signed)
    # Use the SAME dual-grid spacing the divergence terms are divided by
    # (extended to the last level with its edge value): with one common
    # mass convention the column integral of the divergence terms
    # telescopes exactly and the per-level source/sink terms cancel their
    # own conversions, so the water and enthalpy budgets close identically
    # regardless of grid stretching. Mixing the dual spacing (divergences)
    # with a centred spacing (sources) opened the enthalpy budget by ~25 %
    # on a stretched tropical sounding.
    dp_lev = jnp.concatenate([dp_abs, dp_abs[-1:]])
    mass_lev = dp_lev / c.grav  # kg/m², (nlev,)

    # ECHAM cuflx precipitation budget: rain/snow partition, snow melt
    # (pdpmel), sub-cloud Kessler evaporation charged back into pdmfup.
    rain_sfc, snow_sfc, prain, pdpmel, pdmfup_adj = convective_precip_fluxes(
        temperature, humidity, pressure, dp_lev, kbase,
        updraft_state.pdmfup, downdraft_state.pdmfdp, dt,
    )
    plude = updraft_state.plude

    # Per-level source/sink terms of the ledger (cudtdq lines 649-662):
    #   T: +L·(plude + pdmfup + pdmfdp) − alf·pdpmel
    #   q: −(plude + pdmfup + pdmfdp)
    # Heating where precip is generated / condensate detrained (both were
    # condensed inside the plume, and the vapour that made them must be
    # debited from the column — the missing sinks that previously created
    # water at the precipitation rate, review finding 0.1). Negative
    # pdmfup increments (sub-cloud evaporation) and pdmfdp (downdraft
    # evaporation) flip both signs locally: cooling + re-moistening.
    ledger_src = plude + pdmfup_adj + downdraft_state.pdmfdp
    # ECHAM keys the ledger latent heat ``zalv`` to the FULL-LEVEL
    # environment temperature: sublimation heat below the melting point
    # (mo_cufluxdts.f90 — the palvsh/zalv pair), condensation heat above.
    zalv = jnp.where(temperature > c.tmelt, c.alhc, c.alhs)
    dtedt_lev = (zalv * ledger_src - c.alhf * pdpmel) / (c.cpd * mass_lev)
    dqdt_lev = -ledger_src / mass_lev

    # Detrained condensate feeds the stratiform cloud tracers (ECHAM
    # zxtec = g/Δp·plude → pxtecl/pxteci split by the full-level
    # temperature), NOT the vapour budget.
    liquid_frac = jnp.where(temperature > c.tmelt, 1.0, 0.0)
    dqc_dt_plude = liquid_frac * plude / mass_lev
    dqi_dt_plude = (1.0 - liquid_frac) * plude / mass_lev

    # Normalization factor for tendencies (1 / signed layer_mass) — same
    # ordering-agnostic convention as for the temperature/moisture
    # tendency above.
    factor = 1.0 / layer_mass_per_area

    def calculate_momentum_transport():
        # ECHAM cududv (mo_cufluxdts.f90:874-960) deviation-flux form: the
        # tendency is the divergence of ``M·(u_plume − ū_upstream)`` with
        # the environment wind taken one level ABOVE the interface (the
        # deliberate ik = jk−1 upstream offset). The port does not carry a
        # prognostic plume wind (ECHAM builds puu/pvu in cubase/cuasc), so
        # the plume wind is approximated by the cloud-base environment wind
        # — the leading-order cududv behaviour (plume momentum is dominated
        # by its sub-cloud source). The previous invented "PGF relaxation"
        # term (0.3 efficiency toward cloud-base wind) had no ECHAM
        # counterpart and is removed.
        u_cloud_base = u_wind[kbase]
        v_cloud_base = v_wind[kbase]
        u_up = jnp.roll(u_wind, 1)  # upstream (jk−1) environment wind
        v_up = jnp.roll(v_wind, 1)
        net_mf = updraft_state.mfu + downdraft_state.mfd
        zmfu_u = net_mf * (u_cloud_base - u_up)
        zmfu_v = net_mf * (v_cloud_base - v_up)
        dudt_transport = jnp.diff(zmfu_u, axis=0) * factor
        dvdt_transport = jnp.diff(zmfu_v, axis=0) * factor
        return dudt_transport, dvdt_transport

    dudt_k_levels, dvdt_k_levels = lax.cond(
        config.lmfdudv,
        calculate_momentum_transport,
        lambda: (jnp.zeros(nlev-1), jnp.zeros(nlev-1)),
    )
    
    # Mask ONLY the flux-divergence parts to the cloud column (ECHAM's
    # ``IF(ldcum .AND. jk.GE.kctop-1)`` guard — its cudtdq loop then runs
    # all the way DOWN TO THE SURFACE, jk = ktopm2..klev, never truncating
    # at cloud base). The per-level ledger terms (plude, pdmfup, pdmfdp,
    # pdpmel) are already zero wherever the physics is inactive and MUST
    # flow through below cloud base: the sub-cloud Kessler evaporation
    # writes negative pdmfup there, and zeroing its cooling/moistening
    # while the surface precip is still depleted opens the water/energy
    # budgets and dries sub-cloud layers (Codex review on #550).
    k_indices = jnp.arange(nlev)
    cloud_bottom = jnp.maximum(ktop, kbase)
    cloud_top = jnp.minimum(ktop, kbase)
    # Include one level above cloud top for flux divergence (ktop-1 in ECHAM)
    conv_mask = (k_indices >= cloud_top - 1) & (k_indices <= cloud_bottom)

    div_dt = jnp.where(conv_mask, jnp.zeros(nlev).at[:-1].set(dtedt_k_levels), 0.0)
    div_dq = jnp.where(conv_mask, jnp.zeros(nlev).at[:-1].set(dqdt_k_levels), 0.0)
    dtedt = div_dt + dtedt_lev
    dqdt = div_dq + dqdt_lev
    dudt = jnp.where(conv_mask, jnp.zeros(nlev).at[:-1].set(dudt_k_levels), 0.0)
    dvdt = jnp.where(conv_mask, jnp.zeros(nlev).at[:-1].set(dvdt_k_levels), 0.0)

    # Surface precipitation = rain + snow after the full cuflx budget
    # (generation − downdraft consumption − sub-cloud evaporation, with the
    # snow phase carried through melting). Replaces sum(pdmfup), which
    # exported precip the column never paid for (review finding 0.1).
    precip_rate = rain_sfc + snow_sfc

    # In-plume condensate diagnostic (kg/kg where the updraft is active).
    qc_conv = jnp.where(updraft_state.mfu > 0, updraft_state.lu, 0.0)
    qi_conv = jnp.zeros_like(qc_conv)

    # Detrained-condensate tendencies (ECHAM zxtec = g/Δp·plude split by
    # full-level temperature into pxtecl/pxteci). Replaces the previous
    # dimensionless ``lu·0.1/dt`` stubs.
    dqc_dt = dqc_dt_plude
    dqi_dt = dqi_dt_plude
    
    return ConvectionTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dudt=dudt,
        dvdt=dvdt,
        qc_conv=qc_conv,
        qi_conv=qi_conv,
        precip_conv=precip_rate,
        dqc_dt=dqc_dt,
        dqi_dt=dqi_dt
    )


def mass_flux_closure(
    cape: jnp.ndarray,
    cin: jnp.ndarray,
    moisture_conv: jnp.ndarray,
    ktype: int,
    config: ConvectionParameters
) -> jnp.ndarray:
    """Determine cloud base mass flux using appropriate closure
    
    Args:
        cape: Convective available potential energy (J/kg)
        cin: Convective inhibition (J/kg)
        moisture_conv: Low-level moisture convergence (kg/m²/s)
        ktype: Convection type (1=deep, 2=shallow, 3=mid)
        config: Convection configuration
        
    Returns:
        Cloud base mass flux (kg/m²/s)

    """
    # Deep convection: CAPE closure
    def deep_closure():
        # Timescale for CAPE removal
        tau = config.tau
        
        # Mass flux to remove CAPE over timescale
        # Simplified - full version would iterate
        mf_cape = cape / (c.grav * tau)
        
        # Apply limits
        return jnp.clip(mf_cape, config.cmfcmin, config.cmfcmax)
    
    # Shallow convection: moisture convergence closure
    def shallow_closure():
        # Balance low-level moisture convergence
        # For shallow convection, also use CAPE but with different scaling
        # If no moisture convergence, use CAPE-based trigger for shallow convection
        cape_flux = cape / (c.grav * config.tau * 10.0)  # Weaker than deep convection
        moisture_flux = moisture_conv * 0.1  # Efficiency factor
        
        # Use the larger of the two triggers
        base_flux = jnp.maximum(cape_flux, moisture_flux)
        
        return jnp.clip(
            base_flux,
            config.cmfcmin * 10.0,  # Minimum for shallow convection
            config.cmfcmax * 0.3    # Lower limit for shallow
        )
    
    # Mid-level convection: hybrid closure
    def mid_closure():
        # Combination of CAPE and moisture
        return 0.5 * (deep_closure() + shallow_closure())
    
    # Select closure based on convection type using clipped index
    # Ensure index is in valid range [0, 2] for switch
    switch_index = jnp.clip(ktype - 1, 0, 2)
    
    return lax.switch(
        switch_index,
        [deep_closure, shallow_closure, mid_closure],
    )