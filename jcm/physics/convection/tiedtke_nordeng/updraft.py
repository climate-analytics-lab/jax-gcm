"""Updraft calculations for Tiedtke-Nordeng convection scheme

This module implements the updraft calculations including:
- Cloud base determination
- Entrainment and detrainment
- Moist ascent with condensation
- Buoyancy calculations

Based on ICON mo_cuascent.f90

"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple

import jcm.constants as c
from .tiedtke_nordeng import ConvectionParameters
# The ECHAM cuadjtq-style damped Newton adjustment. It lives in
# jcm.physics.convection.saturation so that ``calculate_cape_cin`` (in
# tiedtke_nordeng.py, which this module imports from) can call it too;
# re-exported here under its historical name for callers and tests.
from jcm.physics.convection.saturation import (
    cuadjtq_newton as saturation_adjustment,
)


#: Ceiling on the organized-detrainment tan-profile fractional height
#: ``zzmzk/ztmzk``. ECHAM's ``tan(π·frac/2)`` diverges at ``frac == 1``
#: (cloud top); the ``centrmax`` cap below already saturates the top layers
#: for any physical cloud depth, so clipping the argument strictly below
#: π/2 changes nothing physically while keeping ``tan`` — and its VJP —
#: finite (no ``0·inf`` gradient poison, jax-gcm#558/#559).
_ORG_DETR_FRAC_MAX = 0.98


class UpdatedraftState(NamedTuple):
    """State variables for updraft calculation"""

    tu: jnp.ndarray      # Updraft temperature (K)
    qu: jnp.ndarray      # Updraft specific humidity (kg/kg)
    lu: jnp.ndarray      # Updraft liquid water (kg/kg) — after per-layer precip removal
    mfu: jnp.ndarray     # Updraft mass flux (kg/m²/s)
    entr: jnp.ndarray    # Entrainment rate (1/m)
    detr: jnp.ndarray    # Detrainment rate (1/m)
    buoy: jnp.ndarray    # Buoyancy (m/s²)
    pdmfup: jnp.ndarray  # Precip generated per layer (kg/m²/s) — ECHAM ``pdmfup``
    plude: jnp.ndarray   # Condensate DETRAINED per layer (kg/m²/s) — ECHAM ``plude``.
                         # Feeds the stratiform cloud tracers (ECHAM pxtecl/pxteci
                         # via zxtec = g/Δp·plude) and the cudtdq latent-heat
                         # ledger; includes the cloud-top dump of the remaining
                         # plume condensate when the updraft terminates.
    uu: jnp.ndarray      # Updraft zonal wind (m/s) — ECHAM ``puu`` (cuasc),
                         # mass-weighted mixing of the lifted plume wind and the
                         # entrained environmental wind. Consumed by cududv's
                         # momentum-transport deviation flux ``mfu·(uu − ū)``.
    vu: jnp.ndarray      # Updraft meridional wind (m/s) — ECHAM ``pvu``.


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
) -> UpdatedraftState:
    """Calculate full updraft profile

    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        rho: Air density (kg/m³) [nlev]
        kbase: Cloud base level index
        ktop: Cloud top level index
        ktype: Convection type
        mass_flux_base: Cloud base mass flux (kg/m²/s)
        config: Convection configuration
        land_fraction: Fraction of column underlying land surface (0=open
            ocean, 1=land). Selects ECHAM's per-surface ``zdnoprc``
            threshold via ``config.cu_dnoprc_ocean`` and
            ``config.cu_dnoprc_land``. Defaults to 0 (ocean) so existing
            single-column tests behave as before.
        type_weights: Smooth (deep, shallow, mid) weights; ``None`` falls
            back to the one-hot of ``ktype``.
        lift: ECHAM ``zlift`` [K], the sub-grid buoyancy excess. Applied to
            the buoyancy test at the FIRST ascent level above cloud base and
            ONLY for a mid-level (``ktype == 3``) plume — see the
            termination comment below for why that is the reference's one
            legitimate site for it.

    Returns:
        UpdatedraftState with computed profiles

    """
    nlev = len(temperature)
    # Environmental winds for the prognostic plume wind (cududv). Column
    # callers that do not transport momentum (older tests) pass none; a
    # zero wind then makes the plume wind identically zero, which is inert.
    if u_wind is None:
        u_wind = jnp.zeros(nlev)
    if v_wind is None:
        v_wind = jnp.zeros(nlev)
    # Linear blend of ocean/land precip-zone threshold by land fraction —
    # smooth in land_fraction so the column gradient is well-defined.
    zdnoprc_col = (
        (1.0 - land_fraction) * config.cu_dnoprc_ocean
        + land_fraction * config.cu_dnoprc_land
    )

    # Initialize updraft state at cloud base
    tu_init = jnp.zeros(nlev)
    qu_init = jnp.zeros(nlev)
    lu_init = jnp.zeros(nlev)
    mfu_init = jnp.zeros(nlev)
    entr_init = jnp.zeros(nlev)
    detr_init = jnp.zeros(nlev)
    buoy_init = jnp.zeros(nlev)
    pdmfup_init = jnp.zeros(nlev)
    plude_init = jnp.zeros(nlev)
    uu_init = jnp.zeros(nlev)
    vu_init = jnp.zeros(nlev)

    # Set cloud base values. The parcel arriving at the LCL has the
    # surface mixing ratio (q is conserved during dry-adiabatic ascent).
    # Where ``find_cloud_base`` picks the first discrete level above the
    # true LCL, the parcel is already supersaturated there, so we run the
    # same damped ``cuadjtq`` Newton step the interior of the plume uses:
    # the condensate warms the parcel by L/cp, and the vapour it is left
    # with is the saturation value at that *warmed* temperature. This is
    # ECHAM ``cubase`` (mo_cuinitialize.f90:296-314): ``cuadjtq`` with
    # icall=1 followed by ``plu = plu + zqold - pqu``, i.e. the condensate
    # stays in the plume and total water is conserved exactly.
    #
    # Doing this by hand — condensing all of ``q - qs(T_dry)`` without the
    # ``1 + (L/cp)·dqs/dT`` denominator, then re-saturating at the warmed
    # temperature — created water (up to +42% here, worse on coarse grids)
    # and over-warmed the parcel by several K, so every plume started too
    # buoyant with invented condensate (issue #661). Assigning
    # ``qu = qs(T)`` unconditionally also moistened a *sub*saturated parcel
    # up to saturation with no latent-heat debit at all; the Newton step
    # leaves such a parcel untouched, which is the correct behaviour and
    # matches CAM's UW scheme setting ``qv = qt`` when unsaturated.
    surf_idx = jnp.argmax(pressure)
    surf_temp = temperature[surf_idx]
    surf_humid = humidity[surf_idx]
    surf_press = pressure[surf_idx]

    parcel_T_dry_at_cb = surf_temp * (pressure[kbase] / surf_press) ** (c.rd / c.cpd)
    tu_cb, qu_cb, lu_cb = saturation_adjustment(
        parcel_T_dry_at_cb, surf_humid, pressure[kbase],
    )

    # A MID-LEVEL plume has no surface connection: ECHAM ``cubasmc`` seeds it
    # from the ENVIRONMENT at its own base (mo_cuascent.f90:620-622,
    # ``pqu = pqen(kk)``, ``plu = 0``). Lifting a surface parcel to a base
    # 3+ km up, as the ``cubase`` branch above does, would be the wrong
    # parcel entirely — that is the whole point of the separate trigger.
    #
    # ECHAM's ``ptu`` is the full-level environmental temperature brought
    # adiabatically DOWN to the layer's bottom interface, from which the
    # ascent lifts it back through the layer. On jcm's full-level grid the
    # base *is* the full level, so the two half-layer displacements are
    # absent and the seed is the environment there — the scheme-wide
    # staggering approximation (#530), not something specific to this seed.
    is_midlevel = (ktype == 3)
    tu_cb = jnp.where(is_midlevel, temperature[kbase], tu_cb)
    qu_cb = jnp.where(is_midlevel, humidity[kbase], qu_cb)
    lu_cb = jnp.where(is_midlevel, 0.0, lu_cb)

    tu_init = tu_init.at[kbase].set(tu_cb)
    qu_init = qu_init.at[kbase].set(qu_cb)
    lu_init = lu_init.at[kbase].set(lu_cb)
    mfu_init = mfu_init.at[kbase].set(mass_flux_base)
    # Plume wind at cloud base: the environmental wind there (ECHAM ``cubase``
    # sets ``puu(kcbot)`` from the sub-cloud source). Both surface-triggered
    # and mid-level (cubasmc) plumes start with the base-level environment
    # wind; the ascent then mixes in entrained environmental momentum.
    uu_init = uu_init.at[kbase].set(u_wind[kbase])
    vu_init = vu_init.at[kbase].set(v_wind[kbase])

    buoy_init = buoy_init.at[kbase].set(0.0)  # Neutral at cloud base

    updraft_init = UpdatedraftState(
        tu=tu_init, qu=qu_init, lu=lu_init,
        mfu=mfu_init, entr=entr_init, detr=detr_init,
        buoy=buoy_init,
        pdmfup=pdmfup_init,
        plude=plude_init,
        uu=uu_init, vu=vu_init,
    )
    # Carry = (updraft_state, integrated_buoyancy). The integrated
    # buoyancy drives Nordeng (1994) organized entrainment and is kept
    # *outside* UpdatedraftState so external callers see the same type.
    initial_state = (updraft_init, jnp.zeros(()))
    
    # Prepare inputs for scan (extract config parameters to avoid passing object).
    # ``p_base_const`` carries the cloud-base pressure as a per-level constant
    # so the precip-zone gate (zdnoprc threshold) inside the scan can compare
    # against it.
    k_levels = jnp.arange(nlev)
    p_base_const = jnp.full(nlev, pressure[kbase])
    # Geopotential height of each full level above the surface [m], for the
    # metre-based organized detrainment (mo_cuascent.f90:779-784). Top-first
    # cumulative thickness: ``heights[k]`` is larger for higher levels, so
    # ``heights[ktop] > heights[kbase]`` and the cloud depth is positive.
    heights = jnp.cumsum(layer_thickness[::-1])[::-1]
    z_base_const = jnp.full(nlev, heights[kbase])
    z_top_const = jnp.full(nlev, heights[ktop])
    # Environmental temperature one level ABOVE (smaller index, top-first),
    # for the ``zdrodz`` density-gradient term in the organized entrainment.
    temp_above = jnp.concatenate([temperature[:1], temperature[:-1]])
    # Type-blended base entrainment and deep-convection weight. With the
    # smooth type selection (tiedtke_nordeng.py) the per-type entrainment
    # rates combine by the softmax weights instead of a hard ktype
    # select, so entrpen/entrscv/entrmid keep gradients across the type
    # thresholds; w_deep likewise gates the Nordeng organized
    # entrainment/detrainment smoothly. ``type_weights=None`` (legacy
    # callers/tests) falls back to the hard one-hot of ktype.
    if type_weights is None:
        type_weights = jnp.stack([
            jnp.asarray(ktype == 1, dtype=temperature.dtype),
            jnp.asarray(ktype == 2, dtype=temperature.dtype),
            jnp.asarray(ktype == 3, dtype=temperature.dtype),
        ])
    entr_base_blend = (type_weights[0] * config.entrpen
                       + type_weights[1] * config.entrscv
                       + type_weights[2] * config.entrmid)
    w_deep = type_weights[0]
    level_inputs = (
        k_levels, temperature, humidity, pressure, layer_thickness, rho,
        jnp.full(nlev, kbase), jnp.full(nlev, ktop),
        jnp.full(nlev, ktype),
        jnp.full(nlev, entr_base_blend), jnp.full(nlev, w_deep),
        jnp.full(nlev, config.smooth_term_buoy),
        jnp.full(nlev, config.smooth_term_mf),
        jnp.full(nlev, config.smooth_precip_pa),
        jnp.full(nlev, config.cprcon),
        p_base_const,
        jnp.full(nlev, zdnoprc_col),
        u_wind, v_wind, temp_above, heights, z_base_const, z_top_const,
        jnp.full(nlev, config.cu_centrmax),
    )

    # Create specialized step function with config parameters
    def updraft_step_with_config(carry_tuple, inputs):
        carry, zbuoy_accum = carry_tuple
        (k, env_temp, env_q, pressure, dz, rho, kbase, ktop, ktype,
         entr_base_in, w_deep_in, w_term_buoy, w_term_mf, w_precip,
         cprcon, p_at_base, zdnoprc,
         env_u, env_v, env_temp_above, z_k, z_base, z_top, centrmax) = inputs

        # Skip if outside cloud layer or at cloud base (boundary condition)
        in_cloud_interior = jnp.logical_and(
            jnp.minimum(ktop, kbase) < k,
            k < jnp.maximum(ktop, kbase)
        )
        at_cloud_top = (k == ktop)
        should_compute = jnp.logical_or(in_cloud_interior, at_cloud_top)
        skip = jnp.logical_not(should_compute)

        def compute_updraft():
            # Type-blended base turbulent entrainment rate (see above)
            entr_base = entr_base_in

            # Turbulent entrainment is the PLAIN fractional rate. ECHAM's
            # cuentr (mo_cuascent.f90:746) has NO humidity dependence:
            # zentr = pentr*pmfu*zdprho*zrrho. The previous IFS-style
            # (1 + 2*(1-RH)^2) enhancement tripled entrainment into dry
            # environments, so plumes died before penetrating a dry free
            # troposphere — locking coupled runs in a desiccated fixed
            # point (dry FT -> no deep convection -> no upward moisture
            # transport -> dry FT; TPW pinned at ~1.5 kg/m2). ECHAM's
            # only entrainment ENHANCEMENT is the moisture-convergence
            # term below the cloud-water minimum level (cuentr:758-760,
            # zentest = MAX(pqte,0)/pqenh) — not yet ported (needs the
            # accumulated moisture tendency); tracked as a follow-up.
            entr_turb = jnp.clip(entr_base, 0.0, 0.01)

            # Nordeng (1994) organized entrainment for deep convection
            # (ECHAM ``mo_cuascent.f90:517-526``). The organized rate is the
            # local-buoyancy contribution ``zbuoyz·0.5/(1+∫buoyancy)`` PLUS
            # the density-scale-height gradient ``zdrodz``, then clamped to
            # ``[0, centrmax]`` — ECHAM's exact clamp order (MIN then MAX).
            next_level_for_buoy = jnp.minimum(k + 1, nlev - 1)
            prev_buoy = carry.buoy[next_level_for_buoy]
            # Only positive buoyancy drives organized entrainment (proxy:
            # previous-level buoyancy, one scan step behind).
            zbuoyz = jnp.maximum(prev_buoy, 0.0)
            # ``zdrodz`` (mo_cuascent.f90:521-522): the density lapse of the
            # environment, ≈ −1e-4 m⁻¹ in the lower troposphere — comparable
            # to ``entrpen`` itself. Omitting it (the prior code) left
            # organized entrainment systematically too large and removed the
            # ``MAX(...,0)`` mechanism that clamps entrainment to zero in
            # weakly buoyant layers. Full-level env T/q stand in for ECHAM's
            # half-level ptenh/pqenh (the scheme-wide #530 staggering).
            zdz = jnp.maximum(dz, 1.0)
            zdrodz = (
                -jnp.log(env_temp_above / env_temp) / zdz
                - c.grav / (c.rd * env_temp * (1.0 + c.vtmpc1 * env_q))
            )
            zoentr = jnp.clip(
                zbuoyz * 0.5 / (1.0 + zbuoy_accum) + zdrodz, 0.0, centrmax,
            )
            # Organized entrainment scales with the smooth deep weight
            # (1 for a solidly deep column; fades across the type boundary
            # instead of switching). The previous code capped the SUM
            # entr_turb+entr_org at 0.01 m⁻¹ (33× centrmax): ECHAM caps the
            # ORGANIZED rate alone at centrmax, so that clip is dropped here.
            entr_org = w_deep_in * zoentr
            entr = entr_turb + entr_org

            # Turbulent detrainment equals turbulent entrainment (ECHAM
            # cuentr: zdmfde = zdmfen for the turbulent part — δ = ε; the
            # previous 0.5·ε under-detrained and over-deepened the plume).
            detr_turb = entr_turb

            # Organized detrainment (ECHAM ``mo_cuascent.f90:769-784``): a
            # tan profile in HEIGHT (metres) whose prefactor scales as
            # 1/(cloud depth), capped at ``centrmax``. ``ztmzk`` is the cloud
            # depth and ``zzmzk = z(k) − z(kbase)`` the height above cloud
            # base. ECHAM's lower bound is ``khmin`` (the MSE-minimum onset
            # level); jcm's single-pass ascent does not compute khmin, so
            # cloud base is used as the onset — the tan form itself keeps
            # near-base detrainment negligible (tan(0)=0). This replaces the
            # previous LEVEL-COUNT profile ``0.003·sqrt(depth/10)·tan(...)``,
            # which INCREASED with vertical resolution at fixed physics (the
            # wrong sign) and was uncapped — detraining up to ~19× the plume
            # mass in one layer against ECHAM's ceiling of centrmax·dz.
            ztmzk = jnp.maximum(z_top - z_base, 1.0)          # cloud depth [m]
            zzmzk = jnp.clip(z_k - z_base, 0.0, ztmzk)
            frac_height = jnp.minimum(zzmzk / ztmzk, _ORG_DETR_FRAC_MAX)
            zorgde = jnp.tan(jnp.pi * frac_height * 0.5) * jnp.pi * 0.5 / ztmzk
            detr_org = w_deep_in * jnp.minimum(zorgde, centrmax)

            detr = detr_turb + detr_org

            # Safe array indexing - clamp k+1 to valid range
            next_level = jnp.minimum(k + 1, nlev - 1)

            # Mass flux change. ECHAM ``cuasc`` line 500 caps the detrained
            # MASS at 0.75 of the plume mass entering the layer, so
            # turbulent+organized detrainment can never remove more than 75%
            # of the plume in one layer — and the ``plude`` condensate ledger
            # (∝ dmf_detr) can never exceed the available plume mass.
            dmf_entr = entr * carry.mfu[next_level] * dz
            dmf_detr = jnp.minimum(
                detr * carry.mfu[next_level] * dz,
                0.75 * carry.mfu[next_level],
            )

            # Update mass flux
            mfu_new = jnp.maximum(carry.mfu[next_level] + dmf_entr - dmf_detr, 0.0)
            
            # Proper mixing with entrainment
            # When mass flux is negligible, use environmental values instead of dividing by tiny numbers
            mfu_threshold = 1e-6  # kg/m²/s - below this, updraft is negligible

            def compute_updraft_properties():
                # Mass-weighted mixing of the updraft air (lifted adiabatically
                # from the level below) and entrained environmental air.
                #
                # Dry static energy (DSE = cp·T + g·z) is conserved during
                # adiabatic ascent. Equivalently, a parcel rising by dz cools
                # by g·dz/cp (~9.8 K/km), so mixing is done in DSE, not T
                # directly: mixing T without the adiabatic cooling leaves the
                # parcel ~10 K too warm at each level, the saturation
                # adjustment never sees supersaturation, and no liquid or
                # precipitation forms.
                #
                # Detrainment removes mass at *updraft* properties, so the
                # correct denominator for mixing is the pre-detrainment mass
                # (mfu_below + dmf_entr), NOT mfu_new.
                mfu_mix = jnp.maximum(
                    carry.mfu[next_level] + dmf_entr, 1e-10
                )
                # Adiabatic cooling of the updraft air as it rises by dz
                adiabatic_cooling = c.grav * dz / c.cpd
                tu_lifted = carry.tu[next_level] - adiabatic_cooling

                total_water = (
                    (carry.qu[next_level] + carry.lu[next_level])
                    * carry.mfu[next_level]
                    + env_q * dmf_entr
                ) / mfu_mix
                temp_mix = (
                    tu_lifted * carry.mfu[next_level]
                    + env_temp * dmf_entr
                ) / mfu_mix

                # Saturation adjustment (iterative Newton; cuadjtq kcall=1)
                return saturation_adjustment(temp_mix, total_water, pressure)

            def use_environmental_values():
                # When updraft mass flux is negligible, use environmental values
                return env_temp, env_q, jnp.array(0.0)

            # Use environmental values when mass flux is too small
            tu_new, qu_new, lu_new = lax.cond(
                mfu_new > mfu_threshold,
                compute_updraft_properties,
                use_environmental_values
            )

            # Prognostic plume wind (ECHAM ``cuasc`` ``puu``/``pvu``): the
            # wind is a PASSIVE scalar mass-weighted between the plume air
            # lifted from below and the entrained environmental wind — the
            # same entrainment weighting the DSE/moisture mixing above uses
            # (the pre-detrainment mass ``mfu_below + dmf_entr``). ECHAM's
            # ``cuasc`` adds a momentum-specific ``zz`` entrainment
            # enhancement (lines 494-499) that jcm's simplified entrainment
            # does not carry; using the thermodynamic entrainment weighting
            # here is the leading-order behaviour and is consistent with the
            # rest of the port. Where the plume is negligible the wind
            # follows the environment.
            mfu_mix_wind = jnp.maximum(carry.mfu[next_level] + dmf_entr, 1e-10)
            uu_mixed = (
                carry.uu[next_level] * carry.mfu[next_level] + env_u * dmf_entr
            ) / mfu_mix_wind
            vv_mixed = (
                carry.vu[next_level] * carry.mfu[next_level] + env_v * dmf_entr
            ) / mfu_mix_wind
            uu_new = jnp.where(mfu_new > mfu_threshold, uu_mixed, env_u)
            vu_new = jnp.where(mfu_new > mfu_threshold, vv_mixed, env_v)

            # Per-layer precipitation generation (ECHAM cuasc lines 454-457).
            # The parcel converts a fraction of its liquid water to precip
            # in each layer it ascends through:
            #
            #   zlnew  = plu(jk) / (1 + cprcon * (geoh(jk) - geoh(jk+1)))
            #   pdmfup = max(0, (plu(jk) - zlnew) * pmfu(jk))
            #   plu(jk) = zlnew
            #
            # ECHAM gates this on a thickness threshold ``zdnoprc``: precip
            # is only generated when the level is more than ``zdnoprc`` Pa
            # above cloud base. ECHAM uses different thresholds over
            # ocean vs land (continental convection has a thicker non-
            # precipitating layer near cloud base); the per-column value
            # is built in ``calculate_updraft`` from
            # ``config.cu_dnoprc_ocean`` / ``config.cu_dnoprc_land``
            # blended by ``land_fraction``.
            # Smooth precip onset: the hard ``depth >= zdnoprc`` gate put
            # zdnoprc only in an inequality (identically zero gradient —
            # unlearnable). A sigmoid over the depth excess makes the
            # ocean/land onset thresholds calibratable; width → 0
            # recovers the hard gate.
            precip_zone_w = jax.nn.sigmoid(
                ((p_at_base - pressure) - zdnoprc) / w_precip
            )
            geoh_diff = c.grav * dz  # ≈ pgeoh(jk) - pgeoh(jk+1) in ECHAM
            cprcon_eff = jnp.where(
                mfu_new > mfu_threshold, cprcon * precip_zone_w, 0.0,
            )
            lu_after_precip = lu_new / (1.0 + cprcon_eff * geoh_diff)
            pdmfup = jnp.maximum(
                (lu_new - lu_after_precip) * mfu_new, 0.0,
            )
            lu_new = lu_after_precip

            # Detrained condensate (ECHAM ``plude = zdmfde·plu``,
            # mo_cuascent.f90): the mass detrained in this layer carries the
            # condensate the plume brought into it. This is the source the
            # cudtdq ledger heats with (+L·plude) and the stratiform cloud
            # tracers receive (g/Δp·plude → dqc/dqi) — previously never
            # computed, so detrained condensate simply vanished.
            plude_layer = carry.lu[next_level] * dmf_detr

            # Calculate buoyancy
            virtual_temp_u = tu_new * (1.0 + 0.608 * qu_new - lu_new)
            virtual_temp_e = env_temp * (1.0 + 0.608 * env_q)
            buoy_new = c.grav * (virtual_temp_u - virtual_temp_e) / virtual_temp_e

            # Dynamic cloud-top termination: once above cloud base the parcel
            # becomes negatively buoyant (or the mass flux has already dropped
            # below 1% of the base value — ECHAM's termination criterion in
            # `mo_cuascent.f90`), terminate the updraft here. This replaces the
            # previous fixed `ktop` which ignored the environment.
            # Tapered cloud-top termination (review B.2.3). The hard
            # ``where(buoy < 0 | mfu < 1%·mfb, 0, mfu)`` had a cliff:
            # precip smooth in entrpen up to a critical value, then
            # exactly 0 with zero gradient beyond. The survival fraction
            # is now the product of two sigmoids — buoyancy (width
            # ~3e-4 m/s² ≈ 0.01 K, so a solidly buoyant plume keeps
            # >99.99% of its flux) and the mass-flux floor. The
            # non-surviving fraction detrains its condensate here,
            # exactly like the previous all-or-nothing dump; widths → 0
            # recover the hard termination.
            above_cloud_base = k < kbase
            # ECHAM's sub-grid buoyancy bonus ``zlift`` applies to exactly
            # one ascent test: ``IF(klab(jk+1).EQ.1) zbuo = zbuo + zlift``
            # (mo_cuascent.f90:449) — the level below must still be
            # SUB-CLOUD. ``cubase`` sets ``klab(kcbot) = 2``, so a
            # surface-triggered plume never qualifies at any level; the only
            # ``klab == 1`` above cloud base is the one ``cubasmc`` writes
            # when it seeds a mid-level plume (mo_cuascent.f90:654). Hence
            # the bonus here is gated on BOTH ``ktype == 3`` and the first
            # step above the base, and is otherwise identically zero.
            #
            # The bonus lives only at this one site the reference uses it
            # (the mid-level ``cubasmc`` plume seed); ``find_cloud_base`` does
            # ECHAM's ``klab`` walk and returns the LCL, so no level above the
            # base needs it to stay alive (#684/#691).
            #
            # ``buoy_new`` is an acceleration (g·ΔTv/Tv), ``lift`` a
            # temperature, so the bonus converts with g/Tv_env.
            first_step_above_base = jnp.logical_and(
                above_cloud_base, k == kbase - 1,
            )
            lift_accel = jnp.where(
                jnp.logical_and(first_step_above_base, ktype == 3),
                c.grav * lift / virtual_temp_e,
                0.0,
            )
            surv_buoy = jax.nn.sigmoid((buoy_new + lift_accel) / w_term_buoy)
            surv_mf = jax.nn.sigmoid(
                (carry.mfu[next_level] / jnp.maximum(mass_flux_base, 1e-10)
                 - 0.01) / w_term_mf
            )
            survival = jnp.where(above_cloud_base, surv_buoy * surv_mf, 1.0)
            # Forced total detrainment at the scan ceiling (ECHAM
            # mo_cuasc.f90:540-563: at cloud top the plume fully detrains —
            # ``plude(jk-1) = pmful(jk)`` dumps the ENTIRE remaining condensate
            # flux). With the metre-based capped detrainment a still-buoyant
            # plume can now reach the supplied ``ktop`` with positive mfu/lu;
            # without this, its residual ``lu·mfu`` would leave through the top
            # interface as a flux-boundary loss instead of feeding the
            # stratiform dqc/dqi ledger, under-supplying anvil condensate and
            # breaking column water conservation at the ceiling (Codex P2).
            # Forcing survival to 0 at ``ktop`` routes the whole remaining
            # plume condensate to ``plude`` and terminates mfu there.
            survival = jnp.where(at_cloud_top, 0.0, survival)
            plude_layer = plude_layer + lu_new * mfu_new * (1.0 - survival)
            mfu_new = mfu_new * survival

            # Update state
            new_state = carry._replace(
                tu=carry.tu.at[k].set(tu_new),
                qu=carry.qu.at[k].set(qu_new),
                lu=carry.lu.at[k].set(lu_new),
                mfu=carry.mfu.at[k].set(mfu_new),
                entr=carry.entr.at[k].set(entr),
                detr=carry.detr.at[k].set(detr),
                buoy=carry.buoy.at[k].set(buoy_new),
                pdmfup=carry.pdmfup.at[k].set(pdmfup),
                plude=carry.plude.at[k].set(plude_layer),
                uu=carry.uu.at[k].set(uu_new),
                vu=carry.vu.at[k].set(vu_new),
            )
            # Accumulate integrated positive buoyancy for the next step's
            # organized-entrainment denominator (matches ECHAM `zbuoy`).
            # Use the JUST-COMPUTED ``buoy_new`` rather than ``zbuoyz``
            # (which is the previous level's buoyancy used as a proxy for
            # the local rate). Without this, after a positive-buoyancy
            # step the accumulator stays at zero and the next level's
            # ``entr_org = zbuoyz·0.5/(1+zbuoy_accum)`` saturates against
            # the 1.0 floor — diluting the parcel by ~85% on every step
            # and killing the updraft after 1-2 levels.
            buoy_pos = jnp.maximum(buoy_new, 0.0)
            new_accum = zbuoy_accum + buoy_pos * dz
            return (new_state, new_accum)

        # Skip calculation if below cloud base: state and accumulator unchanged
        updated_tuple = lax.cond(
            skip,
            lambda: (carry, zbuoy_accum),
            compute_updraft,
        )
        return updated_tuple, updated_tuple[0]
    
    # Use scan to compute updraft from bottom to top. The scan carry is
    # (UpdatedraftState, integrated_buoyancy); we return only the state.
    final_carry, _ = lax.scan(
        updraft_step_with_config,
        initial_state,
        level_inputs,
        reverse=True,  # Go from bottom to top
    )
    final_state, _zbuoy_total = final_carry
    return final_state