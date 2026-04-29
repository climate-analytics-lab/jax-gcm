"""Updraft calculations for Tiedtke-Nordeng convection scheme

This module implements the updraft calculations including:
- Cloud base determination
- Entrainment and detrainment
- Moist ascent with condensation
- Buoyancy calculations

Based on ICON mo_cuascent.f90

Date: 2025-01-09
"""

import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple

from jcm.constants import (
    grav, cp, alhc, tmelt, eps, rd
)
from .tiedtke_nordeng import (
    ConvectionParameters, saturation_mixing_ratio, saturation_vapor_pressure
)


# Tetens coefficients matching `saturation_vapor_pressure` in
# tiedtke_nordeng.py: es = 610.78 * exp(a*tc/(tc+C)). The ice-phase `b` here
# matches the existing implementation (35.86) even though canonical Tetens
# ice uses 21.87 — consistency with the qs formula is what matters for the
# Newton step to converge against the same target.
_TETENS_A_WATER = 17.27
_TETENS_C_WATER = 237.3
_TETENS_A_ICE = 35.86
_TETENS_C_ICE = 265.5


def _saturation_mixing_ratio_and_derivative(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (qs, dqs/dT) for the Tetens formulation used in this package.

    This is the `dqsat_dT` that the ECHAM `cuadjtq` Newton step requires.
    Computed analytically so the Newton iteration is bit-reproducible under
    JIT without relying on autodiff through a saturation lookup table.
    """
    # Re-use the bound-safe saturation vapor pressure
    es = saturation_vapor_pressure(temperature)
    # `es_safe` matches the clipping inside `saturation_mixing_ratio`
    p_safe = jnp.maximum(pressure, 1.0)
    es_safe = jnp.minimum(es, 0.99 * p_safe)
    denom = jnp.maximum(p_safe - es_safe * (1.0 - eps), 1.0)
    qs = eps * es_safe / denom

    # des/dT from Tetens: des/dT = es * a*C / (tc + C)**2
    # Use water coefficients above freezing, ice coefficients below
    tc = temperature - tmelt
    a_water, c_water = _TETENS_A_WATER, _TETENS_C_WATER
    a_ice, c_ice = _TETENS_A_ICE, _TETENS_C_ICE
    des_dT_water = es * a_water * c_water / jnp.maximum((tc + c_water) ** 2, 1e-3)
    des_dT_ice = es * a_ice * c_ice / jnp.maximum((tc + c_ice) ** 2, 1e-3)
    des_dT = jnp.where(temperature > tmelt, des_dT_water, des_dT_ice)
    # dqs/dT: differentiate qs = eps*es / (p - (1-eps)*es)
    dqs_dT = eps * p_safe * des_dT / denom ** 2
    return qs, dqs_dT


class UpdatedraftState(NamedTuple):
    """State variables for updraft calculation"""

    tu: jnp.ndarray      # Updraft temperature (K)
    qu: jnp.ndarray      # Updraft specific humidity (kg/kg)
    lu: jnp.ndarray      # Updraft liquid water (kg/kg)
    mfu: jnp.ndarray     # Updraft mass flux (kg/m²/s)
    entr: jnp.ndarray    # Entrainment rate (1/m)
    detr: jnp.ndarray    # Detrainment rate (1/m)
    buoy: jnp.ndarray    # Buoyancy (m/s²)


def saturation_adjustment(
    temperature: jnp.ndarray,
    total_water: jnp.ndarray,
    pressure: jnp.ndarray,
    n_refine: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Newton-Raphson saturation adjustment (cuadjtq, kcall=1 flavour).

    Matches ECHAM/ICON ``mo_cuadjust.f90`` ``cuadjtq`` for the
    "condensation-only" mode used inside updrafts. The first iteration
    clips the Newton step to be non-negative (only condensation, never
    evaporation of pre-existing liquid). Subsequent refinement iterations
    allow both directions so Newton overshoot in one direction can be
    corrected.

    The Newton step:

        Δq = (q - qs(T)) / (1 + (L/cp) * dqs/dT)

    is the linearised solution to ``q - Δq = qs(T + L·Δq/cp)``. With one
    refinement the residual ``q - qs(T_adj)`` typically drops to <~0.5%
    even for strong supersaturation; the old single-pass implementation
    left parcels 3-30% off, under-releasing latent heat and cooling the
    mid-troposphere in RCE.

    Args:
        temperature: Temperature (K)
        total_water: Total water mixing ratio (kg/kg)
        pressure: Pressure (Pa)
        n_refine: Number of refinement iterations after the first
            condensation-only pass (Fortran cuadjtq uses 1 refinement).

    Returns:
        Tuple of (T_adj, vapour, liquid) with ``vapour + liquid == total_water``
        and ``vapour ≈ qs(T_adj)`` to within a fraction of a percent.

    """
    L_cp = alhc / cp

    def _first_pass(T, q_vap, liq):
        """Condensation-only Newton step (kcall=1)."""
        qs, dqs_dT = _saturation_mixing_ratio_and_derivative(T, pressure)
        cond = (q_vap - qs) / (1.0 + L_cp * dqs_dT)
        cond = jnp.maximum(cond, 0.0)
        return T + L_cp * cond, q_vap - cond, liq + cond

    def _refine_body(carry, _):
        """Refinement: allow both directions (kcall=0) to correct Newton
        overshoot, but only while there's liquid available to re-evaporate.
        """
        T, q_vap, liq = carry
        qs, dqs_dT = _saturation_mixing_ratio_and_derivative(T, pressure)
        cond = (q_vap - qs) / (1.0 + L_cp * dqs_dT)
        # Don't evaporate more than available liquid
        cond = jnp.maximum(cond, -liq)
        return (T + L_cp * cond, q_vap - cond, liq + cond), None

    T1, q1, liq1 = _first_pass(temperature,
                               total_water,
                               jnp.zeros_like(total_water))
    (T_adj, vapor, liquid), _ = lax.scan(
        _refine_body, (T1, q1, liq1), None, length=n_refine
    )
    return T_adj, vapor, liquid


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
    config: ConvectionParameters
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
        
    Returns:
        UpdatedraftState with computed profiles

    """
    nlev = len(temperature)

    # Initialize updraft state at cloud base
    tu_init = jnp.zeros(nlev)
    qu_init = jnp.zeros(nlev)
    lu_init = jnp.zeros(nlev)
    mfu_init = jnp.zeros(nlev)
    entr_init = jnp.zeros(nlev)
    detr_init = jnp.zeros(nlev)
    buoy_init = jnp.zeros(nlev)

    # Set cloud base values. The parcel arriving at the LCL has the
    # surface mixing ratio (q is conserved during dry-adiabatic ascent).
    # Where ``find_cloud_base`` picks the first discrete level above the
    # true LCL, the parcel is already supersaturated there, and we apply
    # a one-step ``cuadjtq``-style saturation adjustment so the cloud-
    # base parcel is warmer than the environment by the latent heat of
    # the excess condensate. This matches the LCL handling in
    # ``calculate_cape_cin`` and prevents the updraft from terminating
    # on the very first interior step due to a too-cold initial parcel.
    surf_idx = jnp.argmax(pressure)
    surf_temp = temperature[surf_idx]
    surf_humid = humidity[surf_idx]
    surf_press = pressure[surf_idx]

    parcel_T_dry_at_cb = surf_temp * (pressure[kbase] / surf_press) ** (rd / cp)
    qsat_at_cb = saturation_mixing_ratio(pressure[kbase], parcel_T_dry_at_cb)
    excess = jnp.maximum(surf_humid - qsat_at_cb, 0.0)
    tu_cb = parcel_T_dry_at_cb + (alhc / cp) * excess
    # The parcel is exactly saturated at the (warmer) cb temperature;
    # use that as the cloud-base mixing ratio rather than the raw env q.
    qu_cb = saturation_mixing_ratio(pressure[kbase], tu_cb)

    tu_init = tu_init.at[kbase].set(tu_cb)
    qu_init = qu_init.at[kbase].set(qu_cb)
    mfu_init = mfu_init.at[kbase].set(mass_flux_base)

    buoy_init = buoy_init.at[kbase].set(0.0)  # Neutral at cloud base

    updraft_init = UpdatedraftState(
        tu=tu_init, qu=qu_init, lu=lu_init,
        mfu=mfu_init, entr=entr_init, detr=detr_init,
        buoy=buoy_init,
    )
    # Carry = (updraft_state, integrated_buoyancy). The integrated
    # buoyancy drives Nordeng (1994) organized entrainment and is kept
    # *outside* UpdatedraftState so external callers see the same type.
    initial_state = (updraft_init, jnp.zeros(()))
    
    # Prepare inputs for scan (extract config parameters to avoid passing object)
    k_levels = jnp.arange(nlev)
    level_inputs = (
        k_levels, temperature, humidity, pressure, layer_thickness, rho,
        jnp.full(nlev, kbase), jnp.full(nlev, ktop),
        jnp.full(nlev, ktype),
        jnp.full(nlev, config.entrpen), jnp.full(nlev, config.entrscv),
        jnp.full(nlev, config.entrmid)
    )

    # Create specialized step function with config parameters
    def updraft_step_with_config(carry_tuple, inputs):
        carry, zbuoy_accum = carry_tuple
        k, env_temp, env_q, pressure, dz, rho, kbase, ktop, ktype, entrpen, entrscv, entrmid = inputs

        # Skip if outside cloud layer or at cloud base (boundary condition)
        in_cloud_interior = jnp.logical_and(
            jnp.minimum(ktop, kbase) < k,
            k < jnp.maximum(ktop, kbase)
        )
        at_cloud_top = (k == ktop)
        should_compute = jnp.logical_or(in_cloud_interior, at_cloud_top)
        skip = jnp.logical_not(should_compute)

        def compute_updraft():
            # Base turbulent entrainment rate by convection type
            entr_base = jnp.where(ktype == 1, entrpen,
                                  jnp.where(ktype == 2, entrscv, entrmid))

            # Humidity-dependent turbulent entrainment: drier environment
            # entrains more
            qs_env = saturation_mixing_ratio(pressure, env_temp)
            rh = jnp.clip(env_q / jnp.maximum(qs_env, 1e-10), 0.0, 1.0)
            humidity_factor = 1.0 + 2.0 * (1.0 - rh) ** 2

            entr_turb = jnp.clip(entr_base * humidity_factor, 0.0, 0.01)

            # Nordeng (1994) organized entrainment for deep convection:
            # rate ∝ local buoyancy, suppressed by the running integral of
            # buoyancy below. See ECHAM/ICON `mo_cuascent.f90` lines 511-523.
            # Use previous-level updraft buoyancy as proxy for "local zbuoyz"
            # (computed bottom-up via scan, so one step behind).
            next_level_for_buoy = jnp.minimum(k + 1, nlev - 1)
            prev_buoy = carry.buoy[next_level_for_buoy]
            # Only positive buoyancy drives organized entrainment
            zbuoyz = jnp.maximum(prev_buoy, 0.0)
            # Active only for deep convection (ktype=1)
            entr_org = jnp.where(
                ktype == 1,
                zbuoyz * 0.5 / (1.0 + zbuoy_accum),
                0.0,
            )
            entr = jnp.clip(entr_turb + entr_org, 0.0, 0.01)

            # Turbulent detrainment: fraction of entrainment
            detr_turb = 0.5 * entr

            # Organized detrainment for deep convection (Fortran tan() profile).
            # The ICON cuentr subroutine uses a tan-based profile that produces
            # sharp detrainment near cloud top, unlike a symmetric Gaussian.
            cloud_depth = jnp.maximum(kbase - ktop, 1.0)
            # Fractional distance from base (0 at base, 1 at top)
            frac_height = jnp.clip((kbase - k) / cloud_depth, 0.0, 1.0)
            # tan() profile: gentle in lower cloud, sharp increase near top
            # Argument mapped to (-pi/4, pi/2) so tan ranges from ~-1 to inf
            tan_arg = jnp.pi * (0.75 * frac_height - 0.25)
            org_profile = jnp.maximum(jnp.tan(tan_arg), 0.0)
            # Normalize: peak value of tan(pi/2 * 0.75 - pi/4) is bounded
            # Scale strength with cloud depth
            detr_strength = 0.003 * jnp.sqrt(cloud_depth / 10.0)
            detr_org = jnp.where(ktype == 1, detr_strength * org_profile, 0.0)

            detr = detr_turb + detr_org
            
            # Safe array indexing - clamp k+1 to valid range
            next_level = jnp.minimum(k + 1, nlev - 1)
            
            # Mass flux change
            dmf_entr = entr * carry.mfu[next_level] * dz
            dmf_detr = detr * carry.mfu[next_level] * dz
            
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
                # adiabatic ascent. Equivalently, a parcel rising by dz
                # cools by g·dz/cp (~9.8 K/km). The previous implementation
                # mixed T directly without this adiabatic cooling — the
                # parcel arrived at each level ~10 K too warm, so the
                # saturation adjustment never saw supersaturation, no liquid
                # formed, and no precipitation was produced.
                #
                # Detrainment removes mass at *updraft* properties, so the
                # correct denominator for mixing is the pre-detrainment mass
                # (mfu_below + dmf_entr), NOT mfu_new.
                mfu_mix = jnp.maximum(
                    carry.mfu[next_level] + dmf_entr, 1e-10
                )
                # Adiabatic cooling of the updraft air as it rises by dz
                adiabatic_cooling = grav * dz / cp
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
            
            # Calculate buoyancy
            virtual_temp_u = tu_new * (1.0 + 0.608 * qu_new - lu_new)
            virtual_temp_e = env_temp * (1.0 + 0.608 * env_q)
            buoy_new = grav * (virtual_temp_u - virtual_temp_e) / virtual_temp_e

            # Dynamic cloud-top termination: once above cloud base the parcel
            # becomes negatively buoyant (or the mass flux has already dropped
            # below 1% of the base value — ECHAM's termination criterion in
            # `mo_cuascent.f90`), terminate the updraft here. This replaces the
            # previous fixed `ktop` which ignored the environment.
            above_cloud_base = k < kbase
            mfu_too_small = carry.mfu[next_level] < 0.01 * mass_flux_base
            terminate = jnp.logical_and(
                above_cloud_base,
                jnp.logical_or(buoy_new < 0.0, mfu_too_small),
            )
            mfu_new = jnp.where(terminate, 0.0, mfu_new)

            # Update state
            new_state = carry._replace(
                tu=carry.tu.at[k].set(tu_new),
                qu=carry.qu.at[k].set(qu_new),
                lu=carry.lu.at[k].set(lu_new),
                mfu=carry.mfu.at[k].set(mfu_new),
                entr=carry.entr.at[k].set(entr),
                detr=carry.detr.at[k].set(detr),
                buoy=carry.buoy.at[k].set(buoy_new),
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