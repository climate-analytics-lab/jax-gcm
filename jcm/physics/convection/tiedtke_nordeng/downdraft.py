"""Downdraft calculations for Tiedtke-Nordeng convection scheme

This module implements the downdraft calculations including:
- Level of free sinking (LFS) determination
- Downdraft entrainment and detrainment
- Evaporative cooling
- Moist descent

Based on ICON mo_cudescent.f90

Date: 2025-01-09
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple
from functools import partial

import jcm.constants as c
from .tiedtke_nordeng import (
    ConvectionParameters, saturation_mixing_ratio
)


class DowndraftState(NamedTuple):
    """State variables for downdraft calculation"""

    td: jnp.ndarray      # Downdraft temperature (K)
    qd: jnp.ndarray      # Downdraft specific humidity (kg/kg)
    mfd: jnp.ndarray     # Downdraft mass flux (kg/m²/s) - negative values
    pdmfdp: jnp.ndarray  # Downdraft precip sink per layer (kg/m²/s, ≤ 0) —
                         # ECHAM ``pdmfdp = −pmfd·zcond``: rain evaporated
                         # into the descending parcel, debited from the rain
                         # flux and credited back to the environment through
                         # the cudtdq ledger.
    lfs: int             # Level of free sinking
    active: bool         # Whether downdraft is active


def wetbulb_temperature(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate wet-bulb temperature and humidity
    
    Simplified version - full implementation would iterate
    
    Args:
        temperature: Environmental temperature (K)
        humidity: Environmental humidity (kg/kg)
        pressure: Pressure (Pa)
        
    Returns:
        Tuple of (wetbulb_temp, wetbulb_humidity)

    """
    # Get saturation values
    qs = saturation_mixing_ratio(pressure, temperature)
    
    # If already saturated, wet-bulb equals dry-bulb
    is_saturated = humidity >= qs
    
    def calculate_wetbulb():
        # Simplified: assume wet-bulb is slightly cooler
        # Full version would iterate to find equilibrium
        cooling = (qs - humidity) * c.alhc / c.cpd
        twb = temperature - 0.3 * cooling  # Damping factor
        qwb = saturation_mixing_ratio(pressure, twb)
        return twb.astype(temperature.dtype), qwb.astype(humidity.dtype)

    def already_saturated():
        return temperature, humidity

    # Both branches must return identical dtypes for ``lax.cond``. Tie them to
    # the inputs (not a hardcoded float32) so the convection scheme is correct
    # whether the model runs in float32 or float64.
    return lax.cond(is_saturated, already_saturated, calculate_wetbulb)


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
    config: ConvectionParameters
) -> Tuple[int, bool]:
    """Find level of free sinking for downdraft initiation
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev]
        updraft_temp: Updraft temperature (K) [nlev]
        updraft_humid: Updraft humidity (kg/kg) [nlev]
        updraft_mf: Updraft mass flux (kg/m²/s) [nlev]
        precip_rate: Precipitation rate (kg/m²/s)
        kbase: Cloud base level
        ktop: Cloud top level
        config: Convection configuration
        
    Returns:
        Tuple of (lfs_level, found_lfs)

    """
    nlev = len(temperature)
    
    # Scan from cloud top down to find LFS
    def check_lfs(k):
        # Cloud-bound checks (k within [ktop, kbase]) are the calling
        # function's responsibility.
        # Calculate wet-bulb values for environment
        twb, qwb = wetbulb_temperature(temperature[k], humidity[k], pressure[k])
        
        # Mix 50% cloud air with 50% environmental air at wet-bulb
        t_mix = 0.5 * (updraft_temp[k] + twb)
        q_mix = 0.5 * (updraft_humid[k] + qwb)
        
        # Calculate buoyancy
        vt_mix = t_mix * (1.0 + 0.608 * q_mix)
        vt_env = temperature[k] * (1.0 + 0.608 * humidity[k])
        buoyancy = (vt_mix - vt_env) / vt_env
        
        # Condensation in downdraft
        condensation = humidity[k] - qwb
        
        # Minimum mass flux threshold (Fortran: zmftop = -cmfdeps*pmfub)
        min_flux = config.cmfdeps * updraft_mf[kbase]
        
        # Check LFS criteria:
        # 1. Negative buoyancy
        # 2. Sufficient precipitation to maintain downdraft
        is_lfs = jnp.logical_and(
            buoyancy < 0.0,
            precip_rate > 10.0 * min_flux * condensation
        )
        
        return is_lfs, buoyancy
    
    # Find first level that satisfies LFS criteria using JAX-compatible operations
    # Check all possible levels and find the first one that satisfies LFS
    nlev = len(temperature)
    
    # Create a function to check LFS at each level
    def check_all_levels(k):
        # Only check if k is in valid range
        in_range = (k >= ktop) & (k <= kbase)
        is_lfs, buoy = lax.cond(
            in_range,
            lambda: check_lfs(k),
            lambda: (False, 0.0)
        )
        return is_lfs
    
    # Check all levels from top to base
    all_levels = jnp.arange(nlev)
    lfs_conditions = jax.vmap(check_all_levels)(all_levels)
    
    # Find first level where LFS is satisfied
    lfs_found = jnp.any(lfs_conditions)
    
    # Get the first level index where condition is met
    first_lfs_idx = jnp.argmax(lfs_conditions)  # argmax returns first True
    lfs_level = jnp.where(lfs_found, first_lfs_idx, ktop)
    
    return lfs_level, lfs_found


def downdraft_entrainment_ledger(
    mfd: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    entrdd: float,
) -> jnp.ndarray:
    """Absolute per-layer downdraft entrainment flux [kg/m²/s] (#622).

    ECHAM ``cuddraf`` entrains ``zentr = entrdd·|mfd(k-1)|·dz`` into each
    descent layer (matched by an equal detrainment in the bulk, so
    |mfd| is conserved going down); entrainment is shut off in the two
    surface-taper layers. ``mfd[k]`` is the flux leaving layer k through
    its BOTTOM interface, so the flux entering from above is
    ``mfd[k-1]`` — zero at and above the LFS, which zeroes the ledger
    there without an explicit LFS index. Where the downdraft died
    mid-descent (``mfd[k] == 0`` with inflow above), the ledger is also
    zero so plume continuity dumps the arriving flux as pure
    detrainment, matching the Fortran's buoyancy shut-off.

    Vertical on axis 0; trailing axes broadcast (a ``(nlev,)`` column and
    a ``(nlev, ncols)`` block agree per column).
    """
    nlev = mfd.shape[0]
    mfd_in = jnp.concatenate(
        [jnp.zeros_like(mfd[:1]), mfd[:-1]], axis=0
    )
    levels = jnp.arange(nlev).reshape((nlev,) + (1,) * (mfd.ndim - 1))
    in_bulk = (mfd < 0.0) & (levels < nlev - 2)
    return jnp.where(
        in_bulk, entrdd * jnp.abs(mfd_in) * layer_thickness, 0.0
    )


def downdraft_step(
    carry_and_rain: Tuple,
    level_inputs: Tuple
) -> Tuple[Tuple, DowndraftState]:
    """Single step of downdraft calculation for use with lax.scan

    Mirrors ECHAM ``mo_cudescent.f90::cuddraf``: in the bulk of the
    downdraft column the fractional entrainment ``entrdd`` is matched
    by an equal detrainment, so the downdraft mass flux is conserved
    going down. In the lowest two layers, entrainment is shut off and
    detrainment is set to a linear ramp that drives the mass flux to
    zero at the surface. Without these, the prior implementation only
    entrained (no matching detrainment), so |mfd| ran away by ~50x as
    the downdraft descended a deep RCE column.

    Args:
        carry: Current downdraft state
        level_inputs: Environment variables at current level

    Returns:
        Tuple of (updated_carry, output_state)

    """
    carry, rain_flux = carry_and_rain
    (k, env_temp, env_q, pressure, dz, rho, precip,
     entrdd, cmfcmin, cevapcu, klev_m2, p_taper_frac) = level_inputs

    # Surface-first index convention: k=0 = TOA, k=nlev-1 = surface.
    # Downdraft is active from carry.lfs (somewhere in cloud, lower index)
    # downward. Skip levels at or above the LFS — those are handled by
    # the LFS-init step before the scan.
    skip = jnp.logical_or(~carry.active, k <= carry.lfs)

    def compute_downdraft():
        # State from immediately above (towards LFS).
        prev_mfd = carry.mfd[k - 1]
        prev_td = carry.td[k - 1]
        prev_qd = carry.qd[k - 1]

        # 1) Dry-adiabatic descent: a parcel falling by dz warms by g·dz/cp
        # (~1.95 K per 200 m). ECHAM ``cuddraf`` builds this into the DSE
        # update implicitly through the (pgeoh(k-1)-pgeoh(k))/cp term;
        # we apply it explicitly so the temperature mixing step is just
        # a linear interpolation toward the environment.
        adiabatic_warming = c.grav * dz / c.cpd
        td_desc = prev_td + adiabatic_warming
        qd_desc = prev_qd

        # 2) Entrainment / detrainment magnitude (mass flux per layer,
        # kg/m²/s). ECHAM cuddraf: zentr = entrdd*|mfd(k-1)|*Rd*T/p*pmref;
        # with pmref/rho≈dz this reduces to entrdd*|mfd|*dz.
        zentr = entrdd * jnp.abs(prev_mfd) * dz

        # 3) Surface taper. Fortran ``itopde=klev-2``: in the lowest two
        # layers, shut off entrainment and apply a linear detrainment
        # ramp so the mass flux reaches zero at the surface. In the bulk
        # of the column, entrainment is matched by detrainment so mfd is
        # conserved going down.
        in_surface_taper = k > klev_m2
        # In bulk, zdmfen and zdmfde cancel; mass flux is unchanged.
        # In taper, zdmfde ramps |mfd| down to zero in the lowest layer.
        extra_detr = jnp.where(
            in_surface_taper,
            jnp.abs(prev_mfd) * p_taper_frac,
            0.0,
        )
        # ``prev_mfd`` is negative; detrainment (extra_detr ≥ 0) makes
        # it less negative.
        mfd_new = prev_mfd + extra_detr

        # 4) Mixing: in the bulk, fraction ``zentr/|mfd|`` of environment
        # air is mixed in (matched by the same fraction detrained out
        # of the downdraft). In the surface taper there is no entrainment
        # so no mixing — the parcel just retains its previous-level
        # properties, modified only by adiabatic warming.
        mix_fraction = jnp.where(
            in_surface_taper,
            0.0,
            zentr / jnp.maximum(jnp.abs(prev_mfd), cmfcmin),
        )
        td_mix = (1.0 - mix_fraction) * td_desc + mix_fraction * env_temp
        qd_mix = (1.0 - mix_fraction) * qd_desc + mix_fraction * env_q

        # 5) Saturate the descending parcel by evaporating rain into it —
        # ECHAM cuddraf lines 286-316: cuadjtq(kcall=2) drives (T,q) to
        # saturation (evaporation-only Newton step), and the vapour taken
        # up is debited from the rain flux as ``pdmfdp = −pmfd·zcond`` (a
        # NEGATIVE per-layer precip increment: the environment gets the
        # cooling + moistening back through the cudtdq ledger). The
        # previous cevapcu-scaled pseudo-evaporation warmed the parcel
        # nearly dry-adiabatically, so it lost negative buoyancy within a
        # level or two and the downdraft died at its LFS value; cevapcu
        # itself belongs to the sub-cloud Kessler chain in cuflx, not here.
        from .adjustment import cuadjtq
        td_clipped = jnp.clip(td_mix, 100.0, 400.0)
        td_new, qd_new, zcond = cuadjtq(
            td_clipped, qd_mix, pressure, kcall=2,
        )
        # zcond ≤ 0 is the vapour ADDED to the parcel (q_before − q_after).
        # Rain consumed per layer: zdmfdp = −mfd·zcond ≤ 0 (mfd < 0).
        zdmfdp = -mfd_new * zcond

        # 6) Termination (cuddraf line 310): keep the downdraft only while
        # negatively buoyant AND the rain it would consume is available
        # (rain_flux − mfd·zcond > 0).
        vt_down = td_new * (1.0 + 0.608 * qd_new)
        vt_env = env_temp * (1.0 + 0.608 * env_q)
        rain_available = rain_flux + zdmfdp > 0.0
        keep = jnp.logical_and(vt_down < vt_env, rain_available)
        mfd_final = jnp.where(keep, mfd_new, 0.0)
        zdmfdp = jnp.where(keep, zdmfdp, 0.0)
        rain_flux_new = rain_flux + zdmfdp

        new_state = carry._replace(
            td=carry.td.at[k].set(td_new),
            qd=carry.qd.at[k].set(qd_new),
            mfd=carry.mfd.at[k].set(mfd_final),
            pdmfdp=carry.pdmfdp.at[k].set(zdmfdp),
            active=jnp.abs(mfd_final) > cmfcmin,
        )
        return new_state, rain_flux_new

    def keep_state():
        return carry, rain_flux

    (updated_state, rain_flux_out) = lax.cond(
        skip, keep_state, compute_downdraft,
    )
    return (updated_state, rain_flux_out), updated_state


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
    config: ConvectionParameters
) -> DowndraftState:
    """Calculate full downdraft profile
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        rho: Air density (kg/m³) [nlev]
        updraft_state: Computed updraft state
        precip_rate: Column precipitation rate (kg/m²/s)
        kbase: Cloud base level
        ktop: Cloud top level
        config: Convection configuration
        
    Returns:
        DowndraftState with computed profiles

    """
    nlev = len(temperature)
    
    # Find level of free sinking
    lfs, has_lfs = find_lfs(
        temperature, humidity, pressure,
        updraft_state.tu, updraft_state.qu, updraft_state.mfu,
        precip_rate, kbase, ktop, config
    )
    
    # Initialize downdraft state
    td_init = temperature.copy()
    qd_init = humidity.copy()
    mfd_init = jnp.zeros(nlev)
    
    # Initialize downdraft conditionally using JAX-compatible operations
    def initialize_downdraft():
        # Mix cloud and environmental air at LFS
        twb, qwb = wetbulb_temperature(
            temperature[lfs], humidity[lfs], pressure[lfs]
        )
        td_new = td_init.at[lfs].set(0.5 * (updraft_state.tu[lfs] + twb))
        qd_new = qd_init.at[lfs].set(0.5 * (updraft_state.qu[lfs] + qwb))

        # Initial downdraft mass flux: ECHAM cudlfs uses
        #   zmftop = -cmfdeps * pmfub
        # where pmfub = mfu(kcbot) is the cloud-base mass flux. The
        # previous code used ``cmfctop`` (a different parameter that
        # controls cloud-top mass flux fraction in the updraft) which is
        # numerically similar (~0.2-0.3) but conceptually wrong.
        mfd_new = mfd_init.at[lfs].set(
            -config.cmfdeps * updraft_state.mfu[kbase]
        )
        return td_new, qd_new, mfd_new

    def no_downdraft():
        return td_init, qd_init, mfd_init

    # Apply Pattern 2: Conditional Computation
    td_final, qd_final, mfd_final = lax.cond(
        has_lfs,
        initialize_downdraft,
        no_downdraft
    )

    initial_state = DowndraftState(
        td=td_final,
        qd=qd_final,
        mfd=mfd_final,
        pdmfdp=jnp.zeros(nlev),
        lfs=lfs,
        active=has_lfs
    )

    # Surface-taper geometry (mirrors ``itopde = klev-2`` in cuddraf):
    # in the bottom two layers, entrainment is shut off and detrainment
    # is split linearly across the layers so the mass flux reaches zero
    # at the surface. ``p_taper_frac`` is the fraction of the residual
    # mass-flux to detrain in each surface-taper layer; the simplest
    # uniform split is 0.5 in the second-to-last layer and 1.0 in the
    # last (which fully zeroes mfd at the surface).
    klev_m2 = jnp.array(nlev - 3, dtype=jnp.int32)  # Fortran ``itopde``-equivalent (0-indexed: nlev-3 = top of taper)
    p_taper_frac = jnp.zeros(nlev)
    p_taper_frac = p_taper_frac.at[nlev - 2].set(0.5)
    p_taper_frac = p_taper_frac.at[nlev - 1].set(1.0)

    # Prepare inputs for scan (extract config parameters to avoid passing object)
    k_levels = jnp.arange(nlev)
    level_inputs = (
        k_levels, temperature, humidity, pressure,
        layer_thickness, rho, jnp.full(nlev, precip_rate),
        jnp.full(nlev, config.entrdd),
        jnp.full(nlev, config.cmfcmin),
        jnp.full(nlev, config.cevapcu),
        jnp.full(nlev, klev_m2),
        p_taper_frac,
    )
    
    # Use scan to compute downdraft from LFS downward. The carry threads
    # the remaining rain flux so per-layer cuadjtq evaporation depletes it
    # (cuddraf's ``prfl`` accumulation).
    (final_state, _rain_left), all_states = lax.scan(
        partial(downdraft_step),
        (initial_state, precip_rate),
        level_inputs
    )

    return final_state