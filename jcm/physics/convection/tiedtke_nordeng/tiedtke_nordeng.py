"""Tiedtke-Nordeng Mass-Flux Convection Scheme

This module implements the Tiedtke-Nordeng convection parameterization
in JAX, based on the ICON atmospheric model implementation.

The scheme includes:
- Deep convection with CAPE closure
- Shallow convection with moisture convergence closure  
- Mid-level convection
- Convective momentum transport
- Downdraft processes

References:
- Tiedtke, M. (1989): A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models. Mon. Weather Rev., 117, 1779-1800.
- Nordeng, T. E. (1994): Extended versions of the convective parametrization
  scheme at ECMWF and their impact on the mean and transient activity of the
  model in the tropics. ECMWF Tech. Memo. 206.

Date: 2025-01-09

"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple
import tree_math

from jcm.physics.icon.constants.physical_constants import (
    grav, rd, rv, cp, eps, tmelt, alhc
)

# Import updraft, downdraft and flux modules after they're defined
# This avoids circular imports


@tree_math.struct
class ConvectionParameters:
    """Configuration parameters for Tiedtke-Nordeng convection scheme"""
    
    # Time stepping
    dt_conv: float           # Convection timestep (s)
    
    # Entrainment/detrainment parameters
    entrpen: float           # Entrainment rate for penetrative convection (m⁻¹)
    entrscv: float           # Entrainment rate for shallow convection (m⁻¹) 
    entrmid: float           # Entrainment rate for mid-level convection (m⁻¹)
    
    # CAPE closure
    tau: float               # CAPE adjustment timescale (s)
    
    # Cloud base mass flux
    cmfcmax: float           # Maximum cloud base mass flux (kg/m²/s)
    cmfcmin: float           # Minimum cloud base mass flux (kg/m²/s)
    
    # Precipitation parameters
    cprcon: float            # Coefficient for precipitation conversion
    
    # Evaporation parameters
    cevapcu: float           # Coefficient for rain evaporation
    
    # Numerical parameters
    epsilon: float           # Small number for numerical stability
    
    # Convection type thresholds
    rlcrit: float            # Critical relative humidity for shallow convection
    rhcrit: float            # Critical relative humidity threshold
    
    # Momentum transport
    cmfctop: float           # Mass flux fraction at cloud top

    # Downdraft parameters
    cmfdeps: float           # Downdraft mass flux fraction for LFS threshold

    @classmethod
    def default(cls, dt_conv=3600.0, entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4, # FIXME: validate dt_conv
                 tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10, cprcon=1.4e-3,
                 cevapcu=2.0e-5, epsilon=1.0e-12, rlcrit=8.0e-4, rhcrit=0.9,
                 cmfctop=0.33, cmfdeps=0.33) -> 'ConvectionParameters':
        """Return default convection parameters"""
        return cls(
            dt_conv=jnp.array(dt_conv),
            entrpen=jnp.array(entrpen),
            entrscv=jnp.array(entrscv),
            entrmid=jnp.array(entrmid),
            tau=jnp.array(tau),
            cmfcmax=jnp.array(cmfcmax),
            cmfcmin=jnp.array(cmfcmin),
            cprcon=jnp.array(cprcon),
            cevapcu=jnp.array(cevapcu),
            epsilon=jnp.array(epsilon),
            rlcrit=jnp.array(rlcrit),
            rhcrit=jnp.array(rhcrit),
            cmfctop=jnp.array(cmfctop),
            cmfdeps=jnp.array(cmfdeps)
        )


class ConvectionState(NamedTuple):
    """State variables for convection scheme"""
    
    # Updraft properties
    tu: jnp.ndarray          # Updraft temperature (K)
    qu: jnp.ndarray          # Updraft specific humidity (kg/kg)  
    lu: jnp.ndarray          # Updraft liquid water content (kg/kg)
    uu: jnp.ndarray          # Updraft zonal wind (m/s)
    vu: jnp.ndarray          # Updraft meridional wind (m/s)
    
    # Downdraft properties  
    td: jnp.ndarray          # Downdraft temperature (K)
    qd: jnp.ndarray          # Downdraft specific humidity (kg/kg)
    ud: jnp.ndarray          # Downdraft zonal wind (m/s)
    vd: jnp.ndarray          # Downdraft meridional wind (m/s)
    
    # Mass fluxes
    mfu: jnp.ndarray         # Updraft mass flux (kg/m²/s)
    mfd: jnp.ndarray         # Downdraft mass flux (kg/m²/s)
    
    # Convection diagnostics
    ktype: jnp.ndarray       # Convection type (0=none, 1=deep, 2=shallow, 3=mid)
    kbase: jnp.ndarray       # Cloud base level index
    ktop: jnp.ndarray        # Cloud top level index
    
    # Precipitation
    prate: jnp.ndarray       # Precipitation rate (kg/m²/s)


class ConvectionTendencies(NamedTuple):
    """Tendencies from convection scheme"""
    
    dtedt: jnp.ndarray       # Temperature tendency (K/s)
    dqdt: jnp.ndarray        # Specific humidity tendency (kg/kg/s)
    dudt: jnp.ndarray        # Zonal wind tendency (m/s²)
    dvdt: jnp.ndarray        # Meridional wind tendency (m/s²)
    
    # Convective fluxes
    qc_conv: jnp.ndarray     # Convective cloud water (kg/kg)
    qi_conv: jnp.ndarray     # Convective cloud ice (kg/kg)
    
    # Surface fluxes
    precip_conv: jnp.ndarray # Convective precipitation (kg/m²/s)
    
    # Fixed tracer tendencies (qc, qi only)
    dqc_dt: jnp.ndarray      # Cloud water tendency (kg/kg/s)
    dqi_dt: jnp.ndarray      # Cloud ice tendency (kg/kg/s)


def saturation_vapor_pressure(temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate saturation vapor pressure using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)

    """
    # Tetens formula coefficients
    a = 17.27
    b = 35.86

    # Wide math-safety clip — Tetens denominators t+237.3 and t+265.5 hit
    # zero at T≈36K and T≈8K. Use a loose bound that only catches truly
    # pathological values and doesn't mask upstream physics bugs.
    temperature = jnp.clip(temperature, 50.0, 500.0)
    t_celsius = temperature - tmelt

    # Over water (T > 0°C) — denominator always > 150+237.3-273.15 > 114 when T clipped
    es_water = 610.78 * jnp.exp(a * t_celsius / (t_celsius + 237.3))

    # Over ice (T <= 0°C)
    es_ice = 610.78 * jnp.exp(b * t_celsius / (t_celsius + 265.5))

    # Use water or ice formula depending on temperature
    es = jnp.where(temperature > tmelt, es_water, es_ice)

    return es


def saturation_mixing_ratio(pressure: jnp.ndarray, 
                          temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate saturation mixing ratio
    
    Args:
        pressure: Pressure (Pa)
        temperature: Temperature (K)
        
    Returns:
        Saturation mixing ratio (kg/kg)

    """
    es = saturation_vapor_pressure(temperature)
    # Cap es < 0.99*pressure so denominator can't approach zero at low P / high T
    es_safe = jnp.minimum(es, 0.99 * jnp.maximum(pressure, 1.0))
    qs = eps * es_safe / jnp.maximum(pressure - es_safe * (1.0 - eps), 1.0)
    return jnp.clip(qs, 0.0, 0.5)


def moist_static_energy(temperature: jnp.ndarray,
                       height: jnp.ndarray, 
                       mixing_ratio: jnp.ndarray) -> jnp.ndarray:
    """Calculate moist static energy
    
    Args:
        temperature: Temperature (K)
        height: Geopotential height (m)  
        mixing_ratio: Water vapor mixing ratio (kg/kg)
        
    Returns:
        Moist static energy (J/kg)

    """
    return cp * temperature + grav * height + alhc * mixing_ratio


def initialize_convection(temperature: jnp.ndarray,
                         humidity: jnp.ndarray,
                         pressure: jnp.ndarray,
                         u_wind: jnp.ndarray,
                         v_wind: jnp.ndarray,
                         config: ConvectionParameters) -> ConvectionState:
    """Initialize convection state variables
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        config: Convection configuration
        
    Returns:
        Initial convection state

    """
    nlev = temperature.shape[0]
    
    # Initialize updraft properties with environmental values (ensure float32)
    tu = jnp.array(temperature, dtype=jnp.float32)
    qu = jnp.array(humidity, dtype=jnp.float32)
    lu = jnp.zeros_like(temperature, dtype=jnp.float32)
    uu = jnp.array(u_wind, dtype=jnp.float32)
    vu = jnp.array(v_wind, dtype=jnp.float32)
    
    # Initialize downdraft properties (ensure float32)
    td = jnp.array(temperature, dtype=jnp.float32)
    qd = jnp.array(humidity, dtype=jnp.float32)
    ud = jnp.array(u_wind, dtype=jnp.float32)
    vd = jnp.array(v_wind, dtype=jnp.float32)
    
    # Initialize mass fluxes to zero with explicit dtype
    mfu = jnp.zeros_like(temperature, dtype=jnp.float32)
    mfd = jnp.zeros_like(temperature, dtype=jnp.float32)
    
    # Initialize convection diagnostics
    ktype = jnp.array(0)  # No convection initially
    kbase = jnp.array(nlev - 1)  # Surface level
    ktop = jnp.array(0)   # Top level
    
    # Initialize precipitation
    prate = jnp.array(0.0)
    
    return ConvectionState(
        tu=tu, qu=qu, lu=lu, uu=uu, vu=vu,
        td=td, qd=qd, ud=ud, vd=vd,
        mfu=mfu, mfd=mfd,
        ktype=ktype, kbase=kbase, ktop=ktop,
        prate=prate
    )


def find_cloud_base(temperature: jnp.ndarray,
                   humidity: jnp.ndarray, 
                   pressure: jnp.ndarray,
                   config: ConvectionParameters) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find lifting condensation level (cloud base)
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        config: Convection configuration
        
    Returns:
        Tuple of (cloud_base_level, cloud_base_exists)

    """
    nlev = len(temperature)
    
    # Start from surface (bottom level - highest pressure)
    surf_idx = jnp.argmax(pressure)  # Surface is at highest pressure
    surf_temp = temperature[surf_idx]
    surf_humid = humidity[surf_idx]
    surf_press = pressure[surf_idx]
    
    # Calculate parcel temperature at all levels (dry adiabatic)
    exner_ratios = (pressure / surf_press) ** (rd / cp)
    parcel_temps = surf_temp * exner_ratios
    
    # Calculate saturation mixing ratio at parcel temperatures
    parcel_qs = jax.vmap(saturation_mixing_ratio)(pressure, parcel_temps)
    
    # Check where parcel becomes saturated
    is_saturated = surf_humid >= parcel_qs
    
    # Find first level (from bottom up) where saturation occurs
    # Start from surface and go up
    levels = jnp.arange(nlev)
    
    # Mask for levels where saturation occurs
    # Only consider levels above surface but below very high levels
    valid_levels = jnp.logical_and(levels < nlev - 1, levels > 0)
    saturated_and_valid = jnp.logical_and(is_saturated, valid_levels)

    # Find nearest-to-surface saturated level: the one with the highest pressure
    # This works regardless of index ordering (TOA-first or surface-first)
    saturated_pressure = jnp.where(saturated_and_valid, pressure, -1.0)
    cloud_base_level = jnp.argmax(saturated_pressure)
    cloud_base_found = saturated_pressure[cloud_base_level] > 0.0
    
    # If no cloud base found, set to surface
    cloud_base_level = jnp.where(cloud_base_found, cloud_base_level, nlev - 1)
    
    return cloud_base_level, cloud_base_found


def calculate_cape_cin(temperature: jnp.ndarray,
                      humidity: jnp.ndarray,
                      pressure: jnp.ndarray,
                      layer_thickness: jnp.ndarray,
                      cloud_base: int,
                      config: ConvectionParameters) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate CAPE and CIN for convective instability.

    Lifts a surface parcel: dry-adiabatic up to cloud base, moist-
    adiabatic above. CAPE is the positive-buoyancy work done between the
    LFC (level of free convection) and the EL (equilibrium level); CIN
    is the magnitude of negative-buoyancy work between cloud base and
    LFC. Stratospheric layers above the EL — where the parcel cools to
    absurdly low temperatures and gives massive bogus negative buoyancy
    — are not counted.

    Works for either input ordering:
      * TOA-first (level 0 = TOA, ICON/ECHAM convention used by the
        running model)
      * surface-first (level 0 = surface, used by some unit tests)
    Internally we always work in surface-first form so the moist-adiabat
    scan steps in the natural surface→TOA direction. The output is in
    the input's ordering for backward compatibility.

    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        cloud_base: Cloud base level index in the input ordering
        config: Convection configuration

    Returns:
        Tuple of (CAPE, CIN) in J/kg

    """
    nlev = len(temperature)

    # Reorder to surface-first: level 0 = surface, level nlev-1 = TOA.
    # ``flip`` is a no-op when the input is already surface-first.
    is_surface_first = pressure[0] >= pressure[-1]
    flip = lambda a: jnp.where(is_surface_first, a, a[::-1])
    T_sf = flip(temperature)
    q_sf = flip(humidity)
    p_sf = flip(pressure)
    dz_sf = flip(layer_thickness)
    cb_sf = jnp.where(is_surface_first, cloud_base, nlev - 1 - cloud_base)

    surf_temp = T_sf[0]
    surf_humid = q_sf[0]
    surf_press = p_sf[0]
    k_levels = jnp.arange(nlev)

    # Below cloud base — dry-adiabatic ascent. q is conserved, so the
    # parcel mixing ratio stays at the surface value.
    parcel_temp_dry = surf_temp * (p_sf / surf_press) ** (rd / cp)

    # Above cloud base — moist (pseudoadiabatic) ascent. We scan
    # surface→TOA (increasing index in surface-first) and only step the
    # parcel temperature when we are AT or above cloud base; below cb
    # the parcel just rides the dry adiabat we already computed.
    #
    # If the parcel arrives at cb already supersaturated (surf_q >
    # qsat(parcel_dry_T, p_cb) — common when find_cloud_base picks the
    # next discrete level above the true LCL), do a one-step saturation
    # adjustment: condense the excess water and warm the parcel by L/cp
    # times the condensate. This raises the cloud-base parcel
    # temperature to its physically meaningful value and prevents
    # spurious cold biases that crush CAPE for warm tropical columns.
    parcel_temp_at_cb_dry = parcel_temp_dry[cb_sf]
    p_cb = p_sf[cb_sf]
    qsat_at_cb = saturation_mixing_ratio(p_cb, parcel_temp_at_cb_dry)
    excess = jnp.maximum(surf_humid - qsat_at_cb, 0.0)
    cloud_base_temp = parcel_temp_at_cb_dry + (alhc / cp) * excess

    def _step(parcel_t, args):
        p_curr, p_next, k = args
        dp = p_next - p_curr  # negative going up
        qs = saturation_mixing_ratio(p_curr, parcel_t)
        dTdp = (1.0 / p_curr) * (rd * parcel_t + alhc * qs) / (
            cp + alhc ** 2 * qs / (rv * parcel_t ** 2)
        )
        new_t = parcel_t + dTdp * dp
        # If we haven't reached cloud base yet, hold the parcel at the
        # cloud-base temperature so the moist integration starts from
        # the right pressure when k finally crosses cb.
        below_cb = k < cb_sf
        new_t = jnp.where(below_cb, cloud_base_temp, new_t)
        return new_t, new_t

    p_pairs = (p_sf[:-1], p_sf[1:], k_levels[:-1])
    _, parcel_after = lax.scan(_step, cloud_base_temp, p_pairs)
    parcel_temp_moist_sf = jnp.concatenate(
        [cloud_base_temp[jnp.newaxis], parcel_after],
    )

    is_above_cb = k_levels >= cb_sf
    parcel_temp_sf = jnp.where(
        is_above_cb, parcel_temp_moist_sf, parcel_temp_dry,
    )
    parcel_qs_sf = jax.vmap(saturation_mixing_ratio)(p_sf, parcel_temp_sf)
    parcel_q_sf = jnp.where(is_above_cb, parcel_qs_sf, surf_humid)

    env_tv_sf = T_sf * (1.0 + 0.61 * q_sf)
    parcel_tv_sf = parcel_temp_sf * (1.0 + 0.61 * parcel_q_sf)
    buoyancy_sf = grav * (parcel_tv_sf - env_tv_sf) / env_tv_sf

    # LFC: lowest-altitude (smallest surface-first index) at-or-above cb
    #      where buoyancy first becomes positive.
    pos_above_cb = (buoyancy_sf > 0) & is_above_cb
    has_lfc = jnp.any(pos_above_cb)
    # ``argmax`` returns the FIRST True; surface-first means lowest-
    # altitude True is the smallest index, which is what we want.
    lfc_sf = jnp.argmax(pos_above_cb)
    lfc_sf = jnp.where(has_lfc, lfc_sf, nlev)

    # EL: first level above LFC (larger index in surface-first) where
    # buoyancy turns non-positive again. Without an EL we integrate to
    # TOA (k = nlev - 1).
    above_lfc_mask = k_levels > lfc_sf
    el_candidate_mask = above_lfc_mask & ~(buoyancy_sf > 0)
    has_el = jnp.any(el_candidate_mask)
    el_sf = jnp.argmax(el_candidate_mask)
    el_sf = jnp.where(has_el, el_sf, nlev)

    in_cape_layer = (k_levels >= lfc_sf) & (k_levels < el_sf)
    in_cin_layer = is_above_cb & (k_levels < lfc_sf)

    cape_contrib = jnp.where(
        in_cape_layer & (buoyancy_sf > 0), buoyancy_sf * dz_sf, 0.0,
    )
    cin_contrib = jnp.where(
        in_cin_layer & (buoyancy_sf <= 0), -buoyancy_sf * dz_sf, 0.0,
    )

    cape = jnp.where(has_lfc, jnp.sum(cape_contrib), 0.0)
    cin = jnp.where(has_lfc, jnp.sum(cin_contrib), 0.0)

    return cape, cin


def tiedtke_nordeng_convection(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray, 
    pressure: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    rho: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    dt: float,
    config: ConvectionParameters = None
) -> Tuple[ConvectionTendencies, ConvectionState]:
    """Run Tiedtke-Nordeng convection scheme with fixed qc/qi transport
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        rho: Air density (kg/m³) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        qc: Cloud water mixing ratio (kg/kg) [nlev]
        qi: Cloud ice mixing ratio (kg/kg) [nlev]
        dt: Time step (s)
        config: Convection configuration
        
    Returns:
        Tuple of (tendencies, final_state) with fixed qc/qi transport

    """
    if config is None:
        config = ConvectionParameters.default()
    
    nlev = len(temperature)
    
    # Initialize state
    state = initialize_convection(
        temperature, humidity, pressure, 
        u_wind, v_wind, config
    )
    
    # Find cloud base
    cloud_base, has_cloud_base = find_cloud_base(
        temperature, humidity, pressure, config
    )
    
    # Calculate CAPE and CIN if cloud base exists
    cape, cin = lax.cond(
        has_cloud_base,
        lambda: calculate_cape_cin(temperature, humidity, pressure, layer_thickness, 
                                 cloud_base, config),
        lambda: (jnp.array(0.0), jnp.array(0.0))
    )
    
    # Determine convection type based on CAPE and other criteria
    # 0 = no convection, 1 = deep, 2 = shallow, 3 = mid-level
    # Use more reasonable CAPE thresholds for triggering
    conv_type = lax.cond(
        jnp.logical_and(has_cloud_base, cape > 100.0),  # Minimum CAPE threshold
        lambda: lax.cond(cape > 1000.0, lambda: 1, lambda: 2),  # Deep vs shallow
        lambda: 0  # No convection
    )
    
    # Initialize tendencies to zero with explicit float32 dtype
    dtedt = jnp.zeros_like(temperature, dtype=jnp.float32)
    dqdt = jnp.zeros_like(humidity, dtype=jnp.float32)
    dudt = jnp.zeros_like(u_wind, dtype=jnp.float32)
    dvdt = jnp.zeros_like(v_wind, dtype=jnp.float32)
    qc_conv = jnp.zeros_like(temperature, dtype=jnp.float32)
    qi_conv = jnp.zeros_like(temperature, dtype=jnp.float32)
    precip_conv = jnp.array(0.0, dtype=jnp.float32)
    
    # Import modules here to avoid circular imports
    from .updraft import calculate_updraft
    from .downdraft import calculate_downdraft
    from .flux_tendencies import (
        calculate_tendencies, mass_flux_closure
    )
    from .adjustment import convective_adjustment
    
    # Apply full convection scheme if active (with tracer transport)
    def apply_full_convection():
        # Cloud-top scan ceiling. Deep convection in the tropics commonly
        # reaches the tropopause (~12-15 km, ~15-20 levels above cloud
        # base on the ICON 47-level grid); shallow convection peaks
        # around 2-3 km. Set a generous scan range and let the updraft's
        # dynamic termination (negative buoyancy or mfu < 1% of base —
        # see ``calculate_updraft``) decide where the cloud actually
        # ends. The previous values (6 for deep, 3 for shallow) capped
        # deep convection at ~3 km so it could never properly transport
        # heat / moisture through the troposphere.
        cloud_depth = lax.cond(conv_type == 2, lambda: 5, lambda: 15)

        # Handle level ordering properly
        pressure_increasing = pressure[0] < pressure[-1]

        # Ensure cloud depth is at least 2 levels and doesn't extend to TOA
        # Cloud base must be at least 2 levels from the top to allow for updraft development
        min_top_level = 2  # Don't allow clouds to extend above this level

        ktop = lax.cond(
            pressure_increasing,
            lambda: jnp.maximum(cloud_base - cloud_depth, min_top_level),      # Standard: top = lower index, but not TOA
            lambda: jnp.minimum(cloud_base + cloud_depth, nlev-1-min_top_level)  # Reverse: top = higher index
        )
        
        # Calculate mass flux using appropriate closure
        moisture_conv = jnp.array(0.0)  # Would calculate from large-scale fields
        mass_flux_base = mass_flux_closure(
            cape, cin, moisture_conv, conv_type, config
        )
        
        # Calculate updraft
        updraft_state = calculate_updraft(
            temperature, humidity, pressure, layer_thickness, rho,
            cloud_base, ktop, conv_type, mass_flux_base, config
        )
        
        # Calculate precipitation from updraft
        precip_rate = jnp.sum(updraft_state.lu * updraft_state.mfu) * config.cprcon
        
        # Calculate downdraft (now properly implemented)
        downdraft_state = calculate_downdraft(
            temperature, humidity, pressure, layer_thickness, rho,
            updraft_state, precip_rate, cloud_base, ktop, config
        )
        
        # Calculate final tendencies for basic variables
        tendencies = calculate_tendencies(
            temperature, humidity, u_wind, v_wind, pressure, rho, layer_thickness,
            updraft_state, downdraft_state,
            cloud_base, ktop, dt, config
        )
        
        # Calculate fixed qc/qi transport
        mass_flux_profile = updraft_state.mfu - downdraft_state.mfd
        
        def calculate_tracer_tendency(tracer_profile):
            # Simple finite difference for transport
            tracer_flux = mass_flux_profile * tracer_profile * 0.1  # Mixing efficiency
            # Tendency from flux divergence (simplified)
            return jnp.diff(tracer_flux, append=0.0) * 0.001  # Scale factor
        
        # Calculate fixed qc/qi tendencies
        dqc_dt = calculate_tracer_tendency(qc)
        dqi_dt = calculate_tracer_tendency(qi)
        
        # Enhanced cloud water/ice production from condensation
        qc_conv = jnp.where(updraft_state.mfu > 0, updraft_state.lu * 0.1, 0.0)
        qi_conv = jnp.where(
            (updraft_state.mfu > 0) & (temperature < tmelt),
            updraft_state.lu * 0.05, 0.0
        )
        
        # Final saturation adjustment: apply the convective tendencies,
        # remove any residual supersaturation via iterative saturation
        # adjustment (matches the post-convection `cuadjtq` call in the
        # ECHAM reference), then re-derive the tendencies that produce
        # the adjusted state. Previously this step was missing — the
        # `convective_adjustment` helper existed but was never called,
        # leaving the post-convection state supersaturated.
        t_adj, q_adj, qc_adj, qi_adj = convective_adjustment(
            temperature, humidity, pressure, qc, qi,
            tendencies.dtedt, tendencies.dqdt, dqc_dt, dqi_dt, dt,
        )
        inv_dt = 1.0 / jnp.maximum(dt, 1e-6)
        dtedt_adj = (t_adj - temperature) * inv_dt
        dqdt_adj = (q_adj - humidity) * inv_dt
        dqc_dt_adj = (qc_adj - qc) * inv_dt
        dqi_dt_adj = (qi_adj - qi) * inv_dt

        # Create enhanced tendencies with fixed qc/qi transport and the
        # adjusted saturation state.
        enhanced_tendencies = ConvectionTendencies(
            dtedt=dtedt_adj,
            dqdt=dqdt_adj,
            dudt=tendencies.dudt,
            dvdt=tendencies.dvdt,
            qc_conv=qc_conv,
            qi_conv=qi_conv,
            precip_conv=tendencies.precip_conv,
            dqc_dt=dqc_dt_adj,
            dqi_dt=dqi_dt_adj,
        )
        
        # Update state
        new_state = ConvectionState(
            tu=updraft_state.tu, qu=updraft_state.qu, lu=updraft_state.lu,
            uu=u_wind, vu=v_wind,  # Simplified - would update from momentum transport
            td=downdraft_state.td, qd=downdraft_state.qd,
            ud=u_wind, vd=v_wind,  # Simplified
            mfu=updraft_state.mfu, mfd=downdraft_state.mfd,
            ktype=jnp.array(conv_type), kbase=jnp.array(cloud_base), 
            ktop=jnp.array(ktop), prate=enhanced_tendencies.precip_conv
        )
        
        return enhanced_tendencies, new_state
    
    # No convection case (with fixed qc/qi placeholders)
    def no_convection():
        # Initialize fixed qc/qi tendencies to zero
        dqc_dt = jnp.zeros_like(qc)
        dqi_dt = jnp.zeros_like(qi)
        
        tendencies = ConvectionTendencies(
            dtedt=dtedt, dqdt=dqdt, dudt=dudt, dvdt=dvdt,
            qc_conv=qc_conv, qi_conv=qi_conv, precip_conv=precip_conv,
            dqc_dt=dqc_dt, dqi_dt=dqi_dt
        )
        return tendencies, state
    
    # Apply convection if active
    tendencies, updated_state = lax.cond(
        conv_type > 0,
        apply_full_convection,
        no_convection
    )

    return tendencies, updated_state


