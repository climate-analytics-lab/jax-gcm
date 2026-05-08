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

from jcm.constants import (
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
    cu_dnoprc_ocean: float   # Pressure thickness above cloud base before
                             # precip generation starts, over ocean (Pa)
                             # — ECHAM ``zdnoprc`` ocean default 1.5e4
    cu_dnoprc_land: float    # Same threshold over land (Pa) — ECHAM
                             # ``zdnoprc`` land default 3.0e4 (continental
                             # convection has thicker non-precipitating
                             # cloud-base layer)

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
    entrdd: float            # Downdraft fractional entrainment rate (m⁻¹)

    @classmethod
    def default(cls, dt_conv=3600.0, entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4, # FIXME: validate dt_conv
                 tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10, cprcon=1.4e-3,
                 cu_dnoprc_ocean=1.5e4, cu_dnoprc_land=3.0e4,
                 cevapcu=2.0e-5, epsilon=1.0e-12, rlcrit=8.0e-4, rhcrit=0.9,
                 cmfctop=0.33, cmfdeps=0.33, entrdd=2.0e-4) -> 'ConvectionParameters':
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
            cu_dnoprc_ocean=jnp.array(cu_dnoprc_ocean),
            cu_dnoprc_land=jnp.array(cu_dnoprc_land),
            cevapcu=jnp.array(cevapcu),
            epsilon=jnp.array(epsilon),
            rlcrit=jnp.array(rlcrit),
            rhcrit=jnp.array(rhcrit),
            cmfctop=jnp.array(cmfctop),
            cmfdeps=jnp.array(cmfdeps),
            entrdd=jnp.array(entrdd),
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


@tree_math.struct
class ConvectionData:
    """Diagnostic outputs from the Tiedtke-Nordeng convection scheme.

    Stored in the diagnostics dict under the ``"convection"`` key (no
    leading underscore — flows to user-facing xarray output as
    ``convection.<field>``). The ``mass_flux_*`` / ``cloud_base`` /
    ``cloud_top`` / ``cape`` fields are reserved for the future port of
    the equivalent ECHAM diagnostics; they are zero-filled today.
    """

    mass_flux_up: jnp.ndarray        # Updraft mass flux [kg/m²/s] (nlev, ncols)
    mass_flux_down: jnp.ndarray      # Downdraft mass flux [kg/m²/s] (nlev, ncols)
    cloud_base: jnp.ndarray          # Cloud base level index (ncols,)
    cloud_top: jnp.ndarray           # Cloud top level index (ncols,)
    cape: jnp.ndarray                # CAPE [J/kg] (ncols,)
    precip_conv: jnp.ndarray         # Convective precipitation [kg/m²/s] (ncols,)
    qc_conv: jnp.ndarray             # Convective cloud water [kg/kg] (nlev, ncols)
    qi_conv: jnp.ndarray             # Convective cloud ice [kg/kg] (nlev, ncols)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        """Construct a zero-filled ``ConvectionData`` for the given grid."""
        return cls(
            mass_flux_up=jnp.zeros((nlev,) + nodal_shape),
            mass_flux_down=jnp.zeros((nlev,) + nodal_shape),
            cloud_base=jnp.zeros(nodal_shape, dtype=int),
            cloud_top=jnp.zeros(nodal_shape, dtype=int),
            cape=jnp.zeros(nodal_shape),
            precip_conv=jnp.zeros(nodal_shape),
            qc_conv=jnp.zeros((nlev,) + nodal_shape),
            qi_conv=jnp.zeros((nlev,) + nodal_shape),
        )


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


def cloud_depth_for_target_top(
    pressure: jnp.ndarray,
    cloud_base: jnp.ndarray,
    target_top_pa: float,
    min_layers: int = 2,
) -> jnp.ndarray:
    """Return the number of model levels between ``cloud_base`` and the
    level closest to ``target_top_pa`` from above — used as the updraft
    scan ceiling.

    The scan ceiling is a *maximum* depth the updraft is allowed to
    extend to, NOT the actual cloud top. The actual termination is
    decided dynamically inside ``calculate_updraft`` (negative
    buoyancy, or mfu < 1 % of mfb). ``cloud_depth`` only needs to give
    the scan enough headroom to reach physically plausible cloud tops;
    too small a value silently truncates real convection, too large
    just wastes compute on levels that would terminate dynamically
    anyway.

    A fixed level-count value would be vertical-resolution-dependent in
    surprising ways:

    * On the 47-level ICON hybrid grid we run T85×L47 on, layers are
      ~22 hPa thick in the mid-troposphere; ``cloud_depth=35`` ≈ a
      surface-to-200-hPa scan range.
    * On a coarser 8-level sigma grid (used in some bisection tests),
      ``cloud_depth=35`` would be silently clamped to ``nlev-2`` —
      the cloud is allowed to reach the model top, which both wastes
      compute and risks unphysical extension into the stratosphere.
    * On a 90-level grid, the same ``35`` would only let the cloud
      reach ~700 hPa, cutting off real deep convection.

    Deriving from a target *pressure* makes the value
    resolution-independent. Recommended targets:

    * Deep convection: 15000 Pa (150 hPa) — tropical Cb tops typically
      reach the tropopause around this pressure.
    * Shallow convection: 70000 Pa (700 hPa) — trade-cumulus cloud
      tops at ~3 km.

    Implementation: for any pressure index ordering, find the level
    closest to ``target_top_pa`` from above (i.e. the level with the
    HIGHEST pressure among levels whose pressure ≤ target). That's
    the level we want the scan to reach. ``cloud_depth`` is then the
    integer index distance ``|cloud_base - target_top_idx|``. The
    result is clipped to ``[min_layers, nlev-2]`` so the scan always
    has at least ``min_layers`` levels of headroom and stops short of
    TOA.

    Args:
        pressure: Full-level pressure profile (Pa) [nlev]
        cloud_base: Cloud-base level index (0-indexed)
        target_top_pa: Scan should reach (at least) this pressure level
        min_layers: Minimum scan depth (≥ 2 to avoid degenerate scans)

    Returns:
        Scan-ceiling depth in *levels* (int32), clipped to
        ``[min_layers, nlev-2]``.

    """
    nlev = pressure.shape[0]
    above_target = pressure <= target_top_pa
    # Among levels at or above target, pick the HIGHEST-pressure one —
    # that's the level closest to ``target_top_pa`` from above, where we
    # want the scan to reach. ``argmax`` of ``-inf`` outside the mask
    # returns 0 if no level is above target (clipped to ``min_layers``
    # so it doesn't matter for the result).
    masked_p = jnp.where(
        above_target,
        pressure,
        jnp.array(-jnp.inf, dtype=pressure.dtype),
    )
    target_top_idx = jnp.argmax(masked_p)
    depth = jnp.abs(cloud_base.astype(jnp.int32) - target_top_idx.astype(jnp.int32))
    return jnp.clip(depth, min_layers, nlev - 2)


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
    config: ConvectionParameters = None,
    land_fraction: jnp.ndarray = jnp.array(0.0),
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
        land_fraction: Fraction of column underlying land (0=ocean, 1=land).
            Selects ECHAM's per-surface ``zdnoprc`` precip-zone threshold
            via ``config.cu_dnoprc_ocean`` / ``config.cu_dnoprc_land``.
            Defaults to 0 (ocean).

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
    
    # Determine convection type based on CAPE + free-troposphere moisture.
    # 0 = no convection, 1 = deep, 2 = shallow, 3 = mid-level
    #
    # Mirrors the ECHAM trigger structure (mo_cumastr.f90 ``zktype`` line
    # 276 + ``cubasmc`` line 660). ECHAM activates ktype=3 (mid-level
    # convection) inside ``cuasc`` when no surface-based deep/shallow
    # convection has fired AND a free-tropospheric layer is moist
    # (RH > 90 %), upward-rising and above the boundary layer (z > 1500 m).
    #
    # JAX uses CAPE-based closure (which is more responsive than ECHAM's
    # PBL-moisture-convergence closure on the same column) so ``ktype=1``
    # often fires here when ECHAM would have picked ``ktype=3`` instead.
    # We add ``ktype=3`` as a fallback for *moderate* CAPE columns
    # (100 < CAPE < 1000 J/kg) with high free-tropospheric RH — a proxy
    # for the cubasmc trigger that doesn't require a separate vertical-
    # velocity input. Deep ``ktype=1`` still wins when CAPE is large.
    qsat_env = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
    rh_env = humidity / jnp.maximum(qsat_env, 1e-12)
    # Free-troposphere mask: ~700-300 hPa
    free_trop_mask = jnp.logical_and(pressure < 70_000.0, pressure > 30_000.0)
    has_moist_free_trop = jnp.any(
        jnp.logical_and(free_trop_mask, rh_env > 0.90)
    )

    def select_active_conv_type():
        # Deep convection if CAPE is strong
        def deep_branch():
            return jnp.array(1)
        # Otherwise, mid-level if free-trop moist conditions met,
        # else shallow.
        def mid_or_shallow_branch():
            return lax.cond(has_moist_free_trop, lambda: jnp.array(3),
                            lambda: jnp.array(2))
        return lax.cond(cape > 1000.0, deep_branch, mid_or_shallow_branch)

    conv_type = lax.cond(
        jnp.logical_and(has_cloud_base, cape > 100.0),
        select_active_conv_type,
        lambda: jnp.array(0),  # No convection
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
        # Cloud-top scan ceiling. The ceiling is a *maximum* depth, not
        # the actual cloud top — actual termination is decided
        # dynamically inside ``calculate_updraft`` (negative buoyancy or
        # mfu < 1 % of mfb). Derive the ceiling from a target cloud-top
        # PRESSURE rather than a fixed level count so the value is
        # vertical-resolution-independent. Targets:
        #   * Deep:    150 hPa (tropical Cb tops near the tropopause)
        #   * Shallow: 700 hPa (trade-cumulus tops at ~3 km)
        # See ``cloud_depth_for_target_top`` for the derivation and a
        # detailed discussion of why a fixed level count is wrong.
        cloud_depth = lax.cond(
            conv_type == 2,
            lambda: cloud_depth_for_target_top(pressure, cloud_base, 70_000.0),
            lambda: cloud_depth_for_target_top(pressure, cloud_base, 15_000.0),
        )

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

        # ECHAM mass-flux CFL cap (``mo_cumastr.f90:582-583``):
        #
        #     zmfmax = pmref(jl, ikb-1) / dt
        #     zmfub1 = MIN(zmfub1, zmfmax)
        #
        # The convective updraft cannot evacuate more mass per unit time
        # than the source layer at cloud base contains. Without this cap
        # the closure can return arbitrarily large mass fluxes when CAPE
        # is high relative to the convective timescale, producing run-
        # away latent heating in a single step. We use the air mass of
        # the cloud-base layer itself (``rho * dz``) as the budget.
        layer_mass_at_cb = rho[cloud_base] * layer_thickness[cloud_base]
        mfu_cfl_max = layer_mass_at_cb / dt
        mass_flux_base = jnp.minimum(mass_flux_base, mfu_cfl_max)
        
        # Calculate updraft
        updraft_state = calculate_updraft(
            temperature, humidity, pressure, layer_thickness, rho,
            cloud_base, ktop, conv_type, mass_flux_base, config,
            land_fraction=land_fraction,
        )
        
        # Calculate precipitation from updraft
        # Use the per-layer precip generated inside calculate_updraft (the
        # ECHAM ``pdmfup`` accumulator) rather than the previous
        # ``sum(lu*mfu)*cprcon`` estimator, which was ~60x too small on
        # tropical RCE columns. See ``flux_tendencies.calculate_precipitation_rate``.
        precip_rate = jnp.sum(updraft_state.pdmfup)
        
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
        # the adjusted state.
        #
        # Restrict the adjustment to the cloud column. ECHAM's `cuadjtq`
        # is only called on the cloud levels processed by cuasc /
        # cubase; running it across the whole column makes the
        # saturation adjustment fire on every above-cloud-top level
        # whose initial RH happens to exceed the JAX qsat cutoff
        # (which differs slightly from the lookup-table values ECHAM
        # uses), producing spurious heating tens of times larger than
        # the actual convective flux divergence. See the harness
        # comparison in fortran_harness/compare_cumastr.py for the
        # diagnostic that surfaces this.
        #
        # The cloud column is delimited by the *actual* updraft extent
        # (where mfu is still nonzero), not the scan ceiling ``ktop`` —
        # using the ceiling lets the adjustment fire above the real
        # cloud top, where the env can be supersaturated relative to
        # JAX's qsat formula and the iteration explodes into spurious
        # condensational heating. Derive the actual top from where the
        # updraft mass flux extends.
        _mfu_active_for_mask = updraft_state.mfu > config.cmfcmin
        _has_active_for_mask = jnp.any(_mfu_active_for_mask)
        _candidate_for_mask = jnp.where(
            _mfu_active_for_mask, jnp.arange(nlev),
            jnp.array(nlev, jnp.int32),
        )
        actual_top_for_mask = jnp.where(
            _has_active_for_mask,
            jnp.min(_candidate_for_mask).astype(jnp.int32),
            ktop,
        )
        cloud_top = jnp.minimum(actual_top_for_mask, cloud_base)
        cloud_bottom = jnp.maximum(actual_top_for_mask, cloud_base)
        cloud_mask = (jnp.arange(nlev) >= cloud_top - 1) & (
            jnp.arange(nlev) <= cloud_bottom
        )
        zero_outside = lambda arr: jnp.where(cloud_mask, arr, 0.0)
        t_adj, q_adj, qc_adj, qi_adj = convective_adjustment(
            temperature, humidity, pressure, qc, qi,
            zero_outside(tendencies.dtedt), zero_outside(tendencies.dqdt),
            zero_outside(dqc_dt), zero_outside(dqi_dt), dt,
        )
        # Outside the cloud column, leave the original state untouched
        # (no tendency, no condensation) regardless of what the
        # saturation lookup said.
        t_adj  = jnp.where(cloud_mask, t_adj,  temperature)
        q_adj  = jnp.where(cloud_mask, q_adj,  humidity)
        qc_adj = jnp.where(cloud_mask, qc_adj, qc)
        qi_adj = jnp.where(cloud_mask, qi_adj, qi)
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
        
        # ECHAM-ICON convention: ktop is the smallest level index (highest
        # altitude) where the updraft mass flux is still nonzero — i.e.
        # where the dynamic termination in `calculate_updraft` last left
        # a nonzero `mfu` before zeroing it above. The previous code wrote
        # the *scan ceiling* ``ktop = kbase - cloud_depth``, which masks
        # the actual cloud top whenever the updraft terminates early.
        # Re-derive it from where ``updraft_state.mfu`` is still active.
        mfu_active = updraft_state.mfu > config.cmfcmin
        has_active = jnp.any(mfu_active)
        candidate = jnp.where(
            mfu_active, jnp.arange(nlev), jnp.array(nlev, jnp.int32),
        )
        # ``min(candidate)`` = topmost active level (smallest index in
        # ECHAM ordering). If no level is active, fall back to the scan
        # ceiling so downstream consumers don't see ``nlev``.
        actual_ktop = jnp.where(
            has_active, jnp.min(candidate).astype(jnp.int32), ktop,
        )

        # Update state
        new_state = ConvectionState(
            tu=updraft_state.tu, qu=updraft_state.qu, lu=updraft_state.lu,
            uu=u_wind, vu=v_wind,  # Simplified - would update from momentum transport
            td=downdraft_state.td, qd=downdraft_state.qd,
            ud=u_wind, vd=v_wind,  # Simplified
            mfu=updraft_state.mfu, mfd=downdraft_state.mfd,
            ktype=jnp.array(conv_type), kbase=jnp.array(cloud_base),
            ktop=actual_ktop, prate=enhanced_tendencies.precip_conv,
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


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class TiedtkeConvection(PhysicsTerm):
    """Tiedtke-Nordeng mass-flux convection as a composable PhysicsTerm.

    Operates on column-vectorized state ``(nlev, ncols)``. Calls the
    standalone :func:`tiedtke_nordeng_convection` scheme via ``jax.vmap``
    over columns. Holds its own :class:`ConvectionParameters` as
    ``nnx.Param`` so that gradients flow through them.

    Reads the moist-air diagnostics produced by
    :class:`~jcm.physics.diagnostics.moist_air_state.MoistAirColumnState`
    (``pressure_full``, ``layer_thickness``, ``air_density``) and the
    model timestep from ``diagnostics["_date"].dt_seconds``. Writes the
    :class:`ConvectionData` sub-struct under the public ``"convection"``
    key.
    """

    name: ClassVar[str] = "tiedtke_convection"
    category: ClassVar[str] = "convection"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "layer_thickness", "air_density",
    )
    provides: ClassVar[tuple[str, ...]] = ("convection",)

    def __init__(self, params: ConvectionParameters | None = None):
        """Hold the scheme-native :class:`ConvectionParameters`."""
        self.params = nnx.Param(params or ConvectionParameters.default())

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """Declare ``qc`` and ``qi`` so the dynamics carries them across steps.

        The scheme transports cloud water and ice prognostically inside
        each updraft/downdraft, so the tendencies it returns rely on
        seeing yesterday's qc/qi at the start of every column call.
        """
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
        )

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute convective tendencies; write ``convection`` diagnostics."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_date"].dt_seconds
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        layer_thickness = diagnostics["layer_thickness"]
        air_density = diagnostics["air_density"]

        qc = state.tracers.get("qc", jnp.zeros_like(state.temperature))
        qi = state.tracers.get("qi", jnp.zeros_like(state.temperature))

        # Per-column land fraction selects between ECHAM's ocean and land
        # ``zdnoprc`` precip-zone thresholds inside the updraft.
        land_fraction = terrain.fmask.reshape(ncols)

        column_fn = jax.vmap(
            tiedtke_nordeng_convection,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None, 0),
            out_axes=(0, 0),
        )
        tendencies_all, _state_all = column_fn(
            state.temperature, state.specific_humidity,
            pressure_full, layer_thickness, air_density,
            state.u_wind, state.v_wind, qc, qi,
            dt, params, land_fraction,
        )

        # Hard limit on the convective T tendency: 5 K/hr, applied
        # symmetrically. Healthy deep convection over the warmest tropical
        # SSTs gives ~1 K/hr at the most active level; the cap only fires
        # when the column's parcel-vs-environment energy balance has gone
        # pathological. The companion cloud-base mass-flux CFL cap inside
        # ``tiedtke_nordeng_convection`` bounds the column-integrated mass
        # flux but does not contain per-level latent-heat spikes inside
        # the updraft loop — ECHAM bounds those via the per-level moist-
        # adjustment limits in ``mo_cuadjust.f90`` which we have not yet
        # ported. Until that lands this cap is the safety net.
        _DTDT_MAX = 5.0 / 3600.0  # K/s
        dt_capped = jnp.clip(tendencies_all.dtedt, -_DTDT_MAX, _DTDT_MAX)

        tendency = PhysicsTendency(
            u_wind=tendencies_all.dudt.T,
            v_wind=tendencies_all.dvdt.T,
            temperature=dt_capped.T,
            specific_humidity=tendencies_all.dqdt.T,
            tracers={
                "qc": tendencies_all.dqc_dt.T,
                "qi": tendencies_all.dqi_dt.T,
            },
        )

        # Mass-flux / cloud-base/top / CAPE diagnostics aren't populated
        # by the wrapper today (the scheme returns the per-column state
        # but we don't reduce or surface it yet) — they stay as zeros
        # for back-compat with existing xarray field names.
        convection = ConvectionData(
            mass_flux_up=jnp.zeros_like(pressure_full),
            mass_flux_down=jnp.zeros_like(pressure_full),
            cloud_base=jnp.zeros(ncols, dtype=int),
            cloud_top=jnp.zeros(ncols, dtype=int),
            cape=jnp.zeros(ncols),
            precip_conv=tendencies_all.precip_conv,
            qc_conv=tendencies_all.qc_conv.T,
            qi_conv=tendencies_all.qi_conv.T,
        )

        return tendency, {**diagnostics, "convection": convection}


