"""Cloud microphysics scheme for ECHAM physics

This module implements comprehensive cloud microphysics including:
- Autoconversion of cloud water to rain (Khairoutdinov and Kogan, 2000)
- Accretion of cloud droplets by rain
- Autoconversion of cloud ice to snow 
- Aggregation of ice crystals and accretion by snow
- Melting of snow and freezing of rain
- Sedimentation of cloud ice and snow
- Evaporation of rain and sublimation of snow

Based on the ECHAM6/ICON microphysics as described in:
- Lohmann and Roeckner (1996)
- Levkov et al. (1992) for ice phase
- Beheng (1994) for warm phase

Date: 2025-01-10
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
import tree_math

from jcm.constants import (
    tmelt, alhf, alhc, alhs, cp, rhow, rv, eps
)


@tree_math.struct
class MicrophysicsParameters:
    """Configuration parameters for cloud microphysics"""
    
    # Autoconversion parameters
    ccraut: float        # Critical cloud water for autoconversion (kg/kg)
    ccracl: float        # Accretion coefficient (cloud to rain)
    cauloc: float        # Cloud droplet dispersion parameter
    ceffmin: float       # Minimum cloud droplet radius (microns)
    ceffmax: float       # Maximum cloud droplet radius (microns)
    
    # Ice microphysics parameters
    cn0s: float          # Snow particle number density (1/m^3)
    crhosno: float       # Snow density (kg/m^3)
    cvtfall: float       # Terminal velocity factor for ice
    cthomi: float        # Homogeneous ice nucleation temperature (K)
    csecfrl: float       # Critical ice fraction for Bergeron-Findeisen
    
    # Collection efficiencies
    ccollec: float       # Collection efficiency rain/cloud
    ccollei: float       # Collection efficiency snow/ice
    
    # Time scale parameters
    tau_melt: float      # Melting time scale (s)
    tau_freeze: float    # Freezing time scale (s)
    
    # Evaporation/sublimation parameters
    cevaprain: float     # Rain evaporation coefficient
    cevapsnow: float     # Snow sublimation coefficient
    
    # Sedimentation parameters
    vt_ice: float        # Ice crystal fall speed (m/s)
    vt_snow_a: float     # Snow fall speed coefficient a
    vt_snow_b: float     # Snow fall speed exponent b
    vt_rain_a: float     # Rain fall speed coefficient a
    vt_rain_b: float     # Rain fall speed exponent b
    
    # Cloud droplet number concentration
    base_cdnc: float     # Baseline CDNC in clean air (1/m³), modulated by aerosol cdnc_factor

    # Numerical parameters
    epsilon: float       # Small number for numerical stability
    dt_sedi: float       # Sub-timestep for sedimentation (s)

    # Autoconversion scheme selector (int flag — JAX won't trace strings).
    # 0 = Beheng (1994) implicit form (default; robust at large dt).
    # 1 = Khairoutdinov & Kogan (2000) explicit form (good fit for 2M
    #     microphysics with prognostic Nc).
    # ``ccraut`` is interpreted differently by each scheme: in Beheng
    # it's the rate prefactor (default 15.0); in KK2000 it's the qc
    # threshold above which autoconversion fires (a small g/kg-scale
    # value is appropriate, e.g. 1e-5).
    autoconversion_scheme: int

    SCHEME_BEHENG = 0
    SCHEME_KK2000 = 1

    @classmethod
    def default(cls, ccraut=15.0, ccracl=6.0, cauloc=1.0, ceffmin=10.0, ceffmax=150.0, cn0s=3.0e6,
                 crhosno=100.0, cvtfall=3.29, cthomi=233.15, csecfrl=0.1, ccollec=0.7,
                 ccollei=0.3, tau_melt=100.0, tau_freeze=100.0, cevaprain=1.0e-3,
                 cevapsnow=5.0e-4, vt_ice=0.1, vt_snow_a=8.8, vt_snow_b=0.15,
                 vt_rain_a=386.0, vt_rain_b=0.67, base_cdnc=100.0e6,
                 epsilon=1.0e-12, dt_sedi=10.0,
                 autoconversion_scheme=0) -> 'MicrophysicsParameters':
        """Return default microphysics parameters.

        ``autoconversion_scheme`` accepts either the int constant
        (``SCHEME_BEHENG`` / ``SCHEME_KK2000``) or the string aliases
        ``"beheng"`` / ``"kk2000"``.
        """
        if isinstance(autoconversion_scheme, str):
            scheme_map = {
                "beheng": cls.SCHEME_BEHENG,
                "kk2000": cls.SCHEME_KK2000,
            }
            autoconversion_scheme = scheme_map[autoconversion_scheme]

        return cls(
            ccraut=jnp.array(ccraut),
            ccracl=jnp.array(ccracl),
            cauloc=jnp.array(cauloc),
            ceffmin=jnp.array(ceffmin),
            ceffmax=jnp.array(ceffmax),
            cn0s=jnp.array(cn0s),
            crhosno=jnp.array(crhosno),
            cvtfall=jnp.array(cvtfall),
            cthomi=jnp.array(cthomi),
            csecfrl=jnp.array(csecfrl),
            ccollec=jnp.array(ccollec),
            ccollei=jnp.array(ccollei),
            tau_melt=jnp.array(tau_melt),
            tau_freeze=jnp.array(tau_freeze),
            cevaprain=jnp.array(cevaprain),
            cevapsnow=jnp.array(cevapsnow),
            vt_ice=jnp.array(vt_ice),
            vt_snow_a=jnp.array(vt_snow_a),
            vt_snow_b=jnp.array(vt_snow_b),
            vt_rain_a=jnp.array(vt_rain_a),
            vt_rain_b=jnp.array(vt_rain_b),
            base_cdnc=jnp.array(base_cdnc),
            epsilon=jnp.array(epsilon),
            dt_sedi=jnp.array(dt_sedi),
            autoconversion_scheme=int(autoconversion_scheme),
        )


class MicrophysicsState(NamedTuple):
    """Microphysics state variables and diagnostics"""
    
    # Precipitation fluxes (kg/m²/s)
    rain_flux: jnp.ndarray      # Rain flux at each level
    snow_flux: jnp.ndarray      # Snow flux at each level
    
    # In-cloud values
    qc_in_cloud: jnp.ndarray    # In-cloud liquid water (kg/kg)
    qi_in_cloud: jnp.ndarray    # In-cloud ice (kg/kg)
    
    # Process rates (kg/kg/s)
    autoconv_rate: jnp.ndarray  # Autoconversion rate
    accretion_rate: jnp.ndarray # Accretion rate
    melting_rate: jnp.ndarray   # Melting rate
    freezing_rate: jnp.ndarray  # Freezing rate
    
    # Precipitation at surface
    precip_rain: jnp.ndarray    # Surface rain (kg/m²/s)
    precip_snow: jnp.ndarray    # Surface snow (kg/m²/s)


class MicrophysicsTendencies(NamedTuple):
    """Tendencies from microphysics processes"""
    
    dtedt: jnp.ndarray          # Temperature tendency (K/s)
    dqdt: jnp.ndarray           # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray          # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray          # Cloud ice tendency (kg/kg/s)
    dqrdt: jnp.ndarray          # Rain water tendency (kg/kg/s)
    dqsdt: jnp.ndarray          # Snow tendency (kg/kg/s)


def cloud_droplet_radius(
    cloud_water: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Calculate effective cloud droplet radius
    
    Args:
        cloud_water: Cloud liquid water content (kg/kg)
        air_density: Air density (kg/m³)
        droplet_number: Droplet number concentration (1/kg)
        config: Microphysics configuration
        
    Returns:
        Effective radius (m)

    """
    # Convert mixing ratio to mass concentration
    cloud_water_density = cloud_water * air_density  # kg/m³
    
    # Convert droplet number from per kg to per m³
    droplet_density = droplet_number * air_density  # 1/m³
    
    # Volume of single droplet
    volume_per_droplet = cloud_water_density / (droplet_density + config.epsilon) / rhow  # m³
    
    # Volume mean radius
    radius = (3.0 * volume_per_droplet / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    
    # Apply limits
    radius = jnp.clip(radius, config.ceffmin * 1e-6, config.ceffmax * 1e-6)
    
    return radius


def autoconversion_beheng(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Autoconversion of cloud water to rain — Beheng (1994) implicit form.

    Mirrors ECHAM ``mo_cloud.f90`` lines 841-863. The implicit integration
    is what makes this scheme robust at realistic post-convection cloud
    water values: the depletion fraction stays in [0, 1] even when the
    instantaneous Beheng rate × dt would overshoot.

        zraut_rate = ccraut * 1.2e27 / rho * Nc^-3.3 * rho^4.7 * qc^3.7
        qc_remain  = (1 + zraut_rate * dt * 3.7 * qc^3.7) ^ (-1/3.7)
        autoconv   = qc * (1 - qc_remain) / dt

    Default in the 1M scheme. The KK2000 form
    (``autoconversion_kk2000``) is also available and may be a better
    pairing with explicit-Nc 2M microphysics; pick via
    ``MicrophysicsParameters(autoconversion_scheme="beheng" | "kk2000")``.

    Args:
        cloud_water: Grid-mean cloud water mixing ratio (kg/kg)
        cloud_fraction: Cloud fraction (0-1)
        air_density: Air density (kg/m³)
        droplet_number: Cloud droplet number concentration (1/m³)
        dt: Time step (s)
        config: Microphysics configuration (uses ccraut, epsilon)

    Returns:
        Grid-mean autoconversion rate (kg/kg/s)

    """
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0,
    )

    zexm1 = 3.7  # 4.7 - 1.0
    nc_per_cm3 = droplet_number * air_density * 1e-6  # 1/cm³
    rho_g_cm3 = air_density * 1e-3                    # g/cm³

    # Beheng's Nc^-3.3 dependence blows up for Nc → 0; floor at 1/cm³
    nc_safe = jnp.maximum(nc_per_cm3, 1.0)
    zraut_rate = (
        config.ccraut * 1.2e27 / air_density
        * nc_safe ** (-3.3)
        * rho_g_cm3 ** 4.7
    )

    # Implicit integration: protect against (qc^zexm1) underflow at
    # near-zero qc — the formula gives no autoconv there anyway.
    qc_pow = jnp.where(qc_in_cloud > 1e-12, qc_in_cloud ** zexm1, 0.0)
    denominator = 1.0 + zraut_rate * dt * zexm1 * qc_pow
    qc_remaining_frac = denominator ** (-1.0 / zexm1)
    autoconv_in_cloud = qc_in_cloud * (1.0 - qc_remaining_frac) / dt

    # Convert to grid-mean
    return autoconv_in_cloud * cloud_fraction


def autoconversion_kk2000(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Autoconversion of cloud water to rain — Khairoutdinov & Kogan (2000).

    Explicit-rate form:

        P_aut = 1350 * qc^2.47 * (Nc·1e-6)^(-1.79)   [g/m³/s]

    Activates above the ``ccraut`` threshold. KK2000 was the original
    1M default and remains a good fit for 2M microphysics where the
    droplet number ``Nc`` is a prognostic variable. In the 1M context
    with prescribed ``Nc`` and large dt, the explicit form can produce
    ``rate × dt > qc`` at high cloud water (~37500 % depletion at
    qc = 0.3 g/kg, dt = 1800 s); downstream code must clip to mass
    conservation. ``autoconversion_beheng`` is the more robust 1M
    default; KK2000 is preferred when paired with prognostic ``Nc``.

    Args:
        cloud_water: Grid-mean cloud water mixing ratio (kg/kg)
        cloud_fraction: Cloud fraction (0-1)
        air_density: Air density (kg/m³)
        droplet_number: Cloud droplet number concentration (1/m³)
        dt: Time step (s) — unused (explicit rate); kept in signature
            for parity with ``autoconversion_beheng``.
        config: Microphysics configuration (uses ccraut, epsilon)

    Returns:
        Grid-mean autoconversion rate (kg/kg/s)

    """
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0,
    )

    qc_gm3 = qc_in_cloud * air_density * 1000.0      # g/m³
    nc_cm3 = droplet_number * air_density * 1e-6     # 1/cm³

    # KK2000 rate. The 1e-3 converts g/m³/s → kg/m³/s, then divide
    # by air density to recover the kg/kg/s mixing-ratio tendency.
    autoconv_rate = jnp.where(
        qc_in_cloud > config.ccraut,
        1350.0 * qc_gm3 ** 2.47 * (nc_cm3 + config.epsilon) ** (-1.79)
        * 1e-3 / air_density,
        0.0,
    )

    return autoconv_rate * cloud_fraction


def autoconversion(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters,
) -> jnp.ndarray:
    """Dispatcher — picks Beheng or KK2000 by ``config.autoconversion_scheme``.

    Both schemes have the same signature so ``lax.cond`` can switch
    cleanly between them at runtime.
    """
    return jax.lax.cond(
        config.autoconversion_scheme == MicrophysicsParameters.SCHEME_KK2000,
        lambda: autoconversion_kk2000(
            cloud_water, cloud_fraction, air_density, droplet_number, dt, config,
        ),
        lambda: autoconversion_beheng(
            cloud_water, cloud_fraction, air_density, droplet_number, dt, config,
        ),
    )


def accretion_rain_cloud(
    cloud_water: jnp.ndarray,
    rain_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Accretion of cloud droplets by rain
    
    Args:
        cloud_water: Cloud water mixing ratio (kg/kg)
        rain_water: Rain water mixing ratio (kg/kg)
        cloud_fraction: Cloud fraction (0-1)
        air_density: Air density (kg/m³)
        config: Microphysics configuration
        
    Returns:
        Accretion rate (kg/kg/s)

    """
    # In-cloud values
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0
    )
    
    # Accretion rate following Beheng (1994)
    # Uses collection efficiency and geometric sweep-out
    accretion_rate = config.ccracl * config.ccollec * qc_in_cloud * rain_water * air_density**0.5
    
    # Convert to grid-mean
    accretion_rate = accretion_rate * cloud_fraction
    
    return accretion_rate


def ice_autoconversion(
    cloud_ice: jnp.ndarray,
    temperature: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Autoconversion of cloud ice to snow through aggregation
    
    Args:
        cloud_ice: Cloud ice mixing ratio (kg/kg)
        temperature: Temperature (K)
        cloud_fraction: Cloud fraction (0-1)
        dt: Time step (s)
        config: Microphysics configuration
        
    Returns:
        Ice autoconversion rate (kg/kg/s)

    """
    # Temperature-dependent aggregation efficiency
    # Maximum near -15°C (258K)
    t_celsius = temperature - tmelt
    agg_efficiency = jnp.exp(-0.05 * jnp.abs(t_celsius + 15.0))
    
    # Critical ice content for autoconversion (fixed)
    qi_crit = 0.3e-3  # kg/kg
    
    # In-cloud ice
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / cloud_fraction,
        0.0
    )
    
    # Autoconversion rate with temperature-dependent efficiency
    autoconv_rate = jnp.where(
        qi_in_cloud > qi_crit,
        agg_efficiency * 0.001 * (qi_in_cloud - qi_crit) / dt,
        0.0
    )
    
    # Convert to grid mean
    autoconv_rate = autoconv_rate * cloud_fraction
    
    # Limit to available ice
    max_rate = cloud_ice / dt
    autoconv_rate = jnp.minimum(autoconv_rate, max_rate)
    
    return autoconv_rate


def snow_accretion(
    target: jnp.ndarray,
    snow: jnp.ndarray, 
    temperature: jnp.ndarray,
    air_density: jnp.ndarray,
    is_liquid: bool,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Accretion of cloud water/ice by falling snow
    
    Args:
        target: Target species mixing ratio (cloud water or ice) (kg/kg)
        snow: Snow mixing ratio (kg/kg)
        temperature: Temperature (K)
        air_density: Air density (kg/m³)
        is_liquid: True for riming (liquid), False for aggregation (ice)
        config: Microphysics configuration
        
    Returns:
        Accretion rate (kg/kg/s)

    """
    # Collection efficiency
    efficiency = config.ccollec if is_liquid else config.ccollei
    
    # Temperature factor for aggregation (ice only)
    if not is_liquid:
        t_celsius = temperature - tmelt
        temp_factor = jnp.exp(-0.03 * jnp.abs(t_celsius + 15.0))
        efficiency = efficiency * temp_factor
    
    # Snow fall velocity
    snow_gm3 = snow * air_density * 1000.0  # g/m³
    vt_snow = config.vt_snow_a * snow_gm3**config.vt_snow_b
    
    # Accretion rate
    accretion_rate = efficiency * target * snow * vt_snow / (air_density**0.5)
    
    return accretion_rate


def melting_freezing(
    temperature: jnp.ndarray,
    snow: jnp.ndarray,
    rain: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate melting of snow and freezing of rain
    
    Args:
        temperature: Temperature (K)
        snow: Snow mixing ratio (kg/kg)
        rain: Rain mixing ratio (kg/kg)  
        dt: Time step (s)
        config: Microphysics configuration
        
    Returns:
        Tuple of (melting_rate, freezing_rate) in kg/kg/s

    """
    # Temperature departure from freezing
    dt_freeze = tmelt - temperature
    
    # Melting rate (T > 0°C)
    melt_rate = jnp.where(
        temperature > tmelt,
        snow * (temperature - tmelt) / (config.tau_melt * 10.0),  # Scaled by temp
        0.0
    )
    melt_rate = jnp.minimum(melt_rate, snow / dt)
    
    # Freezing rate (T < 0°C)  
    # Heterogeneous freezing increases rapidly below -5°C
    freeze_efficiency = jnp.where(
        dt_freeze > 5.0,
        1.0 - jnp.exp(-0.5 * (dt_freeze - 5.0)),
        0.0
    )
    
    freeze_rate = freeze_efficiency * rain / config.tau_freeze
    freeze_rate = jnp.minimum(freeze_rate, rain / dt)
    
    return melt_rate, freeze_rate


def evaporation_sublimation(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    rain: jnp.ndarray,
    snow: jnp.ndarray,
    air_density: jnp.ndarray,
    config: MicrophysicsParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate evaporation of rain and sublimation of snow
    
    Args:
        temperature: Temperature (K)
        specific_humidity: Specific humidity (kg/kg)
        pressure: Pressure (Pa)
        rain: Rain mixing ratio (kg/kg)
        snow: Snow mixing ratio (kg/kg)
        air_density: Air density (kg/m³)
        config: Microphysics configuration
        
    Returns:
        Tuple of (rain_evap_rate, snow_sublim_rate) in kg/kg/s

    """
    from .sundqvist import saturation_specific_humidity
    
    # Saturation specific humidity
    qs = saturation_specific_humidity(pressure, temperature)
    
    # Subsaturation
    subsaturation = jnp.maximum(0.0, (qs - specific_humidity) / qs)
    
    # Rain evaporation
    rain_gm3 = rain * air_density * 1000.0
    rain_evap = jnp.where(
        rain > config.epsilon,
        config.cevaprain * subsaturation * rain_gm3**0.5 / air_density,
        0.0
    )
    
    # Snow sublimation
    snow_gm3 = snow * air_density * 1000.0  
    snow_sublim = jnp.where(
        snow > config.epsilon,
        config.cevapsnow * subsaturation * snow_gm3**0.5 / air_density,
        0.0
    )
    
    return rain_evap, snow_sublim


def sedimentation_flux(
    hydrometeor: jnp.ndarray,
    air_density: jnp.ndarray,
    dz: jnp.ndarray,
    terminal_velocity: jnp.ndarray,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate sedimentation flux and tendency for a hydrometeor
    
    Uses upwind differencing with flux limiter to maintain stability.
    JAX-compatible implementation without loops.
    
    Args:
        hydrometeor: Hydrometeor mixing ratio (kg/kg) [nlev]
        air_density: Air density (kg/m³) [nlev]
        dz: Layer thickness (m) [nlev]
        terminal_velocity: Fall velocity (m/s) [nlev]
        dt: Time step (s)
        
    Returns:
        Tuple of (flux [nlev+1], tendency [nlev])

    """
    # Mass content (kg/m³)
    mass_content = hydrometeor * air_density
    
    # Calculate fluxes at each interface (upwind)
    # Flux from level k to k+1
    flux_unlimited = mass_content * terminal_velocity
    
    # CFL limiter to prevent overshooting
    max_flux = mass_content * dz / dt
    flux_limited = jnp.minimum(flux_unlimited, max_flux)
    
    # Build interface fluxes
    # flux[0] = 0 (top), flux[k+1] = flux from level k
    flux = jnp.concatenate([jnp.zeros(1), flux_limited])
    
    # Tendency from flux divergence
    # (flux_in - flux_out) / (dz * rho)
    flux_in = flux[:-1]  # Flux from above
    flux_out = flux[1:]  # Flux to below
    tendency = (flux_in - flux_out) / (dz * air_density)
    
    return flux, tendency


def cloud_microphysics(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: Optional[MicrophysicsParameters] = None,
    rain_water: Optional[jnp.ndarray] = None,
    snow: Optional[jnp.ndarray] = None
) -> Tuple[MicrophysicsTendencies, MicrophysicsState]:
    """Run cloud microphysics scheme
    
    Computes tendencies from all microphysical processes including:
    - Autoconversion and accretion
    - Melting and freezing
    - Evaporation and sublimation
    - Sedimentation
    
    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev]
        cloud_water: Cloud liquid water (kg/kg) [nlev]
        cloud_ice: Cloud ice (kg/kg) [nlev]
        cloud_fraction: Cloud fraction [nlev]
        air_density: Air density (kg/m³) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        droplet_number: Droplet number concentration (1/kg) [nlev]
        dt: Time step (s)
        config: Microphysics configuration
        rain_water: Rain water mixing ratio (kg/kg) [nlev]. If None, initialized to zeros.
        snow: Snow mixing ratio (kg/kg) [nlev]. If None, initialized to zeros.
        
    Returns:
        Tuple of (tendencies, state)

    """
    if config is None:
        config = MicrophysicsParameters.default()
    
    # Ensure all inputs are arrays
    temperature = jnp.atleast_1d(temperature)
    nlev = temperature.shape[0]
    
    # Initialize tendencies
    dtedt = jnp.zeros(nlev)
    dqdt = jnp.zeros(nlev)
    dqcdt = jnp.zeros(nlev)
    dqidt = jnp.zeros(nlev)
    dqrdt = jnp.zeros(nlev)
    dqsdt = jnp.zeros(nlev)
    
    # Initialize precipitation if not provided
    if rain_water is None:
        rain_water = jnp.zeros(nlev)
    if snow is None:
        snow = jnp.zeros(nlev)
    
    # Calculate in-cloud values
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0
    )
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / cloud_fraction,
        0.0
    )
    
    # 1. Autoconversion processes
    qc_auto = autoconversion(
        cloud_water, cloud_fraction, air_density, droplet_number, dt, config
    )
    qi_auto = ice_autoconversion(
        cloud_ice, temperature, cloud_fraction, dt, config
    )
    
    # 2. Accretion processes
    qc_accr = accretion_rain_cloud(
        cloud_water, rain_water, cloud_fraction, air_density, config
    )
    qc_rime = snow_accretion(
        cloud_water, snow, temperature, air_density, True, config
    )
    qi_aggr = snow_accretion(
        cloud_ice, snow, temperature, air_density, False, config
    )
    
    # 3. Melting and freezing
    snow_melt, rain_freeze = melting_freezing(
        temperature, snow, rain_water, dt, config
    )
    
    # 4. Evaporation and sublimation
    rain_evap, snow_sublim = evaporation_sublimation(
        temperature, specific_humidity, pressure,
        rain_water, snow, air_density, config
    )
    
    # 5. Update tendencies from microphysical processes
    # Cloud water: loses to autoconversion, accretion, riming
    dqcdt = -(qc_auto + qc_accr + qc_rime)
    
    # Cloud ice: loses to autoconversion and aggregation
    dqidt = -(qi_auto + qi_aggr)
    
    # Rain: gains from warm processes and melting, loses to evaporation and freezing
    dqrdt = qc_auto + qc_accr + snow_melt - rain_evap - rain_freeze
    
    # Snow: gains from cold processes and freezing, loses to melting and sublimation
    dqsdt = qi_auto + qi_aggr + qc_rime + rain_freeze - snow_melt - snow_sublim
    
    # Humidity: gains from evaporation/sublimation
    dqdt = rain_evap + snow_sublim
    
    # Temperature: latent heat effects
    dtedt = (
        - alhc / cp * (rain_evap - qc_auto - qc_accr)  # Liquid phase changes
        - alhs / cp * (snow_sublim - qi_auto - qi_aggr - qc_rime)  # Ice phase changes  
        - alhf / cp * (snow_melt - rain_freeze)  # Melting/freezing
    )
    
    # 6. Sedimentation (using simple approach for now)
    # Calculate terminal velocities
    rain_gm3 = rain_water * air_density * 1000.0
    vt_rain = config.vt_rain_a * rain_gm3**config.vt_rain_b * 1e-3  # m/s
    
    snow_gm3 = snow * air_density * 1000.0
    vt_snow = config.vt_snow_a * snow_gm3**config.vt_snow_b * 1e-3  # m/s
    
    # Ice sedimentation: ECHAM mo_cloud.f90 lines 472-491 uses
    # ``zxifall = cvtfall * (rho*xi)^0.16`` (Heymsfield-Donner content-
    # dependent fall speed) PLUS an exponential-decay integral form
    # ``zxised = xi*exp(-vt*dt/dz) + flux_in/(rho*vt)*(1-exp(...))`` so
    # the per-timestep depletion is naturally bounded by ``1 - exp(-…)``,
    # never the unbounded ``vt*dt/dz`` of the instantaneous form.
    #
    # The cloud harness flagged the prior implementation as ~3x too
    # slow (fixed vt=0.1 m/s). After switching to the content-dependent
    # vt but keeping the instantaneous rate formula, it became ~3x too
    # fast (the linear rate overshoots for typical dt=1800s timesteps).
    # Use the integral form for ice; keep the linear approximation for
    # rain/snow which usually have shorter residence times than dt.
    rho_qi = jnp.maximum(air_density * cloud_ice, config.epsilon)
    vt_ice = config.cvtfall * rho_qi ** 0.16  # m/s, content-dependent

    # Integral form: per-timestep depletion fraction = 1 - exp(-vt*dt/dz),
    # converted back to a rate by dividing by dt.
    zal1_ice = vt_ice * dt / jnp.maximum(layer_thickness, config.epsilon)
    ice_sedi = cloud_ice * (1.0 - jnp.exp(-zal1_ice)) / jnp.maximum(dt, 1e-6)

    rain_sedi = rain_water * vt_rain / (layer_thickness + config.epsilon)
    snow_sedi = snow * vt_snow / (layer_thickness + config.epsilon)
    
    # Update tendencies
    dqidt = dqidt - ice_sedi
    dqrdt = dqrdt - rain_sedi
    dqsdt = dqsdt - snow_sedi
    
    # Surface precipitation: column-integrated rain/snow production
    # Following ECHAM mo_cloud.f90: zzdrr = zrpr * pmref / pdtime
    # where pmref = air_density * layer_thickness (layer mass per unit area)
    layer_mass = air_density * layer_thickness  # kg/m²

    # Rain production per level: autoconversion + accretion + snow melting
    rain_prod = (qc_auto + qc_accr + snow_melt) * cloud_fraction
    # Snow production per level: ice autoconversion + aggregation + riming + rain freezing
    snow_prod = (qi_auto + qi_aggr + qc_rime + rain_freeze) * cloud_fraction

    # Column-integrated surface flux (kg/m²/s)
    precip_rain = jnp.sum(rain_prod * layer_mass)
    precip_snow = jnp.sum(snow_prod * layer_mass)

    rain_flux = rain_prod * layer_mass
    snow_flux = snow_prod * layer_mass
    
    # Create output structures
    tendencies = MicrophysicsTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dqcdt=dqcdt,
        dqidt=dqidt,
        dqrdt=dqrdt,
        dqsdt=dqsdt
    )
    
    state = MicrophysicsState(
        rain_flux=rain_flux,
        snow_flux=snow_flux,
        qc_in_cloud=qc_in_cloud,
        qi_in_cloud=qi_in_cloud,
        autoconv_rate=qc_auto,
        accretion_rate=qc_accr,
        melting_rate=snow_melt,
        freezing_rate=rain_freeze,
        precip_rain=jnp.array(precip_rain),
        precip_snow=jnp.array(precip_snow)
    )

    return tendencies, state


def _qsat_water(pressure: jnp.ndarray, temperature: jnp.ndarray):
    """Saturation specific humidity over water + the vapor pressure es.

    Uses the same Tetens form as :func:`sundqvist.saturation_vapor_pressure_water`
    so the rain-evaporation step is consistent with the condensation step.
    The conversion from ``es`` to ``qs`` follows the standard mixing-ratio
    formula ``qs = ε·es/(p - (1-ε)·es)`` (equivalent to ICON's
    ``zqsw = uaw/(p - vtmpc1·uaw)`` after expanding ``uaw = ε·es``).
    Returns ``(qsw, esw_pa)``.
    """
    t_c = temperature - tmelt
    es = 610.78 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    es_safe = jnp.minimum(es, 0.5 * pressure)
    qsw = eps * es_safe / jnp.maximum(pressure - (1.0 - eps) * es_safe, 1.0)
    return qsw, es_safe


def cloud_microphysics_column_sweep(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: Optional[MicrophysicsParameters] = None,
) -> Tuple[MicrophysicsTendencies, MicrophysicsState]:
    """ICON ``mo_cloud.f90`` column-sweep microphysics — flux propagation.

    Faithful port of the column structure of ICON's ``mo_cloud.f90``
    (lines 260-1080 in
    ``/home/dwatsonparris/atm_phy_echam/mo_cloud.f90``). Treats rain
    (``zrfl``) and snow (``zsfl``) as **downward fluxes** that propagate
    top-to-bottom through the column within a single timestep:

    1. ``zrfl`` and ``zsfl`` start at 0 at TOA.
    2. For each layer (top → bottom):

       a. **Snow melt** for incoming flux at ``T > 273 K``: convert
          ``zsfl`` → ``zrfl``. Per ICON ``mo_cloud.f90:319-323``.
       b. **Local microphysics**: autoconversion / ice-autoconversion /
          accretion / riming / aggregation produce mass into rain or
          snow, drawing from local ``qc`` / ``qi``.
       c. **Update outgoing flux**:
          ``zrfl_out = zrfl_in + (autoconv + accr) · pmref / dt``
          ``zsfl_out = zsfl_in + (ice_auto + aggr + rime) · pmref / dt``

    3. Bottom-of-column ``zrfl`` / ``zsfl`` become the surface
       precipitation flux (``state.precip_rain`` / ``state.precip_snow``).

    Rain evaporation in subsaturated layers (``mo_cloud.f90:397-435``,
    Rotstayn 1997) is included: as falling rain enters a layer with
    ``q < qsw``, a fraction evaporates into vapour, cooling the layer
    and reducing the propagating ``zrfl``. The implementation tracks
    a ``zclcpre`` carry — the precipitating-area cloud fraction — using
    ICON's flux-mass-weighted recipe (``mo_cloud.f90:1006-1013``).

    What's INTENTIONALLY MISSING from this port:

    * **Snow sublimation in subsaturated layers** (``mo_cloud.f90``
      332-393, Lin et al. 1983). Same structural shape as rain evap;
      tracked as a separate add when stability data justifies it.
    * **Rain freezing** below ``cthomi`` and the **Bergeron-Findeisen**
      ice-from-supercooled-water process.

    The propagating flux means rain produced in a high cloud actually
    exits the column at the surface in the SAME step (rather than being
    a per-level diagnostic that doesn't interact with the layer below).
    With rain evap turned on, falling rain that enters dry air below
    cloud-base re-evaporates, moistening the lower troposphere — this
    is the standard mechanism by which the column moisture cycle closes
    in ECHAM/ICON.

    Same signature as :func:`cloud_microphysics` so call sites can swap.
    """
    if config is None:
        config = MicrophysicsParameters.default()

    nlev = temperature.shape[0]

    # Pre-compute per-level diagnostic quantities used inside the scan.
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon, cloud_water / cloud_fraction, 0.0,
    )
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon, cloud_ice / cloud_fraction, 0.0,
    )
    pmref = air_density * layer_thickness     # kg/m² per layer

    # Per-level autoconversion / accretion sources (depend only on local
    # state, vectorised before the scan). These produce condensate that
    # we add into the falling flux as we sweep down.
    qc_auto = autoconversion(
        cloud_water, cloud_fraction, air_density, droplet_number, dt, config,
    )
    qi_auto = ice_autoconversion(cloud_ice, temperature, cloud_fraction, dt, config)

    # Phase weights for the latent-heat update from snow melting.
    zlsdcp = alhs / cp
    zlvdcp = alhc / cp

    def step(carry, level_inputs):
        zrfl, zsfl, zclcpre = carry
        (T, q, p, qc, qi, cf, rho, dz,
         qcic, qiic, qcaut, qiaut, mref) = level_inputs

        # ---------- (a) snow melt at T > 273 K ----------
        # ICON ``mo_cloud.f90:319-323``:
        #
        #     zcons   = (mref/dt) / (Ls/cp - Lv/cp)
        #     zsnmlt  = MIN(0.99 * zsfl, zcons * MAX(0, T - tmelt))
        #     zrfl   += zsnmlt
        #     zsfl   -= zsnmlt
        #     zsmlt   = zsnmlt / mref * dt   (per-step mass change)
        zcons = (mref / dt) / jnp.maximum(zlsdcp - zlvdcp, 1e-6)
        ztdif = jnp.maximum(0.0, T - tmelt)
        zsnmlt = jnp.minimum(0.99 * zsfl, zcons * ztdif)
        zrfl = zrfl + zsnmlt
        zsfl = zsfl - zsnmlt

        # Latent heat absorbed by the melting snow cools the air.
        # zsmlt (kg/kg/s) → dT/dt = -(Lf/cp)·zsmlt. (zsnmlt is a per-step
        # mass-per-area in kg/m², dividing by mref gives kg/kg per step,
        # dividing by dt gives the rate.)
        zsmlt_rate = zsnmlt / jnp.maximum(mref, config.epsilon) / dt
        dTdt_melt = -(zlsdcp - zlvdcp) * zsmlt_rate

        # ---------- (b) local autoconversion / accretion / riming ----------
        # Cloud water → rain: autoconversion (Beheng / KK2000) + accretion
        # by the falling rain flux (proportional to zrfl).
        rain_accr = jnp.where(
            qc > config.epsilon,
            config.ccracl * zrfl * qc / jnp.maximum(rho, config.epsilon),
            0.0,
        )
        # Cloud water → snow: riming by falling snow.
        snow_rime = jnp.where(
            (qc > config.epsilon) & (T < tmelt),
            config.ccracl * zsfl * qc / jnp.maximum(rho, config.epsilon),
            0.0,
        )
        # Cloud ice → snow: autoconversion + aggregation by falling snow.
        snow_aggr = jnp.where(
            qi > config.epsilon,
            config.ccracl * zsfl * qi / jnp.maximum(rho, config.epsilon),
            0.0,
        )

        # Cloud water / ice tendencies (kg/kg/s).
        dqcdt_local = -(qcaut + rain_accr + snow_rime)
        dqidt_local = -(qiaut + snow_aggr)

        # Latent-heat release from riming (liquid → ice via collection by
        # falling snow). ICON ``mo_cloud.f90:949-952`` carries the riming
        # contribution into the temperature tendency through ``alhf*zsacl``.
        # Autoconversion (qc → rain) and ice autoconv (qi → snow) are
        # phase-preserving and release no latent heat. Snow melt is handled
        # in step (a) above. Rain evap / snow sublim / rain freezing are
        # NOT simulated by this minimal port (see module docstring), so
        # their latent-heat terms are absent here.
        zlfdcp = zlsdcp - zlvdcp        # alhf / cp
        dTdt_rime = zlfdcp * snow_rime  # K/s (snow_rime already in kg/kg/s)

        # ---------- (c-pre) Rotstayn (1997) rain evaporation ----------
        # ICON ``mo_cloud.f90:397-435``. ``zsusatw`` is the (negative)
        # sub-saturation w.r.t. liquid; ``zast+zbst`` are Rotstayn's
        # thermodynamic + vapour-diffusion coefficients; the prefactor
        # ``870`` and the ``zrfl/zclcpre`` exponent ``0.61`` come from
        # the Marshall-Palmer rain spectrum integral. ``zclcpre`` is
        # the propagating precipitating-cloud-area fraction carried
        # from above; we update it after the local rain/snow sources
        # using ICON's flux-mass-weighted recipe (line 1006).
        qsw, esw = _qsat_water(p, T)
        zclcpre_safe = jnp.maximum(zclcpre, config.epsilon)
        zsusatw = jnp.minimum(q / jnp.maximum(qsw, config.epsilon) - 1.0, 0.0)
        zdv = 2.21 / jnp.maximum(p, config.epsilon)
        zast = (
            alhc * (alhc / (rv * jnp.maximum(T, 1.0)) - 1.0)
            / jnp.maximum(T, 1.0) / 0.024
        )
        zbst = T / jnp.maximum(zdv * esw, config.epsilon)
        zthermo = jnp.maximum(zast + zbst, config.epsilon)
        zrfl_in_cf = zrfl / zclcpre_safe
        zqrho_sqrt = jnp.sqrt(jnp.maximum(1.3 / jnp.maximum(rho, config.epsilon), 0.0))
        # Rate of evap per unit grid mass per second (kg/kg/s, negative
        # because zsusatw ≤ 0).
        zzepr_rate = (
            870.0 * zsusatw * jnp.power(jnp.maximum(zrfl_in_cf, 0.0), 0.61)
            * zqrho_sqrt / jnp.sqrt(1.3) / zthermo
        )
        # Convert to per-step grid-mean evap (kg/kg). zevp ≥ 0.
        zevp_unbounded = -zzepr_rate * dt * zclcpre
        # Cap by Rotstayn's per-step rain budget: can't evaporate more
        # rain than what's actually falling through the cloud area.
        zevp_max_rain = zrfl / jnp.maximum(mref, config.epsilon) * dt
        zevp_max_subsat = jnp.maximum(0.99 * (qsw - q), 0.0)
        zevp = jnp.minimum(zevp_unbounded, zevp_max_subsat)
        zevp = jnp.maximum(zevp, 0.0)
        zevp = jnp.minimum(zevp, zevp_max_rain)
        # Apply only when there's a propagating rain flux + a precipitating
        # area to evaporate from.
        zevp = jnp.where((zrfl > config.epsilon) & (zclcpre > config.epsilon), zevp, 0.0)

        dqdt_evap = zevp / dt                                         # kg/kg/s
        dTdt_evap = -zlvdcp * dqdt_evap                               # K/s (cooling)
        rain_evap_flux = zevp * mref / dt                             # kg/m²/s

        # ---------- (c) update fluxes for the next layer ----------
        # ICON ``mo_cloud.f90:984-985 / 1030-1031``:
        #     zzdrr = (zraut + zrac1 + zrac2) * mref / dt
        #     zzdrs = (zspr + zsacl)         * mref / dt
        #     zrfl += zzdrr - zevp * mref / dt
        #     zsfl += zzdrs - zsub * mref / dt
        # (zsub = 0 in this port — snow sublim is a follow-up.)
        rain_source = (qcaut + rain_accr) * mref          # kg/m²/s
        snow_source = (qiaut + snow_aggr + snow_rime) * mref
        # Clamp to ≥ 0 to defend against float round-off when rain
        # evap consumes essentially all of the incoming flux.
        zrfl_out = jnp.maximum(zrfl + rain_source - rain_evap_flux, 0.0)
        zsfl_out = jnp.maximum(zsfl + snow_source, 0.0)

        # ---------- (d) update zclcpre carry per ICON 1006-1013 ----------
        zpretot = zrfl + zsfl                                       # incoming
        zpredel = rain_source + snow_source                         # local add
        zpresum = zpretot + zpredel
        # Flux-mass-weighted blend of incoming-area and local-cf.
        zclcpre1 = jnp.where(
            zpresum > config.epsilon,
            (cf * zpredel + zclcpre * zpretot) / jnp.maximum(zpresum, config.epsilon),
            0.0,
        )
        zclcpre1 = jnp.clip(jnp.maximum(zclcpre, zclcpre1), 0.0, 1.0)
        zclcpre_out = jnp.where(zpresum > config.epsilon, zclcpre1, 0.0)

        # Per-step rates the model integrates (q_new = q + dqdt*dt etc).
        out = (
            dTdt_melt + dTdt_rime + dTdt_evap,    # dT/dt from this layer (K/s)
            dqdt_evap,                            # dq/dt (vapour gain)
            dqcdt_local,
            dqidt_local,
            rain_source,                          # local rain source (kg/m²/s)
            snow_source,
        )
        return (zrfl_out, zsfl_out, zclcpre_out), out

    level_inputs = (
        temperature, specific_humidity, pressure,
        cloud_water, cloud_ice, cloud_fraction,
        air_density, layer_thickness,
        qc_in_cloud, qi_in_cloud,
        qc_auto, qi_auto,
        pmref,
    )
    (zrfl_surface, zsfl_surface, _zclcpre_surface), per_level_out = jax.lax.scan(
        step,
        (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        level_inputs,
    )
    dtedt, dqdt, dqcdt, dqidt, rain_flux, snow_flux = per_level_out

    tendencies = MicrophysicsTendencies(
        dtedt=dtedt, dqdt=dqdt, dqcdt=dqcdt, dqidt=dqidt,
        dqrdt=jnp.zeros(nlev),  # rain/snow live in the flux, not state
        dqsdt=jnp.zeros(nlev),
    )
    state = MicrophysicsState(
        rain_flux=rain_flux, snow_flux=snow_flux,
        qc_in_cloud=qc_in_cloud, qi_in_cloud=qi_in_cloud,
        autoconv_rate=qc_auto, accretion_rate=qc_auto * 0,
        melting_rate=qc_auto * 0, freezing_rate=qc_auto * 0,
        precip_rain=zrfl_surface, precip_snow=zsfl_surface,
    )
    return tendencies, state