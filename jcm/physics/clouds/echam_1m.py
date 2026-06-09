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

import jcm.constants as c


@tree_math.struct
class MicrophysicsParameters:
    """Configuration parameters for cloud microphysics"""
    
    # Autoconversion parameters
    ccraut: float        # Critical cloud water for autoconversion (kg/kg)
    ccracl: float        # Accretion coefficient (cloud to rain)
    cauloc: float        # ECHAM ``zrac2`` local-rain accretion enhancement.
                         # 0.0 is the ECHAM6.3 default (zrac2 disabled); raise to
                         # let the in-step autoconverted rain ALSO collect qc
                         # from its source layer (ECHAM mo_cloud.f90:791).
    clmin: float         # Lower bound on ``zauloc = clip(cauloc·dz/5000, clmin, clmax)``
    clmax: float         # Upper bound on ``zauloc`` (ECHAM6.3: 0.0 / 0.5).
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

    # Mixed-phase split for the saturation-adjustment step. Below
    # ``t_mix_min`` condensate becomes 100% ice; above ``t_mix_max`` it
    # becomes 100% liquid. In between, the partition weighs liquid by
    # ``(T - t_mix_min)/(t_mix_max - t_mix_min)``. These match the
    # defaults previously held on ``CloudParameters``; the values live on
    # MicrophysicsParameters now because cuadjtq-style condensation is
    # part of the merged column-sweep cloud routine (see
    # :func:`cloud_microphysics_column_sweep`).
    t_mix_min: float
    t_mix_max: float

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
    def default(cls, ccraut=15.0, ccracl=6.0, cauloc=0.0, clmin=0.0, clmax=0.5,
                 ceffmin=10.0, ceffmax=150.0, cn0s=3.0e6,
                 crhosno=100.0, cvtfall=3.29, cthomi=233.15, csecfrl=0.1, ccollec=0.7,
                 ccollei=0.3, tau_melt=100.0, tau_freeze=100.0, cevaprain=1.0e-3,
                 cevapsnow=5.0e-4, vt_ice=0.1, vt_snow_a=8.8, vt_snow_b=0.15,
                 vt_rain_a=386.0, vt_rain_b=0.67, base_cdnc=100.0e6,
                 t_mix_min=238.15, t_mix_max=273.15,
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
            clmin=jnp.array(clmin),
            clmax=jnp.array(clmax),
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
            t_mix_min=jnp.array(t_mix_min),
            t_mix_max=jnp.array(t_mix_max),
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
    volume_per_droplet = cloud_water_density / (droplet_density + config.epsilon) / c.rhow  # m³
    
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
    t_celsius = temperature - c.tmelt
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
        t_celsius = temperature - c.tmelt
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
    dt_freeze = c.tmelt - temperature

    # Melting rate (T > 0°C)
    melt_rate = jnp.where(
        temperature > c.tmelt,
        snow * (temperature - c.tmelt) / (config.tau_melt * 10.0),  # Scaled by temp
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
    
    # Temperature: latent heat effects. ECHAM's thermodynamic tendency does
    # not include rain/snow production from autoconversion or aggregation;
    # those are phase-preserving condensate-to-precip conversions. Only
    # evaporation/sublimation, melt/freeze, and liquid riming by snow change
    # phase enthalpy here.
    dtedt = (
        - c.alhc / c.cpd * rain_evap
        - c.alhs / c.cpd * snow_sublim
        - c.alhf / c.cpd * snow_melt
        + c.alhf / c.cpd * rain_freeze
        + c.alhf / c.cpd * qc_rime
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

    # Rain/snow production rates are already grid-mean kg/kg/s. The
    # autoconversion/accretion helpers convert from in-cloud to grid-mean
    # internally, and melt/freeze rates operate on grid-mean precip stores.
    rain_prod = qc_auto + qc_accr + snow_melt
    snow_prod = qi_auto + qi_aggr + qc_rime + rain_freeze

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


def _saturation_adjustment_layer(
    T: jnp.ndarray,
    q: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    p: jnp.ndarray,
    config: MicrophysicsParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-layer cuadjtq-style saturation adjustment.

    Ports the two-pass Newton step from ``sundqvist.condensation_evaporation``
    so the column-sweep microphysics can do its own condensation
    locally — the way ECHAM's ``mo_cloud.f90`` lines 696-784 do. With
    condensation inside the sweep, the autoconv / accretion / rain-evap
    that follow at the same level operate on post-condensation
    ``(T', q', qc', qi')``, which closes the rain-evap ↔ re-condensation
    loop within a single ``dt`` and breaks the two-step feedback that
    forced PR #458 to revert the column-sweep variant.

    Args:
        T: Temperature [K] (per-layer scalar inside the scan).
        q: Specific humidity [kg/kg].
        qc, qi: Cloud water and ice mixing ratios [kg/kg].
        p: Pressure at the layer [Pa].
        config: :class:`MicrophysicsParameters` — only ``t_mix_min`` /
            ``t_mix_max`` are read here; everything else is unused.

    Returns:
        ``(dT, dq, dqc, dqi)`` — per-step absolute increments
        (kg/kg, K) over ``dt``. Add to the input fields to get the
        post-adjustment values:

            T_post = T + dT
            q_post = q + dq
            qc_post = max(qc + dqc, 0)
            qi_post = max(qi + dqi, 0)

        Positive ``dqc`` / ``dqi`` indicate condensation onto cloud
        condensate; negative values are evaporation of cloud water/ice.
        ``dq = -(dqc + dqi)`` by construction so the column-integrated
        vapour balance closes.

    """
    # Imported here rather than at module top to keep the dependency
    # explicit and to avoid pulling sundqvist into the module-load path
    # of every code path that touches echam_1m.
    from jcm.physics.clouds.sundqvist import _qs_and_dqs_dt

    weight_liquid = jnp.clip(
        (T - config.t_mix_min)
        / jnp.maximum(config.t_mix_max - config.t_mix_min, 1e-3),
        0.0, 1.0,
    )
    L_eff = weight_liquid * c.alhc + (1.0 - weight_liquid) * c.alhs
    L_cp = L_eff / c.cpd

    # ---- Pass 1: linearised Newton step (matches sundqvist) ----
    qs, dqs_dt = _qs_and_dqs_dt(p, T)
    q_excess = q - qs
    cond1 = q_excess / jnp.maximum(1.0 + L_cp * dqs_dt, 1e-3)
    total_cloud = qc + qi
    cond1 = jnp.maximum(cond1, -total_cloud)
    cond1 = jnp.minimum(cond1, jnp.maximum(q, 0.0))

    # ---- Pass 2: cleanup any residual super-saturation above 1% qs ----
    T_p1 = T + L_cp * cond1
    q_p1 = q - cond1
    qs_p1, _ = _qs_and_dqs_dt(p, T_p1)
    oversat_tol = 0.01 * qs_p1
    cond2 = jnp.maximum(
        (q_p1 - qs_p1 - oversat_tol) / jnp.maximum(1.0 + L_cp * dqs_dt, 1e-3),
        0.0,
    )
    cond2 = jnp.minimum(cond2, jnp.maximum(q_p1, 0.0))
    cond_total = cond1 + cond2

    # ---- Partition between liquid / ice ----
    safe_total = jnp.where(total_cloud > 0, total_cloud, 1.0)
    qc_frac = jnp.where(total_cloud > 0, qc / safe_total, 0.0)
    qi_frac = jnp.where(total_cloud > 0, qi / safe_total, 0.0)
    L_evap = jnp.where(
        total_cloud > 0,
        (qc * c.alhc + qi * c.alhs) / safe_total,
        L_eff,
    )

    dq = -cond_total
    dqc = jnp.where(
        cond_total > 0,
        weight_liquid * cond_total,
        cond_total * qc_frac,
    )
    dqi = jnp.where(
        cond_total > 0,
        (1.0 - weight_liquid) * cond_total,
        cond_total * qi_frac,
    )
    L_for_dT = jnp.where(cond_total > 0, L_eff, L_evap)
    dT = L_for_dT * cond_total / c.cpd
    return dT, dq, dqc, dqi


def _qsat_water(pressure: jnp.ndarray, temperature: jnp.ndarray):
    """Saturation specific humidity over water + the vapor pressure es.

    Uses the same Tetens form as :func:`sundqvist.saturation_vapor_pressure_water`
    so the rain-evaporation step is consistent with the condensation step.
    The conversion from ``es`` to ``qs`` follows the standard mixing-ratio
    formula ``qs = ε·es/(p - (1-ε)·es)`` (equivalent to ICON's
    ``zqsw = uaw/(p - vtmpc1·uaw)`` after expanding ``uaw = ε·es``).
    Returns ``(qsw, esw_pa)``.
    """
    t_c = temperature - c.tmelt
    es = 610.78 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    es_safe = jnp.minimum(es, 0.5 * pressure)
    qsw = c.eps * es_safe / jnp.maximum(pressure - (1.0 - c.eps) * es_safe, 1.0)
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
    """ECHAM ``mo_cloud.f90`` column-sweep cloud + microphysics routine.

    Faithful port of ICON/ECHAM ``mo_cloud.f90`` lines 260-1080. Treats
    rain (``zrfl``) and snow (``zsfl``) as **downward fluxes** that
    propagate top-to-bottom through the column within a single ``dt``
    and now also does the **per-layer saturation adjustment** (cuadjtq
    Newton step) inside the same column sweep — matching ECHAM's
    structure where condensation, autoconversion, rain evap, and flux
    propagation all live in one routine.

    Per-layer order (top → bottom):

    1. **Snow melt** for incoming flux at ``T > 273 K``: convert
       ``zsfl`` → ``zrfl`` (``mo_cloud.f90:319-323``).
    2. **Saturation adjustment** (``_saturation_adjustment_layer``):
       linearised Newton step on the layer ``(T, q, qc, qi)`` so the
       layer is non-supersaturated *and* any subsaturated cloud
       water/ice evaporates. Mirrors ECHAM ``mo_cloud.f90`` lines
       696-784.
    3. **Microphysics** from the *post-condensation* ``(T', q', qc', qi')``:
       Beheng/KK2000 autoconversion (``qc' → rain``), Lin-style ice
       autoconversion (``qi' → snow``), rain accretion of cloud water,
       snow riming of cloud water (T < ``tmelt``), snow aggregation of
       cloud ice. Accretion / riming / aggregation use ECHAM's
       implicit-Euler form
       ``zrac1 = zxlb·(1 - exp(-ccracl·zxrp1·dt))`` with the
       Marshall-Palmer in-precipitating-area concentration ``zxrp1``
       (mo_cloud.f90:800-877), so per-step depletion is bounded in
       ``[0, qc]`` and can't drive ``qc`` negative even at high
       incoming rain flux.
    4. **Rotstayn (1997) rain evaporation** below cloud, using the
       *post-condensation* ``q'`` so it can't push the layer above
       saturation (``zevp_max_subsat = 0.99·(qs - q')``).
    5. **Flux update** for ``zrfl`` / ``zsfl`` / ``zclcpre`` carry.

    Why no within-step cleanup pass: the 0.99·(qs - q') cap on rain
    evap means the layer cannot be pushed past saturation in step 4,
    so a second saturation-adjustment pass would always be a no-op.
    An earlier draft of this routine ran one anyway as a defensive
    measure; in practice it re-condensed the slight super-saturation
    that rain-evap-cooling produced (qs drops with T → small
    super-saturation appears → cleanup condenses → more autoconv →
    more rain), reigniting the rain-evap ↔ re-condensation feedback
    PR #458 originally caught. The cap alone is sufficient.

    Bottom-of-column ``zrfl`` / ``zsfl`` become the surface precipitation
    flux (``state.precip_rain`` / ``state.precip_snow``).

    The per-layer ``(dT, dq, dqc, dqi)`` returned to the caller pool
    every contribution from steps 1-5 into rate-form tendencies — the
    composable physics integrator applies ``state += dt * tend`` as
    usual.

    What's INTENTIONALLY MISSING from this port:

    * **Snow sublimation in subsaturated layers** (``mo_cloud.f90``
      332-393, Lin et al. 1983). Same structural shape as rain evap;
      tracked as a separate add when stability data justifies it.
    * **Rain freezing** below ``cthomi`` and the **Bergeron-Findeisen**
      ice-from-supercooled-water process (covered by the 2M scheme).

    Same signature as :func:`cloud_microphysics` so call sites can swap.
    """
    if config is None:
        config = MicrophysicsParameters.default()

    nlev = temperature.shape[0]
    pmref = air_density * layer_thickness     # kg/m² per layer

    # Phase weights for the latent-heat update from snow melting / riming.
    zlsdcp = c.alhs / c.cpd
    zlvdcp = c.alhc / c.cpd
    zlfdcp = zlsdcp - zlvdcp        # alhf / cp

    def step(carry, level_inputs):
        zrfl, zsfl, zclcpre = carry
        T0, q0, p, qc0, qi0, cf, rho, dz, ndrop, mref = level_inputs

        # ---------- (1) snow melt at T > tmelt ----------
        # ICON ``mo_cloud.f90:319-323``. Uses the input T (pre-condensation)
        # since snow falling INTO this layer melts based on whether the
        # ambient air is above freezing — condensation hasn't run yet.
        zcons = (mref / dt) / jnp.maximum(zlfdcp, 1e-6)
        ztdif = jnp.maximum(0.0, T0 - c.tmelt)
        zsnmlt = jnp.minimum(0.99 * zsfl, zcons * ztdif)
        zrfl = zrfl + zsnmlt
        zsfl = zsfl - zsnmlt
        zsmlt_rate = zsnmlt / jnp.maximum(mref, config.epsilon) / dt
        dTdt_melt = -zlfdcp * zsmlt_rate

        # ---------- (2) pre-microphysics saturation adjustment ----------
        # Two-pass Newton condensation / evaporation on this layer's
        # ``(T0, q0, qc0, qi0)`` — same logic as ECHAM ``mo_cloud.f90``
        # 696-784. Outputs are absolute increments over ``dt``.
        dT_cond_a, dq_cond_a, dqc_cond_a, dqi_cond_a = _saturation_adjustment_layer(
            T0, q0, qc0, qi0, p, config,
        )
        T1 = T0 + dT_cond_a
        q1 = q0 + dq_cond_a
        qc1 = jnp.maximum(qc0 + dqc_cond_a, 0.0)
        qi1 = jnp.maximum(qi0 + dqi_cond_a, 0.0)

        # ---------- (3) microphysics on POST-condensation (T1, q1, qc1, qi1) ----------
        # Mirrors ECHAM ``mo_cloud.f90:795-879``: sequential depletion of
        # in-cloud ``zxlb`` (= qc/cf) and ``zxib`` (= qi/cf) by
        # autoconversion (zraut), accretion of cloud water by falling
        # rain (zrac1), local-rain accretion by the in-step autoconverted
        # rain (zrac2 — only fires when cauloc > 0, ECHAM default 0),
        # snow riming of cloud water (zsacl), and snow aggregation of
        # cloud ice (zsaci). Each step reads the post-previous-depletion
        # zxlb, so accretion sees the qc that autoconv left behind, not
        # the original. The grid-mean rain/snow source going into the
        # falling flux is cf-weighted ECHAM-style:
        #
        #     zrpr = cf · (zraut + zrac2) + zclcstar · zrac1
        #     zspr = cf · (zsaut + zsaci2) + zclcstar · zsaci1
        #
        # where ``zclcstar = min(cf, zclcpre)`` is the precipitating /
        # cloud area intersection (rain can only accrete from the area
        # where it overlaps cloud), and the in-cloud "wind back" via
        # the implicit-Euler ``1 - exp(-rate·dt)`` form bounds per-step
        # depletion in ``[0, zxlb]`` by construction so neither qc nor
        # qi can be driven negative.

        # Density-correction factors used by accretion (zxrp1 needs the
        # *forward* sqrt(rho/1.3)) and rain evap (Rotstayn needs the
        # *inverse* sqrt(1.3/rho)).
        zclcpre_safe = jnp.maximum(zclcpre, config.epsilon)
        zqrho_sqrt = jnp.sqrt(jnp.maximum(rho / 1.3, config.epsilon))
        zqrho_sqrt_inv = jnp.sqrt(jnp.maximum(1.3 / jnp.maximum(rho, config.epsilon), 0.0))
        zxrp1 = jnp.where(
            (zrfl > config.epsilon) & (zclcpre > config.epsilon),
            jnp.power(
                jnp.maximum(zrfl / zclcpre_safe / (12.45 * zqrho_sqrt), 0.0),
                8.0 / 9.0,
            ),
            0.0,
        )
        zxsp1 = jnp.where(
            (zsfl > config.epsilon) & (zclcpre > config.epsilon),
            jnp.power(
                jnp.maximum(zsfl / zclcpre_safe / config.cvtfall, 0.0),
                1.0 / 1.16,
            ),
            0.0,
        )

        # In-cloud values for the cascade. ECHAM works on ``zxlb`` /
        # ``zxib`` which are in-cloud mixing ratios (qc/cf, qi/cf).
        cf_safe = jnp.maximum(cf, config.epsilon)
        cloud_mask = cf > config.epsilon
        zxlb = jnp.where(cloud_mask, qc1 / cf_safe, 0.0)
        zxib = jnp.where(cloud_mask, qi1 / cf_safe, 0.0)
        zclcstar = jnp.minimum(cf, zclcpre)

        # Numerical safety: clamp the exponent in ``1 - exp(-x)``;
        # float32 overflows to denormalised zero past ~50 and gradients
        # through ``exp`` of a huge negative value are unstable.
        def _impl_depletion(arg):
            return 1.0 - jnp.exp(-jnp.minimum(arg, 50.0))

        # (3a) Beheng autoconversion: in-cloud qc → rain. Reuses the
        # standalone helper at cf=1 so the existing implementation owns
        # the rate formula; the returned value is then ``rate * 1 = rate``
        # in kg/kg/s in-cloud. Per-dt depletion = rate * dt.
        qcaut_rate_in_cloud = autoconversion(
            zxlb, jnp.array(1.0), rho, ndrop, dt, config,
        )
        zraut = jnp.minimum(qcaut_rate_in_cloud * dt, zxlb)  # in-cloud kg/kg over dt
        zxlb = zxlb - zraut

        # (3b) Rain accretion of cloud water (zrac1). Reads post-autoconv zxlb.
        zrac1 = zxlb * _impl_depletion(config.ccracl * zxrp1 * dt)
        zxlb = zxlb - zrac1

        # (3c) Snow riming of cloud water (zsacl-style). Only fires when
        # T1 < tmelt — above freezing the collected liquid stays liquid.
        zsacl = jnp.where(
            T1 < c.tmelt,
            zxlb * _impl_depletion(config.ccracl * zxsp1 * dt),
            0.0,
        )
        zxlb = zxlb - zsacl

        # (3d) Local-rain accretion (zrac2). ECHAM ``mo_cloud.f90:860``:
        # ``ztmp2 = -ccracl · zauloc · rho · zraut · dt`` then
        # ``zrac2 = zxlb · (1 - exp(ztmp2))``. ``zauloc`` scales with
        # layer thickness; clipped to ``[clmin, clmax]``. With the
        # ECHAM6.3 default ``cauloc=0`` this is identically zero — the
        # branch is here for parameter completeness and ICON-style
        # tunings that enable it.
        zauloc = jnp.clip(config.cauloc * dz / 5000.0, config.clmin, config.clmax)
        zrac2 = zxlb * _impl_depletion(
            config.ccracl * zauloc * rho * zraut,
        )
        zxlb = zxlb - zrac2

        # (3e) Ice autoconversion (qi → snow) and snow aggregation
        # (qi by falling snow). Sequential like the warm-rain side.
        qiaut_rate_in_cloud = ice_autoconversion(
            zxib, T1, jnp.array(1.0), dt, config,
        )
        zsaut = jnp.minimum(qiaut_rate_in_cloud * dt, zxib)
        zxib = zxib - zsaut
        zsaci = zxib * _impl_depletion(config.ccracl * zxsp1 * dt)
        zxib = zxib - zsaci

        # Convert in-cloud per-dt depletions to grid-mean tendencies
        # (kg/kg/s). qc depletion happens inside the cloud area, so the
        # grid-mean rate is ``cf · sum(depletions) / dt``.
        dqcdt_micro = -cf * (zraut + zrac1 + zsacl + zrac2) / dt
        dqidt_micro = -cf * (zsaut + zsaci) / dt
        # Riming latent heat: liquid → ice via collection by falling snow.
        # Grid-mean rate; uses zsacl which is already in-cloud per-dt.
        dTdt_rime = zlfdcp * cf * zsacl / dt

        # ---------- (4) Rotstayn rain evaporation on POST-condensation q1 ----------
        # ICON ``mo_cloud.f90:397-435``. ``zsusatw`` is the (negative)
        # sub-saturation w.r.t. liquid; ``zast+zbst`` are Rotstayn's
        # thermodynamic + vapour-diffusion coefficients. Using ``q1``
        # (not ``q0``) means rain evap can't push the layer above
        # saturation — the 0.99·(qs - q1) cap is what enforces this and
        # makes the original PR #458 within-step re-condensation pass
        # unnecessary in this version.
        qsw, esw = _qsat_water(p, T1)
        zsusatw = jnp.minimum(q1 / jnp.maximum(qsw, config.epsilon) - 1.0, 0.0)
        zdv = 2.21 / jnp.maximum(p, config.epsilon)
        zast = (
            c.alhc * (c.alhc / (c.rv * jnp.maximum(T1, 1.0)) - 1.0)
            / jnp.maximum(T1, 1.0) / 0.024
        )
        zbst = T1 / jnp.maximum(zdv * esw, config.epsilon)
        zthermo = jnp.maximum(zast + zbst, config.epsilon)
        zrfl_in_cf = zrfl / zclcpre_safe
        # Rotstayn (1997) per-area rate. The density factor here is the
        # *inverse* of the one accretion uses: see ECHAM mo_cloud.f90:415
        # — ``870 * sub * (zrfl/zclcpre)**0.61 * zqrho/cqtmin / zthermo``
        # where ``zqrho = sqrt(1.3/rho)``. Earlier drafts of this routine
        # mistakenly reused the accretion-direction ``sqrt(rho/1.3)``
        # here, which inverted the density dependence (suppressing
        # rain-evap in low-density layers and amplifying it in dense
        # layers — the opposite of physical).
        zzepr_rate = (
            870.0 * zsusatw * jnp.power(jnp.maximum(zrfl_in_cf, 0.0), 0.61)
            * zqrho_sqrt_inv / jnp.sqrt(1.3) / zthermo
        )
        zevp_unbounded = -zzepr_rate * dt * zclcpre
        zevp_max_rain = zrfl / jnp.maximum(mref, config.epsilon) * dt
        zevp_max_subsat = jnp.maximum(0.99 * (qsw - q1), 0.0)
        zevp = jnp.minimum(zevp_unbounded, zevp_max_subsat)
        zevp = jnp.maximum(zevp, 0.0)
        zevp = jnp.minimum(zevp, zevp_max_rain)
        zevp = jnp.where(
            (zrfl > config.epsilon) & (zclcpre > config.epsilon), zevp, 0.0,
        )
        dq_evap = zevp                                                # kg/kg over dt
        dTdt_evap = -zlvdcp * (dq_evap / dt)                          # K/s
        rain_evap_flux = zevp * mref / dt                             # kg/m²/s

        # ---------- (6) flux update ----------
        # ECHAM ``mo_cloud.f90:879`` rain source:
        #   zrpr = cf · (zraut + zrac2)  +  zclcstar · zrac1
        # and analogously for snow (line 950). Rain produced in-cloud
        # (autoconv, local-rain accretion) covers the full cloud area;
        # rain accretion of cloud water by *falling* rain covers only
        # the intersection of cloud area and the incoming precipitating
        # area (zclcstar). All the in-cloud zXXX values are per-dt
        # depletion amounts in kg/kg, so divide by dt to get rate, then
        # multiply by mref to get the kg/m²/s flux into ``zrfl``.
        rain_source = (
            cf * (zraut + zrac2) + zclcstar * zrac1
        ) * mref / dt
        snow_source = (
            cf * (zsaut + zsacl) + zclcstar * zsaci
        ) * mref / dt
        # Clamp to ≥ 0 against float round-off when rain evap consumes
        # essentially all of the incoming flux.
        zrfl_out = jnp.maximum(zrfl + rain_source - rain_evap_flux, 0.0)
        zsfl_out = jnp.maximum(zsfl + snow_source, 0.0)

        # ---------- (7) zclcpre carry update per ICON 1006-1013 ----------
        zpretot = zrfl + zsfl
        zpredel = rain_source + snow_source
        zpresum = zpretot + zpredel
        zclcpre1 = jnp.where(
            zpresum > config.epsilon,
            (cf * zpredel + zclcpre * zpretot) / jnp.maximum(zpresum, config.epsilon),
            0.0,
        )
        zclcpre1 = jnp.clip(jnp.maximum(zclcpre, zclcpre1), 0.0, 1.0)
        zclcpre_out = jnp.where(zpresum > config.epsilon, zclcpre1, 0.0)

        # Pool every contribution into per-step rates (kg/kg/s, K/s) that
        # the composable physics integrator multiplies by dt and adds to
        # the dynamics state. The single condensation pass returns
        # absolute increments over dt, so divide by dt to convert to a
        # rate.
        dTdt = dTdt_melt + dTdt_rime + dTdt_evap + dT_cond_a / dt
        dqdt = (dq_evap / dt) + dq_cond_a / dt
        dqcdt = dqcdt_micro + dqc_cond_a / dt
        dqidt = dqidt_micro + dqi_cond_a / dt

        # ``zraut`` is the in-cloud per-dt autoconversion depletion
        # (kg/kg over dt). Convert to a grid-mean rate (kg/kg/s) for
        # the public ``autoconv_rate`` diagnostic.
        autoconv_rate_diag = cf * zraut / dt
        out = (dTdt, dqdt, dqcdt, dqidt, rain_source, snow_source, autoconv_rate_diag)
        return (zrfl_out, zsfl_out, zclcpre_out), out

    level_inputs = (
        temperature, specific_humidity, pressure,
        cloud_water, cloud_ice, cloud_fraction,
        air_density, layer_thickness, droplet_number, pmref,
    )
    (zrfl_surface, zsfl_surface, _zclcpre_surface), per_level_out = jax.lax.scan(
        step,
        (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        level_inputs,
    )
    dtedt, dqdt, dqcdt, dqidt, rain_flux, snow_flux, autoconv_rate = per_level_out

    tendencies = MicrophysicsTendencies(
        dtedt=dtedt, dqdt=dqdt, dqcdt=dqcdt, dqidt=dqidt,
        dqrdt=jnp.zeros(nlev),  # rain/snow live in the falling flux, not state
        dqsdt=jnp.zeros(nlev),
    )
    # In-cloud qc / qi from the *input* state — preserves the public
    # diagnostic signature; the within-step post-condensation values are
    # local to the scan and not exposed.
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon, cloud_water / cloud_fraction, 0.0,
    )
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon, cloud_ice / cloud_fraction, 0.0,
    )
    state = MicrophysicsState(
        rain_flux=rain_flux, snow_flux=snow_flux,
        qc_in_cloud=qc_in_cloud, qi_in_cloud=qi_in_cloud,
        autoconv_rate=autoconv_rate, accretion_rate=jnp.zeros(nlev),
        melting_rate=jnp.zeros(nlev), freezing_rate=jnp.zeros(nlev),
        precip_rain=zrfl_surface, precip_snow=zsfl_surface,
    )
    return tendencies, state


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class Echam1MMicrophysics(PhysicsTerm):
    """ECHAM 1-moment cloud microphysics as a composable PhysicsTerm.

    Consumes the post-condensation ``cloud_fraction``, ``qc``, ``qi``
    written to the public ``"clouds"`` key by
    :class:`~jcm.physics.clouds.sundqvist.SundqvistCloudFraction` so it
    must be composed downstream of that term. Reads ``cdnc_factor`` from
    the public ``"aerosol"`` key (set by
    :class:`~jcm.physics.aerosol.Macv2SpAerosol`) to apply the Twomey
    indirect effect on droplet number — when the aerosol term is absent,
    falls back to the bare ``base_cdnc`` from the parameters.

    Reads ``pressure_full``, ``air_density``, ``layer_thickness`` from
    the moist-air diagnostics dict and the model timestep from
    ``diagnostics["_dt_seconds"]`` (injected by ``ComposablePhysics``).
    Writes ``precip_rain``, ``precip_snow``, ``droplet_number`` back
    into the public ``"clouds"`` key (preserving the upstream
    ``cloud_fraction`` / ``qc`` / ``qi`` fields).
    """

    name: ClassVar[str] = "echam_1m_microphysics"
    category: ClassVar[str] = "clouds"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "air_density", "layer_thickness",
        "clouds", "aerosol",
    )
    provides: ClassVar[tuple[str, ...]] = ("clouds",)

    def __init__(self, params: MicrophysicsParameters | None = None):
        """Hold the scheme-native :class:`MicrophysicsParameters`."""
        self.params = nnx.Param(
            params or MicrophysicsParameters.default(),
        )

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """``qc`` / ``qi`` are read each step; declared so dynamics carries them."""
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
        """Compute microphysics tendencies + precip/droplet diagnostics."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        air_density = diagnostics["air_density"]
        layer_thickness = diagnostics["layer_thickness"]
        clouds = diagnostics["clouds"]

        qc_interim = clouds.qc
        qi_interim = clouds.qi
        cloud_fraction = clouds.cloud_fraction

        # Twomey effect: aerosol term provides per-column cdnc_factor
        # (validated as a required upstream key at composition time).
        cdnc_factor = diagnostics["aerosol"].cdnc_factor
        cdnc_m3 = (
            jnp.ones_like(state.temperature)
            * params.base_cdnc
            * cdnc_factor[jnp.newaxis, :]
        )
        droplet_number_per_kg = cdnc_m3 / air_density

        # ECHAM ``mo_cloud.f90`` column-sweep: per-layer saturation
        # adjustment + autoconversion / accretion / riming / rain-evap +
        # rain/snow flux propagation, all top-to-bottom in one ``lax.scan``.
        # The condensation step lives inside the sweep (see
        # :func:`_saturation_adjustment_layer`) so the rain-evap that
        # follows can't push the layer past saturation, and the cleanup
        # pass at the end of each layer's step closes any residual
        # supersat within the same ``dt`` — breaking the rain-evap ↔
        # re-condensation feedback that drove PR #458 to revert to the
        # per-level scheme. With this in place, Sundqvist is a pure
        # cloud-fraction diagnostic upstream — see
        # :class:`~jcm.physics.clouds.sundqvist.SundqvistCloudFraction`.
        micro_tend, micro_state = jax.vmap(
            cloud_microphysics_column_sweep,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),
            out_axes=(0, 0),
        )(
            state.temperature, state.specific_humidity, pressure_full,
            qc_interim, qi_interim, cloud_fraction,
            air_density, layer_thickness,
            droplet_number_per_kg, dt, params,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=micro_tend.dtedt.T,
            specific_humidity=micro_tend.dqdt.T,
            tracers={
                "qc": micro_tend.dqcdt.T,
                "qi": micro_tend.dqidt.T,
            },
        )

        clouds = clouds.copy(
            precip_rain=micro_state.precip_rain,
            precip_snow=micro_state.precip_snow,
            droplet_number=cdnc_m3,
        )

        return tendency, {**diagnostics, "clouds": clouds}
