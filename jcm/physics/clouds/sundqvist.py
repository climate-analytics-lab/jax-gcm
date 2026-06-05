"""Shallow cloud scheme for ECHAM physics

This module implements a simplified cloud scheme focusing on:
- Cloud fraction diagnosis based on relative humidity
- Cloud water and ice content
- Basic condensation/evaporation processes

Based on the Lohmann and Roeckner (1996) scheme used in ICON/ECHAM.

Date: 2025-01-10
"""

import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
import tree_math

import jcm.constants as c


@tree_math.struct
class CloudParameters:
    """Configuration parameters for shallow cloud scheme"""

    # Cloud fraction parameters
    crt: float           # Critical relative humidity aloft
    crs: float           # Critical relative humidity near surface
    nex: float           # Exponent for RH threshold profile

    # Stratocumulus inversion enhancement (ECHAM ``mo_cover.f90:219-244``).
    # When the column has a low-level inversion (or strongest-stable
    # layer) at altitudes between ``inversion_z_max`` and the surface,
    # cf at that level is enhanced by ``zsat = csatsc + zgam`` ≤ 1
    # where zgam captures the lapse-rate stability — at a true inversion
    # zgam=0 and zsat=csatsc, boosting the apparent RH that drives cf.
    csatsc: float        # Saturation factor for stratocumulus (0.7 = strong)
    cinv: float          # dT/dz threshold (fraction of dry adiabatic) below
                         # which a layer is considered too unstable to support
                         # the stratocumulus enhancement
    inversion_z_max: float   # Highest altitude for inversion search (m)
    inversion_z_min: float   # Lowest altitude for inversion search (m)

    # Cloud droplet parameters
    ceffmin: float       # Minimum cloud droplet radius (microns)
    ceffmax: float       # Maximum cloud droplet radius (microns)

    # Numerical parameters
    epsilon: float       # Small number for numerical stability

    # Cloud ice temperature thresholds
    t_ice: float         # Temperature below which all cloud is ice (K)
    t_mix_min: float     # Lower bound of mixed phase (K)
    t_mix_max: float     # Upper bound of mixed phase (K)

    @classmethod
    def default(cls, crt=0.75, crs=0.975, nex=2.0,
                 csatsc=0.7, cinv=0.25,
                 inversion_z_max=2000.0, inversion_z_min=500.0,
                 ceffmin=10.0,
                 ceffmax=150.0, epsilon=1.0e-12,
                 t_ice=238.15, t_mix_min=238.15, t_mix_max=273.15) -> 'CloudParameters':
        """Return default cloud parameters.

        Defaults match ECHAM6.3 T63 ``mo_echam_cloud_params.f90``
        (``crt=0.75``, ``crs=0.975``, ``nex=2``, ``csatsc=0.7``,
        ``cinv=0.25``). The inversion altitude range
        (``inversion_z_max=2000`` m, ``inversion_z_min=500`` m)
        replaces ECHAM's ``jbmin`` / ``jbmax`` level indices with
        a portable height-based equivalent (ECHAM derives its level
        indices from the same 2000 m / 500 m thresholds anyway, see
        ``mo_echam_cloud_params.f90:152-162``).
        """
        return cls(
            crt=jnp.array(crt),
            crs=jnp.array(crs),
            nex=jnp.array(nex),
            csatsc=jnp.array(csatsc),
            cinv=jnp.array(cinv),
            inversion_z_max=jnp.array(inversion_z_max),
            inversion_z_min=jnp.array(inversion_z_min),
            ceffmin=jnp.array(ceffmin),
            ceffmax=jnp.array(ceffmax),
            epsilon=jnp.array(epsilon),
            t_ice=jnp.array(t_ice),
            t_mix_min=jnp.array(t_mix_min),
            t_mix_max=jnp.array(t_mix_max)
        )


def critical_relative_humidity(
    pressure: jnp.ndarray,
    surface_pressure: float,
    config: CloudParameters,
) -> jnp.ndarray:
    """ECHAM critical RH profile from ``mo_cover.f90``.

    ECHAM names ``crs`` as the near-surface value and ``crt`` as the
    free-tropospheric value. The exponent uses surface pressure divided by
    full-level pressure, not a linear sigma interpolation.
    """
    pressure_safe = jnp.maximum(pressure, 1.0)
    surface_pressure_safe = jnp.maximum(surface_pressure, 1.0)
    return config.crt + (config.crs - config.crt) * jnp.exp(
        1.0 - (surface_pressure_safe / pressure_safe) ** config.nex
    )


class CloudState(NamedTuple):
    """Cloud state variables"""
    
    cloud_fraction: jnp.ndarray     # Cloud fraction [0-1]
    cloud_water: jnp.ndarray        # Cloud liquid water content (kg/kg)
    cloud_ice: jnp.ndarray          # Cloud ice content (kg/kg)
    rel_humidity: jnp.ndarray       # Relative humidity [0-1]
    
    # Diagnostics
    total_cloud_cover: jnp.ndarray  # Column total cloud cover
    
    
class CloudTendencies(NamedTuple):
    """Tendencies from cloud condensation processes.

    Precipitation is handled by cloud_microphysics, not here.
    """

    dtedt: jnp.ndarray         # Temperature tendency (K/s)
    dqdt: jnp.ndarray          # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray         # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray         # Cloud ice tendency (kg/kg/s)


def saturation_vapor_pressure_water(temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate saturation vapor pressure over water using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)

    """
    t_celsius = temperature - c.tmelt
    return 610.78 * jnp.exp(17.27 * t_celsius / (t_celsius + 237.3))


def saturation_vapor_pressure_ice(temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate saturation vapor pressure over ice using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)

    """
    t_celsius = temperature - c.tmelt
    return 610.78 * jnp.exp(21.87 * t_celsius / (t_celsius + 265.5))


def saturation_specific_humidity(
    pressure: jnp.ndarray, 
    temperature: jnp.ndarray
) -> jnp.ndarray:
    """Calculate saturation specific humidity
    
    Args:
        pressure: Pressure (Pa)
        temperature: Temperature (K)
        
    Returns:
        Saturation specific humidity (kg/kg)

    """
    # Use appropriate saturation vapor pressure based on temperature
    es_water = saturation_vapor_pressure_water(temperature)
    es_ice = saturation_vapor_pressure_ice(temperature)
    
    # Blend between ice and water saturation in mixed phase region
    # Linear interpolation between t_ice and tmelt
    weight = jnp.clip((temperature - 238.15) / (c.tmelt - 238.15), 0.0, 1.0)
    es = weight * es_water + (1.0 - weight) * es_ice

    # Convert to saturation specific humidity
    # Cap es < pressure so denominator stays positive under extreme T
    es_safe = jnp.minimum(es, 0.99 * jnp.maximum(pressure, 1.0))
    qs = c.eps * es_safe / jnp.maximum(pressure - es_safe * (1.0 - c.eps), 1.0)
    return jnp.clip(qs, 0.0, 0.5)


def _stratocumulus_zsat(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    config: CloudParameters,
) -> jnp.ndarray:
    """Per-layer stratocumulus saturation factor ``zsat`` ∈ (0, 1].

    Ports the ECHAM ``mo_cover.f90:160-244`` low-level inversion
    enhancement: for each column we find the level with the most
    inversion-like lapse rate (``zdtdz`` closest to 0 / most positive)
    inside the boundary layer (between ``inversion_z_min`` and
    ``inversion_z_max`` above the surface), provided it exceeds the
    ECHAM stability threshold ``-cinv·g/cp``. At that single level
    only, ``zsat = min(1, csatsc + max(0, -dT/dz · cp/g))``; everywhere
    else ``zsat = 1`` (no enhancement). Multiplying ``q/qsat`` by
    ``1/zsat`` boosts the apparent RH that drives cf, which is how
    ECHAM injects extra cloud cover at the BL-top inversion that
    persistent stratocumulus decks live on.

    Inputs are single-column arrays of shape ``(nlev,)`` in physics
    convention (level=0 TOA, level=N-1 surface).

    Returns:
        ``zsat`` of shape ``(nlev,)`` — multiply ``qsat`` by this in
        the cf formula.

    """
    nlev = temperature.shape[0]

    # Approximate height per layer using hydrostatic balance from the
    # surface up: ``dz_k = (R_d * T_k / g) * ln(p_lower/p_upper)``.
    # In physics ordering, lower (higher pressure) is at level k+1 and
    # upper (lower pressure) is at level k, so for each layer we use
    # the layer below as the reference. Column-cumulative sum from the
    # surface gives height above surface.
    p_safe = jnp.maximum(pressure, 1.0)
    # ``log(p_below / p_above)`` between adjacent levels (k+1 below, k above).
    # Shape (nlev-1,). Layer thickness assigned to the upper level k.
    log_ratio = jnp.log(p_safe[1:] / p_safe[:-1])  # +ve when going up
    T_avg = 0.5 * (temperature[:-1] + temperature[1:])
    dz_layer = c.rd * T_avg / c.grav * log_ratio       # (nlev-1,), m
    # height above surface at each full level: 0 at surface, accumulating up
    z_full = jnp.concatenate([
        jnp.cumsum(dz_layer[::-1])[::-1],   # height of each upper-level w.r.t. surface
        jnp.zeros(1),                       # surface (level=N-1) has z=0
    ])

    # dT/dz at each interior level k: between level k and k+1 (the layer
    # immediately below k). Positive = inversion (T rises with height),
    # negative = normal lapse.
    dT = temperature[:-1] - temperature[1:]              # (nlev-1,)
    dz = jnp.maximum(dz_layer, 1.0)                      # avoid /0
    dTdz_layer = dT / dz                                 # (nlev-1,) at upper level k
    # Pad to (nlev,): no zdtdz defined at the surface (no layer below it).
    dTdz = jnp.concatenate([dTdz_layer, jnp.zeros(1)])

    # ECHAM's ``zdtdz = MIN(0, zdtdz)`` clip — clips inversions (zdtdz>0)
    # to 0, leaving normal lapses unchanged. The argmax then finds the
    # level with the LEAST-NEGATIVE lapse, which is the BL-top inversion
    # if there is one (clip→0 is the maximum value), otherwise the most
    # stable lapse near the surface.
    dTdz_clipped = jnp.minimum(dTdz, 0.0)

    # Mask: only consider levels in the BL altitude range AND whose
    # zdtdz exceeds the ECHAM stability threshold ``-cinv*g/cp``
    # (otherwise the layer is too unstable to sustain stratocumulus).
    # Use the SAME ``-cinv*g/cp`` initial value ECHAM seeds ``zdtmin``
    # with, so any ``dTdz_clipped > -cinv*g/cp`` qualifies.
    dtdz_threshold = -config.cinv * c.grav / c.cpd
    in_bl = (z_full >= config.inversion_z_min) & (z_full <= config.inversion_z_max)
    valid = in_bl & (dTdz_clipped > dtdz_threshold)
    # Set invalid levels to a strongly-negative sentinel so argmax skips them.
    masked = jnp.where(valid, dTdz_clipped, -1e10)
    knvb = jnp.argmax(masked)
    has_inversion = jnp.any(valid)

    # zgam = max(0, -zdtdz · cp/g). At a TRUE inversion (zdtdz > 0),
    # the MIN(0, zdtdz) clip already drove dTdz_clipped to 0 → zgam = 0
    # → zsat = csatsc (full enhancement). At a "stable but not inverted"
    # layer (zdtdz slightly negative), zgam > 0 → zsat = csatsc + zgam,
    # weakening the enhancement until ``zsat = 1`` and it has no effect.
    zgam_at_knvb = jnp.maximum(-dTdz_clipped[knvb] * c.cpd / c.grav, 0.0)
    zsat_value = jnp.where(
        has_inversion,
        jnp.minimum(1.0, config.csatsc + zgam_at_knvb),
        1.0,
    )

    # Build zsat array: 1 everywhere, zsat_value at knvb only.
    zsat = jnp.ones(nlev)
    zsat = zsat.at[knvb].set(zsat_value)
    return zsat


def calculate_cloud_fraction(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    surface_pressure: float,
    config: CloudParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Diagnose cloud fraction following ECHAM ``mo_cover.f90``.

    Implements the full ECHAM scheme:

    1. ``rhc = crt + (crs - crt)·exp(1 - (p_surf/p)^nex)`` (crit RH profile,
       :func:`critical_relative_humidity`)
    2. Stratocumulus inversion enhancement at the BL-top level (when one
       is present in the column): ``zsat = min(1, csatsc + zgam)`` where
       ``zgam`` captures lapse-rate stability — see
       :func:`_stratocumulus_zsat`. zsat = 1 elsewhere.
    3. ``zqr = q / (qsat · zsat)`` — apparent RH after the inversion boost
    4. ``b₀ = (zqr - rhc) / (1 - rhc)``, clipped to ``[0, 1]``
    5. ``cc = 1 - sqrt(1 - b₀)``

    Returns the diagnosed cloud fraction and a *grid-mean* relative
    humidity ``q/qsat`` (NOT clipped to ≤1; supersaturated cells carry
    RH > 1 so downstream code can act on the actual super-saturation
    rather than seeing a saturated diagnostic).

    Args:
        temperature: Temperature (K) — single column, shape (nlev,).
        specific_humidity: Specific humidity (kg/kg) — shape (nlev,).
        pressure: Full-level pressure (Pa) — shape (nlev,).
        surface_pressure: Surface pressure (Pa) — scalar.
        config: Cloud configuration (``crt``, ``crs``, ``nex``,
            ``csatsc``, ``cinv``, ``inversion_z_min/max``).

    Returns:
        Tuple of ``(cloud_fraction, relative_humidity)`` of shape
        ``(nlev,)``.

    """
    qs = saturation_specific_humidity(pressure, temperature)

    # Diagnostic relative humidity — NOT clipped at 1. ECHAM uses
    # ``zqr = q/(qsat·zsat)`` directly without clipping; super-saturated
    # cells naturally drive ``b₀ > 1`` which gets clipped to 1 below
    # (giving ``cc = 1``). Clipping RH itself loses information that
    # callers (e.g. downstream microphysics) may want to act on.
    rel_humidity = specific_humidity / (qs + config.epsilon)

    rhc = critical_relative_humidity(pressure, surface_pressure, config)

    # Stratocumulus inversion enhancement (1 everywhere except at BL-top
    # inversion where it drops to ``csatsc`` ≤ 1, boosting ``zqr``).
    zsat = _stratocumulus_zsat(temperature, pressure, config)
    zqr = specific_humidity / (qs * zsat + config.epsilon)

    b0 = (zqr - rhc) / (1.0 - rhc + config.epsilon)
    b0 = jnp.clip(b0, 0.0, 1.0)

    # Cloud fraction: cc = 1 - sqrt(1 - b0). Guard sqrt against b0 == 1
    # via the double-where pattern so ``jax.grad`` doesn't pick up a
    # 0*inf from d(sqrt)/dx at 0.
    sqrt_arg_raw = 1.0 - b0
    sqrt_arg_safe = jnp.where(sqrt_arg_raw > 0.0, sqrt_arg_raw, 1.0)
    cloud_fraction = jnp.where(
        sqrt_arg_raw > 0.0,
        1.0 - jnp.sqrt(sqrt_arg_safe),
        1.0,                     # b0 >= 1 → cc = 1
    )

    # Apply minimum cloud fraction threshold (matches ECHAM convention).
    cloud_fraction = jnp.where(cloud_fraction < 0.01, 0.0, cloud_fraction)

    return cloud_fraction, rel_humidity


def partition_cloud_phase(
    temperature: jnp.ndarray,
    total_cloud_water: jnp.ndarray,
    config: CloudParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Partition cloud water between liquid and ice phases
    
    Args:
        temperature: Temperature (K)
        total_cloud_water: Total cloud condensate (kg/kg)
        config: Cloud configuration
        
    Returns:
        Tuple of (cloud_liquid, cloud_ice)

    """
    # Calculate ice fraction based on temperature
    # All ice below t_ice, all liquid above tmelt
    # Linear transition in between
    ice_frac = jnp.clip(
        (config.t_mix_max - temperature) / (config.t_mix_max - config.t_mix_min),
        0.0, 1.0
    )
    
    # Partition cloud water
    cloud_ice = ice_frac * total_cloud_water
    cloud_liquid = (1.0 - ice_frac) * total_cloud_water
    
    return cloud_liquid, cloud_ice


def _qs_and_dqs_dt(
    pressure: jnp.ndarray,
    temperature: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Saturation specific humidity and its temperature derivative.

    Closed-form derivative of :func:`saturation_specific_humidity` for
    the mixed-phase Tetens-style formulation, so the Newton step is
    bit-reproducible under JIT without finite-difference noise. Mirrors
    what ECHAM's ``ua / dua`` lookup tables provide.
    """
    es_water = saturation_vapor_pressure_water(temperature)
    es_ice = saturation_vapor_pressure_ice(temperature)
    weight = jnp.clip((temperature - 238.15) / (c.tmelt - 238.15), 0.0, 1.0)
    es = weight * es_water + (1.0 - weight) * es_ice

    p_safe = jnp.maximum(pressure, 1.0)
    es_safe = jnp.minimum(es, 0.99 * p_safe)
    denom = jnp.maximum(p_safe - es_safe * (1.0 - c.eps), 1.0)
    qs = c.eps * es_safe / denom

    # Tetens d(es)/dT — same coefficients used in
    # ``saturation_vapor_pressure_water`` / ``..._ice``.
    a_water, c_water = 17.27, 237.3
    a_ice, c_ice = 21.875, 265.5
    tc = temperature - c.tmelt
    des_dt_water = es_water * a_water * c_water / jnp.maximum(
        (tc + c_water) ** 2, 1e-3,
    )
    des_dt_ice = es_ice * a_ice * c_ice / jnp.maximum(
        (tc + c_ice) ** 2, 1e-3,
    )
    des_dt = weight * des_dt_water + (1.0 - weight) * des_dt_ice
    dqs_dt = c.eps * p_safe * des_dt / denom ** 2
    return qs, dqs_dt


def condensation_evaporation(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    pressure: jnp.ndarray,
    dt: float,
    config: CloudParameters,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linearised-Newton condensation / evaporation step.

    Faithful port of the per-cell condensation block in ECHAM
    ``mo_cloud.f90`` (lines 696-784 of echam6.3). The previous
    implementation used the instantaneous ``cond = (q - q_s)/dt``
    adjustment that ignores the warming feedback; on highly super-
    saturated columns it released ``L · (q - q_s)/cp`` of latent heat
    per step (60+ K at 100 % supersat), driving the per-level heating
    spike documented in PR #458.

    Newton step::

        cond = (q - q_s(T)) / (1 + L/cp · dq_s/dT)

    The ``1 + L/cp · dq_s/dT`` denominator is the warming-feedback
    damper: a parcel that condenses ``cond`` warms by ``L·cond/cp``,
    which raises ``q_s`` by ``dq_s/dT · L·cond/cp``. The implicit
    equation ``q - cond = q_s(T + L·cond/cp)`` linearised around T
    solves to the Newton form above. In the warm troposphere
    ``L/cp · dq_s/dT ≈ 5`` so the per-step heating drops ~6× compared
    to the bare formula.

    We act on the WHOLE grid (no cloud-fraction weighting). ECHAM
    weights pass 1 by ``zclcaux`` then runs a grid-box-wide pass-2
    cleanup; the net effect for our microphysics chain is closer to
    the unweighted single-pass form (verified by the harness in
    ``/tmp/sundqvist_audit/``). One pass is sufficient because the
    moist-static-energy budget converges at the per-step scale we use
    (verified by ``test_no_oversat_after_step``).

    Args:
        temperature: Temperature (K)
        specific_humidity: Specific humidity (kg/kg)
        cloud_water: Cloud liquid water (kg/kg)
        cloud_ice: Cloud ice (kg/kg)
        cloud_fraction: Cloud fraction [0-1] (currently unused — see
            module docstring on why we don't ECHAM-style weight here)
        pressure: Pressure (Pa)
        dt: Time step (s)
        config: Cloud configuration

    Returns:
        Tuple of (dT/dt, dq/dt, dqc/dt, dqi/dt)

    """
    # Phase weight + latent heat per phase (Sundqvist mixed-phase split).
    weight_liquid = jnp.clip(
        (temperature - config.t_mix_min)
        / (config.t_mix_max - config.t_mix_min),
        0.0, 1.0,
    )
    L_eff = weight_liquid * c.alhc + (1.0 - weight_liquid) * c.alhs
    L_cp = L_eff / c.cpd

    # ---- Pass 1: linearised Newton step ---------------------------------
    # ECHAM's ``cuadjtq`` and ``mo_cloud`` lines 776-779 (``zqcon``).
    qs, dqs_dt = _qs_and_dqs_dt(pressure, temperature)
    q_excess = specific_humidity - qs
    cond1 = q_excess / (1.0 + L_cp * dqs_dt)

    # Cap evaporation at available cloud water/ice.
    total_cloud = cloud_water + cloud_ice
    cond1 = jnp.maximum(cond1, -total_cloud)
    # Cap condensation at available vapour.
    cond1 = jnp.minimum(cond1, jnp.maximum(specific_humidity, 0.0))

    # ---- Pass 2: grid-box super-saturation cleanup ----------------------
    # ECHAM ``mo_cloud`` lines 762-784: re-evaluate q_s at the post-pass-1
    # temperature; condense any residual super-saturation that exceeds the
    # ``zoversat = 1 % · q_s_new`` tolerance. This pass is what stops
    # moisture accumulating in the column when pass 1 is conservative
    # (small per-step condensation due to the warming-feedback denominator).
    T_p1 = temperature + L_cp * cond1
    q_p1 = specific_humidity - cond1
    qs_p1, _ = _qs_and_dqs_dt(pressure, T_p1)
    oversat_tol = 0.01 * qs_p1                   # ECHAM's ``zoversat``
    cond2 = jnp.maximum(
        (q_p1 - qs_p1 - oversat_tol) / (1.0 + L_cp * dqs_dt),
        0.0,                                      # pass 2 only condenses
    )
    cond2 = jnp.minimum(cond2, jnp.maximum(q_p1, 0.0))

    cond_total = cond1 + cond2

    # Convert to rates so the caller (which integrates as
    # ``q_new = q + dqdt*dt``) sees the right magnitude.
    dqdt = -cond_total / dt

    # Partition between liquid and ice. Wrap the evap-branch divisions
    # in a safe double-where pattern so jax.grad through the unused
    # branch doesn't pick up a 0/eps NaN when cloud_water = cloud_ice = 0
    # (the common case at the start of the simulation).
    safe_total = jnp.where(total_cloud > 0, total_cloud, 1.0)
    qc_frac = jnp.where(total_cloud > 0, cloud_water / safe_total, 0.0)
    qi_frac = jnp.where(total_cloud > 0, cloud_ice / safe_total, 0.0)
    L_evap = jnp.where(
        total_cloud > 0,
        (cloud_water * c.alhc + cloud_ice * c.alhs) / safe_total,
        L_eff,                                    # fallback (unused)
    )

    dqcdt = jnp.where(
        cond_total > 0,                           # condensation
        weight_liquid * cond_total / dt,
        cond_total * qc_frac / dt,                # evaporation
    )
    dqidt = jnp.where(
        cond_total > 0,
        (1.0 - weight_liquid) * cond_total / dt,
        cond_total * qi_frac / dt,
    )

    # Temperature tendency. Latent heat uses the same mixed-phase L the
    # Newton step used so the moist static energy budget is consistent.
    L_for_dT = jnp.where(cond_total > 0, L_eff, L_evap)
    dtedt = L_for_dT * cond_total / (c.cpd * dt)

    return dtedt, dqdt, dqcdt, dqidt


def shallow_cloud_scheme(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    surface_pressure: float,
    dt: float,
    config: Optional[CloudParameters] = None
) -> Tuple[CloudTendencies, CloudState]:
    """Run shallow cloud scheme

    Args:
        temperature: Temperature (K) [nlev] or scalar
        specific_humidity: Specific humidity (kg/kg) [nlev] or scalar
        pressure: Pressure (Pa) [nlev] or scalar
        cloud_water: Cloud liquid water (kg/kg) [nlev] or scalar
        cloud_ice: Cloud ice (kg/kg) [nlev] or scalar
        surface_pressure: Surface pressure (Pa)
        dt: Time step (s)
        config: Cloud configuration

    Returns:
        Tuple of (tendencies, cloud_state)

    """
    if config is None:
        config = CloudParameters.default()
    
    # Ensure all inputs are arrays
    temperature = jnp.atleast_1d(temperature)
    specific_humidity = jnp.atleast_1d(specific_humidity)
    pressure = jnp.atleast_1d(pressure)
    cloud_water = jnp.atleast_1d(cloud_water)
    cloud_ice = jnp.atleast_1d(cloud_ice)
        
    # Calculate cloud fraction and relative humidity
    cloud_fraction, rel_humidity = calculate_cloud_fraction(
        temperature, specific_humidity, pressure, surface_pressure, config
    )
    
    # Calculate condensation/evaporation
    dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
        temperature, specific_humidity, cloud_water, cloud_ice,
        cloud_fraction, pressure, dt, config
    )
    
    # Within-timestep condensation: update cloud water/ice with condensation
    # so that microphysics (called next) sees non-zero values.
    # Following ECHAM mo_cloud.f90 where zxlb += zcnd within the same call.
    updated_cloud_water = jnp.maximum(cloud_water + dqcdt * dt, 0.0)
    updated_cloud_ice = jnp.maximum(cloud_ice + dqidt * dt, 0.0)

    # Total cloud cover (maximum overlap assumption)
    total_cloud_cover = jnp.max(cloud_fraction)
    
    # Create output structures
    tendencies = CloudTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dqcdt=dqcdt,
        dqidt=dqidt,
    )

    state = CloudState(
        cloud_fraction=cloud_fraction,
        cloud_water=updated_cloud_water,
        cloud_ice=updated_cloud_ice,
        rel_humidity=rel_humidity,
        total_cloud_cover=jnp.array(total_cloud_cover)
    )

    return tendencies, state


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.clouds.cloud_data import CloudData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class SundqvistCloudFraction(PhysicsTerm):
    """Sundqvist (1989) / Lohmann-Roeckner (1996) diagnostic cloud fraction.

    Pure cloud-fraction diagnostic — operates on column-vectorized state
    ``(nlev, ncols)``. Reads ``pressure_full`` / ``surface_pressure`` from
    the moist-air diagnostics dict and ``qc`` / ``qi`` from
    ``state.tracers``. Writes ``cloud_fraction``, plus a pass-through of
    the input ``qc`` / ``qi``, into the public ``"clouds"`` key
    (:class:`CloudData` typed sub-struct, shared with the downstream
    microphysics terms) and updates the public ``"relative_humidity"``
    key with ``q / qsat``.

    **No q ↔ qc/qi condensation tendency is emitted.** Saturation
    adjustment (cuadjtq Newton step) lives in the downstream microphysics
    term — the 1M scheme
    (:class:`~jcm.physics.clouds.echam_1m.Echam1MMicrophysics`) does it
    inside its column sweep
    (:func:`~jcm.physics.clouds.echam_1m._saturation_adjustment_layer`),
    and the 2M scheme
    (:class:`~jcm.physics.clouds.lohmann_2m.Lohmann2MMicrophysics`) does
    it via :func:`mixed_phase_deposition_and_corrections`. This matches
    ECHAM's ``mo_cloud.f90`` where condensation, autoconversion, rain
    evap, and flux propagation all live in the cloud routine alongside
    the cloud-fraction diagnostic — splitting condensation out into a
    separate upstream term (the previous JCM layout) double-counted
    against 2M and created a rain-evap ↔ re-condensation feedback with
    the 1M column-sweep variant, both of which are resolved by this
    structure.
    """

    name: ClassVar[str] = "sundqvist_cloud_fraction"
    category: ClassVar[str] = "cloud_fraction"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "surface_pressure",
    )
    provides: ClassVar[tuple[str, ...]] = ("clouds", "relative_humidity")
    # Carry seeded as zeros; cloud fraction / qc / qi are rebuilt every
    # step from RH and the dynamics tracers, so the zero seed is
    # overwritten on the first compute call. Downstream microphysics
    # terms write ``precip_*`` / TOA-flux fields on the same key so the
    # carry shape stays stable after step 1.
    carry_slots: ClassVar[dict[str, type]] = {"clouds": CloudData}

    def __init__(self, params: CloudParameters | None = None):
        """Hold the scheme-native :class:`CloudParameters`."""
        self.params = nnx.Param(params or CloudParameters.default())

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
        """Diagnose cloud fraction + relative humidity, no q tendency."""
        nlev, ncols = state.temperature.shape
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        surface_pressure = diagnostics["surface_pressure"]
        qc = state.tracers.get("qc", jnp.zeros_like(state.temperature))
        qi = state.tracers.get("qi", jnp.zeros_like(state.temperature))

        # Cloud fraction is purely diagnostic: ``cc = 1 - sqrt(1 - b0)``
        # with ``b0 = (RH - RH_crit) / (1 - RH_crit)``. Vmap over columns
        # so :func:`calculate_cloud_fraction` works on (nlev,) slices.
        cf_T, rh_T = jax.vmap(
            calculate_cloud_fraction,
            in_axes=(1, 1, 1, 0, None),
            out_axes=(0, 0),
        )(
            state.temperature, state.specific_humidity, pressure_full,
            surface_pressure, params,
        )
        cloud_fraction = cf_T.T  # back to (nlev, ncols)
        rel_humidity = rh_T.T

        # No condensation tendency — the downstream microphysics term
        # owns saturation adjustment now (see class docstring).
        zeros = jnp.zeros_like(state.temperature)
        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=zeros,
            specific_humidity=zeros,
            tracers={"qc": zeros, "qi": zeros},
        )

        # Write cloud_fraction (the only thing this term computes) plus a
        # pass-through of the input qc / qi so downstream terms see a
        # populated CloudData with the latest state values.
        prev_clouds = diagnostics.get(
            "clouds", CloudData.zeros((ncols,), nlev),
        )
        clouds = prev_clouds.copy(
            cloud_fraction=cloud_fraction,
            qc=qc,
            qi=qi,
        )

        return tendency, {
            **diagnostics,
            "clouds": clouds,
            "relative_humidity": rel_humidity,
        }
