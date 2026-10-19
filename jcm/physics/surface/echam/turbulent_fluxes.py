"""Turbulent flux calculations for surface-atmosphere exchange.

This module implements bulk aerodynamic formulations for computing
surface fluxes of momentum, heat, and moisture following ICON's approach.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from jcm.physics.convection.saturation import dewpoint_temperature
from .surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing,
    SurfaceFluxes, SurfaceResistances, SurfaceDiagnostics
)


@jax.jit
def compute_bulk_richardson_number(
    temperature_air: jnp.ndarray,
    temperature_surface: jnp.ndarray,
    humidity_air: jnp.ndarray,
    humidity_surface: jnp.ndarray,
    wind_speed: jnp.ndarray,
    reference_height: float = 10.0
) -> jnp.ndarray:
    """Compute bulk Richardson number for surface layer stability.
    
    Args:
        temperature_air: Air temperature [K] (ncol,)
        temperature_surface: Surface temperature [K] (ncol, nsfc_type)
        humidity_air: Air specific humidity [kg/kg] (ncol,)
        humidity_surface: Surface specific humidity [kg/kg] (ncol, nsfc_type)
        wind_speed: Wind speed [m/s] (ncol,)
        reference_height: Reference height [m]
        
    Returns:
        Bulk Richardson number [-] (ncol, nsfc_type)

    """
    # Virtual potential temperatures
    theta_v_air = temperature_air * (1.0 + 0.608 * humidity_air)
    theta_v_surface = temperature_surface * (1.0 + 0.608 * humidity_surface)
    
    # Mean virtual potential temperature
    theta_v_mean = 0.5 * (theta_v_air[:, None] + theta_v_surface)
    
    # Buoyancy term
    buoyancy = (c.grav * reference_height *
               (theta_v_air[:, None] - theta_v_surface) / theta_v_mean)
    
    # Wind shear term (prevent division by zero)
    wind_shear_squared = jnp.maximum(wind_speed[:, None]**2, 0.01)
    
    # Richardson number
    ri_bulk = buoyancy / wind_shear_squared
    
    return ri_bulk


@jax.jit
def compute_stability_functions(
    richardson_number: jnp.ndarray,
    stable_limit: float = 0.2,
    unstable_coeff: float = 16.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute stability functions following Businger-Dyer relationships.
    
    Args:
        richardson_number: Bulk Richardson number [-] (ncol, nsfc_type)
        stable_limit: Maximum Richardson number for stable conditions
        unstable_coeff: Coefficient for unstable conditions
        
    Returns:
        Tuple of (heat_stability_function, momentum_stability_function)

    """
    # Stable conditions (Ri > 0)
    stable_mask = richardson_number >= 0.0
    ri_stable = jnp.minimum(richardson_number, stable_limit)
    phi_h_stable = 1.0 + 5.0 * ri_stable
    phi_m_stable = phi_h_stable
    
    # Unstable conditions (Ri < 0). Businger-Dyer stability functions:
    #     Φ_m = (1 - 16 ζ)^(-1/4)   = (1 + 16|ζ|)^(-1/4)  for ζ < 0
    #     Φ_h = (1 - 16 ζ)^(-1/2)   = (1 + 16|ζ|)^(-1/2)  for ζ < 0
    # Both are < 1 under unstable conditions, which makes the bulk
    # exchange coefficients (κ² / (ln·Φ_m·Φ_h)·…) LARGER — the standard
    # boundary-layer enhancement of turbulent mixing under buoyant
    # convection. The original implementation here used (1 - 16|Ri|)
    # inside the parenthesis, which flipped the sign and made Φ ≫ 1
    # for unstable cells, suppressing surface fluxes by ~5× over warm
    # surfaces and creating runaway near-surface heat accumulation
    # that drove the day-7 NaN on T63L47 + real terrain.
    ri_unstable = jnp.maximum(richardson_number, -0.5)  # Limit to -0.5
    x_arg = 1.0 + unstable_coeff * jnp.abs(ri_unstable)
    x = x_arg**0.25
    phi_h_unstable = x_arg**(-0.5)
    phi_m_unstable = x**(-1)
    
    # Combine stable and unstable
    phi_h = jnp.where(stable_mask, phi_h_stable, phi_h_unstable)
    phi_m = jnp.where(stable_mask, phi_m_stable, phi_m_unstable)
    
    return phi_h, phi_m


@jax.jit
def compute_exchange_coefficients(
    wind_speed: jnp.ndarray,
    roughness_momentum: jnp.ndarray,
    roughness_heat: jnp.ndarray,
    stability_heat: jnp.ndarray,
    stability_momentum: jnp.ndarray,
    min_wind_speed: float,
    von_karman: float,
    reference_height: float = 10.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute bulk exchange coefficients for momentum, heat, and moisture.
    
    Args:
        wind_speed: Wind speed [m/s] (ncol,)
        roughness_momentum: Momentum roughness length [m] (ncol, nsfc_type)
        roughness_heat: Heat roughness length [m] (ncol, nsfc_type)
        stability_heat: Heat stability function [-] (ncol, nsfc_type)
        stability_momentum: Momentum stability function [-] (ncol, nsfc_type)
        reference_height: Reference height [m]
        params: Surface parameters
        
    Returns:
        Tuple of (momentum_coeff, heat_coeff, moisture_coeff) [m/s]

    """    
    # Ensure minimum wind speed
    wind_speed_safe = jnp.maximum(wind_speed, min_wind_speed)

    # Logarithmic terms — guard against non-positive arguments
    ln_z_z0m = jnp.log(jnp.maximum(reference_height, 0.1)
                       / jnp.maximum(roughness_momentum, 1e-5))
    ln_z_z0h = jnp.log(jnp.maximum(reference_height, 0.1)
                       / jnp.maximum(roughness_heat, 1e-5))
    
    # Exchange coefficients with stability correction
    cd = (von_karman**2 / 
          (ln_z_z0m * stability_momentum)**2)
    ch = (von_karman**2 / 
          (ln_z_z0m * ln_z_z0h * stability_momentum * stability_heat))
    
    # Convert to exchange coefficients [m/s]
    momentum_coeff = cd * wind_speed_safe[:, None]
    heat_coeff = ch * wind_speed_safe[:, None]
    moisture_coeff = heat_coeff  # Assume same as heat
    
    return momentum_coeff, heat_coeff, moisture_coeff


@jax.jit
def compute_surface_humidity(
    temperature_surface: jnp.ndarray,
    pressure: jnp.ndarray,
    surface_type_indices: jnp.ndarray = None
) -> jnp.ndarray:
    """Compute saturation humidity at the surface.
    
    Args:
        temperature_surface: Surface temperature [K] (ncol, nsfc_type)
        pressure: Surface pressure [Pa] (ncol,)
        surface_type_indices: Surface type for each tile (ncol, nsfc_type)
        
    Returns:
        Surface saturation humidity [kg/kg] (ncol, nsfc_type)

    """
    # Saturation vapor pressure (simplified Clausius-Clapeyron)
    # e_sat = e0 * exp(L/Rv * (1/T0 - 1/T))
    e0 = 611.0  # Pa
    L_over_Rv = c.alhc / c.rv  # K
    T0 = c.tmelt  # K
    
    # Wide math-safety clip only — prevents divide-by-zero and exp overflow,
    # NOT a physical-range bound
    T_safe = jnp.clip(temperature_surface, 50.0, 500.0)
    exponent = jnp.clip(L_over_Rv * (1.0/T0 - 1.0/T_safe), -50.0, 50.0)
    e_sat = e0 * jnp.exp(exponent)

    # Saturation mixing ratio — cap e_sat < 0.99 * pressure
    epsilon = c.eps
    p_safe = jnp.maximum(pressure[:, None], 1.0)
    e_sat = jnp.minimum(e_sat, 0.99 * p_safe)
    q_sat = epsilon * e_sat / jnp.maximum(p_safe - (1.0 - epsilon) * e_sat, 1.0)

    # Ensure reasonable bounds
    q_sat = jnp.clip(q_sat, 0.0, 0.1)  # Max 100 g/kg

    return q_sat


@jax.jit
def compute_surface_resistances(
    atmospheric_state: AtmosphericForcing,
    surface_state: SurfaceState,
    richardson_number: jnp.ndarray,
    params: SurfaceParameters | None = None
) -> SurfaceResistances:
    """Compute surface resistances for heat, moisture, and momentum transfer.
    
    Args:
        atmospheric_state: Atmospheric forcing
        surface_state: Surface state
        richardson_number: Bulk Richardson number
        params: Surface parameters
        
    Returns:
        Surface resistances

    """
    if params is None:  # not a def default: it would build jax arrays at import (#859)
        params = SurfaceParameters.default()
    ncol, nsfc_type = surface_state.temperature.shape
    
    # Wind speed
    wind_speed = jnp.sqrt(jnp.maximum(atmospheric_state.u_wind**2 + atmospheric_state.v_wind**2, 1.0e-30))
    wind_speed = jnp.maximum(wind_speed, params.min_wind_speed)
    
    # Stability functions
    stability_heat, stability_momentum = compute_stability_functions(richardson_number)
    
    # Exchange coefficients
    momentum_coeff, heat_coeff, moisture_coeff = compute_exchange_coefficients(
        wind_speed, surface_state.roughness_momentum, surface_state.roughness_heat,
        stability_heat, stability_momentum, params.min_wind_speed, params.von_karman,
    )
    
    # Convert to resistances [s/m]
    aerodynamic_momentum = 1.0 / jnp.maximum(momentum_coeff, 1e-6)
    aerodynamic_heat = 1.0 / jnp.maximum(heat_coeff, 1e-6)
    aerodynamic_moisture = 1.0 / jnp.maximum(moisture_coeff, 1e-6)
    
    # Surface resistances (simplified)
    surface_moisture = jnp.where(
        jnp.arange(nsfc_type)[None, :] == params.iwtr,  # Water
        0.0,  # No surface resistance over water
        100.0  # Simple surface resistance for land/ice
    ) * jnp.ones((ncol, nsfc_type))
    
    # Canopy and soil resistances (placeholders for land model)
    canopy_resistance = jnp.full(ncol, 100.0)  # s/m
    soil_resistance = jnp.full(ncol, 200.0)   # s/m
    
    return SurfaceResistances(
        aerodynamic_heat=aerodynamic_heat,
        aerodynamic_moisture=aerodynamic_moisture,
        aerodynamic_momentum=aerodynamic_momentum,
        surface_moisture=surface_moisture,
        canopy_resistance=canopy_resistance,
        soil_resistance=soil_resistance,
        stability_heat=stability_heat,
        stability_momentum=stability_momentum
    )


@jax.jit
def compute_surface_diagnostics(
    atmospheric_state: AtmosphericForcing,
    surface_state: SurfaceState,
    surface_fluxes: SurfaceFluxes,
    resistances: SurfaceResistances,
    surface_humidity_tile: jnp.ndarray,
    params: SurfaceParameters | None = None,
) -> SurfaceDiagnostics:
    """Compute standard surface diagnostics (2m temperature, u*, etc.).

    The 10 m wind is not among them: the ECHAM family has one 10 m wind,
    diagnosed by the vertical-diffusion term from the per-tile
    surface-layer reduction (``vertical_diffusion.wind_10m*``, ECHAM
    ``nsurf_diag``), which the surface-exchange contract publishes and the
    AeroCom ``uas``/``vas`` apply to the post-physics wind.

    ``surface_humidity_tile`` is this function's ONLY surface-humidity
    input: it no longer derives its own per-tile anchor from
    ``surface_state.temperature``. Three separate Codex review rounds
    (jax-gcm#1140/#1151) each caught a different defect in a previous,
    locally re-derived copy of this quantity — saturation taken at the
    fraction-weighted MEAN tile temperature instead of per-tile, the land
    tile anchored at saturation instead of JSBACH's boundary, and that
    boundary evaluated at the lowest model level's pressure instead of the
    surface's — which is the signature of a quantity with more than one
    definition, not three unrelated bugs. The fix is structural: this
    function now CONSUMES the per-tile anchor its caller
    (:func:`~jcm.physics.surface.echam.surface_physics.surface_physics_step`)
    builds from :func:`~jcm.physics.thermodynamics.
    jsbach_surface_humidity_boundary` — the SAME function the live TTE-TKE
    surface exchange's own Richardson number calls
    (:func:`~jcm.physics.vertical_diffusion.tte_tke.surface_layer.
    surface_bulk_richardson`'s ``qts``) — at the surface (lowest interface)
    pressure, with the land tile's JSBACH ``cair``/``csat`` already
    applied, so there is exactly one place this boundary value is defined.

    Args:
        atmospheric_state: Atmospheric forcing
        surface_state: Surface state
        surface_fluxes: Surface fluxes
        resistances: Surface resistances
        surface_humidity_tile: Per-tile surface-layer humidity anchor
            [kg/kg] (ncol, nsfc_type) — Sonntag saturation over water/ice,
            the JSBACH boundary ``csat·qs + (1−cair)·q_air`` over land, at
            ``atmospheric_state.surface_pressure`` (NOT the lowest model
            level's ``pressure``). Built by the caller via
            :func:`~jcm.physics.thermodynamics.
            jsbach_surface_humidity_boundary`; see
            :func:`~jcm.physics.surface.echam.surface_physics.
            surface_physics_step` for how it assembles the per-tile
            ``cair``/``csat`` this expects water/ice to carry as ``1.0``.
        params: Surface parameters

    Returns:
        Surface diagnostics

    """
    if params is None:  # not a def default: it would build jax arrays at import (#859)
        params = SurfaceParameters.default()
    ncol = surface_state.temperature.shape[0]

    # Reference heights. ``z_ref`` is the ACTUAL height of
    # ``atmospheric_state``'s temperature/humidity/wind above the surface —
    # the model's lowest full level (tens to hundreds of metres, not
    # literally 10 m; a review on #1140 caught this call reading the
    # default ``reference_height=10.0`` while the field it was applied to
    # was the lowest MODEL level). ``AtmosphericForcing.reference_height``
    # carries the real value from the caller (``EchamSurface.__call__``);
    # it defaults to 10.0 only for callers/tests that genuinely hand this
    # function a 10 m input.
    z_2m = 2.0                                   # 2m for temperature/humidity
    z_ref = atmospheric_state.reference_height   # height of the atmospheric_state level

    # Grid-box mean surface temperature
    temp_surface_mean = jnp.sum(surface_state.fraction * surface_state.temperature, axis=1)

    # Simple linear interpolation for 2m temperature
    # T_2m = T_surface + (T_air - T_surface) * (z_2m / z_ref)
    temp_2m = temp_surface_mean + (atmospheric_state.temperature - temp_surface_mean) * (z_2m / z_ref)

    # Tile-specific 2 m diagnostics (moved up from below #1140-review: the
    # grid-mean humidity_2m immediately below is now formed FROM these, not
    # independently). ``z_ref`` may be a bare scalar (callers/tests with no
    # opinion on height) or an ``(ncol,)`` per-column array (what
    # ``EchamSurface.__call__`` hands it, see the Richardson number below
    # for the same shape note); ``jnp.atleast_1d(...)[:, None]`` makes
    # either one broadcast safely against the ``(ncol, nsfc_type)`` tile
    # arrays here, instead of colliding ``ncol`` against ``nsfc_type``.
    z_ref_tile = jnp.atleast_1d(z_ref)[:, None]
    temp_2m_tile = (
        surface_state.temperature
        + (atmospheric_state.temperature[:, None] - surface_state.temperature)
        * (z_2m / z_ref_tile))
    humidity_2m_tile = (
        surface_humidity_tile
        + (atmospheric_state.humidity[:, None] - surface_humidity_tile) * (z_2m / z_ref_tile))

    # Grid-mean 2 m humidity: fraction-weight the PER-TILE 2 m humidities
    # above, rather than interpolating from a fraction-weighted mean
    # surface humidity as the grid-mean temperature does. Saturation (and
    # the JSBACH boundary, which is itself nonlinear through q_sat) is
    # nonlinear in temperature, so averaging tile temperatures before
    # taking the surface humidity does NOT equal averaging the tile
    # surface humidities — a #1140 review reproduced this directly in a
    # warm-water/cold-ice cell, where the two differed materially and the
    # old grid mean disagreed with the published per-tile
    # ``humidity_2m_tile``. ``temperature_2m`` needs no such fix: it is a
    # LINEAR blend of a linear per-tile blend, so fraction-weighting
    # commutes and the mean-first form above already equals the
    # per-tile-then-weighted one exactly.
    humidity_2m = jnp.sum(surface_state.fraction * humidity_2m_tile, axis=1)

    # 2m dew point: the exact analytic inverse of the Tetens saturation law
    # jcm's shared ``saturation`` helper uses (jax-gcm#1140), applied to the
    # 2 m humidity just formed above. Evaluated at
    # ``atmospheric_state.surface_pressure`` — the near-surface pressure
    # ``humidity_2m`` and ``surface_humidity_tile`` are themselves anchored
    # at (the 2 m level sits a few metres above the surface, not partway to
    # the lowest model level tens to hundreds of metres up) — NOT the
    # lowest-model-level ``pressure`` a previous version of this line used,
    # which a #1151 review caught compounding that same pressure mismatch
    # into the dew point on top of the surface-humidity anchor itself.
    # Replaces a placeholder linear-in-humidity formula with no
    # saturation-physics basis (``temp_2m - 20*(1 - q/0.01)``, dimensionally
    # wrong — it compared a specific humidity in kg/kg against a bare
    # 0.01 constant). See :func:`jcm.physics.convection.saturation.
    # dewpoint_temperature` for why this is the exact inverse of jcm's own
    # law rather than a transcription of ECHAM's ``zdew2w``/``zdewl``/
    # ``zdewi`` line.
    #
    # ``temp_2m`` and ``humidity_2m`` are still two INDEPENDENT quantities
    # (one a linear blend of tile temperatures, the other a fraction-
    # weighted blend of each tile's own, separately-anchored 2 m
    # humidity), so nothing stops the pair from implying a dew point above
    # the 2 m temperature it was derived alongside — a physically
    # impossible output a review on #1140 reproduced directly (a cold
    # surface under warmer near-saturated air). This remains true after
    # the per-tile humidity fix above: fixing HOW humidity_2m is averaged
    # does not create a saturation constraint between it and temp_2m: the
    # cap below is still required. ECHAM's own
    # ``postproc_ocean``/``_land``/``_ice`` guard against exactly this:
    # ``zdew2w = MIN(zt2w, ...)`` caps the dew point at the 2 m temperature
    # unconditionally. Applied here for the same reason.
    dewpoint_2m = jnp.minimum(
        dewpoint_temperature(humidity_2m, atmospheric_state.surface_pressure),
        temp_2m)

    wind_speed_atm = jnp.sqrt(jnp.maximum(
        atmospheric_state.u_wind**2 + atmospheric_state.v_wind**2, 1.0e-30))

    # Friction velocity
    momentum_flux_magnitude = jnp.sqrt(jnp.maximum(
        surface_fluxes.momentum_u_mean**2 + surface_fluxes.momentum_v_mean**2, 1.0e-30
    ))
    air_density = atmospheric_state.pressure / (c.rd * atmospheric_state.temperature)
    friction_velocity = jnp.sqrt(jnp.maximum(momentum_flux_magnitude / air_density, 1.0e-30))

    # Richardson number (grid-box mean). Same ``z_ref``/``z_ref_tile`` and
    # per-tile surface-humidity anchor (``surface_humidity_tile``, Sonntag
    # over water/ice, the JSBACH boundary over land, at the surface
    # pressure) as the 2 m blend above — the SAME anchor, passed in once
    # and consumed here and there, not two independent copies. The per-tile
    # Richardson numbers are fraction-weighted AFTER the (nonlinear)
    # per-tile computation, exactly as the tile-then-weight ordering the
    # 2 m humidity needed above.
    ri_mean = jnp.sum(surface_state.fraction * compute_bulk_richardson_number(
        atmospheric_state.temperature, surface_state.temperature,
        atmospheric_state.humidity,
        surface_humidity_tile,
        wind_speed_atm, reference_height=z_ref_tile,
    ), axis=1)

    # Energy balance
    net_radiation = surface_fluxes.shortwave_net + surface_fluxes.longwave_net
    net_radiation_mean = jnp.sum(surface_state.fraction * net_radiation, axis=1)

    energy_balance_residual = (net_radiation_mean - surface_fluxes.sensible_heat_mean -
                              surface_fluxes.latent_heat_mean)

    return SurfaceDiagnostics(
        temperature_2m=temp_2m,
        humidity_2m=humidity_2m,
        dewpoint_2m=dewpoint_2m,
        friction_velocity=friction_velocity,
        richardson_number=ri_mean,
        surface_layer_height=jnp.full(ncol, 100.0),  # Simplified
        net_radiation=net_radiation_mean,
        radiation_balance=net_radiation_mean,
        energy_balance_residual=energy_balance_residual,
        temperature_2m_tile=temp_2m_tile,
        humidity_2m_tile=humidity_2m_tile,
    )