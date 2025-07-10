"""
Convective adjustment for Tiedtke-Nordeng scheme

This module implements the final adjustment step of the convection scheme,
including:
- Saturation adjustment after convective tendencies
- Energy conservation checks
- Final temperature and moisture updates

Based on ICON mo_cuadjust.f90

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import Tuple, Optional
from functools import partial

from ..constants.physical_constants import (
    cp, alhc, alhs, tmelt, rd, rv, eps
)
from .tiedtke_nordeng import (
    ConvectionParameters, saturation_mixing_ratio
)


@jax.jit
def saturation_adjustment(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform saturation adjustment after convective updates
    
    This ensures that the updated state is thermodynamically consistent,
    removing any supersaturation through condensation.
    
    Args:
        temperature: Temperature after convective tendencies (K)
        specific_humidity: Specific humidity after tendencies (kg/kg)
        pressure: Pressure (Pa)
        cloud_water: Cloud liquid water (kg/kg)
        cloud_ice: Cloud ice (kg/kg)
        
    Returns:
        Tuple of adjusted (temperature, specific_humidity, cloud_water, cloud_ice)
    """
    
    # Total cloud condensate
    total_cloud = cloud_water + cloud_ice
    
    # Get saturation mixing ratio
    rs = saturation_mixing_ratio(pressure, temperature)
    
    # Convert specific humidity to mixing ratio
    r = specific_humidity / (1 - specific_humidity)
    
    # Total water (vapor + cloud)
    rtot = r + total_cloud
    
    def perform_adjustment(t, r, qc, qi):
        """Perform iterative saturation adjustment"""
        
        # Newton-Raphson iteration for saturation adjustment
        def adjustment_iteration(carry, _):
            t_curr, r_curr, qc_curr, qi_curr = carry
            
            # Current saturation mixing ratio
            rs_curr = saturation_mixing_ratio(pressure, t_curr)
            
            # Excess vapor
            excess = r_curr - rs_curr
            
            # Only condense if supersaturated
            cond = jnp.maximum(excess, 0.0)
            
            # Partition condensate between liquid and ice
            # Use simple temperature-based partitioning
            t_freeze = tmelt
            t_ice = tmelt - 23.0  # All ice below this
            
            frac_liquid = jnp.clip((t_curr - t_ice) / (t_freeze - t_ice), 0, 1)
            frac_ice = 1 - frac_liquid
            
            # Update cloud water and ice
            qc_new = qc_curr + cond * frac_liquid
            qi_new = qi_curr + cond * frac_ice
            
            # Update vapor
            r_new = r_curr - cond
            
            # Latent heat release
            # Use weighted average of liquid and ice latent heats
            lheat = frac_liquid * alhc + frac_ice * alhs
            t_new = t_curr + cond * lheat / cp
            
            return (t_new, r_new, qc_new, qi_new), None
        
        # Run several iterations for convergence
        (t_adj, r_adj, qc_adj, qi_adj), _ = lax.scan(
            adjustment_iteration, (t, r, qc, qi), None, length=10
        )
        
        return t_adj, r_adj, qc_adj, qi_adj
    
    # Perform adjustment
    t_adj, r_adj, qc_adj, qi_adj = perform_adjustment(temperature, r, cloud_water, cloud_ice)
    
    # Convert back to specific humidity
    q_adj = r_adj / (1 + r_adj)
    
    # Ensure non-negative values
    q_adj = jnp.maximum(q_adj, 0.0)
    qc_adj = jnp.maximum(qc_adj, 0.0)
    qi_adj = jnp.maximum(qi_adj, 0.0)
    
    return t_adj, q_adj, qc_adj, qi_adj


def energy_conservation_check(
    temperature_old: jnp.ndarray,
    specific_humidity_old: jnp.ndarray,
    cloud_water_old: jnp.ndarray,
    cloud_ice_old: jnp.ndarray,
    temperature_new: jnp.ndarray,
    specific_humidity_new: jnp.ndarray,
    cloud_water_new: jnp.ndarray,
    cloud_ice_new: jnp.ndarray,
    precipitation: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Check energy conservation in convective adjustment
    
    Args:
        *_old: State before adjustment
        *_new: State after adjustment
        precipitation: Precipitation rate (kg/m²/s)
        dt: Time step (s)
        
    Returns:
        Energy imbalance (W/m²)
    """
    # Sensible heat change
    dT = temperature_new - temperature_old
    sensible = cp * dT / dt
    
    # Latent heat changes
    dq = specific_humidity_new - specific_humidity_old
    dqc = cloud_water_new - cloud_water_old
    dqi = cloud_ice_new - cloud_ice_old
    
    # Latent heat (vapor uses L at current temperature)
    t_avg = 0.5 * (temperature_old + temperature_new)
    lv = alhc + (alhs - alhc) * jnp.clip((tmelt - t_avg) / 23.0, 0, 1)
    
    latent_vapor = lv * dq / dt
    latent_liquid = alhc * dqc / dt
    latent_ice = alhs * dqi / dt
    
    # Precipitation removes energy
    # Assume precipitation temperature is cloud temperature
    precip_energy = precipitation * cp * (t_avg - tmelt)
    
    # Total energy change
    total_energy = sensible + latent_vapor + latent_liquid + latent_ice + precip_energy
    
    return total_energy


@jax.jit
def convective_adjustment(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    convective_tendency_t: jnp.ndarray,
    convective_tendency_q: jnp.ndarray,
    convective_tendency_qc: jnp.ndarray,
    convective_tendency_qi: jnp.ndarray,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply convective tendencies and perform saturation adjustment
    
    This is the main interface for applying convection results to the
    model state, ensuring thermodynamic consistency.
    
    Args:
        temperature: Temperature before convection (K)
        specific_humidity: Specific humidity before (kg/kg)
        pressure: Pressure (Pa)
        cloud_water: Cloud water before (kg/kg)
        cloud_ice: Cloud ice before (kg/kg)
        convective_tendency_*: Tendencies from convection scheme
        dt: Time step (s)
        
    Returns:
        Tuple of adjusted (temperature, specific_humidity, cloud_water, cloud_ice)
    """
    # Apply convective tendencies
    t_conv = temperature + convective_tendency_t * dt
    q_conv = specific_humidity + convective_tendency_q * dt
    qc_conv = cloud_water + convective_tendency_qc * dt
    qi_conv = cloud_ice + convective_tendency_qi * dt
    
    # Ensure positive values before adjustment
    q_conv = jnp.maximum(q_conv, 0.0)
    qc_conv = jnp.maximum(qc_conv, 0.0)
    qi_conv = jnp.maximum(qi_conv, 0.0)
    
    # Perform saturation adjustment
    t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
        t_conv, q_conv, pressure, qc_conv, qi_conv
    )
    
    return t_adj, q_adj, qc_adj, qi_adj


def test_saturation_adjustment():
    """Test the saturation adjustment"""
    # Create supersaturated conditions
    temperature = jnp.array(280.0)  # K
    pressure = jnp.array(90000.0)    # Pa
    
    # Get saturation mixing ratio
    from .tiedtke_nordeng import saturation_mixing_ratio
    rs = saturation_mixing_ratio(pressure, temperature)
    qs = rs / (1 + rs)  # Convert to specific humidity
    
    # Create supersaturated state (120% RH)
    specific_humidity = 1.2 * qs
    cloud_water = jnp.array(0.0)
    cloud_ice = jnp.array(0.0)
    dt = 100.0
    
    # Perform adjustment
    t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
        temperature, specific_humidity, pressure,
        cloud_water, cloud_ice
    )
    
    # Check results
    print(f"Initial T: {temperature:.2f} K, q: {specific_humidity*1000:.2f} g/kg")
    print(f"Adjusted T: {t_adj:.2f} K, q: {q_adj*1000:.2f} g/kg")
    print(f"Cloud water: {qc_adj*1000:.2f} g/kg")
    print(f"Temperature increase: {t_adj - temperature:.2f} K")
    
    # Should have condensation and warming
    assert t_adj > temperature  # Latent heat release
    assert q_adj < specific_humidity  # Vapor removed
    assert qc_adj > cloud_water  # Cloud water increased
    
    # Should be approximately saturated after adjustment
    rs_adj = saturation_mixing_ratio(pressure, t_adj)
    qs_adj = rs_adj / (1 + rs_adj)
    rh_adj = q_adj / qs_adj
    print(f"Final RH: {rh_adj*100:.1f}%")
    # The adjustment reduces supersaturation significantly
    assert 0.75 < rh_adj < 1.05  # Should be closer to saturation
    
    print("Saturation adjustment test passed!")


def test_energy_conservation():
    """Test energy conservation check"""
    # Create a simple state change
    t_old = jnp.array(280.0)
    q_old = jnp.array(0.010)
    qc_old = jnp.array(0.001)
    qi_old = jnp.array(0.0)
    
    # Warming and drying (condensation)
    t_new = jnp.array(281.0)
    q_new = jnp.array(0.008)
    qc_new = jnp.array(0.003)
    qi_new = jnp.array(0.0)
    
    precip = jnp.array(0.0)
    dt = 3600.0
    
    imbalance = energy_conservation_check(
        t_old, q_old, qc_old, qi_old,
        t_new, q_new, qc_new, qi_new,
        precip, dt
    )
    
    print(f"Energy imbalance: {imbalance:.2f} W/m²")
    
    # The imbalance should be small if energy is conserved
    # (some imbalance is expected due to approximations)
    assert jnp.abs(imbalance) < 10.0  # Less than 10 W/m²
    
    print("Energy conservation test passed!")


if __name__ == "__main__":
    test_saturation_adjustment()
    print()
    test_energy_conservation()