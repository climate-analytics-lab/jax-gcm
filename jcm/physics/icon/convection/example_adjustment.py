"""
Example of using saturation adjustment after convection

This shows how the saturation adjustment can be applied as a 
post-processing step after convective tendencies have been computed.

Date: 2025-01-10
"""

import jax.numpy as jnp
from jcm.physics.icon.convection import (
    tiedtke_nordeng_convection, 
    saturation_adjustment,
    ConvectionParameters
)


def convection_with_adjustment(
    temperature, humidity, pressure, height, u_wind, v_wind,
    cloud_water, cloud_ice, tracers, dt
):
    """
    Apply convection scheme followed by saturation adjustment
    
    This ensures the final state is thermodynamically consistent.
    """
    # Apply convection scheme
    conv_tendencies, conv_state = tiedtke_nordeng_convection(
        temperature, humidity, pressure, height, 
        u_wind, v_wind, tracers, dt
    )
    
    # Update state with convective tendencies
    t_new = temperature + conv_tendencies.dtedt * dt
    q_new = humidity + conv_tendencies.dqdt * dt
    qc_new = cloud_water + conv_tendencies.qc_conv * dt
    qi_new = cloud_ice + conv_tendencies.qi_conv * dt
    
    # Apply saturation adjustment to ensure consistency
    t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
        t_new, q_new, pressure, qc_new, qi_new
    )
    
    # Calculate net tendencies including adjustment
    dtedt_total = (t_adj - temperature) / dt
    dqdt_total = (q_adj - humidity) / dt
    dqcdt_total = (qc_adj - cloud_water) / dt
    dqidt_total = (qi_adj - cloud_ice) / dt
    
    return dtedt_total, dqdt_total, dqcdt_total, dqidt_total, conv_state


def test_example():
    """Test the convection with adjustment"""
    # Create test profile
    nlev = 20
    temperature = jnp.linspace(290, 220, nlev)
    pressure = jnp.linspace(100000, 20000, nlev)
    height = jnp.linspace(0, 15000, nlev)
    
    # Moist lower levels
    humidity = jnp.ones(nlev) * 0.001
    humidity = humidity.at[:10].set(0.008)
    
    u_wind = jnp.ones(nlev) * 10.0
    v_wind = jnp.zeros(nlev)
    
    cloud_water = jnp.zeros(nlev)
    cloud_ice = jnp.zeros(nlev)
    
    # Simple tracers array
    tracers = jnp.zeros((nlev, 3))  # qv, qc, qi
    tracers = tracers.at[:, 0].set(humidity)
    
    dt = 1800.0
    
    # Apply convection with adjustment
    dtedt, dqdt, dqcdt, dqidt, conv_state = convection_with_adjustment(
        temperature, humidity, pressure, height, u_wind, v_wind,
        cloud_water, cloud_ice, tracers, dt
    )
    
    print(f"Temperature tendency range: {dtedt.min():.2e} to {dtedt.max():.2e} K/s")
    print(f"Humidity tendency range: {dqdt.min():.2e} to {dqdt.max():.2e} kg/kg/s")
    print(f"Convection type: {conv_state.ktype}")
    
    # The adjusted state should be thermodynamically consistent
    t_final = temperature + dtedt * dt
    q_final = humidity + dqdt * dt
    print(f"\nFinal temperature range: {t_final.min():.1f} to {t_final.max():.1f} K")
    print(f"Final humidity range: {q_final.min()*1000:.2f} to {q_final.max()*1000:.2f} g/kg")


if __name__ == "__main__":
    test_example()