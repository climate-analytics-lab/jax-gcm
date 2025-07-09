"""
Example usage of ICON Tiedtke-Nordeng convection scheme in JAX-GCM

This script demonstrates how to use the convection parameterization
with a simple atmospheric profile.

Date: 2025-01-09
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jcm.physics.icon.convection import (
    tiedtke_nordeng_convection,
    ConvectionConfig
)


def create_tropical_profile(nlev=40):
    """Create a typical tropical atmospheric profile"""
    # Pressure levels (Pa) - from surface to top
    pressure = jnp.logspace(5, 3, nlev)[::-1]  # 1000 hPa to 10 hPa
    
    # Height (m) - using standard atmosphere
    height = -7000 * jnp.log(pressure / 1e5)
    
    # Temperature profile - tropical with boundary layer
    surface_temp = 302.0  # K (29°C)
    bl_height = 1000.0    # m
    lapse_rate = 6.5e-3   # K/m
    
    # Boundary layer - well mixed
    bl_mask = height < bl_height
    temperature = jnp.where(
        bl_mask,
        surface_temp - 2.0 * height / bl_height,  # Slight cooling in BL
        surface_temp - 2.0 - lapse_rate * (height - bl_height)
    )
    
    # Add tropopause at ~16 km
    trop_height = 16000.0
    trop_mask = height > trop_height
    temperature = jnp.where(
        trop_mask,
        temperature[jnp.argmin(jnp.abs(height - trop_height))],
        temperature
    )
    
    # Humidity profile - moist boundary layer
    surface_rh = 0.85
    bl_rh = 0.80
    free_trop_rh = 0.40
    
    rel_humidity = jnp.where(
        bl_mask,
        surface_rh - (surface_rh - bl_rh) * height / bl_height,
        free_trop_rh * jnp.exp(-(height - bl_height) / 3000.0)
    )
    
    # Convert to specific humidity
    from jcm.physics.icon.convection.tiedtke_nordeng import saturation_mixing_ratio
    qs = jnp.array([saturation_mixing_ratio(p, t) for p, t in zip(pressure, temperature)])
    humidity = rel_humidity * qs
    
    # Simple wind profile
    u_wind = 5.0 - 10.0 * (pressure / 1e5 - 0.5)  # Easterlies at surface
    v_wind = jnp.zeros_like(u_wind)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'height': height,
        'u_wind': u_wind,
        'v_wind': v_wind,
        'rel_humidity': rel_humidity
    }


def plot_results(profile, tendencies, state):
    """Plot atmospheric profile and convection results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert height to km
    height_km = profile['height'] / 1000.0
    
    # Temperature profile
    ax = axes[0, 0]
    ax.plot(profile['temperature'], height_km, 'b-', label='Environment', linewidth=2)
    if state.ktype > 0:
        ax.plot(state.tu, height_km, 'r--', label='Updraft', linewidth=2)
        ax.plot(state.td, height_km, 'g--', label='Downdraft', linewidth=2)
        ax.axhline(height_km[state.kbase], color='k', linestyle=':', label='Cloud base')
        ax.axhline(height_km[state.ktop], color='k', linestyle='-.', label='Cloud top')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Humidity profile
    ax = axes[0, 1]
    ax.plot(profile['humidity'] * 1000, height_km, 'b-', label='Environment', linewidth=2)
    if state.ktype > 0:
        ax.plot(state.qu * 1000, height_km, 'r--', label='Updraft', linewidth=2)
        ax.plot(state.qd * 1000, height_km, 'g--', label='Downdraft', linewidth=2)
    ax.set_xlabel('Specific Humidity (g/kg)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Humidity Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mass flux
    ax = axes[0, 2]
    if state.ktype > 0:
        ax.plot(state.mfu, height_km, 'r-', label='Updraft', linewidth=2)
        ax.plot(-state.mfd, height_km, 'g-', label='Downdraft (−)', linewidth=2)
        ax.plot(state.mfu + state.mfd, height_km, 'k--', label='Net', linewidth=2)
    ax.set_xlabel('Mass Flux (kg/m²/s)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Convective Mass Flux')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    # Temperature tendency
    ax = axes[1, 0]
    heating_rate = tendencies.dtedt * 86400  # K/day
    ax.plot(heating_rate, height_km, 'r-', linewidth=2)
    ax.set_xlabel('Heating Rate (K/day)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Temperature Tendency')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    # Moisture tendency
    ax = axes[1, 1]
    drying_rate = tendencies.dqdt * 86400 * 1000  # g/kg/day
    ax.plot(drying_rate, height_km, 'b-', linewidth=2)
    ax.set_xlabel('Moistening Rate (g/kg/day)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Moisture Tendency')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    # Cloud water/ice
    ax = axes[1, 2]
    ax.plot(tendencies.qc_conv * 1000, height_km, 'b-', label='Cloud water', linewidth=2)
    ax.plot(tendencies.qi_conv * 1000, height_km, 'c-', label='Cloud ice', linewidth=2)
    ax.set_xlabel('Condensate (g/kg)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Convective Condensate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print diagnostics
    print(f"\nConvection Diagnostics:")
    print(f"- Convection type: {state.ktype} (0=none, 1=deep, 2=shallow, 3=mid)")
    print(f"- Cloud base height: {height_km[state.kbase]:.1f} km")
    print(f"- Cloud top height: {height_km[state.ktop]:.1f} km")
    print(f"- Surface precipitation: {tendencies.precip_conv * 3600:.1f} mm/hr")
    print(f"- Max updraft velocity: {jnp.max(state.mfu):.3f} kg/m²/s")
    print(f"- Max heating rate: {jnp.max(heating_rate):.1f} K/day")
    
    return fig


def main():
    """Run convection example"""
    print("ICON Tiedtke-Nordeng Convection Scheme Example")
    print("=" * 50)
    
    # Create atmospheric profile
    print("\nCreating tropical atmospheric profile...")
    profile = create_tropical_profile(nlev=40)
    
    # Configure convection scheme
    config = ConvectionConfig(
        tau=7200.0,           # 2-hour CAPE adjustment
        entrpen=1.0e-4,       # Entrainment for deep convection
        entrscv=3.0e-3,       # Entrainment for shallow convection
        cmfcmax=1.0,          # Maximum mass flux
        cprcon=1.4e-3,        # Precipitation efficiency
        cevapcu=2.0e-5        # Evaporation coefficient
    )
    
    # Run convection scheme
    print("\nRunning convection scheme...")
    tendencies, state = tiedtke_nordeng_convection(
        profile['temperature'],
        profile['humidity'],
        profile['pressure'],
        profile['height'],
        profile['u_wind'],
        profile['v_wind'],
        dt=600.0,  # 10-minute timestep
        config=config
    )
    
    # Plot results
    print("\nPlotting results...")
    fig = plot_results(profile, tendencies, state)
    
    plt.savefig('convection_example.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to 'convection_example.png'")
    
    plt.show()


if __name__ == "__main__":
    main()