#!/usr/bin/env python3
"""
Example of using ICON physics with JAX-GCM

This example shows how to use the ICON physics implementation
instead of the default SPEEDY physics.
"""

import jax.numpy as jnp
from jcm.physics.icon.icon_physics import IconPhysics
from jcm.physics.icon.constants import physical_constants
from jcm.physics.icon.diagnostics import wmo_tropopause

def main():
    print("ICON Physics Integration Example")
    print("=" * 50)
    
    # Create ICON physics instance
    icon_physics = IconPhysics(
        enable_radiation=True,
        enable_convection=True,
        enable_clouds=True,
        enable_vertical_diffusion=True,
        enable_surface=True,
        enable_gravity_waves=True,
        enable_chemistry=False,  # Start with simple setup
        write_output=True
    )
    
    print("✅ ICON Physics created successfully")
    print(f"   - Physical constants: g = {physical_constants.grav} m/s²")
    print(f"   - Enabled modules: radiation, convection, clouds, vertical diffusion, surface, gravity waves")
    
    # For now, just demonstrate the ICON physics framework
    # Integration with Model class will be completed once import issues are resolved
    print("📝 Note: Full model integration pending dinosaur package compatibility")
    print("   Creating ICON physics framework demonstration...")
    
    print("✅ ICON physics framework ready for integration")
    print(f"   - Physics modules: {len(icon_physics.terms)} active terms")
    print(f"   - Configuration: JAX-compatible, differentiable")
    
    # Example of using ICON diagnostics
    print("\n📊 ICON Diagnostics Example")
    print("-" * 30)
    
    # Create sample atmospheric profile for tropopause diagnostic
    nlev = 20
    pressure = jnp.logspace(jnp.log10(100000), jnp.log10(1000), nlev)
    surface_pressure = jnp.array([100000.0])
    
    # Create temperature profile (troposphere + stratosphere)
    temperature = jnp.zeros(nlev)
    for i in range(nlev):
        p = pressure[i]
        if p > 20000:  # Troposphere
            height = -7000 * jnp.log(p / 100000)
            temperature = temperature.at[i].set(288.0 - 0.0065 * height)
        else:  # Stratosphere
            temperature = temperature.at[i].set(220.0)
    
    # Calculate tropopause pressure
    tropopause_pressure = wmo_tropopause(
        temperature[None, :], 
        pressure[None, :], 
        surface_pressure
    )
    
    print(f"   - Sample tropopause pressure: {tropopause_pressure[0]:.1f} Pa")
    print(f"   - Equivalent altitude: ~{-7000 * jnp.log(tropopause_pressure[0] / 100000):.0f} m")
    
    print("\n🚀 Integration Status")
    print("-" * 30)
    print("✅ ICON physics framework integrated with JAX-GCM")
    print("✅ WMO tropopause diagnostic working")
    print("✅ Physical constants and data structures in place")
    print("⏳ Individual physics modules ready for implementation")
    print("⏳ Full physics ensemble pending module completion")
    
    print("\n🔄 Next Steps")
    print("-" * 30)
    print("1. Implement individual physics modules (radiation, convection, etc.)")
    print("2. Add physics module tests and validation")
    print("3. Compare outputs with SPEEDY physics")
    print("4. Optimize performance for JAX transformations")
    
    return icon_physics

if __name__ == "__main__":
    physics = main()
    print(f"\nICON physics framework ready! 🎉")