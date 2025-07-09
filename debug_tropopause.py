#!/usr/bin/env python3
"""Debug WMO tropopause function"""

import jax.numpy as jnp
from jcm.physics.icon.diagnostics.wmo_tropopause import (
    wmo_tropopause, compute_geopotential_height, find_tropopause_level, 
    compute_lapse_rate, GWMO, P_DEFAULT
)

# Create simple test atmosphere
nlev = 20
pressure = jnp.logspace(jnp.log10(100000), jnp.log10(1000), nlev)
surface_pressure = jnp.array([100000.0])

# Create realistic temperature profile
T_surface = 288.0
T_tropopause = 220.0
p_tropopause = 20000.0

temperature = jnp.zeros(nlev)
for i in range(nlev):
    p = pressure[i]
    if p > p_tropopause:
        # Troposphere - linear decrease
        height_approx = -7000 * jnp.log(p / 100000)
        temperature = temperature.at[i].set(T_surface - 0.0065 * height_approx)
    else:
        # Stratosphere - constant
        temperature = temperature.at[i].set(T_tropopause)

# Compute height
height = compute_geopotential_height(pressure[None, :], temperature[None, :], surface_pressure)

# Compute lapse rate
lapse_rate = compute_lapse_rate(temperature[None, :], height)

print("Debug WMO Tropopause")
print("=" * 50)
print(f"Pressure levels: {pressure}")
print(f"Temperature: {temperature}")
print(f"Height: {height[0]}")
print(f"Lapse rate: {lapse_rate[0]}")
print(f"GWMO threshold: {GWMO}")
print(f"Levels where lapse >= GWMO: {jnp.where(lapse_rate[0] >= GWMO)[0]}")

# Find tropopause
tropopause_pressure = find_tropopause_level(
    temperature[None, :], pressure[None, :], height, 
    ncctop=2, nccbot=18
)

print(f"Found tropopause pressure: {tropopause_pressure[0]} Pa")
print(f"Expected around: {p_tropopause} Pa")
print(f"Default pressure: {P_DEFAULT} Pa")