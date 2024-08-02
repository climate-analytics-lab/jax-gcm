'''
Date: 1/25/2024
For storing and initializing physical constants.
'''

import jax.numpy as jnp

from jcm.params import kx

# Physical constants for dynamics
rearth = 6.371e+6 # Radius of Earth (m)
omega = 7.292e-05 # Rotation rate of Earth (rad/s)
grav = 9.81 # Gravitational acceleration (m/s/s)

# Physical constants for thermodynamics
p0 = 1.e+5 # Reference pressure (Pa)
cp = 1004.0 # Specific heat at constant pressure (J/K/kg)
akap = 2.0/7.0 # 1 - 1/gamma where gamma is the heat capacity ratio of a perfect diatomic gas (7/5)
rgas = akap * cp # Gas constant per unit mass for dry air (J/K/kg)
alhc = 2501.0 # Latent heat of condensation, in J/g for consistency with specific humidity in g/Kg
alhs = 2801.0 # Latent heat of sublimation
sbc = 5.67e-8 # Stefan-Boltzmann constant

# Functions of sigma and latitude (initial. in INPHYS)
sigl = jnp.log(jnp.linspace(1.0, 0.0, kx)) # Logarithm of full-level sigma
sigh = jnp.linspace(1.0, 0.0, kx + 1) # Half-level sigma
grdsig = grav / (jnp.gradient(sigh) * p0) # g/(d_sigma p0): to convert fluxes of u,v,q into d(u,v,q)/dt
grdscp = grav / (jnp.gradient(sigh) * p0 * cp) # g/(d_sigma p0 c_p): to convert energy fluxes into dT/dt
wvi = jnp.zeros((kx, 2)) # Weights for vertical interpolation
