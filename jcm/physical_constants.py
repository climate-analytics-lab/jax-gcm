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
solc = 342.0 # Solar constant (area averaged) in W/m^2
epssw = 0.020 #Fraction of incoming solar radiation absorbed by ozone
# Functions of sigma and latitude (initial. in INPHYS)
sigl = jnp.array([-3.6888795, -2.3538785, -1.609438, -1.0788097, -0.67334455, -0.37833643, -0.18032359, -0.05129331]) # Logarithm of full-level sigma
sigh = jnp.array([0., 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.]) # Half-level sigma
grdsig = jnp.array([0.001962, 0.00109, 0.0008175, 0.00061313, 0.000545, 0.00057706, 0.00075462, 0.000981]) # g/(d_sigma p0): to convert fluxes of u,v,q into d(u,v,q)/dt
grdscp = jnp.array([1.9541833e-06, 1.0856575e-06, 8.1424315e-07, 6.1068232e-07, 5.4282856e-07, 5.7475995e-07, 7.5160898e-07, 9.7709153e-07]) # g/(d_sigma p0 c_p): to convert energy fluxes into dT/dt
wvi = jnp.array([[0.74906313, 0.519211  ],
                 [1.3432906 , 0.52088195],
                 [1.8845587 , 0.49444085],
                 [2.4663029 , 0.5211523 ],
                 [3.3897371 , 0.5508966 ],
                 [5.0501776 , 0.59072757],
                 [7.7501183 , 0.58097243],
                 [0.31963795, 0.        ]]) # Weights for vertical interpolation