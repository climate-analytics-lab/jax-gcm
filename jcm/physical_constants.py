'''
Date: 1/25/2024
For storing and initializing physical constants.
'''

import jax.numpy as jnp

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
sigl = None # Logarithm of full-level sigma
sigh = None # Half-level sigma
grdsig = None # g/(d_sigma p0): to convert fluxes of u,v,q into d(u,v,q)/dt
gridscp = None # g/(d_sigma p0 c_p): to convert energy fluxes into dT/dt
wvi = None # Weights for vertical interpolation

def initialize_physics():
    from jcm.geometry import hsg, fsg, dhs
    
    global sigl, sigh, grdsig, grdscp, wvi

    # 1.2 Functions of sigma and latitude
    sigh = hsg
    sigl = jnp.log(fsg)
    grdsig = grav/(dhs*p0)
    grdscp = grdsig/cp

    # Weights for vertical interpolation at half-levels(1,kx) and surface
    # Note that for phys.par. half-lev(k) is between full-lev k and k+1
    # Fhalf(k) = Ffull(k)+WVI(K,2)*(Ffull(k+1)-Ffull(k))
    # Fsurf = Ffull(kx)+WVI(kx,2)*(Ffull(kx)-Ffull(kx-1))
    wvi = jnp.zeros((fsg.shape[0], 2))
    wvi = wvi.at[:-1, 0].set(1./(sigl[1:]-sigl[:-1]))
    wvi = wvi.at[:-1, 1].set((jnp.log(sigh[1:-1])-sigl[:-1])*wvi[:-1, 0])
    wvi = wvi.at[-1, 1].set((jnp.log(0.99)-sigl[-1])*wvi[-2,0])
