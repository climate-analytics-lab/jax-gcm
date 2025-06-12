"""
Date: 1/25/2024
For storing and initializing physical constants.
"""

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

gamma  = 6.0       # Reference temperature lapse rate (-dT/dz in deg/km)
hscale = 7.5       # Reference scale height for pressure (in km)
hshum  = 2.5       # Reference scale height for specific humidity (in km)
refrh1 = 0.7       # Reference relative humidity of near-surface air
thd    = 2.4       # Max damping time (in hours) for horizontal diffusion
                                             # (del^6) of temperature and vorticity
thdd   = 2.4       # Max damping time (in hours) for horizontal diffusion
                                             # (del^6) of divergence
thds   = 12.0      # Max damping time (in hours) for extra diffusion
                                             ## (del^2) in the stratosphere
tdrs   = 24.0*30.0 # Damping time (in hours) for drag on zonal-mean wind
                                             # in the stratosphere

# to prevent blowup of gradients
epsilon = 1e-9

nstrad = 3 # number of timesteps between shortwave evaluations  


frz_fw = 273.15         # freezing point of freshwater at 1 atm (unit: Kelvin)
frz_sw = frz_fw - 1.8   # freezing point of seawater at 1 atm (unit: Kelvin)
cp_ocn = 3996.0 # ocean heat capacity of constant pressure (unit: J / kg)
