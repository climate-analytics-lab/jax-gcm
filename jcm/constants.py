"""Physical constants used by the model regardless of physics package.

These are general physical constants shared across SPEEDY, ICON, and any
future physics packages. Package-specific tunables live with that package.
"""

from typing import NamedTuple


class PhysicalConstants(NamedTuple):
    """Physical constants used across atmospheric physics packages."""

    # Fundamental constants
    rearth: float = 6.371e+6      # Radius of Earth (m)
    omega: float = 7.292e-05      # Rotation rate of Earth (rad/s)
    grav: float = 9.81            # Gravitational acceleration (m/s²)

    # Thermodynamic constants
    p0: float = 1.0e+5            # Reference pressure (Pa)
    cp: float = 1004.0            # Specific heat at constant pressure (J/K/kg)
    akap: float = 2.0 / 7.0       # kappa = R/cp
    rgas: float = 287.0           # Gas constant per unit mass for dry air (J/K/kg)
    karman_const: float = 0.4     # von Kármán constant (dimensionless)

    # Latent heats (J/kg)
    alhc: float = 2.501e6         # Latent heat of condensation
    alhs: float = 2.834e6         # Latent heat of sublimation
    alhf: float = 3.34e5          # Latent heat of fusion

    # Radiation constants
    sbc: float = 5.67e-8          # Stefan-Boltzmann constant (W/m²/K⁴)
    solc: float = 1361.0          # Solar constant (W/m²)

    # Water vapor constants
    rd: float = 287.0             # Gas constant for dry air (J/K/kg)
    rv: float = 461.0             # Gas constant for water vapor (J/K/kg)
    eps: float = 0.622            # Ratio of molecular weights (Md/Mv)

    # Thermodynamic reference values
    t0: float = 273.15            # Reference temperature (K)
    tmelt: float = 273.15         # Melting point of ice (K)

    # Cloud microphysics constants
    rhow: float = 1000.0          # Density of liquid water (kg/m³)
    rhoi: float = 917.0           # Density of ice (kg/m³)

    # Numerical constants
    epsilon: float = 1e-12        # Small number to prevent division by zero

    # ECHAM-6.3 mo_physical_constants additions used by the two-moment scheme
    rgrav: float = 1.0 / 9.81     # Reciprocal of gravitational acceleration (s²/m)
    cpd: float = 1004.64          # Specific heat at constant pressure for dry air (J/K/kg)
    cvd: float = 1004.64 - 287.0  # = cpd - rd; specific heat at constant volume for dry air (J/K/kg)
    cpv: float = 1869.46          # Specific heat at constant pressure for water vapor (J/K/kg)
    cvv: float = 1869.46 - 461.0  # = cpv - rv; specific heat at constant volume for water vapor (J/K/kg)
    vtmpc1: float = 461.0 / 287.0 - 1.0      # = rv/rd - 1; moisture buoyancy parameter
    vtmpc2: float = 1869.46 / 1004.64 - 1.0  # = cpv/cpd - 1; moist heat-capacity coefficient
    ak: float = 1.3806504e-23     # Boltzmann constant (J/K)
    p0s1_bg: float = 101325.0     # Sea level reference pressure (Pa)

    @classmethod
    def default(cls) -> 'PhysicalConstants':
        """Return default physical constants."""
        return cls()


physical_constants = PhysicalConstants.default()

# Fundamental
rearth = physical_constants.rearth
omega = physical_constants.omega
grav = physical_constants.grav

# Thermodynamic
p0 = physical_constants.p0
cp = physical_constants.cp
akap = physical_constants.akap
rgas = physical_constants.rgas
karman_const = physical_constants.karman_const

# Latent heats
alhc = physical_constants.alhc
alhs = physical_constants.alhs
alhf = physical_constants.alhf

# Radiation
sbc = physical_constants.sbc
solc = physical_constants.solc

# Water vapor
rd = physical_constants.rd
rv = physical_constants.rv
eps = physical_constants.eps

# Reference temps
t0 = physical_constants.t0
tmelt = physical_constants.tmelt

# Microphysics densities
rhow = physical_constants.rhow
rhoi = physical_constants.rhoi

# Numerical
epsilon = physical_constants.epsilon

# ECHAM-6.3 additions
rgrav = physical_constants.rgrav
cpd = physical_constants.cpd
cvd = physical_constants.cvd  # = cpd - rd
cpv = physical_constants.cpv
cvv = physical_constants.cvv  # = cpv - rv
vtmpc1 = physical_constants.vtmpc1  # = rv/rd - 1
vtmpc2 = physical_constants.vtmpc2  # = cpv/cpd - 1
ak = physical_constants.ak
p0s1_bg = physical_constants.p0s1_bg

# Aliases
rhoh2o = rhow
alv = alhc
als = alhs
alf = alhf
