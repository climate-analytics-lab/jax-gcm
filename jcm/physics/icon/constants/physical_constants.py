"""
Physical constants for ICON atmospheric physics

This module contains physical constants used throughout the ICON physics
parameterizations. These constants are based on the ICON mo_physical_constants
module and are organized by category.

Date: 2025-01-09
"""

import jax.numpy as jnp
from typing import NamedTuple

class PhysicalConstants(NamedTuple):
    """Physical constants for ICON atmospheric physics"""
    
    # Fundamental constants
    rearth: float = 6.371e+6      # Radius of Earth (m)
    omega: float = 7.292e-05      # Rotation rate of Earth (rad/s)
    grav: float = 9.81            # Gravitational acceleration (m/s²)
    
    # Thermodynamic constants
    p0: float = 1.0e+5            # Reference pressure (Pa)
    cp: float = 1004.0            # Specific heat at constant pressure (J/K/kg)
    akap: float = 2.0/7.0         # 1 - 1/gamma (kappa = R/cp)
    rgas: float = 287.0           # Gas constant per unit mass for dry air (J/K/kg)
    karman_const: float = 0.4     # von Kármán constant (dimensionless)
    
    # Latent heat constants (J/kg)
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
    
    @classmethod
    def default(cls) -> 'PhysicalConstants':
        """Return default physical constants"""
        return cls()

# Global instance of physical constants
physical_constants = PhysicalConstants.default()

# Export individual constants for convenience
rearth = physical_constants.rearth
omega = physical_constants.omega
grav = physical_constants.grav
p0 = physical_constants.p0
cp = physical_constants.cp
akap = physical_constants.akap
rgas = physical_constants.rgas
karman_const = physical_constants.karman_const
alhc = physical_constants.alhc
alhs = physical_constants.alhs
alhf = physical_constants.alhf
sbc = physical_constants.sbc
solc = physical_constants.solc
rd = physical_constants.rd
rv = physical_constants.rv
eps = physical_constants.eps
t0 = physical_constants.t0
tmelt = physical_constants.tmelt
rhow = physical_constants.rhow
rhoi = physical_constants.rhoi
epsilon = physical_constants.epsilon

# Aliases for compatibility
rhoh2o = rhow  # Density of water
alv = alhc     # Latent heat of vaporization
als = alhs     # Latent heat of sublimation  
alf = alhf     # Latent heat of fusion