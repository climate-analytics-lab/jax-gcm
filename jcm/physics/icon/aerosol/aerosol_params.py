"""
JAX-compatible aerosol parameters for MACv2-SP (Simple Plumes) scheme

This module defines the aerosol parameters following the SpeedyPhysics pattern
using tree_math.struct for JAX compatibility. Based on the ICON implementation
in mo_bc_aeropt_splumes.f90.

Date: 2025-01-11
"""

import jax.numpy as jnp
import tree_math
from jax import tree_util


@tree_math.struct
class AerosolParameters:
    """
    Parameters for MACv2-SP (Simple Plumes) aerosol scheme
    
    This implements the simple plume aerosol parametrization based on
    Kinne et al. climatology with 9 anthropogenic plumes and natural
    background aerosol.
    """
    
    # Number of plumes and features
    nplumes: int
    nfeatures: int
    
    # Plume center locations [degrees]
    plume_lat: jnp.ndarray        # (nplumes,) latitude of plume centers
    plume_lon: jnp.ndarray        # (nplumes,) longitude of plume centers
    
    # Vertical distribution parameters (beta function)
    beta_a: jnp.ndarray           # (nplumes,) beta function parameter a
    beta_b: jnp.ndarray           # (nplumes,) beta function parameter b
    
    # Aerosol optical properties at 550nm
    aod_spmx: jnp.ndarray         # (nplumes,) AOD at 550nm for simple plume (maximum)
    aod_fmbg: jnp.ndarray         # (nplumes,) AOD at 550nm for fine mode background
    asy550: jnp.ndarray           # (nplumes,) asymmetry parameter at 550nm
    ssa550: jnp.ndarray           # (nplumes,) single scattering albedo at 550nm
    angstrom: jnp.ndarray         # (nplumes,) Angstrom parameter
    
    # Spatial extent parameters [degrees]
    sig_lon_E: jnp.ndarray        # (nfeatures, nplumes) Eastward longitude extent
    sig_lon_W: jnp.ndarray        # (nfeatures, nplumes) Westward longitude extent
    sig_lat_E: jnp.ndarray        # (nfeatures, nplumes) Southward latitude extent  
    sig_lat_W: jnp.ndarray        # (nfeatures, nplumes) Northward latitude extent
    
    # Feature weights and rotation
    theta: jnp.ndarray            # (nfeatures, nplumes) Rotation angle [radians]
    ftr_weight: jnp.ndarray       # (nfeatures, nplumes) Feature weights
    
    # Natural background AOD
    background_aod: jnp.ndarray   # Background AOD at 550nm (scalar)
    
    @classmethod
    def default(cls):
        """
        Create default MACv2-SP aerosol parameters
        
        These values are representative of the MACv2-SP climatology
        for demonstration purposes. In a full implementation, these
        would be read from netCDF files.
        """
        nplumes = 9
        nfeatures = 2
        
        # Simplified plume centers (representative major emission regions)
        plume_lat = jnp.array([
            25.0,   # East Asia  
            50.0,   # Europe
            35.0,   # North America East
            40.0,   # North America West
            -10.0,  # Biomass burning Africa
            -20.0,  # South America
            20.0,   # India
            15.0,   # Southeast Asia
            30.0    # Middle East
        ])
        
        plume_lon = jnp.array([
            120.0,  # East Asia
            10.0,   # Europe  
            -80.0,  # North America East
            -120.0, # North America West
            20.0,   # Biomass burning Africa
            -60.0,  # South America
            75.0,   # India
            110.0,  # Southeast Asia
            45.0    # Middle East
        ])
        
        # Vertical distribution parameters (beta function)
        # Lower values = more surface-concentrated
        beta_a = jnp.array([1.5, 1.8, 1.6, 1.7, 2.0, 1.9, 1.4, 1.6, 1.5])
        beta_b = jnp.array([3.0, 4.0, 3.5, 3.8, 2.5, 3.2, 4.5, 3.5, 3.8])
        
        # AOD values at 550nm
        aod_spmx = jnp.array([0.30, 0.15, 0.12, 0.08, 0.25, 0.20, 0.35, 0.28, 0.10])
        aod_fmbg = jnp.array([0.02, 0.02, 0.01, 0.01, 0.03, 0.02, 0.04, 0.03, 0.02])
        
        # Optical properties at 550nm
        asy550 = jnp.array([0.65, 0.68, 0.66, 0.67, 0.60, 0.62, 0.63, 0.61, 0.69])
        ssa550 = jnp.array([0.92, 0.95, 0.94, 0.93, 0.85, 0.88, 0.89, 0.86, 0.96])
        angstrom = jnp.array([1.8, 1.5, 1.6, 1.7, 1.2, 1.4, 2.0, 1.9, 1.3])
        
        # Spatial extent parameters [degrees]
        # Feature 1 (primary), Feature 2 (secondary)
        sig_lon_E = jnp.array([
            [15.0, 20.0],  # East Asia
            [12.0, 18.0],  # Europe
            [10.0, 15.0],  # North America East
            [12.0, 20.0],  # North America West
            [20.0, 30.0],  # Biomass burning Africa
            [15.0, 25.0],  # South America
            [8.0, 12.0],   # India
            [12.0, 18.0],  # Southeast Asia
            [10.0, 15.0]   # Middle East
        ]).T
        
        sig_lon_W = jnp.array([
            [12.0, 15.0],  # East Asia
            [10.0, 15.0],  # Europe
            [8.0, 12.0],   # North America East
            [10.0, 18.0],  # North America West
            [18.0, 25.0],  # Biomass burning Africa
            [12.0, 20.0],  # South America
            [6.0, 10.0],   # India
            [10.0, 15.0],  # Southeast Asia
            [8.0, 12.0]    # Middle East
        ]).T
        
        sig_lat_E = jnp.array([
            [8.0, 12.0],   # East Asia
            [10.0, 15.0],  # Europe
            [6.0, 10.0],   # North America East
            [8.0, 12.0],   # North America West
            [15.0, 20.0],  # Biomass burning Africa
            [10.0, 15.0],  # South America
            [5.0, 8.0],    # India
            [8.0, 12.0],   # Southeast Asia
            [6.0, 10.0]    # Middle East
        ]).T
        
        sig_lat_W = jnp.array([
            [6.0, 10.0],   # East Asia
            [8.0, 12.0],   # Europe
            [5.0, 8.0],    # North America East
            [6.0, 10.0],   # North America West
            [12.0, 18.0],  # Biomass burning Africa
            [8.0, 12.0],   # South America
            [4.0, 6.0],    # India
            [6.0, 10.0],   # Southeast Asia
            [5.0, 8.0]     # Middle East
        ]).T
        
        # Rotation angles [radians] 
        theta = jnp.array([
            [0.0, 0.2],    # East Asia
            [0.1, 0.0],    # Europe
            [0.0, 0.1],    # North America East
            [0.2, 0.0],    # North America West
            [0.1, 0.3],    # Biomass burning Africa
            [0.0, 0.1],    # South America
            [0.3, 0.1],    # India
            [0.1, 0.2],    # Southeast Asia
            [0.0, 0.1]     # Middle East
        ]).T
        
        # Feature weights (relative importance of each feature)
        ftr_weight = jnp.array([
            [0.7, 0.3],    # East Asia
            [0.8, 0.2],    # Europe
            [0.75, 0.25],  # North America East
            [0.6, 0.4],    # North America West
            [0.5, 0.5],    # Biomass burning Africa
            [0.7, 0.3],    # South America
            [0.9, 0.1],    # India
            [0.6, 0.4],    # Southeast Asia
            [0.8, 0.2]     # Middle East
        ]).T
        
        return cls(
            nplumes=nplumes,
            nfeatures=nfeatures,
            plume_lat=plume_lat,
            plume_lon=plume_lon,
            beta_a=beta_a,
            beta_b=beta_b,
            aod_spmx=aod_spmx,
            aod_fmbg=aod_fmbg,
            asy550=asy550,
            ssa550=ssa550,
            angstrom=angstrom,
            sig_lon_E=sig_lon_E,
            sig_lon_W=sig_lon_W,
            sig_lat_E=sig_lat_E,
            sig_lat_W=sig_lat_W,
            theta=theta,
            ftr_weight=ftr_weight,
            background_aod=jnp.array(0.02)
        )
    
    def isnan(self):
        """Check for NaN values in parameters"""
        return tree_util.tree_map(jnp.isnan, self)