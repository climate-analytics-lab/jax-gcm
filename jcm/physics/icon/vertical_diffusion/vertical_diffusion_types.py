"""
Data structures and types for vertical diffusion and boundary layer physics.

This module defines the key data structures used in vertical diffusion
calculations, following the ICON model structure.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp
import tree_math


@tree_math.struct
class VDiffParameters:
    """Parameters for vertical diffusion scheme."""
    
    # Implicitness factors (following ICON's tpfac1, tpfac2, tpfac3)
    tpfac1: float       # Factor for new timestep (implicit)
    tpfac2: float       # Factor for old timestep (explicit part)
    tpfac3: float       # Factor for time interpolation
    
    # Turbulence parameters
    totte_min: float    # Minimum TTE value
    z0m_min: float      # Minimum roughness length
    cchar: float        # Charnock constant for ocean roughness
    
    # Surface types
    nsfc_type: int      # Number of surface types (water, ice, land)
    iwtr: int           # Index for water surface
    iice: int           # Index for ice surface
    ilnd: int           # Index for land surface
    
    # Vertical structure
    itop: int           # Top level for turbulence calculation

    @classmethod
    def default(cls, tpfac1=1.0, tpfac2=0.0, tpfac3=0.0,
                 totte_min=1.0e-6, z0m_min=1.0e-5, cchar=0.018,
                 nsfc_type=3, iwtr=0, iice=1, ilnd=2, itop=1) -> 'VDiffParameters':
        """Return default vertical diffusion parameters"""
        return cls(
            tpfac1=jnp.array(tpfac1),
            tpfac2=jnp.array(tpfac2),
            tpfac3=jnp.array(tpfac3),
            totte_min=jnp.array(totte_min),
            z0m_min=jnp.array(z0m_min),
            cchar=jnp.array(cchar),
            nsfc_type=nsfc_type,  # Keep as Python int for static shape
            iwtr=iwtr,  # Keep as Python int for indexing
            iice=iice,  # Keep as Python int for indexing
            ilnd=ilnd,  # Keep as Python int for indexing
            itop=itop   # Keep as Python int for indexing
        )


class VDiffState(NamedTuple):
    """Atmospheric state variables for vertical diffusion."""
    
    # Dynamical variables
    u: jnp.ndarray             # Zonal wind [m/s] (ncol, nlev)
    v: jnp.ndarray             # Meridional wind [m/s] (ncol, nlev)
    temperature: jnp.ndarray    # Temperature [K] (ncol, nlev)
    
    # Moisture variables
    qv: jnp.ndarray            # Water vapor mixing ratio [kg/kg] (ncol, nlev)
    qc: jnp.ndarray            # Cloud water mixing ratio [kg/kg] (ncol, nlev)
    qi: jnp.ndarray            # Cloud ice mixing ratio [kg/kg] (ncol, nlev)
    
    # Atmospheric structure
    pressure_full: jnp.ndarray      # Full level pressure [Pa] (ncol, nlev)
    pressure_half: jnp.ndarray      # Half level pressure [Pa] (ncol, nlev+1)
    geopotential: jnp.ndarray       # Geopotential [m²/s²] (ncol, nlev)
    
    # Air mass
    air_mass: jnp.ndarray          # Moist air mass [kg/m²] (ncol, nlev)
    dry_air_mass: jnp.ndarray      # Dry air mass [kg/m²] (ncol, nlev)
    
    # Surface properties
    surface_temperature: jnp.ndarray  # Surface temperature [K] (ncol, nsfc_type)
    surface_fraction: jnp.ndarray     # Surface type fraction [-] (ncol, nsfc_type)
    roughness_length: jnp.ndarray     # Roughness length [m] (ncol, nsfc_type)
    
    # Geometric heights
    height_full: jnp.ndarray       # Full level height [m] (ncol, nlev)
    height_half: jnp.ndarray       # Half level height [m] (ncol, nlev+1)
    
    # Turbulence variables
    tke: jnp.ndarray              # Turbulent kinetic energy [m²/s²] (ncol, nlev)
    thv_variance: jnp.ndarray     # Variance of theta_v [K²] (ncol, nlev)
    
    # Ocean surface velocities (for momentum exchange)
    ocean_u: jnp.ndarray          # Ocean u-velocity [m/s] (ncol,)
    ocean_v: jnp.ndarray          # Ocean v-velocity [m/s] (ncol,)
    
    # Tracers (optional)
    tracers: Optional[jnp.ndarray] = None  # Additional tracers (ncol, nlev, ntrac)


class VDiffTendencies(NamedTuple):
    """Tendencies computed by vertical diffusion."""
    
    # Momentum tendencies
    u_tendency: jnp.ndarray        # du/dt [m/s²] (ncol, nlev)
    v_tendency: jnp.ndarray        # dv/dt [m/s²] (ncol, nlev)
    
    # Thermodynamic tendencies
    temperature_tendency: jnp.ndarray  # dT/dt [K/s] (ncol, nlev)
    heating_rate: jnp.ndarray         # Heating rate [W/m²] (ncol, nlev)
    
    # Moisture tendencies
    qv_tendency: jnp.ndarray       # dqv/dt [kg/kg/s] (ncol, nlev)
    qc_tendency: jnp.ndarray       # dqc/dt [kg/kg/s] (ncol, nlev)
    qi_tendency: jnp.ndarray       # dqi/dt [kg/kg/s] (ncol, nlev)
    
    # Turbulence tendencies
    tke_tendency: jnp.ndarray      # dTKE/dt [m²/s³] (ncol, nlev)
    thv_var_tendency: jnp.ndarray  # d(theta_v_var)/dt [K²/s] (ncol, nlev)
    
    # Tracer tendencies (optional)
    tracer_tendencies: Optional[jnp.ndarray] = None  # (ncol, nlev, ntrac)


class VDiffDiagnostics(NamedTuple):
    """Diagnostic variables from vertical diffusion."""
    
    # Exchange coefficients
    exchange_coeff_momentum: jnp.ndarray  # Momentum exchange coeff [m²/s] (ncol, nlev)
    exchange_coeff_heat: jnp.ndarray      # Heat exchange coeff [m²/s] (ncol, nlev)
    exchange_coeff_moisture: jnp.ndarray  # Moisture exchange coeff [m²/s] (ncol, nlev)
    
    # Surface exchange coefficients
    surface_exchange_heat: jnp.ndarray    # Surface heat exchange [m²/s] (ncol, nsfc_type)
    surface_exchange_moisture: jnp.ndarray # Surface moisture exchange [m²/s] (ncol, nsfc_type)
    
    # Boundary layer diagnostics
    boundary_layer_height: jnp.ndarray    # PBL height [m] (ncol,)
    friction_velocity: jnp.ndarray        # u* [m/s] (ncol,)
    convective_velocity: jnp.ndarray      # w* [m/s] (ncol,)
    
    # Richardson number
    richardson_number: jnp.ndarray        # Bulk Richardson number [-] (ncol, nlev)
    
    # Mixing length
    mixing_length: jnp.ndarray           # Mixing length [m] (ncol, nlev)
    
    # Surface fluxes
    surface_momentum_flux_u: jnp.ndarray  # u-momentum flux [N/m²] (ncol,)
    surface_momentum_flux_v: jnp.ndarray  # v-momentum flux [N/m²] (ncol,)
    surface_heat_flux: jnp.ndarray        # Sensible heat flux [W/m²] (ncol,)
    surface_moisture_flux: jnp.ndarray    # Latent heat flux [W/m²] (ncol,)
    
    # Energy dissipation
    kinetic_energy_dissipation: jnp.ndarray  # KE dissipation [W/m²] (ncol,)


class VDiffMatrixSystem(NamedTuple):
    """Tridiagonal matrix system for vertical diffusion solver."""
    
    # Coefficient matrices for different variable types
    # Shape: (ncol, nlev, 3, nmatrix) where 3 = [sub, diag, super]
    matrix_coeffs: jnp.ndarray
    
    # Bottom row matrices for surface boundary conditions
    # Shape: (ncol, 3, nsfc_type, nvar_surface)
    matrix_bottom: jnp.ndarray
    
    # Right-hand side vectors
    # Shape: (ncol, nlev, nvar_total)
    rhs_vectors: jnp.ndarray
    
    # Surface RHS vectors
    # Shape: (ncol, nsfc_type, nvar_surface)
    rhs_surface: jnp.ndarray
    
    # Matrix indices for different variable types
    # These map variables to their matrix type
    variable_to_matrix: jnp.ndarray
    
    # Variable indices
    iu: int = 0      # u-wind index
    iv: int = 1      # v-wind index
    ih: int = 2      # heat index
    iqv: int = 3     # moisture index
    iqc: int = 4     # cloud water index
    iqi: int = 5     # cloud ice index
    itke: int = 6    # TKE index
    ithv: int = 7    # theta_v variance index
    
    # Matrix type indices
    imu: int = 0     # momentum matrix
    imh: int = 1     # heat matrix
    imqv: int = 2    # moisture matrix
    imqc: int = 3    # cloud water matrix
    imtke: int = 4   # TKE matrix
    imthv: int = 5   # theta_v variance matrix