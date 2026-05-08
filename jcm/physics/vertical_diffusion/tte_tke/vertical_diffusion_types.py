"""Data structures and types for vertical diffusion and boundary layer physics.

This module defines the key data structures used in vertical diffusion
calculations, following the ICON model structure.
"""

from typing import NamedTuple
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

    # Surface-layer scheme selector (int flag — JAX won't trace strings).
    # 0 = "businger_dyer" (default; preserves original ICON-port behavior)
    # 1 = "echam_louis"   (faithful port of mo_turbulence_diag)
    # See ``surface_layer.py`` for both implementations.
    surface_layer_scheme: int

    # ECHAM-Louis surface-layer scheme tunables (used when
    # ``surface_layer_scheme == SCHEME_ECHAM_LOUIS``). Defaults match
    # ECHAM ``mo_echam_vdiff_params``.
    surface_layer_fsl: float   # mid-surface-layer weighting
                               # (fraction of air; 1-fsl is surface)
    louis_cb: float            # Louis (1979) near-neutrality parameter
    louis_cc: float            # Louis (1979) unstable-branch parameter

    SCHEME_BUSINGER_DYER = 0
    SCHEME_ECHAM_LOUIS = 1

    @classmethod
    def default(cls, tpfac1=1.5, tpfac2=0.667, tpfac3=0.333,
                 totte_min=1.0e-6, z0m_min=1.0e-5, cchar=0.018,
                 nsfc_type=3, iwtr=0, iice=1, ilnd=2, itop=1,
                 surface_layer_scheme=0,
                 surface_layer_fsl=0.4,
                 louis_cb=5.0, louis_cc=5.0) -> 'VDiffParameters':
        """Return default vertical diffusion parameters.

        ``surface_layer_scheme`` accepts either the int constant
        (``SCHEME_BUSINGER_DYER`` / ``SCHEME_ECHAM_LOUIS``) or the
        string aliases ``"businger_dyer"`` / ``"echam_louis"``.
        """
        if isinstance(surface_layer_scheme, str):
            scheme_map = {
                "businger_dyer": cls.SCHEME_BUSINGER_DYER,
                "echam_louis":   cls.SCHEME_ECHAM_LOUIS,
            }
            surface_layer_scheme = scheme_map[surface_layer_scheme]

        return cls(
            tpfac1=jnp.array(tpfac1),
            tpfac2=jnp.array(tpfac2),
            tpfac3=jnp.array(tpfac3),
            totte_min=jnp.array(totte_min),
            z0m_min=jnp.array(z0m_min),
            cchar=jnp.array(cchar),
            nsfc_type=nsfc_type,
            iwtr=iwtr,
            iice=iice,
            ilnd=ilnd,
            itop=itop,
            surface_layer_scheme=int(surface_layer_scheme),
            surface_layer_fsl=jnp.array(surface_layer_fsl),
            louis_cb=jnp.array(louis_cb),
            louis_cc=jnp.array(louis_cc),
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
    roughness_length: jnp.ndarray     # Momentum roughness z0m [m] (ncol, nsfc_type)
    roughness_heat: jnp.ndarray       # Heat roughness z0h [m] (ncol, nsfc_type) —
                                      # tile-specific. ECHAM uses
                                      # ``exp(2 - 86·z0^0.375)`` over open
                                      # water, ``z0`` over ice, and the
                                      # JSBACH ``paz0lh`` over land.
    surface_wetness: jnp.ndarray      # Effective surface saturation [-]
                                      # (ncol, nsfc_type). 1.0 means fully
                                      # saturated (open water / ice); over
                                      # land it's the JSBACH ``cair`` /
                                      # ``csat``-style fraction derived from
                                      # the boundary soil moisture.
    
    # Geometric heights
    height_full: jnp.ndarray       # Full level height [m] (ncol, nlev)
    height_half: jnp.ndarray       # Half level height [m] (ncol, nlev+1)
    
    # Turbulence variables
    tke: jnp.ndarray              # Turbulent kinetic energy [m²/s²] (ncol, nlev)
    thv_variance: jnp.ndarray     # Variance of theta_v [K²] (ncol, nlev)
    
    # Ocean surface velocities (for momentum exchange)
    ocean_u: jnp.ndarray          # Ocean u-velocity [m/s] (ncol,)
    ocean_v: jnp.ndarray          # Ocean v-velocity [m/s] (ncol,)
    


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


@tree_math.struct
class VerticalDiffusionData:
    """Diagnostics produced by the TTE-TKE vertical-diffusion term.

    Lives next to :class:`VDiffParameters` so the TTE-TKE scheme owns
    one self-contained type module.
    """

    # Exchange coefficients
    km: jnp.ndarray                  # Momentum exchange coeff [m²/s] (nlev+1, ncols)
    kh: jnp.ndarray                  # Heat exchange coeff [m²/s] (nlev+1, ncols)

    # Surface exchange coefficients (per surface type)
    surface_exchange_heat: jnp.ndarray      # Surface heat exchange [m²/s] (ncols, nsfc_type)
    surface_exchange_moisture: jnp.ndarray  # Surface moisture exchange [m²/s] (ncols, nsfc_type)
    surface_exchange_momentum: jnp.ndarray  # Surface momentum exchange [m²/s] (ncols, nsfc_type)

    # Turbulent kinetic energy
    tke: jnp.ndarray                 # TKE [m²/s²] (nlev, ncols)

    # Boundary layer diagnostics
    pbl_height: jnp.ndarray          # PBL height [m] (ncols,)
    surface_friction_velocity: jnp.ndarray  # u* [m/s] (ncols,)
    monin_obukhov_length: jnp.ndarray       # L [m] (ncols,)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        nsfc_type = 3  # water, ice, land
        return cls(
            km=jnp.zeros((nlev + 1,) + nodal_shape),
            kh=jnp.zeros((nlev + 1,) + nodal_shape),
            surface_exchange_heat=jnp.zeros(nodal_shape + (nsfc_type,)),
            surface_exchange_moisture=jnp.zeros(nodal_shape + (nsfc_type,)),
            surface_exchange_momentum=jnp.zeros(nodal_shape + (nsfc_type,)),
            tke=jnp.zeros((nlev,) + nodal_shape),
            pbl_height=jnp.zeros(nodal_shape),
            surface_friction_velocity=jnp.zeros(nodal_shape),
            monin_obukhov_length=jnp.zeros(nodal_shape),
        )

    def copy(self, **kwargs):
        new_data = {
            'km': self.km,
            'kh': self.kh,
            'surface_exchange_heat': self.surface_exchange_heat,
            'surface_exchange_moisture': self.surface_exchange_moisture,
            'surface_exchange_momentum': self.surface_exchange_momentum,
            'tke': self.tke,
            'pbl_height': self.pbl_height,
            'surface_friction_velocity': self.surface_friction_velocity,
            'monin_obukhov_length': self.monin_obukhov_length,
        }
        new_data.update(kwargs)
        return VerticalDiffusionData(**new_data)