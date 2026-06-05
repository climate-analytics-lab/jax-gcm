"""Simple chemistry schemes for ECHAM physics

This module implements basic chemistry parameterizations including
fixed ozone distribution and basic methane oxidation.

Date: 2025-01-15
"""

import jax.numpy as jnp
from typing import NamedTuple, Tuple
import tree_math

import jcm.constants as c


@tree_math.struct
class ChemistryParameters:
    """Configuration parameters for chemistry schemes"""
    
    # Ozone parameters
    ozone_scale_height: float      # Ozone scale height (m)
    ozone_max_vmr: float          # Maximum ozone volume mixing ratio (ppbv)
    ozone_tropopause_height: float # Height of ozone maximum (m)
    ozone_stratosphere_coeff: float # Stratospheric ozone coefficient
    
    # Methane parameters  
    methane_surface_vmr: float     # Surface methane VMR (ppbv)
    methane_lifetime: float        # Methane lifetime (s)
    methane_oh_scaling: float      # OH scaling factor
    
    # CO2 parameters
    co2_vmr: float                 # CO2 volume mixing ratio (ppmv)
    co2_growth_rate: float         # CO2 growth rate (ppmv/year)
    
    @classmethod
    def default(cls) -> 'ChemistryParameters':
        """Return default chemistry parameters"""
        return cls(
            ozone_scale_height=jnp.array(7000.0),      # 7 km
            ozone_max_vmr=jnp.array(8000.0),           # 8 ppmv
            ozone_tropopause_height=jnp.array(20000.0), # 20 km
            ozone_stratosphere_coeff=jnp.array(0.1),
            methane_surface_vmr=jnp.array(1900.0),     # 1.9 ppmv
            methane_lifetime=jnp.array(9.0 * 365.25 * 24 * 3600), # 9 years
            methane_oh_scaling=jnp.array(1.0),
            co2_vmr=jnp.array(420.0),                   # 420 ppmv
            co2_growth_rate=jnp.array(2.5)             # 2.5 ppmv/year
        )


class ChemistryState(NamedTuple):
    """Chemistry state variables and diagnostics"""
    
    # Gas concentrations (volume mixing ratios)
    ozone_vmr: jnp.ndarray          # Ozone VMR (ppbv)
    methane_vmr: jnp.ndarray        # Methane VMR (ppbv)
    co2_vmr: jnp.ndarray           # CO2 VMR (ppmv)
    
    # Production/loss rates
    ozone_production: jnp.ndarray   # Ozone production rate (ppbv/s)
    ozone_loss: jnp.ndarray        # Ozone loss rate (ppbv/s)
    methane_loss: jnp.ndarray      # Methane loss rate (ppbv/s)


class ChemistryTendencies(NamedTuple):
    """Tendencies from chemistry processes"""

    # Trace gas tendencies (mixing ratio per second)
    ozone_tend: jnp.ndarray         # Ozone tendency (ppbv/s)
    methane_tend: jnp.ndarray       # Methane tendency (ppbv/s)
    co2_tend: jnp.ndarray          # CO2 tendency (ppmv/s)


@tree_math.struct
class ChemistryData:
    """Diagnostic sub-struct for the chemistry diagnostic-dict slot.

    Seeded by ``EchamBoundaryConditions`` (which fills CO2/CH4/O3 VMRs
    from forcing) and consumed by the radiation terms; the
    ``SimpleChemistry`` term also writes its production/loss diagnostics
    here. Lives next to the chemistry scheme so a future replacement
    chemistry term can extend or replace this struct without reaching
    into the ECHAM tree.
    """

    # Gas concentrations (volume mixing ratios)
    ozone_vmr: jnp.ndarray           # Ozone VMR [ppbv] (nlev, ncols)
    methane_vmr: jnp.ndarray         # Methane VMR [ppbv] (nlev, ncols)
    co2_vmr: jnp.ndarray            # CO2 VMR [ppmv] (nlev, ncols)

    # Production/loss rates
    ozone_production: jnp.ndarray    # Ozone production rate [ppbv/s] (nlev, ncols)
    ozone_loss: jnp.ndarray         # Ozone loss rate [ppbv/s] (nlev, ncols)
    methane_loss: jnp.ndarray       # Methane loss rate [ppbv/s] (nlev, ncols)

    # Surface emissions/deposition
    methane_surface_flux: jnp.ndarray  # Surface methane flux [ppbv m/s] (ncols,)
    ozone_dry_deposition: jnp.ndarray  # Ozone dry deposition velocity [m/s] (ncols,)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        # Chemistry fields should be in column format (nlev, ncols)
        # where ncols = nlat * nlon.
        if len(nodal_shape) == 2:
            nlat, nlon = nodal_shape
            ncols = nlat * nlon
            column_shape = (nlev, ncols)
            surface_shape = (ncols,)
        else:
            column_shape = (nlev,) + nodal_shape
            surface_shape = nodal_shape

        return cls(
            ozone_vmr=jnp.zeros(column_shape),
            methane_vmr=jnp.zeros(column_shape),
            co2_vmr=jnp.zeros(column_shape),
            ozone_production=jnp.zeros(column_shape),
            ozone_loss=jnp.zeros(column_shape),
            methane_loss=jnp.zeros(column_shape),
            methane_surface_flux=jnp.zeros(surface_shape),
            ozone_dry_deposition=jnp.zeros(surface_shape),
        )

    def copy(self, **kwargs):
        new_data = {
            'ozone_vmr': self.ozone_vmr,
            'methane_vmr': self.methane_vmr,
            'co2_vmr': self.co2_vmr,
            'ozone_production': self.ozone_production,
            'ozone_loss': self.ozone_loss,
            'methane_loss': self.methane_loss,
            'methane_surface_flux': self.methane_surface_flux,
            'ozone_dry_deposition': self.ozone_dry_deposition,
        }
        new_data.update(kwargs)
        return ChemistryData(**new_data)


def compute_height_from_pressure(
    pressure: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    temperature: jnp.ndarray
) -> jnp.ndarray:
    """Compute approximate height from pressure using hydrostatic equation
    
    Args:
        pressure: Pressure at each level (Pa) [nlev, ncols]
        surface_pressure: Surface pressure (Pa) [ncols]
        temperature: Temperature (K) [nlev, ncols]
        
    Returns:
        Height (m) [nlev, ncols]

    """
    # Mean temperature for each layer
    temp_mean = jnp.mean(temperature, axis=0, keepdims=True)
    
    # Scale height H = R*T/g
    scale_height = c.rd * temp_mean / c.grav
    
    # Height from hydrostatic equation: h = H * ln(p_sfc/p)
    height = scale_height * jnp.log(surface_pressure[None, :] / pressure)
    
    return height


def fixed_ozone_distribution(
    pressure: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    config: ChemistryParameters
) -> jnp.ndarray:
    """Calculate fixed ozone distribution based on pressure/height
    
    Uses a simple analytical profile with maximum in the stratosphere
    and exponential decay above and below.
    
    Args:
        pressure: Pressure (Pa) [nlev, ncols]
        surface_pressure: Surface pressure (Pa) [ncols]
        temperature: Temperature (K) [nlev, ncols]
        config: Chemistry configuration
        
    Returns:
        Ozone volume mixing ratio (ppbv) [nlev, ncols]

    """
    # Calculate approximate height
    height = compute_height_from_pressure(pressure, surface_pressure, temperature)
    
    # Height relative to ozone maximum
    height_rel = height - config.ozone_tropopause_height
    
    # Ozone profile with maximum at tropopause height
    # Exponential decay above and below with different scale heights
    ozone_profile = jnp.where(
        height_rel <= 0,
        # Troposphere: linear increase to maximum
        config.ozone_max_vmr * (1.0 + height_rel / config.ozone_tropopause_height),
        # Stratosphere: exponential decay
        config.ozone_max_vmr * jnp.exp(-height_rel / config.ozone_scale_height)
    )
    
    # Ensure positive values
    ozone_profile = jnp.maximum(ozone_profile, 10.0)  # Minimum 10 ppbv
    
    return ozone_profile


def simple_methane_chemistry(
    pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    methane_vmr: jnp.ndarray,
    dt: float,
    config: ChemistryParameters
) -> jnp.ndarray:
    """Compute methane chemistry with exponential decay
    
    Args:
        pressure: Pressure (Pa) [nlev, ncols]
        temperature: Temperature (K) [nlev, ncols]
        methane_vmr: Current methane VMR (ppbv) [nlev, ncols]
        dt: Time step (s)
        config: Chemistry configuration
        
    Returns:
        Methane loss rate (ppbv/s) [nlev, ncols]

    """
    # Temperature-dependent loss rate
    # Increases with temperature (simplified OH chemistry)
    temp_factor = jnp.exp((temperature - 273.15) / 50.0)  # Arrhenius-like
    
    # Pressure-dependent loss (more loss at lower pressures)
    pressure_factor = jnp.exp(-pressure / 50000.0)  # More loss in upper atmosphere
    
    # Overall loss rate
    loss_rate_coefficient = (config.methane_oh_scaling * temp_factor * pressure_factor / 
                           config.methane_lifetime)
    
    # Methane loss rate
    methane_loss = methane_vmr * loss_rate_coefficient
    
    return methane_loss


def simple_chemistry(
    pressure: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    current_ozone: jnp.ndarray,
    current_methane: jnp.ndarray,
    dt: float,
    config: ChemistryParameters = None
) -> Tuple[ChemistryTendencies, ChemistryState]:
    """Compute chemistry with fixed ozone and methane decay
    
    Args:
        pressure: Pressure (Pa) [nlev, ncols]
        surface_pressure: Surface pressure (Pa) [ncols]
        temperature: Temperature (K) [nlev, ncols]
        current_ozone: Current ozone VMR (ppbv) [nlev, ncols]
        current_methane: Current methane VMR (ppbv) [nlev, ncols]
        dt: Time step (s)
        config: Chemistry configuration
        
    Returns:
        Tuple of (tendencies, state)

    """
    if config is None:
        config = ChemistryParameters.default()
    
    nlev, ncols = pressure.shape
    
    # Calculate target ozone distribution
    target_ozone = fixed_ozone_distribution(
        pressure, surface_pressure, temperature, config
    )
    
    # Relax current ozone toward target distribution
    # Use 10-day relaxation time scale
    ozone_relaxation_time = 10.0 * 24.0 * 3600.0  # 10 days
    ozone_tendency = (target_ozone - current_ozone) / ozone_relaxation_time
    
    # Calculate methane loss
    methane_loss = simple_methane_chemistry(
        pressure, temperature, current_methane, dt, config
    )
    methane_tendency = -methane_loss
    
    # CO2 is fixed (no tendency)
    co2_tendency = jnp.zeros_like(current_ozone)
    co2_vmr = jnp.ones_like(current_ozone) * config.co2_vmr
    
    # Create tendencies
    tendencies = ChemistryTendencies(
        ozone_tend=ozone_tendency,
        methane_tend=methane_tendency,
        co2_tend=co2_tendency
    )
    
    # Create state
    state = ChemistryState(
        ozone_vmr=current_ozone,
        methane_vmr=current_methane,
        co2_vmr=co2_vmr,
        ozone_production=jnp.maximum(ozone_tendency, 0.0),
        ozone_loss=jnp.maximum(-ozone_tendency, 0.0),
        methane_loss=methane_loss
    )
    
    return tendencies, state


def initialize_chemistry_tracers(
    pressure: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    config: ChemistryParameters = None
) -> ChemistryState:
    """Initialize chemistry tracers with reasonable distributions
    
    Args:
        pressure: Pressure (Pa) [nlev, ncols]
        surface_pressure: Surface pressure (Pa) [ncols]
        temperature: Temperature (K) [nlev, ncols]
        config: Chemistry configuration
        
    Returns:
        Initial chemistry state

    """
    if config is None:
        config = ChemistryParameters.default()
    
    nlev, ncols = pressure.shape
    
    # Initialize ozone with fixed distribution
    ozone_vmr = fixed_ozone_distribution(
        pressure, surface_pressure, temperature, config
    )
    
    # Initialize methane with exponential decay from surface
    # Higher concentration at surface, decreasing with height
    height = compute_height_from_pressure(pressure, surface_pressure, temperature)
    methane_vmr = config.methane_surface_vmr * jnp.exp(-height / 8000.0)  # 8 km scale height
    
    # Initialize CO2 as constant
    co2_vmr = jnp.ones((nlev, ncols)) * config.co2_vmr

    return ChemistryState(
        ozone_vmr=ozone_vmr,
        methane_vmr=methane_vmr,
        co2_vmr=co2_vmr,
        ozone_production=jnp.zeros((nlev, ncols)),
        ozone_loss=jnp.zeros((nlev, ncols)),
        methane_loss=jnp.zeros((nlev, ncols))
    )


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class SimpleChemistry(PhysicsTerm):
    """Simple chemistry as a composable PhysicsTerm.

    Wraps :func:`simple_chemistry`. Reads the current chemistry typed
    sub-struct from ``diagnostics["chemistry"]`` (initialised by
    :class:`~jcm.physics.forcing.echam_boundary_conditions.EchamBoundaryConditions`
    each step), computes the relaxation-to-climatology / linear-decay
    update, and writes the new ``ChemistryState`` back to the same public
    key. Returns zero atmospheric tendency — chemistry doesn't directly
    perturb the dynamics here (its ozone/CO2 fields are consumed by the
    radiation term to influence heating rates).

    Reads ``pressure_full`` and ``surface_pressure`` from the moist-air
    diagnostics dict and the model timestep from
    ``diagnostics["_dt_seconds"]`` (injected by ``ComposablePhysics``).
    """

    name: ClassVar[str] = "simple_chemistry"
    category: ClassVar[str] = "chemistry"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "surface_pressure", "chemistry",
    )
    provides: ClassVar[tuple[str, ...]] = ("chemistry",)

    def __init__(self, params: ChemistryParameters | None = None):
        """Hold the scheme-native :class:`ChemistryParameters`."""
        self.params = nnx.Param(params or ChemistryParameters.default())

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Update chemistry sub-struct from the previous step's values."""
        params = self.params.get_value()
        dt = diagnostics["_dt_seconds"]

        chemistry = diagnostics["chemistry"]
        _tend, new_state = simple_chemistry(
            pressure=diagnostics["pressure_full"],
            surface_pressure=diagnostics["surface_pressure"],
            temperature=state.temperature,
            current_ozone=chemistry.ozone_vmr,
            current_methane=chemistry.methane_vmr,
            dt=dt,
            config=params,
        )

        chemistry = chemistry.copy(
            ozone_vmr=new_state.ozone_vmr,
            methane_vmr=new_state.methane_vmr,
            co2_vmr=new_state.co2_vmr,
            ozone_production=new_state.ozone_production,
            ozone_loss=new_state.ozone_loss,
            methane_loss=new_state.methane_loss,
        )

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {**diagnostics, "chemistry": chemistry}