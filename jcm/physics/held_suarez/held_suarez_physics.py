import jax.numpy as jnp
import tree_math
from typing import Tuple
from dinosaur.scales import units
from dinosaur import coordinate_systems
from jcm.geometry import Geometry
from jcm.boundaries import BoundaryData
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.model import get_coords, PHYSICS_SPECS
from jcm.date import DateData

Quantity = units.Quantity

@tree_math.struct
class SurfaceFluxParameters:
    """Minimal surface flux parameters for boundary compatibility."""
    hdrag: jnp.ndarray  # height scale for orographic drag [m]
    
    @classmethod
    def default(cls):
        """Return default surface flux parameters."""
        return cls(hdrag=jnp.array(2000.0))

@tree_math.struct
class Parameters:
    """Held-Suarez parameters."""
    sigma_b: jnp.ndarray  # sigma level of effective planetary boundary layer
    kf: jnp.ndarray  # coefficient of friction for Rayleigh drag [1/s]
    ka: jnp.ndarray  # coefficient of thermal relaxation in upper atmosphere [1/s]
    ks: jnp.ndarray  # coefficient of thermal relaxation at earth surface on the equator [1/s]
    minT: jnp.ndarray  # lower temperature bound of radiative equilibrium [K]
    maxT: jnp.ndarray  # upper temperature bound of radiative equilibrium [K]
    dTy: jnp.ndarray  # horizontal temperature variation of radiative equilibrium [K]
    dThz: jnp.ndarray  # vertical temperature variation of radiative equilibrium [K]
    surface_flux: SurfaceFluxParameters  # Minimal surface flux params for boundary compatibility
    
    @classmethod
    def default(cls):
        """Return default Held-Suarez parameters."""
        return cls(
            sigma_b=jnp.array(0.7),
            kf=PHYSICS_SPECS.nondimensionalize(1 / (1 * units.day)),
            ka=PHYSICS_SPECS.nondimensionalize(1 / (40 * units.day)),
            ks=PHYSICS_SPECS.nondimensionalize(1 / (4 * units.day)),
            minT=PHYSICS_SPECS.nondimensionalize(200 * units.degK),
            maxT=PHYSICS_SPECS.nondimensionalize(315 * units.degK),
            dTy=PHYSICS_SPECS.nondimensionalize(60 * units.degK),
            dThz=PHYSICS_SPECS.nondimensionalize(10 * units.degK),
            surface_flux=SurfaceFluxParameters.default(),
        )

class HeldSuarezPhysics(Physics):
    parameters: Parameters
    write_output: bool
    coords: coordinate_systems.CoordinateSystem
    sigma: jnp.ndarray
    lat: jnp.ndarray

    def __init__(self,
        coords: coordinate_systems.CoordinateSystem = None,
        parameters: Parameters = None,
        write_output: bool = False,
    ) -> None:
        """Initialize Held-Suarez.

        Args:
            coords: horizontal and vertical discretization. If None, will use get_coords().
            parameters: Held-Suarez parameters. If None, will use default parameters.
            write_output: Flag to indicate whether physics output should be written to predictions.
        """
        self.write_output = write_output
        self.coords = coords if coords is not None else get_coords()
        self.parameters = parameters if parameters is not None else Parameters.default()
        # Coordinates
        self.sigma = self.coords.vertical.centers
        self.lat = self.coords.horizontal.latitudes[jnp.newaxis]

    def equilibrium_temperature(self, normalized_surface_pressure):
        p_over_p0 = (
            self.sigma[:, jnp.newaxis, jnp.newaxis] * normalized_surface_pressure[jnp.newaxis]
        )
        temperature = p_over_p0**PHYSICS_SPECS.kappa * (
            self.parameters.maxT
            - self.parameters.dTy * jnp.sin(self.lat[jnp.newaxis]) ** 2
            - self.parameters.dThz * jnp.log(p_over_p0) * jnp.cos(self.lat[jnp.newaxis]) ** 2
        )
        return jnp.maximum(self.parameters.minT, temperature)

    def kv(self):
        kv_coeff = self.parameters.kf * (
            jnp.maximum(0, (self.sigma - self.parameters.sigma_b) / (1 - self.parameters.sigma_b))
        )
        return kv_coeff[:, jnp.newaxis, jnp.newaxis]

    def kt(self):
        cutoff = jnp.maximum(0, (self.sigma - self.parameters.sigma_b) / (1 - self.parameters.sigma_b))
        return self.parameters.ka + (self.parameters.ks - self.parameters.ka) * (
            cutoff[:, jnp.newaxis, jnp.newaxis] * jnp.cos(self.lat[jnp.newaxis]) ** 4
        )

    def compute_tendencies(
        self,
        state: PhysicsState,
        boundaries: BoundaryData,
        geometry: Geometry,
        date: DateData,
    ) -> Tuple[PhysicsTendency, None]:
        """
        Compute the physical tendencies given the current state and data structs. Tendencies are computed as a Held-Suarez forcing.

        Args:
            state: Current state variables
            boundaries: Boundary data (unused)
            geometry: Geometry data (unused)
            date: Date data (unused)

        Returns:
            Physical tendencies in PhysicsTendency format
            Object containing physics data (unused)
        """
        Teq = self.equilibrium_temperature(state.normalized_surface_pressure)
        d_temperature = -self.kt() * (state.temperature - Teq)

        d_v_wind = -self.kv() * state.v_wind
        d_u_wind = -self.kv() * state.u_wind
        d_spec_humidity = jnp.zeros_like(state.temperature) # just keep the same specific humidity?

        return PhysicsTendency(d_u_wind, d_v_wind, d_temperature, d_spec_humidity), None