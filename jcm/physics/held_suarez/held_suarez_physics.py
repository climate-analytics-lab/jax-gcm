from dinosaur.scales import units
import jax.numpy as jnp
from jcm.boundaries import BoundaryData
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from jcm.geometry import Geometry
from jcm.physics.speedy.params import Parameters
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.model import get_coords, PHYSICS_SPECS
from jcm.date import DateData
from typing import Tuple

Quantity = units.Quantity

class HeldSuarezPhysics(Physics):
    write_output: bool

    def __init__(self,
        coords: coordinate_systems.CoordinateSystem = get_coords(),
        sigma_b: Quantity = 0.7,
        kf: Quantity = 1 / (1 * units.day),
        ka: Quantity = 1 / (40 * units.day),
        ks: Quantity = 1 / (4 * units.day),
        minT: Quantity = 200 * units.degK,
        maxT: Quantity = 315 * units.degK,
        dTy: Quantity = 60 * units.degK,
        dThz: Quantity = 10 * units.degK,
    ) -> None:
        """Initialize Held-Suarez.

        Args:
            coords: horizontal and vertical descritization
            sigma_b: sigma level of effective planetary boundary layer.
            kf: coefficient of friction for Rayleigh drag.
            ka: coefficient of thermal relaxation in upper atmosphere.
            ks: coefficient of thermal relaxation at earth surface on the equator.
            minT: lower temperature bound of radiative equilibrium.
            maxT: upper temperature bound of radiative equilibrium.
            dTy: horizontal temperature variation of radiative equilibrium.
            dThz: vertical temperature variation of radiative equilibrium.
        """
        self.write_output = False
        self.coords = coords
        self.sigma_b = sigma_b
        self.kf = PHYSICS_SPECS.nondimensionalize(kf)
        self.ka = PHYSICS_SPECS.nondimensionalize(ka)
        self.ks = PHYSICS_SPECS.nondimensionalize(ks)
        self.minT = PHYSICS_SPECS.nondimensionalize(minT)
        self.maxT = PHYSICS_SPECS.nondimensionalize(maxT)
        self.dTy = PHYSICS_SPECS.nondimensionalize(dTy)
        self.dThz = PHYSICS_SPECS.nondimensionalize(dThz)
        # Coordinates
        self.sigma = self.coords.vertical.centers
        _, sin_lat = self.coords.horizontal.nodal_mesh
        self.lat = jnp.arcsin(sin_lat)

    def equilibrium_temperature(self, normalized_surface_pressure):
        p_over_p0 = (
            self.sigma[:, jnp.newaxis, jnp.newaxis] * normalized_surface_pressure[jnp.newaxis]
        )
        temperature = p_over_p0**PHYSICS_SPECS.kappa * (
            self.maxT
            - self.dTy * jnp.sin(self.lat[jnp.newaxis]) ** 2
            - self.dThz * jnp.log(p_over_p0) * jnp.cos(self.lat[jnp.newaxis]) ** 2
        )
        return jnp.maximum(self.minT, temperature)

    def kv(self):
        kv_coeff = self.kf * (
            jnp.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        )
        return kv_coeff[:, jnp.newaxis, jnp.newaxis]

    def kt(self):
        cutoff = jnp.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        return self.ka + (self.ks - self.ka) * (
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
        Teq = self.equilibrium_temperature(state.surface_pressure)
        d_temperature = -self.kt() * (state.temperature - Teq)

        d_v_wind = -self.kv() * state.v_wind
        d_u_wind = -self.kv() * state.u_wind
        d_spec_humidity = jnp.zeros_like(state.temperature) # just keep the same specific humidity?

        return PhysicsTendency(d_u_wind, d_v_wind, d_temperature, d_spec_humidity), None