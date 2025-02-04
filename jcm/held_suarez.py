import dinosaur
from dinosaur.scales import units
import jax.numpy as jnp
from jcm.boundaries import BoundaryData
from jcm.physics_data import PhysicsData
from dinosaur.time_integration import ExplicitODE
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import typing
from jcm.params import Parameters
from jcm.physics import PhysicsState, PhysicsTendency

Quantity = units.Quantity

class HeldSuarezForcing:
    def __init__(self,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: primitive_equations.PrimitiveEquationsSpecs,
        reference_temperature: typing.Array,
        p0: Quantity = 1e5 * units.pascal,
        sigma_b: Quantity = 0.7,
        kf: Quantity = 1 / (1 * units.day),
        ka: Quantity = 1 / (40 * units.day),
        ks: Quantity = 1 / (4 * units.day),
        minT: Quantity = 200 * units.degK,
        maxT: Quantity = 315 * units.degK,
        dTy: Quantity = 60 * units.degK,    
        dThz: Quantity = 10 * units.degK,):
        """Initialize Held-Suarez.

        Args:
            coords: horizontal and vertical descritization.
            physics_specs: object holding physical constants and definition of custom
            units to use for initialization of the state.
            reference_temperature: horizontal reference temperature at all altitudes.
            p0: reference surface pressure.
            sigma_b: sigma level of effective planetary boundary layer.
            kf: coefficient of friction for Rayleigh drag.
            ka: coefficient of thermal relaxation in upper atmosphere.
            ks: coefficient of thermal relaxation at earth surface on the equator.
            minT: lower temperature bound of radiative equilibrium.
            maxT: upper temperature bound of radiative equilibrium.
            dTy: horizontal temperature variation of radiative equilibrium.
            dThz: vertical temperature variation of radiative equilibrium.
        """
        self.coords = coords
        self.physics_specs = physics_specs
        self.reference_temperature = reference_temperature
        self.p0 = physics_specs.nondimensionalize(p0)
        self.sigma_b = sigma_b
        self.kf = physics_specs.nondimensionalize(kf)
        self.ka = physics_specs.nondimensionalize(ka)
        self.ks = physics_specs.nondimensionalize(ks)
        self.minT = physics_specs.nondimensionalize(minT)
        self.maxT = physics_specs.nondimensionalize(maxT)
        self.dTy = physics_specs.nondimensionalize(dTy)
        self.dThz = physics_specs.nondimensionalize(dThz)
        # Coordinates
        self.sigma = self.coords.vertical.centers
        _, sin_lat = self.coords.horizontal.nodal_mesh
        self.lat = jnp.arcsin(sin_lat)                                                           


      
    def equilibrium_temperature(self, nodal_surface_pressure):
        p_over_p0 = (
            self.sigma[jnp.newaxis, jnp.newaxis, :] * nodal_surface_pressure[:, :, jnp.newaxis] / self.p0
        )
        temperature = p_over_p0**self.physics_specs.kappa * (
            self.maxT
            - self.dTy * jnp.sin(self.lat[:, :, jnp.newaxis]) ** 2
            - self.dThz * jnp.log(p_over_p0) * jnp.cos(self.lat[:, :, jnp.newaxis]) ** 2
        )
        return jnp.maximum(self.minT, temperature)
   
    def kv(self):
        kv_coeff = self.kf * (
            jnp.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        )
        return kv_coeff[jnp.newaxis, jnp.newaxis, :]

    def kt(self):
        cutoff = jnp.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        return self.ka + (self.ks - self.ka) * (
            cutoff[jnp.newaxis, jnp.newaxis, :] * jnp.cos(self.lat[:, :, jnp.newaxis]) ** 4
    )

    def held_suarez_forcings(self, state: PhysicsState, physics_data: PhysicsData, parameters: Parameters, boundaries: BoundaryData):
        Teq = self.equilibrium_temperature(state.surface_pressure)
        d_temperature = -self.kt() * (state.temperature - Teq)

        d_v_wind = -self.kv() * state.v_wind
        d_u_wind = -self.kv() * state.u_wind
        d_spec_humidity = jnp.zeros_like(state.temperature) # just keep the same specific humidity?

        return PhysicsTendency(d_u_wind, d_v_wind, d_temperature, d_spec_humidity), physics_data
