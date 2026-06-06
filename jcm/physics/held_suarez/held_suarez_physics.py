"""Held-Suarez (1994) idealized forcing as a composable PhysicsTerm.

Provides a single PhysicsTerm (`HeldSuarez`) implementing Newtonian
relaxation toward the analytic Held-Suarez radiative equilibrium plus
Rayleigh friction in the boundary layer, and a `held_suarez_physics()`
factory returning a `ComposablePhysics` with that single term.
"""

from typing import ClassVar
import jax.numpy as jnp
from flax import nnx
from dinosaur.scales import units
from dinosaur import coordinate_systems
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics_interface import PhysicsState, PhysicsTendency
import jcm.constants as jcm_constants
from jcm.dycore.dinosaur.dycore import physics_specs_from_constants
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.composable_physics import ComposablePhysics

Quantity = units.Quantity


class HeldSuarez(PhysicsTerm):
    """Held-Suarez (1994) Newtonian relaxation + Rayleigh friction.

    All parameters use SI Quantity inputs and are non-dimensionalized at
    construction against specs built from the live
    :data:`jcm.constants.physical_constants`, so a prior ``set_constants``
    override is honoured (consistent with the dynamical core).
    """

    name: ClassVar[str] = "held_suarez"
    category: ClassVar[str] = "held_suarez"

    def __init__(
        self,
        sigma_b: float = 0.7,
        kf: Quantity = 1 / (1 * units.day),
        ka: Quantity = 1 / (40 * units.day),
        ks: Quantity = 1 / (4 * units.day),
        minT: Quantity = 200 * units.degK,
        maxT: Quantity = 315 * units.degK,
        dTy: Quantity = 60 * units.degK,
        dThz: Quantity = 10 * units.degK,
    ) -> None:
        """Initialize Held-Suarez forcing parameters."""
        # Build specs from the live constants singleton (read here at
        # construction) so an override set before the model is built is used
        # for nondimensionalisation and kappa.
        specs = physics_specs_from_constants(jcm_constants.physical_constants)
        self.sigma_b = nnx.Variable(jnp.asarray(sigma_b))
        self.kf = nnx.Variable(jnp.asarray(specs.nondimensionalize(kf)))
        self.ka = nnx.Variable(jnp.asarray(specs.nondimensionalize(ka)))
        self.ks = nnx.Variable(jnp.asarray(specs.nondimensionalize(ks)))
        self.minT = nnx.Variable(jnp.asarray(specs.nondimensionalize(minT)))
        self.maxT = nnx.Variable(jnp.asarray(specs.nondimensionalize(maxT)))
        self.dTy = nnx.Variable(jnp.asarray(specs.nondimensionalize(dTy)))
        self.dThz = nnx.Variable(jnp.asarray(specs.nondimensionalize(dThz)))
        self._kappa = float(specs.kappa)
        self._coords_cached = False

    def cache_coords(self, coords: coordinate_systems.CoordinateSystem) -> None:
        """Cache the sigma centers and latitudes used by the analytic forcing."""
        vertical = coords.vertical
        if hasattr(vertical, 'centers'):
            sigma = vertical.centers
        else:
            sigma = vertical.get_sigma_centers(101325.0)
        self._sigma = nnx.Variable(jnp.asarray(sigma))
        self._lat = nnx.Variable(jnp.asarray(coords.horizontal.latitudes))
        self._coords_cached = True

    def _equilibrium_temperature(self, normalized_surface_pressure):
        sigma = self._sigma.get_value()
        lat = self._lat.get_value()
        p_over_p0 = sigma[:, jnp.newaxis, jnp.newaxis] * normalized_surface_pressure
        temperature = p_over_p0 ** self._kappa * (
            self.maxT.get_value()
            - self.dTy.get_value() * jnp.sin(lat) ** 2
            - self.dThz.get_value() * jnp.log(p_over_p0) * jnp.cos(lat) ** 2
        )
        return jnp.maximum(self.minT.get_value(), temperature)

    def _kv(self):
        sigma = self._sigma.get_value()
        kv = self.kf.get_value() * jnp.maximum(
            0.0, (sigma - self.sigma_b.get_value()) / (1.0 - self.sigma_b.get_value())
        )
        return kv[:, jnp.newaxis, jnp.newaxis]

    def _kt(self):
        sigma = self._sigma.get_value()
        lat = self._lat.get_value()
        cutoff = jnp.maximum(
            0.0, (sigma - self.sigma_b.get_value()) / (1.0 - self.sigma_b.get_value())
        )
        return self.ka.get_value() + (
            self.ks.get_value() - self.ka.get_value()
        ) * (cutoff[:, jnp.newaxis, jnp.newaxis] * jnp.cos(lat) ** 4)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute Held-Suarez tendencies; diagnostics dict is passed through unchanged."""
        Teq = self._equilibrium_temperature(state.normalized_surface_pressure)
        d_temperature = -self._kt() * (state.temperature - Teq)
        d_v_wind = -self._kv() * state.v_wind
        d_u_wind = -self._kv() * state.u_wind
        d_spec_humidity = jnp.zeros_like(state.temperature)

        tendencies = PhysicsTendency(
            u_wind=d_u_wind,
            v_wind=d_v_wind,
            temperature=d_temperature,
            specific_humidity=d_spec_humidity,
        )
        return tendencies, diagnostics


def held_suarez_physics(**kwargs) -> ComposablePhysics:
    """Return a ComposablePhysics with the single Held-Suarez forcing term.

    Any keyword arguments are forwarded to `HeldSuarez.__init__`.
    """
    return ComposablePhysics(
        terms=[HeldSuarez(**kwargs)],
        checkpoint_terms=False,
        vectorize_columns=False,
    )
