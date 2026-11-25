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
from jcm.physics.coords_util import column_lat_lon
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.composable_physics import ComposablePhysics

Quantity = units.Quantity


class HeldSuarez(PhysicsTerm):
    """Held-Suarez (1994) Newtonian relaxation + Rayleigh friction.

    All parameters use SI Quantity inputs and are non-dimensionalized at
    construction against specs built from the live
    :data:`jcm.constants.physical_constants`, so a prior ``set_constants``
    override is honoured (consistent with the dynamical core).

    Held-Suarez deliberately does NOT publish the ``surface_exchange``
    coupling struct (jax-gcm#754): the forcing is a bulk relaxation with no
    surface fluxes, precipitation or hydrology to report, so opting out
    (rather than publishing zeros a coupler could mistake for a calm, dry
    atmosphere) is the honest contract.
    ``ComposablePhysics.require_surface_exchange()`` fails loudly for this
    package; see docs/source/design/surface_exchange.md.
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
        """Cache the sigma centers and the per-column latitude.

        Latitude is stored once per physics column, flattened lon-major to
        ``(ncols,)`` by :func:`jcm.physics.coords_util.column_lat_lon` (the
        helper every horizontally-aware term uses, so scattered-column grids
        work too). The forcing is defined on sigma levels: a vertical
        coordinate without ``centers`` is evaluated at a 101325 Pa reference
        surface pressure.
        """
        vertical = coords.vertical
        if hasattr(vertical, 'centers'):
            sigma = vertical.centers
        else:
            sigma = vertical.get_sigma_centers(101325.0)
        lat, _ = column_lat_lon(coords.horizontal)
        self._sigma = nnx.Variable(jnp.asarray(sigma))
        self._lat = nnx.Variable(jnp.asarray(lat))
        self._coords_cached = True

    def _column_geometry(self, normalized_surface_pressure):
        """Return ``(sigma, lat)`` shaped to broadcast against ``(nlev, *horiz)``.

        The horizontal shape is read off the surface pressure, so the one code
        path serves a single column, a ``(ncols,)`` block and an
        ``(nlon, nlat)`` grid. The cached latitude holds the same columns
        flattened lon-major, so reshaping it to the host's horizontal shape
        lays it out exactly as the host laid out its columns.
        """
        horiz = normalized_surface_pressure.shape
        sigma = self._sigma.get_value().reshape((-1,) + (1,) * len(horiz))
        lat = self._lat.get_value().reshape(horiz)
        return sigma, lat

    def _cutoff(self, sigma):
        """Boundary-layer weight ``max(0, (sigma - sigma_b) / (1 - sigma_b))``."""
        sigma_b = self.sigma_b.get_value()
        return jnp.maximum(0.0, (sigma - sigma_b) / (1.0 - sigma_b))

    def _equilibrium_temperature(self, normalized_surface_pressure):
        sigma, lat = self._column_geometry(normalized_surface_pressure)
        p_over_p0 = sigma * normalized_surface_pressure
        temperature = p_over_p0 ** self._kappa * (
            self.maxT.get_value()
            - self.dTy.get_value() * jnp.sin(lat) ** 2
            - self.dThz.get_value() * jnp.log(p_over_p0) * jnp.cos(lat) ** 2
        )
        return jnp.maximum(self.minT.get_value(), temperature)

    def _kv(self, normalized_surface_pressure):
        sigma, _ = self._column_geometry(normalized_surface_pressure)
        return self.kf.get_value() * self._cutoff(sigma)

    def _kt(self, normalized_surface_pressure):
        sigma, lat = self._column_geometry(normalized_surface_pressure)
        return self.ka.get_value() + (
            self.ks.get_value() - self.ka.get_value()
        ) * (self._cutoff(sigma) * jnp.cos(lat) ** 4)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute Held-Suarez tendencies; diagnostics dict is passed through unchanged."""
        ps = state.normalized_surface_pressure
        Teq = self._equilibrium_temperature(ps)
        d_temperature = -self._kt(ps) * (state.temperature - Teq)
        d_v_wind = -self._kv(ps) * state.v_wind
        d_u_wind = -self._kv(ps) * state.u_wind
        d_spec_humidity = jnp.zeros_like(state.temperature)

        tendencies = PhysicsTendency(
            u_wind=d_u_wind,
            v_wind=d_v_wind,
            temperature=d_temperature,
            specific_humidity=d_spec_humidity,
        )
        return tendencies, diagnostics


def held_suarez_physics(vectorize_columns: bool = False, **kwargs) -> ComposablePhysics:
    """Return a ComposablePhysics with the single Held-Suarez forcing term.

    Args:
        vectorize_columns: Run the term on the flattened ``(nlev, ncols)``
            column layout instead of the ``(nlev, nlon, nlat)`` grid. The term
            is layout-agnostic, so both give the same tendencies; the column
            layout is the one to choose when composing with column-only terms
            (gravity-wave drag, tracer terms) via ``+``.
        **kwargs: Forwarded to `HeldSuarez.__init__`.

    """
    return ComposablePhysics(
        terms=[HeldSuarez(**kwargs)],
        checkpoint_terms=False,
        vectorize_columns=vectorize_columns,
    )
