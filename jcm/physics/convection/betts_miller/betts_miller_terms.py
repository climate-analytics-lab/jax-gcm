"""Composable :class:`PhysicsTerm` wrapper for the Betts-Miller scheme."""

from typing import ClassVar

import jax.numpy as jnp
from dinosaur.hybrid_coordinates import HybridCoordinates
from flax import nnx

import jcm.constants as c
from jcm.physics_interface import PhysicsTendency
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.convection.betts_miller.betts_miller import betts_miller_tendencies
from jcm.physics.convection.betts_miller.params import BettsMillerParameters


class BettsMillerConvection(PhysicsTerm):
    """Betts-Miller convective adjustment as a composable physics term.

    Relaxes temperature and humidity toward a moist-adiabatic reference profile
    (target RH ``rhbm``) over ``tau_bm``; the negative-precip behaviour is set by
    ``params.shallow`` (see :class:`BettsMillerParameters`). Works on sigma or
    hybrid vertical grids and at any number of levels.

    The configuration is a static :class:`BettsMillerParameters` (the flavor and
    modifiers select code paths), supplied at construction.
    """

    name: ClassVar[str] = "betts_miller_convection"
    category: ClassVar[str] = "convection"

    def __init__(self, params: BettsMillerParameters | None = None) -> None:
        """Initialize with a static :class:`BettsMillerParameters` configuration."""
        self.params = params or BettsMillerParameters.default()
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Cache the hybrid/sigma half-level coefficients (``p_half = a + b·ps``)."""
        vertical = coords.vertical
        if isinstance(vertical, HybridCoordinates):
            a_half = jnp.asarray(vertical.a_boundaries)        # Pa
            b_half = jnp.asarray(vertical.b_boundaries)        # dimensionless
        else:  # SigmaCoordinates
            sigma_boundaries = jnp.asarray(vertical.boundaries)
            a_half = jnp.zeros_like(sigma_boundaries)
            b_half = sigma_boundaries
        self._a_half = nnx.Variable(a_half)
        self._b_half = nnx.Variable(b_half)
        self._coords_cached = True

    def __call__(self, state, diagnostics, forcing, terrain):
        """Return Betts-Miller (temperature, humidity) tendencies + precip diagnostic."""
        dt = diagnostics["_dt_seconds"]

        ps = state.normalized_surface_pressure * c.p0          # (ix, il) [Pa]
        a_half = self._a_half.get_value()[:, None, None]
        b_half = self._b_half.get_value()[:, None, None]
        phalf = a_half + b_half * ps[None, :, :]               # (kx+1, ix, il)
        pfull = 0.5 * (phalf[:-1] + phalf[1:])                 # (kx, ix, il)

        # PhysicsState carries specific humidity in g/kg; the scheme works in
        # SI kg/kg internally.
        q_kgkg = state.specific_humidity / 1000.0

        dtemp_dt, dq_dt_kgkg, precip = betts_miller_tendencies(
            state.temperature, q_kgkg, pfull, phalf, dt, self.params,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=dtemp_dt,
            specific_humidity=dq_dt_kgkg * 1000.0,             # kg/kg/s -> g/kg/s
        )

        diagnostics = dict(diagnostics)
        diagnostics["betts_miller_precip"] = precip            # [kg/m^2/s]
        return tendency, diagnostics
