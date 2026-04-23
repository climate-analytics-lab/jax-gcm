"""Upper sponge layer — Rayleigh drag on horizontal wind at top levels.

Analog of ECHAM's ``uspnge`` (``mo_upper_sponge.f90``), reworked as a
composable ``PhysicsTerm``. Applies linear relaxation to (u, v) at a
configurable number of top model levels with a timescale that intensifies
toward TOA:

    du/dt += -u / tau(k)
    dv/dt += -v / tau(k)

    tau(k_top)   = sponge_timescale_s
    tau(k_top+i) = sponge_timescale_s * enspodi ** i       (i = 1..n_sponge_levels-1)
    tau(k >= n_sponge_levels) = infinity                   (no damping)

With the ECHAM defaults (spdrag = 0.926e-4 s⁻¹ → 3 h, enspodi = 1.0) all
sponge levels share the same timescale. Our defaults use ``enspodi = 2.0``
so the damping softens by a factor of 2 per level away from TOA; this
gives a smoother transition into the freely-evolving troposphere.

Unlike the ECHAM version we do **not** restrict to non-zonal-mean waves —
we damp the full wind field at the top levels. In steady state that costs
some stratospheric jet but is cheap to implement and robust. Temperature
is not damped: let radiation set the stratospheric equilibrium.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData


class UpperSponge(PhysicsTerm):
    """Rayleigh drag on (u, v) at the top N model levels."""

    name: ClassVar[str] = "upper_sponge"
    category: ClassVar[str] = "dissipation"

    def __init__(
        self,
        n_sponge_levels: int = 5,
        sponge_timescale_s: float = 3 * 3600.0,
        enspodi: float = 2.0,
    ):
        """Configure the sponge.

        Args:
            n_sponge_levels: Number of top levels over which the sponge acts.
                Levels deeper than this see no damping.
            sponge_timescale_s: Rayleigh timescale tau at the topmost level (s).
                ECHAM default spdrag = 0.926e-4 s⁻¹ corresponds to ~3 h.
            enspodi: Multiplicative increase in tau (softening) per level
                away from TOA. enspodi = 1.0 reproduces ECHAM's uniform-
                strength sponge; enspodi > 1 softens the sponge downward.

        """
        self.n_sponge_levels = n_sponge_levels
        self.sponge_timescale_s = sponge_timescale_s
        self.enspodi = enspodi
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Precompute the 1/tau(k) damping profile for every level."""
        nlev = coords.nodal_shape[0]
        inv_tau = jnp.zeros(nlev)
        for i in range(self.n_sponge_levels):
            if i >= nlev:
                break
            tau_i = self.sponge_timescale_s * (self.enspodi ** i)
            inv_tau = inv_tau.at[i].set(1.0 / tau_i)
        self._inv_tau = nnx.Variable(inv_tau)
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Return Rayleigh-drag tendencies on u, v at the top levels."""
        # state here is the column-vectorised state (nlev, ncols) when
        # called from ComposablePhysics with vectorize_columns=True, or
        # the full 3-D (nlev, nlon, nlat) when used without vectorisation.
        # Either way broadcasting against inv_tau[:, None (, None)] works.
        inv_tau = self._inv_tau.get_value()
        shape = state.u_wind.shape
        broadcast = (slice(None),) + (None,) * (state.u_wind.ndim - 1)
        itau = inv_tau[broadcast]

        du = -state.u_wind * itau
        dv = -state.v_wind * itau
        dT = jnp.zeros(shape)
        dq = jnp.zeros(shape)
        tracers = {name: jnp.zeros(shape) for name in state.tracers}

        tend = PhysicsTendency(
            u_wind=du,
            v_wind=dv,
            temperature=dT,
            specific_humidity=dq,
            tracers=tracers,
        )
        return tend, diagnostics
