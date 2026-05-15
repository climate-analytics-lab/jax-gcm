"""Upper sponge layer — Rayleigh drag on horizontal wind + relaxation
of temperature toward its zonal mean at the top model levels.

Analog of ECHAM's ``uspnge`` (``mo_upper_sponge.f90``), reworked as a
composable ``PhysicsTerm``. Applies linear relaxation at a configurable
number of top model levels with a timescale that intensifies toward
TOA::

    du/dt += -u / tau(k)
    dv/dt += -v / tau(k)
    dT/dt += -(T - T_zonal_mean) / tau(k)        (when damp_temperature)

    tau(k_top)   = sponge_timescale_s
    tau(k_top+i) = sponge_timescale_s * enspodi ** i       (i = 1..n_sponge_levels-1)
    tau(k >= n_sponge_levels) = infinity                   (no damping)

The temperature relaxation is toward the zonal-mean profile at each
sponge level — mathematically equivalent (in gridpoint space) to ECHAM's
spectral implicit step that damps only the m≠0 components of T at the
sponge levels (``mo_upper_sponge.f90`` lines 99-110, applied to ``stp``
when ``mymsp(is) /= 0``). The zonal mean is preserved so radiation can
still set the global stratospheric equilibrium structure; only the
wave/Gibbs-ringing component of T is damped.

With the ECHAM defaults (spdrag = 0.926e-4 s⁻¹ → 3 h, enspodi = 1.0,
nlvspd1 = nlvspd2 = 1) all sponge levels share the same timescale and
the sponge acts on level 1 only. This module's defaults use enspodi
= 2.0 so the damping softens by a factor of 2 per level away from TOA;
this gives a smoother transition into the freely-evolving troposphere
and matches what we tend to ramp up via Hydra at runtime.

Note on (u, v): unlike ECHAM (which damps only m≠0 modes of u, v
spectrally) we damp the full wind field at the sponge levels. In steady
state that costs some stratospheric jet strength but is cheap to
implement and operationally robust. If preserving the zonal-mean wind
becomes important for stratospheric climatology, switch the wind path
to use the same zonal-mean relaxation we now apply to T.
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
    """Rayleigh drag on (u, v) and zonal-mean relaxation of T at top N levels."""

    name: ClassVar[str] = "upper_sponge"
    category: ClassVar[str] = "dissipation"

    def __init__(
        self,
        n_sponge_levels: int = 5,
        sponge_timescale_s: float = 3 * 3600.0,
        enspodi: float = 2.0,
        damp_temperature: bool = True,
        target_T_K: float | None = None,
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
            damp_temperature: When True (default, matches ECHAM lmidatm
                behaviour), relax temperature toward its zonal mean at
                the sponge levels with the same tau profile. The zonal
                mean is preserved so radiation continues to set the global
                stratospheric structure; only the wave / spectral-ringing
                component of T is damped. Set False to skip T damping
                entirely.
            target_T_K: Optional absolute temperature target (K) for T
                relaxation at the sponge levels — i.e. ``dT/dt -=
                (T - target_T_K) / tau(k)``. This is an extra term *added
                to* the zonal-mean relaxation (when ``damp_temperature``
                is True) and addresses the m=0 spectral mode that
                zonal-mean relaxation by construction can't touch. Useful
                during spin-up from non-equilibrated initial conditions
                (e.g. JW-dry init with realistic ozone) where the
                top-layer zonal mean drifts uncontrolled toward an
                unphysical equilibrium. Set ``None`` (default) to skip
                the absolute target and behave like ECHAM's sponge — fine
                for runs starting from radiatively-balanced ICs. Picking
                a value: 250-270 K is a reasonable mesospheric target for
                the model top (~1 Pa); aim for whatever the long-term
                radiative-equilibrium would be at the topmost full level.

        """
        self.n_sponge_levels = n_sponge_levels
        self.sponge_timescale_s = sponge_timescale_s
        self.enspodi = enspodi
        self.damp_temperature = damp_temperature
        self.target_T_K = target_T_K
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Precompute the 1/tau(k) damping profile and the (nlon, nlat) shape."""
        nlev = coords.nodal_shape[0]
        inv_tau = jnp.zeros(nlev)
        for i in range(self.n_sponge_levels):
            if i >= nlev:
                break
            tau_i = self.sponge_timescale_s * (self.enspodi ** i)
            inv_tau = inv_tau.at[i].set(1.0 / tau_i)
        self._inv_tau = nnx.Variable(inv_tau)
        # Cache the (nlon, nlat) shape so __call__ can reshape a flattened
        # ncols axis back into (lon, lat) for zonal-mean computation under
        # vectorize_columns=True.
        self._nlon = int(coords.horizontal.nodal_shape[0])
        self._nlat = int(coords.horizontal.nodal_shape[1])
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Return Rayleigh-drag tendencies on u, v and zonal-mean T relaxation."""
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

        if self.damp_temperature:
            T = state.temperature
            # Compute zonal-mean T at each (level, lat). Reshape ncols→(lon, lat)
            # if the state is column-vectorised.
            if T.ndim == 2:
                # (nlev, ncols=nlon*nlat) — reshape, mean over lon, broadcast.
                nlev = T.shape[0]
                T_3d = T.reshape(nlev, self._nlon, self._nlat)
                T_zonal = jnp.mean(T_3d, axis=1, keepdims=True)         # (nlev, 1, nlat)
                T_anomaly_3d = T_3d - T_zonal
                T_anomaly = T_anomaly_3d.reshape(nlev, self._nlon * self._nlat)
            else:
                # (nlev, nlon, nlat) — direct mean over the lon axis.
                T_zonal = jnp.mean(T, axis=1, keepdims=True)            # (nlev, 1, nlat)
                T_anomaly = T - T_zonal
            dT = -T_anomaly * itau
        else:
            dT = jnp.zeros(shape)

        if self.target_T_K is not None:
            # Add an absolute-target relaxation that catches the m=0 mode
            # the zonal-mean relaxation by construction can't touch.
            dT = dT - (state.temperature - float(self.target_T_K)) * itau

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
