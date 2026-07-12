"""Newtonian relaxation of the top-level temperatures toward a reference.

Purpose-built for finite-lid model tops in the mesosphere (the pySES
CAM-SE full-L47 grid ends at ~1 Pa): the shipped radiation schemes are not
valid at those pressures (no non-LTE physics; RRTMGP's lookup tables bottom
out near 160 K) and drive a *horizontal-mean* radiative refrigeration of the
lid that neither the dycore's Laplacian ``nu_top`` sponge nor the ECHAM-style
:class:`~jcm.physics.dissipation.upper_sponge.UpperSponge` can arrest — both
of those only damp horizontal structure and deliberately preserve the mean.
WACCM handles the same problem with extra physics-side damping; ECHAM's
production configurations simply do not extend this high.

The term applies, at the top ``n_levels`` model levels only::

    dT/dt += -(T - T_ref(k)) / tau(k)
    tau(k_top)     = timescale_s
    tau(k_top + i) = timescale_s * ramp ** i        (i = 1..n_levels-1)

``T_ref`` is a fixed per-level reference profile supplied at construction
(e.g. the U.S. Standard Atmosphere evaluated at the level mid-heights — the
same profile the pySES AMIP initial state uses). With the defaults the lid
is relaxed on ~6 h while the sponge base (8 levels down, ~1.5 hPa) sees
~15 days, leaving the scientifically-relevant stratosphere untouched.

Winds are untouched (the dycore's own ``nu_top`` sponge handles wave
momentum); moisture is untouched (negligible at these pressures).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from flax import nnx

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


class UpperTemperatureRelaxation(PhysicsTerm):
    """Relax the top ``n_levels`` temperatures toward a reference profile."""

    name: ClassVar[str] = "upper_temperature_relaxation"
    category: ClassVar[str] = "dissipation"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("upper_t_relaxation",)

    def __init__(self, t_ref_profile, n_levels: int = 8,
                 timescale_s: float = 6.0 * 3600.0, ramp: float = 2.5,
                 wind_timescale_s: float | None = None,
                 wind_center_level: float = 3.0,
                 wind_range_levels: float = 2.0):
        """Configure the relaxation.

        Args:
            t_ref_profile: (nlev,) reference temperature (K) on the model's
                full levels, TOA-first (only the top ``n_levels`` entries are
                used).
            n_levels: How many top levels are relaxed.
            timescale_s: Relaxation timescale at the model top (s).
            ramp: Multiplicative timescale increase per level downward
                (temperature branch).
            wind_timescale_s: Optional Rayleigh-friction timescale for the
                winds at the model top (s); ``None`` (default) leaves winds
                untouched. This is the WACCM-style momentum counterpart of
                the temperature relaxation: nothing else damps the *mean*
                wind at a finite mesospheric lid (``nu_top`` is a Laplacian
                on horizontal structure), and without it lid jets grow
                unopposed — both day-127 (1m) and day-150 (2m) ne30
                blow-ups showed ~100 m/s 5-day-mean winds in the 1-10 Pa
                levels immediately before going non-finite. The coefficient
                follows CAM's ``rayleigh_friction.F90`` tanh profile in
                level index (smooth in height — the original ×``ramp``
                staircase concentrated differential drag between adjacent
                thin lid layers and shear-shocked a 100 m/s jet at
                tau = 12 h) and is applied Euler-backward (implicit), which
                is unconditionally stable at any strength, with the
                dissipated kinetic energy returned as heat (also per CAM).
            wind_center_level: Level index (from the top, 0-based) where the
                wind-damping tanh profile crosses half strength (CAM
                ``rayk0``).
            wind_range_levels: e-folding half-width of the tanh in level
                index (CAM ``raykrange``).

        """
        t_ref = np.asarray(t_ref_profile, dtype=np.float32)
        nlev = t_ref.shape[0]
        inv_tau = np.zeros(nlev, dtype=np.float32)
        for i in range(min(int(n_levels), nlev)):
            inv_tau[i] = 1.0 / (float(timescale_s) * float(ramp) ** i)
        # CAM-style smooth profile: otau(k) = (1/tau0) * (1 + tanh((k0-k)/kr))/2.
        # Full strength at the lid, half at k0, smoothly to ~0 below — no
        # staircase in adjacent-layer drag. Computed over ALL levels (the
        # tanh itself localises it near the top).
        k_idx = np.arange(nlev, dtype=np.float32)
        if wind_timescale_s is not None:
            inv_tau_wind = (
                (1.0 / float(wind_timescale_s))
                * 0.5 * (1.0 + np.tanh(
                    (float(wind_center_level) - k_idx)
                    / float(wind_range_levels)))
            ).astype(np.float32)
        else:
            inv_tau_wind = np.zeros(nlev, dtype=np.float32)
        self._t_ref = nnx.Variable(jnp.asarray(t_ref))
        self._inv_tau = nnx.Variable(jnp.asarray(inv_tau))
        self._inv_tau_wind = nnx.Variable(jnp.asarray(inv_tau_wind))
        self.damps_wind = wind_timescale_s is not None
        self.n_levels = int(n_levels)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Apply the top-level Newtonian temperature relaxation."""
        t_ref = self._t_ref.get_value().astype(state.temperature.dtype)
        inv_tau = self._inv_tau.get_value().astype(state.temperature.dtype)
        shape = (-1,) + (1,) * (state.temperature.ndim - 1)
        dtdt = -(state.temperature - t_ref.reshape(shape)) * inv_tau.reshape(shape)

        tend = PhysicsTendency.zeros(state.temperature.shape,
                                     temperature=dtdt)
        if self.damps_wind:
            # Rayleigh friction toward rest: the lid winds are unphysical
            # anyway (no non-LTE radiation or resolved GW breaking there),
            # and undamped they grow until they break the dycore's vertical
            # numerics in the thin top layers. Euler-backward form per CAM
            # rayleigh_friction.F90: du/dt = -k u / (1 + k dt) — the update
            # u' = c2 u with c2 = 1/(1 + k dt) is unconditionally stable at
            # any strength, unlike the explicit -k u (which shear-shocked
            # the lid at tau = 12 h).
            dt = diagnostics.get("_dt_seconds", 1800.0)
            k = self._inv_tau_wind.get_value().astype(
                state.temperature.dtype).reshape(shape)
            c2 = 1.0 / (1.0 + k * dt)
            tend = tend.copy(u_wind=-k * c2 * state.u_wind,
                             v_wind=-k * c2 * state.v_wind)
            # Return the dissipated kinetic energy as heat (CAM: the
            # discrete-exact KE loss of the implicit update, added to dry
            # static energy): dT = 0.5 (1 - c2^2)(u^2 + v^2) / cp.
            ke_heating = (0.5 * (1.0 - c2 * c2)
                          * (state.u_wind ** 2 + state.v_wind ** 2)
                          / (c.cpd * dt))
            dtdt = dtdt + ke_heating
            tend = tend.copy(temperature=dtdt)
        # Diagnose the applied heating so the effect is visible in output.
        return tend, {**diagnostics, "upper_t_relaxation": dtdt}
