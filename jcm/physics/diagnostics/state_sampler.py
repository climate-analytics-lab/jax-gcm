"""StateSampler: expose state fields to virtual-observation Observers.

Observers (:mod:`jcm.observers`) sample from the per-step physics
diagnostics dict — the only per-``dt`` data channel the integration scan
carries. The raw :class:`~jcm.physics_interface.PhysicsState` fields
(temperature, winds, humidity, tracers) and the vertical coordinates the
observers interpolate against (``z_full``, ``p_full``) are not otherwise in
that dict, so this zero-tendency term copies them in under the single
internal key ``"_sampler_state"``.

The key is deliberately a plain dict: ``data_struct_to_dict`` drops
non-array, non-struct values from the user-facing xarray output, and the
Model strips the key from saved trajectory frames (see
``Model._post_process``) so the regular netCDF output does not duplicate
the dynamics fields. Only the per-timestep observer channel reads it.

:class:`~jcm.model.Model` appends this term automatically when observers
are attached; it is broadcasting-native (works in both the 3-D and the
column-vectorized physics layouts) and adds no measurable cost.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


class StateSampler(PhysicsTerm):
    """Copy state fields + vertical coordinates into the diagnostics dict."""

    name: ClassVar[str] = "state_sampler"
    category: ClassVar[str] = "diagnostics"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("_sampler_state",)

    def __init__(self):
        """Defer coefficient caching until ``cache_coords`` runs."""
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Cache hybrid/sigma (a, b) full-level coefficients for ``p_full``.

        Handles ``HybridCoordinates`` (a, b native) and sigma coordinates
        (a = 0, b = sigma) — same pattern as ``MoistAirColumnState``.
        """
        from dinosaur.hybrid_coordinates import HybridCoordinates

        vertical = coords.vertical
        if isinstance(vertical, HybridCoordinates):
            a_half = jnp.asarray(vertical.a_boundaries)
            b_half = jnp.asarray(vertical.b_boundaries)
        else:
            sigma_boundaries = jnp.asarray(vertical.boundaries)
            a_half = jnp.zeros_like(sigma_boundaries)
            b_half = sigma_boundaries
        self._a_full = nnx.Variable(0.5 * (a_half[:-1] + a_half[1:]))
        self._b_full = nnx.Variable(0.5 * (b_half[:-1] + b_half[1:]))
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Publish ``_sampler_state``; return zero tendencies."""
        ps = state.normalized_surface_pressure * c.p0  # Pa, (*horiz)
        dtype = state.temperature.dtype
        # Broadcasting-native hybrid pressure: works on (kx,), (kx, ncols)
        # and (kx, ix, il) states alike.
        shape = (-1,) + (1,) * ps.ndim
        a_full = self._a_full.get_value().astype(dtype).reshape(shape)
        b_full = self._b_full.get_value().astype(dtype).reshape(shape)
        p_full = a_full + b_full * ps[None]

        sampler_state = {
            "temperature": state.temperature,
            "u_wind": state.u_wind,
            "v_wind": state.v_wind,
            "specific_humidity": state.specific_humidity,
            "surface_pressure": ps,
            # Height above the geoid (the state geopotential includes the
            # surface geopotential), matching obs altitudes above sea level.
            "z_full": state.geopotential / c.grav,
            "p_full": p_full,
            "tracers": dict(state.tracers),
        }
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "_sampler_state": sampler_state}
