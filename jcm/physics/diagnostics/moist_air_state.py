"""``MoistAirColumnState`` — common per-column thermodynamic diagnostics.

Computes the pressure / height / density / humidity diagnostics that
*every* column-based scheme reads at the start of a physics step:

- ``pressure_full`` / ``pressure_half`` from the hybrid (a, b) coefficients
  via ``p(k) = a(k) + b(k) * P_s``. Pure sigma is the special case ``a=0``.
- ``height_full`` / ``height_half`` from geopotential, with the top
  half-level extrapolated using the top-layer thickness and the surface
  half-level extrapolated using the bottom-layer thickness.
- ``air_density`` from the ideal-gas law.
- ``layer_thickness`` from ``Δp / (ρ g)`` with a 10 m floor so that very
  thin uniform sigma layers don't blow up downstream divisions.
- ``surface_pressure`` (Pa).
- ``relative_humidity`` from the Tetens formula, with temperature clipped
  only enough to avoid divide-by-zero at T = 29.65 K.

The numerical implementation matches what was previously in
``_prepare_common_physics_state`` (echam/echam_physics.py:44-132); this
term is the scheme-neutral home for that routine.

Each output is written to the diagnostics dict under a public, top-level
key (e.g. ``"pressure_full"``) — no leading underscore — so that
downstream scheme-named terms can read them by name and so they flow
through to the user-facing xarray output without a ``diagnostics.``
prefix.

Date: 2026-05-07
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm import constants as physical_constants
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


#: The set of public keys produced by :class:`MoistAirColumnState`.
#: Other modules import this to convert between the dict representation
#: and the legacy typed ``DiagnosticData`` sub-struct.
MOIST_AIR_FIELDS: tuple[str, ...] = (
    "pressure_full",
    "pressure_half",
    "height_full",
    "height_half",
    "air_density",
    "layer_thickness",
    "surface_pressure",
    "relative_humidity",
    "thermo_run",
)


def advance_thermo_run(
    diagnostics: dict,
    dt,
    d_temperature=None,
    d_specific_humidity=None,
) -> dict:
    """Advance the running ``thermo_run`` state by a term's tendency.

    ``thermo_run`` is a *running thermodynamic view* (temperature, specific
    humidity) carried through the per-step diagnostics so a term that runs after
    another can see the upstream term's effect. A term that changes T and/or q
    before the cloud microphysics (currently only Tiedtke convection; the
    pattern generalises to any opt-in pre-cloud term) calls this with its own
    tendency, so the 2M cloud scheme's saturation balance condenses against the
    post-convection state — the sequential convection->cloud coupling in
    ``.claude/coupled_cloud_operator_design.md``. ``d_temperature`` /
    ``d_specific_humidity`` are tendencies (per second) in the same layout as
    ``thermo_run`` (the term's ``PhysicsState`` shape).

    Tendency ownership — *nothing is zeroed here or by the caller.* This is an
    operator-split model: every term returns its full tendency, the driver sums
    all of them, and the dycore applies that single sum to the *step-start*
    state exactly once. ``thermo_run`` is a parallel diagnostic view, NOT the
    prognostic state, so the same tendency legitimately plays two roles — it is
    returned for the op-split sum *and* passed here to advance ``thermo_run`` for
    downstream terms. Zeroing it in either place would drop that term's
    contribution from the final state. Because each term's tendency is computed
    against the running ``thermo_run`` and the sum is applied once, additive
    accumulation is algebraically identical to applying the terms sequentially
    (T_final = T0 + dt*(conv_dT + cloud_dT(post-conv)) either way).

    Returns a new diagnostics dict with ``thermo_run`` advanced; a no-op
    (returns the input unchanged) if ``thermo_run`` is absent, so callers stay
    backward-safe.
    """
    tr = diagnostics.get("thermo_run")
    if tr is None:
        return diagnostics
    temperature = tr["temperature"]
    specific_humidity = tr["specific_humidity"]
    if d_temperature is not None:
        temperature = temperature + d_temperature * dt
    if d_specific_humidity is not None:
        specific_humidity = specific_humidity + d_specific_humidity * dt
    return {
        **diagnostics,
        "thermo_run": {
            "temperature": temperature,
            "specific_humidity": specific_humidity,
        },
    }


class MoistAirColumnState(PhysicsTerm):
    """Compute pressure, height, density, and humidity diagnostics.

    Operates on column-vectorized state ``(nlev, ncols)``. Caches the
    hybrid (a, b) coefficients from the dinosaur coordinate system at
    construction time.

    Provides the keys listed in :data:`MOIST_AIR_FIELDS` to every
    downstream term.
    """

    name: ClassVar[str] = "moist_air_column_state"
    category: ClassVar[str] = "prepare"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = MOIST_AIR_FIELDS

    def __init__(self):
        """Defer coefficient caching until ``cache_coords`` runs."""
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Cache the hybrid (a, b) coefficients from the coordinate system.

        Handles both ``SigmaCoordinates`` (a = 0, b = sigma) and
        ``HybridCoordinates`` (a, b in their ICON-native form).
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
        a_full = 0.5 * (a_half[:-1] + a_half[1:])
        b_full = 0.5 * (b_half[:-1] + b_half[1:])
        self._a_half = nnx.Variable(a_half)
        self._b_half = nnx.Variable(b_half)
        self._a_full = nnx.Variable(a_full)
        self._b_full = nnx.Variable(b_full)
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Populate the moist-air diagnostic keys; return zero tendency."""
        p0 = physical_constants.p0
        a_full = self._a_full.get_value()
        b_full = self._b_full.get_value()
        a_half = self._a_half.get_value()
        b_half = self._b_half.get_value()

        surface_pressure = state.normalized_surface_pressure * p0  # Pa
        # Hybrid-coordinate pressure: works for pure sigma (a=0) too.
        pressure_full = (
            a_full[:, None] + b_full[:, None] * surface_pressure[None, :]
        )
        pressure_half = (
            a_half[:, None] + b_half[:, None] * surface_pressure[None, :]
        )

        height_full = state.geopotential / physical_constants.grav
        # Internal half-levels are midpoints between full levels; top and
        # surface half-levels extrapolate using the top/bottom layer
        # thickness so layer thickness stays consistent at the boundaries.
        height_half_internal = (height_full[1:] + height_full[:-1]) / 2
        top_layer_thickness = height_full[0] - height_half_internal[0]
        height_top = height_full[0] + top_layer_thickness
        bottom_layer_thickness = height_half_internal[-1] - height_full[-1]
        height_surface = height_full[-1] - bottom_layer_thickness
        height_half = jnp.concatenate(
            (
                height_top[jnp.newaxis],
                height_half_internal,
                height_surface[jnp.newaxis],
            ),
            axis=0,
        )

        air_density = pressure_full / (
            physical_constants.rd * state.temperature
        )
        # Clamp layer thickness floor at 10 m for numerical stability with
        # very thin uniform sigma layers — matches the legacy behaviour.
        dp = jnp.diff(pressure_half, axis=0)
        layer_thickness = jnp.maximum(
            dp / (air_density * physical_constants.grav), 10.0,
        )

        # Tetens formula. The temperature clip is a wide math-safety bound
        # (avoids divide-by-zero at T = 29.65 K and exp overflow at high T)
        # — NOT a physical-range clip.
        T_clip = jnp.clip(state.temperature, 50.0, 500.0)
        q_clip = jnp.maximum(state.specific_humidity, 0.0)
        es = 611.2 * jnp.exp(
            17.67 * (T_clip - 273.15) / (T_clip - 29.65),
        )
        e = q_clip * pressure_full / (0.622 + 0.378 * q_clip)
        relative_humidity = e / jnp.maximum(es, 1e-3)

        # Running thermodynamic state for sequential convection->cloud coupling.
        # Seeded to the step-start (T, q); terms that change T/q before the cloud
        # microphysics (radiation, convection) advance it by their tendency*dt so
        # the cloud scheme sees the post-(radiation+convection) state — emulating
        # ECHAM's sequential radiation->convection->cloud ordering inside our
        # additive-splitting framework. Summing the sequentially-computed
        # tendencies and applying to the step-start state telescopes to the same
        # result as true sequential operator splitting. See
        # ``.claude/coupled_cloud_operator_design.md``.
        thermo_run = {
            "temperature": state.temperature,
            "specific_humidity": state.specific_humidity,
        }

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {
            **diagnostics,
            "pressure_full": pressure_full,
            "pressure_half": pressure_half,
            "height_full": height_full,
            "height_half": height_half,
            "air_density": air_density,
            "layer_thickness": layer_thickness,
            "surface_pressure": surface_pressure,
            "relative_humidity": relative_humidity,
            "thermo_run": thermo_run,
        }
