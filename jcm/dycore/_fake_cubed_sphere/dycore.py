"""Minimal non-lat/lon dycore used to validate the :class:`DynamicalCore` protocol.

State lives on a fictional cubed-sphere layout: every 3-D field has shape
``(nlev, nelem, gll, gll)``, so the trailing two axes are *not* ``(nlon, nlat)``.
There is no real dynamics — :meth:`step` just forward-Euler-adds any provided
:class:`PhysicsTendency` and advances sim-time. The fake exists purely to
prove that the protocol surface (and the rest of :class:`Model` /
:mod:`physics_interface` that the protocol talks to) doesn't bake in
lat/lon-grid assumptions.

The latitude / longitude arrays the fake exposes through ``self.coords`` have
shape ``(nelem, gll, gll)``. Physics packages broadcast their column-local
calculations against these arrays the same way they would against the 1-D
``latitudes`` arrays of the dinosaur backend — JAX trailing-dim broadcasting
makes the layout transparent as long as ``latitudes.shape`` matches the
trailing-dim of the 3-D state fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import tree_math

from jcm.dycore.base import DynamicalCore, Predictions
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


# ---------------------------------------------------------------------------
# Mock coordinate objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeHorizontalGrid:
    """A minimal stand-in for ``dinosaur.spherical_harmonic.Grid``.

    Exposes only the attributes physics packages actually read off the grid —
    ``latitudes``, ``longitudes``, ``nodal_shape``. Modal transforms
    (``to_modal`` / ``to_nodal``) are deliberately absent; a real SE backend
    would have an analogous "this method makes no sense here" stance.
    """

    nodal_shape: tuple
    latitudes: jnp.ndarray
    longitudes: jnp.ndarray

    def to_modal(self, *args, **kwargs):
        raise NotImplementedError(
            "Cubed-sphere grid has no spherical-harmonic basis — "
            "this code path should never run outside the dinosaur backend.",
        )

    def to_nodal(self, *args, **kwargs):
        raise NotImplementedError(
            "Cubed-sphere grid has no spherical-harmonic basis — "
            "this code path should never run outside the dinosaur backend.",
        )


@dataclass(frozen=True)
class _FakeVerticalGrid:
    """Minimal vertical grid stand-in: sigma centers + layer count."""

    centers: jnp.ndarray

    @property
    def layers(self) -> int:
        return int(self.centers.shape[0])


@dataclass(frozen=True)
class _FakeCoords:
    """Coordinate-system container with the attribute surface the rest of jax-gcm reads."""

    horizontal: _FakeHorizontalGrid
    vertical: _FakeVerticalGrid

    @property
    def nodal_shape(self):
        # ``(nlev, *horizontal_shape)`` — Model._final_dycore_state pytree
        # readers grab this in a few places (e.g. the xarray output path).
        return (self.vertical.layers,) + self.horizontal.nodal_shape


def _build_cubed_sphere_coords(nelem: int, gll: int, nlev: int) -> _FakeCoords:
    """Build coords that pretend each face has ``nelem`` elements with ``gll``
    GLL points each side. Lats/lons are placeholder; the layout — not the
    physical positions — is what the protocol canary cares about.
    """
    rng = np.random.default_rng(0)
    horizontal_shape = (nelem, gll, gll)
    # Uniform random lat/lon in radians, single fixed seed for reproducibility.
    lats = jnp.asarray(rng.uniform(-jnp.pi / 2, jnp.pi / 2, size=horizontal_shape))
    lons = jnp.asarray(rng.uniform(0.0, 2 * jnp.pi, size=horizontal_shape))
    centers = jnp.asarray(np.linspace(0.05, 0.95, nlev))
    return _FakeCoords(
        horizontal=_FakeHorizontalGrid(
            nodal_shape=horizontal_shape, latitudes=lats, longitudes=lons,
        ),
        vertical=_FakeVerticalGrid(centers=centers),
    )


# ---------------------------------------------------------------------------
# State pytree
# ---------------------------------------------------------------------------


@tree_math.struct
class FakeCubedSphereState:
    """Backend-native state for :class:`FakeCubedSphereDycore`.

    Every 3-D field is shaped ``(nlev, nelem, gll, gll)``. ``surface_pressure``
    is 2-D — ``(nelem, gll, gll)`` — and ``sim_time`` is a scalar.
    """

    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray
    sim_time: jnp.ndarray
    tracers: dict


# ---------------------------------------------------------------------------
# The dycore
# ---------------------------------------------------------------------------


class FakeCubedSphereDycore(DynamicalCore):
    """Identity-dynamics cubed-sphere dycore.

    Time-stepping is just ``state + dt * physics_tendency`` (no dynamics,
    no filters). Used by :mod:`jcm.dycore.protocol_test` to prove the
    protocol can host non-lat/lon backends.
    """

    def __init__(
        self,
        nelem: int = 6,
        gll: int = 4,
        nlev: int = 8,
        dt_seconds: float = 1800.0,
    ):
        """Initialise the fake; see the class docstring for argument semantics."""
        self.nelem = int(nelem)
        self.gll = int(gll)
        self.nlev = int(nlev)
        self.coords = _build_cubed_sphere_coords(self.nelem, self.gll, self.nlev)
        # Populated by Model.__init__ from physics.required_tracers(); kept on
        # the dycore so that to_physics_state / step can read it consistently.
        self.tracer_specs = {}
        self.dt_seconds = float(dt_seconds)
        self.terrain = TerrainData.aquaplanet(self.coords)

    # ------------------------------------------------------------------
    # State construction & projection
    # ------------------------------------------------------------------

    def _horizontal_shape(self):
        return self.coords.horizontal.nodal_shape

    def _state_shape(self):
        return (self.nlev,) + self._horizontal_shape()

    def initial_state(
        self,
        physics_state: PhysicsState | None,
        *,
        sim_time: float = 0.0,
        random_seed: int = 0,
        tracer_specs: Mapping[str, Any] | None = None,
    ) -> FakeCubedSphereState:
        shape = self._state_shape()
        if physics_state is None:
            T = jnp.full(shape, 288.0)
            u = jnp.zeros(shape)
            v = jnp.zeros(shape)
            q = jnp.zeros(shape)
            sp = jnp.ones(self._horizontal_shape())
        else:
            T = physics_state.temperature
            u = physics_state.u_wind
            v = physics_state.v_wind
            q = physics_state.specific_humidity
            sp = physics_state.normalized_surface_pressure
        tracers = {}
        for spec in (tracer_specs or {}).values():
            tracers[spec.name] = jnp.full(shape, float(spec.initial_value))
        return FakeCubedSphereState(
            u_wind=u, v_wind=v, temperature=T, specific_humidity=q,
            normalized_surface_pressure=sp,
            sim_time=jnp.asarray(float(sim_time)),
            tracers=tracers,
        )

    def to_physics_state(self, state: FakeCubedSphereState) -> PhysicsState:
        # ``geopotential`` is a diagnostic that the identity dynamics doesn't
        # carry — zero it out for the protocol canary. A real backend would
        # compute it from temperature + surface pressure here.
        phi = jnp.zeros_like(state.temperature)
        return PhysicsState(
            u_wind=state.u_wind,
            v_wind=state.v_wind,
            temperature=state.temperature,
            specific_humidity=state.specific_humidity,
            geopotential=phi,
            normalized_surface_pressure=state.normalized_surface_pressure,
            tracers=dict(state.tracers),
        )

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(
        self,
        state: FakeCubedSphereState,
        physics_tendency: PhysicsTendency | None,
    ) -> FakeCubedSphereState:
        dt = self.dt_seconds
        if physics_tendency is None:
            return state.replace(sim_time=state.sim_time + dt)
        new_tracers = dict(state.tracers)
        for name, tend in physics_tendency.tracers.items():
            if name in new_tracers:
                new_tracers[name] = new_tracers[name] + dt * tend
            else:
                new_tracers[name] = dt * tend
        return state.replace(
            u_wind=state.u_wind + dt * physics_tendency.u_wind,
            v_wind=state.v_wind + dt * physics_tendency.v_wind,
            temperature=state.temperature + dt * physics_tendency.temperature,
            specific_humidity=state.specific_humidity + dt * physics_tendency.specific_humidity,
            sim_time=state.sim_time + dt,
            tracers=new_tracers,
        )

    def sim_time(self, state: FakeCubedSphereState):
        return state.sim_time

    def with_sim_time(self, state: FakeCubedSphereState, sim_time):
        return state.replace(sim_time=jnp.asarray(sim_time))

    # ------------------------------------------------------------------
    # Output (regrid to lat/lon — for the canary this is a stub)
    # ------------------------------------------------------------------

    def to_xarray(self, predictions: Predictions, times, *, additional_coords=None):
        # A real SE backend would precompute a cubed-sphere → lat/lon weight
        # matrix here and emit a regular grid. For protocol-validation the
        # native cubed-sphere layout is enough.
        import xarray as xr
        ds = xr.Dataset(coords={"time": np.asarray(times)})
        ds["sim_time"] = ("time", np.asarray(times))
        return ds

    def build_terrain(self, *, source_file=None, **kwargs) -> TerrainData:
        if source_file is not None:
            raise NotImplementedError(
                "FakeCubedSphereDycore does not load orography from file — "
                "use TerrainData.aquaplanet(coords) or pass a TerrainData "
                "directly to the Model.",
            )
        return TerrainData.aquaplanet(self.coords)
