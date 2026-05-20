"""The :class:`DynamicalCore` protocol.

A dynamical core is anything that integrates the primitive equations forward by one
``dt`` given an optional gridpoint-space physics tendency. The protocol fixes the
contract; the implementation details (spectral / finite-element / finite-volume,
hydrostatic / non-hydrostatic, sigma / hybrid / height coordinates) are entirely
internal to each backend.

The contract is **gridpoint-on-both-sides** on the dycore's native horizontal layout.
That is: :meth:`DynamicalCore.to_physics_state` returns a
:class:`~jcm.physics_interface.PhysicsState` whose 3-D fields are shaped
``(nlev, *horizontal_shape)`` where ``horizontal_shape`` is whatever the dycore
exposes; :meth:`DynamicalCore.step` takes a
:class:`~jcm.physics_interface.PhysicsTendency` shaped identically. There is no
horizontal regrid at the physics-dynamics seam — physics packages run column-local
(or vectorised across columns), and the small number of horizontally non-local
schemes (solar zenith, MACv2-SP, SST forcing) cache against the dycore-supplied
``coords.horizontal.latitudes`` / ``longitudes`` arrays, which are shaped to match
``horizontal_shape``.

This protocol is the only sanctioned bridge between the dycore's internal state and
the rest of jax-gcm. ``Model`` never reaches past it; physics packages never see
spectral coefficients (or whatever the dycore's native representation happens to be).
"""

from __future__ import annotations

import abc
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import jax.numpy as jnp
import tree_math

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

    from jcm.physics_interface import PhysicsState, PhysicsTendency
    from jcm.physics.physics_term import TracerSpec
    from jcm.terrain import TerrainData


# DycoreState is whatever pytree a particular backend uses to carry its prognostic
# variables across one ``dt`` of integration. The protocol places no constraint on
# its concrete shape other than that it must be a valid JAX pytree (so jax.lax.scan,
# tree_map, and flax.serialization work on it without special cases).
DycoreState = Any


@tree_math.struct
class Predictions:
    """Internal container for one frame of model prediction output (a JAX pytree).

    Relocated from :mod:`jcm.model` so that the dycore protocol can reference it
    without inducing a circular import. The user-facing :class:`ModelPredictions`
    wrapper still lives in :mod:`jcm.model`.

    Attributes:
        dynamics: Gridpoint :class:`PhysicsState` projected from the dycore's
            native state via :meth:`DynamicalCore.to_physics_state`.
        physics: Diagnostic physics dict for this frame (per-step snapshot or
            inner-step running mean, depending on the integration mode).
        times: Frame timestamps (filled in by :class:`Model` after the scan).

    """

    dynamics: Any
    physics: Any
    times: Any


class DynamicalCore(abc.ABC):
    """Protocol for a swappable dynamical-core backend.

    Concrete backends subclass this and register themselves with
    :func:`jcm.dycore.registry.register_dycore`. The user-facing :class:`Model`
    constructs a backend (or accepts one) and delegates every state-touching
    operation to it.

    A backend is responsible for:
      * Carrying its own native state representation across ``step`` calls.
      * Projecting that state to and from :class:`PhysicsState` so that physics
        packages can run on a common gridpoint layout (no spectral coefficients
        leak into physics).
      * Applying its own hyperdiffusion / filters as part of ``step``. The
        physics-dynamics seam is purely operator-split (Lie a): the gridpoint
        ``physics_tendency`` is forward-Euler-added to the state and the dycore
        then takes one ``dt`` of dynamics.
      * Building its own terrain (orography is smoothed against the dycore's own
        basis — spectral truncation for dinosaur, SE projection for pyses, ...).
      * Converting a trajectory to xarray for output (the cubed-sphere → lat/lon
        regrid for SE backends lives here, not in the physics path).

    Instance attributes:
        coords: Coordinate system carrying horizontal lat/lon arrays (in the
            dycore's native shape) and the vertical level definitions. For the
            v2 dinosaur backend this is the dinosaur
            ``CoordinateSystem``; SE backends provide a compatible adapter.
        dt_seconds: The integration timestep this dycore was built for. Concrete
            backends may sub-cycle internally but ``step`` advances by exactly
            ``dt_seconds`` (or the override passed to ``step``).
        terrain: Boundary conditions (orography, land/sea mask, SSO descriptors).
            Built via :meth:`build_terrain` or passed in at construction.
        tracer_specs: Public mapping ``name -> TracerSpec`` declaring every
            tracer the dycore should carry. :class:`Model` writes this every
            time it is constructed (so callers who pass a pre-built dycore
            still get the right specs); backends read it from
            :meth:`initial_state`, :meth:`to_physics_state`, and :meth:`step`.
    """

    coords: Any
    dt_seconds: float
    terrain: "TerrainData"
    tracer_specs: dict

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initial_state(
        self,
        physics_state: "PhysicsState | None",
        *,
        sim_time: float = 0.0,
        random_seed: int = 0,
        tracer_specs: Mapping[str, "TracerSpec"] | None = None,
    ) -> DycoreState:
        """Build the dycore's native initial state.

        Args:
            physics_state: Optional gridpoint state to seed from. ``None`` lets
                the backend produce its own default (typically an isothermal
                rest atmosphere).
            sim_time: Initial value for the state's sim-time counter (seconds).
            random_seed: Used to seed any randomised perturbation the backend
                applies to its default initial state.
            tracer_specs: Per-tracer declarations from the physics package; the
                backend uses these to size and initialise its tracer arrays.

        Returns:
            A backend-native state pytree.

        """

    # ------------------------------------------------------------------
    # Gridpoint bridge
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def to_physics_state(self, state: DycoreState) -> "PhysicsState":
        """Project the dycore's native state into a gridpoint :class:`PhysicsState`.

        The returned arrays must be shaped ``(nlev, *horizontal_shape)`` where
        ``horizontal_shape`` matches ``self.coords.horizontal.nodal_shape``.
        No horizontal regrid happens here — that is reserved for
        :meth:`to_xarray` on backends whose native layout differs from the
        target output grid.
        """

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def step(
        self,
        state: DycoreState,
        physics_tendency: "PhysicsTendency | None",
    ) -> DycoreState:
        """Advance the state by one ``dt``.

        Backends apply the gridpoint ``physics_tendency`` as a forward-Euler add
        (operator-split Lie a) before running their own dynamics step. Backends
        may also apply hyperdiffusion / filters / sponge / vertical remap
        internally; the returned state is end-of-step, post-everything.

        Args:
            state: Current dycore-native state.
            physics_tendency: Gridpoint physics tendencies (may be ``None`` for
                a pure-dynamics step, e.g. dry baroclinic-wave tests).

        Returns:
            The dycore-native state at ``t + dt``.

        """

    # ------------------------------------------------------------------
    # Sim-time accounting (so Model can index trajectory frames)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def sim_time(self, state: DycoreState) -> jnp.ndarray:
        """Return the simulation-time (seconds) carried by ``state``."""

    @abc.abstractmethod
    def with_sim_time(self, state: DycoreState, sim_time) -> DycoreState:
        """Return a copy of ``state`` whose sim-time has been replaced."""

    # ------------------------------------------------------------------
    # Tracer compatibility
    # ------------------------------------------------------------------

    def required_tracers_ok(self, specs: Sequence["TracerSpec"]) -> None:
        """Verify the dycore can carry the requested tracers.

        Default is permissive (accept anything). Backends with hard constraints
        (e.g. CAM-SE's dry vs moist mixing-ratio convention) override this to
        raise informatively.
        """
        return None

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def to_xarray(
        self,
        predictions: Predictions,
        times: "np.ndarray",
        *,
        additional_coords: Mapping[str, Any] | None = None,
    ) -> "xr.Dataset":
        """Convert a saved trajectory to an :class:`xarray.Dataset`.

        Backends whose native horizontal layout differs from the desired output
        grid (e.g. a cubed-sphere SE backend producing a lat/lon dataset)
        perform the regrid here. The protocol does not constrain the output
        layout, but callers conventionally expect a regular lat/lon grid so
        existing plotting tooling works without changes.
        """

    # ------------------------------------------------------------------
    # Terrain construction
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_terrain(
        self,
        *,
        source_file: str | None = None,
        **kwargs,
    ) -> "TerrainData":
        """Construct a :class:`TerrainData` against this dycore's basis.

        Orography is smoothed against the dycore's own representation —
        spectral truncation for dinosaur, SE projection for pyses, identity
        for aquaplanet / Held-Suarez runs. ``source_file`` is an optional
        path to a netCDF with raw orography and land-sea mask; backends
        accept their own additional kwargs (envelope wavenumber, smoothing
        radius, …) via ``**kwargs``.
        """
