"""Semi-Lagrangian vertical interpolation in log reference pressure.

Dinosaur's semi-Lagrangian step interpolates every transported field (winds,
``T'``, humidity, tracers) at the departure points with Lagrange weights
computed on the trajectory coordinate itself: the reference sigma
``s = (A + B·pₛ_ref)/pₛ_ref`` on hybrid grids (``σ`` on sigma grids). On a
middle-atmosphere hybrid grid ``s`` is proportional to pressure at the top,
where the levels are spaced geometrically: on ECHAM's L47 the four top full
levels sit at 1.0, 4.3, 11.1 and 23.1 Pa, each layer about twice as thick in
``s`` as the one above it. Two things go wrong there.

* **The linear top cell under-reads vertical advection.** Between the top
  two levels the stencil is linear (the cubic rule degrades to linear in the
  first and last cells). Linear in ``s`` puts a departure point a short way
  below the top level almost entirely on the top level's value: for the L47
  top cell the weight of the second level is 0.44 of the weight a profile
  linear in ``ln p`` (in height) gives it, while the cubic cells below
  over-read the same profile (1.37 and 1.16 of the advection at the 4.3 and
  11.1 Pa levels). Under the summer-mesosphere upwelling the top level
  receives less than half of the advection of warmer air from below while
  the adiabatic cooling acts in full: a numerical cold bias of the top level
  and a warm bias of the one below it.

* **The cubic cells amplify grid-scale vertical structure.** Four-point
  Lagrange weights on nodes whose spacing doubles from one cell to the next
  are not bounded: for a departure point a short way below a node, the
  interpolant of a 2Δz pattern overshoots the node's own value: by up to
  9.9 % per step in the L47 cell between 4.3 and 11.1 Pa, and in every cell
  down to 74 Pa. Under persistent upwelling, which keeps the departure points
  on that side, the step is anti-diffusive.

Interpolating in ``ln s`` removes both. At the pure-pressure top ``ln s`` is
log-pressure, in which the L47/L95 levels are near-uniform and in which the
fields of the stratosphere and mesosphere are smooth (an isothermal or
constant-lapse-rate layer is linear in height, i.e. in ``ln p``). Near the
surface ``ln s ≈ s − 1``, so the troposphere interpolates essentially as
before. The trajectories are unchanged — the departure points are solved in
``s``, with ``ṡ`` diagnosed from the mass flux exactly as dinosaur does — and
only the interpolation coordinate is mapped, monotonically, so the bracketing
cell, the quasi-monotone limiter and the boundary treatment (no extrapolation
past the top and bottom levels) are as before.

ECHAM has no semi-Lagrangian dynamics (its vertical advection is the Simmons
& Burridge (1981) centred difference of ``dyn.f90``), so there is no ECHAM
rule to copy; the choice is the coordinate in which the grid's reference
levels are quasi-uniform. On the SPEEDY/Held-Suarez sigma grids (L8) that is
``σ`` itself, which is why sigma grids keep dinosaur's native interpolation
by default. See ``docs/source/design/sl_vertical_interpolation.md``.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from dinosaur import primitive_equations, semi_lagrangian

def log_interpolation_nodes(
    nodes: semi_lagrangian.VerticalNodes,
) -> semi_lagrangian.VerticalNodes:
    """Map the trajectory nodes to ``ln s``.

    Only ``centers`` is read by the transport (the fields live at layer
    centres). The layer boundaries include ``s = 0`` at a pressure-zero lid,
    which is mapped to ``ln(s₀/2)`` (``ln 2`` above the top centre) so the
    boundaries stay finite and increasing; nothing interpolates on them.
    """
    centers = np.asarray(nodes.centers)
    boundaries = np.asarray(nodes.boundaries)
    log_centers = np.log(centers.astype(np.float64))
    lid = 0.5 * float(centers[0])
    log_boundaries = np.log(np.maximum(boundaries.astype(np.float64), lid))
    return semi_lagrangian.VerticalNodes(
        centers=log_centers.astype(centers.dtype),
        boundaries=log_boundaries.astype(boundaries.dtype),
    )


def log_departure(
    departure: primitive_equations.PrimitiveDeparturePoints,
) -> primitive_equations.PrimitiveDeparturePoints:
    """Departure points with the vertical position mapped to ``ln s``.

    The trajectory solve clips the departure ``s`` to the range of the layer
    centres, all positive, so the logarithm is finite.
    """
    full = departure.full
    return primitive_equations.PrimitiveDeparturePoints(
        full=semi_lagrangian.DeparturePoints(
            cartesian=full.cartesian, sigma=jnp.log(full.sigma)),
        horizontal=departure.horizontal,
    )


class _LogPressureVerticalInterpolation:
    """Mixin: interpolate the transported fields in ``ln s``.

    Overrides only ``semi_lagrangian_transport``; the departure-point solve
    and the non-advective and implicit terms are the parent's. Both stages
    of the Crank–Nicolson RK2 step call this method, so the whole step uses
    one interpolation coordinate.
    """

    def _trajectory_nodes(self) -> semi_lagrangian.VerticalNodes:
        raise NotImplementedError

    def semi_lagrangian_transport(self, state, departure):
        """Remap ``state`` along ``departure``, interpolating in ``ln s``."""
        return primitive_equations._semi_lagrangian_transport(
            self, state, log_departure(departure),
            log_interpolation_nodes(self._trajectory_nodes()),
        )


@dataclasses.dataclass
class LogPressureSemiLagrangianHybrid(
    _LogPressureVerticalInterpolation,
    primitive_equations.SemiLagrangianPrimitiveEquationsHybrid,
):
    """Dinosaur's hybrid SL primitive equations, interpolating in ``ln s``."""

    def _trajectory_nodes(self) -> semi_lagrangian.VerticalNodes:
        return self._reference_vertical_nodes


@dataclasses.dataclass
class LogPressureSemiLagrangianSigma(
    _LogPressureVerticalInterpolation,
    primitive_equations.SemiLagrangianPrimitiveEquations,
):
    """Dinosaur's sigma SL primitive equations, interpolating in ``ln σ``."""

    def _trajectory_nodes(self) -> semi_lagrangian.VerticalNodes:
        return self._vertical_nodes
