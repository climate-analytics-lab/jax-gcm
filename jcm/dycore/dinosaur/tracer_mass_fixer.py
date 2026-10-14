"""Local (Bermejo–Conde) mass fixer for the semi-Lagrangian nodal tracers.

Dinosaur's semi-Lagrangian step interpolates every transported tracer at the
departure points with a quasi-cubic stencil clipped by the Bermejo &
Staniforth (1992) quasi-monotone limiter. The interpolation does not conserve
``∫ q dp``, and the limiter makes the error one-signed where a field is patchy:
clipping the cubic undershoots beside a sharp, near-zero minimum can only add
mass, and it adds it at the edge of the feature. Restoring the global mass by
a single proportional factor (``DinosaurDycore._fix_nodal_tracer_mass``) then
takes that mass back in proportion to the field everywhere — mostly from
where the tracer is large and smooth — so the transport moves mass from
smooth regions into patchy ones every step (#1062). For aerosol this is a
pump into the upper troposphere and lower stratosphere: one step from a
T63L47 JAM state creates 0.6 % of the coarse sea salt held above 300 hPa at
40-60°S and returns 0.22 % of the global mass, so the layer grows by
e-folding times of days wherever its physical sinks are weak (the winter
polar vortex, #1066). The same measurement on ``qi`` gave 0.44 %/step.

The fixer here is the IFS's (Diamantakis & Flemming 2014, GMD 7, 965,
Eqs. 5-6; Bermejo & Conde 2002, MWR 130, 423): the mass error ``δM = M* − M⁰``
of the high-order solution ``φ*`` is removed where it departs from the
low-order (linear) interpolant ``φᴸ`` of the same field at the same departure
points,

    φ¹ = φ* − λ w,   w = max(0, sgn(δM)·(φ* − φᴸ))^β,   λ = δM / Σ w Δm,

with ``β = 1`` as in the IFS (the module default, :data:`BC_BETA`). A
surplus is taken where the cubic sits above the linear solution and a
deficit is restored where it sits below, so the correction stays at the
sharp features that produced it. At ``β = 1``, ``w`` is exactly the
pointwise gap ``|φ* − φᴸ|``, so ``0 ≤ λ ≤ 1`` alone keeps every corrected
value between ``φ*`` and ``φᴸ`` — non-negative and within the limiter's
bounds. At any other ``β``, ``w`` is a nonlinear reshaping of that gap used
only to steer where the correction concentrates, and ``λ·w`` is additionally
capped at the gap itself so the same bound holds regardless of ``β`` (see
:func:`bermejo_conde_correction`). Should the error exceed what the weights'
total capacity or that per-cell cap can absorb, the remainder is left to the
global proportional fixer, which still runs after the whole step to close
the budget exactly (here it only meets the residual of this correction and
of the pressure change between the transport and the end of the step).

ECHAM transports its tracers with the flux-form ``tpcore`` (Lin & Rood 1996),
which conserves mass locally and needs no fixer; an advective-form
semi-Lagrangian scheme needs one, and the local fixer is the one that keeps
the transport from redistributing mass between regions. See
``docs/source/design/tracer_mass_conservation.md``.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from dinosaur import primitive_equations, semi_lagrangian
from dinosaur.hybrid_coordinates import HybridCoordinates

from jcm.filters import stable_quotient

#: IFS value of the Bermejo–Conde weight exponent (Diamantakis & Flemming
#: 2014: the recommended β = 3 brought no benefit and sharper increments).
BC_BETA = 1.0

#: Floor on ``Σ w Δm`` below which no correction is attempted (a field the
#: cubic and linear interpolants transport identically — uniform or empty —
#: has nothing to correct and nothing to correct it with).
_TINY_WEIGHT = 1.0e-30


def cell_air_mass(coords, log_surface_pressure_nodal: jnp.ndarray) -> jnp.ndarray:
    """Per-cell air-mass weight ``Δp · area`` [nondim] on the nodal grid.

    The same weight ``DinosaurDycore._nodal_tracer_column_weight`` integrates
    the proportional fixer with: ``Δa + Δb·pₛ`` on hybrid grids, ``Δσ·pₛ`` on
    sigma grids, times the Gaussian quadrature weight of the column.
    """
    ps = jnp.exp(log_surface_pressure_nodal)
    vert = coords.vertical
    if isinstance(vert, HybridCoordinates):
        a = jnp.asarray(vert.a_boundaries)
        b = jnp.asarray(vert.b_boundaries)
        dp = (a[1:] - a[:-1])[:, None, None] + (b[1:] - b[:-1])[:, None, None] * ps
    else:
        sigma = jnp.asarray(vert.boundaries)
        dp = (sigma[1:] - sigma[:-1])[:, None, None] * ps
    return dp * jnp.asarray(coords.horizontal.quadrature_weights)


def bermejo_conde_correction(
    q_high: jnp.ndarray,
    q_low: jnp.ndarray,
    target_mass: jnp.ndarray,
    cell_mass: jnp.ndarray,
    beta: float = BC_BETA,
) -> jnp.ndarray:
    """Remove ``Σ q_high·Δm − target_mass`` where ``q_high`` departs from ``q_low``.

    Diamantakis & Flemming (2014) Eqs. (5)-(6). At ``beta == 1`` the weight
    ``w`` IS the pointwise gap ``|q_high - q_low|``, so clipping ``λ`` to
    [0, 1] by itself keeps the corrected field between ``q_high`` and
    ``q_low``. At any other ``beta`` the weight is a nonlinear reshaping of
    that gap (``gap ** beta``) used only to steer *where* the correction
    concentrates, not a value the correction may act with directly: for a
    gap > 1 it can exceed the gap itself, and a ``λ`` that is otherwise
    exactly in-bounds would then push a point past ``q_low`` -- including
    past zero. So the per-cell correction is independently capped at the
    gap it is named for correcting, for every ``beta``; whatever that cap or
    the weights' total capacity leaves uncorrected is for the caller's
    global fixer.
    """
    excess = jnp.sum(q_high * cell_mass) - target_mass
    sign = jnp.where(excess >= 0.0, 1.0, -1.0)
    gap = jnp.maximum(sign * (q_high - q_low), 0.0)
    w = gap if beta == 1.0 else gap ** beta
    capacity = jnp.sum(w * cell_mass)
    ok = capacity > _TINY_WEIGHT
    # stable_quotient (#558): a bare excess/capacity differentiates through
    # capacity**-2, which overflows to inf in float32 once capacity underflows
    # 2**-63 while still clearing the _TINY_WEIGHT guard above (the guard
    # keeps the value finite, not the derivative). A uniform or empty field
    # (capacity == 0 exactly) must still back-propagate a finite, zero
    # gradient through this branch -- see test_gradients.
    lam = jnp.where(
        ok,
        stable_quotient(excess * sign, jnp.where(ok, capacity, 1.0)),
        0.0,
    )
    lam = jnp.clip(lam, 0.0, 1.0)
    # min(., gap), not a second clip on lam*w/gap: at beta == 1, w == gap, so
    # this is exactly lam*w (no new branch on the already-covered path).
    correction = jnp.minimum(lam * w, gap)
    return q_high - sign * correction


@dataclasses.dataclass
class BermejoCondeTracerTransport:
    """Mixin: apply the local mass fixer to every nodal tracer after transport.

    Overrides ``semi_lagrangian_transport`` (both stages of the Crank–Nicolson
    RK2 step call it, and the corrector's tracers are the step's final ones —
    tracers have no non-advective or implicit terms). The low-order solution
    is the linear-in-each-direction, unlimited interpolation of the same
    input field at the same departure points, in the same interpolation
    coordinate as the high-order one. ``M⁰`` is the input field's mass on the
    input surface pressure, ``M*`` the output's on the transported surface
    pressure.
    """

    local_mass_fixer: bool = dataclasses.field(default=False, kw_only=True)

    # The class it is mixed into supplies ``_tracer_interpolation_geometry(
    # departure) -> (PrimitiveDeparturePoints, VerticalNodes)``: the departure
    # points and nodes its transport interpolates on (mapped to ``ln s`` by the
    # log-pressure classes). It is not stubbed here, because this mixin sits
    # ahead of the log-pressure mixin in the MRO and a stub would shadow it.

    def semi_lagrangian_transport(self, state, departure):
        """Transport, then restore each nodal tracer's mass locally."""
        out = super().semi_lagrangian_transport(state, departure)
        nodal = [n for n in self.nodal_tracers if n in state.tracers]
        if not self.local_mass_fixer or not nodal:
            return out
        grid = self.coords.horizontal
        dep, nodes = self._tracer_interpolation_geometry(departure)
        low = semi_lagrangian.GridInterpolator(grid, "linear", None)
        m_in = cell_air_mass(self.coords, grid.to_nodal(state.log_surface_pressure))
        m_out = cell_air_mass(self.coords, grid.to_nodal(out.log_surface_pressure))
        fixed = {}
        for name in nodal:
            q_in = state.tracers[name]
            q_low = semi_lagrangian.transport_scalar(
                q_in, dep.full, nodes, low, vertical_order="linear")
            fixed[name] = bermejo_conde_correction(
                out.tracers[name], q_low, jnp.sum(q_in * m_in), m_out)
        return out.replace(tracers={**out.tracers, **fixed})


@dataclasses.dataclass
class LocallyConservativeSemiLagrangianHybrid(
    BermejoCondeTracerTransport,
    primitive_equations.SemiLagrangianPrimitiveEquationsHybrid,
):
    """Dinosaur's hybrid SL primitive equations with the local tracer fixer."""

    def _tracer_interpolation_geometry(self, departure):
        return departure, self._reference_vertical_nodes


@dataclasses.dataclass
class LocallyConservativeSemiLagrangianSigma(
    BermejoCondeTracerTransport,
    primitive_equations.SemiLagrangianPrimitiveEquations,
):
    """Dinosaur's sigma SL primitive equations with the local tracer fixer."""

    def _tracer_interpolation_geometry(self, departure):
        return departure, self._vertical_nodes
