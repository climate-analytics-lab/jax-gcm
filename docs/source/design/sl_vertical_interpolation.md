# Semi-Lagrangian vertical interpolation coordinate

The dinosaur backend's semi-Lagrangian (SL) step transports every prognostic
field — the winds, ``T′``, humidity and the nodal tracers — by interpolating
it at the departure point of the trajectory arriving at each grid point. This
page states the coordinate in which the **vertical** stage of that
interpolation is done, and why it is not the trajectory coordinate itself.

## What the code does

The trajectories are solved in dinosaur's own vertical coordinate: on hybrid
grids the reference sigma ``s = (A + B·pₛ_ref)/pₛ_ref`` with ``ṡ`` diagnosed
from the interface mass flux, on sigma grids ``σ``. The departure ``s`` is
clipped to the range of the layer centres (no extrapolation beyond the top and
bottom levels). Then:

- **hybrid grids** (ECHAM L47/L95) interpolate in ``ln s``
  (``dycore.sl_vertical_coordinate: log_pressure``, the default there): both
  the departure point and the layer-centre nodes are mapped by the logarithm,
  and the Lagrange weights are computed on the mapped nodes. At the
  pure-pressure top ``ln s`` is log-pressure; near the surface
  ``ln s ≈ s − 1``.
- **sigma grids** (SPEEDY / Held-Suarez L8) interpolate in ``σ`` itself
  (``sigma``, dinosaur's native rule), which is also available on hybrid
  grids by override.

Everything else is unchanged: the stencil (4-point cubic Lagrange, linear in
the first and last cells, ``dycore.sl_vertical_interpolation``), the bracketing
cell, the quasi-monotone limiter of the tracers, the mass fixers. The mapping
is monotone, so it moves no departure point to another cell; only the weights
within the stencil change. The code is
``jcm/dycore/dinosaur/log_pressure_interpolation.py``, which overrides
``semi_lagrangian_transport`` of dinosaur's two SL primitive-equation classes
and so applies to both stages of the Crank–Nicolson RK2 step.

## Why not interpolate in ``s``

Lagrange weights are well behaved on nodes that are close to uniformly spaced
in the interpolation coordinate. The L47 grid is not, in ``s``, at its top:
its four top full levels are at 1.0, 4.3, 11.1 and 23.1 Pa, each layer about
twice as thick in ``s`` as the one above it. That is the spacing of a grid
designed to be quasi-uniform in height, i.e. in ``ln p``. Interpolating in
``s`` there has two consequences.

**The linear top cell under-reads vertical advection.** A departure point a
short way ``δ`` below the top level draws on the second level with weight
``δ/(s₂ − s₁)`` when the cell is linear in ``s``, and
``(δ/s₁)/ln(s₂/s₁)`` when it is linear in ``ln s``. For the L47 top cell the
first is 0.44 of the second. A profile that is linear in height (isothermal or
of constant lapse rate) is linear in ``ln p``, so the ``s`` rule credits the
top level with less than half of the advection from below, and the cubic
cells beneath it over-read the same profile: one SL step of the L47 grid
advects 0.44, 1.37 and 1.16 of the exact displacement at the 1.0, 4.3 and
11.1 Pa levels (``ln s``: 1.000, 1.000, 0.998). Under the upwelling of the
summer mesosphere, where temperature increases downward, the top level
receives less than half of the warm-air advection while the adiabatic cooling
of the same upwelling acts in full, and the level below receives too much.

**The cubic cells amplify grid-scale vertical structure.** On geometrically
spaced nodes the 4-point Lagrange interpolant of a 2Δz pattern overshoots
the nodes for departure points a short way below a node. The largest gain
over one interpolation, maximised over the position in the cell
(``|Σ wⱼ(−1)ʲ|``; 1 is neutral):

| L47 cell (levels, Pa) | gain in ``s`` | gain in ``ln s`` |
|---|---|---|
| 4.3 – 11.1 | 1.099 | 0.9997 |
| 11.1 – 23.1 | 1.028 | 0.9994 |
| 23.1 – 42.6 | 1.009 | 0.9991 |
| 42.6 – 73.6 | 1.006 | 0.9990 |
| 73.6 – 122 | 1.002 | 0.9989 |
| lowest interior cell (~972 – 996 hPa) | 1.016 | 1.022 |

Under persistent upwelling the departure points stay on the overshooting side
and the step is anti-diffusive. Iterating the vertical interpolation operator
of the L47 grid (the model's own ``_vertical_stencil``, 12-minute step) under
a uniform upward velocity, an initial 2Δz pattern in the top eight levels
evolves as:

| upwelling | rule | amplitude after 1 / 5 / 10 / 20 days |
|---|---|---|
| 0.5 cm/s | cubic in ``s`` | 1.08 / 1.31 / 1.50 / 2.15 |
| 0.5 cm/s | cubic in ``ln s`` | 0.95 / 0.69 / 0.49 / 0.28 |
| 0.5 cm/s | linear in ``s`` | 0.96 / 0.84 / 0.71 / 0.53 |
| 2 cm/s | cubic in ``s`` | 1.26 / 2.14 / 3.65 / 6.73 |
| 2 cm/s | cubic in ``ln s`` | 0.76 / 0.27 / 0.16 / 0.06 |

The summer-mesosphere residual upwelling is of order 1 cm/s.

The near-surface cell is equally non-uniform in either coordinate (the
boundary-layer levels thin towards the ground), so the change there is
immaterial and dominated by boundary-layer vertical diffusion. Below its top
level the L95 grid is spaced by a ratio of about 1.2 per level, where the
cubic cells are stable in either coordinate; its top cell (0.995 to 2.34 Pa)
gives the second level 0.63 of the log-pressure weight when it is linear in
``s``, so the coordinate matters at the L95 lid level too, by less. The L8 sigma grids are close to
uniform in ``σ`` and slightly less so in ``ln σ`` (largest gain 1.000 in
``σ`` against 1.002 in ``ln σ``), which is why sigma grids keep ``σ``.

## Measured consequence

In ``ma-t63-l47`` from the January and July warm states, with the
level-matched FZJ CMIP7 ozone, the summer polar cap of the 1 Pa level cools
from 247 K to ~165 K in the first eight days under either rule. From there:

| rule | summer cap at 1 / 4.3 / 11 Pa, from day 9 until any runaway | coldest 1 Pa cells | ice at 1-4 Pa |
|---|---|---|---|
| cubic in ``s`` | 156-164 / 206-217 / 220-230 K | 136 K by day 11 (Jan), 13 (Jul) | from day 9 (Jan), 11 (Jul); runaway from day 11 (Jan), 13 (Jul) |
| linear in ``s`` (Jan) | 131-158 / 171-206 / 209-220 K | 128 K by day 19 | from day 10; runaway on day 20 |
| cubic in ``ln s`` | 161-170 / 194-210 / 221-228 K | 156-165 K | none in 20 days |

Cubic interpolation in ``s`` splits the 1 Pa and 4.3 Pa levels apart (the
lid under-reads the advection from below, the next level over-reads it).
Linear interpolation in ``s``, stable in the cubic cells but with the same
linear top cell, is colder still at the lid and has no warm 4.3 Pa level,
which isolates the top cell as the cause of the cold bias and the cubic cells
as the source of the warm 4.3 Pa level. In ``ln s`` the cap keeps a monotone
mesospheric lapse.

## Reference

ECHAM has no semi-Lagrangian dynamics: its vertical advection is the
energy-conserving centred difference of Simmons & Burridge (1981) in
``dyn.f90``, so there is no ECHAM rule to follow. Dinosaur's stencil is the
IFS rule (cubic Lagrange in the hybrid coordinate, linear in the first and
last cells); the IFS applies it on grids whose top levels are far more
closely spaced than L47's. The choice made here is the coordinate in which
the grid's reference levels are quasi-uniform and in which the fields of the
stratosphere and mesosphere vary smoothly; it is a property of the
interpolation, not of the trajectories, and leaves the transport's
no-extrapolation and monotonicity guarantees as they were.
