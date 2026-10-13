# Tracer-mass conservation in the dinosaur backend

The global water budget of a run closes when the dynamics neither creates nor
destroys ``∫ q·dp`` — then ``P − E + dPW/dt = 0``, with ``P`` and ``E`` the
fluxes the physics applies. The physics side closes by construction:
column by column, the applied water tendency is the diagnosed ``E − P`` plus
the ledgered positivity source of {doc}`water_positivity_conservation`. This
page states what the dinosaur backend conserves, how, and why the
semi-Lagrangian (SL) step needs a fixer for every species it transports,
humidity included.

## What each piece conserves

| piece | species | conserved | how |
|---|---|---|---|
| Eulerian transport (IMEX-RK SIL3; SPEEDY) | modal ``specific_humidity`` | ``∫ q·dp`` (measured residual 2e-4 mm/day at T31L8) | spectral flux-consistent transport; no fixer |
| projection of the physics tendency, ``T(P)`` | modal fields | per-level global mean exactly; ``∫ q·dp`` to < 1e-4 mm/day (−3.8e-7 kg/m² per step at T63L47) | the spectral transform keeps the ``(0, 0)`` coefficient |
| hyperdiffusion | modal fields | per-level global mean | the filters do not touch the ``(0, 0)`` coefficient |
| SL transport + ``_fix_nodal_tracer_mass`` | nodal tracers (cloud condensate, number, aerosol, gases) | ``∫ q·dp`` per tracer, to round-off | proportional global fixer every step |
| SL transport + ``_fix_humidity_mass`` | modal ``specific_humidity`` | ``∫ q·dp``, to round-off | cubic vertical interpolation, then the same proportional fixer |
| ``_conserve_global_mean_ps`` | ``ln pₛ`` | the global mean of ``ln pₛ``, not of ``pₛ`` | resets the ``(0, 0)`` coefficient; global-mean ``pₛ`` wanders by ±3 Pa over two weeks with no drift |

Both fixers restore the global mass to its **post-physics, pre-transport**
value: the reference is ``state + dt·T(P)``, so the physics' sources and
sinks stand and only what the transport and filters added or removed is
taken back. Each is one global factor per species per step (Diamantakis &
Flemming 2014, the proportional fixer), clipped to [2/3, 1.5]. A factor
outside that band is not transport error, and the clip lets it surface in the
budget diagnostics rather than be absorbed. A dry or empty field passes
through with factor exactly 1. Humidity is modal (it takes part in the
implicit q↔Tv coupling), so its integral is taken over ``to_nodal`` of the
modal coefficients with the nodal tracers' quadrature and ``dp``. The factor
scales the coefficients; the transform is linear, so the gridpoint field and
its integral scale exactly.

## Why modal humidity needs a fixer

SL transport is advective-form, ``q(arrival) = q(departure point)``.
Interpolation at the departure point is not mass-conserving. The nodal
tracers' fixer exists because, with the quasi-monotone limiter, the error is
one-signed wherever sinks leave sharp minima.

Humidity has a one-signed error of its own, in the vertical. Lagrange-linear
interpolation between two levels over-reads a convex profile by about
½·d·(Δσ − d)·∂²q/∂σ², for a displacement ``d`` within a layer of depth
``Δσ``. That is positive whatever the direction of the vertical motion.
Specific humidity falls off roughly as a power of σ, so it is convex through
the troposphere. Every SL step with linear vertical interpolation therefore
adds water: about 1 % of the column per day.

The physics never sees this water arrive, so it shows up only as a budget
residual: steady precipitable water with ``P − E`` positive. Measured at
T63L47 from the Stage-1 ECHAM warm states (one step attributed piece by piece,
and dynamics-only steps), in mm/day:

| configuration | linear vertical | cubic vertical | cubic + humidity fixer | Eulerian |
|---|---|---|---|---|
| 2M, full model (50 steps) | +0.236 | +0.001 | 0 (round-off) | — |
| 2M, dynamics only (20 steps) | +0.258 | −0.008 | — | −0.0006 |
| 1M, dynamics only | +0.246 | −0.011 | — | +0.0002 |
| JAM, dynamics only | +0.252 | −0.023 | — | — |
| SPEEDY T31L8, full model | +0.201 | — | — | +0.0002 |

The saved output of the same configurations carries the matching residual,
``P − E + dPW/dt = +0.22`` to ``+0.24`` mm/day (about 9 % of ``P``), in every
ECHAM package and season. Turning off the limiter, changing the horizontal
interpolation order, adding departure iterations or removing off-centring
changes nothing; only the vertical order matters.

## Why cubic vertical interpolation as well as the fixer

One global factor cannot undo a spatially structured error. Linear
interpolation's gain sits where vertical displacement and curvature are
large. A fixer alone would leave that local moistening in place and remove
the same amount everywhere else in proportion to ``q``.

Cubic vertical interpolation removes the systematic part at its source. It is
4-point Lagrange in reference σ, degraded to linear in the first and last
cells: dinosaur's ``vertical_order="cubic"`` and the IFS rule (Ritchie et
al. 1995; IFS Documentation Cy48r1, Part III). The fixer then closes the
small remainder to round-off. Cubic needs four levels, so a grid with fewer
levels defaults to linear.

The same interpolation transports temperature and winds. Temperature is
concave in σ through the troposphere, so linear interpolation under-reads it.
In the dynamics-only steps the mass-integrated ``c_p·T`` change over the
dynamics step is:

| | linear vertical | cubic vertical | Eulerian |
|---|---|---|---|
| 2M | −14.2 | −2.0 | −4.1 |
| 1M | −13.6 | −3.0 | −2.6 |
| JAM | −13.2 | −2.1 | — |

All in W/m². Kinetic energy behaves alike across the variants, so the total
energy change is ≈ −11 W/m² with linear and ≈ 0 with cubic.

Cost on CPU at T63L47 (32 cores, ECHAM 2M tracer set): the dynamics step
takes 1.34 s with cubic against 0.84 s with linear, and the humidity fixer
adds 0.02 s. A full ECHAM step is about 3 s, so the whole change costs about
20 %. The GPU cost is not measured here.

## The reference

ECHAM6's default transport, ``tpcore`` (``control.f90``: ``iadvec = tpcore``),
is the flux-form semi-Lagrangian scheme of Lin & Rood (1996). It moves ``q``,
``xl``, ``xi`` and the tracers with mass fluxes consistent with the continuity
equation, and corrects the air mass (``scan1.f90``: ``psm1cor``, ``pscor``;
``mo_tpcore.f90::tpcore_tendencies``), so water is conserved by construction.

ECHAM's semi-Lagrangian option applies a mass fixer to the same species
(``mo_semi_lagrangian.f90::mass_fixer``, ``fixer``). That is the Rasch &
Williamson (1990) form, ``q_fix = α·F·q₂·|q₂ − q₁|^β``, with ``F = 1, β = 1.5``
or ``F = η, β = 1``. It puts the correction where the transport changed the
field. **That alternative is not implemented.** With cubic vertical
interpolation the residual it would redistribute is ~1e-3 of the linear
error, and the proportional form already serves the nodal tracers. It is the
next step if the spatial pattern of the residual ever matters; the Bermejo &
Conde (2002) weighting used for IFS tracers is the other candidate.

## Configuration

- ``dycore.sl_vertical_interpolation``: ``null`` (default; cubic, linear below
  four levels), ``cubic`` or ``linear``. Under ``DinosaurDycore(sl_options=)``
  it is ``vertical_interpolation_order``.
- ``dycore.humidity_mass_fixer``: default ``true``. Under ``sl_options`` it is
  ``humidity_mass_fixer``. ``sl_options["mass_fixer"] = False`` switches every
  fixer off.
- Both are ignored by the Eulerian core, whose step is unchanged: SPEEDY runs
  are bit-identical with or without them.

To check a run, take the global means of the saved daily output:
``S = P − E + dPW/dt``, with ``P`` = ``surface_exchange.precipitation``,
``E`` = ``surface_exchange.evaporation``, and ``PW`` from ``specific_humidity``
plus the prognostic condensate with Δp from ``pressure_half``. ``S`` should be
zero to the precision of the daily means.

## References

- Bermejo, R. & Conde, J. (2002). A conservative quasi-monotone
  semi-Lagrangian scheme. *Mon. Wea. Rev.* 130, 423–430.
- Diamantakis, M. & Flemming, J. (2014). Global mass fixer algorithms for
  conservative tracer transport in the ECMWF model. *Geosci. Model Dev.* 7,
  965–979.
- Lin, S.-J. & Rood, R. B. (1996). Multidimensional flux-form semi-Lagrangian
  transport schemes. *Mon. Wea. Rev.* 124, 2046–2070.
- Rasch, P. J. & Williamson, D. L. (1990). Computational aspects of moisture
  transport in global models of the atmosphere. *Q. J. R. Meteorol. Soc.* 116,
  1071–1090.
- Ritchie, H., Temperton, C., Simmons, A., Hortal, M., Davies, T., Dent, D. &
  Hamrud, M. (1995). Implementation of the semi-Lagrangian method in a
  high-resolution version of the ECMWF forecast model. *Mon. Wea. Rev.* 123,
  489–514.
- ECMWF (2023). IFS Documentation Cy48r1, Part III: Dynamics and numerical
  procedures.
