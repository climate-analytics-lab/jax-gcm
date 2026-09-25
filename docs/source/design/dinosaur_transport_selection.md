# Dinosaur transport selection: semi-Lagrangian vs Eulerian

The dinosaur backend offers two transport schemes, chosen per run by
``DinosaurDycore(advection=...)`` (Hydra: ``dycore.advection``):

| scheme | primitive / step | who gets it |
|---|---|---|
| ``semi_lagrangian`` | ``SemiLagrangianPrimitiveEquations[Hybrid]`` + ``semi_lagrangian_crank_nicolson_rk2`` (off-centred 0.2) | the default; every tracer-carrying composition (ECHAM, JAM, …), Held–Suarez |
| ``eulerian`` | ``PrimitiveEquations[Hybrid]`` + ``imex_rk_sil3`` | SPEEDY (tracer-free) |

## The rule

1. **Tracers force semi-Lagrangian.** Spectral (Eulerian) transport of a sharp
   tracer field rings negative every step; it was the documented cause of the
   aerosol-microphysics NaNs that motivated the SL core (#521, and
   {doc}`dinosaur_sl_jam_configuration`). So no path may put a declared tracer
   on it: ``advection="eulerian"`` with non-empty ``tracer_specs`` raises at
   construction (or when the tracers are registered later).
2. **An explicit choice wins** otherwise.
3. **``advection=None`` (the default) lets the physics decide.**
   ``Model`` calls ``DinosaurDycore.resolve_advection(physics.preferred_advection())``
   once the tracer set is known. ``PhysicsTerm.preferred_advection()`` returns
   ``None`` (no preference) by default; ``ComposablePhysics`` aggregates with
   *any ``semi_lagrangian`` wins, else any ``eulerian``, else ``None``* — a term
   asking for SL may need it for correctness (nodal, monotone transport), while
   an Eulerian preference is only a fidelity/cost one. An unresolved ``None``,
   and an Eulerian preference on a composition that carries tracers, both run
   semi-Lagrangian (rule 1).

The SPEEDY terms (``SpeedyTermBase``) declare ``"eulerian"``. A SPEEDY
composition that adds a tracer-declaring term therefore still resolves to
semi-Lagrangian, automatically and without error.

## Why SPEEDY runs Eulerian

- **SL buys SPEEDY nothing.** SPEEDY declares no extra tracers, and
  ``specific_humidity`` is a *modal* field under both schemes (it takes part in
  the implicit q↔Tᵥ coupling), so the ringing SL removes never arises. SL's
  other benefit, a longer stable timestep, is not used: SPEEDY's step is set by
  its explicit physics ({doc}`speedy_variable_levels`).
- **Heritage.** SPEEDY was formulated and tuned on an Eulerian spectral
  dycore, and the jcm 1.x/2.x SPEEDY climatology was produced on the Eulerian
  core.
- **SL is expensive on CPU.** Departure-point transport is gather-bound under
  XLA:CPU: the cubic-stencil ``interpolate_3d`` gathers write out the full
  stencil tensor for every field, several times per step. At T31L8 with
  realistic forcing, dt = 30 min, float32, on 8 CPU cores:

  | transport | s / simulated day |
  |---|---|
  | Eulerian (IMEX-RK SIL3) | 0.84 |
  | semi-Lagrangian (cubic, CN-RK2) | 3.63 |

  In a profile, SL transport plus departure points is ~64 % of the SL step,
  while physics costs the same under both schemes. SPEEDY is the configuration
  most often run on laptops (teaching, calibration), so this cost fell on
  exactly the users with the least compute.

## Why not Eulerian for every tracer-free configuration

Held–Suarez is also tracer-free, but it has no Eulerian heritage to preserve
and runs mostly on GPU, where SL is cheap. Tying the scheme to a declared
preference, not to the absence of tracers, keeps the SL core the default for
everything except the one package that asks otherwise. Any other package can
opt in by overriding ``preferred_advection``.
