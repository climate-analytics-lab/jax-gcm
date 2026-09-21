# Water-conservative tendency positivity limiter

Operator-split physics computes every term against the step-start state and
sums the tendencies; the dycore applies that sum once. `verify_tendencies`
(`jcm/physics_interface.py`) then guards positivity by capping each
non-negative field's tendency at the drain rate `-max(q, 0)/dt` — just enough
to empty the field, never below zero — so the moist physics downstream (which
divides by and takes logs of `q`) always sees `q >= 0`.

## Why the bare cap is not conservative for water

The cap is only sound for a field whose tendency is a *pure sink plus sources*.
The retained water fields — `specific_humidity`, `qc`, `qi`, `qr`, `qs` — are
not: `TTETKEVerticalDiffusion` mixes `q`/`qc`/`qi` and emits a **conservative
vertical redistribution**, a donor/receiver pair whose column integral is
zero. Because every term reads the step-start state, a donor layer can be
overdrawn by the redistribution *plus* the co-located cloud/convection sinks.
The bare cap then clamps the donor at `-q/dt` while the receiving layers keep
their gain, so `∫ q dp/g` over the column increases: the cap **creates water
mass**. This is the water-side twin of the aerosol defect fixed for the JAM
removal chain in {doc}`jam_aerosol_removal` (issue #806, split from #776).

The interface sees only the *summed* tendency, so it structurally cannot tell a
genuine sink from the donor half of a transfer, nor identify which layers
received the redistributed water.

## The fix: a column-conservative hole-filling reallocation

Where the layer masses `Δp` are available — the gridpoint driver
`compute_physics_step_gridpoint` obtains them from
`ComposablePhysics.pressure_thickness(state)`, built from the hybrid `(a, b)`
coefficients and the surface pressure with the same hydrostatic formula the
`moist_air_state` diagnostic uses for its `pressure_thickness` (with `abs()` so
the weight is orientation-independent) — the water-mass fields' cap is made
column-conservative:

1. Apply the per-cell cap as before; `added = capped − raw ≥ 0` is the
   mixing-ratio rate the cap injected to hold each layer non-negative.
2. Pressure-weight and sum it over the column: `S = Σ added·Δp` is the spurious
   column-integrated source.
3. Remove `S` again, distributed over the water left after the cap
   (`qnext = max(q,0) + dt·capped`), **proportional to each layer's remaining
   mass** — the standard mass-fixer / hole-filling choice, and the only
   defensible one when the individual receiving layers are not identifiable.
   Each layer is scaled toward, never below, zero.

The removed fraction of each layer's post-cap water is
`φ = clip(dt·S / Σ qnext·Δp, 0, 1)`, uniform across the column, giving
`qfinal = qnext·(1 − φ) ≥ 0`. When the column can supply the deficit
(`dt·S ≤ Σ qnext·Δp`) the reallocation is exact: `Σ qfinal_tendency·Δp =
Σ raw_tendency·Δp`, so the column water path tendency is preserved to
round-off. Each water species is conserved **independently** — the vdiff
redistribution conserves each phase separately, and borrowing across phases
would silently change the latent-heat partitioning.

The reallocation borrows from *every* layer that still holds water, in
proportion to how much it holds — not specifically from the layers that
received the redistributed water, because those are not identifiable from the
summed tendency. In the case the fix targets — a donor overdrawn by the vdiff
redistribution **plus** a co-located cloud/convection sink — this means the
borrowed water may come from a layer other than the true receiver. The column
water path is conserved exactly; what is accepted in exchange is that the
compensating removal can be *vertically misplaced* relative to where the cap
added it. That trade (exact column conservation, at the cost of the vertical
distribution of a small correction) is the best available given the interface
sees only the summed tendency, and it is strictly better than the bare cap,
which conserves nothing.

Two limiting cases fall out for free and are the reason this is the right
shape:

- **A genuine column-emptying sink** (e.g. runaway evaporation) drains every
  layer to zero, so there is no remaining water to borrow, `φ = 0`, and the
  result equals the bare cap. Genuine sinks *should* be able to empty the
  column; only the redistribution artefact is undone.
- **A column that cannot supply the whole deficit** (`dt·S >` the remaining
  column water — pathological) is drained to zero and the bounded residual
  stays in the water-positivity ledger as an honest, now-tiny, non-conservation
  that no amount of within-column reallocation can remove.

## Accounting and gradients

The `water_positivity_correction.*` ledger (issue #824) now records the **net**
per-field correction `applied − raw`; its pressure-weighted
`column_water_source` is the residual, ~0 to round-off where the reallocation
succeeds instead of the gross source the bare cap left. The whole
positivity-plus-conservation projection is a **primal-only** correction wrapped
in the same exact-primal straight-through estimator the cap already used
(`stop_gradient(result) + (tend − stop_gradient(tend))`): the forward pass is
the conserved value, the cotangent passes through to the producing tendency
unchanged, and the reallocation's division sits entirely under
`stop_gradient`, so a dry column (`Σ qnext·Δp = 0`, guarded with a safe
denominator) cannot poison the reverse-mode graph (cf. #558/#559).

## Scope

The reallocation applies to the water-mass fields
(`_WATER_CONSERVED_FIELDS = {specific_humidity, qc, qi, qr, qs}`). Number
concentrations (`qnc`, `qni`) and the VMR gases keep the bare per-cell cap:
they are not water mass and their redistribution conservation is a separate
concern. Aerosol and gas tracers are not capped at all — their removal is
bounded where it is produced by the operator split in
{doc}`jam_aerosol_removal`. The standalone `verify_tendencies` entry point has
no `Δp` and therefore applies only the bare cap; it is used by unit tests and
by any host that does not expose its vertical geometry.

## Code pointers

- `jcm/physics_interface.py` — `_conserve_water_column`,
  `_verify_tendencies_with_water_corrections`, `_WATER_CONSERVED_FIELDS`,
  `verify_tendencies`, `compute_physics_step_gridpoint`.
- `jcm/physics/composable_physics.py` — `ComposablePhysics.pressure_thickness`
  and the cached hybrid coefficients.
- `jcm/physics_interface_test.py` — `TestWaterConservativeLimiter` (f64
  column-budget closure, the bare-cap reproducer, broadcasting agreement,
  gradient identity/poison-freedom) and `TestComposablePressureThickness`.
