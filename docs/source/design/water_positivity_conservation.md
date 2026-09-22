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

## Why the cap itself is the reference atmosphere update

Working the sequential (ECHAM `physc`-order: vertical diffusion first, then
the cloud chain) reference through the overdraw case shows something
counter-intuitive: **the bare cap already reproduces the sequential
atmosphere state.** Let the vdiff solve move `t·dt` out of a donor holding
`q`, and a co-located sink (computed on the step-start state, each
individually self-limiting) want `k·dt`, with `t + k > q/dt`. Sequentially,
vdiff moves its full `t·dt` (its implicit solve saw the full `q`), and the
sink — recomputed on what remains — takes the rest: the donor ends at zero
and the receivers keep `t·dt`. The summed-then-capped update gives exactly
the same: donor floored at `−q/dt` → zero, receivers keep their raw gain.
The receivers' water is *real* — it physically left the donor — so no
reallocation may touch it.

Two reallocation designs were tried and **rejected on evidence** before this
was understood. Borrowing the cap's source from the column's remaining water
(mass-fixer hole-filling) drained untouched cloud layers to zero in one step;
borrowing only from the column's same-step gains still zeroed the receivers'
legitimate gains. Both wiped a seeded cloud before radiation could see it —
caught by `cre_guard_test.py` in CI on the first version of this PR, and
reproduced per-layer with an instrumented column. The failure is structural:
at the interface, a donor's cap correction and its receivers' gains look the
same whether the overdraw came from the transfer or from the co-located sink,
and in *both* cases the sequential reference keeps the receivers whole.

What the raw sum actually gets wrong is the **sink's own removal**: it was
computed against water the redistribution had already taken, so its product —
precipitation, or the vapour credited by condensate evaporation — is
over-reported by exactly the cap correction. The budget defect is a
double-counted removal, not wrongly-placed atmosphere water.

## The fix: ECHAM's negative-water correction — charge the local vapour

ECHAM repairs exactly this in `mo_cloud.f90` section 8.4 ("Corrections:
Avoid negative cloud water/ice"): after clipping `xl`/`xi` non-negative it
charges the clip to the same cell's vapour with the matching latent heat,

```fortran
zdxlcor = (zxlp1 - zxlold)/ztmst          ! the positivity correction
pxlte = pxlte + zdxlcor
pxite = pxite + zdxicor
pqte  = pqte  - zdxlcor - zdxicor         ! vapour pays for it
ptte  = ptte  + zlvdcp*zdxlcor + zlsdcp*zdxicor
```

i.e. the condensate the correction *adds* is materialised as a real
condensation/deposition event: vapour is consumed, latent heat is released,
and total water in the cell is unchanged. `verify_tendencies` now follows
this pattern for the water-mass tracers (`qc`/`qr` liquid with `alhc`,
`qi`/`qs` frozen with `alhs`, over `cpd`):

1. Per-cell positivity cap on every non-negative field, unchanged.
2. The summed condensate cap corrections of each cell are subtracted from
   that cell's vapour tendency, bounded by the vapour the cell can still
   supply after its own cap (`max(q,0)/dt + capped_dqdt`, scaled down
   uniformly across both phases where the correction exceeds it), with the
   matching phase-split latent heat added to the temperature tendency.
3. A final exact drain-rate re-cap `max(·, −max(q,0)/dt)` guards the
   ulp-level float undershoot of the separate roundings (a float32 P1 from
   the Codex review of #864).

In the common overdraw case — condensate evaporation double-counted against
a redistribution — the charge removes *precisely* the phantom vapour that
evaporation over-credited, and the heating cancels its phantom evaporative
cooling; the repair is exact, not approximate. Where the sink's product was
precipitation instead, the charge still closes the total (atmosphere +
surface flux) water budget, at the cost of locally converting the
over-report into a vapour draw — ECHAM 8.4 makes the same choice, charging
vapour regardless of which process produced the negative. A cell whose
vapour cannot absorb the whole correction keeps the remainder in the
water-positivity ledger as an honest, bounded artificial source —
`specific_humidity`'s own cap corrections likewise (they have no local
donor; ECHAM prevents negative `q` at the producing terms instead).

The correction is **cell-local**, so it needs no layer masses, no column
reductions, and no host geometry: the standalone `verify_tendencies` and the
gridpoint driver apply the identical treatment, and every other layer's
tendency passes through bitwise — which is what makes the seeded-cloud
regression structurally impossible rather than merely retested.

## Accounting and gradients

The `water_positivity_correction.*` ledger (issue #824) records the **net**
per-field correction `applied − raw`: positive on the capped condensate,
negative on the charged vapour, so the per-cell sum over water fields is the
sign-definite residual — ~0 wherever the vapour absorbed the whole
correction, the honest remainder where it could not. The temperature charge
is deliberately not part of the water ledger (it is energy, not water; it
keeps moist static energy consistent with the materialised phase change).
The whole positivity-plus-charge projection is a **primal-only** correction
wrapped in the same exact-primal straight-through estimator the cap already
used (`stop_gradient(result) + (tend − stop_gradient(tend))`): the forward
pass is the corrected value, the cotangent passes through to the producing
tendency unchanged, and the charge's division sits entirely under
`stop_gradient` with a safe denominator, so a correction-free or dry cell
cannot poison the reverse-mode graph (cf. #558/#559).

## Scope

The vapour charge applies to the water-mass tracers, split by phase for the
latent heat (`_LIQUID_WATER_TRACERS = {qc, qr}` with `alhc`,
`_ICE_WATER_TRACERS = {qi, qs}` with `alhs`). Number concentrations (`qnc`,
`qni`) and the VMR gases keep the bare per-cell cap: they are not water mass,
so there is no vapour to charge. Aerosol and gas tracers are not capped at
all — their removal is bounded where it is produced by the operator split in
{doc}`jam_aerosol_removal`.

## Code pointers

- `jcm/physics_interface.py` — `_verify_tendencies_with_water_corrections`
  (the cap, the vapour charge and the latent heat),
  `_LIQUID_WATER_TRACERS` / `_ICE_WATER_TRACERS`, `verify_tendencies`,
  `compute_physics_step_gridpoint`.
- `jcm/physics_interface_test.py` — `TestWaterPositivityVapourCharge` (f64
  per-cell and column total-water closure, the latent-heat charge by phase,
  the sink-overdraw / seeded-cloud regression, the dry-cell residual, the f32
  drain-rate bound, broadcasting agreement, gradient
  identity/poison-freedom).
- `jcm/physics/echam/cre_guard_test.py` — the end-to-end guard that a seeded
  cloud survives the limiter and stays radiatively active.
- Reference: ECHAM `mo_cloud.f90`, section 8.4 (loop 821), at
  `/data/…/echam6.3.0-ham2.3-moz1.0.r7492/src/`.
