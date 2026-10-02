# Reference-exact values with surrogate gradients

The physics ported from ECHAM and SPEEDY is piecewise. Cloud cover is clipped
to `[0, 1]`, the phase of new condensate is chosen by a temperature threshold,
a fall speed follows a power law whose slope is infinite at zero. Many of these
points lie inside the range the model visits at every step.

Two requirements pull in opposite directions there.

- **Fidelity.** The forward model has to be the reference formulation. The
  climate is validated and tuned against it, and a smoothed clip or a blended
  switch is a different model: a cover that never reaches exactly zero keeps
  condensate in cells the reference clears.
- **Differentiability.** The derivative of the reference formulation is
  useless at exactly these points. On a plateau it is identically zero, so an
  optimiser cannot see that a clipped quantity would respond. At a singular
  point it is unbounded and overflows a multi-step adjoint. Across a switch it
  ignores the switching variable.

`jcm.physics.surrogate_gradient.with_surrogate_gradient(exact, surrogate)`
meets both. The returned function evaluates `exact` and nothing else, so its
value is the reference's bit for bit. Its derivatives, in forward and reverse
mode and at every order, are those of `surrogate`, obtained by differentiating
`surrogate` itself.

This is the arrangement operational variational assimilation uses: the
nonlinear model runs the full physics, and the tangent-linear and adjoint
models run a regularised version of it (Janisková and Lopez 2013, *Linearized
physics for data assimilation at ECMWF*, in *Data Assimilation for
Atmospheric, Oceanic and Hydrologic Applications*, vol. II, Springer). In
machine learning it is the straight-through estimator (Bengio et al. 2013,
arXiv:1308.3432).

## When to use it

Use it where the reference derivative carries no usable information inside
the active range:

| Shape | Reference derivative | Example |
|---|---|---|
| Plateau of a clip or a `max` | identically zero | cloud cover clipped to `[0, 1]` |
| Threshold switch | zero with respect to the switching variable | ice or liquid chosen by temperature |
| Singular point | unbounded | `x ** 0.16` as `x` goes to zero |

Do not use it for an ordinary kink, where the derivative exists on both sides
and is bounded: automatic differentiation already returns a one-sided
derivative there and nothing is gained. Do not use it where the smooth form is
needed for the forward model to be well defined, such as a floor under a
denominator. That is a property of the value, and it stays in the value.

## Rules

1. **The derivative is a function's derivative.** `surrogate` is a named,
   smooth function next to `exact`. No hand-written slopes. The gradient the
   model returns is then the exact gradient of a nearby smooth model, which is
   a statement that can be tested.
2. **The width is static.** A smoothing width changes the derivative and not
   the value, so a gradient with respect to it has no meaning. It is a
   `pytree_node=False` field of the scheme's parameters, reachable from a
   Hydra override like any other field. Zero selects the reference
   derivative: the caller then calls `exact` directly.
3. **One surrogate per shared quantity.** Apply it where the quantity is
   formed, not where it is used. Every term of a budget that reads a clipped
   cover then sees the same derivative, and the tangent-linear budget closes
   as the nonlinear one does.
4. **Three tests.** `jcm.testing.check_surrogate_gradient` checks that the
   value equals `exact`'s exactly, that both derivative modes equal
   `surrogate`'s, and that they are adjoint. `check_gradients` on `surrogate`
   alone checks that the surrogate is smooth. A third test bounds the
   distance between `exact` and `surrogate`, which is what makes the
   gradient relevant to the model that runs.
5. **The science page says so.** Each use is a `differentiability` choice
   recorded in the process section of `docs/source/science/`, with the
   surrogate and its width.

## Chains of decisions

A quantity that is on only when several switches all pass is a product of
their 0/1 values, and the product rule gives it no derivative where two or
more of them are 0: each factor's surrogate slope is multiplied by another
factor's zero. Weight each link only where the links before it passed,
``w = s1·where(s1, s2·where(s2, s3, 1), 1)``. The value is the same product;
where the chain fails, its derivative is that of the first switch that failed,
which is the decision that turned the quantity off.

## Where it is used

| Site | Exact value | Surrogate | Width |
|---|---|---|---|
| Tiedtke ascent test, each interface (`convection/tiedtke_nordeng/switches.py::ascent_test`) | `cuasc`'s `pqu < zqold`, `zbuo > 0`, `pmfu ≥ 0.01·pmfub` (`mo_cuascent.f90:442-451`) | product of a rescaled logistic of the condensate and logistics of `zbuo` and of the flux fraction | `ascent_condensate_width` 1e-8 kg/kg, `ascent_buoyancy_width` 0.01 K, `ascent_mass_flux_width` 2e-3 |
| Tiedtke precipitation onset (`updraft.py`) | `zpbase − paphp1 ≥ zdnoprc` (l.454-455) | logistic of the depth excess | `precip_onset_width` 2000 Pa |
| Tiedtke deep/shallow type (`tiedtke_nordeng.py`) | `zdqcv > zhelp` (`mo_cumastr.f90:571-574`) | logistic | `deep_convergence_width` 2e-7 kg m⁻² s⁻¹ |
| Tiedtke `zlo1` gate (`tiedtke_nordeng.py`) | `zdqpbl > 0` and `zqumqe > zdqmin` (l.563-566) | logistic, and logistic relative to `zdqmin` | `sub_cloud_supply_width` 2e-7 kg m⁻² s⁻¹, `cloud_base_excess_width` 0.1 |
| Tiedtke `ldcum` | the chain `zlo1` → first ascent → final ascent (`mo_cuascent.f90:541`) | the links' surrogates, chained as above; it weights the whole ledger and the published mass fluxes | – |

The cloud cover and the one-moment microphysics use it as recorded in their
science pages.

## What a user has to know

The gradient is not the derivative of the value. A finite-difference check of
the whole model disagrees with automatic differentiation wherever a
surrogate is active, by design.

First-order optimisers and variational assimilation are unaffected in
practice: they need a direction that reduces the cost, and the surrogate's
gradient provides one where the reference's provides none. Methods that
compare the predicted and the actual change of the cost are affected. A line
search with a curvature condition, or a trust-region ratio, can reject a good
step because the two differ near a switch. Setting every width to zero
recovers the reference derivative for such a method, at the price of the
plateaux and singular points described above.
