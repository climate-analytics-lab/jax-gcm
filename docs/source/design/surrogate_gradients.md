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

1. **The derivative is a function's derivative, and there is exactly ONE of
   it.** `surrogate` is a named, smooth function next to `exact`. No
   hand-written slopes, and no mixing-and-matching a different surrogate per
   argument of a multi-argument `exact`: a tangent assembled from different
   functions' partials is not the gradient of any single smooth function, so
   its Hessian need not be symmetric and nested `jax.grad` becomes
   order-dependent (jax-gcm#1087 tried exactly this for `demott2010_inp`'s
   `n_aer_coarse ** exponent` and it was wrong for that reason, not fixable
   by patching the wrapper: `∂/∂exponent = n**exponent·log(n)` pins the
   surrogate to `n**exponent + c(exponent)` on any region where it must hold
   exactly, whose `∂/∂n` is then the UNBOUNDED exact one — the three demands
   "exact ∂/∂exponent", "bounded ∂/∂n" and "one consistent function" cannot
   all hold at once). Where one argument's true derivative is singular and
   another's is exact, pick the region instead: `exact` is usually already
   piecewise (a `where` on the singular argument), and the SAME single
   surrogate can differ by region too, as long as it is the one function
   whose derivative both arguments' tangents come from in each region (see
   `demott2010_inp` below: zero in both partials where the value is
   identically zero, one shared floored surrogate elsewhere).
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

That derivative is the quantity's as if the later links had passed, so the
chain needs the quantity to be one the scheme could produce then. A flux
proportional to the variable a later link tests is not: behind SPEEDY's
shallow-convection test `drh >= 0` the flux `a_shc·drh` is negative where the
test fails, and a chain through it hands the tangent a reversed flux. A block
like that is one surrogate whose smooth function evaluates each branch as the
scheme does (`speedy_vdiff._pbl_moisture_flux`).

## Where it is used

| Site | Exact value | Surrogate | Width |
|---|---|---|---|
| Tiedtke ascent test, each interface (`convection/tiedtke_nordeng/switches.py::ascent_test`) | `cuasc`'s `pqu < zqold`, `zbuo > 0`, `pmfu ≥ 0.01·pmfub` (`mo_cuascent.f90:442-451`) | product of a rescaled logistic of the condensate and logistics of `zbuo` and of the flux fraction | `ascent_condensate_width` 1e-8 kg/kg, `ascent_buoyancy_width` 0.01 K, `ascent_mass_flux_width` 2e-3 |
| Tiedtke precipitation onset (`updraft.py`) | `zpbase − paphp1 ≥ zdnoprc` (l.454-455) | logistic of the depth excess | `precip_onset_width` 2000 Pa |
| Tiedtke deep/shallow type (`tiedtke_nordeng.py`) | `zdqcv > zhelp` (`mo_cumastr.f90:571-574`) | logistic | `deep_convergence_width` 2e-7 kg m⁻² s⁻¹ |
| Tiedtke `zlo1` gate (`tiedtke_nordeng.py`) | `zdqpbl > 0` and `zqumqe > zdqmin` (l.563-566) | logistic, and logistic relative to `zdqmin` | `sub_cloud_supply_width` 2e-7 kg m⁻² s⁻¹, `cloud_base_excess_width` 0.1 |
| Tiedtke `ldcum` | the chain `zlo1` → first ascent → final ascent (`mo_cuascent.f90:541`) | the links' surrogates, chained as above; it weights the whole ledger and the published mass fluxes | – |
| DeMott (2010) INP, `n_aer_coarse ** exponent` (`clouds/lohmann_2m/deposition_freezing.py::demott2010_inp`) | the fractional power, 0 at `n_aer_coarse <= 0` | `where(n_aer_coarse > 0, max(n_aer_coarse, floor) ** exponent, 0)` -- ONE function; at `n_aer_coarse <= 0` the value is identically 0 for every `exponent`, so both partials are exactly 0 there (not DeMott's own, since the fit does not apply); for `0 < n_aer_coarse`, including below the floor where the fit is itself undefined, both partials are this one surrogate's | `_DEMOTT_N_AER_FLOOR_CM3` 0.01 cm⁻³, the fit's own calibration floor |

The cloud cover and the one-moment microphysics use it as recorded in their
science pages.

SPEEDY's switches use it through the helpers of
`jcm.physics.speedy.smoothing` (`surrogate_gate`, `surrogate_pos`,
`surrogate_min`, `surrogate_max`, `surrogate_clip01`, `surrogate_sqrt`), each
the hard operation's value with the derivatives of the matching smooth
function. Every width defaults to 0, the reference derivative. SPEEDY's
parameters are `tree_math` structs, which have no static fields, so its
widths are pytree leaves; each surrogate holds its width fixed
(`stop_gradient`), so a gradient with respect to a width is exactly zero, as
for a static one.

| Site | Exact value | Surrogate | Width |
|---|---|---|---|
| SPEEDY humidity trigger (`speedy_convection.py::_humidity_trigger`) | both RH tests `> rhbl`, gating `qdif` (a value jump) | one surrogate for the whole block: a sigmoid gate on the PBL-top test times a softplus surface excess (the surface test is the excess's own sign), so a column failing both tests keeps a derivative; the plume is computed to `ktop1` across the trigger branch so the tangent's column moisture balances its rain | `convection.trigger_smoothing`, an RH fraction (scaled by `qsat`) |
| SPEEDY convective precipitation onset (`speedy_convection.py::_smooth_precnv`) | `max(fuq − fmass·qsatb, 0)`, the plume's cloud-top moisture flux above saturation | `cbmf · softplus(unit_excess)`, the onset per unit cloud-base mass flux: a switched-off plume sits exactly on the hinge's corner, where AD's tie rule gives its tangent half the plume's rain, in either sign of the excess; per unit, the corner has the plume's own rain yield (at width 0 the reference's right-hand derivative) and a running plume keeps the reference derivative | `convection.precnv_smoothing`, g/kg per unit cloud-base mass flux |
| SPEEDY grid-point-storm heating cap (`speedy_condensation.py`) | `min(-dqlsc, cap)` | hyperbolic minimum | `condensation.cap_smoothing`, a fraction of the cap |
| SPEEDY cloud cover (`speedy_shortwave.py::clouds`) | RH hinge and max over reference levels, the `pmaxcl` cap, `sqrt(max(eps, pr))`, the saturation at 1, the stratocumulus clips | softplus / hyperbolic min-max / softplus-pair clip; for the square root, `sqrt(pr+ + (w*pmaxcl)²)`, whose slope is bounded by `1/(2*w*pmaxcl)` where the reference's reaches ~1.6e4 per mm/day in drizzle | `shortwave_radiation.cover_smoothing`, a cover fraction |
| SPEEDY soil-moisture-limited evaporation (`speedy_surface_flux.py`) | `max(wetness*qsat - q, 0)` and its `evap > 0` activity weight in the skin balance | softplus and its sigmoid derivative | `surface_flux.evap_smoothing`, g/kg |
| SPEEDY shallow convection and moisture diffusion (`speedy_vdiff.py::_pbl_moisture_flux`) | at the lowest interface, `dmse >= 0` (jcm: `> 0`, differing only at a tie) choosing shallow convection `a_shc·max(drh, 0)` or moisture diffusion `a_vdi·drh` where `drh > drh0` (value jumps); the same onset above the PBL | one surrogate for the lowest interface's moisture flux: the two regimes crossfaded by a sigmoid in `dmse`, shallow convection's `max(drh, 0)` a softplus and the onset a sigmoid, so the `dmse` derivative is the jump the flux makes and is zero where neither regime carries a flux (below); sigmoid gates for the onsets above; softplus for the one-sided heat flux | `vertical_diffusion.mse_gate_smoothing` J/kg, `rh_gate_smoothing` an RH fraction |
| SPEEDY boundary-layer condensation threshold (`speedy_condensation.py`) | `max(rhref[-1], rhblsc)`, the bottom level's threshold: the losing side (`rhblsc`, or the profile's `rhlsc` and `drhlsc`) has no derivative there, and below the profile value `rhlsc − drhlsc·(1 − σ²)` `rhblsc` has none at all | hyperbolic maximum | `condensation.bl_threshold_smoothing`, an RH fraction |
| SPEEDY stability correction (`speedy_surface_flux.py::_smooth_clipped_excess`) | the surface-to-air excess clipped to `+dtheta` and, after the `astab` damping, to `−dtheta`: the factor is `1 ± fstab` on the plateaux, with no derivative in `dtheta` | one smooth clip of the damped excess: hyperbolic minimum at `+dtheta`, hyperbolic maximum at `−dtheta`, continuous through zero excess (the damping's kink stays) | `surface_flux.stability_smoothing`, K |
| SPEEDY convection surface-pressure switch (`speedy_convection.py::_smooth_mass_flux_pressure_factor`) | `psa > psmin` masking `qdif`, and `psa·min(1, (psa − psmin)·2/(1 − psmin))` on the cloud-base mass flux: no derivative in `psmin` below the switch or above the cap, which at the default `psmin` is everywhere the surface pressure exceeds 900 hPa | one softplus-pair ramp in surface pressure carrying the switch (the factor is 0 at and below `psmin`); the mass flux is formed from the unmasked excess so the derivative reaches the plume, as for the trigger. The switch and the trigger are independent decisions multiplied together, so a column failing both keeps a zero derivative in each | `convection.psmin_smoothing`, normalised surface pressure |

The square root is the singular-point case: in a multi-step adjoint of a
high-resolution coupled run its reference slope produced isolated grid-point
sensitivities orders of magnitude above their neighbours wherever it barely
rained.

The vertical diffusion's regime split is the case of a jump between two
branches. Its smooth function crossfades the branches as the scheme evaluates
them, so the derivative across the switch is the jump the flux makes, and none
where crossing it changes nothing. Assembled decision by decision instead
(each branch's sigmoid gates chained), it gave a column just below the switch
with `drh < 0`, where neither branch carries a flux, the derivative of a
reversed shallow-convection flux: in the tangent-linear model a positive
feedback on the bottom humidity, since `dmse` grows with it. On T30 L8 twin
forecasts at `mse_gate_smoothing` = 200 J/kg, a column that stayed within a
width or two of the switch for four days multiplied the model's humidity
tangent 2–10× every 12 hours, and the whitened 14-day tangent grew 4–220×
with the forecast bit-identical. The crossfade of the jump the flux makes
keeps that tangent within 3 % of the width-0 one. Across the switch,
shallow convection dries the bottom layer faster than the diffusion does
wherever `a_shc > a_vdi` (at the default relaxation times, unless deep
convection halves `a_shc` under an inversion with `qsat[kx]` below about
0.64·`qsat[nl1]`), so the jump damps the bottom humidity's tangent rather
than feeding it. Where it reverses, the switch repels the forward model too,
which leaves it rather than lingering, and the tangent's growth there is a
sensitivity of the model rather than of the surrogate.

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
