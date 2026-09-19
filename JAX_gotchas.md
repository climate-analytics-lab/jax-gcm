
# JAX Gotchas

## VMap

When using `vmap` in JAX, there are some gotchas to be aware of. One common issue is with scalar values that are looped over using `vmap`. In such cases, `vmap` treats the scalar as a vector, which can lead to unexpected behavior.

For example: in the original code we passed in one element of a vector called sig and checked its value. When vectorizing over the same function with `vmap` the comparison `sig <= 0.0` because `vmap` considers `sig` as a vector, even though conceptually is a scalar. To work around this, you can use alternative approaches like `jnp.less_equal` or `jnp.where` to handle the comparison correctly.

Here's an updated version of the code snippet:
`qsat = jnp.where(sig <= 0.0, 622.0 * qsat / (ps[0,0] - 0.378 * qsat), 622.0 * qsat / (sig * ps - 0.378 * qsat))`

## If/else statements

We can't have traditional if/else statements that depend on jax types. The solution is fairly straightforward, it requires the use of jax.lax.cond(). This requires you to write a function to execute if the conditional is true and a function to execute if it is false. There is an option to pass an operand to both functions (i.e. a tuple, array, etc). Example use cases can be found in surface_flux.py (this works for both forward passes of the function and gradients).

```
flag = True
def pass_fun(operand):
    return operand

def update_fun(operand):
    t, s, u, v = operand
    # some operations inserted here

    return (t,s,u,v)

t,s,u,v = jax.lax.cond(flag, update_fun, pass_fun, operand=(t,s,u,v))
```

## Gradients that are NaN or infinite where the forward pass is fine

Reverse-mode AD differentiates *every* branch, including the one a
`jnp.where`/`jnp.clip` discards, and applies the mask afterwards. So a guard
that makes the forward value finite does nothing for the gradient if the
guarded expression is singular on the untaken branch. Four shapes of this
were found across the ECHAM physics by `jcm.testing.check_gradients` (#820);
each had a correct forward value and a NaN or infinite derivative at an
ordinary operating point:

1. **Masked infinity.** `jnp.clip(a / jnp.maximum(jnp.abs(x), 1e-30), 0, 1)`
   where `x == 0`: the quotient's partial is `a / 1e-60`, which is `inf` in
   float32, and the clip's zero mask multiplies it — `0 * inf = nan`. The same
   happens to `jnp.exp(big)` evaluated on the branch a `where` discards.
2. **Tie at zero.** `jnp.sqrt(jnp.maximum(x, 0.0))` at `x == 0`: both
   arguments of `maximum` tie, JAX splits the derivative half to each, and
   half of `sqrt'(0) = inf` is still `inf`. Zero TKE is an ordinary state
   (laminar layer, cold start, zero-initialised carry), so this fires.
3. **Cone tip.** `jnp.sqrt(u**2 + v**2)` at `u = v = 0` (a calm column): no
   derivative exists there and AD returns `nan`.
4. **Float32 range in the derivative, not the value.** A quotient's VJP is
   `-num / den**2`. For `(h*c) / (k*T)` the denominator squared,
   `(1.38e-23 * 250)**2 ~ 1e-41`, is below float32's smallest normal, so
   `dB/dT` overflows to `inf` at *every* temperature. A floor such as
   `jnp.maximum(den**2, 1e-30)` does not help: the VJP squares the floor too.

The house idiom is the **double `where`**: make the *argument* safe before the
singular operation, then mask the result, so the singular function is never
differentiated at the singular point in either AD mode:

```python
safe_x = jnp.where(x > 0.0, x, 1.0)                 # untaken branch is benign
y = jnp.where(x > 0.0, jnp.sqrt(safe_x), 0.0)       # sqrt' is only ever taken at safe_x
```

For (3) apply it to `r2 = u**2 + v**2` and mask the norm to zero; for (4) fold
the constants so that the differentiated denominator is the O(1)–O(1000)
physical variable itself (`HC_OVER_K / T`), or divide through by the largest
exponential so every intermediate stays in `[0, 1]`.

Check with `jcm.testing.check_gradients(f, args, rtol=...)`. It compares
`jvp` and `vjp` against a central difference whose step is *relative* to each
input leaf's magnitude, so a `stop_gradient`, an integer cast or a masked
infinity on a large-magnitude field cannot hide behind the small ones; with
`reference="adjoint"` and `live_inputs=(...)` it fences finiteness and
liveness where an output is piecewise constant (a level index) and no
difference quotient exists.

## Static arguments hold whole objects, and `Model` is one

`Model._run_from_state` is jitted with `self` static. A static argument is a
Python constant inside the trace, so the entire model (physics terms, their
`nnx.Param` values, the dycore) is baked into the executable when the physics is
first traced. `Model` hashes by identity, so an attribute mutated afterwards is
ignored wherever that executable is reused, and a later retrace may or may not
pick it up: the per-term `jax.checkpoint` wrapper caches its own jaxpr, so the
term may not be re-entered at all. An in-place edit is therefore neither
reliably applied nor reliably ignored:

```python
term.params.set_value(term.params.get_value().replace(trvdi=jnp.array(2.0)))
model.run(...)          # may run the OLD trvdi; no error either way
```

Build a new `Model` to change a parameter, and put the loop that does so inside
one `jax.jit` so the rebuild is traced once rather than compiled per iteration.
The corollary for anything on the `run` path: never require a concrete value
from inside the jitted computation (no `int()`/`float()` on a returned array).
That is fine at top level and raises `ConcretizationTypeError` the moment a
caller wraps the run in their own `jit`.
