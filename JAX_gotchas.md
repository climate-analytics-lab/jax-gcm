
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
