
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