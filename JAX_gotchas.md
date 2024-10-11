
# JAX Gotchas

## VMap

When using `vmap` in JAX, there are some gotchas to be aware of. One common issue is with scalar values that are looped over using `vmap`. In such cases, `vmap` treats the scalar as a vector, which can lead to unexpected behavior.

For example: in the original code we passed in one element of a vector called sig and checked its value. When vectorizing over the same function with `vmap` the comparison `sig <= 0.0` because `vmap` considers `sig` as a vector, even though conceptually is a scalar. To work around this, you can use alternative approaches like `jnp.less_equal` or `jnp.where` to handle the comparison correctly.

Here's an updated version of the code snippet:
`qsat = jnp.where(sig <= 0.0, 622.0 * qsat / (ps[0,0] - 0.378 * qsat), 622.0 * qsat / (sig * ps - 0.378 * qsat))`