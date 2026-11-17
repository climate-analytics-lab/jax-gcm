"""Small column-physics primitives shared by schemes with the vertical on axis 0."""

import jax.numpy as jnp


def take_level_axis0(
    arr: jnp.ndarray, idx: jnp.ndarray, *, keepdims: bool = False,
) -> jnp.ndarray:
    """Pick ``arr[idx]`` along axis 0, with a per-column level index ``idx``.

    ``arr`` is ``(nlev, *horiz)`` and ``idx`` is an integer array of shape
    ``(*horiz)`` (a scalar for a single ``(nlev,)`` column). The result is
    ``(*horiz)``, or ``(1, *horiz)`` with ``keepdims=True``.

    This is a masked sum over the vertical axis rather than ``arr[idx]``:
    under a column-vectorised host, ``arr[idx]`` lowers to a per-column
    dynamic gather, which is slow on accelerators that vectorise the column
    axis (on TPU the ``(ncols, 1, 1)`` gather tile is padded to the
    ``(8, 128)`` register shape). The masked sum keeps the vertical on axis 0
    and lets the columns broadcast, so the same code serves a ``(nlev,)``
    column, an ``(nlev, ncols)`` block and an ``(nlev, nlon, nlat)`` grid.
    The value is exactly ``arr[idx]``: exactly one lane per column is
    non-zero, so the sum adds no rounding.

    A negative ``idx`` counts from the end (as in Python indexing) and an
    out-of-range ``idx`` is clipped to the nearest level, so a sentinel index
    (for example "no cloud base found") reads a finite value instead of
    indexing out of bounds; callers mask such lanes themselves.
    """
    n = arr.shape[0]
    i = jnp.asarray(idx, dtype=jnp.int32)
    i = jnp.where(i < 0, i + n, i)
    i = jnp.clip(i, 0, n - 1)
    levels = jnp.arange(n, dtype=jnp.int32).reshape(
        (n,) + (1,) * (arr.ndim - 1)
    )
    mask = levels == jnp.expand_dims(i, axis=0)
    z = jnp.zeros((), dtype=arr.dtype)
    return jnp.sum(jnp.where(mask, arr, z), axis=0, keepdims=keepdims)


def pin_float_dtype(dtype, carry: tuple, ys: tuple) -> tuple:
    """Cast every floating leaf of a scan step's ``(carry, ys)`` to ``dtype``.

    The two branches of a ``lax.cond`` inside a ``lax.scan`` step (the
    "compute" branch and the "skip" branch) must return identical dtypes, and
    the carry must keep the dtype it entered with. Mixing a Python-float
    constant or a promoted product into one branch can widen a leaf (for
    example to float64 under ``jax_enable_x64``) and make the branches or the
    carry disagree; pinning every floating leaf to the working precision on
    both branches keeps them in step. Integer and boolean leaves (level
    indices, "alive" flags) pass through unchanged.

    Parameters
    ----------
    dtype : the working floating dtype of the column.
    carry, ys : tuples of scalars/arrays (the scan carry and per-level output).

    Returns
    -------
    ``(carry, ys)`` as tuples with floating leaves cast to ``dtype``.

    """
    def pin(x):
        return x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x

    return tuple(pin(x) for x in carry), tuple(pin(x) for x in ys)
