"""ECHAM's total cloud cover ``aclcov``: maximum-random overlap of a profile.

``mo_cloud.f90`` section 10.2, "Total cloud cover". Vertically contiguous
cloud is treated as one maximally overlapped cloud and cloud separated by clear
air combines randomly, through the adjacent-layer product (top to bottom,
``jk = 2, klev``)::

    zclcov = 1 - paclc(1)
    DO jk = 2, klev
      zclcov = zclcov * (1 - MAX(paclc(jk), paclc(jk-1)))
                      / (1 - MIN(paclc(jk-1), zxsec))
    END DO
    aclcov = 1 - zclcov                      ! zxsec = 1 - zepsec, zepsec = 1e-12

There is **one** implementation of that recurrence, :func:`max_random_cover`,
and two callers that differ only in how they hand it the levels:

* :func:`column_max_random_cover` takes jnp arrays inside the jitted physics
  step and fills ``clouds.total_cloud_cover``
  (:class:`~jcm.physics.clouds.cloud_data.CloudData`), the instantaneous cover
  ECHAM accumulates into ``paclcov``.
* :func:`jcm.analysis.total_cloud_cover` takes saved output as lazy xarray
  slices. It must stay lazy on a dask-backed ``open_mfdataset`` window, which a
  jnp implementation could not (it would materialise the window), so it passes
  ``numpy`` and a per-level accessor instead of calling the jnp wrapper.

Sharing the recurrence, and not only the formula, is what stops the in-model
diagnostic and the offline scorer from drifting apart.

**The denominator.** ECHAM's ``1 - MIN(c, zxsec)`` is written here as the
identical ``MAX(1 - c, zepsec)``. The two are equal in real arithmetic, but
``zxsec = 1 - 1e-12`` rounds to exactly 1.0 in float32, where the Fortran form
turns an overcast layer's guarded ``0 / 1e-12`` into ``0 / 0``: a NaN in the
value, and a NaN gradient wherever a gradient path reaches the cover (a zero
cotangent into the Fortran form still gives ``0 * NaN``). ``zepsec`` itself is
representable in
float32, so this form is safe at either precision, and every local derivative
of the recurrence is finite (at most ``1 / zepsec``, in a layer within
``zepsec`` of overcast).

**Derivative at ties.** Every local derivative is finite, and the derivative
is the true one wherever adjacent layers differ. Where two adjacent layers hold
exactly equal fractions (an exactly clear column is the common case: Sundqvist's
cover is exactly zero at or below the critical humidity) the overlap has a kink,
and ``jnp.maximum`` splits the derivative between its arguments; at an
exactly-clear interior layer that cancels against the denominator and the layer
gets zero sensitivity where the one-sided (increase-only) derivative is +1. No
fixed tie rule is right in every direction; the choice is #1013.

**Orientation.** The result does not depend on which end of the column is the
surface: the adjacent-pair factors ``1 - max(c_k, c_{k-1})`` and the interior
levels' ``1 - c_k`` in the denominators are the same set reversed. The
physics-internal frame (top-first), the saved output (surface-first) and files
written before #710 (mixed) therefore all score identically, to rounding. The
``MAX(1 - c, zepsec)`` guard is applied in loop order, so it caps a different
denominator in the reversed column; that differs at ``O(zepsec)`` and only for
a layer within ``zepsec`` of overcast.
"""

from __future__ import annotations

import jax.numpy as jnp

#: ECHAM's ``zepsec`` security epsilon (``mo_cloud.f90``, "Security
#: parameters": ``zepsec = 1.0e-12``): the floor of the overlap denominator
#: ``1 - c``, so a cell with cover exactly 1 divides by 1e-12 rather than zero.
ZEPSEC = 1.0e-12


def max_random_cover(level, nlev, xp):
    """Total cover of a cloud-fraction column under maximum-random overlap.

    The single copy of the ``mo_cloud.f90`` section 10.2 recurrence (see the
    module docstring). It is independent of the array type: ``xp`` is the
    namespace whose ``maximum`` it uses (``numpy`` for saved output,
    ``jax.numpy`` inside the physics step), and the levels reach it through
    ``level``, so the caller decides how a level is sliced, clipped and cast
    and a lazy array stays lazy.

    Parameters
    ----------
    level : callable
        ``level(k)`` returns the cloud fraction of layer ``k`` (``k = 0 ...
        nlev-1``, in the order the column is to be swept), already clipped to
        ``[0, 1]``. Its shape is the shape of the result.
    nlev : int
        Number of layers; at least 1.
    xp : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    array
        Cover in ``[0, 1]``. Each factor's numerator ``1 - max(c_k, c_{k-1})``
        is at most its denominator ``max(1 - c_{k-1}, zepsec)``, so every
        factor, and hence the product, lies in ``[0, 1]`` and no clip is needed
        on the way out. ``NaN`` in any layer propagates to the result.

    """
    lower = level(0)
    clear = 1.0 - lower
    for k in range(1, nlev):
        upper = level(k)
        clear = clear * ((1.0 - xp.maximum(upper, lower))
                         / xp.maximum(1.0 - lower, ZEPSEC))
        lower = upper
    return 1.0 - clear


def column_max_random_cover(cloud_fraction):
    """``aclcov`` of every column of a ``(nlev, *horiz)`` cloud-fraction array.

    Broadcasting-native: the vertical is axis 0 and any trailing axes are
    horizontal, so the same code serves one ``(nlev,)`` column, a
    ``(nlev, ncols)`` block and a ``(nlev, nlat, nlon)`` grid. The level order
    does not matter (see the module docstring); the physics-internal frame is
    top-first, which is the order ECHAM sweeps.

    The fraction is clipped to ``[0, 1]`` first, as the saved-output scorer
    does, so a stray excursion cannot make a "clear-sky fraction" negative.
    The result keeps the dtype of the input.
    """
    cloud_fraction = jnp.asarray(cloud_fraction)
    return max_random_cover(
        lambda k: jnp.clip(cloud_fraction[k], 0.0, 1.0),
        cloud_fraction.shape[0], jnp)
