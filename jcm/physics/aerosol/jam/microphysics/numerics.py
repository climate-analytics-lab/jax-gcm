"""Float32-safe primitives for the modal core's derivatives (AER-11).

JAX differentiates ``x / y`` with the rule ``d(x/y) = dx/y − x·dy/y²`` and,
in reverse mode, forms ``cot · y⁻²`` *before* multiplying by ``x``. For a
coagulation coefficient that is a quotient of two ~1e-15 m³ s⁻¹ limits, and
a cotangent of ~1e19 (the number equation's ``dN/dβ = −Δt N²``), that
intermediate is 1e47 — past float32's 3.4e38 — and the finite derivative
comes back as ``inf``, then ``NaN``. Forward mode is fine; the finite
difference is fine; only the reverse evaluation order is not. Measured
2026-10-01 through ``coagulate`` on a populated column (``d n_acc / d
m_bc_acc``), on the sweep of AER-11.

:func:`divide` is the same quotient with the derivative written as
``(dx − z·dy)/y`` where ``z = x/y`` is the already-computed value: reverse
mode then forms ``cot/y`` (1e34, representable) and ``−z·cot/y`` (1e18),
never ``y⁻²``. It is exact in value and in derivative; only the order of
operations differs. Use it wherever a quotient of small quantities can
receive a large cotangent — the kernels and the mass-budget ratios of this
package — and keep the plain ``/`` for O(1) denominators, where the extra
``custom_jvp`` buys nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.custom_jvp
def divide(x, y):
    """``x / y`` whose reverse pass never forms ``y⁻²``."""
    return x / y


@divide.defjvp
def _divide_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    z = x / y
    return z, (dx - z * dy) / y


def harmonic_mean(a, b):
    """Combine two limits as ``a b / (a + b)``, zero where both vanish.

    Written as ``a · (b / (a + b))`` through :func:`divide`: the inner
    quotient is O(1), so its reverse pass stays in range however small the
    two limits are, and the outer product's cotangent on ``a`` is
    ``cot · (b/(a+b))``, bounded by ``cot``.
    """
    total = a + b
    live = total > 0.0
    safe = jnp.where(live, total, 1.0)
    return jnp.where(live, a * divide(b, safe), 0.0)
