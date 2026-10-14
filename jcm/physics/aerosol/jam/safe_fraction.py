"""``safe_fraction`` — ``a / (a + b)`` for nonnegative ``a``, ``b``, exact and
gradient-safe at every float32 scale, including under ``jax.jit``.

Shared between ``cloud_borne.py`` (the per-phase evaporation fraction,
``f_evap_liq``/``f_evap_ice``) and ``wetdep.wetdep_term`` (the per-phase
condensate-mass split feeding ``phase_split``'s ``p_ice``, #1070 round 7+):
both need this exact ratio, and ``cloud_borne.py`` already imports FROM
``wetdep_term`` (for ``phase_split`` itself), so the helper lives in its own
module rather than either of theirs to avoid a circular import.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.custom_jvp
def safe_fraction(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute ``a / (a + b)`` for nonnegative ``a``, ``b``, at any scale.

    The forward division is already safe — ``a``, ``b`` are both nonnegative,
    so ``a + b`` never cancels — but JAX's default VJP for a quotient goes
    through ``1/(a+b)**2`` computed from an UNMASKED ``a+b``, which both
    underflows float32 once ``a + b`` drops below ~1e-19 (#1070 Codex
    review, found repeatedly across both this helper's call sites) and
    divides by a literal zero where ``a = b = 0``. The custom JVP below masks
    ``a + b`` the same way the forward pass does and computes the two
    partials from that ONE masked denominator — ``∂f/∂a = b/denom²`` and
    ``∂f/∂b = −a/denom²`` — the ordinary quotient-rule partials, not
    rewritten through ``f`` or ``1−f``: an earlier revision of this helper
    computed ``∂f/∂a`` as ``(1−f)/denom`` (algebraically identical, since
    ``1−f = b/denom``), but ``1−f`` catastrophically cancels in float32 once
    ``a ≫ b`` makes ``f`` round to exactly ``1.0``, silently zeroing a
    nonzero gradient (a FOURTH Codex-review finding on this helper).
    Computing ``b/denom²`` and ``−a/denom²`` directly from ``b``/``a`` keeps
    the numerator exact at every scale; the only thing masked is the shared,
    nonneg denominator, which the forward pass already confirms is either
    exactly 0 (the one masked case, tangent 0) or safely positive.

    Even so, ``denom**2`` must never be formed as its own intermediate
    value: ``denom`` can be safely positive and ordinary (e.g.
    ``a = b = 1e-30``, ``denom = 2e-30`` — not even subnormal) while its
    SQUARE underflows float32 to literal 0 (``4e-60`` is far below the
    smallest representable subnormal, ~1.4e-45), after which ``b/denom**2``
    divides by that literal 0 instead of by a merely-small number. A first
    revision of this JVP floored ``denom**2`` at float32's smallest normal
    value to avoid the resulting ``inf``/``nan`` — found wrong by Codex
    review: the floor silently corrupted the still-perfectly-representable
    true derivatives it was never needed for (``2.5e19`` read back as
    ``8.5e17`` at ``a = b = 1e-20``; CLAUDE.md's "no substitute caps"
    prohibition). ``(b / denom) / denom`` is algebraically identical to
    ``b / denom**2`` but never forms the squared intermediate at all: each
    division is by ``denom`` itself, which the ``positive`` mask already
    keeps a safely nonzero, ordinary value, so nothing here is capped —
    the result is the exact derivative at every grid point either call
    site's physics can reach, and genuinely overflows (rather than being
    silently wrong) only beyond it, where the TRUE derivative itself
    exceeds float32's range (e.g. ``a = b = 1e-40``, both already
    subnormal: the true ``2.5e39`` is itself unrepresentable in float32).

    A SIXTH Codex-review finding: under ``jax.jit`` (how this model always
    actually runs), XLA's algebraic simplifier recognises ``(b / denom) /
    denom`` as ``b * denom⁻¹ * denom⁻¹`` and refuses to preserve the
    sequential evaluation — it rewrites it back into ``b / denom**2``,
    re-forming the squared intermediate the sequential form exists to avoid
    and reproducing the exact ``inf`` this docstring just finished
    explaining (verified: ``a = b = 1e-20`` gives the correct ``2.5e19`` in
    eager mode but ``inf`` once compiled). The custom JVP below is unaffected
    by autodiff correctness — the math is still exact — but the *compiler*
    defeats the chosen evaluation order, so the fix has to stop the
    simplifier from seeing across it: ``jax.lax.optimization_barrier`` is a
    value-identity op XLA may not fuse, reorder, or algebraically simplify
    through, so wrapping the two first-stage quotients (``b/denom``,
    ``a/denom``) in one forces the second division to run against the
    already-computed quotient rather than against a re-derived ``denom**2``,
    in both eager and compiled execution.
    """
    denom = jnp.where(a + b > 0.0, a + b, 1.0)
    return jnp.where(a + b > 0.0, a / denom, 0.0)


@safe_fraction.defjvp
def _safe_fraction_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    positive = a + b > 0.0
    denom = jnp.where(positive, a + b, 1.0)
    f = jnp.where(positive, a / denom, 0.0)
    # The barrier blocks XLA from re-associating these into b/denom**2 and
    # a/denom**2 under jit, which would re-underflow exactly as the
    # sequential form above was written to avoid (see docstring).
    quotient_a, quotient_b = jax.lax.optimization_barrier((b / denom, a / denom))
    df_da = jnp.where(positive, quotient_a / denom, 0.0)
    df_db = jnp.where(positive, -(quotient_b / denom), 0.0)
    tangent_out = df_da * da + df_db * db
    return f, tangent_out
