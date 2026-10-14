"""Tests for ``safe_fraction`` (#1070)."""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np


class SafeFractionTest(unittest.TestCase):
    """``safe_fraction(a, b) = a / (a + b)``, swept over the float32 regime.

    Six successive Codex-review rounds on this one helper (#1070, PR
    #1099) each found a different scale — or a different EXECUTION MODE —
    at which SOME formulation of this ratio's derivative misbehaved: an
    absolute floor under the denominator (wrong value), that floor one
    order further out, a 1e-6-relative "negligible" cutoff that swallowed a
    real small value, a squared denominator underflowing float32 before a
    meaningful ratio's gradient did, ``(1 − f)`` catastrophically
    cancelling once ``a ≫ b`` rounds ``f`` to exactly 1.0, a floor meant to
    guard that squared denominator silently corrupting derivatives that
    float32 could represent exactly, and finally XLA's JIT simplifier
    re-deriving that same squared denominator from the sequential-division
    form that round five's fix relied on to avoid it — a bug invisible in
    eager mode (round five's regression test ran eager-only) and present
    only once compiled, which is how this model always actually runs.
    This sweeps the whole ``{a, b}`` regime at once, under BOTH eager and
    ``jax.jit`` execution — every combination of six magnitudes spanning
    34 orders, twice — rather than adding one more point-case test per
    round.

    The helper itself moved from ``cloud_borne.py`` into its own module
    (``safe_fraction.py``) once ``wetdep_term.phase_split`` needed the same
    exact ratio for its own per-phase condensate split (round 7+) — see
    that module's docstring for why (a circular import between the two
    original call sites), and this test file moved with it.
    """

    #: float32-relevant magnitudes: exact 0, deep subnormal-for-the-SQUARE
    #: range (1e-30 -- (1e-30+1e-30)**2 = 4e-60, far below float32's
    #: smallest subnormal, ~1.4e-45, though ``denom`` itself, 2e-30, is not
    #: even subnormal — exactly the gap between "safe forward value" and
    #: "unsafe squared intermediate" the helper's sequential-division JVP
    #: exists to close), the smallest total either call site's physics ever
    #: produces (1e-12, cloud_borne.py's ``_PROCESS_FLOOR``), an ordinary
    #: small mixing ratio (1e-4), O(1), and a magnitude large enough to
    #: make the OTHER operand's relative contribution negligible (1e4).
    _GRID = (0.0, 1.0e-30, 1.0e-12, 1.0e-4, 1.0, 1.0e4)

    @staticmethod
    def _reference(a: float, b: float):
        """Return the exact ``(f, df/da, df/db)`` in Python (C) float64.

        Float64's range (subnormals to ~5e-324) never underflows for any
        ``(a, b)`` in ``_GRID`` — even the pathological ``a = b = 1e-30``
        squares to ``4e-60``, still far inside it — so this is an exact
        reference wherever ``a + b > 0``, independent of the float32
        implementation under test.
        """
        if a + b <= 0.0:
            return 0.0, 0.0, 0.0
        denom = a + b
        f = a / denom
        return f, b / denom ** 2, -a / denom ** 2

    def test_matches_the_exact_reference_or_is_finite_throughout(self):
        from jcm.physics.aerosol.jam.safe_fraction import safe_fraction

        # The specific inputs five rounds of Codex review raised directly,
        # alongside the systematic grid sweep.
        codex_cases = (
            (7.5e-13, 0.0),    # round 3: sub-floor total, pool exactly 0
            (7.5e-13, 7.5e-13),  # round 3 variant: both phases sub-floor
            (1.0e-20, 1.0e-20),  # round 4/5/6: live cell, trace-scale
                                 # ratio -- also round 6's JIT-fusion case
            (1.0e15, 1.0),       # round 4 (as pool/evap): a >> b, the
                                 # (1 - f)-cancellation case
        )
        points = list(itertools.product(self._GRID, self._GRID)) + list(
            codex_cases)

        # The helper's sequential-division JVP (b / denom) / denom never
        # forms a squared intermediate, so it has no floor of its own to
        # match: the true derivative is exactly representable in float32
        # at every grid point here, and genuinely overflows only where the
        # exact float64 reference itself does (comfortably outside this
        # grid -- see the helper's docstring). The comparison is therefore
        # unconditional wherever a + b > 0.
        #
        # Run under BOTH eager execution and jax.jit: round six's finding
        # was invisible in eager mode (XLA's algebraic simplifier -- which
        # only runs for a compiled computation -- is what re-forms the
        # squared denominator from the sequential divisions), so a
        # jit-only variant of this same sweep is the regression test for
        # that round, not a separate one-off case.
        value_and_grad = jax.value_and_grad(safe_fraction, argnums=(0, 1))
        jitted_value_and_grad = jax.jit(value_and_grad)
        for mode, fn in (("eager", value_and_grad),
                          ("jit", jitted_value_and_grad)):
            for a, b in points:
                f32, (df_da, df_db) = fn(
                    jnp.asarray(a, dtype=jnp.float32),
                    jnp.asarray(b, dtype=jnp.float32))
                f32, df_da, df_db = float(f32), float(df_da), float(df_db)
                case = f"a={a:.3e} b={b:.3e} mode={mode}"
                self.assertTrue(
                    np.isfinite(f32), f"value not finite: {case}")
                self.assertTrue(
                    np.isfinite(df_da), f"df/da not finite: {case}")
                self.assertTrue(
                    np.isfinite(df_db), f"df/db not finite: {case}")

                f_ref, da_ref, db_ref = self._reference(a, b)
                np.testing.assert_allclose(
                    f32, f_ref, atol=1e-6,
                    err_msg=f"value disagrees with the exact "
                            f"reference: {case}")
                for ref, got, name in (
                    (da_ref, df_da, "df/da"), (db_ref, df_db, "df/db"),
                ):
                    np.testing.assert_allclose(
                        got, ref, rtol=1e-3, atol=1e-30,
                        err_msg=f"{name} disagrees with the exact "
                                f"reference: {case}")
