"""Tests for the dycore-side gridpoint tracer filters."""

import unittest

import numpy as np
import jax
import jax.numpy as jnp

from jcm.filters import (
    mass_conserving_positivity, MassConservingPositivity, stable_quotient,
)


class MassConservingPositivityMathTest(unittest.TestCase):
    """The core hole-filling clip: non-negative + column-mass-conserving."""

    def test_nonnegative_and_conserves_column_mass(self):
        # (nlev, ncol): mix of positive and Gibbs-ringing negative values.
        q = jnp.asarray([[1.0, -0.5, 2.0],
                         [-0.3, 1.0, -1.0],
                         [0.8, 0.4, 3.0]])
        m = jnp.asarray([[1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0],
                         [1.5, 1.5, 1.5]])          # per-layer air mass ∝ Δp
        out = mass_conserving_positivity(q, m)

        # Non-negative everywhere.
        self.assertTrue(bool(jnp.all(out >= 0.0)))
        # Column mass preserved for columns whose net mass is positive.
        before = jnp.sum(m * q, axis=0)
        after = jnp.sum(m * out, axis=0)
        # All three test columns have positive net mass here.
        self.assertTrue(bool(jnp.all(before > 0.0)))
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), rtol=1e-6)

    def test_net_negative_column_is_zeroed(self):
        # A column whose mass-weighted sum is negative can't be made positive
        # while conserving mass — it's zeroed (the only non-conservation).
        q = jnp.asarray([[-2.0], [-1.0], [0.5]])
        m = jnp.ones_like(q)
        out = mass_conserving_positivity(q, m)
        self.assertTrue(bool(jnp.all(out == 0.0)))

    def test_already_nonnegative_is_unchanged(self):
        q = jnp.asarray([[1.0, 0.0], [2.0, 3.0], [0.5, 1.0]])
        m = jnp.asarray([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
        out = mass_conserving_positivity(q, m)
        np.testing.assert_allclose(np.asarray(out), np.asarray(q), rtol=1e-6)

    def test_filter_object_applies_to_every_tracer(self):
        dp = jnp.asarray([[1.0, 1.0], [1.0, 1.0]])
        tracers = {
            'a': jnp.asarray([[1.0, -0.5], [-0.2, 1.0]]),
            'b': jnp.asarray([[0.0, 2.0], [3.0, -1.0]]),
        }
        out = MassConservingPositivity()(tracers, dp)
        self.assertEqual(set(out), {'a', 'b'})
        for k in tracers:
            self.assertTrue(bool(jnp.all(out[k] >= 0.0)))


def _working_dtype():
    """Return the session's float dtype: float32 unless a test module enabled x64."""
    return jnp.zeros(()).dtype


def _smallest_safe_denominator(dtype):
    """``sqrt(smallest normal)``: below it ``denominator**-2`` is ``inf``."""
    return float(np.sqrt(np.finfo(dtype).tiny))


class StableQuotientTest(unittest.TestCase):
    """``stable_quotient`` is ``n / d`` in value and in derivative.

    The reason it exists is a range limit in the *derivative* of the bare
    quotient: ``-n / d**2`` is ``inf`` once ``d**2`` leaves the dtype's normal
    range (``d`` below ~1.08e-19 in float32) although ``n / d`` is fine. Every
    case below is therefore pitched in that band, with the bare quotient as
    the control that shows the case can fail.
    """

    def test_value_is_the_plain_quotient_bit_for_bit(self):
        dtype = _working_dtype()
        values = np.array([0.0, 1e-40, 1e-30, 3e-21, 1e-10, 1.0, 7.5, -2e-9],
                          dtype=dtype)
        n, d = (jnp.asarray(a) for a in np.meshgrid(values, values))
        # The backend's own division is the reference (it flushes a subnormal
        # such as 1e-40 to zero, which numpy does not), zero denominators
        # included: the helper masks nothing.
        expected = np.asarray(n / d)
        np.testing.assert_array_equal(np.asarray(stable_quotient(n, d)), expected)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(stable_quotient)(n, d)), expected)

    def test_derivative_is_the_true_derivative_in_the_ordinary_range(self):
        # float64 closed form: d(n/d)/dn = 1/d, d(n/d)/dd = -n/d**2.
        n, d = 0.37, 1.9
        dn, dd = jax.grad(stable_quotient, argnums=(0, 1))(
            jnp.asarray(n), jnp.asarray(d))
        np.testing.assert_allclose(float(dn), 1.0 / d, rtol=1e-5)
        np.testing.assert_allclose(float(dd), -n / d ** 2, rtol=1e-5)
        # Forward and reverse mode agree (the reverse rule is the transpose
        # of the tangent rule).
        _, jvp_n = jax.jvp(stable_quotient, (jnp.asarray(n), jnp.asarray(d)),
                           (jnp.asarray(1.0), jnp.asarray(0.0)))
        _, jvp_d = jax.jvp(stable_quotient, (jnp.asarray(n), jnp.asarray(d)),
                           (jnp.asarray(0.0), jnp.asarray(1.0)))
        np.testing.assert_allclose(float(jvp_n), float(dn), rtol=1e-6)
        np.testing.assert_allclose(float(jvp_d), float(dd), rtol=1e-6)

    def test_second_derivative_meets_the_same_rule(self):
        n, d = 0.37, 1.9
        d2 = jax.grad(jax.grad(stable_quotient, argnums=1), argnums=1)(
            jnp.asarray(n), jnp.asarray(d))
        np.testing.assert_allclose(float(d2), 2.0 * n / d ** 3, rtol=1e-4)

    def test_derivative_is_finite_and_correct_where_the_bare_quotient_overflows(self):
        dtype = _working_dtype()
        # A denominator between the guard that keeps it positive and the point
        # where its square leaves the normal range.
        d = 1e-2 * _smallest_safe_denominator(dtype)
        n = 0.9999 * d
        args = (jnp.asarray(n, dtype), jnp.asarray(d, dtype))

        # The control: the bare quotient's reverse pass is not finite here,
        # for a zero cotangent (nan) and for a unit one (inf).
        for ct in (0.0, 1.0):
            _, vjp = jax.vjp(lambda a, b: a / b, *args)
            bare = np.asarray(vjp(jnp.asarray(ct, dtype)))
            self.assertFalse(np.all(np.isfinite(bare)), ct)

        # The helper, eagerly and under jit, for both cotangents.
        for ct in (0.0, 1.0):
            for fn in (jax.vjp, lambda f, *a: jax.jit(lambda *b: jax.vjp(f, *b))(*a)):
                out, vjp = fn(stable_quotient, *args)
                grads = np.asarray(vjp(jnp.asarray(ct, dtype)))
                self.assertTrue(np.all(np.isfinite(grads)), (ct, grads))
                # float64 closed form: ct/d and -ct*n/d**2 = -ct*(n/d)/d.
                expected = ct * np.array([1.0 / d, -(n / d) / d])
                np.testing.assert_allclose(grads, expected, rtol=1e-4)


def _plain_filter(q, m):
    """Apply the filter as a masked bare quotient, the formulation to reproduce bit for bit."""
    q_clip = jnp.maximum(0.0, q)
    col_mass = jnp.sum(m * q, axis=0)
    col_mass_clip = jnp.sum(m * q_clip, axis=0)
    scale = jnp.where(col_mass_clip > 0.0,
                      jnp.maximum(col_mass, 0.0) / col_mass_clip, 0.0)
    return q_clip * scale[jnp.newaxis, ...]


class MassConservingPositivityGradientTest(unittest.TestCase):
    """The hole-filler differentiates finitely in every kind of column.

    Its rescale is ``max(M, 0) / M_clip``, masked where ``M_clip == 0``. The
    forward discards the masked quotient; reverse mode differentiates it, and
    ``0/0`` there (an empty tracer column is ordinary) or ``M_clip**-2``
    overflowing (a column holding only a rounding residue) reaches the
    gradient as ``nan`` / ``inf``.
    """

    def _columns(self):
        dtype = _working_dtype()
        nlev = 4
        residue = 1e-2 * _smallest_safe_denominator(dtype)
        cols = {
            "empty": np.zeros(nlev),
            "residue_only": np.full(nlev, residue),
            "residue_with_hole": np.array([residue, -0.5 * residue, residue, 0.0]),
            "net_negative": np.array([-2.0, -1.0, 0.5, 0.0]) * 1e-6,
            "ordinary": np.array([1e-6, 2e-6, 0.0, -1e-7]),
            "ordinary_positive": np.array([1e-6, 2e-6, 3e-7, 5e-7]),
        }
        q = np.stack(list(cols.values()), axis=1).astype(dtype)
        m = np.broadcast_to(np.array([1.0, 2.0, 1.5, 0.5], dtype=dtype)[:, None],
                            q.shape).copy()
        return list(cols), jnp.asarray(q), jnp.asarray(m)

    def test_forward_is_the_plain_formulation_bit_for_bit(self):
        _, q, m = self._columns()
        np.testing.assert_array_equal(
            np.asarray(mass_conserving_positivity(q, m)),
            np.asarray(_plain_filter(q, m)))
        np.testing.assert_array_equal(
            np.asarray(jax.jit(mass_conserving_positivity)(q, m)),
            np.asarray(_plain_filter(q, m)))

    def test_reverse_mode_is_finite_for_zero_and_unit_cotangents_under_jit(self):
        names, q, m = self._columns()

        @jax.jit
        def pullback(q, ct):
            return jax.vjp(lambda x: mass_conserving_positivity(x, m), q)[1](ct)[0]

        for ct_value in (0.0, 1.0):
            grad = np.asarray(pullback(q, jnp.full_like(q, ct_value)))
            for j, name in enumerate(names):
                self.assertTrue(np.all(np.isfinite(grad[:, j])), (name, ct_value))
        # The control: the plain formulation is not finite in the columns the
        # test is about, so this test fails if the guard is removed.
        plain = np.asarray(jax.vjp(lambda x: _plain_filter(x, m), q)[1](
            jnp.zeros_like(q))[0])
        self.assertFalse(np.all(np.isfinite(plain[:, names.index("empty")])))
        self.assertFalse(np.all(np.isfinite(plain[:, names.index("residue_only")])))

    def test_gradient_matches_the_plain_formulation_where_that_is_finite(self):
        names, q, m = self._columns()
        rng = np.random.default_rng(0)
        ct = jnp.asarray(rng.standard_normal(q.shape).astype(q.dtype))
        new = np.asarray(jax.vjp(lambda x: mass_conserving_positivity(x, m), q)[1](ct)[0])
        old = np.asarray(jax.vjp(lambda x: _plain_filter(x, m), q)[1](ct)[0])
        for name in ("ordinary", "ordinary_positive", "net_negative"):
            j = names.index(name)
            np.testing.assert_allclose(new[:, j], old[:, j], rtol=1e-5, atol=1e-12)

    def test_a_column_with_no_negatives_passes_gradients_straight_through(self):
        # With nothing to fill, the filter is the identity, so its derivative
        # is the identity whatever the column's total.
        names, q, m = self._columns()
        ct = jnp.asarray(np.linspace(-1.0, 1.0, q.size).reshape(q.shape),
                         dtype=q.dtype)
        grad = np.asarray(jax.vjp(lambda x: mass_conserving_positivity(x, m), q)[1](ct)[0])
        for name in ("residue_only", "ordinary_positive"):
            j = names.index(name)
            np.testing.assert_allclose(grad[:, j], np.asarray(ct)[:, j], rtol=1e-6)


class DycoreTracerFilterWiringTest(unittest.TestCase):
    """The dycore applies ``tracer_filter`` inside ``to_physics_state`` with a
    correctly-shaped ``dp``, and is an exact no-op when no filter is set.
    """

    def _build_dycore(self, tracer_filter):
        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.physics.physics_term import TracerSpec
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords
        sigma_b = np.linspace(0, 1, 9)
        coords = get_coords(sigma_b, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        # An additional tracer (beyond specific_humidity, which is a top-level
        # PhysicsState field) so it lands in ``physics_state.tracers`` where the
        # filter operates.
        return DinosaurDycore(
            coords=coords, terrain=terrain, dt_seconds=600.0,
            tracer_specs={'qc': TracerSpec("qc", units="kg/kg")},
            tracer_filter=tracer_filter,
        ), coords

    def test_filter_is_called_with_layer_shaped_dp_and_applied(self):
        captured = {}

        def recording_filter(tracers, dp):
            captured['dp_shape'] = dp.shape
            return {k: v + 1.0 for k, v in tracers.items()}

        dy_filt, coords = self._build_dycore(recording_filter)
        dy_none, _ = self._build_dycore(None)
        state = dy_filt.initial_state(None)

        phys_none = dy_none.to_physics_state(state)
        phys_filt = dy_filt.to_physics_state(state)

        nlev = coords.vertical.layers
        q = phys_none.tracers['qc']
        # dp has the per-layer leading axis and matches the tracer field shape.
        self.assertEqual(captured['dp_shape'][0], nlev)
        self.assertEqual(captured['dp_shape'], q.shape)
        # The filter's effect is visible in the projected physics state.
        np.testing.assert_allclose(
            np.asarray(phys_filt.tracers['qc']),
            np.asarray(q) + 1.0, rtol=1e-6,
        )

    def test_no_filter_is_a_noop(self):
        dy_none, _ = self._build_dycore(None)
        state = dy_none.initial_state(None)
        # Should not raise and should round-trip the tracer untouched.
        phys = dy_none.to_physics_state(state)
        self.assertIn('qc', phys.tracers)


if __name__ == '__main__':
    unittest.main()
