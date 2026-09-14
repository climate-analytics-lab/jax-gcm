"""Tests for the structure-independent gradient checker."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import tree_math

from jcm.testing import (DEFAULT_STEPS, _check_unique, _leaf_names,
                         check_gradients, random_direction)


@tree_math.struct
class _Inner:
    a: jnp.ndarray
    b: jnp.ndarray


@tree_math.struct
class _InnerWithExtra:
    """``_Inner`` with a field inserted at the top — the worst case for keying
    on position, since every later field shifts.
    """

    spare: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray


@tree_math.struct
class _Outer:
    inner: object
    tail: jnp.ndarray


class TestLeafNames(unittest.TestCase):

    def test_tree_math_struct_fields_are_named_not_numbered(self):
        """tree_math registers without key support, so JAX's own path is
        positional (``[<flat index 0>]``) and the whole invariant would rest on
        field order. The names must come back declared.
        """
        names = _leaf_names(_Outer(inner=_Inner(a=jnp.zeros(2), b=jnp.zeros(2)),
                                   tail=jnp.zeros(3)))
        self.assertEqual(names, ["inner/a", "inner/b", "tail"])

    def test_dict_and_sequence_keys_are_kept(self):
        names = _leaf_names({"x": [jnp.zeros(1), jnp.zeros(1)]})
        self.assertEqual(len(names), 2)
        self.assertNotEqual(names[0], names[1])

    def test_a_custom_node_with_more_children_than_fields_stays_unique(self):
        # Falls back to a positional name for the undeclared child rather than
        # repeating the declared one.
        names = _leaf_names(_ExtraChild(x=jnp.zeros(2)))
        self.assertEqual(len(set(names)), 2)

    def test_colliding_names_are_rejected(self):
        with self.assertRaises(ValueError):
            _check_unique(["inner/a", "inner/a", "tail"])

    def test_unique_names_are_accepted(self):
        _check_unique(["inner/a", "inner/b", "tail"])


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _ExtraChild:
    """One declared field but two children, so both resolve to the same name."""

    x: jnp.ndarray

    def tree_flatten(self):
        return (self.x, self.x), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(children[0])


class TestRandomDirection(unittest.TestCase):

    def test_direction_survives_a_field_inserted_above_it(self):
        """The property the whole module exists for: #753 added one field to
        SWRadiationData and re-rolled the perturbation of every test that
        passes a PhysicsData, breaking four checks in unrelated schemes.
        """
        before = random_direction(
            _Outer(inner=_Inner(a=jnp.zeros(2), b=jnp.zeros(2)),
                   tail=jnp.zeros(3)))
        after = random_direction(
            _Outer(inner=_InnerWithExtra(spare=jnp.zeros(5), a=jnp.zeros(2),
                                         b=jnp.zeros(2)),
                   tail=jnp.zeros(3)))
        np.testing.assert_array_equal(before.inner.a, after.inner.a)
        np.testing.assert_array_equal(before.inner.b, after.inner.b)
        np.testing.assert_array_equal(before.tail, after.tail)

    def test_different_leaves_get_different_directions(self):
        d = random_direction({"a": jnp.zeros(8), "b": jnp.zeros(8)})
        self.assertFalse(np.array_equal(d["a"], d["b"]))

    def test_seed_changes_the_direction(self):
        a = random_direction({"a": jnp.zeros(8)}, seed=0)
        b = random_direction({"a": jnp.zeros(8)}, seed=1)
        self.assertFalse(np.array_equal(a["a"], b["a"]))

    def test_non_float_leaves_get_a_float0_tangent(self):
        # jax.jvp rejects any other tangent dtype for an integer primal.
        d = random_direction({"i": jnp.arange(5), "f": jnp.zeros(5)})
        self.assertEqual(d["i"].dtype, jax.dtypes.float0)
        self.assertTrue(np.any(np.asarray(d["f"]) != 0))

    def test_integer_arguments_are_accepted_end_to_end(self):
        f = lambda x, i: jnp.sum(x**2) + jnp.sum(i).astype(x.dtype)
        check_gradients(f, (jnp.linspace(0.5, 2.0, 4), jnp.arange(4)), rtol=1e-3)


def _smooth(x, y):
    return jnp.sum(jnp.sin(x) * y**2), jnp.cos(x) @ y


class TestCheckGradients(unittest.TestCase):

    def setUp(self):
        self.args = (jnp.linspace(0.1, 1.0, 6), jnp.linspace(-1.0, 2.0, 6))

    def test_smooth_function_passes(self):
        check_gradients(_smooth, self.args, rtol=1e-3)

    def test_a_second_direction_also_passes(self):
        check_gradients(_smooth, self.args, rtol=1e-3, seed=11)

    def test_a_wrong_gradient_is_caught(self):
        @jax.custom_jvp
        def f(x, y):
            return jnp.sum(jnp.sin(x) * y**2)

        @f.defjvp
        def f_jvp(primals, tangents):
            x, y = primals
            dx, _ = tangents
            return f(x, y), 3.0 * jnp.sum(jnp.cos(x) * dx)

        with self.assertRaises(AssertionError):
            check_gradients(f, self.args, rtol=1e-3)

    def test_a_gradient_lost_on_a_small_output_is_caught(self):
        """Contracting onto one unscaled cotangent lets an O(1e4) output mask a
        small one; the per-leaf primal scaling is what keeps this visible.
        """

        @jax.custom_jvp
        def f(x):
            return 1e4 * jnp.sum(x**2), jnp.sum(jnp.sin(x))

        @f.defjvp
        def f_jvp(primals, tangents):
            (x,), (dx,) = primals, tangents
            # Correct on the large output, zeroed on the small one.
            return f(x), (1e4 * 2.0 * jnp.sum(x * dx), jnp.zeros(()))

        with self.assertRaises(AssertionError):
            check_gradients(f, (self.args[0],), rtol=1e-3)

    def test_a_straddled_jump_is_reported_not_compared(self):
        """A step function has no valid difference at any step, and saying so
        is more useful than comparing the gradient against jump/eps.
        """
        step = lambda x: jnp.sum(jnp.where(x > 0.5, 1.0, 0.0) + 1e-9 * x)
        with self.assertRaises(AssertionError) as caught:
            check_gradients(step, (jnp.full((64,), 0.5),), rtol=1e-1)
        self.assertIn("usable", str(caught.exception))

    def test_a_kink_is_reported_not_averaged_across(self):
        """The case neighbour-agreement alone cannot see: a central difference
        across a kink converges stably, at every rung, to the mean of the two
        one-sided derivatives — which is not what AD computes.
        """
        # |x| at x ~ 0: every step on the ladder straddles the kink, the
        # central secant is a stable 0 and the one-sided ones are -1 and +1.
        with self.assertRaises(AssertionError) as caught:
            check_gradients(lambda x: jnp.sum(jnp.abs(x)),
                            (jnp.full((32,), 1e-8),), rtol=1e-1)
        self.assertIn("kink", str(caught.exception))

    def test_a_legitimately_zero_gradient_is_accepted_on_atol(self):
        """Smooth, with an exactly zero directional derivative: the relative
        spread is meaningless there, so the ladder must not be exhausted and
        the failure blamed on a discontinuity.
        """
        f = lambda x: jnp.sum(x)**2
        check_gradients(f, (jnp.array([1.0, -1.0, 2.0, -2.0]),),
                        rtol=1e-3, atol=1e-5)

    def test_adjoint_reference_needs_no_smoothness(self):
        f = lambda x: jnp.sum(jnp.floor(x) + 1e-3 * x)
        check_gradients(f, (jnp.linspace(0.0, 4.0, 32),), reference="adjoint")

    def test_adjoint_reference_rejects_a_dead_output_beside_a_live_one(self):
        """diagnose_convection returns a level index *and* a moisture excess;
        the live one must not satisfy the guard on the dead one's behalf.
        """
        f = lambda x: (jnp.floor(x).sum(), jnp.sum(x**2))
        with self.assertRaises(AssertionError):
            check_gradients(f, (jnp.linspace(0.0, 4.0, 32),),
                            reference="adjoint")

    def test_difference_reference_requires_rtol(self):
        with self.assertRaises(ValueError):
            check_gradients(_smooth, self.args)

    def test_unknown_reference_is_rejected_before_any_work(self):
        with self.assertRaises(ValueError):
            check_gradients(_smooth, self.args, rtol=1e-3, reference="spline")

    def test_a_single_step_is_rejected_with_a_clear_message(self):
        with self.assertRaises(ValueError):
            check_gradients(_smooth, self.args, rtol=1e-3, steps=(1e-4,))

    def test_static_dataclass_fields_do_not_misname_children(self):
        """A node that flattens to fewer children than it declares must not
        pair them off by position: a real leaf named after a static field would
        re-acquire the dependence on unrelated fields this module removes.
        """
        names = _leaf_names(_WithStatic(a=jnp.zeros(2), b=jnp.zeros(2)))
        self.assertNotIn("mode", names)

    def test_steps_form_a_halving_ladder(self):
        ratios = [b / a for a, b in zip(DEFAULT_STEPS, DEFAULT_STEPS[1:])]
        np.testing.assert_allclose(ratios, 0.5)


if __name__ == '__main__':
    unittest.main()


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _WithStatic:
    """Two children but three declared fields, the static one in the middle."""

    a: jnp.ndarray
    mode: str = "fixed"
    b: jnp.ndarray = None

    def tree_flatten(self):
        return (self.a, self.b), self.mode

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], aux, children[1])
