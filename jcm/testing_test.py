"""Tests for the structure-independent gradient checker."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import tree_math

from jcm.testing import (DEFAULT_STEPS, _check_unique, _freeze, _leaf_names,
                         _tangent, check_gradients, random_direction)


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

    def test_the_step_is_relative_to_each_leaf_magnitude(self):
        """``random_direction`` stays the unscaled building block; ``_tangent``
        is what weights it, per leaf, by that leaf's own RMS.
        """
        tree = {"big": jnp.full((256,), 3.1e5), "small": jnp.full((256,), 2e-5)}
        direction, tangent = random_direction(tree), _tangent(tree, 0)
        for name in tree:
            np.testing.assert_allclose(
                np.asarray(tangent[name]),
                np.asarray(direction[name]) * float(abs(tree[name][0])),
                rtol=1e-5)

    def test_non_float_leaves_get_a_float0_tangent(self):
        # jax.jvp rejects any other tangent dtype for an integer primal.
        d = random_direction({"i": jnp.arange(5), "f": jnp.zeros(5)})
        self.assertEqual(d["i"].dtype, jax.dtypes.float0)
        self.assertTrue(np.any(np.asarray(d["f"]) != 0))

    def test_integer_arguments_are_accepted_end_to_end(self):
        f = lambda x, i: jnp.sum(x**2) + jnp.sum(i).astype(x.dtype)
        check_gradients(f, (jnp.linspace(0.5, 2.0, 4), jnp.arange(4)), rtol=1e-3)

    def test_an_empty_leaf_costs_no_nan(self):
        """An empty leaf has no RMS, and must not reach one through ``mean``.

        ``ForcingData`` carries a zero-length ozone climatology whenever none
        is loaded, so every ECHAM fixture hands ``check_gradients`` an empty
        leaf. A mean over no elements is NaN, which would abort the check
        under ``jax_debug_nans`` — exactly the flag a caller turns on to chase
        the non-finite gradient a check has just reported.
        """
        with jax.debug_nans(True):
            tangent = _tangent({"empty": jnp.zeros((0,)),
                                "full": jnp.full((4,), 2.0)}, 0)
        self.assertEqual(tangent["empty"].shape, (0,))
        self.assertTrue(np.all(np.isfinite(np.asarray(tangent["full"]))))


def _smooth(x, y):
    return jnp.sum(jnp.sin(x) * y**2), jnp.cos(x) @ y


class TestCheckGradients(unittest.TestCase):

    # Tolerance for the positive smooth-function checks, which compare AD
    # against a float32 central difference. ``self.args`` holds a sub-unit leaf
    # (``x`` spans 0.1-1.0, RMS 0.63), and ``check_gradients`` displaces each
    # leaf by a fraction of its own RMS, so that leaf moves by only ~0.63*eps —
    # a smaller absolute step than the unit-scale direction the harness took
    # before it went relative. The float32 secant's relative error is
    # ~ulp(f)/(2*eps*RMS*|f'|), which for a sub-unit leaf floors the reference
    # near 2e-3; rtol=1e-3 sat just under that floor and its pass/fail was
    # decided by float32 rounding order (it passed on one platform and failed
    # on CI by ~1.4e-3). 5e-3 clears the floor with margin on every seed while
    # still catching any gradient error >= 1 % — a wrong gradient disagrees by
    # O(1), as ``test_a_wrong_gradient_is_caught`` confirms. Real physics
    # callers keep rtol=1e-3: their leaves are >= O(1), where the relative step
    # scales the displacement UP and the secant is correspondingly cleaner.
    SMOOTH_RTOL = 5e-3

    def setUp(self):
        self.args = (jnp.linspace(0.1, 1.0, 6), jnp.linspace(-1.0, 2.0, 6))

    def test_smooth_function_passes(self):
        check_gradients(_smooth, self.args, rtol=self.SMOOTH_RTOL)

    def test_a_second_direction_also_passes(self):
        check_gradients(_smooth, self.args, rtol=self.SMOOTH_RTOL, seed=11)

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

    def test_a_gradient_lost_on_a_large_input_is_caught(self):
        """The input-side mirror, and the sharper case: a unit-scale direction
        times an absolute step is under a float32 ulp of an O(1e5) leaf, so the
        shifted argument rounds back to the original bit for bit and the secant
        learns nothing about it. SPEEDY's dry static energy (se ~ 3.1e5, ulp
        ~0.03) is the real instance — a stop_gradient on it left this green.

        ``se`` enters the scheme only through vertical differences
        (``speedy_vdiff.py`` forms ``se0 - se``), which is what keeps the
        output O(100) while the leaf is O(1e5): the mismatch is invisible in
        the output magnitude and only the input scaling exposes it.
        """
        # A large leaf with O(100) structure on top of it.
        se = 3.1e5 + jnp.linspace(0.0, 100.0, 8)

        def f(x_small, se_in):
            # stop_gradient stands in for every way a large leaf's gradient can
            # go missing: an integer cast, a dropped term, a wrong custom rule.
            return (jnp.sum(x_small**2)
                    + jnp.sum(jnp.diff(jax.lax.stop_gradient(se_in))))

        with self.assertRaises(AssertionError):
            check_gradients(f, (jnp.linspace(0.5, 2.0, 8), se), rtol=1e-3)

    def test_an_absolute_step_could_not_have_seen_that_large_input(self):
        """The other half of the discrimination above, stated directly on the
        arithmetic: at unit direction scale every rung of the ladder leaves a
        3.1e5 leaf bit-identical, so no comparison downstream could have told
        its gradient from zero.
        """
        big = jnp.full((8,), 3.1e5)
        unscaled = random_direction((big,))[0]
        for eps in DEFAULT_STEPS:
            np.testing.assert_array_equal(big + eps * unscaled, big)
        # Scaled by the leaf's RMS, even the smallest rung moves it.
        scaled = _tangent((big,), 0)[0]
        self.assertFalse(
            np.array_equal(big + DEFAULT_STEPS[-1] * scaled, big))

    def test_the_step_is_the_rung_at_both_ends_of_the_float32_range(self):
        """The RMS is a mean of *squares*, so it needs twice the leaf's dynamic
        range and cannot be taken in float32: a squared 1.0e-25 underflows to
        zero, which reads as "no magnitude" and falls through to the unit
        scale, and a squared 1.0e+25 overflows to infinity.

        Both ends are inside the range of ordinary model fields, which is what
        makes this more than an arithmetic curiosity — a spectral-ringing
        condensate tail at 4e-30 kg/kg is the state ``echam_1m`` is written
        for, and it was being displaced by ~1e22 times its own value.
        """
        top = DEFAULT_STEPS[0]
        for magnitude in (1e-25, 1e25):
            leaf = jnp.full((64,), magnitude)
            tangent = _tangent((leaf,), 0)[0]
            self.assertTrue(np.all(np.isfinite(np.asarray(tangent))), magnitude)
            displaced = np.asarray(leaf + top * tangent, np.float64)
            relative = float(np.sqrt(np.mean((displaced / magnitude - 1.0)**2)))
            # The draw is standard normal, so its own RMS is ~1 over 64 samples.
            self.assertAlmostEqual(relative / top, 1.0, delta=0.5, msg=magnitude)

    def test_a_gradient_lost_at_either_end_of_that_range_is_caught(self):
        """The same discrimination as the O(1e5) input above, at both extremes:
        a leaf is only checked at all while its own perturbation is the rung,
        so these two are the cases a float32 RMS could not reach. With the
        scale fallen back to 1.0 a 1e-25 leaf moved by 1e22 times its value,
        and a 1e25 leaf got an infinite tangent; either way the *correct*
        gradient was rejected as having no usable reference, and no wrong one
        could be told apart from it.
        """
        x = jnp.linspace(0.5, 2.0, 16)
        tail, big = jnp.full((16,), 1e-25), jnp.full((16,), 1e25)

        def lost(x_in, extreme, weight):
            return (jnp.sum(x_in**2)
                    + jnp.sum(jnp.sin(weight * jax.lax.stop_gradient(extreme))))

        for extreme, weight in ((tail, 1e25), (big, 1e-25)):
            # Live, the same function is an ordinary smooth check.
            check_gradients(lambda a, b, w=weight: jnp.sum(a**2)
                            + jnp.sum(jnp.sin(w * b)), (x, extreme), rtol=1e-3)
            with self.assertRaises(AssertionError):
                check_gradients(lambda a, b, w=weight: lost(a, b, w),
                                (x, extreme), rtol=1e-3)

    def test_a_zero_leaf_falls_back_to_an_absolute_step(self):
        """An identically-zero leaf — a cloud-water field a fixture never fills
        — has no magnitude to be relative to, and its primal says nothing about
        the scale f responds on, so the tangent keeps unit scale.
        """
        zeros = jnp.zeros((8,))
        np.testing.assert_array_equal(_tangent((zeros,), 0)[0],
                                      random_direction((zeros,))[0])
        # And the fallback still drives a real check: exp is O(1)-sensitive at 0.
        check_gradients(lambda x: jnp.sum(jnp.exp(x)), (zeros,), rtol=1e-3)

    def test_a_named_input_that_is_dead_is_caught(self):
        """live_inputs is the input-side mirror of the per-output liveness
        guard: one projection only reports that *something* moved.
        """
        f = lambda x, y: jnp.sum(x**2) + jnp.sum(jax.lax.stop_gradient(y))
        with self.assertRaises(AssertionError):
            check_gradients(f, self.args, rtol=1e-3, live_inputs=["[1]"])

    def test_a_live_named_input_passes(self):
        check_gradients(_smooth, self.args, rtol=self.SMOOTH_RTOL,
                        live_inputs=["[0]", "[1]"])

    def test_a_named_input_that_does_not_exist_is_rejected(self):
        """A renamed field must fail loudly, not silently check nothing."""
        with self.assertRaises(ValueError):
            check_gradients(_smooth, self.args, rtol=1e-3,
                            live_inputs=["geopotential"])

    def test_a_fixed_input_is_not_perturbed(self):
        """A structural leaf — a grid descriptor that selects a branch rather
        than scaling a result — has no two-sided derivative, and holding it
        fixed must take it out of AD and the difference together.
        """
        # y only ever enters through a step, so any direction that moves it has
        # no usable difference; holding it fixed leaves a smooth function of x.
        f = lambda x, y: jnp.sum(x**2) + jnp.sum(jnp.where(y > 1.0, 3.0, 0.0))
        args = (jnp.linspace(0.5, 2.0, 8), jnp.ones((8,)))
        with self.assertRaises(AssertionError):
            check_gradients(f, args, rtol=1e-3)
        check_gradients(f, args, rtol=1e-3, fixed_inputs=["[1]"])

    def test_freezing_every_differentiable_leaf_is_rejected(self):
        """The zero direction: both the derivative and its reference are 0, so
        any gradient whatever agrees and the check asserts nothing. Reachable
        by naming an interior node that happens to cover the whole argument
        tuple, which is why it is caught rather than left to the caller.
        """
        with self.assertRaises(ValueError) as caught:
            check_gradients(_smooth, self.args, rtol=1e-3,
                            fixed_inputs=["[0]", "[1]"])
        self.assertIn("every differentiable", str(caught.exception))

    def test_a_fixed_input_is_not_checked_at_all(self):
        """Freezing takes the leaf out of the reverse projection as well as the
        tangent, so nothing about its gradient is asserted. This is the cost
        that makes "it selects a code path" the only admissible reason — the
        same stop_gradient that fails the check live passes once frozen.
        """
        f = lambda x, y: jnp.sum(x**2) + jnp.sum(jnp.sin(jax.lax.stop_gradient(y)))
        with self.assertRaises(AssertionError):
            check_gradients(f, self.args, rtol=1e-2)
        check_gradients(f, self.args, rtol=1e-2, fixed_inputs=["[1]"])

    def test_a_fixed_input_that_does_not_exist_is_rejected(self):
        with self.assertRaises(ValueError):
            check_gradients(_smooth, self.args, rtol=1e-3,
                            fixed_inputs=["speedy_coords"])

    def test_an_interior_name_reaches_a_whole_subtree(self):
        """``fixed_inputs=["speedy_coords"]`` has to reach every leaf of that
        struct, not just one that happens to be called exactly that.
        """
        tree = (_Outer(inner=_Inner(a=jnp.ones(4), b=jnp.ones(4)),
                       tail=jnp.ones(4)),)
        frozen = _freeze(_tangent(tree, 0), tree, ["inner"])
        np.testing.assert_array_equal(frozen[0].inner.a, np.zeros(4))
        np.testing.assert_array_equal(frozen[0].inner.b, np.zeros(4))
        self.assertTrue(np.any(np.asarray(frozen[0].tail) != 0))

    def test_live_inputs_names_a_leaf_by_its_field_name(self):
        f = lambda s: jnp.sum(s.inner.a**2) + jnp.sum(jnp.sin(s.tail))
        args = (_Outer(inner=_Inner(a=jnp.linspace(1.0, 2.0, 4),
                                    b=jnp.linspace(1.0, 2.0, 4)),
                       tail=jnp.linspace(0.1, 0.4, 4)),)
        check_gradients(f, args, rtol=1e-3, live_inputs=["inner/a", "tail"])
        # ``b`` is genuinely unused, so naming it must fail.
        with self.assertRaises(AssertionError):
            check_gradients(f, args, rtol=1e-3, live_inputs=["inner/b"])

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
        # |x - 1| evaluated exactly at the kink. The step is a fraction of the
        # leaf's own RMS (1.0 here), so every rung on the ladder straddles it:
        # the central secant is a stable 0 and the one-sided ones are -1 and +1.
        with self.assertRaises(AssertionError) as caught:
            check_gradients(lambda x: jnp.sum(jnp.abs(x - 1.0)),
                            (jnp.full((32,), 1.0),), rtol=1e-1)
        self.assertIn("kink", str(caught.exception))

    def test_a_legitimately_zero_gradient_is_accepted_on_atol(self):
        """Smooth, with an exactly zero directional derivative: the relative
        spread is meaningless there, so the ladder must not be exhausted and
        the failure blamed on a discontinuity.
        """
        f = lambda x: jnp.sum(x)**2
        check_gradients(f, (jnp.array([1.0, -1.0, 2.0, -2.0]),),
                        rtol=1e-3, atol=1e-5)

    def test_atol_does_not_wave_through_a_symmetric_kink(self):
        """The central secant of |x| at 0 is exactly 0 at every step, so an
        absolute-tolerance escape that looked only at the central pair would
        accept a zero reference — and a wrongly-zero gradient with it — where
        no derivative exists.

        """

        @jax.custom_jvp
        def bad_abs(x):
            return jnp.sum(jnp.abs(x))

        @bad_abs.defjvp
        def _jvp(primals, _tangents):
            (x,) = primals
            return bad_abs(x), jnp.zeros(())

        with self.assertRaises(AssertionError):
            check_gradients(bad_abs, (jnp.zeros(32),), rtol=1e-3, atol=1e-5)

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
