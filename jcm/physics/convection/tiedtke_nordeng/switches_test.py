"""ECHAM's convective decisions: exact values and surrogate derivatives.

For every switch in ``switches.py`` the four checks of
``docs/source/design/surrogate_gradients.md``: the value is the exact
function's bit for bit, both derivative modes are the surrogate's
(``check_surrogate_gradient``), the surrogate itself is smooth
(``check_gradients`` against a central difference), and the surrogate stays
within a stated distance of the exact switch. Each exact function is also
pinned against ECHAM's comparison, strict or inclusive, at the threshold
itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.convection.tiedtke_nordeng.switches import (
    ASCENT_MIN_FLUX_FRACTION,
    ascent_test,
    ascent_test_pair,
    relative_threshold_pair,
    relative_threshold_switch,
    rescaled_sigmoid,
    threshold_pair,
    threshold_switch,
)
from jcm.physics.convection.tiedtke_nordeng.types import ConvectionParameters
from jcm.physics.surrogate_gradient import with_surrogate_gradient
from jcm.testing import check_gradients, check_surrogate_gradient

_DEFAULT = ConvectionParameters.default()
_ASCENT_WIDTHS = (_DEFAULT.ascent_condensate_width,
                  _DEFAULT.ascent_buoyancy_width,
                  _DEFAULT.ascent_mass_flux_width)


def _ascent_args(n=41, seed=0):
    """Ascent-test arguments straddling each threshold."""
    rng = np.random.default_rng(seed)
    cond = jnp.asarray(np.concatenate([np.zeros(5), 10.0 ** rng.uniform(
        -10, -4, n - 5)]))
    zbuo = jnp.asarray(rng.uniform(-0.05, 0.05, n))
    mfub = jnp.asarray(rng.uniform(1e-3, 5e-2, n))
    mfu = mfub * jnp.asarray(rng.uniform(0.0, 0.03, n))
    return cond, zbuo, mfu, mfub


class TestThresholdSwitch:
    width = _DEFAULT.deep_convergence_width

    def test_value_is_echams_comparison(self):
        d = jnp.asarray([-1e-6, -1e-12, 0.0, 1e-12, 1e-6])
        np.testing.assert_array_equal(
            threshold_switch(d, self.width), [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(
            threshold_switch(d, self.width, inclusive=True), [0, 0, 1, 1, 1])

    @pytest.mark.parametrize("inclusive", [False, True])
    def test_derivative_is_the_surrogates(self, inclusive):
        exact, surrogate = threshold_pair(self.width, inclusive)
        f = with_surrogate_gradient(exact, surrogate)
        d = jnp.linspace(-5 * self.width, 5 * self.width, 41)
        check_surrogate_gradient(f, exact, surrogate, (d,))
        # ... and it is not the reference's, which is zero.
        assert float(jax.grad(lambda x: f(x))(jnp.asarray(0.0))) > 0.0

    def test_surrogate_is_smooth(self):
        _, surrogate = threshold_pair(self.width)
        d = jnp.linspace(-3 * self.width, 3 * self.width, 13)
        with jax.enable_x64(True):
            check_gradients(surrogate, (jnp.asarray(d, jnp.float64),),
                            rtol=1e-6)

    def test_surrogate_is_close_to_the_switch(self):
        exact, surrogate = threshold_pair(self.width)
        d = jnp.linspace(-50 * self.width, 50 * self.width, 2001)
        # (a margin over ten widths absorbs float32 rounding of ``d``)
        far = jnp.abs(d) >= 10.01 * self.width
        gap = jnp.abs(exact(d) - surrogate(d))
        assert float(jnp.max(gap[far])) <= float(jax.nn.sigmoid(-10.0))
        assert float(jnp.max(gap)) <= 0.5

    def test_zero_width_is_the_reference_derivative(self):
        grad = jax.grad(lambda x: threshold_switch(x, 0.0))(jnp.asarray(1e-9))
        assert float(grad) == 0.0


class TestRelativeThresholdSwitch:
    width = _DEFAULT.cloud_base_excess_width

    def test_value_is_echams_strict_comparison(self):
        floor = jnp.asarray(2.0e-4)
        v = floor * jnp.asarray([0.5, 1.0 - 1e-6, 1.0, 1.0 + 1e-6, 2.0])
        np.testing.assert_array_equal(
            relative_threshold_switch(v, floor, self.width), [0, 0, 0, 1, 1])

    def test_derivative_is_the_surrogates(self):
        exact, surrogate = relative_threshold_pair(self.width)
        f = with_surrogate_gradient(exact, surrogate)
        floor = jnp.full(21, 2.0e-4)
        value = floor * jnp.linspace(0.5, 1.5, 21)
        check_surrogate_gradient(f, exact, surrogate, (value, floor))

    def test_surrogate_is_smooth_and_close(self):
        exact, surrogate = relative_threshold_pair(self.width)
        with jax.enable_x64(True):
            floor = jnp.full(9, 2.0e-4, jnp.float64)
            value = floor * jnp.linspace(0.8, 1.2, 9)
            check_gradients(surrogate, (value, floor), rtol=1e-6)
        floor = jnp.full(401, 2.0e-4)
        value = floor * jnp.linspace(-1.0, 3.0, 401)
        far = jnp.abs(value / floor - 1.0) >= 10.01 * self.width
        gap = jnp.abs(exact(value, floor) - surrogate(value, floor))
        assert float(jnp.max(gap[far])) <= float(jax.nn.sigmoid(-10.0))


class TestAscentTest:
    def test_value_is_echams_conjunction(self):
        # mo_cuascent.f90:442-451: pqu < zqold (cond > 0), zbuo > 0 strictly,
        # pmfu >= 0.01·pmfub inclusively.
        mfub = 0.02
        cases = [
            # cond,  zbuo,   mfu,                     passes
            (1e-6, 1e-3, mfub, 1.0),
            (0.0, 1e-3, mfub, 0.0),
            (1e-6, 0.0, mfub, 0.0),
            (1e-6, -1e-9, mfub, 0.0),
            (1e-6, 1e-3, ASCENT_MIN_FLUX_FRACTION * mfub, 1.0),
            (1e-6, 1e-3, 0.999 * ASCENT_MIN_FLUX_FRACTION * mfub, 0.0),
        ]
        for cond, zbuo, mfu, want in cases:
            got = ascent_test(jnp.asarray(cond), jnp.asarray(zbuo),
                              jnp.asarray(mfu), jnp.asarray(mfub),
                              *_ASCENT_WIDTHS)
            assert float(got) == want, (cond, zbuo, mfu)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_derivative_is_the_surrogates(self, seed):
        exact, surrogate = ascent_test_pair(*_ASCENT_WIDTHS)
        f = with_surrogate_gradient(exact, surrogate)
        check_surrogate_gradient(f, exact, surrogate, _ascent_args(seed=seed))

    def test_buoyancy_carries_a_derivative_across_the_threshold(self):
        # A plume that just fails (zbuo < 0) and one that just passes both
        # respond to buoyancy; the reference derivative is zero on both sides.
        mfub = jnp.asarray(0.02)
        for zbuo in (-0.005, 0.005):
            grad = jax.grad(lambda b: ascent_test(
                jnp.asarray(1e-5), b, mfub, mfub, *_ASCENT_WIDTHS))(
                    jnp.asarray(zbuo))
            assert float(grad) > 0.0, zbuo
            ref = jax.grad(lambda b: ascent_test(
                jnp.asarray(1e-5), b, mfub, mfub, 0.0, 0.0, 0.0))(
                    jnp.asarray(zbuo))
            assert float(ref) == 0.0

    def test_surrogate_is_smooth(self):
        # Away from the condensate gate's kink at zero condensate (where the
        # plume stops condensing and the exact test fails too).
        _, surrogate = ascent_test_pair(*_ASCENT_WIDTHS)
        with jax.enable_x64(True):
            n = 9
            args = (jnp.full(n, 3e-8, jnp.float64),
                    jnp.linspace(-0.02, 0.02, n, dtype=jnp.float64),
                    jnp.full(n, 2.0e-4, jnp.float64),
                    jnp.full(n, 0.02, jnp.float64))
            # A product of three logistics: the central difference settles
            # to 3e-6 of the derivative at the steps the check tries.
            check_gradients(surrogate, args, rtol=1e-4)

    def test_surrogate_is_close_to_the_test(self):
        exact, surrogate = ascent_test_pair(*_ASCENT_WIDTHS)
        wc, wb, wm = _ASCENT_WIDTHS
        rng = np.random.default_rng(3)
        n = 5000
        cond = jnp.asarray(10.0 ** rng.uniform(-10, -3, n))
        zbuo = jnp.asarray(rng.uniform(-1.0, 1.0, n))
        mfub = jnp.asarray(rng.uniform(1e-3, 5e-2, n))
        ratio = rng.uniform(0.0, 1.0, n)
        mfu = mfub * jnp.asarray(ratio)
        # Far from every threshold (ten widths) the two differ by at most
        # three logistic tails.
        far = ((np.asarray(cond) > 10 * wc) & (np.abs(np.asarray(zbuo)) > 10 * wb)
               & (np.abs(ratio - ASCENT_MIN_FLUX_FRACTION) > 10 * wm))
        gap = np.abs(np.asarray(exact(cond, zbuo, mfu, mfub))
                     - np.asarray(surrogate(cond, zbuo, mfu, mfub)))
        assert far.sum() > 1000
        assert gap[far].max() <= 3 * float(jax.nn.sigmoid(-9.0))

    def test_condensate_gate_is_exactly_zero_without_condensation(self):
        x = jnp.asarray([-1e-9, 0.0])
        np.testing.assert_array_equal(rescaled_sigmoid(x, 1e-8), [0.0, 0.0])
        assert float(rescaled_sigmoid(jnp.asarray(1e-7), 1e-8)) > 0.9998


class TestWidthsAreStatic:
    """The widths configure derivatives only: static, and never negative."""

    def test_widths_are_not_pytree_leaves(self):
        leaves = jax.tree_util.tree_leaves(_DEFAULT)
        n_fields = len(_DEFAULT.__dataclass_fields__)
        from jcm.physics.convection.tiedtke_nordeng.types import (
            SURROGATE_WIDTH_FIELDS,
        )
        assert len(leaves) == n_fields - len(SURROGATE_WIDTH_FIELDS)

    def test_negative_width_is_rejected(self):
        with pytest.raises(ValueError):
            _DEFAULT.replace(ascent_buoyancy_width=-1.0).validate()
        with pytest.raises(ValueError):
            ConvectionParameters.default(ascent_buoyancy_width=-1.0)
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            TiedtkeConvection,
        )
        with pytest.raises(ValueError):
            TiedtkeConvection(_DEFAULT.replace(precip_onset_width=-1.0))

    def test_unknown_width_is_rejected(self):
        with pytest.raises(TypeError):
            ConvectionParameters.default(smooth_term_buoy=3e-4)
