"""Tests for ``with_field_overrides``: an override takes the kind of its field.

The scheme tests that run a real scheme with overridden parameters live with
the schemes (``clouds/echam_1m_test.py``, ``clouds/lohmann_2m/scheme_test.py``)
and the Hydra door in ``jcm/runners_test.py``; this module pins the conversion
rule itself on a small parameter class with one leaf of each kind.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jcm.physics.physics_term import with_field_overrides


@struct.dataclass
class _Toy:
    """One leaf of each kind a scheme's ``Parameters`` holds."""

    scalar: jnp.ndarray        # a tunable: 0-d float array (weakly typed)
    profile: jnp.ndarray       # a per-level profile: 1-d float array
    flag: jnp.ndarray          # a boolean 0-d array
    code: jnp.ndarray          # an integer selector held as a 0-d array
    plain_float: float         # a Python-float leaf
    plain_int: int = struct.field(pytree_node=False, default=2)
    plain_flag: bool = struct.field(pytree_node=False, default=True)
    mode: str = struct.field(pytree_node=False, default="a")


@struct.dataclass
class _CheckedToy(_Toy):
    """``_Toy`` with the opt-in cross-field ``validate`` hook."""

    def validate(self):
        if float(self.scalar) > 100.0:
            raise ValueError("scalar too large")


def _base(cls=_Toy):
    return cls(scalar=jnp.array(15.0), profile=jnp.array([1.0, 2.0, 3.0]),
               flag=jnp.array(True), code=jnp.array(1), plain_float=3.0)


def _override(base=None, **fields):
    return with_field_overrides(base or _base(), fields, scheme="toy")


@pytest.mark.parametrize("value", [
    20.0, 20, np.float64(20.0), np.float32(20.0), jnp.float32(20.0),
    np.array(20.0)])
def test_numeric_override_of_an_array_leaf_is_an_array_of_its_dtype(value):
    base = _base()
    out = _override(base, scalar=value)
    assert isinstance(out.scalar, jax.Array)
    assert out.scalar.dtype == base.scalar.dtype
    assert out.scalar.shape == base.scalar.shape == ()
    assert out.scalar.weak_type == base.scalar.weak_type
    assert float(out.scalar) == 20.0
    # Fields that are not overridden are the very same leaves.
    assert out.profile is base.profile and out.flag is base.flag


def test_weakly_typed_leaf_does_not_promote_a_float32_state_under_x64():
    # The defaults are weak (built from Python numbers), so the override must
    # be too: a strongly typed float64 leaf would promote float32 physics to
    # float64 under ``jax_enable_x64`` (the pySES configuration).
    with jax.enable_x64():
        out = _override(scalar=20)
        assert out.scalar.dtype == jnp.float64 and out.scalar.weak_type
        assert (out.scalar * jnp.ones(2, jnp.float32)).dtype == jnp.float32


def test_profile_override_takes_the_leaf_shape_and_dtype():
    for value in ([4.0, 5.0, 6.0], (4, 5, 6), np.array([4.0, 5.0, 6.0])):
        out = _override(profile=value)
        assert out.profile.dtype == _base().profile.dtype
        np.testing.assert_array_equal(np.asarray(out.profile), [4.0, 5.0, 6.0])


@pytest.mark.parametrize("field,value", [
    ("profile", 4.0),                  # a scalar is not broadcast over a profile
    ("profile", [4.0, 5.0]),
    ("profile", [[4.0, 5.0, 6.0]]),
    ("scalar", [20.0]),                # nor a length-1 list onto a scalar
    ("scalar", [20.0, 21.0]),
    ("plain_float", [1.0]),
])
def test_wrong_shape_is_an_error_naming_the_field(field, value):
    with pytest.raises(ValueError, match=rf"toy\.{field}: this field is"):
        _override(**{field: value})


def test_boolean_and_integer_leaves_take_only_representable_values():
    base = _base()
    for value in (False, 0, np.bool_(False), 0.0):
        out = _override(flag=value)
        assert out.flag.dtype == base.flag.dtype and not bool(out.flag)
    for value in (3, 3.0, np.int64(3), True):
        out = _override(code=value)
        assert out.code.dtype == base.code.dtype
        assert int(out.code) == int(value)
    for field, value in (("flag", 2), ("flag", 0.5), ("code", 2.5),
                         ("code", 2 ** 40)):
        with pytest.raises(ValueError, match="not representable"):
            _override(**{field: value})


@pytest.mark.parametrize("value", [["a", "b", "c"], 1 + 2j])
def test_non_real_values_are_an_error(value):
    with pytest.raises(ValueError, match="not a real numeric value"):
        _override(profile=value)


def test_python_scalar_leaves_keep_their_python_type():
    out = _override(plain_float=3600, plain_int=5, plain_flag=False)
    assert out.plain_float == 3600.0 and type(out.plain_float) is float
    assert out.plain_int == 5 and type(out.plain_int) is int
    assert out.plain_flag is False
    out = _override(plain_int=3.0, plain_flag=0, mode="b")
    assert out.plain_int == 3 and type(out.plain_int) is int
    assert out.plain_flag is False
    out = _override(plain_flag=1)
    assert out.plain_flag is True
    # A string is a spelling the class interprets itself: passed through.
    assert _override(mode="b").mode == "b"


@pytest.mark.parametrize("field,value", [
    ("plain_int", 2.5),       # an integer selector is not truncated
    ("plain_flag", 2),        # a flag is not silently true
    ("plain_flag", 0.5),
])
def test_python_integer_and_boolean_leaves_take_only_representable_values(
        field, value):
    with pytest.raises(ValueError, match=rf"toy\.{field}: .* not representable"):
        _override(**{field: value})
    with pytest.raises(ValueError, match=rf"toy\.{field}: this field is a scalar"):
        _override(**{field: [value]})


def test_overridden_leaf_is_a_differentiable_pytree_leaf():
    out = _override(scalar=20.0)
    assert any(leaf is out.scalar for leaf in jax.tree_util.tree_leaves(out))
    grads = jax.grad(lambda p: p.scalar ** 2, allow_int=True)(out)
    assert float(grads.scalar) == pytest.approx(40.0)


def test_conversion_accepts_a_tracer():
    # The dtype and shape checks use only static information.
    grad = jax.grad(lambda x: with_field_overrides(
        _base(), {"scalar": x}, scheme="toy").scalar ** 2)(3.0)
    assert float(grad) == pytest.approx(6.0)


def test_unknown_field_and_the_validate_hook_still_apply():
    with pytest.raises(ValueError, match=r"toy: unknown _Toy field\(s\) \['x'\]"):
        _override(x=1.0)
    # ``validate`` sees the converted (array) leaves.
    checked = _base(_CheckedToy)
    with pytest.raises(ValueError, match="scalar too large"):
        _override(checked, scalar=1000.0)
    assert _override(checked, scalar=50.0).scalar == 50.0
    assert with_field_overrides(checked, None, scheme="toy").scalar == 15.0
