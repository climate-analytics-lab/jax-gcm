"""Tests for :mod:`jcm.dycore.dinosaur.dot_precision`."""

from __future__ import annotations

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from dinosaur import jax_numpy_utils, spherical_harmonic

from jcm.dycore.dinosaur import dot_precision

X6 = jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
X3 = jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3
HIGHEST = jax.lax.Precision.HIGHEST


class JaxlibGateTest(unittest.TestCase):

    def test_affected_versions(self):
        for version in ("0.10.2", "0.11.0", "0.11.1"):
            self.assertTrue(dot_precision.jaxlib_is_affected(version), version)

    def test_fixed_versions(self):
        for version in ("0.11.2", "0.11.3", "0.12.0"):
            self.assertFalse(dot_precision.jaxlib_is_affected(version), version)


def _dinosaur_defaults():
    """Patch both dinosaur defaults back to their upstream values for a test."""
    return (mock.patch.object(jax_numpy_utils, "FLOAT32_DOT_ALGORITHM", X6),
            mock.patch.object(spherical_harmonic, "FAST_TRANSFORM_DOT_ALGORITHM", X3))


class ApplyWorkaroundTest(unittest.TestCase):

    def test_replaces_both_defaults_on_affected_jaxlib(self):
        p1, p2 = _dinosaur_defaults()
        with p1, p2:
            changed = dot_precision.apply_dot_precision_workaround("0.11.1")
            self.assertEqual(changed, {"float32": True, "fast_transform": True})
            self.assertEqual(jax_numpy_utils.FLOAT32_DOT_ALGORITHM, HIGHEST)
            self.assertEqual(spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM, HIGHEST)

    def test_fixed_jaxlib_keeps_x6_but_still_raises_spmd_transforms(self):
        # The X6 override is for the jaxlib bug; the X3 one is not version-gated.
        p1, p2 = _dinosaur_defaults()
        with p1, p2:
            changed = dot_precision.apply_dot_precision_workaround("0.11.2")
            self.assertEqual(changed, {"float32": False, "fast_transform": True})
            self.assertEqual(jax_numpy_utils.FLOAT32_DOT_ALGORITHM, X6)
            self.assertEqual(spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM, HIGHEST)

    def test_respects_caller_chosen_algorithms(self):
        chosen = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
        with mock.patch.object(jax_numpy_utils, "FLOAT32_DOT_ALGORITHM", chosen), \
                mock.patch.object(spherical_harmonic, "FAST_TRANSFORM_DOT_ALGORITHM", chosen):
            changed = dot_precision.apply_dot_precision_workaround("0.11.1")
            self.assertEqual(changed, {"float32": False, "fast_transform": False})
            self.assertEqual(jax_numpy_utils.FLOAT32_DOT_ALGORITHM, chosen)
            self.assertEqual(spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM, chosen)

    def test_applied_on_import(self):
        # jcm imports jcm.dycore.dinosaur, which applies the workaround.
        self.assertEqual(spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM, HIGHEST)
        if dot_precision.jaxlib_is_affected():
            self.assertEqual(jax_numpy_utils.FLOAT32_DOT_ALGORITHM, HIGHEST)


@unittest.skipUnless(jax.default_backend() == "gpu", "the miscompile is GPU-only")
class SingleLevelTransformGpuTest(unittest.TestCase):
    """The shape that triggers the XLA bug must transform accurately.

    The inverse transform of a single-level (size-1 leading dim) field, with
    dinosaur's constant Legendre basis, is what drifted log surface pressure.
    Compare against a float64 numpy evaluation of the same basis.
    """

    def test_single_level_inverse_transform(self):
        grid = spherical_harmonic.Grid.T42()
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1,) + grid.modal_shape).astype(np.float32)
        basis = grid.spherical_harmonics.basis
        p = np.asarray(basis.p, np.float64)
        f = np.asarray(basis.f, np.float64)
        expected = np.einsum("im,...mj->...ij", f,
                             np.einsum("mjl,...ml->...mj", p, x.astype(np.float64)))
        actual = np.asarray(jax.jit(grid.to_nodal)(jnp.asarray(x)), np.float64)
        rel_err = np.abs(actual - expected).max() / np.abs(expected).max()
        # float32 accuracy is ~4e-7 here; the miscompile gives ~2e-5.
        self.assertLess(rel_err, 2e-6)


if __name__ == "__main__":
    unittest.main()
