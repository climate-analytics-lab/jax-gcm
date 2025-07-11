"""Tests whether the optics utilities are functioning properly."""

import collections
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from jcm.rrtmgp.optics import lookup_gas_optics_longwave
from jcm.rrtmgp.optics import optics_utils

IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight
Interpolant: TypeAlias = optics_utils.Interpolant
OrderedDict: TypeAlias = collections.OrderedDict

ROOT_DIR = epath.resource_path('jcm.rrtmgp.optics.rrtmgp_data')
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp-gas-lw-g256.nc'
_LW_LOOKUP_TABLE_FILEPATH = ROOT_DIR / _LW_LOOKUP_TABLE_FILENAME


def assert_interpolant_allclose(i1: Interpolant, i2: Interpolant):
	rtol = 1e-5
	atol = 0
	np.testing.assert_allclose(i1.interp_low.idx, i2.interp_low.idx, rtol, atol)
	np.testing.assert_allclose(
		i1.interp_low.weight, i2.interp_low.weight, rtol, atol
	)
	np.testing.assert_allclose(i1.interp_high.idx, i2.interp_high.idx, rtol, atol)
	np.testing.assert_allclose(
		i1.interp_high.weight, i2.interp_high.weight, rtol, atol
	)


class OpticsUtilsTest(parameterized.TestCase):

	@parameterized.parameters(True, False)
	def test_lookup_values(self, use_direct_indexing: bool):
		"""Tests the `lookup_values` operation with different tensor shapes."""
		if use_direct_indexing:
			# Use direct indexing for lookup.
			lookup_fn = optics_utils.lookup_values_direct_indexing
		else:
			# Use one-hot vectors and einsum for lookup via matrix multiplication.
			lookup_fn = optics_utils.lookup_values

		with self.subTest('1DCoeffs1DIndex'):
			coeffs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
			idx = [8, 7, 6, 5, 4, 3, 2, 1, 0]
			expected = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
			result = lookup_fn(coeffs, (idx,))
			np.testing.assert_equal(result, expected)
			
		with self.subTest('1DCoeffs2DIndex'):
			coeffs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
			idx = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
			expected = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
			result = lookup_fn(coeffs, (idx,))
			np.testing.assert_equal(result, expected)
			
		with self.subTest('2DCoeffs1DIndex'):
			coeffs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
			idx0 = [2, 2, 2, 1, 1, 1, 0, 0, 0]
			idx1 = [2, 1, 0, 2, 1, 0, 2, 1, 0]
			expected = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
			result = lookup_fn(coeffs, (idx0, idx1))
			np.testing.assert_equal(result, expected)
			
		with self.subTest('2DCoeffs2DIndex'):
			coeffs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
			idx0 = [[2, 2, 2], [1, 1, 1], [0, 0, 0]]
			idx1 = [[2, 1, 0], [2, 1, 0], [2, 1, 0]]
			expected = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
			result = lookup_fn(coeffs, (idx0, idx1))
			np.testing.assert_equal(result, expected)
			
		with self.subTest('3DCoeffs3DIndex'):
			coeffs = jnp.array([
				[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
				[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
			])
			idx0 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
			idx1 = [[2, 2, 2], [1, 1, 1], [0, 0, 0]]
			idx2 = [[2, 1, 0], [2, 1, 0], [2, 1, 0]]
			
			expected = np.array(
				[[9.0, 80.0, 7.0], [60.0, 5.0, 40.0], [3.0, 20.0, 1.0]]
			)
			result = lookup_fn(coeffs, (idx0, idx1, idx2))
			np.testing.assert_equal(result, expected)

	def test_evaluate_weighted_lookup(self):
		"""Test whether the `weighted_lookup` yields the correct scaled values."""
		coeffs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
		idx = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
		weight = jnp.array(
			[[1.0, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6], [1 / 7, 1 / 8, 1 / 9]]
		)
		result = optics_utils.evaluate_weighted_lookup(
			coeffs, [IndexAndWeight(idx, weight)]
		)
		expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
		np.testing.assert_equal(result, expected)

	def test_floor_idx(self):
		"""Tests whether `floor_idx` returns the floor index of given values."""
		ref_vals = jnp.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
		vals = jnp.array((1.0, 1.5, 5.6, 7.8, 9.0, 10.01))
		expected_floor_idx = np.array((0, 0, 4, 6, 8, 9))
		floor_idx = optics_utils.floor_idx(vals, ref_vals)
		np.testing.assert_equal(floor_idx, expected_floor_idx)

	def test_create_linear_interpolant(self):
		"""Tests the creation of a linear interpolant and out of range exception."""
		ref_vals = jnp.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
		vals = jnp.array((1.0, 1.5, 5.6, 7.8, 9.0, 10.0))
		idx_low = jnp.array((0, 0, 4, 6, 8, 9))
		idx_high = jnp.array((1, 1, 5, 7, 9, 9))
		weight_low = jnp.array((1.0, 0.5, 0.4, 0.2, 1.0, 1.0))
		weight_high = jnp.array((0.0, 0.5, 0.6, 0.8, 0.0, 0.0))
		idx_and_weight_low = IndexAndWeight(idx_low, weight_low)
		idx_and_weight_high = IndexAndWeight(idx_high, weight_high)
		expected_interpolant = Interpolant(idx_and_weight_low, idx_and_weight_high)
		
		with self.subTest('ValuesWithinReferenceRange'):
			interpolant = optics_utils.create_linear_interpolant(vals, ref_vals)
			assert_interpolant_allclose(interpolant, expected_interpolant)

		with self.subTest('WithOffset'):
			offset = jnp.array((1, 0, 1, 0, 1, 0))
			interpolant = optics_utils.create_linear_interpolant(
				vals, ref_vals, offset
			)
			idx_and_weight_low_offset = IndexAndWeight(idx_low + offset, weight_low)
			idx_and_weight_high_offset = IndexAndWeight(
				idx_high + offset, weight_high
			)
			expected_interpolant_offset = Interpolant(
				idx_and_weight_low_offset, idx_and_weight_high_offset
			)
			assert_interpolant_allclose(interpolant, expected_interpolant_offset)

	@parameterized.parameters(True, False)
	def test_interpolate(self, use_optimized_interpolation: bool):
		"""Tests `interpolate` on a given lookup array and list of interpolants."""
		# SETUP
		if use_optimized_interpolation:
			interpolate_fn = optics_utils.interpolate_optimized
		else:
			interpolate_fn = optics_utils.interpolate_orig
		
		coeffs = jnp.array([
			[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
			[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
		])
		idx1_low = [[0, 1], [0, 1]]
		idx1_low_weight = 0.2 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx1_high = [[1, 1], [1, 1]]
		idx1_high_weight = 0.8 * jnp.ones((2, 2), dtype=jnp.float32)
	
		idx2_low = [[1, 2], [0, 2]]
		idx_2_low_weight = 0.4 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx2_high = [[2, 2], [1, 2]]
		idx_2_high_weight = 0.6 * jnp.ones((2, 2), dtype=jnp.float32)

		idx3_low = [[1, 2], [2, 0]]
		idx_3_low_weight = 0.9 * jnp.ones((2, 2), dtype=jnp.float32)

		idx3_high = [[2, 2], [2, 1]]
		idx_3_high_weight = 0.1 * jnp.ones((2, 2), dtype=jnp.float32)

		element00 = (0.2 * 0.4 * 0.9 * 5.0 +  # low, low, low
	                 0.8 * 0.4 * 0.9 * 50.0 +  # high, low, low
	                 0.2 * 0.6 * 0.9 * 8.0 +  # low, high, low
	                 0.2 * 0.4 * 0.1 * 6.0 +  # low, low, high
	                 0.8 * 0.6 * 0.9 * 80.0 +  # high, high, low
	                 0.8 * 0.4 * 0.1 * 60.0 +  # high, low, high
	                 0.2 * 0.6 * 0.1 * 9.0 +  # low, high, high
	                 0.8 * 0.6 * 0.1 * 90.0)  # high, high, high

		idx1_weight_low = IndexAndWeight(idx1_low, idx1_low_weight)
		idx1_weight_high = IndexAndWeight(idx1_high, idx1_high_weight)
		interpolant1 = Interpolant(idx1_weight_low, idx1_weight_high)

		idx2_weight_low = IndexAndWeight(idx2_low, idx_2_low_weight)
		idx2_weight_high = IndexAndWeight(idx2_high, idx_2_high_weight)
		interpolant2 = Interpolant(idx2_weight_low, idx2_weight_high)

		idx3_weight_low = IndexAndWeight(idx3_low, idx_3_low_weight)
		idx3_weight_high = IndexAndWeight(idx3_high, idx_3_high_weight)
		interpolant3 = Interpolant(idx3_weight_low, idx3_weight_high)
		
		interpolant_fns = OrderedDict((
	        ('x', lambda: interpolant1),
	        ('y', lambda: interpolant2),
	        ('z', lambda: interpolant3),
	    ))
		
		# ACTION
		interpolated_values = interpolate_fn(coeffs, interpolant_fns)
		
		# VERIFICATION
		self.assertEqual(interpolated_values.shape, (2, 2))
		self.assertEqual(interpolated_values[0, 0], element00)

	@parameterized.parameters(True, False)
	def test_interpolate_with_dependency(self, use_optimized_interpolation: bool):
		"""Tests `interpolate` on a given lookup tensor and list of interpolants."""
		# SETUP
		if use_optimized_interpolation:
		  interpolate_fn = optics_utils.interpolate_optimized
		else:
		  interpolate_fn = optics_utils.interpolate_orig
		
		coeffs = jnp.array([
			[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
			[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
		])
		idx1_low = [[0, 1], [0, 1]]
		idx1_low_weight = 0.2 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx1_high = [[1, 1], [1, 1]]
		idx1_high_weight = 0.8 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx2_low = [[1, 2], [0, 2]]
		idx_2_low_weight = 0.4 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx2_high = [[2, 2], [1, 2]]
		idx_2_high_weight = 0.6 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx3_low = [[1, 2], [2, 0]]
		idx_3_low_weight = 0.9 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx3_high = [[2, 2], [2, 1]]
		idx_3_high_weight = 0.1 * jnp.ones((2, 2), dtype=jnp.float32)
		
		idx1_weight_low = IndexAndWeight(idx1_low, idx1_low_weight)
		idx1_weight_high = IndexAndWeight(idx1_high, idx1_high_weight)
		interpolant1 = Interpolant(idx1_weight_low, idx1_weight_high)
		
		idx2_weight_low = IndexAndWeight(idx2_low, idx_2_low_weight)
		idx2_weight_high = IndexAndWeight(idx2_high, idx_2_high_weight)
		interpolant2 = Interpolant(idx2_weight_low, idx2_weight_high)
		
		# Create a dependent interpolant for the 3-rd axis that adds the `x` weights
		# to the lower index weight and adds the `y` weights to the upper index
		# weights.
		def interpolant_3_fn(x: IndexAndWeight, y: IndexAndWeight) -> Interpolant:
		  idx3_weight_low = IndexAndWeight(
			  idx3_low, x.weight + idx_3_low_weight
		  )
		  idx3_weight_high = IndexAndWeight(
			  idx3_high, y.weight + idx_3_high_weight
		  )
		  return Interpolant(idx3_weight_low, idx3_weight_high)
		
		interpolant_fns = OrderedDict((
			('x', lambda: interpolant1),
			('y', lambda: interpolant2),
			('z', interpolant_3_fn),
		))
		
		# ACTION
		interpolated_values = interpolate_fn(coeffs, interpolant_fns)
		
		# VERIFICATION
		expected_element00 = (
			0.2 * 0.4 * (0.2 + 0.9) * 5.0 +  # low, low, low
			0.8 * 0.4 * (0.8 + 0.9) * 50.0 +  # high, low, low
			0.2 * 0.6 * (0.2 + 0.9) * 8.0 +  # low, high, low
			0.2 * 0.4 * (0.4 + 0.1) * 6.0 +  # low, low, high
			0.8 * 0.6 * (0.8 + 0.9) * 80.0 +  # high, high, low
			0.8 * 0.4 * (0.4 + 0.1) * 60.0 +  # high, low, high
			0.2 * 0.6 * (0.6 + 0.1) * 9.0 +  # low, high, high
			0.8 * 0.6 * (0.6 + 0.1) * 90.0)  # high, high, high
		self.assertEqual(interpolated_values.shape, (2, 2))
		self.assertAlmostEqual(
			interpolated_values[0, 0], expected_element00, delta=1e-4
		)

	@parameterized.parameters(True, False)
	def test_recover_original_values_via_interpolation(
	  self, use_optimized_interpolation: bool
	):
		"""Tests whether interpolation recovers original values."""
		if use_optimized_interpolation:
			interpolate_fn = optics_utils.interpolate_optimized
		else:
			interpolate_fn = optics_utils.interpolate_orig
		gas_optics = lookup_gas_optics_longwave.from_nc_file(
			_LW_LOOKUP_TABLE_FILEPATH
		)
		t_ref = gas_optics.t_ref
		reversed_t_ref = jnp.flip(t_ref, axis=0)
		
		key = jax.random.key(42)
		t = jax.random.uniform(
			key, (20, 20), minval=jnp.min(t_ref), maxval=jnp.max(t_ref)
		)
		
		with self.subTest('IncreasingOrderRefValues'):
			interpolant = optics_utils.create_linear_interpolant(t, t_ref)
			interpolant_fns = OrderedDict({'x': lambda: interpolant})
			interpolated = interpolate_fn(t_ref, interpolant_fns)
			np.testing.assert_allclose(interpolated, t, rtol=2e-7)
		
		with self.subTest('DecreasingOrderRefValues'):
			interpolant = optics_utils.create_linear_interpolant(t, reversed_t_ref)
			interpolant_fns = OrderedDict({'x': lambda: interpolant})
			interpolated = interpolate_fn(reversed_t_ref, interpolant_fns)
			np.testing.assert_allclose(interpolated, t, rtol=2e-7)


if __name__ == '__main__':
	absltest.main()
