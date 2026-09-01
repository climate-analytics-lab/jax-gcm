"""Tests for jcm/utils.py

Tests for utility functions including coordinate systems, spectral truncation,
dataset validation, pytree operations, and xarray conversion.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from dinosaur.hybrid_coordinates import HybridCoordinates
from dinosaur.sigma_coordinates import SigmaCoordinates
from jcm.utils import (
    get_coords,
    spectral_truncation,
    validate_ds,
    ones_like,
    ones_like_tangent,
    convert_to_float,
    convert_back,
    data_to_xarray,
    load_states_from_xarray,
)
from jcm.physics.speedy.physical_constants import SIGMA_LAYER_BOUNDARIES


class TestGetCoords(unittest.TestCase):
    """Tests for get_coords function."""

    def test_get_coords_with_spectral_truncation(self):
        """get_coords should create CoordinateSystem with spectral truncation."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=31)

        self.assertEqual(coords.horizontal.nodal_shape, (96, 48))
        self.assertEqual(coords.vertical.layers, 8)

    def test_get_coords_with_nodal_shape(self):
        """get_coords should infer truncation from nodal_shape."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, nodal_shape=(64, 32))

        # T21 has nodal shape (64, 32)
        self.assertEqual(coords.horizontal.nodal_shape, (64, 32))
        self.assertEqual(coords.horizontal.total_wavenumbers, 23)  # T21 + 2

    def test_get_coords_t63_uses_explicit_construct(self):
        """T63 has no Grid.T63 factory — it takes the Grid.construct branch.

        The 192x96 ECHAM grid is built with gaussian_nodes=48 so it matches
        the production boundary-condition files.
        """
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=63)
        self.assertEqual(coords.horizontal.nodal_shape, (192, 96))

    def test_get_coords_builds_spmd_mesh(self):
        """A 1x1x1 spmd_mesh exercises the mesh-build path on one device.

        The mesh product must equal the device count; (1, 1, 1) needs only the
        single device the fast suite has, while still running the Auto-axis
        ``make_mesh`` branch that multi-device runs rely on.
        """
        from jax.sharding import AxisType

        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(
            sigma_boundaries, spectral_truncation=21, spmd_mesh=(1, 1, 1),
        )
        self.assertIsNotNone(coords.spmd_mesh)
        self.assertEqual(coords.spmd_mesh.axis_names, ("x", "y", "z"))
        # Constraints require Auto axes (Explicit would make them raise).
        self.assertTrue(
            all(t == AxisType.Auto for t in coords.spmd_mesh.axis_types)
        )
        # Grid is unchanged by a unit mesh.
        self.assertEqual(coords.horizontal.nodal_shape, (64, 32))

    def test_get_coords_invalid_spectral_truncation(self):
        """get_coords should raise error for invalid spectral truncation."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]

        with self.assertRaises(ValueError) as context:
            get_coords(sigma_boundaries, spectral_truncation=50)  # Invalid
        self.assertIn("Invalid horizontal resolution", str(context.exception))

    def test_get_coords_invalid_nodal_shape(self):
        """get_coords should raise error for invalid nodal shape."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]

        with self.assertRaises(ValueError) as context:
            get_coords(sigma_boundaries, nodal_shape=(100, 50))  # Invalid
        self.assertIn("Invalid nodal shape", str(context.exception))

    def test_get_coords_different_vertical_levels(self):
        """get_coords should work with different numbers of vertical levels."""
        for layers in [7, 8]:
            sigma_boundaries = SIGMA_LAYER_BOUNDARIES[layers]
            coords = get_coords(sigma_boundaries, spectral_truncation=21)

            self.assertEqual(coords.vertical.layers, layers)

    def test_get_coords_nodal_shape_overrides_truncation(self):
        """nodal_shape should override spectral_truncation when both provided."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        # Provide T31 truncation but T21 nodal shape
        coords = get_coords(sigma_boundaries, spectral_truncation=31, nodal_shape=(64, 32))

        # Should use T21 from nodal_shape
        self.assertEqual(coords.horizontal.nodal_shape, (64, 32))

    def test_get_coords_with_sigma_coordinates_instance(self):
        """get_coords should accept a SigmaCoordinates instance directly."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        sigma_coords = SigmaCoordinates(sigma_boundaries)
        coords = get_coords(sigma_coords, spectral_truncation=21)

        self.assertIsInstance(coords.vertical, SigmaCoordinates)
        self.assertEqual(coords.vertical.layers, 8)

    def test_get_coords_with_hybrid_coordinates(self):
        """get_coords should accept a HybridCoordinates instance."""
        # Simple hybrid coords: 4 layers, a + b*ps
        a_boundaries = np.array([0.0, 100.0, 200.0, 300.0, 0.0])
        b_boundaries = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
        hybrid = HybridCoordinates(a_boundaries, b_boundaries)

        coords = get_coords(hybrid, spectral_truncation=21)

        self.assertIsInstance(coords.vertical, HybridCoordinates)
        self.assertEqual(coords.vertical.layers, 4)
        self.assertIs(coords.vertical, hybrid)
        self.assertEqual(coords.horizontal.nodal_shape, (64, 32))

    def test_get_coords_hybrid_with_nodal_shape(self):
        """get_coords should accept HybridCoordinates with nodal_shape arg."""
        a_boundaries = np.array([0.0, 100.0, 200.0, 300.0, 0.0])
        b_boundaries = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
        hybrid = HybridCoordinates(a_boundaries, b_boundaries)

        coords = get_coords(hybrid, nodal_shape=(64, 32))

        self.assertIsInstance(coords.vertical, HybridCoordinates)
        self.assertEqual(coords.horizontal.nodal_shape, (64, 32))


class TestSpectralTruncation(unittest.TestCase):
    """Tests for spectral_truncation function."""

    def test_spectral_truncation_uniform_field(self):
        """Uniform field should be preserved after spectral truncation."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)
        grid = coords.horizontal

        uniform_field = jnp.ones(grid.nodal_shape) * 100.0
        truncated = spectral_truncation(grid, uniform_field)

        # Uniform field should be preserved (only wavenumber 0 is non-zero)
        self.assertTrue(jnp.allclose(truncated, 100.0, rtol=1e-5))

    def test_spectral_truncation_removes_high_frequencies(self):
        """High-frequency noise should be reduced after truncation."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)
        grid = coords.horizontal

        # Create field with high-frequency variation
        lon, lat = grid.nodal_axes
        high_freq_field = jnp.sin(20 * lon[:, None]) * jnp.ones_like(lat)[None, :]

        truncated = spectral_truncation(grid, high_freq_field)

        # High frequency (wavenumber 20) should be removed for T21
        # The result should have lower variance than the input
        input_var = jnp.var(high_freq_field)
        output_var = jnp.var(truncated)
        self.assertTrue(output_var < input_var)

    def test_spectral_truncation_preserves_shape(self):
        """Output should have same shape as input."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=31)
        grid = coords.horizontal

        field = jnp.ones(grid.nodal_shape)
        truncated = spectral_truncation(grid, field)

        self.assertEqual(truncated.shape, field.shape)


class TestValidateDs(unittest.TestCase):
    """Tests for validate_ds function."""

    def test_validate_ds_valid_dataset(self):
        """validate_ds should not raise for valid dataset."""
        ds = xr.Dataset({
            'temp': (['lon', 'lat'], np.zeros((10, 5))),
            'pressure': (['lon', 'lat', 'level'], np.zeros((10, 5, 3))),
        })

        expected_structure = {
            'temp': ('lon', 'lat'),
            'pressure': ('lon', 'lat', 'level'),
        }

        # Should not raise
        validate_ds(ds, expected_structure)

    def test_validate_ds_missing_variable(self):
        """validate_ds should raise for missing variable."""
        ds = xr.Dataset({
            'temp': (['lon', 'lat'], np.zeros((10, 5))),
        })

        expected_structure = {
            'temp': ('lon', 'lat'),
            'humidity': ('lon', 'lat'),  # Missing
        }

        with self.assertRaises(ValueError) as context:
            validate_ds(ds, expected_structure)
        self.assertIn("Missing variables", str(context.exception))
        self.assertIn("humidity", str(context.exception))

    def test_validate_ds_wrong_dimensions(self):
        """validate_ds should raise for wrong dimensions."""
        ds = xr.Dataset({
            'temp': (['lon', 'lat', 'time'], np.zeros((10, 5, 3))),  # Has time dim
        })

        expected_structure = {
            'temp': ('lon', 'lat'),  # Expected no time dim
        }

        with self.assertRaises(ValueError) as context:
            validate_ds(ds, expected_structure)
        self.assertIn("temp", str(context.exception))
        self.assertIn("dims", str(context.exception))

    def test_validate_ds_empty_expected(self):
        """validate_ds should pass for empty expected structure."""
        ds = xr.Dataset({
            'temp': (['lon', 'lat'], np.zeros((10, 5))),
        })

        # Should not raise
        validate_ds(ds, {})


class TestOnesLike(unittest.TestCase):
    """Tests for ones_like function."""

    def test_ones_like_array(self):
        """ones_like should create ones with same shape."""
        x = jnp.zeros((3, 4, 5))
        result = ones_like(x)

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(jnp.allclose(result, 1.0))

    def test_ones_like_pytree(self):
        """ones_like should work with pytrees."""
        pytree = {
            'a': jnp.zeros((2, 3)),
            'b': jnp.zeros((4,)),
        }
        result = ones_like(pytree)

        self.assertTrue(jnp.allclose(result['a'], 1.0))
        self.assertTrue(jnp.allclose(result['b'], 1.0))
        self.assertEqual(result['a'].shape, (2, 3))
        self.assertEqual(result['b'].shape, (4,))

    def test_ones_like_nested_pytree(self):
        """ones_like should work with nested pytrees."""
        pytree = {
            'outer': {
                'inner': jnp.zeros((2, 2)),
            },
            'value': jnp.zeros((3,)),
        }
        result = ones_like(pytree)

        self.assertTrue(jnp.allclose(result['outer']['inner'], 1.0))
        self.assertTrue(jnp.allclose(result['value'], 1.0))


class TestOnesLikeTangent(unittest.TestCase):
    """Tests for ones_like_tangent function."""

    def test_ones_like_tangent_float_array(self):
        """ones_like_tangent should create ones for float arrays."""
        x = jnp.zeros((3, 4), dtype=jnp.float32)
        result = ones_like_tangent(x)

        self.assertTrue(jnp.allclose(result, 1.0))
        self.assertEqual(result.shape, x.shape)

    def test_ones_like_tangent_int_array(self):
        """ones_like_tangent should create float0 for int arrays."""
        x = jnp.zeros((3, 4), dtype=jnp.int32)
        result = ones_like_tangent(x)

        # Result should be float0 (no tangent space for integers)
        self.assertEqual(result.dtype, jax.dtypes.float0)

    def test_ones_like_tangent_bool_array(self):
        """ones_like_tangent should create float0 for bool arrays."""
        x = jnp.zeros((3, 4), dtype=jnp.bool_)
        result = ones_like_tangent(x)

        self.assertEqual(result.dtype, jax.dtypes.float0)

    def test_ones_like_tangent_pytree(self):
        """ones_like_tangent should work with pytrees."""
        pytree = {
            'float': jnp.zeros((2, 3), dtype=jnp.float32),
            'int': jnp.zeros((2,), dtype=jnp.int32),
        }
        result = ones_like_tangent(pytree)

        self.assertTrue(jnp.allclose(result['float'], 1.0))
        self.assertEqual(result['int'].dtype, jax.dtypes.float0)


class TestConvertToFloat(unittest.TestCase):
    """Tests for convert_to_float function."""

    def test_convert_to_float_from_int(self):
        """convert_to_float should convert int to float32."""
        x = jnp.array([1, 2, 3], dtype=jnp.int32)
        result = convert_to_float(x)

        self.assertEqual(result.dtype, jnp.float32)
        self.assertTrue(jnp.allclose(result, jnp.array([1., 2., 3.])))

    def test_convert_to_float_from_float64(self):
        """convert_to_float should convert float64 to float32."""
        x = jnp.array([1., 2., 3.], dtype=jnp.float64)
        result = convert_to_float(x)

        self.assertEqual(result.dtype, jnp.float32)

    def test_convert_to_float_pytree(self):
        """convert_to_float should work with pytrees."""
        pytree = {
            'int': jnp.array([1, 2], dtype=jnp.int32),
            'float64': jnp.array([1., 2.], dtype=jnp.float64),
        }
        result = convert_to_float(pytree)

        self.assertEqual(result['int'].dtype, jnp.float32)
        self.assertEqual(result['float64'].dtype, jnp.float32)


class TestConvertBack(unittest.TestCase):
    """Tests for convert_back function."""

    def test_convert_back_preserves_float(self):
        """convert_back should preserve float values."""
        x = jnp.array([1., 2., 3.], dtype=jnp.float32)
        x0 = jnp.array([0., 0., 0.], dtype=jnp.float32)

        result = convert_back(x, x0)

        self.assertTrue(jnp.allclose(result, x))

    def test_convert_back_restores_original_for_non_float(self):
        """convert_back should restore original value for non-float types."""
        x = jnp.array([1., 2., 3.], dtype=jnp.float32)  # Converted version
        x0 = jnp.array([10, 20, 30], dtype=jnp.int32)  # Original int

        result = convert_back(x, x0)

        # Should return x0 since x0 is not float32
        self.assertTrue(jnp.allclose(result, x0))
        self.assertEqual(result.dtype, jnp.int32)

    def test_convert_back_pytree(self):
        """convert_back should work with pytrees."""
        x = {
            'float': jnp.array([1., 2.], dtype=jnp.float32),
            'was_int': jnp.array([1., 2.], dtype=jnp.float32),
        }
        x0 = {
            'float': jnp.array([0., 0.], dtype=jnp.float32),
            'was_int': jnp.array([10, 20], dtype=jnp.int32),
        }

        result = convert_back(x, x0)

        self.assertTrue(jnp.allclose(result['float'], x['float']))
        self.assertTrue(jnp.allclose(result['was_int'], x0['was_int']))


class TestDataToXarray(unittest.TestCase):
    """Tests for data_to_xarray function."""

    def test_data_to_xarray_basic(self):
        """data_to_xarray should create xarray Dataset from data dict."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 5

        data = {
            'temperature': np.ones((n_times, n_levels) + nodal_shape),
        }
        times = np.arange(n_times)

        ds = data_to_xarray(data, coords=coords, times=times)

        self.assertIn('temperature', ds.data_vars)
        self.assertIn('time', ds.coords)
        self.assertIn('lon', ds.coords)
        self.assertIn('lat', ds.coords)

    def test_data_to_xarray_with_tracers(self):
        """data_to_xarray should handle tracers dict."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 3

        data = {
            'temperature': np.ones((n_times, n_levels) + nodal_shape),
            'tracers': {
                'specific_humidity': np.ones((n_times, n_levels) + nodal_shape),
            },
        }
        times = np.arange(n_times)

        ds = data_to_xarray(data, coords=coords, times=times)

        self.assertIn('temperature', ds.data_vars)
        self.assertIn('specific_humidity', ds.data_vars)

    def test_data_to_xarray_with_diagnostics(self):
        """data_to_xarray should handle diagnostics dict."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 3

        data = {
            'temperature': np.ones((n_times, n_levels) + nodal_shape),
            'diagnostics': {
                'precipitation': np.ones((n_times,) + nodal_shape),
            },
        }
        times = np.arange(n_times)

        ds = data_to_xarray(data, coords=coords, times=times)

        self.assertIn('temperature', ds.data_vars)
        self.assertIn('precipitation', ds.data_vars)

    def test_data_to_xarray_tracer_collision_raises(self):
        """data_to_xarray should raise if tracer name collides with prognostic."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 3

        data = {
            'temperature': np.ones((n_times, n_levels) + nodal_shape),
            'tracers': {
                'temperature': np.ones((n_times, n_levels) + nodal_shape),  # Collision
            },
        }
        times = np.arange(n_times)

        with self.assertRaises(ValueError) as context:
            data_to_xarray(data, coords=coords, times=times)
        self.assertIn("collide", str(context.exception).lower())

    def test_data_to_xarray_diagnostic_collision_raises(self):
        """data_to_xarray should raise if diagnostic name collides with prognostic."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 3

        data = {
            'temperature': np.ones((n_times, n_levels) + nodal_shape),
            'diagnostics': {
                'temperature': np.ones((n_times, n_levels) + nodal_shape),  # Collision
            },
        }
        times = np.arange(n_times)

        with self.assertRaises(ValueError) as context:
            data_to_xarray(data, coords=coords, times=times)
        self.assertIn("collide", str(context.exception).lower())

    def test_data_to_xarray_unrecognized_shape_raises(self):
        """data_to_xarray should raise for unrecognized data shape."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        data = {
            'weird_data': np.ones((7, 11, 13)),  # Not a valid shape
        }
        times = np.arange(5)

        with self.assertRaises(ValueError) as context:
            data_to_xarray(data, coords=coords, times=times)
        self.assertIn("shape", str(context.exception).lower())

    def test_data_to_xarray_without_times(self):
        """data_to_xarray should work without times."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers

        data = {
            'temperature': np.ones((n_levels,) + nodal_shape),
        }

        ds = data_to_xarray(data, coords=coords, times=None)

        self.assertIn('temperature', ds.data_vars)
        self.assertNotIn('time', ds.coords)

    def test_data_to_xarray_with_attrs(self):
        """data_to_xarray should include custom attrs."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers

        data = {
            'temperature': np.ones((n_levels,) + nodal_shape),
        }

        custom_attrs = {'description': 'Test dataset', 'version': '1.0'}
        ds = data_to_xarray(data, coords=coords, times=None, attrs=custom_attrs)

        self.assertEqual(ds.attrs['description'], 'Test dataset')
        self.assertEqual(ds.attrs['version'], '1.0')

    def test_data_to_xarray_serialize_coords_false(self):
        """data_to_xarray should not serialize coords when disabled."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers

        data = {
            'temperature': np.ones((n_levels,) + nodal_shape),
        }

        ds = data_to_xarray(
            data, coords=coords, times=None, serialize_coords_to_attrs=False
        )

        # Should not have coordinate system serialization in attrs
        self.assertNotIn('horizontal', ds.attrs)


class TestDataToXarrayEdgeCases(unittest.TestCase):
    """Additional edge case tests for data_to_xarray function."""

    def test_data_to_xarray_with_sample_ids(self):
        """data_to_xarray should handle sample_ids parameter."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 3
        n_samples = 2

        data = {
            'temperature': np.ones((n_samples, n_times, n_levels) + nodal_shape),
        }
        times = np.arange(n_times)
        sample_ids = np.arange(n_samples)

        ds = data_to_xarray(data, coords=coords, times=times, sample_ids=sample_ids)

        self.assertIn('temperature', ds.data_vars)
        self.assertIn('sample', ds.coords)
        self.assertIn('time', ds.coords)

    def test_data_to_xarray_tracer_unrecognized_shape_raises(self):
        """data_to_xarray should raise for unrecognized tracer shape."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 3

        data = {
            'temperature': np.ones((n_times, n_levels) + nodal_shape),
            'tracers': {
                'bad_tracer': np.ones((7, 11, 13)),  # Invalid shape
            },
        }
        times = np.arange(n_times)

        with self.assertRaises(ValueError) as context:
            data_to_xarray(data, coords=coords, times=times)
        self.assertIn("shape", str(context.exception).lower())

    def test_data_to_xarray_diagnostic_unrecognized_shape_raises(self):
        """data_to_xarray should raise for unrecognized diagnostic shape."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers
        n_times = 3

        data = {
            'temperature': np.ones((n_times, n_levels) + nodal_shape),
            'diagnostics': {
                'bad_diagnostic': np.ones((7, 11, 13)),  # Invalid shape
            },
        }
        times = np.arange(n_times)

        with self.assertRaises(ValueError) as context:
            data_to_xarray(data, coords=coords, times=times)
        self.assertIn("shape", str(context.exception).lower())

    def test_data_to_xarray_additional_coords_non_1d_raises(self):
        """data_to_xarray should raise for non-1d additional_coords."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers

        data = {
            'temperature': np.ones((n_levels,) + nodal_shape),
        }

        # 2D array is invalid for additional_coords
        additional_coords = {'bad_coord': np.ones((3, 4))}

        with self.assertRaises(ValueError) as context:
            data_to_xarray(
                data, coords=coords, times=None, additional_coords=additional_coords
            )
        self.assertIn("1d", str(context.exception).lower())

    def test_data_to_xarray_additional_coords_level_collision_raises(self):
        """data_to_xarray should raise if additional_coords shape collides with level."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers

        data = {
            'temperature': np.ones((n_levels,) + nodal_shape),
        }

        # Shape matches n_levels which would cause collision
        additional_coords = {'my_coord': np.ones((n_levels,))}

        with self.assertRaises(ValueError) as context:
            data_to_xarray(
                data, coords=coords, times=None, additional_coords=additional_coords
            )
        self.assertIn("collide", str(context.exception).lower())

    def test_data_to_xarray_attrs_collision_raises(self):
        """data_to_xarray should raise if attrs key collides with serialized coords."""
        sigma_boundaries = SIGMA_LAYER_BOUNDARIES[8]
        coords = get_coords(sigma_boundaries, spectral_truncation=21)

        nodal_shape = coords.horizontal.nodal_shape
        n_levels = coords.vertical.layers

        data = {
            'temperature': np.ones((n_levels,) + nodal_shape),
        }

        # 'longitude_wavenumbers' is a serialized coords key
        attrs = {'longitude_wavenumbers': 'bad value'}

        with self.assertRaises(ValueError) as context:
            data_to_xarray(
                data, coords=coords, times=None, attrs=attrs,
                serialize_coords_to_attrs=True
            )
        self.assertIn("not allowed", str(context.exception).lower())


class TestLoadStatesFromXarray(unittest.TestCase):
    """``load_states_from_xarray`` must always return a top-first state.

    Output files are surface-first (``jcm.cf_metadata``, #710); the loader
    detects that from the ``level`` coordinate values and flips back to the
    top-first physics frame (#741).
    """

    @staticmethod
    def _make_dataset(level_coord, nlev=4, nlon=3, nlat=2,
                      tracers=(), pressure_full=None):
        """Build a minimal state Dataset with a distinct value per level.

        ``temperature[k] == 200 + 10*k`` so the vertical profile is
        unambiguous and a flip is detectable. ``level_coord`` sets (or omits)
        the ``level`` coordinate; the *data* is written in whatever order
        matches ``level_coord``.

        ``tracers`` names extra level-dimensioned variables to add (each with
        its own distinct profile, so a mis-mapped tracer is detectable).
        ``pressure_full``, when given, is the per-level pressure in the same
        file order as the rest of the data — the loader cross-checks it
        against the state's own surface pressure (#718). Omitting it models a
        trimmed restart file, where that check cannot run.
        """
        prof = 200.0 + 10.0 * np.arange(nlev)  # increases with array index
        col = np.broadcast_to(prof[:, None, None], (nlev, nlon, nlat))
        surf = np.ones((nlon, nlat))
        data_vars = {
            "u_wind": (("level", "lon", "lat"), np.zeros((nlev, nlon, nlat))),
            "v_wind": (("level", "lon", "lat"), np.zeros((nlev, nlon, nlat))),
            "temperature": (("level", "lon", "lat"), col.copy()),
            "specific_humidity": (
                ("level", "lon", "lat"), np.zeros((nlev, nlon, nlat))),
            "geopotential": (
                ("level", "lon", "lat"), np.zeros((nlev, nlon, nlat))),
            "normalized_surface_pressure": (("lon", "lat"), surf),
        }
        for i, name in enumerate(tracers):
            tracer_prof = (i + 1) * 1e-4 * (1.0 + np.arange(nlev))
            data_vars[name] = (
                ("level", "lon", "lat"),
                np.broadcast_to(
                    tracer_prof[:, None, None], (nlev, nlon, nlat)).copy(),
            )
        if pressure_full is not None:
            data_vars["pressure_full"] = (
                ("level", "lon", "lat"),
                np.broadcast_to(
                    np.asarray(pressure_full, dtype=float)[:, None, None],
                    (nlev, nlon, nlat)).copy(),
            )
        coords = {} if level_coord is None else {"level": level_coord}
        return xr.Dataset(data_vars=data_vars, coords=coords)

    @staticmethod
    def _surface_first_level(nlev=4):
        """Descending sigma: the on-disk convention every JCM file uses."""
        return np.linspace(0.99, 0.01, nlev)

    def test_surface_first_file_is_flipped_to_top_first(self):
        # Descending sigma coordinate (index 0 ≈ surface) marks a
        # surface-first file; the loaded state must be reversed relative to it.
        nlev = 4
        level = np.linspace(0.99, 0.01, nlev)  # descending → surface-first
        ds = self._make_dataset(level, nlev=nlev)
        file_profile = ds["temperature"].values[:, 0, 0]

        state = load_states_from_xarray(ds)
        loaded_profile = np.asarray(state.temperature)[:, 0, 0]

        np.testing.assert_allclose(loaded_profile, file_profile[::-1])
        # Top-first contract: index 0 is now the model top (smallest value
        # here, since the file put the largest at index 0).
        self.assertEqual(loaded_profile[0], file_profile[-1])

    def test_top_first_file_passes_through_unchanged(self):
        # Ascending sigma coordinate is already top-first; no flip.
        nlev = 4
        level = np.linspace(0.01, 0.99, nlev)  # ascending → top-first
        ds = self._make_dataset(level, nlev=nlev)
        file_profile = ds["temperature"].values[:, 0, 0]

        state = load_states_from_xarray(ds)
        loaded_profile = np.asarray(state.temperature)[:, 0, 0]

        np.testing.assert_allclose(loaded_profile, file_profile)

    def test_bare_index_level_passes_through_and_warns(self):
        # No numeric sigma coordinate: orientation cannot be determined, so
        # the data passes through unflipped and a warning is emitted.
        ds = self._make_dataset(level_coord=None, nlev=4)
        file_profile = ds["temperature"].values[:, 0, 0]

        with self.assertLogs(level="WARNING") as cm:
            state = load_states_from_xarray(ds)
        loaded_profile = np.asarray(state.temperature)[:, 0, 0]

        np.testing.assert_allclose(loaded_profile, file_profile)
        self.assertTrue(
            any("orientation" in msg.lower() for msg in cm.output))


class TestLoadStatesTracerResolution(unittest.TestCase):
    """Which file variables become ``state.tracers`` (#718).

    A saved state carries its cloud condensate, but the loader used to drop
    it unless the caller named every variable by hand — so radiation driven
    on a saved state saw a clear sky and returned plausible, wrong fluxes.
    The names now come from what the configured physics declares.
    """

    _make_dataset = staticmethod(TestLoadStatesFromXarray._make_dataset)
    _surface_first_level = staticmethod(
        TestLoadStatesFromXarray._surface_first_level)

    def _dataset(self, nlev=4, tracers=("qc", "qi")):
        return self._make_dataset(
            self._surface_first_level(nlev), nlev=nlev, tracers=tracers)

    def test_declared_tracers_are_loaded_and_logged(self):
        # The default path: physics declares qc/qi, the file has them, so
        # they are loaded without the caller naming a single variable.
        from jcm.physics.physics_term import TracerSpec

        ds = self._dataset()
        with self.assertLogs(level="INFO") as cm:
            state = load_states_from_xarray(
                ds, required_tracers=(TracerSpec(name="qc"),
                                      TracerSpec(name="qi")),
            )

        self.assertEqual(set(state.tracers), {"qc", "qi"})
        # Tracers must be flipped with the rest of the state, not left in
        # file order — a tracer paired with the wrong level is the same bug
        # in a different variable.
        np.testing.assert_allclose(
            np.asarray(state.tracers["qc"])[:, 0, 0],
            ds["qc"].values[::-1, 0, 0])
        self.assertTrue(any("qc" in msg and "qi" in msg for msg in cm.output))

    def test_bare_tracer_names_are_accepted(self):
        # Callers without a physics object (a notebook, a test) may pass
        # names instead of TracerSpecs.
        state = load_states_from_xarray(
            self._dataset(), required_tracers=["qc", "qi"])
        self.assertEqual(set(state.tracers), {"qc", "qi"})

    def test_declared_tracer_absent_from_file_warns(self):
        # A dynamics-only state file is a legitimate starting point, so this
        # is a warning rather than an error -- but it must not be silent:
        # physics will run with that tracer at zero.
        ds = self._dataset(tracers=("qc",))
        with self.assertLogs(level="WARNING") as cm:
            state = load_states_from_xarray(
                ds, required_tracers=["qc", "qi"])

        self.assertEqual(set(state.tracers), {"qc"})
        self.assertTrue(any("qi" in msg for msg in cm.output))

    def test_explicit_mapping_wins_and_warns_about_drops(self):
        # An explicit mapping is honoured verbatim (it may rename), but
        # dropping a declared tracer the file actually carries is exactly the
        # silent failure #718 was about, so it is reported.
        ds = self._dataset()
        with self.assertLogs(level="WARNING") as cm:
            state = load_states_from_xarray(
                ds, tracer_vars={"qc": "qc"}, required_tracers=["qc", "qi"])

        self.assertEqual(set(state.tracers), {"qc"})
        self.assertTrue(any("qi" in msg for msg in cm.output))

    def test_empty_mapping_opts_out_of_inference(self):
        # ``{}`` is how a caller says "no tracers" and must not be treated as
        # "unset"; only the explicit-drop warning fires.
        with self.assertLogs(level="WARNING"):
            state = load_states_from_xarray(
                self._dataset(), tracer_vars={}, required_tracers=["qc", "qi"])
        self.assertEqual(state.tracers, {})

    def test_no_declaration_loads_no_tracers(self):
        # The file cannot say which of its variables are prognostic tracers,
        # so a caller that declares nothing gets nothing.
        state = load_states_from_xarray(self._dataset())
        self.assertEqual(state.tracers, {})


class TestLoadStatesPressureGuard(unittest.TestCase):
    """The loaded state must be top-first by its own pressures too (#718).

    Orientation is inferred from the ``level`` coordinate, which a file may
    not have; ``pressure_full`` is the independent check on that inference.
    """

    _make_dataset = staticmethod(TestLoadStatesFromXarray._make_dataset)
    _surface_first_level = staticmethod(
        TestLoadStatesFromXarray._surface_first_level)

    # Surface-first pressures for a 4-level column, ~1e5 Pa at the surface,
    # matching the factory's normalized_surface_pressure of 1.0 (p0 = 1e5).
    _SURFACE_FIRST_PRESSURE = np.array([9.9e4, 7.0e4, 4.0e4, 1.0e4])

    def test_consistent_surface_first_file_loads(self):
        ds = self._make_dataset(
            self._surface_first_level(), pressure_full=self._SURFACE_FIRST_PRESSURE)
        state = load_states_from_xarray(ds)
        # Flipped to top-first, so index 0 holds what the file's last index did.
        self.assertEqual(
            np.asarray(state.temperature)[0, 0, 0],
            ds["temperature"].values[-1, 0, 0])

    def test_inverted_column_is_rejected(self):
        # No usable ``level`` coordinate, so the loader assumes top-first and
        # does not flip -- but the pressures say the file is surface-first.
        # This is the #718 failure, and it must now be loud.
        ds = self._make_dataset(
            level_coord=None, pressure_full=self._SURFACE_FIRST_PRESSURE)
        with self.assertRaisesRegex(ValueError, "does not increase"):
            load_states_from_xarray(ds)

    def test_pressures_inconsistent_with_surface_pressure_are_rejected(self):
        # Monotonic the right way, but on a scale that cannot belong to this
        # state: the surface level is 1e-3 of the surface pressure carried in
        # the file.
        ds = self._make_dataset(
            self._surface_first_level(),
            pressure_full=np.array([100.0, 40.0, 10.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "factor of two"):
            load_states_from_xarray(ds)

    def test_file_without_pressures_falls_back_to_the_level_coordinate(self):
        # A trimmed restart file (only the prognostic fields) has no
        # ``pressure_full``; the level-coordinate inference still applies.
        ds = self._make_dataset(self._surface_first_level())
        state = load_states_from_xarray(ds)
        np.testing.assert_allclose(
            np.asarray(state.temperature)[:, 0, 0],
            ds["temperature"].values[::-1, 0, 0])


if __name__ == '__main__':
    unittest.main()
