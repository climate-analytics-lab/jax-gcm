"""Tests whether the shortwavewave optics data for atmospheric gases are loaded properly."""

import functools
from typing import Literal, TypeAlias
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from jcm.rrtmgp.config import radiative_transfer
from jcm.rrtmgp.optics import cloud_optics
from jcm.rrtmgp.optics import gas_optics
from jcm.rrtmgp.optics import lookup_gas_optics_longwave
from jcm.rrtmgp.optics import lookup_volume_mixing_ratio
from jcm.rrtmgp.optics import optics
from jcm.rrtmgp.optics import optics_base

Array: TypeAlias = jax.Array
LookupGasOpticsLongwave: TypeAlias = (lookup_gas_optics_longwave.LookupGasOpticsLongwave)
LookupVolumeMixingRatio: TypeAlias = (lookup_volume_mixing_ratio.LookupVolumeMixingRatio)

ROOT_RRTMGP_DIR = epath.resource_path('jcm.rrtmgp.optics.rrtmgp_data')
ROOT_TEST_DIR = epath.resource_path('jcm.rrtmgp.optics.test_data')

_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp-gas-sw-g224.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'cloudysky_sw.nc'
_GLOBAL_MEANS_FILENAME = 'vmr_global_means.json'

_LW_LOOKUP_TABLE_FILEPATH = ROOT_RRTMGP_DIR / _LW_LOOKUP_TABLE_FILENAME
_SW_LOOKUP_TABLE_FILEPATH = ROOT_RRTMGP_DIR / _SW_LOOKUP_TABLE_FILENAME
_CLD_LW_LOOKUP_TABLE_FILEPATH = ROOT_RRTMGP_DIR / _CLD_LW_LOOKUP_TABLE_FILENAME
_CLD_SW_LOOKUP_TABLE_FILEPATH = ROOT_RRTMGP_DIR / _CLD_SW_LOOKUP_TABLE_FILENAME
_GLOBAL_MEANS_FILEPATH = ROOT_TEST_DIR / _GLOBAL_MEANS_FILENAME

radiation_params_gray = radiative_transfer.OpticsParameters(
	optics=radiative_transfer.GrayAtmosphereOptics(
		p0=1e5, alpha=3.5, d0_lw=5.5536, d0_sw=0.22
	)
)


def convert_to_3d_array_and_tile(
    f_1d: Array, dim: Literal[0, 1, 2], num_repeats: int
) -> Array:
	"""Converts 1D array `f_1d` to a tiled 3D array.
	
	For example, if len(f_1d) == 8, dim == 1, and num_repeats == 4, then
	f.shape = (4, 8, 4).
	
	Args:
	f_1d: The 1D tensor to convert to 3D.
	dim: The dimension along which `f_1d` is laid out.
	num_repeats: The number of times to repeat the array in dimensions other
	  than `dim`.
	
	Returns:
	The 3D tensor `f_3d`, where f_3d.shape[dim] == len(f_1d), and
	  f_3d.shape[j] == num_repeats for j != dim.
	"""
	# Convert f_1d to 3D tensor f where the direction of variation is along
	# dim. Result: f.shape[dim] = len(f_1d), and f.shape[j] = 1 for j != dim.
	slices = [jnp.newaxis, jnp.newaxis, jnp.newaxis]
	slices[dim] = slice(None)
	f = f_1d[tuple(slices)]
	
	# Tile f to a 3D tensor that is repeated in other dimensions.
	repeats = [num_repeats, num_repeats, num_repeats]
	repeats[dim] = 1
	f_3d = jnp.tile(f, reps=repeats)
	return f_3d
	
def _strip_top(f: Array) -> Array:
	"""Remove the topmost plane."""
	return f[:, :, :-1]


def _strip_bottom(f: Array) -> Array:
	"""Remove the bottommost plane."""
	return f[:, :, 1:]


def _remove_halos(f: Array) -> Array:
	"""Remove the halos from the output."""
	return f[:, :, 1:-1]


class RRTMOpticsTest(parameterized.TestCase):

	def setUp(self):
		super().setUp()
		radiation_params_rrtm = radiative_transfer.OpticsParameters(
			optics=radiative_transfer.RRTMOptics(
				longwave_nc_filepath=_LW_LOOKUP_TABLE_FILEPATH,
				shortwave_nc_filepath=_SW_LOOKUP_TABLE_FILEPATH,
				cloud_longwave_nc_filepath=_CLD_LW_LOOKUP_TABLE_FILEPATH,
				cloud_shortwave_nc_filepath=_CLD_SW_LOOKUP_TABLE_FILEPATH,
			)
		)
		atmospheric_state_cfg = radiative_transfer.AtmosphericStateCfg(
			vmr_global_mean_filepath=_GLOBAL_MEANS_FILEPATH
		)
		self.vmr_lib = lookup_volume_mixing_ratio.from_config(atmospheric_state_cfg)
		self.rrtm_lib = optics.RRTMOptics(self.vmr_lib, radiation_params_rrtm)
		
		# Create mock functions for use in tests.
		self.mock_major_optical_depth_fn = self.enter_context(
			mock.patch.object(
				gas_optics, 'compute_major_optical_depth', autospec=True
			)
		)
		self.mock_minor_optical_depth_fn = self.enter_context(
			mock.patch.object(
				gas_optics, 'compute_minor_optical_depth', autospec=True
			)
		)
		self.mock_rayleigh_optical_depth_fn = self.enter_context(
			mock.patch.object(
				gas_optics, 'compute_rayleigh_optical_depth', autospec=True
			)
		)
		self.mock_planck_fraction_fn = self.enter_context(
			mock.patch.object(gas_optics, 'compute_planck_fraction', autospec=True)
		)
		self.mock_planck_source_fn = self.enter_context(
			mock.patch.object(gas_optics, 'compute_planck_sources', autospec=True)
		)
		self.mock_cloud_optical_props_fn = self.enter_context(
			mock.patch.object(
				cloud_optics, 'compute_optical_properties', autospec=True
			)
		)
		# End mock functions

	def test_reconstruct_face_values(self):
		# SETUP
		n_horiz = 5
		nz = 18  # 16 layers + halo_width of 1.
		
		# Create a linear temperature profile = [299, 298, 297, ..., 282]
		temperature = jnp.arange(299, 281, -1, dtype=jnp.float32)
		self.assertLen(temperature, nz)
		
		# Convert from 1D to 3D array.
		temperature = convert_to_3d_array_and_tile(
			temperature, dim=2, num_repeats=n_horiz
		)
		
		# ACTION
		f_lower_bc = 299.0 * jnp.ones((n_horiz, n_horiz), dtype=jnp.float32)
		temperature_bottom, temperature_top = (
			optics_base.reconstruct_face_values(temperature, f_lower_bc)
		)
		
		# VERIFICATION
		expected_temperature_bottom = temperature + 0.5
		expected_temperature_top = temperature - 0.5
		
		# Remove halos before comparing. Also remove one layer of interior...
		temperature_bottom = temperature_bottom[:, :, 2:-2]
		temperature_top = temperature_top[:, :, 2:-2]
		expected_temperature_bottom = expected_temperature_bottom[:, :, 2:-2]
		expected_temperature_top = expected_temperature_top[:, :, 2:-2]
		
		np.testing.assert_allclose(
			temperature_top, expected_temperature_top, rtol=2e-4, atol=0
		)
		np.testing.assert_allclose(
			temperature_bottom, expected_temperature_bottom, rtol=2e-4, atol=0
		)

	def test_compute_lw_optical_properties_rrtm(self):
		"""Check the computed lw optical depth, albedo, and asymmetry factor."""
		# SETUP
		n = 4
		ones = jnp.ones((n, n, n), dtype=jnp.float32)
		
		self.mock_major_optical_depth_fn.return_value = 0.5 * ones
		self.mock_minor_optical_depth_fn.return_value = 0.2 * ones
		
		pressure = 1e5 * jnp.ones((n, n, n), dtype=jnp.float32)
		temperature = 290.0 * jnp.ones_like(pressure)
		molecules = 1e24 * jnp.ones_like(pressure)
		
		# ACTION
		output = self.rrtm_lib.compute_lw_optical_properties(
			pressure, temperature, molecules, igpt=1
		)
		
		# VERIFICATION
		expected_lw_optical_depth = 0.7 * jnp.ones_like(pressure)
		expected_ssa = jnp.zeros_like(pressure)
		expected_g = jnp.zeros_like(pressure)
		
		np.testing.assert_allclose(
			output['optical_depth'], expected_lw_optical_depth, rtol=1e-5, atol=0
		)
		np.testing.assert_allclose(output['ssa'], expected_ssa, rtol=1e-5, atol=0)
		np.testing.assert_allclose(
			output['asymmetry_factor'], expected_g, rtol=1e-5, atol=0
		)

	def test_compute_cloud_optical_properties_rrtm_lw(self):
		"""Check the lw optical depth, albedo, and asymmetry factor with clouds."""
		# SETUP
		n = 4
		ones = jnp.ones((n, n, n), dtype=jnp.float32)
		
		self.mock_major_optical_depth_fn.return_value = 0.5 * ones
		self.mock_minor_optical_depth_fn.return_value = 0.2 * ones
		self.mock_cloud_optical_props_fn.return_value = {
			'optical_depth': 0.35 * ones,
			'ssa': 0.24 * ones,
			'asymmetry_factor': 0.12 * ones,
		}
		
		pressure = 1e5 * jnp.ones((n, n, n), dtype=jnp.float32)
		temperature = 290.0 * jnp.ones_like(pressure)
		molecules = 1e24 * jnp.ones_like(pressure)
		cloud_r_eff_liq = 1e-5 * jnp.ones_like(pressure)
		cloud_path_liq = 0.1 * jnp.ones_like(pressure)
		cloud_r_eff_ice = 1e-3 * jnp.ones_like(pressure)
		cloud_path_ice = 0.1 * jnp.ones_like(pressure)
		
		# ACTION
		output = self.rrtm_lib.compute_lw_optical_properties(
			pressure,
			temperature,
			molecules,
			igpt=1,
			cloud_r_eff_liq=cloud_r_eff_liq,
			cloud_path_liq=cloud_path_liq,
			cloud_r_eff_ice=cloud_r_eff_ice,
			cloud_path_ice=cloud_path_ice,
		)
		
		# VERIFICATION
		expected_lw_optical_depth = 1.05 * jnp.ones_like(pressure)
		expected_ssa = 0.24 * (0.35 / 1.05) * jnp.ones_like(pressure)
		expected_asy = 0.12 * jnp.ones_like(pressure)
		
		np.testing.assert_allclose(
			output['optical_depth'], expected_lw_optical_depth, rtol=1e-5, atol=0
		)
		np.testing.assert_allclose(output['ssa'], expected_ssa, rtol=1e-5, atol=0)
		np.testing.assert_allclose(
			output['asymmetry_factor'], expected_asy, rtol=1e-5, atol=0
		)

	def test_compute_sw_optical_properties_rrtm(self):
		"""Checks the computed sw optical depth, albedo, and asymmetry factor."""
		# SETUP
		n = 4
		ones = jnp.ones((n, n, n), dtype=jnp.float32)
		
		self.mock_rayleigh_optical_depth_fn.return_value = 0.1 * ones
		self.mock_major_optical_depth_fn.return_value = 0.6 * ones
		self.mock_minor_optical_depth_fn.return_value = 0.24 * ones
		
		pressure = 1e5 * ones
		temperature = 290.0 * jnp.ones_like(pressure)
		molecules = 1e24 * jnp.ones_like(pressure)
		
		# ACTION
		output = self.rrtm_lib.compute_sw_optical_properties(
			pressure, temperature, molecules, igpt=1
		)
		
		# VERIFICATION
		expected_sw_optical_depth = 0.94 * jnp.ones_like(pressure)
		expected_ssa = 0.1 / 0.94 * jnp.ones_like(pressure)
		expected_g = jnp.zeros_like(pressure)
		
		np.testing.assert_allclose(
			output['optical_depth'], expected_sw_optical_depth, rtol=1e-5, atol=0
		)
		np.testing.assert_allclose(output['ssa'], expected_ssa, rtol=1e-5, atol=0)
		np.testing.assert_allclose(
			output['asymmetry_factor'], expected_g, rtol=1e-5, atol=0
		)
		
		# SETUP
		# Make the Rayleigh scattering 0 and check the single-scattering albedo.
		self.mock_rayleigh_optical_depth_fn.return_value = 0.0 * ones
		self.mock_minor_optical_depth_fn.return_value = 0.0 * ones
		# ACTION
		output = self.rrtm_lib.compute_sw_optical_properties(
			pressure, temperature, molecules, igpt=1
		)
		# VERIFICATION
		# The division by zero should make the ssa default to 0.
		np.testing.assert_array_equal(output['ssa'], jnp.zeros_like(pressure))

	def test_compute_cloud_optical_properties_rrtm_sw(self):
		"""Checks the sw optical depth, albedo, and asymmetry factor with clouds."""
		# SETUP
		n = 4
		ones = jnp.ones((n, n, n), dtype=jnp.float32)
		
		self.mock_rayleigh_optical_depth_fn.return_value = 0.1 * ones
		self.mock_major_optical_depth_fn.return_value = 0.6 * ones
		self.mock_minor_optical_depth_fn.return_value = 0.24 * ones
		self.mock_cloud_optical_props_fn.return_value = {
			'optical_depth': 0.12 * ones,
			'ssa': 0.15 * ones,
			'asymmetry_factor': 0.1 * ones,
		}
		
		pressure = 1e5 * ones
		temperature = 290.0 * jnp.ones_like(pressure)
		molecules = 1e24 * jnp.ones_like(pressure)
		cloud_r_eff_liq = 1e-5 * jnp.ones_like(pressure)
		cloud_path_liq = 0.1 * jnp.ones_like(pressure)
		cloud_r_eff_ice = 1e-3 * jnp.ones_like(pressure)
		cloud_path_ice = 0.1 * jnp.ones_like(pressure)
		
		# ACTION
		output = self.rrtm_lib.compute_sw_optical_properties(
			pressure,
			temperature,
			molecules,
			igpt=1,
			cloud_r_eff_liq=cloud_r_eff_liq,
			cloud_path_liq=cloud_path_liq,
			cloud_r_eff_ice=cloud_r_eff_ice,
			cloud_path_ice=cloud_path_ice,
		)
		
		# VERIFICATION
		expected_sw_optical_depth = 1.05982 * jnp.ones_like(pressure)
		expected_ssa = 0.111170 * jnp.ones_like(pressure)
		expected_asy = 0.0137498 * jnp.ones_like(pressure)
		
		np.testing.assert_allclose(
			output['optical_depth'], expected_sw_optical_depth, rtol=1e-5, atol=0
		)
		np.testing.assert_allclose(output['ssa'], expected_ssa, rtol=1e-5, atol=0)
		np.testing.assert_allclose(
			output['asymmetry_factor'], expected_asy, rtol=1e-5, atol=0
		)
		
		# SETUP
		# Make the Rayleigh scattering 0 and cloud ssa 0 and check the
		# single-scattering albedo.
		self.mock_rayleigh_optical_depth_fn.return_value = jnp.zeros_like(ones)
		self.mock_cloud_optical_props_fn.return_value['ssa'] = jnp.zeros_like(ones)
		# ACTION
		output = self.rrtm_lib.compute_sw_optical_properties(
			pressure,
			temperature,
			molecules,
			igpt=1,
			cloud_r_eff_liq=cloud_r_eff_liq,
			cloud_path_liq=cloud_path_liq,
			cloud_r_eff_ice=cloud_r_eff_ice,
			cloud_path_ice=cloud_path_ice,
		)
		# VERIFICATION
		# The division by zero should make the ssa default to 0.
		np.testing.assert_allclose(output['ssa'], jnp.zeros_like(pressure))

	def test_compute_planck_sources_rrtm(self):
		"""Checks the computed Planck sources at cell center and face."""
		# SETUP
		n = 4
		
		def mock_planck_fraction_fn(
			optics_lib, vmr_lib, pressure, temperature, igpt, vmr_fields
		):
			del optics_lib, vmr_lib, igpt
			return pressure * 1e-6 + temperature * 1e-3 + vmr_fields[1]
		
		def mock_planck_source_fn(optics_lib, planck_fraction, temperature, igpt):
			del optics_lib, igpt
			return 1e-2 * temperature * planck_fraction
		
		self.mock_planck_fraction_fn.side_effect = mock_planck_fraction_fn
		self.mock_planck_source_fn.side_effect = mock_planck_source_fn
		
		temperature = jnp.array(
			[430.0, 400.0, 370.0, 340.0, 310.0, 280.0, 250.0, 220.0, 190.0],
			dtype=jnp.float32,
		)
		temperature = convert_to_3d_array_and_tile(
			temperature, dim=2, num_repeats=n
		)
		pressure = 1e5 * jnp.ones_like(temperature)
		vmr_fields = {1: 1.2e-3 * jnp.ones_like(temperature)}
		
		sfc_temperature = 440.0 * jnp.ones((n, n), dtype=jnp.float32)
		
		# ACTION
		output = self.rrtm_lib.compute_planck_sources(
			pressure,
			temperature,
			igpt=100,
			vmr_fields=vmr_fields,
			sfc_temperature=sfc_temperature,
		)
		
		# VERIFICATION
		# Extract the output plane at layer index 4 with local temperature 310 K,
		# skipping the surface source.
		planck_src_layer1 = output['planck_src'][:, :, 4]
		planck_src_top_layer1 = output['planck_src_top'][:, :, 4]
		planck_src_bottom_layer1 = output['planck_src_bottom'][:, :, 4]
		
		ones_2d = np.ones((n, n))
		
		# Planck source corresponding to temperature 310 K.
		np.testing.assert_allclose(
			planck_src_layer1, 1.27472 * ones_2d, rtol=1e-5, atol=0
		)
		
		# Planck source corresponding to temperature 295 K at the top half-level.
		np.testing.assert_allclose(
			planck_src_top_layer1, 1.21304 * ones_2d, rtol=1e-5, atol=0
		)
		
		# Planck source corresponding to temperature 325 K at the bottom face.
		np.testing.assert_allclose(
			planck_src_bottom_layer1, 1.3364 * ones_2d, rtol=1e-5, atol=0
		)
		
		# Planck source corresponding to temperature 350 K at the surface.
		# Commenting out test because we don't have a value to check against.
		# This depends on the exact interpolation used at the boundary, which is
		# not well unique, and is different here from SwirlLM.
		# np.testing.assert_allclose(
		#     output['planck_src_sfc'], 1.3342 * ones_2d, rtol=1e-5, atol=0
		# )

	def test_n_gpt(self):
		"""Test that spectral dimensions are correctly set."""
		self.assertEqual(self.rrtm_lib.n_gpt_lw, 256)
		self.assertEqual(self.rrtm_lib.n_gpt_sw, 224)


class GrayAtmosphereOpticsTest(parameterized.TestCase):

	def test_compute_optical_properties_gray_atmosphere(self):
		"""Checks gray atmosphere optical depth, albedo, and asymmetry factor."""
		# SETUP
		n = 4
		# Create a pressure profile.  Note that the pressure profile is linear and
		# dp across a grid cell is 200 Pa.
		pressure = jnp.array([
			100000.0, 99800.0, 99600.0, 99400.0,
			99200.0, 99000.0, 98800.0, 98600.0,
		])  # pyformat: disable
		# Convert from 1D to 3D arrays.
		convert_to_3d = functools.partial(
			convert_to_3d_array_and_tile, dim=2, num_repeats=n
		)
		pressure = convert_to_3d(pressure)
		
		# Using gray atmosphere parameters:
		# {'p0': 1e5, 'alpha': 3.5, 'd0_lw': 5.5536, 'd0_sw': 0.22}.
		gray_atmosphere = optics.GrayAtmosphereOptics(radiation_params_gray)
		
		# ACTION
		lw_output = gray_atmosphere.compute_lw_optical_properties(pressure)
		sw_output = gray_atmosphere.compute_sw_optical_properties(pressure)
		
		# VERIFICATION
		expected_lw_optical_depth = jnp.array([
			0.0388752, 0.03868112, 0.03848761, 0.0382947,
			0.03810235, 0.0379106, 0.03771941, 0.03752882,
		])  # pyformat: disable
		expected_sw_optical_depth = jnp.array([
			0.00088, 0.00087824, 0.00087648, 0.00087472,
			0.00087296, 0.0008712, 0.00086944, 0.00086768,
		])  # pyformat: disable
		# Convert from 1D to 3D arrays.
		expected_lw_optical_depth = convert_to_3d(expected_lw_optical_depth)
		expected_sw_optical_depth = convert_to_3d(expected_sw_optical_depth)
		
		expected_ssa = jnp.zeros_like(expected_lw_optical_depth)
		expected_g = jnp.zeros_like(expected_lw_optical_depth)
		
		# strip_outermost = self.strip_outermost_fn[g_dim]
		with self.subTest('OpticalDepthLW'):
			np.testing.assert_allclose(
				_remove_halos(lw_output['optical_depth']),
				_remove_halos(expected_lw_optical_depth),
				rtol=1e-5,
				atol=0,
			)
		
		with self.subTest('OpticalDepthSW'):
			np.testing.assert_allclose(
				_remove_halos(sw_output['optical_depth']),
				_remove_halos(expected_sw_optical_depth),
				rtol=1e-5,
				atol=0,
			)
		
		with self.subTest('SSA'):
			np.testing.assert_allclose(
				lw_output['ssa'], expected_ssa, rtol=1e-5, atol=0
			)
			np.testing.assert_allclose(
				sw_output['ssa'], expected_ssa, rtol=1e-5, atol=0
			)
		
		with self.subTest('AsymmetryFactor'):
			np.testing.assert_allclose(
				lw_output['asymmetry_factor'], expected_g, rtol=1e-5, atol=0
			)
			np.testing.assert_allclose(
				sw_output['asymmetry_factor'], expected_g, rtol=1e-5, atol=0
			)

	def test_compute_planck_sources_gray_atmosphere(self):
		"""Checks the Planck source computations for a gray atmosphere."""
		# SETUP
		nx = ny = n = 4
		temperature = jnp.array(
			[430.0, 400.0, 370.0, 340.0, 310.0, 280.0, 250.0, 220.0, 190.0],
			dtype=jnp.float32,
		)
		
		# Convert from 1D to 3D array.
		temperature = convert_to_3d_array_and_tile(
			temperature, dim=2, num_repeats=n
		)
		
		sfc_temperature = 350.0 * jnp.ones((nx, ny), dtype=jnp.float32)
		
		# Using gray atmosphere parameters:
		# {'p0': 1e5, 'alpha': 3.5, 'd0_lw': 5.5536, 'd0_sw': 0.22}.
		gray_atmosphere = optics.GrayAtmosphereOptics(radiation_params_gray)
		
		# ACTION
		pressure = jnp.zeros_like(temperature)  # Unused in the function.
		output = gray_atmosphere.compute_planck_sources(
			pressure, temperature, sfc_temperature=sfc_temperature
		)
		
		# VERIFICATION
		# Extract the output plane at layer index 4 with local temperature 310 K,
		# skipping the surface source.
		planck_src_layer1 = output['planck_src'][:, :, 4]
		planck_src_top_layer1 = output['planck_src_top'][:, :, 4]
		planck_src_bottom_layer1 = output['planck_src_bottom'][:, :, 4]
		
		ones_2d = np.ones((nx, ny))
		
		with self.subTest('PlanckSrc'):
			# Planck source corresponding to temperature 310 K.
			np.testing.assert_allclose(
				planck_src_layer1, 166.6786451 * ones_2d, rtol=1e-5, atol=0
			)
		
		with self.subTest('PlanckSrcTop'):
			# Planck source corresponding to temperature 295 K at the top half-level.
			np.testing.assert_allclose(
				planck_src_top_layer1, 136.6851237 * ones_2d, rtol=1e-5, atol=0
			)
		
		with self.subTest('PlanckSrcBottom'):
			# Planck source corresponding to temperature 325 K at the bottom face.
			np.testing.assert_allclose(
				planck_src_bottom_layer1, 201.3569527 * ones_2d, rtol=1e-5, atol=0
			)
		
		with self.subTest('PlanckSrcSurface'):
			# Planck source corresponding to temperature 350 K at the surface.
			np.testing.assert_allclose(
				output['planck_src_sfc'], 270.83536 * ones_2d, rtol=1e-5, atol=0
			)
	
	def test_n_gpt(self):
		"""Test that spectral dimensions are correctly set."""
		gray_atmosphere = optics.GrayAtmosphereOptics(radiation_params_gray)
		self.assertEqual(gray_atmosphere.n_gpt_lw, 1)
		self.assertEqual(gray_atmosphere.n_gpt_sw, 1)


if __name__ == '__main__':
	absltest.main()
