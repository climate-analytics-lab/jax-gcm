"""Tests whether the shortwavewave optics data for atmospheric gases are loaded properly."""

from absl.testing import absltest
from etils import epath
import jax.numpy as jnp
import numpy as np

from jcm.rrtmgp.optics import cloud_optics
from jcm.rrtmgp.optics import lookup_cloud_optics

ROOT_DIR = epath.resource_path('jcm.rrtmgp.optics.rrtmgp_data')
_LW_LOOKUP_TABLE_FILENAME = 'cloudysky_sw.nc'
_LW_LOOKUP_TABLE_FILEPATH = ROOT_DIR / _LW_LOOKUP_TABLE_FILENAME


class CloudOpticsTest(absltest.TestCase):

	def test_compute_optical_properties(self):
		"""Checks the cloud optics calculations of optical depth, `ssa`, and `g`."""
		cloud_optics_lw = lookup_cloud_optics.from_nc_file(_LW_LOOKUP_TABLE_FILEPATH)
		ones_2d = jnp.ones((2, 2), dtype=jnp.float32)
		
		# Fixed spectral band index.
		ibnd = 5
		# Roughness index for medium roughness.
		rgh_idx = 1
		# Pick effective droplet radius halfway between the reference values at
		# index 8 (10.5e-6 m) and index 9 (11.5e-6 m).
		radius_eff_liq = 11e-6 * ones_2d
		# Pick effective ice particle radius halfway between the reference values at
		# index 9 (100e-6 m) and index 10 (110e-6 m).
		radius_eff_ice = 105e-6 / 2 * ones_2d
		# Cloud liquid path for an atmospheric grid cell in kg/m².
		cld_path_liq = 6e-4 * ones_2d
		# Cloud ice path for an atmospheric grid cell.
		cld_path_ice = 1.2e-3 * ones_2d
		
		# Lookup tables.
		ext_liq = cloud_optics_lw.ext_liq
		ssa_liq = cloud_optics_lw.ssa_liq
		asy_liq = cloud_optics_lw.asy_liq
		ext_ice = cloud_optics_lw.ext_ice
		ssa_ice = cloud_optics_lw.ssa_ice
		asy_ice = cloud_optics_lw.asy_ice
		
		# Compute arithmetic mean of liquid lookup values at indices 8 and 9 of effective radius dimension.
		interpolated_ext = (ext_liq[ibnd, 8] + ext_liq[ibnd, 9]) / 2.0
		interpolated_ssa = (ssa_liq[ibnd, 8] + ssa_liq[ibnd, 9]) / 2.0
		interpolated_g = (asy_liq[ibnd, 8] + asy_liq[ibnd, 9]) / 2.0
		
		# Expected values for liquid only. Use the cloud path in g/m² to scale the
		# table coefficients, which are in units of m²/g.
		expected_optical_depth_liq = 1000.0 * cld_path_liq * interpolated_ext * ones_2d
		expected_ssa_liq = interpolated_ssa * ones_2d
		expected_g_liq = interpolated_g * ones_2d
		
		# Compute arithmetic mean of ice lookup values at indices 9 and 10 of
		# effective radius dimension.
		interpolated_ext = (ext_ice[rgh_idx, ibnd, 9] + ext_ice[rgh_idx, ibnd, 10]) / 2.0
		interpolated_ssa = (ssa_ice[rgh_idx, ibnd, 9] + ssa_ice[rgh_idx, ibnd, 10]) / 2.0
		interpolated_g = (asy_ice[rgh_idx, ibnd, 9] + asy_ice[rgh_idx, ibnd, 10]) / 2.0
		
		# Expected values for ice only. Use the cloud path in g/m²
		expected_optical_depth_ice = 1000.0 * cld_path_ice * interpolated_ext * ones_2d
		expected_ssa_ice = interpolated_ssa * ones_2d
		expected_g_ice = interpolated_g * ones_2d
		
		# Cloud path that is too small to be considered. Note this is in SI units,
		# and only cloud paths that are greater than 1e-6 g/m² are accounted for.
		small_cld_path = 9.9e-10 * jnp.ones_like(cld_path_liq)
		
		rtol = 1e-5
		atol = 0

		with self.subTest('NonzeroLiquidCloudPath'):
			optical_props = cloud_optics.compute_optical_properties(
				cloud_optics_lw,
				cld_path_liq,
				small_cld_path,
				radius_eff_liq,
				radius_eff_ice,
				ibnd=ibnd,
			)
			
			np.testing.assert_allclose(
				expected_optical_depth_liq, optical_props['optical_depth'], rtol, atol
			)
			np.testing.assert_allclose(
				expected_ssa_liq, optical_props['ssa'], rtol, atol
			)
			np.testing.assert_allclose(
				expected_g_liq, optical_props['asymmetry_factor'], rtol, atol
			)

		with self.subTest('NonzeroIceCloudPath'):
			optical_props = cloud_optics.compute_optical_properties(
				cloud_optics_lw,
				small_cld_path,
				cld_path_ice,
				radius_eff_liq,
				radius_eff_ice,
				ibnd=ibnd,
			)

			np.testing.assert_allclose(
				expected_optical_depth_ice, optical_props['optical_depth'], rtol, atol
			)
			np.testing.assert_allclose(
				expected_ssa_ice, optical_props['ssa'], rtol, atol
			)
			np.testing.assert_allclose(
				expected_g_ice, optical_props['asymmetry_factor'], rtol, atol
			)

		with self.subTest('NonzeroTwoPhaseCloudPath'):
			optical_props = cloud_optics.compute_optical_properties(
				cloud_optics_lw,
				cld_path_liq,
				cld_path_ice,
				radius_eff_liq,
				radius_eff_ice,
				ibnd=ibnd,
			)
			
			expected_optical_depth = expected_optical_depth_liq + expected_optical_depth_ice
			weighted_ssa = (
				expected_optical_depth_liq * expected_ssa_liq
				+ expected_optical_depth_ice * expected_ssa_ice
			)
			expected_ssa = weighted_ssa / expected_optical_depth
			expected_g = (
				expected_optical_depth_liq * expected_ssa_liq * expected_g_liq
				+ expected_optical_depth_ice * expected_ssa_ice * expected_g_ice
			) / weighted_ssa
			
			np.testing.assert_allclose(
				expected_optical_depth * ones_2d,
				optical_props['optical_depth'],
				rtol,
				atol,
			)
			np.testing.assert_allclose(
				expected_ssa * ones_2d, optical_props['ssa'], rtol, atol
			)
			np.testing.assert_allclose(
				expected_g * ones_2d, optical_props['asymmetry_factor'], rtol, atol
			)


if __name__ == '__main__':
	absltest.main()
