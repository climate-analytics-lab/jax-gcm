"""Tests whether the atmospheric conditions are loaded properly from a proto."""

from typing import TypeAlias
from absl.testing import absltest
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from jcm.rrtmgp.config import radiative_transfer
from jcm.rrtmgp.optics import gas_optics
from jcm.rrtmgp.optics import lookup_gas_optics_longwave
from jcm.rrtmgp.optics import lookup_gas_optics_shortwave
from jcm.rrtmgp.optics import lookup_volume_mixing_ratio
from jcm.rrtmgp.optics import optics_utils

Array: TypeAlias = jax.Array
IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight
Interpolant: TypeAlias = optics_utils.Interpolant

ROOT_RRTMGP_DIR = epath.resource_path('jcm.rrtmgp.optics.rrtmgp_data')
ROOT_TEST_DIR = epath.resource_path('jcm.rrtmgp.optics.test_data')
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp-gas-sw-g224.nc'
_GLOBAL_MEANS_FILENAME = 'vmr_global_means.json'


_LW_LOOKUP_TABLE_FILEPATH = ROOT_RRTMGP_DIR / _LW_LOOKUP_TABLE_FILENAME
_SW_LOOKUP_TABLE_FILEPATH = ROOT_RRTMGP_DIR / _SW_LOOKUP_TABLE_FILENAME
_GLOBAL_MEANS_FILEPATH = ROOT_TEST_DIR / _GLOBAL_MEANS_FILENAME


def assert_interpolant_allclose(i1: Interpolant, i2: Interpolant):
	rtol = 1e-5
	atol = 1e-6
	np.testing.assert_allclose(i1.interp_low.idx, i2.interp_low.idx, rtol, atol)
	np.testing.assert_allclose(i1.interp_low.weight, i2.interp_low.weight, rtol, atol)
	np.testing.assert_allclose(i1.interp_high.idx, i2.interp_high.idx, rtol, atol)
	np.testing.assert_allclose(i1.interp_high.weight, i2.interp_high.weight, rtol, atol)


class GasOpticsTest(absltest.TestCase):

	def setUp(self):
		super(GasOpticsTest, self).setUp()
		self.gas_optics_lw = lookup_gas_optics_longwave.from_nc_file(_LW_LOOKUP_TABLE_FILEPATH)
		self.gas_optics_sw = lookup_gas_optics_shortwave.from_nc_file(_SW_LOOKUP_TABLE_FILEPATH)
		atmospheric_state_cfg = radiative_transfer.AtmosphericStateCfg(vmr_global_mean_filepath=_GLOBAL_MEANS_FILEPATH)
		self.vmr_lib = lookup_volume_mixing_ratio.from_config(atmospheric_state_cfg)

	def test_get_vmr(self):
		"""Tests that correct variable and global mean vmr's are returned."""
		major_species_idx_h20 = jnp.ones((1, 4), dtype=jnp.int32)
		major_species_idx_co2 = 2 * jnp.ones((1, 4), dtype=jnp.int32)
		major_species_idx_o3 = 3 * jnp.ones((1, 4), dtype=jnp.int32)
		major_species_idx = jnp.concatenate(
			[major_species_idx_h20, major_species_idx_co2, major_species_idx_o3],
			axis=0,
		)
		precomputed_vmr_h2o = jnp.array(
			[[3.7729078e-06, 1.61512e-05, 0.00273486, 0.018282978]],
			dtype=jnp.float32,
		)
		precomputed_vmr_o3 = jnp.array(
			[[1.9249276e-06, 4.4498346e-08, 4.7968513e-08, 3.5275427e-08]]
		)
		vmr_fields = {
			self.gas_optics_lw.idx_h2o: precomputed_vmr_h2o,
			self.gas_optics_lw.idx_o3: precomputed_vmr_o3,
		}
		
		# A global mean is used for CO2, so vmr does not change with pressure level.
		expected_vmr_co2 = 3.9754697e-4 * jnp.ones((1, 4), dtype=jnp.float32)
		expected_vmr = jnp.concatenate(
			[precomputed_vmr_h2o, expected_vmr_co2, precomputed_vmr_o3], axis=0
		)

		with self.subTest('VMRWithinRange'):
			vmr = gas_optics.get_vmr(
				self.gas_optics_lw, self.vmr_lib, major_species_idx, vmr_fields
			)
			
			np.testing.assert_allclose(vmr, expected_vmr, rtol=1e-5, atol=1e-6)

	def test_compute_relative_abundance_interpolant(self):
		"""Tests that the correct relative abundance interpolants are computed."""
		troposphere_idx = jnp.ones((3, 4), dtype=jnp.int32)
		temperature_idx = 10 * jnp.ones((3, 4), dtype=jnp.int32)
		ibnd = 4
		relative_abundance_interp = (
			gas_optics._compute_relative_abundance_interpolant(
				self.gas_optics_lw,
				self.vmr_lib,
				troposphere_idx,
				temperature_idx,
				ibnd,
				True,
			)
		)
		idx_low = jnp.zeros((3, 4), dtype=jnp.int32)
		idx_high = jnp.ones((3, 4), dtype=jnp.int32)
		weight_low = 5.2951004292976034e-06 * jnp.ones((3, 4), dtype=jnp.float32)
		weight_high = 3.5275427000000043e-07 * jnp.ones((3, 4), dtype=jnp.float32)
		idx_and_weight_low = IndexAndWeight(idx_low, weight_low)
		idx_and_weight_high = IndexAndWeight(idx_high, weight_high)
		expected_interp = Interpolant(idx_and_weight_low, idx_and_weight_high)
		assert_interpolant_allclose(relative_abundance_interp, expected_interp)

	def test_compute_major_optical_depth(self):
		"""Checks the optical depth computation for different values of t and p."""
		temperature = jnp.array(
			[[160.0, 200.0, 300.0], [280.0, 290.0, 355.0]], dtype=jnp.float32
		)
		pressure = jnp.array(
			[[1.09663316e5, 90000.0, 80000.0], [7.35095189e04, 30000.0, 1.00518357]], dtype=jnp.float32,
		)
		molecules = jnp.array([[1e24]], dtype=jnp.float32)
		major_optical_depth = gas_optics.compute_major_optical_depth(
			self.gas_optics_lw, self.vmr_lib, molecules, temperature, pressure, 70
		)
		expected_major_optical_depth = jnp.array(
			[
				[1.071459e-5, 2.313674e-5, 1.053062e-4],
				[7.901242e-5, 4.897409e-5, 1.766592e-7],
			],
			dtype=jnp.float32,
		)
		np.testing.assert_allclose(
			expected_major_optical_depth, major_optical_depth, rtol=1e-5, atol=1e-6
		)

	def test_compute_minor_optical_depth(self):
		"""Checks the minor optical depth computation for a particular g-point."""
		# Temperature corresponding to the 10th reference point.
		temperature = jnp.array([[310.0]], dtype=jnp.float32)
		p = jnp.array([[73509.51892419]], dtype=jnp.float32)
		molecules = jnp.array([[1e24]], dtype=jnp.float32)
		
		with self.subTest('PrecomputedVmrH2O'):
			# Precomputed VMR for H2O.
			vmr_fields = {self.gas_optics_lw.idx_h2o: jnp.array([[1.2e-3]], dtype=jnp.float32)}
			minor_optical_depth = gas_optics.compute_minor_optical_depth(
				self.gas_optics_lw,
				self.vmr_lib,
				molecules,
				temperature,
				p,
				100,
				vmr_fields
			)
			np.testing.assert_allclose(
				minor_optical_depth, [[3.755038e-7]], rtol=1e-5, atol=1e-12
			)
		
		with self.subTest('AbsentH2OVmr'):
			vmr_fields = {
				self.gas_optics_lw.idx_o3: jnp.array([[1.2e-3]], dtype=jnp.float32),
			}
			minor_optical_depth = gas_optics.compute_minor_optical_depth(
				self.gas_optics_lw,
				self.vmr_lib,
				molecules,
				temperature,
				p,
				100,
				vmr_fields
			)
			np.testing.assert_allclose([[1.768588e-7]], minor_optical_depth, rtol=1e-5, atol=1e-12)

	def test_compute_rayleigh_optical_depth(self):
		"""Checks the Rayleigh scattering contribution for a particular g-point."""
		# Temperature corresponding to the 10th reference point.
		temperature = jnp.array([[310.0]], dtype=jnp.float32)
		p = jnp.array([[1e5]], dtype=jnp.float32)
		molecules = jnp.array([[1e24]], dtype=jnp.float32)
		# Precomputed VMR for H2O.
		vmr_fields = {self.gas_optics_lw.idx_h2o: jnp.array([[1.2e-5]], dtype=jnp.float32)}
		
		with self.subTest('PrecomputedVmrH2O'):
			expected_minor_optical_depth = jnp.array([[9.647629e-9]], dtype=jnp.float32)
			
			minor_optical_depth = gas_optics.compute_rayleigh_optical_depth(
				self.gas_optics_sw,
				self.vmr_lib,
				molecules,
				temperature,
				p,
				100,
				vmr_fields
			)
			
			np.testing.assert_allclose(
				minor_optical_depth,
				expected_minor_optical_depth,
				rtol=1e-5,
				atol=1e-14,
			)
		
		with self.subTest('AbsentVMRFields'):
			expected_minor_optical_depth = jnp.array([[9.642172e-9]], dtype=jnp.float32)
			minor_optical_depth = gas_optics.compute_rayleigh_optical_depth(
				self.gas_optics_sw,
				self.vmr_lib,
				molecules,
				temperature,
				p,
				100,
			)
			np.testing.assert_allclose(
				minor_optical_depth,
				expected_minor_optical_depth,
				rtol=1e-5,
				atol=1e-14,
			)

	def test_compute_planck_fraction(self):
		"""Tests the Planck fraction computation."""
		# Temperature corresponding to the 10th reference point.
		temperature = jnp.array([[310.0]], dtype=jnp.float32)
		# Pressure corresponding to pressure index 4.
		pressure = jnp.array([[4.92749041e+04]], dtype=jnp.float32)
		vmr_fields = {
			self.gas_optics_lw.idx_h2o: jnp.array([[1.2e-3]], dtype=jnp.float32),
			self.gas_optics_lw.idx_o3: jnp.array([[3.124e-6]], dtype=jnp.float32),
		}
		
		planck_fraction = gas_optics.compute_planck_fraction(
			self.gas_optics_lw,
			self.vmr_lib,
			pressure,
			temperature,
			100,
			vmr_fields,
		)
		np.testing.assert_allclose(planck_fraction, [[0.116895]], rtol=1e-5, atol=0)

	def test_compute_planck_sources(self):
		"""Checks the Planck source computation for different temperature fields."""
		# Temperature corresponding to the 10th reference point.
		temperature_center = jnp.array([[310.0]], dtype=jnp.float32)
		# Temperature corresponding to the 135th reference Planck temperature.
		temperature_top = jnp.array([[295.0]], dtype=jnp.float32)
		planck_fraction = jnp.array([[0.116895]], dtype=jnp.float32)
		
		def planck_src_fn(temp: Array) -> Array:
			return gas_optics.compute_planck_sources(
				self.gas_optics_lw,
				planck_fraction,
				temp,
				100
			)
		np.testing.assert_allclose(
			planck_src_fn(temperature_center), [[1.287805]], rtol=1e-5, atol=0
		)
		np.testing.assert_allclose(
			planck_src_fn(temperature_top), [[1.008423]], rtol=1e-5, atol=0
		)

if __name__ == '__main__':
	absltest.main()
