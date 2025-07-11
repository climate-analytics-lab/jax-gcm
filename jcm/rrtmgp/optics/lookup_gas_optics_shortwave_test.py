"""Tests whether the shortwavewave optics data for atmospheric gases are loaded properly."""

from absl.testing import absltest
from etils import epath
import jax

from jcm.rrtmgp.optics import lookup_gas_optics_shortwave

ROOT_DIR = epath.resource_path('jcm.rrtmgp.optics.rrtmgp_data')
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp-gas-sw-g224.nc'
_SW_LOOKUP_TABLE_FILEPATH = ROOT_DIR / _SW_LOOKUP_TABLE_FILENAME


class LookupGasOpticsShortwaveTest(absltest.TestCase):

	def test_shortwave_optics_lookup_loads_data(self):
		lookup_shortwave = lookup_gas_optics_shortwave.from_nc_file(
			_SW_LOOKUP_TABLE_FILEPATH
		)
		expected_dims = {
			'n_gases': 19,
			'n_maj_absrb': 19,
			'n_atmos_layers': 2,
			'n_bnd': 14,
			'n_contrib_lower': 544,
			'n_contrib_upper': 384,
			'n_gpt': 224,
			'n_minor_absrb': 21,
			'n_minor_absrb_lower': 34,
			'n_minor_absrb_upper': 24,
			'n_mixing_fraction': 9,
			'n_p_ref': 59,
			'n_t_ref': 14,
		}
		actual_dims = {
			'n_gases': lookup_shortwave.n_gases,
			'n_maj_absrb': lookup_shortwave.n_maj_absrb,
			'n_atmos_layers': lookup_shortwave.n_atmos_layers,
			'n_bnd': lookup_shortwave.n_bnd,
			'n_contrib_lower': lookup_shortwave.n_contrib_lower,
			'n_contrib_upper': lookup_shortwave.n_contrib_upper,
			'n_gpt': lookup_shortwave.n_gpt,
			'n_minor_absrb': lookup_shortwave.n_minor_absrb,
			'n_minor_absrb_lower': lookup_shortwave.n_minor_absrb_lower,
			'n_minor_absrb_upper': lookup_shortwave.n_minor_absrb_upper,
			'n_mixing_fraction': lookup_shortwave.n_mixing_fraction,
			'n_p_ref': lookup_shortwave.n_p_ref,
			'n_t_ref': lookup_shortwave.n_t_ref,
		}
		self.assertEqual(actual_dims, expected_dims)
		self.assertEqual(lookup_shortwave.kmajor.shape, (14, 60, 9, 224))
		self.assertEqual(lookup_shortwave.solar_src_scaled.shape, (224,))
		self.assertEqual(lookup_shortwave.rayl_lower.shape, (14, 9, 224))
		self.assertEqual(lookup_shortwave.rayl_upper.shape, (14, 9, 224))


if __name__ == '__main__':
	jax.config.update('jax_enable_x64', True)
	absltest.main()
