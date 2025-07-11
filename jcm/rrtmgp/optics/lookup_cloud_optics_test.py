from absl.testing import absltest
from etils import epath
from jcm.rrtmgp.optics import lookup_cloud_optics

ROOT_DIR = epath.resource_path('jcm.rrtmgp.optics.rrtmgp_data')
_LW_LOOKUP_TABLE_FILENAME = 'cloudysky_lw.nc'
_LW_LOOKUP_TABLE_FILEPATH = ROOT_DIR / _LW_LOOKUP_TABLE_FILENAME

class LookupCloudOpticsTest(absltest.TestCase):

	def test_longwave_optics_lookup_loads_data(self):
		# ACTION
		lookup_cld = lookup_cloud_optics.from_nc_file(_LW_LOOKUP_TABLE_FILEPATH)
		
		# VERIFICATION
		self.assertEqual(lookup_cld.n_size_liq, 20)
		self.assertEqual(lookup_cld.n_size_ice, 18)
		self.assertEqual(lookup_cld.ext_liq.shape, (16, 20))
		self.assertEqual(lookup_cld.ssa_liq.shape, (16, 20))
		self.assertEqual(lookup_cld.asy_liq.shape, (16, 20))
		self.assertEqual(lookup_cld.ext_ice.shape, (3, 16, 18))
		self.assertEqual(lookup_cld.ssa_ice.shape, (3, 16, 18))
		self.assertEqual(lookup_cld.asy_ice.shape, (3, 16, 18))
		self.assertEqual(lookup_cld.ice_roughness.value, 1)


if __name__ == '__main__':
	absltest.main()
