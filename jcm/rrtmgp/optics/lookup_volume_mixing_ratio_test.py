from absl.testing import absltest
from etils import epath
import numpy as np
from jcm.rrtmgp.config import radiative_transfer
from jcm.rrtmgp.optics import lookup_volume_mixing_ratio

ROOT_DIR = epath.resource_path('jcm.rrtmgp.optics.test_data')
_GLOBAL_MEANS_FILENAME = 'vmr_global_means.json'
_SOUNDING_CSV_FILENAME = 'vmr_interpolated.csv'

_GLOBAL_MEANS_FILEPATH = ROOT_DIR / _GLOBAL_MEANS_FILENAME
_SOUNDING_CSV_FILEPATH = ROOT_DIR / _SOUNDING_CSV_FILENAME

# First 10 vmr values from the sounding csv file.
_EXPECTED_VMR_CH4_FROM_SOUNDING = [
    1.52882521e-07, 1.52882521e-07, 1.52882521e-07, 1.52882521e-07,
    1.52882521e-07, 1.54751092e-07, 1.58478497e-07, 1.62205903e-07,
    1.65933308e-07, 1.69660714e-07
]


class LookupVolumeMixingRatioTest(absltest.TestCase):

	def test_volume_mixing_ratio_lookup_loads_data(self):
		atmospheric_state_cfg = radiative_transfer.AtmosphericStateCfg(
			vmr_global_mean_filepath=_GLOBAL_MEANS_FILEPATH,
			vmr_sounding_filepath=_SOUNDING_CSV_FILEPATH,
		)
		lookup_vmr = lookup_volume_mixing_ratio.from_config(atmospheric_state_cfg)
		np.testing.assert_allclose(
			lookup_vmr.profiles['ch4'][:10],
			np.array(_EXPECTED_VMR_CH4_FROM_SOUNDING),
			rtol=1e-5,
			atol=0,
		)
		self.assertEqual(lookup_vmr.global_means['co2'], 3.9754697e-4)
		self.assertAlmostEqual(
			lookup_vmr.global_means['n2o'], 3.2698801e-7, delta=1e-12
		)
		self.assertEqual(lookup_vmr.global_means['co'], 1.2e-7)


if __name__ == '__main__':
	absltest.main()
