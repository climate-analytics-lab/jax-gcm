"""CAM prescribed surface-cover mapping and inventory validation."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from jcm.data.bc import cam_landuse


class CAMLandUseTest(unittest.TestCase):
    def test_pft_mapping_ocean_residual_and_overlapping_cover(self):
        pft=np.zeros((17,1,4))
        pft[0,0,0]=100  # ignored because LANDMASK is ocean
        pft[4,0,1]=100  # deciduous forest
        pft[9,0,2]=50   # shrubland, residual half water
        pft[0,0,3]=100  # bare ground plus overlap with urban cover
        ds=xr.Dataset({"PCT_PFT":(("pft","lat","lon"),pft),
                       "LANDMASK":(("lat","lon"),[[0,1,1,1]]),
                       "PCT_LAKE":(("lat","lon"),np.zeros((1,4))),
                       "PCT_WETLAND":(("lat","lon"),np.zeros((1,4))),
                       "PCT_URBAN":(("lat","lon"),[[0,0,0,50]])},
                      coords={"lat":[0],"lon":[0,90,180,270]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"input.nc"
            ds.to_netcdf(path)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(cam_landuse,"SOURCE_SHA256",sha):
                actual=cam_landuse.prepare(path).fraction_landuse.values
        expected=np.zeros((11,1,4))
        expected[6,0,0]=1
        expected[3,0,1]=1
        expected[10,0,2]=.5
        expected[6,0,2]=.5
        expected[7,0,3]=1
        expected[0,0,3]=.5
        np.testing.assert_array_equal(actual,expected)
        # Normalize after target remapping, like CAM, not at source cells.
        self.assertEqual(actual[:,0,3].sum(),1.5)

    def test_wrong_inventory_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"input.nc"
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                cam_landuse.prepare(path)
