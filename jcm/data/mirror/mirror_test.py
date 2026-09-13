"""Unit tests for the mirror builders' pure-math pieces.

The builders themselves read multi-GB Glade sources; these tests cover
the statistics/registry machinery on synthetic inputs. The shared
regridding helpers are tested in ``jcm/data/regridding_test.py``, the
downloader in ``jcm/data/remote_test.py``.
"""

import os
import tempfile
import unittest

import numpy as np

from jcm.data.mirror.sso import finalize
from jcm.data.mirror.registry import build_registry, write_registry


class SsoFinalizeTest(unittest.TestCase):
    def _acc(self, **over):
        n = np.full(4, 10.0)
        acc = {"n": n, "land": n.copy(), "sh": np.zeros(4),
               "sh2": np.zeros(4), "shx2": np.zeros(4),
               "shy2": np.zeros(4), "shxy": np.zeros(4),
               "pic": np.zeros(4), "val": np.zeros(4)}
        acc.update(over)
        return acc

    def test_isotropic_slope_has_zero_anisotropy(self):
        # equal x/y gradient variance, no correlation -> gamma = 1 (round)
        acc = self._acc(shx2=np.full(4, 10.0), shy2=np.full(4, 10.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 1.0)
        np.testing.assert_allclose(out["orosig"], 1.0)  # sqrt(K+L=1)

    def test_pure_xslope_is_fully_anisotropic(self):
        # gradient variance only in x -> gamma = 0, theta = 0 deg
        acc = self._acc(shx2=np.full(4, 20.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 0.0)
        np.testing.assert_allclose(out["orothe"], 0.0)
        np.testing.assert_allclose(out["orosig"], np.sqrt(2.0))

    def test_ocean_cells_are_zeroed(self):
        acc = self._acc(land=np.array([10.0, 0.0, 10.0, 0.0]),
                        sh=np.full(4, 50.0), shx2=np.full(4, 4.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], [5.0, 0.0, 5.0, 0.0])
        np.testing.assert_allclose(out["lsm"], [1.0, 0.0, 1.0, 0.0])

    def test_variance_from_sufficient_statistics(self):
        h = np.array([1.0, 3.0, 5.0, 7.0])
        acc = self._acc(n=np.full(4, 2.0), land=np.full(4, 2.0),
                        sh=2 * h, sh2=2 * h ** 2 + 2.0)
        # per-cell: two samples h +/- 1 -> mean h, std 1
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], h)
        np.testing.assert_allclose(out["orostd"], 1.0)


class RegistryTest(unittest.TestCase):
    def test_registry_hashes_files(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sub").mkdir()
            (Path(d) / "sub" / "a.nc").write_bytes(b"hello")
            path = write_registry(d)
            reg = json.loads(Path(path).read_text())
            self.assertIn("sub/a.nc", reg["files"])
            self.assertEqual(reg["files"]["sub/a.nc"]["size"], 5)
            # registry.json itself is excluded
            reg2 = build_registry(d)
            self.assertNotIn("registry.json", reg2["files"])


if __name__ == "__main__":
    unittest.main()


class DustProductTest(unittest.TestCase):
    """The dust bundle builder on a synthetic stand-in for the HAMMOZ files.

    The real inputs are on Glade; what needs covering is the orientation flip,
    the nearest-neighbour regrid and the two build-time guards.
    """

    NLAT, NLON = 96, 192

    def _sources(self, tmp, regions=None, soils=None):
        import xarray as xr

        from jcm.data.regridding import gaussian_latlon
        lats, lons = gaussian_latlon(self.NLAT)
        # The HAMMOZ files store latitude DESCENDING; the builder flips it.
        desc = lats[::-1]
        shape = (1, self.NLAT, self.NLON)
        month = (12, self.NLAT, self.NLON)
        # pot_source varies with latitude so the flip is observable.
        pot = np.broadcast_to(np.abs(desc)[None, :, None], month).copy()
        if regions is None:
            regions = np.tile(np.arange(1, 9).repeat(self.NLAT // 8)[:, None],
                              (1, self.NLON))[None]
        if soils is None:
            soils = {f"type{i}": np.full(shape, 0.2 if i in (2, 3, 4, 6) else 0.0)
                     for i in (2, 3, 4, 6, 13, 14, 15, 16, 17)}
        # Month starts on a Gregorian year, like the HAMMOZ files: written as
        # datetime64 so the file carries real CF units for the builder to copy.
        months = np.array([np.datetime64(f"2000-{m:02d}-01") for m in range(1, 13)])
        coords = {"lat": desc, "lon": lons}
        files = {
            "dust_potential_sources_T63.nc": xr.Dataset(
                {"pot_source": (("time", "lat", "lon"), pot)},
                coords={**coords, "time": months}),
            "dust_preferential_sources_T63.nc": xr.Dataset(
                {"source": (("time", "lat", "lon"), np.full(shape, 0.3))},
                coords={**coords, "time": months[:1]}),
            "soil_type_all_T63.nc": xr.Dataset(
                {k: (("time", "lat", "lon"), v) for k, v in soils.items()},
                coords={**coords, "time": months[:1]}),
            "dust_regions_T63.nc": xr.Dataset(
                {"regions": (("time", "lat", "lon"), regions.astype(float))},
                coords={**coords, "time": months[:1]}),
            "surface_rough_12m_T63.nc": xr.Dataset(
                {"surfrough": (("time", "lat", "lon"), np.full(month, 0.02))},
                coords={**coords, "time": months}),
        }
        for name, ds in files.items():
            ds.to_netcdf(os.path.join(tmp, name))
        return lats, lons

    def test_native_grid_is_copied_with_ascending_latitude(self):
        import xarray as xr

        from jcm.data.mirror.dust import build_dust_product
        with tempfile.TemporaryDirectory() as tmp:
            lats, _ = self._sources(tmp)
            out = os.path.join(tmp, "pot.nc")
            build_dust_product("dust_potential_sources", self.NLAT, out,
                               source_dir=tmp)
            with xr.open_dataset(out, decode_times=False) as ds:
                np.testing.assert_allclose(ds.lat.values, lats)
                np.testing.assert_allclose(
                    ds.pot_source.values[0, :, 0], np.abs(lats), atol=1e-9)
                self.assertNotIn("regrid_approximation", ds.attrs)

    def test_regrid_is_nearest_and_keeps_the_mask_categorical(self):
        import xarray as xr

        from jcm.data.mirror.dust import build_dust_product
        with tempfile.TemporaryDirectory() as tmp:
            self._sources(tmp)
            out = os.path.join(tmp, "reg.nc")
            build_dust_product("dust_regions", 32, out, source_dir=tmp)
            with xr.open_dataset(out, decode_times=False) as ds:
                values = ds.regions.values
                self.assertEqual(values.shape, (32, 64))
                np.testing.assert_array_equal(values, np.round(values))
                self.assertTrue(set(np.unique(values)) <= set(range(1, 9)))
                self.assertIn("regrid_approximation", ds.attrs)

    def test_a_built_product_round_trips_through_its_reader(self):
        # The build and the read are the two halves of one contract; nothing
        # else checks that a published file is loadable, decodable and on the
        # model grid.
        import jax.numpy as jnp
        import xarray as xr

        from jcm.forcing import (WRAP_YEAR, read_dust_preferential,
                                 read_dust_regions, read_dust_roughness,
                                 read_dust_soil_types, read_dust_source)
        from jcm.data.mirror.dust import build_dust_product
        readers = {"dust_potential_sources": read_dust_source,
                   "dust_preferential_sources": read_dust_preferential,
                   "dust_soil_types": read_dust_soil_types,
                   "dust_regions": read_dust_regions,
                   "dust_surface_roughness": read_dust_roughness}
        with tempfile.TemporaryDirectory() as tmp:
            lats, lons = self._sources(tmp)
            for name, reader in readers.items():
                out = os.path.join(tmp, f"{name}.nc")
                build_dust_product(name, self.NLAT, out, source_dir=tmp)
                # Decoded (not decode_times=False): a monthly product must carry
                # real CF time units, not bare numbers read as nanoseconds.
                with xr.open_dataset(out) as ds:
                    if "time" in ds.dims and ds.sizes["time"] > 1:
                        self.assertEqual(
                            str(ds.time.values[1])[:10], "2000-02-01",
                            f"{name}: time axis lost its units")
                    leaf = reader(ds, lat_deg=lats, lon_deg=lons)
                values = (next(iter(leaf.values())) if isinstance(leaf, dict)
                          else getattr(leaf, "values", leaf))
                self.assertEqual(jnp.asarray(values).shape[-2:],
                                 (self.NLON, self.NLAT), name)
                if name in ("dust_potential_sources", "dust_surface_roughness"):
                    self.assertEqual(int(leaf.align_mode), WRAP_YEAR, name)

    def test_a_broken_global_partition_is_refused(self):
        from jcm.data.mirror.dust import build_dust_product
        soils = {f"type{i}": np.full((1, self.NLAT, self.NLON), 0.3)
                 for i in (2, 3, 4, 6, 13, 14, 15, 16, 17)}
        with tempfile.TemporaryDirectory() as tmp:
            self._sources(tmp, soils=soils)
            with self.assertRaisesRegex(ValueError, "would go negative"):
                build_dust_product("dust_soil_types", self.NLAT,
                                   os.path.join(tmp, "soil.nc"),
                                   source_dir=tmp)

    def test_a_non_integer_region_mask_is_refused(self):
        from jcm.data.mirror.dust import build_dust_product
        regions = np.full((1, self.NLAT, self.NLON), 2.5)
        with tempfile.TemporaryDirectory() as tmp:
            self._sources(tmp, regions=regions)
            with self.assertRaisesRegex(ValueError, "lost integrality"):
                build_dust_product("dust_regions", self.NLAT,
                                   os.path.join(tmp, "reg.nc"),
                                   source_dir=tmp)
