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


class LandChannelCoverageTest(unittest.TestCase):
    """Every surface-forcing writer must carry each ``translate_land`` channel.

    ``translate_land`` is the single land translation, but three builders
    serialize its output independently — the climatological bundles, the
    transient ERA5 years and the transient AMIP years — and a builder that
    silently omits a channel produces files that load with that channel
    ``None``. For ``soilw_rel`` that means the dust saturation cut-off runs
    inert with no other symptom (#787), which is exactly the failure this
    guards: the builders read multi-GB Glade sources, so no unit test runs
    them end to end.
    """

    _WRITERS = ("bundles", "era5_yearly", "amip_yearly")

    def test_no_writer_drops_a_channel(self):
        import ast
        import inspect
        import importlib

        translated = set(_translate_land_keys())
        self.assertIn("soilw_rel", translated)
        for name in self._WRITERS:
            module = importlib.import_module(f"jcm.data.mirror.{name}")
            # Dict KEYS only: every one of these builders assembles the
            # variables it writes as a dict literal, and a channel merely
            # mentioned in a comment or read back out of ``land[...]`` is
            # exactly the case that passes review and writes nothing.
            written = {key.value for node in
                       ast.walk(ast.parse(inspect.getsource(module)))
                       if isinstance(node, ast.Dict)
                       for key in node.keys
                       if isinstance(key, ast.Constant)
                       and isinstance(key.value, str)}
            missing = sorted(translated - written)
            self.assertEqual(missing, [], f"{name}.py never names {missing}")

    def test_every_writer_carries_the_land_cover_maps(self):
        """``forest``/``glac`` reach every surface-forcing file (#672).

        They come from :func:`land_cover_fields` rather than
        ``translate_land`` (static, not monthly), so the dict-key check above
        cannot see them: each writer must splice that helper into the dict it
        serializes.
        """
        import ast
        import inspect
        import importlib

        for name in self._WRITERS:
            module = importlib.import_module(f"jcm.data.mirror.{name}")
            spliced = any(
                key is None and isinstance(value, ast.Call)
                and getattr(value.func, "id", None) == "land_cover_fields"
                for node in ast.walk(ast.parse(inspect.getsource(module)))
                if isinstance(node, ast.Dict)
                for key, value in zip(node.keys, node.values))
            self.assertTrue(spliced, f"{name}.py never writes forest/glac")


class LandCoverFieldsTest(unittest.TestCase):
    """``forest`` = ERA5 ``cvh``; ``glac`` = the permanent-snow mask (#672)."""

    def test_values_on_the_target_grid(self):
        import xarray as xr

        from jcm.data.mirror.bundles import land_cover_fields

        lat = np.linspace(-90.0, 90.0, 7)
        lon = np.arange(0.0, 360.0, 60.0)
        grid = dict(dims=("latitude", "longitude"),
                    coords={"latitude": lat, "longitude": lon})
        lsm = np.ones((7, 6))
        lsm[:, 3:] = 0.0             # the eastern half is sea
        cvh = np.where(lsm > 0.5, 0.6, 0.0)   # ERA5: cvh is zero at sea
        snow = np.zeros((7, 6), dtype=bool)
        snow[0] = True               # south-pole row: an ice sheet
        era5 = xr.Dataset({"cvh": xr.DataArray(cvh, **grid),
                           "lsm": xr.DataArray(lsm, **grid)})
        # Targets: pure land, a coastal cell half-way to the sea, pure sea.
        out = land_cover_fields(era5, xr.DataArray(snow, **grid),
                                lats=np.array([-90.0, 0.0]),
                                lons=np.array([60.0, 150.0, 240.0]))
        # Per LAND: the coastal cell keeps its land's 0.6, not a diluted 0.3.
        np.testing.assert_allclose(out["forest"].values,
                                   [[0.6, 0.6, 0.0], [0.6, 0.6, 0.0]])
        np.testing.assert_allclose(out["glac"].values,
                                   [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def _translate_land_keys():
    """Return the channel names ``translate_land`` produces on a minimal input."""
    import xarray as xr

    from jcm.data.mirror.bundles import translate_land

    base = xr.DataArray(np.ones((1, 2)), dims=("time", "cell"))
    cell = lambda v: xr.DataArray(np.full(2, v), dims=("cell",))  # noqa: E731
    era5 = xr.Dataset({"sd": base * 0.0, "stl1": base * 280.0,
                       "swvl1": base * 0.2, "swvl2": base * 0.2,
                       "cvh": cell(0.0), "cvl": cell(0.0), "slt": cell(2.0)})
    return translate_land(
        era5, permanent_snow=xr.DataArray(np.zeros(2, dtype=bool),
                                          dims=("cell",))).keys()
