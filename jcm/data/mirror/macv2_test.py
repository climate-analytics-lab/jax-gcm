"""Tests for the MACv2 natural-aerosol background bundle builder."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from jcm.data.mirror import macv2, sites
from jcm.data.mirror.macv2 import (
    ALTITUDE_FILES,
    LW_WAVELENGTH_NM,
    LW_WAVENUMBER_BOUNDS,
    OPTICS_FILES,
    PRODUCT,
    SW_WAVELENGTH_NM,
    SW_WAVENUMBER_BOUNDS,
    build_macv2_natural,
    source_files,
)
from jcm.data.regridding import gaussian_latlon, latitude_bounds

# A 10° stand-in for the 1° source, in the source's own orientation: latitude
# north to south, longitude from the date line.
_LAT = np.arange(85.0, -90.0, -10.0)
_LON = np.arange(-175.0, 180.0, 10.0)
_TOP = np.array([500.0, 1000.0, 2000.0, 5000.0])
_BOTTOM = np.array([0.0, 500.0, 1000.0, 2000.0])


def _history(tag, wavelengths):
    files = " ".join(f"gt_{tag}_{nm:05d}nm.nc" for nm in wavelengths)
    return f"Mon Aug 13 14:27:21 2018: ncecat {files} ../out.nc"


def _optics(key, rng, wavelengths=None, months=12):
    """One optics file: every band a different, spatially varying field."""
    _, tag = OPTICS_FILES[key]
    expected = SW_WAVELENGTH_NM if key.startswith("sw") else LW_WAVELENGTH_NM
    wavelengths = expected if wavelengths is None else wavelengths
    shape = (len(expected), months, _LAT.size, _LON.size)
    band = np.arange(1, len(expected) + 1)[:, None, None, None]
    aod = 0.01 * band + rng.uniform(0.0, 0.2, shape)
    return xr.Dataset(
        {"aod": (("record", "time", "lat", "lon"), aod.astype(np.float32)),
         "ssa": (("record", "time", "lat", "lon"),
                 rng.uniform(0.5, 1.0, shape).astype(np.float32)),
         "asy": (("record", "time", "lat", "lon"),
                 rng.uniform(0.1, 0.9, shape).astype(np.float32))},
        coords={"lat": _LAT, "lon": _LON,
                "time": np.arange(months, dtype=float)},
        attrs={"history": _history(tag, wavelengths)})


def _altitude(mode, rng, months=12):
    _, var = ALTITUDE_FILES[mode]
    raw = rng.uniform(0.0, 1.0, (months, _TOP.size, _LAT.size, _LON.size))
    frac = raw / raw.sum(axis=1, keepdims=True)
    return xr.Dataset(
        {var: (("time", "lay", "lat", "lon"), frac.astype(np.float32)),
         "ztop": ("lay", _TOP), "zbot": ("lay", _BOTTOM),
         "asl": (("lat", "lon"),
                 rng.uniform(0.0, 400.0, (_LAT.size, _LON.size)))},
        coords={"lat": _LAT, "lon": _LON, "lay": _TOP,
                "time": np.arange(months, dtype=float)})


def _write_tree(root, seed=0, **overrides):
    """Write a synthetic ``for_ICON`` tree; ``overrides`` replace datasets by key."""
    rng = np.random.default_rng(seed)
    datasets = {key: _optics(key, rng) for key in OPTICS_FILES}
    datasets.update({mode: _altitude(mode, rng) for mode in ALTITUDE_FILES})
    datasets.update(overrides)
    relpaths = {**{k: v[0] for k, v in OPTICS_FILES.items()},
                **{k: v[0] for k, v in ALTITUDE_FILES.items()}}
    for key, ds in datasets.items():
        path = Path(root) / relpaths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
    return datasets


def _area_mean(field, lats):
    w = np.diff(np.sin(np.deg2rad(latitude_bounds(np.sort(lats)))))
    if lats[0] > lats[-1]:
        field = field[..., ::-1, :]
    return (field.mean(axis=-1) * w).sum(axis=-1) / w.sum()


class TestBandTables(unittest.TestCase):
    def test_bands_ascend_in_wavelength_and_tile_the_spectrum(self):
        for wavelengths, bounds in ((SW_WAVELENGTH_NM, SW_WAVENUMBER_BOUNDS),
                                    (LW_WAVELENGTH_NM, LW_WAVENUMBER_BOUNDS)):
            self.assertEqual(len(wavelengths), len(bounds))
            self.assertEqual(list(wavelengths), sorted(wavelengths))
            lower = [b[0] for b in bounds]
            self.assertTrue(all(lo < hi for lo, hi in bounds))
            # Ascending wavelength is descending wavenumber; the last
            # shortwave band (820-2600 cm-1) is RRTMG's out-of-sequence one.
            body = lower[:-1] if bounds is SW_WAVENUMBER_BOUNDS else lower
            self.assertEqual(body, sorted(body, reverse=True))

    def test_nominal_wavelengths_sit_in_their_band_bar_two_labels(self):
        outside = []
        for wavelengths, bounds in ((SW_WAVELENGTH_NM, SW_WAVENUMBER_BOUNDS),
                                    (LW_WAVELENGTH_NM, LW_WAVENUMBER_BOUNDS)):
            for nm, (lo, hi) in zip(wavelengths, bounds):
                if not lo <= 1.0e7 / nm <= hi:
                    outside.append(nm)
        # The source labels two longwave bands just past their upper edge.
        self.assertEqual(outside, [4200, 30000])

    def test_history_parser_reads_the_stacking_order(self):
        history = _history("c", (230, 300, 8000))
        self.assertEqual(macv2._history_wavelengths(history, "c"),
                         (230, 300, 8000))
        self.assertEqual(macv2._history_wavelengths(history, "p"), ())


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "for_ICON"
        cls.src = _write_tree(cls.root)
        cls.nlat = 8
        cls.out = Path(cls._tmp.name) / "bundles" / "t" / f"{PRODUCT}.nc"
        build_macv2_natural(cls.nlat, cls.out, cls.root)
        cls.ds = xr.open_dataset(cls.out).load()

    @classmethod
    def tearDownClass(cls):
        cls.ds.close()
        cls._tmp.cleanup()

    def test_grid_is_the_ascending_gaussian_grid(self):
        lats, lons = gaussian_latlon(self.nlat)
        np.testing.assert_allclose(self.ds["lat"], lats)
        np.testing.assert_allclose(self.ds["lon"], lons)
        self.assertEqual(self.ds["lat"].attrs["units"], "degrees_north")

    def test_coordinates_label_bands_months_and_layers(self):
        ds = self.ds
        np.testing.assert_array_equal(ds["month"], np.arange(1, 13))
        np.testing.assert_array_equal(ds["wavelength_sw"], SW_WAVELENGTH_NM)
        np.testing.assert_array_equal(ds["wavelength_lw"], LW_WAVELENGTH_NM)
        np.testing.assert_array_equal(
            ds["wavenumber_lower_sw"], [b[0] for b in SW_WAVENUMBER_BOUNDS])
        np.testing.assert_array_equal(
            ds["wavenumber_upper_lw"], [b[1] for b in LW_WAVENUMBER_BOUNDS])
        self.assertEqual(ds["wavelength_sw"].attrs["units"], "nm")
        self.assertEqual(ds["wavenumber_lower_sw"].attrs["units"], "cm-1")
        np.testing.assert_array_equal(ds["layer_bottom"], _BOTTOM)
        np.testing.assert_array_equal(ds["layer_top"], _TOP)
        self.assertEqual(ds["layer_top"].attrs["units"], "m")
        self.assertEqual(ds["aod_sw_coarse"].dims,
                         ("month", "band_sw", "lat", "lon"))
        self.assertEqual(ds["aod_lw_coarse"].dims,
                         ("month", "band_lw", "lat", "lon"))
        self.assertEqual(ds["layer_fraction_fine"].dims,
                         ("month", "layer", "lat", "lon"))

    def test_optical_depth_global_mean_is_conserved_per_band_and_month(self):
        lats = self.ds["lat"].values
        for key in OPTICS_FILES:
            src = self.src[key]["aod"].values.astype(np.float64)
            before = _area_mean(src, _LAT)                    # (band, month)
            after = _area_mean(self.ds[f"aod_{key}"].values, lats).T
            np.testing.assert_allclose(after, before, rtol=2e-6)

    def test_band_order_is_kept(self):
        # Each band's source field has its own offset, so a permutation of the
        # band axis would show up in the band means.
        src = self.src["sw_coarse"]["aod"].values.astype(np.float64)
        before = _area_mean(src.mean(axis=1), _LAT)
        after = _area_mean(self.ds["aod_sw_coarse"].mean("month").values,
                           self.ds["lat"].values)
        np.testing.assert_allclose(after, before, rtol=2e-6)
        self.assertTrue(np.all(np.diff(after) > 0.0))

    def test_ssa_and_asy_are_optical_depth_weighted(self):
        lats = self.ds["lat"].values
        for key in OPTICS_FILES:
            s = {n: self.src[key][n].values.astype(np.float64)
                 for n in ("aod", "ssa", "asy")}
            new = {n: np.moveaxis(self.ds[f"{n}_{key}"].values, 0, 1)
                   .astype(np.float64) for n in ("aod", "ssa", "asy")}
            scat = _area_mean(s["aod"] * s["ssa"], _LAT)
            np.testing.assert_allclose(
                _area_mean(new["aod"] * new["ssa"], lats), scat, rtol=1e-5)
            np.testing.assert_allclose(
                _area_mean(new["aod"] * new["ssa"] * new["asy"], lats),
                _area_mean(s["aod"] * s["ssa"] * s["asy"], _LAT), rtol=1e-5)
            self.assertTrue((new["ssa"] <= 1.0).all() and (new["ssa"] >= 0.0).all())

    def test_layer_fractions_sum_to_one_and_conserve_layer_optical_depth(self):
        lats = self.ds["lat"].values
        for mode, key in (("coarse", "sw_coarse"), ("fine", "sw_fine")):
            frac = self.ds[f"layer_fraction_{mode}"].values.astype(np.float64)
            np.testing.assert_allclose(frac.sum(axis=1), 1.0, atol=1e-5)
            i550 = SW_WAVELENGTH_NM.index(550)
            src_frac = self.src[mode][ALTITUDE_FILES[mode][1]].values
            src_aod = self.src[key]["aod"].values[i550].astype(np.float64)
            new_aod = self.ds[f"aod_{key}"].values[:, i550].astype(np.float64)
            np.testing.assert_allclose(
                _area_mean(frac * new_aod[:, None], lats),
                _area_mean(src_frac * src_aod[:, None], _LAT), rtol=1e-5)

    def test_orientation_follows_the_coordinates_not_the_index(self):
        # A northern-hemisphere-only source must land at positive latitudes
        # although the source's first row is the north and the bundle's the
        # south, and a source confined to 0-180°E at the bundle's first half.
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(1)
            ds = _optics("sw_coarse", rng)
            north_east = (ds["lat"] > 0) & (ds["lon"] > 0)
            ds["aod"] = ds["aod"].where(north_east, 0.0)
            _write_tree(tmp, sw_coarse=ds)
            out = Path(tmp) / "out.nc"
            build_macv2_natural(8, out, tmp)
            with xr.open_dataset(out) as built:
                aod = built["aod_sw_coarse"].mean(("month", "band_sw"))
                self.assertGreater(float(aod.sel(lat=slice(10, 90),
                                                 lon=slice(10, 170)).min()), 0.0)
                self.assertEqual(float(aod.sel(lat=slice(-90, -10)).max()), 0.0)
                self.assertEqual(float(aod.sel(lon=slice(190, 350)).max()), 0.0)
                # No optical depth: the reference's limits.
                south = built.sel(lat=slice(-90, -10))
                self.assertTrue((south["ssa_sw_coarse"] == 1.0).all())
                self.assertTrue((south["asy_sw_coarse"] == 0.0).all())

    def test_surface_altitude_is_the_area_mean_on_the_target_grid(self):
        src = self.src["fine"]["asl"].values.astype(np.float64)
        out = self.ds["source_surface_altitude"]
        self.assertEqual(out.dims, ("lat", "lon"))
        self.assertEqual(out.attrs["units"], "m")
        self.assertIn("not a floor", out.attrs["comment"])
        np.testing.assert_allclose(
            _area_mean(out.values.astype(np.float64), self.ds["lat"].values),
            _area_mean(src, _LAT), rtol=1e-6)

    def test_surface_altitude_follows_the_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(9)
            alts = {mode: _altitude(mode, rng) for mode in ALTITUDE_FILES}
            for alt in alts.values():
                north_east = (alt["lat"] > 0) & (alt["lon"] > 0)
                alt["asl"] = xr.where(north_east, 1000.0, 0.0)
            _write_tree(tmp, **alts)
            out = Path(tmp) / "out.nc"
            build_macv2_natural(8, out, tmp)
            with xr.open_dataset(out) as built:
                surface = built["source_surface_altitude"]
                np.testing.assert_allclose(
                    surface.sel(lat=slice(10, 90), lon=slice(10, 170)), 1000.0)
                self.assertEqual(float(surface.sel(lat=slice(-90, -10)).max()),
                                 0.0)
                self.assertEqual(float(surface.sel(lon=slice(190, 350)).max()),
                                 0.0)

    def test_fractions_without_optical_depth_fall_back_to_the_plain_remap(self):
        from jcm.data.regridding import conservative_overlap

        rng = np.random.default_rng(10)
        raw = rng.uniform(0.0, 1.0, (2, _TOP.size, _LAT.size, _LON.size))
        frac = (raw / raw.sum(axis=1, keepdims=True))[..., ::-1, :]
        src_lats, (lats, lons) = np.sort(_LAT), gaussian_latlon(8)
        weight = np.zeros((2, _LAT.size, _LON.size))
        weight[1] = 1.0                 # month 0 has no optical depth at all
        out = macv2._regrid_fraction(frac, weight, src_lats, _LON, lats, lons)
        plain = conservative_overlap(frac, src_lats, _LON, lats, lons)
        np.testing.assert_allclose(out[0], plain[0])
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)

    def test_attributes_record_provenance_and_global_means(self):
        attrs = self.ds.attrs
        self.assertEqual(attrs["product"], PRODUCT)
        self.assertIn(macv2.SOURCE_URL, attrs["source"])
        for rel in source_files():
            self.assertIn(rel, attrs["source"])
        self.assertIn("conservative", attrs["history"])
        self.assertIn("ascending wavelength", attrs["band_note"])
        i550 = SW_WAVELENGTH_NM.index(550)
        src = self.src["sw_coarse"]["aod"].values[i550].astype(np.float64)
        np.testing.assert_allclose(attrs["global_mean_aod_550nm_coarse"],
                                   _area_mean(src.mean(axis=0), _LAT),
                                   rtol=1e-6)
        # 8 Gaussian latitudes are coarser than the 18-row source.
        self.assertNotIn("regrid_approximation", attrs)

    def test_finer_target_says_it_carries_source_detail(self):
        out = Path(self._tmp.name) / "fine.nc"
        build_macv2_natural(32, out, self.root)
        with xr.open_dataset(out) as built:
            self.assertIn("piecewise constant",
                          built.attrs["regrid_approximation"])
            self.assertEqual(built.sizes["lat"], 32)


class TestRefusals(unittest.TestCase):
    def _build(self, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp, **overrides)
            build_macv2_natural(8, Path(tmp) / "out.nc", tmp)

    def test_band_order_other_than_labelled_is_refused(self):
        rng = np.random.default_rng(2)
        reordered = _optics("sw_coarse", rng,
                            wavelengths=SW_WAVELENGTH_NM[::-1])
        with self.assertRaisesRegex(ValueError, "file history lists the bands"):
            self._build(sw_coarse=reordered)

    def test_missing_history_is_refused(self):
        ds = _optics("lw_coarse", np.random.default_rng(3))
        ds.attrs = {}
        with self.assertRaisesRegex(ValueError, "file history lists the bands"):
            self._build(lw_coarse=ds)

    def test_wrong_month_count_is_refused(self):
        ds = _optics("sw_fine", np.random.default_rng(4), months=11)
        with self.assertRaisesRegex(ValueError, "expected 14 × 12"):
            self._build(sw_fine=ds)
        alt = _altitude("fine", np.random.default_rng(4), months=11)
        with self.assertRaisesRegex(ValueError, "11 months"):
            self._build(fine=alt)

    def test_non_finite_optics_are_refused(self):
        ds = _optics("sw_coarse", np.random.default_rng(5))
        ds["ssa"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._build(sw_coarse=ds)

    def test_fractions_not_summing_to_one_are_refused(self):
        alt = _altitude("coarse", np.random.default_rng(6))
        alt["zkm_aer_coarse_mo"] = alt["zkm_aer_coarse_mo"] * 0.5
        with self.assertRaisesRegex(ValueError, "does not sum to one"):
            self._build(coarse=alt)

    def test_gapped_layers_are_refused(self):
        alt = _altitude("fine", np.random.default_rng(7))
        alt["zbot"] = ("lay", _BOTTOM + 100.0)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            self._build(fine=alt)

    def test_altitude_on_another_grid_is_refused(self):
        alt = _altitude("coarse", np.random.default_rng(8))
        alt = alt.assign_coords(lon=alt["lon"] + 1.0)
        with self.assertRaisesRegex(ValueError, "is not on the grid of"):
            self._build(coarse=alt)


class TestSite(unittest.TestCase):
    def test_no_declared_site_and_the_override(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JCM_MACV2_DIR", None)
            for name in sites.SITES:
                self.assertIsNone(sites.SITES[name]().macv2)
        with patch.dict(os.environ, {"JCM_MACV2_DIR": "/copy/for_ICON"}):
            self.assertEqual(sites.current().macv2, "/copy/for_ICON")

    def test_default_source_needs_the_override(self):
        bare = replace(sites.current(), macv2=None)
        with patch.object(sites, "current", return_value=bare), \
                tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(FileNotFoundError, "JCM_MACV2_DIR"):
            build_macv2_natural(8, Path(tmp) / "out.nc")

    def test_default_source_is_the_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tree(tmp)
            here = replace(sites.current(), macv2=tmp)
            with patch.object(sites, "current", return_value=here):
                build_macv2_natural(8, Path(tmp) / "out.nc")
            self.assertTrue((Path(tmp) / "out.nc").is_file())


class TestStage(unittest.TestCase):
    def test_stage_is_opt_in_and_checks_its_sources(self):
        from jcm.data.mirror import build_mirror as bm

        self.assertIn("macv2", bm.STAGES)
        self.assertIn("macv2", bm._NOT_IN_ALL)
        absent = bm._stage_sources(replace(bm.SITE, macv2=None))["macv2"]
        self.assertEqual(absent, (("MACv2 for_ICON tree", None),))
        present = bm._stage_sources(replace(bm.SITE, macv2="/copy"))["macv2"]
        self.assertEqual(sorted(p for _, p in present),
                         [f"/copy/{rel}" for rel in source_files()])
        with patch.object(bm, "SITE", replace(bm.SITE, macv2=None)), \
                self.assertRaises(SystemExit) as ctx:
            bm.check_sources(["macv2"])
        self.assertIn("JCM_MACV2_DIR", str(ctx.exception))

    def test_stage_writes_one_bundle_per_selected_grid(self):
        from jcm.data.mirror import build_mirror as bm

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            _write_tree(src)
            upload = Path(tmp) / "upload"
            with patch.object(bm, "SITE", replace(bm.SITE, macv2=str(src))), \
                    patch.object(bm, "UPLOAD", upload), \
                    patch.object(bm, "GRIDS", {"t63": 8, "t106": 12}), \
                    patch.object(bm, "_SELECTED", frozenset({"t63"})):
                bm.stage_macv2()
            self.assertTrue(
                (upload / "bundles" / "t63" / f"{PRODUCT}.nc").is_file())
            self.assertFalse((upload / "bundles" / "t106").exists())


if __name__ == "__main__":
    unittest.main()
