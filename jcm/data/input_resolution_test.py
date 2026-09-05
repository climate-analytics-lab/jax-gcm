"""Standalone tests for the typed forcing-input resolver (issue #751, phase A).

The engine is built and tested at the library level here; ``jcm.runners`` is not
yet rewired to it (phase B), so these mirror the runner/forcing behaviours the
resolver subsumes — year expansion + coverage clamps, list-into-products
splitting, merge compatibility, publication gating and eager-fetch-loud-offline —
against the engine directly, so it is proven before the wiring lands.
"""

import unittest
from unittest import mock

import numpy as np
import xarray as xr

from jcm.data import input_resolution as ir
from jcm.data import mirror_manifest as mm


class TestParseSpec(unittest.TestCase):
    def test_auto_none_explicit(self):
        self.assertIs(ir.InputSpec.parse("k", "auto").kind, ir.SpecKind.AUTO)
        for v in (None, "", "null", "none"):
            self.assertIs(ir.InputSpec.parse("k", v).kind, ir.SpecKind.NONE)
        s = ir.InputSpec.parse("k", "/a.nc")
        self.assertIs(s.kind, ir.SpecKind.EXPLICIT)
        self.assertFalse(s.is_list or s.has_pattern)

    def test_list_and_pattern_flags(self):
        s = ir.InputSpec.parse("k", ["/bb_{year}.nc", "/anthro.nc"])
        self.assertIs(s.kind, ir.SpecKind.EXPLICIT)
        self.assertTrue(s.is_list and s.has_pattern)
        # An all-null list is opt-out.
        self.assertIs(ir.InputSpec.parse("k", [None, ""]).kind,
                      ir.SpecKind.NONE)


class TestForcingProducts(unittest.TestCase):
    """Port of runners TestYearExpansionAndStartDate list/coverage cases."""

    def test_list_spec_splits_into_per_element_products(self):
        self.assertEqual(
            ir.forcing_products(["/bb_{year}.nc", "/anthro.nc"],
                                [2000, 2001], None),
            [["/bb_2000.nc", "/bb_2001.nc"], "/anthro.nc"])
        self.assertEqual(
            ir.forcing_products(["/a.nc", "/b.nc"], [2000, 2001], None),
            ["/a.nc", "/b.nc"])
        self.assertEqual(
            ir.forcing_products("/anthro.nc", [2000, 2001], None), ["/anthro.nc"])
        self.assertEqual(
            ir.forcing_products("/bb_{year}.nc", [2000, 2001], None),
            [["/bb_2000.nc", "/bb_2001.nc"]])

    def test_coverage_clamps_expansion(self):
        # emissions coverage ends 2022 while the requested range runs to 2024.
        self.assertEqual(
            ir.forcing_products("/emis/{year}.nc", [2023, 2024], [1850, 2022]),
            [["/emis/2022.nc"]])
        # The by-date bracket pads one year each side, clipped to coverage.
        self.assertEqual(
            ir.expand_yearly("/o3/{year}.nc", [2022, 2022], [1850, 2022]),
            ["/o3/2021.nc", "/o3/2022.nc"])

    def test_coverage_from_manifest(self):
        manifest = mm.load_manifest()
        cov = mm.coverage(manifest, "emissions_amip")
        self.assertEqual(
            ir.forcing_products("/emis/{year}.nc", [2023, 2024], cov),
            [["/emis/2022.nc"]])


class TestMergeCompatibility(unittest.TestCase):
    """Port of runners _product_time_axis / _assert_uniform_time_axis cases."""

    def _open_datetime(self, year_by_path):
        def _open(path, **_kw):
            for frag, times in year_by_path.items():
                if frag in str(path):
                    return xr.Dataset(coords={"time": np.array(
                        times, dtype="datetime64[ns]")})
            return xr.Dataset()
        return _open

    def test_product_time_axis_object_cftime(self):
        class _CFTime:
            def __init__(self, m):
                self.month = m

            def __str__(self):
                return f"2000-{self.month:02d}-15"

        def _open(_p, **_kw):
            return xr.Dataset(coords={"time": ("time", np.array(
                [_CFTime(m) for m in range(1, 13)], dtype=object))})
        dtype, key, _ = ir.product_time_axis(["/clim.nc"], open_dataset=_open)
        self.assertEqual(dtype, "datetime")
        self.assertEqual(key, frozenset(f"2000-{m:02d}-15" for m in range(1, 13)))

    def test_product_time_axis_no_time_is_none(self):
        def _open(_p, **_kw):
            return xr.Dataset(coords={"lat": [0.0, 1.0]})
        self.assertIsNone(ir.product_time_axis(["/static.nc"], open_dataset=_open))

    def test_identical_axis_and_single_product_pass(self):
        _open = self._open_datetime({"a": ["2000-06-15"], "b": ["2000-06-15"]})
        # (a) two products, identical axis.
        ir.assert_uniform_time_axis([["/a.nc"], ["/b.nc"]],
                                    config_key="k", open_dataset=_open)
        # (b) a single product's yearly files.
        ir.assert_uniform_time_axis([["/a.nc", "/b.nc"]],
                                    config_key="k", open_dataset=_open)

    def test_one_year_transient_plus_offyear_climatology_rejects(self):
        _open = self._open_datetime({
            "bb_2000": ["2000-06-15"],
            "clim": [f"1850-{m:02d}-15" for m in range(1, 13)]})
        with self.assertRaisesRegex(ValueError, "incompatible time axes"):
            ir.assert_uniform_time_axis(
                [["/bb_2000.nc"], ["/clim.nc"]],
                config_key="forcing.emissions_file", open_dataset=_open)


class TestResolveInput(unittest.TestCase):
    def setUp(self):
        self.manifest = mm.load_manifest()

    def _resolve(self, key, value, **kw):
        kw.setdefault("grid_token", "t63")
        kw.setdefault("nlev", 47)
        kw.setdefault("manifest", self.manifest)
        return ir.resolve_input(key, value, **kw)

    def test_none_opts_out(self):
        r = self._resolve("emissions_file", "null")
        self.assertTrue(r.is_none)

    def test_auto_published_fetches_and_carries_alignment(self):
        fetch = mock.Mock(return_value="/cache/emissions_pd.nc")
        r = self._resolve("emissions_file", "auto", fetch=fetch)
        self.assertFalse(r.is_none)
        self.assertEqual(r.paths, ("/cache/emissions_pd.nc",))
        self.assertEqual(r.alignment, ir.WRAP_YEAR)
        self.assertEqual(r.source, "auto:emissions_pd")
        self.assertEqual(r.provenance, ("hf://bundles/t63/emissions_pd.nc",))
        fetch.assert_called_once_with("bundles/t63/emissions_pd.nc")

    def test_auto_unpublished_grid_nulls_without_fetch(self):
        fetch = mock.Mock(side_effect=AssertionError("fetch must not run"))
        r = self._resolve("emissions_file", "auto", grid_token="t42",
                          fetch=fetch)
        self.assertTrue(r.is_none)
        fetch.assert_not_called()

    def test_auto_oxidants_unpublished_level_and_sigma_null(self):
        fetch = mock.Mock(side_effect=AssertionError("fetch must not run"))
        # Unpublished layer count on a published grid.
        self.assertTrue(self._resolve(
            "oxidants_file", "auto", nlev=8, fetch=fetch).is_none)
        # Published (token, nlev) but sigma vertical.
        self.assertTrue(self._resolve(
            "oxidants_file", "auto", nlev=47, vertical="sigma",
            fetch=fetch).is_none)

    def test_auto_disabled_consumer_nulls_without_fetch(self):
        fetch = mock.Mock(side_effect=AssertionError("fetch must not run"))
        r = self._resolve("emissions_file", "auto", enabled=False, fetch=fetch)
        self.assertTrue(r.is_none)
        fetch.assert_not_called()

    def test_auto_not_yet_staged_raises_precisely(self):
        with self.assertRaisesRegex(FileNotFoundError, "yet published"):
            self._resolve("macv2_file", "auto",
                          fetch=mock.Mock(return_value="/x"))

    def test_auto_offline_raises_actionable(self):
        fetch = mock.Mock(side_effect=OSError("offline"))
        with self.assertRaisesRegex(FileNotFoundError, "could not be downloaded"):
            self._resolve("emissions_file", "auto", fetch=fetch)

    def test_explicit_hf_path_fetched(self):
        fetch = mock.Mock(return_value="/cache/f.nc")
        r = self._resolve("emissions_file", "hf://bundles/t63/emissions_pd.nc",
                          fetch=fetch)
        self.assertEqual(r.paths, ("/cache/f.nc",))
        self.assertEqual(r.provenance, ("hf://bundles/t63/emissions_pd.nc",))
        self.assertEqual(r.source, "explicit")

    def test_explicit_year_pattern_expands_one_product(self):
        r = self._resolve("emissions_file", "/emis/{year}.nc",
                          years=[2000, 2001], fetch=lambda p: p)
        self.assertEqual(r.products, (["/emis/2000.nc", "/emis/2001.nc"],))
        self.assertEqual(r.paths, ("/emis/2000.nc", "/emis/2001.nc"))

    def test_explicit_list_keeps_products_separate(self):
        r = self._resolve("emissions_file", ["/bb_{year}.nc", "/anthro.nc"],
                          years=[2000, 2001], fetch=lambda p: p)
        self.assertEqual(
            r.products, (["/bb_2000.nc", "/bb_2001.nc"], "/anthro.nc"))


if __name__ == "__main__":
    unittest.main()
