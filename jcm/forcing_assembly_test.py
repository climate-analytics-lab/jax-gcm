"""Tests for the natural-aerosol background's path through the forcing engine.

From a ``macv2_natural_aerosol`` bundle (built here from a synthetic source
by the mirror builder itself, so reader and writer cannot drift apart) through
key resolution, the reader, the time selection and the term that consumes it.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from jcm import forcing_assembly as fa
from jcm.data.mirror.macv2 import SW_WAVELENGTH_NM, build_macv2_natural
from jcm.data.mirror.macv2_test import _write_tree
from jcm.date import DateData
from jcm.forcing import (
    MID_MONTH,
    NATURAL_AEROSOL_MONTHLY,
    NATURAL_AEROSOL_STATIC,
    WRAP_YEAR,
    ForcingData,
    TimeSeries,
    read_natural_aerosol,
)
from jcm.physics.aerosol.aerosol_types import AerosolData
from jcm.physics.aerosol.macv2_natural import Macv2NaturalAerosol
from jcm.physics.speedy.speedy_coords import get_speedy_coords

_TERM_TARGET = "jcm.physics.aerosol.Macv2NaturalAerosol"


def _cfg(physics=None, **forcing):
    return OmegaConf.create({"physics": physics or {}, "forcing": forcing})


def _terms_physics(*targets):
    return {"terms": {f"t{i}": {"_target_": t} for i, t in enumerate(targets)}}


class _BundleCase(unittest.TestCase):
    """A T21 model grid and a bundle built for it from a synthetic source."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        cls.nlon, cls.nlat = cls.coords.horizontal.nodal_shape
        source = Path(cls._tmp.name) / "source"
        _write_tree(source, seed=11)
        cls.bundle = Path(cls._tmp.name) / "macv2_natural_aerosol.nc"
        build_macv2_natural(cls.nlat, cls.bundle, source)
        cls.lat_deg, cls.lon_deg = fa._model_latlon_deg(cls.coords)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _read(self, **kwargs):
        with xr.open_dataset(self.bundle) as ds:
            return read_natural_aerosol(ds, lat_deg=self.lat_deg,
                                        lon_deg=self.lon_deg, **kwargs)


class TestReadNaturalAerosol(_BundleCase):
    def test_monthly_fields_are_mid_month_wrap_year_leaves_on_the_model_grid(self):
        clim = self._read()
        self.assertEqual(set(clim), {*NATURAL_AEROSOL_MONTHLY,
                                     *NATURAL_AEROSOL_STATIC})
        for name in NATURAL_AEROSOL_MONTHLY:
            leaf = clim[name]
            self.assertIsInstance(leaf, TimeSeries)
            self.assertEqual(leaf.values.shape[0], 12)
            self.assertEqual(leaf.values.shape[-2:], (self.nlon, self.nlat))
            self.assertEqual(int(leaf.align_mode), WRAP_YEAR)
            self.assertEqual(int(leaf.anchors), MID_MONTH)
            self.assertEqual(leaf.values.dtype, jnp.float32)
        self.assertEqual(clim["aod_sw_coarse"].values.shape[1], 14)
        self.assertEqual(clim["aod_lw_coarse"].values.shape[1], 16)
        self.assertEqual(clim["layer_fraction_fine"].values.shape[1],
                         clim["layer_top"].shape[0])

    def test_values_are_the_bundles_in_lon_lat_order(self):
        clim = self._read()
        with xr.open_dataset(self.bundle) as ds:
            expected = ds["aod_sw_fine"].transpose(
                "month", "band_sw", "lon", "lat").values
            np.testing.assert_array_equal(clim["aod_sw_fine"].values, expected)
            np.testing.assert_array_equal(clim["layer_bottom"],
                                          ds["layer_bottom"].values)
            np.testing.assert_array_equal(clim["wavenumber_upper_lw"],
                                          ds["wavenumber_upper_lw"].values)

    def test_month_start_anchors_can_be_declared(self):
        clim = self._read(anchors="month_start")
        self.assertNotEqual(int(clim["aod_sw_coarse"].anchors), MID_MONTH)

    def test_file_on_another_grid_is_refused(self):
        with xr.open_dataset(self.bundle) as ds, \
                self.assertRaisesRegex(ValueError, "model grid"):
            read_natural_aerosol(ds, lat_deg=self.lat_deg[::2],
                                 lon_deg=self.lon_deg)

    def test_incomplete_or_misordered_files_are_refused(self):
        with xr.open_dataset(self.bundle) as ds:
            ds = ds.load()
        with self.assertRaisesRegex(ValueError, "lacks ssa_lw_coarse"):
            read_natural_aerosol(ds.drop_vars("ssa_lw_coarse"))
        with self.assertRaisesRegex(ValueError, "months 1..12"):
            read_natural_aerosol(ds.assign_coords(month=np.arange(12)))
        bad = ds.copy()
        bad["aod_sw_coarse"] = bad["aod_sw_coarse"].where(
            bad["aod_sw_coarse"] < 1e9, np.nan)
        bad["aod_sw_coarse"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            read_natural_aerosol(bad)
        with self.assertRaisesRegex(ValueError, "expected dims"):
            read_natural_aerosol(ds.transpose("band_sw", "month", ...))


class TestTimeSelection(_BundleCase):
    def _selected(self, when):
        forcing = ForcingData.zeros((self.nlon, self.nlat)).copy(
            natural_aerosol=self._read())
        return forcing.select(
            DateData.set_date(jdt.to_datetime(when))).natural_aerosol

    def test_mid_month_returns_that_months_record(self):
        clim = self._read()
        # 16 January 12:00 is the middle of a 31-day month.
        chosen = self._selected("2031-01-16T12:00:00")
        np.testing.assert_allclose(chosen["aod_sw_coarse"],
                                   clim["aod_sw_coarse"].values[0], rtol=1e-6)
        self.assertEqual(chosen["aod_sw_coarse"].shape,
                         (14, self.nlon, self.nlat))

    def test_month_boundary_blends_the_two_neighbours(self):
        clim = self._read()
        chosen = self._selected("2031-02-01T00:00:00")
        jan = clim["layer_fraction_coarse"].values[0]
        feb = clim["layer_fraction_coarse"].values[1]
        w_feb = 15.5 / (15.5 + 14.0)        # half of January over the span
        np.testing.assert_allclose(chosen["layer_fraction_coarse"],
                                   w_feb * feb + (1.0 - w_feb) * jan,
                                   rtol=1e-5)

    def test_static_metadata_passes_through(self):
        chosen = self._selected("2031-07-04T06:00:00")
        np.testing.assert_array_equal(chosen["layer_top"],
                                      self._read()["layer_top"])


class TestComposesNaturalAerosol(unittest.TestCase):
    def test_term_list_presets_are_read_off_their_targets(self):
        with_term = _terms_physics("jcm.physics.aerosol.Macv2SpAerosol",
                                   _TERM_TARGET)
        self.assertTrue(fa._composes_natural_aerosol(_cfg(with_term)))
        without = _terms_physics("jcm.physics.aerosol.Macv2SpAerosol")
        self.assertFalse(fa._composes_natural_aerosol(_cfg(without)))

    def test_factory_presets_follow_the_factory_defaults(self):
        for physics, expected in (
                ({"builder": "echam_physics"}, True),
                ({"builder": "echam_physics", "radiation_scheme": "emulated"},
                 False),
                ({"builder": "echam_physics", "aerosol_module": "jam"}, False),
                ({"builder": "echam_physics", "natural_aerosol": False}, False),
                ({"builder": "echam_physics", "radiation_scheme": "emulated",
                  "natural_aerosol": True}, True)):
            with self.subTest(physics=physics):
                self.assertEqual(fa._composes_natural_aerosol(_cfg(physics)),
                                 expected)

    def test_other_physics_blocks_state_it_or_do_not_have_it(self):
        self.assertFalse(fa._composes_natural_aerosol(_cfg({})))
        self.assertFalse(fa._composes_natural_aerosol(None))
        self.assertTrue(fa._composes_natural_aerosol(
            _cfg({"natural_aerosol": True})))
        self.assertFalse(fa._composes_natural_aerosol(
            _cfg({"builder": "speedy_physics"})))


class TestResolveNaturalAerosolInput(_BundleCase):
    physics = _terms_physics(_TERM_TARGET)

    def _resolve(self, value, *, physics=None, kind="from_file",
                 is_pyses=False, coords=None):
        cfg = _cfg(self.physics if physics is None else physics,
                   kind=kind, natural_aerosol_file=value)
        return fa._resolve_natural_aerosol_input(
            cfg.forcing, cfg, coords or self.coords, is_pyses)

    def test_null_and_explicit_paths(self):
        for value in (None, "", "null", "none"):
            self.assertIsNone(self._resolve(value))
        self.assertEqual(self._resolve("/data/bundle.nc"), "/data/bundle.nc")
        # An explicit path is honoured on an idealised surface too.
        self.assertEqual(self._resolve("/data/bundle.nc", kind="default"),
                         "/data/bundle.nc")

    def test_explicit_path_on_the_column_backend_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pySES column forcing"):
            self._resolve("/data/bundle.nc", is_pyses=True)

    def test_auto_is_none_without_the_term_a_real_surface_or_a_spectral_grid(self):
        with patch.object(fa.ir, "resolve_input") as resolve:
            self.assertIsNone(self._resolve("auto", physics={}))
            self.assertIsNone(self._resolve("auto", kind="default"))
            self.assertIsNone(self._resolve("auto", is_pyses=True))
            resolve.assert_not_called()

    def test_auto_on_an_unpublished_grid_is_none_and_says_so(self):
        with patch.object(fa.ir, "resolve_input") as resolve, \
                self.assertLogs(fa.logger, level="WARNING") as logs:
            self.assertIsNone(self._resolve("auto"))       # T21
            resolve.assert_not_called()
        self.assertIn("no natural-aerosol background for grid t21",
                      logs.output[0])

    def test_auto_on_a_published_grid_fetches_the_bundle(self):
        t31 = get_speedy_coords(layers=8, spectral_truncation=31)
        resolved = SimpleNamespace(paths=["/cache/t31/bundle.nc"],
                                   is_none=False)
        with patch.object(fa.ir, "resolve_input",
                          return_value=resolved) as resolve:
            self.assertEqual(self._resolve("auto", coords=t31),
                             "/cache/t31/bundle.nc")
        self.assertEqual(resolve.call_args.args[:2],
                         ("natural_aerosol_file", "auto"))
        self.assertEqual(resolve.call_args.kwargs["grid_token"], "t31")

    def test_manifest_publishes_the_product_on_the_low_resolution_grids_too(self):
        manifest = fa.mm.load_manifest()
        name = fa.mm.product_for_key(manifest, "natural_aerosol_file")
        self.assertEqual(name, "macv2_natural_aerosol")
        for grid in ("t31", "t42", "t63", "t106", "t127", "t255"):
            self.assertTrue(fa.mm.is_published(manifest, name, grid))
        self.assertFalse(fa.mm.is_published(manifest, name, "t85"))
        # Nothing else is published on the two SPEEDY-family grids.
        for other, rec in manifest["products"].items():
            if other != name and rec["grids"] is not None:
                self.assertNotIn("t31", rec["grids"])
                self.assertNotIn("t42", rec["grids"])

    def test_emission_resolution_carries_the_key(self):
        cfg = _cfg(self.physics, kind="from_file",
                   natural_aerosol_file=str(self.bundle))
        out = fa._resolve_emission_inputs(cfg.forcing, cfg, self.coords,
                                          is_pyses=False)
        self.assertEqual(out.natural_aerosol_file, str(self.bundle))
        cfg = _cfg(self.physics, kind="default", natural_aerosol_file="auto")
        out = fa._resolve_emission_inputs(cfg.forcing, cfg, self.coords,
                                          is_pyses=False)
        self.assertIsNone(out.natural_aerosol_file)


class TestAttachNaturalAerosol(_BundleCase):
    def test_unset_and_unresolved_keys_attach_nothing(self):
        base = ForcingData.zeros((self.nlon, self.nlat))
        self.assertIs(fa._attach_natural_aerosol(base, None, self.coords), base)
        for value in (None, "null", "auto"):
            cfg = _cfg(natural_aerosol_file=value).forcing
            out = fa._attach_natural_aerosol(base, cfg, self.coords)
            self.assertIsNone(out.natural_aerosol)

    def test_a_path_attaches_the_climatology(self):
        base = ForcingData.zeros((self.nlon, self.nlat))
        cfg = _cfg(natural_aerosol_file=str(self.bundle)).forcing
        out = fa._attach_natural_aerosol(base, cfg, self.coords)
        self.assertEqual(set(out.natural_aerosol),
                         {*NATURAL_AEROSOL_MONTHLY, *NATURAL_AEROSOL_STATIC})
        self.assertEqual(int(out.natural_aerosol["aod_sw_fine"].anchors),
                         MID_MONTH)
        # The declared monthly reconstruction reaches the reader.
        cfg = _cfg(natural_aerosol_file=str(self.bundle),
                   monthly_anchors="month_start").forcing
        out = fa._attach_natural_aerosol(base, cfg, self.coords)
        self.assertNotEqual(int(out.natural_aerosol["aod_sw_fine"].anchors),
                            MID_MONTH)

    def test_no_parent_forcing_gets_the_default_surface(self):
        cfg = _cfg(natural_aerosol_file=str(self.bundle)).forcing
        out = fa._attach_natural_aerosol(None, cfg, self.coords)
        self.assertIsNotNone(out.natural_aerosol)

    def test_yearly_pattern_is_refused(self):
        cfg = _cfg(natural_aerosol_file="/data/{year}.nc").forcing
        with self.assertRaisesRegex(ValueError, "year"):
            fa._attach_natural_aerosol(None, cfg, self.coords)


class TestBundleToRadiationOptics(_BundleCase):
    """The whole path: bundle, forcing, time selection, term."""

    def test_selected_climatology_reaches_the_aerosol_slot(self):
        cfg = _cfg(natural_aerosol_file=str(self.bundle)).forcing
        forcing = fa._attach_natural_aerosol(
            ForcingData.zeros((self.nlon, self.nlat)), cfg, self.coords)
        forcing = forcing.select(
            DateData.set_date(jdt.to_datetime("2031-04-16T00:00:00")))
        ncols, nlev = self.nlon * self.nlat, 10
        state = SimpleNamespace(temperature=jnp.zeros((nlev, ncols)))
        half = jnp.broadcast_to(jnp.linspace(30000.0, 0.0, nlev + 1)[:, None],
                                (nlev + 1, ncols))
        diagnostics = {"aerosol": AerosolData.zeros((ncols,), nlev,
                                                    n_bnd_sw=1, n_bnd_lw=0),
                       "height_half": half}
        _, out = Macv2NaturalAerosol()(state, diagnostics, forcing, None)
        i550 = SW_WAVELENGTH_NM.index(550)
        april = self._read()
        expected = (april["aod_sw_coarse"].values[3, i550]
                    + april["aod_sw_fine"].values[3, i550]).reshape(-1)
        np.testing.assert_allclose(
            out["aerosol"].aod_sw_per_band[0].sum(axis=0), expected, rtol=1e-4)
        np.testing.assert_allclose(
            out["natural_od550_coarse"],
            april["aod_sw_coarse"].values[3, i550].reshape(-1), rtol=1e-5)


class TestPythonDoor(unittest.TestCase):
    def test_from_bundles_asks_for_the_background_with_the_plumes_only(self):
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        seen = {}

        def capture(cfg, _coords):
            seen["forcing"] = dict(cfg.forcing)
            seen["physics"] = dict(cfg.physics)
            return None

        for aerosol, kwargs, key, composed in (
                ("macv2sp", {}, "auto", True),
                ("macv2sp", {"natural_aerosol": False}, None, False),
                ("jam", {}, None, False),
                (None, {}, None, False)):
            with self.subTest(aerosol=aerosol, **kwargs), \
                    patch.object(fa, "build_forcing", side_effect=capture), \
                    patch("jcm.runners.warn_emission_config_traps"):
                ForcingData.from_bundles(coords, aerosol=aerosol, surface=None,
                                         **kwargs)
                self.assertEqual(seen["forcing"]["natural_aerosol_file"], key)
                self.assertEqual(
                    fa._composes_natural_aerosol(
                        OmegaConf.create({"physics": seen["physics"]})),
                    composed)


if __name__ == "__main__":
    unittest.main()
