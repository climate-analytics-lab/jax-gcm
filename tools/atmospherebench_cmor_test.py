"""Unit tests for ``tools/atmospherebench_cmor.py``."""

from __future__ import annotations

import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import xarray as xr

import jcm.constants as c
from tools import atmospherebench_cmor as ab_cmor


def _make_synthetic_jcm_dataset(
    times: np.ndarray,
    bounds: np.ndarray,
    *,
    descending_lat: bool = False,
) -> xr.Dataset:
    """Create a realistic JCM output Dataset in JCM's native ``(..., lon, lat)`` order."""
    x, _ = np.polynomial.legendre.leggauss(8)
    lat = np.degrees(np.arcsin(x))
    if descending_lat:
        lat = lat[::-1]
    lon = np.arange(16, dtype=float) * (360.0 / 16.0)
    nt, nlev, nlon, nlat = len(times), 4, len(lon), len(lat)

    # Surface-first pressure levels: 1000, 850, 500, 200 hPa in Pa.
    p_1d = np.array([100000.0, 85000.0, 50000.0, 20000.0], dtype=np.float32)
    p_full = np.broadcast_to(p_1d[None, :, None, None], (nt, nlev, nlon, nlat)).copy()
    dp_1d = np.array([15000.0, 25000.0, 35000.0, 25000.0], dtype=np.float32)
    dp = np.broadcast_to(dp_1d[None, :, None, None], (nt, nlev, nlon, nlat)).copy()

    t_3d = np.broadcast_to(
        np.array([290.0, 280.0, 255.0, 220.0], dtype=np.float32)[None, :, None, None],
        (nt, nlev, nlon, nlat),
    ).copy()
    q_3d = np.full((nt, nlev, nlon, nlat), 0.005, dtype=np.float32)
    rh_3d = np.full((nt, nlev, nlon, nlat), 0.65, dtype=np.float32)
    u_3d = np.full((nt, nlev, nlon, nlat), 10.0, dtype=np.float32)
    v_3d = np.full((nt, nlev, nlon, nlat), -2.0, dtype=np.float32)
    z_3d = np.broadcast_to(
        np.array([100.0, 1500.0, 5500.0, 12000.0], dtype=np.float32)[None, :, None, None],
        (nt, nlev, nlon, nlat),
    ).copy()

    # Land mask: northern half is land (1.0), southern half is ocean (0.0).
    land_2d = np.where(lat[None, :] > 0.0, 1.0, 0.0).astype(np.float32)
    land_frac = np.broadcast_to(land_2d[None, :, :], (nt, nlon, nlat)).copy()

    return xr.Dataset(
        data_vars={
            "temperature": (("time", "level", "lon", "lat"), t_3d),
            "specific_humidity": (("time", "level", "lon", "lat"), q_3d),
            "relative_humidity": (("time", "level", "lon", "lat"), rh_3d),
            "u_wind": (("time", "level", "lon", "lat"), u_3d),
            "v_wind": (("time", "level", "lon", "lat"), v_3d),
            "height_full": (("time", "level", "lon", "lat"), z_3d),
            "pressure_full": (("time", "level", "lon", "lat"), p_full),
            "pressure_thickness": (("time", "level", "lon", "lat"), dp),
            "surface_pressure": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 101325.0, dtype=np.float32)),
            "surface.temperature_2m": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 288.0, dtype=np.float32)),
            "surface.surface_temperature": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 289.0, dtype=np.float32)),
            "surface.humidity_2m": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 0.01, dtype=np.float32)),
            "surface.evaporation": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 3e-5, dtype=np.float32)),
            "surface.latent_heat_flux": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 80.0, dtype=np.float32)),
            "surface.sensible_heat_flux_dry": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 20.0, dtype=np.float32)),
            "clouds.precip_rain": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 2e-5, dtype=np.float32)),
            "clouds.precip_snow": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 0.5e-5, dtype=np.float32)),
            "convection.precip_conv": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 1e-5, dtype=np.float32)),
            "clouds.total_cloud_cover": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 0.62, dtype=np.float32)),
            "radiation.toa_sw_down": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 340.0, dtype=np.float32)),
            "radiation.toa_sw_up": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 100.0, dtype=np.float32)),
            "radiation.toa_sw_up_clear": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 55.0, dtype=np.float32)),
            "radiation.toa_lw_up": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 240.0, dtype=np.float32)),
            "radiation.toa_lw_up_clear": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 265.0, dtype=np.float32)),
            "radiation.surface_sw_down": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 190.0, dtype=np.float32)),
            "radiation.surface_sw_up": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 25.0, dtype=np.float32)),
            "radiation.surface_lw_down": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 340.0, dtype=np.float32)),
            "radiation.surface_lw_up": (("time", "lon", "lat"), np.full((nt, nlon, nlat), 395.0, dtype=np.float32)),
            "surface_exchange.tile_fraction.2": (
                ("time", "lon", "lat"),
                land_frac,
                {"surface_type": "land", "units": "1"},
            ),
            "time_bounds": (("time", "bounds"), bounds),
        },
        coords={
            "time": ("time", times, {"bounds": "time_bounds"}),
            "level": np.arange(nlev),
            "lon": lon,
            "lat": lat,
        },
    )


def _monthly_dataset(*months: str, **kwargs) -> xr.Dataset:
    """Synthetic monthly-mean Dataset with one step per ``YYYY-MM`` in ``months``."""
    starts = np.array(months or ("1979-01",), dtype="datetime64[M]")
    bounds = np.stack([starts, starts + np.timedelta64(1, "M")], axis=-1).astype("datetime64[ns]")
    times = bounds[:, 0] + np.timedelta64(15, "D")
    return _make_synthetic_jcm_dataset(times, bounds, **kwargs)


def _stepped_dataset(start: str, n_steps: int, hours: int) -> xr.Dataset:
    """Synthetic Dataset of ``n_steps`` consecutive ``hours``-long intervals."""
    step = np.timedelta64(hours, "h")
    times = (np.datetime64(start, "h") + step * np.arange(n_steps)).astype("datetime64[ns]")
    return _make_synthetic_jcm_dataset(times, np.stack([times, times + step], axis=-1))


class _CmorTestCase(unittest.TestCase):
    """Convert into a per-test temporary directory and read the results back."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        self.outdir = self.tmp / "out"

    def convert(self, ds: xr.Dataset, **kwargs) -> dict[str, str]:
        """Run ``convert_dataset`` into ``self.outdir``; return ``{cmor: filename}``."""
        kwargs = {"model": "JCM", "experiment": "amip", "outdir": self.outdir, **kwargs}
        self.written, self.missing = ab_cmor.convert_dataset(ds, **kwargs)
        return self.written

    def read(self, cmor: str) -> xr.Dataset:
        """Load the file written for ``cmor`` by the last :meth:`convert`."""
        with xr.open_dataset(self.outdir / self.written[cmor]) as out:
            return out.load()


class MonthlyConversionTest(_CmorTestCase):
    """CMOR naming, CF metadata, pressure interpolation, and derived fields."""

    def setUp(self):
        super().setUp()
        self.convert(
            _monthly_dataset("1979-01", "1979-02", descending_lat=True),
            model="JCM-ECHAM-1M",
            member="r1i1p1f1",
            table="Amon",
            plevs_hpa=(850.0, 500.0, 200.0),
        )

    def test_filenames_follow_cmor_convention(self):
        for var in ("tas", "pr", "prw", "ta", "hur", "zg"):
            self.assertEqual(
                self.written[var],
                f"{var}_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc",
            )
        for var in ("sftlf", "areacella"):
            self.assertEqual(self.written[var], f"{var}_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc")
        for fname in self.written.values():
            self.assertTrue((self.outdir / fname).is_file(), fname)

    def test_2d_variable_layout_and_cf_metadata(self):
        pr_ds = self.read("pr")
        self.assertEqual(pr_ds["pr"].dims, ("time", "lat", "lon"))
        self.assertEqual(pr_ds["pr"].attrs["units"], "kg m-2 s-1")
        self.assertEqual(pr_ds["pr"].attrs["standard_name"], "precipitation_flux")
        self.assertEqual(pr_ds["pr"].attrs["cell_methods"], "area: time: mean")
        self.assertEqual(pr_ds.attrs["variable_id"], "pr")
        self.assertEqual(pr_ds.attrs["realm"], "atmos")
        self.assertEqual(pr_ds["lat"].attrs["standard_name"], "latitude")
        self.assertEqual(pr_ds["lon"].attrs["standard_name"], "longitude")
        self.assertEqual(pr_ds["time"].attrs["standard_name"], "time")
        self.assertEqual(pr_ds["time"].attrs["bounds"], "time_bounds")
        self.assertIn("time_bounds", pr_ds.data_vars)
        # The input latitude is north-to-south; CMOR output is south-to-north.
        self.assertTrue(np.all(np.diff(pr_ds["lat"].values) > 0))

    def test_pr_sums_rain_snow_and_convective_precipitation(self):
        np.testing.assert_allclose(self.read("pr")["pr"].values, 3.5e-5, rtol=1e-5)

    def test_flux_positive_direction(self):
        self.assertEqual(self.read("rlut")["rlut"].attrs["positive"], "up")
        self.assertEqual(self.read("rsdt")["rsdt"].attrs["positive"], "down")

    def test_clt_is_scaled_to_percent(self):
        clt = self.read("clt")["clt"]
        self.assertEqual(clt.attrs["units"], "%")
        np.testing.assert_allclose(clt.values, 62.0, rtol=1e-5)

    def test_prw_is_column_integral_of_specific_humidity(self):
        # prw = sum(q * dp) / g, using the live jcm.constants.grav.
        np.testing.assert_allclose(
            self.read("prw")["prw"].values, 0.005 * 100000.0 / c.grav, rtol=1e-4
        )

    def test_3d_variables_are_interpolated_to_cf_pressure_levels(self):
        ta_ds = self.read("ta")
        self.assertEqual(ta_ds["ta"].dims, ("time", "plev", "lat", "lon"))
        self.assertEqual(ta_ds["plev"].attrs["standard_name"], "air_pressure")
        self.assertEqual(ta_ds["plev"].attrs["units"], "Pa")
        np.testing.assert_allclose(ta_ds["plev"].values, [85000.0, 50000.0, 20000.0])
        zg = self.read("zg")["zg"]
        for i, (want_ta, want_zg) in enumerate([(280.0, 1500.0), (255.0, 5500.0), (220.0, 12000.0)]):
            np.testing.assert_allclose(ta_ds["ta"].isel(plev=i).values, want_ta, rtol=1e-5)
            np.testing.assert_allclose(zg.isel(plev=i).values, want_zg, rtol=1e-5)

    def test_hur_is_scaled_to_percent(self):
        hur = self.read("hur")["hur"]
        self.assertEqual(hur.attrs["units"], "%")
        np.testing.assert_allclose(hur.values, 65.0, rtol=1e-5)

    def test_fx_land_fraction(self):
        sftlf = self.read("sftlf")["sftlf"]
        self.assertEqual(sftlf.dims, ("lat", "lon"))
        self.assertEqual(sftlf.attrs["units"], "%")
        self.assertEqual(sftlf.attrs["standard_name"], "land_area_fraction")
        self.assertEqual(sftlf.attrs["cell_methods"], "area: mean")
        # Ascending lat: southern half 0%, northern half 100%.
        np.testing.assert_allclose(sftlf.isel(lat=0).values, 0.0)
        np.testing.assert_allclose(sftlf.isel(lat=-1).values, 100.0)

    def test_fx_cell_area_tiles_the_sphere(self):
        area = self.read("areacella")["areacella"]
        self.assertEqual(area.dims, ("lat", "lon"))
        self.assertEqual(area.attrs["units"], "m2")
        self.assertEqual(area.attrs["standard_name"], "cell_area")
        self.assertEqual(area.attrs["cell_methods"], "area: sum")
        np.testing.assert_allclose(float(area.sum()), 4.0 * np.pi * c.rearth**2, rtol=1e-6)


class CadenceTest(_CmorTestCase):
    """Cadence validation and sub-daily -> daily resampling."""

    def setUp(self):
        super().setUp()
        # 8 x 6-hourly steps spanning 2 days with a diurnal cycle in tas.
        self.ds = _stepped_dataset("1979-01-01", 8, hours=6)
        diurnal = np.array([280.0, 286.0, 294.0, 288.0, 282.0, 288.0, 296.0, 290.0], dtype=np.float32)
        self.ds["surface.temperature_2m"].values[:] = diurnal[:, None, None]

    def test_subdaily_input_is_rejected_for_amon(self):
        with self.assertRaisesRegex(ValueError, "Expected monthly"):
            self.convert(self.ds, table="Amon")

    def test_subdaily_input_is_rejected_without_resampling(self):
        with self.assertRaisesRegex(ValueError, "sub-daily cadence"):
            self.convert(self.ds, table="day", resample_daily=False)

    def test_monthly_input_is_rejected_for_day(self):
        with self.assertRaisesRegex(ValueError, "Expected daily or sub-daily"):
            self.convert(_monthly_dataset(), table="day")

    def test_subdaily_input_is_resampled_to_daily_mean_max_and_min(self):
        written = self.convert(
            self.ds, table="day", variables=["tas", "tasmax", "tasmin"], fx_vars=()
        )
        self.assertEqual(
            written,
            {v: f"{v}_day_JCM_amip_r1i1p1f1_gn_19790101-19790102.nc"
             for v in ("tas", "tasmax", "tasmin")},
        )
        expected = {
            "tas": ("area: time: mean", [287.0, 289.0]),
            "tasmax": ("area: mean time: maximum", [294.0, 296.0]),
            "tasmin": ("area: mean time: minimum", [280.0, 282.0]),
        }
        for var, (cell_methods, daily_values) in expected.items():
            da = self.read(var)[var]
            self.assertEqual(da.sizes["time"], 2)
            self.assertEqual(da.attrs["cell_methods"], cell_methods)
            for day, want in enumerate(daily_values):
                np.testing.assert_allclose(da.isel(time=day).values, want, rtol=1e-5)


class SourceFallbackTest(_CmorTestCase):
    """Fallback sources, unit handling, and pre-interpolated inputs."""

    def test_speedy_precipitation_and_land_tile_found_by_metadata(self):
        times = np.array(["1979-01-16"], dtype="datetime64[ns]")
        bounds = np.array([["1979-01-01", "1979-02-01"]], dtype="datetime64[ns]")
        lat = np.linspace(-60.0, 60.0, 5)
        lon = np.linspace(0.0, 300.0, 6)
        land = np.where(lat[None, :] >= 0.0, 1.0, 0.0).astype(np.float32)

        def tile(fraction, surface_type):
            return (
                ("time", "lon", "lat"),
                np.broadcast_to(fraction[None, :, :], (1, 6, 5)).copy(),
                {"surface_type": surface_type, "units": "1"},
            )

        ds = xr.Dataset(
            data_vars={
                "condensation.precls": (("time", "lon", "lat"), np.full((1, 6, 5), 0.02, dtype=np.float32)),
                "convection.precnv": (("time", "lon", "lat"), np.full((1, 6, 5), 0.01, dtype=np.float32)),
                "surface_exchange.tile_fraction.0": tile(1.0 - land, "sea"),
                "surface_exchange.tile_fraction.1": tile(land, "land"),
                "time_bounds": (("time", "bounds"), bounds),
            },
            coords={"time": ("time", times, {"bounds": "time_bounds"}), "lon": lon, "lat": lat},
        )
        written = self.convert(ds, model="JCM-SPEEDY", table="Amon")
        self.assertEqual(written["pr"], "pr_Amon_JCM-SPEEDY_amip_r1i1p1f1_gn_197901-197901.nc")
        self.assertEqual(written["sftlf"], "sftlf_fx_JCM-SPEEDY_amip_r1i1p1f1_gn.nc")
        # SPEEDY precipitation is g m-2 s-1; the land tile is index 1, not ECHAM's 2.
        np.testing.assert_allclose(self.read("pr")["pr"].values, 3e-5, rtol=1e-5)
        sftlf = self.read("sftlf")["sftlf"]
        np.testing.assert_allclose(sftlf.isel(lat=0).values, 0.0)
        np.testing.assert_allclose(sftlf.isel(lat=-1).values, 100.0)

    def test_tas_falls_back_to_lowest_model_level_with_a_warning(self):
        ds = _monthly_dataset().drop_vars("surface.temperature_2m")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.convert(ds, variables=["tas"])
        self.assertTrue(any("lowest-model-level" in str(w.message) for w in caught))
        np.testing.assert_allclose(self.read("tas")["tas"].values, 290.0, rtol=1e-5)

    def test_sources_already_in_percent_are_not_rescaled(self):
        ds = _monthly_dataset()
        for name in ("clouds.total_cloud_cover", "relative_humidity"):
            ds[name] = ds[name] * 100.0
            ds[name].attrs["units"] = "%"
        self.convert(ds, variables=["clt", "hur"], plevs_hpa=(850.0, 500.0))
        np.testing.assert_allclose(self.read("clt")["clt"].values, 62.0, rtol=1e-5)
        np.testing.assert_allclose(self.read("hur")["hur"].values, 65.0, rtol=1e-5)

    def test_zg_falls_back_to_geopotential_over_gravity(self):
        ds = _monthly_dataset()
        ds["geopotential"] = ds["height_full"] * c.grav
        self.convert(ds.drop_vars("height_full"), variables=["zg"], plevs_hpa=(850.0, 500.0))
        zg = self.read("zg")["zg"]
        np.testing.assert_allclose(zg.isel(plev=0).values, 1500.0, rtol=1e-5)
        np.testing.assert_allclose(zg.isel(plev=1).values, 5500.0, rtol=1e-5)

    def test_preinterpolated_hpa_levels_become_pa_and_are_subset(self):
        pre_ds = xr.Dataset(
            data_vars={"temperature": (("time", "plev", "lat", "lon"), np.full((1, 3, 2, 2), 260.0, dtype=np.float32))},
            coords={
                "time": np.array(["1979-01-16"], dtype="datetime64[ns]"),
                "plev": ("plev", [850.0, 500.0, 200.0], {"units": "hPa"}),
                "lat": [-10.0, 10.0],
                "lon": [0.0, 180.0],
            },
        )
        self.convert(pre_ds, variables=["ta"], plevs_hpa=(850.0, 200.0))
        ta_pre = self.read("ta")
        self.assertEqual(ta_pre["plev"].attrs["units"], "Pa")
        np.testing.assert_allclose(ta_pre["plev"].values, [85000.0, 20000.0])

    def test_unavailable_variables_are_reported_and_warned_about(self):
        with self.assertWarnsRegex(UserWarning, "not available in input.*psl"):
            written = self.convert(_monthly_dataset(), variables=["tas", "psl"])
        self.assertEqual(list(written), ["tas"])
        self.assertEqual(self.missing, ["psl"])


class ConvertOptionsTest(_CmorTestCase):
    """Variable/fx selection, year slicing, dry runs, and argument validation."""

    def test_dry_run_reports_filenames_without_touching_the_filesystem(self):
        ds = _monthly_dataset("1978-12", "1979-01", "1980-01")
        written = self.convert(
            ds, variables=["tas"], start_year=1979, end_year=1979, dry_run=True
        )
        self.assertFalse(self.outdir.exists())
        # The year filter kept only 1979-01.
        self.assertEqual(written, {"tas": "tas_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc"})

    def test_fixed_fields_follow_variables_unless_fx_vars_is_given(self):
        ds = _monthly_dataset()
        self.assertEqual(set(self.convert(ds, variables=["tas"], dry_run=True)), {"tas"})
        self.assertEqual(
            set(self.convert(ds, variables=["tas", "sftlf"], dry_run=True)), {"tas", "sftlf"}
        )
        self.assertEqual(
            set(self.convert(ds, variables=["tas"], fx_vars=["areacella"], dry_run=True)),
            {"tas", "areacella"},
        )
        self.assertFalse(
            ab_cmor.FX_VARS & set(self.convert(ds, fx_vars=(), dry_run=True))
        )

    def test_bad_tokens_and_unknown_variables_are_rejected(self):
        ds = _monthly_dataset()
        for bad in ({"model": "bad_name"}, {"member": "mem1"}, {"variables": ["not_a_var"]},
                    {"table": "6hr"}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.convert(ds, **bad)

    def test_aerocom_and_atmospherebench_share_clt_source_priority(self):
        """Both CMOR exporters select the same ``clt`` source field."""
        from tools import aerocom_cmor

        ds = _monthly_dataset()
        # Deliberately set radiation.total_cloud_cover differently from
        # clouds.total_cloud_cover (0.62) to prove both tools pick the same source.
        ds["radiation.total_cloud_cover"] = (
            ds["clouds.total_cloud_cover"].dims,
            np.full(ds["clouds.total_cloud_cover"].shape, 0.72, dtype=np.float32),
        )
        self.convert(ds, table="Amon", variables=["clt"])
        np.testing.assert_allclose(self.read("clt")["clt"].values, 62.0, rtol=1e-5)

        ac_dir = self.tmp / "ac"
        ac_written, _ = aerocom_cmor.convert(ds, "JCM", "CTRL", "1979", "monthly", ac_dir)
        ac_clt_file = next(f for f in ac_written if "_clt_" in f)
        with xr.open_dataset(ac_dir / ac_clt_file) as ac_ds:
            np.testing.assert_allclose(ac_ds["clt"].values, 0.62, rtol=1e-5)


class CliTest(_CmorTestCase):
    """Input discovery and the submission directory layout."""

    def setUp(self):
        super().setUp()
        # Two monthly files and one 2-day daily chunk file.
        self.run_dir = self.tmp / "jcm-run"
        self.run_dir.mkdir()
        self.monthly = self.run_dir / "jcm_monthly_1979-01.nc"
        self.daily = self.run_dir / "jcm_day2.nc"
        _monthly_dataset("1979-01").to_netcdf(self.monthly)
        _monthly_dataset("1979-02").to_netcdf(self.run_dir / "jcm_monthly_1979-02.nc")
        _stepped_dataset("1979-01-01", 2, hours=24).to_netcdf(self.daily)

    def run_cli(self, *argv) -> None:
        self.assertEqual(
            ab_cmor.main([*map(str, argv), "--model", "JCM-ECHAM-1M", "--out", str(self.outdir)]),
            0,
        )

    def test_run_directory_writes_monthly_daily_and_fx_under_member_dir(self):
        self.run_cli(self.run_dir, "--experiment", "amip", "--member", "r2i1p1f1",
                     "--plevs", "850,500", "--daily-plevs", "850,200")
        member_dir = self.outdir / "amip" / "r2i1p1f1"
        for fname in (
            "pr_Amon_JCM-ECHAM-1M_amip_r2i1p1f1_gn_197901-197902.nc",
            "pr_day_JCM-ECHAM-1M_amip_r2i1p1f1_gn_19790101-19790102.nc",
            "ua_day_JCM-ECHAM-1M_amip_r2i1p1f1_gn_19790101-19790102.nc",
            "sftlf_fx_JCM-ECHAM-1M_amip_r2i1p1f1_gn.nc",
        ):
            self.assertTrue((member_dir / fname).is_file(), fname)

    def test_single_member_experiment_layout_and_variables_emit_no_fx(self):
        self.run_cli(self.monthly, "--experiment", "amip-p4K", "--variables", "ts,rsdt,rsut,rlut")
        exp_dir = self.outdir / "amip-p4K"
        self.assertTrue((exp_dir / "ts_Amon_JCM-ECHAM-1M_amip-p4K_r1i1p1f1_gn_197901-197901.nc").is_file())
        self.assertFalse((exp_dir / "sftlf_fx_JCM-ECHAM-1M_amip-p4K_r1i1p1f1_gn.nc").exists())

    def test_flat_writes_directly_into_out(self):
        self.run_cli(self.monthly, "--experiment", "amip", "--flat", "--no-fx", "--variables", "tas")
        self.assertEqual(
            [p.name for p in self.outdir.iterdir()],
            ["tas_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197901.nc"],
        )

    def test_explicit_monthly_and_daily_files_partition_by_name(self):
        grouped = ab_cmor.discover_inputs([self.monthly, self.daily], freq="both")
        self.assertEqual(grouped, {"Amon": [self.monthly], "day": [self.daily]})

    def test_freq_both_fails_when_one_frequency_is_missing(self):
        with self.assertRaises(FileNotFoundError):
            ab_cmor.discover_inputs([self.monthly], freq="both")

    def test_multiple_run_prefixes_require_prefix(self):
        other = self.run_dir / "other_monthly_1979-01.nc"
        _monthly_dataset("1979-01").to_netcdf(other)
        with self.assertRaisesRegex(ValueError, "Multiple JCM run prefixes"):
            ab_cmor.discover_inputs([self.run_dir])
        self.assertEqual(ab_cmor.discover_inputs([self.run_dir], prefix="other"), {"Amon": [other]})


if __name__ == "__main__":
    unittest.main()
