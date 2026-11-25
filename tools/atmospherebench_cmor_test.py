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


class AtmosphereBenchCmorTest(unittest.TestCase):
    """Verify CMOR naming, CF metadata, pressure interpolation, and derived fields."""

    def test_monthly_conversion_writes_valid_cmor_files_and_fx(self):
        times = np.array(["1979-01-16", "1979-02-15"], dtype="datetime64[ns]")
        bounds = np.array(
            [["1979-01-01", "1979-02-01"], ["1979-02-01", "1979-03-01"]],
            dtype="datetime64[ns]",
        )
        ds = _make_synthetic_jcm_dataset(times, bounds, descending_lat=True)

        with tempfile.TemporaryDirectory() as tmp:
            outdir = pathlib.Path(tmp) / "amip" / "r1i1p1f1"
            written = ab_cmor.convert_dataset(
                ds,
                model="JCM-ECHAM-1M",
                experiment="amip",
                member="r1i1p1f1",
                table="Amon",
                outdir=outdir,
                plevs_hpa=(850.0, 500.0, 200.0),
            )
            self.assertIn("tas_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc", written)
            self.assertIn("pr_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc", written)
            self.assertIn("prw_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc", written)
            self.assertIn("ta_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc", written)
            self.assertIn("hur_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc", written)
            self.assertIn("zg_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc", written)
            self.assertIn("sftlf_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc", written)
            self.assertIn("areacella_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc", written)

            # Check 2D variable dims, lat ordering, CF attrs, and derived pr sum.
            with xr.open_dataset(outdir / "pr_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as pr_ds:
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
                self.assertTrue(np.all(np.diff(pr_ds["lat"].values) > 0))
                np.testing.assert_allclose(pr_ds["pr"].values, 3.5e-5, rtol=1e-5)

            # Check flux positive direction attributes.
            with xr.open_dataset(outdir / "rlut_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as rlut_ds:
                self.assertEqual(rlut_ds["rlut"].attrs["positive"], "up")
            with xr.open_dataset(outdir / "rsdt_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as rsdt_ds:
                self.assertEqual(rsdt_ds["rsdt"].attrs["positive"], "down")

            # Check clt scaled to percent.
            with xr.open_dataset(outdir / "clt_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as clt_ds:
                self.assertEqual(clt_ds["clt"].attrs["units"], "%")
                np.testing.assert_allclose(clt_ds["clt"].values, 62.0, rtol=1e-5)

            # Check column water vapor prw = sum(q * dp) / g (using jcm.constants.grav).
            with xr.open_dataset(outdir / "prw_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as prw_ds:
                expected_prw = 0.005 * 100000.0 / c.grav
                np.testing.assert_allclose(prw_ds["prw"].values, expected_prw, rtol=1e-4)

            # Check 3D pressure-level variables (ta, hur, zg) and CF plev coordinate.
            with xr.open_dataset(outdir / "ta_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as ta_ds:
                self.assertEqual(ta_ds["ta"].dims, ("time", "plev", "lat", "lon"))
                self.assertEqual(ta_ds["plev"].attrs["standard_name"], "air_pressure")
                self.assertEqual(ta_ds["plev"].attrs["units"], "Pa")
                np.testing.assert_allclose(ta_ds["plev"].values, [85000.0, 50000.0, 20000.0])
                np.testing.assert_allclose(ta_ds["ta"].isel(plev=0).values, 280.0, rtol=1e-5)
                np.testing.assert_allclose(ta_ds["ta"].isel(plev=1).values, 255.0, rtol=1e-5)
                np.testing.assert_allclose(ta_ds["ta"].isel(plev=2).values, 220.0, rtol=1e-5)

            with xr.open_dataset(outdir / "hur_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as hur_ds:
                self.assertEqual(hur_ds["hur"].attrs["units"], "%")
                np.testing.assert_allclose(hur_ds["hur"].values, 65.0, rtol=1e-5)

            with xr.open_dataset(outdir / "zg_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as zg_ds:
                np.testing.assert_allclose(zg_ds["zg"].isel(plev=0).values, 1500.0, rtol=1e-5)
                np.testing.assert_allclose(zg_ds["zg"].isel(plev=1).values, 5500.0, rtol=1e-5)
                np.testing.assert_allclose(zg_ds["zg"].isel(plev=2).values, 12000.0, rtol=1e-5)

            # Check fixed fields sftlf and areacella.
            with xr.open_dataset(outdir / "sftlf_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc") as sftlf_ds:
                self.assertEqual(sftlf_ds["sftlf"].dims, ("lat", "lon"))
                self.assertEqual(sftlf_ds["sftlf"].attrs["units"], "%")
                self.assertEqual(sftlf_ds["sftlf"].attrs["standard_name"], "land_area_fraction")
                self.assertEqual(sftlf_ds["sftlf"].attrs["cell_methods"], "area: mean")
                # Ascending lat: southern half 0%, northern half 100%.
                np.testing.assert_allclose(sftlf_ds["sftlf"].isel(lat=0).values, 0.0)
                np.testing.assert_allclose(sftlf_ds["sftlf"].isel(lat=-1).values, 100.0)

            with xr.open_dataset(outdir / "areacella_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc") as area_ds:
                self.assertEqual(area_ds["areacella"].dims, ("lat", "lon"))
                self.assertEqual(area_ds["areacella"].attrs["units"], "m2")
                self.assertEqual(area_ds["areacella"].attrs["standard_name"], "cell_area")
                self.assertEqual(area_ds["areacella"].attrs["cell_methods"], "area: sum")
                expected_earth_area = 4.0 * np.pi * c.rearth**2
                np.testing.assert_allclose(
                    float(area_ds["areacella"].sum()), expected_earth_area, rtol=1e-6
                )

    def test_subdaily_resampling_and_cadence_checks(self):
        # 8 x 6-hourly steps spanning 2 days with a diurnal cycle in tas.
        times = np.arange(
            np.datetime64("1979-01-01T00:00"),
            np.datetime64("1979-01-03T00:00"),
            np.timedelta64(6, "h"),
        )
        bounds = np.stack([times, times + np.timedelta64(6, "h")], axis=-1)
        ds = _make_synthetic_jcm_dataset(times.astype("datetime64[ns]"), bounds.astype("datetime64[ns]"))
        diurnal = np.array([280.0, 286.0, 294.0, 288.0, 282.0, 288.0, 296.0, 290.0], dtype=np.float32)
        ds["surface.temperature_2m"].values[:] = diurnal[:, None, None]

        with tempfile.TemporaryDirectory() as tmp:
            outdir = pathlib.Path(tmp)
            # Sub-daily rejected for Amon.
            with self.assertRaisesRegex(ValueError, "Expected monthly"):
                ab_cmor.convert_dataset(
                    ds, model="JCM", experiment="amip", table="Amon", outdir=outdir
                )
            # Sub-daily rejected for day when resample_daily=False.
            with self.assertRaisesRegex(ValueError, "sub-daily cadence"):
                ab_cmor.convert_dataset(
                    ds,
                    model="JCM",
                    experiment="amip",
                    table="day",
                    outdir=outdir,
                    resample_daily=False,
                )
            # Default resample_daily=True aggregates to daily means + tasmax/tasmin.
            written = ab_cmor.convert_dataset(
                ds,
                model="JCM",
                experiment="amip",
                table="day",
                outdir=outdir,
                variables=["tas", "tasmax", "tasmin"],
                write_fx=False,
            )
            self.assertIn("tas_day_JCM_amip_r1i1p1f1_gn_19790101-19790102.nc", written)
            self.assertIn("tasmax_day_JCM_amip_r1i1p1f1_gn_19790101-19790102.nc", written)
            self.assertIn("tasmin_day_JCM_amip_r1i1p1f1_gn_19790101-19790102.nc", written)

            with xr.open_dataset(outdir / "tas_day_JCM_amip_r1i1p1f1_gn_19790101-19790102.nc") as d_tas:
                self.assertEqual(d_tas.sizes["time"], 2)
                np.testing.assert_allclose(d_tas["tas"].isel(time=0).values, 287.0, rtol=1e-5)
                np.testing.assert_allclose(d_tas["tas"].isel(time=1).values, 289.0, rtol=1e-5)
            with xr.open_dataset(outdir / "tasmax_day_JCM_amip_r1i1p1f1_gn_19790101-19790102.nc") as d_max:
                self.assertEqual(d_max["tasmax"].attrs["cell_methods"], "area: mean time: maximum")
                np.testing.assert_allclose(d_max["tasmax"].isel(time=0).values, 294.0, rtol=1e-5)
                np.testing.assert_allclose(d_max["tasmax"].isel(time=1).values, 296.0, rtol=1e-5)
            with xr.open_dataset(outdir / "tasmin_day_JCM_amip_r1i1p1f1_gn_19790101-19790102.nc") as d_min:
                self.assertEqual(d_min["tasmin"].attrs["cell_methods"], "area: mean time: minimum")
                np.testing.assert_allclose(d_min["tasmin"].isel(time=0).values, 280.0, rtol=1e-5)
                np.testing.assert_allclose(d_min["tasmin"].isel(time=1).values, 282.0, rtol=1e-5)

    def test_speedy_precip_and_dynamic_land_tile_discovery(self):
        times = np.array(["1979-01-16"], dtype="datetime64[ns]")
        bounds = np.array([["1979-01-01", "1979-02-01"]], dtype="datetime64[ns]")
        lat = np.linspace(-60.0, 60.0, 5)
        lon = np.linspace(0.0, 300.0, 6)
        land = np.where(lat[None, :] >= 0.0, 1.0, 0.0).astype(np.float32)
        ds = xr.Dataset(
            data_vars={
                "condensation.precls": (("time", "lon", "lat"), np.full((1, 6, 5), 0.02, dtype=np.float32)),
                "convection.precnv": (("time", "lon", "lat"), np.full((1, 6, 5), 0.01, dtype=np.float32)),
                "surface_exchange.tile_fraction.0": (
                    ("time", "lon", "lat"),
                    np.broadcast_to((1.0 - land)[None, :, :], (1, 6, 5)).copy(),
                    {"surface_type": "sea", "units": "1"},
                ),
                "surface_exchange.tile_fraction.1": (
                    ("time", "lon", "lat"),
                    np.broadcast_to(land[None, :, :], (1, 6, 5)).copy(),
                    {"surface_type": "land", "units": "1"},
                ),
                "time_bounds": (("time", "bounds"), bounds),
            },
            coords={"time": ("time", times, {"bounds": "time_bounds"}), "lon": lon, "lat": lat},
        )
        with tempfile.TemporaryDirectory() as tmp:
            outdir = pathlib.Path(tmp)
            written = ab_cmor.convert_dataset(
                ds, model="JCM-SPEEDY", experiment="amip", table="Amon", outdir=outdir
            )
            self.assertIn("pr_Amon_JCM-SPEEDY_amip_r1i1p1f1_gn_197901-197901.nc", written)
            self.assertIn("sftlf_fx_JCM-SPEEDY_amip_r1i1p1f1_gn.nc", written)
            with xr.open_dataset(outdir / "pr_Amon_JCM-SPEEDY_amip_r1i1p1f1_gn_197901-197901.nc") as pr_ds:
                np.testing.assert_allclose(pr_ds["pr"].values, 3e-5, rtol=1e-5)
            with xr.open_dataset(outdir / "sftlf_fx_JCM-SPEEDY_amip_r1i1p1f1_gn.nc") as sftlf_ds:
                np.testing.assert_allclose(sftlf_ds["sftlf"].isel(lat=0).values, 0.0)
                np.testing.assert_allclose(sftlf_ds["sftlf"].isel(lat=-1).values, 100.0)

    def test_fallbacks_geopotential_tas_percent_units_and_preinterp_plev(self):
        times = np.array(["1979-01-16"], dtype="datetime64[ns]")
        bounds = np.array([["1979-01-01", "1979-02-01"]], dtype="datetime64[ns]")
        ds = _make_synthetic_jcm_dataset(times, bounds)
        # Drop 2m tas and height_full; provide geopotential and already-% clt/hur.
        ds = ds.drop_vars(["surface.temperature_2m", "height_full"])
        ds["geopotential"] = (
            ds["temperature"].dims,
            np.broadcast_to(
                (np.array([100.0, 1500.0, 5500.0, 12000.0], dtype=np.float32) * c.grav)[
                    None, :, None, None
                ],
                ds["temperature"].shape,
            ).copy(),
        )
        ds["clouds.total_cloud_cover"] = ds["clouds.total_cloud_cover"] * 100.0
        ds["clouds.total_cloud_cover"].attrs["units"] = "%"
        ds["relative_humidity"] = ds["relative_humidity"] * 100.0
        ds["relative_humidity"].attrs["units"] = "%"

        with tempfile.TemporaryDirectory() as tmp:
            outdir = pathlib.Path(tmp)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ab_cmor.convert_dataset(
                    ds,
                    model="JCM",
                    experiment="amip",
                    table="Amon",
                    outdir=outdir,
                    variables=["tas", "clt", "hur", "zg"],
                    plevs_hpa=(850.0, 500.0),
                )
            self.assertTrue(any("lowest-model-level" in str(w.message) for w in caught))

            with xr.open_dataset(outdir / "tas_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc") as tas_ds:
                np.testing.assert_allclose(tas_ds["tas"].values, 290.0, rtol=1e-5)
            with xr.open_dataset(outdir / "clt_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc") as clt_ds:
                np.testing.assert_allclose(clt_ds["clt"].values, 62.0, rtol=1e-5)
            with xr.open_dataset(outdir / "hur_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc") as hur_ds:
                np.testing.assert_allclose(hur_ds["hur"].values, 65.0, rtol=1e-5)
            with xr.open_dataset(outdir / "zg_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc") as zg_ds:
                np.testing.assert_allclose(zg_ds["zg"].isel(plev=0).values, 1500.0, rtol=1e-5)
                np.testing.assert_allclose(zg_ds["zg"].isel(plev=1).values, 5500.0, rtol=1e-5)

            # Pre-interpolated plev in hPa should normalize to Pa and subset to --plevs.
            pre_ds = xr.Dataset(
                data_vars={"temperature": (("time", "plev", "lat", "lon"), np.ones((1, 3, 2, 2), dtype=np.float32) * 260.0)},
                coords={
                    "time": times,
                    "plev": ("plev", [850.0, 500.0, 200.0], {"units": "hPa"}),
                    "lat": [-10.0, 10.0],
                    "lon": [0.0, 180.0],
                },
            )
            outdir_pre = pathlib.Path(tmp) / "pre"
            ab_cmor.convert_dataset(
                pre_ds,
                model="JCM",
                experiment="amip",
                table="Amon",
                outdir=outdir_pre,
                variables=["ta"],
                plevs_hpa=(850.0, 200.0),
            )
            with xr.open_dataset(outdir_pre / "ta_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc") as ta_pre:
                self.assertEqual(ta_pre["plev"].attrs["units"], "Pa")
                np.testing.assert_allclose(ta_pre["plev"].values, [85000.0, 20000.0])

    def test_dry_run_has_no_filesystem_side_effects_and_year_slicing(self):
        times = np.array(["1978-12-16", "1979-01-16", "1980-01-16"], dtype="datetime64[ns]")
        bounds = np.array(
            [
                ["1978-12-01", "1979-01-01"],
                ["1979-01-01", "1979-02-01"],
                ["1980-01-01", "1980-02-01"],
            ],
            dtype="datetime64[ns]",
        )
        ds = _make_synthetic_jcm_dataset(times, bounds)
        with tempfile.TemporaryDirectory() as tmp:
            dry_dir = pathlib.Path(tmp) / "should_not_exist"
            written = ab_cmor.convert_dataset(
                ds,
                model="JCM",
                experiment="amip",
                table="Amon",
                outdir=dry_dir,
                variables=["tas"],
                start_year=1979,
                end_year=1979,
                dry_run=True,
            )
            self.assertFalse(dry_dir.exists())
            self.assertEqual(written, ["tas_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc"])

    def test_cli_discovers_run_directory_monthly_and_daily_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run_dir = root / "jcm-run"
            run_dir.mkdir()

            # Write two monthly files and one 2-day daily chunk file.
            m1 = _make_synthetic_jcm_dataset(
                np.array(["1979-01-16"], dtype="datetime64[ns]"),
                np.array([["1979-01-01", "1979-02-01"]], dtype="datetime64[ns]"),
            )
            m2 = _make_synthetic_jcm_dataset(
                np.array(["1979-02-15"], dtype="datetime64[ns]"),
                np.array([["1979-02-01", "1979-03-01"]], dtype="datetime64[ns]"),
            )
            d1 = _make_synthetic_jcm_dataset(
                np.array(["1979-01-01", "1979-01-02"], dtype="datetime64[ns]"),
                np.array(
                    [["1979-01-01", "1979-01-02"], ["1979-01-02", "1979-01-03"]],
                    dtype="datetime64[ns]",
                ),
            )
            m1.to_netcdf(run_dir / "jcm_monthly_1979-01.nc")
            m2.to_netcdf(run_dir / "jcm_monthly_1979-02.nc")
            d1.to_netcdf(run_dir / "jcm_day2.nc")

            out_root = root / "submission"
            rc = ab_cmor.main(
                [
                    str(run_dir),
                    "--model",
                    "JCM-ECHAM-1M",
                    "--experiment",
                    "amip",
                    "--member",
                    "r2i1p1f1",
                    "--plevs",
                    "850,500",
                    "--daily-plevs",
                    "850,200",
                    "--out",
                    str(out_root),
                ]
            )
            self.assertEqual(rc, 0)
            member_dir = out_root / "amip" / "r2i1p1f1"
            self.assertTrue(
                (member_dir / "pr_Amon_JCM-ECHAM-1M_amip_r2i1p1f1_gn_197901-197902.nc").is_file()
            )
            self.assertTrue(
                (member_dir / "pr_day_JCM-ECHAM-1M_amip_r2i1p1f1_gn_19790101-19790102.nc").is_file()
            )
            self.assertTrue(
                (member_dir / "ua_day_JCM-ECHAM-1M_amip_r2i1p1f1_gn_19790101-19790102.nc").is_file()
            )
            self.assertTrue(
                (member_dir / "sftlf_fx_JCM-ECHAM-1M_amip_r2i1p1f1_gn.nc").is_file()
            )

            # Non-member experiment writes flat under <out>/<experiment>/, and
            # --variables does not unexpectedly emit fx files.
            out_p4k = root / "submission_p4k"
            rc2 = ab_cmor.main(
                [
                    str(run_dir / "jcm_monthly_1979-01.nc"),
                    "--model",
                    "JCM-ECHAM-1M",
                    "--experiment",
                    "amip-p4K",
                    "--variables",
                    "ts,rsdt,rsut,rlut",
                    "--out",
                    str(out_p4k),
                ]
            )
            self.assertEqual(rc2, 0)
            self.assertTrue(
                (out_p4k / "amip-p4K" / "ts_Amon_JCM-ECHAM-1M_amip-p4K_r1i1p1f1_gn_197901-197901.nc").is_file()
            )
            self.assertFalse(
                (out_p4k / "amip-p4K" / "sftlf_fx_JCM-ECHAM-1M_amip-p4K_r1i1p1f1_gn.nc").exists()
            )

            # --flat writes directly into --out.
            out_flat = root / "flat_out"
            rc3 = ab_cmor.main(
                [
                    str(run_dir / "jcm_monthly_1979-01.nc"),
                    "--model",
                    "JCM-ECHAM-1M",
                    "--experiment",
                    "amip",
                    "--flat",
                    "--no-fx",
                    "--variables",
                    "tas",
                    "--out",
                    str(out_flat),
                ]
            )
            self.assertEqual(rc3, 0)
            self.assertTrue(
                (out_flat / "tas_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197901.nc").is_file()
            )

            # Mixed explicit monthly + daily files partition cleanly.
            grouped = ab_cmor.discover_inputs(
                [run_dir / "jcm_monthly_1979-01.nc", run_dir / "jcm_day2.nc"],
                freq="both",
            )
            self.assertEqual(set(grouped), {"Amon", "day"})

            # --freq both fails if one frequency is missing.
            with self.assertRaises(FileNotFoundError):
                ab_cmor.discover_inputs([run_dir / "jcm_monthly_1979-01.nc"], freq="both")

            # Multiple prefixes in one directory raise ValueError unless --prefix is given.
            m1.to_netcdf(run_dir / "other_monthly_1979-01.nc")
            with self.assertRaisesRegex(ValueError, "Multiple JCM run prefixes"):
                ab_cmor.discover_inputs([run_dir])
            filtered = ab_cmor.discover_inputs([run_dir], prefix="other")
            self.assertEqual(filtered["Amon"], [run_dir / "other_monthly_1979-01.nc"])

    def test_validation_rejects_bad_tokens(self):
        times = np.array(["1979-01-16"], dtype="datetime64[ns]")
        bounds = np.array([["1979-01-01", "1979-02-01"]], dtype="datetime64[ns]")
        ds = _make_synthetic_jcm_dataset(times, bounds)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ab_cmor.convert_dataset(
                    ds, model="bad_name", experiment="amip", outdir=pathlib.Path(tmp)
                )
            with self.assertRaises(ValueError):
                ab_cmor.convert_dataset(
                    ds, model="JCM", experiment="amip", member="mem1", outdir=pathlib.Path(tmp)
                )
            with self.assertRaises(ValueError):
                ab_cmor.convert_dataset(
                    ds, model="JCM", experiment="amip", variables=["not_a_var"], outdir=pathlib.Path(tmp)
                )

    def test_aerocom_and_atmospherebench_share_clt_source_priority(self):
        """Both CMOR exporters select the same ``clt`` source field."""
        from tools import aerocom_cmor

        times = np.array(["1979-01-16"], dtype="datetime64[ns]")
        bounds = np.array([["1979-01-01", "1979-02-01"]], dtype="datetime64[ns]")
        ds = _make_synthetic_jcm_dataset(times, bounds)
        # Deliberately set radiation.total_cloud_cover differently from
        # clouds.total_cloud_cover (0.62) to prove both tools pick the same source.
        ds["radiation.total_cloud_cover"] = (
            ds["clouds.total_cloud_cover"].dims,
            np.full(ds["clouds.total_cloud_cover"].shape, 0.72, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ab_dir = root / "ab"
            ac_dir = root / "ac"
            ab_cmor.convert_dataset(
                ds, model="JCM", experiment="amip", table="Amon",
                variables=["clt"], outdir=ab_dir,
            )
            ac_written, _ = aerocom_cmor.convert(
                ds, "JCM", "CTRL", "1979", "monthly", ac_dir,
            )
            ac_clt_file = next(f for f in ac_written if "_clt_" in f)
            with (
                xr.open_dataset(ab_dir / "clt_Amon_JCM_amip_r1i1p1f1_gn_197901-197901.nc") as ab_ds,
                xr.open_dataset(ac_dir / ac_clt_file) as ac_ds,
            ):
                np.testing.assert_allclose(ab_ds["clt"].values, 62.0, rtol=1e-5)
                np.testing.assert_allclose(ac_ds["clt"].values, 0.62, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
