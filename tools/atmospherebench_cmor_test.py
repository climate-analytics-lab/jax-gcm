"""Unit tests for ``tools/atmospherebench_cmor.py``."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
import xarray as xr

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
            "surface_exchange.tile_fraction.2": (("time", "lon", "lat"), land_frac),
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
            self.assertIn("sftlf_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc", written)
            self.assertIn("areacella_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc", written)

            # Check 2D variable dims, lat ordering, CF attrs, and derived pr sum.
            with xr.open_dataset(outdir / "pr_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as pr_ds:
                self.assertEqual(pr_ds["pr"].dims, ("time", "lat", "lon"))
                self.assertEqual(pr_ds["pr"].attrs["units"], "kg m-2 s-1")
                self.assertEqual(pr_ds["pr"].attrs["standard_name"], "precipitation_flux")
                self.assertEqual(pr_ds["lat"].attrs["standard_name"], "latitude")
                self.assertEqual(pr_ds["lon"].attrs["standard_name"], "longitude")
                self.assertEqual(pr_ds["time"].attrs["standard_name"], "time")
                self.assertEqual(pr_ds["time"].attrs["bounds"], "time_bounds")
                self.assertIn("time_bounds", pr_ds.data_vars)
                self.assertTrue(np.all(np.diff(pr_ds["lat"].values) > 0))
                np.testing.assert_allclose(pr_ds["pr"].values, 3.5e-5, rtol=1e-5)

            # Check clt scaled to percent.
            with xr.open_dataset(outdir / "clt_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as clt_ds:
                self.assertEqual(clt_ds["clt"].attrs["units"], "%")
                np.testing.assert_allclose(clt_ds["clt"].values, 62.0, rtol=1e-5)

            # Check column water vapor prw = sum(q * dp) / g (using jcm.analysis.column_burden's g=9.81).
            with xr.open_dataset(outdir / "prw_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as prw_ds:
                expected_prw = 0.005 * 100000.0 / 9.81
                np.testing.assert_allclose(prw_ds["prw"].values, expected_prw, rtol=1e-4)

            # Check 3D pressure-level variable dims and CF plev coordinate.
            with xr.open_dataset(outdir / "ta_Amon_JCM-ECHAM-1M_amip_r1i1p1f1_gn_197901-197902.nc") as ta_ds:
                self.assertEqual(ta_ds["ta"].dims, ("time", "plev", "lat", "lon"))
                self.assertEqual(ta_ds["plev"].attrs["standard_name"], "air_pressure")
                self.assertEqual(ta_ds["plev"].attrs["units"], "Pa")
                np.testing.assert_allclose(ta_ds["plev"].values, [85000.0, 50000.0, 20000.0])
                np.testing.assert_allclose(ta_ds["ta"].isel(plev=0).values, 280.0, rtol=1e-5)
                np.testing.assert_allclose(ta_ds["ta"].isel(plev=1).values, 255.0, rtol=1e-5)
                np.testing.assert_allclose(ta_ds["ta"].isel(plev=2).values, 220.0, rtol=1e-5)

            # Check fixed fields sftlf and areacella.
            with xr.open_dataset(outdir / "sftlf_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc") as sftlf_ds:
                self.assertEqual(sftlf_ds["sftlf"].dims, ("lat", "lon"))
                self.assertEqual(sftlf_ds["sftlf"].attrs["units"], "%")
                self.assertEqual(sftlf_ds["sftlf"].attrs["standard_name"], "land_area_fraction")
                # Ascending lat: southern half 0%, northern half 100%.
                np.testing.assert_allclose(sftlf_ds["sftlf"].isel(lat=0).values, 0.0)
                np.testing.assert_allclose(sftlf_ds["sftlf"].isel(lat=-1).values, 100.0)

            with xr.open_dataset(outdir / "areacella_fx_JCM-ECHAM-1M_amip_r1i1p1f1_gn.nc") as area_ds:
                self.assertEqual(area_ds["areacella"].dims, ("lat", "lon"))
                self.assertEqual(area_ds["areacella"].attrs["units"], "m2")
                self.assertEqual(area_ds["areacella"].attrs["standard_name"], "cell_area")
                expected_earth_area = 4.0 * np.pi * ab_cmor.EARTH_RADIUS_M**2
                np.testing.assert_allclose(
                    float(area_ds["areacella"].sum()), expected_earth_area, rtol=1e-6
                )

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

            # Non-member experiment writes flat under <out>/<experiment>/.
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


if __name__ == "__main__":
    unittest.main()
