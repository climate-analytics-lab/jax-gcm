"""Tests for the pyses JAM forcing attachment (emissions/DMS/dust/oxidants).

Covers :func:`jcm.dycore.pyses.forcing.attach_jam_forcing` — the column
analogue of ``jcm.runners``' ``_attach_*`` helpers — with synthetic
reader-contract files on tiny regular grids, checking that every field
arrives on the ``(1, ncol)`` column layout non-zero (the exact gap that made
the first ne30 JAM year aerosol-dark: physics wired, forcing absent).
No dycore is needed: the attach helper takes the column lon/lats directly.
"""

import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from jcm.forcing import ForcingData, TimeSeries, WRAP_YEAR

# Tiny source grid and a handful of scattered "columns" inside it.
_LON = np.arange(0.0, 360.0, 45.0)            # 8
_LAT = np.array([-60.0, -30.0, 0.0, 30.0, 60.0, 80.0])   # 6, ascending
_COL_LON = np.array([10.0, 100.0, 200.0, 300.0, 355.0])
_COL_LAT = np.array([-45.0, 0.0, 15.0, 50.0, 70.0])
_NCOL = _COL_LON.size
_TIME = np.array([np.datetime64(f"2014-{m:02d}-15") for m in range(1, 13)])


def _attach(**kwargs):
    from jcm.dycore.pyses.forcing import attach_jam_forcing

    forcing = ForcingData.zeros(nodal_shape=(1, _NCOL))
    return attach_jam_forcing(forcing, _COL_LON, _COL_LAT, nlev=4, **kwargs)


def _write(tmp, name, ds):
    path = os.path.join(tmp, name)
    ds.to_netcdf(path)
    return path


class AttachJamForcingTest(unittest.TestCase):
    def test_emissions_attach_nonzero_columns(self):
        ds = xr.Dataset(
            {"emis_surface_combustion_so2": (
                ("time", "lon", "lat"),
                np.full((12, _LON.size, _LAT.size), 2.0e-12),
                {"units": "kg m-2 s-1"})},
            coords={"time": _TIME, "lon": _LON, "lat": _LAT},
        )
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _attach(emissions_file=_write(tmp, "emis.nc", ds))
        leaf = forcing.anthropogenic_emissions["emis_surface_combustion_so2"]
        self.assertIsInstance(leaf, TimeSeries)
        self.assertEqual(leaf.values.shape, (12, 1, _NCOL))
        self.assertEqual(int(leaf.align_mode), WRAP_YEAR)
        np.testing.assert_allclose(np.asarray(leaf.values), 2.0e-12)
        self.assertIsNone(forcing.prescribed_aerosol_emissions)

    def test_emissions_speciated_and_latlon_order_rejected(self):
        ds = xr.Dataset(
            {"aero_emis_m_so4_acc": (
                ("time", "lon", "lat"),
                np.full((12, _LON.size, _LAT.size), 1.0e-13))},
            coords={"time": _TIME, "lon": _LON, "lat": _LAT},
        )
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _attach(emissions_file=_write(tmp, "spec.nc", ds))
            self.assertEqual(
                forcing.prescribed_aerosol_emissions["m_so4_acc"].values.shape,
                (12, 1, _NCOL))
            # (lat, lon)-ordered fields must be rejected, not silently
            # transposed into the wrong columns.
            bad = xr.Dataset(
                {"emis_shipping_so2": (
                    ("time", "lat", "lon"),
                    np.zeros((12, _LAT.size, _LON.size)))},
                coords={"time": _TIME, "lat": _LAT, "lon": _LON},
            )
            with self.assertRaisesRegex(ValueError, "lon.*lat"):
                _attach(emissions_file=_write(tmp, "bad.nc", bad))

    def test_dms_converted_and_on_columns(self):
        # (time, lat, lon) with descending latitude — the reader flips it;
        # value 10 nmol/L should arrive converted to kg/m³ everywhere.
        ds = xr.Dataset(
            {"DMS_sea": (
                ("time", "lat", "lon"),
                np.full((12, _LAT.size, _LON.size), 10.0),
                {"units": "nanomol l-1"})},
            coords={"time": _TIME, "lat": _LAT[::-1], "lon": _LON},
        )
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _attach(dms_file=_write(tmp, "dms.nc", ds))
        leaf = forcing.dms_seawater
        self.assertEqual(leaf.values.shape, (12, 1, _NCOL))
        np.testing.assert_allclose(
            np.asarray(leaf.values), 10.0 * 1.0e-6 * 0.0621324, rtol=1e-6)

    def test_static_dust_map_on_columns(self):
        ds = xr.Dataset(
            {"pot_source": (("lat", "lon"),
                            np.full((_LAT.size, _LON.size), 0.5))},
            coords={"lat": _LAT, "lon": _LON},
        )
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _attach(dust_file=_write(tmp, "dust.nc", ds))
        self.assertEqual(forcing.dust_source.shape, (1, _NCOL))
        np.testing.assert_allclose(np.asarray(forcing.dust_source), 0.5)

    def test_oxidants_per_level_on_columns(self):
        nlev = 4
        # Distinct per-level values so we can check levels stay level-for-level
        # through the horizontal interpolation.
        base = np.arange(1, nlev + 1, dtype=float).reshape(1, nlev, 1, 1)
        data = np.broadcast_to(base * 1.0e-9,
                               (12, nlev, _LAT.size, _LON.size)).copy()
        ds = xr.Dataset(
            {f"{n}_VMR_avrg": (("time", "mlev", "lat", "lon"), data,
                               {"units": "mole/mole"})
             for n in ("OH", "NO3", "O3", "H2O2")},
            coords={"time": _TIME, "mlev": np.arange(1, nlev + 1),
                    "lat": _LAT, "lon": _LON},
        )
        ds["hybm"] = ("mlev", np.array([0.0, 0.1, 0.5, 1.0]))  # top→bottom
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _attach(oxidants_file=_write(tmp, "oxid.nc", ds))
        self.assertEqual(sorted(forcing.oxidant_vmr), ["h2o2", "no3", "o3", "oh"])
        oh = forcing.oxidant_vmr["oh"]
        self.assertEqual(oh.values.shape, (12, nlev, 1, _NCOL))
        np.testing.assert_allclose(
            np.asarray(oh.values[0, :, 0, 0]),
            np.arange(1, nlev + 1) * 1.0e-9, rtol=1e-6)

    def test_select_slices_all_jam_leaves(self):
        """`ForcingData.select` must slice the attached dict leaves too."""
        import jax_datetime as jdt

        from jcm.date import DateData

        ds_e = xr.Dataset(
            {"emis_biomass_burning_bc": (
                ("time", "lon", "lat"),
                np.full((12, _LON.size, _LAT.size), 3.0e-12))},
            coords={"time": _TIME, "lon": _LON, "lat": _LAT},
        )
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _attach(emissions_file=_write(tmp, "emis.nc", ds_e))
        date = DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(jdt.to_datetime("2014-07-01")),
            calendar="gregorian",
        )
        sliced = forcing.select(date, calendar="gregorian")
        leaf = sliced.anthropogenic_emissions["emis_biomass_burning_bc"]
        self.assertEqual(leaf.shape, (1, _NCOL))
        np.testing.assert_allclose(np.asarray(leaf), 3.0e-12)


if __name__ == "__main__":
    unittest.main()
