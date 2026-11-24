"""Phase D: prescribed anthropogenic emissions wired through ForcingData.

Fast tests cover the emissions-file contract round-trip (``read_anthropogenic_
emissions`` → ``ForcingData`` → ``select(date)`` slicing). A slow test runs the
full JAM stack on a T21 aquaplanet and checks the prescribed emission actually
raises the relevant aerosol burden vs an emission-free control.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from jcm.forcing import (
    ForcingData,
    read_anthropogenic_emissions,
    read_prescribed_aerosol_emissions,
)

_NLON, _NLAT = 64, 32  # T21 nodal shape
_NLEV = 20


def _synthetic_emissions_ds(nmonths: int = 12, so2=2.0e-11, bc=1.0e-11, oc=3.0e-11):
    """Build a minimal contract-conformant emissions Dataset on the T21 grid."""
    shape = (_NLON, _NLAT, nmonths)
    coords = {
        "lon": np.linspace(0, 360, _NLON, endpoint=False),
        "lat": np.linspace(-87, 87, _NLAT),
        "time": np.arange(nmonths),
    }
    data = {
        "emis_surface_combustion_so2": (("lon", "lat", "time"), np.full(shape, so2)),
        "emis_surface_combustion_bc": (("lon", "lat", "time"), np.full(shape, bc)),
        "emis_surface_combustion_oc": (("lon", "lat", "time"), np.full(shape, oc)),
    }
    return xr.Dataset(data, coords=coords)


class ContractRoundTripTest(unittest.TestCase):
    def _date(self):
        from jcm.date import DateData
        import jax_datetime as jdt
        return DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(jdt.to_datetime("2001-07-02")),
        )

    def test_returns_none_without_emis_vars(self):
        # A dataset with no emis_* variable yields None (so .copy is a no-op).
        ds = xr.Dataset({"sst": (("lon", "lat"), np.zeros((4, 4)))})
        self.assertIsNone(read_anthropogenic_emissions(ds))

    def test_reads_all_channels(self):
        emis = read_anthropogenic_emissions(_synthetic_emissions_ds(),
                                            align_mode="wrap_year")
        self.assertEqual(
            set(emis),
            {"emis_surface_combustion_so2", "emis_surface_combustion_bc",
             "emis_surface_combustion_oc"},
        )

    def test_select_slices_channels_to_grid(self):
        # After select(date), each per-channel TimeSeries collapses to the bare
        # (lon, lat) grid the term consumes (ravels to ncols = lon*lat).
        emis = read_anthropogenic_emissions(_synthetic_emissions_ds(),
                                            align_mode="wrap_year")
        forcing = ForcingData.zeros((_NLON, _NLAT)).copy(
            anthropogenic_emissions=emis)
        sliced = forcing.select(self._date())
        bc = sliced.anthropogenic_emissions["emis_surface_combustion_bc"]
        self.assertEqual(bc.shape, (_NLON, _NLAT))
        self.assertEqual(jnp.ravel(bc).size, _NLON * _NLAT)
        self.assertTrue(np.allclose(np.asarray(bc), 1.0e-11))

    def test_single_year_dated_axis_is_transient(self):
        """Twelve real dates are never inferred to be a climatology (#884).

        ``auto`` refuses an in-memory dataset outright; declared dated, the
        year keeps its exact dates rather than repeating annually.
        """
        from jcm.forcing import BY_DATE

        ds = _synthetic_emissions_ds().assign_coords(
            time=np.arange("2001-01", "2002-01", dtype="datetime64[M]"))
        with self.assertRaisesRegex(ValueError, "emissions_align=auto"):
            read_anthropogenic_emissions(ds)
        emissions = read_anthropogenic_emissions(ds, align_mode="by_date")
        leaf = emissions["emis_surface_combustion_bc"]
        self.assertEqual(int(leaf.align_mode), BY_DATE)
        np.testing.assert_array_equal(
            leaf.times.to_datetime64().astype("datetime64[M]"),
            np.arange("2001-01", "2002-01", dtype="datetime64[M]"))

    def test_static_field_passthrough(self):
        # A time-less emissions field is carried as a bare array, still sliced
        # to a no-op by select.
        ds = _synthetic_emissions_ds().isel(time=0)  # drop time dim
        emis = read_anthropogenic_emissions(ds, align_mode="wrap_year")
        bc = emis["emis_surface_combustion_bc"]
        self.assertEqual(bc.shape, (_NLON, _NLAT))


def _synthetic_speciated_ds(nmonths=12):
    """Build a pre-speciated (aero_emis_*) Dataset: a 2-D and a 3-D channel."""
    coords = {
        "lon": np.linspace(0, 360, _NLON, endpoint=False),
        "lat": np.linspace(-87, 87, _NLAT),
        "lev": np.arange(_NLEV),
        "time": np.arange(nmonths),
    }
    surf = np.full((_NLON, _NLAT, nmonths), 2.0e-11)
    vol = np.full((_NLEV, _NLON, _NLAT, nmonths), 1.0e-13)
    return xr.Dataset(
        {
            "aero_emis_m_bc_pcm": (("lon", "lat", "time"), surf),
            "aero_emis_m_so4_acc": (("lev", "lon", "lat", "time"), vol),
        },
        coords=coords,
    )


class PreSpeciatedContractTest(unittest.TestCase):
    def _date(self):
        from jcm.date import DateData
        import jax_datetime as jdt
        return DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(jdt.to_datetime("2001-07-02")))

    def test_returns_none_without_vars(self):
        ds = xr.Dataset({"emis_surface_combustion_so2":
                         (("lon", "lat"), np.zeros((4, 4)))})
        self.assertIsNone(read_prescribed_aerosol_emissions(ds))

    def test_keys_strip_prefix(self):
        emis = read_prescribed_aerosol_emissions(
            _synthetic_speciated_ds(), align_mode="wrap_year")
        self.assertEqual(set(emis), {"m_bc_pcm", "m_so4_acc"})

    def test_select_slices_surface_and_volume(self):
        emis = read_prescribed_aerosol_emissions(
            _synthetic_speciated_ds(), align_mode="wrap_year")
        forcing = ForcingData.zeros((_NLON, _NLAT)).copy(
            prescribed_aerosol_emissions=emis)
        sliced = forcing.select(self._date())
        got = sliced.prescribed_aerosol_emissions
        # 2-D surface channel → (lon, lat); 3-D volume channel → (lev, lon, lat).
        self.assertEqual(got["m_bc_pcm"].shape, (_NLON, _NLAT))
        self.assertEqual(got["m_so4_acc"].shape, (_NLEV, _NLON, _NLAT))


@pytest.mark.slow
class EmissionsRaiseBurdenTest(unittest.TestCase):
    """End-to-end: a prescribed BC/OC source raises the primary-carbon burden."""

    _model = None  # built once; "on" and "off" share it (see _run)

    def _run(self, with_emissions: bool):
        from jcm.model import Model
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        if type(self)._model is None:
            sigma_boundaries = np.linspace(0, 1, 21)
            coords = get_coords(sigma_boundaries, spectral_truncation=21)
            terrain = TerrainData.aquaplanet(coords)
            type(self)._model = Model(
                coords=coords, time_step=30, terrain=terrain,
                physics=idealized_echam_physics(
                    aerosol_module="jam", cloud_scheme="2m",
                    jam_anthropogenic=True),
            )
        model = type(self)._model

        from jcm.forcing import default_forcing
        # Both branches populate the SAME ``anthropogenic_emissions`` channels
        # (on: a strong uniform source; off: the identical channels at
        # exactly zero), rather than leaving the "off" forcing's field at its
        # ``None`` default. ``AnthropogenicEmissions._flux`` branches at
        # TRACE time on whether a channel is present at all (present ->
        # read the array; absent -> a baked-in zero constant), so a
        # structurally-identical forcing pytree for both calls lets
        # ``model.run``'s jit reuse the one compiled executable instead of
        # retracing the whole ECHAM+JAM term graph a second time -- the
        # physical result (zero flux emits nothing) is unchanged either way.
        emis = read_anthropogenic_emissions(
            _synthetic_emissions_ds(
                bc=5.0e-9 if with_emissions else 0.0,
                oc=5.0e-9 if with_emissions else 0.0,
                so2=5.0e-9 if with_emissions else 0.0),
            align_mode="wrap_year")
        forcing = default_forcing(model.coords.horizontal).copy(
            anthropogenic_emissions=emis)
        preds = model.run(forcing=forcing, save_interval=0.0625, total_time=0.0625)
        return preds

    def test_bc_burden_increases_with_emissions(self):
        from jcm.physics.aerosol.jam import mass_name

        on = self._run(with_emissions=True).dynamics.tracers
        off = self._run(with_emissions=False).dynamics.tracers
        bc = mass_name("bc", "pcm")  # primary-carbon BC: the anthropogenic sink

        burden_on = float(np.sum(np.asarray(on[bc])))
        burden_off = float(np.sum(np.asarray(off[bc])))
        self.assertTrue(np.all(np.isfinite(np.asarray(on[bc]))))
        # The prescribed BC source must raise the primary-carbon BC burden.
        self.assertGreater(burden_on, burden_off)


@pytest.mark.slow
class PreSpeciatedRaisesBurdenTest(unittest.TestCase):
    """End-to-end: the CAM-faithful pre-speciated path injects into MAM4 tracers."""

    _model = None  # built once; "on" and "off" share it (see _run)

    def _run(self, with_emissions: bool):
        from jcm.model import Model
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords
        from jcm.forcing import default_forcing

        if type(self)._model is None:
            sigma_boundaries = np.linspace(0, 1, 21)
            coords = get_coords(sigma_boundaries, spectral_truncation=21)
            type(self)._model = Model(
                coords=coords, time_step=30, terrain=TerrainData.aquaplanet(coords),
                physics=idealized_echam_physics(
                    aerosol_module="jam", cloud_scheme="2m",
                    jam_prescribed_speciated=True),
            )
        model = type(self)._model

        # Both branches populate the SAME aero_emis_m_so4_acc channel (on: a
        # uniform source; off: the identical channel at exactly zero) rather
        # than leaving the "off" forcing's ``prescribed_aerosol_emissions`` at
        # its ``None`` default -- see emissions_pipeline_test.py's sibling
        # EmissionsRaiseBurdenTest for why a structurally-identical forcing
        # for both calls lets ``model.run``'s jit reuse one compile instead
        # of retracing the ECHAM+JAM term graph a second time.
        shape = (_NLON, _NLAT, 12)
        ds = xr.Dataset(
            {"aero_emis_m_so4_acc": (("lon", "lat", "time"),
                                     np.full(shape, 5.0e-9 if with_emissions else 0.0))},
            coords={"lon": np.linspace(0, 360, _NLON, endpoint=False),
                    "lat": np.linspace(-87, 87, _NLAT),
                    "time": np.arange(12)})
        forcing = default_forcing(model.coords.horizontal).copy(
            prescribed_aerosol_emissions=read_prescribed_aerosol_emissions(
                ds, align_mode="wrap_year"))
        return model.run(forcing=forcing, save_interval=0.0625, total_time=0.0625)

    def test_so4_burden_increases(self):
        from jcm.physics.aerosol.jam import mass_name

        on = self._run(with_emissions=True).dynamics.tracers
        off = self._run(with_emissions=False).dynamics.tracers
        so4 = mass_name("so4", "acc")
        self.assertTrue(np.all(np.isfinite(np.asarray(on[so4]))))
        self.assertGreater(float(np.sum(np.asarray(on[so4]))),
                           float(np.sum(np.asarray(off[so4]))))


if __name__ == "__main__":
    unittest.main()
