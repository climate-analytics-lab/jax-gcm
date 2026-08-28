"""Tests for :mod:`jcm.cf_metadata` and the output vertical convention.

The end-to-end test asserts the convention on a *written* netCDF rather than
on an in-memory Dataset, because that is the artefact the trap bites in: #710
was invisible until a reader paired a ``level`` field with a ``level_i`` field
from the same file.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from dinosaur.hybrid_coordinates import HybridCoordinates
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm import cf_metadata


def _hybrid_2level():
    """Build a tiny TOA-first hybrid table: 2 layers, 3 interfaces."""
    return HybridCoordinates(
        a_boundaries=np.array([0.0, 100.0, 0.0]),
        b_boundaries=np.array([0.0, 0.5, 1.0]),
    )


def _toa_first_dataset():
    """Build a 2-layer dataset in the physics-internal (TOA-first) frame.

    Pressures are the ones :func:`_hybrid_2level`'s table gives at
    ``p_s = 1000``, so the hybrid round-trip is a real check and not a
    tautology: interfaces ``a + b·p_s``, mid-levels their means.
    """
    return xr.Dataset(
        {
            "pressure_full": (("level",), np.array([300.0, 800.0])),
            "pressure_half": (("level_i",), np.array([0.0, 600.0, 1000.0])),
            "surface_pressure": ((), 1000.0),
        }
    )


class TestHybridBoundaries(unittest.TestCase):
    def test_hybrid_coordinates(self):
        a, b = cf_metadata.hybrid_boundaries(_hybrid_2level())
        np.testing.assert_allclose(a, [0.0, 100.0, 0.0])
        np.testing.assert_allclose(b, [0.0, 0.5, 1.0])

    def test_sigma_coordinates_are_the_a_equals_zero_case(self):
        sigma = SigmaCoordinates(boundaries=np.array([0.0, 0.5, 1.0]))
        a, b = cf_metadata.hybrid_boundaries(sigma)
        np.testing.assert_allclose(a, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(b, [0.0, 0.5, 1.0])


class TestOrientSurfaceFirst(unittest.TestCase):
    def test_flips_both_vertical_axes(self):
        """The #710 regression guard: flipping only ``level`` is the bug."""
        out = cf_metadata.orient_surface_first(_toa_first_dataset())
        np.testing.assert_allclose(out["pressure_full"].values, [800.0, 300.0])
        np.testing.assert_allclose(
            out["pressure_half"].values, [1000.0, 600.0, 0.0])

    def test_no_vertical_axis_is_a_no_op(self):
        ds = xr.Dataset({"precip": (("lat",), np.arange(3.0))})
        np.testing.assert_allclose(
            cf_metadata.orient_surface_first(ds)["precip"].values, [0, 1, 2])


class TestAttachVerticalCoordinates(unittest.TestCase):
    def setUp(self):
        ds = cf_metadata.orient_surface_first(_toa_first_dataset())
        self.ds = cf_metadata.attach_vertical_coordinates(
            ds, *cf_metadata.hybrid_boundaries(_hybrid_2level()), p0=1000.0)

    def test_sigma_values_are_surface_first(self):
        # sigma = a/p0 + b, reversed: interfaces 1.0, 0.6, 0.0
        np.testing.assert_allclose(self.ds["level_i"].values, [1.0, 0.6, 0.0])
        np.testing.assert_allclose(self.ds["level"].values, [0.8, 0.3])

    def test_hybrid_tables_reproduce_the_pressures(self):
        a = self.ds[cf_metadata.HYBRID_A_HALF].values
        b = self.ds[cf_metadata.HYBRID_B_HALF].values
        np.testing.assert_allclose(
            a + b * 1000.0, self.ds["pressure_half"].values)

    def test_full_levels_bracketed_by_their_interfaces(self):
        sigma_full = self.ds["level"].values
        sigma_half = self.ds["level_i"].values
        self.assertTrue(np.all(sigma_half[:-1] > sigma_full))
        self.assertTrue(np.all(sigma_full > sigma_half[1:]))


class TestApplyCfAttributes(unittest.TestCase):
    def setUp(self):
        self.ds = cf_metadata.finalize_output(
            _toa_first_dataset(), vertical=_hybrid_2level(), p0=1000.0)

    def test_vertical_axes_declare_their_direction(self):
        for dim in ("level", "level_i"):
            attrs = self.ds[dim].attrs
            self.assertEqual(attrs["positive"], "down")
            self.assertEqual(attrs["axis"], "Z")
            self.assertEqual(
                attrs["standard_name"],
                "atmosphere_hybrid_sigma_pressure_coordinate")
            self.assertIn("surface-first", attrs["long_name"])

    def test_formula_terms_name_the_variables_in_the_file(self):
        for dim in ("level", "level_i"):
            terms = self.ds[dim].attrs["formula_terms"].split()
            for name in terms[1::2]:
                self.assertIn(name, self.ds.variables, msg=f"{dim}: {name}")

    def test_formula_terms_omitted_without_surface_pressure(self):
        """A dynamics-only file must not carry a dangling ``ps:`` reference."""
        ds = _toa_first_dataset().drop_vars("surface_pressure")
        ds = cf_metadata.finalize_output(
            ds, vertical=_hybrid_2level(), p0=1000.0)
        self.assertNotIn("formula_terms", ds["level"].attrs)
        self.assertEqual(ds["level"].attrs["positive"], "down")

    def test_pressure_variables_get_standard_names(self):
        for name in ("pressure_full", "pressure_half"):
            self.assertEqual(self.ds[name].attrs["standard_name"], "air_pressure")
            self.assertEqual(self.ds[name].attrs["units"], "Pa")
        # ``positive`` belongs on the coordinate variable, not on a data
        # variable that happens to be vertical (CF-1.11 4.3).
        self.assertNotIn("positive", self.ds["pressure_full"].attrs)

    def test_conventions_stamped(self):
        self.assertTrue(self.ds.attrs["Conventions"].startswith("CF-"))

    def test_datetime_time_axis_gets_the_cf_time_attributes(self):
        ds = _toa_first_dataset().assign_coords(
            time=("time", np.array(["2000-01-01"], dtype="datetime64[ns]")))
        ds = cf_metadata.finalize_output(
            ds, vertical=_hybrid_2level(), p0=1000.0)
        self.assertEqual(ds["time"].attrs["standard_name"], "time")
        self.assertEqual(ds["time"].attrs["axis"], "T")
        # xarray's datetime encoding owns ``units`` on write.
        self.assertNotIn("units", ds["time"].attrs)

    def test_numeric_time_axis_is_not_claimed_as_a_cf_time_coordinate(self):
        """A bare elapsed-days axis has no reference-time units to decode.

        Claiming ``standard_name = "time"`` on it would announce CF
        conformance a reader cannot honour.
        """
        ds = _toa_first_dataset().assign_coords(
            time=("time", np.array([0.0, 0.5])))
        ds = cf_metadata.finalize_output(
            ds, vertical=_hybrid_2level(), p0=1000.0)
        self.assertNotIn("standard_name", ds["time"].attrs)
        self.assertEqual(ds["time"].attrs["units"], "d")
        self.assertEqual(ds["time"].attrs["axis"], "T")

    def test_flip_vertical_false_leaves_data_alone(self):
        """The pyses backend emits surface-first already."""
        ds = xr.Dataset({"pressure_half": (("level_i",),
                                           np.array([1000.0, 600.0, 0.0]))})
        out = cf_metadata.finalize_output(
            ds, vertical=_hybrid_2level(), p0=1000.0, flip_vertical=False)
        np.testing.assert_allclose(
            out["pressure_half"].values, [1000.0, 600.0, 0.0])
        np.testing.assert_allclose(out["level_i"].values, [1.0, 0.6, 0.0])


class TestWrittenFileConvention(unittest.TestCase):
    """Both vertical axes of a real run's netCDF must run the same way."""

    @classmethod
    def setUpClass(cls):
        import logging

        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        model = Model(
            coords=coords,
            physics=echam_physics(radiation_scheme="grey",
                                  checkpoint_terms=False),
            time_step=3.0,
            log_level=logging.CRITICAL,
        )
        preds = model.run(save_interval=1.0 / 24.0, total_time=1.0 / 24.0)
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "traj.nc"
        preds.to_xarray().to_netcdf(path)
        cls.ds = xr.open_dataset(path)

    @classmethod
    def tearDownClass(cls):
        cls.ds.close()
        cls._tmp.cleanup()

    def _column(self, name):
        da = self.ds[name].isel(time=0)
        return da.mean([d for d in da.dims if d in ("lon", "lat")]).values

    def test_both_axes_are_surface_first(self):
        for name in ("pressure_full", "pressure_half"):
            column = self._column(name)
            self.assertGreater(column[0], column[-1], msg=name)
        self.assertGreater(float(self.ds["level"][0]), 0.9)
        self.assertLess(float(self.ds["level"][-1]), 1e-3)
        self.assertAlmostEqual(float(self.ds["level_i"][0]), 1.0)
        self.assertAlmostEqual(float(self.ds["level_i"][-1]), 0.0)

    def test_full_levels_lie_between_their_interfaces(self):
        """The pairing that #710 silently reversed, asserted per column."""
        pf = self.ds["pressure_full"].isel(time=0).values
        ph = self.ds["pressure_half"].isel(time=0).values
        self.assertTrue(np.all(ph[:-1] > pf))
        self.assertTrue(np.all(pf > ph[1:]))

    def test_layer_thickness_from_interfaces_is_positive(self):
        dp = -np.diff(self.ds["pressure_half"].isel(time=0).values, axis=0)
        self.assertTrue(np.all(dp > 0))

    def test_hybrid_tables_survive_the_round_trip(self):
        ps = self.ds["surface_pressure"].isel(time=0).values
        a = self.ds[cf_metadata.HYBRID_A_HALF].values
        b = self.ds[cf_metadata.HYBRID_B_HALF].values
        expected = a[:, None, None] + b[:, None, None] * ps[None]
        np.testing.assert_allclose(
            expected, self.ds["pressure_half"].isel(time=0).values, rtol=1e-5)

    def test_orientation_is_declared_in_the_metadata(self):
        for dim in ("level", "level_i"):
            self.assertEqual(self.ds[dim].attrs["positive"], "down")
            self.assertEqual(self.ds[dim].attrs["axis"], "Z")
        for axis, std in (("lat", "latitude"), ("lon", "longitude")):
            self.assertEqual(self.ds[axis].attrs["standard_name"], std)


if __name__ == "__main__":
    unittest.main()
