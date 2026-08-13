"""Tests for ``jcm.column_coordinates.ColumnCoordinates``."""

import dataclasses
import unittest

import numpy as np
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.column_coordinates import ColumnCoordinates
from jcm.single_column_model import _make_single_column_coords


class ColumnCoordinatesTest(unittest.TestCase):

    def setUp(self):
        self.vertical = SigmaCoordinates.equidistant(12)
        self.coords = ColumnCoordinates.at_location(
            self.vertical, lat_deg=15.5, lon_deg=-140.625
        )

    def test_nodal_shape_follows_grid_convention(self):
        """(nlev, nlon, nlat), as EchamTermBase.cache_coords assumes."""
        self.assertEqual(self.coords.nodal_shape, (12, 1, 1))
        self.assertEqual(self.coords.horizontal.nodal_shape, (1, 1))

    def test_location_is_stored_in_radians(self):
        np.testing.assert_allclose(
            np.asarray(self.coords.horizontal.latitudes),
            [np.deg2rad(15.5)], rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(self.coords.horizontal.longitudes),
            [np.deg2rad(-140.625)], rtol=1e-6,
        )

    def test_nodal_axes_convention(self):
        """(longitudes, sin(latitudes)) — the pair predictions unpacks."""
        lon, sin_lat = self.coords.horizontal.nodal_axes
        np.testing.assert_allclose(
            np.asarray(lon), np.asarray(self.coords.horizontal.longitudes)
        )
        np.testing.assert_allclose(
            np.asarray(sin_lat),
            np.sin(np.asarray(self.coords.horizontal.latitudes)),
        )

    def test_vertical_passes_through_unchanged(self):
        self.assertIs(self.coords.vertical, self.vertical)

    def test_spectral_attributes_fail_with_explanation(self):
        """A spectral-only physics package must learn WHY it cannot run.

        The SimpleNamespace stub raised a bare AttributeError from deep
        inside the requesting term; the class names the actual problem.
        """
        with self.assertRaisesRegex(
            AttributeError, "no spectral truncation"
        ):
            _ = self.coords.horizontal.longitude_wavenumbers

    def test_unknown_attributes_still_raise_attribute_error(self):
        """hasattr()-probing (used across jcm) must keep working."""
        self.assertFalse(hasattr(self.coords.horizontal, "no_such_attr"))

    def test_scm_helper_returns_the_class(self):
        made = _make_single_column_coords(self.vertical, 15.5, -140.625)
        self.assertIsInstance(made, ColumnCoordinates)
        self.assertEqual(made.nodal_shape, self.coords.nodal_shape)

    def test_physics_term_shape_contract(self):
        """The exact reads PhysicsTerm.initial_carry_state performs."""
        ncols = (
            self.coords.horizontal.nodal_shape[0]
            * self.coords.horizontal.nodal_shape[1]
        )
        nlev = self.coords.nodal_shape[0]
        self.assertEqual((nlev, ncols), (12, 1))

    def test_is_a_frozen_value_type(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.coords.vertical = None


if __name__ == "__main__":
    unittest.main()
