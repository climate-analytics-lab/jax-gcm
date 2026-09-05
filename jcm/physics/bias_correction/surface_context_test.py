"""Tests for the surface-state context features (sice, snowc, stl, soilw).

These are the runnable form of "give the network the ERA5 land fields": ERA5
itself does not exist during a free run, so the same quantities are read from
the model's own boundary conditions, which ForcingData.select repopulates every
step.

`sice` is the time-varying counterpart to the static land mask (`fmask`), which
calls winter sea ice "ocean" exactly where the DJF bias lives. These tests cover
the plumbing (values, normalisation, finiteness, the zero-row warm start); which
feature helps the climate score is an experimental question, not a test.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.bias_correction.nn_bias_correction import (
    CONTEXT_FEATURES,
    _SURFACE_FEATURES,
    build_context,
)
from jcm.physics_interface import PhysicsState

NLEV, NLON, NLAT = 8, 6, 4


class _Forcing:
    """Minimal stand-in carrying just the surface fields."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


def _state():
    shape = (NLEV, NLON, NLAT)
    return PhysicsState(
        u_wind=jnp.zeros(shape), v_wind=jnp.zeros(shape),
        temperature=jnp.full(shape, 260.0),
        specific_humidity=jnp.full(shape, 3.0),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones((NLON, NLAT)),
    )


def _forcing(sice=0.0, snowc=0.0, stl=273.15, soilw=0.0):
    horiz = (NLON, NLAT)
    return _Forcing(
        sice_am=jnp.full(horiz, sice), snowc_am=jnp.full(horiz, snowc),
        stl_am=jnp.full(horiz, stl), soilw_am=jnp.full(horiz, soilw),
    )


class TestRegistry(unittest.TestCase):
    def test_surface_features_are_in_the_public_tuple(self):
        for name in _SURFACE_FEATURES:
            self.assertIn(name, CONTEXT_FEATURES)

    def test_new_features_are_appended_last(self):
        # Load-bearing: a term's feature order is re-derived by filtering
        # CONTEXT_FEATURES, and widen_first_layer only appends rows at the
        # bottom. Inserting mid-tuple would silently pair existing trained
        # weights with the wrong inputs.
        self.assertEqual(CONTEXT_FEATURES[:6],
                         ("fmask", "orog", "sin_lat", "cos_lat", "ps",
                          "insol"))
        self.assertEqual(set(CONTEXT_FEATURES[6:]), set(_SURFACE_FEATURES))


class TestValues(unittest.TestCase):
    def test_shape_is_columns_by_features(self):
        ctx = build_context(_state(), ("sice", "snowc"), forcing=_forcing())
        self.assertEqual(ctx.shape, (NLON * NLAT, 2))

    def test_sea_ice_is_centred(self):
        # Open ocean and full ice sit either side of zero, so the feature
        # carries sign rather than magnitude alone.
        open_ocean = build_context(_state(), ("sice",),
                                   forcing=_forcing(sice=0.0))
        full_ice = build_context(_state(), ("sice",),
                                 forcing=_forcing(sice=1.0))
        self.assertAlmostEqual(float(open_ocean[0, 0]), -0.5, places=6)
        self.assertAlmostEqual(float(full_ice[0, 0]), +0.5, places=6)

    def test_surface_temperature_is_centred_on_freezing(self):
        at_freezing = build_context(_state(), ("stl",),
                                    forcing=_forcing(stl=273.15))
        self.assertAlmostEqual(float(at_freezing[0, 0]), 0.0, places=5)
        warm = build_context(_state(), ("stl",), forcing=_forcing(stl=303.15))
        self.assertAlmostEqual(float(warm[0, 0]), 1.0, places=5)

    def test_all_surface_features_land_near_order_one(self):
        # Extremes must not blow the input scale, since these are appended to
        # a standardised feature vector.
        ctx = build_context(
            _state(), tuple(_SURFACE_FEATURES),
            forcing=_forcing(sice=1.0, snowc=200.0, stl=320.0, soilw=1.0))
        self.assertTrue(bool(jnp.all(jnp.abs(ctx) <= 3.0)),
                        f"feature out of range: {np.asarray(ctx[0])}")

    def test_features_are_finite_at_the_cold_extreme(self):
        ctx = build_context(
            _state(), tuple(_SURFACE_FEATURES),
            forcing=_forcing(sice=1.0, snowc=0.0, stl=220.0, soilw=0.0))
        self.assertTrue(bool(jnp.all(jnp.isfinite(ctx))))


class TestErrors(unittest.TestCase):
    def test_missing_forcing_raises_rather_than_silently_zeroing(self):
        for name in _SURFACE_FEATURES:
            with self.assertRaises(ValueError, msg=name):
                build_context(_state(), (name,), forcing=None)

    def test_unknown_feature_still_rejected(self):
        with self.assertRaises(ValueError):
            build_context(_state(), ("not_a_feature",), forcing=_forcing())


if __name__ == "__main__":
    unittest.main()
