"""Phase 1 tests: interim prescribed oxidant fields (#496)."""

import types
import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.chemistry.oxidants import (
    OxidantParameters,
    PrescribedOxidants,
    air_number_density,
    oxidant_field,
    oxidant_field_from_vmr,
)
from jcm.physics_interface import PhysicsState


class AirNumberDensityTest(unittest.TestCase):
    def test_surface_value(self):
        n = air_number_density(jnp.asarray(288.0), jnp.asarray(1.0e5))
        # ~2.5e19 molec/cm^3 at the surface.
        self.assertAlmostEqual(float(n) / 1.0e19, 2.5, delta=0.2)

    def test_decreases_with_pressure(self):
        n_sfc = air_number_density(jnp.asarray(250.0), jnp.asarray(1.0e5))
        n_top = air_number_density(jnp.asarray(250.0), jnp.asarray(1.0e4))
        self.assertGreater(float(n_sfc), float(n_top))


class OxidantFieldTest(unittest.TestCase):
    def _inputs(self, cosz):
        shape = (3, 2)
        t = jnp.full(shape, 280.0)
        p = jnp.full(shape, 9.0e4)
        o3 = jnp.full(shape, 0.04)  # ppmv-style (40 ppbv)
        return t, p, o3, jnp.full(shape, cosz)

    def test_fields_finite_and_nonnegative(self):
        t, p, o3, cz = self._inputs(0.5)
        f = oxidant_field(t, p, o3, cz, OxidantParameters.default())
        for arr in (f.oh, f.no3, f.o3, f.h2o2):
            a = np.asarray(arr)
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertTrue(np.all(a >= 0.0))

    def test_oh_is_daytime(self):
        params = OxidantParameters.default()
        day = oxidant_field(*self._inputs(1.0), params)
        night = oxidant_field(*self._inputs(0.0), params)
        self.assertGreater(float(day.oh.mean()), float(night.oh.mean()))
        self.assertAlmostEqual(float(night.oh.mean()), 0.0)

    def test_no3_is_nighttime(self):
        params = OxidantParameters.default()
        day = oxidant_field(*self._inputs(1.0), params)
        night = oxidant_field(*self._inputs(0.0), params)
        self.assertGreater(float(night.no3.mean()), float(day.no3.mean()))

    def test_oh_magnitude_reasonable(self):
        # Daytime OH should be O(1e6) molec/cm^3, not orders off.
        f = oxidant_field(*self._inputs(1.0), OxidantParameters.default())
        self.assertTrue(1.0e5 < float(f.oh.mean()) < 1.0e7)


class PrescribedOxidantsTermTest(unittest.TestCase):
    def _state(self, nlev=3, ncols=2):
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 280.0),
        )
        return state

    def test_with_carried_chemistry_and_radiation(self):
        state = self._state()
        diagnostics = {
            "pressure_full": jnp.full((3, 2), 9.0e4),
            "chemistry": types.SimpleNamespace(
                ozone_vmr=jnp.full((3, 2), 0.05)
            ),
            "radiation": types.SimpleNamespace(
                cos_zenith=jnp.full((2,), 0.8)
            ),
        }
        _, diags = PrescribedOxidants()(state, diagnostics, None, None)
        ox = diags["oxidants"]
        self.assertEqual(ox.oh.shape, (3, 2))
        for arr in (ox.oh, ox.no3, ox.o3, ox.h2o2):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
        self.assertGreater(float(ox.oh.mean()), 0.0)

    def test_fallback_when_carry_absent(self):
        state = self._state()
        diagnostics = {"pressure_full": jnp.full((3, 2), 9.0e4)}
        _, diags = PrescribedOxidants()(state, diagnostics, None, None)
        ox = diags["oxidants"]
        for arr in (ox.oh, ox.no3, ox.o3, ox.h2o2):
            a = np.asarray(arr)
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertTrue(np.all(a >= 0.0))
        # Fallback O3 is finite and positive.
        self.assertGreater(float(ox.o3.mean()), 0.0)


class PrescribedOxidantsFromFileTest(unittest.TestCase):
    """The prescribed-climatology path (``forcing.oxidant_vmr``)."""

    def _state(self, nlev=3, ncols=2):
        return PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 280.0),
        )

    def _vmr(self, shape):
        return {
            "oh": jnp.full(shape, 1.0e-13),
            "no3": jnp.full(shape, 2.0e-14),
            "o3": jnp.full(shape, 4.0e-8),
            "h2o2": jnp.full(shape, 5.0e-10),
        }

    def test_vmr_converts_with_air_density(self):
        t = jnp.full((3, 2), 280.0)
        p = jnp.full((3, 2), 9.0e4)
        f = oxidant_field_from_vmr(t, p, self._vmr((3, 2)))
        n_air = air_number_density(t, p)
        np.testing.assert_allclose(
            np.asarray(f.o3), np.asarray(4.0e-8 * n_air), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(f.oh), np.asarray(1.0e-13 * n_air), rtol=1e-6
        )

    def test_negative_vmr_clamped(self):
        t = jnp.full((3, 2), 280.0)
        p = jnp.full((3, 2), 9.0e4)
        vmr = self._vmr((3, 2))
        vmr["no3"] = jnp.full((3, 2), -1.0e-12)
        f = oxidant_field_from_vmr(t, p, vmr)
        self.assertEqual(float(f.no3.max()), 0.0)

    def test_grid_layout_reshapes_to_columns(self):
        # A (nlev, lon, lat) forcing slice against a column-vectorized
        # (nlev, ncols) state reshapes level-preserving (C order).
        t = jnp.full((3, 6), 280.0)
        p = jnp.full((3, 6), 9.0e4)
        f = oxidant_field_from_vmr(t, p, self._vmr((3, 3, 2)))
        self.assertEqual(f.o3.shape, (3, 6))

    def test_term_prefers_file_climatology_over_proxies(self):
        state = self._state()
        diagnostics = {
            "pressure_full": jnp.full((3, 2), 9.0e4),
            # Carry chemistry/radiation so the proxy path *would* produce a
            # different (nonzero-OH daytime) answer if it were taken.
            "chemistry": types.SimpleNamespace(
                ozone_vmr=jnp.full((3, 2), 0.05)
            ),
            "radiation": types.SimpleNamespace(
                cos_zenith=jnp.full((2,), 0.8)
            ),
        }
        forcing = types.SimpleNamespace(oxidant_vmr=self._vmr((3, 2)))
        _, diags = PrescribedOxidants()(state, diagnostics, forcing, None)
        ox = diags["oxidants"]
        n_air = air_number_density(
            state.temperature, diagnostics["pressure_full"]
        )
        np.testing.assert_allclose(
            np.asarray(ox.o3), np.asarray(4.0e-8 * n_air), rtol=1e-6
        )
        # OH from the file (1e-13 VMR · n_air ≈ 2e6) — not the cos-zenith
        # proxy, which for these inputs would be 2.5e6 · 0.8 · 1.25 = 2.5e6.
        proxy = oxidant_field(
            state.temperature, diagnostics["pressure_full"],
            jnp.full((3, 2), 0.05), jnp.full((3, 2), 0.8),
            OxidantParameters.default(),
        )
        self.assertFalse(np.allclose(np.asarray(ox.oh), np.asarray(proxy.oh)))

    def test_term_falls_back_without_field(self):
        # ``oxidant_vmr=None`` (the ForcingData default) keeps the proxies.
        state = self._state()
        diagnostics = {"pressure_full": jnp.full((3, 2), 9.0e4)}
        forcing = types.SimpleNamespace(oxidant_vmr=None)
        _, diags = PrescribedOxidants()(state, diagnostics, forcing, None)
        for arr in (diags["oxidants"].oh, diags["oxidants"].o3):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))


if __name__ == "__main__":
    unittest.main()
