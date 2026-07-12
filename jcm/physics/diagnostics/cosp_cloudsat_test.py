"""Tests for the CloudSat COSP diagnostic hook (requires jax-cosp)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

try:
    import jcosp  # noqa: F401

    HAVE_JCOSP = True
except ImportError:
    HAVE_JCOSP = False

from jcm.forcing import ForcingData
from jcm.physics.clouds.cloud_data import CloudData
from jcm.physics.convection.tiedtke_nordeng.types import ConvectionData
from jcm.physics.diagnostics.moist_air_state import MoistAirColumnState
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData
from jcm.utils import get_coords

# T21 is the smallest supported grid; the term runs column-vectorized.
NLEV, NLAT, NLON = 10, 64, 32
NCOLS = NLAT * NLON


def _setup():
    """Column-vectorized state with a raining warm cloud in half the columns."""
    coords = get_coords(np.linspace(0, 1, NLEV + 1), nodal_shape=(NLAT, NLON))
    terrain = TerrainData.aquaplanet(coords)
    forcing = ForcingData.zeros((NLAT, NLON))

    temperature = jnp.broadcast_to(
        jnp.linspace(210.0, 290.0, NLEV)[:, None], (NLEV, NCOLS))
    q = 0.8 * 6.11e2 * 0.622 / 1e5 * jnp.exp(
        17.27 * (temperature - 273.15) / (temperature - 35.85))
    state = PhysicsState.zeros(
        (NLEV, NCOLS),
        temperature=temperature,
        specific_humidity=q,
        geopotential=jnp.broadcast_to(
            9.81 * jnp.linspace(18000.0, 100.0, NLEV)[:, None], (NLEV, NCOLS)),
        normalized_surface_pressure=jnp.ones((NCOLS,)),
    )

    prep = MoistAirColumnState()
    prep.cache_coords(coords)
    _, diagnostics = prep(state, {"_dt_seconds": 900.0}, forcing, terrain)

    # A warm liquid cloud (levels 7-8, T > 273 K) raining to the surface in
    # the even columns; ice cloud aloft (levels 3-4) in column 0 only.
    clouds = CloudData.zeros((NCOLS,), NLEV)
    cf = np.zeros((NLEV, NCOLS))
    qc = np.zeros((NLEV, NCOLS))
    qi = np.zeros((NLEV, NCOLS))
    rain = np.zeros((NLEV, NCOLS))
    cols = np.arange(0, NCOLS, 2)
    cf[7:9, cols] = 0.7
    qc[7:9, cols] = 4e-4
    rain[7:, cols] = 4e-4  # kg m-2 s-1 below cloud base
    cf[3:5, 0] = 0.6
    qi[3:5, 0] = 1e-4
    clouds = clouds.copy(cloud_fraction=jnp.asarray(cf), qc=jnp.asarray(qc),
                         qi=jnp.asarray(qi), rain_flux=jnp.asarray(rain),
                         snow_flux=jnp.zeros((NLEV, NCOLS)),
                         precip_rain=jnp.asarray(rain[-1]))
    convection = ConvectionData.zeros((NCOLS,), NLEV)
    diagnostics = {**diagnostics, "clouds": clouds, "convection": convection}
    return state, diagnostics, forcing, terrain


@unittest.skipUnless(HAVE_JCOSP, "jax-cosp not installed")
class CloudsatCospTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jcm.physics.diagnostics.cosp_cloudsat import CloudsatCosp

        cls.term = CloudsatCosp(ncolumns=20, seed=1)
        cls.setup = _setup()

    def test_outputs_shapes_and_ranges(self):
        state, diagnostics, forcing, terrain = self.setup
        tend, diag = self.term(state, diagnostics, forcing, terrain)
        for key in ("cosp_warm_rain", "cosp_cold_rain", "cosp_warm_drizzle",
                    "cosp_cold_drizzle", "cosp_pia"):
            self.assertEqual(diag[key].shape, (NCOLS,), key)
            self.assertTrue(bool(jnp.isfinite(diag[key]).all()), key)
        self.assertEqual(diag["cosp_precip_cover"].shape, (NCOLS, 10))
        frac = np.stack([np.asarray(diag[k]) for k in (
            "cosp_warm_rain", "cosp_cold_rain", "cosp_warm_drizzle",
            "cosp_cold_drizzle")])
        self.assertTrue(((frac >= 0) & (frac <= 1)).all())
        # Zero physics tendency: this is a pure diagnostic.
        self.assertEqual(float(jnp.abs(tend.temperature).max()), 0.0)

    def test_warm_rain_detected_in_raining_columns(self):
        state, diagnostics, forcing, terrain = self.setup
        _, diag = self.term(state, diagnostics, forcing, terrain)
        rainy = np.asarray(diag["cosp_warm_rain"]
                           + diag["cosp_warm_drizzle"]
                           + diag["cosp_cold_rain"]
                           + diag["cosp_cold_drizzle"])
        # Even columns rain; odd columns are clear.
        self.assertTrue((rainy[np.arange(2, NCOLS, 2)] > 0).all(), rainy)
        self.assertTrue((rainy[1::2] == 0).all(), rainy)

    def test_jit(self):
        state, diagnostics, forcing, terrain = self.setup

        @jax.jit
        def run(s):
            _, d = self.term(s, diagnostics, forcing, terrain)
            return d["cosp_warm_rain"]

        out = run(state)
        self.assertTrue(bool(jnp.isfinite(out).all()))


@unittest.skipUnless(HAVE_JCOSP, "jax-cosp not installed")
class FactoryWiringTest(unittest.TestCase):
    def test_enable_cosp_adds_term_after_microphysics(self):
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(enable_cosp=True, checkpoint_terms=False)
        names = [t.name for t in physics.terms]
        self.assertIn("cloudsat_cosp", names)
        self.assertGreater(names.index("cloudsat_cosp"),
                           names.index("echam_1m_microphysics"))

    def test_disabled_by_default(self):
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(checkpoint_terms=False)
        self.assertNotIn("cloudsat_cosp", [t.name for t in physics.terms])


if __name__ == "__main__":
    unittest.main()
