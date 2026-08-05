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
    # Keep ``thermo_run`` consistent with the seeded CloudData. In the real
    # model these agree by construction — CloudData holds the STEP-START
    # condensate and ``thermo_run`` the post-microphysics values, and the
    # simulators read the latter. A fixture that seeds only CloudData would
    # hand the simulators a cloud-free thermo_run and quietly test the old
    # (stale-condensate) behaviour.
    tr = diagnostics.get("thermo_run")
    if tr is not None:
        diagnostics = {**diagnostics,
                       "thermo_run": {**tr, "qc": jnp.asarray(qc),
                                      "qi": jnp.asarray(qi)}}
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


@unittest.skipUnless(HAVE_JCOSP, "jax-cosp not installed")
class CalipsoModisTest(unittest.TestCase):
    """CALIPSO and MODIS run on the radar's SCOPS realization."""

    @classmethod
    def setUpClass(cls):
        state, diagnostics, forcing, terrain = _setup()
        # The shared fixture leaves the effective radii at zero, which the
        # LIDAR reads as "no particles" (unlike the radar, for which zero
        # selects the PSD defaults). Give the cloudy layers realistic radii
        # so the lidar has something to detect.
        clouds = diagnostics["clouds"]
        reff_liq = jnp.where(clouds.qc > 0.0, 10.0, 0.0)   # microns
        reff_ice = jnp.where(clouds.qi > 0.0, 30.0, 0.0)
        diagnostics = {**diagnostics,
                       "clouds": clouds.copy(r_eff_liq=reff_liq,
                                             r_eff_ice=reff_ice)}
        cls.setup = (state, diagnostics, forcing, terrain)

    def _run(self, **kw):
        from jcm.physics.diagnostics.cosp_cloudsat import CloudsatCosp
        term = CloudsatCosp(ncolumns=20, seed=1, **kw)
        state, diagnostics, forcing, terrain = self.setup
        return term(state, diagnostics, forcing, terrain)[1]

    def test_simulators_use_post_microphysics_condensate(self):
        """The simulators must read ``thermo_run`` qc/qi, not the step-start
        CloudData values.

        CloudData holds the condensate as it was at the START of the step:
        microphysics returns its effect as a tendency (operator splitting),
        so the struct is stale by the time this diagnostic runs. Reading it
        makes every satellite product disagree with the saved tracer state
        whenever microphysics is active — which is essentially always.

        Here the two are driven APART deliberately: thermo_run is emptied
        while CloudData keeps its cloud. A simulator reading CloudData would
        still report cloud; one reading thermo_run correctly reports none.
        """
        state, diagnostics, forcing, terrain = self.setup
        tr = diagnostics.get("thermo_run")
        self.assertIsNotNone(tr, "fixture must carry thermo_run")
        emptied = {**diagnostics,
                   "thermo_run": {**tr,
                                  "qc": jnp.zeros_like(tr["qc"]),
                                  "qi": jnp.zeros_like(tr["qi"])}}
        from jcm.physics.diagnostics.cosp_cloudsat import CloudsatCosp
        term = CloudsatCosp(ncolumns=20, seed=1, enable_calipso=True,
                            enable_modis=True)
        out = term(state, emptied, forcing, terrain)[1]
        # CloudData still has cloud; thermo_run does not. Post-microphysics
        # wins, so the retrievals must be empty.
        self.assertEqual(float(np.asarray(out["cltcalipso"]).max()), 0.0)
        self.assertEqual(float(np.asarray(out["cltmodis"]).max()), 0.0)
        # ...and with thermo_run carrying the cloud, they are not.
        full = term(state, diagnostics, forcing, terrain)[1]
        self.assertGreater(float(np.asarray(full["cltcalipso"]).max()), 0.0)

    def test_calipso_layered_cover_is_a_fraction(self):
        diag = self._run(enable_calipso=True)
        for key in ("cltcalipso", "cllcalipso", "clmcalipso", "clhcalipso"):
            arr = np.asarray(diag[key])
            self.assertEqual(arr.shape, (NCOLS,), key)
            self.assertTrue(np.isfinite(arr).all(), key)
            # jcosp reports percent; the term converts to a [0,1] fraction.
            self.assertGreaterEqual(arr.min(), -1e-6, key)
            self.assertLessEqual(arr.max(), 1.0 + 1e-6, key)

    def test_calipso_sees_the_seeded_cloud(self):
        """The fixture has cloud, so total lidar cover must be non-zero."""
        diag = self._run(enable_calipso=True)
        self.assertGreater(float(np.asarray(diag["cltcalipso"]).max()), 0.0)

    def test_zero_effective_radius_gives_no_lidar_cloud(self):
        """Documents the radar/lidar convention difference.

        ``lidar_optics`` treats radius <= 0 as "class absent", whereas the
        radar treats reff == 0 as "use PSD defaults". A configuration that
        never sets the effective radii therefore reports zero lidar cover —
        surprising enough to pin down so it is not mistaken for a bug.
        """
        from jcm.physics.diagnostics.cosp_cloudsat import CloudsatCosp
        state, diagnostics, forcing, terrain = _setup()  # radii left at zero
        term = CloudsatCosp(ncolumns=20, seed=1, enable_calipso=True)
        _, diag = term(state, diagnostics, forcing, terrain)
        self.assertEqual(float(np.asarray(diag["cltcalipso"]).max()), 0.0)

    def test_modis_outputs_are_finite_and_physical(self):
        diag = self._run(enable_modis=True)
        for key in ("cltmodis", "clwmodis", "climodis"):
            arr = np.asarray(diag[key])
            self.assertEqual(arr.shape, (NCOLS,), key)
            self.assertGreaterEqual(arr.min(), -1e-6, key)
            self.assertLessEqual(arr.max(), 1.0 + 1e-6, key)
        for key in ("tauwmodis", "tauimodis", "reffclwmodis", "reffclimodis",
                    "lwpmodis", "iwpmodis"):
            arr = np.asarray(diag[key])
            self.assertTrue(np.isfinite(arr).all(), key)
            self.assertGreaterEqual(arr.min(), -1e-9, key)

    def test_radar_diagnostics_unchanged_by_the_extra_simulators(self):
        """Adding MODIS/CALIPSO must not perturb the CloudSat results.

        They share one SCOPS draw, so the radar output has to be identical —
        if it moves, the subcolumn realization is being regenerated rather
        than reused, which would both cost more and decouple the instruments.
        """
        base = self._run()
        both = self._run(enable_calipso=True, enable_modis=True)
        for key in ("cosp_warm_rain", "cosp_cold_rain", "cosp_pia"):
            np.testing.assert_allclose(
                np.asarray(both[key]), np.asarray(base[key]), rtol=1e-6,
                err_msg=f"{key} changed when extra simulators were enabled")

    def test_disabled_by_default(self):
        diag = self._run()
        for key in ("cltcalipso", "cltmodis"):
            self.assertNotIn(key, diag)
