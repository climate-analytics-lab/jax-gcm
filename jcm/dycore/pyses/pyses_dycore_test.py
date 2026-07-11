"""Protocol / configuration / coupling tests for :class:`PysesCamSEDycore`.

Mirrors the conventions of ``jcm/dycore/protocol_test.py`` (shapes on a
non-lat/lon layout, tracer seeding, forward stepping) plus the developer
prototype's self-checks (finite-top L47 grid, periodic interpolation, real
geography, native nu_top sponge, bounded coupled steps).

Skipped automatically when the optional ``pyses`` dependency is missing.
Run on CPU: ``JAX_PLATFORMS=cpu pytest jcm/dycore/pyses -q``. The heavy
dycore fixture (SE grid + USSA initial state, ~30 s) is built once per
class; the coupled ECHAM smokes are ``@pytest.mark.slow``.
"""

import os
import unittest

os.environ.setdefault("PYSES_BACKEND", "jax")
os.environ.setdefault("PYSES_USE_CPU", "1")

import numpy as np
import pytest

pytest.importorskip("pyses")

import jax.numpy as jnp

from jcm.dycore import list_dycores
from jcm.dycore.base import DynamicalCore, Predictions
from jcm.dycore.pyses import PysesCamSEDycore, build_forcing, full_echam_hybrid
from jcm.dycore.pyses.initial_states import ussa_pressure, ussa_temperature
from jcm.dycore.pyses.interp import interp_grid_to_points
from jcm.physics.echam.echam_coords import EchamCoords
from jcm.physics.physics_term import TracerSpec
from jcm.physics_interface import PhysicsState, PhysicsTendency

_T63_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bc", "t63")
T63_TERRAIN = os.path.abspath(os.path.join(_T63_DIR, "terrain.nc"))
T63_FORCING = os.path.abspath(os.path.join(_T63_DIR, "forcing.nc"))

_TRACER_SPECS = {
    "qc": TracerSpec("qc", units="kg/kg"),
    "qi": TracerSpec("qi", units="kg/kg"),
    "co2_vmr": TracerSpec("co2_vmr", units="ppmv", initial_value=400.0,
                          nondimensionalize=False),
}


class TestHelpers(unittest.TestCase):
    """Pure helpers: finite-top hybrid table, interpolation, USSA profile."""

    def test_full_hybrid_grid_47_levels(self):
        a, b = full_echam_hybrid(47)
        self.assertEqual(a.shape, (48,))
        self.assertEqual(b.shape, (48,))
        self.assertAlmostEqual(float(b[-1]), 1.0, places=6)   # surface
        self.assertAlmostEqual(float(b[0]), 0.0, places=6)    # pure-pressure top
        self.assertGreater(float(a[0]), 0.0)                  # finite top
        self.assertLess(float(a[0]), float(a[1]))             # increasing at top
        self.assertLess(float(a[0]), 5.0)                     # ~1 Pa tiny top

    def test_interpolation_exact_on_nodes_and_periodic(self):
        lon = np.linspace(0.0, 360.0, 9)[:-1]
        lat = np.linspace(-80.0, 80.0, 5)
        field = np.outer(np.cos(np.radians(lon)), np.ones_like(lat))
        v = interp_grid_to_points(lon, lat, field, [45.0], [0.0])
        self.assertAlmostEqual(float(v[0]), np.cos(np.radians(45.0)), places=6)
        v0 = interp_grid_to_points(lon, lat, field, [0.0], [0.0])
        v360 = interp_grid_to_points(lon, lat, field, [360.0], [0.0])
        self.assertAlmostEqual(float(v0[0]), float(v360[0]), places=6)
        # Latitude clamped beyond the grid (no polar extrapolation blow-up).
        vp = interp_grid_to_points(lon, lat, field, [10.0], [89.9])
        self.assertTrue(np.isfinite(vp).all())

    def test_ussa_profile_positive_and_monotone(self):
        # Stay within the tabulated profile (top 84.852 km geopotential;
        # beyond it T and p are deliberately clamped, so p goes flat).
        z = jnp.linspace(0.0, 80e3, 500)
        T = np.asarray(ussa_temperature(z))
        p = np.asarray(ussa_pressure(z))
        self.assertTrue(np.isfinite(T).all() and np.isfinite(p).all())
        self.assertGreater(T.min(), 150.0)
        self.assertLess(T.max(), 300.0)
        self.assertTrue(np.all(np.diff(p) < 0.0))             # monotone p(z)
        self.assertAlmostEqual(float(p[0]), 101325.0, delta=1.0)
        # Clamped-but-finite above the table.
        self.assertTrue(np.isfinite(float(ussa_pressure(jnp.asarray(90e3)))))


class TestPysesDycoreProtocol(unittest.TestCase):
    """Protocol conformance on the real backend (ne3, real T63 geography)."""

    @classmethod
    def setUpClass(cls):
        cls.dycore = PysesCamSEDycore(
            nx=3, npt=4, dt_seconds=900.0,
            terrain_file=T63_TERRAIN, tracer_specs=_TRACER_SPECS,
        )
        cls.ncol = cls.dycore.colmap.num_cols
        cls.state = cls.dycore.initial_state(None, tracer_specs=_TRACER_SPECS)

    def test_is_dynamical_core_and_registered(self):
        self.assertIsInstance(self.dycore, DynamicalCore)
        self.assertIn("pyses_cam_se", list_dycores())

    def test_coords_adapter_shapes(self):
        coords = self.dycore.coords
        self.assertEqual(coords.horizontal.nodal_shape, (1, self.ncol))
        self.assertEqual(coords.nodal_shape, (47, 1, self.ncol))
        self.assertEqual(coords.horizontal.latitudes.shape, (self.ncol,))
        self.assertEqual(coords.horizontal.longitudes.shape, (1,))
        self.assertEqual(coords.horizontal.column_longitudes.shape, (self.ncol,))
        # The ECHAM physics coords struct must build cleanly off the adapter
        # and reproduce the dynamics' hybrid pressure at the reference ps.
        ec = EchamCoords.from_coordinate_system(coords)
        self.assertEqual(ec.a_half.shape, (48,))
        p_full = ec.calculate_pressure_full(
            jnp.full((1, self.ncol), self.dycore.p0))
        self.assertEqual(p_full.shape, (47, 1, self.ncol))
        # Top-first: index 0 is the ~1 Pa model top, index -1 near-surface.
        self.assertLess(float(p_full[0, 0, 0]), 5.0)
        self.assertGreater(float(p_full[-1, 0, 0]), 9.5e4)

    def test_ussa_initial_state_and_tracers(self):
        ps = self.dycore.to_physics_state(self.state)
        self.assertEqual(ps.temperature.shape, (47, 1, self.ncol))
        self.assertEqual(ps.temperature.dtype, jnp.float32)
        T = np.asarray(ps.temperature)
        self.assertTrue(np.isfinite(T).all())
        self.assertGreater(T.min(), 150.0)
        self.assertLess(T.max(), 340.0)
        # Resting and dry.
        self.assertEqual(float(jnp.abs(ps.u_wind).max()), 0.0)
        self.assertEqual(float(np.asarray(ps.specific_humidity).max()), 0.0)
        # Real orography imprints a land/sea surface-pressure contrast.
        nsp = np.asarray(ps.normalized_surface_pressure)
        self.assertLess(nsp.min(), 0.9)
        self.assertGreater(nsp.max(), 1.0)
        # Declared tracers are seeded at their initial values.
        self.assertEqual(set(ps.tracers), set(_TRACER_SPECS))
        np.testing.assert_allclose(np.asarray(ps.tracers["co2_vmr"]), 400.0,
                                   rtol=1e-6)
        np.testing.assert_allclose(np.asarray(ps.tracers["qc"]), 0.0)
        # Geopotential grows monotonically toward the (top-first) index 0.
        phi = np.asarray(ps.geopotential)
        self.assertGreater(phi[0].mean(), phi[-1].mean())

    def test_real_geography(self):
        orog = np.asarray(self.dycore._orog_col)
        fmask = np.asarray(self.dycore.terrain.fmask).reshape(-1)
        self.assertGreaterEqual(orog.min(), 0.0)
        self.assertGreater(orog.max(), 500.0)     # real mountains present
        self.assertLess(orog.max(), 6500.0)       # plausible peak
        self.assertGreater(fmask.max(), 0.5)      # some land columns
        self.assertLess(fmask.min(), 0.5)         # some ocean columns

    def test_native_upper_sponge_configured(self):
        dc = self.dycore.diffusion_config
        self.assertIn("sponge_layer", dc)
        self.assertIn("nu_top", dc)
        nu_ramp = np.asarray(dc["nu_ramp"]).reshape(-1)
        self.assertEqual(nu_ramp.size, self.dycore.n_sponge)
        self.assertEqual(float(np.asarray(dc["nu_top"])), self.dycore.nu_top)
        # Ramp strongest at the model top (index 0), positive throughout.
        self.assertGreaterEqual(nu_ramp[0], nu_ramp[-1])
        self.assertGreater(float(nu_ramp.min()), 0.0)

    def test_required_tracers_ok(self):
        self.dycore.required_tracers_ok(_TRACER_SPECS.values())  # no raise
        with self.assertRaises(ValueError):
            self.dycore.required_tracers_ok([TracerSpec("water_vapor")])
        with self.assertRaises(ValueError):
            self.dycore.required_tracers_ok(
                [TracerSpec("qc"), TracerSpec("qc")])

    def test_forced_step_advances_and_stays_finite(self):
        dc = self.dycore
        shape = (47, 1, self.ncol)
        tend = PhysicsTendency(
            u_wind=jnp.zeros(shape, jnp.float32),
            v_wind=jnp.zeros(shape, jnp.float32),
            temperature=jnp.full(shape, 1e-5, jnp.float32),   # ~0.01 K/step
            specific_humidity=jnp.full(shape, 1e-10, jnp.float32),
            tracers={name: jnp.zeros(shape, jnp.float32) for name in _TRACER_SPECS},
        )
        next_state = dc.step(self.state, tend)
        self.assertAlmostEqual(
            float(dc.sim_time(next_state)) - float(dc.sim_time(self.state)),
            dc.dt_seconds,
        )
        ps = dc.to_physics_state(next_state)
        T = np.asarray(ps.temperature)
        self.assertTrue(np.isfinite(T).all())
        self.assertGreater(T.min(), 150.0)
        self.assertLess(T.max(), 340.0)
        # The moisture forcing reached the (dry) state.
        self.assertGreater(float(np.asarray(ps.specific_humidity).max()), 0.0)
        # with_sim_time round trip.
        rewound = dc.with_sim_time(next_state, 0.0)
        self.assertEqual(float(dc.sim_time(rewound)), 0.0)

    def test_initial_state_from_physics_state_round_trips(self):
        """Constant gridpoint fields survive the columns->GLL->columns seam.

        Not bitwise: the FV->GLL reconstruction is J-density-weighted and the
        DSS mixes element boundaries, so pointwise agreement is ~1% at ne3
        (shrinking with resolution). The element-mean identity (exactness of
        gather∘scatter) is asserted separately in physics_grid_test.
        """
        shape = (47, 1, self.ncol)
        target = PhysicsState(
            u_wind=jnp.full(shape, 1.5, jnp.float32),
            v_wind=jnp.full(shape, -2.5, jnp.float32),
            temperature=jnp.full(shape, 271.0, jnp.float32),
            specific_humidity=jnp.full(shape, 1e-3, jnp.float32),
            geopotential=jnp.zeros(shape, jnp.float32),
            normalized_surface_pressure=jnp.full((1, self.ncol), 0.98, jnp.float32),
            tracers={"qc": jnp.zeros(shape, jnp.float32),
                     "qi": jnp.zeros(shape, jnp.float32),
                     "co2_vmr": jnp.full(shape, 400.0, jnp.float32)},
        )
        state = self.dycore.initial_state(target, tracer_specs=_TRACER_SPECS)
        rec = self.dycore.to_physics_state(state)
        np.testing.assert_allclose(np.asarray(rec.u_wind), 1.5, rtol=0.02)
        np.testing.assert_allclose(np.asarray(rec.temperature), 271.0, rtol=0.02)
        np.testing.assert_allclose(np.asarray(rec.specific_humidity), 1e-3,
                                   rtol=0.02)
        np.testing.assert_allclose(
            np.asarray(rec.normalized_surface_pressure), 0.98, rtol=0.02)
        np.testing.assert_allclose(np.asarray(rec.tracers["co2_vmr"]), 400.0,
                                   rtol=0.02)

    def test_build_forcing_monthly_columns(self):
        from jcm.date import DateData
        import jax_datetime as jdt

        forcing = build_forcing(T63_FORCING, self.dycore)
        self.assertEqual(forcing.sea_surface_temperature.values.shape,
                         (12, 1, self.ncol))
        self.assertEqual(forcing.alb0.shape, (1, self.ncol))
        date = DateData.set_date(model_time=jdt.to_datetime("2000-07-15"))
        sliced = forcing.select(date, calendar="365_day")
        sst = np.asarray(sliced.sea_surface_temperature)
        self.assertEqual(sst.shape, (1, self.ncol))
        self.assertGreater(sst.min(), 200.0)
        self.assertLess(sst.max(), 320.0)
        ice = np.asarray(sliced.sice_am)
        self.assertGreaterEqual(ice.min(), 0.0)
        self.assertLessEqual(ice.max(), 1.0)

    def test_to_xarray_regrids_to_latlon(self):
        import jax

        ps0 = self.dycore.to_physics_state(self.state)
        stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), ps0, ps0)
        # Physics diagnostics ride along when their trailing shape matches
        # the column layout — both the flattened (ncols,) layout the
        # column-vectorized physics produces and (nlev, ncols).
        physics = {
            "radiation": {
                "toa_lw_up": jnp.ones((2, self.ncol)) * 240.0,
                "sw_heating": jnp.zeros((2, 47, self.ncol)),
            },
            "skipped_scalar": jnp.zeros((2,)),
        }
        preds = Predictions(dynamics=stacked, physics=physics, times=None)
        ds = self.dycore.to_xarray(
            preds, np.array([0.0, self.dycore.dt_seconds / 86400.0]))
        self.assertIn("lat", ds.coords)
        self.assertIn("lon", ds.coords)
        self.assertEqual(ds["temperature"].dims, ("time", "level", "lon", "lat"))
        self.assertTrue(np.isfinite(ds["temperature"].values).all())
        self.assertTrue(np.isfinite(ds["normalized_surface_pressure"].values).all())
        # Output is surface-first per the repo convention: level[0] ~ sigma 1.
        self.assertGreater(float(ds["level"][0]), 0.9)
        self.assertLess(float(ds["level"][-1]), 1e-4)
        # Selecting a level by coordinate value works (no blind indexing).
        surf_T = ds["temperature"].sel(level=1.0, method="nearest")
        self.assertGreater(float(surf_T.mean()), 240.0)
        # Physics diagnostics with column-shaped trailing dims are included;
        # non-spatial leaves are skipped rather than crashing the export.
        self.assertIn("radiation.toa_lw_up", ds)
        self.assertEqual(ds["radiation.sw_heating"].dims,
                         ("time", "level", "lon", "lat"))
        self.assertNotIn("skipped_scalar", ds)


@pytest.mark.slow
class TestCoupledEchamSmoke(unittest.TestCase):
    """Coupled CAM-SE + moist-ECHAM (grey radiation) smoke tests at ne3.

    Two drivers are exercised:

    * the full :class:`jcm.model.Model` (with ``physics_dtype=float64`` —
      see the module docstring of ``dycore.py`` for why Model's carry
      templating fixes the physics working dtype to the process default);
    * the direct-drive loop with the production float32 physics split.
    """

    def test_model_drives_coupled_run(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        dycore = PysesCamSEDycore(
            nx=3, npt=4, dt_seconds=900.0, terrain_file=T63_TERRAIN,
            physics_dtype=jnp.float64,
        )
        model = Model(
            dycore=dycore,
            time_step=dycore.dt_seconds / 60.0,
            physics=echam_physics(radiation_scheme="grey"),
        )
        forcing = build_forcing(T63_FORCING, dycore)
        dt_days = dycore.dt_seconds / 86400.0
        model.run(forcing=forcing, save_interval=dt_days, total_time=2 * dt_days)
        ps_end = dycore.to_physics_state(model._final_dycore_state)
        T = np.asarray(ps_end.temperature)
        self.assertTrue(np.isfinite(T).all())
        self.assertGreater(T.min(), 140.0)
        self.assertLess(T.max(), 360.0)
        # Moisture spun up from the dry start (evaporation active).
        self.assertGreater(float(np.asarray(ps_end.specific_humidity).max()), 0.0)

    def test_model_drives_float32_physics(self):
        """The production split: float32 physics carry on the float64 core.

        Exercises the two fixes that unlocked it: radiation lax.cond
        branches pinned to the carry dtype, and Model casting its carry
        template to ``dycore.physics_dtype``.
        """
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        dycore = PysesCamSEDycore(
            nx=3, npt=4, dt_seconds=900.0, terrain_file=T63_TERRAIN,
            physics_dtype=jnp.float32,
        )
        model = Model(
            dycore=dycore,
            time_step=dycore.dt_seconds / 60.0,
            physics=echam_physics(radiation_scheme="grey"),
        )
        forcing = build_forcing(T63_FORCING, dycore)
        dt_days = dycore.dt_seconds / 86400.0
        model.run(forcing=forcing, save_interval=dt_days, total_time=2 * dt_days)
        ps_end = dycore.to_physics_state(model._final_dycore_state)
        self.assertEqual(ps_end.temperature.dtype, jnp.float32)
        T = np.asarray(ps_end.temperature)
        self.assertTrue(np.isfinite(T).all())
        self.assertGreater(T.min(), 140.0)
        self.assertLess(T.max(), 360.0)

    def test_direct_drive_loop_without_model(self):
        """Caller-owned production loop: to_physics_state → physics → step.

        Runs the ECHAM physics at float64. A pure-float32 ECHAM pass is
        currently impossible with x64 enabled regardless of the driver: the
        shipped grey-radiation term produces *mixed* f32/f64 outputs from
        f32 inputs (strong float64 table constants promote some leaves),
        so its compute-vs-cached ``lax.cond`` cannot type-check against any
        uniform-dtype carry. The float32 split therefore lives at the
        state/tendency seam (``test_forced_step_advances_and_stays_finite``
        exercises it); full-f32 ECHAM physics needs a physics-side
        dtype-stability fix and is tracked as an open issue.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics_interface import compute_physics_step_gridpoint

        dycore = PysesCamSEDycore(
            nx=3, npt=4, dt_seconds=900.0, terrain_file=T63_TERRAIN,
            physics_dtype=jnp.float64,
        )
        physics = echam_physics(radiation_scheme="grey")
        specs = {s.name: s for s in physics.required_tracers()}
        dycore.tracer_specs = specs
        physics.cache_coords(dycore.coords)
        physics.dt_seconds = dycore.dt_seconds

        forcing_all = build_forcing(T63_FORCING, dycore)
        from jcm.date import DateData
        import jax_datetime as jdt

        state = dycore.initial_state(None, tracer_specs=specs)
        carry = physics.initial_carry_state(dycore.coords)
        for step in range(2):
            ps = dycore.to_physics_state(state)
            date = DateData.set_date(
                model_time=jdt.to_datetime("2000-01-01")
                + jdt.Timedelta(seconds=int(dycore.sim_time(state))))
            forcing_now = forcing_all.select(date, calendar="365_day")
            tend, carry = compute_physics_step_gridpoint(
                ps, forcing_now, dycore.terrain, carry,
                physics=physics, time_step=dycore.dt_seconds,
            )
            state = dycore.step(state, tend)
        ps_end = dycore.to_physics_state(state)
        T = np.asarray(ps_end.temperature)
        self.assertTrue(np.isfinite(T).all())
        self.assertGreater(T.min(), 140.0)
        self.assertLess(T.max(), 360.0)


if __name__ == "__main__":
    unittest.main()
