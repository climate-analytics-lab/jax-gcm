"""Tests for composable SPEEDY physics (speedy_terms.py).

Smoke tests: build via the speedy_physics() factory, run through Model,
exercise nnx.grad and the replace() composition operator.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jcm.physics.speedy.speedy_terms import (
    _call_legacy_speedy,
    speedy_physics,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.params import Parameters
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData


def _make_test_state(coords):
    """Create a test PhysicsState with physically plausible values."""
    nlev = coords.nodal_shape[0]
    nodal_shape = coords.horizontal.nodal_shape
    shape_3d = (nlev,) + nodal_shape

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    return PhysicsState(
        u_wind=5.0 * jax.random.normal(keys[0], shape_3d),
        v_wind=5.0 * jax.random.normal(keys[1], shape_3d),
        temperature=250.0 + 20.0 * jax.random.normal(keys[2], shape_3d),
        specific_humidity=jnp.abs(3e-3 * jax.random.normal(keys[3], shape_3d)),
        geopotential=jnp.broadcast_to(
            jnp.linspace(50000, 0, nlev)[:, None, None], shape_3d
        ),
        normalized_surface_pressure=(
            1.0 + 0.01 * jax.random.normal(keys[4], nodal_shape)
        ),
    )


class TestSpeedyNumericalEquivalence(unittest.TestCase):
    """Smoke tests for composable SPEEDY physics."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.params = Parameters.default()
        self.state = _make_test_state(self.coords)
        self.forcing = ForcingData.zeros(self.coords.horizontal.nodal_shape)
        self.terrain = TerrainData.aquaplanet(self.coords)
        self.date = DateData.zeros()

    def test_composable_with_model(self):
        """ComposablePhysics can be passed to Model and run."""
        from jcm.model import Model

        composable = speedy_physics(parameters=self.params)
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=composable,
        )
        # Just verify it doesn't crash during a very short run
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    def test_adapter_preserves_legacy_humidity_numerics(self):
        """Only q crosses SPEEDY's kg/kg <-> g/kg compatibility boundary."""
        shape = self.state.specific_humidity.shape
        canonical = self.state.copy(
            specific_humidity=jnp.full(shape, 7.5e-3, dtype=jnp.float32),
            tracers={"signed": jnp.full(shape, -2.0, dtype=jnp.float32)},
        )

        def legacy_routine(legacy_state):
            # A stand-in for any translated SPEEDY routine: observe its input
            # and emit a g/kg/s q tendency plus unrelated fields/tracers.
            tendency = PhysicsTendency.zeros(
                shape,
                temperature=jnp.full(shape, 4.0, dtype=jnp.float32),
                specific_humidity=jnp.full(shape, 2.5, dtype=jnp.float32),
                tracers={"signed": jnp.full(shape, -3.0, dtype=jnp.float32)},
            )
            return tendency, {"legacy_state": legacy_state}

        tendency, data = _call_legacy_speedy(legacy_routine, canonical)
        legacy_state = data["legacy_state"]

        np.testing.assert_allclose(
            legacy_state.specific_humidity, 7.5, rtol=1e-6,
        )
        np.testing.assert_allclose(tendency.specific_humidity, 2.5e-3)
        np.testing.assert_allclose(tendency.temperature, 4.0)
        np.testing.assert_allclose(
            legacy_state.tracers["signed"], canonical.tracers["signed"],
        )
        np.testing.assert_allclose(tendency.tracers["signed"], -3.0)

    def test_nnx_grad_through_composable_speedy(self):
        """Gradients flow through the composable SPEEDY physics."""
        composable = speedy_physics(parameters=self.params, checkpoint_terms=False)
        composable.cache_coords(self.coords)

        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(self.state, self.forcing, self.terrain)
            return jnp.sum(tend.temperature ** 2)

        grads = nnx.grad(loss_fn)(composable)

        # Verify at least some parameter gradients are non-zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        any_nonzero = any(
            jnp.any(leaf != 0.0) for leaf in grad_leaves
            if hasattr(leaf, 'shape')
        )
        self.assertTrue(any_nonzero, "All gradients are zero")

    def test_replace_term(self):
        """Verify we can replace a SPEEDY term category."""
        composable = speedy_physics(parameters=self.params, checkpoint_terms=False)

        # Replace convection with different convection params
        from jcm.physics.speedy.speedy_terms import SpeedyConvection
        from jcm.physics.speedy.params import ConvectionParameters

        new_conv_params = ConvectionParameters(
            psmin=jnp.array(0.8),
            trcnv=jnp.array(12.0),  # doubled relaxation time
            rhil=jnp.array(0.7),
            rhbl=jnp.array(0.9),
            entmax=jnp.array(0.5),
            smf=jnp.array(0.8),
        )
        replaced = composable.replace("convection", SpeedyConvection(new_conv_params))

        replaced.cache_coords(self.coords)
        tend, _ = replaced.compute_tendencies(self.state, self.forcing, self.terrain)

        # Should produce valid (non-NaN) tendencies
        self.assertFalse(jnp.any(jnp.isnan(tend.temperature)))

    def test_speedy_humidity_publishes_bolton_rh_without_touching_physics_input(self):
        """SpeedyHumidity must leave ``humidity.rh`` (a physics input read by
        speedy_vdiff/speedy_shortwave/speedy_surface_flux) exactly as the
        reference computes it, however extreme, while publishing a
        separate ``relative_humidity`` diagnostic computed from the SAME
        Bolton formula ``MoistAirColumnState`` uses for every ECHAM
        composition (#407 and its Codex review: a first version instead
        published SPEEDY's own ice/water-switching ratio, sanity-capped,
        under this name -- silently redefining an existing cross-package
        output contract that ``tools/aerocom_cmor.py`` and the science docs
        both rely on).
        """
        from jcm.physics.diagnostics.moist_air_state import (
            bolton_relative_humidity,
        )
        from jcm.physics.speedy.speedy_terms import SpeedyHumidity

        term = SpeedyHumidity()
        term.cache_coords(self.coords)

        state = self.state.copy(
            temperature=self.state.temperature.at[0].set(126.0),
        )

        _, diagnostics = term(state, {}, self.forcing, self.terrain)

        physics_rh = diagnostics["_humidity"].rh
        reported_rh = diagnostics["relative_humidity"]

        # The physics-facing field is the raw, uncapped ratio -- the same
        # pathological value speedy_vdiff.py etc. would read unmodified --
        # and is wildly different from the Bolton-based public field, since
        # the two are now genuinely different formulas (not the same ratio
        # under two names).
        self.assertTrue(float(jnp.max(physics_rh)) > 1e4)
        self.assertFalse(
            jnp.allclose(reported_rh, physics_rh, rtol=1e-2).item(),
        )

        # relative_humidity matches a direct call to the shared function on
        # SPEEDY's own pure-sigma pressure -- i.e. it is exactly the same
        # computation an ECHAM composition's MoistAirColumnState would do
        # given the same (T, q, p), not a rescaled version of humidity.rh.
        import jcm.constants as c
        from jcm.physics.speedy.speedy_coords import SpeedyCoords
        fsg = SpeedyCoords.from_coordinate_system(self.coords).fsg
        pressure_full = (
            fsg.reshape((-1,) + (1,) * state.normalized_surface_pressure.ndim)
            * (state.normalized_surface_pressure * c.p0)[jnp.newaxis]
        )
        expected = bolton_relative_humidity(
            state.temperature, state.specific_humidity, pressure_full,
        )
        np.testing.assert_allclose(reported_rh, expected, rtol=1e-6)

    def test_every_output_variable_is_documented(self):
        """Every published variable carries units and a description.

        The SPEEDY units table reaches the output through the terms that
        publish the diagnostics, so a term family with no table — or a new
        diagnostic added without a row — ships an undocumented variable.
        """
        from jcm.model import Model

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        model = Model(coords=coords, terrain=TerrainData.from_coords(coords),
                      physics=speedy_physics(checkpoint_terms=False))
        ds = model.run(save_interval=1.0, total_time=1.0).to_xarray()

        undocumented = sorted(str(v) for v in ds.data_vars
                              if not ds[v].attrs.get("description"))
        self.assertEqual(undocumented, [],
                         "add a row to jcm/physics/speedy/units_table.csv")
        self.assertEqual(ds["specific_humidity"].attrs["units"], "kg kg-1")


class TestSpeedyPublishedFields(unittest.TestCase):
    """The package-independent fields a SPEEDY run publishes, read off one model day."""

    @classmethod
    def setUpClass(cls):
        from jcm.model import Model
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        model = Model(coords=coords, terrain=TerrainData.from_coords(coords),
                      physics=speedy_physics(checkpoint_terms=False))
        cls.ds = model.run(save_interval=1.0, total_time=1.0).to_xarray().isel(time=-1)

    def test_radiation_fields_are_speedys_own_fluxes(self):
        ds = self.ds
        np.testing.assert_allclose(ds["radiation.toa_lw_up"], ds["longwave_rad.ftop"], rtol=1e-6)
        np.testing.assert_allclose(
            ds["radiation.toa_sw_down"] - ds["radiation.toa_sw_up"], ds["shortwave_rad.ftop"],
            rtol=1e-5, atol=1e-3)
        # the clear-sky fields are populated (the cloud effects themselves are tested in
        # speedy_shortwave_test; a day from isothermal rest has no cloud yet)
        self.assertGreater(float(ds["radiation.toa_lw_up_clear"].min()), 100.0)
        self.assertGreater(float(ds["radiation.toa_sw_up_clear"].max()), 0.0)
        cover = ds["radiation.total_cloud_cover"]
        self.assertTrue(bool(((cover >= ds["shortwave_rad.cloudc"] - 1e-6) & (cover <= 1.0 + 1e-6)).all()))

    def test_surface_longwave_fields_are_the_fluxes_through_the_surface(self):
        """SPEEDY's rlds / rlus are the absorbed and emitted parts; the published
        fields are the full fluxes (CMIP rlds / rlus), with the same net.
        """
        ds = self.ds
        emisfc = float(Parameters.default().mod_radcon.emisfc)
        absorbed, emitted = ds["surface_flux.rlds"], ds["surface_flux.rlus"]
        down, up = ds["radiation.surface_lw_down"], ds["radiation.surface_lw_up"]
        np.testing.assert_allclose(down, absorbed / emisfc, rtol=1e-6)
        np.testing.assert_allclose(up, emitted + (1.0 - emisfc) * down, rtol=1e-6)
        np.testing.assert_allclose(up - down, emitted - absorbed, rtol=1e-4, atol=1e-2)

    def test_layer_pressures_span_the_column(self):
        ds = self.ds
        ps = ds["normalized_surface_pressure"] * 1.0e5
        # the diagnostics are from the physics call, the state's pressure from after the step
        np.testing.assert_allclose(ds["pressure_thickness"].sum("level"), ps, rtol=2e-4)
        self.assertTrue(bool((ds["pressure_full"] < ps).all()))

if __name__ == "__main__":
    unittest.main()
