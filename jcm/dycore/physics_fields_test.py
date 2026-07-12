"""Tests for the dycore-supplied physics-fields contract (CAM pbuf pattern).

Covers the protocol default, Model construction-time validation, the
dinosaur frontogenesis provider, per-step injection under
``"_dycore_fields"`` (including scan-carry structure stability in both
snapshot and averaged modes), and the ``echam_physics(gw_scheme=...)``
composition switch.
"""

import unittest
from typing import ClassVar

import numpy as np
import jax.numpy as jnp

from jcm.dycore.dinosaur.dycore import DinosaurDycore
from jcm.model import Model
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


def _coords():
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    return get_speedy_coords(layers=8, spectral_truncation=21)


class _FrontgfRecorder(PhysicsTerm):
    """Test term: republish the injected dycore field as a diagnostic."""

    name: ClassVar[str] = "frontgf_recorder"
    category: ClassVar[str] = "test"
    provides: ClassVar[tuple[str, ...]] = ("frontgf_absmax",)
    requires_dycore_fields: ClassVar[tuple[str, ...]] = ("frontogenesis",)

    def __call__(self, state, diagnostics, forcing, terrain):
        fields = diagnostics.get("_dycore_fields")
        frontgf = fields["frontogenesis"] if isinstance(fields, dict) else None
        absmax = (jnp.max(jnp.abs(frontgf)) if frontgf is not None
                  else jnp.asarray(jnp.nan, state.temperature.dtype))
        diag = absmax * jnp.ones_like(state.normalized_surface_pressure)
        tend = PhysicsTendency.zeros(state.temperature.shape)
        return tend, {**diagnostics, "frontgf_absmax": diag}


class ProtocolTest(unittest.TestCase):
    def test_default_backend_provides_nothing(self):
        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0)
        self.assertEqual(dycore.physics_field_names(), ())
        self.assertEqual(dycore.physics_fields(None, None), {})

    def test_model_validation_fails_without_provider(self):
        coords = _coords()
        physics = ComposablePhysics(terms=[_FrontgfRecorder()])
        with self.assertRaisesRegex(ValueError, "frontogenesis"):
            Model(coords=coords, physics=physics)

    def test_upstream_provider_term_satisfies_requirement(self):
        # A physics-side term whose ``provides`` names the field counts as
        # a provider, so composition without the dycore flag succeeds.
        class _Provider(PhysicsTerm):
            name: ClassVar[str] = "frontgf_provider"
            category: ClassVar[str] = "prepare"
            provides: ClassVar[tuple[str, ...]] = ("frontogenesis",)

            def __call__(self, state, diagnostics, forcing, terrain):
                tend = PhysicsTendency.zeros(state.temperature.shape)
                return tend, {**diagnostics,
                              "frontogenesis": jnp.zeros_like(
                                  state.temperature)}

        physics = ComposablePhysics(
            terms=[_Provider(), _FrontgfRecorder()])
        self.assertEqual(physics.required_dycore_fields(), ())


class DinosaurProviderTest(unittest.TestCase):
    def test_provider_matches_direct_frontogenesis_function(self):
        from jcm.physics.gravity_waves.spectral.frontogenesis import (
            frontogenesis_function,
        )
        import jcm.constants as c

        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0,
                                compute_frontogenesis=True)
        self.assertEqual(dycore.physics_field_names(), ("frontogenesis",))

        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape
        rng = np.random.default_rng(0)
        lats = np.asarray(coords.horizontal.latitudes)
        lons = np.asarray(coords.horizontal.longitudes)
        u = 20.0 * np.cos(lats)[None, None, :] * np.ones((nlev, nlon, nlat))
        v = 5.0 * np.sin(lons)[None, :, None] * np.ones((nlev, nlon, nlat))
        temp = 250.0 + 5.0 * rng.standard_normal((nlev, nlon, nlat))
        state = PhysicsState.zeros(
            (nlev, nlon, nlat),
            u_wind=jnp.asarray(u, jnp.float32),
            v_wind=jnp.asarray(v, jnp.float32),
            temperature=jnp.asarray(temp, jnp.float32),
            normalized_surface_pressure=jnp.ones((nlon, nlat), jnp.float32),
        )
        out = dycore.physics_fields(None, state)
        self.assertEqual(set(out), {"frontogenesis"})
        got = np.asarray(out["frontogenesis"])
        self.assertEqual(got.shape, (nlev, nlon, nlat))
        self.assertTrue(np.isfinite(got).all())

        # Reference: identical theta computation, direct function call.
        p0 = float(dycore.constants.p0)
        boundaries = np.asarray(coords.vertical.boundaries)
        sigma_full = 0.5 * (boundaries[:-1] + boundaries[1:])
        p_full = sigma_full[:, None, None] * p0
        theta = temp * (p0 / p_full) ** float(c.akap)
        want = np.asarray(frontogenesis_function(
            jnp.asarray(u, jnp.float32), jnp.asarray(v, jnp.float32),
            jnp.asarray(theta, jnp.float32),
            lons=jnp.asarray(lons), lats=jnp.asarray(lats)))
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=1e-12)


class InjectionTest(unittest.TestCase):
    def _run(self, output_averages):
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0,
                                compute_frontogenesis=True)
        physics = held_suarez_physics() + _FrontgfRecorder()
        model = Model(dycore=dycore, physics=physics)
        dt_days = 30.0 / (60.0 * 24.0)
        return model.run(save_interval=2 * dt_days, total_time=4 * dt_days,
                         output_averages=output_averages)

    def test_injection_reaches_terms_snapshot_and_averaged(self):
        for output_averages in (False, True):
            preds = self._run(output_averages)
            absmax = np.asarray(preds.physics["frontgf_absmax"])
            # The recorder saw a real injected field on every saved frame:
            # finite, non-NaN (NaN would mean the key was missing).
            self.assertTrue(np.isfinite(absmax).all(),
                            msg=f"averaged={output_averages}")
            # The plumbing key itself must not leak into saved output.
            self.assertNotIn("_dycore_fields", preds.physics)

    def test_column_vectorized_fields_are_flattened(self):
        # Codex P1 on #568: with vectorize_columns=True the state terms see
        # is (nlev, ncols) while the injected dycore fields arrived
        # grid-shaped; ComposablePhysics must reshape them consistently.
        # The checker term adds frontgf's source level to the (ncols,)
        # surface pressure — a shape mismatch fails loudly at trace.
        class _ShapeChecker(PhysicsTerm):
            name: ClassVar[str] = "shape_checker"
            category: ClassVar[str] = "test"
            provides: ClassVar[tuple[str, ...]] = ("frontgf_plus_ps",)
            requires_dycore_fields: ClassVar[tuple[str, ...]] = (
                "frontogenesis",)

            def __call__(self, state, diagnostics, forcing, terrain):
                # ``.get`` fallback: get_empty_data's construction-time
                # probe runs terms WITHOUT dycore-field injection (part of
                # the contract — consumers must tolerate absence).
                fields = diagnostics.get("_dycore_fields", {})
                frontgf = fields.get("frontogenesis",
                                     jnp.zeros_like(state.temperature))
                combined = frontgf[0] + state.normalized_surface_pressure
                tend = PhysicsTendency.zeros(state.temperature.shape)
                return tend, {**diagnostics, "frontgf_plus_ps": combined}

        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0,
                                compute_frontogenesis=True)
        physics = ComposablePhysics(terms=[_ShapeChecker()],
                                    vectorize_columns=True)
        model = Model(dycore=dycore, physics=physics)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=2 * dt_days)
        out = np.asarray(preds.physics["frontgf_plus_ps"])
        self.assertTrue(np.isfinite(out).all())

    def test_frontal_gw_term_runs_end_to_end(self):
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.physics.gravity_waves.spectral.term import (
            FrontalGravityWaveDrag,
        )
        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0,
                                compute_frontogenesis=True)
        physics = held_suarez_physics() + FrontalGravityWaveDrag()
        model = Model(dycore=dycore, physics=physics)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=2 * dt_days)
        temp = np.asarray(preds.dynamics.temperature)
        self.assertTrue(np.isfinite(temp).all())


class EchamFactoryTest(unittest.TestCase):
    def test_gw_scheme_switch(self):
        from jcm.physics.echam.echam_terms import echam_physics

        frontal = echam_physics(radiation_scheme="grey",
                                gw_scheme="frontal")
        names = [t.name for t in frontal.terms]
        self.assertIn("frontal_gravity_wave_drag", names)
        self.assertNotIn("hines_gwd", " ".join(names))
        self.assertEqual(frontal.required_dycore_fields(),
                         ("frontogenesis",))

        hines = echam_physics(radiation_scheme="grey", gw_scheme="hines")
        self.assertEqual(hines.required_dycore_fields(), ())

        none = echam_physics(radiation_scheme="grey", gw_scheme="none")
        none_names = [t.name for t in none.terms]
        self.assertNotIn("frontal_gravity_wave_drag", none_names)

        with self.assertRaises(ValueError):
            echam_physics(radiation_scheme="grey", gw_scheme="bogus")


if __name__ == "__main__":
    unittest.main()
