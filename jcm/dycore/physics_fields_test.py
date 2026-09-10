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

    def test_model_enables_the_provider_it_needs(self):
        # A declared requirement RESOLVES against a capable backend rather
        # than rejecting it; only genuine incapability is an error (see
        # OmegaDiagnosticTermTest for that half).
        coords = _coords()
        physics = ComposablePhysics(terms=[_FrontgfRecorder()])
        model = Model(coords=coords, physics=physics)
        self.assertIn("frontogenesis", model._dycore_field_names)

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


class DinosaurOmegaProviderTest(unittest.TestCase):
    """The omega (Dp/Dt) provider against closed forms (jax-gcm#409).

    A horizontally uniform divergence ``D0`` over flat surface pressure
    has an exact omega on both verticals: the continuity integrals reduce
    to sums of constants, so the only error is the spectral round-trip.
    """

    D0_SI = 1e-5    # 1/s
    PS_SI = 97000.0  # Pa

    def _dycore(self, coords):
        return DinosaurDycore(coords=coords,
                              terrain=TerrainData.aquaplanet(coords),
                              dt_seconds=600.0, compute_omega=True)

    def _uniform_divergence_state(self, dycore, d0_si, ps_si):
        from dinosaur.hybrid_coordinates import HybridCoordinates
        from dinosaur.primitive_equations import State
        from dinosaur.scales import units
        coords = dycore.coords
        hor = coords.horizontal
        nlev = coords.vertical.layers
        nlon, nlat = hor.nodal_shape
        specs = dycore.physics_specs
        d0 = float(specs.nondimensionalize(d0_si / units.second))
        # Each coordinate family's OWN log-ps convention (state_bridge):
        # hybrid stores log(P_s) in nondim pressure units, sigma stores
        # the normalized log(P_s / p0). Building the state any other way
        # validates the math under a convention no real run uses (which
        # is how the sigma-path p0 factor initially slipped through).
        if isinstance(coords.vertical, HybridCoordinates):
            lsp = np.log(float(specs.nondimensionalize(
                ps_si * units.pascal)))
        else:
            lsp = np.log(ps_si / float(dycore.constants.p0))
        zeros = np.zeros((nlev, nlon, nlat))
        return State(
            vorticity=hor.to_modal(zeros),
            divergence=hor.to_modal(d0 * np.ones((nlev, nlon, nlat))),
            temperature_variation=hor.to_modal(zeros),
            log_surface_pressure=hor.to_modal(
                lsp * np.ones((1, nlon, nlat))),
            tracers={}, sim_time=0.0,
        )

    def test_declaration(self):
        coords = _coords()
        dycore = self._dycore(coords)
        self.assertEqual(dycore.physics_field_names(), ("omega",))

    def test_rest_state_omega_is_zero(self):
        coords = _coords()
        dycore = self._dycore(coords)
        state = self._uniform_divergence_state(dycore, 0.0, self.PS_SI)
        omega = np.asarray(dycore._compute_omega(state))
        self.assertEqual(np.max(np.abs(omega)), 0.0)

    def test_sigma_uniform_divergence_closed_form(self):
        # On sigma levels with flat ps, uniform D0 gives sigma_dot = 0 and
        # d ln ps/dt = -D0, so omega(sigma) = -sigma ps D0 exactly.
        coords = _coords()
        dycore = self._dycore(coords)
        state = self._uniform_divergence_state(dycore, self.D0_SI, self.PS_SI)
        omega = np.asarray(dycore._compute_omega(state))
        sigma = np.asarray(coords.vertical.centers)
        want = -sigma[:, None, None] * self.PS_SI * self.D0_SI
        want = np.broadcast_to(want, omega.shape)
        # The only error source is the f32 spectral round-trip, which is
        # absolute (proportional to the field scale), so pair a loose
        # rtol with an atol at ~1e-4 of the field maximum.
        np.testing.assert_allclose(omega, want, rtol=1e-4,
                                   atol=1e-4 * np.abs(want).max())

    def test_omega_ignores_nodal_tracers(self):
        # Under semi-Lagrangian advection extra tracers are stored NODAL
        # in state.tracers; the diagnostic-state helpers' unconditional
        # to_nodal over them crashes on the shape mismatch unless
        # _compute_omega strips tracers first. Same closed form must come
        # out with a nodal tracer along for the ride.
        coords = _coords()
        dycore = self._dycore(coords)
        state = self._uniform_divergence_state(dycore, self.D0_SI, self.PS_SI)
        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape
        state = state.replace(
            tracers={"dust": jnp.ones((nlev, nlon, nlat))})
        omega = np.asarray(dycore._compute_omega(state))
        sigma = np.asarray(coords.vertical.centers)
        want = -sigma[:, None, None] * self.PS_SI * self.D0_SI
        np.testing.assert_allclose(
            omega, np.broadcast_to(want, omega.shape), rtol=1e-4,
            atol=1e-4 * np.abs(want).max())

    def test_hybrid_uniform_divergence_matches_numpy_reference(self):
        # Same setup on the ECHAM L47 hybrid grid, checked against an
        # independent numpy evaluation of the mass-flux formula from the
        # (a, b) tables.
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.utils import get_coords
        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        dycore = self._dycore(coords)
        state = self._uniform_divergence_state(dycore, self.D0_SI, self.PS_SI)
        omega = np.asarray(dycore._compute_omega(state))

        a = np.asarray(coords.vertical.a_boundaries)
        b = np.asarray(coords.vertical.b_boundaries)
        dp = np.diff(a) + np.diff(b) * self.PS_SI
        d = dp * self.D0_SI
        dps_dt = -d.sum()
        mass_flux = -np.concatenate([[0.0], np.cumsum(d)]) + b * d.sum()
        b_full = 0.5 * (b[:-1] + b[1:])
        want = b_full * dps_dt + 0.5 * (mass_flux[:-1] + mass_flux[1:])
        want = np.broadcast_to(want[:, None, None], omega.shape)
        np.testing.assert_allclose(omega, want, rtol=1e-4,
                                   atol=1e-4 * np.abs(want).max())
        # omega < 0 everywhere: parcels ride their coordinate surface
        # while uniform divergence drains the column, so the pressure
        # above each parcel (and hence its own pressure) falls.
        self.assertLess(omega.max(), 0.0)


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

    def test_omega_injection_reaches_terms_in_a_real_run(self):
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )

        class _OmegaRecorder(PhysicsTerm):
            name: ClassVar[str] = "omega_recorder"
            category: ClassVar[str] = "test"
            provides: ClassVar[tuple[str, ...]] = ("omega_absmax",)
            requires_dycore_fields: ClassVar[tuple[str, ...]] = ("omega",)

            def __call__(self, state, diagnostics, forcing, terrain):
                fields = diagnostics.get("_dycore_fields", {})
                omega = fields.get("omega")
                absmax = (jnp.max(jnp.abs(omega)) if omega is not None
                          else jnp.asarray(jnp.nan, state.temperature.dtype))
                diag = absmax * jnp.ones_like(
                    state.normalized_surface_pressure)
                tend = PhysicsTendency.zeros(state.temperature.shape)
                return tend, {**diagnostics, "omega_absmax": diag}

        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0, compute_omega=True)
        physics = held_suarez_physics() + _OmegaRecorder()
        model = Model(dycore=dycore, physics=physics)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=4 * dt_days)
        absmax = np.asarray(preds.physics["omega_absmax"])
        self.assertTrue(np.isfinite(absmax).all())
        # A baroclinically forced spun-up-from-rest run develops real
        # vertical motion within a couple of steps; a provider that
        # silently returned zeros would fail here.
        self.assertGreater(float(absmax[-1].max()), 0.0)
        # Sanity on magnitude: Pa/s, not nondimensional or hPa/day.
        self.assertLess(float(absmax.max()), 1e3)

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


class OmegaDiagnosticTermTest(unittest.TestCase):
    """The model-agnostic omega output term (jax-gcm#409's original ask)."""

    def test_factories_expose_the_flag(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        self.assertEqual(speedy_physics().required_dycore_fields(), ())
        physics = speedy_physics(diagnose_omega=True)
        self.assertEqual(physics.required_dycore_fields(), ("omega",))
        self.assertEqual(physics.terms[-1].name, "omega_diagnostic")

    def test_echam_factory_keeps_aerocom_terminal(self):
        from jcm.physics.echam.echam_terms import echam_physics
        physics = echam_physics(radiation_scheme="grey",
                                diagnose_omega=True, enable_aerocom=True)
        names = [t.name for t in physics.terms]
        self.assertIn("omega_diagnostic", names)
        self.assertEqual(names[-1], "aerocom_diagnostics")
        # AerocomDiagnostics only consumes opportunistically, so the
        # requirement comes from the omega term alone.
        self.assertIn("omega", physics.required_dycore_fields())

    def test_model_enables_a_switched_off_provider(self):
        """A capable backend gets its provider turned on, not rejected.

        The flags are pure cost knobs and a term that declares a field
        cannot run without it, so ``Model`` resolves the contract rather
        than making every caller pre-configure the dycore. Before this,
        ``Model(coords=..., physics=echam_physics())`` raised — which is
        every library caller of the ECHAM package, since Tiedtke's
        ``cubasmc`` trigger requires omega (#697).
        """
        from jcm.physics.speedy.speedy_terms import speedy_physics
        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0)
        self.assertEqual(dycore.physics_field_names(), ())
        model = Model(dycore=dycore,
                      physics=speedy_physics(diagnose_omega=True))
        self.assertTrue(dycore.compute_omega)
        self.assertIn("omega", model._dycore_field_names)

    def test_model_validation_fails_when_backend_cannot_provide(self):
        """A backend with no such provider at all still fails loudly."""
        from jcm.physics.speedy.speedy_terms import speedy_physics
        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0)
        # Stand in for a backend like pySES that computes omega internally
        # but exposes no ``compute_omega`` provider flag (#698).
        dycore.physics_field_names = lambda: ()
        del dycore.compute_omega
        with self.assertRaisesRegex(ValueError, "omega"):
            Model(dycore=dycore, physics=speedy_physics(diagnose_omega=True))

    def test_speedy_run_emits_omega(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        coords = _coords()
        dycore = DinosaurDycore(coords=coords,
                                terrain=TerrainData.aquaplanet(coords),
                                dt_seconds=1800.0, compute_omega=True)
        model = Model(dycore=dycore,
                      physics=speedy_physics(diagnose_omega=True))
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=4 * dt_days)
        omega = np.asarray(preds.physics["omega"])
        nlev = coords.vertical.layers
        self.assertIn(nlev, omega.shape)
        self.assertTrue(np.isfinite(omega).all())
        # Real vertical motion in Pa/s: nonzero, and nowhere near the
        # ~1e-5 of a normalized-pressure mistake or the ~1e5 of a
        # double-scaled one.
        absmax = float(np.abs(omega[-1]).max())
        self.assertGreater(absmax, 1e-6)
        self.assertLess(absmax, 1e3)


class EchamFactoryTest(unittest.TestCase):
    def test_gw_scheme_switch(self):
        from jcm.physics.echam.echam_terms import echam_physics

        frontal = echam_physics(radiation_scheme="grey",
                                gw_scheme="frontal")
        names = [t.name for t in frontal.terms]
        self.assertIn("frontal_gravity_wave_drag", names)
        self.assertNotIn("hines_gwd", " ".join(names))
        # ``omega`` comes from Tiedtke convection: ECHAM's ``lmfmid``
        # mid-level trigger (``cubasmc``) is on by default and gates on the
        # resolved vertical velocity, so every ECHAM package requires it.
        self.assertEqual(frontal.required_dycore_fields(),
                         ("omega", "frontogenesis"))

        hines = echam_physics(radiation_scheme="grey", gw_scheme="hines")
        self.assertEqual(hines.required_dycore_fields(), ("omega",))

        none = echam_physics(radiation_scheme="grey", gw_scheme="none")
        none_names = [t.name for t in none.terms]
        self.assertNotIn("frontal_gravity_wave_drag", none_names)

        with self.assertRaises(ValueError):
            echam_physics(radiation_scheme="grey", gw_scheme="bogus")


if __name__ == "__main__":
    unittest.main()
