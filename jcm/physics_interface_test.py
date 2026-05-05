import unittest
import jax.numpy as jnp
from dinosaur import primitive_equations_states
from dinosaur.scales import units
from jcm.constants import p0
from jcm.physics_interface import PhysicsState, physics_state_to_dynamics_state, dynamics_state_to_physics_state

class TestPhysicsInterfaceUnit(unittest.TestCase):
    def test_initial_state_conversion(self):
        from dinosaur.scales import SI_SCALE
        from dinosaur import primitive_equations
        from dinosaur import xarray_utils
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)
        kx, ix, il = 8, 96, 48
        temp = 288 * jnp.ones((kx, ix, il))
        u = jnp.ones((kx, ix, il)) * 0.5
        v = jnp.ones((kx, ix, il)) * -0.5
        q = jnp.ones((kx, ix, il)) * 0.5
        phi = jnp.ones((kx, ix, il)) * 5000
        sp = jnp.ones((kx, ix, il))

        coords = get_speedy_coords()
        _, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=coords,
            physics_specs=PHYSICS_SPECS,
            p0=p0*units.pascal,
        )
        ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
        truncated_orography = primitive_equations.truncated_modal_orography(aux_features[xarray_utils.OROGRAPHY], coords)

        primitive = primitive_equations.PrimitiveEquations(
            ref_temps,
            truncated_orography,
            coords,
            PHYSICS_SPECS)

        state = PhysicsState.zeros((kx, ix, il), u, v, temp, q, phi, sp)

        dynamics_state = physics_state_to_dynamics_state(state, primitive)
        physics_state_recovered = dynamics_state_to_physics_state(dynamics_state, primitive)

        self.assertTrue(jnp.allclose(state.temperature, physics_state_recovered.temperature))

    def test_verify_state(self):
        from jcm.physics_interface import verify_state, PhysicsState
        import jax.numpy as jnp

        kx, ix, il = 8, 96, 48
        qa = jnp.ones((kx, il, ix)) * -1

        state = PhysicsState.zeros((kx,ix,il), specific_humidity=qa)

        updated_state = verify_state(state)

        self.assertTrue(jnp.all(updated_state.specific_humidity >= 0))

        qa = jnp.ones((kx, il, ix)) * -1e-5

        state = PhysicsState.zeros((kx,ix,il), specific_humidity=qa)

        updated_state = verify_state(state)


class TestVerifyState(unittest.TestCase):
    """verify_state only enforces q >= 0; no upper cap (by design)."""

    def test_negative_q_clipped_to_zero(self):
        from jcm.physics_interface import verify_state
        kx, ix, il = 4, 8, 8
        q = jnp.array([-0.5, -1e-5, 0.005, 0.0])[:, None, None] * jnp.ones((kx, ix, il))
        state = PhysicsState.zeros((kx, ix, il), specific_humidity=q)
        out = verify_state(state)
        self.assertTrue(jnp.all(out.specific_humidity >= 0.0))
        self.assertTrue(jnp.allclose(out.specific_humidity[2], 0.005))

    def test_unphysically_high_q_not_capped(self):
        """Unphysically high q should NOT be silently clamped — we want the
        model to surface the bug, not hide it with a cap.
        """
        from jcm.physics_interface import verify_state
        kx, ix, il = 4, 8, 8
        q = jnp.full((kx, ix, il), 0.5)  # 500 g/kg — unphysical but uncapped
        state = PhysicsState.zeros((kx, ix, il), specific_humidity=q)
        out = verify_state(state)
        self.assertTrue(jnp.allclose(out.specific_humidity, 0.5))


class TestVerifyTendencies(unittest.TestCase):
    """verify_tendencies only enforces q_next >= 0; no upper cap (by design)."""

    def _make_state_and_tendency(self, q_init, dqdt):
        from jcm.physics_interface import PhysicsTendency
        shape = (8, 4, 4)
        state = PhysicsState.zeros(shape, specific_humidity=jnp.full(shape, q_init))
        tendency = PhysicsTendency.zeros(
            shape, specific_humidity=jnp.full(shape, dqdt)
        )
        return state, tendency

    def test_positive_tendency_within_bounds(self):
        """Normal positive tendency should pass through unchanged."""
        from jcm.physics_interface import verify_tendencies
        state, tend = self._make_state_and_tendency(q_init=0.005, dqdt=1e-5)
        result = verify_tendencies(state, tend, time_step=1800.0)
        self.assertTrue(jnp.allclose(result.specific_humidity, tend.specific_humidity))

    def test_negative_tendency_clipped_at_zero(self):
        """Tendency that would make q negative is clipped to exactly drain q."""
        from jcm.physics_interface import verify_tendencies
        state, tend = self._make_state_and_tendency(q_init=0.001, dqdt=-0.01)
        result = verify_tendencies(state, tend, time_step=1800.0)
        q_next = 0.001 + 1800.0 * result.specific_humidity
        self.assertTrue(jnp.all(q_next >= 0))

    def test_large_positive_tendency_not_capped(self):
        """A tendency that would drive q very high is NOT silently clamped
        (by design — masking would hide upstream bugs).
        """
        from jcm.physics_interface import verify_tendencies
        state, tend = self._make_state_and_tendency(q_init=0.001, dqdt=1.0)
        result = verify_tendencies(state, tend, time_step=1800.0)
        # Unchanged tendency passes through
        self.assertTrue(jnp.allclose(result.specific_humidity, tend.specific_humidity))


class TestVerifyTracerNonNegativity(unittest.TestCase):
    """``verify_state`` and ``verify_tendencies`` clip every positive-
    definite tracer (cloud water/ice/rain/snow/number, GHG mixing ratios)
    to ``>= 0`` — not just ``specific_humidity``.

    This is the defensive layer that catches small-magnitude negatives
    from the spectral round-trip of horizontally advected tracers (the
    same mechanism documented in PR #458 for the ``q`` cycle).
    """

    def test_negative_qc_clipped_to_zero(self):
        from jcm.physics_interface import verify_state
        kx, ix, il = 4, 8, 8
        state = PhysicsState.zeros(
            (kx, ix, il),
            tracers={
                "qc": jnp.full((kx, ix, il), -1e-7),
                "qi": jnp.full((kx, ix, il), -2e-9),
            },
        )
        out = verify_state(state)
        self.assertTrue(jnp.all(out.tracers["qc"] >= 0.0))
        self.assertTrue(jnp.all(out.tracers["qi"] >= 0.0))

    def test_unknown_tracer_passes_through_unchanged(self):
        """Tracers not in the positive-definite set must pass through
        unchanged — we don't want to silently clamp e.g. anomaly fields
        or signed perturbations a future module might add.
        """
        from jcm.physics_interface import verify_state
        kx, ix, il = 4, 8, 8
        signed = jnp.full((kx, ix, il), -0.3)
        state = PhysicsState.zeros(
            (kx, ix, il),
            tracers={"some_signed_diagnostic": signed},
        )
        out = verify_state(state)
        self.assertTrue(jnp.allclose(out.tracers["some_signed_diagnostic"], signed))

    def test_microphysics_tracers_clipped_in_state(self):
        """All ECHAM microphysics tracers must be clipped to ``>= 0``."""
        from jcm.physics_interface import verify_state
        shape = (4, 8, 8)
        tracers = {
            name: jnp.full(shape, -1e-8)
            for name in ("qc", "qi", "qr", "qs", "qnc", "qni")
        }
        state = PhysicsState.zeros(shape, tracers=tracers)
        out = verify_state(state)
        for name in tracers:
            self.assertTrue(
                jnp.all(out.tracers[name] >= 0.0),
                msg=f"tracer {name!r} not clipped to >= 0",
            )

    def test_negative_tracer_tendency_caps_at_zero(self):
        """A microphysics tendency that would make ``qc`` negative is
        capped at ``-qc / dt`` so the next step lands at exactly 0.
        """
        from jcm.physics_interface import verify_tendencies, PhysicsTendency
        shape = (4, 8, 8)
        state = PhysicsState.zeros(
            shape, tracers={"qc": jnp.full(shape, 1e-5)},
        )
        tend = PhysicsTendency.zeros(
            shape, tracers={"qc": jnp.full(shape, -1.0)},
        )
        result = verify_tendencies(state, tend, time_step=1800.0)
        qc_next = state.tracers["qc"] + 1800.0 * result.tracers["qc"]
        self.assertTrue(jnp.all(qc_next >= 0.0))

    def test_unknown_tracer_tendency_passes_through(self):
        """Tendencies of tracers not in the positive-definite set must
        pass through unchanged (same rationale as ``test_unknown_tracer
        _passes_through_unchanged``).
        """
        from jcm.physics_interface import verify_tendencies, PhysicsTendency
        shape = (4, 8, 8)
        state = PhysicsState.zeros(
            shape, tracers={"signed_diag": jnp.full(shape, 1.0)},
        )
        tend = PhysicsTendency.zeros(
            shape, tracers={"signed_diag": jnp.full(shape, -100.0)},
        )
        result = verify_tendencies(state, tend, time_step=1800.0)
        self.assertTrue(
            jnp.allclose(result.tracers["signed_diag"], tend.tracers["signed_diag"])
        )
