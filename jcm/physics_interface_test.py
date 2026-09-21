import unittest
import jax.numpy as jnp
import numpy as np
from dinosaur import primitive_equations_states
from dinosaur.scales import units
from jcm.constants import p0
from jcm.forcing import ForcingData
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData
from jcm.dycore.dinosaur.state_bridge import (
    dynamics_state_to_physics_state, physics_state_to_dynamics_state,
)

class TestPhysicsInterfaceUnit(unittest.TestCase):
    def test_initial_state_conversion(self):
        from dinosaur import primitive_equations
        from dinosaur import xarray_utils
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        # Use the constants-derived specs so coords.radius (from get_coords,
        # sourced from PhysicalConstants) matches physics_specs.radius.
        from jcm.dycore.dinosaur.dycore import PHYSICS_SPECS
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
        """A large positive tendency passes through so upstream bugs remain visible."""
        from jcm.physics_interface import verify_tendencies
        state, tend = self._make_state_and_tendency(q_init=0.001, dqdt=1.0)
        result = verify_tendencies(state, tend, time_step=1800.0)
        self.assertTrue(jnp.allclose(
            result.specific_humidity, tend.specific_humidity,
        ))


class _WaterDrainTerm(PhysicsTerm):
    """Toy operator-split sink used to exercise the interface accounting."""

    name = "water_drain"
    category = "microphysics"

    def __init__(self, drain_fraction: float):
        self.drain_fraction = float(drain_fraction)

    def required_tracers(self):
        """Declare the water fields whose tendencies this term emits."""
        return tuple(
            TracerSpec(name=name) for name in ("qc", "qi", "qr", "qs")
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        del forcing, terrain
        dt = diagnostics["_dt_seconds"]
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            specific_humidity=(
                -self.drain_fraction * state.specific_humidity / dt
            ),
            tracers={
                name: -self.drain_fraction * value / dt
                for name, value in state.tracers.items()
            },
        )
        # Two unequal layers make the pressure-weighted source distinguishable
        # from an unweighted vertical sum.
        dp = jnp.array([10_000.0, 30_000.0], dtype=state.temperature.dtype)
        return tendency, {
            **diagnostics,
            "pressure_thickness": dp[:, jnp.newaxis],
        }


class TestWaterPositivityAccounting(unittest.TestCase):
    """The retained water cap publishes the exact artificial source."""

    def _run(self, drain_fraction):
        from jcm.physics_interface import compute_physics_step_gridpoint

        shape = (2, 1, 1)
        q = jnp.array([1.0e-3, 2.0e-3]).reshape(shape)
        tracers = {
            "qc": jnp.array([2.0e-4, 4.0e-4]).reshape(shape),
            "qi": jnp.array([3.0e-4, 6.0e-4]).reshape(shape),
            "qr": jnp.array([4.0e-4, 8.0e-4]).reshape(shape),
            "qs": jnp.array([5.0e-4, 1.0e-3]).reshape(shape),
        }
        state = PhysicsState.zeros(
            shape,
            temperature=jnp.full(shape, 280.0),
            normalized_surface_pressure=jnp.ones((1, 1)),
            specific_humidity=q,
            tracers=tracers,
        )
        physics = ComposablePhysics(
            terms=[_WaterDrainTerm(drain_fraction)],
            checkpoint_terms=False,
            vectorize_columns=True,
            dt_seconds=10.0,
        )
        applied, diagnostics = compute_physics_step_gridpoint(
            state,
            ForcingData.zeros((1, 1)),
            TerrainData.single_column(),
            {},
            physics=physics,
            time_step=10.0,
        )
        return state, applied, diagnostics, physics

    def test_exact_per_field_and_pressure_weighted_source(self):
        from jcm import constants

        state, applied, diagnostics, physics = self._run(drain_fraction=2.0)
        correction = diagnostics["water_positivity_correction"]
        raw_q = -2.0 * state.specific_humidity / 10.0
        q_correction = correction["specific_humidity_tendency"]

        np.testing.assert_array_equal(
            np.asarray(applied.specific_humidity),
            np.asarray(-state.specific_humidity / 10.0),
        )
        np.testing.assert_allclose(
            np.asarray(raw_q.reshape(q_correction.shape) + q_correction),
            np.asarray(applied.specific_humidity.reshape(q_correction.shape)),
            rtol=0.0,
            atol=0.0,
        )

        expected_total = state.specific_humidity / 10.0
        for name, value in state.tracers.items():
            field = correction[f"{name}_tendency"]
            raw = -2.0 * value / 10.0
            np.testing.assert_allclose(
                np.asarray(raw.reshape(field.shape) + field),
                np.asarray(applied.tracers[name].reshape(field.shape)),
                rtol=0.0,
                atol=0.0,
            )
            expected_total = expected_total + value / 10.0

        expected_total = expected_total.reshape((2, 1))
        np.testing.assert_allclose(
            np.asarray(correction["total_water_tendency"]),
            np.asarray(expected_total),
            rtol=1e-7,
        )
        dp = jnp.array([10_000.0, 30_000.0])[:, jnp.newaxis]
        expected_column_source = jnp.sum(
            expected_total * dp / constants.grav, axis=0,
        )
        np.testing.assert_allclose(
            np.asarray(correction["column_water_source"]),
            np.asarray(expected_column_source),
            rtol=1e-7,
        )
        self.assertTrue(bool(jnp.all(q_correction >= 0.0)))

        # The carry used by Tiedtke on the next step must subtract what the
        # host actually integrated, not the overdrawn raw term sum.
        np.testing.assert_array_equal(
            np.asarray(diagnostics["_prev_step"]["q_tendency"]),
            np.asarray(applied.specific_humidity.reshape((2, 1))),
        )

        flattened = physics.data_struct_to_dict(
            diagnostics, nodal_shape=(2, 1, 1),
        )
        self.assertIn(
            "water_positivity_correction.specific_humidity_tendency",
            flattened,
        )
        self.assertEqual(
            flattened["water_positivity_correction.column_water_source"].shape,
            (1, 1),
        )

    def test_no_correction_when_cap_is_inactive(self):
        _, _, diagnostics, _ = self._run(drain_fraction=0.5)
        correction = diagnostics["water_positivity_correction"]
        for field in correction.values():
            np.testing.assert_array_equal(
                np.asarray(field), np.zeros_like(np.asarray(field)),
            )

    def test_correction_is_stop_gradient(self):
        import jax

        from jcm.physics_interface import (
            PhysicsTendency,
            _verify_tendencies_with_water_corrections,
        )

        def correction_sum(raw_rate):
            state = PhysicsState.zeros(
                (2, 1, 1),
                specific_humidity=jnp.full((2, 1, 1), 1.0e-3),
            )
            raw = PhysicsTendency.zeros(
                (2, 1, 1), specific_humidity=jnp.full((2, 1, 1), raw_rate),
            )
            _, correction = _verify_tendencies_with_water_corrections(
                state, raw, 10.0,
            )
            return jnp.sum(correction["specific_humidity"])

        self.assertEqual(float(jax.grad(correction_sum)(-1.0)), 0.0)


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

    def test_conservative_transport_of_aerosol_survives_the_interface(self):
        """A column-conserving redistribution must pass through untouched.

        Tracer vertical diffusion and convective transport both read the
        step-start state and both return conservative redistributions; their
        SUM can drive a donor cell negative while the column total is exact.
        A per-cell positivity cap would clamp the donor and leave the
        receiving cells' gain, creating mass — so aerosol and gas tracers are
        not capped here at all.
        """
        from jcm.physics_interface import PhysicsTendency, verify_tendencies
        dt = 1800.0
        shape = (2, 1, 1)
        q0 = jnp.array([0.0, 1.0]).reshape(shape)
        # Two individually valid updates, [0,1] -> [1,0] and -> [0.083,0.917].
        moved = (jnp.array([1.0, -1.0]).reshape(shape)
                 + jnp.array([0.083, -0.083]).reshape(shape)) / dt
        for name in ("m_ss_cor", "n_cor", "g_so2"):
            state = PhysicsState.zeros(shape, tracers={name: q0})
            tend = PhysicsTendency.zeros(shape, tracers={name: moved})
            out = verify_tendencies(state, tend, time_step=dt).tracers[name]
            self.assertAlmostEqual(
                float(jnp.sum(out)) * dt, 0.0, places=6,
                msg=f"{name}: the interface broke column conservation")
            np.testing.assert_allclose(np.asarray(out), np.asarray(moved))

    def test_water_tracers_capped_by_the_geometry_free_fallback(self):
        """Pins the bare per-cell cap of the standalone entry point.

        :func:`verify_tendencies` has no layer masses, so it applies only the
        per-cell positivity cap — the documented geometry-free fallback for
        unit tests and hosts that do not expose their vertical geometry. The
        column-conservative treatment that removes the cap's spurious water
        source (#806) lives on the gridpoint driver, where Δp is available, and
        is covered by ``TestWaterConservativeLimiter``.
        """
        from jcm.physics_interface import PhysicsTendency, verify_tendencies
        shape = (4, 8, 8)
        state = PhysicsState.zeros(shape, tracers={"qc": jnp.full(shape, 1e-5)})
        tend = PhysicsTendency.zeros(shape, tracers={"qc": jnp.full(shape, -1.0)})
        result = verify_tendencies(state, tend, time_step=1800.0)
        nxt = state.tracers["qc"] + 1800.0 * result.tracers["qc"]
        self.assertTrue(bool(jnp.all(nxt >= 0.0)))
        self.assertTrue(bool(jnp.allclose(nxt, 0.0)))

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


class TestWaterConservativeLimiter(unittest.TestCase):
    """#806: the water-mass positivity cap is column-conservative given Δp.

    The mechanism: vertical diffusion redistributes q/qc/qi (a conservative
    donor/receiver transfer), and a co-located sink can overdraw the donor
    layer. The bare per-cell cap clamps the donor at ``-q/dt`` while the
    receiver keeps its gain — creating column water. With the layer masses the
    limiter removes exactly that spurious column source again, so column water
    is conserved to round-off while every layer stays non-negative.
    """

    def setUp(self):
        import jax
        # f64 so the conservation closure can be asserted to round-off; the
        # session-level fixture restores the flag after the test.
        self._x64 = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)

    def tearDown(self):
        import jax
        jax.config.update("jax_enable_x64", self._x64)

    def _scenario(self):
        """Two-level column with an overdrawn donor.

        vdiff exports qc donor->receiver and a co-located sink overdraws the
        donor, so the summed donor tendency drives qc0 < 0.
        """
        dt = 100.0
        dp = jnp.array([10_000.0, 30_000.0], dtype=jnp.float64).reshape(2, 1, 1)
        qc = jnp.array([1.0e-3, 5.0e-3], dtype=jnp.float64).reshape(2, 1, 1)
        redistribution = 2.0e-5      # conservative donor export rate
        sink = 2.0e-5                # genuine microphysics sink on the donor
        donor = -redistribution - sink
        receiver = redistribution * dp[0, 0, 0] / dp[1, 0, 0]  # mass-conserving
        raw = jnp.array(
            [float(donor), float(receiver)], dtype=jnp.float64,
        ).reshape(2, 1, 1)
        state = PhysicsState.zeros((2, 1, 1), tracers={"qc": qc})
        tend = PhysicsTendency.zeros((2, 1, 1), tracers={"qc": raw})
        return state, tend, dp, dt

    def test_bare_cap_reproduces_the_mass_creation(self):
        # Reproduce the defect: without Δp the fallback cap creates column
        # water in the overdrawn donor layer.
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        state, tend, dp, dt = self._scenario()
        applied, _ = _verify_tendencies_with_water_corrections(state, tend, dt)
        raw_col = float(jnp.sum(tend.tracers["qc"] * dp))
        bare_col = float(jnp.sum(applied.tracers["qc"] * dp))
        self.assertGreater(bare_col - raw_col, 1.0e-4)

    def test_column_water_conserved_with_thickness(self):
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        state, tend, dp, dt = self._scenario()
        applied, corrections = _verify_tendencies_with_water_corrections(
            state, tend, dt, pressure_thickness=dp,
        )
        raw_col = float(jnp.sum(tend.tracers["qc"] * dp))
        conserved_col = float(jnp.sum(applied.tracers["qc"] * dp))
        # Column water tendency preserved to f64.
        self.assertAlmostEqual(conserved_col, raw_col, places=12)
        # Non-negativity preserved (the load-bearing property the cap exists
        # for): every layer's next-step value is >= 0.
        qnext = state.tracers["qc"] + dt * applied.tracers["qc"]
        self.assertTrue(bool(jnp.all(qnext >= -1.0e-18)))
        # The net (post-reallocation) ledger source integrates to ~0: the cap's
        # source was reallocated, not invented.
        net_source = float(jnp.sum(corrections["qc"] * dp))
        self.assertAlmostEqual(net_source, 0.0, places=12)

    def test_specific_humidity_is_conserved_too(self):
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        dt = 100.0
        dp = jnp.array([10_000.0, 30_000.0], dtype=jnp.float64).reshape(2, 1, 1)
        q = jnp.array([2.0e-3, 8.0e-3], dtype=jnp.float64).reshape(2, 1, 1)
        raw = jnp.array([-1.0e-4, 2.0e-5], dtype=jnp.float64).reshape(2, 1, 1)
        state = PhysicsState.zeros((2, 1, 1), specific_humidity=q)
        tend = PhysicsTendency.zeros((2, 1, 1), specific_humidity=raw)
        applied, _ = _verify_tendencies_with_water_corrections(
            state, tend, dt, pressure_thickness=dp,
        )
        raw_col = float(jnp.sum(tend.specific_humidity * dp))
        conserved_col = float(jnp.sum(applied.specific_humidity * dp))
        self.assertAlmostEqual(conserved_col, raw_col, places=12)
        qnext = state.specific_humidity + dt * applied.specific_humidity
        self.assertTrue(bool(jnp.all(qnext >= -1.0e-18)))

    def test_pure_sink_matches_bare_cap(self):
        # A genuine column-emptying sink leaves no water to borrow, so the
        # conservative path must equal the bare cap (and the ledger keeps the
        # full correction #824 recorded for the truly-unrecoverable case).
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        dt = 100.0
        dp = jnp.array([10_000.0, 30_000.0], dtype=jnp.float64).reshape(2, 1, 1)
        qc = jnp.array([1.0e-3, 2.0e-3], dtype=jnp.float64).reshape(2, 1, 1)
        raw = jnp.array([-1.0, -1.0], dtype=jnp.float64).reshape(2, 1, 1)
        state = PhysicsState.zeros((2, 1, 1), tracers={"qc": qc})
        tend = PhysicsTendency.zeros((2, 1, 1), tracers={"qc": raw})
        bare, _ = _verify_tendencies_with_water_corrections(state, tend, dt)
        conserved, _ = _verify_tendencies_with_water_corrections(
            state, tend, dt, pressure_thickness=dp,
        )
        np.testing.assert_allclose(
            np.asarray(conserved.tracers["qc"]),
            np.asarray(bare.tracers["qc"]),
        )

    def test_broadcasting_native_columns_agree(self):
        # Vertical on axis 0, horizontal broadcast: a single (2, 1) column and
        # a (2, ncols) block of identical columns must agree per column.
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        dt = 100.0
        dp_col = jnp.array([10_000.0, 30_000.0], dtype=jnp.float64).reshape(2, 1)
        qc_col = jnp.array([1.0e-3, 5.0e-3], dtype=jnp.float64).reshape(2, 1)
        raw_col = jnp.array([-4.0e-5, 6.6667e-6], dtype=jnp.float64).reshape(2, 1)

        col_state = PhysicsState.zeros((2, 1), tracers={"qc": qc_col})
        col_tend = PhysicsTendency.zeros((2, 1), tracers={"qc": raw_col})
        col_applied, _ = _verify_tendencies_with_water_corrections(
            col_state, col_tend, dt, pressure_thickness=dp_col,
        )

        ncols = 4
        tile = lambda x: jnp.broadcast_to(x, (2, ncols))
        blk_state = PhysicsState.zeros((2, ncols), tracers={"qc": tile(qc_col)})
        blk_tend = PhysicsTendency.zeros(
            (2, ncols), tracers={"qc": tile(raw_col)},
        )
        blk_applied, _ = _verify_tendencies_with_water_corrections(
            blk_state, blk_tend, dt, pressure_thickness=tile(dp_col),
        )
        for c in range(ncols):
            np.testing.assert_allclose(
                np.asarray(blk_applied.tracers["qc"][:, c]),
                np.asarray(col_applied.tracers["qc"][:, 0]),
            )

    def test_gradient_is_identity_and_poison_free(self):
        # The whole positivity+conservation projection is a primal-only
        # correction with a straight-through (identity) gradient to the
        # producing tendency; the reallocation's division sits under
        # stop_gradient, so a dry column (0/0 guard) cannot poison the graph
        # (#558/#559).
        import jax
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        from jcm.testing import check_gradients
        state, tend, dp, dt = self._scenario()

        def applied_qc(raw_qc):
            t = PhysicsTendency.zeros((2, 1, 1), tracers={"qc": raw_qc})
            out, _ = _verify_tendencies_with_water_corrections(
                state, t, dt, pressure_thickness=dp,
            )
            return out.tracers["qc"]

        jac = jax.jacobian(lambda r: jnp.sum(applied_qc(r)))(
            tend.tracers["qc"]
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(jac))))
        np.testing.assert_allclose(
            np.asarray(jac), np.ones_like(np.asarray(jac)),
        )
        check_gradients((lambda r: applied_qc(r)), (tend.tracers["qc"],),
                        reference="adjoint")

        # Dry column: removable water is zero; the gradient must stay finite.
        dry = PhysicsState.zeros((2, 1, 1), tracers={"qc": jnp.zeros((2, 1, 1))})

        def dry_sum(raw_qc):
            t = PhysicsTendency.zeros((2, 1, 1), tracers={"qc": raw_qc})
            out, _ = _verify_tendencies_with_water_corrections(
                dry, t, dt, pressure_thickness=dp,
            )
            return jnp.sum(out.tracers["qc"])

        grad_dry = jax.grad(dry_sum)(
            jnp.array([-1.0, -1.0], dtype=jnp.float64).reshape(2, 1, 1)
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad_dry))))


class TestComposablePressureThickness(unittest.TestCase):
    """The Δp weights the conservative limiter consumes (#806)."""

    def test_column_sum_recovers_surface_pressure(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.constants import p0

        coords = get_speedy_coords()
        physics = speedy_physics()
        physics.cache_coords(coords)

        nlev, nlon, nlat = coords.nodal_shape
        state = PhysicsState.zeros(
            (nlev, nlon, nlat),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
        )
        dp = physics.pressure_thickness(state)
        self.assertEqual(dp.shape, (nlev, nlon, nlat))
        self.assertTrue(bool(jnp.all(dp > 0.0)))
        # Pure-sigma column: the |Δp| stack integrates to the surface pressure
        # (here p0, since normalized_surface_pressure == 1).
        np.testing.assert_allclose(
            np.asarray(jnp.sum(dp, axis=0)),
            np.full((nlon, nlat), float(p0)),
            rtol=1e-6,
        )
