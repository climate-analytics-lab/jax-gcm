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

    def test_water_tracer_cap_with_no_vapour_matches_bare_cap(self):
        """A dry cell's condensate cap is the bare drain-to-zero cap.

        With no vapour to charge (``specific_humidity`` is zero here), the
        ECHAM 8.4 vapour charge is inactive and the cap's correction stays in
        the ledger; the qc tendency itself is floored at exactly the drain
        rate. The moist behaviour — the local vapour charge with latent heat —
        is covered by ``TestWaterPositivityVapourCharge`` (#806).
        """
        from jcm.physics_interface import PhysicsTendency, verify_tendencies
        shape = (4, 8, 8)
        state = PhysicsState.zeros(shape, tracers={"qc": jnp.full(shape, 1e-5)})
        tend = PhysicsTendency.zeros(shape, tracers={"qc": jnp.full(shape, -1.0)})
        result = verify_tendencies(state, tend, time_step=1800.0)
        nxt = state.tracers["qc"] + 1800.0 * result.tracers["qc"]
        self.assertTrue(bool(jnp.all(nxt >= 0.0)))
        self.assertTrue(bool(jnp.allclose(nxt, 0.0)))
        # No vapour available: q and T pass through untouched.
        self.assertTrue(bool(jnp.all(result.specific_humidity == 0.0)))
        self.assertTrue(bool(jnp.all(result.temperature == 0.0)))

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



class TestWaterPositivityVapourCharge(unittest.TestCase):
    """#806: condensate positivity corrections are charged to local vapour.

    The reference pattern is ECHAM mo_cloud.f90 section 8.4 ("Corrections:
    Avoid negative cloud water/ice"): the condensate a positivity cap ADDS is
    materialised as condensation/deposition of the same cell's vapour
    (``pqte -= zdxlcor + zdxicor``) with the matching latent heat on
    temperature. Total water per cell is conserved exactly wherever the
    vapour can supply the correction; the capped field itself and every other
    layer keep the bare-cap (sequential-reference) values — in particular a
    conservative redistribution's receivers keep the water that genuinely
    left the donor (the seeded-cloud CRE regression on #864 pinned this).
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
        """Two-level column with an overdrawn donor and moist cells.

        vdiff exports qc donor->receiver (mass-conserving against the layer
        thicknesses used for the budget check) and a co-located sink overdraws
        the donor, so the summed donor qc tendency drives qc0 < 0. Both cells
        carry ample vapour, so the donor's correction can be charged locally.
        """
        dt = 100.0
        dp = jnp.array([10_000.0, 30_000.0], dtype=jnp.float64).reshape(2, 1, 1)
        qc = jnp.array([1.0e-3, 5.0e-3], dtype=jnp.float64).reshape(2, 1, 1)
        q = jnp.array([5.0e-3, 5.0e-3], dtype=jnp.float64).reshape(2, 1, 1)
        redistribution = 2.0e-5      # conservative donor export rate
        sink = 0.5e-5                # genuine microphysics sink on the donor
        donor = -redistribution - sink
        receiver = redistribution * dp[0, 0, 0] / dp[1, 0, 0]  # mass-conserving
        raw_qc = jnp.array(
            [float(donor), float(receiver)], dtype=jnp.float64,
        ).reshape(2, 1, 1)
        state = PhysicsState.zeros(
            (2, 1, 1), specific_humidity=q, tracers={"qc": qc},
        )
        tend = PhysicsTendency.zeros(
            (2, 1, 1), specific_humidity=jnp.zeros((2, 1, 1)),
            tracers={"qc": raw_qc},
        )
        return state, tend, dp, dt

    def test_condensate_cap_charged_to_local_vapour_with_latent_heat(self):
        from jcm import constants as c
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        state, tend, dp, dt = self._scenario()
        applied, corrections = _verify_tendencies_with_water_corrections(
            state, tend, dt,
        )
        qc0 = float(state.tracers["qc"][0, 0, 0])
        raw0 = float(tend.tracers["qc"][0, 0, 0])
        correction = -qc0 / dt - raw0            # what the cap added, > 0
        # The capped field gets exactly the bare cap (drain to zero).
        self.assertAlmostEqual(
            float(applied.tracers["qc"][0, 0, 0]), -qc0 / dt, places=18,
        )
        # The receiver keeps its raw gain untouched (sequential reference).
        self.assertEqual(
            float(applied.tracers["qc"][1, 0, 0]),
            float(tend.tracers["qc"][1, 0, 0]),
        )
        # The correction is charged to the donor cell's vapour...
        self.assertAlmostEqual(
            float(applied.specific_humidity[0, 0, 0]), -correction, places=18,
        )
        # ... with condensation latent heat on temperature (ECHAM zlvdcp).
        self.assertAlmostEqual(
            float(applied.temperature[0, 0, 0]),
            c.alhc * correction / c.cpd, places=12,
        )
        # And the net ledger closes per cell: q correction == -qc correction.
        np.testing.assert_allclose(
            np.asarray(corrections["specific_humidity"]),
            -np.asarray(corrections["qc"]),
            rtol=0.0, atol=1e-24,
        )

    def test_column_total_water_conserved(self):
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        state, tend, dp, dt = self._scenario()
        applied, _ = _verify_tendencies_with_water_corrections(state, tend, dt)
        total = lambda t: float(jnp.sum(
            (t.specific_humidity + t.tracers["qc"]) * dp,
        ))
        # Column-integrated total water tendency preserved to f64 round-off.
        self.assertAlmostEqual(total(applied), total(tend), places=12)
        # Positivity preserved on both fields.
        qc_next = state.tracers["qc"] + dt * applied.tracers["qc"]
        q_next = state.specific_humidity + dt * applied.specific_humidity
        self.assertTrue(bool(jnp.all(qc_next >= -1e-18)))
        self.assertTrue(bool(jnp.all(q_next >= -1e-18)))

    def test_sink_overdraw_does_not_touch_other_layers(self):
        # The seeded-cloud CRE regression (#864 CI): the first version of this
        # fix reallocated the correction across the COLUMN, draining untouched
        # cloud layers (and, in the gains-bounded variant, the receivers'
        # legitimate gains) to zero in one step. The vapour charge is
        # cell-local, so other layers' tendencies pass through bitwise.
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        dt = 100.0
        qc = jnp.array([1.0e-5, 5.0e-3], dtype=jnp.float64).reshape(2, 1, 1)
        q = jnp.array([2.0e-3, 2.0e-3], dtype=jnp.float64).reshape(2, 1, 1)
        # Layer 0: violent sink overdraw. Layer 1: physics leaves it alone.
        raw = jnp.array([-1.0, 0.0], dtype=jnp.float64).reshape(2, 1, 1)
        state = PhysicsState.zeros(
            (2, 1, 1), specific_humidity=q, tracers={"qc": qc},
        )
        tend = PhysicsTendency.zeros((2, 1, 1), tracers={"qc": raw})
        applied, _ = _verify_tendencies_with_water_corrections(state, tend, dt)
        # The untouched cloud layer keeps a zero qc tendency and an untouched
        # vapour tendency.
        self.assertEqual(float(applied.tracers["qc"][1, 0, 0]), 0.0)
        self.assertEqual(float(applied.specific_humidity[1, 0, 0]), 0.0)
        # The overdrawn layer gets the bare cap; its vapour absorbs what it
        # can (bounded by q/dt) and the rest stays ledgered.
        self.assertAlmostEqual(
            float(applied.tracers["qc"][0, 0, 0]), -1.0e-5 / dt, places=18,
        )
        self.assertAlmostEqual(
            float(applied.specific_humidity[0, 0, 0]), -2.0e-3 / dt, places=15,
        )

    def test_dry_cell_leaves_residual_ledgered(self):
        # No vapour to charge: the correction stays in the ledger as an
        # honest artificial source; q and T pass through unchanged.
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        dt = 100.0
        qc = jnp.full((2, 1, 1), 1.0e-5, dtype=jnp.float64)
        raw = jnp.full((2, 1, 1), -1.0, dtype=jnp.float64)
        state = PhysicsState.zeros((2, 1, 1), tracers={"qc": qc})
        tend = PhysicsTendency.zeros((2, 1, 1), tracers={"qc": raw})
        applied, corrections = _verify_tendencies_with_water_corrections(
            state, tend, dt,
        )
        self.assertTrue(bool(jnp.all(applied.specific_humidity == 0.0)))
        self.assertTrue(bool(jnp.all(applied.temperature == 0.0)))
        self.assertGreater(float(jnp.min(corrections["qc"])), 0.9)

    def test_ice_correction_heats_with_sublimation(self):
        from jcm import constants as c
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        dt = 100.0
        shape = (1, 1, 1)
        qi = jnp.full(shape, 1.0e-4, dtype=jnp.float64)
        q = jnp.full(shape, 5.0e-3, dtype=jnp.float64)
        raw = jnp.full(shape, -2.0e-6, dtype=jnp.float64)   # overdraw: qi/dt=1e-6
        state = PhysicsState.zeros(shape, specific_humidity=q, tracers={"qi": qi})
        tend = PhysicsTendency.zeros(shape, tracers={"qi": raw})
        applied, _ = _verify_tendencies_with_water_corrections(state, tend, dt)
        correction = 1.0e-6                                  # -qi/dt - raw
        self.assertAlmostEqual(
            float(applied.temperature[0, 0, 0]),
            c.alhs * correction / c.cpd, places=12,
        )
        self.assertAlmostEqual(
            float(applied.specific_humidity[0, 0, 0]), -correction, places=18,
        )

    def test_charge_never_exceeds_the_drain_rate(self):
        # Codex P1 (#864): the vapour tendency after the charge must never
        # fall below the exact drain rate -max(q,0)/dt, even in float32 where
        # the separate roundings of ``capped - charge`` could land an ulp low.
        # Randomized f32 cells, including corrections larger than the vapour.
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        import jax
        jax.config.update("jax_enable_x64", False)
        rng = np.random.default_rng(806)
        dt = 657.57404
        nlev, ncols = 8, 32
        q = rng.uniform(0.0, 1e-2, (nlev, ncols)).astype(np.float32)
        qc = rng.uniform(0.0, 1e-3, (nlev, ncols)).astype(np.float32)
        raw_q = rng.uniform(-3e-5, 1e-6, (nlev, ncols)).astype(np.float32)
        raw_qc = rng.uniform(-3e-5, 1e-6, (nlev, ncols)).astype(np.float32)
        state = PhysicsState.zeros(
            (nlev, ncols), specific_humidity=jnp.asarray(q),
            tracers={"qc": jnp.asarray(qc)},
        )
        tend = PhysicsTendency.zeros(
            (nlev, ncols), specific_humidity=jnp.asarray(raw_q),
            tracers={"qc": jnp.asarray(raw_qc)},
        )
        applied, _ = _verify_tendencies_with_water_corrections(state, tend, dt)
        out_q = np.asarray(applied.specific_humidity)
        out_qc = np.asarray(applied.tracers["qc"])
        self.assertTrue(bool(np.all(out_q >= -np.maximum(q, 0.0) / np.float32(dt))))
        self.assertTrue(bool(np.all(out_qc >= -np.maximum(qc, 0.0) / np.float32(dt))))
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(applied.temperature)))))

    def test_broadcasting_native_columns_agree(self):
        # Vertical on axis 0, horizontal broadcast: a single (2, 1) column and
        # a (2, ncols) block of identical columns must agree per column.
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        dt = 100.0
        qc_col = jnp.array([1.0e-3, 5.0e-3], dtype=jnp.float64).reshape(2, 1)
        q_col = jnp.array([5.0e-3, 5.0e-3], dtype=jnp.float64).reshape(2, 1)
        raw_col = jnp.array([-4.0e-5, 6.6667e-6], dtype=jnp.float64).reshape(2, 1)

        col_state = PhysicsState.zeros(
            (2, 1), specific_humidity=q_col, tracers={"qc": qc_col},
        )
        col_tend = PhysicsTendency.zeros((2, 1), tracers={"qc": raw_col})
        col_applied, _ = _verify_tendencies_with_water_corrections(
            col_state, col_tend, dt,
        )

        ncols = 4
        tile = lambda x: jnp.broadcast_to(x, (2, ncols))
        blk_state = PhysicsState.zeros(
            (2, ncols), specific_humidity=tile(q_col),
            tracers={"qc": tile(qc_col)},
        )
        blk_tend = PhysicsTendency.zeros(
            (2, ncols), tracers={"qc": tile(raw_col)},
        )
        blk_applied, _ = _verify_tendencies_with_water_corrections(
            blk_state, blk_tend, dt,
        )
        for c in range(ncols):
            np.testing.assert_allclose(
                np.asarray(blk_applied.tracers["qc"][:, c]),
                np.asarray(col_applied.tracers["qc"][:, 0]),
            )
            np.testing.assert_allclose(
                np.asarray(blk_applied.specific_humidity[:, c]),
                np.asarray(col_applied.specific_humidity[:, 0]),
            )

    def test_gradient_is_identity_and_poison_free(self):
        # The whole positivity+charge projection is a primal-only correction
        # with a straight-through (identity) gradient to the producing
        # tendencies; the charge's division sits under stop_gradient, so a
        # dry cell (0/0 guard) cannot poison the graph (#558/#559).
        import jax
        from jcm.physics_interface import (
            _verify_tendencies_with_water_corrections,
        )
        from jcm.testing import check_gradients
        state, tend, dp, dt = self._scenario()

        def apply_fields(raw_qc, raw_q, raw_t):
            t = PhysicsTendency.zeros(
                (2, 1, 1), specific_humidity=raw_q, temperature=raw_t,
                tracers={"qc": raw_qc},
            )
            out, _ = _verify_tendencies_with_water_corrections(state, t, dt)
            return out.tracers["qc"], out.specific_humidity, out.temperature

        args = (
            tend.tracers["qc"], tend.specific_humidity, tend.temperature,
        )
        jac = jax.jacobian(
            lambda a, b, c: sum(jnp.sum(x) for x in apply_fields(a, b, c)),
            argnums=(0, 1, 2),
        )(*args)
        for j in jac:
            self.assertTrue(bool(jnp.all(jnp.isfinite(j))))
            np.testing.assert_allclose(np.asarray(j), np.ones_like(np.asarray(j)))
        check_gradients(apply_fields, args, reference="adjoint")

        # Dry cell (charge guard branch): the gradient must stay finite.
        dry = PhysicsState.zeros((2, 1, 1), tracers={"qc": jnp.zeros((2, 1, 1))})

        def dry_sum(raw_qc):
            t = PhysicsTendency.zeros((2, 1, 1), tracers={"qc": raw_qc})
            out, _ = _verify_tendencies_with_water_corrections(dry, t, dt)
            return jnp.sum(out.tracers["qc"]) + jnp.sum(out.specific_humidity)

        grad_dry = jax.grad(dry_sum)(
            jnp.array([-1.0, -1.0], dtype=jnp.float64).reshape(2, 1, 1)
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad_dry))))
