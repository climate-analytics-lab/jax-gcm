"""Tests for ``jcm.prescribed_state_model.PrescribedStateModel``."""

import unittest
import warnings
from typing import ClassVar

import jax.numpy as jnp
import jax_datetime as jdt
import pytest

from jcm.constants import grav
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.held_suarez.utils import get_held_suarez_coords
from jcm.prescribed_state_model import (
    PrescribedStateModel,
    PrescribedStatePredictions,
)


class _CounterCarryTerm(PhysicsTerm):
    """Toy carry-owning term: increments its carry slot by 1 every call.

    Minimal stand-in for a real carry-owning term (TTE-TKE, JAM's cloud-
    borne store) that makes the scan-vs-vmap distinction mechanically
    checkable: after N sequential steps the carry must read exactly N, no
    matter what the (irrelevant, here) atmospheric state is. The
    ``diagnostics.get(..., <zero-shaped fallback>)`` read (rather than a
    bare subscript) matters for the SAME reason it does in
    ``TteTkeVerticalDiffusion`` (see ``vertical_diffusion.py``): the
    structural probe behind ``get_empty_data`` calls ``compute_tendencies``
    with no carry at all (``prev_physics_data=None``), so a bare
    ``diagnostics["_counter"]`` would ``KeyError`` there.
    """

    name: ClassVar[str] = "counter"
    category: ClassVar[str] = "test_carry"
    provides: ClassVar[tuple[str, ...]] = ("_counter",)

    def initial_carry_state(self, coords):
        nlon, nlat = coords.horizontal.nodal_shape
        return {"_counter": jnp.zeros((nlon, nlat))}

    def __call__(self, state, diagnostics, forcing, terrain):
        nlon, nlat = state.temperature.shape[-2:]
        prev = diagnostics.get("_counter", jnp.zeros((nlon, nlat)))
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "_counter": prev + 1.0}


class _CounterReaderTerm(PhysicsTerm):
    """Toy carry-FREE second term: doubles whatever ``_CounterCarryTerm`` wrote.

    Exists so the carry-owning composition has TWO terms (as a real
    composition always does), exercising the same-step term-ordering
    contract (this term runs after ``_CounterCarryTerm`` and sees ITS
    updated counter, not the stale one) without owning any carry itself.
    """

    name: ClassVar[str] = "counter_reader"
    category: ClassVar[str] = "test_carry"

    def __call__(self, state, diagnostics, forcing, terrain):
        nlon, nlat = state.temperature.shape[-2:]
        counter = diagnostics.get("_counter", jnp.zeros((nlon, nlat)))
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "counter_doubled": counter * 2.0}


def _counter_physics() -> ComposablePhysics:
    """Two-term toy physics with one carry slot (see the classes above)."""
    return ComposablePhysics([_CounterCarryTerm(), _CounterReaderTerm()])


def _make_test_state(coords) -> PhysicsState:
    nlev = coords.nodal_shape[0]
    nlon, nlat = coords.horizontal.nodal_shape
    shape = (nlev, nlon, nlat)
    z = jnp.linspace(0, 30000, nlev)[::-1]
    t_profile = jnp.maximum(288.0 - 6.5e-3 * z, 200.0)
    q_profile = 0.012 * jnp.exp(-z / 3000.0)
    return PhysicsState(
        u_wind=jnp.full(shape, 5.0),
        v_wind=jnp.zeros(shape),
        temperature=jnp.broadcast_to(t_profile[:, None, None], shape),
        specific_humidity=jnp.broadcast_to(q_profile[:, None, None], shape),
        geopotential=jnp.broadcast_to((grav * z)[:, None, None], shape),
        normalized_surface_pressure=jnp.ones((nlon, nlat)),
        tracers={},
    )


class TestPrescribedStateModel(unittest.TestCase):
    def setUp(self):
        self.coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        self.state = _make_test_state(self.coords)

    def test_run_smoke(self):
        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        states = [self.state] * 3
        predictions = model.run(states)
        self.assertIsInstance(predictions, PrescribedStatePredictions)
        self.assertEqual(predictions.tendencies.temperature.shape[0], 3)
        self.assertEqual(predictions.times.shape[0], 3)

    def test_run_accepts_stacked_state_and_explicit_times(self):
        # A single PhysicsState whose leading axis is time must behave the
        # same as the list-of-states form, and user-provided ``times`` must
        # be passed through untouched.
        from jax.tree_util import tree_map

        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
            dt_seconds=900.0,
        )
        stacked = tree_map(lambda a: jnp.stack([a, a], axis=0), self.state)
        times = jnp.array([0.0, 0.5])  # days
        predictions = model.run(stacked, times=times)
        self.assertEqual(predictions.tendencies.temperature.shape[0], 2)
        self.assertTrue(jnp.array_equal(
            predictions.times.delta.seconds, jnp.array([0, 43200])))
        # Identical prescribed states must yield identical tendencies —
        # there is no carry between steps in prescribed mode.
        t_tend = predictions.tendencies.temperature
        self.assertTrue(jnp.allclose(t_tend[0], t_tend[1]))

    def test_default_times_use_dt_seconds(self):
        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
            dt_seconds=43200.0,  # half a day per step
        )
        predictions = model.run([self.state] * 3)
        self.assertTrue(jnp.array_equal(
            predictions.times.delta.days, jnp.array([10957, 10957, 10958])))
        self.assertTrue(jnp.array_equal(
            predictions.times.delta.seconds, jnp.array([0, 43200, 0])))

    def test_to_xarray_layout_and_diagnostics(self):
        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        predictions = model.run([self.state] * 2)
        ds = predictions.to_xarray()

        nlev = self.coords.nodal_shape[0]
        nlon, nlat = self.coords.horizontal.nodal_shape
        # Column state variables use the prescribed-mode vmap layout.
        self.assertEqual(
            ds["state.temperature"].dims, ("time", "level", "lon", "lat")
        )
        self.assertEqual(
            ds["state.temperature"].shape, (2, nlev, nlon, nlat)
        )
        # Surface variables drop the level axis.
        self.assertEqual(
            ds["state.normalized_surface_pressure"].dims,
            ("time", "lon", "lat"),
        )
        # Tendencies serialised alongside the states.
        self.assertIn("tend.temperature", ds)
        # The prescribed states round-trip bit-exact into the dataset.
        import numpy as np
        np.testing.assert_array_equal(
            ds["state.u_wind"].values[0],
            np.asarray(self.state.u_wind),
        )
        self.assertEqual(ds.sizes["time"], 2)

    def test_to_xarray_uses_shared_exact_time_serialization(self):
        import numpy as np

        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
            dt_seconds=500,
            start_time="2000-01-01T00:00:00",
        )
        ds = model.run([self.state] * 2).to_xarray()

        assert ds.time.dtype == np.dtype("datetime64[ms]")
        np.testing.assert_array_equal(
            ds.time.values,
            np.array(
                ["2000-01-01T00:00:00.000", "2000-01-01T00:08:20.000"],
                dtype="datetime64[ms]",
            ),
        )
        assert ds.time.encoding["units"] == (
            "seconds since 1970-01-01 00:00:00"
        )
        assert ds.time.encoding["calendar"] == "proleptic_gregorian"

    def test_to_xarray_physics_data_dict_handling(self):
        # Private ("_"-prefixed) diagnostics are dropped; struct-valued
        # entries expand via asdict; plain arrays serialise directly.
        import numpy as np
        from jax.tree_util import tree_map

        nlev = self.coords.nodal_shape[0]
        base_state = _make_test_state(self.coords)
        # Give the struct a tracer so the nested-dict expansion runs too.
        base_state = PhysicsState(
            **{**base_state.asdict(),
               "tracers": {"qc": jnp.zeros_like(base_state.temperature)}},
        )
        stacked = tree_map(lambda a: jnp.stack([a, a], axis=0), base_state)
        preds = PrescribedStatePredictions(
            states=stacked,
            tendencies=stacked,
            physics_data={
                "_private": jnp.zeros((2,)),
                "scalar_series": jnp.arange(2.0),
                # Struct-valued diagnostics expand field-by-field.
                "blob": stacked,
                # (time, level) columns and rank>4 fall back gracefully.
                "column_series": jnp.zeros((2, nlev)),
                "rank5": jnp.zeros((2, 1, 1, 1, 1)),
            },
            times=jdt.to_datetime('2000-01-01') + jdt.Timedelta(
                days=jnp.array([0, 1], dtype=jnp.int32)),
        )
        ds = preds.to_xarray()
        self.assertNotIn("diag._private", ds)
        self.assertIn("diag.scalar_series", ds)
        np.testing.assert_allclose(
            ds["diag.scalar_series"].values, [0.0, 1.0]
        )
        self.assertIn("diag.blob.temperature", ds)
        self.assertEqual(
            ds["diag.blob.temperature"].dims,
            ("time", "level", "lon", "lat"),
        )
        # Nested tracer dict flattens with a dotted name.
        self.assertIn("diag.blob.tracers.qc", ds)
        self.assertEqual(
            ds["diag.column_series"].dims, ("time", "level"),
        )
        self.assertEqual(
            ds["diag.rank5"].dims,
            ("dim_0", "dim_1", "dim_2", "dim_3", "dim_4"),
        )

    def test_to_xarray_level_is_surface_first_sigma(self):
        # ``run`` threads the vertical table through, so ``to_xarray`` writes a
        # real descending sigma ``level`` coordinate with CF metadata (#739).
        import numpy as np

        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        predictions = model.run([self.state] * 2)
        ds = predictions.to_xarray()

        level = ds["level"].values
        # Surface-first file convention: sigma descends from ~1 to ~0.
        self.assertTrue(np.all(np.diff(level) < 0))
        self.assertEqual(ds["level"].attrs.get("positive"), "down")

    def test_to_xarray_round_trips_through_load_states(self):
        # A file written by ``to_xarray`` (surface-first) must come back
        # top-first via ``load_states_from_xarray`` (#739 + #741), recovering
        # the original per-level physics-frame values exactly.
        import numpy as np

        from jcm.utils import load_states_from_xarray

        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        predictions = model.run([self.state] * 2)
        ds = predictions.to_xarray()

        reloaded = load_states_from_xarray(
            ds,
            u_wind_var="state.u_wind",
            v_wind_var="state.v_wind",
            temperature_var="state.temperature",
            specific_humidity_var="state.specific_humidity",
            geopotential_var="state.geopotential",
            surface_pressure_var="state.normalized_surface_pressure",
        )
        # ``self.state`` is a single (nlev, ...) column; the predictions stack
        # it over 2 times, so compare each recovered time slice to it.
        np.testing.assert_allclose(
            np.asarray(reloaded.temperature)[0],
            np.asarray(self.state.temperature),
        )
        np.testing.assert_allclose(
            np.asarray(reloaded.temperature)[1],
            np.asarray(self.state.temperature),
        )


class TestPrescribedStateModelCarryWarning(unittest.TestCase):
    """Carry classification + the two warnings it can produce (#623).

    A "none" (no carry) or "scan" (named dict carry) composition never
    warns at CONSTRUCTION time — the one remaining construction-time
    warning is for "unthreadable" (a non-dict carry a scan cannot be
    seeded for), which still falls back to the pre-#623 vmap/cold-start
    behaviour. The "scan" case's own warning (that it cold-starts) is
    per-``run()``-call, covered by ``TestPrescribedStateModelCarryThreading``
    below — it is not raised at construction.
    """

    @staticmethod
    def _carry_warnings(caught):
        return [str(w.message) for w in caught
                if "cross-step carry" in str(w.message)]

    def _construct(self, physics, coords):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PrescribedStateModel(physics=physics, coords=coords)
        return model, self._carry_warnings(caught)

    def test_no_carry_slots_no_warning(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics

        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        for physics in (held_suarez_physics(), speedy_physics()):
            with self.subTest(physics=type(physics).__name__):
                model, messages = self._construct(physics, coords)
                self.assertEqual(messages, [])
                self.assertEqual(model._carry_mode, "none")

    def test_a_raw_physics_carry_of_any_pytree_is_handled(self):
        """``initial_carry_state`` may return any pytree, or ``None``.

        A raw ``Physics`` whose carry is an array has cross-step state with
        no slot names: it cannot be threaded (no keys to union against a
        structural template), so it keeps the pre-#623 behaviour and warns
        at construction. One whose carry is ``None`` or empty has none and
        stays silent.
        """
        from jcm.physics_interface import Physics

        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)

        class _Raw(Physics):
            def __init__(self, carry):
                self._carry = carry

            def initial_carry_state(self, coords):
                return self._carry

        for carry, expected_mode, expected_n in (
            (None, "none", 0), ({}, "none", 0), ((), "none", 0),
            (jnp.zeros((8, 2048)), "unthreadable", 1),
        ):
            with self.subTest(carry=type(carry).__name__):
                model, messages = self._construct(_Raw(carry), coords)
                self.assertEqual(model._carry_mode, expected_mode)
                self.assertEqual(len(messages), expected_n)
                if expected_n:
                    self.assertIn("unnamed", messages[0])

    def test_named_carry_classified_as_scan_with_no_construction_warning(self):
        """A dict carry (TTE-TKE + JAM's cloud-borne store) is now threaded.

        Construction classifies it "scan" and stays silent — #969's
        construction-time warning for this case is gone; the live
        cold-start warning moved to ``run()`` (see
        ``TestPrescribedStateModelCarryThreading``).
        """
        from jcm.physics.aerosol.jam.cloud_borne_store import (
            CARRY_KEY, CloudBorneCarryStore)
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        physics = idealized_echam_physics(checkpoint_terms=False)
        physics = physics + CloudBorneCarryStore()
        model, messages = self._construct(physics, coords)
        self.assertEqual(messages, [])
        self.assertEqual(model._carry_mode, "scan")
        for slot in ("vertical_diffusion", CARRY_KEY):
            self.assertIn(slot, model._carry_slot_names)
        self.assertIn(CARRY_KEY, model._prognostic_carry_slots)


class TestPrescribedStateModelCarryThreading(unittest.TestCase):
    """``run()`` threads a "scan"-mode carry sequentially across times (#623)."""

    def test_toy_carry_advances_by_one_per_time_and_vmap_is_unused(self):
        """Mechanical check: N sequential steps must read counter 1..N.

        If ``run`` were still vmapping independently, every time would see
        the SAME cold carry (0) and report counter == 1 at every time —
        the pre-#623 behaviour this PR replaces for "scan"-mode physics.
        """
        physics = _counter_physics()
        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        state = _make_test_state(coords)
        n_times = 5

        model = PrescribedStateModel(physics=physics, coords=coords)
        self.assertEqual(model._carry_mode, "scan")
        self.assertEqual(model._carry_slot_names, ("_counter",))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictions = model.run([state] * n_times)
        spin_up = [str(w.message) for w in caught if "spin-up" in str(w.message)]
        self.assertEqual(len(spin_up), 1)  # cold-started: no donor passed.

        counter = predictions.physics_data["_counter"]
        self.assertEqual(counter.shape[0], n_times)
        expected = jnp.arange(1.0, n_times + 1.0)
        np_counter_by_time = jnp.asarray(
            [counter[i, 0, 0] for i in range(n_times)])
        self.assertTrue(jnp.array_equal(np_counter_by_time, expected))
        # Every column advances identically (the toy carry is spatially
        # uniform), so the whole-field values match too.
        self.assertTrue(jnp.allclose(
            counter, expected[:, None, None] * jnp.ones_like(counter[0])))
        # The second term reads the FIRST term's updated value within the
        # SAME step (same-step term ordering), so it is always 2x the
        # counter reported for that step, not 2x the previous step's.
        self.assertTrue(jnp.allclose(
            predictions.physics_data["counter_doubled"], 2.0 * counter))

    def test_donor_carry_skips_spin_up_and_seeds_the_scan(self):
        """A supplied ``initial_physics_state`` seeds the scan; no warning."""
        from jcm.physics_interface import build_initial_physics_carry

        physics = _counter_physics()
        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        state = _make_test_state(coords)
        nlon, nlat = coords.horizontal.nodal_shape
        # A real donor (e.g. ``jcm.initial_states.checkpoint_state``'s
        # ``physics_carry``) is the FULL carry ``build_initial_physics_carry``
        # builds, not just the named slot — the scan's per-step diagnostic
        # keys (``_prev_step``, ...) ride along too. Override only the
        # counter slot to make the seed recognisable.
        donor = {**build_initial_physics_carry(physics, coords),
                 "_counter": jnp.full((nlon, nlat), 100.0)}

        model = PrescribedStateModel(physics=physics, coords=coords)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictions = model.run(
                [state] * 3, initial_physics_state=donor)
        spin_up = [str(w.message) for w in caught if "spin-up" in str(w.message)]
        self.assertEqual(spin_up, [])

        counter = predictions.physics_data["_counter"]
        # Seeded at 100 rather than 0: times read 101, 102, 103.
        expected = jnp.array([101.0, 102.0, 103.0])
        self.assertTrue(jnp.allclose(counter[:, 0, 0], expected))

    def test_out_of_order_times_are_refused_in_scan_mode(self):
        """Threading a carry between unrelated (non-chronological) states
        would be physically meaningless, so a decreasing time axis is
        refused rather than silently mis-threaded.
        """
        physics = _counter_physics()
        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        state = _make_test_state(coords)
        model = PrescribedStateModel(physics=physics, coords=coords)

        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            model.run([state] * 3, times=jnp.array([0.0, 1.0, 0.5]))

    @pytest.mark.slow
    def test_tte_tke_threads_a_real_carry_through_the_scan(self):
        """A real carry-owning term must show the scan actually threaded.

        Two scenarios on the SAME 3 identical prescribed states:

        * Cold-started (no donor): warns (naming the slots, including the
          prognostic cloud-borne one). ``TteTkeVerticalDiffusion`` seeds TKE
          at 0 (ECHAM's own cold-start value, ``vdiff.f90``'s ``lstart``
          branch — see its ``initial_carry_state`` docstring), so time 0's
          TKE is a genuine first-step production/dissipation balance built
          from nothing. The load-bearing assertion: time 1's TKE differs
          substantially from time 0's — under the pre-#623 vmap path every
          time would independently re-diagnose from the SAME zero seed and
          time 0 could never influence time 1 at all, so this would be
          impossible; threading it sequentially makes time 1 see time 0's
          OWN built-up TKE as its previous-step value instead of zero.
        * Donor-seeded well above anything the cold start reaches: no
          spin-up warning, and the same threading signature (time 1 differs
          from time 0) reproduces starting from a different, non-zero seed.
        """
        from jcm.model import Model
        from jcm.physics.aerosol.jam.cloud_borne_store import CloudBorneCarryStore
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.physics_interface import build_initial_physics_carry
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        physics = idealized_echam_physics(checkpoint_terms=False)
        physics = physics + CloudBorneCarryStore()

        # A physically valid column comes from Model's own initial-state
        # construction (the Jablonowski-Williamson profile) rather than a
        # hand-rolled one — ``model`` itself is discarded, used only to
        # build one realistic ``PhysicsState``.
        model = Model(coords=coords, physics=physics, time_step=900.0)
        base_state = model.dycore.to_physics_state(
            model._prepare_initial_dycore_state())

        prescribed = PrescribedStateModel(
            physics=physics, coords=coords, dt_seconds=900.0)
        self.assertEqual(prescribed._carry_mode, "scan")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cold_predictions = prescribed.run([base_state] * 3)
        spin_up = [str(w.message) for w in caught if "spin-up" in str(w.message)]
        self.assertEqual(len(spin_up), 1)
        self.assertIn("_jam_cloud_borne", spin_up[0])
        cold_tke = cold_predictions.physics_data["vertical_diffusion"].tke
        self.assertFalse(jnp.allclose(cold_tke[0], cold_tke[1]))

        elevated_seed = 1.0
        donor = build_initial_physics_carry(physics, coords)
        vdiff_donor = donor["vertical_diffusion"]
        donor = {**donor, "vertical_diffusion": vdiff_donor.copy(
            tke=jnp.full_like(vdiff_donor.tke, elevated_seed))}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warm_predictions = prescribed.run(
                [base_state] * 3, initial_physics_state=donor)
        self.assertEqual(
            [str(w.message) for w in caught if "spin-up" in str(w.message)], [])
        warm_tke = warm_predictions.physics_data["vertical_diffusion"].tke
        self.assertFalse(jnp.allclose(warm_tke[0], warm_tke[1]))


class TestPrescribedStateModelVmapBitIdentical(unittest.TestCase):
    """A carry-free composition still takes the exact pre-#623 vmap path."""

    def test_vmap_output_matches_direct_per_state_call(self):
        """``run`` must agree with calling ``compute_tendencies`` directly.

        Every state is evaluated at the SAME time (``times`` all zero) so
        the comparison needs no independent reimplementation of ``run``'s
        per-index time/forcing arithmetic — only the one ``DateData`` that
        offset zero always produces. Two DIFFERENT states (not three
        copies of one) keep the comparison from being a trivial no-op.
        """
        from jcm.date import DateData
        from jcm.forcing import default_forcing
        from jcm.physics_interface import verify_state

        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        physics = held_suarez_physics()
        physics.cache_coords(coords)
        base_state = _make_test_state(coords)
        perturbed = base_state.copy(temperature=base_state.temperature + 3.0)
        states = [base_state, perturbed, base_state]

        model = PrescribedStateModel(physics=physics, coords=coords)
        self.assertEqual(model._carry_mode, "none")
        predictions = model.run(states, times=jnp.zeros(3))

        forcing_now = default_forcing(coords.horizontal).select(
            DateData.set_date(
                model_time=model.start_time, model_step=jnp.int32(0),
                dt_seconds=model.dt_seconds))
        expected = [
            physics.compute_tendencies(
                verify_state(s), forcing_now, model.terrain)[0]
            for s in states
        ]
        expected_temperature = jnp.stack(
            [e.temperature for e in expected], axis=0)
        # ``vmap`` vs a plain per-state Python loop reorders XLA's fused
        # reductions, so compare to float roundoff rather than bit-exact.
        self.assertTrue(jnp.allclose(
            predictions.tendencies.temperature, expected_temperature,
            atol=1e-9, rtol=1e-9))


# Slow-marked companion — see jcm/runners_test.py for rationale.

@pytest.mark.slow
class TestPrescribedStateModelSlow(TestPrescribedStateModel):
    pass
