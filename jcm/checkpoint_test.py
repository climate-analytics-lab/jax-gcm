"""Tests for ``jcm/checkpoint.py``.

Covers four contracts:

1. ``save_checkpoint`` → fresh ``Model`` → ``bootstrap_state`` →
   ``load_checkpoint`` reproduces the original state pytrees
   element-wise (round-trip fidelity).
2. A continuous N-day integration matches a ``(N/2 days, checkpoint,
   load on a fresh Model, N/2 days)`` split to numerical roundoff
   (resumption equivalence — the real use case from issue #128).
3. The schema stamp and its metadata (#731), and the forward migration it
   enables: a physics-carry field the file lacks is seeded from the fresh
   carry, one the model no longer carries is dropped, and anything else —
   a shape, a dtype, a dycore-state field — is still refused.
4. An unstamped (pre-policy) file is refused unless the caller asserts
   its unit convention, because PR #824 changed what the dycore state's
   mass mixing ratios mean.

Uses Held-Suarez physics for speed: no moisture, no radiation, deterministic
forcing. The composition-coverage tests build one SPEEDY and one ECHAM model
(bootstrap only, no integration) to exercise a moist tracer set.
"""

import tempfile
import unittest
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tree_math

from jcm.checkpoint import (
    SCHEMA_VERSION, _named_leaves, load_checkpoint, parse_unstamped_scale,
    save_checkpoint,
)
from jcm.model import Model
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.held_suarez.utils import get_held_suarez_coords
from jcm.physics_interface import PhysicsTendency
from jcm.physics.physics_term import PhysicsTerm
from jcm.terrain import TerrainData


def _build_model(physics=None, spectral_truncation=31) -> Model:
    coords = get_held_suarez_coords(spectral_truncation=spectral_truncation)
    return Model(
        coords=coords,
        terrain=TerrainData.from_coords(coords),
        time_step=180,
        physics=physics if physics is not None else held_suarez_physics(),
    )


@tree_math.struct
class _ExtraCarryData:
    """A one-field diagnostic carry struct, standing in for a new field."""

    counter: jnp.ndarray

    @classmethod
    def zeros(cls, col_shape, nlev):
        return cls(counter=jnp.zeros((nlev,) + tuple(col_shape)))


class _ExtraCarryTerm(PhysicsTerm):
    """Carries one extra diagnostic slot and no physics.

    Stands in for the upgrade this migration exists for: a jcm version
    whose diagnostic structs have one more field than the version that
    wrote the checkpoint (issue #731).
    """

    name = "extra_carry"
    category = "test_extra_carry"
    carry_slots = {"extra_carry": _ExtraCarryData}

    def __call__(self, state, diagnostics, forcing, terrain):
        """No tendency, no diagnostic writes — the slot is the point."""
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


class _PrognosticCarryTerm(_ExtraCarryTerm):
    """An extra slot that is the only copy of what it holds.

    The real case is JAM's cloud-borne aerosol phase, which lives in the
    carry and nowhere else (#602); this stands in for it so the guard can
    be tested without composing JAM.
    """

    name = "prognostic_carry"
    category = "test_prognostic_carry"
    carry_slots = {"reservoir": _ExtraCarryData}
    prognostic_carry_slots = ("reservoir",)


def _read_payload(path) -> dict:
    return flax.serialization.msgpack_restore(Path(path).read_bytes())


def _write_payload(path, payload) -> None:
    # ``to_bytes`` (not ``msgpack_serialize``) so a list is stored the way
    # the writer stores one: a dict keyed by the stringified index.
    Path(path).write_bytes(flax.serialization.to_bytes(payload))


def _write_unstamped(model, path, *, elapsed_days: float) -> None:
    """Write the pre-#731 payload: two positional lists, no stamp."""
    _write_payload(path, {
        "elapsed_days": float(elapsed_days),
        "dycore_leaves": [np.asarray(x)
                          for x in jax.tree_util.tree_leaves(model.dycore_state)],
        "physics_leaves": [np.asarray(x)
                           for x in jax.tree_util.tree_leaves(model.physics_carry)],
    })


def _max_abs_diff(tree_a, tree_b) -> float:
    def leaf_diff(a, b):
        a, b = jnp.asarray(a), jnp.asarray(b)
        # An ECHAM carry holds empty-shaped slots (e.g. a band axis a grey
        # scheme leaves at zero length); ``max`` has no identity on those.
        if a.size == 0:
            return jnp.zeros((), dtype=jnp.float32)
        if a.dtype == jnp.bool_ or b.dtype == jnp.bool_:
            # SPEEDY's carry holds boolean flags; "differs" is all that
            # means for them.
            return jnp.max((a != b).astype(jnp.float32))
        return jnp.max(jnp.abs(a - b))

    diffs = jax.tree.leaves(jax.tree.map(leaf_diff, tree_a, tree_b))
    return float(max(float(d) for d in diffs)) if diffs else 0.0


class TestCheckpointRoundTrip(unittest.TestCase):

    def test_save_load_reproduces_state(self):
        model = _build_model()
        model.run(save_interval=1, total_time=2)
        dycore_before = model.dycore_state
        physics_before = model.physics_carry

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(model, path, elapsed_days=2.0)
            self.assertTrue(path.exists())

            fresh = _build_model()
            template_state, template_carry = fresh.bootstrap_state()
            self.assertIs(template_state, fresh.dycore_state)
            self.assertIs(template_carry, fresh.physics_carry)
            elapsed = load_checkpoint(fresh, path)

        self.assertAlmostEqual(elapsed, 2.0)
        self.assertEqual(_max_abs_diff(dycore_before, fresh.dycore_state), 0.0)
        self.assertEqual(_max_abs_diff(physics_before, fresh.physics_carry), 0.0)

    def test_save_without_state_raises(self):
        model = _build_model()  # never run / bootstrapped
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_checkpoint(model, Path(tmp) / "x.msgpack", elapsed_days=0.0)

    def test_load_without_template_raises(self):
        # First produce a checkpoint to load.
        donor = _build_model()
        donor.run(save_interval=1, total_time=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=1.0)

            target = _build_model()  # not bootstrapped — no templates
            with self.assertRaises(ValueError):
                load_checkpoint(target, path)


class TestCheckpointResumptionEquivalence(unittest.TestCase):
    """A split (run → ckpt → fresh model → load → resume) run matches a continuous run."""

    def test_split_resume_matches_continuous(self):
        # Baseline: continuous 4-day integration.
        baseline = _build_model()
        baseline.run(save_interval=1, total_time=4)
        baseline_dycore = baseline.dycore_state
        baseline_physics = baseline.physics_carry

        # Split: 2 days → checkpoint → new model → load → resume 2 days.
        first_half = _build_model()
        first_half.run(save_interval=1, total_time=2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(first_half, path, elapsed_days=2.0)

            second_half = _build_model()
            second_half.bootstrap_state()
            elapsed = load_checkpoint(second_half, path)
            self.assertAlmostEqual(elapsed, 2.0)

            second_half.resume(save_interval=1, total_time=2)

        modal_diff = _max_abs_diff(baseline_dycore, second_half.dycore_state)
        physics_diff = _max_abs_diff(baseline_physics, second_half.physics_carry)

        # Equivalence is exact in the absence of host-side RNG: Held-
        # Suarez and the dynamical core are deterministic given the same
        # state and forcing. Allow only float32 accumulation noise.
        self.assertLess(modal_diff, 1e-5, f"modal state diverged by {modal_diff}")
        self.assertLess(physics_diff, 1e-5, f"physics state diverged by {physics_diff}")


class TestCheckpointSchemaStamp(unittest.TestCase):
    """What ``save_checkpoint`` records so a later jcm can migrate (#731)."""

    @classmethod
    def setUpClass(cls):
        cls.model = _build_model()
        cls.model.bootstrap_state()

    def test_stamp_and_migration_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(self.model, path, elapsed_days=1.5)
            payload = _read_payload(path)

        self.assertEqual(int(payload["schema_version"]), SCHEMA_VERSION)
        self.assertIsInstance(payload["jcm_version"], str)
        self.assertAlmostEqual(float(payload["elapsed_days"]), 1.5)
        # Arrays are keyed by their pytree NAME, which is what makes a
        # field-set change migratable rather than fatal.
        self.assertIn("vorticity", payload["dycore"])
        self.assertIn("tracers.specific_humidity", payload["dycore"])
        self.assertIn("_prev_step.specific_humidity", payload["physics"])
        # Per-group field names, including the root group's members.
        self.assertIn("_prev_step", payload["physics_fields"]["<root>"].values())
        # ``specific_humidity`` is a kg/kg mass mixing ratio even though no
        # term declares it — the flag a unit migration keys on.
        self.assertTrue(payload["dycore_tracers"]["specific_humidity"])

    def test_newer_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(self.model, path, elapsed_days=0.0)
            payload = _read_payload(path)
            payload["schema_version"] = SCHEMA_VERSION + 7
            _write_payload(path, payload)

            fresh = _build_model()
            fresh.bootstrap_state()
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(fresh, path)
        self.assertIn(f"schema {SCHEMA_VERSION + 7}", str(ctx.exception))
        self.assertIn("Upgrade jcm", str(ctx.exception))


class TestPhysicsCarryFieldMigration(unittest.TestCase):
    """Name matching migrates a changed carry field set (#731, option 1)."""

    def test_extra_field_in_model_is_seeded_from_fresh_carry(self):
        """A model whose carry gained a field restores an older file."""
        donor = _build_model()
        donor.bootstrap_state()
        upgraded = _build_model(
            physics=held_suarez_physics() + _ExtraCarryTerm())
        _, fresh_carry = upgraded.bootstrap_state()
        expected = fresh_carry["extra_carry"].counter

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=3.0)
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                elapsed = load_checkpoint(upgraded, path)

        self.assertAlmostEqual(elapsed, 3.0)
        restored = upgraded.physics_carry["extra_carry"].counter
        self.assertEqual(restored.shape, expected.shape)
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(expected))
        self.assertTrue(
            any("extra_carry.counter" in line and "seeded" in line
                for line in logs.output),
            logs.output,
        )
        # The fields both versions share still come from the file.
        self.assertEqual(
            _max_abs_diff(donor.physics_carry["_prev_step"],
                          upgraded.physics_carry["_prev_step"]),
            0.0,
        )

    def test_the_seed_comes_from_a_fresh_carry_not_the_current_one(self):
        """The restore template may be an evolved carry; the seed must not be.

        ``load_checkpoint`` accepts a model whose state came from an
        earlier ``Model.run``, so the pytree it deserializes against can
        hold that run's evolved values. A field the checkpoint predates
        must still be filled with the term's documented seed, not with
        the unrelated run's state.
        """
        donor = _build_model()
        donor.bootstrap_state()
        upgraded = _build_model(
            physics=held_suarez_physics() + _ExtraCarryTerm())
        state, carry = upgraded.bootstrap_state()
        # Stand in for a carry an earlier integration left behind.
        evolved = dict(carry)
        evolved["extra_carry"] = _ExtraCarryData(
            counter=jnp.full_like(carry["extra_carry"].counter, 7.0))
        upgraded.restore_state(state, evolved)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=1.0)
            load_checkpoint(upgraded, path)

        restored = np.asarray(upgraded.physics_carry["extra_carry"].counter)
        np.testing.assert_array_equal(restored, np.zeros_like(restored))

    def test_field_the_model_no_longer_carries_is_dropped(self):
        """A file from a jcm whose carry had one more field still loads."""
        donor = _build_model(physics=held_suarez_physics() + _ExtraCarryTerm())
        donor.bootstrap_state()
        target = _build_model()
        target.bootstrap_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=4.0)
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                elapsed = load_checkpoint(target, path)

        self.assertAlmostEqual(elapsed, 4.0)
        self.assertNotIn("extra_carry", target.physics_carry)
        self.assertTrue(
            any("extra_carry.counter" in line and "dropped" in line
                for line in logs.output),
            logs.output,
        )

    def test_a_prognostic_slot_is_not_seeded(self):
        """Nothing recomputes it, so a zero seed would invent mass."""
        donor = _build_model()
        donor.bootstrap_state()
        upgraded = _build_model(
            physics=held_suarez_physics() + _PrognosticCarryTerm())
        upgraded.bootstrap_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=1.0)
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(upgraded, path)
        message = str(ctx.exception)
        self.assertIn("reservoir", message)
        self.assertIn("prognostic state", message)

    def test_a_prognostic_slot_is_not_dropped(self):
        """The file records the slot, so a reader without the term still refuses."""
        donor = _build_model(
            physics=held_suarez_physics() + _PrognosticCarryTerm())
        donor.bootstrap_state()
        target = _build_model()
        target.bootstrap_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=1.0)
            payload = _read_payload(path)
            self.assertIn(
                "reservoir",
                list(payload["prognostic_carry_slots"].values()),
            )
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(target, path)
        self.assertIn("reservoir", str(ctx.exception))

    def test_jam_cloud_borne_store_declares_its_slot(self):
        """The real prognostic carry (#602) is declared, not just the stand-in."""
        from jcm.physics.aerosol.jam.cloud_borne_store import (
            CARRY_KEY, CloudBorneCarryStore,
        )

        self.assertEqual(
            CloudBorneCarryStore.prognostic_carry_slots, (CARRY_KEY,))

    def test_same_name_wrong_shape_still_errors(self):
        """A shared field whose shape changed names the file and the field."""
        model = _build_model()
        model.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(model, path, elapsed_days=0.0)
            payload = _read_payload(path)
            name = "_prev_step.specific_humidity"
            payload["physics"][name] = np.zeros((3, 4), dtype=np.float32)
            _write_payload(path, payload)

            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(model, path)
        message = str(ctx.exception)
        self.assertIn(name, message)
        self.assertIn("ckpt.msgpack", message)
        self.assertIn("shape", message)

    def test_same_name_wrong_dtype_still_errors(self):
        """Precision is not migrated: a float64 array is not a float32 one."""
        model = _build_model()
        model.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(model, path, elapsed_days=0.0)
            payload = _read_payload(path)
            name = "_prev_step.specific_humidity"
            payload["physics"][name] = np.asarray(
                payload["physics"][name], dtype=np.float64)
            _write_payload(path, payload)

            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(model, path)
        self.assertIn(name, str(ctx.exception))
        self.assertIn("dtype", str(ctx.exception))

    def test_missing_dycore_field_is_refused(self):
        """Prognostic state is never invented, only diagnostics are."""
        model = _build_model()
        model.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(model, path, elapsed_days=0.0)
            payload = _read_payload(path)
            del payload["dycore"]["tracers.specific_humidity"]
            _write_payload(path, payload)

            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(model, path)
        self.assertIn("tracers.specific_humidity", str(ctx.exception))
        self.assertIn("dycore state", str(ctx.exception))

    def test_extra_dycore_field_is_refused(self):
        """A tracer this composition does not carry is a composition change."""
        model = _build_model()
        model.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(model, path, elapsed_days=0.0)
            payload = _read_payload(path)
            payload["dycore"]["tracers.qc"] = np.asarray(
                payload["dycore"]["tracers.specific_humidity"])
            _write_payload(path, payload)

            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(model, path)
        self.assertIn("tracers.qc", str(ctx.exception))

    def test_grid_change_is_refused(self):
        """A different truncation is a shape mismatch, named on the file."""
        donor = _build_model(spectral_truncation=31)
        donor.bootstrap_state()
        other_grid = _build_model(spectral_truncation=21)
        other_grid.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.0)
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(other_grid, path)
        message = str(ctx.exception)
        self.assertIn("ckpt.msgpack", message)
        self.assertIn("wrong grid/levels/physics", message)


class TestUnstampedCheckpoints(unittest.TestCase):
    """Pre-policy files: refused unless the caller states their units.

    PR #824 made every mass mixing ratio cross the dycore boundary as the
    physical kg/kg value. An unstamped file records neither its schema nor
    the physics package that wrote it, so which factor each array needs is
    not recoverable — see ``docs/source/design/checkpoint_compatibility.md``.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _build_model()
        cls.model.bootstrap_state()
        cls.q_name = "tracers.specific_humidity"

    def test_unstamped_is_refused_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(self.model, path, elapsed_days=7.0)
            target = _build_model()
            target.bootstrap_state()
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(target, path)
        message = str(ctx.exception)
        self.assertIn("no schema_version stamp", message)
        self.assertIn("unstamped_scale", message)
        self.assertIn("#824", message)

    def test_unstamped_loads_on_an_explicit_no_op_assertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(self.model, path, elapsed_days=7.0)
            target = _build_model()
            target.bootstrap_state()
            elapsed = load_checkpoint(target, path, unstamped_scale={})
        self.assertAlmostEqual(elapsed, 7.0)
        self.assertEqual(
            _max_abs_diff(self.model.dycore_state, target.dycore_state), 0.0)

    def test_unstamped_scale_rescales_only_the_named_leaves(self):
        donor = _build_model()
        donor.bootstrap_state()
        # Give the humidity tracer a non-zero value so a factor is visible.
        state = donor.dycore_state
        moist = state.replace(tracers={
            **state.tracers,
            "specific_humidity": jnp.full_like(
                state.tracers["specific_humidity"], 2.0),
        })
        donor.restore_state(moist, donor.physics_carry)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(donor, path, elapsed_days=1.0)
            target = _build_model()
            target.bootstrap_state()
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                load_checkpoint(target, path,
                                unstamped_scale={self.q_name: 1000.0})

        restored = np.asarray(target.dycore_state.tracers["specific_humidity"])
        np.testing.assert_allclose(restored, 2000.0, rtol=1e-6)
        # An unnamed leaf keeps the file's value.
        np.testing.assert_array_equal(
            np.asarray(target.dycore_state.vorticity),
            np.asarray(donor.dycore_state.vorticity),
        )
        self.assertTrue(
            any(self.q_name in line and "1000" in line for line in logs.output),
            logs.output,
        )

    def test_unstamped_scale_rejects_an_unknown_leaf_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(self.model, path, elapsed_days=0.0)
            target = _build_model()
            target.bootstrap_state()
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(target, path,
                                unstamped_scale={"tracers.nope": 1000.0})
        self.assertIn("tracers.nope", str(ctx.exception))

    def test_unstamped_scale_is_refused_for_a_stamped_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(self.model, path, elapsed_days=0.0)
            target = _build_model()
            target.bootstrap_state()
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(target, path,
                                unstamped_scale={self.q_name: 1000.0})
        self.assertIn("stamped schema", str(ctx.exception))

    def test_unstamped_structural_mismatch_cannot_be_migrated(self):
        """With no names in the file there is nothing to match on."""
        donor = _build_model(physics=held_suarez_physics() + _ExtraCarryTerm())
        donor.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(donor, path, elapsed_days=0.0)
            target = _build_model()
            target.bootstrap_state()
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(target, path, unstamped_scale={})
        self.assertIn("carries no field names", str(ctx.exception))


class TestParseUnstampedScale(unittest.TestCase):
    """The config-facing form of the explicit unit assertion."""

    def test_none_stays_none(self):
        self.assertIsNone(parse_unstamped_scale(None))

    def test_mapping_is_coerced_to_floats(self):
        self.assertEqual(
            parse_unstamped_scale({"tracers.qc": 1000}),
            {"tracers.qc": 1000.0},
        )

    def test_entry_sequence_is_parsed(self):
        self.assertEqual(
            parse_unstamped_scale(["tracers.qc=1000", "tracers.qi=1e3"]),
            {"tracers.qc": 1000.0, "tracers.qi": 1000.0},
        )

    def test_empty_sequence_is_an_explicit_no_op(self):
        self.assertEqual(parse_unstamped_scale([]), {})

    def test_malformed_entries_raise(self):
        for bad in (["tracers.qc"], ["=1000"], ["tracers.qc=x"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_unstamped_scale(bad)

    def test_bare_string_raises(self):
        with self.assertRaises(ValueError):
            parse_unstamped_scale("tracers.qc=1000")


class TestCompositionCoverage(unittest.TestCase):
    """The stamp and the round trip on a moist SPEEDY and ECHAM state."""

    def _round_trip(self, build):
        donor = build()
        donor.bootstrap_state()
        target = build()
        target.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=2.0)
            payload = _read_payload(path)
            elapsed = load_checkpoint(target, path)
        self.assertAlmostEqual(elapsed, 2.0)
        self.assertEqual(
            _max_abs_diff(donor.dycore_state, target.dycore_state), 0.0)
        self.assertEqual(
            _max_abs_diff(donor.physics_carry, target.physics_carry), 0.0)
        return payload, donor, target

    def test_speedy_round_trip_and_tracer_metadata(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics

        def build():
            coords = get_speedy_coords(layers=8, spectral_truncation=21)
            return Model(coords=coords,
                         terrain=TerrainData.from_coords(coords),
                         physics=speedy_physics())

        payload, donor, target = self._round_trip(build)
        # SPEEDY declares no extra tracers: humidity is the whole set.
        self.assertEqual(list(payload["dycore_tracers"].values()), [True])
        self.assertIn("tracers.specific_humidity", payload["dycore"])

        # SPEEDY's carry holds integer and boolean sub-cycle state
        # (``_shortwave_rad.step``, ``_convection.iptop``). A unit rescale
        # on one of those would quietly corrupt it, so asserting a factor
        # for a non-float leaf is refused rather than applied.
        integer_leaf = next(
            name for name, leaf in _named_leaves(donor.physics_carry)
            if not np.issubdtype(leaf.dtype, np.floating)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(donor, path, elapsed_days=0.0)
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(target, path,
                                unstamped_scale={integer_leaf: 1000.0})
        self.assertIn("floating-point", str(ctx.exception))

    def test_echam_round_trip_and_condensate_metadata(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.utils import get_coords

        def build():
            coords = get_coords(get_echam_levels(47), spectral_truncation=21)
            return Model(coords=coords,
                         terrain=TerrainData.aquaplanet(coords),
                         physics=echam_physics(radiation_scheme="grey"))

        payload, _, _ = self._round_trip(build)
        tracers = payload["dycore_tracers"]
        # The condensate species PR #824's rescale applies to, flagged as
        # the kg/kg mass mixing ratios they are.
        for name in ("qc", "qi"):
            self.assertTrue(tracers[name], tracers)
        # ECHAM's carry holds real diagnostic structs, whose field names
        # are what a future field-set change migrates on.
        self.assertIn("radiation", payload["physics_fields"]["<root>"].values())
        self.assertTrue(
            any(key.startswith("radiation.") for key in payload["physics"]),
            sorted(payload["physics"])[:10],
        )



class TestSurfaceOpticsAcrossRestart(unittest.TestCase):
    """The radiation's surface optics survive a restart bit for bit (#672).

    The boundary-condition term hands the radiation this step's surface
    albedo / emissivity under a step-local key that ``ComposablePhysics``
    drops before the carry (so it is never checkpointed); the radiation
    publishes the values it solved with in ``radiation.surface_*`` and holds
    them between solves. A restart in the middle of a radiation interval
    must therefore replay the held, solve-time albedo exactly, and a file
    carrying the step-local key (written by an intermediate build) must load
    to the same result.
    """

    DT_MIN = 30          # model step [min]
    EVERY = 4            # radiation solves every 4 steps (2 h)

    def _build(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.radiation_types import RadiationParameters
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        return Model(
            coords=coords, terrain=TerrainData.aquaplanet(coords),
            time_step=self.DT_MIN,
            physics=echam_physics(
                radiation_scheme="grey",
                radiation=RadiationParameters.default(
                    radiation_interval=self.EVERY * self.DT_MIN * 60.0)))

    @staticmethod
    def _radiation(model):
        rad = model.physics_carry["radiation"]
        return {name: np.asarray(getattr(rad, name)) for name in (
            "surface_albedo_vis", "surface_albedo_nir", "surface_emissivity",
            "sw_flux_up", "sw_flux_down", "surface_sw_up", "surface_sw_down",
            "toa_sw_up", "sw_heating_rate")}

    @pytest.mark.slow
    def test_mid_interval_restart_is_bit_identical(self):
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.radiation import SURFACE_OPTICS_KEY

        self.assertIn(SURFACE_OPTICS_KEY, ComposablePhysics._STEP_LOCAL_KEYS)
        step_days = self.DT_MIN / 1440.0
        total = 2 * self.EVERY            # two radiation intervals
        split = self.EVERY + 2            # mid-interval: a cached step next

        baseline = self._build()
        baseline.run(save_interval=step_days, total_time=total * step_days)
        want = self._radiation(baseline)
        # Anti-vacuity: the sun is up somewhere and the ocean albedo varies.
        self.assertGreater(float(want["surface_sw_down"].max()), 100.0)
        self.assertGreater(float(np.ptp(want["surface_albedo_vis"])), 1e-3)

        first = self._build()
        first.run(save_interval=step_days, total_time=split * step_days)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(first, path, elapsed_days=split * step_days)
            payload = _read_payload(path)
            # Step-local: never part of the checkpointed carry.
            self.assertFalse(
                [k for k in payload["physics"] if SURFACE_OPTICS_KEY in k])

            resumed = self._build()
            resumed.bootstrap_state()
            load_checkpoint(resumed, path)
            resumed.resume(save_interval=step_days,
                           total_time=(total - split) * step_days)

            # A file that does carry the step-local key (written by a build
            # that checkpointed it) loads with the key dropped: zeros there
            # can never reach a solve.
            stale = dict(payload)
            stale["physics"] = dict(payload["physics"])
            ncols_shape = want["surface_albedo_vis"].shape
            for field in ("albedo_vis", "albedo_nir", "emissivity"):
                stale["physics"][f"{SURFACE_OPTICS_KEY}.{field}"] = (
                    np.zeros(ncols_shape, np.float32))
            stale_path = Path(tmp) / "stale.msgpack"
            _write_payload(stale_path, stale)
            from_stale = self._build()
            from_stale.bootstrap_state()
            load_checkpoint(from_stale, stale_path)
            from_stale.resume(save_interval=step_days,
                              total_time=(total - split) * step_days)

        for label, model in (("resumed", resumed), ("stale-key", from_stale)):
            got = self._radiation(model)
            for name, value in want.items():
                np.testing.assert_array_equal(
                    got[name], value, err_msg=f"{label}: radiation.{name}")


if __name__ == "__main__":
    unittest.main()
