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
5. A save that fails part-way (quota, full disk) leaves the previous
   checkpoint and its ``.prev`` exactly as they were (#1006), and a file
   that cannot be decoded is told apart from one that does not fit.

Uses Held-Suarez physics for speed: no moisture, no radiation, deterministic
forcing. The composition-coverage tests build one SPEEDY and one ECHAM model
(bootstrap only, no integration) to exercise a moist tracer set.
"""

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tree_math

from jcm.checkpoint import (
    SCHEMA_VERSION, CheckpointUnreadableError, _named_leaves, atomic_open,
    load_checkpoint, parse_unstamped_scale, save_checkpoint,
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
            save_checkpoint(self.model, path, elapsed_days=0.0)
            payload = _read_payload(path)

        self.assertEqual(int(payload["schema_version"]), SCHEMA_VERSION)
        self.assertIsInstance(payload["jcm_version"], str)
        self.assertAlmostEqual(float(payload["elapsed_days"]), 0.0)
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

    def test_mirror_revision_round_trips_through_metadata(self):
        from unittest import mock

        from jcm.data import remote
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {remote.REVISION_ENV: "a" * 40}):
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(self.model, path, elapsed_days=0.0)
            fresh = _build_model()
            fresh.bootstrap_state()
            meta = {}
            load_checkpoint(fresh, path, metadata=meta)
        self.assertEqual(meta["data_mirror_revision"], "a" * 40)

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
            save_checkpoint(donor, path, elapsed_days=0.0)
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                elapsed = load_checkpoint(upgraded, path)

        self.assertAlmostEqual(elapsed, 0.0)
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
        upgraded.restore_state(state, evolved, time=upgraded.run_state.time,
                               step=upgraded.run_state.step)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.0)
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
            save_checkpoint(donor, path, elapsed_days=0.0)
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                elapsed = load_checkpoint(target, path)

        self.assertAlmostEqual(elapsed, 0.0)
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
            save_checkpoint(donor, path, elapsed_days=0.0)
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
            save_checkpoint(donor, path, elapsed_days=0.0)
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
        self.assertIn("no exact Gregorian clock", message)
        self.assertIn("unstamped_scale", message)
        self.assertIn("as_initial_condition", message)

    def test_unstamped_loads_on_an_explicit_no_op_assertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(self.model, path, elapsed_days=7.0)
            target = _build_model()
            target.bootstrap_state()
            elapsed = load_checkpoint(target, path, unstamped_scale={},
                                      as_initial_condition=True)
        # Documented return contract: the donor's recorded elapsed days,
        # while the imported state's clock restarts at start_time.
        self.assertAlmostEqual(elapsed, 7.0)
        run_state = target.run_state
        restarted = run_state.time - target.start_time
        self.assertEqual((int(restarted.days), int(restarted.seconds)), (0, 0))
        self.assertEqual(int(run_state.step), 0)
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
        donor.restore_state(moist, donor.physics_carry, time=donor.run_state.time,
                            step=donor.run_state.step)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(donor, path, elapsed_days=0.0)
            target = _build_model()
            target.bootstrap_state()
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                load_checkpoint(target, path,
                                unstamped_scale={self.q_name: 1000.0},
                                as_initial_condition=True)

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
                                unstamped_scale={"tracers.nope": 1000.0},
                                as_initial_condition=True)
        self.assertIn("tracers.nope", str(ctx.exception))

    def test_unstamped_scale_is_refused_for_a_stamped_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(self.model, path, elapsed_days=0.0)
            target = _build_model()
            target.bootstrap_state()
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(target, path,
                                unstamped_scale={self.q_name: 1000.0},
                                as_initial_condition=True)
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
                load_checkpoint(target, path, unstamped_scale={},
                                      as_initial_condition=True)
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
            save_checkpoint(donor, path, elapsed_days=0.0)
            payload = _read_payload(path)
            elapsed = load_checkpoint(target, path)
        self.assertAlmostEqual(elapsed, 0.0)
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
                                unstamped_scale={integer_leaf: 1000.0},
                                as_initial_condition=True)
        self.assertIn("floating-point", str(ctx.exception))

    def test_echam_round_trip_and_condensate_metadata(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.utils import get_coords

        def build():
            coords = get_coords(get_echam_levels(47), spectral_truncation=21)
            return Model(coords=coords,
                         terrain=TerrainData.aquaplanet(coords),
                         physics=idealized_echam_physics())

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
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.physics.radiation.radiation_types import RadiationParameters
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        return Model(
            coords=coords, terrain=TerrainData.aquaplanet(coords),
            time_step=self.DT_MIN,
            physics=idealized_echam_physics(
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
        """One Model object, reset between phases, drives this comparison.

        Compile (tracing the idealized ECHAM physics graph once per distinct
        ``(model object, save_interval, total_time)``) dominates this test's
        wall time far more than the handful of 30-minute steps it executes,
        so the lever that shrinks it is calls, not simulated time. But the
        reference ("want") must come from a GENUINELY UNINTERRUPTED 8-step
        ``run()`` -- not a 6-step ``run()`` followed by a 2-step ``resume()``
        on the same object, which an earlier version of this test did to
        save a compile (Codex P2, PR #1143 comment 4295019375): every
        compared trajectory (the reference AND both checkpoint-resume legs)
        would then cross the same ``resume()`` boundary, so a bug that
        shifted the radiation sub-cycle schedule or mishandled the held
        surface-optics carry across THAT specific API seam could make the
        reference wrong in exactly the way the checkpoint-loaded legs are
        also wrong, and this test would still pass. (The general
        resumption-equivalence test, ``TestCheckpointResumptionEquivalence``,
        uses Held-Suarez and never exercises the subcycled-radiation carry
        this test is about, so it cannot stand in for that check.) The
        reference therefore pays its own, distinct compile:
          - ``donor.run(6 steps)`` writes the checkpoint at the mid-interval
            mark this test needs (``split``).
          - ``donor.bootstrap_state()`` + ``donor.run(8 steps)``, reusing the
            same object, is a fresh, single, uninterrupted call from the
            same deterministic initial condition -- the genuine continuous
            reference. Reusing the object only resets what ``bootstrap_
            state``/``run`` themselves reset; it shares no trace with the
            6-step donor run (different ``total_time``) or with the resume
            legs below, but does avoid a second ``Model`` CONSTRUCTION.
          - The checkpoint is then loaded into the SAME object twice in a
            row (``bootstrap_state`` + ``load_checkpoint`` + ``resume(2
            steps)``), once from the good payload and once from the "stale
            step-local key" payload: same object, same static
            ``total_time``, so the second load's trace is a pure dispatch-
            cache hit (measured ~0.3s vs ~20s for the first -- this part of
            the original shrink stands, since it is the resume legs
            sharing a trace with EACH OTHER, not with the reference).
        Net: 3 distinct traces (6-step donor, 8-step reference, 2-step
        resume) instead of the original 4 (the same 3 plus a 2-step resume
        for EACH checkpoint leg) -- a real but smaller saving than the
        previous version of this shrink claimed, with the same
        exact-equality assertions on the same fields. Measured locally:
        124s before any shrink, 81s with this fix (the 2-step resume-leg
        sharing -- resumed and stale-key hitting the same dispatch-cache
        entry -- is the saving that survives review).
        """
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.radiation import SURFACE_OPTICS_KEY

        self.assertIn(SURFACE_OPTICS_KEY, ComposablePhysics._STEP_LOCAL_KEYS)
        step_days = self.DT_MIN / 1440.0
        total = 2 * self.EVERY            # two radiation intervals
        split = self.EVERY + 2            # mid-interval: a cached step next

        probe = self._build()
        probe.run(save_interval=step_days, total_time=split * step_days)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(probe, path, elapsed_days=split * step_days)
            payload = _read_payload(path)
            # Step-local: never part of the checkpointed carry.
            self.assertFalse(
                [k for k in payload["physics"] if SURFACE_OPTICS_KEY in k])

            # The reference: ONE uninterrupted run from a fresh bootstrap,
            # never through resume() -- see the docstring on why this must
            # not reuse the 6-step donor's in-flight state.
            probe.bootstrap_state()
            probe.run(save_interval=step_days, total_time=total * step_days)
            want = self._radiation(probe)
            # Anti-vacuity: the sun is up somewhere and the ocean albedo varies.
            self.assertGreater(float(want["surface_sw_down"].max()), 100.0)
            self.assertGreater(float(np.ptp(want["surface_albedo_vis"])), 1e-3)

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

            results = {}
            for label, load_path in (("resumed", path), ("stale-key", stale_path)):
                probe.bootstrap_state()
                load_checkpoint(probe, load_path)
                probe.resume(save_interval=step_days,
                            total_time=(total - split) * step_days)
                results[label] = self._radiation(probe)

        for label, got in results.items():
            for name, value in want.items():
                np.testing.assert_array_equal(
                    got[name], value, err_msg=f"{label}: radiation.{name}")


if __name__ == "__main__":
    unittest.main()


class TestExactCheckpointClock(unittest.TestCase):
    """Resumes preserve exact dates; legacy fields are initialization only."""

    def test_clock_round_trip_and_reject_inconsistent_elapsed_metadata(self):
        import jax_datetime as jdt

        donor = _build_model()
        state, carry = donor.bootstrap_state()
        # An exact sub-day offset must survive without an epoch float.
        donor.restore_state(state, carry,
                            time=donor.start_time + jdt.Timedelta(seconds=10800),
                            step=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clock.msgpack"
            with self.assertRaisesRegex(ValueError, "elapsed_days"):
                save_checkpoint(donor, path, elapsed_days=1.0)
            save_checkpoint(donor, path, elapsed_days=0.125)
            target = _build_model()
            target.bootstrap_state()
            load_checkpoint(target, path)
            self.assertEqual(int(target.run_state.step), 1)
            self.assertEqual(int((target.run_state.time - donor.run_state.time).seconds), 0)
            self.assertEqual(int((target.run_state.time - donor.run_state.time).days), 0)
            payload = _read_payload(path)
            payload["clock"]["step"] = np.int32(2)
            _write_payload(path, payload)
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_checkpoint(target, path)

    def test_schema_one_requires_new_experiment(self):
        donor = _build_model()
        donor.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.0)
            payload = _read_payload(path)
            payload["schema_version"] = 1
            del payload["clock"]
            _write_payload(path, payload)
            target = _build_model()
            target.bootstrap_state()
            with self.assertRaisesRegex(ValueError, "as_initial_condition"):
                load_checkpoint(target, path)
            load_checkpoint(target, path, as_initial_condition=True)
            self.assertEqual(int(target.run_state.step), 0)


class _AnchorReader(PhysicsTerm):
    """Asks for the carried post-physics state and does nothing else.

    Stands in for a tendency-driven cloud scheme (the Lohmann 2M declares the
    same), so the slot's checkpoint behaviour is tested on Held-Suarez.
    """

    name = "anchor_reader"
    category = "test_anchor_reader"
    requires_post_physics_fields = ("temperature", "specific_humidity")

    def __call__(self, state, diagnostics, forcing, terrain):
        """No tendency; the carry slot is the point."""
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


class TestPostPhysicsSlotAcrossRestart(unittest.TestCase):
    """The carried post-physics state restores by name, like every carry slot."""

    def test_checkpoint_without_the_slot_restores_it_invalid(self):
        """A file written before the slot existed: template seed, valid = 0."""
        donor = _build_model()
        donor.run(save_interval=0.25, total_time=0.25)
        upgraded = _build_model(physics=held_suarez_physics() + _AnchorReader())
        upgraded.bootstrap_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.25)
            self.assertNotIn("_post_physics_state", donor.physics_carry)
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                load_checkpoint(upgraded, path)

        slot = upgraded.physics_carry["_post_physics_state"]
        self.assertEqual(float(slot["valid"]), 0.0)
        np.testing.assert_array_equal(np.asarray(slot["temperature"]), 0.0)
        self.assertTrue(
            any("_post_physics_state" in line and "seeded" in line
                for line in logs.output), logs.output)
        # The run continues: the first step falls back, the next ones are valid.
        upgraded.resume(save_interval=0.25, total_time=0.25)
        slot = upgraded.physics_carry["_post_physics_state"]
        self.assertEqual(float(slot["valid"]), 1.0)
        self.assertTrue(np.isfinite(np.asarray(slot["temperature"])).all())
        self.assertGreater(float(np.min(np.asarray(slot["temperature"]))), 100.0)

    def test_slot_round_trips_valid(self):
        first = _build_model(physics=held_suarez_physics() + _AnchorReader())
        first.run(save_interval=0.25, total_time=0.25)
        saved = first.physics_carry["_post_physics_state"]
        self.assertEqual(float(saved["valid"]), 1.0)
        second = _build_model(physics=held_suarez_physics() + _AnchorReader())
        second.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(first, path, elapsed_days=0.25)
            load_checkpoint(second, path)
        self.assertEqual(
            _max_abs_diff(saved, second.physics_carry["_post_physics_state"]), 0.0)


class _TteTkeCarryTerm(PhysicsTerm):
    """Carries the TTE-TKE term's slot and no physics (see the class below)."""

    name = "tte_tke_carry"
    category = "test_tte_tke_carry"

    def __init__(self):
        """Declare the ``vertical_diffusion`` slot of the TTE-TKE term."""
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )
        self.carry_slots = {"vertical_diffusion": VerticalDiffusionData}

    def __call__(self, state, diagnostics, forcing, terrain):
        """No tendency, no diagnostic writes — the slot is the point."""
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


class TestTteTkeInterfaceRowMigration(unittest.TestCase):
    """A file written before #996 has its TTE-TKE carry on full levels.

    Schema 4 moved TKE, the θ_v variance and ``km``/``kh`` onto ECHAM's
    interface rows: row ``k`` is the interface below full level ``k``, where a
    schema-3 file's row ``k`` held the interface above it. The rows are moved
    up one on load, and the surface row repeats the lowest interior interface.
    """

    FIELDS = ("tke", "thv_variance", "km", "kh")

    def _model(self):
        return _build_model(physics=held_suarez_physics() + _TteTkeCarryTerm(),
                            spectral_truncation=21)

    def _donor_payload(self, path, schema):
        donor = self._model()
        donor.bootstrap_state()
        save_checkpoint(donor, path, elapsed_days=0.0)
        payload = _read_payload(path)
        nlev = np.asarray(payload["physics"]["vertical_diffusion.tke"]).shape[0]
        profile = {}
        for i, name in enumerate(self.FIELDS):
            key = f"vertical_diffusion.{name}"
            old = np.asarray(payload["physics"][key])
            values = (np.arange(nlev, dtype=old.dtype) + 1.0 + 100.0 * i)
            values = values.reshape((nlev,) + (1,) * (old.ndim - 1)) * np.ones_like(old)
            payload["physics"][key] = values
            profile[name] = values
        payload["schema_version"] = schema
        _write_payload(path, payload)
        return profile

    def test_a_schema_3_file_moves_up_one_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pre996.msgpack"
            old = self._donor_payload(path, schema=3)
            target = self._model()
            target.bootstrap_state()
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                load_checkpoint(target, path)
        self.assertTrue(any("interface rows" in line for line in logs.output),
                        logs.output)
        carry = target.physics_carry["vertical_diffusion"]
        for name in self.FIELDS:
            got = np.asarray(getattr(carry, name))
            np.testing.assert_array_equal(got[:-1], old[name][1:])
            # The surface row is what the scheme writes there: 0 for the
            # exchange coefficients, the lowest interior interface otherwise.
            want = 0.0 if name in ("km", "kh") else old[name][-1]
            np.testing.assert_array_equal(got[-1], want)

    def test_a_schema_4_file_is_read_as_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "post996.msgpack"
            new = self._donor_payload(path, schema=SCHEMA_VERSION)
            target = self._model()
            target.bootstrap_state()
            load_checkpoint(target, path)
        carry = target.physics_carry["vertical_diffusion"]
        for name in self.FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(carry, name)), new[name])

    def test_an_unstamped_file_moves_up_one_row_too(self):
        donor = self._model()
        donor.bootstrap_state()
        tke = np.asarray(donor.physics_carry["vertical_diffusion"].tke)
        nlev = tke.shape[0]
        profile = (np.arange(nlev, dtype=tke.dtype) + 1.0)[:, None] * np.ones_like(tke)
        donor.restore_state(
            donor.dycore_state,
            {**donor.physics_carry,
             "vertical_diffusion": donor.physics_carry["vertical_diffusion"].copy(
                 tke=jnp.asarray(profile))},
            time=donor.run_state.time, step=donor.run_state.step)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.msgpack"
            _write_unstamped(donor, path, elapsed_days=0.0)
            target = self._model()
            target.bootstrap_state()
            load_checkpoint(target, path, unstamped_scale={}, as_initial_condition=True)
        got = np.asarray(target.physics_carry["vertical_diffusion"].tke)
        np.testing.assert_array_equal(got[:-1], profile[1:])


class TestLandSkinTemperatureMigration(unittest.TestCase):
    """A checkpoint written before the prognostic land skin temperature (#979).

    ``SurfaceData`` gained ``land_surface_temperature``, the land-balance
    diagnostics and ``land_albedo_at_solve``. A file without them restores
    through the name-matched carry migration: the new leaves take the
    bootstrapped zeros, which ``EchamBoundaryConditions`` reads as "unset" for
    the skin and seeds from ``stl_am`` on the first step — the value the land
    had when it was prescribed — and which the land balance reads as "unset"
    for the albedo and replaces with the current step's until the radiation
    next solves.
    """

    def _model(self):
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.diagnostics.moist_air_state import MoistAirColumnState
        from jcm.physics.forcing.echam_boundary_conditions import EchamBoundaryConditions

        coords = get_held_suarez_coords(spectral_truncation=21)
        physics = ComposablePhysics(
            terms=[MoistAirColumnState(), EchamBoundaryConditions()],
            vectorize_columns=True)
        return Model(coords=coords, terrain=TerrainData.from_coords(coords), time_step=30,
                     physics=physics)

    def test_pre_979_file_restores_and_seeds_the_skin_from_stl(self):
        from jcm.physics.surface.echam.surface_types import LAND_FIELDS

        donor = self._model()
        donor.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.0)
            payload = _read_payload(path)
            # Strip the fields #979 added, as a file from before it holds them.
            for name in LAND_FIELDS:
                del payload["physics"][f"surface.{name}"]
            fields = payload["physics_fields"]["surface"]
            payload["physics_fields"]["surface"] = {
                k: v for k, v in fields.items() if v not in LAND_FIELDS}
            _write_payload(path, payload)

            target = self._model()
            target.bootstrap_state()
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                load_checkpoint(target, path)
        self.assertTrue(any("surface.land_surface_temperature" in line and "seeded" in line
                            for line in logs.output), logs.output)
        restored = np.asarray(target.physics_carry["surface"].land_surface_temperature)
        np.testing.assert_array_equal(restored, 0.0)
        # The land albedo of the last radiation solve is among those fields: a
        # file without it restores it as unset (0), which the land balance
        # reads as "use the current albedo" until the next solve.
        self.assertTrue(any("surface.land_albedo_at_solve" in line and "seeded" in line
                            for line in logs.output), logs.output)
        np.testing.assert_array_equal(
            np.asarray(target.physics_carry["surface"].land_albedo_at_solve), 0.0)

        from jcm.forcing import default_forcing

        target.resume(save_interval=1, total_time=1)
        skin = np.asarray(target.physics_carry["surface"].land_surface_temperature).ravel()
        stl = np.asarray(default_forcing(target.coords.horizontal).stl_am).ravel()
        assert np.all(skin > 100.0), "the skin was not seeded"
        np.testing.assert_allclose(skin, stl, rtol=1e-6)

    def test_the_held_land_albedo_round_trips_by_name(self):
        """A file that has the slot restores the albedo the radiation last solved with."""
        donor = self._model()
        donor.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.0)
            payload = _read_payload(path)
            key = "surface.land_albedo_at_solve"
            payload["physics"][key] = np.full_like(np.asarray(payload["physics"][key]), 0.63)
            _write_payload(path, payload)

            target = self._model()
            target.bootstrap_state()
            load_checkpoint(target, path)
        np.testing.assert_allclose(
            np.asarray(target.physics_carry["surface"].land_albedo_at_solve), 0.63)


_ALBEDO_QUARTET_FIELDS = tuple(
    f"radiation.surface_albedo_{band}_{beam}"
    for band in ("vis", "nir") for beam in ("dir", "dif")
)


class TestAlbedoQuartetMigration(unittest.TestCase):
    """A checkpoint written before the per-band albedo quartet (jax-gcm#959).

    ``RadiationData`` gained ``surface_albedo_{vis,nir}_{dir,dif}``
    (schema 3). A file at an earlier schema restores through the generic
    name-matched carry migration like any other new field, EXCEPT that this
    one is not left to seed from the fresh carry's zero: it is registered in
    ``jcm.checkpoint._DERIVED_ON_MIGRATION`` (alongside ``mod_radcon.emis2``,
    #571) so each missing quartet leaf is derived by asking the RESUMED
    model's own composed radiation term how it would have published the
    quartet for a legacy (broadband-only) carry
    (``legacy_surface_albedo_quartet``) -- never a fixed formula baked into
    ``checkpoint.py``, because which value each backend's pre-quartet solve
    actually used is backend-specific: grey two-stream solved its two SW
    bands with the RAW vis/nir pair, while RRTMGP and the NN emulator
    BLENDED both into one scalar for every band (Codex review, three
    rounds on this PR: first an identity-per-sibling bug wrong for
    RRTMGP/the emulator when ``vis != nir``, then a fixed-blend bug wrong
    for a resumed GREY carry -- both the same root cause, guessing instead
    of asking the live backend). ``TestModRadConEmis2Migration`` above
    covers the single-source, backend-independent half of the same
    mechanism; this class covers the backend-dependent half, across all
    three radiation backends. The schema-2 fixed January warm states this
    A/B gate resumes from predate the quartet, so they depend on this
    migration to start at all under schema 3.
    """

    DT_MIN = 30
    EVERY = 4  # radiation solves every 4 steps (2 h)
    SCHEMES = ("grey", "rrtmgp", "emulated")

    def _build(self, scheme):
        """Build a model whose radiation term is ``scheme``.

        Real ``EchamBoundaryConditions`` + the named radiation
        ``PhysicsTerm`` (not a hand-built carry), so the quartet is
        exercised through the actual plumbing each backend runs in
        production. ``echam_physics()`` rejects ``radiation_scheme="grey"``
        by policy (it is test-only and never ECHAM-validated), so that one
        case goes through ``idealized_echam_physics()`` instead -- the same
        substitution ``TestSurfaceOpticsAcrossRestart`` makes, and the only
        difference from the other two cases is which factory builds the
        term stack, not what the test then does with it.
        """
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.physics.radiation.radiation_types import RadiationParameters
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        radiation_params = RadiationParameters.default(
            radiation_interval=self.EVERY * self.DT_MIN * 60.0)
        if scheme == "grey":
            physics = idealized_echam_physics(radiation=radiation_params)
        else:
            physics = echam_physics(
                radiation=radiation_params, radiation_scheme=scheme)
        return Model(
            coords=coords, terrain=TerrainData.aquaplanet(coords),
            time_step=self.DT_MIN, physics=physics)

    @staticmethod
    def _radiation_term(scheme):
        """Build the bare radiation ``PhysicsTerm`` for ``scheme``.

        Used to ask its own ``legacy_surface_albedo_quartet`` directly --
        the ground truth this test checks the migration against, since a
        pre-#959 file's lost per-tile information makes anything beyond
        this reconstruction (what the backend's own method says it would
        have used) both unrecoverable and not the thing under test.
        """
        if scheme == "grey":
            from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
            return GreyTwoStreamRadiation()
        if scheme == "rrtmgp":
            from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
            return RRTMGPRadiation()
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
        return NNEmulatorRadiation()

    def _pre_schema3_checkpoint(self, path, scheme):
        """Build a real run's checkpoint, stripped to schema-2's field set.

        Strips the quartet from a genuine, forcing-driven solve rather than
        hand-writing a broadband value, so the migrated result the test
        checks against is whatever that real run's own open-water/land/ice
        mix actually produced -- not a value chosen to make the assertion
        trivially true.
        """
        donor = self._build(scheme)
        step_days = self.DT_MIN / 1440.0
        # Stop mid-interval (the same "split = EVERY + 2" offset
        # TestSurfaceOpticsAcrossRestart validates as leaving a CACHED step
        # next): a resume whose first step is cached republishes the HELD
        # carry values untouched (rescale_cached_radiation never touches
        # the albedo fields), so an unmigrated zero would appear directly
        # in that step's output, not get overwritten by a fresh solve.
        split = self.EVERY + 2
        donor.run(save_interval=step_days, total_time=split * step_days)
        broadband = {
            name: np.asarray(getattr(donor.physics_carry["radiation"], name)).copy()
            for name in ("surface_albedo_vis", "surface_albedo_nir")
        }
        # Anti-vacuity: a flat ocean/zero albedo would pass even a broken
        # migration (zero happens to equal the broadband value too).
        self.assertGreater(
            float(np.ptp(broadband["surface_albedo_vis"])), 1e-3, scheme)
        # Anti-vacuity for the SPECIFIC bug Codex caught: vis and nir must
        # actually differ somewhere, or a formula that mixes up which
        # sibling (or which backend's convention) feeds which quartet slot
        # would coincide with the correct answer and this test would pass
        # either way.
        self.assertGreater(
            float(np.max(np.abs(
                broadband["surface_albedo_vis"] - broadband["surface_albedo_nir"]))),
            1e-3, scheme)
        save_checkpoint(donor, path, elapsed_days=split * step_days)
        payload = _read_payload(path)
        payload["schema_version"] = 2
        for name in _ALBEDO_QUARTET_FIELDS:
            del payload["physics"][name]
        fields = payload["physics_fields"]["radiation"]
        payload["physics_fields"]["radiation"] = {
            k: v for k, v in fields.items() if v not in _ALBEDO_QUARTET_FIELDS}
        _write_payload(path, payload)
        expected = self._radiation_term(scheme).legacy_surface_albedo_quartet(
            broadband["surface_albedo_vis"], broadband["surface_albedo_nir"])
        return dict(zip(
            ("surface_albedo_vis_dir", "surface_albedo_vis_dif",
            "surface_albedo_nir_dir", "surface_albedo_nir_dif"),
            expected)), step_days

    @pytest.mark.slow
    def test_migrated_quartet_and_the_next_cached_step_both_match(self):
        """Merges what were two tests sharing the same expensive setup.

        Both halves need, per scheme, the SAME pre-schema-3 donor checkpoint
        (a real 6-step solve written by ``_build(scheme)``, which for
        ``rrtmgp``/``emulated`` compiles the full correlated-k radiation
        graph) and the SAME freshly-loaded ``target`` Model. Building either
        twice -- once per test method, as two separate tests previously did
        -- paid that compile twice for no additional coverage, so this one
        test does the donor run and the target build/load ONCE per scheme
        and then asserts both invariants against them:
          1. immediately after ``load_checkpoint``, the quartet was DERIVED
             from the resumed backend's own legacy-publication method, not
             seeded with the generic migration's zero (jax-gcm#959);
          2. one further resumed step that is CACHED (no fresh radiation
             solve -- ``rescale_cached_radiation`` never touches the
             albedo fields) still publishes exactly that migrated quartet,
             not some other backend's convention.
        """
        for scheme in self.SCHEMES:
            with self.subTest(scheme=scheme):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "pre_schema3.msgpack"
                    expected, step_days = self._pre_schema3_checkpoint(
                        path, scheme)

                    target = self._build(scheme)
                    target.bootstrap_state()
                    with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                        load_checkpoint(target, path)

                    # --- invariant 1: migrated (not seeded) right after load ---
                    # The generic migration's "seeded from a freshly built
                    # carry" log names every field it had to invent a value
                    # for. There must be none of the quartet fields: a name
                    # appearing there would mean this fell through to the
                    # zero this migration exists to avoid. (The phrase
                    # "seeded from a freshly built carry" -- not a bare
                    # "seeded" substring -- distinguishes this from the
                    # DERIVED log line below, which also explains itself in
                    # terms of "seeded".)
                    generic_seed_lines = [
                        line for line in logs.output
                        if "seeded from a freshly built carry" in line]
                    rad = target.physics_carry["radiation"]
                    for name, want in expected.items():
                        self.assertFalse(
                            any(name in line for line in generic_seed_lines),
                            f"{scheme}: {name} was seeded from the fresh "
                            f"carry, not migrated: {generic_seed_lines}")
                        # And positively: each one DOES appear on the
                        # derived-field log line _DERIVED_ON_MIGRATION
                        # emits (shared with mod_radcon.emis2, #571) --
                        # confirming the quartet rides that mechanism, not
                        # some other path that happens to avoid the
                        # generic-seed phrase too.
                        self.assertTrue(
                            any(name in line and "derived" in line
                                for line in logs.output),
                            f"{scheme}: {name} was not named on a 'derived' "
                            f"log line: {logs.output}")
                        np.testing.assert_array_equal(
                            np.asarray(getattr(rad, name)), want,
                            err_msg=f"{scheme}: {name}")

                    # --- invariant 2: still correct after one CACHED step ---
                    # The donor stopped at "split = EVERY + 2" steps --
                    # mid-interval, same as TestSurfaceOpticsAcrossRestart
                    # -- so this one resumed step is CACHED, not a fresh
                    # solve. Continuing on the SAME target object (rather
                    # than building a second one, as the two separate
                    # tests used to) avoids paying for a second Model
                    # construction + checkpoint load for no new coverage.
                    target.resume(save_interval=step_days, total_time=step_days)
                    rad = target.physics_carry["radiation"]
                    for name, want in expected.items():
                        np.testing.assert_array_equal(
                            np.asarray(getattr(rad, name)), want,
                            err_msg=f"{scheme}: {name}")


_EFFECTIVE_RADII_AT_SOLVE_FIELDS = (
    "radiation.r_eff_liq_at_solve", "radiation.r_eff_ice_at_solve")


class TestEffectiveRadiiAtSolveMigration(unittest.TestCase):
    """A checkpoint written before the host-effective-radii carry (#1135).

    ``RadiationData`` gained ``r_eff_liq_at_solve``/``r_eff_ice_at_solve`` so
    a CACHED radiation step reports the exact radius its last SOLVE used,
    rather than re-reading ``clouds.r_eff_liq``/``r_eff_ice`` (which a host
    can overwrite every step regardless of whether this step solves). A file
    written before this pair existed has neither leaf: the generic
    fresh-carry zero seed is wrong for the same reason the #959 albedo
    quartet's was (``TestAlbedoQuartetMigration`` above) -- a resumed
    CACHED step would republish zero until the next solve recomputes it
    (Codex review on PR #1146). Registered in
    ``jcm.checkpoint._DERIVED_ON_MIGRATION``: each missing leaf is derived
    by asking the RESUMED model's own composed radiation term
    (``legacy_effective_radii_at_solve``) what it would have held, given
    the restored ``clouds.r_eff_liq``/``r_eff_ice`` -- never a fixed
    formula baked into ``checkpoint.py``, the same contract as
    ``legacy_surface_albedo_quartet``. RRTMGP and the NN emulator always
    published their last solve's radius onto ``clouds.r_eff_liq``/
    ``r_eff_ice`` even before this PR (no host-radii door existed
    pre-#1135), so the identity is correct for both; grey two-stream never
    forms an effective radius at all, so the migration returns zero for
    it too, not the (irrelevant) restored pair.

    Unlike ``TestAlbedoQuartetMigration``, this test reuses ONE model as
    both the donor (the real run that writes the checkpoint) and the
    resumed target (the reload + cached-step replay), per the slow-test
    budget in the dev-workflow skill: ``Model.run`` calls ``self.resume``
    internally, so reloading into and resuming the SAME already-run
    instance reuses that instance's already-compiled step rather than
    paying for a second one.

    The donor's own ``clouds.r_eff_liq``/``r_eff_ice`` are overwritten with
    synthetic, clearly-distinct constants right after the run rather than
    left at whatever the real physics produced: measured empirically, this
    aquaplanet setup's default (dry) initial condition has formed no cloud
    condensate at all by hour 3 -- or even by simulated day 3 -- so the
    real values would be zero, making the migration indistinguishable from
    a bug that always returns zero. The synthetic pair isolates exactly
    what this test checks (the migration mechanism, not whether this short
    a spin-up grows clouds) and keeps the donor run itself as cheap as the
    quartet test's.
    """

    DT_MIN = 30
    EVERY = 4  # radiation solves every 4 steps (2 h) -- same cadence as the quartet test
    SCHEMES = ("grey", "rrtmgp", "emulated")

    def _build(self, scheme):
        """Build a model whose radiation term is ``scheme``.

        Same substitution ``TestAlbedoQuartetMigration`` makes: the grey
        case goes through ``idealized_echam_physics()`` since
        ``echam_physics()`` rejects ``radiation_scheme="grey"`` by policy.
        """
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.physics.radiation.radiation_types import RadiationParameters
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        radiation_params = RadiationParameters.default(
            radiation_interval=self.EVERY * self.DT_MIN * 60.0)
        if scheme == "grey":
            physics = idealized_echam_physics(radiation=radiation_params)
        else:
            physics = echam_physics(
                radiation=radiation_params, radiation_scheme=scheme)
        return Model(
            coords=coords, terrain=TerrainData.aquaplanet(coords),
            time_step=self.DT_MIN, physics=physics)

    @staticmethod
    def _radiation_term(scheme):
        """Build the bare radiation ``PhysicsTerm`` for ``scheme``.

        Used to ask its own ``legacy_effective_radii_at_solve`` directly --
        the ground truth this test checks the migration against.
        """
        if scheme == "grey":
            from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
            return GreyTwoStreamRadiation()
        if scheme == "rrtmgp":
            from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
            return RRTMGPRadiation()
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
        return NNEmulatorRadiation()

    @pytest.mark.slow
    def test_migrated_radii_and_the_next_cached_step_both_match(self):
        for scheme in self.SCHEMES:
            with self.subTest(scheme=scheme):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "pre_1135.msgpack"
                    model = self._build(scheme)
                    step_days = self.DT_MIN / 1440.0
                    # Mid-interval, the same offset TestAlbedoQuartetMigration
                    # uses, so the step this test resumes into below is
                    # CACHED (no fresh radiation solve) -- the exact case an
                    # unmigrated zero would leak into.
                    split = self.EVERY + 2
                    model.run(save_interval=step_days, total_time=split * step_days)

                    # Overwrite with synthetic, clearly-distinct constants --
                    # see the class docstring for why the real (all-zero)
                    # values would make this test vacuous. Even for "grey",
                    # which never reads these back, patching them keeps the
                    # setup uniform across schemes; its expected migration
                    # is zero regardless (checked below via its own
                    # ``legacy_effective_radii_at_solve``, not by this
                    # patch).
                    orig_r_eff_liq = model.physics_carry["clouds"].r_eff_liq
                    synthetic_liq = jnp.full(
                        orig_r_eff_liq.shape, 12.5, dtype=orig_r_eff_liq.dtype)
                    synthetic_ice = jnp.full(
                        orig_r_eff_liq.shape, 45.0, dtype=orig_r_eff_liq.dtype)
                    model.physics_carry["clouds"] = model.physics_carry[
                        "clouds"].copy(r_eff_liq=synthetic_liq, r_eff_ice=synthetic_ice)
                    clouds_radii = {
                        "r_eff_liq": np.asarray(synthetic_liq),
                        "r_eff_ice": np.asarray(synthetic_ice),
                    }

                    save_checkpoint(model, path, elapsed_days=split * step_days)
                    payload = _read_payload(path)
                    for name in _EFFECTIVE_RADII_AT_SOLVE_FIELDS:
                        del payload["physics"][name]
                    _write_payload(path, payload)

                    expected = dict(zip(
                        _EFFECTIVE_RADII_AT_SOLVE_FIELDS,
                        self._radiation_term(scheme).legacy_effective_radii_at_solve(
                            clouds_radii["r_eff_liq"], clouds_radii["r_eff_ice"])))

                    # Reload into the SAME model (it already has the
                    # populated final states load_checkpoint requires, from
                    # the run above) rather than a second, freshly built
                    # target -- see the class docstring.
                    with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                        load_checkpoint(model, path)

                    # --- invariant 1: migrated (not seeded) right after load ---
                    generic_seed_lines = [
                        line for line in logs.output
                        if "seeded from a freshly built carry" in line]
                    rad = model.physics_carry["radiation"]
                    for name, want in expected.items():
                        leaf = name.split(".", 1)[1]
                        self.assertFalse(
                            any(name in line for line in generic_seed_lines),
                            f"{scheme}: {name} was seeded from the fresh "
                            f"carry, not migrated: {generic_seed_lines}")
                        self.assertTrue(
                            any(name in line and "derived" in line
                                for line in logs.output),
                            f"{scheme}: {name} was not named on a 'derived' "
                            f"log line: {logs.output}")
                        np.testing.assert_array_equal(
                            np.asarray(getattr(rad, leaf)), want,
                            err_msg=f"{scheme}: {name}")

                    # --- invariant 2: still correct after one CACHED step ---
                    model.resume(save_interval=step_days, total_time=step_days)
                    rad = model.physics_carry["radiation"]
                    for name, want in expected.items():
                        leaf = name.split(".", 1)[1]
                        np.testing.assert_array_equal(
                            np.asarray(getattr(rad, leaf)), want,
                            err_msg=f"{scheme}: {name}")


class TestModRadConEmis2Migration(unittest.TestCase):
    """A checkpoint written before ``mod_radcon.emis2`` existed (#571).

    ``tau2`` (the longwave layer transmissivity) was always carried across
    steps; ``emis2``, its ``-expm1``-based non-cancelling complement, is
    new. A file without it restores through the name-matched carry
    migration (``fill_missing=True``): the generic fresh-carry seed is the
    class default ("fully emissive", ``emis2=1``), which mismatches
    whatever ``tau2`` actually evolved to -- wrong for the one or two steps
    before the next radiation sub-step recomputes both together (Codex
    review, PR #1124 thread 4275798697). The migration instead derives
    ``emis2 = 1 - tau2`` -- the complement relation the field exists to
    carry -- from the RESTORED ``tau2``, via ``_DERIVED_ON_MIGRATION``.
    """

    def _build(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        return Model(coords=coords, terrain=TerrainData.from_coords(coords),
                     physics=speedy_physics())

    def test_pre_571_file_derives_emis2_from_the_restored_tau2(self):
        donor = self._build()
        # One step is enough: set_physics_flags computes shortwave on step
        # 0, which populates tau2/emis2 with evolved, non-default values.
        donor.run(save_interval=1 / 48., total_time=1 / 48.)

        with tempfile.TemporaryDirectory() as tmp:
            full_path = Path(tmp) / "full.msgpack"
            save_checkpoint(donor, full_path, elapsed_days=1 / 48.)
            payload = _read_payload(full_path)

            # Strip emis2, as a file from before #571 lacks it.
            stripped_path = Path(tmp) / "stripped.msgpack"
            del payload["physics"]["_mod_radcon.emis2"]
            payload["physics_fields"]["_mod_radcon"] = [
                name for name in payload["physics_fields"]["_mod_radcon"]
                if name != "emis2"
            ]
            _write_payload(stripped_path, payload)

            derived_target = self._build()
            derived_target.bootstrap_state()
            with self.assertLogs("jcm.checkpoint", level="INFO") as logs:
                load_checkpoint(derived_target, stripped_path)
            self.assertTrue(
                any("_mod_radcon.emis2" in line and "derived" in line
                    for line in logs.output),
                logs.output)

            control_target = self._build()
            control_target.bootstrap_state()
            load_checkpoint(control_target, full_path)

        tau2 = np.asarray(derived_target.physics_carry["_mod_radcon"].tau2)
        emis2 = np.asarray(derived_target.physics_carry["_mod_radcon"].emis2)
        np.testing.assert_array_equal(emis2, 1.0 - tau2)
        # The control (loaded from the unstripped file) carries exactly the
        # tau2/emis2 the donor computed; the derived target's tau2 restores
        # identically (it was never stripped) and its emis2 is the
        # migration's derivation -- both compared below.
        np.testing.assert_array_equal(
            tau2, np.asarray(control_target.physics_carry["_mod_radcon"].tau2))

        # One replay step: SPEEDY's nstrad sub-cycle means step 1 skips the
        # shortwave recompute, so the longwave tendency is formed directly
        # from the carried tau2/emis2 -- this must agree whether emis2 was
        # derived by the migration or literally restored from the file.
        derived_target.resume(save_interval=1 / 48., total_time=1 / 48.)
        control_target.resume(save_interval=1 / 48., total_time=1 / 48.)
        derived_dfabs = np.asarray(
            derived_target.physics_carry["_longwave_rad"].dfabs)
        control_dfabs = np.asarray(
            control_target.physics_carry["_longwave_rad"].dfabs)
        # Not bit-identical: emis2 = 1 - tau2 reconstructs the complement
        # from the already float32-rounded tau2 (the checkpoint's own
        # precision, per the migration's docstring), rather than from the
        # original optical depth -expm1 was built from (#571) -- the exact
        # cancellation #571 avoids when both are computed together, which a
        # migration from a file that never recorded the optical depth
        # cannot fully recover. Measured here: max relative difference
        # ~3.6e-4, comfortably inside this bound.
        np.testing.assert_allclose(derived_dfabs, control_dfabs,
                                   rtol=1e-3, atol=1e-6)


def _quota_exceeded(*_args, **_kwargs):
    raise OSError(errno.EDQUOT, "Disk quota exceeded")


class TestAtomicOpen(unittest.TestCase):
    """``atomic_open`` changes ``path`` only once the new content is whole."""

    def test_success_replaces_and_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            with atomic_open(path) as handle:
                handle.write(b"one")
            with atomic_open(path) as handle:
                handle.write(b"two")
            self.assertEqual(path.read_bytes(), b"two")
            self.assertEqual(sorted(os.listdir(tmp)), ["f"])

    def test_keep_previous_rotates_only_after_the_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            prev = Path(f"{path}.prev")
            with atomic_open(path, keep_previous=True) as handle:
                handle.write(b"one")
            self.assertFalse(prev.exists())     # nothing to rotate yet
            with atomic_open(path, keep_previous=True) as handle:
                handle.write(b"two")
                # Mid-write the live file is still the old one.
                self.assertEqual(path.read_bytes(), b"one")
                self.assertFalse(prev.exists())
            self.assertEqual((path.read_bytes(), prev.read_bytes()),
                             (b"two", b"one"))

    def test_a_failure_in_the_body_leaves_live_and_prev_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            prev = Path(f"{path}.prev")
            path.write_bytes(b"live")
            prev.write_bytes(b"older")
            with self.assertRaises(RuntimeError):
                with atomic_open(path, keep_previous=True) as handle:
                    handle.write(b"half a checkpoin")
                    raise RuntimeError("killed mid-write")
            self.assertEqual((path.read_bytes(), prev.read_bytes()),
                             (b"live", b"older"))
            self.assertEqual(sorted(os.listdir(tmp)), ["f", "f.prev"])

    def test_a_quota_error_at_fsync_leaves_live_and_prev_untouched(self):
        """The Nautilus failure: the volume fills as the new file is flushed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            path.write_bytes(b"live")
            with mock.patch("jcm.checkpoint.os.fsync", _quota_exceeded):
                with self.assertRaises(OSError) as ctx:
                    with atomic_open(path, keep_previous=True) as handle:
                        handle.write(b"new")
            self.assertEqual(ctx.exception.errno, errno.EDQUOT)
            self.assertEqual(path.read_bytes(), b"live")
            self.assertEqual(sorted(os.listdir(tmp)), ["f"])


class TestFailedSaveKeepsThePreviousCheckpoint(unittest.TestCase):
    """``save_checkpoint(keep_previous=True)`` never leaves a run without one."""

    def test_failed_save_leaves_the_loadable_live_checkpoint_and_prev(self):
        model = _build_model()
        model.run(save_interval=1, total_time=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(model, path, keep_previous=True)       # day 1
            model.resume(save_interval=1, total_time=1)
            save_checkpoint(model, path, keep_previous=True)       # day 2
            model.resume(save_interval=1, total_time=1)
            with mock.patch("jcm.checkpoint.os.fsync", _quota_exceeded):
                with self.assertRaises(OSError):
                    save_checkpoint(model, path, keep_previous=True)  # day 3

            self.assertEqual(sorted(os.listdir(tmp)),
                             ["ckpt.msgpack", "ckpt.msgpack.prev"])
            elapsed = {}
            for name in ("ckpt.msgpack", "ckpt.msgpack.prev"):
                target = _build_model()
                target.bootstrap_state()
                elapsed[name] = load_checkpoint(target, Path(tmp) / name)
            self.assertAlmostEqual(elapsed["ckpt.msgpack"], 2.0)
            self.assertAlmostEqual(elapsed["ckpt.msgpack.prev"], 1.0)


class TestUnreadableVersusIncompatible(unittest.TestCase):
    """Only an undecodable file is "unreadable"; a mismatch is not."""

    def _load(self, data: bytes):
        target = _build_model()
        target.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            path.write_bytes(data)
            load_checkpoint(target, path)

    def test_empty_truncated_and_foreign_files_are_unreadable(self):
        donor = _build_model()
        donor.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.msgpack"
            save_checkpoint(donor, good, elapsed_days=0.0)
            blob = good.read_bytes()
        foreign = flax.serialization.msgpack_serialize({"hello": np.zeros(2)})
        for label, data in (("empty", b""), ("truncated", blob[: len(blob) // 2]),
                            ("foreign", foreign)):
            with self.subTest(label), self.assertRaises(CheckpointUnreadableError):
                self._load(data)

    def test_a_readable_mismatch_is_a_plain_value_error(self):
        donor = _build_model(spectral_truncation=31)
        donor.bootstrap_state()
        other_grid = _build_model(spectral_truncation=21)
        other_grid.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.0)
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(other_grid, path)
        self.assertNotIsInstance(ctx.exception, CheckpointUnreadableError)
