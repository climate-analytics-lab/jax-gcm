"""Tests for ``jcm/checkpoint.py``.

Covers two contracts:

1. ``save_checkpoint`` → fresh ``Model`` → ``bootstrap_state`` →
   ``load_checkpoint`` reproduces the original state pytrees
   element-wise (round-trip fidelity).
2. A continuous N-day integration matches a ``(N/2 days, checkpoint,
   load on a fresh Model, N/2 days)`` split to numerical roundoff
   (resumption equivalence — the real use case from issue #128).

Uses Held-Suarez physics for speed: no moisture, no radiation, deterministic
forcing. ``setUpClass`` builds the model once per class so JAX compile cost is
paid a single time across the suite.
"""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

from jcm.checkpoint import load_checkpoint, save_checkpoint
from jcm.model import Model
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.held_suarez.utils import get_held_suarez_coords
from jcm.terrain import TerrainData


def _build_model() -> Model:
    coords = get_held_suarez_coords()
    return Model(
        coords=coords,
        terrain=TerrainData.from_coords(coords),
        time_step=180,
        physics=held_suarez_physics(),
    )


def _max_abs_diff(tree_a, tree_b) -> float:
    diffs = jax.tree.leaves(
        jax.tree.map(lambda a, b: jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))),
                     tree_a, tree_b)
    )
    return float(max(float(d) for d in diffs)) if diffs else 0.0


class TestCheckpointRoundTrip(unittest.TestCase):

    def test_save_load_reproduces_state(self):
        model = _build_model()
        model.run(save_interval=1, total_time=2)
        dycore_before = model._final_dycore_state
        physics_before = model._final_physics_state

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(model, path, elapsed_days=2.0)
            self.assertTrue(path.exists())

            fresh = _build_model()
            fresh.bootstrap_state()
            elapsed = load_checkpoint(fresh, path)

        self.assertAlmostEqual(elapsed, 2.0)
        self.assertEqual(_max_abs_diff(dycore_before, fresh._final_dycore_state), 0.0)
        self.assertEqual(_max_abs_diff(physics_before, fresh._final_physics_state), 0.0)

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
        baseline_dycore = baseline._final_dycore_state
        baseline_physics = baseline._final_physics_state

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

        modal_diff = _max_abs_diff(baseline_dycore, second_half._final_dycore_state)
        physics_diff = _max_abs_diff(baseline_physics, second_half._final_physics_state)

        # Equivalence is exact in the absence of host-side RNG: Held-
        # Suarez and the dynamical core are deterministic given the same
        # state and forcing. Allow only float32 accumulation noise.
        self.assertLess(modal_diff, 1e-5, f"modal state diverged by {modal_diff}")
        self.assertLess(physics_diff, 1e-5, f"physics state diverged by {physics_diff}")


if __name__ == "__main__":
    unittest.main()
