"""Tight-tolerance refactor regression for the dinosaur dycore.

The cross-hardware regression in ``jcm/physics/composable_physics_regression_test.py``
runs at ``rtol=1e-3, atol=1e-4`` so that CPU↔GPU↔TPU XLA reduction-order drift
doesn't fail the build. That tolerance is fine for catching meaningful science
regressions but is loose enough to absorb a Phase-1 refactor that accidentally
reorders an op.

This file is the **same-host pure-rearrangement** check: a 1-day T21L8 SPEEDY
integration must produce trajectories that match an on-disk baseline at
``rtol=1e-10, atol=1e-12``. The baseline is captured *before* the dycore refactor
lands (Phase 1) so that the refactor PR has a hard guard against drifting the
numerics. Re-run with ``REGENERATE=1`` to refresh the baseline on the host
that wrote it (CPU-only, single-precision XLA backend).

Reference file: ``jcm/data/test/dycore_refactor_baseline/speedy_t21l8_1day.npz``.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import jax
import numpy as np
import pytest

from jcm.forcing import ForcingData
from jcm.terrain import TerrainData


REFERENCE_DIR = (
    Path(__file__).parent.parent.parent
    / "data" / "test" / "dycore_refactor_baseline"
)

# Same-host, same-JAX-build tolerance. If this fails, the refactor is not a
# pure rearrangement — investigate the diff before raising the tolerance.
RTOL = 1e-10
ATOL = 1e-12


def _flatten_leaves(pytree) -> list[np.ndarray]:
    """Flatten a pytree to a numpy-array leaf list (one entry per leaf)."""
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(pytree)]


def _capture_arrays(model, predictions) -> dict[str, np.ndarray]:
    """Bundle the artefacts we freeze as the baseline.

    We freeze three things:

    * ``_final_dycore_state.leaf{i}`` — the dycore-native state at end of run.
      Catches any drift in the integration itself.
    * ``_final_physics_state.leaf{i}`` — the cross-step physics carry. Catches
      any drift in the sub-cycled radiation cache / prior-step TKE / etc.
    * ``predictions.leaf{i}`` — every leaf of the full saved trajectory.
      Catches drift in the gridpoint-bridge projection used at save time.

    Numbered ``leaf{i}`` rather than named because the pytree shapes are stable
    for a given (coords, physics) configuration and the structure-comparison in
    ``_check_or_regenerate`` already fails loudly on key drift.
    """
    arrays: dict[str, np.ndarray] = {}
    final_state = model._final_dycore_state
    for i, leaf in enumerate(_flatten_leaves(final_state)):
        arrays[f"final_state.leaf{i}"] = leaf
    for i, leaf in enumerate(_flatten_leaves(model._final_physics_state)):
        arrays[f"final_physics_state.leaf{i}"] = leaf
    for i, leaf in enumerate(_flatten_leaves(predictions._predictions)):
        arrays[f"trajectory.leaf{i}"] = leaf
    return arrays


def _check_or_regenerate(name: str, actual: dict[str, np.ndarray]) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCE_DIR / f"{name}.npz"

    if os.environ.get("REGENERATE") == "1" or not path.exists():
        np.savez(path, **actual)
        pytest.skip(
            f"Wrote dycore-refactor baseline at {path}; commit and rerun "
            "without REGENERATE=1 to verify against it."
        )

    expected = np.load(path)
    if set(expected.files) != set(actual):
        only_ref = set(expected.files) - set(actual)
        only_act = set(actual) - set(expected.files)
        raise AssertionError(
            f"Pytree shape drift in {name}: only-in-reference={sorted(only_ref)}, "
            f"only-in-actual={sorted(only_act)}. The dycore state pytree changed "
            "structurally; regenerate with REGENERATE=1 if intentional."
        )
    for key in actual:
        np.testing.assert_allclose(
            actual[key], expected[key],
            rtol=RTOL, atol=ATOL,
            err_msg=(
                f"Same-host refactor regression failed on {key!r}. "
                f"The refactor drifted numerics beyond rtol={RTOL}. "
                "Investigate before regenerating."
            ),
        )


def _build_speedy_t21l8_aquaplanet():
    """The single canonical configuration the baseline freezes.

    Kept simple so changes to any default elsewhere don't accidentally
    invalidate the baseline.
    """
    from jcm.model import Model
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics

    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    terrain = TerrainData.aquaplanet(coords)
    forcing = ForcingData.zeros(coords.horizontal.nodal_shape)
    model = Model(coords=coords, terrain=terrain, physics=speedy_physics())
    return model, forcing


class TestDycoreRefactorBaseline(unittest.TestCase):
    """Same-host bit-level guard for the Phase-1 dynamical-core refactor."""

    @pytest.mark.slow
    def test_speedy_t21l8_1day_baseline(self):
        model, forcing = _build_speedy_t21l8_aquaplanet()
        preds = model.run(
            forcing=forcing,
            save_interval=0.25,
            total_time=1.0,
        )
        _check_or_regenerate(
            "speedy_t21l8_1day", _capture_arrays(model, preds),
        )


if __name__ == "__main__":
    unittest.main()
