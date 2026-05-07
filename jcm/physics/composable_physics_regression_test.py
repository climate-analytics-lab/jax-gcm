"""Reference-trajectory regression tests for the composable physics factories.

These tests act as the safety net for the multi-phase refactor that moves
ECHAM terms out of the ``apply_*`` wrapper layer in ``echam_physics.py``
into scheme-named ``PhysicsTerm`` classes. Each refactor PR must keep
these trajectories bit-exact (or document why a numerical change is
intentional and regenerate the reference).

The references are small ``.npz`` files in
``jcm/data/test/composable_physics_regression/`` that capture the final
saved state of a 1-day aquaplanet run at low resolution (T31L8 for ECHAM,
T21L8 for SPEEDY). They are tiny — a few hundred kilobytes — and exercise
every term in the default factory at least once.

To regenerate after an intentional numerical change, run::

    REGENERATE=1 JAX_PLATFORMS=cpu pytest jcm/physics/composable_physics_regression_test.py -v

then commit the updated ``.npz`` files. The test will skip with a message
on regeneration so it is impossible to silently overwrite the safety net.

For changes that touch numerics at production resolution, also run a full
T63L47 5-day ECHAM integration on a GPU before merging — see the
verification section of the refactor design doc.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import pytest

from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.utils import get_coords


REFERENCE_DIR = (
    Path(__file__).parent.parent / "data" / "test"
    / "composable_physics_regression"
)


def _final_state_arrays(predictions):
    """Return the final-step nodal arrays we freeze as the reference fingerprint.

    We freeze the full final-step state of each prognostic field rather than
    a summary statistic, so that a refactor that perturbs *any* term in *any*
    cell shows up as a per-element diff. At T31L8 / T21L8 the total payload is
    well under 1 MB.
    """
    dyn = predictions.dynamics
    arrays = {
        "u_wind": np.asarray(dyn.u_wind[-1]),
        "v_wind": np.asarray(dyn.v_wind[-1]),
        "temperature": np.asarray(dyn.temperature[-1]),
        "specific_humidity": np.asarray(dyn.specific_humidity[-1]),
        "normalized_surface_pressure": np.asarray(
            dyn.normalized_surface_pressure[-1],
        ),
    }
    for name, tracer in dyn.tracers.items():
        arrays[f"tracer_{name}"] = np.asarray(tracer[-1])
    return arrays


def _check_or_regenerate(name: str, actual_arrays: dict) -> None:
    """Assert ``actual_arrays`` matches the on-disk reference, or regenerate.

    Behaviour:
      - If ``REGENERATE=1`` in the environment, or the reference file does
        not yet exist, write the current arrays as the new reference and
        skip the test with a message that points at the file path.
      - Otherwise, load the reference and assert per-element bit-equality
        for every field.

    The skip-on-write behaviour means a missing reference is loud — the
    test never silently auto-blesses output.
    """
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCE_DIR / f"{name}.npz"

    if os.environ.get("REGENERATE") == "1" or not path.exists():
        np.savez(path, **actual_arrays)
        pytest.skip(
            f"Wrote reference at {path}; commit and rerun without "
            "REGENERATE=1 to verify against it.",
        )

    expected = np.load(path)
    expected_keys = set(expected.files)
    actual_keys = set(actual_arrays.keys())
    if expected_keys != actual_keys:
        raise AssertionError(
            f"Reference {path.name} field set drift: "
            f"only-in-reference={expected_keys - actual_keys}, "
            f"only-in-actual={actual_keys - expected_keys}. "
            "If this is intentional, regenerate with REGENERATE=1.",
        )
    for key in actual_arrays:
        np.testing.assert_array_equal(
            actual_arrays[key], expected[key],
            err_msg=(
                f"Reference mismatch in {name!r} field {key!r}: "
                "the refactor changed numerics. If intentional, regenerate "
                "with REGENERATE=1 and commit."
            ),
        )


class TestEchamReferenceTrajectory(unittest.TestCase):
    """T31L8 aquaplanet 1-day ECHAM reference trajectory."""

    @pytest.mark.slow
    def test_echam_default_reference(self):
        """Bit-exact match for ``echam_physics()`` defaults."""
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        sigma_boundaries = np.linspace(0, 1, 9)  # 8 levels
        coords = get_coords(sigma_boundaries, nodal_shape=(64, 32))
        terrain = TerrainData.aquaplanet(coords)
        forcing = ForcingData.zeros((64, 32))

        model = Model(
            coords=coords,
            terrain=terrain,
            physics=echam_physics(),
        )
        preds = model.run(
            forcing=forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        _check_or_regenerate(
            "echam_t31l8_1day", _final_state_arrays(preds),
        )


class TestSpeedyReferenceTrajectory(unittest.TestCase):
    """T21L8 aquaplanet 1-day SPEEDY reference trajectory."""

    @pytest.mark.slow
    def test_speedy_default_reference(self):
        """Bit-exact match for ``speedy_physics()`` defaults."""
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        forcing = ForcingData.zeros(coords.horizontal.nodal_shape)

        model = Model(
            coords=coords,
            terrain=terrain,
            physics=speedy_physics(),
        )
        preds = model.run(
            forcing=forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        _check_or_regenerate(
            "speedy_t21l8_1day", _final_state_arrays(preds),
        )


if __name__ == "__main__":
    unittest.main()
