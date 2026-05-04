r"""Regression: ECHAM physics on T63L47 hybrid + real terrain stays finite.

Originally a failing reproduction of the surface-scheme runaway over
real orography. The fix landed alongside this file: ``land.py`` and
``sea_ice.py`` now use the same ``surface - atmosphere`` flux convention
as ``ocean.py`` and as ``apply_surface``'s positive-upward formula. The
old ``atmosphere - surface`` convention produced a positive-feedback
runaway over high-altitude land (NaN by step 34 on T63L47) — see the
commit message for the bisection trail.

T63L47 hybrid is too heavy to compile on CPU within the regular test
budget, so this module is gated behind ``JCM_RUN_GPU_INTEGRATION_TESTS=1``
and only meaningful when an accelerator is available.

Run with::

    JCM_RUN_GPU_INTEGRATION_TESTS=1 CUDA_VISIBLE_DEVICES=4 \\
      pytest jcm/physics/echam/echam_t63_land_repro_test.py -v
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import jax.numpy as jnp
import pytest

from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.runners import inject_balanced_isothermal_profile
from jcm.terrain import TerrainData
from jcm.utils import get_coords


_T63_BC_DIR = Path("jcm/data/bc/t63")
_REQUIRED_FILES = ["terrain.nc", "forcing.nc"]
_GPU_ENV = "JCM_RUN_GPU_INTEGRATION_TESTS"


def _t63l47_coords():
    return get_coords(get_echam_levels(47), spectral_truncation=63)


def _state_is_finite(state) -> bool:
    fields = (
        state.vorticity, state.divergence, state.temperature_variation,
        state.log_surface_pressure,
    )
    return all(bool(jnp.isfinite(f).all()) for f in fields)


def _run_steps(physics, terrain, forcing, n_steps: int):
    coords = _t63l47_coords()
    model = Model(
        coords=coords, terrain=terrain, physics=physics, time_step=12,
    )
    model._final_modal_state = model._prepare_initial_modal_state()
    inject_balanced_isothermal_profile(model)
    dt_days = 12.0 / (60.0 * 24.0) * n_steps
    model.resume(forcing=forcing, save_interval=dt_days, total_time=dt_days)
    return model._final_modal_state


def _gpu_required():
    if os.environ.get(_GPU_ENV) != "1":
        pytest.skip(f"set {_GPU_ENV}=1 to run; T63L47 is too heavy for CPU CI")
    for fname in _REQUIRED_FILES:
        if not (_T63_BC_DIR / fname).exists():
            pytest.skip(
                f"{_T63_BC_DIR / fname} missing; run "
                f"utils/convert_echam_bc.py to generate it"
            )


@pytest.mark.slow
class TestEchamLandT63L47Hybrid(unittest.TestCase):
    """Failing T63L47 reproduction. See module docstring for context."""

    def setUp(self):
        _gpu_required()
        self.terrain_real = TerrainData.from_file(
            _T63_BC_DIR / "terrain.nc", coords=_t63l47_coords(),
        )
        self.terrain_aqua = TerrainData.aquaplanet(_t63l47_coords())
        self.forcing = ForcingData.from_file(
            _T63_BC_DIR / "forcing.nc", coords=_t63l47_coords(),
        )

    def test_aquaplanet_t63l47_baseline(self):
        """T63L47 hybrid aquaplanet must pass (control)."""
        final = _run_steps(
            echam_physics(radiation_scheme="grey"),
            self.terrain_aqua, self.forcing, n_steps=40,
        )
        self.assertTrue(_state_is_finite(final))

    def test_real_terrain_does_not_nan_at_step60(self):
        """Failing reproduction: T63L47 + real terrain NaNs by step 34.

        Once the underlying issue is fixed this test should start passing
        with no other change. 60 steps comfortably covers the observed
        blow-up window so a fix that delays (rather than resolves) the
        instability won't accidentally make the test pass.
        """
        final = _run_steps(
            echam_physics(radiation_scheme="grey"),
            self.terrain_real, self.forcing, n_steps=60,
        )
        self.assertTrue(_state_is_finite(final))

    def test_real_terrain_minus_surface_survives(self):
        """Removing the ``surface`` term lets the run survive — confirms
        the surface scheme is the failing component.
        """
        physics = echam_physics(radiation_scheme="grey").remove("surface")
        final = _run_steps(physics, self.terrain_real, self.forcing, n_steps=60)
        self.assertTrue(_state_is_finite(final))

    def test_real_orog_zero_fmask_does_not_nan(self):
        """Even with ``fmask=0`` (no land tiles), real orography + surface
        scheme blows up — currently between steps 60 and 120. Documents
        that the failure is orog × surface, not the land-tile fluxes
        themselves.

        Currently failing for the same root cause as
        ``test_real_terrain_does_not_nan_at_step60``; both should start
        passing together when the underlying issue is fixed.
        """
        terrain_no_land = self.terrain_real.copy(
            fmask=jnp.zeros_like(self.terrain_real.fmask),
        )
        final = _run_steps(
            echam_physics(radiation_scheme="grey"),
            terrain_no_land, self.forcing, n_steps=120,
        )
        self.assertTrue(_state_is_finite(final))


if __name__ == "__main__":
    unittest.main()
