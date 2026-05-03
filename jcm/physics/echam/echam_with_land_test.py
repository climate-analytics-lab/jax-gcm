"""Regression: ECHAM physics integration over real terrain.

PR #455 documented that switching ``terrain=aquaplanet`` to a file-loaded
real terrain breaks the ECHAM model.

The failure does NOT manifest at small grid + sigma + isothermal init —
T30L8 sigma + real T30 terrain runs cleanly through ``echam_physics()``.
It also does not manifest at T31L47 hybrid + real T30 terrain. The
production failure is at T63L47 hybrid + real T63 terrain +
``balanced_isothermal`` init, where the model NaNs at step 34
(~6.8 simulated hours) — slow enough to integrate forward, fast enough
that "land breaks very fast" is a fair description.

Bisection results (T63 GPU, see ``utils/scratch/echam_land_bisect/``):

- All physics terms removed one-by-one. Removing ``surface`` (and only
  ``surface``) makes the run survive 1+ days.
- Within the surface step, zeroing the heat / moisture tendencies makes
  the run survive 60 steps; zeroing only the momentum drag makes the
  failure HAPPEN SOONER (step 27 vs step 34) — wind drag is mildly
  stabilizing, so the destabilizing piece is the bottom-level
  thermodynamic tendencies.
- Zeroing the land-tile fraction (``fmask=0``) while keeping real
  orography STILL NaNs. Conversely, real ``fmask`` with zeroed
  orography survives. So the failure is **orog × surface**, not land
  tiles per se.

The most likely root cause: the bottom-layer ``temp_tend_sfc`` /
``qv_tend_sfc`` (``apply_surface`` in ``jcm.physics.echam.echam_physics``)
have small but spatially noisy values over orography that excite a
numerically-unstable spectral mode at T63 hybrid resolution. First-step
magnitudes are ~1e-4 K/s — physically tiny — but the dynamics integrate
them into a global blow-up over ~30 steps.

The two tests in this file are the cheap-CPU bisection harness:
``test_aquaplanet_baseline`` (control, must always pass) and
``test_real_terrain_at_t31l47_hybrid`` (currently passing — confirms
that T31L47 is too coarse to reproduce). The actual failing T63L47
reproduction is in ``echam_t63_land_repro_test.py`` (slow + GPU only).
"""
from __future__ import annotations

import unittest
from importlib import resources
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


def _t31l47_coords():
    """T31 spectral × 47-level ECHAM hybrid grid."""
    return get_coords(get_echam_levels(47), spectral_truncation=31)


def _t31l47_terrain_real():
    data_dir = resources.files("jcm.data.bc.t30.clim")
    return TerrainData.from_file(
        Path(data_dir / "terrain.nc"), coords=_t31l47_coords(),
    )


def _t31l47_terrain_aqua():
    return TerrainData.aquaplanet(_t31l47_coords())


def _t31l47_forcing():
    data_dir = resources.files("jcm.data.bc.t30.clim")
    return ForcingData.from_file(
        Path(data_dir / "forcing.nc"), coords=_t31l47_coords(),
    )


def _state_is_finite(state) -> bool:
    fields = (
        state.vorticity, state.divergence, state.temperature_variation,
        state.log_surface_pressure,
    )
    if not all(bool(jnp.isfinite(f).all()) for f in fields):
        return False
    if hasattr(state, "tracers"):
        for v in state.tracers.values():
            if not bool(jnp.isfinite(v).all()):
                return False
    return True


def _run_one_step(physics, terrain):
    """Build a Model with given physics + terrain, run 1 step, return final state."""
    coords = _t31l47_coords()
    forcing = _t31l47_forcing()
    model = Model(
        coords=coords, terrain=terrain, physics=physics, time_step=12,
    )
    model._final_modal_state = model._prepare_initial_modal_state()
    inject_balanced_isothermal_profile(model)
    dt_days = 12.0 / (60.0 * 24.0)
    model.resume(forcing=forcing, save_interval=dt_days, total_time=dt_days)
    return model._final_modal_state


@pytest.mark.slow
class TestEchamLandT31L47Hybrid(unittest.TestCase):
    """Cheap-CPU control: T31L47 hybrid does NOT reproduce the T63 failure."""

    def test_aquaplanet_baseline(self):
        """Sanity: aquaplanet T31L47 hybrid + balanced_isothermal must
        succeed (otherwise the test harness itself is broken).
        """
        final = _run_one_step(
            echam_physics(radiation_scheme="grey"),
            _t31l47_terrain_aqua(),
        )
        self.assertTrue(_state_is_finite(final))

    def test_real_terrain_at_t31l47_hybrid(self):
        """T31L47 hybrid + real T30 terrain runs cleanly at step 1.

        Documents that this configuration is too coarse to reproduce
        the production T63 failure. The actual failing reproduction is
        in ``echam_t63_land_repro_test.py``.
        """
        final = _run_one_step(
            echam_physics(radiation_scheme="grey"),
            _t31l47_terrain_real(),
        )
        self.assertTrue(_state_is_finite(final))


if __name__ == "__main__":
    unittest.main()
