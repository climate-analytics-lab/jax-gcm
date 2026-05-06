r"""Regression: ECHAM physics on T63L47 hybrid + real terrain stays finite.

Originally a failing reproduction of the surface-scheme runaway over
real orography. The fixes that landed in this branch:

* ``land.py``/``sea_ice.py`` now use the same ``surface - atmosphere``
  positive-upward flux convention as ``ocean.py``. The old
  ``atm - surf`` convention created a positive feedback over cold land
  that NaN'd by step 34.
* ``apply_surface`` damps the explicit bottom-level surface tendencies
  by the implicit-Euler factor ``1 / (1 + K*dt/dz_sfc)``. Over rough
  terrain the ECHAM-tuned exchange coefficients give ``K*dt/dz > 2``,
  which the old explicit step couldn't survive past ~step 600.
* ``apply_surface`` reads ``ocean_temp`` and ``land_temp`` straight
  from forcing instead of routing through the upstream-blended
  ``physics_data.surface.surface_temperature``, which had snapped to
  the dominant tile via ``where(fmask>0.5)``.
* ``ComposableEchamPhysics.__add__`` overrides the parent so that
  ``echam_physics() + UpperSponge(...)`` (the production sponge wiring)
  preserves the type and ``Model.__init__`` still calls
  ``apply_timestep`` on it.

* ``apply_convection`` clips the bottom-level convective T tendency to
  ±5 K/hr. The TN scheme over a 5 km mountain develops a 320 K hot
  spot at ~800 hPa in a single column (Tibetan Plateau), driven by a
  q ~ 38 g/kg supersaturation. The cap is a workaround that prevents
  the column NaN-cascade without distorting the well-behaved 99 % of
  grid points; the underlying moisture-balance question is its own
  follow-up.

With all five fixes the production T63L47 + real terrain + sponge run
is stable for 30 simulated days at dt=12 min on GPU.

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

    def test_real_terrain_stable_for_24h(self):
        """T63L47 + real terrain: 240 steps (1 day at dt=12 min) clean.

        Originally NaN'd at step 34. After the surface fixes the run
        stays finite for ~15 days; we check 1 day here as a fast,
        clearly-past-the-old-failure regression. A multi-day production
        check lives in ``test_real_terrain_stable_for_5_days_with_sponge``.
        """
        final = _run_steps(
            echam_physics(radiation_scheme="grey"),
            self.terrain_real, self.forcing, n_steps=240,
        )
        self.assertTrue(_state_is_finite(final))

    def test_real_terrain_with_sponge_stable_5_days(self):
        """The full production wiring: ECHAM physics + UpperSponge.

        Catches the ``ComposableEchamPhysics + UpperSponge`` regression
        where the parent ``__add__`` returned a plain ``ComposablePhysics``
        and ``Model.__init__`` skipped ``apply_timestep`` — leaving ECHAM
        at the default ``dt_conv = 3600 s`` regardless of model dt.
        Without that fix this test NaNs by step ~95.
        """
        from jcm.physics.dissipation import UpperSponge
        physics = echam_physics(radiation_scheme="grey") + UpperSponge(
            n_sponge_levels=5, sponge_timescale_s=3 * 3600.0, enspodi=2.0,
        )
        final = _run_steps(
            physics, self.terrain_real, self.forcing, n_steps=600,
        )
        self.assertTrue(_state_is_finite(final))

    def test_real_terrain_with_sponge_stable_30_days(self):
        """Production wiring, full month.

        Restored to a hard pass after the real ECHAM land surface
        temperature climatology landed in ``jcm/data/bc/t63``
        (``surf_temp`` from JSBACH IC ``ic_land_soil_T63GR15_*``,
        wired by ``utils/convert_echam_bc.py --land-init``). Earlier
        iterations of this branch were xfail because the bundled BCs
        used ``stl ≈ sst`` extrapolated over land — over the Antarctic
        plateau that gave a ~+50 K bias in winter, driving the day-7
        runaway sensible-heat NaN.
        """
        from jcm.physics.dissipation import UpperSponge
        physics = echam_physics(radiation_scheme="grey") + UpperSponge(
            n_sponge_levels=5, sponge_timescale_s=3 * 3600.0, enspodi=2.0,
        )
        final = _run_steps(
            physics, self.terrain_real, self.forcing, n_steps=30 * 120,
        )
        self.assertTrue(_state_is_finite(final))

    def test_real_terrain_minus_surface_survives(self):
        """Removing the ``surface`` term lets the run survive — control
        from the original bisection (kept for reference even though the
        full physics now also passes the 24h test above).
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
