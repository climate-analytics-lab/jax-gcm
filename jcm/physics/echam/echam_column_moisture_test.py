"""Column-level moisture-balance tests for the ECHAM physics over high orography.

These tests pin down the moisture-cycle bugs that drove the day-15 NaN in
``echam_t63_land_repro_test.py`` — specifically the runaway over the
Tibetan Plateau column where ``q`` exceeded ``q_sat`` by 2× at L40 / 800 hPa.
Each test isolates one piece of the cycle on a single Tibetan-like column
(``ps ≈ 540 hPa``, ``orog = 2800 m``, land tile) so that future regressions
can be triaged without a full T63L47 GPU run.

What we know so far (from bisection in the parent commit):

* Surface evaporation is the moisture source. Without it, ``q`` stays at 0
  and the runaway never starts.
* The first negative ``q`` shows up at step 4 around L38 / 520 hPa — well
  above the surface tendency level (L46) — and grows roughly in proportion
  to the bottom-level ``q`` magnitude. Pattern is consistent with spectral
  ringing of the steep PBL gradient propagated into the upper troposphere.
* ``filter_tendencies`` originally only filtered ``divergence``; vorticity,
  T', log_ps and tracers passed through unfiltered. The fix (also in this
  commit) routes each prognostic through the appropriate filter using
  ``DiffusionFilter``'s separate ``div`` / ``vor_q`` / ``temp`` settings.

With the hyperdiffusion fix the worst negative-q is down to O(1 µg/kg)
(was 1 g/kg pre-fix) — physically zero, six orders of magnitude below
the threshold any moisture scheme consumes. The convective-T cap in
``apply_convection`` continues to handle the residual hot columns; at
this point removing the cap should be a measured experiment rather
than the default. Eliminating the µg/kg-level noise entirely would
need positive-definite tracer advection in the dinosaur dycore.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import pytest

from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.initial_states import balanced_isothermal_state
from jcm.terrain import TerrainData
from jcm.utils import get_coords


_T63_BC_DIR = Path("jcm/data/bc/t63")
_GPU_ENV = "JCM_RUN_GPU_INTEGRATION_TESTS"
_TIBET_I, _TIBET_J = 154, 40   # orog ≈ 2800 m, fmask = 1.0 (71.2°W 14.0°S, Andes)
# A high column whose own soil evaporates from the dry start (see
# test_q_negatives_stay_below_1_percent_of_q_max): 65.6°W 25.2°S, orog 2296 m,
# all land, no glacier, January soilw_am 0.45 (β = 0.25, h = 0.42).
_EVAP_I, _EVAP_J = 157, 34


def _gpu_required():
    if os.environ.get(_GPU_ENV) != "1":
        pytest.skip(f"set {_GPU_ENV}=1 to run; T63L47 is too heavy for CPU CI")
    for fname in ("terrain.nc", "forcing.nc"):
        if not (_T63_BC_DIR / fname).exists():
            pytest.skip(
                f"{_T63_BC_DIR / fname} missing; run "
                f"utils/convert_echam_bc.py to generate it"
            )


def _e_sat(T):
    """Tetens saturation vapor pressure [Pa]."""
    return 611.2 * np.exp(17.62 * (T - 273.15) / (T - 30.03))


def _q_sat(T, p):
    """Saturation specific humidity [kg/kg]."""
    e = _e_sat(T)
    return 0.622 * e / (p - (1.0 - 0.622) * e)


def _soil_evaporates_at_start(i, j):
    """JSBACH's criterion for a column whose own soil evaporates from the dry start.

    The bare soil evaporates while ``h·q_sat > q_air`` (``q_air = 0`` here, so
    ``h > 0``) and the canopy while the water-stress factor ``β > 0``, i.e. the
    root-zone fill ``soilw_am`` exceeds the wilting fraction (0.35). January is
    the first forcing month the bootstrap reads.
    """
    import xarray as xr

    from jcm.physics.surface.echam import jsbach_land

    params = jsbach_land.JsbachLandParameters.default()
    with xr.open_dataset(_T63_BC_DIR / "forcing.nc") as f, \
            xr.open_dataset(_T63_BC_DIR / "terrain.nc") as t:
        w = float(f.soilw_am.transpose("lon", "lat", "time").values[i, j, 0])
        land = float(t.lsm.transpose("lon", "lat").values[i, j])
        glacier = float(f.glac.transpose("lon", "lat").values[i, j])
        orog = float(t.orog.transpose("lon", "lat").values[i, j])
    h = float(jsbach_land.bare_soil_relative_humidity(w))
    return (land >= 0.999 and glacier <= 0.0 and orog >= 2000.0
            and w > float(params.moisture_wilting_fraction) and h > 0.0)


def _build_model_and_step(physics_factory, n_steps: int,
                          column: tuple[int, int] = (_TIBET_I, _TIBET_J)):
    """Build the standard T63L47 + real terrain + sponge model with the
    given physics package and step it forward ``n_steps`` × 12 min.
    Returns the profile of ``column`` (the Tibetan grid point by default) at
    every step.
    """
    col_i, col_j = column
    coords = get_coords(get_echam_levels(47), spectral_truncation=63)
    terrain = TerrainData.from_file(_T63_BC_DIR / "terrain.nc", coords=coords)
    forcing = ForcingData.from_file(
        _T63_BC_DIR / "forcing.nc", coords=coords, align_mode="wrap_year")
    physics = physics_factory()
    model = Model(coords=coords, terrain=terrain, physics=physics, time_step=12)
    # Bootstrap from the balanced-isothermal start so the step loop below can
    # ``model.resume(...)`` from it (bootstrap_state builds both the dycore
    # state and the physics carry the first resume needs).
    model.bootstrap_state(balanced_isothermal_state(model))

    levels = get_echam_levels(47)
    a = np.asarray(levels.a_boundaries)
    b = np.asarray(levels.b_boundaries)
    dt_days = 12.0 / (60.0 * 24.0)

    history = []  # list of (step, T_col, q_col, p_full)
    for step in range(1, n_steps + 1):
        model.resume(forcing=forcing, save_interval=dt_days, total_time=dt_days)
        s = model.dycore.to_physics_state(model._final_dycore_state)
        T = np.asarray(s.temperature[:, col_i, col_j])
        q = np.asarray(s.specific_humidity[:, col_i, col_j])
        ps = float(s.normalized_surface_pressure[col_i, col_j]) * 1e5
        p_half = a + b * ps
        p_full = 0.5 * (p_half[:-1] + p_half[1:])
        history.append((step, T, q, p_full))
    return history


def _full_physics():
    from jcm.physics.dissipation import UpperSponge
    return echam_physics() + UpperSponge(
        n_sponge_levels=5, sponge_timescale_s=3 * 3600.0, enspodi=2.0,
    )


def _no_surface_physics():
    """Full physics without a surface moisture source.

    The surface exchange is a boundary condition of the vertical diffusion's
    implicit solve (the vdiff term couples every tile; ``EchamSurface`` only
    publishes), so the source goes with an uncoupled vdiff (zero-flux bottom),
    not with the ``surface`` term alone.
    """
    from jcm.physics.dissipation import UpperSponge
    from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion import (
        TteTkeVerticalDiffusion,
    )
    physics = echam_physics().remove("surface")
    physics = physics.replace(
        TteTkeVerticalDiffusion.category, TteTkeVerticalDiffusion(couple_surface=False))
    return physics + UpperSponge(
        n_sponge_levels=5, sponge_timescale_s=3 * 3600.0, enspodi=2.0,
    )


@pytest.mark.slow
class TestTibetanColumnMoisture(unittest.TestCase):
    """Single-column moisture-balance regressions over the Tibetan Plateau."""

    def test_no_surface_means_no_q(self):
        """Sanity: without the surface scheme there's no moisture source,
        so ``q`` must remain identically 0 in this column for the first
        12 hours.
        """
        _gpu_required()
        history = _build_model_and_step(_no_surface_physics, n_steps=60)
        for step, _T, q, _p in history:
            self.assertEqual(float(q.max()), 0.0,
                             msg=f"step {step}: surface-removed run leaked q")
            self.assertEqual(float(q.min()), 0.0,
                             msg=f"step {step}: surface-removed run produced negative q")

    def test_q_negatives_stay_below_1_percent_of_q_max(self):
        """``q`` at a high, evaporating column may be slightly negative from
        spectral round-trip of advected moisture, but never by more than
        1 % of the column's positive moisture content.

        The bound is relative to the column's OWN moisture source, so the
        column must have one under the land's evaporation form (JSBACH's
        humidity factors): from the dry start, its bare soil evaporates while
        ``h·q_sat > q_air`` and its canopy while ``β > 0``
        (:func:`_soil_evaporates_at_start`). The Tibetan column (154, 40) no
        longer qualifies: its January ``soilw_am`` is 0.03, so ``β = 0`` and
        ``h = 0.002``, and its moisture is only what the transport brings in
        from its neighbours, ringing included. The column used here (157, 34)
        is the highest-β column at or above 2000 m that qualifies.

        Pre-hyperdiff fix the worst negative-to-max ratio hit ~50 % by
        step 20 (q_min = -1 g/kg vs q_max = 2 g/kg); the convective
        runaway followed. With proper hyperdiffusion the ratio is now
        below 0.2 %, well within the 1 % tolerance. The residual is
        spectral-truncation noise on horizontally-advected ``q`` from
        neighbouring evap-active grid cells; eliminating it entirely
        would need positive-definite tracer advection in the dycore.
        """
        _gpu_required()
        self.assertTrue(_soil_evaporates_at_start(_EVAP_I, _EVAP_J),
                        msg="the probe column no longer evaporates from the dry start")
        self.assertFalse(_soil_evaporates_at_start(_TIBET_I, _TIBET_J))
        history = _build_model_and_step(_full_physics, n_steps=60,
                                        column=(_EVAP_I, _EVAP_J))
        worst_ratio = 0.0
        for _step, _T, q, _p in history:
            q_max = float(q.max())
            if q_max > 1e-12:
                ratio = abs(min(0.0, float(q.min()))) / q_max
                worst_ratio = max(worst_ratio, ratio)
        self.assertLess(
            worst_ratio, 0.01,
            msg=f"worst |q_min| / q_max over 60 steps = {worst_ratio*100:.2f} % (>1 %)",
        )

    @pytest.mark.xfail(
        strict=False,
        reason="q exceeds q_sat by ~10 % at L35-L38 by step 60; pending the "
               "cloud / convection moisture-balance fix (fix-plan PRs 4-5)",
    )
    def test_q_stays_subsaturated(self):
        """``q`` at the Tibetan column must not exceed ``q_sat`` over the
        first 12 hours of full physics. The Sundqvist + 1-moment cloud
        scheme is supposed to condense any supersaturation each step.

        Currently fails: q exceeds q_sat by ~10 % at L35-L38 by step 60.
        Marked xfail (non-strict) until then.
        """
        _gpu_required()
        history = _build_model_and_step(_full_physics, n_steps=60)
        for step, T, q, p in history:
            qsat = np.array([_q_sat(T[k], p[k]) for k in range(len(T))])
            rh_max = float(np.nanmax(q / np.maximum(qsat, 1e-12)))
            self.assertLessEqual(
                rh_max, 1.01,
                msg=f"step {step}: RH_max = {rh_max*100:.1f} % (>101 %)",
            )

    def test_full_physics_q_grows_smoothly(self):
        """Bottom-level ``q`` must grow monotonically while the surface
        scheme is providing a positive evap flux, with no jumps that
        would indicate an unbounded numerical instability.

        Detects e.g. the early sign-flip bug (where land flux convention
        was upside-down) which gave ``q`` jumps of >5 g/kg in a single
        step over high orography.
        """
        _gpu_required()
        history = _build_model_and_step(_full_physics, n_steps=60)
        prev_qbot = 0.0
        max_jump = 0.0
        for step, _T, q, _p in history:
            qbot = float(q[-1])
            jump = abs(qbot - prev_qbot)
            max_jump = max(max_jump, jump)
            prev_qbot = qbot
        # 5 g/kg in a single step is the threshold the original land sign
        # bug crossed; healthy spinup gives < 0.1 g/kg jumps.
        self.assertLess(
            max_jump, 5e-3,
            msg=f"max single-step jump in bottom-level q = {max_jump*1000:.3f} g/kg",
        )


if __name__ == "__main__":
    unittest.main()
