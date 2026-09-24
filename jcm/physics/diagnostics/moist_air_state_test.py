"""Tests for MoistAirColumnState.

The term had no tests. Its pressure construction assumed exactly one trailing
horizontal axis, so it worked on a column-vectorized ``(kx, ncols)`` host and
raised on a whole ``(kx, nlon, nlat)`` grid. Every existing caller (the ECHAM
terms, rce.py, Tiedtke-Nordeng) runs column-vectorized, so nothing caught it
until the term was composed onto SPEEDY to supply pressures for the Lott-Miller
SSO drag.

The rule these tests enforce is the one in CLAUDE.md: a column-physics term
must be broadcasting-native, with the vertical on axis 0 and any trailing axes
horizontal, so the identical code runs on a column, a vectorized block, or a
whole grid.
"""

import unittest

import numpy as np

from jcm.physics.diagnostics.moist_air_state import (
    MoistAirColumnState,
    advance_thermo_run,
)
from jcm.physics_interface import PhysicsState
from jcm.utils import get_coords

NLEV = 8


def _coords():
    # Sigma boundaries; get_coords wraps a bare array in SigmaCoordinates,
    # which cache_coords reads as a = 0, b = sigma.
    return get_coords(np.linspace(0.0, 1.0, NLEV + 1), spectral_truncation=21)


def _state(horiz):
    """Build a plausible state with ``horiz`` as the trailing shape."""
    shape = (NLEV,) + tuple(horiz)
    lev = np.linspace(0.1, 1.0, NLEV).reshape((NLEV,) + (1,) * len(horiz))
    return PhysicsState(
        u_wind=np.zeros(shape, dtype=np.float32),
        v_wind=np.zeros(shape, dtype=np.float32),
        temperature=np.broadcast_to(
            200.0 + 90.0 * lev, shape).astype(np.float32),
        specific_humidity=np.broadcast_to(
            5.0 * lev, shape).astype(np.float32),
        # Geopotential decreasing downward, so height_full does too.
        geopotential=np.broadcast_to(
            9.81 * 16000.0 * (1.0 - lev), shape).astype(np.float32),
        normalized_surface_pressure=np.ones(
            tuple(horiz), dtype=np.float32),
    )


def _run(horiz):
    term = MoistAirColumnState()
    coords = _coords()
    term.cache_coords(coords)
    _, diags = term(_state(horiz), {}, None, None)
    return diags


class TestHostLayouts(unittest.TestCase):
    """The same code must run on a vectorized block and on a whole grid."""

    def test_column_vectorized_host(self):
        diags = _run((37,))
        self.assertEqual(diags["pressure_full"].shape, (NLEV, 37))
        self.assertEqual(diags["pressure_half"].shape, (NLEV + 1, 37))

    def test_whole_grid_host(self):
        # The case that used to raise: two horizontal axes.
        diags = _run((12, 6))
        self.assertEqual(diags["pressure_full"].shape, (NLEV, 12, 6))
        self.assertEqual(diags["pressure_half"].shape, (NLEV + 1, 12, 6))

    def test_single_column_host(self):
        # No horizontal axis at all.
        diags = _run(())
        self.assertEqual(diags["pressure_full"].shape, (NLEV,))
        self.assertEqual(diags["pressure_half"].shape, (NLEV + 1,))

    def test_layouts_agree_column_for_column(self):
        # A grid column and a vectorized column with the same inputs must give
        # the same pressures; that equivalence is what "broadcasting-native"
        # buys and what a rank-specific reshape would silently break.
        grid = _run((12, 6))["pressure_full"]
        block = _run((37,))["pressure_full"]
        np.testing.assert_allclose(np.asarray(grid)[:, 0, 0],
                                   np.asarray(block)[:, 0], rtol=1e-6)


class TestDiagnosticValues(unittest.TestCase):
    def setUp(self):
        self.diags = _run((12, 6))

    def test_pressure_increases_downward_and_is_finite(self):
        pf = np.asarray(self.diags["pressure_full"])
        self.assertTrue(np.isfinite(pf).all())
        self.assertTrue((np.diff(pf, axis=0) > 0).all())

    def test_expected_keys_are_published(self):
        for key in ("pressure_full", "pressure_half", "height_full",
                    "height_half", "air_density", "pressure_thickness",
                    "layer_thickness", "surface_pressure"):
            self.assertIn(key, self.diags)

    def test_pressure_thickness_is_positive_dp_summing_to_the_column(self):
        # pressure_thickness is the per-layer Δp [Pa] on the level axis, the
        # weight for mass-integrating a level field. It must be positive, share
        # the mid-level (nlev) shape, and telescope to the column pressure span
        # surface_pressure - pressure_half[top] per column (axis 0 is top-first
        # in the physics frame, so index 0 of the interface axis is the top).
        dp = np.asarray(self.diags["pressure_thickness"])
        ph = np.asarray(self.diags["pressure_half"])
        ps = np.asarray(self.diags["surface_pressure"])
        self.assertEqual(dp.shape, np.asarray(self.diags["pressure_full"]).shape)
        self.assertTrue((dp > 0).all())
        np.testing.assert_allclose(
            dp.sum(axis=0), ps - ph[0], rtol=1e-5, atol=1e-3)

    def test_half_levels_bracket_full_levels(self):
        pf = np.asarray(self.diags["pressure_full"])
        ph = np.asarray(self.diags["pressure_half"])
        self.assertEqual(ph.shape[0], pf.shape[0] + 1)
        self.assertTrue((ph[:-1] <= pf + 1e-3).all())
        self.assertTrue((ph[1:] >= pf - 1e-3).all())

    def test_layer_thickness_respects_its_floor(self):
        # The 10 m floor is a documented numerical-stability clamp.
        lt = np.asarray(self.diags["layer_thickness"])
        self.assertTrue(np.isfinite(lt).all())
        self.assertTrue((lt >= 10.0 - 1e-6).all())


class TestAdvanceThermoRun(unittest.TestCase):
    """Provisional condensate guards stay separate from the final ledger."""

    def test_qc_qi_floor_only_the_returned_diagnostic_view(self):
        qc = np.array([1.0e-5, 2.0e-5], dtype=np.float32)
        qi = np.array([3.0e-5, 4.0e-5], dtype=np.float32)
        diagnostics = {
            "thermo_run": {
                "temperature": np.array([280.0, 270.0], dtype=np.float32),
                "specific_humidity": np.array([1.0e-3, 2.0e-3], dtype=np.float32),
                "qc": qc,
                "qi": qi,
            },
        }
        updated = advance_thermo_run(
            diagnostics,
            10.0,
            d_qc=np.array([-1.0, 0.0], dtype=np.float32),
            d_qi=np.array([0.0, -1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(updated["thermo_run"]["qc"]),
            np.array([0.0, 2.0e-5], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(updated["thermo_run"]["qi"]),
            np.array([3.0e-5, 0.0], dtype=np.float32),
        )
        # The input is immutable and no tendency is returned or rewritten;
        # final prognostic accounting belongs to the interface cap.
        np.testing.assert_array_equal(diagnostics["thermo_run"]["qc"], qc)
        np.testing.assert_array_equal(diagnostics["thermo_run"]["qi"], qi)


if __name__ == "__main__":
    unittest.main()


class TestOneRelativeHumidity(unittest.TestCase):
    """The published ``relative_humidity`` is the documented water-RH (#615).

    The Sundqvist cover computes its own ``q/qsat`` with ECHAM's binary ice
    switch, which jumps by tens of percent across the cloud-ice threshold in
    adjacent cells; it once overwrote this key after MoistAirColumnState set
    it, so the saved field's meaning depended on the composition. Pin that it
    no longer can.
    """

    def test_only_the_moist_air_term_provides_it_in_echam_physics(self):
        from jcm.physics.echam.echam_terms import echam_physics

        for kw in ({}, {"cloud_scheme": "2m"}):
            providers = [t.name for t in echam_physics(**kw).terms
                         if "relative_humidity" in t.provides]
            self.assertEqual(providers, [MoistAirColumnState.name], kw)

    def test_published_rh_is_water_rh_after_the_cover_runs(self):
        import jax.numpy as jnp

        from jcm.physics.clouds.sundqvist import SundqvistCloudFraction

        ncols = 2
        state = _state((ncols,))
        # Cold (235 K) upper levels carrying cloud ice in column 1 only: the
        # cover's ice switch fires there and not in column 0.
        qi = np.zeros((NLEV, ncols), dtype=np.float32)
        qi[:3, 1] = 1.0e-5
        state = state.copy(
            specific_humidity=np.full((NLEV, ncols), 1.0e-4, np.float32),
            tracers={"qc": np.zeros_like(qi), "qi": qi},
        )
        moist = MoistAirColumnState()
        moist.cache_coords(_coords())
        _, diags = moist(state, {}, None, None)

        # The documented definition, computed independently: e / e_s,w with
        # the Bolton (1980) water saturation, at every temperature.
        t = np.asarray(state.temperature, np.float64)
        q = np.asarray(state.specific_humidity, np.float64)
        p = np.asarray(diags["pressure_full"], np.float64)
        e = q * p / (0.622 + 0.378 * q)
        es_w = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
        np.testing.assert_allclose(
            np.asarray(diags["relative_humidity"]), e / es_w, rtol=1e-4)

        # Minimal stand-ins for the two fields the cover reads (land mask,
        # sea ice) — the Sc enhancement is irrelevant to this check.
        forcing = type("F", (), {"sice_am": None})()
        terrain = type("T", (), {"fmask": jnp.ones(ncols)})()
        _, after = SundqvistCloudFraction()(state, diags, forcing, terrain)
        np.testing.assert_array_equal(
            np.asarray(after["relative_humidity"]),
            np.asarray(diags["relative_humidity"]))
        # The cover's own humidity is published separately and does differ
        # (ice saturation) in the icy cold cells.
        cover = np.asarray(after["cover_relative_humidity"])
        water = np.asarray(after["relative_humidity"])
        self.assertGreater(float(np.max(cover[:3, 1] / water[:3, 1])), 1.1)
