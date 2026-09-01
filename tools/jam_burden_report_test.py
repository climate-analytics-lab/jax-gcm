"""Tests for the JAM burden report's layer-Δp helper.

Only ``_layer_dp`` is covered here: it is the piece with a real correctness
choice (prefer the model's own ``pressure_thickness`` diagnostic, fall back to
differencing ``pressure_half``), and the one that #710-era orientation bugs hid
in. Synthetic Datasets keep the test GPU-free and model-free; ``tools/`` is
outside the ``jcm`` coverage target, so this stays deliberately light.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import jam_burden_report as jbr  # noqa: E402


def _base_vars(dp_level):
    # A surface-first interface profile whose -diff equals dp_level, so the two
    # code paths have a known common answer.
    ph = np.concatenate([[1000.0], 1000.0 - np.cumsum(dp_level)])
    return ph


def test_prefers_pressure_thickness_when_present():
    dp_level = np.array([100.0, 150.0, 250.0])
    ph = _base_vars(dp_level)
    ds = xr.Dataset(
        {
            "pressure_thickness": (("time", "level"), dp_level[None]),
            "pressure_half": (("time", "level_i"), ph[None]),
        }
    )
    out = jbr._layer_dp(ds)
    # Must return the diagnostic itself (on the level axis), not the diff.
    assert "level" in out.dims
    np.testing.assert_allclose(np.asarray(out), dp_level)


def test_falls_back_to_diff_of_pressure_half():
    dp_level = np.array([100.0, 150.0, 250.0])
    ph = _base_vars(dp_level)
    ds = xr.Dataset({"pressure_half": (("time", "level_i"), ph[None])})
    out = jbr._layer_dp(ds)
    assert "level" in out.dims
    np.testing.assert_allclose(np.asarray(out), dp_level)


def test_both_paths_agree():
    dp_level = np.array([80.0, 120.0, 300.0, 400.0])
    ph = _base_vars(dp_level)
    with_diag = xr.Dataset(
        {
            "pressure_thickness": (("time", "level"), dp_level[None]),
            "pressure_half": (("time", "level_i"), ph[None]),
        }
    )
    without = xr.Dataset({"pressure_half": (("time", "level_i"), ph[None])})
    np.testing.assert_allclose(
        np.asarray(jbr._layer_dp(with_diag)),
        np.asarray(jbr._layer_dp(without)),
    )
