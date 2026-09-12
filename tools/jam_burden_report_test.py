"""Tests for the JAM burden report's layer-Δp and burden helpers.

These are the two pieces with a real correctness choice: ``_layer_dp`` picks
between the model's own ``pressure_thickness`` diagnostic and differencing
``pressure_half``, and orients the result for both output conventions; and
``burden`` has to find the cloud-borne phase under the namespace the writer
actually emits. Synthetic Datasets keep the test GPU-free and model-free;
``tools/`` is outside the ``jcm`` coverage target, so this stays deliberately
light.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import jam_burden_report as jbr  # noqa: E402
import jcm.constants as c  # noqa: E402


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
    # Must return the diagnostic itself (on the level axis), not the diff; the
    # (length-1) time axis is preserved so Δp can track an evolving ps.
    assert out.dims == ("time", "level")
    np.testing.assert_allclose(np.asarray(out), dp_level[None])


def test_falls_back_to_diff_of_pressure_half():
    dp_level = np.array([100.0, 150.0, 250.0])
    ph = _base_vars(dp_level)
    ds = xr.Dataset({"pressure_half": (("time", "level_i"), ph[None])})
    out = jbr._layer_dp(ds)
    assert out.dims == ("time", "level")
    np.testing.assert_allclose(np.asarray(out), dp_level[None])


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


def test_pre_710_interfaces_are_reversed_to_match_level_fields():
    """Pre-#710 files store interfaces TOA-first, ``level`` fields surface-first.

    The Δp handed back must be on the ``level`` axis' orientation, so a
    surface-heavy layer lines up with the boundary layer, not the stratosphere.
    """
    dp_level = np.array([300.0, 250.0, 150.0, 100.0])   # surface-first
    ph_toa_first = np.concatenate([[0.0], np.cumsum(dp_level[::-1])])
    ds = xr.Dataset({"pressure_half": (("time", "level_i"), ph_toa_first[None])})
    np.testing.assert_allclose(np.asarray(jbr._layer_dp(ds)), dp_level[None])


def test_post_710_interfaces_are_left_alone():
    dp_level = np.array([300.0, 250.0, 150.0, 100.0])
    ph_sfc_first = np.concatenate([[800.0], 800.0 - np.cumsum(dp_level)])
    ds = xr.Dataset(
        {"pressure_half": (("time", "level_i"), ph_sfc_first[None])},
        coords={"level_i": ("level_i", np.linspace(1.0, 0.0, 5),
                            {"positive": "down"})},
    )
    np.testing.assert_allclose(np.asarray(jbr._layer_dp(ds)), dp_level[None])


def _two_phase_dataset():
    """One layer, one column: interstitial 2 kg/kg + cloud-borne 3 kg/kg."""
    return xr.Dataset(
        {
            "pressure_thickness": (("time", "level", "lat"),
                                   np.full((1, 1, 1), float(c.grav))),
            "m_so4_acc": (("time", "level", "lat"), np.full((1, 1, 1), 2.0)),
            "jam_cloud_borne.mc_so4_acc": (("time", "level", "lat"),
                                           np.full((1, 1, 1), 3.0)),
        },
        coords={"lat": [0.0]},
    )


def test_burden_includes_the_cloud_borne_namespace():
    # dp/g = 1 kg/m² of air, so the burden is (2 + 3) kg/m² -> 5e6 mg/m².
    col = jbr.burden(_two_phase_dataset(), "so4", ("acc",))
    np.testing.assert_allclose(np.asarray(col), 5.0e6)


def test_burden_omitting_cloud_borne_would_be_lower():
    """Guards the regression: interstitial alone is a different number."""
    ds = _two_phase_dataset().drop_vars("jam_cloud_borne.mc_so4_acc")
    np.testing.assert_allclose(np.asarray(jbr.burden(ds, "so4", ("acc",))),
                               2.0e6)


def test_burden_is_none_without_tracers():
    assert jbr.burden(_two_phase_dataset(), "du", ("acc",)) is None
