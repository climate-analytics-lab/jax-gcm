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


def test_species_table_covers_the_m7_population():
    """An M7 run's tracers (HAM's modes, and its ``oc``) reach every burden."""
    from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC

    tracers = {(sp, mode.short) for mode in M7_SPEC.modes
               for sp in mode.species}
    for sp, mode in tracers:
        assert sp in jbr._SPECIES, sp
        assert mode in jbr._SPECIES[sp][0], (sp, mode)
    one = np.full((1, 1, 1), 1.0)
    ds = xr.Dataset(
        {"pressure_thickness": (("time", "level", "lat"), one * float(c.grav)),
         **{f"m_du_{m}": (("time", "level", "lat"), one)
            for m in ("as", "cs", "ai", "ci")}},
        coords={"lat": [0.0]})
    # Four 1 kg/kg dust tracers over 1 kg/m² of air -> 4e6 mg/m².
    np.testing.assert_allclose(
        np.asarray(jbr.burden(ds, "du", jbr._SPECIES["du"][0])), 4.0e6)


def _so4_dataset(*modes):
    one = np.full((1, 1, 1), 1.0)
    return xr.Dataset(
        {"pressure_thickness": (("time", "level", "lat"), one * float(c.grav)),
         **{f"m_so4_{m}": (("time", "level", "lat"), one) for m in modes}},
        coords={"lat": [0.0]})


def test_sulfate_molar_mass_is_the_populations_own():
    """MAM4's ``so4`` is NH4HSO4, M7's is SO4: the tracers say which."""
    from jcm.physics.aerosol.jam.microphysics import M7_SPEC, MAM4_SPEC

    mam4 = MAM4_SPEC.species_props("so4").molar_mass
    m7 = M7_SPEC.species_props("so4").molar_mass
    assert mam4 == 0.115 and m7 == 0.0960631
    for mode in MAM4_SPEC.classes_for("so4"):
        assert jbr.sulfate_molar_mass(_so4_dataset(mode.short)) == mam4
    for mode in M7_SPEC.classes_for("so4"):
        assert jbr.sulfate_molar_mass(_so4_dataset(mode.short)) == m7
    np.testing.assert_allclose(jbr.sulfate_per_so2(_so4_dataset("as")),
                               0.0960631 / 0.0640648)


def test_sulfate_molar_mass_is_none_without_one_population():
    """No sulfate tracer, or two populations' at once: nothing to convert to."""
    assert jbr.sulfate_molar_mass(_so4_dataset()) is None
    assert jbr.sulfate_molar_mass(_so4_dataset("acc", "as")) is None
    assert jbr.sulfate_per_so2(_so4_dataset("acc", "as")) is None


def test_the_populations_sulfate_modes_are_disjoint():
    """What lets a file's ``m_so4_<mode>`` tracers name their population."""
    seen = {}
    for spec in jbr._POPULATIONS:
        for mode in spec.classes_for("so4"):
            assert mode.short not in seen, (mode.short, seen)
            seen[mode.short] = spec
