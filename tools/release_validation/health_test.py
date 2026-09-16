"""Tests for the release-validation health gates (#782).

The subject is *which* cover each field dialect scores and which it only
reports, at two levels: ``cloud_cover_fields`` in isolation, and ``main()``
driven over a directory of synthetic chunks. The second is not redundant —
the helper can be perfectly correct while the call site gates on the column
max, and only an end-to-end run of ``main()`` reads the line the release
report actually carries.

Synthetic Datasets are the only way to know the right answer, and they make
the two failure modes that matter explicit — a radiation-view cover that a run
never saved, and one that is present but identically zero because the
configuration's radiation samples no sub-columns. Neither may be reported as a
measured zero, and the two must not be reported as each other.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import xarray as xr

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import health as H  # noqa: E402

_LAT = np.array([-45.0, 45.0])
_LON = np.array([0.0, 180.0])

# One 4-layer profile per column, chosen so the three definitions separate:
# a contiguous 0.6/0.5 deck, a clear layer, then a 0.4 deck. Max-random cover
# is 1 - (1-0.6)(1-0.4) = 0.76, strictly between the column max (0.6) and
# random overlap (1 - 0.4*0.5*0.6 = 0.88).
_PROFILE = np.array([0.6, 0.5, 0.0, 0.4])
_MAXRANDOM = 0.76
_COLMAX = 0.6


def _levels(values):
    """Broadcast a per-level profile onto (time, level, lat, lon)."""
    return np.broadcast_to(np.asarray(values, float)[None, :, None, None],
                           (1, len(values), len(_LAT), len(_LON))).copy()


def echam_chunk(radiation_cover=None, profile=_PROFILE) -> xr.Dataset:
    """Build a minimal ECHAM-dialect output chunk.

    ``radiation_cover=None`` leaves ``radiation.total_cloud_cover`` out of the
    file entirely (output predating the diagnostic); a float writes it as a
    uniform field, including the 0.0 that grey two-stream publishes.
    """
    data = {
        "clouds.cloud_fraction": (("time", "level", "lat", "lon"),
                                  _levels(profile)),
    }
    if radiation_cover is not None:
        data["radiation.total_cloud_cover"] = (
            ("time", "lat", "lon"),
            np.full((1, len(_LAT), len(_LON)), float(radiation_cover)))
    return xr.Dataset(data, coords={"lat": _LAT, "lon": _LON})


def speedy_chunk() -> xr.Dataset:
    """Build a minimal SPEEDY-dialect chunk (its own 2-D column cover)."""
    return xr.Dataset(
        {"shortwave_rad.cloudc": (("time", "lat", "lon"),
                                  np.full((1, len(_LAT), len(_LON)), 0.55))},
        coords={"lat": _LAT, "lon": _LON})


def _scalar(da) -> float:
    return H.wmean(da, H.area_weights(da))


class TestEchamDialect:
    """The gated cover is ECHAM's aclcov; the other two are context."""

    def test_gated_cover_is_max_random_not_the_column_max(self):
        fields, note = H.cloud_cover_fields(echam_chunk(0.71), speedy=False)
        np.testing.assert_allclose(_scalar(fields["cloud_cover"]), _MAXRANDOM)
        np.testing.assert_allclose(_scalar(fields["cloud_cover_colmax"]),
                                   _COLMAX)
        # The two are genuinely different numbers, which is the whole point:
        # the old gate scored a lower bound on the quantity it named.
        assert _MAXRANDOM > _COLMAX

    def test_radiation_cover_is_reported_when_saved(self):
        fields, note = H.cloud_cover_fields(echam_chunk(0.71), speedy=False)
        np.testing.assert_allclose(_scalar(fields["cloud_cover_radiation"]),
                                   0.71)
        assert note is None

    def test_radiation_cover_is_omitted_when_absent(self):
        fields, note = H.cloud_cover_fields(echam_chunk(None), speedy=False)
        assert "cloud_cover_radiation" not in fields
        # The reason names the *absent* case, not the all-zero one: they are
        # different states of the run and the reader acts on them differently.
        assert "saves no" in note and "b772ffec" in note
        # The gated cover is unaffected by the diagnostic's absence.
        np.testing.assert_allclose(_scalar(fields["cloud_cover"]), _MAXRANDOM)

    def test_identically_zero_radiation_cover_is_omitted_not_reported(self):
        # Grey two-stream writes zeros because it samples no sub-columns.
        # Reporting 0.00 would read as a run with no cloud at all.
        fields, note = H.cloud_cover_fields(echam_chunk(0.0), speedy=False)
        assert "cloud_cover_radiation" not in fields
        assert "identically zero" in note and "grey two-stream" in note

    def test_a_partly_zero_radiation_cover_is_still_a_measurement(self):
        # Clear columns are physical; only an all-zero field is the "this
        # scheme does not produce the diagnostic" signature.
        ds = echam_chunk(0.0)
        ds["radiation.total_cloud_cover"][dict(lat=0)] = 0.8
        fields, note = H.cloud_cover_fields(ds, speedy=False)
        assert "cloud_cover_radiation" in fields
        assert note is None

    def test_cover_is_computed_before_the_time_mean(self):
        # The overlap product is non-linear, so the fields returned must still
        # carry time: averaging cloud_fraction first would give a different
        # (here: larger) answer than averaging the per-step cover.
        clear = np.zeros(len(_PROFILE))
        ds = xr.Dataset(
            {"clouds.cloud_fraction": (
                ("time", "level", "lat", "lon"),
                np.concatenate([_levels(_PROFILE), _levels(clear)], axis=0))},
            coords={"lat": _LAT, "lon": _LON})
        fields, note = H.cloud_cover_fields(ds, speedy=False)
        assert "time" in fields["cloud_cover"].dims
        np.testing.assert_allclose(_scalar(fields["cloud_cover"]),
                                   _MAXRANDOM / 2.0)


class TestSpeedyDialect:
    """SPEEDY publishes its own column cover; nothing to overlap."""

    def test_speedy_scores_its_own_cloudc_and_reports_nothing_else(self):
        fields, note = H.cloud_cover_fields(speedy_chunk(), speedy=True)
        assert set(fields) == {"cloud_cover"}
        np.testing.assert_allclose(_scalar(fields["cloud_cover"]), 0.55)

    def test_speedy_does_not_need_the_echam_cloud_fraction(self):
        # A SPEEDY run saves no clouds.cloud_fraction at all; asking for the
        # ECHAM dialect on one is a KeyError, which is why the caller passes
        # the dialect flag rather than sniffing for either field.
        with pytest.raises(KeyError):
            H.cloud_cover_fields(speedy_chunk(), speedy=False)


def _write_run(tmp_path, radiation_cover=None, days=(30, 60),
               profile=_PROFILE):
    """Write a minimal ECHAM-dialect run directory for ``health.main()``.

    Only the fields the gates touch — enough to reach the cloud block and to
    leave every other gate scoring a defined number.
    """
    for i, day in enumerate(days):
        ds = echam_chunk(radiation_cover, profile=profile)
        ds = ds.assign_coords(time=[np.datetime64("2000-01-01") +
                                    np.timedelta64(30 * i, "D")])
        shape = (1, len(_LAT), len(_LON))
        for name, value in (("radiation.toa_sw_down", 340.0),
                            ("radiation.toa_sw_up", 100.0),
                            ("radiation.toa_lw_up", 240.0),
                            ("clouds.precip_rain", 3.0 / 86400.0),
                            ("clouds.precip_snow", 0.0),
                            ("convection.precip_conv", 0.0)):
            ds[name] = (("time", "lat", "lon"), np.full(shape, value))
        ds["temperature"] = (
            ("time", "level", "lat", "lon"),
            _levels(np.linspace(288.0, 220.0, len(_PROFILE))))
        ds.to_netcdf(tmp_path / f"run_day{day}.nc")
    return str(tmp_path)


def _run_main(tmp_path, monkeypatch, capsys, **kwargs):
    run_dir = _write_run(tmp_path, **kwargs)
    monkeypatch.setattr(sys, "argv", ["health.py", run_dir])
    status = H.main()
    return status, capsys.readouterr().out


class TestGateWiring:
    """What ``main()`` actually gates on — not just what the helper returns.

    Without this, reverting the call site to the column max leaves every other
    test in this file passing, which is exactly the regression the change is
    about.
    """

    def test_the_gate_scores_the_max_random_cover(self, tmp_path, monkeypatch,
                                                  capsys):
        status, out = _run_main(tmp_path, monkeypatch, capsys,
                                radiation_cover=0.71)
        assert f"cloud_cover = {_MAXRANDOM:.2f}" in out
        # ...and specifically NOT the column max, which is also printed but
        # only as context.
        assert f"cloud_cover = {_COLMAX:.2f}" not in out
        assert (f"INFO  cloud_cover_colmax = {_COLMAX:.2f} "
                "(reported, not gated)") in out
        assert "INFO  cloud_cover_radiation = 0.71 (reported, not gated)" in out
        assert "NOTE  the window saves no radiation" not in out
        assert status == 0

    def test_an_absent_radiation_cover_says_so_and_does_not_fail_the_run(
            self, tmp_path, monkeypatch, capsys):
        status, out = _run_main(tmp_path, monkeypatch, capsys,
                                radiation_cover=None)
        assert "INFO  cloud_cover_radiation" not in out
        assert "NOTE  the window saves no radiation.total_cloud_cover" in out
        assert status == 0

    def test_an_all_zero_radiation_cover_names_the_other_reason(
            self, tmp_path, monkeypatch, capsys):
        _status, out = _run_main(tmp_path, monkeypatch, capsys,
                                 radiation_cover=0.0)
        assert "NOTE  radiation.total_cloud_cover is identically zero" in out
        assert "INFO  cloud_cover_radiation" not in out

    def test_a_cover_outside_the_band_fails_the_run(self, tmp_path,
                                                    monkeypatch, capsys):
        # A cloudless column scores 0, below the floor: the gate has to be
        # live, not merely printed.
        status, out = _run_main(tmp_path, monkeypatch, capsys,
                                profile=np.zeros(len(_PROFILE)))
        assert "FAIL  cloud_cover = 0.00" in out
        assert status == 1
