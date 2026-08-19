"""Tests for standard ARMBE temporal-cadence summaries."""

import numpy as np

import summarize_armbe_cadence as cadence


def test_available_by_time_profiles() -> None:
    values = np.asarray([[np.nan, np.nan], [1.0, np.nan], [np.nan, 2.0]])
    np.testing.assert_array_equal(cadence.available_by_time(values), [False, True, True])


def test_temporal_semantics() -> None:
    assert cadence.temporal_semantics(
        "tot_cld", {"long_name": "Total cloud fraction, hourly mean"}, "armbecldrad", ("time",)
    ) == "one_hour_mean"
    assert cadence.temporal_semantics(
        "temperature_p",
        {"source_comment": "SONDE, 2 sec data"},
        "armbeatm",
        ("time", "pressure"),
    ) == "sounding_associated_hourly_cell"
    assert cadence.temporal_semantics(
        "qc_tot_cld", {"long_name": "Quality check results"}, "armbecldrad", ("time",)
    ) == "quality_flag_for_hourly_cell"
    assert cadence.temporal_semantics(
        "latent_heat_flux_baebbr", {"source_comment": "BAEBBR, 30 min data"}, "armbeatm", ("time",)
    ) == "one_hour_mean"
