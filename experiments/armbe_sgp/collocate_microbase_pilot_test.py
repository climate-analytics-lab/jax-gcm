"""Tests for the bounded MICROBASE/ARMBE pilot collocation."""

import numpy as np
import xarray as xr

from collocate_microbase_pilot import (
    _canonicalize_atmosphere,
    build_hourly_cache,
    reconstruct_hydrostatic_state,
)
from microbase_physics import AKAP, CPD, EPS, GRAV, RD, RV


def test_portable_constants_match_jcm_contract():
    assert GRAV == 9.81
    assert CPD == 1004.64
    assert AKAP == 2.0 / 7.0
    assert np.isclose(RD, 287.04)
    assert RV == 461.0
    assert EPS == 0.622


def test_canonicalize_legacy_armbeatm_schema():
    legacy = xr.Dataset(
        {
            "T_z": (("time", "z"), [[280.0]]),
            "Td_z": (("time", "z"), [[270.0]]),
            "rh_z": (("time", "z"), [[50.0]]),
            "p_sfc": ("time", [1000.0]),
        },
        coords={"time": [np.datetime64("2011-01-01")], "z": [15.0]},
    )
    canonical = _canonicalize_atmosphere(legacy)
    assert canonical.temperature_h.item() == 280.0
    assert canonical.dewpoint_h.item() == 270.0
    assert canonical.relative_humidity_h.item() == 50.0
    assert canonical.pressure_sfc.item() == 1000.0
    assert canonical.height.item() == 15.0


def test_hydrostatic_reconstruction_matches_dry_isothermal_column():
    height = np.array([100.0, 500.0, 1000.0])
    temperature = np.full(3, 280.0)
    pressure, density, specific_humidity = reconstruct_hydrostatic_state(
        height, temperature, np.zeros(3), 100000.0
    )
    expected = 100000.0 * np.exp(-9.81 * height / (287.04 * 280.0))
    np.testing.assert_allclose(pressure, expected, rtol=1e-12)
    np.testing.assert_allclose(density, pressure / (287.04 * 280.0), rtol=1e-12)
    np.testing.assert_array_equal(specific_humidity, 0.0)


def test_build_hourly_cache_preserves_missing_and_qc_semantics():
    day = np.datetime64("2018-06-01")
    micro_time = day + np.arange(48) * np.timedelta64(30, "m")
    hourly_time = day + np.arange(24) * np.timedelta64(1, "h") + np.timedelta64(30, "m")
    height = np.array([160.0, 190.0])
    retrieval = np.zeros((48, 2), dtype=float)
    retrieval[0, 0] = 1
    retrieval[1, 0] = 2
    retrieval[2, 1] = np.nan
    liquid = np.where(retrieval == 1, 0.2, 0.0)
    ice = np.where(retrieval == 1, 0.1, 0.0)
    qc = np.zeros((48, 2), dtype=int)
    precip = np.full(48, np.nan)
    microbase = xr.Dataset(
        {
            "retrieval_flag": (("time", "height"), retrieval),
            "liquid_water_content": (("time", "height"), liquid),
            "ice_water_content": (("time", "height"), ice),
            "qc_liquid_water_content": (("time", "height"), qc),
            "qc_ice_water_content": (("time", "height"), qc),
            "precip_flag": ("time", precip),
        },
        coords={"time": micro_time, "height": height},
    )
    microbase.liquid_water_content.attrs["units"] = "g m-3"
    microbase.ice_water_content.attrs["units"] = "g m-3"
    cldrad = xr.Dataset(
        {
            "cld_frac": (("time", "height"), np.zeros((24, 2))),
            "qc_cld_frac": (("time", "height"), np.zeros((24, 2), dtype=int)),
        },
        coords={"time": hourly_time, "height": height},
    )
    temperature = np.full((24, 3), np.nan)
    relative_humidity = np.full((24, 3), np.nan)
    dewpoint = np.full((24, 3), np.nan)
    observed = np.array([5, 11, 17, 23])
    temperature[observed] = 280.0
    relative_humidity[observed] = 50.0
    dewpoint[observed] = 270.0
    atmosphere = xr.Dataset(
        {
            "temperature_h": (("time", "height"), temperature),
            "relative_humidity_h": (("time", "height"), relative_humidity),
            "dewpoint_h": (("time", "height"), dewpoint),
            "pressure_sfc": ("time", np.full(24, 1000.0)),
        },
        coords={"time": hourly_time, "height": np.array([145.0, 175.0, 205.0])},
    )

    hourly, paired, audit = build_hourly_cache(
        microbase, atmosphere, cldrad, "2018-06-01"
    )

    assert hourly.sizes == {"time": 24, "height": 2}
    assert paired.sizes == {"time": 4, "height": 2}
    assert "temperature" not in hourly
    assert np.array_equal(paired.time.values, hourly.time.values[observed])
    assert np.all(paired.temperature.values == 280.0)
    assert hourly.cloud_fraction_strict.values[0, 0] == 1.0
    assert hourly.cloud_fraction_inclusive.values[0, 0] == 1.0
    assert hourly.strict_occurrence_count.values[0, 0] == 1
    assert hourly.inclusive_occurrence_count.values[0, 0] == 2
    assert hourly.strict_occurrence_count.values[1, 1] == 1
    assert hourly.liquid_water_concentration.values[0, 0] == 0.2
    assert audit["precip_flag"]["status"] == "unknown"
    assert audit["conversion_gates"]["condensate_convention_resolved"]
    assert audit["conversion_gates"]["pressure_profile_available"]
    np.testing.assert_allclose(
        paired.qc.values,
        1.0e-3 * paired.liquid_water_concentration.values / paired.air_density.values,
    )
    assert np.all(np.diff(paired.pressure.values, axis=1) < 0.0)
