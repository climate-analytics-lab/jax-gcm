"""Regression coverage for exact Gregorian forcing selection (#449, #805)."""

import cftime
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

from jcm.date import DateData
from jcm.forcing import (
    BY_DATE,
    MONTHLY_CLIMATOLOGY,
    ForcingData,
    make_time_series,
    read_anthropogenic_emissions,
)


def _date(value, *, step=0, dt_seconds=1800):
    return DateData.set_date(
        jdt.to_datetime(value), model_step=jnp.int32(step),
        dt_seconds=dt_seconds)


def _monthly(values=None):
    values = jnp.arange(12, dtype=jnp.float32) if values is None else values
    return make_time_series(
        values,
        np.arange("2001-01", "2002-01", dtype="datetime64[M]"),
        MONTHLY_CLIMATOLOGY,
    )


@pytest.mark.parametrize("year", [2005, 2000, 1900, 2100])
def test_issue_805_all_civil_month_boundaries_under_jit(year):
    """#805: all 12 records switch on month starts, including century years."""
    series = _monthly()

    @jax.jit
    def select(date):
        forcing = ForcingData.zeros((1, 1), sea_surface_temperature=series)
        return forcing.select(date).sea_surface_temperature

    for month in range(1, 13):
        at_start = float(select(_date(f"{year:04d}-{month:02d}-01")))
        assert at_start == month - 1
        if month > 1:
            previous_day = np.datetime64(f"{year:04d}-{month:02d}-01") - np.timedelta64(1, "D")
            before = float(select(_date(str(previous_day))))
            assert before == month - 2


def test_issue_805_shared_select_slices_all_nested_consumers_together():
    """#805: one selection path covers surface, dust, emissions and oxidants."""
    monthly = _monthly()
    forcing = ForcingData.zeros(
        (1, 1),
        sea_surface_temperature=monthly,
        sice_am=_monthly(jnp.arange(12) + 100),
    ).copy(
        dust_source=_monthly(jnp.arange(12) + 200),
        anthropogenic_emissions={"emis_surface_combustion_bc":
                                 _monthly(jnp.arange(12) + 300)},
        oxidant_vmr={"oh": _monthly(jnp.arange(12) + 400)},
    )

    @jax.jit
    def select(date):
        selected = forcing.select(date)
        return (selected.sea_surface_temperature, selected.sice_am,
                selected.dust_source,
                selected.anthropogenic_emissions["emis_surface_combustion_bc"],
                selected.oxidant_vmr["oh"])

    got = tuple(float(x) for x in select(_date("2000-03-01")))
    assert got == (2.0, 102.0, 202.0, 302.0, 402.0)


def _noleap_emissions(values, mode):
    dates = [
        cftime.DatetimeNoLeap(1999, 1, 1),
        cftime.DatetimeNoLeap(2000, 2, 28),
        cftime.DatetimeNoLeap(2000, 3, 1),
        cftime.DatetimeNoLeap(2001, 1, 1),
    ]
    ds = xr.Dataset(
        {"emis_surface_combustion_bc":
         (("time", "lon", "lat"), np.asarray(values)[:, None, None])},
        coords={"time": dates, "lon": [0.0], "lat": [0.0]},
    )
    return read_anthropogenic_emissions(ds, align_mode=mode)[
        "emis_surface_combustion_bc"]


def test_issue_449_noleap_nominal_dates_have_no_accumulated_drift():
    """#449: noleap inputs map by nominal Y/M/D, not noleap elapsed days."""
    series = _noleap_emissions([1999.0, 228.0, 301.0, 2001.0], "by_date")
    forcing = ForcingData.zeros((1, 1)).copy(
        anthropogenic_emissions={"emis_surface_combustion_bc": series})

    @jax.jit
    def select(value):
        return forcing.select(value).anthropogenic_emissions[
            "emis_surface_combustion_bc"][0, 0]

    assert float(select(_date("2000-02-28"))) == 228.0
    assert float(select(_date("2000-03-01"))) == 301.0
    assert float(select(_date("2001-01-01"))) == 2001.0


def test_issue_449_feb29_hold_and_continuous_interpolation_are_explicit():
    """#449: noleap Feb 29 is held or interpolated according to alignment."""
    held = _noleap_emissions([0.0, 10.0, 30.0, 40.0], "by_date")
    interp = _noleap_emissions([0.0, 10.0, 30.0, 40.0], "by_date_interp")
    held_forcing = ForcingData.zeros((1, 1), co2_vmr=held)
    interp_forcing = ForcingData.zeros((1, 1), co2_vmr=interp)

    @jax.jit
    def select(forcing, date):
        selected = forcing.select(date)
        return selected.co2_vmr[0, 0], date.model_step, date.dt_seconds

    feb28, step, timestep = select(held_forcing, _date(
        "2000-02-28", step=7, dt_seconds=3600))
    feb29, _, _ = select(held_forcing, _date("2000-02-29"))
    mar1, _, _ = select(held_forcing, _date("2000-03-01"))
    assert (float(feb28), float(feb29), float(mar1)) == (10.0, 10.0, 30.0)
    assert int(step) == 7
    assert float(timestep) == 3600.0

    assert float(select(interp_forcing, _date("2000-02-28"))[0]) == 10.0
    assert float(select(interp_forcing, _date("2000-02-29"))[0]) == 20.0
    assert float(select(interp_forcing, _date("2000-03-01"))[0]) == 30.0


def test_exact_hourly_dated_lookup_under_jit_avoids_float_epoch_aliasing():
    """Exact datetime comparison keeps adjacent modern-hour samples distinct."""
    series = make_time_series(
        jnp.asarray([1.0, 2.0]),
        np.asarray(["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
                   dtype="datetime64[s]"),
        BY_DATE,
    )
    forcing = ForcingData.zeros((1, 1), co2_vmr=series)
    select = jax.jit(lambda date: forcing.select(date).co2_vmr)
    assert float(select(_date("2026-01-01T00:59:59"))) == 1.0
    assert float(select(_date("2026-01-01T01:00:00"))) == 2.0
