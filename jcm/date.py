from __future__ import annotations

import jax.numpy as jnp
import tree_math
import jax_datetime as jdt

# Calendar lengths used for `tyear` / `model_year` / `model_day` derivations.
# `gregorian` keeps the long-standing Julian-mean approximation that smooths
# leap years across all dates; `365_day` is the no-leap calendar SPEEDY's
# climatologies and solar tables assume by construction. We deliberately do
# not support full CFTime calendars (#287) — pick whichever of these matches
# the physics being run.
DAYS_PER_YEAR_BY_CALENDAR = {
    "gregorian": 365.2425,
    "365_day":   365.0,
}
# Module-level default keeps the legacy gregorian approximation so callers that
# don't go through `Model` (e.g. ad-hoc usage in tests / notebooks) see no
# behavior change. `Model.__init__` overrides this to `365_day` for SPEEDY.
DEFAULT_CALENDAR = "gregorian"

# Reference epoch for converting absolute model dates to seconds for forcing
# alignment. Picked to match jax_datetime's own zero (`1970-01-01 UTC`).
MODEL_EPOCH = jdt.to_datetime('1970-01-01')

# Backwards-compatible single-value alias still imported elsewhere; will be
# removed once every caller passes `calendar` explicitly.
_DAYS_YEAR = DAYS_PER_YEAR_BY_CALENDAR["gregorian"]


def days_per_year(calendar: str = DEFAULT_CALENDAR) -> float:
    """Return the days-per-year used by `calendar`."""
    try:
        return DAYS_PER_YEAR_BY_CALENDAR[calendar]
    except KeyError as exc:
        raise ValueError(
            f"Unknown calendar {calendar!r}; expected one of "
            f"{sorted(DAYS_PER_YEAR_BY_CALENDAR)}"
        ) from exc


@tree_math.struct
class DateData:
    tyear: float # Fractional time of year, should possibly be part of the model itself (i.e. not in physics_data)
    dt: jdt.Datetime  # Actual datetime object, may be None if not needed
    model_year: jnp.int32
    model_step: jnp.int32
    dt_seconds: float # Model timestep in seconds

    @classmethod
    def zeros(cls, dt=None, tyear=None, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=tyear if tyear is not None else 0.0,
          # FIXME: dt should be required and tyear and model_year derived from it (as properties)
          dt=dt if dt is not None else jdt.Datetime.from_pydatetime(jdt.to_datetime('1950-01-01')),
          model_year=model_year if model_year is not None else jnp.int32(1950),
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else 1800.0)

    @classmethod
    def set_date(cls, model_time, model_step=None, dt_seconds=None, calendar=DEFAULT_CALENDAR):
        return cls(
          tyear=fraction_of_year_elapsed(model_time, calendar=calendar),
          dt=model_time,
          model_year=get_year(model_time, calendar=calendar),
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else 1800.0)

    @classmethod
    def ones(cls, tyear=None, dt=None, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=tyear if tyear is not None else 1.0,
          dt=dt if dt is not None else jdt.Datetime.from_pydatetime(jdt.to_datetime('1950-01-01')),
          model_year=model_year if model_year is not None else jnp.int32(1950),
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else 1800.0)

    def model_day(self, calendar: str = DEFAULT_CALENDAR):
        return jnp.round(self.tyear * days_per_year(calendar)).astype(jnp.int32)

    def copy(self, tyear=None, dt=None, model_year=None, model_step=None, dt_seconds=None):
        return DateData(
          tyear=tyear if tyear is not None else self.tyear,
          dt=dt if dt is not None else self.dt,
          model_year=model_year if model_year is not None else self.model_year,
          model_step=model_step if model_step is not None else self.model_step,
          dt_seconds=dt_seconds if dt_seconds is not None else self.dt_seconds)

def get_year(dt: jdt.Datetime, calendar: str = DEFAULT_CALENDAR):
    """Get the year from a Datetime JAX object using the given calendar."""
    return jnp.int32(1970 + dt.delta.days // days_per_year(calendar))

def fraction_of_year_elapsed(dt: jdt.Datetime, calendar: str = DEFAULT_CALENDAR):
    """Fraction of the year that has elapsed at `dt` under the given calendar.

    Both supported calendars treat every year as having a fixed number of days
    (365.0 for `365_day`, 365.2425 for `gregorian`) — there is no real Feb 29
    handling, by design. This is sufficient for annual solar lookups and for
    indexing climatological forcing tables.
    """
    dpy = days_per_year(calendar)
    days_elapsed_in_year = jnp.floor(dt.delta.days % dpy)
    days_elapsed_in_year += dt.delta.seconds / (24 * 60 * 60)
    return days_elapsed_in_year / dpy


def absolute_seconds_since_epoch(dt: jdt.Datetime) -> jnp.ndarray:
    """Total seconds between `dt` and `MODEL_EPOCH` (1970-01-01).

    Used to align forcing time axes with the model clock under
    ``align_mode='by_date'``. Returns a JAX-traceable scalar.
    """
    delta = dt - jdt.Datetime.from_pydatetime(MODEL_EPOCH)
    return delta.days * 86400.0 + delta.seconds


# Mapping of accepted unit aliases to their conversion factor in days. Months
# and years are calendar-dependent so they're handled separately.
_FIXED_UNIT_DAYS: dict[str, float] = {
    "sec": 1.0 / 86400.0, "secs": 1.0 / 86400.0,
    "second": 1.0 / 86400.0, "seconds": 1.0 / 86400.0,
    "min": 1.0 / 1440.0, "mins": 1.0 / 1440.0,
    "minute": 1.0 / 1440.0, "minutes": 1.0 / 1440.0,
    "h": 1.0 / 24.0, "hr": 1.0 / 24.0, "hrs": 1.0 / 24.0,
    "hour": 1.0 / 24.0, "hours": 1.0 / 24.0,
    "d": 1.0, "day": 1.0, "days": 1.0,
    "w": 7.0, "wk": 7.0, "wks": 7.0, "week": 7.0, "weeks": 7.0,
}
_MONTH_ALIASES = {"mo", "mon", "mons", "month", "months"}
_YEAR_ALIASES = {"y", "yr", "yrs", "year", "years"}


def parse_duration_days(value, calendar: str = DEFAULT_CALENDAR) -> float:
    """Parse a duration spec into a float number of days.

    Numeric input (int / float) is returned as-is — assumed to be days.
    Strings are parsed as `<number> <unit>`, e.g. `'1 month'`,
    `'5 years'`, `'30 days'`, `'12 hours'`. Months and years are mapped
    via the chosen `calendar` (so under ``'365_day'``, '1 month' is
    365/12 ≈ 30.4167 days; under ``'gregorian'`` it's 365.2425/12).

    This intentionally does *not* align to calendar month/year
    boundaries — each "month" is a fixed-length chunk. For
    calendar-aligned aggregation use ``ModelPredictions.resample``.
    """
    if isinstance(value, (int, float)):
        return float(value)

    import re
    s = str(value).strip().lower()
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-z]+)\s*$", s)
    if not m:
        raise ValueError(
            f"Cannot parse duration {value!r}. Expected '<number> <unit>' "
            "with unit in {seconds, minutes, hours, days, weeks, months, years}."
        )
    n = float(m.group(1))
    unit = m.group(2)

    if unit in _FIXED_UNIT_DAYS:
        return n * _FIXED_UNIT_DAYS[unit]
    if unit in _MONTH_ALIASES:
        return n * days_per_year(calendar) / 12.0
    if unit in _YEAR_ALIASES:
        return n * days_per_year(calendar)

    raise ValueError(
        f"Unknown duration unit {unit!r} in {value!r}. "
        f"Accepted units: {sorted(_FIXED_UNIT_DAYS) + sorted(_MONTH_ALIASES) + sorted(_YEAR_ALIASES)}"
    )
