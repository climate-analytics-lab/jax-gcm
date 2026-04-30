from __future__ import annotations

import jax.numpy as jnp
import tree_math
import jax_datetime as jdt


# `gregorian` carries true Y/M/D arithmetic with leap years (Fliegel & Van
# Flandern below); `365_day` is the no-leap calendar SPEEDY's climatologies
# and solar tables assume by construction. Full CFTime calendars are out of
# scope (#287).
SUPPORTED_CALENDARS = ("gregorian", "365_day")
# Days-per-year used for `365_day`-mode arithmetic and for any caller that
# wants a single number (e.g. parsing `'1 year'` to days).
_DAYS_PER_YEAR_BY_CALENDAR = {
    "gregorian": 365.2425,
    "365_day":   365.0,
}
# Module-level default. `Model.__init__` overrides this to `365_day` for
# SPEEDY; ad-hoc callers (tests, notebooks) get gregorian.
DEFAULT_CALENDAR = "gregorian"

# Reference epoch for converting absolute model dates to seconds for forcing
# alignment. Picked to match jax_datetime's own zero (`1970-01-01 UTC`).
MODEL_EPOCH = jdt.to_datetime('1970-01-01')

# Backwards-compatible single-value alias still imported elsewhere.
_DAYS_YEAR = _DAYS_PER_YEAR_BY_CALENDAR["gregorian"]


def days_per_year(calendar: str = DEFAULT_CALENDAR) -> float:
    """Return the days-per-year used by `calendar`."""
    try:
        return _DAYS_PER_YEAR_BY_CALENDAR[calendar]
    except KeyError as exc:
        raise ValueError(
            f"Unknown calendar {calendar!r}; expected one of {SUPPORTED_CALENDARS}"
        ) from exc


# ---------------------------------------------------------------------------
# Gregorian Y/M/D from days since 1970-01-01 (Fliegel & Van Flandern, 1968)
# ---------------------------------------------------------------------------

# Julian Day Number of 1970-01-01 — the Unix epoch.
_UNIX_EPOCH_JDN = 2440588


def gregorian_ymd_from_days(days_since_epoch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert days-since-1970 to a proper Gregorian (year, month, day).

    Uses the Fliegel & Van Flandern (1968) integer algorithm — JAX-friendly
    (only int arithmetic, no Python ``datetime``) and exact for any year in
    the proleptic Gregorian calendar. References:
        - Fliegel, H. F., & Van Flandern, T. C. (1968).
        - https://aa.usno.navy.mil/faq/JD_formula

    Variable names match the published algorithm verbatim; intentionally
    not renamed (`# noqa: E741` for the lowercase `l`).
    """
    jdn = days_since_epoch + _UNIX_EPOCH_JDN
    l = jdn + 68569                              # noqa: E741
    n = (4 * l) // 146097
    l = l - (146097 * n + 3) // 4                # noqa: E741
    i = (4000 * (l + 1)) // 1461001
    l = l - (1461 * i) // 4 + 31                 # noqa: E741
    j = (80 * l) // 2447
    day = l - (2447 * j) // 80
    l = j // 11                                  # noqa: E741
    month = j + 2 - 12 * l
    year = 100 * (n - 49) + i + l
    return year, month, day


def is_leap_year(year: jnp.ndarray) -> jnp.ndarray:
    """Gregorian leap-year predicate (returns a JAX boolean array)."""
    return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)


def _gregorian_day_of_year(year, month, day) -> jnp.ndarray:
    """Zero-indexed day-of-year for a Gregorian date (Jan 1 → 0)."""
    days_in_month = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cum_no_leap = jnp.concatenate([jnp.array([0]), jnp.cumsum(days_in_month)[:-1]])
    leap_offset = jnp.where(jnp.arange(12) >= 2, is_leap_year(year).astype(jnp.int32), 0)
    cum = cum_no_leap + leap_offset
    return cum[month - 1] + (day - 1)


# ---------------------------------------------------------------------------
# DateData
# ---------------------------------------------------------------------------


@tree_math.struct
class DateData:
    """Per-step time info threaded through the model.

    Carries an absolute `jdt.Datetime` (`dt`) plus the integer step counter
    and timestep length used by physics. Fraction-of-year and model-year are
    not stored — they are derived from `dt` on demand via the
    ``tyear``/``model_year`` methods, so they cannot drift out of sync
    (#352).
    """

    dt: jdt.Datetime
    model_step: jnp.int32
    dt_seconds: float

    @classmethod
    def zeros(cls, dt=None, model_step=None, dt_seconds=None):
        return cls(
            dt=dt if dt is not None else jdt.Datetime.from_pydatetime(jdt.to_datetime('1950-01-01')),
            model_step=model_step if model_step is not None else jnp.int32(0),
            dt_seconds=dt_seconds if dt_seconds is not None else 1800.0,
        )

    @classmethod
    def set_date(cls, model_time, model_step=None, dt_seconds=None, calendar=DEFAULT_CALENDAR):
        # `calendar` is accepted for backward compatibility but is no longer
        # used at construction time — `tyear`/`model_year` are derived from
        # `dt` on demand and take their own calendar argument.
        del calendar  # unused
        return cls(
            dt=model_time,
            model_step=model_step if model_step is not None else jnp.int32(0),
            dt_seconds=dt_seconds if dt_seconds is not None else 1800.0,
        )

    @classmethod
    def ones(cls, dt=None, model_step=None, dt_seconds=None):
        return cls(
            dt=dt if dt is not None else jdt.Datetime.from_pydatetime(jdt.to_datetime('1950-01-01')),
            model_step=model_step if model_step is not None else jnp.int32(1),
            dt_seconds=dt_seconds if dt_seconds is not None else 1800.0,
        )

    def tyear(self, calendar: str = DEFAULT_CALENDAR) -> jnp.ndarray:
        """Fraction of the year elapsed at `self.dt` under `calendar`."""
        return fraction_of_year_elapsed(self.dt, calendar=calendar)

    def model_year(self, calendar: str = DEFAULT_CALENDAR) -> jnp.ndarray:
        """Year extracted from `self.dt` under `calendar`."""
        return get_year(self.dt, calendar=calendar)

    def model_day(self, calendar: str = DEFAULT_CALENDAR):
        """Integer day-of-year (rounded) under `calendar`."""
        return jnp.round(self.tyear(calendar) * days_per_year(calendar)).astype(jnp.int32)

    def copy(self, dt=None, model_step=None, dt_seconds=None):
        return DateData(
            dt=dt if dt is not None else self.dt,
            model_step=model_step if model_step is not None else self.model_step,
            dt_seconds=dt_seconds if dt_seconds is not None else self.dt_seconds,
        )


# ---------------------------------------------------------------------------
# Calendar-aware date math
# ---------------------------------------------------------------------------


def get_year(dt: jdt.Datetime, calendar: str = DEFAULT_CALENDAR) -> jnp.ndarray:
    """Year of `dt` under `calendar`."""
    if calendar == "gregorian":
        year, _, _ = gregorian_ymd_from_days(dt.delta.days)
        return year.astype(jnp.int32)
    if calendar == "365_day":
        return jnp.int32(1970 + dt.delta.days // 365)
    raise ValueError(
        f"Unknown calendar {calendar!r}; expected one of {SUPPORTED_CALENDARS}"
    )


def fraction_of_year_elapsed(dt: jdt.Datetime, calendar: str = DEFAULT_CALENDAR) -> jnp.ndarray:
    """Fraction of the year elapsed at `dt` under the given calendar.

    Under ``'gregorian'`` this is the *true* day-of-year divided by the
    actual length of the year (365 or 366) — leap years are honoured (#410).
    Under ``'365_day'`` every year is exactly 365 days, no leap days exist,
    and `tyear = (days_since_year_start) / 365`.
    """
    fraction_of_day = dt.delta.seconds / 86400.0

    if calendar == "gregorian":
        year, month, day = gregorian_ymd_from_days(dt.delta.days)
        doy = _gregorian_day_of_year(year, month, day)
        days_in_year = jnp.where(is_leap_year(year), 366, 365)
        return (doy + fraction_of_day) / days_in_year

    if calendar == "365_day":
        days_into_year = dt.delta.days % 365
        return (days_into_year + fraction_of_day) / 365.0

    raise ValueError(
        f"Unknown calendar {calendar!r}; expected one of {SUPPORTED_CALENDARS}"
    )


def absolute_seconds_since_epoch(dt: jdt.Datetime) -> jnp.ndarray:
    """Total seconds between `dt` and `MODEL_EPOCH` (1970-01-01).

    Used to align forcing time axes with the model clock under
    ``align_mode='by_date'``. Returns a JAX-traceable scalar.
    """
    delta = dt - jdt.Datetime.from_pydatetime(MODEL_EPOCH)
    return delta.days * 86400.0 + delta.seconds


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


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
