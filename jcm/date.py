from __future__ import annotations

import jax.numpy as jnp
import tree_math
import jax_datetime as jdt
import math
import numbers
import datetime as pydt
import re


SECONDS_PER_DAY = 86_400

def to_datetime(value, *, name: str = "time") -> jdt.Datetime:
    """Normalize one scalar timestamp to the whole-second model clock."""
    if isinstance(value, jdt.Datetime):
        if value.delta.days.shape or value.delta.seconds.shape:
            raise ValueError(f"{name} must be a scalar datetime.")
        return value
    if isinstance(value, str):
        if value.strip().lower() == "nat":
            raise ValueError(f"{name} cannot be NaT.")
        match = re.search(r"\.(\d+)", value)
        if match and int(match.group(1)) != 0:
            raise ValueError(f"{name} must have whole-second precision.")
        parsed = pydt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, pydt.datetime):
        parsed = value
    else:
        raise TypeError(f"{name} must be a datetime or ISO datetime string.")
    if parsed.microsecond:
        raise ValueError(f"{name} must have whole-second precision.")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(pydt.timezone.utc).replace(tzinfo=None)
    return jdt.to_datetime(parsed.isoformat(sep=" "))


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
    def set_date(cls, model_time, model_step=None, dt_seconds=None):
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

    def tyear(self) -> jnp.ndarray:
        """Fraction of the proleptic Gregorian year elapsed."""
        return fraction_of_year_elapsed(self.dt)

    def model_year(self) -> jnp.ndarray:
        """Proleptic Gregorian year containing ``self.dt``."""
        return get_year(self.dt)

    def model_day(self):
        """Zero-based Gregorian day of year (rounded for compatibility)."""
        year, month, day = gregorian_ymd_from_days(self.dt.delta.days)
        fraction = self.dt.delta.seconds / SECONDS_PER_DAY
        return jnp.round(_gregorian_day_of_year(year, month, day) + fraction).astype(jnp.int32)

    def copy(self, dt=None, model_step=None, dt_seconds=None):
        return DateData(
            dt=dt if dt is not None else self.dt,
            model_step=model_step if model_step is not None else self.model_step,
            dt_seconds=dt_seconds if dt_seconds is not None else self.dt_seconds,
        )


# ---------------------------------------------------------------------------
# Calendar-aware date math
# ---------------------------------------------------------------------------


def get_year(dt: jdt.Datetime) -> jnp.ndarray:
    """Proleptic Gregorian year of ``dt``."""
    year, _, _ = gregorian_ymd_from_days(dt.delta.days)
    return year.astype(jnp.int32)


def fraction_of_year_elapsed(dt: jdt.Datetime) -> jnp.ndarray:
    """Fraction of the actual proleptic Gregorian year elapsed at ``dt``."""
    fraction_of_day = dt.delta.seconds / SECONDS_PER_DAY
    year, month, day = gregorian_ymd_from_days(dt.delta.days)
    doy = _gregorian_day_of_year(year, month, day)
    days_in_year = jnp.where(is_leap_year(year), 366, 365)
    return (doy + fraction_of_day) / days_in_year


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


_FIXED_UNIT_SECONDS: dict[str, float] = {
    "sec": 1.0, "secs": 1.0,
    "second": 1.0, "seconds": 1.0,
    "min": 60.0, "mins": 60.0,
    "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0,
    "hour": 3600.0, "hours": 3600.0,
    "d": SECONDS_PER_DAY, "day": SECONDS_PER_DAY, "days": SECONDS_PER_DAY,
    "w": 7 * SECONDS_PER_DAY, "wk": 7 * SECONDS_PER_DAY,
    "wks": 7 * SECONDS_PER_DAY, "week": 7 * SECONDS_PER_DAY,
    "weeks": 7 * SECONDS_PER_DAY,
}
_MONTH_ALIASES = {"mo", "mon", "mons", "month", "months"}
_YEAR_ALIASES = {"y", "yr", "yrs", "year", "years"}


def parse_duration_seconds(value) -> int:
    """Parse a fixed duration and return an exact whole-second count.

    Numeric input remains days for compatibility. Strings accept only fixed
    seconds, minutes, hours, days, and weeks. Calendar months and years are
    scheduling concepts and are deliberately rejected.
    """
    if isinstance(value, numbers.Real):
        seconds = float(value) * SECONDS_PER_DAY
        source = repr(value)
    else:
        import re
        s = str(value).strip().lower()
        m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-z]+)\s*$", s)
        if not m:
            raise ValueError(
                f"Cannot parse duration {value!r}. Expected '<number> <unit>' "
                "with a fixed unit in {seconds, minutes, hours, days, weeks}."
            )
        n = float(m.group(1))
        unit = m.group(2)
        if unit in _MONTH_ALIASES | _YEAR_ALIASES:
            raise ValueError(
                f"Calendar duration {value!r} is not fixed; use end_time for "
                "named month/year boundaries."
            )
        if unit not in _FIXED_UNIT_SECONDS:
            raise ValueError(f"Unknown duration unit {unit!r} in {value!r}.")
        seconds = n * _FIXED_UNIT_SECONDS[unit]
        source = repr(value)

    if not math.isfinite(seconds):
        raise ValueError(f"Duration must be finite, got {source}.")
    rounded = round(seconds)
    if seconds <= 0:
        raise ValueError(f"Duration must be positive, got {source}.")
    if abs(seconds - rounded) > 1e-9:
        raise ValueError(
            f"Duration {source} is not representable at whole-second precision."
        )
    return int(rounded)


def calendar_duration_months(value) -> int | None:
    """Whole months in a calendar duration string, or ``None`` if not one.

    ``"12 months"`` -> 12, ``"1 year"`` -> 12. Only positive whole numbers
    are accepted; anything that is not a month/year string (numbers, fixed
    units) returns ``None`` so callers fall through to
    :func:`parse_duration_seconds`.
    """
    if isinstance(value, numbers.Real) or value is None:
        return None
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-z]+)\s*$",
                 str(value).strip().lower())
    if not m or m.group(2) not in _MONTH_ALIASES | _YEAR_ALIASES:
        return None
    n = float(m.group(1))
    if n <= 0 or not n.is_integer():
        raise ValueError(
            f"Calendar duration {value!r} must be a positive whole number of "
            "months or years.")
    return int(n) * (12 if m.group(2) in _YEAR_ALIASES else 1)


def add_calendar_months(start: pydt.datetime, months: int) -> pydt.datetime:
    """``start`` moved by whole Gregorian months, keeping day and time.

    Refuses a start day that does not exist in the target month (e.g. Jan 31
    + 1 month) rather than guessing an end-of-month convention.
    """
    year, month0 = divmod(start.month - 1 + months, 12)
    try:
        return start.replace(year=start.year + year, month=month0 + 1)
    except ValueError as error:
        raise ValueError(
            f"{start.isoformat()} + {months} month(s) has no matching day; "
            "start calendar-duration runs on a day that exists in every "
            "month (1-28), or set run.end_time explicitly.") from error


def resolve_calendar_end(value, start) -> str | None:
    """ISO end time for a calendar ``total_time``, or ``None`` if not one.

    ``start`` is the run's start (``jax_datetime.Datetime``, datetime64,
    datetime or ISO string). Calendar durations are not fixed lengths, so
    they are resolved against the start into the equivalent exact
    ``end_time`` at the configuration boundary; the model clock itself only
    ever sees fixed seconds.
    """
    months = calendar_duration_months(value)
    if months is None:
        return None
    import numpy as np
    if start is None:                       # Model's default start
        start = "2000-01-01"
    if isinstance(start, np.datetime64):
        start = str(start.astype("datetime64[s]"))
    start64 = np.datetime64(to_datetime(start).to_datetime64(), "s")
    end = add_calendar_months(start64.astype(pydt.datetime), months)
    return end.isoformat()


def parse_duration_days(value) -> float:
    """Compatibility adapter returning fixed duration days."""
    return parse_duration_seconds(value) / SECONDS_PER_DAY
