"""Bounds-aware temporal aggregation for model output.

The routines in this module run explicitly on the host, after a trajectory has
been converted to xarray.  Model integration and raw ``ModelPredictions`` stay
JAX pytrees, so using these helpers cannot accidentally put calendar operations
inside a differentiated computation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


_TICK = np.timedelta64(1, "ms")
_CF_TIME_ENCODING = {
    "units": "seconds since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
}


def set_cf_datetime_encoding(ds: xr.Dataset, *names: str) -> xr.Dataset:
    """Give related datetime variables one deterministic CF file encoding."""
    for name in names:
        if name in ds and np.issubdtype(ds[name].dtype, np.datetime64):
            ds[name].encoding.update(_CF_TIME_ENCODING)
    return ds


def _time_operations(cell_methods: str) -> set[str]:
    """Return every operation declared for the CF time axis."""
    words = cell_methods.replace(":", " : ").split()
    return {
        words[index + 2]
        for index in range(len(words) - 2)
        if words[index:index + 2] == ["time", ":"]
    }


def _validated_intervals(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "time" not in ds.coords:
        raise ValueError("Monthly means require a 'time' coordinate.")
    if ds.sizes.get("time", 0) == 0:
        raise ValueError("Monthly means require at least one output interval.")
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        raise TypeError("time must contain exact datetime64 values.")
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    if bounds_name not in ds:
        raise ValueError(
            "Monthly means require interval means with a time_bounds variable."
        )
    bounds = np.asarray(ds[bounds_name].values)
    if (bounds.shape != (ds.sizes["time"], 2)
            or ds[bounds_name].dims[0] != "time"):
        raise ValueError(
            f"{bounds_name!r} must have shape (time, 2); got {bounds.shape}."
        )
    if not np.issubdtype(bounds.dtype, np.datetime64):
        raise TypeError("time bounds must be exact datetime64 values.")
    bounds = bounds.astype("datetime64[ms]")
    starts, ends = bounds[:, 0], bounds[:, 1]
    if np.any(np.isnat(bounds)) or np.any(np.isnat(ds.time.values)):
        raise ValueError("time and time bounds may not contain NaT.")
    if np.any(ends <= starts):
        raise ValueError("Every output interval must have positive duration.")
    if len(starts) > 1 and (np.any(starts[1:] < starts[:-1])
                            or np.any(starts[1:] < ends[:-1])):
        raise ValueError("Output intervals must be ordered and non-overlapping.")

    # An interval ending exactly at the first instant of the next month belongs
    # wholly to its starting month.  Anything extending beyond that boundary
    # cannot be split faithfully from a pre-computed interval mean.
    start_months = starts.astype("datetime64[M]")
    next_months = (start_months + np.timedelta64(1, "M")).astype(
        "datetime64[ms]")
    if np.any(ends > next_months):
        bad = int(np.flatnonzero(ends > next_months)[0])
        raise ValueError(
            "Cannot aggregate an interval that crosses a month boundary; "
            f"interval {bad} is [{starts[bad]}, {ends[bad]}]. Produce bounded "
            "daily means whose edges land on the boundary."
        )
    return bounds, start_months


def monthly_means(ds: xr.Dataset) -> xr.Dataset:
    """Reduce bounded interval means into real Gregorian monthly means.

    Durations are used as weights, independently for every variable.  Missing
    values therefore remove only that variable's duration from its denominator.
    Partial first/last months and gaps are retained and described by
    ``time_coverage`` and ``time_coverage_fraction``.

    Instantaneous samples and intervals crossing a month boundary are rejected:
    neither contains enough information to reconstruct a true monthly mean.
    """
    bounds, interval_months = _validated_intervals(ds)
    time_vars = [name for name, var in ds.data_vars.items()
                 if "time" in var.dims and name != ds["time"].attrs.get(
                     "bounds", "time_bounds")]
    if not time_vars:
        raise ValueError("Dataset contains no time-dependent variables to average.")
    invalid = [name for name in time_vars
               if _time_operations(
                   ds[name].attrs.get("cell_methods", "")) != {"mean"}]
    if invalid:
        raise ValueError(
            "Monthly aggregation accepts interval means only; missing "
            f"cell_methods='time: mean' on {invalid}."
        )

    durations_ns = (bounds[:, 1] - bounds[:, 0]) / _TICK
    month_values = np.unique(interval_months)
    monthly_parts = []
    monthly_bounds = []
    coverages = []
    fractions = []
    for month in month_values:
        indices = np.flatnonzero(interval_months == month)
        actual_start = bounds[indices, 0].min()
        actual_end = bounds[indices, 1].max()
        monthly_bounds.append((actual_start, actual_end))
        coverage_ns = durations_ns[indices].sum()
        full_start = month.astype("datetime64[ms]")
        full_end = (month + np.timedelta64(1, "M")).astype("datetime64[ms]")
        coverages.append(np.timedelta64(int(coverage_ns), "ms"))
        fractions.append(float(coverage_ns / ((full_end - full_start) / _TICK)))

        variables = {}
        for name in time_vars:
            var = ds[name].isel(time=indices)
            if not (np.issubdtype(var.dtype, np.number)
                    or np.issubdtype(var.dtype, np.bool_)):
                raise TypeError(
                    f"Time-dependent variable {name!r} is not numeric and "
                    "cannot be duration-averaged."
                )
            weights = xr.DataArray(
                durations_ns[indices], dims=("time",),
                coords={"time": var["time"]})
            valid_weights = weights.where(var.notnull())
            reduced = (var * valid_weights).sum("time", skipna=True) / (
                valid_weights.sum("time", skipna=True))
            reduced.attrs = dict(var.attrs)
            reduced.attrs["cell_methods"] = var.attrs["cell_methods"]
            variables[name] = reduced

        midpoint = actual_start + (actual_end - actual_start) // 2
        monthly_parts.append(
            xr.Dataset(variables, attrs=dict(ds.attrs)).expand_dims(time=[midpoint])
        )

    out = xr.concat(monthly_parts, dim="time", combine_attrs="override")
    for name, var in ds.data_vars.items():
        if "time" not in var.dims:
            out[name] = var
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    out[bounds_name] = (("time", "bounds"),
                        np.asarray(monthly_bounds, dtype="datetime64[ms]"))
    out["time_coverage"] = ("time", np.asarray(coverages))
    out["time_coverage_fraction"] = ("time", np.asarray(fractions))
    out["time"].attrs.update(ds["time"].attrs)
    out["time"].attrs["bounds"] = bounds_name
    out["time_coverage"].attrs.update(
        long_name="duration represented by source intervals")
    out["time_coverage_fraction"].attrs.update(
        long_name="fraction of the Gregorian month represented")
    return set_cf_datetime_encoding(out, "time", bounds_name)


class MonthlyMeanAccumulator:
    """Streaming monthly reducer retaining only sums and valid durations.

    ``update`` accepts any sequential chunk of bounded interval means and
    returns months that became closed by that chunk.  ``finish`` emits the
    final (possibly partial) month.  The accumulator never retains source
    frames, which bounds device-to-host output memory independently of month
    length and chunk boundaries.
    """

    def __init__(self):
        """Create an empty streaming accumulator."""
        self._month = None
        self._start = None
        self._end = None
        self._coverage_ns = 0
        self._sums = {}
        self._valid_ns = {}
        self._templates = {}
        self._static = None
        self._attrs = {}
        self._time_attrs = {}
        self._bounds_name = None
        self._variable_names = None

    def update(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Consume a chunk and return any month(s) closed by later input."""
        bounds, months = _validated_intervals(ds)
        if self._end is not None and bounds[0, 0] < self._end:
            raise ValueError(
                "Output intervals must remain ordered and non-overlapping "
                "across streaming chunks."
            )
        bounds_name = ds.time.attrs.get("bounds", "time_bounds")
        variables = [name for name, var in ds.data_vars.items()
                     if "time" in var.dims and name != bounds_name]
        if (self._variable_names is not None
                and set(variables) != self._variable_names):
            raise ValueError(
                "Every streaming chunk must contain the same time-dependent "
                "variables."
            )
        invalid = [name for name in variables
                   if _time_operations(
                       ds[name].attrs.get("cell_methods", "")) != {"mean"}]
        if invalid:
            raise ValueError(
                "Monthly aggregation accepts interval means only; missing "
                f"cell_methods='time: mean' on {invalid}."
            )
        # Dropping the dimension also drops the potentially large source time
        # coordinate.  Only true non-time variables/coordinates are retained.
        static = ds.drop_dims("time")
        if self._static is None:
            self._static = static
            self._attrs = dict(ds.attrs)
            self._time_attrs = dict(ds.time.attrs)
            self._bounds_name = bounds_name
            self._variable_names = set(variables)
        else:
            try:
                xr.testing.assert_identical(self._static, static)
            except AssertionError as error:
                raise ValueError(
                    "Non-time variables and spatial coordinates must be "
                    "identical across streaming chunks."
                ) from error
        emitted = []
        for i, month in enumerate(months):
            month_text = str(month)
            if self._month is not None and month_text != self._month:
                emitted.append(self._emit())
                self._reset_month()
            if self._month is None:
                self._month = month_text
                self._start = bounds[i, 0]
            duration_ns = int((bounds[i, 1] - bounds[i, 0]) / _TICK)
            self._end = bounds[i, 1]
            self._coverage_ns += duration_ns
            for name in variables:
                value = ds[name].isel(time=i, drop=True)
                if not (np.issubdtype(value.dtype, np.number)
                        or np.issubdtype(value.dtype, np.bool_)):
                    raise TypeError(f"Time-dependent variable {name!r} is not numeric.")
                valid = value.notnull()
                contribution = value.fillna(0) * duration_ns
                valid_duration = valid.astype(np.int64) * duration_ns
                if name not in self._sums:
                    self._sums[name] = contribution
                    self._valid_ns[name] = valid_duration
                    self._templates[name] = dict(value.attrs)
                else:
                    try:
                        total, contribution = xr.align(
                            self._sums[name], contribution, join="exact")
                        valid_total, valid_duration = xr.align(
                            self._valid_ns[name], valid_duration, join="exact")
                    except ValueError as error:
                        raise ValueError(
                            f"Variable {name!r} changed dimensions or spatial "
                            "coordinates across streaming chunks."
                        ) from error
                    self._sums[name] = total + contribution
                    self._valid_ns[name] = valid_total + valid_duration
        if len(emitted) == 1:
            return emitted[0]
        return xr.concat(emitted, dim="time") if emitted else None

    def finish(self) -> xr.Dataset | None:
        """Emit and clear the pending, possibly partial, final month."""
        if self._month is None:
            return None
        result = self._emit()
        self._reset_month()
        return result

    def _emit(self) -> xr.Dataset:
        variables = {}
        for name, total in self._sums.items():
            mean = total / self._valid_ns[name].where(self._valid_ns[name] != 0)
            mean.attrs = self._templates[name]
            variables[name] = mean
        midpoint = self._start + (self._end - self._start) // 2
        result = xr.Dataset(variables, attrs=self._attrs).expand_dims(time=[midpoint])
        if self._static is not None:
            for name, var in self._static.data_vars.items():
                result[name] = var
        result[self._bounds_name] = (("time", "bounds"),
                                     np.asarray([[self._start, self._end]]))
        month = np.datetime64(self._month, "M")
        full_ns = int(((month + np.timedelta64(1, "M")).astype("datetime64[ms]")
                       - month.astype("datetime64[ms]")) / _TICK)
        result["time_coverage"] = ("time",
                                   [np.timedelta64(self._coverage_ns, "ms")])
        result["time_coverage_fraction"] = (
            "time", [self._coverage_ns / full_ns])
        result.time.attrs.update(self._time_attrs)
        result.time.attrs["bounds"] = self._bounds_name
        result["time_coverage"].attrs.update(
            long_name="duration represented by source intervals")
        result["time_coverage_fraction"].attrs.update(
            long_name="fraction of the Gregorian month represented")
        return set_cf_datetime_encoding(result, "time", self._bounds_name)

    def _reset_month(self):
        self._month = self._start = self._end = None
        self._coverage_ns = 0
        self._sums = {}
        self._valid_ns = {}
        self._templates = {}

    def state_dict(self) -> dict:
        """Return a checkpointable mapping containing the pending month only."""
        return {
            "month": self._month,
            "start": None if self._start is None else str(self._start),
            "end": None if self._end is None else str(self._end),
            "coverage_ns": self._coverage_ns,
            "sums": {name: value.to_dict() for name, value in self._sums.items()},
            "valid_ns": {name: value.to_dict()
                         for name, value in self._valid_ns.items()},
            "templates": self._templates,
            "static": None if self._static is None else self._static.to_dict(),
            "attrs": self._attrs,
            "time_attrs": self._time_attrs,
            "bounds_name": self._bounds_name,
            "variable_names": (None if self._variable_names is None
                               else sorted(self._variable_names)),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "MonthlyMeanAccumulator":
        """Restore :meth:`state_dict` without retaining any source frames."""
        obj = cls()
        obj._month = state["month"]
        obj._start = (None if state["start"] is None
                      else np.datetime64(state["start"], "ms"))
        obj._end = (None if state["end"] is None
                    else np.datetime64(state["end"], "ms"))
        obj._coverage_ns = int(state["coverage_ns"])
        obj._sums = {name: xr.DataArray.from_dict(value)
                     for name, value in state["sums"].items()}
        obj._valid_ns = {name: xr.DataArray.from_dict(value)
                         for name, value in state["valid_ns"].items()}
        obj._templates = state["templates"]
        obj._static = (None if state["static"] is None
                       else xr.Dataset.from_dict(state["static"]))
        obj._attrs = state["attrs"]
        obj._time_attrs = state.get("time_attrs", {})
        obj._bounds_name = state.get("bounds_name", "time_bounds")
        names = state.get("variable_names")
        obj._variable_names = None if names is None else set(names)
        if set(obj._sums) != set(obj._valid_ns):
            raise ValueError("Accumulator state has inconsistent variable statistics.")
        if obj._month is None:
            if obj._sums or obj._coverage_ns or obj._start is not None or obj._end is not None:
                raise ValueError("Empty accumulator state contains pending statistics.")
        elif obj._start is None or obj._end is None or obj._coverage_ns <= 0:
            raise ValueError("Pending accumulator state is missing its interval metadata.")
        return obj
