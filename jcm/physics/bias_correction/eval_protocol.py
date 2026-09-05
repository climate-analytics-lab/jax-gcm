"""What an evaluation run is scored over, and how a long one is reduced.

Two halves. Which observed years the reference averages, which is the older and
more dangerous half, and how the scored span is cut into windows and the run
into segments, which is what makes a run longer than seven years possible.

Small enough to look trivial, load-bearing enough to be worth its own module.
Every reported number in this project comes from one specific protocol: a
450-day free run from 1 January 2001, first 90 days discarded, scored against a
**2001-only** ERA5 reference with 2002 as an out-of-sample spot check.

A 450-day run does spill about 85 days into 2002, so deriving the reference
span from the run length -- which is the obvious thing to do when generalising
the script to score a held-out period -- produces a 2001-2002 average instead.
That is arguably more faithful to the run, and it silently changes every figure
in the paper, poster, deck and docx. It also fails quietly, because a cached
reference file masks it entirely.

So the canonical protocol is pinned here, the derived span is used only for
other periods, and both are covered by tests. The logic lives in the package
rather than in the evaluation driver because that script chdirs and runs a
model at import time, which makes it untestable. The same argument brought the windowing and
the season-mean accumulator here later: a segmented run has to produce the
means an unsegmented one would, and that is not something to leave untested.
"""

from __future__ import annotations

import numpy as np

# The WeatherBench2 store ends 10 January 2023, so 2022 is the last year with a
# full climatology. Averaging a ten-day stub into a reference would bias it
# toward early January.
LAST_FULL_YEAR = 2022

# The canonical protocol every reported figure was produced with.
CANONICAL_YEAR = 2001
CANONICAL_DAYS = 450.0
CANONICAL_REF_YEARS = (2001, 2001)
CANONICAL_REF2_YEAR = 2002


def is_canonical(year: int, total_days: float) -> bool:
    """Is this the protocol every reported number was produced with?"""
    return year == CANONICAL_YEAR and float(total_days) == CANONICAL_DAYS


def resolve_reference(year: int, total_days: float, *, drop_days: float = 90.0,
                      ref_years_env: str | None = None,
                      ref2_year_env: str | None = None,
                      last_full_year: int = LAST_FULL_YEAR):
    """Pick the ERA5 reference span and out-of-sample spot check for a run.

    Args:
        year: calendar year the run starts on, 1 January.
        total_days: run length including spin-up.
        drop_days: spin-up discarded before scoring.
        ref_years_env: explicit override, ``"2016-2022"`` or ``"2018"``.
        ref2_year_env: explicit spot-check override, or ``"none"`` to disable.
        last_full_year: last year the source store covers completely.

    Returns:
        ``(ref_years, ref2_year, canonical)`` where ``ref_years`` is an
        inclusive ``(first, last)`` pair and ``ref2_year`` may be ``None`` when
        the store has no year to spare.

    """
    canonical = is_canonical(year, total_days)

    if ref_years_env:
        lo, _, hi = ref_years_env.partition("-")
        ref_years = (int(lo), int(hi or lo))
    elif canonical:
        ref_years = CANONICAL_REF_YEARS
    else:
        ref_years = (year + int(drop_days // 365),
                     min(year + int((float(total_days) - 1) // 365),
                         last_full_year))

    if ref2_year_env:
        ref2 = (None if ref2_year_env.strip().lower() == "none"
                else int(ref2_year_env))
    elif canonical:
        ref2 = CANONICAL_REF2_YEAR
    else:
        ref2 = (ref_years[1] + 1
                if ref_years[1] + 1 <= last_full_year else None)

    return ref_years, ref2, canonical


def reference_label(ref_years: tuple[int, int]) -> str:
    """Render a reference span for logs: ``2001`` or ``2016-2022``."""
    lo, hi = ref_years
    return str(lo) if lo == hi else f"{lo}-{hi}"


def period_suffix(year: int, total_days: float) -> str:
    """Filename suffix tying a cached product to its evaluated period.

    Empty for the canonical protocol so the existing ``eval_out/`` files stay
    valid; anything else is suffixed so a holdout run launched without
    ``M4_OUT_DIR`` cannot overwrite a canonical file with fields from a
    different period.
    """
    if is_canonical(year, total_days):
        return ""
    return f"_{year}_{int(total_days)}d"


# The model runs on a 365-day calendar, so a "year" of run length is exact.
DAYS_PER_YEAR = 365

# How long a scored window is by default. Seven years is the holdout span every
# reported holdout number used, and it is also the block size a longer run gets
# cut into, so the blocks of a long run are directly comparable to it.
BLOCK_YEARS = 7


# The shortest scored span the season masks can be read off. The canonical
# protocol scores 360 days, which reaches every day of the year once, so it is
# both the shortest run we have ever scored and the natural floor. Anything
# shorter leaves DJF or JJA with no samples at all, which used to average to
# NaN and print as though it were a number.
MIN_SCORED_DAYS = 360.0


def too_short_to_score(total_days: float, *, drop_days: float = 90.0,
                       minimum: float = MIN_SCORED_DAYS):
    """Explain why a run cannot be scored, or return None if it can.

    Checked before the model runs. The failure it prevents otherwise surfaces
    as a missing season after the whole rollout has been paid for.
    """
    scored = float(total_days) - float(drop_days)
    if scored >= minimum:
        return None
    return ("a %g-day run leaves %.0f days after the %g-day spin-up drop, and "
            "the seasonal windows need at least %g. Raise M4_DAYS to %g or "
            "more." % (total_days, scored, drop_days, minimum,
                       minimum + drop_days))


def scoring_blocks(total_days: float, *, drop_days: float = 90.0,
                   block_years: int = BLOCK_YEARS,
                   days_per_year: int = DAYS_PER_YEAR):
    """Split the scored part of a run into equal, non-overlapping windows.

    A metric read off a 7-year free-running climatology carries sampling noise
    from the model's own internal variability, and we have no direct measure of
    how large that is: reseeding training moves the scores, but a reseed changes
    the network too, so the two causes are confounded. Cutting ONE run into
    consecutive blocks separates them. The blocks share a network, so their
    spread is sampling noise alone, and it is the floor any claim about one term
    beating another has to clear.

    The blocks start after the spin-up drop and each covers a whole number of
    calendar years, so every block sees the same seasons the same number of
    times. A trailing remainder shorter than a block is left unscored rather
    than folded into the last block, which would give it a different seasonal
    composition from the others.

    Args:
        total_days: run length including spin-up.
        drop_days: spin-up discarded before scoring.
        block_years: block length in 365-day years.
        days_per_year: days per model year.

    Returns:
        Tuple of ``(start_day, end_day)`` pairs measured from the run start,
        end exclusive. Empty when fewer than two whole blocks fit, because one
        block is just the run itself and measures no spread.

    """
    span = float(block_years * days_per_year)
    scored = float(total_days) - float(drop_days)
    n = int(scored // span)
    if n < 2:
        return ()
    return tuple((drop_days + i * span, drop_days + (i + 1) * span)
                 for i in range(n))


def chunk_spans(total_days: float, chunk_days: float, save_days: float):
    """Split a run into segments to integrate one at a time.

    A 21-year run saved every 5 days is 1533 snapshots held at once, which is
    why the run is done in segments and reduced to per-season sums as it goes.
    ``Model.resume`` threads the cross-step physics carry, so the segments are
    one continuous trajectory rather than a set of restarts.

    Spans are equal to within one save interval so the integrator compiles for
    at most two segment lengths. ``chunk_days <= 0`` means one segment, which is
    what the 450-day and 2645-day protocols use: their published products came
    from a single call and there is no reason to perturb them by roundoff.

    Raises:
        ValueError: if the run length is not a whole number of save intervals,
            which would silently truncate the tail.

    """
    units = round(float(total_days) / float(save_days))
    if abs(units * float(save_days) - float(total_days)) > 1e-6:
        raise ValueError(
            f"total_days={total_days} is not a whole number of {save_days}-day "
            "save intervals; the model would drop the remainder")
    if chunk_days is None or float(chunk_days) <= 0:
        return (float(total_days),)
    per_chunk = max(1, int(round(float(chunk_days) / float(save_days))))
    n = max(1, -(-units // per_chunk))
    base, extra = divmod(units, n)
    return tuple((base + (1 if i < extra else 0)) * float(save_days)
                 for i in range(n))


# Day-of-year windows on the 365-day calendar. "annual" is every sample, so a
# season here is a filter rather than a partition.
SEASONS = {
    "annual": lambda doy: np.ones_like(doy, dtype=bool),
    "djf": lambda doy: (doy >= 334) | (doy < 59),
    "jja": lambda doy: (doy >= 151) & (doy < 243),
}

# Humidity is only ever reported annually, so the seasonal sums skip it.
VARS = {"annual": ("temperature", "specific_humidity"),
        "djf": ("temperature",),
        "jja": ("temperature",)}


class SeasonMeans:
    """Per-season time means over one or more windows, built segment by segment.

    Holds running sums instead of snapshots, so a long run never has to be in
    memory at once: a 21-year run saved every 5 days is 1533 snapshots. Summing
    then dividing gives the same mean a single-segment run computes, to
    roundoff, which is what lets a long run be cut into segments at all.

    Windows are ``(name, first_day, last_day)`` with ``last_day`` exclusive,
    measured in elapsed run days. The caller supplies those days with each
    segment rather than having them derived here, because anchoring them on the
    run's start date is the script's business and getting it wrong has bitten
    this project before (the July 9 audit: absolute datetimes cast to float
    disabled both the spin-up drop and the season masks).
    """

    def __init__(self, windows, seasons=None, variables=None):
        """Accumulate over ``windows``; season and variable tables are overridable."""
        self.windows = tuple(windows)
        self.seasons = SEASONS if seasons is None else seasons
        self.variables = VARS if variables is None else variables
        self._sum = {}
        self._n = {}

    def add(self, ds, days):
        """Fold one segment in. ``days`` is elapsed run days per time sample."""
        days = np.asarray(days, dtype=float)
        doy = days % float(DAYS_PER_YEAR)
        for name, first, last in self.windows:
            in_window = (days >= first) & (days < last)
            for season, mask_fn in self.seasons.items():
                idx = np.where(in_window & mask_fn(doy))[0]
                if idx.size == 0:
                    continue
                for var in self.variables[season]:
                    part = ds[var].isel(time=idx).sum("time")
                    # Pull it off the device now. Keeping a lazy handle would
                    # pin the whole segment, which is the memory this class
                    # exists to avoid.
                    part = part.copy(data=np.asarray(part.data))
                    key = (name, season, var)
                    prev = self._sum.get(key)
                    self._sum[key] = part if prev is None else prev + part
                    self._n[key] = self._n.get(key, 0) + idx.size

    def mean(self, window, season, var):
        """Time mean of ``var`` over ``season`` within ``window``."""
        key = (window, season, var)
        if key not in self._sum:
            raise KeyError("no %s samples of %s in window %r"
                           % (season, var, window or "full"))
        return self._sum[key] / self._n[key]

    def samples(self, window, season="annual", var="temperature"):
        """How many time samples went into one mean. 0 if the window is empty."""
        return self._n.get((window, season, var), 0)
