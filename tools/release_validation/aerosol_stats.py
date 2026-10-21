"""Aerosol regression statistics for a finished JAM run (#762).

Reduces a year-long JAM run directory to a small set of scalars that a
release gate can score, and applies the two tolerance tiers described in
``docs/source/design/jam_regression.md``:

* **absolute physics gates** — the exponential drift ``|d ln B/dt|`` of every
  species' burden (a straight line over the final six months of a record
  shorter than a year, held to 0.002 /day; over its final year once it covers
  one, fit jointly with the annual harmonic because the sources are seasonal,
  and held to 0.003 /day, 3 sigma of that fit's scatter on stationary years), the
  mass-budget residual, and the
  per-step dynamics residual from the #713 in-step gauge. These are the runaway
  detector: an aerosol runaway grows multiplicatively with the meteorology
  entirely normal, so a climatological range gate does not notice it until the
  final fortnight of a year that was unusable months earlier.

  Which window a record gets is not left to how much of it happens to be
  loaded: a **warm-started or second year** (:func:`warm_start_kind`, detected
  from the run's own ``.hydra/config.yaml``, or an explicit ``--warm-start``)
  is always scored over its *whole* record, ignoring an operator's
  ``--last-n`` — the six-month window exists to skip a cold start's spin-up
  ramp, which a warm start never has (:func:`resolve_warm_start`). For **POA
  and BC** — the two species with a prescribed (not interactive) emission —
  the whole-year slope additionally has a periodic box model's own fit
  artefact subtracted before it is gated (:func:`box_model_drift`); both the
  raw and the box-corrected slope are reported. Dust and sea salt are
  unaffected: their emission is interactive, so a box driven by "the run's own
  emission series" would not be driven by an external forcing at all. See
  ``docs/source/design/jam_regression.md`` ("Which window, and the box-model
  correction").
* **climatological anchors** — the shared ``jam_burden_report._SPECIES``
  ranges widened by the release gate's slack, unchanged.
* **regression tolerances** — ``max(3 sigma, 15 %)`` against a reference run
  (``--reference``), with ``sigma`` the standard error of the annual mean
  estimated from the chunk series' own lag-1 autocorrelation. No reference is
  vendored: the reference has to be a run that passes the absolute gates
  above, and none does yet (see the design doc's "Not yet done").

Everything is computed chunk by chunk: a year of T63 L47 JAM output is ~60 GB
and must never be held open as one array. :func:`collect` reduces each chunk
file to scalars and returns the 5-day series; :func:`summarize` turns that
series into the statistic set.

Both output vertical conventions are supported. Levels are chosen by their
*pressure*, read from the file's own ``pressure_full``, never by a bare index
(see the "Inspecting model output" rule in ``CLAUDE.md``), and the layer
thicknesses come from :func:`jam_burden_report._layer_dp`, which orients
pre-#710 TOA-first interfaces onto the surface-first ``level`` axis.

**Two different windows, read from the same chunk files (#1010).** The drift,
ledger and mean statistics above are scored on whatever window the caller
asks for (``--last-n``, or the whole run directory without it) — that is the
"settled window" the design doc means by it. The *annual-budget* statistics
(currently just ``dust_emission_tg_per_yr``; see :data:`MIN_DUST_WINDOW_DAYS`)
are drawn on a full T63 year and are scored on the WHOLE run directory's
trailing year (:func:`annual_window_bounds`) regardless of ``--last-n``: the
documented ``health.py --last-n 40`` recipe is a 200-day window, below
:data:`MIN_DUST_WINDOW_DAYS`, so scoring the annual statistic on it rather
than on the run's actual year reported it UNSCORED on every confirmation-year
report of the #682 retune. :func:`collect_windows` and :func:`slice_windows`
read or slice both windows from one record and hand each to
:func:`summarize`/:func:`unscored_gates` as a separate ``annual_*`` triple;
``main`` below is the one caller that needs this, and ``health.py`` uses the
same two functions for its own aerosol block.

Usage:
    python tools/release_validation/aerosol_stats.py <run_dir>
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Source-checkout bootstrap: repo root (for ``jcm``) and tools/ (for the
# sibling ``jam_burden_report``) before the imports that need them.
_TOOLS = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jcm.analysis import area_weights, column_integral, global_mean  # noqa: E402
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES  # noqa: E402
from jcm.physics.aerosol.jam.microphysics import MAM4_SPEC  # noqa: E402
from jam_burden_report import (  # noqa: E402
    _POPULATIONS,
    _SPECIES,
    _layer_dp,
    burden,
    sulfate_molar_mass,
)

#: Species carrying a lifetime statistic — those whose removal is dominated by
#: the two published deposition ledgers (``dry_*`` sedimentation and ``wet_*``
#: scavenging, which already includes in-plume convective removal).
LIFETIME_SPECIES = ("so4", "bc", "du", "ss")

#: Species whose emitted mass is a complete source inventory, so
#: ``emi - dep - dB`` is expected to close. ``soa`` is excluded: its source is
#: condensation of the SOAG gas, which has no ``emi_*`` channel, so a residual
#: computed for it would measure a missing diagnostic, not a mass leak.
#: ``moa`` is excluded for a different reason — it has emission and deposition
#: ledgers but no entry in the shared ``_SPECIES`` anchor table, so no burden
#: is computed for it and there is no storage term to close against. Adding it
#: needs a climatological anchor range, which is a calibration decision.
#: ``oc`` is M7's whole organic aerosol, emitted as primary (HAM's default
#: emits biogenic SOA as primary too), so ``emi_oc`` is its complete source; a
#: MAM4 run carries no ``oc`` tracer and is skipped as not carrying it.
PRIMARY_BUDGET_SPECIES = ("bc", "du", "ss", "poa", "oc")

#: Species whose emission is PRESCRIBED — an external climatology, not a
#: function of this run's own winds, soil moisture or SST — and so the only
#: ones #1081's periodic-box correction (:func:`box_model_drift`) applies to.
#: POA and BC both come from the MACv2-SP/AeroCom biomass-burning and
#: fossil/biofuel emission climatology; dust and sea salt are emitted
#: INTERACTIVELY (saltation on the model's own 10 m wind and soil wetness for
#: dust, whitecap fraction on wind and SST for sea salt), so a box driven by
#: "the run's own emission series" would be driven by the run's own dynamics,
#: not an external forcing the box model can honestly claim to replay — the
#: maintainer's explicit decision is to correct only these two (#1081).
BOX_CORRECTED_SPECIES = ("poa", "bc")

#: Gas reservoirs and emission channels of the sulfur family. The gases are
#: stored as raw column burdens by :func:`chunk_reduction`; the conversion to
#: sulfate (:func:`sulfur_carriers`) is applied in the ledger, so the
#: reduction stays unit-neutral.
_S_FAMILY_GASES = ("so2", "dms", "h2so4")
_S_FAMILY_EMIS = ("so2", "dms", "so4")


def sulfur_carriers(m_sulfate: float) -> dict[str, float]:
    """Kilograms of the run's sulfate per kilogram of each sulfur carrier.

    One sulfur atom per molecule throughout, so each factor is a ratio of
    molar masses, which lets the whole family be totalled in the unit the
    sulfate burden and deposition ledgers are already in. The sulfate's molar
    mass ``m_sulfate`` [kg/mol] is the population's, read from the output by
    :func:`jam_burden_report.sulfate_molar_mass`: MAM4's ``so4`` tracer is
    ammonium bisulfate (115 g/mol, so SO2 converts at 1.795) and M7's is SO4
    (96.0631 g/mol, 1.499). The model converts with the population's value, so
    a ledger built on the other one misstates every carrier by their 19.7 %
    ratio and reads that as a leak or a source.
    """
    factors = {gas: m_sulfate / GAS_SPECIES[gas].molar_mass
               for gas in _S_FAMILY_GASES}
    factors["so4"] = 1.0
    return factors


def _record_sulfate_molar_mass(series) -> float | None:
    """Return the one sulfate molar mass every chunk of the record carries.

    ``None`` when any chunk could not be attributed to a population (a chunk
    with no or mixed sulfate tracers) or when chunks disagree: a record
    spanning two populations has no single sulfate unit to close the family
    in.

    A series cached (``--series-out``) before the reduction recorded the unit
    has no ``so4_molar_mass`` at all. The tools that wrote those knew only
    MAM4's species until M7's ``oc`` joined them, so such a record with a
    sulfate burden but no ``burden_oc`` key is MAM4's and closes in MAM4's
    sulfate, as it always did; one carrying ``burden_oc`` stays unscored.
    """
    m = series.get("so4_molar_mass")
    if m is None:
        if "burden_so4" in series and "burden_oc" not in series:
            return MAM4_SPEC.species_props("so4").molar_mass
        return None
    m = np.asarray(m, dtype=float)
    if m.size == 0 or not np.all(np.isfinite(m)) or np.ptp(m) != 0.0:
        return None
    return float(m[0])


#: The population each anchor species belongs to (an index into
#: ``_POPULATIONS``), or ``None`` for one both carry: a species is one
#: population's when every mode it lists is that population's (MAM4's ``poa``
#: and ``soa``, M7's ``oc``); ``so4``, ``bc``, ``du`` and ``ss`` are both's.
_SPECIES_POPULATION = {
    species: next((i for i, spec in enumerate(_POPULATIONS)
                   if set(modes) <= set(spec.mode_shorts)), None)
    for species, (modes, _range) in _SPECIES.items()}


def other_population_species(series: dict) -> set[str]:
    """Return the anchor species of a population the record does not carry.

    Their absence is a different model, not a missing measurement, so it is
    neither an unscored gate nor a NOTE: a MAM4 run has no ``oc`` and an M7
    run no ``poa``/``soa``. Read from which population's own species the
    record has burdens of; empty when it has none (nothing to tell the two
    apart by, so every absence is still reported).
    """
    carried = {pop for species, pop in _SPECIES_POPULATION.items()
               if pop is not None and f"burden_{species}" in series}
    if not carried:
        return set()
    return {species for species, pop in _SPECIES_POPULATION.items()
            if pop is not None and pop not in carried}

#: Dynamics-conservation gate. ``budget_dyn_<sp>`` (#713) is the mass the
#: transport created or destroyed per step, so ``|dyn|*dt/mass`` is the
#: fractional error semi-Lagrangian advection commits each step. Set AT that
#: error's honest magnitude, so it trips on a transport error that has become
#: systematic rather than on ordinary interpolation noise, and ~500x tighter
#: than the [2/3, 1.5] clip on the proportional mass fixer.
DYN_RESIDUAL_PER_STEP = 1.0e-3

#: Absolute physics gates (see the design doc). Both drift limits read burdens
#: only, so they stay trustworthy when the deposition ledger is incomplete. The
#: limit that applies to a slope is set by the fit that produced it
#: (:func:`drift_limit_per_day`), because the two fits scatter differently on a
#: stationary burden.
#:
#: ``DRIFT_LIMIT_PER_DAY`` is the limit for a slope fit over a year of record
#: with the annual harmonic fitted jointly (:func:`fit_log_drift`). Over exactly
#: one cycle a trend cannot be told apart from the cycle's non-sinusoidal part
#: and its interannual variability, so the slope of a stationary burden
#: scatters even with the harmonic removed: over 30 species-years of
#: equilibrated runs it lies in -0.0023..+0.0015 /day with RMS 0.00098 (a plain
#: whole-year line: -0.0021..+0.0008, RMS 0.00069). The limit is 3 sigma of that
#: harmonic-fit scatter; 2 sigma (0.002) is exceeded by noise alone in about one
#: stationary species-year in 24, 3 sigma in about one in 450 (Gaussian). It
#: still admits no more than a factor e over ~330 days, and a x60-in-40-days
#: runaway fits 0.0082 /day over the final year, 2.7x the limit. The year-over-year
#: drift of the species-year that read furthest out (POA, -0.0023 /day with the
#: harmonic) is -0.00007 /day, so the scatter is the fit's, not the burden's.
DRIFT_LIMIT_PER_DAY = 0.003

#: The limit for a straight-line slope: a record shorter than a year, scored over
#: its final six months, and a year-long window with too few finite samples to
#: identify the harmonic (:func:`fit_log_drift` falls back to the line there, and
#: the 3-sigma basis above is the harmonic fit's). It admits a factor e over ~500
#: days — slower than any runaway, faster than a burden that has genuinely
#: settled — and 3 sigma of the line's own fit noise falls below it only past ~90
#: days (``MIN_WINDOW_DAYS``).
SHORT_RECORD_DRIFT_LIMIT_PER_DAY = 0.002
BUDGET_RESIDUAL_LIMIT = 0.05

#: The closure gate's sign carries the diagnosis. NEGATIVE means more mass was
#: deposited than entered the column, which no missing ledger entry can produce
#: — it is mass creation. POSITIVE means emitted mass is unaccounted for, which
#: a missing sink *diagnostic* looks exactly like: before the #722 ledger fix
#: ``dry_*`` omits Slinn deposition, so a positive residual on such output is
#: expected rather than evidence of a leak.
_POSITIVE_RESIDUAL_CAVEAT = (
    "unaccounted emission; on output written before the #722 removal-ledger "
    "fix, dry_* omits Slinn dry deposition and reads positive by construction")

#: Shortest window on which the drift and closure statistics are scored. A
#: least-squares slope has noise ``sigma_resid / (dt * sqrt(N(N^2-1)/12))``; at
#: the 5-day output cadence and the ~0.07 log-burden scatter of a settled
#: species, 3 sigma of that falls below ``SHORT_RECORD_DRIFT_LIMIT_PER_DAY``
#: only past ~90 days. On a shorter fit the statistic is noise, so it is
#: reported UNSCORED rather than gated — the same "not measurable is not
#: scored" rule the drift statistic already applies to a species the run does
#: not carry. The closure residual shares the limit: its storage term is one
#: endpoint difference, which over a few chunks swamps the flux integral it is
#: compared against.
MIN_WINDOW_DAYS = 90.0

#: Shortest window on which the annual dust-emission budget is scored. Dust
#: emission is strongly seasonal, so annualising a partial year measures the
#: season rather than the year. A scored member has to cover most of one.
MIN_DUST_WINDOW_DAYS = 300.0

#: Earth's surface area [m²] — turns a global-mean flux into a global total.
EARTH_AREA_M2 = 4.0 * np.pi * 6.371e6 ** 2

#: Release band on the annual D < 10 µm dust emission [Tg/yr] — the size range
#: jcm's MAM4 population emits (coarser mass is discarded and published
#: separately as ``dust_supercoarse_flux``); for the M7 population the same
#: statistic is HAM's own window, whose references are HAM2.3's budgets
#: directly (see ``docs/source/science/aerosol.md``). The parent model converted to this window is
#: the lower anchor: ECHAM6.3-HAM2.3 emits 1221 Tg/yr present-day (Krätschmer
#: et al. 2022), which is the mass reaching M7 — tracers 1-4 of ``mo_ham_dust``,
#: everything below 15.887 µm — and 47.4 % of that window is the 10-15.887 µm
#: slice this port does not carry, so the comparable budget is 642 Tg/yr
#: present-day and 485 pre-industrial. The shipped dust threshold scale is
#: fitted to the observed dust AOD rather than to that budget (JAM aerosol
#: retune, ``docs/source/design/jam_aerosol_retune.md``) and a T63 year emits
#: 2351 Tg/yr, above the AeroCom phase-I median of 1640 (Textor et al. 2006;
#: Huneeus et al. 2011 give 1123 for their 15 models, at different size
#: cut-offs). The band spans the converted budget and that year; its upper edge
#: is 1.11 times the calibrated year, against a run-to-run spread of about 6 %
#: (two free-running members that differ only in start date), so the calibrated
#: configuration passes with a margin of under two spreads. The band is a check
#: on vanishing and runaway dust, not a tuning target. The derivation and the
#: literature it is read against are in ``docs/source/science/aerosol.md``.
#: It is the check that dust has neither vanished (HAM's untuned threshold
#: gives jcm 5.7 Tg/yr, #808) nor run away (the same model's LGM run emits
#: 5159 Tg/yr in HAM's size window, about 2700 in this one). The documented
#: ``health.py --last-n 40`` recipe's 200-day window is shorter than
#: ``MIN_DUST_WINDOW_DAYS``, but the statistic is scored anyway: it is built
#: from the WHOLE run directory's trailing year, not from the ``--last-n``
#: slice (:func:`annual_window_bounds`, #1010) — the band is drawn on a full
#: T63 year, including its first months, so the cold-start ramp is part of
#: the year the band was calibrated on rather than a transient to discard.
DUST_EMISSION_TG_PER_YR = (400.0, 2600.0)

#: T63's Gaussian latitude count. The dust band is a T63 calibration (#810:
#: the HAMMOZ source maps exist only at T63), so it is scored only there.
_T63_NLAT = 96

#: Regression tolerance: whichever of a 3-sigma excursion and a 15 % relative
#: change is larger. 3 sigma alone is far too tight for a well-sampled mean
#: (sigma is ~1 % of the annual mean); 15 % alone would let a slow bias
#: through on a noisy statistic.
_N_SIGMA = 3.0
_REL_TOLERANCE = 0.15

#: Statistics the regression tier does NOT score: each has an absolute physics
#: gate of its own, and scoring them twice would replace that gate with a
#: tighter one. For the drift, budget and dynamics residuals the references
#: are ~1e-3, so a 15 % relative tolerance is tighter than the gate and would
#: fail every real run. ``dust_emission_tg_per_yr`` is exempt for the opposite
#: reason: its band is deliberately six times wide (#808), because the target
#: itself is uncertain, and comparing to one reference run at 15 % would turn
#: a non-tuning release gate into a tuning target.
_GATED_PREFIXES = ("dlnB_dt_", "budget_residual", "dyn_frac_per_step_",
                   "dust_emission_tg_per_yr")

#: Bookkeeping entries of the statistic set that describe the record rather than
#: measure the model (matched by prefix): a run scored over a different window
#: than its reference, or by a different fit, is not a regression.
#: ``dust_window_days`` names the annual window ``dust_emission_tg_per_yr``
#: (itself already exempt, via ``_GATED_PREFIXES``) was scored on (#1010).
#: ``raw_dlnB_dt_``/``box_dlnB_dt_`` (#1081) are the UNCORRECTED slope and the
#: box model's own fit artefact that ``dlnB_dt_<sp>_per_day`` (already exempt,
#: above) is built from for POA/BC -- companions of an already-exempt gate,
#: not an independent measurement of anything, so scoring them against a
#: reference would be scoring the same slope twice under two different names.
_RECORD_METADATA = ("record_days", "drift_harmonic_", "dust_window_days",
                    "raw_dlnB_dt_", "box_dlnB_dt_")

#: Absolute tolerance floors [statistic units], so a reference that is legibly
#: zero (an unused species' burden) does not collapse the tolerance to zero and
#: fail on any nonzero value. Matched by prefix, longest first.
_TOLERANCE_FLOORS = {
    "burden_": 0.05,            # mg/m2
    "so4_burden_": 0.05,        # mg/m2
    "lifetime_": 0.1,           # days
    "aod_550": 0.002,
    "angstrom": 0.02,
    "cdnc_": 1.0,               # cm-3
    "N100_": 1.0,               # cm-3
    "r_dry_": 0.001,            # um
    "so4_frac_above_500hPa": 0.01,
    "so4_nh_sh_ratio": 0.05,
}

#: Pressure [Pa] used for the near-surface aerosol-number diagnostics, and the
#: divide separating the free troposphere reservoir from the boundary layer.
_CDNC_PRESSURE = 90000.0
_UPPER_LEVEL_PRESSURE = 50000.0

_AOD_KEYS = ("jam_optics.aod_550", "jam_band_optics.aod_550")
_ANGSTROM_KEYS = ("jam_optics.angstrom", "jam_band_optics.angstrom",
                  "ang4487aer")


#: ``np.trapezoid`` is numpy >= 2.0; ``np.trapz`` is its pre-2.0 spelling.
#: numpy is unpinned in requirements.txt, so accept either.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def chunk_centres(days: np.ndarray, start: float | None = None) -> np.ndarray:
    """Mid-times of the averaging windows whose labels are ``days``.

    Output chunks are labelled by their END day and hold time *averages*, so a
    chunk mean belongs at the window's centre. Each window starts at its
    predecessor's label; the first starts at ``start`` — the label of the chunk
    just before the record, which a ``--last-n`` slice must pass along, or day
    0 for a record that begins at the run start. Guessing 0 for a sliced
    record puts the first node at ``days[0] / 2`` and stretches the first
    trapezoid over most of the run, which mis-weights the first flux sample by
    an order of magnitude and fabricates a closure error.
    """
    return 0.5 * (_window_starts(days, start) + np.asarray(days, dtype=float))


def _window_starts(days: np.ndarray, start: float | None) -> np.ndarray:
    """Start day of each averaging window: its predecessor's label.

    The first window starts at ``start``. ``None`` means infer it: a record
    whose chunks are uniformly spaced but whose first label exceeds that
    spacing evidently does not begin at day 0 — a resumed run writing into a
    fresh output directory, or early chunks deleted to save disk — so its
    first window starts one cadence before its first label. The last chunk may
    legitimately be short, so it is left out of the uniformity test; a record
    too short or too irregular to judge keeps day 0, which is exact for a run
    start and only ever an approximation where no caller knows better.
    """
    days = np.asarray(days, dtype=float)
    if start is None:
        start = 0.0
        gaps = np.diff(days)
        if gaps.size >= 3 and np.allclose(gaps[:-1], gaps[0]) \
                and days[0] > gaps[0] * (1 + 1e-9):
            start = days[0] - gaps[0]
    return np.concatenate([[float(start)], days[:-1]])


def chunk_durations(days: np.ndarray, start: float | None = None
                    ) -> np.ndarray:
    """Length [days] of each averaging window whose labels are ``days``.

    A time mean over chunks of unequal length weights by these rather than
    counting them equally: ``run/longrun.yaml`` writes twelve 30-day chunks
    and a final 5-day one, and for a strongly seasonal quantity the short tail
    chunk carrying a full month's weight moves the annual mean.
    """
    return np.asarray(days, dtype=float) - _window_starts(days, start)


def _uneven_window_reason(days: np.ndarray,
                          window_start: float | None) -> str | None:
    """Why this record's averaging windows cannot be duration-weighted.

    Duration weighting assumes each chunk's value averages the window between
    its own label and its predecessor's. A *deleted* interior chunk breaks
    that silently: the next file still averages only its own cadence, but its
    window now spans the gap, so it would be weighted twice over. No NaN
    appears, so the finiteness check cannot see it. jcm writes a fixed
    ``chunk_days`` with at most a short final chunk, so an interior window
    that is not the common cadence — or a final one longer than it — means a
    missing file, and the annual budget is reported unscored rather than
    computed from a record with a hole in it.
    """
    durations = chunk_durations(days, window_start)
    if durations.size < 3:
        return None
    cadence = float(np.median(durations[:-1]))
    if cadence <= 0.0:
        return "chunk labels are not increasing, so no averaging window " \
               "can be derived"
    odd = ~np.isclose(durations[:-1], cadence, rtol=1e-6)
    if odd.any() or durations[-1] > cadence * (1.0 + 1e-6):
        where = float(days[int(np.argmax(odd))]) if odd.any() else float(
            days[-1])
        return (f"the chunk cadence is {cadence:.0f} days but the window "
                f"ending on day {where:.0f} spans "
                f"{float(durations[int(np.argmax(odd)) if odd.any() else -1]):.0f}"
                " — a chunk file is missing, and the budget of a record with "
                "a hole in it is not this run's year")
    return None


def covered_days(days: np.ndarray,
                 window_start: float | None = None) -> float:
    """Days of simulation a record actually covers, first window included.

    ``days[-1] - days[0]`` is the distance between chunk LABELS, which drops
    the first window: a complete 300-day run written in 30-day chunks is
    labelled 30…300 and spans only 270 by that measure. For a threshold that
    decides whether a year is long enough to annualise, that difference
    decides whether the gate runs at all.
    """
    if np.asarray(days).size == 0:
        return 0.0
    return float(np.sum(chunk_durations(days, window_start)))


def chunk_day(path, index: int | None = None) -> float | None:
    """Return the END day a chunk file is labelled with (``*_day<N>*.nc``).

    Read from the file NAME only: a run directory such as ``spinup_day0/``
    would otherwise label every chunk 0. Names without a day fall back to
    ``index`` (the file's position, so a record of unlabelled chunks keeps a
    monotone abscissa) or to ``None`` when no position is given.
    """
    match = re.search(r"day(\d+)", Path(str(path)).name)
    if match:
        return float(match.group(1))
    return None if index is None else float(index)


def _finite_mean(values) -> float:
    """Mean of the finite samples, NaN when there are none (no empty-slice warning)."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def positive_int(text: str) -> int:
    """Parse ``--last-n``; a negative count would silently invert the slice."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("--last-n must be a positive chunk count")
    return value


def is_jam_run(ds: xr.Dataset) -> bool:
    """Report whether ``ds`` carries JAM's prognostic modal aerosol tracers.

    Detected from the variables present rather than from a config file, so a
    run directory alone is enough. The test is the interstitial mass tracers
    alone: they are what every statistic here is built from, and a MACv2-SP run
    (which publishes AOD but no ``m_<sp>_<mode>``) has none. Requiring a
    ``jam_*`` diagnostic namespace as well would let a JAM run with a trimmed
    output set skip the aerosol gates silently, which is the one outcome this
    block exists to prevent — a trimmed run is a misconfiguration to name (see
    :func:`missing_jam_diagnostics`), not a reason to skip.
    """
    return any(re.fullmatch(r"m_[a-z0-9]+_[a-z]+", str(v))
               for v in ds.data_vars)


def missing_jam_diagnostics(ds: xr.Dataset) -> list[str]:
    """JAM diagnostic namespaces absent from a run that carries JAM tracers.

    Their absence does not stop the burdens being computed, but it does silently
    drop the cloud-borne phase, the optics and the modal radii from the report,
    so it is named rather than left to be inferred from missing rows.
    """
    present = {str(v).split(".", 1)[0] for v in ds.data_vars if "." in str(v)}
    wanted = ("jam_cloud_borne", "jam_state")
    optics = ("jam_optics", "jam_band_optics")
    missing = [n for n in wanted if n not in present]
    if not any(n in present for n in optics):
        missing.append("/".join(optics))
    return missing


def _band_indices(ds: xr.Dataset, lo: float, hi: float) -> np.ndarray:
    lat = np.asarray(ds["lat"].values, dtype=float)
    return np.nonzero((lat >= lo) & (lat <= hi))[0]


def _band_mean(da: xr.DataArray, weights: xr.DataArray,
               ds: xr.Dataset, lo: float, hi: float) -> float:
    """Area-weighted mean of ``da`` over the latitude band [lo, hi]."""
    idx = _band_indices(ds, lo, hi)
    if idx.size == 0:
        return float("nan")
    return float(global_mean(da.isel(lat=idx), weights.isel(lat=idx)))


def nearest_pressure_level(ds: xr.Dataset, pressure: float) -> int:
    """Index on the ``level`` axis whose mean pressure is nearest ``pressure``.

    The index is *derived* from the file's own ``pressure_full`` rather than
    assumed, which is what makes the selection valid under either vertical
    convention.
    """
    pf = ds["pressure_full"]
    prof = pf.mean([d for d in pf.dims if d != "level"])
    return int(np.argmin(np.abs(np.asarray(prof.values) - pressure)))


def _optional_key(ds: xr.Dataset, keys) -> str | None:
    for key in keys:
        if key in ds:
            return key
    return None


def chunk_reduction(ds: xr.Dataset) -> dict[str, float]:
    """Reduce one output chunk to the scalars the statistics are built from.

    Returns global (and, for sulfate, hemispheric/polar/upper-level) means of
    burdens, the emission and deposition fluxes, and the optics/number
    diagnostics. Time is averaged out here: a chunk is one 5-day sample.
    """
    weights = area_weights(ds)
    dp = _layer_dp(ds)
    out: dict[str, float] = {}

    def tmean(da):
        return da.mean("time") if "time" in da.dims else da

    burdens = {}
    for species, (modes, _range) in _SPECIES.items():
        col = burden(ds, species, modes)
        if col is None:
            continue
        col = col.compute()
        burdens[species] = col
        out[f"burden_{species}"] = float(global_mean(col, weights))

    if "so4" in burdens:
        col = burdens["so4"]
        out["so4_nh"] = _band_mean(col, weights, ds, 0.0, 90.0)
        out["so4_sh"] = _band_mean(col, weights, ds, -90.0, 0.0)
        out["so4_60_90N"] = _band_mean(col, weights, ds, 60.0, 90.0)
        # Upper-level share: mask on the file's own pressure field, so no
        # level ordering is assumed.
        names = [f"{p}_so4_{m}"
                 for m in _SPECIES["so4"][0]
                 for p in ("m", "mc", "jam_cloud_borne.mc")]
        q = sum(ds[n] for n in names if n in ds)
        aloft = q.where(ds["pressure_full"] < _UPPER_LEVEL_PRESSURE, 0.0)
        col_aloft = tmean(column_integral(aloft, dp)) * 1e6
        out["so4_above_500hPa"] = float(global_mean(col_aloft, weights))

    # Sulfur-family gas burdens [mg/m2] as carried, for the closure test; the
    # conversion to sulfate-aerosol mass belongs to the ledger, which needs to
    # know which sulfate the file's ``so4`` tracers carry (MAM4's ammonium
    # bisulfate or M7's SO4). Recorded per chunk, like ``nlat``, so a record
    # stitched from two populations is caught rather than averaged.
    m_sulfate = sulfate_molar_mass(ds)
    if m_sulfate is not None:
        out["so4_molar_mass"] = m_sulfate
    for gas in _S_FAMILY_GASES:
        name = f"g_{gas}"
        if name in ds:
            col = tmean(column_integral(ds[name], dp)) * 1e6
            out[f"gas_{gas}"] = float(global_mean(col, weights))

    # #713 in-step budget gauges: column mass of the ADVECTED tracers and the
    # dynamics residual. Interstitial only (the host never transports the
    # cloud-borne carry), so the two share a denominator.
    for prefix in ("budget_mass", "budget_dyn"):
        for var in ds.data_vars:
            name = str(var)
            if name.startswith(f"{prefix}_") and ds[var].ndim <= 3:
                out[name] = float(global_mean(tmean(ds[var]), weights))

    for prefix in ("emi", "dry", "wet"):
        for var in ds.data_vars:
            name = str(var)
            # ``emi_bb_*`` comes along as a separate key; it is a *split* of
            # ``emi_*``, never added to it.
            if name.startswith(f"{prefix}_") and ds[var].ndim <= 3:
                out[name] = float(global_mean(tmean(ds[var]), weights))

    # Grid size, so the dust band — a T63 calibration — knows whether it
    # applies to this member at all rather than failing a T106 run for a
    # number that was never tuned there (#810).
    out["nlat"] = float(ds.sizes.get("lat", 0))

    aod_key = _optional_key(ds, _AOD_KEYS)
    if aod_key:
        out["aod_550"] = float(global_mean(tmean(ds[aod_key]), weights))
    ang_key = _optional_key(ds, _ANGSTROM_KEYS)
    if ang_key:
        out["angstrom"] = float(global_mean(tmean(ds[ang_key]), weights))

    k_cdnc = (nearest_pressure_level(ds, _CDNC_PRESSURE)
              if "pressure_full" in ds else None)
    if k_cdnc is not None and "activated_cdnc" in ds:
        # m^-3 -> cm^-3.
        cdnc = tmean(ds["activated_cdnc"]).isel(level=k_cdnc)
        out["cdnc_900hPa_cm3"] = float(global_mean(cdnc, weights)) * 1e-6
    if k_cdnc is not None and "aerocom_N100" in ds:
        n100 = tmean(ds["aerocom_N100"]).isel(level=k_cdnc)
        out["N100_900hPa_cm3"] = float(global_mean(n100, weights)) * 1e-6

    if "jam_state.r_dry" in ds and "pressure_full" in ds:
        k_sfc = int(np.argmax(np.asarray(
            ds["pressure_full"].mean([d for d in ds["pressure_full"].dims
                                      if d != "level"]).values)))
        r_dry = tmean(ds["jam_state.r_dry"]).isel(level=k_sfc)
        for i, mode in enumerate(np.asarray(ds["mode"].values)):
            out[f"r_dry_{str(mode)}_um"] = float(
                global_mean(r_dry.isel(mode=i), weights)) * 1e6

    return out


def collect(files) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reduce every chunk file to scalars; return ``(days, series)``.

    Files are opened and closed one at a time — a year of JAM output does not
    fit in memory, and the statistics only ever need per-chunk reductions.
    """
    days: list[float] = []
    rows: list[dict[str, float]] = []
    for path in files:
        with xr.open_dataset(path) as ds:
            rows.append(chunk_reduction(ds))
        days.append(chunk_day(path, len(days)))
    keys = sorted({k for row in rows for k in row})
    series = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float)
              for k in keys}
    return np.asarray(days, dtype=float), series


#: Length of a seasonal cycle. A record that covers this many days is scored
#: over its final year, which holds each season exactly once (see
#: :func:`drift_window_days`).
YEAR_DAYS = 365.0

#: The fit window [days] for a record shorter than a year: its final six months.
SHORT_RECORD_WINDOW_DAYS = 182.5


# --- Annual window selection (#1010) ---------------------------------------
#
# The drift/ledger/mean statistics are scored on whatever window the caller
# asked for (``--last-n``, or the whole record without it). The annual-budget
# statistics (``dust_emission_tg_per_yr``; see ``MIN_DUST_WINDOW_DAYS``) are
# drawn on a full T63 year and must be scored on the record's actual year
# regardless of that request — the functions below resolve the two windows
# from one record (:func:`annual_window_bounds`) and read or slice both from
# it in one pass (:func:`collect_windows`, :func:`slice_windows`), so
# ``main`` (here and in ``health.py``) reads the chunk files it needs once.


def annual_window_bounds(days: np.ndarray,
                         window_start: float | None = None
                         ) -> tuple[int, float | None]:
    """First index, and that index's ``window_start``, of the annual window.

    The annual window is the trailing slice of ``days`` covering at least
    :data:`YEAR_DAYS`, or the whole record when it covers less — a record
    shorter than a year stays whole (and may still fail
    ``MIN_DUST_WINDOW_DAYS`` downstream; that is a genuinely short run, not an
    artefact of this selection) and a record longer than a year is trimmed to
    its final one, so a multi-year run is not annualised over its entire
    history.

    The slice is whole CHUNKS: the smallest trailing run of them whose
    combined duration (:func:`chunk_durations`) is at least ``YEAR_DAYS``,
    never a fraction of one, because the statistics the window feeds are
    chunk means. On a coarse cadence this can run a little past 365 days
    (13 trailing 30-day chunks is 390), which is immaterial next to the
    seasonal cycle the window exists to average over.

    Returns ``(start_index, new_window_start)`` into ``days``, in the same
    convention :func:`_window_starts` uses for a ``--last-n`` slice:
    ``new_window_start`` is the label of the chunk immediately before the
    slice, carried through unchanged (``None`` usually means "the run's own
    start") when the slice is the whole record.
    """
    days = np.asarray(days, dtype=float)
    durations = chunk_durations(days, window_start)
    if float(np.sum(durations)) <= YEAR_DAYS + 1e-6:
        return 0, window_start
    cumulative = 0.0
    start = days.size - 1
    for start in range(days.size - 1, -1, -1):
        cumulative += durations[start]
        if cumulative >= YEAR_DAYS - 1e-6:
            break
    new_window_start = window_start if start == 0 else float(days[start - 1])
    return start, new_window_start


def _window_indices(days: np.ndarray, last_n: int | None,
                    window_start: float | None = None
                    ) -> tuple[int, float | None, int, float | None]:
    """Start index (and its ``window_start``) of the request and annual windows.

    Shared by :func:`collect_windows` (reading from chunk files) and
    :func:`slice_windows` (re-slicing an already-loaded series): both windows
    end at the record's last sample, so separating "which indices" from "how
    to extract them" lets a file-backed and an in-memory record share one
    definition. ``window_start`` is the label of the chunk before ``days[0]``
    when ``days`` is itself already a slice of a longer record (as a loaded
    ``--series-in`` may be) — the request/annual windows this returns are
    slices of ``days``, so one that would need to reach past ``days[0]`` is
    capped there; the ANNUAL window is then shorter than a year through no
    fault of the run, only of a cache that was itself saved pre-sliced (see
    :func:`slice_windows`).
    """
    annual_start, annual_window_start = annual_window_bounds(days, window_start)
    n = days.size
    if last_n and last_n < n:
        req_start = n - last_n
        req_window_start = float(days[req_start - 1])
    else:
        req_start = 0
        req_window_start = window_start
    return req_start, req_window_start, annual_start, annual_window_start


def collect_windows(files: list,
                    last_n: int | None) -> tuple[tuple, tuple]:
    """Read the request (``--last-n``) and annual windows in ONE pass.

    Returns ``(request, annual)``, each a ``(days, series, window_start)``
    triple ready for :func:`summarize`/:func:`unscored_gates`. ``request`` is
    exactly the pre-#1010 ``--last-n`` slice — what the drift/ledger/mean
    statistics read; ``annual`` is :func:`annual_window_bounds`' trailing
    slice — what the annual-budget statistics read regardless of ``--last-n``.

    The two windows share their last chunk, so their union is one trailing
    slice of ``files``, read with a single :func:`collect` call: reading the
    whole record on every invocation would defeat ``--last-n`` on the common
    case where the request already covers a year, so this reads only the
    chunks the annual window needs beyond the request (zero of them, when the
    request is already that long or longer).
    """
    all_days = np.array([chunk_day(f, i) for i, f in enumerate(files)],
                        dtype=float)
    req_start, req_window_start, annual_start, annual_window_start = \
        _window_indices(all_days, last_n)
    union_start = min(req_start, annual_start)
    union_days, union_series = collect(files[union_start:])

    def extract(start):
        offset = start - union_start
        return union_days[offset:], {k: v[offset:]
                                     for k, v in union_series.items()}

    req_days, req_series = extract(req_start)
    annual_days, annual_series = extract(annual_start)
    return ((req_days, req_series, req_window_start),
            (annual_days, annual_series, annual_window_start))


def slice_windows(days: np.ndarray, series: dict[str, np.ndarray],
                  last_n: int | None, window_start: float | None = None
                  ) -> tuple[tuple, tuple]:
    """Split an already-loaded series into the same ``(request, annual)``
    pair :func:`collect_windows` returns — the ``--series-in`` path.

    The record is fully in memory, so both windows are produced by array
    slicing alone — no second read. If ``days``/``series`` is ITSELF a slice
    saved by an earlier ``--last-n ... --series-out`` run (see
    ``TestSeriesRoundTrip.test_window_start_travels_with_a_sliced_series``,
    which pins that ``--series-out`` keeps saving exactly the requested
    slice, not the annual window, so this case is expected rather than a
    bug), the annual window can only be as long as what was saved: a cache
    sliced short of a year before saving has no more record to recover, and
    the annual statistic reads UNSCORED exactly as it would from a run
    directory with chunks genuinely missing. Save with ``--series-out`` and
    no ``--last-n`` to keep the whole record available to every later
    re-score.
    """
    req_start, req_window_start, annual_start, annual_window_start = \
        _window_indices(days, last_n, window_start)

    def extract(start):
        return days[start:], {k: v[start:] for k, v in series.items()}

    req_days, req_series = extract(req_start)
    annual_days, annual_series = extract(annual_start)
    return ((req_days, req_series, req_window_start),
            (annual_days, annual_series, annual_window_start))


def drift_window_days(days: np.ndarray,
                      window_start: float | None = None) -> float:
    """Fit window [days] of the drift statistic for this record.

    The sources behind most of the burdens are seasonal (dust, biomass-burning
    BC, sulfate), so a straight-line fit over a window shorter than a year reads
    the seasonal swing as growth or decay. Once the record covers a year the
    window is its final 365 days, which hold every season exactly once, and the
    fit removes the annual harmonic (:func:`log_drift`); a shorter record (a
    ``--last-n`` slice, or a cold-start year scored over its settled months)
    keeps the final six months with a straight line, where a from-zero spin-up
    ramp is already behind it. A record that starts from a cold init and covers a
    whole year therefore has to be sliced with ``--last-n`` to its settled
    months: the whole-year rule is for a warm or second year.
    """
    covered = covered_days(days, window_start)
    return YEAR_DAYS if covered >= YEAR_DAYS - 1e-6 \
        else SHORT_RECORD_WINDOW_DAYS


def drift_limit_per_day(harmonic_fit: bool) -> float:
    """Return the drift limit [1/day] for a slope from the given kind of fit.

    A slope from the annual-harmonic fit is held to :data:`DRIFT_LIMIT_PER_DAY`;
    a straight-line slope keeps :data:`SHORT_RECORD_DRIFT_LIMIT_PER_DAY`. Keyed
    on the fit that actually produced the slope (:func:`fit_log_drift`), not on
    the window it was asked for, because the 3-sigma basis of the whole-year
    limit is the harmonic fit's scatter and a year-long window with too few
    samples is fit by a line.
    """
    return DRIFT_LIMIT_PER_DAY if harmonic_fit \
        else SHORT_RECORD_DRIFT_LIMIT_PER_DAY


def yoy_burden_ratio(days: np.ndarray, values: np.ndarray) -> float:
    """Mean burden of the final 365 days over that of the 365 days before it.

    Reported when the record holds two years; the same seasons are compared, so
    the seasonal cycle cancels and what is left is interannual variability plus
    any real drift. NaN when there is no earlier year or no finite samples.
    """
    days, values = np.asarray(days, float), np.asarray(values, float)
    last = days > days[-1] - YEAR_DAYS
    prev = (days > days[-1] - 2 * YEAR_DAYS) & ~last
    ok = np.isfinite(values)
    if not (ok & last).any() or not (ok & prev).any():
        return float("nan")
    previous = float(np.mean(values[ok & prev]))
    return float(np.mean(values[ok & last]) / previous) if previous > 0 \
        else float("nan")


def log_drift(days: np.ndarray, values: np.ndarray,
              window_days: float = SHORT_RECORD_WINDOW_DAYS) -> float:
    """``d ln B / dt`` [1/day] from a least-squares fit over the last window.

    The slope of :func:`fit_log_drift`, which documents the fit.
    """
    return fit_log_drift(days, values, window_days)[0]


def fit_log_drift(days: np.ndarray, values: np.ndarray,
                  window_days: float = SHORT_RECORD_WINDOW_DAYS
                  ) -> tuple[float, bool]:
    """Return ``(d ln B / dt [1/day], harmonic)`` from a fit over the last window.

    ``harmonic`` says which fit produced the slope, because the drift limit
    depends on it (:func:`drift_limit_per_day`).

    A logarithmic slope is the right statistic for an aerosol burden because
    the failure mode is multiplicative: a runaway grows by a fixed factor per
    unit time, so it shows up as a slope that is large regardless of the
    species' absolute loading, while a merely noisy but stationary burden
    averages to zero.

    A window of a year or more is fit jointly with the annual harmonic,
    ``ln B = a + b t + c cos(2 pi t / 365) + d sin(2 pi t / 365)``, and ``b`` is
    returned. A straight line over one cycle only cancels a sinusoidal seasonal
    swing for particular phases (a stationary burden of amplitude 0.5 in
    ``ln B`` fits up to 0.0026 /day of "drift" at the worst phase), and the
    fitted harmonic absorbs such a swing whatever its phase. The cycle's higher
    harmonics are not removed: over exactly one period a trend and an arbitrary
    periodic function cannot be told apart, so a one-year slope of a stationary
    burden still scatters by ~0.001 /day (``jam_regression.md``, "What a
    one-year record resolves"); :func:`yoy_burden_ratio` is the cycle-free
    comparison. A shorter window is a straight line: it cannot hold a cycle, so
    the harmonic is not identifiable there. A window of a year that the finite
    samples do not span is a straight line for the same reason, and is reported
    as one: fewer than eight finite samples, or a first-to-last span of less than
    a year minus two chunk cadences. The cadence is read from the record's chunk
    labels (finite or not), so a sample that is NaN or non-positive does not
    widen the allowance and let a sparse series pass as a year. The labels are all
    the record says about its cadence: a chunk file deleted from the run directory
    is indistinguishable from a coarser output interval, and a record of eight
    45-day means that covers a year is scored as one.
    """
    good = np.isfinite(values) & (values > 0)
    if good.sum() < 3:
        return float("nan"), False
    cadence = float(np.median(np.diff(days)))
    d, v = days[good], values[good]
    sel = d >= (d[-1] - window_days)
    if sel.sum() < 3:
        sel = np.ones_like(d, dtype=bool)
    t, y = d[sel], np.log(v[sel])
    if window_days >= YEAR_DAYS and t.size >= 8 \
            and t[-1] - t[0] >= YEAR_DAYS - 2 * cadence:
        w = 2.0 * np.pi * t / YEAR_DAYS
        design = np.column_stack([np.ones_like(t), t - t.mean(), np.cos(w), np.sin(w)])
        return float(np.linalg.lstsq(design, y, rcond=None)[0][1]), True
    return float(np.polyfit(t, y, 1)[0]), False


#: Substeps per output chunk, and periodic cycles to spin the box model up by,
#: in :func:`box_model_drift`. 50 substeps resolves the exact-exponential step
#: far finer than the forcing varies within one chunk; 20 annual cycles decays
#: the arbitrary initial condition by ``exp(-20*cadence/tau)`` -- for a 5-day
#: cadence and POA/BC's 4-7 day lifetime that is below 1e-5 of the signal, so
#: the sampled solution is the periodic steady state to well past the
#: precision this statistic is read to.
_BOX_SUBSTEPS_PER_CHUNK = 50
_BOX_SPINUP_CYCLES = 20


def box_model_drift(days: np.ndarray, emission: np.ndarray, burden: np.ndarray,
                    dry: np.ndarray, wet: np.ndarray,
                    window_days: float = YEAR_DAYS) -> float:
    """Return the ``d ln B/dt`` the gate's own fit reads off a box with NO real drift.

    POA and BC are a lopsided seasonal pulse (biomass burning): a least-squares
    fit over any *finite* window of a periodic signal is biased by the part of
    the cycle it does not evenly sample, even the annual-harmonic fit
    (:func:`fit_log_drift`'s own docstring: "over exactly one period a trend
    and an arbitrary periodic function cannot be told apart"). This builds the
    box ``dB/dt = E(t) - B/tau`` that is EXACTLY periodic by construction (its
    steady-state solution, reached after :data:`_BOX_SPINUP_CYCLES` annual
    cycles) and reads the SAME fit off it, so the returned number is that
    fit's own artefact on this run's actual emission shape -- not a model of
    what the run did, a model of what the GATE's STATISTIC does to a signal
    that, by construction, has no drift at all.

    ``emission`` drives the box with the run's own chunk-mean ``emi_<sp>``
    series, repeated periodically (period :data:`YEAR_DAYS`) by reading it as
    a function of day-of-year. ``tau`` -- the e-folding removal time -- is a
    SINGLE CONSTANT, the record's own mean of the elementwise
    ``burden / (dry + wet)``, not the run's time-varying lifetime. That choice
    is deliberate: a box driven by the run's own tau(t) would absorb whatever
    the sink actually did that year, including a genuine sink-strength trend,
    and "correct" it away along with the artefact -- which is exactly the
    failure mode a correction must not have. The constant-tau box isolates
    only what the emission's periodic SHAPE does to the fit, independent of
    any information about how the run's own sink evolved.

    Validated against an independent reduction of 12 JAM T63 L47/L95 years on
    disk (six stage-3 years, a cold/warm-start pair and two consecutive-year
    pairs): this function reproduces the investigation's box-model slope to
    machine precision on every one of them (year D POA: box -0.00145,
    raw -0.00324, corrected -0.00179 -- see
    ``docs/source/design/jam_regression.md``, "Which window, and the
    box-model correction").
    """
    sink = np.asarray(dry, dtype=float) + np.asarray(wet, dtype=float)
    emission = np.asarray(emission, dtype=float)
    days = np.asarray(days, dtype=float)
    # ``burden`` is the ``burden_<sp>`` series as reduced, in mg/m2 (see the
    # ``* 1e6`` where the column integral is formed), while the ledgers are
    # kg/m2/s, so the lifetime needs the same mg -> kg conversion the
    # ``lifetime_<sp>_days`` statistic applies. Elementwise mean, not
    # ratio-of-means: the statistic box_and_figs.py validated this against
    # uses ``(B/sink).mean()``, and the two differ whenever burden and sink
    # co-vary seasonally (which they do).
    burden_kg = np.asarray(burden, dtype=float) * 1e-6
    tau_seconds = float(np.mean(burden_kg / sink))
    cadence = float(np.median(np.diff(days)))
    # The fine grid is an integer number of substeps per chunk AND an integer
    # number of chunks per year (the nearest whole count for a cadence that
    # does not divide the year, e.g. 10 or 30 days), so pooling the fine
    # solution back to chunk means below is always an exact reshape.
    n_chunks = max(int(round(YEAR_DAYS / cadence)), 1)
    n_per_chunk = _BOX_SUBSTEPS_PER_CHUNK
    n_fine = n_chunks * n_per_chunk
    fine_step = YEAR_DAYS / n_fine
    fine = (np.arange(n_fine) + 0.5) * fine_step
    # Chunks are labelled by their END day and hold an average over the
    # cadence before it (see chunk_centres); read as a function of
    # day-of-year via periodic interpolation so this works for any chunk
    # offset, not only a record that starts exactly on day ``cadence``.
    centres = days - 0.5 * cadence
    e_fine = np.interp(fine, centres, emission, period=YEAR_DAYS)
    equilibrium = e_fine * tau_seconds       # instantaneous eq. B = E*tau
    decay = float(np.exp(-fine_step * 86400.0 / tau_seconds))
    b = float(np.mean(emission)) * tau_seconds       # arbitrary starting B
    out = np.empty(n_fine)
    for _ in range(_BOX_SPINUP_CYCLES):
        for i in range(n_fine):
            # Exact exponential step for E, tau constant over one substep.
            b = equilibrium[i] + (b - equilibrium[i]) * decay
            out[i] = b
    # One canonical year of chunk means (the fine steady solution pooled back
    # to the native cadence), then read off at the record's ACTUAL chunk
    # labels by day-of-year -- so a record that is not exactly one
    # ``cadence``-aligned year still gets a box sample for every real sample.
    canonical = out.reshape(n_chunks, n_per_chunk).mean(axis=1)
    canonical_days = (np.arange(n_chunks) + 0.5) * (YEAR_DAYS / n_chunks)
    box = np.interp(np.mod(centres, YEAR_DAYS), canonical_days, canonical,
                    period=YEAR_DAYS)
    return fit_log_drift(days, box, window_days)[0]


def standard_error(values: np.ndarray) -> float:
    """Estimate the standard error of the mean of an autocorrelated series.

    ``N_eff = N (1 - a) / (1 + a)`` with ``a`` the lag-1 autocorrelation of the
    chunk series — the usual first-order correction, which is what makes the
    3-sigma tier meaningful for a burden whose 5-day samples are anything but
    independent. Negative ``a`` is clipped to zero (never claim *more*
    independent samples than there are chunks).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4 or np.allclose(v, v[0]):
        return 0.0 if n else float("nan")
    a = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    a = min(max(a, 0.0), 0.99)
    n_eff = max(n * (1.0 - a) / (1.0 + a), 1.0)
    return float(np.std(v, ddof=1) / np.sqrt(n_eff))


def tolerance_floor(name: str) -> float:
    """Absolute tolerance floor for a statistic, by longest matching prefix."""
    matches = [v for k, v in _TOLERANCE_FLOORS.items() if name.startswith(k)]
    return max(matches) if matches else 0.0


def regression_tolerance(reference: float, sigma: float,
                         floor: float = 0.0) -> float:
    """``max(3 sigma, 15 % of the reference, an absolute floor)``.

    The floor is what keeps a legitimately-zero reference — an unused species'
    burden — from collapsing the tolerance to zero and failing on any nonzero
    value. See the design doc.
    """
    return max(_N_SIGMA * sigma, _REL_TOLERANCE * abs(reference), floor)


def _is_t63(series: dict[str, np.ndarray]) -> bool:
    """Was every scored chunk written on the T63 grid the dust band assumes?

    Unknown (output without the ``nlat`` reduction) counts as NOT T63: a band
    applied to an unidentified grid would be scoring a number it cannot
    attribute.
    """
    nlat = series.get("nlat")
    return nlat is not None and bool(np.all(nlat == _T63_NLAT))


def _budget_residual(days, series, species,
                     window_start: float | None = None) -> float | None:
    """Compute ``(emitted - deposited - dB) / emitted`` over the record.

    The fluxes are chunk means of ``kg/m2/s``, integrated trapezoidally over
    ``days`` between the first and last chunk means; ``dB`` differences the
    chunk-mean burden at those same two endpoints. Every component must be
    present: a missing one returns ``None`` (reported unscored) rather than
    counting as zero. For sulfate the ledger is the
    whole sulfur family in the run's own sulfate mass (SO2 and DMS emissions
    are the real sulfate source; ``emi_so4`` alone is a few per cent of it),
    the stored mass includes the gas-phase reservoirs, and the sulfate is the
    one the reduction identified (``so4_molar_mass``): a record without it,
    or spanning two populations, is unscored.
    """
    span = float(days[-1] - days[0])
    if span <= 0 or days.size < 2:
        return None

    nodes = chunk_centres(days, window_start)

    def integral(key):
        # Trapezoidal between the first and last chunk MEANS — the same two
        # endpoints the storage term differences. A right-endpoint rectangle
        # rule (dropping the first sample) integrates a half-chunk-shifted
        # window instead, short by half a chunk times the endpoint change: on
        # a 90-day window a seasonal swing in the sink can exceed the closure
        # threshold on that alone. The abscissa is chunk CENTRES, which is
        # where a chunk mean belongs; under a uniform cadence the offset from
        # the filename's end-of-chunk day cancels, but under a varying one it
        # does not, and only the centres put the nodes in the right places.
        v = series.get(key)
        if v is None or not np.all(np.isfinite(v)):
            return None
        return float(_trapezoid(v, nodes)) * 86400e6

    # EVERY component is required. A missing ledger field is not a zero
    # contribution: treating it as one fabricates a residual and then gates
    # it, which is a false pass or a false failure rather than the "unscored"
    # the caller would get from None.
    if species == "so4":
        # Every carrier converted to the run's own sulfate mass, the unit the
        # burden and the deposition ledger are already in. Without a known
        # sulfate there is no unit to convert to: unscored, not guessed.
        m_sulfate = _record_sulfate_molar_mass(series)
        if m_sulfate is None:
            return None
        carriers = sulfur_carriers(m_sulfate)
        contributions = [integral(f"emi_{c}") for c in _S_FAMILY_EMIS]
        if any(e is None for e in contributions):
            return None
        emitted = sum(e * carriers[c]
                      for e, c in zip(contributions, _S_FAMILY_EMIS))
        stored = series.get("burden_so4")
        if stored is None:
            return None
        for gas in _S_FAMILY_GASES:
            reservoir = series.get(f"gas_{gas}")
            if reservoir is None:
                return None     # a gas reservoir omitted, not empty
            stored = stored + reservoir * carriers[gas]
    else:
        emitted = integral(f"emi_{species}")
        stored = series.get(f"burden_{species}")
        if emitted is None or stored is None:
            return None

    deposited = 0.0
    for prefix in ("dry", "wet"):
        contribution = integral(f"{prefix}_{species}")
        if contribution is None:
            return None         # a removal ledger omitted, not zero
        deposited += contribution
    if not np.isfinite(emitted) or emitted <= 0:
        return None
    residual = float((emitted - deposited - (stored[-1] - stored[0])) / emitted)
    # The storage endpoints are as capable of being NaN as the emission was: a
    # chunk missing a burden or a gas reservoir poisons the difference. A NaN
    # here would score as a gate FAIL for something never measured, and would
    # win ``max(..., key=abs)`` too, since every comparison against NaN is
    # False.
    return residual if np.isfinite(residual) else None


def _box_slope(days: np.ndarray, series: dict[str, np.ndarray], species: str,
               burden: np.ndarray, window_days: float) -> float | None:
    """:func:`box_model_drift` for ``species``, or ``None`` if it cannot be built.

    Needs the emission and both deposition ledgers, every sample finite (a
    ledger that is present but holed is not usable, the same bar
    :func:`_budget_residual` and the ``lifetime_*`` statistic hold the removal
    series to) and a positive sink (an undefined lifetime). Any of those
    missing falls back to the UNCORRECTED slope in :func:`summarize` --
    exactly today's behaviour -- rather than a fabricated correction.
    """
    emission = series.get(f"emi_{species}")
    dry = series.get(f"dry_{species}")
    wet = series.get(f"wet_{species}")
    if emission is None or dry is None or wet is None:
        return None
    if not (np.all(np.isfinite(emission)) and np.all(np.isfinite(dry))
            and np.all(np.isfinite(wet)) and np.all(np.isfinite(burden))):
        return None
    if np.any((dry + wet) <= 0):
        return None
    # No emission at all (a run with the prescribed anthropogenic sources
    # switched off still carries a residual burden) drives a zero box whose
    # fit is undefined; that is "no correction available", not a failure.
    if not np.any(emission > 0):
        return None
    slope = box_model_drift(days, emission, burden, dry, wet, window_days)
    return slope if np.isfinite(slope) else None


def summarize(days: np.ndarray, series: dict[str, np.ndarray],
              timestep_seconds: float | None = None,
              window_start: float | None = None,
              annual_days: np.ndarray | None = None,
              annual_series: dict[str, np.ndarray] | None = None,
              annual_window_start: float | None = None) -> dict[str, float]:
    """Build the statistic set from the series returned by :func:`collect`.

    Every entry is a single number a gate or a stored reference can be
    compared against; the rationale for each is in the design doc.
    ``window_start`` is the label of the chunk preceding ``days[0]`` when the
    record is a slice of a longer run (see :func:`chunk_centres`).

    ``annual_days``/``annual_series``/``annual_window_start`` are the record
    the annual-budget statistics (``dust_emission_tg_per_yr``) are scored on
    — the whole run directory's trailing year (:func:`annual_window_bounds`),
    independent of whatever ``days``/``series``/``window_start`` were sliced
    to by ``--last-n`` (#1010: the band is drawn on a full year, and scoring
    it on a ``--last-n`` short of one measures the season, which is exactly
    what left it UNSCORED on every documented recipe). They default to
    ``days``/``series``/``window_start`` when omitted (keyed on
    ``annual_series is None``, since ``annual_days`` alone says nothing), so
    a direct call — as every existing test, and the reference path in
    :func:`compare_to_reference`, make — scores the annual statistics on
    whatever record it was given, unchanged.
    """
    if annual_series is None:
        annual_days, annual_series, annual_window_start = (
            days, series, window_start)
    stats: dict[str, float] = {}
    span_days = float(days[-1] - days[0]) if days.size > 1 else 0.0
    fit_window = drift_window_days(days, window_start)

    for species in _SPECIES:
        b = series.get(f"burden_{species}")
        if b is None:
            continue
        stats[f"burden_{species}_mg_m2"] = float(np.nanmean(b))
        # Emit the drift only where it is measurable: a species the run does
        # not carry yields NaN, and a window shorter than MIN_WINDOW_DAYS
        # yields a slope dominated by its own noise. Both are reported
        # unscored (see :func:`unscored_gates`) rather than gated, because a
        # gate FAIL for something never measured is worse than no number.
        drift, harmonic = fit_log_drift(days, b, fit_window)
        if np.isfinite(drift) and span_days >= MIN_WINDOW_DAYS:
            gated = drift
            # The box-model correction (#1081) only ever applies to the
            # whole-year harmonic fit: that is the only fit a periodic box can
            # be validated against (the box needs a full annual emission
            # cycle to build, which a straight-line fit's shorter window does
            # not have), and it is the only one the maintainer asked
            # corrected, for POA and BC alone (BOX_CORRECTED_SPECIES).
            if species in BOX_CORRECTED_SPECIES and harmonic:
                box = _box_slope(days, series, species, b, fit_window)
                if box is not None:
                    stats[f"raw_dlnB_dt_{species}_per_day"] = drift
                    stats[f"box_dlnB_dt_{species}_per_day"] = box
                    gated = drift - box
            stats[f"dlnB_dt_{species}_per_day"] = gated
            # Which fit produced this slope, because the gate's limit depends
            # on it and the slope alone cannot say. Per species: whether a
            # year-long window held enough finite samples for the harmonic is
            # a property of that species' series.
            stats[f"drift_harmonic_{species}"] = float(harmonic)
        if covered_days(days, window_start) >= 2 * YEAR_DAYS - 1e-6:
            yoy = yoy_burden_ratio(days, b)
            if np.isfinite(yoy):
                stats[f"yoy_burden_ratio_{species}"] = yoy

    for species in LIFETIME_SPECIES:
        b = series.get(f"burden_{species}")
        if b is None:
            continue
        # ``wet_*`` already includes in-plume convective scavenging (the
        # transport term's flux is folded into it in wetdep_term.py), so
        # adding ``conv_scav_flux.*`` here would double-count that sink.
        # BOTH ledgers are required: a removal series absent from trimmed or
        # mixed-version output is an omission, not a zero sink, and a lifetime
        # formed from the other alone is a wrong number that a reference
        # comparison would then score (see :func:`unscored_gates`).
        # As strict as ``_budget_residual``: a ledger that is present but has
        # a non-finite sample (a diagnostic missing from some chunks) would
        # average a different set of chunks than the burden does.
        dry, wet = series.get(f"dry_{species}"), series.get(f"wet_{species}")
        if dry is None or wet is None or not (
                np.all(np.isfinite(dry)) and np.all(np.isfinite(wet))):
            continue
        sink = (float(np.mean(dry)) + float(np.mean(wet))) * 86400e6
        if sink > 0:                                # kg/m2/s -> mg/m2/day
            stats[f"lifetime_{species}_days"] = float(np.nanmean(b)) / sink

    residuals = {}
    for species in ("so4",) + PRIMARY_BUDGET_SPECIES:
        r = (_budget_residual(days, series, species, window_start)
             if span_days >= MIN_WINDOW_DAYS else None)
        if r is not None:
            residuals[species] = r
            stats[f"budget_residual_{species}"] = r
    if residuals:
        stats["budget_residual_max"] = max(residuals.values(), key=abs)

    # Annual dust emission [Tg/yr] in the population's own size window: D < 10
    # µm for MAM4, M7's AI and CI modes (to ``dust.py``'s ``M7_CI_UM`` edge,
    # HAM's own window) for M7. ``emi_du`` is a global-MEAN flux, so the total
    # is that times the Earth's area; mass above the window never enters a
    # tracer and is not in this number. Read from
    # ``annual_series``/``annual_days`` — the whole run directory's trailing
    # year, regardless of what ``--last-n`` sliced ``days``/``series`` to
    # (#1010) — and emitted only for a window long enough to mean the year
    # rather than a season, and only on the grid the band was calibrated for;
    # an unscored statistic is dropped here and explained in
    # :func:`unscored_gates`, exactly like the drifts.
    # Every chunk must contribute: ``nanmean`` over a series with a missing
    # chunk would annualise the chunks that happen to be present, so a year
    # whose emission diagnostic vanished half-way — a mixed-version rerun, a
    # corrupted file — could still score inside the band.
    # Weighted by each chunk's own averaging window, not counted equally: a
    # year written as twelve 30-day chunks and a final 5-day one would
    # otherwise give those five days a month's weight, and dust is seasonal
    # enough for that to move the annual total across the band.
    if ("emi_du" in annual_series
            and covered_days(annual_days, annual_window_start)
            >= MIN_DUST_WINDOW_DAYS
            and _is_t63(annual_series)
            and np.all(np.isfinite(annual_series["emi_du"]))
            and _uneven_window_reason(annual_days, annual_window_start)
            is None):
        stats["dust_emission_tg_per_yr"] = (
            float(np.average(annual_series["emi_du"],
                             weights=chunk_durations(annual_days,
                                                     annual_window_start)))
            * EARTH_AREA_M2 * 86400.0 * 365.0 / 1e9)
        # The window actually scored, in days, so a reader of the printed
        # table or health.json can tell this annual budget's window apart
        # from ``record_days`` (the drift/ledger window, which may be the
        # much shorter ``--last-n`` slice) rather than assuming they match.
        stats["dust_window_days"] = covered_days(annual_days,
                                                 annual_window_start)

    if "so4_above_500hPa" in series and "burden_so4" in series:
        aloft = float(np.nanmean(series["so4_above_500hPa"]))
        total = float(np.nanmean(series["burden_so4"]))
        if total > 0:
            stats["so4_frac_above_500hPa"] = aloft / total
    if "so4_nh" in series and "so4_sh" in series:
        sh = float(np.nanmean(series["so4_sh"]))
        if sh > 0:
            stats["so4_nh_sh_ratio"] = float(np.nanmean(series["so4_nh"])) / sh
    if "so4_60_90N" in series:
        stats["so4_burden_60_90N_mg_m2"] = float(np.nanmean(
            series["so4_60_90N"]))

    for key in ("aod_550", "angstrom", "cdnc_900hPa_cm3", "N100_900hPa_cm3"):
        if key in series:
            stats[key] = float(np.nanmean(series[key]))
    for key in series:
        if key.startswith("r_dry_"):
            stats[key] = float(np.nanmean(series[key]))

    # Dynamics conservation: the transport residual as a fraction of the
    # advected mass, per step. Needs the run's timestep; without it the
    # statistic is not emitted (and its gate is therefore not scored).
    if timestep_seconds:
        for key in series:
            if not key.startswith("budget_dyn_"):
                continue
            species = key[len("budget_dyn_"):]
            mass = series.get(f"budget_mass_{species}")
            if mass is None:
                continue
            mean_mass = _finite_mean(mass)
            if mean_mass > 0:
                stats[f"dyn_frac_per_step_{species}"] = abs(
                    float(np.nanmean(series[key])) * timestep_seconds
                    / mean_mass)

    stats["record_days"] = span_days
    return stats


def _hydra_config(run_dir: str) -> dict | None:
    """Load ``<run_dir>/.hydra/config.yaml``; ``None`` when it is absent.

    Shared by :func:`timestep_seconds` and :func:`warm_start_kind`: both read
    provenance Hydra itself saved beside the run's output, and both treat its
    absence (a hand-assembled directory, or output copied without it) the
    same way — "unknown", never a guess.
    """
    cfg = Path(run_dir) / ".hydra" / "config.yaml"
    if not cfg.is_file():
        return None
    import yaml
    return yaml.safe_load(cfg.read_text()) or {}


def timestep_seconds(run_dir: str) -> float | None:
    """Read the run.s timestep [s] from the Hydra config saved beside it.

    Returned as ``None`` when the run directory has no ``.hydra/config.yaml``
    (a hand-assembled directory, or output copied without it), which makes the
    dynamics-conservation gate unscored rather than wrong.
    """
    loaded = _hydra_config(run_dir)
    if loaded is None:
        return None
    minutes = (loaded.get("run") or {}).get("time_step")
    return float(minutes) * 60.0 if minutes else None


def warm_start_kind(run_dir: str) -> bool | None:
    """Was ``run_dir`` launched with ``init=from_state`` (a warm start)?

    Read from the ``init.kind`` Hydra records in ``.hydra/config.yaml``
    (``jcm/config/init/from_state.yaml`` sets ``kind: from_state``; every
    other ``init/*.yaml`` names its own cold-start kind). A "second year" in
    this codebase is produced exactly this way — a fresh run warm-started from
    the first year's end state (``docs/source/design/jam_regression.md``,
    "The spun-up state") — so this one check answers both "warm-started" and
    "second year" questions at once; there is no separate provenance marker
    for the latter.

    Returns ``None`` when the run directory carries no ``.hydra/config.yaml``
    to read (predates it, or was hand-assembled/copied without it): the
    caller then has only an explicit ``--warm-start`` to go on, exactly as
    :func:`timestep_seconds` falls back to ``--timestep-minutes``.
    """
    loaded = _hydra_config(run_dir)
    if loaded is None:
        return None
    return (loaded.get("init") or {}).get("kind") == "from_state"


def resolve_warm_start(flag: str, run_dir: str | None) -> tuple[bool, str]:
    """Decide warm-start status from an explicit flag, or else provenance.

    ``flag`` is one of ``"yes"``, ``"no"`` or ``"auto"`` (a CLI
    ``--warm-start`` argument's value). ``"auto"`` defers to
    :func:`warm_start_kind`, which needs ``run_dir``; when that is ``None`` or
    the directory carries no provenance to detect from, the record defaults
    to cold-start — the one choice that reproduces every existing run's
    current behaviour unchanged, so adding this detection is never a silent
    behaviour change for output it cannot read. Returns ``(warm, reason)``:
    ``reason`` is printed so a report says *why* a run was scored the way it
    was, not just that it was.
    """
    if flag == "yes":
        return True, "--warm-start yes"
    if flag == "no":
        return False, "--warm-start no"
    detected = warm_start_kind(run_dir) if run_dir else None
    if detected is None:
        return False, ("no .hydra/config.yaml to detect a warm start from "
                       "(scored as cold-start; pass --warm-start yes to "
                       "override)")
    if detected:
        return True, "init.kind=from_state in .hydra/config.yaml"
    return False, "init.kind != from_state in .hydra/config.yaml"


def compare_to_reference(stats: dict[str, float],
                         reference: dict[str, float],
                         series: dict[str, np.ndarray] | None = None
                         ) -> list[tuple[str, float, str, bool]]:
    """Score this run's statistics against a stored reference (tier 3).

    The tolerance is ``max(3 sigma, 15 %)``. ``sigma`` comes from the *current*
    run's own chunk series where one is available, so a statistic that is
    genuinely noisy in this configuration is not held to the precision of a
    quiet one; without a series the 15 % floor decides. Returns the same
    ``(name, value, limit, ok)`` rows the gate functions use.
    """
    rows = []
    for key in sorted(reference):
        if key.startswith(_RECORD_METADATA):
            continue
        # The exemption comes FIRST. These carry their own absolute gates, and
        # ``summarize`` omits them precisely when they could not be measured —
        # so scoring their absence here would hard-fail a run for a gate it has
        # already, correctly, reported as UNSCORED.
        if key.startswith(_GATED_PREFIXES):
            continue
        if key not in stats:
            # The reference has it and this run does not: the diagnostic was
            # removed. Skipping would report PASS for the regression that
            # deleted the very statistic being compared.
            rows.append((key, float("nan"),
                         f"{reference[key]:.5g} (absent from this run)", False))
            continue
        value, ref = stats[key], reference[key]
        sigma = 0.0
        if series is not None:
            for candidate in (key, key.removesuffix("_mg_m2"),
                              f"burden_{key.removeprefix('burden_')}"):
                if candidate in series:
                    sigma = standard_error(series[candidate])
                    break
        tol = regression_tolerance(ref, sigma, tolerance_floor(key))
        ok = abs(value - ref) <= tol
        rows.append((key, value, f"{ref:.5g} +- {tol:.3g}", bool(ok)))
    return rows


def unscored_gates(days: np.ndarray, series: dict[str, np.ndarray],
                   timestep_seconds: float | None = None,
                   window_start: float | None = None,
                   annual_days: np.ndarray | None = None,
                   annual_series: dict[str, np.ndarray] | None = None,
                   annual_window_start: float | None = None
                   ) -> list[tuple[str, str]]:
    """Gates that could NOT be evaluated, each with the reason.

    A gate that is absent from :func:`physics_gates` has passed nothing — it
    was never run. Reporting the absence is what stops a run with no dynamics
    gauge, or a window too short to fit a slope, from reading as clean.
    Returns ``(gate_name, reason)`` pairs for the caller to print.

    ``annual_days``/``annual_series``/``annual_window_start`` name the record
    the ``dust_emission_tg_per_yr`` reasons below are checked against — see
    :func:`summarize` for why it differs from ``days``/``series``/
    ``window_start`` (#1010); they default the same way.
    """
    if annual_series is None:
        annual_days, annual_series, annual_window_start = (
            days, series, window_start)
    span = float(days[-1] - days[0]) if days.size > 1 else 0.0
    short_slope = (f"window spans {span:.0f} days; a slope needs "
                   f"{MIN_WINDOW_DAYS:.0f} to rise above its own fit noise")
    short_budget = (f"window spans {span:.0f} days; the storage endpoints "
                    f"swamp the flux integral below {MIN_WINDOW_DAYS:.0f}")
    rows: list[tuple[str, str]] = []
    other = other_population_species(series)

    for species in _SPECIES:
        b = series.get(f"burden_{species}")
        if b is None:
            if species in other:
                continue        # the other population's species, not a gap
            # No tracers for it at all. Reported here so the standalone
            # scorer says what health.py says.
            rows.append((f"burden_{species}_mg_m2",
                         "the output carries no mass tracers for this species"))
            continue
        name = f"dlnB_dt_{species}_per_day"
        if not np.any(np.isfinite(b) & (b > 0)):
            reason = "the run carries no burden of this species"
            rows.append((name, reason))
            # The anchor gate skips it on the same grounds, so say so rather
            # than leaving a silently missing tier-2 row.
            rows.append((f"burden_{species}_mg_m2", reason))
        elif span < MIN_WINDOW_DAYS:
            rows.append((name, short_slope))

    # The dust budget: say which of the three preconditions is missing, so a
    # silently unscored release blocker (#808) cannot look like a pass.
    # Checked against the ANNUAL window (the whole run directory's trailing
    # year), not the ``--last-n`` slice ``days``/``series`` may be (#1010) —
    # see :func:`summarize`.
    if "emi_du" not in annual_series:
        rows.append(("dust_emission_tg_per_yr",
                     "the run publishes no emi_du flux, so no dust was "
                     "emitted or the emission diagnostic is absent"))
    elif not _is_t63(annual_series):
        rows.append(("dust_emission_tg_per_yr",
                     "the band is a T63 calibration (#808/#810) and this "
                     "output is on another grid, where nduscale_reg keeps "
                     "HAM's untuned value"))
    elif covered_days(annual_days, annual_window_start) < MIN_DUST_WINDOW_DAYS:
        rows.append(("dust_emission_tg_per_yr",
                     "the run directory spans "
                     f"{covered_days(annual_days, annual_window_start):.0f} "
                     "days; dust emission is "
                     f"seasonal, so annualising below {MIN_DUST_WINDOW_DAYS:.0f} "
                     "days measures the season, not the year"))
    elif not np.all(np.isfinite(annual_series["emi_du"])):
        missing = int(np.sum(~np.isfinite(np.asarray(annual_series["emi_du"]))))
        rows.append(("dust_emission_tg_per_yr",
                     f"{missing} of {len(annual_series['emi_du'])} chunks "
                     "carry no finite emi_du, and an annual budget averaged "
                     "over the rest would not be this run's year"))
    elif _uneven_window_reason(annual_days, annual_window_start) is not None:
        rows.append(("dust_emission_tg_per_yr",
                     _uneven_window_reason(annual_days, annual_window_start)))

    if span < MIN_WINDOW_DAYS:
        rows.append(("budget_residual_max", short_budget))
    else:
        # A species missing from ``budget_residual_max`` is a species whose
        # mass was never checked, and the maximum over an incomplete subset
        # cannot see a leak confined to the species it omits. Report each
        # omission, and the aggregate when nothing at all could be closed.
        closed = []
        for species in ("so4",) + PRIMARY_BUDGET_SPECIES:
            if f"burden_{species}" not in series:
                continue        # the run does not carry it; nothing to close
            if _budget_residual(days, series, species, window_start) is None:
                if (species == "so4"
                        and _record_sulfate_molar_mass(series) is None):
                    reason = (
                        "the sulfate the so4 tracers carry is not identified "
                        "(a series saved before the reduction recorded "
                        "so4_molar_mass, or chunks with no, mixed or "
                        "different populations' sulfate tracers), so the "
                        "sulfur family has no unit to close in; re-read the "
                        "run directory")
                else:
                    reason = ("the mass ledger is incomplete — a missing "
                              "emi_*/dry_*/wet_* diagnostic, or a non-finite "
                              "burden endpoint")
                rows.append((f"budget_residual_{species}", reason))
            else:
                closed.append(species)
        if not closed:
            rows.append(("budget_residual_max",
                         "no species had a complete mass ledger to close "
                         "against; no closure was checked"))

    # A lifetime is formed only from a complete deposition ledger (see
    # :func:`summarize`); name the species whose ledger is not.
    for species in LIFETIME_SPECIES:
        if f"burden_{species}" not in series:
            continue
        ledgers = {f"{p}_{species}": series.get(f"{p}_{species}")
                   for p in ("dry", "wet")}
        missing = [k for k, v in ledgers.items() if v is None]
        if missing:
            rows.append((f"lifetime_{species}_days",
                         f"the deposition ledger is incomplete ({', '.join(missing)} "
                         "absent); a lifetime from the remaining sink alone would "
                         "be wrong, not approximate"))
            continue
        holed = [k for k, v in ledgers.items() if not np.all(np.isfinite(v))]
        if holed:
            rows.append((f"lifetime_{species}_days",
                         f"{', '.join(holed)} has a non-finite sample (a "
                         "diagnostic missing from some chunks); the sink would "
                         "average a different set of chunks than the burden"))
        elif (float(np.mean(ledgers[f"dry_{species}"]))
              + float(np.mean(ledgers[f"wet_{species}"]))) <= 0:
            rows.append((f"lifetime_{species}_days",
                         "the deposition ledger records no removal, so the "
                         "lifetime is undefined"))

    # The dynamics gate needs, PER CARRIED SPECIES, the #713 in-step gauge and
    # its advected-mass denominator, and a timestep to express it per step.
    # :func:`summarize` emits no gate for a species missing either, so a
    # run-wide "some gauge exists" test would let a transport leak confined to
    # the species without one pass unseen — trimmed or mixed-version output
    # can carry the gauge for one species and not another.
    gauged = {k[len("budget_dyn_"):] for k in series if k.startswith("budget_dyn_")}
    if not gauged:
        rows.append(("dyn_frac_per_step", "the run publishes no budget_dyn_* "
                                          "gauge (output predates #713)"))
        return rows
    for species in _SPECIES:
        if f"burden_{species}" not in series:
            continue
        if species not in gauged:
            rows.append((f"dyn_frac_per_step_{species}",
                         f"no budget_dyn_{species} gauge in the output (trimmed "
                         "or mixed-version chunks); the transport residual of "
                         "this species was not measured"))
        elif f"budget_mass_{species}" not in series:
            rows.append((f"dyn_frac_per_step_{species}",
                         f"budget_dyn_{species} has no budget_mass_{species} "
                         "denominator; the residual cannot be expressed as a "
                         "fraction"))
        else:
            # Mirror the condition ``summarize`` emits the gate under, so a
            # denominator that is present but all-NaN or zero is named rather
            # than silently producing no gate.
            mean_mass = _finite_mean(series[f"budget_mass_{species}"])
            if not (np.isfinite(mean_mass) and mean_mass > 0):
                rows.append((f"dyn_frac_per_step_{species}",
                             f"budget_mass_{species} has no finite positive "
                             "mean; the residual cannot be expressed as a "
                             "fraction"))
    if not timestep_seconds:
        rows.append(("dyn_frac_per_step", "no run timestep is available "
                     "(.hydra/config.yaml absent, or run.time_step is null as "
                     "on the pySES backend), and the gate is per step"))
    return rows


#: Climatological anchor gate (tier 2): the shared ``_SPECIES`` ranges widened
#: each way by this factor. A "did the model produce a plausible planetary
#: loading" check, not a tuning target.
BURDEN_SLACK = 3.0

BURDEN_RANGES = {sp: (lo / BURDEN_SLACK, hi * BURDEN_SLACK)
                 for sp, (_modes, (lo, hi)) in _SPECIES.items()}


def anchor_gates(stats: dict[str, float]) -> list[tuple[str, float, str, bool]]:
    """Score the annual burdens against the climatological anchor ranges.

    Shared so the standalone scorer and ``health.py`` apply the same tier-2
    check: a stationary run with a wildly excessive burden has zero drift and
    can close its ledger, so the physics gates alone would pass it. A species
    the run does not carry is left to :func:`unscored_gates`.
    """
    rows = []
    for species, (lo, hi) in BURDEN_RANGES.items():
        key = f"burden_{species}_mg_m2"
        value = stats.get(key)
        # A NaN burden would score FAIL through ``lo <= nan <= hi`` being
        # False — a gate failure for something never measured, the mirror of
        # the case the closure residual already guards.
        if value is None or value == 0.0 or not np.isfinite(value):
            continue
        rows.append((key, value, f"[{lo:g}, {hi:g}]", bool(lo <= value <= hi)))
    return rows


def physics_gates(stats: dict[str, float]) -> list[tuple[str, float, str, bool]]:
    """Score the absolute physics gates; returns ``(name, value, gate, ok)``.

    These are not tuning targets: a burden that grows exponentially, or a
    species whose mass ledger does not close, is a defect at any loading.
    """
    rows = []
    for key, value in sorted(stats.items()):
        if key.startswith("dlnB_dt_"):
            # The limit follows the fit that produced the slope, which
            # :func:`summarize` records beside it. A slope whose fit is not
            # named was not built by ``summarize``; it is held to the stricter
            # straight-line limit rather than credited with a harmonic fit
            # nobody recorded.
            species = key.removeprefix("dlnB_dt_").removesuffix("_per_day")
            limit = drift_limit_per_day(
                bool(stats.get(f"drift_harmonic_{species}", 0.0)))
            ok = np.isfinite(value) and abs(value) < limit
            rows.append((key, value, f"|x| < {limit:g} /day", bool(ok)))
    for key, value in sorted(stats.items()):
        if key.startswith("dyn_frac_per_step_"):
            ok = np.isfinite(value) and value < DYN_RESIDUAL_PER_STEP
            rows.append((key, value,
                         f"x < {DYN_RESIDUAL_PER_STEP:.1%}/step", bool(ok)))
    if "dust_emission_tg_per_yr" in stats:
        value = stats["dust_emission_tg_per_yr"]
        lo, hi = DUST_EMISSION_TG_PER_YR
        ok = np.isfinite(value) and lo <= value <= hi
        rows.append(("dust_emission_tg_per_yr", value,
                     f"{lo:g} <= x <= {hi:g} Tg/yr", bool(ok)))
    if "budget_residual_max" in stats:
        value = stats["budget_residual_max"]
        ok = np.isfinite(value) and abs(value) < BUDGET_RESIDUAL_LIMIT
        limit = f"|x| < {BUDGET_RESIDUAL_LIMIT:.0%}"
        if not ok and np.isfinite(value) and value > 0:
            limit += f" ({_POSITIVE_RESIDUAL_CAVEAT})"
        rows.append(("budget_residual_max", value, limit, bool(ok)))
    return rows


def format_table(stats: dict[str, float],
                 gates: list[tuple[str, float, str, bool]]) -> str:
    """Human-readable report of the statistic set and the physics gates."""
    lines = [f"{'statistic':<34}{'value':>14}", "-" * 48]
    for key, value in sorted(stats.items()):
        if isinstance(value, str):
            lines.append(f"{key:<34}{value:>14}")
        else:
            lines.append(f"{key:<34}{value:>14.5g}")
    if not gates:
        return "\n".join(lines)
    lines += ["", f"{'physics gate':<34}{'value':>14}  {'limit':<20}result",
              "-" * 76]
    notes = []
    for name, value, limit, ok in gates:
        head, _, note = limit.partition(" (")
        lines.append(f"{name:<34}{value:>14.5g}  {head:<20}"
                     f"{'PASS' if ok else 'FAIL'}")
        if note:
            notes.append(f"NOTE  {name}: {note.rstrip(')')}")
    lines += notes
    return "\n".join(lines)


#: A saved chunk is ``<prefix>_day<N>.nc``. ``run.snapshot_interval`` writes
#: ``<prefix>_day<N>_snapshots.nc`` beside it, which a ``*_day*.nc`` glob also
#: matches — those are 2-D snapshot streams with no ``pressure_half``, so they
#: would break the column integral, and any that did parse would double-count
#: the day. Anchored on the extension, they are excluded.
CHUNK_FILE = re.compile(r"_day(\d+)\.nc$")


def run_files(run_dir: str) -> list[str]:
    """Chunk files of a run directory, in day order (snapshots excluded)."""
    matched = ((m, f) for f, m in
               ((f, CHUNK_FILE.search(f)) for f in glob.glob(f"{run_dir}/*.nc"))
               if m)
    return [f for _m, f in sorted(matched, key=lambda mf: int(mf[0].group(1)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--last-n", type=positive_int, default=None,
                    help="use only the last N chunks for the drift/ledger/"
                         "mean statistics (default: all); the annual-budget "
                         "statistics (dust_emission_tg_per_yr) still read "
                         "the whole run directory's trailing year, #1010")
    ap.add_argument("--warm-start", choices=("auto", "yes", "no"),
                    default="auto",
                    help="is this record warm-started or a second year (no "
                         "cold-start spin-up ramp)? When yes, the REQUEST "
                         "window (the drift/ledger/mean statistics) is "
                         "scored over the WHOLE record -- --last-n is "
                         "ignored for it -- and POA/BC get the #1081 "
                         "box-model correction. 'auto' (default) detects "
                         "init.kind=from_state from the run's "
                         ".hydra/config.yaml and falls back to cold-start "
                         "when that is absent, e.g. under --series-in")
    ap.add_argument("--series-out", default=None,
                    help="save the per-chunk reductions to this .npz "
                         "(exactly the --last-n window, if given: omit "
                         "--last-n here to keep the whole record cached)")
    ap.add_argument("--series-in", default=None,
                    help="re-score a saved .npz instead of reading the run "
                         "(reducing a year of JAM output takes ~40 min)")
    ap.add_argument("--reference", default=None,
                    help="a .npz of a reference run's series; adds the "
                         "regression comparison (max(3 sigma, 15 %%))")
    ap.add_argument("--timestep-minutes", type=float, default=None,
                    help="override the run timestep the dynamics gate needs "
                         "(default: the run's Hydra config, or the one saved "
                         "with --series-out)")
    args = ap.parse_args()

    # A warm-started or second year (#1081) is scored, on the REQUEST window,
    # over its whole record rather than an operator's ``--last-n`` -- that
    # window exists to skip a cold-start's spin-up ramp, which a warm start
    # never has. Resolved ONCE, before either branch below builds the
    # request/annual windows, so both read the same decision and the printed
    # NOTE always reflects what actually happened. The ANNUAL window (#1010,
    # dust_emission_tg_per_yr) is unaffected either way -- it already ignores
    # --last-n unconditionally.
    warm, warm_reason = resolve_warm_start(args.warm_start, args.run_dir)
    last_n = None if warm else args.last_n
    if warm and args.last_n:
        print(f"NOTE  --last-n {args.last_n} ignored for the request window "
              f"(drift/ledger/mean statistics): {warm_reason} -- scored "
              "over the whole record (docs/source/design/jam_regression.md, "
              "\"Which window, and the box-model correction\")")

    dt = None
    # (days, series, window_start) for the request (--last-n, or the whole
    # record without it — unchanged from before #1010) and the annual window
    # (the whole run directory's trailing year, which the annual-budget
    # statistics read regardless of --last-n: see collect_windows/
    # slice_windows and the summarize()/unscored_gates() docstrings).
    if args.series_in:
        loaded = np.load(args.series_in)
        full_days = loaded["_days"]
        # Underscore keys are the reduction's own metadata, not statistics.
        full_series = {k: loaded[k] for k in loaded.files
                       if not k.startswith("_")}
        full_window_start = None
        if "_window_start" in loaded.files:
            stored = float(loaded["_window_start"])
            full_window_start = stored if np.isfinite(stored) else None
        # The timestep travels WITH the reduction: re-scoring a saved series
        # has no run directory to read it from, and without it every
        # dynamics-conservation gate would silently drop out of a re-score
        # that the original scored.
        if "_timestep_seconds" in loaded.files:
            stored = float(loaded["_timestep_seconds"])
            dt = stored if np.isfinite(stored) and stored > 0 else None
        request, annual = slice_windows(full_days, full_series, last_n,
                                        full_window_start)
    else:
        files = run_files(args.run_dir)
        if not files:
            print(f"FAIL  no chunk files in {args.run_dir}")
            return 1
        request, annual = collect_windows(files, last_n)
        dt = timestep_seconds(args.run_dir)
    days, series, window_start = request
    annual_days, annual_series, annual_window_start = annual
    if args.timestep_minutes:
        dt = args.timestep_minutes * 60.0
    if args.series_out:
        # Exactly the REQUEST window, not the annual one: a reader who sliced
        # with --last-n before saving asked for that slice to be what the
        # cache holds (test_window_start_travels_with_a_sliced_series pins
        # this). Save without --last-n to keep the whole record available to
        # every later re-score's annual window too.
        np.savez(args.series_out, _days=days,
                 _timestep_seconds=np.array(dt if dt else np.nan),
                 _window_start=np.array(np.nan if window_start is None
                                        else window_start),
                 **series)
    stats = summarize(days, series, dt, window_start,
                      annual_days, annual_series, annual_window_start)
    gates = anchor_gates(stats) + physics_gates(stats)
    print(format_table(stats, gates))
    unscored = unscored_gates(days, series, dt, window_start,
                              annual_days, annual_series, annual_window_start)
    for name, reason in unscored:
        print(f"UNSCORED  {name}: {reason}")
    # An individual UNSCORED gate is deliberately NOT a failure: the commonest
    # cause is the user's own ``--last-n``, or a species the configuration does
    # not carry, and failing those would make the tool unusable for the windows
    # it is documented to support. Scoring NOTHING is different — a report that
    # measured nothing has not passed anything, and a harness reading only the
    # exit code would otherwise see success.
    ok = bool(gates) and all(row[3] for row in gates)

    if args.reference:
        loaded = np.load(args.reference)
        # Same metadata filter as the --series-in path: underscore keys belong
        # to the reduction, not to the statistics. The reference's own
        # timestep is used so its dynamics gates are built the same way.
        ref_series = {k: loaded[k] for k in loaded.files
                      if not k.startswith("_")}
        ref_dt = None
        if "_timestep_seconds" in loaded.files:
            stored = float(loaded["_timestep_seconds"])
            ref_dt = stored if np.isfinite(stored) and stored > 0 else None
        ref = summarize(loaded["_days"], ref_series, ref_dt)
        rows = compare_to_reference(stats, ref, series)
        print(f"\n{'regression vs reference':<34}{'value':>14}  "
              f"{'expected':<24}result")
        print("-" * 80)
        for name, value, limit, good in rows:
            print(f"{name:<34}{value:>14.5g}  {limit:<24}"
                  f"{'PASS' if good else 'FAIL'}")
        ok = ok and all(row[3] for row in rows)

    # Say how much was actually measured: a PASS over zero scored gates is
    # not the same result as a PASS over all of them.
    print(f"\nAEROSOL: {'PASS' if ok else 'FAIL'} "
          f"({len(gates)} gate(s) scored, {len(unscored)} unscored)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
