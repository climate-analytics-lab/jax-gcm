"""Tests for the JAM aerosol regression statistics (#762).

Everything here runs on synthetic Datasets: the point of the module is the
arithmetic (orientation handling, log-drift, budget closure, the gates), and a
hand-built Dataset is the only way to know the right answer. Both output
vertical conventions are exercised, because the module's whole job on a real
run directory is to give the same numbers for either.
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

import aerosol_stats as A  # noqa: E402
import jcm.constants as c  # noqa: E402

_LAT = np.array([-45.0, 45.0])
_LON = np.array([0.0, 180.0])
_DP = np.array([300.0, 250.0, 250.0, 200.0]) * 100.0   # surface-first, Pa


def _pressures(surface_first: bool):
    """Mid-level and interface pressures for a 4-layer synthetic column."""
    half_sfc_first = np.concatenate([[1000e2], 1000e2 - np.cumsum(_DP)])
    full_sfc_first = 0.5 * (half_sfc_first[:-1] + half_sfc_first[1:])
    if surface_first:
        return full_sfc_first, half_sfc_first
    return full_sfc_first, half_sfc_first[::-1]


def _field(values):
    """Broadcast a per-level profile onto (time, level, lon, lat)."""
    return np.broadcast_to(np.asarray(values, float)[None, :, None, None],
                           (1, len(_DP), len(_LON), len(_LAT))).copy()


def synthetic_chunk(*, surface_first_interfaces=True, so4_profile=None,
                    aloft_only=False, extra=None) -> xr.Dataset:
    """Build a minimal JAM-like output chunk in one of the two conventions.

    ``surface_first_interfaces=False`` reproduces a pre-#710 file: ``level``
    fields stay surface-first while ``pressure_half`` runs TOA-first and
    ``level_i`` is a bare index carrying no attributes.
    """
    full, half = _pressures(surface_first_interfaces)
    if so4_profile is None:
        so4_profile = np.array([4.0, 0.0, 0.0, 0.0]) if aloft_only is False \
            else np.array([0.0, 0.0, 0.0, 4.0])
    q_dims = ("time", "level", "lon", "lat")
    data = {
        "pressure_full": (q_dims, _field(full)),
        "pressure_half": (("time", "level_i", "lon", "lat"),
                          np.broadcast_to(half[None, :, None, None],
                                          (1, len(_DP) + 1, len(_LON),
                                           len(_LAT))).copy()),
        "m_so4_acc": (q_dims, _field(so4_profile)),
        "jam_cloud_borne.mc_so4_acc": (q_dims, _field(so4_profile)),
        "jam_state.r_dry": (("time", "mode", "level", "lon", "lat"),
                            np.full((1, 1, len(_DP), len(_LON), len(_LAT)),
                                    1e-7)),
        "activated_cdnc": (q_dims, _field(np.full(len(_DP), 1.0e8))),
        "jam_optics.aod_550": (("time", "lon", "lat"),
                               np.full((1, len(_LON), len(_LAT)), 0.12)),
        "emi_so2": (("time", "lon", "lat"),
                    np.zeros((1, len(_LON), len(_LAT)))),
    }
    coords = {"lat": _LAT, "lon": _LON, "mode": ["acc"]}
    if surface_first_interfaces:
        coords["level_i"] = ("level_i", np.linspace(1.0, 0.0, len(_DP) + 1),
                             {"positive": "down"})
    if extra:
        data.update(extra)
    return xr.Dataset(data, coords=coords)


class TestConventions:
    def test_both_conventions_give_the_same_burden(self):
        post = A.chunk_reduction(synthetic_chunk(surface_first_interfaces=True))
        pre = A.chunk_reduction(synthetic_chunk(surface_first_interfaces=False))
        assert np.isclose(post["burden_so4"], pre["burden_so4"])

    def test_burden_value_includes_both_phases(self):
        # 4 kg/kg in the bottom layer only, dp = 300 hPa, both phases:
        # 2 x 4 x 3e4 / g kg/m² -> mg/m².
        stats = A.chunk_reduction(synthetic_chunk())
        expected = 2 * 4.0 * 3.0e4 / float(c.grav) * 1e6
        assert np.isclose(stats["burden_so4"], expected, rtol=1e-6)

    def test_the_two_fixtures_really_are_different_conventions(self):
        """Guards the fixtures: the convention marker CLAUDE.md names.

        Post-#710 output carries a real ``level_i`` sigma coordinate with a
        ``positive`` attribute and surface-first interfaces; pre-#710 output
        has a bare index and TOA-first interfaces. If the fixtures ever stop
        differing, the agreement test above becomes vacuous.
        """
        post = synthetic_chunk(surface_first_interfaces=True)
        pre = synthetic_chunk(surface_first_interfaces=False)
        assert post["level_i"].attrs.get("positive") == "down"
        assert "level_i" not in pre.coords
        post_half = np.asarray(post["pressure_half"][0, :, 0, 0])
        pre_half = np.asarray(pre["pressure_half"][0, :, 0, 0])
        assert post_half[0] > post_half[-1]      # surface-first
        assert pre_half[0] < pre_half[-1]        # TOA-first

    def test_nearest_pressure_level_is_orientation_free(self):
        for surface_first in (True, False):
            ds = synthetic_chunk(surface_first_interfaces=surface_first)
            k = A.nearest_pressure_level(ds, 85000.0)
            # 850 hPa sits in the lowest layer, which is index 0 on the
            # surface-first ``level`` axis under BOTH conventions.
            assert k == 0

    def test_upper_level_fraction_uses_pressure_not_index(self):
        for surface_first in (True, False):
            low = A.chunk_reduction(synthetic_chunk(
                surface_first_interfaces=surface_first, aloft_only=False))
            high = A.chunk_reduction(synthetic_chunk(
                surface_first_interfaces=surface_first, aloft_only=True))
            assert np.isclose(low["so4_above_500hPa"], 0.0)
            assert np.isclose(high["so4_above_500hPa"], high["burden_so4"])


class TestDetection:
    def test_jam_run_detected(self):
        assert A.is_jam_run(synthetic_chunk())

    def test_non_jam_run_not_detected(self):
        ds = xr.Dataset({"macsp.od550aer": (("lat",), np.zeros(2))},
                        coords={"lat": _LAT})
        assert not A.is_jam_run(ds)


def _series(days, values, **extra):
    series = {"burden_so4": np.asarray(values, float)}
    series.update({k: np.asarray(v, float) for k, v in extra.items()})
    return np.asarray(days, float), series


class TestDrift:
    def test_stationary_series_has_no_drift(self):
        rng = np.random.default_rng(0)
        days = np.arange(5, 370, 5, dtype=float)
        values = 3.0 + 0.1 * rng.standard_normal(days.size)
        assert abs(A.log_drift(days, values)) < A.DRIFT_LIMIT_PER_DAY

    def test_super_exponential_tail_is_caught(self):
        """A #658-class runaway: flat for most of the year, then blows up."""
        days = np.arange(5, 370, 5, dtype=float)
        values = np.full(days.size, 3.0)
        tail = days >= 320
        values[tail] = 3.0 * np.exp(0.12 * (days[tail] - 320.0))
        drift = A.log_drift(days, values)
        assert drift > A.DRIFT_LIMIT_PER_DAY

    def test_drift_uses_only_the_final_window(self):
        """A spin-up ramp in the first half must not count as drift."""
        days = np.arange(5, 370, 5, dtype=float)
        values = np.where(days < 180, np.exp(0.02 * days), np.exp(0.02 * 180))
        assert abs(A.log_drift(days, values)) < A.DRIFT_LIMIT_PER_DAY


class TestUncertainty:
    def test_autocorrelation_inflates_the_standard_error(self):
        rng = np.random.default_rng(1)
        white = rng.standard_normal(200)
        red = np.empty_like(white)
        red[0] = white[0]
        for i in range(1, red.size):
            red[i] = 0.9 * red[i - 1] + white[i]
        assert A.standard_error(red) > A.standard_error(white)

    def test_constant_series_has_zero_error(self):
        assert A.standard_error(np.full(20, 2.0)) == 0.0

    def test_regression_tolerance_takes_the_larger_tier(self):
        # Tiny sigma: the 15 % floor wins.
        assert np.isclose(A.regression_tolerance(10.0, 1e-6), 1.5)
        # Large sigma: 3 sigma wins.
        assert np.isclose(A.regression_tolerance(10.0, 1.0), 3.0)


class TestBudget:
    def _closed(self, leak_fraction=0.0):
        """Emission 1 mg/m²/day of bc, all deposited, burden steady.

        Chunk labels start at the first chunk's END day, as the output files
        do; a day-0 label would name a zero-length averaging window.
        """
        days = np.arange(5.0, 371.0, 5.0)
        n = days.size
        emi = np.full(n, 1.0 / 86400e6)
        dep = emi * (1.0 - leak_fraction)
        return days, {"burden_bc": np.full(n, 5.0),
                      "emi_bc": emi,
                      "dry_bc": np.zeros(n),
                      "wet_bc": dep}

    def test_closed_budget_has_no_residual(self):
        days, series = self._closed()
        assert abs(A._budget_residual(days, series, "bc")) < 1e-9

    def test_the_flux_integral_matches_the_storage_interval(self):
        """A short record must not read as a leak from a window mismatch.

        The storage term spans chunk CENTRES, so the flux integral must too;
        averaging all N chunks over an (N-1)-chunk span understates it by 1/N,
        which on a four-chunk window is 25 % into a gate set at 5 %.
        """
        for n_chunks in (4, 10, 73):
            days = np.arange(5.0, 5.0 * n_chunks + 5.0, 5.0)
            emi = np.full(n_chunks, 1.0 / 86400e6)      # 1 mg/m²/day
            series = {"burden_bc": np.full(n_chunks, 5.0),
                      "emi_bc": emi, "dry_bc": np.zeros(n_chunks),
                      "wet_bc": emi}
            residual = A._budget_residual(days, series, "bc")
            assert abs(residual) < 1e-9, n_chunks

    def test_a_leak_is_reported_at_its_size(self):
        days, series = self._closed(leak_fraction=0.3)
        assert np.isclose(A._budget_residual(days, series, "bc"), 0.3,
                          rtol=1e-6)

    def test_storage_change_counts_against_the_residual(self):
        """Mass that accumulates is not a leak — it is in the burden."""
        days, series = self._closed(leak_fraction=0.3)
        span = days[-1] - days[0]
        growth = 0.3 * 1.0 * span          # mg/m², the un-deposited mass
        series["burden_bc"] = 5.0 + growth * (days - days[0]) / span
        assert abs(A._budget_residual(days, series, "bc")) < 1e-6

    def test_sulfur_family_uses_so2_and_dms_emissions(self):
        days = np.arange(0.0, 366.0, 5.0)
        n = days.size
        # 1 mg of SULFATE AEROSOL per m² per day, arriving as SO2 and all
        # deposited as so4. The conversion is the sulfate-aerosol molar mass
        # (NH4HSO4, 115 g/mol) over SO2's, not 96.06 over SO2's.
        emi_so2 = np.full(n, 1.0 / 86400e6) / A._SULFUR_CARRIERS["so2"]
        series = {"burden_so4": np.full(n, 4.0),
                  "gas_so2": np.zeros(n), "gas_dms": np.zeros(n),
                  "gas_h2so4": np.zeros(n),
                  "emi_so2": emi_so2, "emi_dms": np.zeros(n),
                  "emi_so4": np.zeros(n),
                  "dry_so4": np.zeros(n), "wet_so4": np.full(n, 1.0 / 86400e6)}
        assert abs(A._budget_residual(days, series, "so4")) < 1e-6

    def test_sulfate_carrier_ratios_are_the_aerosol_molar_mass(self):
        """NH4HSO4 (115 g/mol), the species jcm's ``so4`` tracer carries."""
        assert np.isclose(A._SULFUR_CARRIERS["so2"], 1.795, atol=1e-3)
        assert np.isclose(A._SULFUR_CARRIERS["dms"], 1.851, atol=1e-3)

    def test_gas_reservoirs_convert_to_sulfate_mass(self):
        """Sulfur stored as SO2 counts as the sulfate it will become."""
        days = np.arange(0.0, 366.0, 5.0)
        n = days.size
        # Nothing emitted or deposited; the aerosol burden falls by exactly
        # the sulfate equivalent of the SO2 the column gained.
        gas = np.linspace(0.0, 10.0, n)
        series = {"burden_so4": 40.0 - gas * A._SULFUR_CARRIERS["so2"],
                  "gas_so2": gas, "gas_dms": np.zeros(n),
                  "gas_h2so4": np.zeros(n),
                  "emi_so2": np.full(n, 1.0 / 86400e6),
                  "emi_dms": np.zeros(n), "emi_so4": np.zeros(n),
                  "dry_so4": np.zeros(n), "wet_so4": np.zeros(n)}
        residual = A._budget_residual(days, series, "so4")
        # The storage change cancels exactly, leaving only the emission.
        assert np.isclose(residual, 1.0, atol=1e-9)

    def test_soa_is_not_scored(self):
        """SOA's source is the SOAG gas, which has no ``emi_*`` channel."""
        assert "soa" not in A.PRIMARY_BUDGET_SPECIES

    def test_every_budget_species_has_a_burden_to_close_against(self):
        """A species with no burden can never be scored — do not advertise it.

        ``moa`` has emi/dry/wet ledgers but no entry in the shared anchor
        table, so ``burden_moa`` is never computed and its residual would be
        silently dropped from ``budget_residual_max``.
        """
        from jam_burden_report import _SPECIES
        for species in A.PRIMARY_BUDGET_SPECIES:
            assert species in _SPECIES, species
        for species in A.LIFETIME_SPECIES:
            assert species in _SPECIES, species


class TestDynamicsConservation:
    """The #713 in-step gauge: transport that does not conserve mass.

    Transport that creates mass is a different failure from aerosol physics
    that creates mass, and needs its own gate: the burden drift sees the
    symptom, this sees the cause.
    """

    def _series(self, dyn_per_second, mass=4.0e-6, n=40):
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        return days, {"budget_mass_so4": np.full(n, mass),
                      "budget_dyn_so4": np.full(n, dyn_per_second),
                      "burden_so4": np.full(n, mass * 1e6)}

    def test_a_conservative_core_passes(self):
        # 1e-5 of the mass per 720 s step.
        days, series = self._series(4.0e-6 * 1e-5 / 720.0)
        stats = A.summarize(days, series, timestep_seconds=720.0)
        assert np.isclose(stats["dyn_frac_per_step_so4"], 1e-5, rtol=1e-6)
        assert all(ok for name, _v, _l, ok in A.physics_gates(stats)
                   if name.startswith("dyn_"))

    def test_a_leaking_transport_trips_the_gate(self):
        # 1 % of the mass created per step — ten times the limit.
        days, series = self._series(4.0e-6 * 1e-2 / 720.0)
        stats = A.summarize(days, series, timestep_seconds=720.0)
        failed = [name for name, _v, _l, ok in A.physics_gates(stats)
                  if not ok]
        assert "dyn_frac_per_step_so4" in failed

    def test_destruction_trips_it_too(self):
        """The residual is signed; either direction is non-conservation."""
        days, series = self._series(-4.0e-6 * 1e-2 / 720.0)
        stats = A.summarize(days, series, timestep_seconds=720.0)
        assert stats["dyn_frac_per_step_so4"] > A.DYN_RESIDUAL_PER_STEP

    def test_unscored_without_a_timestep(self):
        """Pre-#713 output has no gauge, and a copied rundir has no config."""
        days, series = self._series(4.0e-6 * 1e-2 / 720.0)
        stats = A.summarize(days, series, timestep_seconds=None)
        assert not any(k.startswith("dyn_frac_per_step_") for k in stats)
        assert not any(name.startswith("dyn_")
                       for name, *_rest in A.physics_gates(stats))

    def test_timestep_is_read_from_the_saved_hydra_config(self, tmp_path):
        (tmp_path / ".hydra").mkdir()
        (tmp_path / ".hydra" / "config.yaml").write_text(
            "run:\n  time_step: 12\n")
        assert A.timestep_seconds(str(tmp_path)) == 720.0

    def test_timestep_is_none_without_a_config(self, tmp_path):
        assert A.timestep_seconds(str(tmp_path)) is None

    def test_chunk_reduction_picks_up_the_gauge_fields(self):
        n_lon, n_lat = len(_LON), len(_LAT)
        extra = {
            "budget_mass_so4": (("time", "lon", "lat"),
                                np.full((1, n_lon, n_lat), 4.0e-6)),
            "budget_dyn_so4": (("time", "lon", "lat"),
                               np.full((1, n_lon, n_lat), 5.0e-14)),
        }
        out = A.chunk_reduction(synthetic_chunk(extra=extra))
        assert np.isclose(out["budget_mass_so4"], 4.0e-6)
        assert np.isclose(out["budget_dyn_so4"], 5.0e-14)


class TestGates:
    def _stats(self, values, **extra):
        days, series = _series(np.arange(5, 370, 5, dtype=float), values,
                               **extra)
        return A.summarize(days, series)

    def test_clean_run_passes(self):
        rng = np.random.default_rng(2)
        days = np.arange(5, 370, 5, dtype=float)
        values = 3.0 + 0.05 * rng.standard_normal(days.size)
        stats = self._stats(values)
        rows = A.physics_gates(stats)
        assert rows and all(ok for *_rest, ok in rows)

    def test_runaway_run_fails_the_drift_gate(self):
        days = np.arange(5, 370, 5, dtype=float)
        values = np.full(days.size, 3.0)
        tail = days >= 320
        values[tail] = 3.0 * np.exp(0.12 * (days[tail] - 320.0))
        rows = A.physics_gates(self._stats(values))
        failed = [name for name, _v, _lim, ok in rows if not ok]
        assert "dlnB_dt_so4_per_day" in failed

    def test_species_the_run_never_carries_has_no_drift_statistic(self):
        """A correctly-zero burden must not fail the drift gate on a NaN."""
        n = 73
        days = np.arange(5, 370, 5, dtype=float)[:n]
        stats = A.summarize(days, {"burden_soa": np.zeros(n)})
        assert stats["burden_soa_mg_m2"] == 0.0
        assert "dlnB_dt_soa_per_day" not in stats
        assert all(ok for *_rest, ok in A.physics_gates(stats))

    def test_a_record_too_short_to_fit_is_not_scored(self):
        """``--last-n 2`` on a healthy run must not report an aerosol failure."""
        stats = A.summarize(np.array([355.0, 360.0]),
                            {"burden_so4": np.array([2.0, 2.1])})
        assert "dlnB_dt_so4_per_day" not in stats
        assert all(ok for *_rest, ok in A.physics_gates(stats))

    def test_lifetime_is_burden_over_deposition(self):
        n = 73
        days = np.arange(5, 370, 5, dtype=float)[:n]
        series = {"burden_so4": np.full(n, 4.0),
                  "dry_so4": np.zeros(n),
                  "wet_so4": np.full(n, 1.0 / 86400e6)}   # 1 mg/m²/day
        stats = A.summarize(days, series)
        assert np.isclose(stats["lifetime_so4_days"], 4.0)

    def test_lifetime_needs_both_removal_ledgers(self):
        """An absent ledger is an omission, not a zero sink.

        Trimmed output that keeps ``wet_so4`` but not ``dry_so4`` must not
        yield a lifetime from the wet sink alone — a reference comparison would
        score that number — and the omission is named in the report.
        """
        n = 73
        days = np.arange(5, 370, 5, dtype=float)[:n]
        series = {"burden_so4": np.full(n, 4.0),
                  "wet_so4": np.full(n, 1.0 / 86400e6)}      # no dry_so4
        stats = A.summarize(days, series)
        assert "lifetime_so4_days" not in stats
        reason = dict(A.unscored_gates(days, series))["lifetime_so4_days"]
        assert "dry_so4" in reason

    def test_lifetime_needs_usable_ledgers(self):
        """Present but holed (a chunk without the diagnostic) or zero: named, not scored."""
        n = 73
        days = np.arange(5, 370, 5, dtype=float)[:n]
        holed = np.full(n, 1.0 / 86400e6)
        holed[7] = np.nan
        series = {"burden_so4": np.full(n, 4.0), "dry_so4": np.zeros(n),
                  "wet_so4": holed}
        assert "lifetime_so4_days" not in A.summarize(days, series)
        assert "non-finite" in dict(A.unscored_gates(days, series))["lifetime_so4_days"]
        series = {"burden_so4": np.full(n, 4.0), "dry_so4": np.zeros(n),
                  "wet_so4": np.zeros(n)}
        assert "lifetime_so4_days" not in A.summarize(days, series)
        assert "no removal" in dict(A.unscored_gates(days, series))["lifetime_so4_days"]

    def test_hemispheric_ratio_and_upper_fraction(self):
        n = 20
        days = np.arange(5, 5 * n + 5, 5, dtype=float)
        series = {"burden_so4": np.full(n, 4.0),
                  "so4_nh": np.full(n, 6.0), "so4_sh": np.full(n, 2.0),
                  "so4_60_90N": np.full(n, 1.5),
                  "so4_above_500hPa": np.full(n, 1.0)}
        stats = A.summarize(days, series)
        assert np.isclose(stats["so4_nh_sh_ratio"], 3.0)
        assert np.isclose(stats["so4_frac_above_500hPa"], 0.25)
        assert np.isclose(stats["so4_burden_60_90N_mg_m2"], 1.5)

    def test_a_positive_residual_names_the_incomplete_ledger(self):
        """Emitted mass unaccounted for looks exactly like a missing sink.

        On output written before the #722 removal-ledger fix, ``dry_*`` omits
        Slinn dry deposition, so the gate must say so rather than report a leak.
        """
        stats = {"budget_residual_max": 0.4}
        (name, value, limit, ok), = A.physics_gates(stats)
        assert name == "budget_residual_max" and not ok
        assert "#722" in limit
        assert "#722" in A.format_table(stats, A.physics_gates(stats))

    def test_a_negative_residual_is_unambiguous_mass_creation(self):
        """No missing ledger entry can deposit more mass than entered."""
        stats = {"budget_residual_max": -0.4}
        (_name, _value, limit, ok), = A.physics_gates(stats)
        assert not ok and "#722" not in limit

    def test_regression_comparison_passes_a_matching_run(self):
        stats = self._stats(np.full(73, 3.0))
        rows = A.compare_to_reference(stats, dict(stats))
        assert rows and all(ok for *_rest, ok in rows)

    def test_regression_comparison_catches_a_shifted_burden(self):
        """A 30 % shift is past the 15 % floor even on a quiet statistic."""
        stats = self._stats(np.full(73, 3.0))
        reference = dict(stats, burden_so4_mg_m2=3.0 * 1.3)
        failed = [name for name, _v, _l, ok
                  in A.compare_to_reference(stats, reference) if not ok]
        assert "burden_so4_mg_m2" in failed

    def test_a_noisy_statistic_gets_the_wider_tolerance(self):
        """3 sigma wins over the 15 % floor when the series is noisy."""
        rng = np.random.default_rng(3)
        days = np.arange(5, 370, 5, dtype=float)
        values = 3.0 + 2.0 * rng.standard_normal(days.size)
        series = {"burden_so4": values}
        stats = A.summarize(days, series)
        reference = dict(stats, burden_so4_mg_m2=stats["burden_so4_mg_m2"] * 1.2)
        without = A.compare_to_reference(stats, reference)
        with_series = A.compare_to_reference(stats, reference, series)
        picked = lambda rows: dict(  # noqa: E731
            (name, ok) for name, _v, _l, ok in rows)["burden_so4_mg_m2"]
        assert not picked(without)      # 15 % floor alone rejects a 20 % shift
        assert picked(with_series)      # 3 sigma of a noisy series accepts it

    def test_format_table_reports_every_gate(self):
        stats = self._stats(np.full(73, 3.0))
        text = A.format_table(stats, A.physics_gates(stats))
        assert "dlnB_dt_so4_per_day" in text and "PASS" in text


class TestMinimumWindow:
    """No statistic whose fit is noise-dominated may reach a gate."""

    def _series(self, n):
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        rng = np.random.default_rng(4)
        return days, {"burden_bc": 3.0 * np.exp(0.05 * rng.standard_normal(n))}

    def test_short_window_is_unscored_not_failed(self):
        days, series = self._series(4)               # 15 days
        stats = A.summarize(days, series)
        assert "dlnB_dt_bc_per_day" not in stats
        assert not any(name.startswith("dlnB_dt_")
                       for name, *_ in A.physics_gates(stats))
        reasons = dict(A.unscored_gates(days, series))
        assert "dlnB_dt_bc_per_day" in reasons
        assert "90" in reasons["dlnB_dt_bc_per_day"]

    def test_long_window_is_scored(self):
        days, series = self._series(40)              # 195 days
        stats = A.summarize(days, series)
        assert "dlnB_dt_bc_per_day" in stats
        assert not any(name.startswith("dlnB_dt_")
                       for name, _r in A.unscored_gates(days, series))

    def test_short_window_does_not_score_the_residual_either(self):
        n = 4
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        emi = np.full(n, 1.0 / 86400e6)
        series = {"burden_bc": np.full(n, 5.0), "emi_bc": emi,
                  "dry_bc": np.zeros(n), "wet_bc": np.zeros(n)}
        stats = A.summarize(days, series)
        assert "budget_residual_max" not in stats
        assert "budget_residual_max" in dict(A.unscored_gates(days, series))

    def test_a_stationary_species_is_not_failed_by_a_short_fit(self):
        """The regression this guard exists for: --last-n on a settled run."""
        rng = np.random.default_rng(5)
        for n in (3, 4, 6):
            days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
            series = {"burden_bc": 3.0 * np.exp(0.05 *
                                                rng.standard_normal(n))}
            stats = A.summarize(days, series)
            assert all(ok for *_r, ok in A.physics_gates(stats)), n


class TestNothingPassesByAbsence:
    def test_missing_gauge_is_reported_even_with_a_timestep(self):
        """Pre-#713 output has a config but no gauge: still must not be silent."""
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_bc": np.full(n, 3.0)}
        stats = A.summarize(days, series, timestep_seconds=720.0)
        assert not any(name.startswith("dyn_")
                       for name, *_ in A.physics_gates(stats))
        reason = dict(A.unscored_gates(days, series, 720.0))["dyn_frac_per_step"]
        assert "budget_dyn" in reason and "713" in reason

    def test_missing_timestep_is_reported_when_the_gauge_is_present(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_bc": np.full(n, 3.0),
                  "budget_mass_bc": np.full(n, 4.0e-6),
                  "budget_dyn_bc": np.zeros(n)}
        reason = dict(A.unscored_gates(days, series, None))["dyn_frac_per_step"]
        assert "timestep" in reason

    def test_a_scored_dynamics_gate_is_not_reported_unscored(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_bc": np.full(n, 3.0),
                  "budget_mass_bc": np.full(n, 4.0e-6),
                  "budget_dyn_bc": np.zeros(n)}
        assert "dyn_frac_per_step" not in dict(
            A.unscored_gates(days, series, 720.0))

    def test_a_carried_species_without_its_own_gauge_is_reported(self):
        """One species gauged, another carried without one: per species, not run-wide.

        A run-wide "some gauge exists" test let a transport leak confined to
        the ungauged species pass unseen, since ``summarize`` emits no gate
        for it.
        """
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_bc": np.full(n, 3.0),
                  "budget_mass_bc": np.full(n, 4.0e-6),
                  "budget_dyn_bc": np.zeros(n),
                  "burden_so4": np.full(n, 4.0)}             # carried, ungauged
        stats = A.summarize(days, series, timestep_seconds=720.0)
        assert "dyn_frac_per_step_bc" in stats
        assert "dyn_frac_per_step_so4" not in stats
        reasons = dict(A.unscored_gates(days, series, 720.0))
        assert "budget_dyn_so4" in reasons["dyn_frac_per_step_so4"]
        assert "dyn_frac_per_step_bc" not in reasons
        assert "dyn_frac_per_step" not in reasons

    def test_a_gauge_without_its_mass_denominator_is_reported(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_bc": np.full(n, 3.0), "budget_dyn_bc": np.zeros(n)}
        stats = A.summarize(days, series, timestep_seconds=720.0)
        assert "dyn_frac_per_step_bc" not in stats
        reason = dict(A.unscored_gates(days, series, 720.0))["dyn_frac_per_step_bc"]
        assert "budget_mass_bc" in reason

    def test_an_unusable_mass_denominator_is_reported(self):
        """All-NaN or zero mass: ``summarize`` emits no gate, so the row must say why."""
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        for mass in (np.full(n, np.nan), np.zeros(n)):
            series = {"burden_bc": np.full(n, 3.0), "budget_dyn_bc": np.zeros(n),
                      "budget_mass_bc": mass}
            assert "dyn_frac_per_step_bc" not in A.summarize(days, series, 720.0)
            reason = dict(A.unscored_gates(days, series, 720.0))["dyn_frac_per_step_bc"]
            assert "finite positive" in reason

    def test_species_the_run_does_not_carry_is_reported(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_soa": np.zeros(n)}
        assert "carries no burden" in dict(
            A.unscored_gates(days, series))["dlnB_dt_soa_per_day"]


class TestResidualNaNGuard:
    def test_a_nan_storage_endpoint_yields_no_residual(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        burden = np.full(n, 5.0)
        burden[-1] = np.nan
        emi = np.full(n, 1.0 / 86400e6)
        series = {"burden_bc": burden, "emi_bc": emi,
                  "dry_bc": np.zeros(n), "wet_bc": emi}
        assert A._budget_residual(days, series, "bc") is None
        stats = A.summarize(days, series)
        assert "budget_residual_bc" not in stats
        assert "budget_residual_max" not in stats
        assert all(ok for *_r, ok in A.physics_gates(stats))


class TestRegressionTiers:
    def test_gated_statistics_are_not_double_scored(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_bc": np.full(n, 3.0)}
        stats = A.summarize(days, series)
        names = [name for name, *_ in A.compare_to_reference(stats, stats)]
        assert not any(n_.startswith(A._GATED_PREFIXES) for n_ in names)
        assert "burden_bc_mg_m2" in names

    def test_a_zero_reference_gets_an_absolute_floor(self):
        """A zero tolerance would fail any nonzero value against a zero reference."""
        assert A.tolerance_floor("burden_soa_mg_m2") > 0
        tol = A.regression_tolerance(0.0, 0.0,
                                     A.tolerance_floor("burden_soa_mg_m2"))
        assert tol > 0
        rows = A.compare_to_reference({"burden_soa_mg_m2": 0.0},
                                      {"burden_soa_mg_m2": 0.0})
        assert all(ok for *_r, ok in rows)

    def test_the_floor_does_not_swallow_a_real_change(self):
        rows = A.compare_to_reference({"burden_so4_mg_m2": 3.0},
                                      {"burden_so4_mg_m2": 0.0})
        assert not rows[0][3]


class TestJamDetection:
    def test_a_trimmed_jam_run_is_still_detected(self):
        """Mass tracers alone: a trimmed output set must not skip the gates."""
        ds = synthetic_chunk().drop_vars(
            ["jam_state.r_dry", "jam_cloud_borne.mc_so4_acc",
             "jam_optics.aod_550"])
        assert A.is_jam_run(ds)
        assert A.missing_jam_diagnostics(ds)

    def test_a_complete_run_reports_nothing_missing(self):
        assert A.missing_jam_diagnostics(synthetic_chunk()) == []


class TestChunkDiscovery:
    """``run.snapshot_interval`` writes a second stream beside the chunks."""

    def _run_dir(self, tmp_path):
        for name in ("mx_day5.nc", "mx_day10.nc", "mx_day100.nc",
                     "mx_day5_snapshots.nc", "mx_day10_snapshots.nc",
                     "mx.nc", "mx_day5.nc.provenance.json"):
            (tmp_path / name).write_text("")
        return tmp_path

    def test_snapshot_streams_are_not_chunks(self, tmp_path):
        found = [pathlib.Path(f).name
                 for f in A.run_files(str(self._run_dir(tmp_path)))]
        assert found == ["mx_day5.nc", "mx_day10.nc", "mx_day100.nc"]

    def test_ordering_is_numeric_not_lexical(self, tmp_path):
        found = A.run_files(str(self._run_dir(tmp_path)))
        assert [int(A.CHUNK_FILE.search(f).group(1)) for f in found] == [
            5, 10, 100]


class TestIncompleteLedgerIsUnscored:
    """A species absent from ``budget_residual_max`` was never checked."""

    def _days(self, n=40):
        return np.arange(5.0, 5.0 * n + 5.0, 5.0)

    def test_a_species_with_no_ledger_is_reported(self):
        days = self._days()
        n = days.size
        emi = np.full(n, 1.0 / 86400e6)
        series = {"burden_bc": np.full(n, 5.0), "emi_bc": emi,
                  "dry_bc": np.zeros(n), "wet_bc": emi,
                  # du carries a burden but no emission diagnostic at all.
                  "burden_du": np.full(n, 20.0)}
        reasons = dict(A.unscored_gates(days, series))
        assert "budget_residual_du" in reasons
        assert "ledger is incomplete" in reasons["budget_residual_du"]
        assert "budget_residual_bc" not in reasons

    def test_no_closable_species_reports_the_aggregate_gate(self):
        """Otherwise a run passes its drift gates with no closure at all."""
        days = self._days()
        series = {"burden_bc": np.full(days.size, 5.0)}   # no ledger anywhere
        stats = A.summarize(days, series)
        assert "budget_residual_max" not in stats
        reasons = dict(A.unscored_gates(days, series))
        assert "no species had a complete mass ledger" in reasons[
            "budget_residual_max"]

    def test_a_complete_ledger_is_not_reported_unscored(self):
        days = self._days()
        n = days.size
        emi = np.full(n, 1.0 / 86400e6)
        series = {"burden_bc": np.full(n, 5.0), "emi_bc": emi,
                  "dry_bc": np.zeros(n), "wet_bc": emi}
        reasons = dict(A.unscored_gates(days, series))
        assert not any(k.startswith("budget_residual") for k in reasons)


class TestSeriesRoundTrip:
    """The reduction must carry everything a re-score needs."""

    def _write(self, tmp_path, dt_minutes):
        run = tmp_path / "run"
        run.mkdir()
        (run / ".hydra").mkdir()
        (run / ".hydra" / "config.yaml").write_text(
            f"run:\n  time_step: {dt_minutes}\n")
        return run

    def test_timestep_travels_with_the_series(self, tmp_path):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        out = tmp_path / "reduced.npz"
        np.savez(out, _days=days, _timestep_seconds=np.array(720.0),
                 burden_bc=np.full(n, 3.0),
                 budget_mass_bc=np.full(n, 4.0e-6),
                 budget_dyn_bc=np.zeros(n))
        loaded = np.load(out)
        series = {k: loaded[k] for k in loaded.files if not k.startswith("_")}
        dt = float(loaded["_timestep_seconds"])
        # With the timestep the dynamics gate is scored; without it, it is not.
        assert "dyn_frac_per_step_bc" in A.summarize(days, series, dt)
        assert "dyn_frac_per_step_bc" not in A.summarize(days, series, None)
        assert "dyn_frac_per_step" not in dict(
            A.unscored_gates(days, series, dt))

    @staticmethod
    def _closed_record(n=73):
        """Fluxes and a burden consistent with them at the true chunk centres."""
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        centres = A.chunk_centres(days)
        emi = np.full(n, 2.0 / 86400e6)
        wet = np.linspace(1.0, 3.0, n) / 86400e6
        net = (emi - wet) * 86400e6
        stored = 5.0 + np.concatenate(
            [[0.0], np.cumsum(0.5 * (net[1:] + net[:-1]) * np.diff(centres))])
        return days, {"burden_bc": stored, "emi_bc": emi,
                      "dry_bc": np.zeros(n), "wet_bc": wet}

    def test_window_start_travels_with_a_sliced_series(self, tmp_path,
                                                        monkeypatch, capsys):
        """``--series-in X --last-n 19 --series-out Y``: Y knows where its window began."""
        days, series = self._closed_record()
        src, dst = tmp_path / "full.npz", tmp_path / "sliced.npz"
        np.savez(src, _days=days, _timestep_seconds=np.array(720.0), **series)
        monkeypatch.setattr(sys, "argv", ["aerosol_stats", "--series-in", str(src),
                                          "--last-n", "19", "--series-out", str(dst)])
        A.main()
        capsys.readouterr()
        out = np.load(dst)
        assert float(out["_window_start"]) == days[-20]
        assert out["_days"][0] == days[-19]
        sliced = {k: out[k] for k in out.files if not k.startswith("_")}
        # Re-scoring the saved slice reproduces the closed budget exactly.
        monkeypatch.setattr(sys, "argv", ["aerosol_stats", "--series-in", str(dst)])
        A.main()
        assert "budget_residual_bc" in capsys.readouterr().out
        assert abs(A._budget_residual(out["_days"], sliced, "bc",
                                      window_start=float(out["_window_start"]))) < 1e-9

    def test_a_negative_last_n_is_rejected(self, tmp_path, monkeypatch):
        days, series = self._closed_record()
        src = tmp_path / "full.npz"
        np.savez(src, _days=days, **series)
        monkeypatch.setattr(sys, "argv", ["aerosol_stats", "--series-in", str(src),
                                          "--last-n", "-5"])
        with pytest.raises(SystemExit):
            A.main()

    def test_chunk_day_reads_the_file_name_only(self):
        assert A.chunk_day("/scratch/spinup_day0/out_day365.nc") == 365.0
        assert A.chunk_day("/runs/x/unlabelled.nc") is None
        assert A.chunk_day("/runs/x/unlabelled.nc", 7) == 7.0

    def test_metadata_keys_never_become_statistics(self, tmp_path):
        out = tmp_path / "reduced.npz"
        np.savez(out, _days=np.arange(3.0), _timestep_seconds=np.array(720.0),
                 burden_bc=np.zeros(3))
        loaded = np.load(out)
        series = {k: loaded[k] for k in loaded.files if not k.startswith("_")}
        assert set(series) == {"burden_bc"}


class TestEveryClosureComponentRequired:
    """A missing ledger field is not a zero contribution."""

    def _closed(self, n=40):
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        emi = np.full(n, 1.0 / 86400e6)
        return days, {"burden_bc": np.full(n, 5.0), "emi_bc": emi,
                      "dry_bc": np.zeros(n), "wet_bc": emi}

    def test_baseline_closes(self):
        days, series = self._closed()
        assert abs(A._budget_residual(days, series, "bc")) < 1e-9

    def test_a_missing_removal_ledger_is_unscored_not_fabricated(self):
        for dropped in ("dry_bc", "wet_bc"):
            days, series = self._closed()
            del series[dropped]
            assert A._budget_residual(days, series, "bc") is None, dropped

    def test_dropping_wet_would_otherwise_fabricate_a_full_leak(self):
        """Guards the specific wrong answer: wet carries the whole sink."""
        days, series = self._closed()
        del series["wet_bc"]
        # Treating the absence as zero deposition would give residual == 1.0;
        # the point is that no residual is produced at all.
        assert A._budget_residual(days, series, "bc") is None
        assert "budget_residual_bc" not in A.summarize(days, series)

    def test_a_missing_gas_reservoir_is_unscored(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        emi = np.full(n, 1.0 / 86400e6)
        base = {"burden_so4": np.full(n, 4.0),
                "gas_so2": np.zeros(n), "gas_dms": np.zeros(n),
                "gas_h2so4": np.zeros(n),
                "emi_so2": emi, "emi_dms": np.zeros(n), "emi_so4": np.zeros(n),
                "dry_so4": np.zeros(n), "wet_so4": np.zeros(n)}
        assert A._budget_residual(days, dict(base), "so4") is not None
        for gas in ("gas_so2", "gas_dms", "gas_h2so4"):
            series = dict(base)
            del series[gas]
            assert A._budget_residual(days, series, "so4") is None, gas


class TestFluxIntegration:
    """The flux integral must span the same interval as the storage term.

    Every assertion here has to DISCRIMINATE the quadrature. Setting the sink
    equal to the source makes the residual identically zero for any linear
    functional, so such a test passes under the rectangle rule too and pins
    nothing; these use a source and sink with different shapes.
    """

    @staticmethod
    def _rectangle_residual(days, series, species):
        """Compute the superseded right-endpoint rule, for tests to reject."""
        span = float(days[-1] - days[0])

        def integral(key):
            return float(np.mean(series[key][1:])) * span * 86400e6

        emitted = integral(f"emi_{species}")
        deposited = integral(f"dry_{species}") + integral(f"wet_{species}")
        stored = series[f"burden_{species}"]
        return (emitted - deposited - (stored[-1] - stored[0])) / emitted

    def _ramped_sink(self, n=19):
        """Constant source, sink ramping 1->3 over a 90-day window.

        Emission and deposition integrate to the same total and the burden is
        steady, so the budget closes exactly — under the correct quadrature.
        """
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        return days, {"burden_bc": np.full(n, 5.0),
                      "emi_bc": np.full(n, 2.0 / 86400e6),
                      "dry_bc": np.zeros(n),
                      "wet_bc": np.linspace(1.0, 3.0, n) / 86400e6}

    def test_a_ramped_sink_closes(self):
        days, series = self._ramped_sink()
        assert abs(A._budget_residual(days, series, "bc")) < 1e-12

    def test_the_superseded_rule_would_not_close(self):
        """Guards the fix: reverting `integral()` must break this suite.

        The rectangle rule reads a 2.8 % leak on a budget that closes — the
        half-chunk-times-endpoint-change error, at a size comparable to the
        5 % gate it feeds.
        """
        days, series = self._ramped_sink()
        rect = self._rectangle_residual(days, series, "bc")
        assert abs(rect) > 0.02
        # ... and large enough to matter against the gate it feeds.
        assert abs(rect) > 0.5 * A.BUDGET_RESIDUAL_LIMIT

    def test_a_ramped_source_closes_too(self):
        """The same, with the shapes swapped, so neither side is privileged."""
        n = 19
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        series = {"burden_bc": np.full(n, 5.0),
                  "emi_bc": np.linspace(1.0, 3.0, n) / 86400e6,
                  "dry_bc": np.zeros(n),
                  "wet_bc": np.full(n, 2.0 / 86400e6)}
        assert abs(A._budget_residual(days, series, "bc")) < 1e-12
        assert abs(self._rectangle_residual(days, series, "bc")) > 0.02

    def test_real_storage_growth_is_accounted(self):
        """A burden that genuinely grows must show up, not cancel."""
        n = 19
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        emi = np.full(n, 2.0 / 86400e6)
        span = A.chunk_centres(days)[-1] - A.chunk_centres(days)[0]
        growth = 0.10 * 2.0 * span          # 10 % of the emitted mass retained
        series = {"burden_bc": 5.0 + growth * np.linspace(0.0, 1.0, n),
                  "emi_bc": emi, "dry_bc": np.zeros(n),
                  "wet_bc": emi * 0.9}
        assert abs(A._budget_residual(days, series, "bc")) < 1e-9

    def test_chunk_centres_are_the_window_midpoints(self):
        """The abscissa: filenames give the END day, a mean sits at the middle."""
        np.testing.assert_allclose(A.chunk_centres(np.array([5.0, 10.0, 15.0])),
                                   [2.5, 7.5, 12.5])
        # Non-uniform cadence: the centres track the varying window lengths,
        # which is the case a fixed end-day abscissa gets wrong.
        np.testing.assert_allclose(A.chunk_centres(np.array([10.0, 15.0, 35.0])),
                                   [5.0, 12.5, 25.0])
        # A slice of a longer record: the first window starts at the label of
        # the chunk before it, not at day 0.
        np.testing.assert_allclose(
            A.chunk_centres(np.array([335.0, 345.0, 355.0, 365.0]), start=325.0),
            [330.0, 340.0, 350.0, 360.0])

    def test_a_sliced_record_keeps_its_window_start(self):
        """``--last-n`` hands the preceding chunk's label to the quadrature.

        The burden is built to be exactly consistent with the fluxes at the
        true chunk centres, so the budget closes over ANY window — provided
        the window's first node is where the retained chunk really sits. With
        the start guessed as day 0 the first node lands at ``days[0] / 2`` and
        the first trapezoid spans most of the run: a closed budget reads as a
        large leak.
        """
        n = 73
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        centres = A.chunk_centres(days)
        emi = np.full(n, 2.0 / 86400e6)
        wet = np.linspace(1.0, 3.0, n) / 86400e6
        net = (emi - wet) * 86400e6                     # mg/m2/day at the centres
        stored = 5.0 + np.concatenate(
            [[0.0], np.cumsum(0.5 * (net[1:] + net[:-1]) * np.diff(centres))])
        series = {"burden_bc": stored, "emi_bc": emi,
                  "dry_bc": np.zeros(n), "wet_bc": wet}
        assert abs(A._budget_residual(days, series, "bc")) < 1e-9

        keep = 19
        sliced = {k: v[-keep:] for k, v in series.items()}
        start = float(days[-keep - 1])
        assert abs(A._budget_residual(days[-keep:], sliced, "bc",
                                      window_start=start)) < 1e-9
        # The defect this guards against: a day-0 start for the slice puts the
        # first node at days[0]/2 and reads a large leak on a closed budget.
        assert abs(A._budget_residual(days[-keep:], sliced, "bc",
                                      window_start=0.0)) > 0.05
        # A uniformly spaced slice with no start given infers it (see
        # test_a_record_that_starts_mid_run_infers_its_start), so the
        # unannotated call closes too.
        assert abs(A._budget_residual(days[-keep:], sliced, "bc")) < 1e-9

    def test_a_record_that_starts_mid_run_infers_its_start(self):
        """A resumed run in a fresh directory: uniform chunks, first label 190.

        Nothing sliced it, so no caller can pass the start; the cadence says
        the first window began one chunk earlier. A record whose first label
        IS the cadence starts at day 0, and an irregular one keeps day 0 too.
        """
        resumed = np.arange(190.0, 370.0, 5.0)
        np.testing.assert_allclose(A.chunk_centres(resumed)[0], 187.5)
        np.testing.assert_allclose(A.chunk_centres(np.arange(5.0, 100.0, 5.0))[0],
                                   2.5)
        # Short final chunk: still recognised as uniform.
        np.testing.assert_allclose(
            A.chunk_centres(np.array([190.0, 195.0, 200.0, 205.0, 207.0]))[0], 187.5)
        # An explicit start always wins over the inference.
        np.testing.assert_allclose(A.chunk_centres(resumed, start=0.0)[0], 95.0)

    def test_a_nan_flux_sample_is_unscored(self):
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        emi = np.full(n, 1.0 / 86400e6)
        emi[5] = np.nan
        series = {"burden_bc": np.full(n, 5.0), "emi_bc": emi,
                  "dry_bc": np.zeros(n), "wet_bc": np.full(n, 1.0 / 86400e6)}
        assert A._budget_residual(days, series, "bc") is None


class TestReferenceCompleteness:
    def test_a_statistic_the_run_lost_fails(self):
        """A regression that deletes a diagnostic must not report PASS."""
        reference = {"burden_so4_mg_m2": 3.0, "aod_550": 0.12}
        stats = {"burden_so4_mg_m2": 3.0}          # aod_550 no longer emitted
        rows = A.compare_to_reference(stats, reference)
        failed = {name: limit for name, _v, limit, ok in rows if not ok}
        assert "aod_550" in failed
        assert "absent" in failed["aod_550"]

    def test_a_matching_run_still_passes(self):
        reference = {"burden_so4_mg_m2": 3.0, "aod_550": 0.12}
        rows = A.compare_to_reference(dict(reference), reference)
        assert rows and all(ok for *_r, ok in rows)


class TestAnchorGates:
    """Tier 2 must apply wherever the statistics are scored."""

    def test_an_excessive_burden_fails_the_anchor(self):
        stats = {"burden_so4_mg_m2": 500.0}
        (name, _v, _l, ok), = A.anchor_gates(stats)
        assert name == "burden_so4_mg_m2" and not ok

    def test_a_plausible_burden_passes(self):
        assert all(ok for *_r, ok in A.anchor_gates({"burden_so4_mg_m2": 3.0}))

    def test_a_species_the_run_lacks_is_not_anchor_scored(self):
        assert A.anchor_gates({"burden_soa_mg_m2": 0.0}) == []
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        reasons = dict(A.unscored_gates(days, {"burden_soa": np.zeros(n)}))
        assert "burden_soa_mg_m2" in reasons

    def test_a_stationary_excessive_run_is_not_a_clean_pass(self):
        """Zero drift and a closed ledger must not excuse a huge burden."""
        n = 40
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        emi = np.full(n, 1.0 / 86400e6)
        series = {"burden_so4": np.full(n, 500.0), "emi_so4": emi,
                  "emi_so2": np.zeros(n), "emi_dms": np.zeros(n),
                  "gas_so2": np.zeros(n), "gas_dms": np.zeros(n),
                  "gas_h2so4": np.zeros(n),
                  "dry_so4": np.zeros(n), "wet_so4": emi}
        stats = A.summarize(days, series)
        assert all(ok for *_r, ok in A.physics_gates(stats))    # drift/closure fine
        assert not all(ok for *_r, ok in A.anchor_gates(stats))  # anchor is not


class TestDustEmissionBand:
    """The release gate on the annual dust budget (#808).

    Dust vanishing is the failure this exists to catch: HAM's untuned
    threshold gave jcm 5 Tg/yr against a mid-hundreds target, and every other
    statistic in this module was happy about it — the burden was stationary,
    the ledger closed, the drift was zero.
    """

    def _series(self, tg_per_yr, n=74, nlat=96):
        days = np.arange(5.0, 5.0 * n + 5.0, 5.0)
        flux = tg_per_yr * 1e9 / (A.EARTH_AREA_M2 * 86400.0 * 365.0)
        return days, {"emi_du": np.full(n, flux),
                      "nlat": np.full(n, float(nlat))}

    def test_a_budget_in_band_passes(self):
        days, series = self._series(450.0)
        stats = A.summarize(days, series)
        assert stats["dust_emission_tg_per_yr"] == pytest.approx(450.0,
                                                                 rel=1e-6)
        rows = dict((name, ok) for name, _v, _lim, ok in A.physics_gates(stats))
        assert rows["dust_emission_tg_per_yr"]

    def test_the_band_brackets_the_documented_anchors(self):
        # The band exists to sit around the parent model's budget converted to
        # this port's sub-10 um window (642 Tg/yr present-day, 485
        # pre-industrial) and to admit the calibrated T63 year (829). If any
        # of those moves outside, the band and the science register have
        # drifted apart.
        for anchor in (485.0, 642.0, 829.0):
            days, series = self._series(anchor)
            rows = dict((name, ok) for name, _v, _lim, ok
                        in A.physics_gates(A.summarize(days, series)))
            assert rows["dust_emission_tg_per_yr"], anchor

    def test_dust_that_vanished_fails(self):
        days, series = self._series(5.0)
        rows = dict((name, ok) for name, _v, _lim, ok
                    in A.physics_gates(A.summarize(days, series)))
        assert rows["dust_emission_tg_per_yr"] is False

    def test_dust_that_ran_away_fails(self):
        days, series = self._series(4000.0)
        rows = dict((name, ok) for name, _v, _lim, ok
                    in A.physics_gates(A.summarize(days, series)))
        assert rows["dust_emission_tg_per_yr"] is False

    def test_a_partial_year_is_unscored_not_failed(self):
        # 30 days of a dust season annualises to nonsense either way.
        days, series = self._series(5.0, n=6)
        stats = A.summarize(days, series)
        assert "dust_emission_tg_per_yr" not in stats
        reasons = dict(A.unscored_gates(days, series))
        assert "seasonal" in reasons["dust_emission_tg_per_yr"]

    def test_another_grid_is_unscored_not_failed(self):
        # T106 keeps HAM's untuned nduscale_reg (#810), so the T63 band must
        # not be applied to it.
        days, series = self._series(450.0, nlat=160)
        stats = A.summarize(days, series)
        assert "dust_emission_tg_per_yr" not in stats
        assert "T63" in dict(A.unscored_gates(days, series))[
            "dust_emission_tg_per_yr"]

    def test_a_run_without_the_emission_diagnostic_is_unscored(self):
        days = np.arange(5.0, 5.0 * 74 + 5.0, 5.0)
        series = {"nlat": np.full(74, 96.0)}
        assert "dust_emission_tg_per_yr" not in A.summarize(days, series)
        assert "emi_du" in dict(A.unscored_gates(days, series))[
            "dust_emission_tg_per_yr"]

    def test_uneven_chunks_are_weighted_by_their_own_window(self):
        # run/longrun.yaml writes twelve 30-day chunks and a final 5-day one.
        # A seasonal quantity whose last five days are quiet must not have
        # them counted as a full month: the unweighted mean of this series is
        # 1073 Tg/yr, the time mean 1183.
        days = np.array([30.0 * (i + 1) for i in range(12)] + [365.0])
        per_chunk = np.array([1200.0] * 12 + [0.0])
        flux = per_chunk * 1e9 / (A.EARTH_AREA_M2 * 86400.0 * 365.0)
        series = {"emi_du": flux, "nlat": np.full(13, 96.0)}
        got = A.summarize(days, series)["dust_emission_tg_per_yr"]
        assert got == pytest.approx(1200.0 * 360.0 / 365.0, rel=1e-6)
        assert got != pytest.approx(float(np.mean(per_chunk)), rel=1e-3)

    def test_a_deleted_interior_chunk_is_unscored_not_reweighted(self):
        # A missing file leaves no NaN: the next chunk still averages only
        # its own five days, but its window would span ten. Weighting it
        # twice over is exactly the silent bias the finiteness check cannot
        # see, so the gate declines to score the record at all.
        days, series = self._series(450.0)
        keep = days != 100.0
        days = days[keep]
        series = {k: v[keep] for k, v in series.items()}
        assert "dust_emission_tg_per_yr" not in A.summarize(days, series)
        reason = dict(A.unscored_gates(days, series))[
            "dust_emission_tg_per_yr"]
        assert "missing" in reason and "day 105" in reason

    def test_a_300_day_run_in_30_day_chunks_is_scored(self):
        # Labels 30..300 are 270 days apart but cover 300; measuring the gap
        # between labels would let a complete 300-day run slip past the gate
        # as "unscored", which does not fail the command.
        days = np.array([30.0 * (i + 1) for i in range(10)])
        flux = 450.0 * 1e9 / (A.EARTH_AREA_M2 * 86400.0 * 365.0)
        series = {"emi_du": np.full(10, flux), "nlat": np.full(10, 96.0)}
        stats = A.summarize(days, series)
        assert stats["dust_emission_tg_per_yr"] == pytest.approx(450.0,
                                                                 rel=1e-6)

    def test_a_short_final_chunk_is_still_scored(self):
        # run/longrun.yaml's twelve 30-day chunks plus a 5-day tail is a
        # legitimate record, not a hole.
        days = np.array([30.0 * (i + 1) for i in range(12)] + [365.0])
        flux = 450.0 * 1e9 / (A.EARTH_AREA_M2 * 86400.0 * 365.0)
        series = {"emi_du": np.full(13, flux), "nlat": np.full(13, 96.0)}
        assert "dust_emission_tg_per_yr" in A.summarize(days, series)

    def test_the_band_is_not_also_a_regression_target(self):
        # The band is deliberately wide because the target is uncertain;
        # scoring it against one reference at 15 % would quietly replace it
        # with a tuning target six times tighter.
        days, series = self._series(450.0)
        stats = A.summarize(days, series)
        reference = dict(stats, dust_emission_tg_per_yr=1200.0)
        names = [name for name, _v, _lim, _ok
                 in A.compare_to_reference(stats, reference, series)]
        assert "dust_emission_tg_per_yr" not in names

    def test_a_year_missing_chunks_is_unscored_not_averaged_over_the_rest(self):
        # A year whose emission diagnostic vanished part-way must not be
        # scored from the chunks that survived: that is a different year.
        days, series = self._series(450.0)
        series["emi_du"][10:20] = np.nan
        assert "dust_emission_tg_per_yr" not in A.summarize(days, series)
        reason = dict(A.unscored_gates(days, series))[
            "dust_emission_tg_per_yr"]
        assert "10 of 74" in reason
