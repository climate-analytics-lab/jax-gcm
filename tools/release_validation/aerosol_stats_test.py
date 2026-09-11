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
        """Emission 1 mg/m²/day of bc, all deposited, burden steady."""
        days = np.arange(0.0, 366.0, 5.0)
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

    def test_metadata_keys_never_become_statistics(self, tmp_path):
        out = tmp_path / "reduced.npz"
        np.savez(out, _days=np.arange(3.0), _timestep_seconds=np.array(720.0),
                 burden_bc=np.zeros(3))
        loaded = np.load(out)
        series = {k: loaded[k] for k in loaded.files if not k.startswith("_")}
        assert set(series) == {"burden_bc"}
