"""Tests for the benchmark harness's throughput reduction.

These guard the *methodology*, which is where benchmarking goes wrong
silently: a bug here does not crash, it reports a plausible wrong number.
"""

import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmark import (  # noqa: E402
    PRESETS,
    _auto_emission_files,
    _compose_preset,
    _preset_data_files,
    _summarize_gpu,
    should_keep_output,
)
from chunk_timing import analyse as _analyse_chunks  # noqa: E402
import gpu_util  # noqa: E402


class AnalyseChunksTest(unittest.TestCase):
    def test_compile_chunk_is_discarded(self):
        """Chunk 1 contains compilation. Including it is the single most
        common way to report a wrong throughput.
        """
        r = _analyse_chunks([900.0, 200.0, 199.0], days_per_chunk=1)
        self.assertEqual(r["compile_chunk_s"], 900.0)
        self.assertAlmostEqual(r["steady_chunk_s"], 199.5)   # mean of last two
        self.assertAlmostEqual(r["s_per_sim_day"], 199.5)

    def test_requires_two_consecutive_agreeing_chunks(self):
        """A monotonically falling series has not converged, however long it
        runs — reporting its last chunk as the rate is exactly the error that
        produced a retracted jax-rrtmgp regression report.
        """
        r = _analyse_chunks([900.0, 500.0, 400.0, 330.0], days_per_chunk=1)
        self.assertFalse(r["converged"])
        self.assertIn("still settling", r["reason"])
        # It still reports something, but flagged, so a caller cannot mistake
        # silence for success.
        self.assertAlmostEqual(r["steady_chunk_s"], 365.0)

    def test_convergence_is_judged_on_the_last_two_chunks(self):
        """Not the first agreeing pair: a noisy series can match by accident
        early while still drifting, and what matters is that the run had
        settled by the time it ended. The first settle point is still
        reported, for information.
        """
        r = _analyse_chunks([900.0, 500.0, 300.0, 299.0, 298.0],
                            days_per_chunk=1)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["steady_chunk_s"], 298.5)
        self.assertEqual(r["first_settled_chunk"], 3)

    def test_early_coincidence_does_not_certify_a_drifting_run(self):
        """Chunks 2 and 3 agree, then the run drifts away again. The first
        agreeing pair would call this converged; judging on the last two
        correctly does not.
        """
        r = _analyse_chunks([900.0, 400.0, 399.0, 350.0, 300.0],
                            days_per_chunk=1)
        self.assertFalse(r["converged"])
        self.assertEqual(r["first_settled_chunk"], 2)

    def test_too_few_chunks_is_not_a_number(self):
        r = _analyse_chunks([900.0], days_per_chunk=5)
        self.assertFalse(r["converged"])
        self.assertNotIn("s_per_sim_day", r)

    def test_single_post_compile_chunk_is_provisional(self):
        """One chunk after compile has nothing to be compared against, so it
        may be shown but never called converged.
        """
        r = _analyse_chunks([900.0, 156.0], days_per_chunk=1)
        self.assertFalse(r["converged"])
        self.assertAlmostEqual(r["s_per_sim_day"], 156.0)

    def test_rates_scale_with_chunk_length(self):
        """s_per_sim_day must divide by the chunk length, or a 30-day-chunk
        run reads 30x slower than a 1-day-chunk run of the same speed.
        """
        r = _analyse_chunks([900.0, 300.0, 300.0], days_per_chunk=30)
        self.assertAlmostEqual(r["s_per_sim_day"], 10.0)
        self.assertAlmostEqual(r["sim_days_per_hour"], 360.0)

    def test_reproduces_the_published_rrtmgp_ab_numbers(self):
        """Guards the shared module against changing the conclusion of the
        jax-rrtmgp#22 A/B: the uncontended pair measured 3.5x.
        """
        base = _analyse_chunks([610.3, 446.5, 363.7, 360.8, 360.6, 361.7], 5)
        perf = _analyse_chunks([345.3, 187.6, 102.3, 103.1, 102.2, 102.5], 5)
        self.assertTrue(base["converged"] and perf["converged"])
        self.assertAlmostEqual(base["s_per_sim_day"], 72.23, places=2)
        self.assertAlmostEqual(perf["s_per_sim_day"], 20.47, places=2)
        self.assertAlmostEqual(
            base["s_per_sim_day"] / perf["s_per_sim_day"], 3.53, places=2)


class SummarizeGpuTest(unittest.TestCase):
    def test_utilisation_is_not_diluted_by_compile_idle(self):
        """Regression: a positional "drop the first 20 %" cut reported 0 %
        utilisation for a real run that was pegged at 82-98 % whenever it was
        integrating. XLA compile leaves the GPU genuinely idle, and on a short
        run compile is most of the wall clock, so the idle samples are neither
        contiguous nor a small leading fraction. This is the actual sample
        series from that run.
        """
        import tempfile
        series = ([0.0] * 24 + [1.0] + [0.0] * 3 + [83.0, 85.0, 83.0, 82.0,
                  84.0, 85.0, 82.0] + [0.0] * 8 + [85.0, 98.0, 82.0, 84.0,
                  94.0, 81.0, 82.0] + [8.0])
        rows = ["timestamp,mem_used_mib,mem_total_mib,util_gpu_pct,"
                "util_mem_pct,power_w"]
        for u in series:
            rows.append(f"t,10000,81920,{u},50,{80 if u < 5 else 250}")
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "gpu.csv"
            p.write_text("\n".join(rows) + "\n")
            g = _summarize_gpu(p)
        self.assertGreater(g["median_util_active_pct"], 80.0)
        self.assertEqual(g["max_util_pct"], 98.0)
        # 15 of 51 samples active -> the run spent most of its wall clock
        # compiling, which is exactly what a short benchmark should surface.
        self.assertAlmostEqual(g["active_fraction"], 15 / 51, places=2)
        self.assertEqual(g["median_power_active_w"], 250.0)

    def test_peak_memory_spans_compile_but_utilisation_does_not(self):
        """Peak memory must include the compile-time spike (it is a real
        provisioning requirement), while utilisation must exclude it (the GPU
        is idle or autotuning then, so including it understates the load).
        """
        import tempfile
        rows = ["timestamp,mem_used_mib,mem_total_mib,util_gpu_pct,"
                "util_mem_pct,power_w"]
        # 5 compile samples: huge memory, idle GPU. Then 5 steady samples.
        for _ in range(5):
            rows.append("t,80000,81920,0,0,60")
        for _ in range(20):
            rows.append("t,40000,81920,95,50,300")
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "gpu.csv"
            p.write_text("\n".join(rows) + "\n")
            g = _summarize_gpu(p)
        self.assertAlmostEqual(g["peak_mem_gib"], 80000 / 1024, places=2)
        self.assertEqual(g["median_util_active_pct"], 95.0)
        self.assertEqual(g["median_power_active_w"], 300.0)

    def test_missing_file_is_not_fatal(self):
        self.assertEqual(_summarize_gpu(pathlib.Path("/nonexistent.csv")), {})


if __name__ == "__main__":
    unittest.main()


class KeepOutputTest(unittest.TestCase):
    """A benchmark discards its model fields; a FAILED benchmark must not.

    Deleting the output of a run that NaN'd or stopped short destroys the only
    evidence of why — so the discard is gated on the run having completed
    cleanly, not merely on having finished.
    """

    def test_clean_run_output_is_discardable(self):
        self.assertFalse(should_keep_output(
            {"exit_code": 0, "nan_any": False, "unhealthy": False,
             "truncated": False}))

    def test_failure_modes_all_retain_output(self):
        base = {"exit_code": 0, "nan_any": False, "unhealthy": False,
                "truncated": False}
        for key, val in (("nan_any", True), ("unhealthy", True),
                         ("truncated", True), ("exit_code", 1)):
            with self.subTest(key):
                self.assertTrue(should_keep_output({**base, key: val}),
                                f"{key} must retain the output")

    def test_missing_keys_default_to_discardable(self):
        """A result dict from an older run has no unhealthy/truncated keys;
        absence must not be read as failure.
        """
        self.assertFalse(should_keep_output({"exit_code": 0}))


#: Prescribed-emission bundles that actually exist on the data mirror. The
#: mirror builds per-grid bundles only for the Gaussian grids in
#: jcm/data/mirror/build_mirror.py's ``GRIDS`` (t63, t106) at L47/L95; T119 and
#: the ne30 native columns carry no emission bundles. A JAM preset that resolves
#: ``auto`` to anything outside this set would abort build_forcing on a fetch of
#: a non-existent file — the failure Codex flagged for the T119 presets.
_MIRROR_EMISSION_BUNDLES = frozenset(
    [f"hf://bundles/{g}/{p}"
     for g in ("t63", "t106")
     for p in ("emissions_pd.nc", "dms.nc", "dust.nc")]
    + [f"hf://bundles/{g}_l{lv}/oxidants_pd.nc"
       for g in ("t63", "t106") for lv in (47, 95)]
)


class AutoEmissionPrefetchTest(unittest.TestCase):
    """The JAM ``auto`` emission inputs must join the pre-GPU prefetch.

    ``_preset_data_files`` sees only literal paths; the four emission keys
    default to ``auto`` and are resolved lazily by ``jcm.runners`` during model
    construction — after the GPU is claimed. ``_auto_emission_files`` enumerates
    them up front (Codex P2), and no shipped preset may enumerate a bundle the
    mirror does not carry (Codex P1, the T119 abort).
    """

    def test_every_shipped_preset_prefetches_only_real_bundles(self):
        """Class sweep: over every preset, the auto-emission bundles that would
        be prefetched are all real mirror products (or none).
        """
        for name in sorted(PRESETS):
            with self.subTest(name):
                auto = _auto_emission_files(_compose_preset(PRESETS[name]))
                unknown = [p for p in auto
                           if p not in _MIRROR_EMISSION_BUNDLES]
                self.assertEqual(
                    unknown, [],
                    f"{name} would prefetch non-mirror bundle(s): {unknown}")

    def test_jam_spectral_preset_enumerates_its_four_bundles(self):
        """A t63 JAM preset resolves all four keys to the t63 bundles, and they
        reach the prefetch list ``run`` iterates.
        """
        cfg = _compose_preset(PRESETS["t63-echam-jam"])
        self.assertEqual(sorted(_auto_emission_files(cfg)), sorted([
            "hf://bundles/t63/emissions_pd.nc",
            "hf://bundles/t63/dms.nc",
            "hf://bundles/t63/dust.nc",
            "hf://bundles/t63_l47/oxidants_pd.nc",
        ]))
        prefetched = _preset_data_files(PRESETS["t63-echam-jam"])
        self.assertIn("hf://bundles/t63/emissions_pd.nc", prefetched)
        self.assertIn("hf://bundles/t63_l47/oxidants_pd.nc", prefetched)

    def test_extra_nulling_emissions_drops_them_from_prefetch(self):
        """``--extra`` that nulls the auto emissions removes them from the
        prefetch list, while the preset alone still enumerates them (Codex F2).

        ``run`` prefetches from the config composed with ``args.extra`` folded
        in — the same override list the real command builds from — so an
        ``--extra`` that nulls an auto input must not leave the harness
        downloading (or failing offline on) a bundle the run never uses.
        """
        preset = PRESETS["t63-echam-jam"]
        # Preset alone: the four auto bundles are enumerated for prefetch.
        base = _preset_data_files(preset)
        self.assertIn("hf://bundles/t63/emissions_pd.nc", base)
        self.assertIn("hf://bundles/t63_l47/oxidants_pd.nc", base)
        # Same mechanism the real run uses: preset + extra nulling the inputs.
        extra = ["forcing.emissions_file=null", "forcing.dms_file=null",
                 "forcing.dust_file=null", "forcing.oxidants_file=null"]
        with_extra = _preset_data_files([*preset, *extra])
        for bundle in ("hf://bundles/t63/emissions_pd.nc",
                       "hf://bundles/t63/dms.nc", "hf://bundles/t63/dust.nc",
                       "hf://bundles/t63_l47/oxidants_pd.nc"):
            self.assertNotIn(bundle, with_extra)

    def test_t119_jam_preset_is_emission_free(self):
        """T119 has no mirror bundle; the preset nulls the four keys so nothing
        is prefetched (and build_forcing does not abort on a t119 fetch).
        """
        for name in ("ma-t119-l47", "ma-t119-l95"):
            with self.subTest(name):
                self.assertEqual(
                    _auto_emission_files(_compose_preset(PRESETS[name])), [])

    def test_non_jam_and_pyses_prefetch_no_emissions(self):
        """A non-JAM package consumes no emissions; the pySES backend's native
        grids are not the spectral-token bundles — both resolve ``auto`` to
        nothing, matching jcm.runners.
        """
        for name in ("t63-echam-rrtmgp", "t63-echam-2m", "speedy-t31",
                     "ma-ne30-l47", "ma-ne30-l95"):
            with self.subTest(name):
                self.assertEqual(
                    _auto_emission_files(_compose_preset(PRESETS[name])), [])


class GpuTenantTest(unittest.TestCase):
    """The free-GPU gate must not mistake the harness for a rival tenant.

    A JAX backend preallocates ~75 % of the device the instant it is
    touched, so any import that reaches jax before the gate makes the
    harness look like a 61 GiB occupant of the card it is about to claim.
    The gate then waits out its timeout against itself and refuses. That
    took out a whole six-job sweep, and the log ("GPU 0 is busy ... tenants:
    python(pid 364)") reads like a contended cluster rather than a bug here.
    """

    def _table(self, apps):
        """Build a gpu_table() with a stubbed nvidia-smi."""
        gpu = "0, GPU-abc, 61305, 81920, 0"

        def fake_smi(args):
            return apps if "compute-apps" in args[0] else gpu

        orig = gpu_util._smi
        gpu_util._smi = fake_smi
        try:
            return gpu_util.gpu_table()[0]
        finally:
            gpu_util._smi = orig

    def test_own_allocation_does_not_make_a_card_busy(self):
        mine = os.getpid()
        g = self._table(f"GPU-abc, {mine}, python, 61294 MiB")
        self.assertEqual(g["procs"], [], "own process must not be a tenant")
        # ...and its memory must be netted out, or the mem_used_mib half of
        # the free test still fails the card.
        self.assertLess(g["mem_used_mib"], gpu_util.FREE_MEM_MIB)
        self.assertTrue(gpu_util.is_free(g))

    def test_a_real_tenant_still_makes_a_card_busy(self):
        """The exclusion must not blind the gate to somebody else's run --
        that would be far worse than the deadlock it fixes.
        """
        g = self._table("GPU-abc, 999999, python, 61294 MiB")
        self.assertEqual(len(g["procs"]), 1)
        self.assertFalse(gpu_util.is_free(g))

    def test_mixed_tenancy_reports_the_other_process(self):
        mine = os.getpid()
        g = self._table(f"GPU-abc, {mine}, python, 61000 MiB\n"
                        f"GPU-abc, 999999, python, 305 MiB")
        self.assertEqual([p["pid"] for p in g["procs"]], ["999999"])
        self.assertFalse(gpu_util.is_free(g))
