"""Tests for the benchmark harness's throughput reduction.

These guard the *methodology*, which is where benchmarking goes wrong
silently: a bug here does not crash, it reports a plausible wrong number.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmark import _summarize_gpu, should_keep_output  # noqa: E402
from chunk_timing import analyse as _analyse_chunks  # noqa: E402


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
