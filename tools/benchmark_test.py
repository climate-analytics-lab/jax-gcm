"""Tests for the benchmark harness's throughput reduction.

These guard the *methodology*, which is where benchmarking goes wrong
silently: a bug here does not crash, it reports a plausible wrong number.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmark import _analyse_chunks, _summarize_gpu  # noqa: E402


class AnalyseChunksTest(unittest.TestCase):
    def test_compile_chunk_is_discarded(self):
        """Chunk 1 contains compilation. Including it is the single most
        common way to report a wrong throughput.
        """
        r = _analyse_chunks([900.0, 200.0, 199.0], days_per_chunk=1)
        self.assertEqual(r["compile_chunk_s"], 900.0)
        self.assertEqual(r["steady_chunk_s"], 199.0)
        self.assertAlmostEqual(r["s_per_sim_day"], 199.0)

    def test_requires_two_consecutive_agreeing_chunks(self):
        """A monotonically falling series has not converged, however long it
        runs — reporting its last chunk as the rate is exactly the error that
        produced a retracted jax-rrtmgp regression report.
        """
        r = _analyse_chunks([900.0, 500.0, 400.0, 330.0], days_per_chunk=1)
        self.assertFalse(r["converged"])
        self.assertIn("no two consecutive chunks", r["reason"])
        # It still reports something, but flagged, so a caller cannot mistake
        # silence for success.
        self.assertEqual(r["steady_chunk_s"], 330.0)

    def test_converges_on_first_agreeing_pair(self):
        r = _analyse_chunks([900.0, 500.0, 300.0, 299.0, 298.0],
                            days_per_chunk=1)
        self.assertTrue(r["converged"])
        self.assertEqual(r["steady_chunk_s"], 299.0)

    def test_too_few_chunks_is_not_a_number(self):
        r = _analyse_chunks([900.0], days_per_chunk=5)
        self.assertFalse(r["converged"])
        self.assertNotIn("s_per_sim_day", r)

    def test_rates_scale_with_chunk_length(self):
        """s_per_sim_day must divide by the chunk length, or a 30-day-chunk
        run reads 30x slower than a 1-day-chunk run of the same speed.
        """
        r = _analyse_chunks([900.0, 300.0, 300.0], days_per_chunk=30)
        self.assertAlmostEqual(r["s_per_sim_day"], 10.0)
        self.assertAlmostEqual(r["sim_days_per_hour"], 360.0)


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
