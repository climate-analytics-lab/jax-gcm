"""Tests for run provenance capture (#591)."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from jcm import provenance


class ProbeCodeTest(unittest.TestCase):
    def test_jcm_probed_with_git_state(self):
        code = provenance.probe_code()
        self.assertIn("jcm", code)
        entry = code["jcm"]
        self.assertTrue(Path(entry["path"]).is_dir())
        # This test runs from a git worktree, so the git triple must be
        # present and well-formed.
        self.assertRegex(entry["sha"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(entry["dirty"], bool)
        if entry["dirty"]:
            self.assertRegex(entry["dirty_diff_sha"], r"^[0-9a-f]{12}$")

    def test_version_libs_recorded(self):
        code = provenance.probe_code()
        self.assertIn("jax", code)
        self.assertIn("version", code["jax"])


class ProbeEnvironmentTest(unittest.TestCase):
    def test_environment_fields(self):
        env = provenance.probe_environment()
        self.assertIn(env["jax_enable_x64"], (True, False))
        self.assertGreaterEqual(env["device_count"], 1)
        self.assertTrue(env["hostname"])
        self.assertIn("platform", env)


class DescribeInputTest(unittest.TestCase):
    def test_size_mtime_default_and_hash_opt_in(self):
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"hello provenance")
            f.flush()
            d = provenance.describe_input(f.name)
            self.assertEqual(d["size"], 16)
            self.assertIn("mtime", d)
            self.assertNotIn("sha256", d)
            try:
                os.environ["JCM_HASH_INPUTS"] = "1"
                d = provenance.describe_input(f.name)
            finally:
                os.environ.pop("JCM_HASH_INPUTS", None)
            self.assertEqual(
                d["sha256"],
                hashlib.sha256(b"hello provenance").hexdigest())

    def test_missing_file_flagged(self):
        d = provenance.describe_input("/nonexistent/never.nc")
        self.assertTrue(d["missing"])


class RegistryAndHashTest(unittest.TestCase):
    def setUp(self):
        provenance.start_run()

    def test_record_input_dedup_and_missing_skipped(self):
        provenance.record_input("/nonexistent/never.nc")
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"x")
            f.flush()
            provenance.record_input(f.name)
            provenance.record_input("hf://bundles/x.nc", f.name)
            prov = provenance.collect()
        self.assertEqual(len(prov["inputs"]), 1)
        entry = next(iter(prov["inputs"].values()))
        self.assertEqual(entry["requested"], "hf://bundles/x.nc")

    def test_run_hash_stable_and_input_sensitive(self):
        h0 = provenance.collect()["run_hash"]
        self.assertEqual(provenance.collect()["run_hash"], h0)
        self.assertRegex(h0, r"^[0-9a-f]{12}$")
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"data")
            f.flush()
            provenance.record_input(f.name)
            self.assertNotEqual(provenance.collect()["run_hash"], h0)

    def test_facts_reach_attrs_and_summary(self):
        provenance.record_fact("ozone_source", "prescribed:/x/ozone.nc")
        attrs = provenance.attrs()
        self.assertEqual(attrs["jcm_prov_ozone_source"],
                         "prescribed:/x/ozone.nc")
        self.assertIn("ozone=prescribed:/x/ozone.nc", provenance.summary())

    def test_attrs_are_netcdf_safe_strings(self):
        attrs = provenance.attrs()
        for key, value in attrs.items():
            self.assertTrue(key.startswith("jcm_prov_"), key)
            self.assertIsInstance(value, str)
        # created (run start) and written (output time) bracket the run.
        self.assertIn("jcm_prov_created", attrs)
        self.assertIn("jcm_prov_written", attrs)
        # nested parts round-trip through JSON
        self.assertIn("jcm", json.loads(attrs["jcm_prov_code"]))

    def test_config_hashed_into_attrs_and_sidecar(self):
        from omegaconf import OmegaConf
        provenance.start_run(OmegaConf.create({"run": {"total_time": 10}}))
        attrs = provenance.attrs()
        self.assertRegex(attrs["jcm_prov_config_sha"], r"^[0-9a-f]{12}$")
        with tempfile.TemporaryDirectory() as d:
            sidecar = provenance.write_sidecar(Path(d) / "out.nc")
            self.assertEqual(sidecar.name, "out.nc.provenance.json")
            full = json.loads(sidecar.read_text())
            self.assertIn("total_time: 10", full["config_yaml"])
            self.assertEqual(full["run_hash"], attrs["jcm_prov_run_hash"])


class SavePredictionsAttachesTest(unittest.TestCase):
    def test_netcdf_attrs_and_sidecar_written(self):
        import xarray as xr

        from jcm.runners import save_predictions

        class _Preds:
            def to_xarray(self):
                return xr.Dataset({"t": ("x", [1.0])})

        provenance.start_run()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run.nc"
            save_predictions(_Preds(), out)
            with xr.open_dataset(out) as ds:
                self.assertIn("jcm_prov_run_hash", ds.attrs)
                self.assertIn("jcm_prov_code", ds.attrs)
            sidecar = Path(d) / "run.nc.provenance.json"
            self.assertTrue(sidecar.exists())
            self.assertEqual(json.loads(sidecar.read_text())["run_hash"],
                             ds.attrs["jcm_prov_run_hash"])


class ResolveDataPathRecordsTest(unittest.TestCase):
    def test_resolver_feeds_the_registry(self):
        from jcm.runners import _resolve_data_path
        provenance.start_run()
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"terrain")
            f.flush()
            out = _resolve_data_path(f.name)
            self.assertEqual(out, f.name)
            self.assertIn(f.name, provenance.collect()["inputs"])
        # non-file strings pass through without registering
        provenance.start_run()
        _resolve_data_path("auto")
        self.assertEqual(provenance.collect()["inputs"], {})


if __name__ == "__main__":
    unittest.main()
