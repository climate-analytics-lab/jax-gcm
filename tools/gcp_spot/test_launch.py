"""Exercise source isolation and paid-resource cleanup without GCP access."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import launch


class SpotLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Spot Test")
        self.git("config", "user.email", "spot-test@example.invalid")
        (self.source / "tracked.py").write_text("original\n")
        (self.source / ".gitignore").write_text("large-data/\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        (self.source / "tracked.py").write_text("local edit\n")
        (self.source / "echam_speed_test.py").write_text("# untracked benchmark\n")
        (self.source / "secret.txt").write_text("must not upload\n")
        (self.source / "large-data").mkdir()
        (self.source / "large-data" / "ignored.nc").write_text("must not upload\n")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(launch.DEFAULT_CONFIG))

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args],
                              check=True, capture_output=True)

    def test_snapshot_captures_edits_and_benchmark_without_local_data(self):
        archive = self.root / "source.tar.gz"
        sha, info = launch.snapshot(self.source, archive)
        self.assertEqual(sha, hashlib.sha256(archive.read_bytes()).hexdigest())
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            self.assertIn("jax-gcm/echam_speed_test.py", names)
            self.assertIn("benchmark.py", names)
            self.assertNotIn("jax-gcm/secret.txt", names)
            self.assertNotIn("jax-gcm/large-data/ignored.nc", names)
            self.assertEqual(tar.extractfile("jax-gcm/tracked.py").read(), b"local edit\n")
            self.assertEqual(json.load(tar.extractfile("source.json"))["revision"], info["revision"])

    def test_snapshot_rejects_missing_benchmark(self):
        (self.source / "echam_speed_test.py").unlink()
        with self.assertRaisesRegex(ValueError, "echam_speed_test"):
            launch.snapshot(self.source, self.root / "source.tar.gz")

    def test_reviewed_archive_hash_is_enforced(self):
        archive = self.root / "source.tar.gz"
        sha, info = launch.snapshot(self.source, archive)
        self.assertEqual(launch.prepared_snapshot(archive, sha), (sha, info))
        with self.assertRaisesRegex(ValueError, "differs"):
            launch.prepared_snapshot(archive, "0" * 64)

    def test_snapshot_rejects_symlink_to_outside_source(self):
        (self.source / "tracked.py").unlink()
        (self.source / "tracked.py").symlink_to(self.config)
        with self.assertRaisesRegex(ValueError, "regular files"):
            launch.snapshot(self.source, self.root / "source.tar.gz")

    def invoke(self, mode="success", dry=False, extra_args=()):
        calls = []
        actual_run = launch.run

        def fake_run(command, **kwargs):
            command = [str(part) for part in command]
            if command[0] == "git":
                return actual_run(command, **kwargs)
            calls.append(command)
            if "create" in command and mode == "create_timeout":
                raise subprocess.TimeoutExpired(command, 120)
            if "cat" in command:
                if mode == "preempted":
                    raise subprocess.CalledProcessError(1, command, stderr="No such object")
                return json.dumps({"exit_code": 1 if mode == "benchmark_failure" else 0})
            if "list" in command:
                return "[]"
            if "cp" in command and command[-1].endswith("results.tar.gz"):
                Path(command[-1]).write_bytes(b"bundle")
            return ""

        argv = ["launch.py", "--source", str(self.source), "--config", str(self.config),
                "--output", str(self.root / "output")]
        if dry:
            argv.append("--dry-run")
        argv.extend(extra_args)
        with mock.patch.object(sys, "argv", argv), mock.patch.object(launch, "run", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                launch.main()
            finally:
                self.calls = calls
        return calls

    def assert_cleanup(self):
        self.assertTrue(any("delete" in call and "instances" in call for call in self.calls))
        create = next(call for call in self.calls if "create" in call)
        self.assertIn("--provisioning-model=SPOT", create)
        self.assertIn("--max-run-duration=2h", create)
        self.assertIn("--boot-disk-auto-delete", create)

    def test_dry_run_never_calls_gcloud(self):
        self.assertEqual(self.invoke(dry=True), [])

    def test_cloud_launch_sends_recorded_dependency_constraints(self):
        self.invoke()
        create = next(call for call in self.calls if "create" in call)
        metadata = next(part for part in create if part.startswith("--metadata-from-file="))
        self.assertIn(f"benchmark-constraints={launch.HERE / 'constraints.txt'}", metadata)
        self.assertIn("jax==0.10.2", (launch.HERE / "constraints.txt").read_text())

    def test_success_downloads_and_cleans_up(self):
        self.invoke()
        self.assertEqual(len(list((self.root / "output").glob("*/results.tar.gz"))), 1)
        self.assert_cleanup()

    def test_failed_benchmark_downloads_evidence_and_cleans_up(self):
        with self.assertRaisesRegex(RuntimeError, "exit code 1"):
            self.invoke("benchmark_failure")
        self.assertEqual(len(list((self.root / "output").glob("*/results.tar.gz"))), 1)
        self.assert_cleanup()

    def test_preemption_is_failure_and_cleans_up(self):
        with self.assertRaisesRegex(RuntimeError, "preemption or startup failure"):
            self.invoke("preempted")
        self.assert_cleanup()

    def test_ambiguous_create_timeout_still_cleans_up(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke("create_timeout")
        self.assert_cleanup()

    def test_g4_month_uses_full_gpu_shape_and_hyperdisk(self):
        self.invoke(extra_args=("--machine-type", "g4-standard-48", "--mode", "month",
                               "--total-days", "30", "--repeats", "1"))
        create = next(call for call in self.calls if "create" in call)
        self.assertIn("--boot-disk-type=hyperdisk-balanced", create)
        self.assertFalse(any(part.startswith("--accelerator=") for part in create))
        self.assertTrue(any("month-runner=" in part for part in create))
        self.assertTrue(any("job-mode=month" in part for part in create))
        self.assert_cleanup()


if __name__ == "__main__":
    unittest.main()
