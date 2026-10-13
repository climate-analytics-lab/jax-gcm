#!/usr/bin/env python
"""Checks for watch_job.sh's failure-signature matching.

    python watch_job_test.py

Dependency-free and standalone, like ``mkjob_test.py``: pytest does not
collect ``.claude`` (its default norecursedirs skips dotted directories).

The guarded property is that a *clean* run is never reported as failed.
``jcm.main`` echoes the whole composed config on stdout, so every run log
contains ``bail_on_unhealthy: true``; a ``FAIL_RE`` containing a bare
``unhealthy`` matched that line and reported every clean run unhealthy after
two polls. The watcher must key on the runner's health *verdicts* instead.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "watch_job.sh")

# Lines a healthy run really prints: the Hydra config echo (jcm/main.py's
# ``print(OmegaConf.to_yaml(cfg))``) plus ordinary chunk progress.
CLEAN_LOG = """\
run:
  chunk_days: 5
  bail_on_unhealthy: true
  archive_ckpt_every: 30
  mode: full
  Saved checkpoint to /scratch/run/checkpoint.msgpack
  NaN vars: 0/252
  Wall: 120.0s this chunk
Chunk done: 12.5 sim days/hr
MX_ECHAM_JAM_T63_L47_DEADBEEF_COMPLETE
"""

# The three verdicts the tooling actually emits when a run IS unhealthy:
# jcm/runners.py's bail message and its checkpoint notice, and
# tools/benchmark.py's kept-scratch line.
VERDICTS = [
    "*** atmosphere unhealthy at day 45: ['q_max'] ***",
    "  Checkpoint NOT updated (unhealthy chunk) — restart from ckpt",
    "- **kept** at /scratch/x because the run was unhealthy — investigate",
]


def fail_re() -> str:
    """Extract FAIL_RE from the script, so the test scores the real value."""
    text = open(SCRIPT).read()
    match = re.search(r"^FAIL_RE='([^']*)'", text, re.M)
    assert match, "FAIL_RE not found in watch_job.sh"
    return match.group(1)


def _grep(pattern: str, content: str) -> list[str]:
    """Reproduce the script's own ``grep -aiE ... | grep -av Lmod`` filter."""
    rx = re.compile(pattern, re.I)
    return [ln for ln in content.splitlines()
            if rx.search(ln) and "Lmod" not in ln]


def test_config_echo_does_not_match():
    hits = _grep(fail_re(), CLEAN_LOG)
    assert not hits, f"clean log matched FAIL_RE: {hits}"


def test_real_verdicts_still_match():
    pattern = fail_re()
    for line in VERDICTS:
        assert _grep(pattern, line), f"verdict no longer matched: {line}"


def _watch(log_text: str, marker: str = "RUN_COMPLETE"):
    """Run watch_job.sh once against a static log; return (code, stdout)."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "job.log")
        open(log, "w").write(log_text)
        # Stub qstat so the watcher never shells out to a real scheduler.
        bindir = os.path.join(tmp, "bin")
        os.mkdir(bindir)
        qstat = os.path.join(bindir, "qstat")
        open(qstat, "w").write("#!/bin/sh\nexit 0\n")
        os.chmod(qstat, 0o755)
        env = dict(os.environ, PATH=bindir + os.pathsep + os.environ["PATH"])
        proc = subprocess.run(
            ["bash", SCRIPT, "12345.desched", log, marker, "0"],
            capture_output=True, text=True, env=env, timeout=60)
        return proc.returncode, proc.stdout


def test_end_to_end_clean_run_completes():
    code, out = _watch(CLEAN_LOG, marker="MX_ECHAM_JAM_T63_L47_DEADBEEF_COMPLETE")
    assert code == 0, out
    assert "COMPLETE" in out and "FAILED" not in out


def test_end_to_end_unhealthy_run_fails():
    log = CLEAN_LOG.replace("Chunk done", VERDICTS[0] + "\nChunk done")
    code, out = _watch(log, marker="MX_ECHAM_JAM_T63_L47_DEADBEEF_COMPLETE")
    assert code == 1, out
    assert "FAILED" in out


def test_end_to_end_nan_count_still_fails():
    log = CLEAN_LOG.replace("NaN vars: 0/252", "NaN vars: 7/252")
    code, out = _watch(log, marker="MX_ECHAM_JAM_T63_L47_DEADBEEF_COMPLETE")
    assert code == 1, out


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print("OVERALL:", "PASS" if not failures else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
