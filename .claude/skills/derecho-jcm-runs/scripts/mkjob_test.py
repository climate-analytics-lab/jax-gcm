#!/usr/bin/env python
"""Checks for mkjob.py's existing-checkpoint handling.

    python mkjob_test.py

Dependency-free and standalone: pytest does not collect ``.claude`` (its
default norecursedirs skips dotted directories), so this runs itself.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run(tmp, name, *flags):
    """Generate a job under SCRATCH=tmp; return (stdout, stderr, exit code)."""
    os.environ["SCRATCH"] = tmp
    import importlib

    import mkjob
    importlib.reload(mkjob)          # SCRATCH is read at import time
    out, err, code = io.StringIO(), io.StringIO(), 0
    # --aquaplanet/--no-emissions keep generation offline: no bundle fetch,
    # no JAM aux-input check, so the test needs nothing but a tmp dir.
    argv = ["--name", name, "--days", "1", "--aquaplanet", "--no-emissions",
            *flags]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            mkjob.main(argv)
    except SystemExit as e:
        code = e.code
    return out.getvalue(), err.getvalue(), code


def _seed(tmp, name):
    d = os.path.join(tmp, "jam_runs", name)
    os.makedirs(d, exist_ok=True)
    ckpt = os.path.join(d, "checkpoint.msgpack")
    open(ckpt, "wb").write(b"stale")
    return ckpt


def test_quiet_for_a_new_rundir(tmp):
    _, err, code = _run(tmp, "new_run")
    assert code == 0, code
    assert "checkpoint.msgpack" not in err, err


def test_warns_that_the_job_will_resume(tmp):
    ckpt = _seed(tmp, "old_run")
    out, err, code = _run(tmp, "old_run", "--resume")
    assert code == 0, code
    assert ckpt in err and "RESUME FROM it" in err, err
    # --resume means the script keeps the checkpoint rather than deleting it.
    assert "rm -f" not in out


def test_warns_that_the_job_will_delete_the_checkpoint(tmp):
    ckpt = _seed(tmp, "old_run")
    out, err, code = _run(tmp, "old_run")
    assert code == 0, code
    assert ckpt in err and "DELETE it at startup" in err, err
    assert 'rm -f "$RUNDIR"/checkpoint.msgpack' in out


def test_fresh_refuses_and_emits_no_script(tmp):
    ckpt = _seed(tmp, "old_run")
    out, err, code = _run(tmp, "old_run", "--fresh")
    assert isinstance(code, str) and ckpt in code, code
    assert "--fresh" in code and "--resume" in code
    assert out == "", out
    assert err == "", err


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_"):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(tmp)
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
