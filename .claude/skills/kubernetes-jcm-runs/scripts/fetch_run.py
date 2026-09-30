#!/usr/bin/env python
"""Copy a production run's directory off the runs PVC.

    python fetch_run.py <run> <dest-dir> [--with-checkpoints] [--site nautilus]

A run written by ``mkrun.py`` (or by ``tools/release_validation/launch.py``)
lives at ``/runs/<run>/`` on the site's runs PVC, which only a pod can mount.
This starts a small CPU pod that mounts the volume READ-ONLY, streams the
files through ``kubectl exec ... tar``, and always deletes the pod — a leaked
one keeps the PVC attached.

Why copy the run rather than score it in a pod: every scorer (``health.py``,
``aerosol_stats.py``, jcm-monitor's ingest) then runs on the local copy
exactly as it runs on a Derecho or dev-box run directory, with no image,
checkout or environment to reproduce inside a pod.

The copy is incremental: a file already in ``dest`` at the volume's size is
skipped, and every copied file's size is checked afterwards, so re-running
after an interrupted stream (``kubectl exec`` has no resume) copies only what
is missing or short. Checkpoints are skipped unless asked for — they are the
bulk of a year's run directory (the JAM members archive one every 30 days)
and no scorer reads them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sites as site_profile  # noqa: E402  (local module)

#: A shell and tar are all the reader needs; a small image starts in seconds
#: where the multi-GB jcm image may first have to be pulled onto the node.
READER_IMAGE = "busybox:1.36"
#: jcm's restart state: the rotating checkpoint and its ``.prev``/``.monthly``
#: companions (``*.msgpack*``, ``<name>.ckpt*`` under mkrun.py) and the
#: permanent ``<prefix>_day<N>.ckpt`` archives.
CHECKPOINT = re.compile(r"\.(msgpack|ckpt)(\.|$)")


def _kubectl(site: dict, *args, **kw):
    return subprocess.run(["kubectl", "-n", site["namespace"], *args],
                          capture_output=True, text=True, **kw)


def reader_pod(site: dict, pod: str, run: str) -> dict:
    """Build the throwaway pod that mounts the runs PVC read-only."""
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": pod, "namespace": site["namespace"],
                     "labels": {"jcm-run": run}},
        "spec": {
            "restartPolicy": "Never",
            "containers": [{
                "name": "reader", "image": READER_IMAGE,
                # Bounded, so a pod this process fails to delete still ends.
                "command": ["sleep", "21600"],
                "volumeMounts": [{"name": "runs", "mountPath": "/runs",
                                  "readOnly": True}],
                "resources": {"limits": {"cpu": "1", "memory": "1Gi"},
                              "requests": {"cpu": "250m", "memory": "256Mi"}},
            }],
            "volumes": [{"name": "runs", "persistentVolumeClaim": {
                "claimName": site["runs_pvc"], "readOnly": True}}],
        },
    }


def _start(site: dict, pod: str, run: str, wait_s: float = 300.0) -> None:
    _kubectl(site, "delete", "pod", pod, "--ignore-not-found", "--wait=true",
             timeout=180)
    r = _kubectl(site, "apply", "-f", "-",
                 input=json.dumps(reader_pod(site, pod, run)), timeout=120)
    if r.returncode:
        raise SystemExit(f"could not start the reader pod {pod}: "
                         f"{r.stderr.strip()}")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        phase = _kubectl(site, "get", "pod", pod, "-o",
                         "jsonpath={.status.phase}", timeout=60).stdout.strip()
        if phase == "Running":
            return
        if phase in ("Failed", "Succeeded"):
            break
        time.sleep(3)
    raise SystemExit(f"the reader pod {pod} never reached Running "
                     f"(`kubectl -n {site['namespace']} describe pod {pod}`)")


def remote_files(site: dict, pod: str,
                 run: str) -> dict[str, tuple[int, int]]:
    """``{relative path: (size, mtime)}`` of every file under ``/runs/<run>``."""
    r = _kubectl(site, "exec", pod, "--", "sh", "-c",
                 f"cd /runs/{run} && find . -type f -exec stat -c '%s %Y %n' {{}} +",
                 timeout=600)
    if r.returncode:
        raise SystemExit(f"cannot list /runs/{run} on {site['runs_pvc']}: "
                         f"{r.stderr.strip() or r.stdout.strip()}")
    out = {}
    for line in r.stdout.splitlines():
        size, mtime, name = line.split(" ", 2)
        out[name.removeprefix("./")] = (int(size), int(mtime))
    return out


def _same(local: Path, size: int, mtime: int) -> bool:
    """Whether ``local`` is the copy tar made of a file of this size and mtime.

    Size alone is not enough: the rotating checkpoint is rewritten in place at
    a fixed size, and a retried chunk file at the same size. ``tar`` restores
    the modification time it recorded, so a file unchanged since the last
    copy matches on both.
    """
    if not local.is_file():
        return False
    st = local.stat()
    return st.st_size == size and int(st.st_mtime) == mtime


def fetch_run(run: str, dest, *, site: dict, pod: str,
              with_checkpoints: bool = False, keep=()) -> list[str]:
    """Copy ``/runs/<run>/`` into ``dest``; return what is still missing or short.

    An empty list means ``dest`` now holds every (non-checkpoint) file at
    least at the size and modification time the volume listed — at least,
    because a run still being written grows between the listing and the copy
    (see :func:`_incomplete`). A name in ``keep`` that already exists in
    ``dest`` is a caller's own record: it is never overwritten, and when the
    volume's copy differs or is absent nothing is copied and that is
    returned.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        # Inside the try: a pod that never reaches Running must be deleted too.
        _start(site, pod, run)
        files = remote_files(site, pod, run)
        kept = [f for f in keep if (dest / f).is_file()]
        wanted = {f: sm for f, sm in files.items()
                  if (with_checkpoints or not CHECKPOINT.search(f))
                  and f not in kept}
        # A differing record means the volume holds another launch under this
        # run name, and an absent one a run no launch recorded (the pod writes
        # it before anything else): copying either's outputs next to this
        # record would pass them off as the recorded launch's, so nothing is
        # copied.
        foreign = [f"{f}: /runs/{run}/{f} " + (
            "is absent, so the run on the volume is not the launch recorded "
            "here" if f not in files else
            f"differs from {dest / f}, so the run on the volume is a "
            "different launch") + "; nothing copied"
            for f in kept
            if f not in files or _kubectl(
                site, "exec", pod, "--", "cat", f"/runs/{run}/{f}",
                timeout=120).stdout != (dest / f).read_text()]
        if foreign:
            return foreign
        todo = sorted(f for f, (n, t) in wanted.items()
                      if not _same(dest / f, n, t))
        gib = sum(wanted[f][0] for f in todo) / 2**30
        print(f"# {run}: {len(todo)} of {len(wanted)} files to copy "
              f"({gib:.2f} GiB; {len(files) - len(wanted)} kept or checkpoint "
              "files skipped)", file=sys.stderr)
        if todo:
            # stderr to a file, not a pipe: a pipe nobody reads until the
            # stream ends fills after ~64 KiB of warnings and stalls kubectl.
            with tempfile.TemporaryFile() as err:
                src = subprocess.Popen(
                    ["kubectl", "-n", site["namespace"], "exec", pod, "--",
                     "tar", "cf", "-", "-C", f"/runs/{run}", *todo],
                    stdout=subprocess.PIPE, stderr=err)
                unpack = subprocess.run(["tar", "xf", "-", "-C", str(dest)],
                                        stdin=src.stdout, capture_output=True)
                src.stdout.close()
                if src.wait() or unpack.returncode:
                    err.seek(0)
                    print("# the copy stream failed: "
                          f"{err.read().decode(errors='replace').strip()} "
                          f"{unpack.stderr.decode(errors='replace').strip()}",
                          file=sys.stderr)
    finally:
        _kubectl(site, "delete", "pod", pod, "--ignore-not-found",
                 "--wait=false", timeout=120)
    return [f"{f} ({why})" for f, (n, t) in sorted(wanted.items())
            if (why := _incomplete(dest / f, n, t, f in todo))]


def _incomplete(local: Path, size: int, mtime: int, copied: bool) -> str | None:
    """Why ``local`` is not a complete copy of the listed file (None: it is).

    A file this fetch set out to copy must now carry at least the listed
    mtime, not only the listed size: a same-size rewrite (the rotating
    checkpoint, a retried chunk) whose stream failed leaves the stale local
    copy at exactly that size. tar sets a file's mtime once it is written
    whole, so a truncated one is short and a complete one has the volume's
    mtime — or a later one, with a larger size, when the run was still
    growing it.
    """
    if not local.is_file():
        return "missing"
    st = local.stat()
    if st.st_size < size:
        return "short"
    if copied and int(st.st_mtime) < mtime:
        return "stale: the copy of the rewritten file did not complete"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", help="the run's directory name under /runs")
    p.add_argument("dest", help="local directory to copy it into")
    p.add_argument("--with-checkpoints", action="store_true")
    p.add_argument("--site", default="nautilus")
    p.add_argument("--pod", default=None,
                   help="reader pod name (default jcm-fetch-<run>)")
    a = p.parse_args()
    pod = a.pod or f"jcm-fetch-{a.run}".lower().replace("_", "-")
    bad = fetch_run(a.run, a.dest, site=site_profile.get(a.site), pod=pod,
                    with_checkpoints=a.with_checkpoints)
    for line in bad:
        print(f"INCOMPLETE  {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
