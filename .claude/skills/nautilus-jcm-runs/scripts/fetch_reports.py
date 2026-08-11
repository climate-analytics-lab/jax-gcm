#!/usr/bin/env python
"""Read jcm benchmark reports off the Nautilus PVC.

    python fetch_reports.py             # print a summary table
    python fetch_reports.py --copy ./   # also copy the report files locally
    python fetch_reports.py --raw <label>   # dump one full report

Prefers `kubectl logs`: jobs echo their report to stdout on completion, so
the common case needs no cluster storage access at all. That is both simpler
and what NRP recommends — their data-movement guidance reserves `kubectl cp`
for small files and points at S3 for bulk.

`--from-pvc` falls back to mounting the volume in a short-lived pod, for
reports whose job has been garbage-collected (TTL is 24 h). That pod is
always deleted, since a leaked one holds a PVC attachment and can block
later jobs.

For production netCDF, use neither: see SKILL.md, those go to S3.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

NAMESPACE = "climate-analytics"
PVC = "jcm-bench"
POD = "jcm-report-reader"
# Small CPU-only image: this pod only cats text files, and asking for a GPU
# would consume quota and could sit Pending behind real work.
IMAGE = "busybox:1.36"


def _kubectl(*args, timeout=120, check=False):
    return subprocess.run(["kubectl", "-n", NAMESPACE, *args],
                          capture_output=True, text=True,
                          timeout=timeout, check=check)


def _spawn() -> bool:
    spec = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": POD, "namespace": NAMESPACE},
        "spec": {
            "restartPolicy": "Never",
            "containers": [{
                "name": "reader", "image": IMAGE,
                "command": ["sleep", "600"],
                "volumeMounts": [{"name": "r", "mountPath": "/reports"}],
                "resources": {"limits": {"cpu": "1", "memory": "512Mi"},
                              "requests": {"cpu": "100m", "memory": "128Mi"}},
            }],
            "volumes": [{"name": "r",
                         "persistentVolumeClaim": {"claimName": PVC}}],
        },
    }
    _kubectl("delete", "pod", POD, "--ignore-not-found", "--wait=true",
             timeout=180)
    p = subprocess.run(["kubectl", "-n", NAMESPACE, "apply", "-f", "-"],
                       input=json.dumps(spec), capture_output=True, text=True,
                       timeout=120)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
        return False
    for _ in range(60):
        r = _kubectl("get", "pod", POD, "-o",
                     "jsonpath={.status.phase}", timeout=60)
        if r.stdout.strip() == "Running":
            return True
        time.sleep(3)
    print("reader pod never reached Running", file=sys.stderr)
    return False


def _sh(cmd: str) -> str:
    return _kubectl("exec", POD, "--", "sh", "-c", cmd, timeout=180).stdout


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--copy", metavar="DIR")
    p.add_argument("--from-pvc", action="store_true",
                   help="mount the PVC in a throwaway pod instead of reading "
                        "job logs; needed once a job's TTL has expired")
    p.add_argument("--raw", metavar="LABEL")
    a = p.parse_args()

    if not a.from_pvc:
        # Job logs: no pod, no exec, no PVC mount.
        r = subprocess.run(
            ["kubectl", "-n", NAMESPACE, "get", "jobs",
             "-o", "jsonpath={.items[*].metadata.name}"],
            capture_output=True, text=True, timeout=120)
        jobs = [x for x in r.stdout.split() if x.startswith("jcm-bench")]
        if not jobs:
            print("no benchmark jobs found; --from-pvc to read older reports "
                  "off the volume")
            return 0
        for j in jobs:
            # Distinguish "still running" from "finished without a rate".
            # Both show no throughput, but only the second is a refusal —
            # conflating them makes an in-flight job look like a failure.
            st = subprocess.run(
                ["kubectl", "-n", NAMESPACE, "get", "job", j, "-o",
                 "jsonpath={.status.succeeded}/{.status.failed}/"
                 "{.status.active}"],
                capture_output=True, text=True, timeout=60).stdout
            succ, fail, active = (st.split("/") + ["", "", ""])[:3]
            log = subprocess.run(
                ["kubectl", "-n", NAMESPACE, "logs", f"job/{j}", "--tail=400"],
                capture_output=True, text=True, timeout=180).stdout
            body = log.split("===== REPORT BEGIN =====")[-1].split(
                "===== REPORT END =====")[0] if "REPORT BEGIN" in log else ""
            rate = next((ln.split("**")[1].split(" sim")[0]
                         for ln in body.splitlines()
                         if "sim days/hr" in ln and "**" in ln), None)
            if rate is None:
                rate = ("RUNNING" if active.strip() else
                        "REFUSED" if (succ.strip() or fail.strip())
                        else "PENDING")
            print(f"{j:44s} {rate:>12s}")
            if a.raw and a.raw in j:
                print(body)
        return 0

    if not _spawn():
        return 1
    try:
        if a.raw:
            print(_sh(f"cat /reports/{a.raw}/report.md 2>/dev/null"
                      f" || echo 'no report for {a.raw}'"))
            return 0

        labels = [x for x in _sh("ls /reports 2>/dev/null").split() if x]
        if not labels:
            print("no reports on the PVC yet")
            return 0

        print(f"{'label':32s} {'sim days/hr':>12s} {'peak GiB':>9s}  card")
        print("-" * 78)
        for lab in sorted(labels):
            body = _sh(f"cat /reports/{lab}/report.md 2>/dev/null")
            card = _sh(f"cat /reports/{lab}/gpu_product.txt 2>/dev/null"
                       ).strip().split(",")[0]
            rate = mem = "-"
            for line in body.splitlines():
                if "sim days/hr" in line and rate == "-":
                    rate = line.split("**")[1].split(" sim")[0] \
                        if "**" in line else "-"
                if "peak memory:" in line and mem == "-":
                    mem = line.split("peak memory:")[1].split("GiB")[0].strip()
            # A report with no rate is a run the harness REFUSED to quote —
            # truncated, NaN'd or unconverged. Say so rather than showing a
            # blank that reads as "still running".
            if rate == "-":
                rate = "REFUSED"
            print(f"{lab:32s} {rate:>12s} {mem:>9s}  {card or '?'}")

        if a.copy:
            import pathlib
            out = pathlib.Path(a.copy)
            out.mkdir(parents=True, exist_ok=True)
            for lab in labels:
                d = out / lab
                d.mkdir(exist_ok=True)
                for f in ("report.md", "result.json", "gpu_product.txt"):
                    txt = _sh(f"cat /reports/{lab}/{f} 2>/dev/null")
                    if txt.strip():
                        (d / f).write_text(txt)
            print(f"\ncopied to {out}")
        return 0
    finally:
        # Always: a leaked pod keeps the PVC attached and can block later
        # jobs from mounting it.
        _kubectl("delete", "pod", POD, "--ignore-not-found", "--wait=false",
                 timeout=120)


if __name__ == "__main__":
    raise SystemExit(main())
