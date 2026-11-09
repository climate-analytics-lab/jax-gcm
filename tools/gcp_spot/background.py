#!/usr/bin/env python3
"""Detached supervisor for one month-long L4 or RTX PRO 6000 Spot job."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ZONES = {
    "l4": ["us-central1-a", "us-central1-b", "us-central1-c", "us-west1-b",
           "us-west1-a", "us-west1-c", "us-east1-b", "us-east1-c", "us-east1-d",
           "us-east4-c", "us-east4-a", "us-west4-a", "us-west4-c"],
    "rtx": ["us-central1-b", "us-central1-c", "us-east4-c"],
}


def simulation_arguments(continuous):
    """An uninterrupted trial uses the original single-call benchmark protocol."""
    return ["--mode", "benchmark" if continuous else "month",
            "--total-days", "30", "--save-days", "30" if continuous else "1",
            "--repeats", "1", "--truncation", "63"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=ZONES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resume-from", help="Resume a saved month checkpoint in this GCS day directory")
    parser.add_argument("--continuous", action="store_true",
                        help="One uninterrupted 30-day timed call after an identical warmup; endpoint output only")
    parser.add_argument("--preemption-retries", type=int, choices=(0, 1), default=0,
                        help="Allow at most one fresh trial after a confirmed preemption")
    args = parser.parse_args()
    if args.continuous and args.resume_from:
        parser.error("Continuous timing trials cannot resume a checkpoint")
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = {"gpu": args.gpu, "machine_type": "g2-standard-8" if args.gpu == "l4" else "g4-standard-48",
             "phase": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
             "total_days": 30, "truncation": 63, "attempts": []}
    state["protocol"] = "uninterrupted_30day" if args.continuous else "daily_chunks"
    if args.continuous:
        state.update(warmup_days=30, save_interval_days=30, timed_repeats=1,
                     checkpoint_resume_allowed=False)
    if args.resume_from:
        state["resume_from"] = args.resume_from

    def write():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    child = None

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write()
    deadline = time.monotonic() + 30 * 60
    preemption_retries = 0
    try:
        # At most three passes, only retrying allocation stockouts. Once work
        # starts, completion or failure ends this supervisor; no surprise reruns.
        for cycle in range(3 + args.preemption_retries):
            for zone in ZONES[args.gpu]:
                if time.monotonic() >= deadline:
                    break
                attempt = {"zone": zone}
                state["attempts"].append(attempt)
                state["phase"] = "requesting_capacity"
                write()
                command = [sys.executable, "-u", str(HERE / "launch.py"),
                           "--config", str(args.config.resolve()),
                           "--source-archive", str(args.source_archive.resolve()),
                           "--source-sha256", args.source_sha256,
                           "--zone", zone, "--machine-type", state["machine_type"],
                           *simulation_arguments(args.continuous), "--output", str(root / "results")]
                if args.resume_from:
                    command += ["--resume-from", args.resume_from]
                print(f"Requesting {args.gpu} in {zone}", flush=True)
                child = subprocess.Popen(command, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                lines = []
                for line in child.stdout:
                    lines.append(line)
                    print(line, end="", flush=True)
                    if line.startswith("Results: "):
                        attempt["gcs_output"] = line.removeprefix("Results: ").strip()
                        attempt["run_id"] = attempt["gcs_output"].split("/")[4]
                        write()
                    if line.startswith("Running echam-spot-"):
                        attempt["instance"] = line.split()[1].rstrip(";")
                        state["phase"] = "running"
                        write()
                code = child.wait()
                child = None
                attempt["exit_code"] = code
                output = "".join(lines)
                if code == 0:
                    directory = root / "results" / attempt["run_id"]
                    with tarfile.open(directory / "results.tar.gz") as archive:
                        archive.extractall(directory / "artifacts", filter="data")
                    state["phase"] = "complete"
                    state["result_directory"] = str(directory)
                    state["summary"] = json.loads((directory / "artifacts/results.json").read_text())
                    write()
                    return
                if "ZONE_RESOURCE_POOL_EXHAUSTED" not in output:
                    # Never combine a preempted timing with another VM. One
                    # explicitly bounded retry is a fresh warmup and full month.
                    if args.continuous and "instance" in attempt:
                        config = json.loads(args.config.read_text())
                        try:
                            operations = json.loads(subprocess.check_output([
                                "gcloud", "compute", "operations", "list",
                                f"--project={config['project']}",
                                f"--filter=targetLink:{attempt['instance']}",
                                "--format=json(operationType,status)"], text=True, timeout=60))
                            preempted = any(op.get("operationType") == "compute.instances.preempted"
                                            and op.get("status") == "DONE" for op in operations)
                        except (subprocess.SubprocessError, ValueError):
                            preempted = False
                        if preempted:
                            attempt["timing_valid"] = False
                            attempt["error"] = "preempted_discarded_trial"
                            if preemption_retries < args.preemption_retries:
                                preemption_retries += 1
                                state["fresh_preemption_retries"] = preemption_retries
                                state["phase"] = "retrying_fresh_trial"
                                deadline = time.monotonic() + 30 * 60
                                write()
                                continue
                    state["phase"] = "failed"
                    state["error"] = output[-6000:]
                    write()
                    return
                attempt["error"] = "capacity_stockout"
                write()
            if time.monotonic() >= deadline:
                break
            state["phase"] = "waiting_for_capacity"
            write()
            time.sleep(60)
        state["phase"] = "capacity_unavailable"
        write()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=150)
            except subprocess.TimeoutExpired:
                child.kill()
        state["phase"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = str(exc)
        write()
        raise


if __name__ == "__main__":
    main()
