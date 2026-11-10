#!/usr/bin/env python3
"""Stage ECHAM, run one GPU Spot attempt, collect results and clean up."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
import uuid

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "project": "fluid-script-453604-u5",
    "bucket": "fluid-script-453604-u5-spot-jobs",
    "service_account": "jax-gcm-spot-runner@fluid-script-453604-u5.iam.gserviceaccount.com",
    "image": "projects/ml-images/global/images/common-cu129-ubuntu-2404-nvidia-580-v20261019",
    "zone": "us-central1-c",
}


def run(args, timeout=120, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, text=True,
                          capture_output=True, timeout=timeout, **kwargs).stdout.strip()


def snapshot(source, destination):
    names = run(["git", "-C", source, "ls-files", "-z"]).split("\0")
    # Include these specific untracked files; never recursively archive ignored
    # datasets, virtualenvs, credentials, or arbitrary untracked files.
    names += ["echam_speed_test.py", "docs/echam_speed_test.md"]
    files = sorted({name for name in names if name and (source / name).exists()})
    if not (source / "echam_speed_test.py").is_file():
        raise ValueError("Source checkout must contain echam_speed_test.py")
    total = 0
    for name in files:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Snapshot requires regular files: {name}")
        total += path.stat().st_size
    if total > 256 * 1024**2:
        raise ValueError(f"Refusing unexpectedly large source snapshot: {total / 1024**2:.1f} MiB")
    info = {
        "revision": run(["git", "-C", source, "rev-parse", "HEAD"]),
        "git_status": run(["git", "-C", source, "status", "--short", "--untracked-files=no"]),
        "snapshot_files": files,
        "uncompressed_source_bytes": total,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with tarfile.open(destination, "w:gz") as archive:
        for name in files:
            archive.add(source / name, arcname=f"jax-gcm/{name}", recursive=False)
        archive.add(HERE / "benchmark.py", arcname="benchmark.py")
        payload = (json.dumps(info, indent=2) + "\n").encode()
        entry = tarfile.TarInfo("source.json")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    return hashlib.sha256(destination.read_bytes()).hexdigest(), info


def prepared_snapshot(archive, expected_sha):
    """Use the exact locally reviewed archive, without rebuilding its contents."""
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if sha != expected_sha:
        raise ValueError("Prepared source archive SHA256 differs from the reviewed value")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
        if sum(member.size for member in members) > 256 * 1024**2:
            raise ValueError("Prepared archive exceeds the source size limit")
        for required in ("source.json", "benchmark.py", "jax-gcm/echam_speed_test.py"):
            if required not in tar.getnames():
                raise ValueError(f"Prepared archive is missing {required}")
        info = json.load(tar.extractfile("source.json"))
    return sha, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="terraform output -json spot_job_config")
    parser.add_argument("--source", type=Path, default=HERE.parents[1])
    parser.add_argument("--source-archive", type=Path, help="Upload an exact, previously reviewed source archive")
    parser.add_argument("--source-sha256", help="Required SHA256 for --source-archive")
    parser.add_argument("--zone", help="Override zone to try another L4 capacity pool")
    parser.add_argument("--machine-type", choices=("g2-standard-8", "g4-standard-48",
                                                  "c4d-standard-8", "c4d-standard-32"), default="g2-standard-8")
    parser.add_argument("--mode", choices=("benchmark", "month"), default="benchmark")
    parser.add_argument("--resume-from", help="GCS day directory containing a month checkpoint and progress.json")
    parser.add_argument("--total-days", type=float, default=5)
    parser.add_argument("--save-days", type=float, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--truncation", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("spot-results"))
    parser.add_argument("--dry-run", action="store_true", help="Build/inspect snapshot; no GCP calls")
    args = parser.parse_args()
    cpu = args.machine_type.startswith("c4d-")
    if cpu and args.mode != "benchmark":
        parser.error("CPU screening supports benchmark mode only")
    if bool(args.source_archive) != bool(args.source_sha256):
        parser.error("--source-archive and --source-sha256 must be supplied together")
    if not args.dry_run and args.config is None:
        parser.error("--config is required for launch; export the applied Terraform output first")
    if not (0 < args.save_days <= args.total_days) or args.repeats < 1 or args.truncation < 1:
        parser.error("Require 0 < save-days <= total-days, repeats >= 1 and truncation >= 1")
    if args.mode == "month" and (args.total_days != 30 or args.repeats != 1 or args.save_days != 1):
        parser.error("Month mode requires --total-days 30 --save-days 1 --repeats 1")
    config = json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG.copy()
    for key in DEFAULT_CONFIG:
        if not isinstance(config.get(key), str) or not config[key]:
            parser.error(f"Missing config field: {key}")
    if args.resume_from and (args.mode != "month" or not args.resume_from.startswith(f"gs://{config['bucket']}/runs/")):
        parser.error("--resume-from requires month mode and the configured job bucket")
    zone = args.zone or config["zone"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    name = f"echam-spot-{run_id}"
    prefix = f"gs://{config['bucket']}/runs/{run_id}"
    source_uri = f"{prefix}/source.tar.gz"
    result_uri = f"{prefix}/output"
    base = ["gcloud", "--quiet", f"--project={config['project']}"]
    with tempfile.TemporaryDirectory(prefix="echam-spot-") as tmp:
        if args.source_archive:
            archive = args.source_archive.resolve()
            sha, info = prepared_snapshot(archive, args.source_sha256)
        else:
            archive = Path(tmp) / "source.tar.gz"
            sha, info = snapshot(args.source.resolve(), archive)
        metadata = {
            "source-uri": source_uri, "source-sha256": sha, "result-uri": result_uri,
            "total-days": str(args.total_days), "save-days": str(args.save_days),
            "repeats": str(args.repeats), "truncation": str(args.truncation),
            "job-mode": args.mode,
            "expected-backend": "cpu" if cpu else "gpu",
        }
        startup = HERE / ("startup_cpu.sh" if cpu else "startup.sh")
        metadata_files = f"startup-script={startup},benchmark-constraints={HERE / 'constraints.txt'}"
        if args.resume_from:
            metadata["resume-from"] = args.resume_from.rstrip("/")
        if args.mode == "month":
            metadata_files += f",month-runner={HERE / 'month.py'}"
            metadata["runner-sha256"] = hashlib.sha256((HERE / "month.py").read_bytes()).hexdigest()
        g4 = args.machine_type.startswith("g4-")
        create = base + [
            "compute", "instances", "create", name, f"--zone={zone}",
            f"--machine-type={args.machine_type}",
            "--provisioning-model=SPOT", "--no-restart-on-failure",
            "--maintenance-policy=TERMINATE", "--instance-termination-action=DELETE",
            "--max-run-duration=2h", "--boot-disk-size=50GB",
            f"--boot-disk-type={'hyperdisk-balanced' if g4 or cpu else 'pd-balanced'}",
            "--boot-disk-auto-delete", f"--image={config['image']}",
            f"--service-account={config['service_account']}", "--scopes=cloud-platform",
            "--network=default", "--no-shielded-secure-boot",
            "--labels=workload=echam-speed-test,provisioning=spot",
            "--metadata=" + ",".join(f"{key}={value}" for key, value in metadata.items()),
            f"--metadata-from-file={metadata_files}",
        ]
        if not g4 and not cpu:
            create.append("--accelerator=type=nvidia-l4,count=1")
        print(f"Source: {info['revision']}; {info['uncompressed_source_bytes'] / 1024**2:.1f} MiB tracked source")
        print(f"Archive: {archive.stat().st_size / 1024**2:.1f} MiB; SHA256 {sha}")
        print(f"Results: {result_uri}")
        print(shlex.join(create), flush=True)
        if args.dry_run:
            print("Dry run complete: no cloud calls or resources created.")
            return

        # This fails before upload/compute allocation if auth or bucket setup is missing.
        run(base + ["storage", "buckets", "describe", f"gs://{config['bucket']}"])
        run(base + ["storage", "cp", archive, source_uri])
        created = False
        try:
            # Mark before the API call: a timeout can leave a successfully created
            # VM. The random name is unique to this invocation, so cleanup is safe.
            created = True
            run(create, timeout=600)
            print(f"Running {name}; waiting for durable completion marker...", flush=True)
            deadline = time.monotonic() + 2 * 3600 + 600
            status = None
            last_error = ""
            while time.monotonic() < deadline:
                try:
                    status = json.loads(run(base + ["storage", "cat", f"{result_uri}/status.json"]))
                    break
                except subprocess.CalledProcessError as exc:
                    last_error = exc.stderr
                # Distinguish deletion/preemption from an auth/network failure;
                # list succeeding with zero results confirms the VM disappeared.
                instances = json.loads(run(base + [
                    "compute", "instances", "list", f"--zones={zone}",
                    f"--filter=name={name}", "--format=json(name,status)",
                ]))
                if not instances or instances[0]["status"] == "TERMINATED":
                    # Re-read on next iteration once to allow a final marker upload.
                    try:
                        status = json.loads(run(base + ["storage", "cat", f"{result_uri}/status.json"]))
                    except subprocess.CalledProcessError:
                        raise RuntimeError(f"VM stopped without a completion marker (preemption or startup failure). Logs: {result_uri}/logs/\n{last_error}")
                    break
                time.sleep(20)
            if status is None:
                raise TimeoutError("No completion marker within runtime limit")
            destination = args.output.resolve() / run_id
            destination.mkdir(parents=True, exist_ok=False)
            run(base + ["storage", "cp", f"{result_uri}/results.tar.gz", destination / "results.tar.gz"])
            (destination / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(f"Result bundle: {destination / 'results.tar.gz'}", flush=True)
            if status["exit_code"] != 0:
                raise RuntimeError(f"Benchmark failed with exit code {status['exit_code']}; inspect bundle job.log")
            print("Benchmark succeeded, including finite-output validation.")
        finally:
            if created:
                try:
                    run(base + ["compute", "instances", "delete", name, f"--zone={zone}"])
                    print(f"Deleted {name} and its auto-delete boot disk.")
                except (subprocess.SubprocessError, OSError) as exc:
                    print(f"Cleanup could not confirm deletion: {exc}. Check {name} in {zone}. "
                          "A running VM has a 2h auto-delete limit; delete any stopped VM/disk manually.")


if __name__ == "__main__":
    # Ensure normal termination, like Ctrl-C, unwinds the VM cleanup finally.
    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or str(exc)) from exc
    except (ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from exc
