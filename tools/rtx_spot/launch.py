"""Run a tracked, isolated RTX Spot experiment with a two-hour deletion cap."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(command, timeout=120):
    result = subprocess.run([str(x) for x in command], text=True,
                            capture_output=True, timeout=timeout, cwd=ROOT)
    if result.returncode:
        print(result.stderr, flush=True)
        result.check_returncode()
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = json.loads((HERE / "plan.json").read_text())
    if run(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("Commit the experiment before launching")
    revision = run(["git", "rev-parse", "HEAD"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output.resolve() / run_id
    destination.mkdir(parents=True)
    archive = destination / "source.tar.gz"
    run(["git", "archive", "--format=tar.gz", f"--output={archive}", revision])
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    name = f"jcm-rtx-perf-{run_id.lower()}"
    prefix = f"gs://{config['bucket']}/rtx-performance/{run_id}"
    base = ["gcloud", "--quiet", f"--project={config['project']}"]
    command = base + ["compute", "instances", "create", name,
        f"--zone={plan['zone']}", f"--machine-type={plan['machine_type']}",
        "--provisioning-model=SPOT", "--no-restart-on-failure",
        "--maintenance-policy=TERMINATE", "--instance-termination-action=DELETE",
        "--max-run-duration=2h", "--boot-disk-size=50GB",
        "--boot-disk-type=hyperdisk-balanced", "--boot-disk-auto-delete",
        f"--image={config['image']}", f"--service-account={config['service_account']}",
        "--scopes=cloud-platform", "--network=default", "--no-shielded-secure-boot",
        "--labels=workload=jcm-rtx-performance,provisioning=spot",
        f"--metadata=source-uri={prefix}/source.tar.gz,source-sha256={sha},result-uri={prefix}/output",
        f"--metadata-from-file=startup-script={HERE / 'startup.sh'},constraints={HERE / 'constraints.txt'}"]
    manifest = {"name": name, "revision": revision, "archive_sha256": sha,
                "zone": plan["zone"], "result_uri": f"{prefix}/output", "command": command}
    (destination / "launch.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    run(base + ["storage", "cp", archive, f"{prefix}/source.tar.gz"])
    try:
        run(command, timeout=600)
        deadline = time.monotonic() + 2 * 3600 + 300
        while time.monotonic() < deadline:
            try:
                status = run(base + ["storage", "cat", f"{prefix}/output/status.json"])
                (destination / "status.json").write_text(status)
                run(base + ["storage", "cp", f"{prefix}/output/results.tar.gz", destination / "results.tar.gz"], timeout=240)
                print(status, flush=True)
                if json.loads(status)["exit_code"]:
                    raise RuntimeError("Experiment failed; inspect the downloaded job.log")
                return
            except subprocess.CalledProcessError:
                instances = json.loads(run(base + ["compute", "instances", "list",
                    f"--zones={plan['zone']}", f"--filter=name={name}", "--format=json(status)"]))
                if not instances or instances[0]["status"] == "TERMINATED":
                    raise RuntimeError("VM stopped without durable completion; inspect serial/log objects")
            time.sleep(30)
        raise TimeoutError("Experiment exceeded its two-hour limit")
    finally:
        present = json.loads(run(base + ["compute", "instances", "list",
            f"--zones={plan['zone']}", f"--filter=name={name}", "--format=json(name)"]))
        if present:
            run(base + ["compute", "instances", "delete", name, f"--zone={plan['zone']}"])


if __name__ == "__main__":
    main()
