"""Download one month from a staged ARM FTP order, then process it."""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


EXPECTED_MICROBASE_SIZE = 670_485_004


def _dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


def _download(url: str, destination: Path) -> str:
    if destination.exists() and destination.stat().st_size == EXPECTED_MICROBASE_SIZE:
        return f"already complete {destination.name}"
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "10",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(destination),
        url,
    ]
    subprocess.run(command, check=True)
    size = destination.stat().st_size
    if size != EXPECTED_MICROBASE_SIZE:
        raise RuntimeError(
            f"unexpected size for {destination.name}: {size}, "
            f"expected {EXPECTED_MICROBASE_SIZE}"
        )
    return f"downloaded {destination.name} ({size} bytes)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-id", required=True)
    parser.add_argument("--user-directory", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--delete-raw-after-verify", action="store_true")
    args = parser.parse_args(argv)
    if args.end <= args.start:
        parser.error("--end must be later than --start")
    if args.parallel < 1:
        parser.error("--parallel must be positive")

    args.staging.mkdir(parents=True, exist_ok=True)
    days = _dates(args.start, args.end)
    jobs: list[tuple[str, Path]] = []
    for day in days:
        filename = f"sgpmicrobaseC1.c1.{day:%Y%m%d}.000000.nc"
        url = (
            f"ftp://ftp.archive.arm.gov/{args.user_directory}/{args.order_id}/"
            f"sgpmicrobaseC1.c1/{filename}"
        )
        jobs.append((url, args.staging / filename))

    print(f"download stage: {len(jobs)} files, parallel={args.parallel}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(_download, url, path) for url, path in jobs]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)

    process_command = [
        sys.executable,
        str(Path(__file__).with_name("process_microbase_month.py")),
        "--microbase-dir",
        str(args.staging),
        "--start",
        args.start.isoformat(),
        "--end",
        args.end.isoformat(),
        "--output",
        str(args.output),
    ]
    if args.delete_raw_after_verify:
        process_command.append("--delete-raw-after-verify")
    print("processing stage", flush=True)
    subprocess.run(process_command, check=True)
    print("month pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
