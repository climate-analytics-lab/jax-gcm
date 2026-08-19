"""Download ARM datastreams via the ARM Live Data Web Service.

Uses the raw REST webservice (https://adc.arm.gov/armlive/) through ``requests``
only — no heavy toolkit needed. Two datastreams are pulled by default:

- ``sgparmbeatmC1.c1``     ARMBEATM: atmospheric state (T/q/u/v profiles, ps)
                           -> prescribed into the single-column model.
- ``sgparmbecldradC1.c1``  ARMBECLDRAD: cloud + radiation (SW/LW irradiance,
                           cloud fraction, LWP) -> evaluation targets.

Get your access token by logging in at https://adc.arm.gov/armlive/home (you
need an ARM account). Then supply credentials by env var or flag:

    export ARM_USERID=yourid
    export ARM_TOKEN=xxxxxxxx...
    source env.sh
    python download.py --start 2018-06-01 --end 2018-07-01

Files land in ``--output`` (default ``./data``), one subfolder per datastream.
Already-present files are skipped, so re-running resumes an interrupted pull.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

ARMLIVE = "https://adc.arm.gov/armlive"
DEFAULT_DATASTREAMS = ("sgparmbeatmC1.c1", "sgparmbecldradC1.c1")


def _user_param(userid: str, token: str) -> str:
    return f"{userid}:{token}"


def list_files(userid: str, token: str, datastream: str,
               start: str, end: str) -> list[str]:
    """Return the filenames available for a datastream over [start, end]."""
    r = requests.get(
        f"{ARMLIVE}/query",
        params={"user": _user_param(userid, token), "ds": datastream,
                "start": start, "end": end, "wt": "json"},
        timeout=120,
    )
    r.raise_for_status()
    payload = r.json()
    files = payload.get("files", [])
    status = payload.get("status")
    if status and status != "success":
        raise RuntimeError(f"armlive query status={status!r} for {datastream}: "
                           f"{payload}")
    return files


def download_file(userid: str, token: str, filename: str, dest: Path) -> None:
    """Stream one file to ``dest`` via the saveData service."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(
        f"{ARMLIVE}/saveData",
        params={"user": _user_param(userid, token), "file": filename},
        stream=True, timeout=600,
    ) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dest)


def pull_datastream(userid: str, token: str, datastream: str,
                    start: str, end: str, output: Path) -> list[Path]:
    outdir = output / datastream
    files = list_files(userid, token, datastream, start, end)
    print(f"[{datastream}] {len(files)} file(s) in {start}..{end}")
    if not files:
        print(f"[{datastream}] nothing to download — check the date range / "
              "that this datastream exists for SGP.")
        return []
    got: list[Path] = []
    for i, fname in enumerate(files, 1):
        dest = outdir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ({i}/{len(files)}) skip (exists) {fname}")
        else:
            print(f"  ({i}/{len(files)}) get  {fname}")
            download_file(userid, token, fname, dest)
        got.append(dest)
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--userid", default=os.environ.get("ARM_USERID"),
                    help="ARM user id (or env ARM_USERID)")
    ap.add_argument("--token", default=os.environ.get("ARM_TOKEN"),
                    help="ARM access token (or env ARM_TOKEN)")
    ap.add_argument("--datastreams", nargs="+", default=list(DEFAULT_DATASTREAMS))
    ap.add_argument("--start", default="2018-06-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default="2018-07-01", help="YYYY-MM-DD (exclusive-ish)")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="query and print matching filenames without downloading",
    )
    args = ap.parse_args(argv)

    if not args.userid or not args.token:
        ap.error("ARM credentials required: set ARM_USERID and ARM_TOKEN "
                 "(get the token at https://adc.arm.gov/armlive/home) or pass "
                 "--userid/--token.")

    print(f"output dir: {args.output.resolve()}")
    total = 0
    for ds in args.datastreams:
        if args.list_only:
            files = list_files(args.userid, args.token, ds, args.start, args.end)
            print(f"[{ds}] {len(files)} file(s) in {args.start}..{args.end}")
            for filename in files:
                print(f"  {filename}")
            total += len(files)
            continue
        got = pull_datastream(args.userid, args.token, ds,
                              args.start, args.end, args.output)
        total += len(got)
    action = "listed" if args.list_only else "present"
    print(f"\ndone: {total} file(s) {action} across {len(args.datastreams)} "
          "datastream(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
