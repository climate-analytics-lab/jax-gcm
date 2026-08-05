"""Find a genuinely free GPU on a shared, unscheduled workstation.

Used by ``tools/benchmark.py`` as a hard pre-flight gate, and runnable
directly to pick a card or wait for one::

    python tools/gpu_util.py                # list every GPU and its tenants
    python tools/gpu_util.py --free         # print free indices, exit 1 if none
    python tools/gpu_util.py --wait 3600    # block until one frees, print it

**Both signals must be checked.** Memory alone is not enough: an idle process
can sit at 0 % utilisation and spike later. Utilisation alone is not enough
either: a process can hold a CUDA context with no current work. The specific
incident this guards against is a colleague's Jupyter kernel that had been
parked on a card for weeks holding ~1.2 GiB -- the card read "1182 MiB, 0 %",
was judged free, and a 3-hour benchmark was started on top of it.

``nvidia-smi --query-compute-apps`` reports GPU *uuids*, not indices, so the
two queries have to be joined on uuid. That join is the whole reason this
lives in one place rather than being re-typed as a shell one-liner.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# A card is "free" below this much resident memory. Not zero: an idle A100 on
# this box still reports ~1 GiB of driver/ECC overhead.
FREE_MEM_MIB = 2048.0


def _smi(args: list[str]) -> str:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True,
                              text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_table() -> list[dict]:
    """One entry per GPU: index, uuid, memory, utilisation, tenant processes."""
    rows = []
    out = _smi(["--query-gpu=index,uuid,memory.used,memory.total,"
                "utilization.gpu", "--format=csv,noheader,nounits"])
    apps = _smi(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader"])
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, uuid, used, total, util = parts[:5]
        procs = []
        for a in apps.splitlines():
            f = [x.strip() for x in a.split(",")]
            if len(f) >= 4 and f[0] == uuid:
                procs.append({"pid": f[1], "name": f[2], "mem": f[3]})
        rows.append({
            "index": int(idx), "uuid": uuid,
            "mem_used_mib": float(used or 0), "mem_total_mib": float(total or 0),
            "util_pct": float(util or 0), "procs": procs,
        })
    return rows


def is_free(g: dict) -> bool:
    """Report whether a GPU is free.

    Requires BOTH no compute apps and near-zero resident memory.
    """
    return not g["procs"] and g["mem_used_mib"] < FREE_MEM_MIB


def free_indices() -> list[int]:
    return [g["index"] for g in gpu_table() if is_free(g)]


def describe(idx: int) -> str:
    for g in gpu_table():
        if g["index"] == idx:
            if is_free(g):
                return f"GPU {idx} is free ({g['mem_used_mib']:.0f} MiB)"
            who = ", ".join(f"{p['name'].split('/')[-1]}(pid {p['pid']}, "
                            f"{p['mem']})" for p in g["procs"]) or "no apps"
            return (f"GPU {idx} is busy: {g['mem_used_mib']:.0f} MiB used, "
                    f"{g['util_pct']:.0f}% util, tenants: {who}")
    return f"GPU {idx} not found"


def wait_for_free(timeout_s: float, poll_s: float = 30.0,
                  prefer: list[int] | None = None) -> int | None:
    """Block until some GPU is free; return its index, or None on timeout."""
    deadline = time.monotonic() + timeout_s
    while True:
        free = free_indices()
        if prefer:
            for p in prefer:
                if p in free:
                    return p
        if free:
            return free[0]
        if time.monotonic() >= deadline:
            return None
        time.sleep(min(poll_s, max(1.0, deadline - time.monotonic())))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--free", action="store_true",
                   help="print free indices only; exit 1 if none")
    p.add_argument("--wait", type=float, default=None, metavar="SECONDS",
                   help="block until a GPU frees, then print its index")
    a = p.parse_args(argv)

    if a.wait is not None:
        idx = wait_for_free(a.wait)
        if idx is None:
            print("no GPU became free within the timeout", file=sys.stderr)
            return 1
        print(idx)
        return 0

    if a.free:
        free = free_indices()
        print(" ".join(str(i) for i in free))
        return 0 if free else 1

    for g in gpu_table():
        tag = "FREE" if is_free(g) else "busy"
        who = ", ".join(f"{p['name'].split('/')[-1]}({p['mem']})"
                        for p in g["procs"])
        print(f"GPU {g['index']}  {tag:4s}  {g['mem_used_mib']:6.0f}/"
              f"{g['mem_total_mib']:.0f} MiB  {g['util_pct']:3.0f}%  {who}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
