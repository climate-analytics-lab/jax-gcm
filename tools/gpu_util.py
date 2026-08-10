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


# A process older than this holding a small allocation is almost certainly a
# parked notebook kernel or an abandoned session, not work in progress.
STALE_AGE_HOURS = 24.0
STALE_MAX_MIB = 4096.0


def stale_processes() -> list[dict]:
    """Long-lived, low-memory GPU processes — likely abandoned.

    On a shared box these are the real cost: a Jupyter kernel holding 1.2 GiB
    of an 80 GiB card contributes no compute contention but pins the GPU for
    anyone whose scheduler (or safety gate) treats "has a tenant" as "busy".
    Reported so the owners can be asked to clear them, rather than each user
    quietly working around it.

    Age comes from ``ps``; memory and GPU index from ``nvidia-smi``.
    """
    out = []
    for g in gpu_table():
        for proc in g["procs"]:
            try:
                mib = float(str(proc["mem"]).split()[0])
            except (ValueError, IndexError):
                continue
            if mib > STALE_MAX_MIB:
                continue
            r = subprocess.run(
                ["ps", "-o", "etimes=,user=,comm=", "-p", str(proc["pid"])],
                capture_output=True, text=True, timeout=20, check=False)
            parts = r.stdout.split()
            if len(parts) < 3:
                continue
            try:
                hours = float(parts[0]) / 3600.0
            except ValueError:
                continue
            if hours < STALE_AGE_HOURS:
                continue
            out.append({"gpu": g["index"], "pid": proc["pid"],
                        "user": parts[1], "comm": parts[2],
                        "mem_mib": mib, "age_hours": round(hours, 1),
                        "cmd": proc["name"]})
    return sorted(out, key=lambda d: -d["age_hours"])


def emptiest(need_mib: float = 0.0) -> int | None:
    """GPU with the most free memory, or None if none can fit ``need_mib``.

    For the case where nothing is idle and the run must share anyway: a
    hardcoded fallback index is worse than useless once that card fills up
    (it turns "wait" into "OOM"). Reports by FREE memory, so the choice
    degrades sensibly as the box fills.
    """
    best, best_free = None, -1.0
    for g in gpu_table():
        free = g["mem_total_mib"] - g["mem_used_mib"]
        if free > best_free:
            best, best_free = g["index"], free
    if need_mib and best_free < need_mib:
        return None
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--free", action="store_true",
                   help="print free indices only; exit 1 if none")
    p.add_argument("--wait", type=float, default=None, metavar="SECONDS",
                   help="block until a GPU frees, then print its index")
    p.add_argument("--emptiest", type=float, default=None, metavar="NEED_MIB",
                   help="print the GPU with the most free memory that can fit "
                        "NEED_MIB; exit 1 if none can")
    p.add_argument("--stale", action="store_true",
                   help=f"list GPU processes older than {STALE_AGE_HOURS:.0f} h "
                        f"holding under {STALE_MAX_MIB:.0f} MiB — likely "
                        "abandoned, worth asking the owner to clear")
    a = p.parse_args(argv)

    if a.wait is not None:
        idx = wait_for_free(a.wait)
        if idx is None:
            print("no GPU became free within the timeout", file=sys.stderr)
            return 1
        print(idx)
        return 0

    if a.emptiest is not None:
        idx = emptiest(a.emptiest)
        if idx is None:
            print("no GPU has that much free", file=sys.stderr)
            return 1
        print(idx)
        return 0

    if a.stale:
        rows = stale_processes()
        if not rows:
            print("no stale GPU processes")
            return 0
        print(f"{'GPU':>3}  {'PID':>8}  {'USER':<12} {'AGE':>9}  {'MEM':>9}  CMD")
        for r in rows:
            print(f"{r['gpu']:>3}  {r['pid']:>8}  {r['user']:<12} "
                  f"{r['age_hours'] / 24:>6.1f} d  {r['mem_mib']:>6.0f} MiB  "
                  f"{r['cmd']}")
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
