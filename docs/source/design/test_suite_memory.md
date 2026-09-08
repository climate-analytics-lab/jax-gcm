# Running the test suite: memory

The jcm test suite is memory-bound, not CPU-bound, which changes how you have
to run it — especially on a Derecho login node. This page explains what the
root `conftest.py` does about it and how to run the CI gates without fighting
the machine.

## Why a pytest process grows without bound

Every `jit`/`pmap` trace in a session leaves a compiled XLA executable in
JAX's caches, and the executable keeps its donated buffers and constants
alive. Nothing evicts them: a pytest process that has run 500 physics tests
is holding 500 executables, even though only the current test needs one. The
arrays involved are tiny — `rrtmgp_test.py` works on a handful of 10-level
columns — so the footprint is almost entirely compiled code and its
constants.

When the process hits a hard memory ceiling it is *killed*, and that never
looks like a test failure:

* under `pytest -n`, the symptom is `worker gwN crashed` and an arbitrary,
  run-dependent subset of "failures" (issue #704);
* in the single-process CI slow job, it is exit 143/137 with every completed
  test passing (issue #745);
* in a long single-process run on Derecho, it is a segfault inside
  `backend_compile_and_load` (issue #729).

Tests within one class share their compilations (same fixtures, same shapes),
so the cache is worth keeping *within* a class and worth much less across
classes. The root `conftest.py` therefore calls `jax.clear_caches()` +
`gc.collect()` from a `pytest_runtest_teardown` hook at each class/module
boundary — but only once the process has grown by more than
`JCM_TEST_CACHE_GROWTH_MB` (default 1024) since the last clear. The gate
matters: clearing costs recompilation, and on light modules that cost is real
while the memory saved is not. It reads RSS from `/proc/self/statm`, falling
back to `getrusage`; where neither is available the gate cannot be evaluated
and the caches are dropped at every boundary.

Measured on this login node, `JAX_PLATFORMS=cpu pytest -q -m "not slow"`:

| | peak RSS | wall time |
| --- | --- | --- |
| **`jcm/physics/radiation/rrtmgp_test.py`** (41 tests) | | |
| no clearing | 7.66 GB | died at test 32/41 |
| clear at every boundary | 3.41 GB | 442 s |
| clear when 1 GB has accumulated | 4.85 GB | 439 s |
| **`clouds/sundqvist_test.py` + `convection/betts_miller`** (33 tests) | | |
| no clearing | 2.86 GB | 32.2 s |
| clear at every boundary | 2.08 GB | 47.3 s |
| clear when 1 GB has accumulated | 2.38 GB | 33.1 s |

The "no clearing" rrtmgp run is the bug itself: it was OOM-killed nine tests
short of the end, as a segfault inside `backend_compile_and_load`. Clearing at
*every* boundary caps memory hardest but costs 47% on the light modules, where
consecutive classes reuse the same executables; the growth gate buys almost
all of the memory back for no measurable time anywhere.

The clear applies to every suite, not just the `slow` one: the fast xdist
suite hits the same ceiling, and a worker that dies takes its tests with it.

## Derecho login nodes: a 10 GiB cgroup, not a slow CPU

Interactive work on a Derecho login node runs inside a per-user memory
cgroup:

```bash
cat /sys/fs/cgroup/memory/user.slice/user-$(id -u).slice/memory.limit_in_bytes
# 10737418240   (10 GiB, shared by every process you have running there)
```

That is the real cause of the "nondeterministic radiation failures" and the
"single-process run segfaults" reports: with several xdist workers each
holding a few GB of executables, the cgroup OOM killer takes whichever worker
asks for memory next, which is why the failing set changes run to run and why
every one of those tests passes in isolation.

Two related login-node observations that are *not* the problem:

* `nproc` reports **1** on a login node. That is `OMP_NUM_THREADS=1` set by
  `ncarenv` — `nproc --all` and `len(os.sched_getaffinity(0))` both report
  128. Oversubscription is not what makes `-n 8` fail there.
* `place=scatter:exclhost` PBS jobs with 200 GB have also segfaulted; they
  inherit the same `OMP_NUM_THREADS=1`, but their failures were likewise
  memory-shaped, not CPU-shaped.

### How to run the gates

* **Preferred:** on a compute node, via the `jcm-local-ci` skill's PBS flow.
  A `-q develop` node gives you the whole node's memory and `-n 12` behaves.
* **On a login node:** `-n 2` at most, and expect to run heavy packages
  (`jcm/physics/radiation/`, the JAM tests) separately. Watch the cgroup with
  `cat /sys/fs/cgroup/memory/user.slice/user-$(id -u).slice/memory.usage_in_bytes`
  while it runs.
* `JCM_TEST_CACHE_GROWTH_MB=0` makes the conftest clear at every class/module
  boundary — the tightest memory setting, at the recompilation cost in the
  table above. Use it if a run is still being killed.
* Never run `pytest -n 12` on a login node: 12 workers × a few GB against a
  10 GiB cap is a guaranteed OOM, and the resulting red run carries no
  information.
