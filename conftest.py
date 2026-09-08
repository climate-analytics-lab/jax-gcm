"""Session-wide pytest hooks.

Rationale, and how to run the gates on a memory-capped host such as a
Derecho login node: ``docs/source/design/test_suite_memory.md``.
"""

import gc
import os
import sys

import pytest


def _memory_group(item):
    """Nodeid of the class (or module) the item belongs to."""
    cls = item.getparent(pytest.Class)
    if cls is not None:
        return cls.nodeid
    module = item.getparent(pytest.Module)
    return module.nodeid if module is not None else item.nodeid


def _rss_bytes():
    """Resident set size of this pytest process in bytes, or None.

    ``/proc`` is Linux-only, so fall back to ``ru_maxrss`` — a high-water
    mark, reported in KiB on Linux but in bytes on macOS. None means the
    process size cannot be measured here at all.
    """
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    return maxrss if sys.platform == "darwin" else maxrss * 1024


# How far the process may grow between cache clears. Clearing is not free —
# it forces later tests to recompile — so it is worth doing only once the
# retained executables are actually costing memory. Zero disables the gate
# and clears at every boundary, as does an unmeasurable process size.
_MAX_GROWTH_BYTES = int(os.environ.get("JCM_TEST_CACHE_GROWTH_MB", "1024")) * 2**20
_rss_at_last_clear = None


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Release JAX's compiled executables as the process grows (#745, #704).

    A pytest process retains every executable it compiles, so a long session
    accumulates them until it is OOM-killed — which surfaces as a crashed
    xdist worker or a SIGTERM'd CI job, never as a test failure. Tests within
    a class share their compilations, so the caches are dropped only at a
    class/module boundary, and only once the process has grown by
    ``JCM_TEST_CACHE_GROWTH_MB`` since the last drop. A zero budget, or a
    platform whose RSS we cannot read, drops at every boundary instead.

    Runs ``trylast`` so pytest's own teardown has already dropped the
    class-scoped fixtures' references by the time the GC runs.
    """
    global _rss_at_last_clear
    if nextitem is not None and _memory_group(item) == _memory_group(nextitem):
        return
    rss = _rss_bytes()
    if _MAX_GROWTH_BYTES > 0 and rss is not None:
        if _rss_at_last_clear is None:
            # First boundary: everything collected is imported, so this is the
            # floor the compilation caches grow from.
            _rss_at_last_clear = rss
            return
        if rss - _rss_at_last_clear < _MAX_GROWTH_BYTES:
            return
    import jax
    jax.clear_caches()
    gc.collect()
    _rss_at_last_clear = _rss_bytes()
