"""Session-wide pytest hooks: memory ceiling and global-config isolation.

Rationale, and how to run the gates on a memory-capped host such as a
Derecho login node: ``docs/source/design/test_suite_memory.md``.
"""

import gc
import os
import sys

import pytest

# The pySES backend needs float64 for the whole life of the objects its
# ``setUpClass`` fixtures build, so its tests are exempt from the x64 pinning
# below; ``jcm/dycore/pyses/conftest.py`` schedules them last instead.
_PYSES_TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "jcm", "dycore", "pyses") + os.sep

_X64_BASELINE = False


def pytest_configure(config):
    """Record the session's starting ``jax_enable_x64`` (issue #729).

    Imported here rather than lazily at the first test because the baseline
    has to predate every test-module import: a module that flips the flag at
    collection time would otherwise define the baseline meant to detect it.
    """
    global _X64_BASELINE
    import jax
    _X64_BASELINE = bool(jax.config.read("jax_enable_x64"))


@pytest.fixture(autouse=True)
def _pin_jax_x64(request):
    """Hold ``jax_enable_x64`` at the session default around each test (#729).

    Constructing the MAM4-JAX adapter (and importing some optional
    dependencies) flips the flag process-wide, which silently runs every later
    test in that process/xdist worker in float64 and fails dtype assertions
    that have nothing to do with aerosols.
    """
    import jax
    if str(getattr(request.node, "path", "")).startswith(_PYSES_TESTS):
        yield
        return

    def _restore():
        if bool(jax.config.read("jax_enable_x64")) != _X64_BASELINE:
            jax.config.update("jax_enable_x64", _X64_BASELINE)

    _restore()
    yield
    _restore()


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
