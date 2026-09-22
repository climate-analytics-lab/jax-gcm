"""Session-wide pytest hooks: memory ceiling and global-config isolation.

Rationale, and how to run the gates on a memory-capped host such as a
Derecho login node: ``docs/source/design/test_suite_memory.md``.
"""

import gc
import logging
import os
import sys

import pytest

# The pySES backend needs float64 for the whole life of the objects its
# ``setUpClass`` fixtures build, so its tests are exempt from the x64 pinning
# below; ``jcm/dycore/pyses/conftest.py`` schedules them last instead.
_PYSES_TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "jcm", "dycore", "pyses") + os.sep

_X64_BASELINE = False

# Level and propagation of every ``jcm`` logger as the session found them,
# keyed by name. Normally empty: ``pytest_configure`` runs before collection
# imports ``jcm``, and nothing in the package sets a level at import time, so
# in practice every logger restores to the ``NOTSET`` default below. It is
# captured anyway so that a package that *did* set one (issue #817 weighs
# attaching jcm's handler to the ``jcm`` logger, which would) is preserved
# rather than silently flattened by the restore. See ``_pin_logging_levels``.
_LOGGING_BASELINE = {}


def pytest_configure(config):
    """Record the session's starting global config (#729, #815).

    Both baselines are taken here rather than lazily at the first test
    because they have to predate every test-module import: a module (or a
    ``setUpClass``, which runs before any function-scoped fixture) that flips
    ``jax_enable_x64`` — or builds a quiet ``Model`` — would otherwise define
    the baseline meant to detect it.
    """
    # Never preallocate the GPU for a test session. XLA's default is to claim
    # 75 % of the card at backend initialisation, and merely *importing* a
    # test module that reaches jcm is enough to trigger that (#859: the
    # SPEEDY lookup tables are built on jcm's import chain) — measured at
    # 61,214 MiB of an 80 GB A100 for a process whose test then does no device
    # work at all. On a shared box that locks out colleagues; worse, it
    # starves this session's own subprocesses, which is how the T106 members
    # of the release-matrix regression came to fail under pytest while passing
    # when run directly. Read at backend init rather than at jax import, so
    # setting it here — before collection imports anything — takes effect.
    # ``setdefault`` leaves an operator's explicit choice alone.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    global _X64_BASELINE
    import jax
    _X64_BASELINE = bool(jax.config.read("jax_enable_x64"))

    for name in _jcm_logger_names():
        logger = logging.getLogger(name)
        _LOGGING_BASELINE[name] = (logger.level, logger.propagate)


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


def _jcm_logger_names():
    """Every logger in the ``jcm`` hierarchy that currently exists."""
    return [name for name in list(logging.Logger.manager.loggerDict)
            if name == "jcm" or name.startswith("jcm.")]


def _restore_logging():
    """Put the ``jcm`` logger levels back to the session baseline.

    A logger with no baseline entry — which is every one of them in a normal
    run, see ``_LOGGING_BASELINE`` — goes back to ``NOTSET`` and propagating,
    i.e. deferring to its ancestors, which is how a freshly imported module's
    logger starts out.

    The root logger is deliberately left alone: pytest owns it (``--log-level``
    and ``caplog`` set and restore it around each test phase), so pinning it
    here would quietly override ``--log-level`` for the whole session.
    """
    for name in _jcm_logger_names():
        logger = logging.getLogger(name)
        level, propagate = _LOGGING_BASELINE.get(name, (logging.NOTSET, True))
        if logger.level != level:
            logger.setLevel(level)  # also clears the manager's level cache
        logger.propagate = propagate


@pytest.fixture(autouse=True)
def _pin_logging_levels():
    """Hold the ``jcm`` logger levels at the session default (#815).

    ``runners.run()`` sets the level on the ``jcm`` logger from
    ``run.log_level`` — the CLI is the application, so that is where the
    knob belongs — and every test that drives a run therefore leaves one
    behind. That breaks any later test asserting a warning fires, because
    ``assertLogs(level=...)`` and ``caplog.at_level(...)`` raise only the
    ROOT logger's level: the record is filtered at its own logger and never
    propagates. Under xdist it depends on which worker drew the run, so it
    surfaces as an unreproducible failure in an unrelated module.

    This is what #815 was, in its original form: ``Model.__init__`` used to
    set the level too, so merely *constructing* a quiet model leaked one.
    That is gone — jcm the library configures no logging — but the runners
    layer still legitimately sets a level, so the isolation is still needed.

    Restored before as well as after the test, so a leak from a test that
    errored out of its own teardown does not travel any further either.
    """
    _restore_logging()
    yield
    _restore_logging()


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
