"""The session-wide isolation fixtures in ``conftest.py`` actually isolate."""

import logging
import sys


class TestLoggingLevelIsolation:
    """A ``jcm``-hierarchy level must not outlive the test that set it (#815).

    ``runners.run()`` sets the level on the ``jcm`` logger from
    ``run.log_level``, so every test that drives a run leaves one behind.
    Later tests asserting that a warning fires then fail, because
    ``assertLogs(level=...)`` and ``caplog.at_level(...)`` raise only the
    ROOT logger's level — the record is filtered at its own logger and never
    reaches the capture.

    Neither test here leaks a level of its own. A canary that deliberately
    left one behind would, if ``_pin_logging_levels`` ever regressed, seed
    exactly the cross-module failure this fixture exists to prevent — and it
    could not detect the regression anyway, since under ``--dist load`` the
    test observing the leak may run on a different worker.
    """

    def test_restoring_clears_a_leaked_level(self):
        """The restore itself: silenced loggers go back to the baseline."""
        conftest = sys.modules["conftest"]
        leaked = ("jcm", "jcm.made_up_for_this_test")
        for name in leaked:
            logging.getLogger(name).setLevel(logging.CRITICAL)

        conftest._restore_logging()

        for name in leaked:
            # Compared against the baseline rather than a hardcoded NOTSET,
            # so this keeps testing the restore even if the package one day
            # sets a level at import (which #817 weighs) instead of failing
            # and pointing at the wrong thing.
            expected, _ = conftest._LOGGING_BASELINE.get(
                name, (logging.NOTSET, True))
            assert logging.getLogger(name).level == expected

    def test_the_restore_runs_around_every_test(self, request):
        """...and is wired in as autouse, not merely defined.

        Checked through ``request``, so renaming, un-``autouse``-ing or
        shadowing the fixture fails here rather than silently going quiet.
        """
        assert "_pin_logging_levels" in request.fixturenames


class TestGpuPreallocationDisabled:
    """The session must never preallocate the GPU, even if told to."""

    def test_hook_overrides_an_explicit_true(self, monkeypatch):
        # An exported ``true`` must not survive: the parent pytest process
        # would then hold 75 % of the card, which the release-matrix workers
        # cannot reclaim with their own setting.
        import os

        conftest = sys.modules["conftest"]
        monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        conftest._disable_gpu_preallocation()
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    def test_session_runs_with_preallocation_off(self):
        import os

        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"


class TestFreedHeapIsReturned:
    """Class/module boundaries hand freed heap back to the OS.

    A worker whose RSS only ratchets up reaches the CI runner's ceiling and
    is SIGTERM'd mid-suite (exit 143), which never reads as a test failure.
    """

    class _Item:
        def __init__(self, cls):
            self._cls = cls

        def getparent(self, kind):
            import pytest
            return self._cls if kind is pytest.Class else None

        def iter_markers(self, name):
            return iter(())

    class _Parent:
        def __init__(self, nodeid):
            self.nodeid = nodeid

    def _teardown(self, monkeypatch, same_group):
        conftest = sys.modules["conftest"]
        calls = []
        monkeypatch.setattr(conftest, "_malloc_trim", calls.append)
        # A budget the boundary cannot reach, so only the trim can fire, and
        # no resident ceiling, so this worker's own size cannot make one.
        monkeypatch.setattr(conftest, "_MAX_GROWTH_BYTES", 2**62)
        monkeypatch.setattr(conftest, "_MAX_RSS_BYTES", 0)
        monkeypatch.setattr(conftest, "_rss_at_last_clear", 0)
        a = self._Item(self._Parent("m.py::A"))
        b = self._Item(self._Parent("m.py::A" if same_group else "m.py::B"))
        conftest.pytest_runtest_teardown(a, b)
        return calls

    def test_trimmed_at_a_class_boundary(self, monkeypatch):
        assert self._teardown(monkeypatch, same_group=False) == [0]

    def test_not_trimmed_within_a_class(self, monkeypatch):
        assert self._teardown(monkeypatch, same_group=True) == []

    def test_release_is_safe_without_glibc(self, monkeypatch):
        conftest = sys.modules["conftest"]
        monkeypatch.setattr(conftest, "_malloc_trim", None)
        conftest._release_freed_heap()


class TestResidentCeilingActsInsideAGroup:
    """Past ``JCM_TEST_MAX_RSS_MB`` a test boundary inside a class counts.

    A module of parametrized functions has no class boundary, so without the
    ceiling it keeps every executable it compiles until its last test, and a
    single-process CI shard reaches the runner's memory that way (#1214).
    """

    _GIB = 2**30

    def _teardown(self, monkeypatch, sizes, last_clear, ceiling_gib=8):
        """Run the hook between two tests of one class; return what it did.

        ``sizes`` are the resident sizes (GiB) the hook reads, in order; the
        last one repeats.
        """
        import jax

        conftest = sys.modules["conftest"]
        done = []
        readings = list(sizes)

        def rss():
            size = readings.pop(0) if len(readings) > 1 else readings[0]
            return int(size * self._GIB)

        monkeypatch.setattr(conftest, "_rss_bytes", rss)
        monkeypatch.setattr(conftest, "_memory_map_count", lambda: 0)
        monkeypatch.setattr(conftest, "_malloc_trim",
                            lambda _: done.append("trim"))
        monkeypatch.setattr(jax, "clear_caches", lambda: done.append("clear"))
        monkeypatch.setattr(conftest, "_MAX_GROWTH_BYTES", self._GIB)
        monkeypatch.setattr(conftest, "_MAX_RSS_BYTES",
                            ceiling_gib * self._GIB)
        monkeypatch.setattr(
            conftest, "_rss_at_last_clear",
            None if last_clear is None else int(last_clear * self._GIB))
        item = TestFreedHeapIsReturned._Item(
            TestFreedHeapIsReturned._Parent("m.py::A"))
        conftest.pytest_runtest_teardown(item, item)
        return done, conftest._rss_at_last_clear

    def test_nothing_happens_under_the_ceiling(self, monkeypatch):
        done, _ = self._teardown(monkeypatch, [7.0], last_clear=1.0)
        assert done == []

    def test_caches_are_dropped_past_the_ceiling(self, monkeypatch):
        # 10 GiB before and after the trim, 2 GiB once the caches are gone.
        done, floor = self._teardown(
            monkeypatch, [10.0, 10.0, 2.0], last_clear=1.0)
        assert done == ["trim", "clear", "trim"]
        assert floor == 2 * self._GIB

    def test_the_first_group_of_a_session_is_covered(self, monkeypatch):
        # No group boundary has set a floor yet: the size itself is the reason.
        done, floor = self._teardown(
            monkeypatch, [10.0, 10.0, 2.0], last_clear=None)
        assert done == ["trim", "clear", "trim"]
        assert floor == 2 * self._GIB

    def test_returning_the_heap_can_be_enough(self, monkeypatch):
        done, _ = self._teardown(monkeypatch, [10.0, 6.0], last_clear=1.0)
        assert done == ["trim"]

    def test_live_data_is_not_cleared_after_every_test(self, monkeypatch):
        # Past the ceiling, but grown by less than the budget since the last
        # drop: what is resident is not executables, so dropping them again
        # would only cost the recompilation.
        done, floor = self._teardown(
            monkeypatch, [10.0, 10.0], last_clear=9.5)
        assert done == ["trim"]
        assert floor == int(9.5 * self._GIB)

    def test_zero_disables_the_ceiling(self, monkeypatch):
        done, _ = self._teardown(
            monkeypatch, [10.0], last_clear=1.0, ceiling_gib=0)
        assert done == []


class TestPysesTestsHoldX64:
    """pySES-backend tests run in float64 for their class, and only they do.

    xdist's ``--dist loadscope`` dispatches the largest scopes first, so a
    pySES class can be built after an unrelated test restored the session
    default, and an unrelated class can be built straight after a pySES
    test. Both hooks are exercised directly, whatever order this run has.
    """

    class _Item:
        nodeid = "m.py::T::test"

        def __init__(self, extras=()):
            import pytest
            self._marks = ([pytest.mark.requires_extra(*extras).mark]
                           if extras else [])

        def iter_markers(self, name):
            return iter([m for m in self._marks if m.name == name])

        def getparent(self, kind):
            return None

    def test_setup_turns_x64_on_for_a_pyses_test_only(self, monkeypatch):
        import jax

        conftest = sys.modules["conftest"]
        monkeypatch.setattr(conftest, "_X64_BASELINE", False)
        try:
            jax.config.update("jax_enable_x64", False)
            conftest.pytest_runtest_setup(self._Item(("mam4",)))
            assert not jax.config.read("jax_enable_x64")
            conftest.pytest_runtest_setup(self._Item(("pyses",)))
            assert jax.config.read("jax_enable_x64")
            conftest.pytest_runtest_setup(self._Item(("pyses", "mam4")))
            assert jax.config.read("jax_enable_x64")
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_teardown_restores_the_default_leaving_a_pyses_run(
            self, monkeypatch):
        import jax

        conftest = sys.modules["conftest"]
        monkeypatch.setattr(conftest, "_X64_BASELINE", False)
        monkeypatch.setattr(conftest, "_malloc_trim", None)
        monkeypatch.setattr(conftest, "_MAX_GROWTH_BYTES", 2**62)
        monkeypatch.setattr(conftest, "_rss_at_last_clear", 0)
        pyses, other = self._Item(("pyses",)), self._Item()
        try:
            jax.config.update("jax_enable_x64", True)
            conftest.pytest_runtest_teardown(pyses, self._Item(("pyses",)))
            assert jax.config.read("jax_enable_x64")     # still in the run
            conftest.pytest_runtest_teardown(pyses, other)
            assert not jax.config.read("jax_enable_x64")
            jax.config.update("jax_enable_x64", True)
            conftest.pytest_runtest_teardown(pyses, None)  # end of queue
            assert not jax.config.read("jax_enable_x64")
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_pin_exempts_only_pyses_tests(self, request):
        conftest = sys.modules["conftest"]
        assert "_pin_jax_x64" in request.fixturenames
        assert conftest._builds_pyses_backend(self._Item(("pyses",)))
        assert not conftest._builds_pyses_backend(self._Item(("cosp",)))
        assert not conftest._builds_pyses_backend(self._Item())


class TestRunsDoNotEnableTheSharedCompilationCache:
    """``runners.run()`` in a test leaves JAX's persistent cache alone (#880)."""

    def test_maybe_enable_is_a_no_op_under_the_fixture(self):
        import jax

        from jcm.runners import maybe_enable_compilation_cache

        before = jax.config.jax_compilation_cache_dir
        try:
            maybe_enable_compilation_cache()
            assert jax.config.jax_compilation_cache_dir == before
        finally:
            # A regressed fixture must not leave the cache pointed at the
            # shared directory for the rest of this worker.
            jax.config.update("jax_compilation_cache_dir", before)
