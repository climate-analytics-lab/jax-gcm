"""The session-wide isolation fixtures in ``conftest.py`` actually isolate."""

import logging
import sys


class TestLoggingLevelIsolation:
    """A ``jcm``-hierarchy level must not outlive the test that set it (#815).

    ``Model(log_level=...)`` sets the level on the ``jcm`` logger, so a test
    that builds a deliberately quiet model silences the whole hierarchy for
    every later test in the same process. Later tests asserting that a
    warning fires then fail, because ``assertLogs(level=...)`` and
    ``caplog.at_level(...)`` raise only the ROOT logger's level — the record
    is filtered at its own logger and never reaches the capture.

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
