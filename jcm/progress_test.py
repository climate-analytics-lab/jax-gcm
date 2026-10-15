"""Tests for the opt-in progress bar (#320)."""

import contextlib
import io
import threading
import unittest
from unittest import mock

from jcm import progress


class BarDisabledTest(unittest.TestCase):
    """``enabled=False`` must be a true no-op: no output, no active bar."""

    def test_disabled_bar_prints_nothing_and_updates_are_noop(self):
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            with progress.bar(10, desc="x", enabled=False) as bar:
                self.assertIsNone(progress._active)
                bar.update(1)
                bar.update()
        self.assertEqual(captured.getvalue(), "")
        self.assertIsNone(progress._active)

    def test_disabled_bar_never_touches_active_even_when_nested(self):
        sentinel = object()
        progress._active = sentinel
        try:
            with progress.bar(5, enabled=False):
                self.assertIs(progress._active, sentinel)
        finally:
            progress._active = None


class BarNestingTest(unittest.TestCase):
    """The caller ticks the yielded bar directly (host-only design, #320).

    There is no module-level tick function any more: three rounds of
    Codex review on an in-jit ``jax.debug.callback`` (ordered effects fail
    under multi-device sharding; unordered delivery has no scheduling
    guarantee; even a data-dependent unordered callback is still a *pure*
    primitive the surrounding scan/lowering may drop or duplicate) found
    the mechanism itself wrong, not any one patch to it — see
    ``jcm/progress.py``'s module docstring. Progress is now ticked only by
    plain Python holding the bar object returned by :func:`bar`.
    """

    def test_bar_object_ticks_directly(self):
        with progress.bar(3, desc="t", enabled=True) as bar:
            self.assertIs(progress._active, bar)
            bar.update(1)
            bar.update(1)
            self.assertEqual(bar.n, 2)
        self.assertIsNone(progress._active)

    def test_nested_bars_restore_the_previous_active_bar_on_exit(self):
        with progress.bar(2, desc="outer", enabled=True) as outer:
            self.assertIs(progress._active, outer)
            with progress.bar(2, desc="inner", enabled=True) as inner:
                self.assertIs(progress._active, inner)
                inner.update(1)
            # Exiting the inner context restores the outer bar as active,
            # not None -- a bar opened while some caller's own bar is
            # already active must resume reporting into the caller's bar
            # rather than going silent (see the module docstring).
            self.assertIs(progress._active, outer)
            outer.update(1)
        self.assertIsNone(progress._active)
        self.assertEqual(outer.n, 1)
        self.assertEqual(inner.n, 1)


class ConcurrencyTest(unittest.TestCase):
    """One active run per process, not per thread (Codex review on #320).

    Same-thread nesting (the chunk bar + its nested per-interval bar) is
    the supported, tested composition above; a SECOND thread trying to
    start its own bar while one is active must be refused rather than
    silently interleaving ticks onto the wrong bar — see jcm/progress.py's
    module docstring for why routing is not supported instead.
    """

    def tearDown(self):
        # Belt: a failed assertion inside the ``with`` below could leave
        # ``_active``/``_active_thread`` set if the context's own cleanup
        # didn't run (it does, via ``finally``, but this keeps a bug in
        # that cleanup from corrupting every later test in this process).
        progress._active = None
        progress._active_thread = None

    def test_second_thread_is_refused_while_a_bar_is_active(self):
        errors = []

        def other_thread():
            try:
                with progress.bar(1, desc="other", enabled=True):
                    pass
            except RuntimeError as exc:
                errors.append(exc)

        with progress.bar(10, desc="main", enabled=True):
            t = threading.Thread(target=other_thread)
            t.start()
            t.join()

        self.assertEqual(len(errors), 1)
        self.assertIn("another thread", str(errors[0]))
        # The refusal didn't touch the main thread's bar/state.
        self.assertIsNone(progress._active)

    def test_same_thread_nesting_is_unaffected_by_the_guard(self):
        # The guard must not turn the ALREADY-supported same-thread nesting
        # into a false positive.
        with progress.bar(2, desc="outer", enabled=True) as outer:
            with progress.bar(2, desc="inner", enabled=True) as inner:
                inner.update(1)
            self.assertIs(progress._active, outer)
        self.assertEqual(inner.n, 1)


class FallbackPrinterTest(unittest.TestCase):
    """``jcm.progress`` must work without ``tqdm`` installed (#320)."""

    def test_bar_falls_back_to_a_plain_printer_without_tqdm(self):
        with mock.patch.object(progress, "_tqdm", None):
            with progress.bar(2, desc="noT", enabled=True) as bar:
                self.assertIsInstance(bar, progress._FallbackPrinter)
                captured = io.StringIO()
                with contextlib.redirect_stderr(captured):
                    bar.update(1)
                self.assertIn("1/2", captured.getvalue())
                self.assertEqual(bar.n, 1)
                bar.update()
                self.assertEqual(bar.n, 2)

    def test_bar_uses_tqdm_when_importable(self):
        if progress._tqdm is None:
            self.skipTest("tqdm is not installed in this environment")
        with progress.bar(2, desc="withT", enabled=True) as bar:
            self.assertIsInstance(bar, progress._tqdm)


class NullBarTest(unittest.TestCase):
    def test_null_bar_update_and_close_accept_any_args(self):
        bar = progress._NullBar()
        bar.update()
        bar.update(5)
        bar.close()


if __name__ == "__main__":
    unittest.main()
