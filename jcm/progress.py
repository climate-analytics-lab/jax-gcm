"""Opt-in, HOST-ONLY progress reporting for long integrations (#320).

Progress is reported only from plain Python running on the host, never
from a callback embedded inside a jitted/traced computation:

* :func:`jcm.runners.run_chunked`'s host loop ticks once per chunk, right
  after each chunk's integration returns.
* :meth:`jcm.model.Model.run` / ``resume`` — one call, whether or not it is
  reached through ``run_chunked`` — shows as a single unit, ticked from 0
  to 1 once the call returns. There is no reporting at a finer grain
  *within* one such call (no per-saved-interval tick): see "Why no
  in-jit reporting" below. A chunked run's real granularity therefore
  comes entirely from ``run_chunked`` ticking once per chunk, which is as
  fine as any one ``Model.run``/``resume`` call can honestly report
  anyway.

Why no in-jit reporting
------------------------
An earlier version of this feature reported once per saved interval from
**inside** the ``jax.lax.scan`` that runs under ``jax.jit``
(``_op_split_trajectory`` in ``jcm/model.py``), via ``jax.debug.callback``.
Three rounds of Codex review each found this broken a different way:

1. ``jax.debug.callback(..., ordered=True)`` raises ``NotImplementedError``
   ("ordered effects are not supported for more than 1 device") under a
   multi-device ``NamedSharding`` — exactly the configuration a progress
   bar matters most for.
2. Dropping ``ordered=True`` makes the callback unordered, so nothing in
   the traced graph forces it to run after the frame it is meant to
   report on; threading the frame's own output through as a data
   dependency fixed that scheduling problem specifically.
3. But ``jax.debug.callback`` is still, in JAX's own terms, a *pure*
   primitive from the compiler's point of view: the surrounding scan or a
   multi-device lowering remains free to drop or duplicate it as a
   dead-code-elimination / rematerialization candidate, so the bar could
   end short even on a single device, with nothing to detect it.

Three failures of the same mechanism — a callback living inside the
traced graph at all — is the signal that the mechanism was wrong, not
any one patch to it (see ``CLAUDE.md``: question the feature once a
review keeps hitting the same one). Reporting only from the host, around
whole calls that already return to plain Python, cannot be dropped,
duplicated or reordered, because nothing inside a trace is involved at
all.

``enabled=False`` (the default throughout jcm) never even calls
:func:`tqdm`/constructs a bar — it is a true no-op, not merely a silent
one; see ``jcm/progress_test.py`` and the chunk wall-time check in
``jcm/runners_test.py``.

**One active bar per process, not per thread.** :data:`_active` is
ordinary (non-JAX) process-global mutable state, so two threads each
calling ``Model.run(progress=True)`` concurrently would otherwise race on
it — whichever thread's bar got overwritten last would receive the
other's updates. :func:`bar` instead detects a second thread entering
while a bar from a DIFFERENT thread is active and raises, rather than
silently interleaving reports onto the wrong bar: see
:data:`_active_thread`, :data:`_lock` and the
``jcm/progress_test.py`` concurrency test. Nesting on the SAME thread (one
bar opened while another of that thread's own bars is already active) is
fine and intentional — see :data:`_active`'s docstring.
Concurrent runs in one process must pass ``progress=False`` for all but one.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised via a monkeypatched _tqdm
    _tqdm = None

#: The one bar/printer "live" on the host right now. Set only inside
#: :func:`bar`'s context, restored (not just cleared) on exit so a bar
#: opened while another of the SAME thread's own bars is already active
#: (nesting) resumes ticking the outer one when the inner exits, rather
#: than going silent. Plain Python state — nothing here is read or written
#: from inside a jitted/traced computation; see the module docstring for
#: why that is now a hard rule, not an optimisation.
_active: Any = None

#: Which thread (``threading.get_ident()``) owns :data:`_active`, so
#: :func:`bar` can tell a same-thread NESTING (fine) from a different
#: thread trying to start a SECOND, unrelated bar while one is already
#: active (not supported — see the module docstring).
_active_thread: int | None = None

#: Guards the check-then-publish of :data:`_active`/:data:`_active_thread`
#: in :func:`bar` so two threads racing to start the first bar cannot both
#: pass the "nothing active yet" check before either publishes.
_lock = threading.Lock()


class _NullBar:
    """What :func:`bar` yields when ``enabled=False``: every call a no-op."""

    def update(self, n: int = 1) -> None:
        del n

    def close(self) -> None:
        pass


class _FallbackPrinter:
    """Minimal progress printer used when ``tqdm`` is not installed.

    Prints one line per update to stderr rather than redrawing a single
    line the way ``tqdm`` does — good enough for the opt-in, no-hard-dep
    fallback (#320); not meant to visually match ``tqdm``.
    """

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n
        print(f"{self.desc}: {self.n}/{self.total}", file=sys.stderr, flush=True)

    def close(self) -> None:
        pass


@contextlib.contextmanager
def bar(total: int, desc: str = "", enabled: bool = True):
    """Yield a progress bar, or a silent no-op when ``enabled`` is False.

    Uses ``tqdm.auto`` when importable, otherwise :class:`_FallbackPrinter`
    — ``tqdm`` (``pip install tqdm``) is a convenience, never a hard
    dependency. While the context is open, the yielded bar is also the
    process-wide :data:`_active` bar, which the caller ticks directly
    (``bar_obj.update(1)``) — see the module docstring for why nothing
    ticks it from inside a jitted/traced computation; on exit the previous
    active bar (``None`` unless this context is nested inside another) is
    restored and this one is closed.

    ``enabled=False`` never touches :data:`_active` and yields a
    :class:`_NullBar`, so a disabled call is entirely side-effect-free.

    Raises:
        RuntimeError: if a bar from a DIFFERENT thread is already active.
            Nesting on the SAME thread is the supported, tested
            composition and proceeds normally — only a second, unrelated
            thread trying to start its own bar concurrently is refused.
            See the module docstring for why routing is not supported
            instead: pass ``progress=False`` for every concurrent run but
            one.

    """
    global _active, _active_thread
    if not enabled:
        yield _NullBar()
        return
    this_thread = threading.get_ident()
    # The check-then-publish below must be one atomic step: two threads
    # racing to start the FIRST bar could otherwise both observe
    # ``_active is None`` before either publishes, especially since
    # constructing a ``tqdm`` instance can release the GIL (Codex review
    # on #320). ``_lock`` is held only across the check and the publish,
    # never across the ``yield`` -- the whole point is reporting on a long
    # integration without serializing it.
    with _lock:
        if _active is not None and _active_thread != this_thread:
            raise RuntimeError(
                "jcm.progress: a progress bar from another thread "
                f"(id={_active_thread}) is already active; this thread "
                f"(id={this_thread}) cannot start a second one. The "
                "progress bar supports one active model run per process "
                "(see jcm/progress.py's module docstring for why) — pass "
                "progress=False on every concurrent Model.run/resume call "
                "but one."
            )
        cls = _tqdm if _tqdm is not None else _FallbackPrinter
        progress_bar = cls(total=total, desc=desc)
        previous, previous_thread = _active, _active_thread
        _active, _active_thread = progress_bar, this_thread
    try:
        yield progress_bar
    finally:
        with _lock:
            _active, _active_thread = previous, previous_thread
        progress_bar.close()
