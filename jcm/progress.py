"""Opt-in progress reporting for long integrations (#320).

Two call sites tick a bar:

* :func:`jcm.runners.run_chunked`'s host loop ticks once per chunk — plain
  Python, so it costs nothing when disabled and needs no JAX machinery.
* :meth:`jcm.model.Model.run` / ``resume`` tick once per *saved interval*
  from **inside** the ``jax.lax.scan`` that runs under ``jax.jit`` (see
  ``_op_split_trajectory`` in ``jcm/model.py``). The only way to surface a
  host-visible side effect from there without changing the traced
  computation is ``jax.debug.callback`` — see ``JAX_gotchas.md``.

The callback embedded in the compiled graph is :func:`tick`, a single fixed
module-level function, rather than a closure over a particular bar object.
This is deliberate, not an accident of convenience: ``Model._run_from_state``
is jitted with ``self`` static, and ``Model`` hashes by identity (see
``JAX_gotchas.md``, "Static arguments hold whole objects"), so a cache hit on
an already-compiled signature reuses the FIRST trace's executable byte for
byte, including whatever Python callable got baked into it at trace time. If
that callable closed over a specific ``tqdm`` instance, every later call that
hits the cache — e.g. every chunk of ``run_chunked``, which builds a fresh
bar per chunk — would silently keep ticking the first chunk's (by then
closed) bar instead of the current one. Routing every call through one fixed
function that *dereferences* whatever bar is "active" sidesteps this
entirely: the compiled graph's callback identity never changes, only what it
finds when it fires, so swapping the active bar between calls (even on a
retained compilation) does the right thing.

``enabled=False`` (the default throughout jcm) adds no callback to the
traced graph at all — not a disabled one — so it cannot perturb compilation
or runtime; see the "no compiled graph change when off" test in
``jcm/progress_test.py`` and the chunk wall-time check in
``jcm/runners_test.py``.

**One active run per process, not per thread.** The module-global
:data:`_active` indirection above is exactly what makes a single call's
nested bars (``run_chunked``'s per-chunk bar, with each chunk's
``Model.run``/``resume`` opening its own per-saved-interval bar inside it)
work correctly on one thread. It is also why ``progress=True`` on
*overlapping* ``Model.run``/``resume`` calls in two different threads of the
same process is not supported: the callback ``jax.debug.callback`` embeds in
the compiled graph is the fixed function :func:`tick`, with no per-call
identity, and it cannot be given one — a jit cache hit reuses the first
trace's executable (and the single ``tick`` reference baked into it)
regardless of which thread's call triggers it, so there is no compiled hook
to route a second thread's ticks to a second bar. Building one would mean
threading a run identity through the jit cache key, which is a far bigger
change for a feature nobody has asked for. :func:`bar` instead detects a
second thread entering while a bar from a different thread is active and
raises, rather than silently interleaving ticks onto the wrong bar: see
:data:`_active_thread` and the ``jcm/progress_test.py`` concurrency test.
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

#: The one bar/printer "live" during a traced scan. Set only inside
#: :func:`bar`'s context, restored (not just cleared) on exit so a nested
#: ``run_chunked`` chunk inside some outer progress scope would resume
#: ticking the outer bar rather than going silent.
_active: Any = None

#: Which thread (``threading.get_ident()``) owns :data:`_active`, so
#: :func:`bar` can tell a same-thread NESTING (fine: a chunk bar opening a
#: nested per-interval bar) from a different thread trying to start a
#: SECOND, unrelated run while one is already active (not supported — see
#: the module docstring).
_active_thread: int | None = None

#: Guards the check-then-publish of :data:`_active`/:data:`_active_thread`
#: in :func:`bar` so two threads racing to start the first bar cannot both
#: pass the "nothing active yet" check before either publishes.
_lock = threading.Lock()


def tick(_completed=None) -> None:
    """Advance the active bar by one step.

    The sole function ``jax.debug.callback`` ever embeds in a compiled
    graph (see the module docstring for why that matters); a deliberate
    no-op when no bar is active, so code that calls it unconditionally
    (there is none in jcm today, but a future caller might) is still safe.

    ``_completed`` is accepted and ignored: every call site passes some
    array that is a true data-dependent output of the frame the tick
    reports on (a step counter, say), never read here. Without it, nothing
    in the traced graph forces this callback to run *after* that frame's
    computation -- ``jax.debug.callback(..., ordered=True)`` only orders
    callbacks relative to EACH OTHER, not relative to an unrelated
    computation with no data edge to it, so XLA is free to schedule an
    argument-less callback whenever it likes, including before the
    integration it is meant to report on (Codex review on #320: a
    single-interval run could show 100% before the scan that produces it
    had actually run). Threading the frame's own output through as an
    (unused) argument is what gives the scheduler that edge.
    """
    del _completed
    if _active is not None:
        _active.update(1)


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
    process-wide :data:`_active` target that a traced :func:`tick` callback
    (see the module docstring) advances; on exit the previous active bar
    (``None`` unless this context is nested inside another) is restored and
    this one is closed.

    ``enabled=False`` never touches :data:`_active` and yields a
    :class:`_NullBar`, so a disabled call is side-effect-free both on the
    host and (via the caller never attaching :func:`tick` to its jit call in
    the first place) in any enclosing traced computation.

    Raises:
        RuntimeError: if a bar from a DIFFERENT thread is already active.
            Nesting on the SAME thread (a chunk bar opening a nested
            per-saved-interval bar) is the supported, tested composition and
            proceeds normally — only a second, unrelated thread trying to
            start its own run concurrently is refused. See the module
            docstring for why routing is not supported instead: pass
            ``progress=False`` for every concurrent run but one.

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
