"""Opt-in progress reporting for long integrations (#320).

Two call sites report progress:

* :func:`jcm.runners.run_chunked`'s host loop ticks once per chunk — plain
  Python (``chunk_bar.update(1)`` directly on the bar object), so it costs
  nothing when disabled and needs no JAX machinery.
* :meth:`jcm.model.Model.run` / ``resume`` report once per *saved interval*
  from **inside** the ``jax.lax.scan`` that runs under ``jax.jit`` (see
  ``_op_split_trajectory`` in ``jcm/model.py``). The only way to surface a
  host-visible side effect from there without changing the traced
  computation is ``jax.debug.callback`` — see ``JAX_gotchas.md``.

**The in-scan callback is UNORDERED and reports an absolute position, not a
delta.** An ``ordered=True`` callback (the first version of this feature)
raises ``NotImplementedError("ordered effects are not supported for more
than 1 device")`` the moment the model runs under a multi-device
``NamedSharding`` — exactly the configuration a progress bar matters most
for (Codex review on #320). Dropping ``ordered=True`` means JAX no longer
guarantees the callback fires in scan-iteration order (an unordered
callback from one shard's device program can in principle be scheduled
before or after another's), so each call instead carries, as a genuine
data dependency the callback discards, this frame's own absolute interval
index (``(step_next - initial_step) // inner_steps`` in ``model.py`` —
monotonically increasing by construction, never by how many callbacks have
fired so far) and :func:`set_position` sets the bar's count to
``max(current, idx)`` rather than incrementing it. A callback that arrives
out of order, or is ever delivered more than once for the same frame (not
something unordered dispatch is documented to do, but nothing this module
relies on rules out), therefore cannot move the bar backwards or double count
— the worst a reordering can do is a visually out-of-order tick, never a
wrong final count. See ``jcm/model_test.py``'s sharding test.

The callback embedded in the compiled graph is :func:`set_position`, a
single fixed module-level function, rather than a closure over a particular
bar object. This is deliberate, not an accident of convenience:
``Model._run_from_state`` is jitted with ``self`` static, and ``Model``
hashes by identity (see ``JAX_gotchas.md``, "Static arguments hold whole
objects"), so a cache hit on an already-compiled signature reuses the FIRST
trace's executable byte for byte, including whatever Python callable got
baked into it at trace time. If that callable closed over a specific
``tqdm`` instance, every later call that hits the cache — e.g. every chunk
of ``run_chunked``, which builds a fresh bar per chunk — would silently
keep reporting into the first chunk's (by then closed) bar instead of the
current one. Routing every call through one fixed function that
*dereferences* whatever bar is "active" sidesteps this entirely: the
compiled graph's callback identity never changes, only what it finds when
it fires, so swapping the active bar between calls (even on a retained
compilation) does the right thing.

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
the compiled graph is the fixed function :func:`set_position`, with no
per-call identity, and it cannot be given one — a jit cache hit reuses the
first trace's executable (and the single ``set_position`` reference baked
into it) regardless of which thread's call triggers it, so there is no
compiled hook to route a second thread's reports to a second bar. Building
one would mean threading a run identity through the jit cache key, which is
a far bigger change for a feature nobody has asked for. :func:`bar` instead
detects a second thread entering while a bar from a different thread is
active and raises, rather than silently interleaving reports onto the wrong
bar: see :data:`_active_thread` and the ``jcm/progress_test.py`` concurrency
test.
Concurrent runs in one process must pass ``progress=False`` for all but one.

**Also unsupported, and rejected the same way (loudly, not silently):**
``progress=True`` where ``Model.run``/``resume`` itself runs inside an
ENCLOSING ``jax.jit``/``grad``/``vmap``. That outer transformation traces
this call's whole body -- including opening and closing the bar -- exactly
once, before the actual device execution it wraps ever happens; by the
time the compiled outer function is later invoked and ``set_position``
actually fires, the bar it would report to has already closed on a Python
stack that no longer exists. ``jcm/model.py`` detects this (the same way
``Model.resume`` detects a traced carry: checking whether
``_run_from_state``'s own OUTPUT is a tracer, since under an outer
transformation it always is, whatever its inputs were) and raises
``ValueError`` rather than silently reporting nothing.
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


def set_position(idx) -> None:
    """Advance the active bar's position to ``max(current position, idx)``.

    The sole function ``jax.debug.callback`` ever embeds in a compiled
    graph (see the module docstring for why that matters, including why it
    is UNORDERED and why ``idx`` is an absolute position rather than a
    delta); a deliberate no-op when no bar is active.

    ``idx`` is a true data-dependent output of the frame being reported on
    (the scan's own absolute saved-interval index, computed in
    ``model.py`` as ``(step_next - initial_step) // inner_steps``), not a
    counter this function increments. Without a real argument, nothing in
    the traced graph forces this callback to run *after* the frame's
    computation -- an argument-less callback has no data edge to the
    computation it is meant to report on, so XLA (or, under an unordered
    callback with multiple devices, the dispatch order across devices)
    would be free to run it whenever convenient, including before that
    frame's integration had actually happened. Setting the bar's count to
    ``max(current, idx)`` rather than adding to it is what makes an
    out-of-order or duplicate delivery harmless: the bar can only move
    forward, and to the position its own frame actually reached, never past
    it and never below where a later frame already put it.
    """
    if _active is None:
        return
    idx = int(idx)
    current = int(_active.n)
    if idx > current:
        _active.update(idx - current)


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
    process-wide :data:`_active` target that a traced :func:`set_position` callback
    (see the module docstring) advances; on exit the previous active bar
    (``None`` unless this context is nested inside another) is restored and
    this one is closed.

    ``enabled=False`` never touches :data:`_active` and yields a
    :class:`_NullBar`, so a disabled call is side-effect-free both on the
    host and (via the caller never attaching :func:`set_position` to its jit call in
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
