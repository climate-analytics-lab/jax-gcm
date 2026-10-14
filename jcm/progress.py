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
"""

from __future__ import annotations

import contextlib
import sys
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


def tick() -> None:
    """Advance the active bar by one step.

    The sole function ``jax.debug.callback`` ever embeds in a compiled
    graph (see the module docstring for why that matters); a deliberate
    no-op when no bar is active, so code that calls it unconditionally
    (there is none in jcm today, but a future caller might) is still safe.
    """
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
    """
    global _active
    if not enabled:
        yield _NullBar()
        return
    cls = _tqdm if _tqdm is not None else _FallbackPrinter
    progress_bar = cls(total=total, desc=desc)
    previous, _active = _active, progress_bar
    try:
        yield progress_bar
    finally:
        _active = previous
        progress_bar.close()
