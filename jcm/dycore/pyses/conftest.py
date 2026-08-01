"""Pytest hooks for the pyses-backend test package.

Why the reordering hook exists
------------------------------
Building the pyses jax backend (the first ``FVPhysicsGrid`` /
``PysesCamSEDycore`` construction in a process) calls
``jax.config.update("jax_enable_x64", True)`` **process-wide** — the CAM-SE
explicit core requires float64. That flag flip cannot be undone safely (the
backend and its jitted functions are built against it), so any
dtype-sensitive test that runs *after* a pyses test in the same process (or
the same pytest-xdist worker) would suddenly see float64 defaults — e.g. the
``utils_test`` tangent-dtype tests and the SPEEDY shortwave solar test fail
exactly this way.

Placing every test from this directory at the *end* of the collection means
that by the time any worker executes a pyses test, all remaining tests in
the queue are also pyses tests, so the x64 flip can no longer contaminate a
foreign test. This holds for single-process runs trivially and for xdist's
default ``--dist load`` scheduling because tests are dispatched in
collection order.
"""

import os

# pyses freezes its array backend on first import; select jax/CPU before any
# collection-time import can touch it. (Unit tests never need the GPU.)
os.environ.setdefault("PYSES_BACKEND", "jax")
os.environ.setdefault("PYSES_USE_CPU", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_collection_modifyitems(session, config, items):
    """Move pyses-backend tests to the end of the run (see module docstring)."""
    pyses_items = [it for it in items if str(it.path).startswith(_HERE)]
    if not pyses_items:
        return
    others = [it for it in items if not str(it.path).startswith(_HERE)]
    items[:] = others + pyses_items
