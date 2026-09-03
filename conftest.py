"""Session-wide pytest hooks.

Slow-suite memory ceiling (issue #745)
--------------------------------------
The CI slow job runs the WHOLE suite in a single process
(``pytest -v -s -m slow ...`` — no xdist), so JAX's compilation cache and the
device arrays it retains accumulate ~unboundedly from module to module. On the
hosted runner's hard memory ceiling that reproducibly OOM-kills the
memory-heaviest test (the RRTMGP RCE integration,
``rce_test.py::TestRceIntegrationRrtmgp``) after ~70 slow tests have run — the
process is killed (exit 137/143), not a test assertion. Clearing JAX's caches
and forcing a GC at each module's teardown releases the retained compiled
executables between modules, capping peak RSS at roughly one module's working
set instead of the whole suite's cumulative footprint.

Scoped to modules that actually contain a slow-marked test: the fast (xdist)
suite deselects slow tests, so this never fires there — each short-lived xdist
worker frees its own memory on exit anyway, and a per-module cache clear would
only force needless recompilation.
"""

import gc

import pytest


@pytest.fixture(scope="module", autouse=True)
def _bound_slow_suite_memory(request):
    """Clear JAX caches + GC at each slow module's teardown (issue #745)."""
    yield
    module = request.module
    if not any(
        item.get_closest_marker("slow")
        for item in request.session.items
        if getattr(item, "module", None) is module
    ):
        return
    import jax
    # Release the compiled executables + retained arrays this module's slow
    # tests accumulated before the next module adds its own, so a single-process
    # slow run's peak RSS does not grow with the whole suite (#745).
    jax.clear_caches()
    gc.collect()
