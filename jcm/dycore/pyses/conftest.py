"""Pytest setup for the pyses-backend test package.

pyses freezes its array backend on first import, so the jax/CPU backend is
selected here, before any collection-time import can touch it (unit tests
never need the GPU).

The float64 the CAM-SE backend needs is handled by the root ``conftest.py``,
for every test marked ``requires_extra("pyses")`` wherever it runs: the flag
is turned on before such a test's class fixtures are built and restored to the
session default when the run of pySES tests ends, and every other test is
pinned to that default. Test order therefore does not matter, which it must
not: xdist's ``--dist loadscope`` dispatches the largest scopes first.
"""

import os

os.environ.setdefault("PYSES_BACKEND", "jax")
os.environ.setdefault("PYSES_USE_CPU", "1")
