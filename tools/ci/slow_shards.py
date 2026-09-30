"""The two shards the CI slow suite runs in, and the check that they partition it.

The slow suite (``pytest -m slow``) is split across two parallel CI jobs in
``.github/workflows/run_test.yaml`` because it no longer fits one job's timeout
on the hosted runner: the RRTMGP-based ECHAM gradient harnesses alone take
~60 min there. The split is by path and is defined only here:

``radiation``
    The ECHAM gradient harnesses and the radiation package — the RRTMGP-heavy
    tests (``RADIATION_PATHS``).
``rest``
    Every other slow test, selected as the complement (``--ignore`` of each
    ``RADIATION_PATHS`` entry), so a new slow test file lands here by
    construction and cannot fall out of CI.

``check`` collects ``-m slow`` three times (whole, each shard) and fails unless
the two shards are disjoint, non-empty and together equal the whole suite, and
every listed path exists. It runs in the ``radiation`` job before its tests.

Usage::

    pytest -m slow $(python tools/ci/slow_shards.py args radiation)
    pytest -m slow $(python tools/ci/slow_shards.py args rest)
    python tools/ci/slow_shards.py check
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: Paths of the ``radiation`` shard, relative to the repository root. Chosen so
#: the two shards take a similar time in CI (~68 and ~62 min of tests,
#: measured on PR #970).
RADIATION_PATHS = (
    "jcm/physics/echam/term_gradients_test.py",
    "jcm/physics/echam/gradient_finiteness_test.py",
    "jcm/physics/radiation",
)

SHARDS = ("radiation", "rest")


def pytest_args(shard: str) -> list[str]:
    """Return the pytest selection arguments for ``shard``."""
    if shard == "radiation":
        return list(RADIATION_PATHS)
    if shard == "rest":
        return [f"--ignore={path}" for path in RADIATION_PATHS]
    raise ValueError(f"unknown shard {shard!r}; choose one of {SHARDS}")


def _collect(args: list[str]) -> set[str]:
    """Node ids ``pytest --collect-only -m slow`` selects with ``args``."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-m", "slow", *args],
        capture_output=True, text=True, check=False)
    # Exit code 5 is "no tests collected", which the checks below report.
    if result.returncode not in (0, 5):
        raise RuntimeError(
            f"collection failed for {args}:\n{result.stdout}\n{result.stderr}")
    return {line.strip() for line in result.stdout.splitlines()
            if "::" in line}


def partition_errors(whole: set[str], shards: dict[str, set[str]]) -> list[str]:
    """Why ``shards`` do not partition ``whole`` (empty when they do)."""
    errors = []
    for name, ids in shards.items():
        if not ids:
            errors.append(f"shard {name!r} selects no slow tests")
    names = list(shards)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = shards[a] & shards[b]
            if both:
                errors.append(f"{len(both)} tests run in both {a!r} and {b!r}, "
                              f"e.g. {sorted(both)[0]}")
    union = set().union(*shards.values())
    missing = whole - union
    if missing:
        errors.append(f"{len(missing)} slow tests run in no shard, "
                      f"e.g. {sorted(missing)[0]}")
    extra = union - whole
    if extra:
        errors.append(f"{len(extra)} shard tests are not in the slow suite, "
                      f"e.g. {sorted(extra)[0]}")
    return errors


def check(root: Path = Path(".")) -> int:
    """Collect the whole suite and both shards; print and return the verdict."""
    errors = [f"{path} does not exist" for path in RADIATION_PATHS
              if not (root / path).exists()]
    whole = _collect([])
    shards = {name: _collect(pytest_args(name)) for name in SHARDS}
    errors += partition_errors(whole, shards)
    for error in errors:
        print(f"::error::slow-suite shards: {error}")
    sizes = ", ".join(f"{name} {len(ids)}" for name, ids in shards.items())
    print(f"slow suite {len(whole)} tests; shards: {sizes}")
    return 1 if errors else 0


def main(argv: list[str]) -> int:
    """CLI: ``args <shard>`` prints the selection, ``check`` verifies it."""
    if len(argv) == 2 and argv[0] == "args":
        print(" ".join(pytest_args(argv[1])))
        return 0
    if argv == ["check"]:
        return check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
