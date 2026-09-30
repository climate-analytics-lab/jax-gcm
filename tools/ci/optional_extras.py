"""The optional extras the test suite gates on, and the checks that keep that gating visible.

The default CI jobs install ``pip install -e .`` with no extras, so coverage is
measured at dependency parity (see ``.github/actions/setup-jcm``). Every test
that needs an extra therefore skips there, and a separate ``extras-tests`` job
in ``.github/workflows/run_test.yaml`` installs them all and runs exactly those
tests. This module is what makes that job trustworthy, so it cannot rot
silently in either direction:

**One gate.** A test declares the extras it needs with
``@pytest.mark.requires_extra("pyses")`` (or a module-level ``pytestmark``).
The marker is the only sanctioned gate: :func:`pytest_runtest_setup` skips the
test when an extra is absent, and the extras job selects its tests with
``-m requires_extra``, so a marked test is picked up by construction.

**Missing is a failure where it must not be missing.** With
``JCM_REQUIRE_EXTRAS=1`` (set by the extras job) the session refuses to start
unless every extra in :data:`EXTRAS` is importable, any marked test (or
subtest of one) that skips, for whatever reason, is reported as failed, and
so is a module skipped at collection: the job exists to run those tests, so a
skip there is a test that silently did not run. An ``xfail`` stays an
``xfail``; it is a declared expectation, visible in the report.

**An unmarked gate cannot hide.** Two checks catch a test that gates on an
extra some other way (a bare ``importorskip``, a ``try: import`` with a
``skipUnless``, a broad ``except`` that skips), because such a test would be
skipped by the default jobs *and* never selected by the extras job:

* at run time, in every job, :func:`pytest_runtest_makereport` and
  :func:`pytest_make_collect_report` turn into a failure any skip of an
  unmarked test (or a whole module) whose reason names an extra's package
  (:data:`PACKAGE_NAMES`, in any case). A subtest of an unmarked test may
  still skip for an extra: the test runs, and the per-case skip is reported;
* statically, :func:`gate_violations` scans every test file for
  ``importorskip`` / ``find_spec`` of an extra's module and for an import of
  one under a guard that swallows ``ImportError`` — including the forms that
  would never report a skip at all. ``optional_extras_test.py`` runs it over
  the repository in the fast suite, and the extras job runs ``check``.

What neither check can see is a gate whose module the scan cannot resolve
statically (a computed name, or an import of some other module that imports
the extra) *and* that skips with a reason not naming the package. Write the
marker instead.

Failing every collection-time skip under ``JCM_REQUIRE_EXTRAS=1`` is
deliberate and applies to modules the selection would deselect too: a module
skipped at collection cannot be inspected for marked tests, so the job cannot
tell a hidden gated test from an unrelated one. A module that must skip for
another reason (no GPU, say) should skip per test, where the marker is visible.

:data:`EXTRAS` must name every extra ``pyproject.toml`` declares (``check``
enforces it), and the extras job installs exactly ``pip-extras``, so a new
extra is installed and required there as soon as it is declared.

Usage::

    python tools/ci/optional_extras.py pip-extras   # "cosp,era5,mam4,pyses"
    python tools/ci/optional_extras.py check        # registry + static scan
    JCM_REQUIRE_EXTRAS=1 pytest -m requires_extra   # the extras job's run
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

#: Each optional extra in ``pyproject.toml`` -> the module whose presence
#: means it is installed.
EXTRAS = {
    "cosp": "jcosp",
    "era5": "gcsfs",
    "mam4": "mam4_jax",
    "pyses": "pyses",
}

#: Every name by which a skip reason can refer to an extra's package: its
#: import name and its distribution name(s). A skip reason that contains one
#: of these (as a whole word) is taken to be a skip for that extra.
PACKAGE_NAMES = {
    "cosp": ("jcosp", "jax-cosp", "jax_cosp"),
    "era5": ("gcsfs",),
    "mam4": ("mam4_jax", "mam4-jax"),
    "pyses": ("pyses",),
}

MARKER = "requires_extra"
REQUIRE_ENV = "JCM_REQUIRE_EXTRAS"

# A name counts wherever it stands as a word, in any case ("pySES",
# "MAM4-JAX"), except as a later component of a dotted or slashed path: in
# "jcm/dycore/pyses/x.nc" or "jcm.dycore.pyses.dycore" it names jcm's own
# package, not the extra.
_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])("
    + "|".join(sorted({re.escape(n) for names in PACKAGE_NAMES.values()
                       for n in names}, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])", re.IGNORECASE)

ROOT = Path(__file__).resolve().parents[2]


def required() -> bool:
    """Whether this session must have every extra (``JCM_REQUIRE_EXTRAS=1``)."""
    return os.environ.get(REQUIRE_ENV) == "1"


def is_installed(extra: str) -> bool:
    """Whether ``extra``'s module is importable, without importing it.

    ``find_spec`` on the top-level name only: importing an extra has side
    effects (``mam4_jax`` turns ``jax_enable_x64`` on process-wide), and the
    test that needs it imports it anyway.
    """
    return importlib.util.find_spec(EXTRAS[extra]) is not None


def missing_extras() -> list[str]:
    """Return the registered extras that are not installed here."""
    return [extra for extra in sorted(EXTRAS) if not is_installed(extra)]


def extras_named_in(text: str) -> set[str]:
    """Return the extras whose package ``text`` names."""
    found = set()
    for match in _NAME_RE.finditer(text or ""):
        word = match.group(1).lower()
        for extra, names in PACKAGE_NAMES.items():
            if word in names:
                found.add(extra)
    return found


def pip_extras() -> str:
    """Return every registered extra as the list ``pip install -e ".[...]"`` takes."""
    return ",".join(sorted(EXTRAS))


def pyproject_extras(root: Path = ROOT) -> set[str]:
    """Return the optional-dependency groups ``pyproject.toml`` declares."""
    import tomllib

    with open(root / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return set(data["project"].get("optional-dependencies", {}))


# ---------------------------------------------------------------------------
# Static scan
# ---------------------------------------------------------------------------

_EXTRA_MODULES = {module: extra for extra, module in EXTRAS.items()}
_GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "Exception",
                     "BaseException"}
_SKIP_DIRS = {"__pycache__", "node_modules", "build", "dist", "site-packages"}


def _extra_of_module(name: str | None) -> str | None:
    """Return the extra whose module ``name`` (possibly dotted) belongs to."""
    if not name:
        return None
    return _EXTRA_MODULES.get(name.split(".")[0])


def _called_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _handler_guards_import(handler: ast.ExceptHandler) -> bool:
    """Whether ``handler`` would swallow a failed import."""
    if handler.type is None:
        return True
    types = (handler.type.elts if isinstance(handler.type, ast.Tuple)
             else [handler.type])
    return any(isinstance(t, ast.Name) and t.id in _GUARD_EXCEPTIONS
               or isinstance(t, ast.Attribute) and t.attr in _GUARD_EXCEPTIONS
               for t in types)


#: Calls that exist to ask whether a module is there: a gate wherever they are.
_PROBE_CALLS = ("importorskip", "find_spec")
#: Calls that import a module named by their argument: a gate under a guard.
_IMPORT_CALLS = ("import_module", "__import__")
#: jcm's own loaders that import an extra: guarding one is gating on it.
_EXTRA_LOADERS = {"require_pyses": "pyses"}
#: jcm modules that import an extra at import time: guarding one is gating on
#: that extra. (``jcm.dycore.pyses`` imports ``pyses`` lazily, so it is not
#: one.)
_EXTRA_BACKED_MODULES = {"jcm.physics.aerosol.jam.microphysics.mam4_jax": "mam4"}


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "string"`` bindings, to resolve probe arguments."""
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if (isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            consts[target.id] = node.value.value
    return consts


def _guarded_bodies(tree: ast.Module):
    """Statement lists run under a guard that swallows a failed import.

    A ``try`` with an ``except`` catching ``ImportError`` (or a superclass,
    or bare), and ``with contextlib.suppress(ImportError, ...)``.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.TryStar)) and any(
                _handler_guards_import(h) for h in node.handlers):
            yield node.body
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                ctx = item.context_expr
                if (isinstance(ctx, ast.Call) and _called_name(ctx) == "suppress"
                        and any(isinstance(a, (ast.Name, ast.Attribute))
                                and _handler_guards_import(
                                    ast.ExceptHandler(type=a))
                                for a in ctx.args)):
                    yield node.body


def gate_violations(source: str, filename: str = "<string>") -> list[str]:
    """Every way ``source`` gates on an extra other than the marker.

    Flags ``importorskip`` / ``find_spec`` of an extra's module (named by a
    string literal, or by a module-level constant), and, under a guard that
    swallows ``ImportError`` (``try``/``except`` or ``contextlib.suppress``),
    any import of one: the statement, ``importlib.import_module`` /
    ``__import__``, or a jcm loader that imports it (``require_pyses``). A plain import is not
    flagged: in a marked test it is how the test uses the extra, and anywhere
    else it fails loudly when the extra is absent rather than skipping.
    """
    tree = ast.parse(source, filename=filename)
    consts = _module_constants(tree)
    found = []

    def extra_of_arg(call):
        # The module is the first positional argument, or passed by keyword
        # (importorskip's ``modname``, import_module's and find_spec's
        # ``name``).
        arg = call.args[0] if call.args else next(
            (k.value for k in call.keywords if k.arg in ("modname", "name")),
            None)
        if arg is None:
            return None, None
        name = None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            name = arg.value
        elif isinstance(arg, ast.Name):
            name = consts.get(arg.id)
        return name, _extra_of_module(name)

    def flag(node, what, extra):
        found.append(f"{filename}:{node.lineno}: {what} gates on the "
                     f"{extra!r} extra; use @pytest.mark.{MARKER}({extra!r}) "
                     "instead")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _called_name(node) in _PROBE_CALLS:
            name, extra = extra_of_arg(node)
            if extra:
                flag(node, f"{_called_name(node)}({name!r})", extra)
    for body in _guarded_bodies(tree):
        for stmt in body:
            for inner in ast.walk(stmt):
                names = []
                if isinstance(inner, ast.Import):
                    names = [a.name for a in inner.names]
                elif isinstance(inner, ast.ImportFrom) and not inner.level:
                    names = [inner.module]
                for name in names:
                    extra = _extra_of_module(name) or _EXTRA_BACKED_MODULES.get(
                        name)
                    if extra:
                        flag(inner, f"a guarded `import {name}`", extra)
                if not isinstance(inner, ast.Call):
                    continue
                called = _called_name(inner)
                if called in _EXTRA_LOADERS:
                    flag(inner, f"a guarded `{called}()`",
                         _EXTRA_LOADERS[called])
                elif called in _IMPORT_CALLS:
                    name, extra = extra_of_arg(inner)
                    if extra:
                        flag(inner, f"a guarded `{called}({name!r})`", extra)
    return found


def test_files(root: Path = ROOT):
    """Every file pytest may load as a test or conftest under ``root``."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith("_test.py") or name == "conftest.py":
                yield Path(dirpath) / name


def scan(root: Path = ROOT) -> list[str]:
    """:func:`gate_violations` over every test file under ``root``."""
    found = []
    for path in test_files(root):
        found += gate_violations(path.read_text(),
                                 str(path.relative_to(root)))
    return found


def check(root: Path = ROOT) -> int:
    """Registry matches ``pyproject.toml`` and no test gates around the marker."""
    errors = []
    declared = pyproject_extras(root)
    if declared != set(EXTRAS):
        errors.append(
            f"EXTRAS {sorted(EXTRAS)} differs from pyproject.toml's "
            f"optional-dependencies {sorted(declared)}: add the new extra's "
            "probe module to EXTRAS and PACKAGE_NAMES")
    if set(PACKAGE_NAMES) != set(EXTRAS):
        errors.append("PACKAGE_NAMES and EXTRAS name different extras")
    errors += scan(root)
    for error in errors:
        print(f"::error::optional extras: {error}")
    print(f"optional extras: {pip_extras()}; "
          f"{len(errors)} problem(s)")
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# pytest plugin (registered by the root conftest.py)
# ---------------------------------------------------------------------------

def _marked_extras(item) -> list[str]:
    extras = []
    for mark in item.iter_markers(MARKER):
        extras.extend(mark.args)
    return extras


def pytest_configure(config):
    """Declare the marker; refuse a ``JCM_REQUIRE_EXTRAS=1`` session lacking an extra."""
    config.addinivalue_line(
        "markers",
        f"{MARKER}(*extras): needs the named optional extras "
        f"({', '.join(sorted(EXTRAS))}); skipped without them, failed under "
        f"{REQUIRE_ENV}=1 (tools/ci/optional_extras.py)")
    if required():
        missing = missing_extras()
        if missing:
            raise pytest.UsageError(
                f"{REQUIRE_ENV}=1 but these optional extras are not "
                f"installed: {', '.join(missing)} "
                f"(pip install -e \".[{pip_extras()}]\")")


def pytest_collection_modifyitems(config, items):
    """Reject a ``requires_extra`` marker that names no, or an unknown, extra."""
    for item in items:
        for mark in item.iter_markers(MARKER):
            unknown = [a for a in mark.args if a not in EXTRAS]
            if not mark.args or unknown:
                raise pytest.UsageError(
                    f"{item.nodeid}: @pytest.mark.{MARKER}{mark.args} must "
                    f"name extras from {sorted(EXTRAS)}")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Skip a marked test whose extra is absent, before its fixtures are built.

    Under ``JCM_REQUIRE_EXTRAS=1`` the session never gets this far without
    every extra, and a skip would be failed by the report hook anyway.
    """
    missing = [e for e in _marked_extras(item) if not is_installed(e)]
    if missing:
        pytest.skip(
            f"optional extra(s) not installed: {', '.join(missing)} "
            f"(pip install -e \".[{','.join(missing)}]\")")


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:   # a subtest's skip carries the exception's text instead
        reason = str(longrepr)
    return reason.removeprefix("Skipped: ")


_IN_CALL = pytest.StashKey[bool]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Mark the call phase, so a report made during it is known a subtest's.

    Subtests (``unittest``'s ``subTest``, pytest's ``subtests`` fixture) are
    reported through ``pytest_runtest_makereport`` while the test is still
    running, as plain ``TestReport`` objects; the test's own call report is
    made after this hook returns.
    """
    item.stash[_IN_CALL] = True
    try:
        return (yield)
    finally:
        item.stash[_IN_CALL] = False


def unmarked_skip_message(reason: str) -> str | None:
    """Return the failure for an unmarked skip naming an extra, else None."""
    extras = sorted(extras_named_in(reason))
    if not extras:
        return None
    marks = ", ".join(repr(e) for e in extras)
    return (f"skipped for the optional extra(s) {marks} without "
            f"@pytest.mark.{MARKER}({marks}), so no CI job runs it: the "
            f"default jobs lack the extra and the extras job selects "
            f"-m {MARKER}. Replace the gate with the marker. "
            f"Skip reason: {reason}")


def _fail(report, message: str) -> None:
    report.outcome = "failed"
    report.longrepr = message


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Fail the skips that would let an extras-gated test go unrun.

    A marked test that skips under ``JCM_REQUIRE_EXTRAS=1`` (or any of its
    subtests does) did not run in the one job that exists to run it. An
    unmarked test that skips for an extra is invisible to that job. An
    unmarked test's *subtest* that skips for one is left alone: the test
    itself runs in the default jobs, and gating part of it per case is a
    declared partial result, not a hidden test (the release-matrix
    regression records members whose extra is absent this way). An
    ``xfail`` is exempt too: it is a declared expectation, reported as such.
    """
    report = yield
    if not report.skipped or hasattr(report, "wasxfail"):
        return report
    reason = _skip_reason(report)
    if item.get_closest_marker(MARKER) is not None:
        if required():
            _fail(report,
                  f"@pytest.mark.{MARKER} test skipped under "
                  f"{REQUIRE_ENV}=1, so it did not run: {reason}")
        return report
    if call.when == "call" and item.stash.get(_IN_CALL, False):
        return report
    message = unmarked_skip_message(reason)
    if message:
        _fail(report, message)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """Fail a collection-time skip that could hide an extras-gated test.

    A module that skips at import takes every test in it out of every
    selection, its marker included. For an extra that is always wrong (the
    marker is the gate), and under ``JCM_REQUIRE_EXTRAS=1`` any such skip is
    a set of tests the extras job silently did not collect.
    """
    report = yield
    if report.skipped:
        reason = _skip_reason(report)
        message = unmarked_skip_message(reason)
        if message:
            _fail(report, "module " + message)
        elif required():
            _fail(report, f"collection skipped under {REQUIRE_ENV}=1, so "
                          f"its tests were never selected: {reason}")
    return report


def main(argv: list[str]) -> int:
    """CLI: ``pip-extras`` prints the install list, ``check`` verifies."""
    if argv == ["pip-extras"]:
        print(pip_extras())
        return 0
    if argv == ["check"]:
        return check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
