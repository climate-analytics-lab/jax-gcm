"""Tests for the optional-extras gate and its anti-rot checks."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import optional_extras  # noqa: E402


def test_registry_names_every_declared_extra():
    assert optional_extras.pyproject_extras() == set(optional_extras.EXTRAS)
    assert set(optional_extras.PACKAGE_NAMES) == set(optional_extras.EXTRAS)
    assert optional_extras.pip_extras() == ",".join(
        sorted(optional_extras.EXTRAS))


def test_repository_gates_only_through_the_marker():
    """The static half of the anti-rot check, run in every fast suite."""
    assert optional_extras.scan() == []


def test_the_root_conftest_registers_the_plugin(request):
    assert request.config.pluginmanager.has_plugin("jcm_optional_extras")


@pytest.mark.parametrize("source", [
    'import pytest\npytest.importorskip("pyses")\n',
    'import pytest\ndef test():\n    pytest.importorskip("mam4_jax.coupling")\n',
    'from pytest import importorskip\nimportorskip("jcosp")\n',
    'import importlib.util\nOK = importlib.util.find_spec("gcsfs") is not None\n',
    'try:\n    import jcosp\nexcept ImportError:\n    jcosp = None\n',
    'try:\n    from mam4_jax.core import data\nexcept (OSError, ModuleNotFoundError):\n    pass\n',
    'def f():\n    try:\n        import pyses.dynamical_cores\n    except Exception:\n        return\n',
    'try:\n    import gcsfs\nexcept:\n    pass\n',
    'M = "jcosp"\nimport pytest\npytest.importorskip(M)\n',
    'import importlib\ntry:\n    importlib.import_module("mam4_jax")\n'
    'except ImportError:\n    pass\n',
    'import contextlib\nwith contextlib.suppress(ImportError):\n    import pyses\n',
    'from contextlib import suppress\nwith suppress(OSError, ModuleNotFoundError):\n'
    '    from jcosp import config\n',
    'from jcm.dycore.pyses._pyses import require_pyses\ntry:\n'
    '    require_pyses()\nexcept ImportError:\n    pass\n',
    'M: str = "pyses"\nimport importlib.util\nimportlib.util.find_spec(M)\n',
    'import pytest\npytest.importorskip(modname="mam4_jax")\n',
    'try:\n    from jcm.physics.aerosol.jam.microphysics.mam4_jax import X\n'
    'except ImportError:\n    X = None\n',
])
def test_each_unmarked_gate_form_is_flagged(source):
    assert len(optional_extras.gate_violations(source)) == 1


@pytest.mark.parametrize("source", [
    'import pytest\npytest.importorskip("yaml")\n',
    'def test():\n    import pyses\n    from mam4_jax.coupling import amicphys\n',
    'import importlib.util\ndef has(m):\n    return importlib.util.find_spec(m)\n',
    'try:\n    import pyses\nexcept KeyError:\n    pass\n',
    'try:\n    import numpy\nexcept ImportError:\n    pass\n',
    'import importlib\ndef test():\n    importlib.import_module("mam4_jax")\n',
    'import contextlib\nwith contextlib.suppress(KeyError):\n    import pyses\n',
])
def test_uses_that_are_not_gates_are_not_flagged(source):
    assert optional_extras.gate_violations(source) == []


@pytest.mark.parametrize("reason, extras", [
    ("could not import 'pyses': No module named 'pyses'", {"pyses"}),
    ("jax-cosp not installed", {"cosp"}),
    ("WB2 store unreachable: Please install gcsfs to access Google Storage",
     {"era5"}),
    ("mam4-jax and pyses both missing", {"mam4", "pyses"}),
    ("pySES not installed", {"pyses"}),
    ("MAM4-JAX missing; JAX-COSP too", {"mam4", "cosp"}),
    ("could not import 'mam4_jax.coupling'", {"mam4"}),
    ("needs >= 2 devices", set()),
    ("pysesx and xgcsfs are other packages", set()),
    ("requires pySES-0.1.3", {"pyses"}),
    # jcm's own modules and paths that merely contain a package's name.
    ("jcm/dycore/pyses/forcing.nc absent", set()),
    ("could not import jcm.dycore.pyses.x: CUDA not available", set()),
    ("not importable: ['jcm.physics.aerosol.jam.microphysics.mam4_jax']",
     set()),
])
def test_skip_reasons_are_attributed_to_their_extra(reason, extras):
    assert optional_extras.extras_named_in(reason) == extras


def _run(tmp_path, files, *args, env=None):
    """Run pytest on ``files`` with only this plugin, return (code, output).

    ``conftest.py`` in ``files`` can replace ``optional_extras.is_installed``
    to fix which extras count as present, whatever this venv has.
    """
    for name, body in files.items():
        (tmp_path / name).write_text(textwrap.dedent(body))
    full_env = {k: v for k, v in os.environ.items()
                if k != optional_extras.REQUIRE_ENV}
    full_env.update(env or {})
    full_env["PYTHONPATH"] = os.pathsep.join(
        [str(_HERE), full_env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "optional_extras",
         "-p", "no:cacheprovider", "-p", "no:xdist", "-p", "no:cov",
         "-rA", "--rootdir", str(tmp_path), *args, str(tmp_path)],
        capture_output=True, text=True, cwd=tmp_path, env=full_env,
        timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


_ONLY_COSP = """
    import optional_extras
    optional_extras.is_installed = lambda extra: extra == "cosp"
"""

_ALL = """
    import optional_extras
    optional_extras.is_installed = lambda extra: True
"""

_CASES = """
    import pytest

    @pytest.mark.requires_extra("pyses")
    def test_marked_missing():
        raise AssertionError("must not run without its extra")

    @pytest.mark.requires_extra("cosp")
    def test_marked_present():
        pass

    def test_unmarked_gate():
        pytest.skip("could not import 'pyses': No module named 'pyses'")

    def test_unrelated_skip():
        pytest.skip("needs >= 2 devices")

    @pytest.mark.xfail(reason="pyses is expected to fail this", strict=True)
    def test_xfail_naming_an_extra():
        raise AssertionError
"""


def _outcomes(output):
    """{test name: outcome} from pytest's ``-rA`` short summary."""
    found = {}
    for line in output.splitlines():
        for outcome in ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL"):
            if line.startswith(outcome + " "):
                rest = line[len(outcome) + 1:]
                name = rest.split("::")[-1].split(" ")[0]
                if outcome == "SKIPPED":
                    # "SKIPPED [1] path:line: reason" carries no test name.
                    name = rest.split(": ", 1)[-1]
                found[name] = outcome
    return found


def test_default_session_skips_marked_and_fails_unmarked_gates(tmp_path):
    code, out = _run(tmp_path, {"conftest.py": _ONLY_COSP,
                                "test_cases.py": _CASES})
    outcomes = _outcomes(out)
    assert code == 1, out
    assert outcomes["test_marked_present"] == "PASSED", out
    assert outcomes["test_unmarked_gate"] == "FAILED", out
    assert outcomes["test_xfail_naming_an_extra"] == "XFAIL", out
    assert "needs >= 2 devices" in outcomes, out          # left skipped
    assert any("not installed: pyses" in k and v == "SKIPPED"
               for k, v in outcomes.items()), out
    assert "without @pytest.mark.requires_extra('pyses')" in out, out


def test_required_session_refuses_to_start_without_every_extra(tmp_path):
    code, out = _run(tmp_path, {"conftest.py": _ONLY_COSP,
                                "test_cases.py": _CASES},
                     env={optional_extras.REQUIRE_ENV: "1"})
    assert code == 4, out                                   # usage error
    assert "era5, m7, mam4, pyses" in out, out


def test_required_session_fails_any_skip_of_a_marked_test(tmp_path):
    code, out = _run(tmp_path, {"conftest.py": _ALL, "test_req.py": """
        import pytest

        pytestmark = pytest.mark.requires_extra("mam4")

        def test_runs():
            pass

        def test_skips_for_another_reason():
            pytest.skip("the network is down")
    """}, env={optional_extras.REQUIRE_ENV: "1"})
    outcomes = _outcomes(out)
    assert code == 1, out
    assert outcomes["test_runs"] == "PASSED", out
    assert outcomes["test_skips_for_another_reason"] == "FAILED", out
    assert "did not run: " in out and "the network is down" in out, out


def test_selection_by_marker_includes_every_marked_test(tmp_path):
    code, out = _run(tmp_path, {"conftest.py": _ALL, "test_sel.py": """
        import pytest

        @pytest.mark.slow
        @pytest.mark.requires_extra("pyses")
        def test_slow_marked():
            pass

        @pytest.mark.requires_extra("cosp", "mam4")
        def test_fast_marked():
            pass

        def test_unmarked():
            pass
    """}, "-m", "requires_extra", "-p", "no:warnings")
    outcomes = _outcomes(out)
    assert code == 0, out
    assert outcomes == {"test_slow_marked": "PASSED",
                        "test_fast_marked": "PASSED"}, out


_SUBTESTS = """
    import unittest

    import pytest

    class Unmarked(unittest.TestCase):
        def test_cases(self):
            for extra in ("pyses", "none"):
                with self.subTest(extra=extra):
                    if extra == "pyses":
                        self.skipTest("optional extra 'pyses' not installed")

    def test_fixture_cases(subtests):
        with subtests.test(msg="mam4"):
            pytest.skip("mam4_jax not installed")

    @pytest.mark.requires_extra("cosp")
    class Marked(unittest.TestCase):
        def test_cases(self):
            with self.subTest(case=1):
                self.skipTest("the network is down")
"""


def _summary(out):
    return out.split("short test summary info")[-1]


def test_an_unmarked_tests_subtest_may_skip_for_an_extra(tmp_path):
    # -v: pytest reports subtest outcomes only when verbose.
    code, out = _run(tmp_path, {"conftest.py": _ALL, "test_sub.py": _SUBTESTS},
                     "-v")
    summary = _summary(out)
    assert code == 0, out
    assert "SUBFAILED" not in summary and "FAILED" not in summary, out
    assert "optional extra 'pyses' not installed" in summary, out
    assert "mam4_jax not installed" in summary, out


def test_a_required_marked_tests_subtest_may_not_skip(tmp_path):
    code, out = _run(tmp_path, {"conftest.py": _ALL, "test_sub.py": _SUBTESTS},
                     "-v", env={optional_extras.REQUIRE_ENV: "1"})
    summary = _summary(out)
    assert code == 1, out
    assert "SUBFAILED(case=1) test_sub.py::Marked::test_cases" in summary, out
    assert "did not run: the network is down" in out, out
    # The unmarked tests' per-case skips are still only skips.
    assert "optional extra 'pyses' not installed" in summary, out
    assert "mam4_jax not installed" in summary, out


def test_required_session_fails_any_collection_skip(tmp_path):
    code, out = _run(tmp_path, {"conftest.py": _ALL, "test_mod.py": """
        import pytest

        pytest.skip("this platform is unsupported", allow_module_level=True)

        def test_hidden():
            pass
    """}, env={optional_extras.REQUIRE_ENV: "1"})
    assert code == 2, out
    assert "collection skipped under JCM_REQUIRE_EXTRAS=1" in out, out


def test_module_level_skip_for_an_extra_is_a_collection_error(tmp_path):
    code, out = _run(tmp_path, {"conftest.py": _ONLY_COSP, "test_mod.py": """
        import pytest

        pytest.skip("jax-cosp not installed", allow_module_level=True)

        def test_hidden():
            pass
    """})
    assert code == 2, out                                   # collection error
    assert "module skipped for the optional extra(s) 'cosp'" in out, out


@pytest.mark.parametrize("args", ["", '"nonsense"'])
def test_marker_must_name_known_extras(tmp_path, args):
    code, out = _run(tmp_path, {"conftest.py": _ALL, "test_bad.py": f"""
        import pytest

        @pytest.mark.requires_extra({args})
        def test_bad():
            pass
    """})
    assert code == 4, out
    assert "must name extras from" in out, out


def test_cli_prints_the_install_list():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "optional_extras.py"), "pip-extras"],
        capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == optional_extras.pip_extras()
