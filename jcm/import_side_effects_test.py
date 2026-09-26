"""Importing jcm must not initialise a JAX backend or touch the GPU (#859).

JAX initialises its backends lazily, on the first query that needs a device
(``jnp.array``, ``jax.devices()``, ...). ``import jax`` alone does not do it, so
if ``import jcm`` does, some jcm module is building a jax array at import time
— a module-level table, a ``def`` default argument, a class attribute. On a GPU
host that first query brings up the CUDA backend, which under JAX's default
``XLA_PYTHON_CLIENT_PREALLOCATE`` claims 75 % of the card for a process that
may never do device work (an orchestrator whose integrations run in
subprocesses, a REPL inspecting output).

How the check works, and why it is robust:

* Each check runs in a **fresh subprocess**: in the pytest process some other
  test has long since initialised the backend, and ``import`` is cached.
* The CPU-safe checks ask JAX's own backend registry whether it has been
  initialised (``jax._src.xla_bridge.backends_are_initialized``). That is the
  single choke point every device-array creation passes through, so it catches
  every way a jax array can be made — not just the ``jnp.array`` spellings a
  monkeypatch of individual constructors would see.
* The GPU check does not diff total card memory (a shared card's other tenants
  make that noisy); it asks ``nvidia-smi`` whether the *child's own PID* holds
  a CUDA context, with a positive control (the same child after creating one
  array) proving the query can see the process at all — otherwise it skips
  rather than passing vacuously.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shared prelude: resolve the backend-initialised predicate once. It is private
# JAX API, so fall back to the registry dict it reads if it is ever renamed.
_PRELUDE = textwrap.dedent(
    """
    import jax
    from jax._src import xla_bridge as _xb

    def backend_initialised():
        fn = getattr(_xb, "backends_are_initialized", None)
        return bool(fn()) if fn is not None else bool(_xb._backends)

    assert not backend_initialised(), "import jax alone initialised a backend"
    x64_before = bool(jax.config.read("jax_enable_x64"))
    """
)


def _run_child(body: str, *, env_overrides=None, timeout=600):
    env = {**os.environ, **(env_overrides or {})}
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


def _json_tail(proc):
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_jcm_does_not_initialise_a_backend():
    """``import jcm`` builds no jax array and leaves ``jax_enable_x64`` alone."""
    proc = _run_child(
        """
        import json
        import jcm  # noqa: F401
        print(json.dumps({
            "initialised": backend_initialised(),
            "x64_changed": bool(jax.config.read("jax_enable_x64")) != x64_before,
        }))
        """,
        env_overrides={"JAX_PLATFORMS": "cpu"},
    )
    result = _json_tail(proc)
    assert not result["initialised"], (
        "`import jcm` initialised a JAX backend: some module on jcm's import "
        "chain builds a jax array at import time (#859)."
    )
    assert not result["x64_changed"]


# Adapters around optional third-party cores that build device arrays in their
# own import. Not imported by the walk below (doing so would initialise the
# backend and blind the check for every later module). Accepted because the
# adapter is itself imported lazily — only when a configuration selects that
# core, i.e. when the model is about to use the device anyway — so it never
# runs on ``import jcm`` or on importing the physics packages that can select it.
_THIRD_PARTY_BACKEND_AT_IMPORT = {
    # mam4_jax (jcm[mam4] extra) builds its coagulation tables with
    # ``jnp.asarray`` at module level (mam4_jax/physics/coag.py).
    "jcm.physics.aerosol.jam.microphysics.mam4_jax",
}


def test_no_jcm_module_initialises_a_backend_at_import():
    """Every jcm module is import-side-effect free, not just ``jcm/__init__``.

    Users import physics packages directly (``from jcm.physics.echam.echam_terms
    import echam_physics``), so the property has to hold for the whole package.
    Modules are imported one by one and the predicate checked after each, so a
    failure names the first offender. A module whose *optional* dependency is
    not installed (e.g. the ``jcm[mam4]`` adapter) raises ``ImportError`` and is
    skipped: it cannot be checked here, and the dependency is third-party code.
    Where the optional dependency *is* installed, the adapters in
    ``_THIRD_PARTY_BACKEND_AT_IMPORT`` are skipped for the reason given there.
    """
    proc = _run_child(
        """
        import importlib, json, pkgutil
        import jcm
        excluded = EXCLUDED
        offender, skipped, checked = None, [], 0
        for info in pkgutil.walk_packages(jcm.__path__, "jcm."):
            leaf = info.name.rsplit(".", 1)[-1]
            if (leaf.endswith("_test") or leaf in ("conftest", "__main__")
                    or info.name in excluded):
                continue
            try:
                importlib.import_module(info.name)
            except ImportError as exc:
                skipped.append(f"{info.name}: {exc}")
                continue
            checked += 1
            if backend_initialised():
                offender = info.name
                break
        print(json.dumps({
            "offender": offender, "skipped": skipped, "checked": checked,
            "x64_changed": bool(jax.config.read("jax_enable_x64")) != x64_before,
        }))
        """.replace("EXCLUDED", repr(sorted(_THIRD_PARTY_BACKEND_AT_IMPORT))),
        env_overrides={"JAX_PLATFORMS": "cpu"},
    )
    result = _json_tail(proc)
    assert result["offender"] is None, (
        f"importing {result['offender']} initialised a JAX backend: it (or a "
        "module it imports) builds a jax array at import time (#859)."
    )
    # Guard against the walk silently checking nothing.
    assert result["checked"] > 100, result
    assert not result["x64_changed"], "a jcm module flips jax_enable_x64 on import"


def _gpu_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import jax_plugins  # noqa: F401  (namespace the CUDA plugin installs into)
    except ImportError:
        return False
    return True


def _pids_with_cuda_context():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        return None
    return {int(line) for line in out.stdout.split() if line.strip().isdigit()}


@pytest.mark.skipif(not _gpu_available(), reason="needs nvidia-smi and a CUDA jax plugin")
def test_import_jcm_takes_no_gpu_memory():
    """With the CUDA backend available, ``import jcm`` opens no CUDA context.

    The child imports jcm, reports its PID and waits; the parent reads which
    PIDs hold a CUDA context. The child then makes one array (the positive
    control: now it *must* appear) and waits again. Preallocation is disabled
    in the child so the control does not claim 75 % of a shared card.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(
            """
            import os, sys
            import jcm  # noqa: F401
            print(os.getpid(), backend_initialised(), flush=True)
            sys.stdin.readline()
            import jax.numpy as jnp
            x = jnp.ones(4).block_until_ready()
            print(jax.default_backend(), flush=True)
            sys.stdin.readline()
            """
        )],
        cwd=_REPO_ROOT, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**{k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"},
             "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    try:
        pid_str, initialised = child.stdout.readline().split()
        pid = int(pid_str)
        after_import = _pids_with_cuda_context()
        child.stdin.write("\n")
        child.stdin.flush()
        backend = child.stdout.readline().strip()
        after_array = _pids_with_cuda_context()
        child.stdin.write("\n")
        child.stdin.flush()
        child.wait(timeout=120)
    finally:
        if child.poll() is None:
            child.kill()
    if backend != "gpu" or after_array is None or pid not in after_array:
        pytest.skip(
            f"cannot observe this process's CUDA context (backend={backend!r}); "
            "e.g. nvidia-smi in a separate PID namespace"
        )
    assert initialised == "False"
    assert pid not in after_import, "`import jcm` opened a CUDA context (#859)"
