"""Run provenance capture (#591).

Records what actually produced an output file — code versions (git SHA +
dirty state of every editable install *actually imported*), precision and
hardware, the resolved boundary files opened, the ozone source, and a
single run hash — and attaches it as netCDF global attributes plus a
``<output>.provenance.json`` sidecar carrying the fully composed config.

The probe inspects ``sys.modules`` in the running process, so a
``PYTHONPATH`` override is reflected as *what was imported*, not what was
requested. Input files funnel through ``runners._resolve_data_path`` and
are recorded here as they resolve; content hashing of multi-GB inputs is
opt-in (``JCM_HASH_INPUTS=1``) — size + mtime is the default descriptor.

Lifecycle: :func:`start_run` at run start (captures code/env/config and
resets the input registry), :func:`record_input` / :func:`record_fact`
during model build, :func:`attrs` / :func:`write_sidecar` at output time
(inputs and the run hash are finalized lazily, after the build has
touched every file).
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: Libraries whose working tree changes what a run computes. Probed only
#: when already imported — an absent entry means "not part of this run".
_EDITABLE_LIBS = ("jcm", "dinosaur", "rrtmgp", "mam4_jax", "jax_cosp",
                  "pyses")
#: Module-name -> distribution-name where they differ.
_DIST_NAMES = {"rrtmgp": "jax-rrtmgp", "mam4_jax": "mam4-jax",
               "jax_cosp": "jax-cosp"}
#: Packaged dependencies recorded by version only.
_VERSION_LIBS = ("jax", "jaxlib", "numpy", "xarray", "flax")
#: Environment variables that silently change precision or compilation.
_ENV_FLAGS = ("JAX_ENABLE_X64", "MAM4_JAX_ENABLE_X64", "XLA_FLAGS",
              "JCM_CACHE_DIR")

_state: dict = {"base": None, "inputs": {}, "facts": {}}


def _git(path: str, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def probe_code() -> dict:
    """Path + version + git SHA/branch/dirty for every imported key library."""
    from importlib import metadata

    code: dict = {}
    for name in _EDITABLE_LIBS:
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "__file__", None):
            continue
        path = str(Path(mod.__file__).resolve().parent)
        entry: dict = {"path": path}
        # Module and distribution names differ (rrtmgp -> jax-rrtmgp).
        for dist in (_DIST_NAMES.get(name, name), name.replace("_", "-")):
            try:
                entry["version"] = metadata.version(dist)
                break
            except metadata.PackageNotFoundError:
                pass
        top = _git(path, "rev-parse", "--show-toplevel")
        if top:
            entry["sha"] = _git(top, "rev-parse", "HEAD")
            entry["branch"] = _git(top, "rev-parse", "--abbrev-ref", "HEAD")
            status = _git(top, "status", "--porcelain")
            entry["dirty"] = bool(status)
            if status:
                # Distinguish two dirty trees without shipping the diff.
                diff = _git(top, "diff", "HEAD") or ""
                entry["dirty_diff_sha"] = hashlib.sha256(
                    diff.encode()).hexdigest()[:12]
        code[name] = entry
    for name in _VERSION_LIBS:
        try:
            code[name] = {"version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            pass
    return code


def probe_environment() -> dict:
    """Precision flags, devices, host — the silent run-shapers."""
    import jax

    devices = jax.devices()
    env = {
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "env": {k: os.environ[k] for k in _ENV_FLAGS if k in os.environ},
        "platform": devices[0].platform,
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
    }
    try:
        # On GPU this carries the CUDA runtime/driver versions.
        env["platform_version"] = (
            jax.extend.backend.get_backend().platform_version)
    except Exception:  # noqa: BLE001 — best-effort, backend-dependent
        pass
    return env


def describe_input(path: str) -> dict:
    """Size + mtime descriptor; content sha256 when JCM_HASH_INPUTS=1."""
    d: dict = {}
    try:
        st = os.stat(path)
        d["size"] = st.st_size
        d["mtime"] = datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        d["missing"] = True
        return d
    if os.environ.get("JCM_HASH_INPUTS") == "1":
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        d["sha256"] = h.hexdigest()
    return d


def start_run(cfg=None) -> None:
    """Reset the registries and capture the config at run start.

    Code and environment are probed lazily (and re-probed on every
    :func:`collect`): the model build imports configuration-selected
    libraries (pyses, mam4_jax, …) and can flip the live x64 setting
    *after* run start, so an eager snapshot here would miss them.
    """
    config_yaml = None
    if cfg is not None:
        from omegaconf import OmegaConf
        from omegaconf.errors import OmegaConfBaseException
        try:
            config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        except OmegaConfBaseException:
            # hydra:-runtime interpolations only resolve inside a live
            # Hydra app; the unresolved form still records every choice.
            config_yaml = OmegaConf.to_yaml(cfg, resolve=False)
    _state["base"] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_yaml": config_yaml,
    }
    _state["inputs"] = {}
    _state["facts"] = {}


def record_input(requested, resolved=None) -> None:
    """Record a boundary/input file the run resolved (deduped by path)."""
    resolved = str(resolved if resolved is not None else requested)
    if not os.path.isfile(resolved):
        return
    entry = describe_input(resolved)
    if str(requested) != resolved:
        entry["requested"] = str(requested)
    _state["inputs"][resolved] = entry


def record_fact(key: str, value) -> None:
    """Record a run-shaping fact (e.g. ozone_source = prescribed/analytic)."""
    _state["facts"][key] = value


def collect() -> dict:
    """Return the full provenance record.

    Code and environment are probed fresh each call (a handful of git
    subprocesses — negligible next to a netCDF write), so libraries the
    model build imported lazily and any post-start x64 flip are captured.
    """
    if _state["base"] is None:
        start_run()
    prov = dict(_state["base"])
    prov["code"] = probe_code()
    prov["environment"] = probe_environment()
    prov["inputs"] = dict(_state["inputs"])
    prov["facts"] = dict(_state["facts"])
    hash_material = {
        # A dirty tree is a different code state than its HEAD: fold the
        # working-tree diff fingerprint into the identity.
        "code": {k: (v.get("sha", v.get("version")),
                     v.get("dirty_diff_sha"))
                 for k, v in prov["code"].items()},
        "config": prov["config_yaml"],
        "inputs": {k: v.get("sha256", (v.get("size"), v.get("mtime")))
                   for k, v in prov["inputs"].items()},
        "x64": prov["environment"]["jax_enable_x64"],
    }
    prov["run_hash"] = hashlib.sha256(
        json.dumps(hash_material, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return prov


def summary() -> str:
    """One log line: SHAs, precision, ozone source (probed fresh)."""
    code = probe_code()
    env = probe_environment()
    shas = ", ".join(
        f"{k}={v['sha'][:8]}{'+dirty' if v.get('dirty') else ''}"
        for k, v in code.items() if "sha" in v)
    ozone = _state["facts"].get("ozone_source", "?")
    return (f"{shas or 'no git trees'}; "
            f"x64={env.get('jax_enable_x64')}; "
            f"{env.get('device_count')}x{env.get('device_kind')}; "
            f"ozone={ozone}")


def attrs() -> dict:
    """Flat netCDF-safe global attributes (nested parts as JSON strings)."""
    prov = collect()
    out = {
        "jcm_prov_created": prov["created"],
        # attrs() runs at output time, so this stamps the write — with
        # ``created`` it brackets the run without a separate end hook.
        "jcm_prov_written": datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        "jcm_prov_run_hash": prov["run_hash"],
        "jcm_prov_code": json.dumps(prov["code"], sort_keys=True),
        "jcm_prov_environment": json.dumps(prov["environment"],
                                           sort_keys=True),
        "jcm_prov_inputs": json.dumps(prov["inputs"], sort_keys=True),
    }
    for key, value in prov["facts"].items():
        out[f"jcm_prov_{key}"] = str(value)
    if prov["config_yaml"] is not None:
        # The composed config is too large for attributes (it goes in the
        # sidecar); the hash makes "same config" checkable from attrs.
        out["jcm_prov_config_sha"] = hashlib.sha256(
            prov["config_yaml"].encode()).hexdigest()[:12]
    return out


def write_sidecar(output_path) -> Path:
    """Write ``<output>.provenance.json`` with the full record + config."""
    output_path = Path(output_path)
    sidecar = output_path.with_suffix(output_path.suffix +
                                      ".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(collect(), indent=1, sort_keys=True,
                                  default=str))
    return sidecar
