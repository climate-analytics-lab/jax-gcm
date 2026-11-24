"""Run provenance capture (#591, #732).

Records what actually produced an output file — code versions (git SHA +
dirty state of every editable install *actually imported*), precision and
hardware, the resolved boundary files opened, the ozone source, the
parameter values the model actually ran with, and a single run hash — and
attaches it as netCDF global attributes plus a
``<output>.provenance.json`` sidecar carrying the fully composed config.

The probe inspects ``sys.modules`` in the running process, so a
``PYTHONPATH`` override is reflected as *what was imported*, not what was
requested. Input files funnel through ``runners._resolve_data_path`` and
are recorded here as they resolve; content hashing of multi-GB inputs is
opt-in (``JCM_HASH_INPUTS=1``) — size + mtime is the default descriptor.
The one exception is a warm start's donor state (``init=from_state``,
#937): it is always one file, size + mtime do not survive a copy to
another machine, and the state *is* the run's initial condition, so
``runners._preflight_from_state_init`` (called from ``runners.run``) always
hashes it via ``record_input(..., force_hash=True)`` — once per process,
before the fresh-start/resume branch is chosen, so a chunked run's
post-preemption chunks hash the same donor as its pre-preemption ones
(#1112).

Parameters are captured separately from the config and by a different
route, because the config does not determine them (#732). A scheme's
``params`` block is deliberately absent from the shipped yamls so each
field falls back to ``Parameters.default()`` in code, so the composed
config records the *overrides* and says nothing about the effective
values; and a model built in Python, or one whose parameters were
replaced after construction (what a calibration loop does), has no
config behind it at all. :func:`describe_params` therefore reads the
built physics, and only the parameters: the rest of what a run is (the
term composition, the dycore, the resolution) is already in the config.
It is called from inside ``Model._run_from_state`` while that jit is
tracing, which is where the values become the ones the executable will
use — see :class:`jcm.predictions.ModelPredictions` for why reading them
afterwards is not the same thing.

Lifecycle: :func:`start_run` at run start (captures the config and resets
the registries — it is the only call that empties them),
:func:`record_input` / :func:`record_fact` during model build,
:func:`attrs` / :func:`write_sidecar` at output time (inputs and the run
hash are finalized lazily, after the build has touched every file). A
process that never calls :func:`start_run` (a model built from Python)
still gets a record: the first thing recorded, or the first
:func:`collect`, opens it without clearing anything, so what the build
recorded reaches the output. Probes are best-effort and cannot fail a run:
a probe that cannot produce its value leaves a ``<probe failed: ...>``
marker in the record rather than raising. The parameter record travels on the predictions
object rather than through this module's registry: two models built in
one process would otherwise overwrite each other's parameters, and the
one that wrote the file need not be the one that ran last.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: ``object.__repr__``'s ``at 0x7f...`` tail, which is run-dependent.
_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")

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

#: Parameter arrays up to this many elements are recorded by value; larger
#: ones become a shape/dtype/hash summary. Embedded NN weights are
#: parameters too, and a few million of them must not land in a netCDF
#: attribute — the hash still distinguishes one weight set from another.
#: 64 rather than a smaller round number so that the per-plume MACv2-SP
#: shape parameters (2x9) and per-level profiles are kept as values: they
#: are tuned, so a hash of them is not a useful record.
_PARAM_ARRAY_MAX_ELEMS = 64
#: Depth cap on the parameter walk. A guard against a cyclic or
#: pathologically nested container, not a real structural limit (the
#: deepest shipped parameter struct nests two levels).
_PARAM_MAX_DEPTH = 6

#: What an environment-dependent probe can raise: a missing or unreadable
#: file, a process that will not start or finish, undecodable bytes, a
#: broken install's metadata, an absent backend. These are the failures a
#: probe records and survives. ``TypeError`` / ``AttributeError`` /
#: ``NameError`` are deliberately absent: those are bugs in this module and
#: must stay loud.
_PROBE_ERRORS = (OSError, ValueError, RuntimeError, KeyError,
                 subprocess.SubprocessError)

_state: dict = {"base": None, "inputs": {}, "facts": {}}


def _probe_failed(exc: BaseException) -> str:
    """Return the sentinel stored where a probe could not produce its value.

    Provenance is best-effort: no probe may stop an integration. A probe
    that fails leaves this marker in the record, so the gap shows in the
    netCDF attributes and the sidecar, instead of raising or silently
    dropping the entry.
    """
    logger.warning("provenance probe failed: %s: %s", type(exc).__name__, exc)
    return f"<probe failed: {type(exc).__name__}>"


def _git_bytes(path: str, *args: str) -> bytes | None:
    """Return a git subcommand's stdout as bytes, or ``None`` on any failure.

    Bytes, not text: ``git diff`` of a modified tracked binary file (or of
    a text file in a non-UTF-8 encoding) is not decodable, and a decoding
    error here would escape into the run.
    """
    try:
        out = subprocess.run(["git", "-C", path, *args],
                             capture_output=True, timeout=10)
    except (OSError, ValueError, subprocess.SubprocessError):
        # OSError: no git binary / unreadable cwd; ValueError: an embedded
        # NUL in *path*; SubprocessError includes TimeoutExpired.
        return None
    return out.stdout if out.returncode == 0 else None


def _git(path: str, *args: str) -> str | None:
    """Return a git subcommand's stripped stdout as text, or ``None``.

    Undecodable bytes (a non-UTF-8 branch or path name) become U+FFFD
    rather than raising. A caller that only fingerprints the output hashes
    the bytes from :func:`_git_bytes` instead.
    """
    out = _git_bytes(path, *args)
    return None if out is None else out.decode("utf-8", "replace").strip()


def _dist_version(dist: str) -> str | None:
    """Return the installed version of *dist*, or ``None`` if not installed.

    A present-but-broken distribution (unreadable or malformed metadata)
    yields the probe-failed marker rather than an exception.
    """
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None
    except _PROBE_ERRORS as exc:  # UnicodeDecodeError is a ValueError
        return _probe_failed(exc)


def _probe_library(name: str, mod) -> dict:
    """Path + version + git SHA/branch/dirty for one imported library."""
    path = str(Path(mod.__file__).resolve().parent)
    entry: dict = {"path": path}
    # Module and distribution names differ (rrtmgp -> jax-rrtmgp).
    for dist in (_DIST_NAMES.get(name, name), name.replace("_", "-")):
        version = _dist_version(dist)
        if version is not None:
            entry["version"] = version
            break
    top = _git(path, "rev-parse", "--show-toplevel")
    if top:
        entry["sha"] = _git(top, "rev-parse", "HEAD")
        entry["branch"] = _git(top, "rev-parse", "--abbrev-ref", "HEAD")
        status = _git(top, "status", "--porcelain")
        # ``None`` is "git could not say", not "clean": a failed status
        # must not read as a pristine tree.
        entry["dirty"] = "unknown" if status is None else bool(status)
        if status:
            # Distinguish two dirty trees without shipping the diff. Hashed
            # as bytes: the diff can contain a binary file's raw content, and
            # the fingerprint must not depend on a text codec. The
            # external-diff and textconv hooks are off so the bytes are
            # git's own and do not vary with the user's git config.
            diff = _git_bytes(top, "diff", "--no-ext-diff", "--no-textconv",
                              "HEAD")
            entry["dirty_diff_sha"] = (
                "<probe failed: git diff>" if diff is None
                else hashlib.sha256(diff).hexdigest()[:12])
    return entry


def probe_code() -> dict:
    """Path + version + git SHA/branch/dirty for every imported key library.

    Never raises for an environmental failure: a library whose probe fails
    is recorded as ``{"probe_error": "<probe failed: ...>"}``.
    """
    code: dict = {}
    for name in _EDITABLE_LIBS:
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "__file__", None):
            continue
        try:
            code[name] = _probe_library(name, mod)
        except _PROBE_ERRORS as exc:
            code[name] = {"probe_error": _probe_failed(exc)}
    for name in _VERSION_LIBS:
        version = _dist_version(name)
        if version is not None:
            code[name] = {"version": version}
    return code


def _probed(fn):
    """Call *fn*; on an environmental failure return the probe-failed marker."""
    try:
        return fn()
    except _PROBE_ERRORS as exc:
        return _probe_failed(exc)


def probe_environment() -> dict:
    """Precision flags, devices, host — the silent run-shapers.

    Each fact is probed on its own: a container with no passwd entry for
    the uid (``getpass.getuser``) or a backend that will not enumerate
    devices leaves a probe-failed marker for that field only.
    """
    import jax

    env = {
        "python": platform.python_version(),
        "hostname": _probed(socket.gethostname),
        "user": _probed(getpass.getuser),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "env": {k: os.environ[k] for k in _ENV_FLAGS if k in os.environ},
    }
    devices = _probed(jax.devices)
    if isinstance(devices, str):
        env.update(platform=devices, device_kind=devices, device_count=None)
    else:
        env.update(platform=devices[0].platform,
                   device_kind=devices[0].device_kind,
                   device_count=len(devices))
    try:
        # On GPU this carries the CUDA runtime/driver versions.
        env["platform_version"] = (
            jax.extend.backend.get_backend().platform_version)
    except Exception:  # noqa: BLE001 — best-effort, backend-dependent
        pass
    return env


def describe_input(path: str, *, force_hash: bool = False) -> dict:
    """Size + mtime descriptor; content sha256 when hashed.

    Content hashing is opt-in by default (``JCM_HASH_INPUTS=1``), because
    most inputs (JAM's gas/aerosol boundary files, …) run to several GB and
    hashing every one would make every run pay for it. ``force_hash=True``
    hashes this one call regardless of the env switch — for a single input
    whose identity matters enough to always record (see
    :func:`record_input`'s ``force_hash`` and #937), streamed in the same
    4 MiB blocks so the cost doesn't scale with a Python-level buffer.
    """
    d: dict = {}
    try:
        st = os.stat(path)
        d["size"] = st.st_size
        d["mtime"] = datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except (OSError, ValueError):  # ValueError: embedded NUL in the path
        d["missing"] = True
        return d
    if force_hash or os.environ.get("JCM_HASH_INPUTS") == "1":
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 22), b""):
                    h.update(block)
        except OSError as exc:
            # Stat succeeded but the content cannot be read (permissions, a
            # file that vanished, an I/O error mid-read): keep the
            # size + mtime descriptor and mark the digest unavailable.
            d["sha256_error"] = _probe_failed(exc)
            return d
        d["sha256"] = h.hexdigest()
    return d


def _describe_leaf(value):
    """JSON-safe description of one parameter leaf."""
    import jax
    import numpy as np

    if isinstance(value, jax.core.Tracer):
        # Captured inside jit/grad/vmap — a calibration loop differentiating
        # through the model. The value is not concrete, and unlike a lazy
        # array it cannot be forced.
        return "<traced>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value):
        # A parameter block may hold a schedule or an activation.
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001 — anything unconvertible is described
        return _stable_repr(value)
    if arr.dtype == object:
        return _stable_repr(value)
    if arr.ndim == 0:
        # ``item()`` on a float32 widens to the exact float64 that value
        # represents (0.1 -> 0.10000000149011612). That is the number the
        # model used, so record it rather than a prettier rounding.
        return arr.item()
    # Shape and dtype travel with the values, not only with the hashed
    # summary: a bare flat list makes a (2, 3) parameter identical to a
    # (3, 2) one with the same row-major bytes, and a float32 vector
    # identical to its float64 twin.
    described = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size <= _PARAM_ARRAY_MAX_ELEMS:
        described["values"] = arr.ravel().tolist()
    else:
        described["sha256"] = hashlib.sha256(
            np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
    return described


def _stable_repr(value) -> str:
    """Return ``repr(value)`` with the ``at 0x...`` object address removed.

    The default ``object.__repr__`` embeds ``id(value)``, which differs
    between two runs of the identical configuration. Left in, it would
    make ``params_sha`` (and through it ``run_hash``) non-reproducible.
    """
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 — a user object's __repr__ may raise anything
        return f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"
    return _ADDRESS_RE.sub("", text)[:200]


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _is_knob(value) -> bool:
    """Report whether a leaf is a tuning knob rather than cached grid data.

    Scalars, enum members and 0-d arrays. Terms cache their coordinates in
    plain variables sitting beside their knobs with no naming convention
    between the two (Held-Suarez keeps ``kf``/``ka``/``ks`` next to its
    ``sigma`` and ``latitudes`` caches), so the split has to be structural.
    ``ndim`` is read rather than converted, to avoid pulling a device array
    to the host only to discard it.
    """
    import enum

    return (_is_scalar(value) or isinstance(value, enum.Enum)
            or getattr(value, "ndim", None) == 0)


def _describe_value(value, prefix: str, out: dict, depth: int = 0,
                    knobs_only: bool = False) -> None:
    """Flatten *value* into *out* under the dotted key *prefix*.

    With *knobs_only*, bulk arrays are skipped but containers are still
    walked for the knobs inside them, since a parameter block mixes the
    two.
    """
    import dataclasses
    from collections.abc import Mapping

    if depth > _PARAM_MAX_DEPTH:
        out[prefix] = f"<truncated {type(value).__name__}>"
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # ``dataclasses.fields``, not ``tree_flatten_with_path``: the
        # tree_math structs these parameter blocks use register as pytrees
        # without key paths, so flattening yields "[<flat index 3>]" where
        # the field name is the entire point of the record.
        for f in dataclasses.fields(value):
            _describe_value(getattr(value, f.name), f"{prefix}.{f.name}",
                            out, depth + 1, knobs_only)
        return
    if hasattr(value, "_fields") and isinstance(value, tuple):  # NamedTuple
        for name in value._fields:
            _describe_value(getattr(value, name), f"{prefix}.{name}",
                            out, depth + 1, knobs_only)
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            _describe_value(v, f"{prefix}.{k}", out, depth + 1, knobs_only)
        return
    if isinstance(value, (list, tuple)):
        # A short all-scalar sequence is one value (a per-level profile);
        # anything else is a container to walk.
        if all(_is_scalar(v) for v in value):
            if len(value) <= _PARAM_ARRAY_MAX_ELEMS:
                out[prefix] = list(value)
            else:
                out[prefix] = {
                    "length": len(value),
                    "sha256": hashlib.sha256(
                        json.dumps(list(value), default=str).encode()
                    ).hexdigest()[:12],
                }
            return
        for i, v in enumerate(value):
            _describe_value(v, f"{prefix}.{i}", out, depth + 1, knobs_only)
        return
    if knobs_only and not _is_knob(value):
        return
    out[prefix] = _describe_leaf(value)


def describe_params(physics=None) -> dict:
    """Return the parameter values a run actually used, by dotted key.

    Reads the *built* physics rather than the requested config, because
    the config does not determine the values (see the module docstring).
    It deliberately stops at parameters: everything else about a run (the
    term composition, the dycore, the resolution, the physical constants)
    is already in the config record this sits beside, so recording it a
    second time would add bulk without adding information.

    Keys are ``<term name>.<variable>.<field>``, for example
    ``tiedtke_convection.params.entrpen``. That names where the value
    *lives*, which is close to but not always the Hydra override path:
    the override addresses the term's *constructor keyword*, and
    ``SpeedyConvection`` takes ``convection_params=`` while storing it as
    ``params``.

    ``nnx.Param`` variables are recorded in full, since a declared
    differentiable parameter is a tuned quantity even when it is an array
    (the MACv2-SP plume shapes, a per-level profile). Plain
    ``nnx.Variable``s are recorded only where they are knob-shaped: a
    parameter block containing a bool cannot be an ``nnx.Param``, so the
    schemes hold those as plain variables (Held-Suarez's whole tuning set,
    ``SpeedySurfaceFlux.surface_params``, ``EchamSurface.params``), but so
    are the terms' cached coordinates, and all eleven SPEEDY terms cache
    the same vertical grid.

    Arrays over ``_PARAM_ARRAY_MAX_ELEMS`` elements are summarized by
    shape, dtype and hash, which keeps embedded NN weights out of a
    netCDF attribute while still telling one weight set from another.
    Values captured under ``jit`` or ``grad`` read ``"<traced>"``.
    """
    if physics is None:
        return {}
    try:
        from flax import nnx
    except ImportError:
        return {}

    out: dict = {}
    seen: dict = {}
    for term in getattr(physics, "terms", None) or [physics]:
        name = getattr(term, "name", None) or type(term).__name__
        # A composition may hold two instances of one term (e.g. a
        # double-call radiation A/B); keep both rather than clobbering.
        seen[name] = seen.get(name, -1) + 1
        if seen[name]:
            name = f"{name}#{seen[name]}"
        try:
            # to_flat_state, not the State.flat_state() method: the latter
            # is deprecated and warns once per call. Present since flax
            # 0.12.1, which requirements.txt already floors us to.
            # nnx.Param is a Variable subclass, so `declared` is a subset
            # of `flat`; the difference is what gets the knob filter.
            flat = nnx.to_flat_state(nnx.state(term, nnx.Variable))
            declared = {tuple(path) for path, _ in
                        nnx.to_flat_state(nnx.state(term, nnx.Param))}
        except Exception:  # noqa: BLE001 — a non-nnx term has no params
            continue
        for path, var in flat:
            key = ".".join(str(p) for p in (name, *path))
            _describe_value(var.get_value(), key, out,
                            knobs_only=tuple(path) not in declared)
    return out


def params_attrs(params: dict | None) -> dict:
    """netCDF-safe global attributes for a parameter record."""
    if not params:
        return {}
    blob = json.dumps(params, sort_keys=True, default=str)
    return {"jcm_prov_params": blob,
            "jcm_prov_params_sha":
                hashlib.sha256(blob.encode()).hexdigest()[:12]}


def read_params(attrs) -> dict:
    """Recover a parameter record from a dataset's global attributes.

    Returns ``{}`` when the file carries no parameter record (anything
    written before #732).
    """
    blob = attrs.get("jcm_prov_params")
    return json.loads(blob) if blob else {}


def start_run(cfg=None) -> None:
    """Reset the registries and capture the config at run start.

    Code and environment are probed lazily (and re-probed on every
    :func:`collect`): the model build imports configuration-selected
    libraries (pyses, mam4_jax, …) and can flip the live x64 setting
    *after* run start, so an eager snapshot here would miss them.
    """
    _state["base"] = _new_base(cfg)
    _state["inputs"] = {}
    _state["facts"] = {}


def _new_base(cfg=None) -> dict:
    """Return the ``{"created", "config_yaml"}`` header of a new record."""
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
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_yaml": config_yaml,
    }


def _ensure_record() -> None:
    """Open the record if none is open, without touching the registries.

    A model built from Python never calls :func:`start_run`, yet the build
    already records inputs and facts. Whatever opens the record implicitly
    must therefore keep what was recorded before it; only :func:`start_run`
    empties the registries. ``created`` is the time of the first thing
    recorded (or of the first :func:`collect`, whichever comes first).
    """
    if _state["base"] is None:
        _state["base"] = _new_base()


def _mirror_revision_facts() -> dict:
    """Return the data-mirror commit this process reads, and where it came from.

    One commit per process (:func:`jcm.data.remote.mirror_revision`), recorded
    for every run: the same code and config at another commit read different
    inputs. An override that is not a commit sha has already failed any
    mirror read; it is recorded as given rather than failing the output.
    """
    from jcm.data import remote
    try:
        revision = remote.mirror_revision()
    except ValueError:
        revision = "invalid: " + os.environ.get(remote.REVISION_ENV, "")
    return {"data_mirror_revision": revision,
            "data_mirror_revision_source": remote.revision_source()}


def record_input(requested, resolved=None, *, force_hash: bool = False) -> None:
    """Record a boundary/input file the run resolved (deduped by path).

    ``force_hash=True`` records a content sha256 for *this* input even
    without ``JCM_HASH_INPUTS=1``. Used for the ``init=from_state`` warm
    start (:func:`jcm.runners._preflight_from_state_init`, #937, #1112): the
    initial state is one file (not the whole boundary-condition set), and
    unlike size + mtime its content digest survives a copy to another
    machine — which is exactly how a run gets ingested into monitoring
    away from where it ran.

    A path already recorded WITH a digest keeps it: a later unhashed call
    (``_resolve_data_path`` records every path it resolves, unhashed, and
    ``init.file`` is resolved more than once per process — once in the
    preflight that force-hashes it, again inside ``_state_from_file``)
    must not silently downgrade the entry back to ``(size, mtime)``, which
    would both drop the digest from ``run_hash`` and pay the streamed-hash
    cost a second time for nothing (#1112).
    """
    resolved = str(resolved if resolved is not None else requested)
    if not os.path.isfile(resolved):
        return
    existing = _state["inputs"].get(resolved)
    if existing is not None and "sha256" in existing and not force_hash:
        return
    entry = describe_input(resolved, force_hash=force_hash)
    if str(requested) != resolved:
        entry["requested"] = str(requested)
    _ensure_record()
    _state["inputs"][resolved] = entry


def record_fact(key: str, value) -> None:
    """Record a run-shaping fact (e.g. ozone_source = prescribed/analytic)."""
    _ensure_record()
    _state["facts"][key] = value


def collect(params: dict | None = None) -> dict:
    """Return the full provenance record.

    Code and environment are probed fresh each call (a handful of git
    subprocesses — negligible next to a netCDF write), so libraries the
    model build imported lazily and any post-start x64 flip are captured.

    *params* is the record from :func:`describe_params`, carried by the
    predictions object rather than by this module's registry.
    """
    _ensure_record()
    prov = dict(_state["base"])
    prov["code"] = probe_code()
    prov["environment"] = probe_environment()
    prov["inputs"] = dict(_state["inputs"])
    prov["facts"] = {**_mirror_revision_facts(), **_state["facts"]}
    if params:
        prov["params"] = params
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
        # The mirror commit decides the content of every hf:// input, and an
        # input described only by size + mtime would not separate two.
        "data_mirror_revision": prov["facts"]["data_mirror_revision"],
        # Without this every member of a parameter sweep shares one run
        # hash: the config, code and inputs are identical across them and
        # the parameters are the only thing that differs.
        "params": params or None,
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
        for k, v in code.items() if v.get("sha"))
    ozone = _state["facts"].get("ozone_source", "?")
    return (f"{shas or 'no git trees'}; "
            f"x64={env.get('jax_enable_x64')}; "
            f"{env.get('device_count')}x{env.get('device_kind')}; "
            f"ozone={ozone}")


def attrs(params: dict | None = None) -> dict:
    """Flat netCDF-safe global attributes (nested parts as JSON strings).

    *params* is optional because the parameter attributes are normally
    stamped earlier, by ``ModelPredictions.to_xarray``; pass it here only
    when building attributes for a dataset that did not come through
    there.
    """
    prov = collect(params)
    out = {
        "jcm_prov_created": prov["created"],
        # attrs() runs at output time, so this stamps the write — with
        # ``created`` it brackets the run without a separate end hook.
        "jcm_prov_written": datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        "jcm_prov_run_hash": prov["run_hash"],
        "jcm_prov_code": json.dumps(prov["code"], sort_keys=True,
                                    default=str),
        "jcm_prov_environment": json.dumps(prov["environment"],
                                           sort_keys=True, default=str),
        "jcm_prov_inputs": json.dumps(prov["inputs"], sort_keys=True,
                                      default=str),
    }
    out.update(params_attrs(params))
    for key, value in prov["facts"].items():
        out[f"jcm_prov_{key}"] = str(value)
    if prov["config_yaml"] is not None:
        # The composed config is too large for attributes (it goes in the
        # sidecar); the hash makes "same config" checkable from attrs.
        out["jcm_prov_config_sha"] = hashlib.sha256(
            prov["config_yaml"].encode()).hexdigest()[:12]
    return out


def write_sidecar(output_path, params: dict | None = None) -> Path:
    """Write ``<output>.provenance.json`` with the full record + config.

    Pass the same *params* given to :func:`attrs` for the same file, or
    the sidecar's ``run_hash`` will not match the one in the attributes.
    """
    output_path = Path(output_path)
    sidecar = output_path.with_suffix(output_path.suffix +
                                      ".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(collect(params), indent=1, sort_keys=True,
                                  default=str))
    return sidecar
