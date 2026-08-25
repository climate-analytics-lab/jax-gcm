"""Run provenance capture (#591).

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

Parameters are captured separately from the config and by a different
route, because the config does not determine them. A scheme's
``params`` block is deliberately absent from the shipped yamls so each
field falls back to ``Parameters.default()`` in code, so the composed
config records the *overrides* and says nothing about the effective
values; and a model built in Python, or one whose parameters were
replaced after construction (what a calibration loop does), has no
config behind it at all. :func:`describe_params` therefore reads the
built objects. It is called at the model-to-user handoff — see
:class:`jcm.predictions.ModelPredictions` — because that is the last
moment the values are still the ones that produced the trajectory.

Lifecycle: :func:`start_run` at run start (captures code/env/config and
resets the input registry), :func:`record_input` / :func:`record_fact`
during model build, :func:`attrs` / :func:`write_sidecar` at output time
(inputs and the run hash are finalized lazily, after the build has
touched every file). The parameter record travels on the predictions
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
#: Cap on the parameter JSON carried *in netCDF attributes*. The sidecar
#: is unconditional; a record over the cap keeps its hash in the
#: attributes and points at the sidecar for the values.
_PARAM_ATTR_MAX_CHARS = 64_000

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
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001 — anything unconvertible is described
        return repr(value)[:200]
    if arr.dtype == object:
        return repr(value)[:200]
    if arr.ndim == 0:
        # ``item()`` on a float32 widens to the exact float64 that value
        # represents (0.1 -> 0.10000000149011612). That is the number the
        # model used, so record it rather than a prettier rounding.
        return arr.item()
    if arr.size <= _PARAM_ARRAY_MAX_ELEMS:
        return arr.ravel().tolist()
    return {"shape": list(arr.shape), "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]}


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _describe_value(value, prefix: str, out: dict, depth: int = 0) -> None:
    """Flatten *value* into *out* under the dotted key *prefix*."""
    import dataclasses

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
                            out, depth + 1)
        return
    if hasattr(value, "_fields") and isinstance(value, tuple):  # NamedTuple
        for name in value._fields:
            _describe_value(getattr(value, name), f"{prefix}.{name}",
                            out, depth + 1)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _describe_value(v, f"{prefix}.{k}", out, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        # A short all-scalar sequence is one value (a per-level profile);
        # anything else is a container to walk.
        if all(_is_scalar(v) for v in value):
            if len(value) <= _PARAM_ARRAY_MAX_ELEMS:
                out[prefix] = list(value)
            else:
                out[prefix] = f"<{len(value)} scalars>"
            return
        for i, v in enumerate(value):
            _describe_value(v, f"{prefix}.{i}", out, depth + 1)
        return
    out[prefix] = _describe_leaf(value)


def _describe_physics_params(physics) -> dict:
    """Every ``nnx.Param`` on every physics term, by dotted name.

    Keyed on the term's own ``name`` (``tiedtke_convection``) and the
    variable it hangs off: ``tiedtke_convection.params.entrpen``.

    That is close to, but not always identical to, the Hydra override
    path minus its ``physics.terms.`` prefix. Hydra addresses the term's
    *constructor keyword*, which for the ECHAM terms is also the variable
    name but for the SPEEDY ones is not (``SpeedyConvection`` takes
    ``convection_params=`` and stores it as ``params``, so the override
    is ``...speedy_convection.convection_params.entmax`` while the record
    says ``...speedy_convection.params.entmax``). The record names where
    the value *lives*, which is the thing that must be unambiguous;
    reproducing it may need the constructor signature.

    Keying on ``nnx.Param`` rather than a ``.params`` attribute is
    deliberate — terms also carry ``mod_radcon_params``, ``sw_params``,
    ``surface_optics`` and bare tuning scalars, and a scheme that adds
    another must not silently drop out of the record.
    """
    if physics is None:
        return {}
    try:
        from flax import nnx
    except ImportError:
        return {}

    terms = getattr(physics, "terms", None)
    modules = list(terms) if terms is not None else [physics]
    out: dict = {}
    seen: dict = {}
    for term in modules:
        name = getattr(term, "name", None) or type(term).__name__
        # A composition may hold two instances of one term (e.g. a
        # double-call radiation A/B); keep both rather than clobbering.
        if name in seen:
            seen[name] += 1
            name = f"{name}#{seen[name]}"
        else:
            seen[name] = 0
        try:
            # to_flat_state, not the State.flat_state() method: the latter
            # is deprecated and warns once per call. Present since flax
            # 0.12.1, which requirements.txt already floors us to.
            flat = nnx.to_flat_state(nnx.state(term, nnx.Param))
        except Exception:  # noqa: BLE001 — a non-nnx term has no params
            continue
        for path, var in flat:
            key = ".".join(str(p) for p in (name, *path))
            _describe_value(var.get_value(), key, out)
    return out


def _all_scalar_leaves(value, depth: int = 0) -> bool:
    """Report whether *value* is built purely from scalars.

    Screens a dycore attribute *before* describing it, so ``coords`` and
    ``terrain`` are rejected on the first array field rather than having
    their grids pulled off the device to be summarized.
    """
    import dataclasses

    if _is_scalar(value):
        return True
    if depth > _PARAM_MAX_DEPTH:
        return False
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return all(_all_scalar_leaves(getattr(value, f.name), depth + 1)
                   for f in dataclasses.fields(value))
    if isinstance(value, dict):
        return all(_all_scalar_leaves(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_scalar_leaves(v, depth + 1) for v in value)
    return False


def _describe_dycore_params(dycore) -> dict:
    """Scalar dycore knobs: timestep, diffusion filter, transport options.

    A dycore declares no parameters the way a physics term does (there is
    no ``nnx.Param`` to key on), so the rule here is structural: instance
    attributes that are scalars, or containers built purely from scalars.
    That keeps ``dt_seconds``, the ``DiffusionFilter`` timescales and
    orders, the ``compute_*`` flags and the semi-Lagrangian options, and
    drops ``coords`` / ``terrain`` / the precomputed vertical coefficients,
    whose arrays are grid data rather than knobs. Private attributes are
    included (``_sl_options`` is as much a knob as ``dt_seconds``); the
    leading underscore is kept so the key names a real attribute.
    """
    if dycore is None:
        return {}
    try:
        attributes = vars(dycore)
    except TypeError:  # a __slots__ backend exposes no instance __dict__
        return {}
    out: dict = {}
    for name, value in sorted(attributes.items()):
        if name.startswith("__") or name == "constants":
            continue  # constants get their own block
        if not _all_scalar_leaves(value):
            continue
        _describe_value(value, name, out)
    return out


def _describe_constants(dycore) -> dict:
    """Record the live physical constants, and the dycore's if they differ.

    ``jcm.constants`` is a process-global singleton read *live* by
    attribute-access physics but captured *at construction* by the dycore,
    so a ``set_constants`` call made after the model was built leaves the
    two genuinely disagreeing. Recording only the live values would hide
    that, so a differing dycore copy is recorded field by field.
    """
    out: dict = {}
    try:
        import jcm.constants as _c
        live = _c.physical_constants
    except Exception:  # noqa: BLE001 — never fail a run over provenance
        return out
    _describe_value(live, "constants", out)
    built = getattr(dycore, "constants", None)
    if built is None or built == live:
        return out
    built_fields: dict = {}
    _describe_value(built, "constants", built_fields)
    for key, value in built_fields.items():
        if out.get(key) != value:
            out[key.replace("constants", "constants_dycore", 1)] = value
    return out


def describe_params(physics=None, dycore=None) -> dict:
    """Return the parameter values a run actually used, by dotted key.

    Reads the *built* objects, not the requested config, because the two
    are not the same thing (see the module docstring): the shipped yamls
    omit each scheme's ``params`` block so the effective values live in
    ``Parameters.default()``, and a Python-built or post-hoc-modified
    model has no config at all.

    Returns a dict with ``physics`` (every ``nnx.Param`` on every term),
    ``dycore`` (scalar backend knobs) and ``constants`` blocks; each maps
    dotted names to JSON-safe values. Large arrays are summarized by
    shape/dtype/hash and values captured under ``jit``/``grad`` read
    ``"<traced>"``.
    """
    out: dict = {}
    physics_params = _describe_physics_params(physics)
    if physics_params:
        out["physics"] = physics_params
    dycore_params = _describe_dycore_params(dycore)
    if dycore_params:
        out["dycore"] = dycore_params
    constants = _describe_constants(dycore)
    if constants:
        out["constants"] = constants
    return out


def params_attrs(params: dict | None) -> dict:
    """netCDF-safe global attributes for a parameter record."""
    if not params:
        return {}
    blob = json.dumps(params, sort_keys=True, default=str)
    out = {"jcm_prov_params_sha":
           hashlib.sha256(blob.encode()).hexdigest()[:12]}
    if len(blob) <= _PARAM_ATTR_MAX_CHARS:
        out["jcm_prov_params"] = blob
    else:
        out["jcm_prov_params"] = (
            f"<{len(blob)} chars, over the {_PARAM_ATTR_MAX_CHARS}-char "
            "attribute cap; full record in the .provenance.json sidecar>")
    return out


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


def collect(params: dict | None = None) -> dict:
    """Return the full provenance record.

    Code and environment are probed fresh each call (a handful of git
    subprocesses — negligible next to a netCDF write), so libraries the
    model build imported lazily and any post-start x64 flip are captured.

    *params* is the record from :func:`describe_params`, carried by the
    predictions object rather than by this module's registry.
    """
    if _state["base"] is None:
        start_run()
    prov = dict(_state["base"])
    prov["code"] = probe_code()
    prov["environment"] = probe_environment()
    prov["inputs"] = dict(_state["inputs"])
    prov["facts"] = dict(_state["facts"])
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
        for k, v in code.items() if "sha" in v)
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
        "jcm_prov_code": json.dumps(prov["code"], sort_keys=True),
        "jcm_prov_environment": json.dumps(prov["environment"],
                                           sort_keys=True),
        "jcm_prov_inputs": json.dumps(prov["inputs"], sort_keys=True),
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
