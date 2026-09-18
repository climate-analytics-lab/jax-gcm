"""Model state checkpointing for long, preemptible runs.

Persists ``Model.dycore_state`` and ``Model.physics_carry``
plus an elapsed sim-day count to a single file using flax's msgpack
serialization. ``run_chunked`` (in :mod:`jcm.runners`) integrates with
these primitives via ``cfg.run.checkpoint_path`` — when set, it writes a
checkpoint after each chunk and restores from one at startup if the file
exists, so an integration interrupted by spot-instance preemption resumes
without redoing completed chunks.

The state pytrees are flattened to plain arrays before serialization
because flax's msgpack codec can't handle ``tree_math`` structs (e.g.
dinosaur's ``primitive_equations.State``) directly. Each array is stored
under a *name* derived from its position in the pytree (``tracers.qc``,
``radiation.lw_flux_up``), and the file carries a ``schema_version``
stamp plus the metadata a migration needs. Restoring rebuilds the
pytrees from the destination model's bootstrapped templates and matches
the file's arrays to them **by name**, so a jcm upgrade that adds or
removes a diagnostic field in the physics carry no longer invalidates a
checkpoint (issue #731).

What migrates automatically, what is refused, and how to bump the schema
for a future change are the checkpoint compatibility policy:
``docs/source/design/checkpoint_compatibility.md``.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

import flax.serialization
import jax
import numpy as np


logger = logging.getLogger(__name__)


#: On-disk layout version. Bump when a change to the *meaning* of the
#: stored values needs a migration on load; see the design doc above for
#: the checklist. Files written before the stamp existed read as schema 0.
#:
#: 1 — named arrays, ``jcm`` version, physics-carry struct fields and the
#:     dycore tracer set; every mass mixing ratio stored as the physical
#:     kg/kg value (the contract PR #824 settled).
SCHEMA_VERSION = 1

#: Schema of a file with no stamp: anything written before this policy.
_UNSTAMPED_SCHEMA = 0

_POLICY_DOC = "docs/source/design/checkpoint_compatibility.md"


def _jcm_version() -> str:
    """Installed jcm version, or ``"unknown"`` outside an installed tree."""
    try:
        return metadata.version("jcm")
    except Exception:  # pragma: no cover - source tree without metadata
        return "unknown"


def _child_field_names(node) -> list[str] | None:
    """Ordered names of a dataclass pytree node's children, else ``None``.

    ``tree_math.struct`` registers its structs with
    ``register_pytree_node`` (no key paths), so ``tree_flatten_with_path``
    labels their children by position. The children are the dataclass
    fields in declaration order, which is what lets a positional path be
    recovered as a field *name* — the whole basis of name matching here.
    ``flax.struct`` fields marked ``pytree_node=False`` are static aux
    data rather than children and are skipped.
    """
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return None
    return [
        field.name
        for field in dataclasses.fields(node)
        if field.metadata.get("pytree_node", True)
    ]


def _leaf_name(root, path) -> str:
    """Dotted name for one ``tree_flatten_with_path`` key path.

    Walks ``root`` alongside the path so an unnamed (positional) node can
    be resolved to its dataclass field name. A node that cannot be named
    keeps its index as ``#i`` — stable, but positional.
    """
    parts: list[str] = []
    node = root
    for key in path:
        if isinstance(key, jax.tree_util.DictKey):
            parts.append(str(key.key))
            node = node[key.key] if node is not None else None
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(f"[{key.idx}]")
            node = node[key.idx] if node is not None else None
        elif isinstance(key, jax.tree_util.GetAttrKey):
            parts.append(key.name)
            node = getattr(node, key.name, None)
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            names = _child_field_names(node)
            if names is not None and key.key < len(names):
                parts.append(names[key.key])
                node = getattr(node, names[key.key], None)
            else:
                parts.append(f"#{key.key}")
                node = None
        else:  # pragma: no cover - jax adds a key type
            parts.append(str(key))
            node = None
    name = ""
    for part in parts:
        if part.startswith("["):
            name += part
        else:
            name = f"{name}.{part}" if name else part
    return name or "<leaf>"


def _named_leaves(tree) -> list[tuple[str, np.ndarray]]:
    """``(name, array)`` for every leaf, in ``tree_leaves`` order.

    The order matters: it is the order ``tree_unflatten`` consumes, and it
    is the positional correspondence an unstamped file is read with.
    """
    named = [
        (_leaf_name(tree, path), np.asarray(leaf))
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    ]
    counts = collections.Counter(name for name, _ in named)
    duplicates = sorted(name for name, n in counts.items() if n > 1)
    if duplicates:
        # Name matching is only sound while names are unique. A duplicate
        # means a container this module cannot name (see ``_leaf_name``),
        # and silently matching the wrong array would corrupt a restart.
        raise ValueError(
            "Cannot name every checkpoint leaf uniquely; duplicated "
            f"name(s): {duplicates}. The state pytree contains a node "
            f"this version cannot label — see {_POLICY_DOC}."
        )
    return named


def _struct_fields(node, prefix: str = "", out: dict | None = None) -> dict:
    """Map each struct/group node's path to its ordered child names.

    Recorded in the file as migration metadata: it says which fields each
    physics-carry struct (and each dict group, e.g. ``_prev_step``) held
    when the checkpoint was written, including fields that contributed no
    leaf at all — an empty container — which the array names alone cannot
    show. The root mapping is recorded as ``<root>``.
    """
    if out is None:
        out = {}
    names = _child_field_names(node)
    if names is not None:
        out[prefix or "<root>"] = names
        for name in names:
            child = f"{prefix}.{name}" if prefix else name
            _struct_fields(getattr(node, name, None), child, out)
    elif isinstance(node, Mapping):
        out[prefix or "<root>"] = sorted(str(key) for key in node)
        for key in sorted(node, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            _struct_fields(node[key], child, out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _struct_fields(value, f"{prefix}[{index}]", out)
    return out


def _dycore_tracers(model) -> dict[str, bool]:
    """``name -> nondimensionalize`` for the tracers in the saved state.

    The flag is what a unit migration keys on: a
    ``nondimensionalize=True`` tracer is a kg/kg mass mixing ratio stored
    unscaled, while ``False`` (number concentrations, VMRs) passes the
    dycore boundary untouched. Names come from the state actually being
    written; the flag from the composition's ``TracerSpec``s, defaulting
    to ``True`` for a tracer with no spec — ``specific_humidity``, which
    every dycore carries without declaring.
    """
    specs = getattr(getattr(model, "dycore", None), "tracer_specs", None) or {}
    tracers = getattr(model.dycore_state, "tracers", None)
    names = tracers if isinstance(tracers, Mapping) else specs
    return {
        str(name): bool(getattr(specs.get(name), "nondimensionalize", True))
        for name in names
    }


def parse_unstamped_scale(value) -> dict[str, float] | None:
    """Coerce a config value into ``load_checkpoint``'s ``unstamped_scale``.

    Accepts ``None`` (refuse unstamped files — the default), a mapping, or
    a sequence of ``"name=factor"`` strings. The string form exists
    because the leaf names contain dots (``tracers.qc``), which a Hydra
    override cannot express as dictionary keys.

    Args:
        value: ``None``, a ``{name: factor}`` mapping, or a sequence of
            ``"name=factor"`` strings (an empty sequence or mapping means
            "the file is already in the current convention").

    Returns:
        The mapping, or ``None``.

    """
    if value is None:
        return None
    # Duck-typed rather than ``isinstance(value, Mapping)``: an OmegaConf
    # ``DictConfig`` from a Hydra config is mapping-shaped either way.
    if hasattr(value, "items"):
        return {str(name): float(factor) for name, factor in value.items()}
    if isinstance(value, str):
        raise ValueError(
            "unstamped_scale must be a mapping or a sequence of "
            f"'name=factor' entries, not the bare string {value!r}."
        )
    scale = {}
    for entry in value:
        name, _, factor = str(entry).partition("=")
        if not name or not factor:
            raise ValueError(
                f"unstamped_scale entry {entry!r} is not 'name=factor'."
            )
        try:
            scale[name.strip()] = float(factor)
        except ValueError as exc:
            raise ValueError(
                f"unstamped_scale entry {entry!r} has a non-numeric factor."
            ) from exc
    return scale


def save_checkpoint(model, path, *, elapsed_days: float) -> Path:
    """Persist the model's current dycore + physics state to ``path``.

    Writes schema ``SCHEMA_VERSION``: every state array under its pytree
    name, the ``jcm`` version, the physics-carry struct fields and the
    dycore tracer set (with each tracer's ``nondimensionalize`` flag), so
    a later jcm can migrate the file rather than only reject it.

    Args:
        model: A ``jcm.model.Model`` whose ``dycore_state`` and
            ``physics_carry`` have been populated, either by a
            prior ``run`` / ``resume`` call or by ``bootstrap_state``.
        path: Output file path (parent directories are created).
        elapsed_days: Sim-day count to record alongside the state so a
            chunked driver can resume at the correct offset.

    Returns:
        ``Path(path)`` for chaining.

    """
    if model.dycore_state is None or model.physics_carry is None:
        raise ValueError(
            "Model has no state to checkpoint — call Model.run(...), "
            "Model.resume(...), or Model.bootstrap_state(...) first."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "jcm_version": _jcm_version(),
        "elapsed_days": float(elapsed_days),
        "dycore": dict(_named_leaves(model.dycore_state)),
        "physics": dict(_named_leaves(model.physics_carry)),
        "physics_fields": _struct_fields(model.physics_carry),
        "dycore_tracers": _dycore_tracers(model),
    }
    # Write to a sibling tmp file then rename atomically. If the run is
    # killed mid-write (the whole point of checkpointing for preemptible
    # workloads), the previous checkpoint is left intact rather than
    # truncated to a half-serialized blob that would fail to load.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(flax.serialization.to_bytes(payload))
    tmp_path.replace(path)
    return path


def _incompatible_dtype(got: np.ndarray, want: np.ndarray) -> bool:
    """Report whether ``got`` needs a real cast to stand in for ``want``.

    A different kind (float vs int vs bool) is always a mismatch. A
    different float width is one too for an array — a float64 pySES state
    is not a float32 state — but not for a scalar: a bootstrapped
    template's ``sim_time`` is a plain Python float (float64) where a run
    leaves a float32 0-d array, and that pairing has always round-tripped.
    """
    if got.dtype.kind != want.dtype.kind:
        return True
    return want.ndim > 0 and got.dtype != want.dtype


def _check_leaf(got: np.ndarray, want: np.ndarray, name: str, path, group: str):
    """Reject a same-named leaf whose shape or dtype cannot be restored."""
    if got.shape != want.shape:
        raise ValueError(
            f"Checkpoint {path} does not match the composed model: "
            f"{group} {name!r} has shape {got.shape}, model expects "
            f"{want.shape} (wrong grid/levels/physics for this file)."
        )
    if _incompatible_dtype(got, want):
        raise ValueError(
            f"Checkpoint {path} does not match the composed model: "
            f"{group} {name!r} has dtype {got.dtype}, model expects "
            f"{want.dtype}. Precision changes are not migrated — see "
            f"{_POLICY_DOC}."
        )


def _match_by_name(
    stored: Mapping,
    template: list[tuple[str, np.ndarray]],
    *,
    path,
    group: str,
    fill_missing: bool,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Order the file's arrays to the template, matching on name.

    Returns ``(leaves, seeded, dropped)``: the leaves in template order,
    the template names the file had no array for, and the file names this
    model has no leaf for.

    ``fill_missing`` is the forward-migration switch. The physics carry
    sets it: a field a newer jcm added takes the freshly bootstrapped
    template's value. The dycore state does not — its leaves are the
    prognostic state, which is never invented.
    """
    leaves: list[np.ndarray] = []
    seeded: list[str] = []
    template_names = {name for name, _ in template}
    dropped = sorted(str(name) for name in stored if str(name) not in template_names)
    for name, want in template:
        if name not in stored:
            if not fill_missing:
                raise ValueError(
                    f"Checkpoint {path} does not match the composed model: "
                    f"{group} has no stored {name!r}. It was written by a "
                    "different physics composition or dycore backend, whose "
                    f"prognostic state this model cannot reconstruct — see "
                    f"{_POLICY_DOC}."
                )
            seeded.append(name)
            leaves.append(want)
            continue
        got = np.asarray(stored[name])
        _check_leaf(got, want, name, path, group)
        leaves.append(got.astype(want.dtype) if got.dtype != want.dtype else got)
    if dropped and not fill_missing:
        raise ValueError(
            f"Checkpoint {path} does not match the composed model: "
            f"{group} stores {dropped} which this model does not carry. "
            "Dropping prognostic state would silently change the run — see "
            f"{_POLICY_DOC}."
        )
    return leaves, seeded, dropped


def _ordered_legacy(group, path, key: str) -> list[np.ndarray]:
    """Read one unstamped group, which stored a plain list of arrays.

    ``flax.serialization.to_bytes`` turns a list into a dict keyed by the
    stringified index, so the order has to be recovered numerically.
    """
    if not isinstance(group, Mapping):
        raise ValueError(
            f"Checkpoint {path} is not a jcm checkpoint: {key!r} is "
            f"{type(group).__name__}, expected a group of arrays."
        )
    try:
        return [np.asarray(group[k]) for k in sorted(group, key=lambda k: int(k))]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint {path} is not a jcm checkpoint: {key!r} is not "
            f"an indexed group of arrays ({exc})."
        ) from exc


def _load_unstamped(
    raw: Mapping,
    path,
    dycore_template: list[tuple[str, np.ndarray]],
    physics_template: list[tuple[str, np.ndarray]],
    unstamped_scale: Mapping[str, float] | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read a pre-stamp checkpoint, only against an explicit scale assertion.

    An unstamped file records neither its schema nor the physics package
    that wrote it, and PR #824 changed what the dycore state's mass
    mixing-ratio arrays *mean* (see the policy doc). The correct factor
    per array is therefore not derivable from the file or from this
    model, so the default is to refuse. ``unstamped_scale`` is the
    caller's explicit statement of the file's convention: ``{}`` asserts
    "already current", and a ``{name: factor}`` mapping rescales exactly
    those arrays.
    """
    if unstamped_scale is None:
        raise ValueError(
            f"Checkpoint {path} carries no schema_version stamp, so it was "
            "written before the checkpoint compatibility policy (jcm 3.0). "
            "Such a file does not record which unit convention its dycore "
            "state uses: PR #824 made every mass mixing-ratio tracer cross "
            "the dycore boundary as the physical kg/kg value, and before "
            "that the stored scale depended on the physics package, so "
            "these values cannot be interpreted safely. Start the run from "
            "a fresh initial state, or — if you know how this file was "
            "written — assert its convention explicitly with "
            "load_checkpoint(..., unstamped_scale={'tracers.qc': 1000.0, "
            f"...}}) (use {{}} for a file already in the current "
            f"convention). See {_POLICY_DOC}."
        )
    names = {name for name, _ in dycore_template} | {
        name for name, _ in physics_template
    }
    unknown = sorted(str(k) for k in unstamped_scale if str(k) not in names)
    if unknown:
        raise ValueError(
            f"unstamped_scale names {unknown}, which are not leaves of this "
            "model's state. Nothing was loaded rather than silently "
            f"ignoring the factor. Known names: {sorted(names)[:8]}..."
        )

    groups = []
    for key, template, group_label in (
        ("dycore_leaves", dycore_template, "dycore state"),
        ("physics_leaves", physics_template, "physics carry"),
    ):
        if key not in raw:
            raise ValueError(
                f"Checkpoint {path} carries no schema_version stamp and no "
                f"{key!r} group either, so it is not a jcm checkpoint."
            )
        stored = _ordered_legacy(raw[key], path, key)
        if len(stored) != len(template):
            raise ValueError(
                f"Checkpoint {path} does not match the composed model: "
                f"{group_label} stores {len(stored)} arrays, model expects "
                f"{len(template)}. An unstamped file carries no field names, "
                "so a structural difference cannot be migrated — see "
                f"{_POLICY_DOC}."
            )
        leaves = []
        for (name, want), got in zip(template, stored):
            _check_leaf(got, want, name, path, group_label)
            factor = unstamped_scale.get(name)
            if factor is not None:
                got = got * np.asarray(factor, dtype=want.dtype)
                logger.info(
                    "Checkpoint %s: scaled unstamped %s %r by %g "
                    "(caller-asserted unit convention)",
                    path, group_label, name, float(factor),
                )
            leaves.append(
                got.astype(want.dtype) if got.dtype != want.dtype else got
            )
        groups.append(leaves)
    logger.info(
        "Checkpoint %s: read as unstamped (schema 0) against a "
        "caller-asserted scale for %d leaf/leaves",
        path, len(unstamped_scale),
    )
    return groups[0], groups[1]


def load_checkpoint(model, path, *, unstamped_scale=None) -> float:
    """Restore ``dycore_state`` + ``physics_carry`` from ``path``.

    The model must already have been bootstrapped (e.g. by an earlier
    ``Model.run``, ``Model.bootstrap_state``, or one of the initial-state
    builders in :mod:`jcm.initial_states`) so that its state pytrees provide
    the per-leaf names, shapes and dtypes the file is matched against.

    Leaves are matched **by name**, so an upgrade that adds or removes a
    field on a physics-carry struct still restores: a field the file does
    not have takes the freshly bootstrapped model's value for it (the
    term's documented seed — ``.zeros()`` for most carry slots), and a
    field the file has but this model does not is dropped. Both are
    logged at INFO on the ``jcm.checkpoint`` logger so a resume is
    auditable. Carry fields are diagnostics recomputed within a step or
    two, with one bounded exception: a radiation sub-cycle cache seeded
    this way starts one radiation interval stale, which is accepted.

    A leaf present in both whose shape or dtype differs is an error
    naming the file and the leaf — that is a different grid, level count
    or precision, not a field-set change. The dycore state (the
    prognostic state) is never filled or dropped: any name difference
    there is refused.

    Args:
        model: A ``jcm.model.Model`` with populated final states to use
            as restore templates. Their values are overwritten.
        path: Checkpoint file path written by :func:`save_checkpoint`.
        unstamped_scale: Only for a file with no ``schema_version`` stamp
            (anything written before jcm 3.0), which is otherwise
            refused: a ``{leaf_name: factor}`` mapping asserting that
            file's unit convention, or ``{}`` to assert it needs no
            rescale. See the compatibility policy in
            ``docs/source/design/checkpoint_compatibility.md``.

    Returns:
        The ``elapsed_days`` count recorded when the checkpoint was
        saved.

    """
    if model.dycore_state is None or model.physics_carry is None:
        raise ValueError(
            "Model state is uninitialised — call Model.bootstrap_state(...) "
            "before load_checkpoint so the destination has templates to "
            "rebuild the pytrees from."
        )
    path = Path(path)
    try:
        raw = flax.serialization.msgpack_restore(path.read_bytes())
    except Exception as exc:
        raise ValueError(
            f"Checkpoint {path} could not be decoded as a flax msgpack "
            f"payload: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Checkpoint {path} is not a jcm checkpoint (decoded as "
            f"{type(raw).__name__}, expected a mapping)."
        )

    if "elapsed_days" not in raw:
        raise ValueError(
            f"Checkpoint {path} is not a jcm checkpoint: it records no "
            "'elapsed_days'."
        )

    try:
        schema = int(np.asarray(raw.get("schema_version", _UNSTAMPED_SCHEMA)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint {path} is not a jcm checkpoint: its "
            f"'schema_version' is not an integer ({exc})."
        ) from exc
    # Accept the config-facing forms too, so a caller threading a value
    # through from Hydra cannot trip over a list where a mapping is read.
    unstamped_scale = parse_unstamped_scale(unstamped_scale)
    if schema > SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint {path} was written with schema {schema} by jcm "
            f"{raw.get('jcm_version', 'unknown')}; this build reads up to "
            f"schema {SCHEMA_VERSION}. Upgrade jcm to resume this file — a "
            "newer schema may store values this version would misread. See "
            f"{_POLICY_DOC}."
        )

    dycore_template = _named_leaves(model.dycore_state)
    physics_template = _named_leaves(model.physics_carry)

    if schema == _UNSTAMPED_SCHEMA:
        dycore_leaves, physics_leaves = _load_unstamped(
            raw, path, dycore_template, physics_template, unstamped_scale,
        )
    else:
        if unstamped_scale is not None:
            raise ValueError(
                f"unstamped_scale was given for {path}, which is stamped "
                f"schema {schema}: its unit convention is recorded, so a "
                "caller-asserted rescale would corrupt it. Drop the "
                "argument."
            )
        for key in ("dycore", "physics"):
            if not isinstance(raw.get(key), Mapping):
                raise ValueError(
                    f"Checkpoint {path} claims schema {schema} but has no "
                    f"{key!r} group of named arrays."
                )
        dycore_leaves, _, _ = _match_by_name(
            raw["dycore"], dycore_template,
            path=path, group="dycore state", fill_missing=False,
        )
        physics_leaves, seeded, dropped = _match_by_name(
            raw["physics"], physics_template,
            path=path, group="physics carry", fill_missing=True,
        )
        if seeded:
            logger.info(
                "Checkpoint %s: physics carry field(s) %s absent from the "
                "file; seeded from this model's fresh carry",
                path, seeded,
            )
        if dropped:
            logger.info(
                "Checkpoint %s: stored physics carry field(s) %s are not "
                "carried by this model; dropped",
                path, dropped,
            )
        logger.info(
            "Checkpoint %s: restored schema %d written by jcm %s "
            "(%d carry field(s) seeded, %d dropped)",
            path, schema, raw.get("jcm_version", "unknown"),
            len(seeded), len(dropped),
        )

    _, dycore_treedef = jax.tree_util.tree_flatten(model.dycore_state)
    _, physics_treedef = jax.tree_util.tree_flatten(model.physics_carry)
    restored_dycore_state = jax.tree_util.tree_unflatten(
        dycore_treedef, dycore_leaves)
    restored_physics_carry = jax.tree_util.tree_unflatten(
        physics_treedef, physics_leaves)
    model.restore_state(restored_dycore_state, restored_physics_carry)
    return float(np.asarray(raw["elapsed_days"]))
