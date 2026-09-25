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
import functools
import logging
import os
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

import flax.serialization
import jax
import jax_datetime as jdt
import numpy as np


logger = logging.getLogger(__name__)


#: On-disk layout version. Bump when a change to the *meaning* of the
#: stored values needs a migration on load; see the design doc above for
#: the checklist. Files written before the stamp existed read as schema 0.
#:
#: 1 — named arrays, ``jcm`` version, physics-carry struct fields and the
#:     dycore tracer set; every mass mixing ratio stored as the physical
#:     kg/kg value (the contract PR #824 settled).
# 2 — exact Gregorian run clock and its origin/timestep. Older states may
# be imported as initial conditions, but cannot resume a different season.
SCHEMA_VERSION = 2

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


def _prognostic_carry_slots(model) -> list[str]:
    """Carry keys the composition declares as prognostic state.

    These hold the only copy of a physical quantity (JAM's cloud-borne
    aerosol phase, #602), so the name migration must neither seed nor drop
    them — see :attr:`jcm.physics.physics_term.PhysicsTerm.prognostic_carry_slots`.
    """
    physics = getattr(model, "physics", None)
    declared = getattr(physics, "prognostic_carry_slots", None)
    if declared is None:
        return []
    return [str(key) for key in (declared() if callable(declared) else declared)]


def _mirror_revision() -> str:
    """Return the data-mirror commit this process reads (see jcm.data.remote)."""
    from jcm.data import remote
    try:
        return remote.mirror_revision()
    except ValueError:      # an invalid override: record it as given
        return "invalid: " + os.environ.get(remote.REVISION_ENV, "")


def save_checkpoint(model, path, *, elapsed_days: float | None = None) -> Path:
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
        elapsed_days: Optional consistency assertion; the saved count is
            derived from the exact model clock, in elapsed days.

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
    run_state = model.run_state
    if run_state is None:
        raise ValueError("Model has no exact run clock to checkpoint.")
    delta = run_state.time - model.start_time
    clock_elapsed = int(delta.days) + int(delta.seconds) / 86400.0
    if elapsed_days is not None and not np.isclose(
            float(elapsed_days), clock_elapsed, rtol=0, atol=1e-9):
        raise ValueError("elapsed_days does not match the model's exact run clock: "
                         f"{elapsed_days} vs {clock_elapsed}.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "jcm_version": _jcm_version(),
        "elapsed_days": clock_elapsed,
        "clock": {
            "start_days": np.asarray(model.start_time.delta.days),
            "start_seconds": np.asarray(model.start_time.delta.seconds),
            "days": np.asarray(run_state.time.delta.days),
            "seconds": np.asarray(run_state.time.delta.seconds),
            "step": np.asarray(run_state.step),
            "dt_seconds": float(model.dt_si.m),
        },
        # With the clock, the run's identity: the data-mirror commit its
        # inputs were read at, so a resume can refuse different inputs.
        "data_mirror_revision": _mirror_revision(),
        "dycore": dict(_named_leaves(model.dycore_state)),
        "physics": dict(_named_leaves(model.physics_carry)),
        "physics_fields": _struct_fields(model.physics_carry),
        "dycore_tracers": _dycore_tracers(model),
        # Recorded so a *reader* that no longer composes the owning term
        # still knows this file's carry held state nothing recomputes, and
        # refuses to drop it rather than migrating it away.
        "prognostic_carry_slots": _prognostic_carry_slots(model),
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


def _as_sequence(value) -> list:
    """Read back a stored list, which msgpack holds as an index-keyed map."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value[key] for key in sorted(value, key=str)]
    return list(value)


def _slot_of(name: str) -> str:
    """Top-level carry key a leaf name belongs to (``a.b.c`` -> ``a``)."""
    return name.split(".", 1)[0].split("[", 1)[0]


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


def _fresh_carry_seeds(model):
    """Build ``{name: array}`` from a genuinely fresh physics carry.

    ``Model.initial_physics_carry`` is a pure builder: it returns the
    per-term documented seed (``.zeros()`` for most slots, TTE-TKE's
    turbulence floor, …) without touching the model's own carry. That is
    what a field the checkpoint predates must be filled with — the
    destination model's *current* carry is an evolved state whenever it
    came from an earlier ``Model.run``, and injecting that into a
    restored run would change its results.
    """
    builder = getattr(model, "initial_physics_carry", None)
    if builder is None:  # pragma: no cover - non-Model host
        return {}
    return dict(_named_leaves(builder()))


def _seed_value(name, want, seeds, path, group):
    """Fill value for one absent leaf, from the fresh carry when possible."""
    fresh = seeds() if callable(seeds) else (seeds or {})
    value = fresh.get(name)
    if value is None or value.shape != want.shape or (
            value.dtype.kind != want.dtype.kind):
        # No fresh counterpart (a host without the builder, or a carry
        # this composition sizes differently): fall back to the template,
        # but say so — the value is then whatever the model held.
        logger.warning(
            "Checkpoint %s: %s %r has no fresh-carry seed; filled from the "
            "destination model's current value instead",
            path, group, name,
        )
        return want
    return value.astype(want.dtype) if value.dtype != want.dtype else value


def _match_by_name(
    stored: Mapping,
    template: list[tuple[str, np.ndarray]],
    *,
    path,
    group: str,
    fill_missing: bool,
    protected: frozenset[str] = frozenset(),
    seeds=None,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Order the file's arrays to the template, matching on name.

    Returns ``(leaves, seeded, dropped)``: the leaves in template order,
    the template names the file had no array for, and the file names this
    model has no leaf for.

    ``fill_missing`` is the forward-migration switch. The physics carry
    sets it: a field a newer jcm added takes the freshly bootstrapped
    template's value. The dycore state does not — its leaves are the
    prognostic state, which is never invented.

    ``protected`` names carry slots that are prognostic state despite
    living in the carry (the cloud-borne aerosol phase). A leaf under one
    of those is refused rather than seeded or dropped, whichever way the
    field sets differ, because nothing recomputes it.

    ``seeds`` supplies the fill values: a callable returning
    ``{name: array}`` from a *freshly built* carry, evaluated only if
    something actually has to be seeded. The template cannot serve that
    role — it is whatever the destination model currently holds, which
    for a model populated by an earlier ``Model.run`` is an evolved state
    from an unrelated integration, not the term's documented seed.
    """
    leaves: list[np.ndarray] = []
    seeded: list[str] = []
    template_names = {name for name, _ in template}
    dropped = sorted(str(name) for name in stored if str(name) not in template_names)
    for name, want in template:
        if name not in stored:
            if fill_missing and _slot_of(name) in protected:
                raise ValueError(
                    f"Checkpoint {path} does not match the composed model: "
                    f"{group} {name!r} is missing, and its slot "
                    f"{_slot_of(name)!r} holds prognostic state this model "
                    "cannot reconstruct — seeding it would invent physical "
                    "mass. Resume with a jcm that carries the same slot, or "
                    f"start from a fresh initial state. See {_POLICY_DOC}."
                )
            if not fill_missing:
                raise ValueError(
                    f"Checkpoint {path} does not match the composed model: "
                    f"{group} has no stored {name!r}. It was written by a "
                    "different physics composition or dycore backend, whose "
                    f"prognostic state this model cannot reconstruct — see "
                    f"{_POLICY_DOC}."
                )
            seeded.append(name)
            leaves.append(_seed_value(name, want, seeds, path, group))
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
    dropped_prognostic = [n for n in dropped if _slot_of(n) in protected]
    if dropped_prognostic:
        raise ValueError(
            f"Checkpoint {path} does not match the composed model: "
            f"{group} stores {dropped_prognostic}, whose slot holds "
            "prognostic state this model does not carry. Dropping it would "
            "silently destroy physical mass the file is the only record of. "
            f"See {_POLICY_DOC}."
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


def _mass_mixing_ratio_leaves(
    model, dycore_template: list[tuple[str, np.ndarray]],
) -> list[str]:
    """Dycore-state leaf names holding a kg/kg mass mixing ratio.

    These are the leaves PR #824's contract change applies to, so they are
    the ones a caller reading an unstamped file has to make a decision
    about. Derived from the composed model's tracer specs — a statement
    about the reader, offered as a starting point, never as an inference
    about what the file contains.
    """
    mixing_ratios = {name for name, nondim in _dycore_tracers(model).items()
                     if nondim}
    return [name for name, _ in dycore_template
            if name.rsplit(".", 1)[-1] in mixing_ratios]


def _load_unstamped(
    raw: Mapping,
    path,
    dycore_template: list[tuple[str, np.ndarray]],
    physics_template: list[tuple[str, np.ndarray]],
    unstamped_scale: Mapping[str, float] | None,
    candidates: list[str],
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
        shown = ", ".join(candidates[:12])
        if len(candidates) > 12:
            shown += f", ... ({len(candidates)} in total)"
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
            "load_checkpoint(..., unstamped_scale={...}), which takes a "
            "factor per leaf ({} asserts the file is already in the "
            "current convention). The leaves this model holds a mass "
            f"mixing ratio in are: {shown}. See {_POLICY_DOC}."
        )
    # One namespace across both groups. The leaves a unit assertion
    # applies to are the dycore state's ``tracers.*``, which no carry slot
    # shares a name with, so a factor cannot land on an unintended leaf;
    # if that ever changed the name would have to carry its group.
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
    floating = {name for name, leaf in (*dycore_template, *physics_template)
                if np.issubdtype(leaf.dtype, np.floating)}
    non_numeric = sorted(str(k) for k in unstamped_scale if str(k) not in floating)
    if non_numeric:
        # A scale factor only means anything on a float leaf; applying one
        # to an integer counter or a boolean flag would quietly corrupt it.
        raise ValueError(
            f"unstamped_scale names {non_numeric}, which are not "
            "floating-point leaves. A unit rescale applies to a float "
            "array only."
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
        "caller-asserted scale for %d leaf name(s)",
        path, len(unstamped_scale),
    )
    return groups[0], groups[1]


def load_checkpoint(model, path, *, unstamped_scale=None,
                    as_initial_condition=False,
                    metadata: dict | None = None) -> float:
    """Restore ``dycore_state`` + ``physics_carry`` from ``path``.

    The model must already have been bootstrapped (e.g. by an earlier
    ``Model.run``, ``Model.bootstrap_state``, or one of the initial-state
    builders in :mod:`jcm.initial_states`) so that its state pytrees provide
    the per-leaf names, shapes and dtypes the file is matched against.

    Leaves are matched **by name**, so an upgrade that adds or removes a
    field on a physics-carry struct still restores: a field the file does
    not have is filled from a *freshly built* carry
    (:meth:`Model.initial_physics_carry` — the term's documented seed,
    ``.zeros()`` for most slots), not from whatever this model currently
    holds, which for a model populated by an earlier ``Model.run`` is an
    evolved state from an unrelated integration. A field the file has but
    this model does not is dropped. Both are
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
        as_initial_condition: Import the fields at this model's start time,
            resetting the exact clock and dycore counter. Required for files
            predating the exact-clock schema; their seasonal interpretation
            cannot be continued as a v3 run.
        metadata: Optional dict, filled with the file's non-array records
            (currently ``data_mirror_revision``, the mirror commit the saving
            run read, or ``None`` for a file that predates the record), so a
            caller can check it without decoding the file twice.

    Returns:
        The ``elapsed_days`` recorded in the file, i.e. the *donor's*
        elapsed time. On a resume this equals the restored clock's elapsed
        time (the two are cross-checked), so a chunked loop may use it to
        skip completed chunks. With ``as_initial_condition=True`` the clock
        restarts at this model's ``start_time`` (elapsed zero, step zero)
        and the value describes the donor only — for logging/provenance,
        as :func:`jcm.initial_states.checkpoint_state` uses it; schedule
        any further run from ``model.run_state.time`` instead.

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

    if metadata is not None:
        rev = raw.get("data_mirror_revision")
        metadata["data_mirror_revision"] = None if rev is None else str(rev)
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

    if schema < 2 and not as_initial_condition:
        raise ValueError(
            f"Checkpoint {path} has no exact Gregorian clock (schema {schema}). "
            "It cannot resume a v3 run. Import it with "
            "as_initial_condition=True to start a new run; an unstamped file "
            "also requires unstamped_scale to declare its unit convention.")

    if as_initial_condition:
        restored_time = model.start_time
        restored_step = np.int32(0)
    else:
        clock = raw.get("clock")
        required = {"start_days", "start_seconds", "days", "seconds", "step",
                    "dt_seconds"}
        if not isinstance(clock, Mapping) or not required.issubset(clock):
            raise ValueError(f"Checkpoint {path} has an incomplete exact clock.")
        for name in required - {"dt_seconds"}:
            value = np.asarray(clock[name])
            if (value.shape != () or value.dtype.kind not in "iu"
                    or not np.iinfo(np.int32).min <= int(value) <= np.iinfo(np.int32).max):
                raise ValueError(f"Checkpoint clock {name!r} must be a scalar integer.")
        if not (0 <= int(clock["seconds"]) < 86400
                and 0 <= int(clock["start_seconds"]) < 86400
                and int(clock["step"]) >= 0):
            raise ValueError("Checkpoint clock is not normalized or has a negative step.")
        if (int(clock["start_days"]) != int(model.start_time.delta.days)
                or int(clock["start_seconds"]) != int(model.start_time.delta.seconds)
                or float(clock["dt_seconds"]) != float(model.dt_si.m)):
            raise ValueError("Checkpoint start_time/timestep does not match the model; "
                             "use as_initial_condition=True for a new experiment.")
        restored_time = jdt.Datetime(jdt.Timedelta(
            days=np.asarray(clock["days"], dtype=np.int32),
            seconds=np.asarray(clock["seconds"], dtype=np.int32)))
        restored_step = np.asarray(clock["step"], dtype=np.int32)
        delta = restored_time - model.start_time
        elapsed_seconds = int(delta.days) * 86400 + int(delta.seconds)
        if (elapsed_seconds != int(restored_step) * int(model.dt_si.m)
                or not np.isclose(float(raw["elapsed_days"]),
                                  elapsed_seconds / 86400.0, rtol=0, atol=1e-9)):
            raise ValueError("Checkpoint clock, step and elapsed_days disagree.")

    dycore_template = _named_leaves(model.dycore_state)
    physics_template = _named_leaves(model.physics_carry)

    if schema == _UNSTAMPED_SCHEMA:
        dycore_leaves, physics_leaves = _load_unstamped(
            raw, path, dycore_template, physics_template, unstamped_scale,
            _mass_mixing_ratio_leaves(model, dycore_template),
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
        # Protect what either side calls prognostic: this model's
        # declaration covers a slot the file predates, the file's covers
        # one this model no longer composes.
        protected = frozenset(_prognostic_carry_slots(model)) | frozenset(
            str(key) for key in _as_sequence(raw.get("prognostic_carry_slots"))
        )
        # Built at most once, and only if a field actually has to be
        # seeded — it re-runs the structural probe.
        seeds = functools.lru_cache(maxsize=1)(
            lambda: _fresh_carry_seeds(model))
        physics_leaves, seeded, dropped = _match_by_name(
            raw["physics"], physics_template,
            path=path, group="physics carry", fill_missing=True,
            protected=protected, seeds=seeds,
        )
        if seeded:
            logger.info(
                "Checkpoint %s: physics carry field(s) %s absent from the "
                "file; seeded from a freshly built carry (the terms' "
                "documented initial values)",
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
    if as_initial_condition:
        restored_dycore_state = model.dycore.with_sim_time(
            restored_dycore_state,
            np.zeros_like(model.dycore.sim_time(restored_dycore_state)))
    model.restore_state(restored_dycore_state, restored_physics_carry,
                        time=restored_time, step=restored_step)
    return float(np.asarray(raw["elapsed_days"]))
