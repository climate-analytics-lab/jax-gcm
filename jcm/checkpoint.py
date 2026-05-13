"""Model state checkpointing for long, preemptible runs.

Persists ``Model._final_modal_state`` and ``Model._final_physics_state``
plus an elapsed sim-day count to a single file using flax's msgpack
serialization. ``run_chunked`` (in :mod:`jcm.runners`) integrates with
these primitives via ``cfg.run.checkpoint_path`` — when set, it writes a
checkpoint after each chunk and restores from one at startup if the file
exists, so an integration interrupted by spot-instance preemption resumes
without redoing completed chunks.

The state pytrees are flattened to plain lists of arrays before
serialization because flax's msgpack codec can't handle ``tree_math``
structs (e.g. ``primitive_equations.State``) directly. The ``treedef``
is reconstructed at load time from the destination model's bootstrapped
templates — this makes a checkpoint portable only across runs with
matching coords + physics term composition (where the leaf order and
dtypes line up), which is the intended usage.
"""

from __future__ import annotations

from pathlib import Path

import flax.serialization
import jax
import numpy as np


def _flatten_arrays(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def save_checkpoint(model, path, *, elapsed_days: float) -> Path:
    """Persist the model's current modal + physics state to ``path``.

    Args:
        model: A ``jcm.model.Model`` whose ``_final_modal_state`` and
            ``_final_physics_state`` have been populated, either by a
            prior ``run`` / ``resume`` call or by ``bootstrap_state``.
        path: Output file path (parent directories are created).
        elapsed_days: Sim-day count to record alongside the state so a
            chunked driver can resume at the correct offset.

    Returns:
        ``Path(path)`` for chaining.

    """
    if model._final_modal_state is None or model._final_physics_state is None:
        raise ValueError(
            "Model has no state to checkpoint — call Model.run(...), "
            "Model.resume(...), or Model.bootstrap_state(...) first."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_days": float(elapsed_days),
        "modal_leaves": _flatten_arrays(model._final_modal_state),
        "physics_leaves": _flatten_arrays(model._final_physics_state),
    }
    # Write to a sibling tmp file then rename atomically. If the run is
    # killed mid-write (the whole point of checkpointing for preemptible
    # workloads), the previous checkpoint is left intact rather than
    # truncated to a half-serialized blob that would fail to load.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(flax.serialization.to_bytes(payload))
    tmp_path.replace(path)
    return path


def load_checkpoint(model, path) -> float:
    """Restore ``_final_modal_state`` + ``_final_physics_state`` from ``path``.

    The model must already have been bootstrapped (e.g. by an earlier
    ``Model.run``, ``Model.bootstrap_state``, or one of the
    ``inject_*_profile`` helpers in :mod:`jcm.runners`) so that its
    state pytrees provide a treedef + per-leaf shape/dtype templates
    that match the checkpoint.

    Args:
        model: A ``jcm.model.Model`` with populated final states to use
            as deserialization templates. Their values are overwritten.
        path: Checkpoint file path written by :func:`save_checkpoint`.

    Returns:
        The ``elapsed_days`` count recorded when the checkpoint was
        saved.

    """
    if model._final_modal_state is None or model._final_physics_state is None:
        raise ValueError(
            "Model state is uninitialised — call Model.bootstrap_state(...) "
            "before load_checkpoint so the destination has templates to "
            "rebuild the pytrees from."
        )
    modal_leaves_template = _flatten_arrays(model._final_modal_state)
    physics_leaves_template = _flatten_arrays(model._final_physics_state)
    template = {
        "elapsed_days": 0.0,
        "modal_leaves": modal_leaves_template,
        "physics_leaves": physics_leaves_template,
    }
    payload = flax.serialization.from_bytes(template, Path(path).read_bytes())

    _, modal_treedef = jax.tree_util.tree_flatten(model._final_modal_state)
    _, physics_treedef = jax.tree_util.tree_flatten(model._final_physics_state)
    model._final_modal_state = jax.tree_util.tree_unflatten(
        modal_treedef, payload["modal_leaves"])
    model._final_physics_state = jax.tree_util.tree_unflatten(
        physics_treedef, payload["physics_leaves"])
    return float(payload["elapsed_days"])
