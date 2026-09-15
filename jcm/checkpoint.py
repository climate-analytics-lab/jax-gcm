"""Model state checkpointing for long, preemptible runs.

Persists ``Model.dycore_state`` and ``Model.physics_carry``
plus an elapsed sim-day count to a single file using flax's msgpack
serialization. ``run_chunked`` (in :mod:`jcm.runners`) integrates with
these primitives via ``cfg.run.checkpoint_path`` — when set, it writes a
checkpoint after each chunk and restores from one at startup if the file
exists, so an integration interrupted by spot-instance preemption resumes
without redoing completed chunks.

The state pytrees are flattened to plain lists of arrays before
serialization because flax's msgpack codec can't handle ``tree_math``
structs (e.g. dinosaur's ``primitive_equations.State``) directly. The
``treedef`` is reconstructed at load time from the destination model's
bootstrapped templates — this makes a checkpoint portable only across
runs with matching dycore + coords + physics term composition (where the
leaf order and dtypes line up), which is the intended usage.
"""

from __future__ import annotations

from pathlib import Path

import flax.serialization
import jax
import numpy as np


def _flatten_arrays(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def save_checkpoint(model, path, *, elapsed_days: float) -> Path:
    """Persist the model's current dycore + physics state to ``path``.

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
        "elapsed_days": float(elapsed_days),
        "dycore_leaves": _flatten_arrays(model.dycore_state),
        "physics_leaves": _flatten_arrays(model.physics_carry),
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
    """Restore ``dycore_state`` + ``physics_carry`` from ``path``.

    The model must already have been bootstrapped (e.g. by an earlier
    ``Model.run``, ``Model.bootstrap_state``, or one of the initial-state
    builders in :mod:`jcm.initial_states`) so that its state pytrees provide
    a treedef + per-leaf shape/dtype templates that match the checkpoint.

    Args:
        model: A ``jcm.model.Model`` with populated final states to use
            as deserialization templates. Their values are overwritten.
        path: Checkpoint file path written by :func:`save_checkpoint`.

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
    dycore_leaves_template = _flatten_arrays(model.dycore_state)
    physics_leaves_template = _flatten_arrays(model.physics_carry)
    template = {
        "elapsed_days": 0.0,
        "dycore_leaves": dycore_leaves_template,
        "physics_leaves": physics_leaves_template,
    }
    try:
        payload = flax.serialization.from_bytes(template, Path(path).read_bytes())
    except ValueError as exc:
        # from_bytes reports a bare leaf-count mismatch with no file name.
        # A count mismatch means a different physics composition wrote the
        # file, or a struct gained/lost a field since it was written.
        raise ValueError(
            f"Checkpoint {path} does not match the composed model: {exc}. "
            "It was written by a different physics composition, or by a "
            "jcm version whose diagnostic structs had different fields."
        ) from exc

    # from_bytes validates structure (leaf count) but not leaf shapes: a
    # same-composition state for the WRONG grid/levels deserializes cleanly
    # and only explodes later inside the jitted step, far from the cause.
    # Check every leaf against the template so the error names the file.
    for group, tmpl in (("dycore_leaves", dycore_leaves_template),
                        ("physics_leaves", physics_leaves_template)):
        for i, (got, want) in enumerate(zip(payload[group], tmpl)):
            if hasattr(want, "shape") and got.shape != want.shape:
                raise ValueError(
                    f"Checkpoint {path} does not match the composed model: "
                    f"{group}[{i}] has shape {got.shape}, model expects "
                    f"{want.shape} (wrong grid/levels/physics for this file)."
                )

    _, dycore_treedef = jax.tree_util.tree_flatten(model.dycore_state)
    _, physics_treedef = jax.tree_util.tree_flatten(model.physics_carry)
    restored_dycore_state = jax.tree_util.tree_unflatten(
        dycore_treedef, payload["dycore_leaves"])
    restored_physics_carry = jax.tree_util.tree_unflatten(
        physics_treedef, payload["physics_leaves"])
    model.restore_state(restored_dycore_state, restored_physics_carry)
    return float(payload["elapsed_days"])
