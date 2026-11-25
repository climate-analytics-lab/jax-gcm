"""Choosing which variables a run writes (``run.output_keep`` / ``run.output_drop``).

A run's trajectory output is every prognostic plus every diagnostic the
composed physics publishes (~130 for the ECHAM package). Summing and stacking
all of them on the device, then transferring and serializing them, is wasted
work for a caller that wants a handful. :class:`OutputSelection` names the
variables to write, and this module applies it **before** the output
accumulation, so a dropped diagnostic is never summed or stacked:

* physics diagnostics are pruned out of the dict the running mean accumulates
  and the scan stacks (:func:`make_physics_pruner`);
* dynamics tracers are pruned out of the gridpoint state the same way, and the
  fixed prognostics (``u_wind`` ... ``normalized_surface_pressure``) are
  replaced by ``None`` leaves (:func:`select_dynamics`), which carry no
  arrays through the scan;
* :meth:`OutputSelection.filter_dataset` removes whatever is still left
  unselected from the Dataset (the one prognostic kept as the grid-shape
  reference, channel-expanded names, a backend's extras), so the Dataset a
  caller sees holds exactly the selected names however the pruning went.

Names are the variable names of the output Dataset (``temperature``,
``radiation.toa_sw_up``, ``surface_exchange.tile_fraction.0``). The pruning
acts only on the saved copy, never on the carry the scan threads between
steps, so a checkpoint has the same structure and a run resumes the same way
whatever is selected. Removing outputs does change the program XLA compiles
(different fusion), so a selected run agrees with an unselected one to
round-off and not bit for bit; a run that must reproduce bit for bit across
segments uses one selection throughout. With no selection the program is the
unselected one.
"""

from __future__ import annotations

import dataclasses
import difflib
from typing import Any, Callable, Iterable

import jax

#: Variables every output Dataset carries whatever is selected. ``time_bounds``
#: is the interval-mean frame metadata :meth:`ModelPredictions.to_xarray`
#: attaches; the time and level axes and the hybrid coefficients are
#: coordinates, which a selection never touches.
STRUCTURAL_NAMES = frozenset({"time_bounds"})

#: The prognostic fields of :class:`jcm.physics_interface.PhysicsState`
#: (tracers are a dict and are selected by their own names).
DYNAMICS_FIELDS = (
    "u_wind", "v_wind", "temperature", "specific_humidity", "geopotential",
    "normalized_surface_pressure",
)

#: The 3-D prognostics. Every consumer of a trajectory takes its frame count
#: and its ``(level, lon, lat)`` grid shape from the first of these present, so
#: :func:`select_dynamics` always leaves one of them in the saved state.
SHAPE_REFERENCE_FIELDS = DYNAMICS_FIELDS[:5]


def _as_names(value) -> tuple[str, ...] | None:
    """Normalise a config/argument value to a tuple of unique names."""
    if value is None:
        return None
    if isinstance(value, str):
        value = (value,)
    return tuple(dict.fromkeys(str(v) for v in value))


@dataclasses.dataclass(frozen=True)
class OutputSelection:
    """Which output variables a run writes.

    Hashable, so it can be a static argument of the jitted run.

    Attributes:
        keep: ``None`` writes every variable; otherwise a tuple of names, and
            only those are written.
        drop: Names never written. Applied after ``keep``, so a name in both
            is dropped.

    """

    keep: tuple[str, ...] | None = None
    drop: tuple[str, ...] = ()

    @classmethod
    def from_args(cls, keep=None, drop=None) -> "OutputSelection":
        """Build a selection from ``output_keep`` / ``output_drop`` values.

        Args:
            keep: ``None`` (everything), a name, or an iterable of names.
            drop: ``None``, a name, or an iterable of names.

        Returns:
            The normalised selection.

        Raises:
            ValueError: If ``keep`` is an empty list. Writing nothing is never
                what is meant; ``None`` is how to say "everything".

        """
        keep_names = _as_names(keep)
        if keep_names is not None and not keep_names:
            raise ValueError(
                "output_keep is an empty list, which would write no "
                "variables. Use null/None to write everything, or list the "
                "variables to keep.")
        return cls(keep=keep_names, drop=_as_names(drop) or ())

    @property
    def is_default(self) -> bool:
        """Whether this selection writes everything (the unchanged output)."""
        return self.keep is None and not self.drop

    def selected(self, name: str) -> bool:
        """Whether the variable ``name`` is written."""
        if name in STRUCTURAL_NAMES:
            return True
        if self.keep is not None and name not in self.keep:
            return False
        return name not in self.drop

    def requiring(self, names: Iterable[str]) -> "OutputSelection":
        """Return the selection that additionally writes ``names``.

        A runner that reads some variables back between chunks (the health
        gate) uses this to make the model compute them whatever the user
        selected; the user's own selection then strips them from what is
        written.
        """
        names = tuple(names)
        if not names:
            return self
        keep = (None if self.keep is None
                else tuple(dict.fromkeys(self.keep + names)))
        return OutputSelection(
            keep=keep, drop=tuple(n for n in self.drop if n not in names))

    def validate(self, available: Iterable[str]) -> None:
        """Raise if a listed name is not an output variable of this run.

        Args:
            available: The names the run can write.

        Raises:
            ValueError: Naming the unknown entries, the closest matches, and
                every available name. A typo must fail before the integration,
                not silently write an empty file.

        """
        available = set(available) | STRUCTURAL_NAMES
        problems = []
        for option, names in (("output_keep", self.keep or ()),
                              ("output_drop", self.drop)):
            for name in names:
                if name in available:
                    continue
                close = difflib.get_close_matches(name, sorted(available), n=3)
                hint = f" (did you mean {', '.join(close)}?)" if close else ""
                problems.append(f"{option}: {name!r}{hint}")
        if problems:
            raise ValueError(
                "Unknown output variable name(s): " + "; ".join(problems)
                + ". Names are those of the output Dataset. Available: "
                + ", ".join(sorted(available - STRUCTURAL_NAMES)) + ".")

    def filter_dataset(self, ds):
        """``ds`` without the data variables this selection does not write."""
        if self.is_default:
            return ds
        unselected = [v for v in ds.data_vars if not self.selected(str(v))]
        return ds.drop_vars(unselected) if unselected else ds


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def select_dynamics(dynamics, selection: OutputSelection):
    """Prune the gridpoint state a trajectory saves.

    Unselected tracers are removed. Unselected fixed prognostics become
    ``None``, which has no pytree leaves, so nothing is accumulated or stacked
    for them. One 3-D prognostic is always retained (``temperature`` when the
    selection would drop them all): trajectory consumers read the frame count
    and grid shape off it, and :meth:`OutputSelection.filter_dataset` removes
    it from the Dataset again.
    """
    if selection.is_default:
        return dynamics
    from jcm.physics_interface import PhysicsState

    fields = {
        name: (getattr(dynamics, name) if selection.selected(name) else None)
        for name in DYNAMICS_FIELDS}
    if all(fields[name] is None for name in SHAPE_REFERENCE_FIELDS):
        fields["temperature"] = dynamics.temperature
    tracers = {k: v for k, v in dynamics.tracers.items()
               if selection.selected(k)}
    return PhysicsState(tracers=tracers, **fields)


def dynamics_output_names(dynamics) -> set[str]:
    """Output names of a gridpoint state: the prognostics and the tracers."""
    return set(DYNAMICS_FIELDS) | set(dynamics.tracers)


def reference_field(dynamics):
    """Return the first 3-D prognostic present in a (possibly pruned) saved state."""
    for name in SHAPE_REFERENCE_FIELDS:
        value = getattr(dynamics, name, None)
        if value is not None:
            return value
    raise ValueError(
        "The saved dynamics hold none of "
        f"{', '.join(SHAPE_REFERENCE_FIELDS)}; no frame count or grid shape "
        "can be read from them.")


# ---------------------------------------------------------------------------
# Physics diagnostics
# ---------------------------------------------------------------------------

def _with_frame_axis(tree):
    """``tree`` with a leading length-1 time axis on every array.

    A trajectory's diagnostics are stacked over frames before they are
    named, and the flattening reads the stacked shapes (it expands a trailing
    channel axis, and unflattens a column axis, by comparing against the
    grid). Naming one step's arrays at that shape gives the names the saved
    trajectory will carry.
    """
    return jax.tree_util.tree_map(lambda x: x[None], tree)


def _entry_names(physics, key, value, nodal_shape) -> set[str] | None:
    """Output names one entry of the diagnostics dict produces.

    Asks the physics package itself (``data_struct_to_dict``), under
    ``jax.eval_shape`` so nothing is computed, so renames, channel expansion
    and withheld fields are exactly what the Dataset will carry. ``None``
    when the entry cannot be flattened abstractly (plumbing objects): such an
    entry is never pruned.
    """
    try:
        out = jax.eval_shape(
            lambda v: physics.data_struct_to_dict(
                {key: _with_frame_axis(v)}, nodal_shape),
            value)
    except Exception:  # noqa: BLE001 — plumbing entries are simply left alone
        return None
    return set(out)


def physics_output_names(physics, physics_state, nodal_shape) -> set[str]:
    """Every output name the diagnostics dict ``physics_state`` will produce."""
    if not isinstance(physics_state, dict):
        try:
            out = jax.eval_shape(
                lambda s: physics.data_struct_to_dict(
                    _with_frame_axis(s), nodal_shape),
                physics_state)
        except Exception:  # noqa: BLE001
            return set()
        return set(out)
    names: set[str] = set()
    for key, value in physics_state.items():
        found = _entry_names(physics, key, value, nodal_shape)
        if found:
            names |= found
    return names


_KEEP = "keep"
_DROP = "drop"


def _struct_fields(value) -> list[str]:
    if dataclasses.is_dataclass(value):
        return [f.name for f in dataclasses.fields(value)]
    return []


def _plan_struct(names_of, struct, selection, rebuild):
    """Plan which fields of a typed sub-struct are never written.

    ``names_of(candidate)`` returns the output names a candidate struct
    produces; ``rebuild(candidate)`` embeds it back into its dict entry. A
    field's names are the difference between the struct with and without it,
    so renames, channel expansion and nested structs need no special case.
    Returns ``{field: plan}`` for the fields that are not kept whole.
    """
    plan: dict[str, Any] = {}
    all_names = names_of(rebuild(struct))
    for field in _struct_fields(struct):
        fv = getattr(struct, field)
        if fv is None or isinstance(fv, (str, tuple)):
            continue
        without = dataclasses.replace(struct, **{field: None})
        mine = all_names - names_of(rebuild(without))
        if not mine:
            continue
        if not any(selection.selected(n) for n in mine):
            plan[field] = _DROP
        elif not all(selection.selected(n) for n in mine):
            if dataclasses.is_dataclass(fv):
                sub = _plan_struct(
                    names_of, fv, selection,
                    lambda c, _s=struct, _f=field: rebuild(
                        dataclasses.replace(_s, **{_f: c})))
                if sub:
                    plan[field] = sub
    return plan


def _apply_struct(struct, plan):
    updates = {}
    for field, sub in plan.items():
        if sub == _DROP:
            updates[field] = None
        else:
            updates[field] = _apply_struct(getattr(struct, field), sub)
    return dataclasses.replace(struct, **updates)


def make_physics_pruner(
    physics, selection: OutputSelection, template, nodal_shape,
) -> Callable[[Any], Any]:
    """Build the function that prunes a diagnostics dict to the selection.

    The plan is decided once, from ``template`` (any dict with the structure
    of the per-step diagnostics), at trace time; the returned function applies
    it to live values. A dict entry no selected name comes from is removed, a
    typed struct entry loses its unselected fields (they become ``None``,
    which :meth:`Physics.data_struct_to_dict` already omits), and an entry
    that produces no output name (carry plumbing) is removed, since it is not
    written under any selection; one that cannot be flattened abstractly is
    left alone.

    Args:
        physics: The physics package; it names its own output variables.
        selection: What to write.
        template: A diagnostics dict of the run's structure.
        nodal_shape: ``(level, lon, lat)`` shape of a gridpoint field.

    Returns:
        ``prune(physics_state) -> physics_state``, the identity for a default
        selection or a diagnostics container that is not a dict.

    """
    if selection.is_default or not isinstance(template, dict):
        return lambda state: state

    plan: dict[str, Any] = {}
    for key, value in template.items():
        names = _entry_names(physics, key, value, nodal_shape)
        if names is None:
            continue
        if not names:
            # Flattens to no output variable: carry plumbing (for example the
            # previous step's humidity handoff), never written under any
            # selection, so it is not worth stacking either.
            plan[key] = _DROP
            continue
        if all(selection.selected(n) for n in names):
            continue
        if not any(selection.selected(n) for n in names):
            plan[key] = _DROP
        elif isinstance(value, dict):
            drop_keys = []
            for sub_key, sub_value in value.items():
                sub_names = _entry_names(
                    physics, key, {sub_key: sub_value}, nodal_shape)
                if sub_names and not any(
                        selection.selected(n) for n in sub_names):
                    drop_keys.append(sub_key)
            if drop_keys:
                plan[key] = ("dict", tuple(drop_keys))
        elif dataclasses.is_dataclass(value):
            sub = _plan_struct(
                lambda cand, _k=key: _entry_names(
                    physics, _k, cand, nodal_shape) or set(),
                value, selection, lambda cand: cand)
            if sub:
                plan[key] = ("struct", sub)

    def prune(state):
        if not plan or not isinstance(state, dict):
            return state
        out = {}
        for key, value in state.items():
            action = plan.get(key)
            if action is None:
                out[key] = value
            elif action == _DROP:
                continue
            elif action[0] == "dict":
                out[key] = {k: v for k, v in value.items()
                            if k not in action[1]}
            else:
                out[key] = _apply_struct(value, action[1])
        return out

    return prune
