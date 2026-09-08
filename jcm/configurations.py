"""Load a validated configuration recipe as built Python objects (issue #751).

The *recipe door*: ``jcm/config/configuration/*.yaml`` stays the single recipe
store (no parallel Python dict — that is how ``benchmark.PRESETS`` drifted).
``load(name)`` composes that yaml through Hydra INTERNALLY and hands back a frozen
:class:`LoadedConfiguration` of built objects — a :class:`~jcm.model.Model`, its
:class:`~jcm.forcing.ForcingData`, and the ``run_kwargs`` a caller passes to
``model.run`` — with Hydra/omegaconf invisible to the caller (only a plain-dict
``.config`` is exposed for introspection). So ``model.run(**exp.run_kwargs)``
reproduces ``python -m jcm.main +configuration=<name>``'s single integration.

Both doors (this one and :meth:`jcm.forcing.ForcingData.from_bundles`) route
through the SAME :mod:`jcm.runners` builders the CLI uses, so a recipe means one
thing whether it is composed from the shell or from a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The Hydra config root and its configuration group — the single recipe store.
CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIGURATION_DIR = CONFIG_DIR / "configuration"


@dataclass(frozen=True)
class LoadedConfiguration:
    """A composed configuration, built and Hydra-free.

    ``run_kwargs`` already carries the applied initial state (e.g. ``jw`` →
    ``initial_state=jw_state(model, rh)``), so ``model.run(**run_kwargs)``
    reproduces the CLI integration. ``config`` is a plain resolved dict for
    introspection — no ``DictConfig`` leaks out.
    """

    name: str
    model: Any
    forcing: Any
    run_kwargs: dict
    config: dict = field(default_factory=dict)


def _summary(path: Path) -> str:
    """First human comment line of a configuration yaml (its one-line summary).

    The yamls open with ``# @package _global_``, a blank ``#``, then the summary;
    return that first real comment, or ``""`` if none is present.
    """
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s.startswith("#"):
            break
        body = s[1:].strip()
        if not body or body.startswith("@package"):
            continue
        return body
    return ""


def available() -> dict[str, str]:
    """Map each configuration name to its one-line summary (sorted by name)."""
    return {p.stem: _summary(p)
            for p in sorted(CONFIGURATION_DIR.glob("*.yaml"))}


def _compose(name: str, overrides: list[str]):
    """Compose ``+configuration=<name>`` (+ dotted overrides) against the root.

    ``initialize_config_dir`` clears the global Hydra on exit, and we also clear
    a pre-existing one up front, so ``load`` is safe to call repeatedly. To not
    leave a *host* application's own initialised Hydra cleared, we snapshot all
    Hydra singletons first and restore them in a ``finally`` — the host's
    context still composes after ``load`` returns (F3).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.singleton import Singleton

    saved = Singleton.get_state()
    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None,
                                   config_dir=str(CONFIG_DIR)):
            return compose(config_name="config",
                           overrides=[f"+configuration={name}", *overrides])
    finally:
        Singleton.set_state(saved)


def _override_str(key: str, value) -> str:
    """One Hydra override token from a ``**overrides`` item.

    ``None`` → ``null``. A ``str`` value is emitted as a Hydra *quoted string* so
    grammar characters (commas, ``=``, braces — ordinary in paths/filenames) are
    carried literally rather than parsed as list/sweep/assignment syntax;
    :meth:`QuotedString.with_quotes` is Hydra's own serializer, so the token
    parses back to the exact string (embedded quotes/backslashes handled).
    Non-string scalars pass through unquoted so ``run.total_time=10`` stays the
    number ``10`` (and a Python list/dict keeps its native override meaning).
    """
    from hydra.core.override_parser.types import Quote, QuotedString

    if value is None:
        return f"{key}=null"
    if isinstance(value, str):
        return f"{key}={QuotedString(text=value, quote=Quote.single).with_quotes()}"
    return f"{key}={value}"


def _run_kwargs(cfg, model) -> dict:
    """Assemble ``model.run(**...)`` kwargs from ``cfg.run`` + the applied init.

    Mirrors the dispatch in :func:`jcm.runners._run_full` so the returned kwargs
    reproduce the CLI's single integration: the run-section values plus the
    ``init.kind`` state applied (``jw``/``balanced_isothermal``/``era5`` set
    ``initial_state``; ``from_state`` additionally carries the donor physics
    carry). Chunking/checkpointing is a runner-loop concern and stays out here.
    """
    from jcm import runners
    from jcm.initial_states import balanced_isothermal_state, jw_state

    run = cfg.run
    kwargs: dict = {
        "forcing": None,  # filled by the caller after build_forcing
        # Pass total_time/save_interval through as composed: ``Model.run`` parses
        # them with ``parse_duration_days`` (accepts int/float days OR a duration
        # string like ``"1 day"``/``"12 hours"``), so a float() cast here would
        # reject the string form the CLI accepts and break door equivalence.
        "save_interval": run.save_interval,
        "total_time": run.total_time,
        "output_averages": bool(run.output_averages),
        "snapshot_interval": run.get("snapshot_interval"),
        "snapshot_variables": tuple(run.get("snapshot_variables") or ()),
    }
    init = cfg.get("init", {})
    kind = init.get("kind", "isothermal")
    if kind == "isothermal":
        pass
    elif kind == "jw":
        kwargs["initial_state"] = jw_state(model, rh=float(init.get("rh", 0.6)))
    elif kind == "balanced_isothermal":
        kwargs["initial_state"] = balanced_isothermal_state(model)
    elif kind == "from_state":
        state, carry = runners._state_from_file(model, cfg)
        kwargs["initial_state"] = state
        kwargs["initial_physics_state"] = carry
    elif kind == "era5":
        kwargs["initial_state"] = runners._state_from_era5(model, cfg)
    else:
        raise ValueError(f"Unknown init.kind={kind!r} in configuration recipe")
    return kwargs


def load(name: str, **overrides) -> LoadedConfiguration:
    """Compose a named configuration recipe and return its built objects.

    ``name`` is any :func:`available` key (a ``jcm/config/configuration/*.yaml``
    stem). ``**overrides`` is the optional escape hatch: Hydra dotted overrides
    passed straight into compose — e.g.
    ``load("t63-echam-jam", **{"run.total_time": 10})`` — where the dict keys
    carry the dots (``run.total_time``) because Python kwargs cannot. The
    returned :class:`LoadedConfiguration` is Hydra-free: ``model``/``forcing`` are
    built and ``config`` is a plain resolved dict.

    ``model.run(**exp.run_kwargs)`` reproduces the CLI's single integration for
    the recipe; the same config-trap warnings the CLI fires are emitted here.
    """
    from omegaconf import OmegaConf

    from jcm import runners

    names = available()
    if name not in names:
        raise ValueError(
            f"Unknown configuration {name!r}. Available: {sorted(names)}")

    overrides_list = [_override_str(k, v) for k, v in overrides.items()]
    cfg = _compose(name, overrides_list)

    # pySES recipes need the optional backend; fail with a clear message rather
    # than the raw ImportError build_model would surface (guarded like the tests).
    if cfg.get("dycore", {}).get("name", "dinosaur") == "pyses":
        import importlib.util
        if importlib.util.find_spec("pyses") is None:
            raise ModuleNotFoundError(
                f"configuration {name!r} uses the pySES CAM-SE backend, which "
                "is not installed. Install pyses (>=0.1.3.1) to load it.")

    # Apply any ``+constants.*`` overrides to the process-global jcm.constants
    # singleton BEFORE build, exactly as the CLI (runners.run) does — the dycore
    # reads the live singleton at construction, so skipping this would build with
    # default physics while ``.config`` claimed otherwise. Of runners.run()'s
    # other pre-build steps only this one bears on the built model, so it is the
    # sole one the door mirrors: configure_host_device_count (SPMD device count)
    # and maybe_enable_compilation_cache (JAX compile cache) are process-global,
    # perf-only side effects — XLA_FLAGS is the reliable device-count lever and a
    # notebook can enable the cache itself; provenance.start_run and the
    # from_state fail-fast path check are run-loop concerns (provenance is
    # attached by _run_full/run_chunked, which the door does not use, and a bad
    # init.file still raises via _state_from_file in _run_kwargs).
    runners.apply_constants_overrides(cfg)

    model = runners.build_model(cfg)
    dycore = getattr(model, "dycore", None)
    forcing = runners.build_forcing(cfg, model.coords, dycore=dycore)
    forcing = runners._maybe_attach_nudging_target(forcing, cfg, model)
    # Same guards/warnings the CLI runs after model+forcing construction.
    runners.guard_emulator_ghg_forcing(model.physics, forcing)
    runners.warn_on_config_traps(cfg, model.physics, forcing,
                                 coords=model.coords, dycore=dycore)

    run_kwargs = _run_kwargs(cfg, model)
    run_kwargs["forcing"] = forcing
    # Plain resolved dict for introspection; a still-unfilled ``???`` key stays a
    # string rather than raising on this read-only copy.
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    return LoadedConfiguration(name=name, model=model, forcing=forcing,
                               run_kwargs=run_kwargs, config=config)
