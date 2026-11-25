"""Load a validated configuration recipe as built Python objects (issue #751).

The *recipe door*: ``jcm/config/configuration/*.yaml`` stays the single recipe
store (no parallel Python dict — that is how ``benchmark.PRESETS`` drifted).
``load(name)`` composes that yaml through Hydra INTERNALLY and hands back a frozen
:class:`LoadedConfiguration` of built objects — a :class:`~jcm.model.Model`, its
:class:`~jcm.forcing.ForcingData`, and the ``run_kwargs`` a caller passes to
``model.run`` — with Hydra/omegaconf invisible to the caller (only a plain-dict
``.config`` is exposed for introspection). So ``model.run(**exp.run_kwargs)``
reproduces ``python -m jcm.main +configuration=<name>``'s single integration.

``from_config(cfg)`` is the engine ``load`` calls after composing: it takes any
already-composed config (a downstream's own overlay included) and builds the
same objects, so a host process needs no ``jcm.runners`` internals. See
``docs/source/design/packaged_config_tree.md`` ("Driving jcm from a host
process"). ``runners.run`` keeps its own pre-build sequence (provenance reset
before the build, the ``from_state`` fail-fast, mode dispatch) and shares the
forcing and initial-state helpers with this module.

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


def _compose(name: str, overrides: list[str], extra_config_dir=None):
    """Compose ``+configuration=<name>`` (+ dotted overrides) against the root.

    ``initialize_config_dir`` clears the global Hydra on exit, and we also clear
    a pre-existing one up front, so ``load`` is safe to call repeatedly. To not
    leave a *host* application's own initialised Hydra cleared, we snapshot all
    Hydra singletons first and restore them in a ``finally`` — the host's
    context still composes after ``load`` returns (F3).

    ``extra_config_dir`` joins Hydra's search path (``hydra.searchpath``), so a
    downstream's own groups and recipes compose with jcm's: a file there that
    shares a group/name with jcm's wins, new groups and names are added.
    """
    if extra_config_dir is not None:
        overrides = [f"hydra.searchpath=[{Path(extra_config_dir).resolve().as_uri()}]",
                     *overrides]
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

    run = cfg.run
    output_selection = runners._output_selection_from_cfg(cfg)
    # A calendar ``total_time`` ("12 months", run/longrun.yaml) is resolved
    # against the start into the exact end_time, as the CLI does.
    total_time, end_time = runners._run_duration(
        cfg, run.get("start_time") or None)
    kwargs: dict = {
        "forcing": None,  # filled by the caller after build_forcing
        # Pass total_time/save_interval through as composed: ``Model.run`` parses
        # them with ``parse_duration_days`` (accepts int/float days OR a duration
        # string like ``"1 day"``/``"12 hours"``), so a float() cast here would
        # reject the string form the CLI accepts and break door equivalence.
        "save_interval": run.save_interval,
        "total_time": total_time,
        "end_time": end_time,
        "output_averages": bool(run.output_averages),
        "snapshot_interval": run.get("snapshot_interval"),
        "snapshot_variables": tuple(run.get("snapshot_variables") or ()),
        "output_keep": output_selection.keep,
        "output_drop": output_selection.drop,
    }
    kwargs.update(runners.initial_state_kwargs(cfg, model))
    return kwargs


#: Config sections a composed config must carry for the build to make sense;
#: ``init``, ``nudging``, ``constants`` and ``observers`` are optional (each
#: has a documented default when absent). Extra sections are never rejected.
_REQUIRED_SECTIONS = ("run", "grid", "physics", "terrain", "forcing")
_REQUIRED_RUN_KEYS = ("save_interval",)


def _check_required(cfg) -> None:
    """Raise ``ValueError`` naming every required key the composed ``cfg`` lacks."""
    missing = [k for k in _REQUIRED_SECTIONS if cfg.get(k) is None]
    if cfg.get("run") is not None:
        missing += [f"run.{k}" for k in _REQUIRED_RUN_KEYS
                    if cfg.run.get(k) is None]
        # The run's extent is ``total_time`` or, instead, an ``end_time``.
        if cfg.run.get("total_time") is None and cfg.run.get("end_time") is None:
            missing.append("run.total_time (or run.end_time)")
    if missing:
        raise ValueError(
            "composed config is missing required key(s): "
            f"{', '.join(missing)}. Compose it on top of jcm's own config "
            "(jcm.configurations.CONFIG_DIR) so its defaults supply them.")


def from_config(cfg, *, name: str | None = None, apply_log_level: bool = False,
                enable_compilation_cache: bool = False) -> LoadedConfiguration:
    """Build a model, forcing and ``run_kwargs`` from an already-composed config.

    The engine behind :func:`load`: a host that composes its own Hydra config
    (a ``--config-dir`` overlay on :data:`CONFIG_DIR`, see :func:`load`'s
    ``extra_config_dir``) calls this directly, so it needs no ``jcm.runners``
    internals. ``model.run(**result.run_kwargs)`` reproduces
    ``python -m jcm.main`` for ``cfg``'s single full-mode integration, and the
    config-trap warnings the CLI fires are emitted here too.

    The nudging target, when ``nudging.enabled``, comes back host-resident
    (numpy leaves) in ``result.forcing``; ``model.run(**run_kwargs)`` accepts
    that whole. A caller that steps the model itself keeps it on the host and
    puts one segment's window on the device with
    :func:`jcm.nudging.window_forcing`.

    Process-global steps are opt-in, so a library caller never has its own
    process reconfigured by default: ``apply_log_level=True`` sets the ``jcm``
    logger level from ``run.log_level`` (:func:`jcm.runners.apply_log_level`),
    and ``enable_compilation_cache=True`` turns on JAX's persistent compilation
    cache (:func:`jcm.runners.maybe_enable_compilation_cache`). Both are public
    for a host that wants them at another moment. ``constants.*`` overrides are
    always applied, because the dynamical core reads the live
    :mod:`jcm.constants` singleton at construction and the built model would
    otherwise disagree with ``config``. The CLI's other pre-build steps stay
    with :func:`jcm.runners.run`: the SPMD device count
    (``configure_host_device_count``; ``XLA_FLAGS`` is the reliable lever),
    ``provenance.start_run`` and the ``from_state`` fail-fast path check (a bad
    ``init.file`` still raises, from the state loader).

    Args:
        cfg: a composed config, an OmegaConf ``DictConfig`` or a plain nested
            dict. It is not modified. Sections jcm's defaults always supply
            (``run``, ``grid``, ``physics``, ``terrain``, ``forcing``,
            ``run.save_interval`` and one of ``run.total_time`` /
            ``run.end_time``) must be present, else
            ``ValueError`` lists what is missing; extra sections are carried
            through untouched.
        name: label for :attr:`LoadedConfiguration.name` (default ``"custom"``).
        apply_log_level: see above.
        enable_compilation_cache: see above.

    Returns:
        The same :class:`LoadedConfiguration` :func:`load` returns.

    """
    from omegaconf import DictConfig, OmegaConf

    from jcm import runners

    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    _check_required(cfg)

    if apply_log_level:
        runners.apply_log_level(cfg)
    if enable_compilation_cache:
        runners.maybe_enable_compilation_cache()

    label = "custom" if name is None else name
    # pySES recipes need the optional backend; fail with a clear message rather
    # than the raw ImportError build_model would surface (guarded like the tests).
    if cfg.get("dycore", {}).get("name", "dinosaur") == "pyses":
        import importlib.util
        if importlib.util.find_spec("pyses") is None:
            raise ModuleNotFoundError(
                f"configuration {label!r} uses the pySES CAM-SE backend, which "
                "is not installed. Install pyses (>=0.1.3.1) to load it.")

    runners.apply_constants_overrides(cfg)
    model = runners.build_model(cfg)
    forcing = runners.build_run_forcing(cfg, model)

    run_kwargs = _run_kwargs(cfg, model)
    run_kwargs["forcing"] = forcing
    # Plain resolved dict for introspection; a still-unfilled ``???`` key stays a
    # string rather than raising on this read-only copy.
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    return LoadedConfiguration(name=label, model=model, forcing=forcing,
                               run_kwargs=run_kwargs, config=config)


def load(name: str, *, extra_config_dir=None, **overrides) -> LoadedConfiguration:
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
    ``load`` is "compose, then :func:`from_config`" with its process-global
    steps off.

    A downstream that keeps its own config directory (its recipes, extra
    groups) passes it as ``extra_config_dir``: it joins Hydra's search path
    behind jcm's, so its new groups and recipes compose with jcm's and a recipe
    there may be named by ``name`` (the :func:`available` check then covers
    only jcm's own recipes, so it is skipped). To compose by other means (the
    host's own primary config, ``hydra.main``), compose against
    :data:`CONFIG_DIR` and hand the result to :func:`from_config`::

        from hydra import compose, initialize_config_dir
        from jcm.configurations import CONFIG_DIR, from_config

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose("config", overrides=[
                "+configuration=t63-echam-jam", "hydra.searchpath=[file:///my/conf]"])
        exp = from_config(cfg, apply_log_level=True)
    """
    if extra_config_dir is None and name not in (names := available()):
        raise ValueError(
            f"Unknown configuration {name!r}. Available: {sorted(names)}")

    overrides_list = [_override_str(k, v) for k, v in overrides.items()]
    cfg = _compose(name, overrides_list, extra_config_dir=extra_config_dir)
    return from_config(cfg, name=name)
