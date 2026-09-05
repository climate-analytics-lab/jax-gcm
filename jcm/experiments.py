"""Load a validated experiment recipe as built Python objects (issue #751).

The *recipe door*: ``jcm/config/experiment/*.yaml`` stays the single recipe store
(no parallel Python dict — that is how ``benchmark.PRESETS`` drifted). ``load(name)``
composes that yaml through Hydra INTERNALLY and hands back a frozen
:class:`LoadedExperiment` of built objects — a :class:`~jcm.model.Model`, its
:class:`~jcm.forcing.ForcingData`, and the ``run_kwargs`` a caller passes to
``model.run`` — with Hydra/omegaconf invisible to the caller (only a plain-dict
``.config`` is exposed for introspection). So ``model.run(**exp.run_kwargs)``
reproduces ``python -m jcm.main +experiment=<name>``'s single integration.

Both doors (this one and :meth:`jcm.forcing.ForcingData.from_bundles`) route
through the SAME :mod:`jcm.runners` builders the CLI uses, so a recipe means one
thing whether it is composed from the shell or from a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The Hydra config root and its experiment group — the single recipe store.
CONFIG_DIR = Path(__file__).resolve().parent / "config"
EXPERIMENT_DIR = CONFIG_DIR / "experiment"


@dataclass(frozen=True)
class LoadedExperiment:
    """A composed experiment, built and Hydra-free.

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
    """First human comment line of an experiment yaml (its one-line summary).

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
    """Map each experiment name to its one-line summary (sorted by name)."""
    return {p.stem: _summary(p) for p in sorted(EXPERIMENT_DIR.glob("*.yaml"))}


def _compose(name: str, overrides: list[str]):
    """Compose ``+experiment=<name>`` (+ dotted overrides) against the config root.

    ``initialize_config_dir`` clears the global Hydra on exit; we also clear a
    pre-existing one up front so ``load`` is safe to call repeatedly and from a
    context where a prior compose left Hydra initialised.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config",
                       overrides=[f"+experiment={name}", *overrides])


def _override_str(key: str, value) -> str:
    """One Hydra override token from a ``**overrides`` item (``None`` → ``null``)."""
    return f"{key}=null" if value is None else f"{key}={value}"


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
        "save_interval": float(run.save_interval),
        "total_time": float(run.total_time),
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
        raise ValueError(f"Unknown init.kind={kind!r} in experiment recipe")
    return kwargs


def load(name: str, **overrides) -> LoadedExperiment:
    """Compose a named experiment recipe and return its built objects.

    ``name`` is any :func:`available` key (a ``jcm/config/experiment/*.yaml``
    stem). ``**overrides`` is the optional escape hatch: Hydra dotted overrides
    passed straight into compose — e.g.
    ``load("t63-echam-jam", **{"run.total_time": 10})`` — where the dict keys
    carry the dots (``run.total_time``) because Python kwargs cannot. The
    returned :class:`LoadedExperiment` is Hydra-free: ``model``/``forcing`` are
    built and ``config`` is a plain resolved dict.

    ``model.run(**exp.run_kwargs)`` reproduces the CLI's single integration for
    the recipe; the same config-trap warnings the CLI fires are emitted here.
    """
    from omegaconf import OmegaConf

    from jcm import runners

    names = available()
    if name not in names:
        raise ValueError(
            f"Unknown experiment {name!r}. Available: {sorted(names)}")

    overrides_list = [_override_str(k, v) for k, v in overrides.items()]
    cfg = _compose(name, overrides_list)

    # pySES recipes need the optional backend; fail with a clear message rather
    # than the raw ImportError build_model would surface (guarded like the tests).
    if cfg.get("dycore", {}).get("name", "dinosaur") == "pyses":
        import importlib.util
        if importlib.util.find_spec("pyses") is None:
            raise ModuleNotFoundError(
                f"experiment {name!r} uses the pySES CAM-SE backend, which is "
                "not installed. Install pyses (>=0.1.3.1) to load it.")

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
    return LoadedExperiment(name=name, model=model, forcing=forcing,
                            run_kwargs=run_kwargs, config=config)
