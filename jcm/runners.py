"""Build models and run simulations from a Hydra ``DictConfig``.

This is the bridge between the Hydra config groups in ``jcm/config/`` and the
construction of ``Model``, ``TerrainData``, ``DiffusionFilter`` and the various
physics packages. Keeps ``main.py`` minimal so other harnesses (notebooks,
integration tests) can import the same builders directly without going through
Hydra's CLI machinery.

By design (#640) this module contains **no science**: it parses config and
calls the library — the initial-state builders in :mod:`jcm.initial_states`, the weights and
burdens in :mod:`jcm.analysis`, the forcing helpers in :mod:`jcm.forcing`, the
relaxation profiles in :mod:`jcm.nudging`, and the various scheme constructors.
Every scientific choice lives in one of those homes with its own tests, so a
diff of this file should never need a scientific reviewer — only a config one.
New behaviour is added by promoting the science into a library home and calling
it from here, not by growing logic in the runner.
"""

from __future__ import annotations

import logging
import os
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import jax
from omegaconf import DictConfig

from jcm import provenance
from jcm.data import bundle_names
from jcm.diffusion import DiffusionFilter
from jcm.forcing import expand_yearly_files
from jcm.initial_states import (
    balanced_isothermal_state,
    jw_state,
)
from jcm.model import Model, ModelPredictions
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.single_column_model import select_column
from jcm.terrain import TerrainData
from jcm.utils import get_coords


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Host (CPU) device topology
# ---------------------------------------------------------------------------

def configure_host_device_count(n: int | None) -> None:
    """Expose ``n`` CPU devices to JAX so an ``spmd_mesh`` can shard over cores.

    JAX presents a *single* CPU device by default regardless of how many cores
    the host has, so multi-CPU SPMD needs the device count raised *before* the
    CPU backend initialises. This sets ``jax_num_cpu_devices``, which only
    takes effect if no JAX device has been touched yet — i.e. when called as
    the very first thing after ``import jax`` in a script/notebook (before
    importing ``jcm``).

    We also append ``--xla_cpu_enable_concurrency_optimized_scheduler=false``
    to ``XLA_FLAGS`` (idempotently): without it, complex graphs (e.g. ECHAM
    physics) crash at >= 8 CPU devices because the spectral transform's
    concurrent ``collective permute`` ops over-subscribe the XLA-CPU thread
    rendezvous. Like the device count, this only takes effect when set before
    the backend initialises.

    ``None`` or ``<= 1`` is a no-op (single-device run). If the backend is
    already live (e.g. under the CLI, where importing the model stack
    initialises it) neither the count nor the flag can change: we leave them,
    then validate and log a warning if the count falls short, pointing at the
    env-var lever the shell can set before the process starts:
    ``XLA_FLAGS="--xla_force_host_platform_device_count=N
    --xla_cpu_enable_concurrency_optimized_scheduler=false"``.
    """
    if not n or int(n) <= 1:
        return
    n = int(n)
    import jax

    # Serialise CPU collectives (idempotent). Harmless on GPU. Must precede any
    # device touch to take effect, hence set on the env before the calls below.
    _flag = "--xla_cpu_enable_concurrency_optimized_scheduler=false"
    if _flag not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _flag).strip()

    try:
        jax.config.update("jax_num_cpu_devices", n)
    except RuntimeError:
        # The CPU backend is already live (importing the model stack touches
        # it), so the count can no longer be raised from here. If the env var
        # was set before the process started we may already have the devices
        # we need — only warn when we actually fall short.
        pass

    got = jax.device_count()
    if got != n:
        logger.warning(
            "Requested %d CPU devices but JAX exposes %d — the backend was "
            "already initialised. Set it before the process starts, e.g. "
            "`export XLA_FLAGS=--xla_force_host_platform_device_count=%d`.",
            n, got, n,
        )


# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------

def build_coords(cfg: DictConfig):
    """Build a ``CoordinateSystem`` from ``cfg.grid``.

    ``cfg.grid.vertical`` is the coordinate *family* — ``sigma`` for
    equidistant sigma coordinates, ``hybrid`` for an ICON-style
    ``HybridCoordinates`` table. Layer count is independent of physics: each
    physics package is responsible for raising if it can't accept the chosen
    ``cfg.grid.layers`` (SPEEDY, for instance, only supports a fixed set).

    ``cfg.grid.spmd_mesh`` (optional) is a ``[x, y, z]`` triple specifying
    the SPMD device mesh over (longitude, latitude, vertical); pass ``null``
    or omit to run on a single device.
    """
    grid = cfg.grid
    layers = grid.layers
    truncation = grid.spectral_truncation
    spmd_mesh = grid.get("spmd_mesh", None)
    spmd_mesh = tuple(spmd_mesh) if spmd_mesh is not None else None

    vertical = grid.vertical
    if vertical == "sigma":
        from dinosaur.sigma_coordinates import SigmaCoordinates
        return get_coords(
            vertical_coords=SigmaCoordinates.equidistant(layers),
            spectral_truncation=truncation,
            spmd_mesh=spmd_mesh,
        )
    if vertical == "hybrid":
        # ECHAM/ICON ship pre-tuned full-depth hybrid tables for 47 / 95
        # levels; for any other count the user has to drop the table in by
        # hand. Keep the error chatty so the failure mode is obvious.
        from jcm.physics.echam.echam_levels import get_echam_levels
        try:
            vert = get_echam_levels(layers)
        except ValueError as exc:
            raise ValueError(
                f"hybrid coords with {layers} levels are not pre-configured. "
                "Use one of the supported counts (47, 95) or extend "
                "jcm.physics.echam.echam_levels.get_echam_levels."
            ) from exc
        return get_coords(
            vertical_coords=vert,
            spectral_truncation=truncation,
            spmd_mesh=spmd_mesh,
        )
    raise ValueError(
        f"Unknown grid.vertical={vertical!r}; expected 'sigma' or 'hybrid'."
    )


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _parameters_specs_from_init(term_cls) -> dict[str, type]:
    """Discover Parameters-typed kwargs on a term's ``__init__``.

    Returns a mapping from ``__init__`` kwarg name to the
    Parameters-like class declared as its (possibly ``Optional``) type
    annotation. A class is considered Parameters-like if it exposes a
    ``default`` classmethod — the structural marker used uniformly by
    every scheme (``ConvectionParameters.default()``,
    ``ModRadConParameters.default()``, …).

    The runner uses this mapping to decide which YAML blocks should be
    interpreted as Parameters field-override dicts (defaulted via
    ``ParamsCls.default()``) versus plain pass-through kwargs.
    """
    import inspect
    import typing

    try:
        hints = typing.get_type_hints(term_cls.__init__)
    except (NameError, TypeError):
        # Forward refs that fail to resolve, or no annotations: treat
        # everything as plain kwargs.
        return {}

    sig_params = inspect.signature(term_cls.__init__).parameters
    specs: dict[str, type] = {}
    for kwarg_name in sig_params:
        if kwarg_name == "self":
            continue
        annot = hints.get(kwarg_name)
        if annot is None:
            continue
        # Strip Optional[X] / Union[X, None] / X | None.
        origin = typing.get_origin(annot)
        if origin in (typing.Union, types.UnionType):
            non_none = [a for a in typing.get_args(annot) if a is not type(None)]
            if len(non_none) != 1:
                continue
            annot = non_none[0]
        # Structural test: anything with a ``default`` classmethod is a
        # Parameters dataclass for our purposes.
        if isinstance(annot, type) and callable(getattr(annot, "default", None)):
            specs[kwarg_name] = annot
    return specs


def _build_term(term_name: str, term_entry: dict):
    """Instantiate a single ``PhysicsTerm`` from a YAML term entry.

    Each ``cfg.physics.terms.<name>`` block names a term class via
    ``_target_`` plus optional kwargs. The runner introspects the
    term's ``__init__`` annotations: kwargs typed with a Parameters
    class (``ConvectionParameters | None``, …) are treated as
    field-override dicts — defaults come from ``ParamsCls.default()``,
    the user only has to supply the fields they want to tune. Any
    other kwargs are passed through as plain ``__init__`` arguments
    (used by terms like ``UpperSponge`` that take primitive values
    rather than Parameters dataclasses).
    """
    from hydra.utils import get_class

    if not isinstance(term_entry, dict) or "_target_" not in term_entry:
        raise ValueError(
            f"physics.terms.{term_name!r} must be a dict containing "
            f"'_target_'; got {term_entry!r}"
        )
    entry = dict(term_entry)
    target = entry.pop("_target_")
    term_cls = get_class(target)

    init_kwargs: dict = {}
    for kwarg_name, params_cls in _parameters_specs_from_init(term_cls).items():
        overrides = entry.pop(kwarg_name, None) or {}
        base = params_cls.default()
        init_kwargs[kwarg_name] = base.__class__(
            **{**base.__dict__, **dict(overrides)}
        )

    # Anything left is a plain-kwarg pass-through (e.g. UpperSponge's
    # n_sponge_levels, sponge_timescale_s).
    init_kwargs.update(entry)
    return term_cls(**init_kwargs)


def build_physics(cfg: DictConfig):
    r"""Build a ``ComposablePhysics`` from ``cfg.physics.terms``.

    ``cfg.physics.terms`` is an ordered mapping from term name to a
    Hydra-style entry::

        physics:
          checkpoint_terms: true
          vectorize_columns: true
          terms:
            tiedtke_convection:
              _target_: jcm.physics.convection.tiedtke_nordeng.TiedtkeConvection
              params:
                entrpen: 4.0e-4
            grey_two_stream_radiation:
              _target_: jcm.physics.radiation.grey_two_stream.GreyTwoStreamRadiation

    Override individual fields from the CLI without editing YAML, e.g.::

        python -m jcm.main physics=echam \
            physics.terms.tiedtke_convection.params.entrpen=4e-4

    Swap a term for an alternative by overriding its ``_target_`` (and
    optionally its kwargs) at the CLI, or by composing a preset YAML
    that pulls in ``physics: echam`` via ``defaults`` and then
    overrides individual term entries.
    """
    from omegaconf import OmegaConf

    from jcm.physics.composable_physics import ComposablePhysics

    physics_cfg = cfg.physics
    # Two physics-config styles: an explicit ``terms`` list (the default), or a
    # ``builder`` that delegates to a factory which already encodes the term
    # ordering. The factory style is how multi-term, order-sensitive packages
    # (notably the JAM aerosol chain, which is split around the cloud term) are
    # configured without re-expressing that ordering as flat YAML.
    if physics_cfg.get("builder", None) is not None:
        return _build_physics_from_factory(physics_cfg)

    terms_raw = physics_cfg.get("terms", None)
    if terms_raw is None:
        raise ValueError(
            "cfg.physics.terms is required (unless physics.builder is set). "
            "Each entry must declare a _target_ pointing at a PhysicsTerm "
            "subclass."
        )
    terms_cfg = OmegaConf.to_container(terms_raw, resolve=True) or {}

    terms = []
    for term_name, term_entry in terms_cfg.items():
        if term_entry is None:
            # Allow turning a term off via Hydra's `~` removal idiom or
            # an explicit ``null`` in the YAML — useful when inheriting
            # a default term list and dropping a term in the override.
            continue
        terms.append(_build_term(term_name, term_entry))

    physics = ComposablePhysics(
        terms=terms,
        checkpoint_terms=physics_cfg.get("checkpoint_terms", True),
        vectorize_columns=physics_cfg.get("vectorize_columns", False),
        band_config=RadiationBandConfig.for_terms(terms),
    )
    return physics


#: Physics ``builder`` names → factory callables returning a ``ComposablePhysics``
#: with its own validated term ordering (and band_config/vectorize handled
#: internally). The factory already orders the JAM aerosol chain (incl. the
#: pre/post-cloud split), so the preset YAML only carries scalar flags.
def _physics_factories():
    from jcm.physics.echam.echam_terms import echam_physics
    return {"echam_physics": echam_physics}


#: Yaml keys consumed by the runner itself, not the physics factory.
_CONFIG_ONLY_PHYSICS_KEYS = frozenset({
    "builder", "radiation_chunk_size", "defaults",
})

def _build_physics_from_factory(physics_cfg):
    """Build physics by delegating to a factory named by ``physics.builder``.

    The factory keyword args present in the YAML are forwarded; keys the
    runner itself consumes (``_CONFIG_ONLY_PHYSICS_KEYS``) are skipped.
    Anything else is an ERROR — a typo'd or removed key silently falling
    back to defaults invalidates the experiment that set it.
    """
    import inspect

    from omegaconf import OmegaConf

    factories = _physics_factories()
    builder = physics_cfg.get("builder")
    factory = factories.get(builder)
    if factory is None:
        raise ValueError(
            f"Unknown physics.builder={builder!r}; expected one of "
            f"{sorted(factories)}."
        )
    cfg_dict = OmegaConf.to_container(physics_cfg, resolve=True) or {}
    accepted = set(inspect.signature(factory).parameters)
    unknown = set(cfg_dict) - accepted - _CONFIG_ONLY_PHYSICS_KEYS
    if unknown:
        raise ValueError(
            f"physics config keys not accepted by {builder}: "
            f"{sorted(unknown)}. Fix or delete them — a typo'd or "
            "removed key silently falling back to the default would "
            "invalidate the experiment that set it."
        )
    kwargs = {k: v for k, v in cfg_dict.items()
              if k in accepted and v is not None}
    return factory(**kwargs)


#: Radiation band-config selection; the science lives in
#: :meth:`jcm.physics.radiation.band_config.RadiationBandConfig.for_terms`.
#: Aliased for backward compatibility (nn_emulator_scheme_test imports it).
_band_config_for_terms = RadiationBandConfig.for_terms


# The emulator GHG guard now lives with the scheme whose training it encodes
# (jax-gcm#738); runners keeps the historical name so existing callers/tests
# (``from jcm.runners import guard_emulator_ghg_forcing``) keep working.
from jcm.physics.radiation.nn_emulator_scheme import (  # noqa: E402
    guard_ghg_forcing as guard_emulator_ghg_forcing,
)


def maybe_add_sponge(physics, cfg: DictConfig):
    """Append an ``UpperSponge`` term if ``cfg.run.sponge.levels > 0``."""
    sponge = cfg.run.get("sponge", None)
    if sponge is None or sponge.get("levels", 0) <= 0:
        return physics
    from jcm.physics.dissipation import UpperSponge
    raw_target_T_K = sponge.get("target_T_K", None)
    target_T_K = None if raw_target_T_K is None else float(raw_target_T_K)
    return physics + UpperSponge(
        n_sponge_levels=int(sponge.levels),
        sponge_timescale_s=float(sponge.timescale_h) * 3600.0,
        enspodi=float(sponge.enspodi),
        damp_temperature=bool(sponge.get("damp_temperature", True)),
        target_T_K=target_T_K,
    )


def _nudging_inv_tau(nudging_cfg, vertical):
    """Adapt the ``nudging`` config to :func:`jcm.nudging.inv_tau_profile`.

    Returns ``(inv_tau, nlev)`` for :func:`maybe_add_nudging`.
    """
    from jcm.nudging import inv_tau_profile

    inv_tau = inv_tau_profile(
        vertical,
        tau_hours=float(nudging_cfg.get("tau_hours", 6.0)),
        min_pressure_hpa=float(nudging_cfg.get("min_pressure_hpa", 60.0)),
        pbl_levels=int(nudging_cfg.get("pbl_levels", 0)),
    )
    return inv_tau, inv_tau.size


def maybe_add_nudging(physics, cfg: DictConfig, coords):
    """Append a ``NudgingTerm`` when ``cfg.nudging.enabled`` (#610).

    Timescale config only — the ERA5 reference target is attached to
    forcing at run time (``_maybe_attach_nudging_target``), windowed to
    ``run.start_date + run.total_time``.
    """
    nudging_cfg = cfg.get("nudging", None)
    if nudging_cfg is None or not nudging_cfg.get("enabled", False):
        return physics
    import jax.numpy as jnp

    from jcm.nudging import NudgingConfig, with_nudging
    inv_tau, nlev = _nudging_inv_tau(nudging_cfg, coords.vertical)
    config = NudgingConfig(
        inv_tau_wind=jnp.asarray(inv_tau),
        inv_tau_temperature=(jnp.asarray(inv_tau)
                             if nudging_cfg.get("nudge_temperature", False)
                             else jnp.zeros(nlev)),
    )
    return with_nudging(physics, config)


def _maybe_attach_nudging_target(forcing, cfg: DictConfig, model):
    """Attach the windowed ERA5 nudging target to forcing (#610).

    The window is ``[run.start_date, start + total_time]`` padded by a
    day each side. Requires internet (or a warm ``jcm.data.era5``
    cache — prefetch on a login node for compute-node runs).
    """
    nudging_cfg = cfg.get("nudging", None)
    if nudging_cfg is None or not nudging_cfg.get("enabled", False):
        return forcing
    if nudging_cfg.get("source", "era5") != "era5":
        raise ValueError(
            f"Unknown nudging.source={nudging_cfg.get('source')!r} — "
            "only 'era5' (WeatherBench2) is implemented.")
    import datetime as _dt

    from jcm.data import era5
    start_raw = cfg.get("run", {}).get("start_date", None) or "2000-01-01"
    start = _dt.date.fromisoformat(str(start_raw)[:10])
    days = float(cfg.run.total_time)
    window = (str(start - _dt.timedelta(days=1)),
              str(start + _dt.timedelta(days=int(days) + 2)))
    target = era5.nudging_target(
        model.coords, *window, freq=str(nudging_cfg.get("freq", "6h")))
    forcing = _ensure_parent_forcing(forcing, model.coords)
    provenance.record_fact(
        "nudging", f"era5 {window[0]}..{window[1]} "
                   f"tau={nudging_cfg.get('tau_hours', 6.0)}h")
    return forcing.copy(nudging_target=target)


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def _resolve_data_path(path):
    """Resolve a boundary-file path from config.

    ``hf://<path-in-dataset>`` fetches (or reuses from the local HF cache)
    the file from the project data mirror via :mod:`jcm.data.remote`, e.g.
    ``hf://bundles/t63/terrain.nc``. Anything else passes through
    unchanged. Fetch on a login/head node first — compute nodes usually
    have no internet, but a warm cache needs none.
    """
    if isinstance(path, str) and path.startswith("hf://"):
        from jcm.data.remote import fetch
        resolved = fetch(path[len("hf://"):])
        provenance.record_input(path, resolved)
        return resolved
    if (not isinstance(path, (str, bytes, Mapping))
            and isinstance(path, Iterable)):
        # emissions_file may be a list of paths (incl. Hydra ListConfig).
        # Mappings/bytes pass through untouched — iterating them would
        # silently turn a mis-typed config into a list of keys/ints.
        return [_resolve_data_path(p) for p in path]
    if isinstance(path, str):
        provenance.record_input(path)   # no-op unless it is a real file
    return path


#: Yearly ``{year}`` file-pattern expansion; the science lives in
#: :func:`jcm.forcing.expand_yearly_files`. Aliased for the many call sites
#: (and TestYearExpansionAndStartDate) that reference the private name.
_expand_years = expand_yearly_files


def _forcing_products(file_spec, years, available):
    """Split a forcing file spec into independent products, each expanded.

    Each element of a **list** spec (e.g.
    ``emissions_file: [bb_{year}.nc, anthro.nc]``) names one product that is
    opened and time-aligned on its own — so a transient ``{year}`` product and
    a 12-month climatology in the same list keep their *distinct* time axes
    instead of being outer-joined into one incompatible axis. That outer-join
    is the bug this replaces: ``open_mfdataset(combine="by_coords")`` over
    disjoint time axes either NaN-fills every non-overlapping step (silently
    corrupting most timesteps) or raises on an integer-month-vs-datetime clash.
    The per-variable ``TimeSeries`` machinery already carries a *per-leaf* time
    axis and ``align_mode`` (see :func:`jcm.forcing._select_time_series`), so
    products with different alignments merge cleanly into one ``ForcingData``.

    Returns a list of *products*, each a scalar path or, for a ``{year}``
    element, its list of yearly files (opened together and concatenated along
    one time axis — the intended multi-year transient). A scalar spec is a
    single product.
    """
    if (not isinstance(file_spec, (str, bytes, Mapping))
            and isinstance(file_spec, Iterable)):
        return [expand_yearly_files(element, years, available)
                for element in file_spec]
    return [expand_yearly_files(file_spec, years, available)]


def _open_forcing_dataset(path):
    """Open one product's file(s) as a single xarray dataset.

    A product may be a list of yearly files (a ``{year}`` expansion) — those
    share one time axis and are concatenated with
    ``open_mfdataset(combine="by_coords")``. A scalar path opens directly.
    """
    import xarray as xr
    if isinstance(path, (list, tuple)):
        paths = [str(p) for p in path]
        return (xr.open_mfdataset(paths, combine="by_coords") if len(paths) > 1
                else xr.open_dataset(paths[0]))
    return xr.open_dataset(str(path))


def _product_available_years(forcing_cfg, key: str):
    """Per-product source coverage, falling back to ``available_years``.

    A preset can mix yearly products with different coverages (e.g.
    ``forcing_era5`` runs to 2024 while the FZJ ozone product ends in
    2022); a per-product override keeps each pattern's expansion inside
    the files that actually exist — for run dates beyond it, the time
    lookup clamps to the last sample.
    """
    avail = forcing_cfg.get(key, None)
    return avail if avail is not None else forcing_cfg.get(
        "available_years", None)


def build_terrain(cfg: DictConfig, coords) -> TerrainData:
    terrain_cfg = cfg.terrain
    kind = terrain_cfg.kind
    if kind == "aquaplanet":
        return TerrainData.aquaplanet(coords)
    if kind == "auto":
        return TerrainData.from_coords(
            coords, terrain_file=_resolve_auto_terrain(coords),
        )
    if kind == "from_file":
        return TerrainData.from_coords(
            coords,
            terrain_file=_resolve_data_path(terrain_cfg.file),
            interpolate=terrain_cfg.get("interpolate", True),
        )
    if kind == "from_file_enveloped":
        return TerrainData.from_file(
            _resolve_data_path(terrain_cfg.file), coords=coords,
            orog_envelope_wavenumber=terrain_cfg.get(
                "orog_envelope_wavenumber", None),
        )
    raise ValueError(f"Unknown terrain.kind={kind!r}")


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def build_diffusion(cfg: DictConfig) -> DiffusionFilter:
    """Build a ``DiffusionFilter`` honouring ``cfg.diffusion`` + the grid.

    Resolution selector: when ``cfg.diffusion.kind`` is ``"auto"`` (the
    default) and the grid is a hybrid grid with a level count ECHAM
    tabulates (L47 or L95), return the ECHAM ``lmidatm`` level-dependent
    profile for that ``(truncation, layers)`` — del² near the model top
    grading to del⁶/del⁸ below, with the ``setdyn.f90`` base timescale.
    That's the stability stack these grids were tuned for in ECHAM, and it
    is what the L95 middle-atmosphere grids exist to exploit. Any other
    grid — SPEEDY T31L8, Held-Suarez, a hybrid grid at an untabulated level
    count — gets the uniform SPEEDY del² profile, with a warning in the
    hybrid case since that is unlikely to be what was intended (#579).

    Set ``cfg.diffusion.kind: default`` to force the uniform SPEEDY profile
    (24h temp / 12h vor_q / 2h div), ``echam_lmidatm`` to force the ECHAM
    profile for the configured grid, or ``echam_t63_l47`` / ``echam_t85_l47``
    to pin a specific named profile regardless of grid.
    ``cfg.diffusion.scale`` still multiplies the chosen profile's timescales
    — keep the existing SPEEDY-tuned configs working unchanged.
    """
    diffusion = cfg.get("diffusion", None)
    kind = "auto" if diffusion is None else str(diffusion.get("kind", "auto"))
    scale = 1.0 if diffusion is None else float(diffusion.get("scale", 1.0))

    grid_cfg = cfg.get("grid", None)
    layers = int(grid_cfg.get("layers", 0)) if grid_cfg is not None else 0
    truncation = int(grid_cfg.get("spectral_truncation", 0)) if grid_cfg is not None else 0
    vertical = str(grid_cfg.get("vertical", "")) if grid_cfg is not None else ""

    if kind == "auto":
        base = DiffusionFilter.auto(truncation, layers, vertical)
    elif kind == "default":
        base = DiffusionFilter.default()
    elif kind == "echam_lmidatm":
        base = DiffusionFilter.echam_lmidatm(truncation, layers)
    elif kind == "echam_t63_l47":
        base = DiffusionFilter.echam_t63_l47()
    elif kind == "echam_t85_l47":
        base = DiffusionFilter.echam_t85_l47()
    else:
        raise ValueError(
            f"Unknown diffusion.kind={kind!r}; expected one of "
            "'auto', 'default', 'echam_lmidatm', 'echam_t63_l47', "
            "'echam_t85_l47'."
        )

    base.validate_layers(layers)
    return base.scaled(scale)


# ---------------------------------------------------------------------------
# Initial state — thin config adapters
#
# The state-builder science lives in ``jcm.initial_states.injectors``. The
# profile builders (JW, balanced-isothermal) take no config and are
# re-exported unchanged above; the file/era5 states need Hydra-config
# adaptation and get the thin adapters below. Each returns a state to hand
# to ``model.run(initial_state=...)``.
# ---------------------------------------------------------------------------


def _state_from_file(model: Model, cfg: DictConfig):
    """Config adapter for the ``init.kind=from_state`` warm start.

    Resolves ``init.file`` to a local path, rejects the case where it
    collides with ``run.checkpoint_path`` (the first-chunk checkpoint would
    overwrite the donor init state), then delegates to
    :func:`jcm.initial_states.checkpoint_state` for the load and clock-reset
    semantics. Returns ``(state, physics_carry)`` for the caller to hand to
    ``model.run(initial_state=..., initial_physics_state=...)`` — the donor's
    physics carry is threaded through so the warm start keeps its radiation
    sub-cycle cache / prior-step TKE rather than resetting them.
    """
    from jcm.initial_states import checkpoint_state

    path = _resolve_data_path(cfg.init.file)
    ckpt = cfg.run.get("checkpoint_path", None)
    if ckpt and Path(ckpt).resolve() == Path(path).resolve():
        raise ValueError(
            "init.file and run.checkpoint_path point at the same file: the "
            "first chunk checkpoint would overwrite the donor init state. "
            "Give the run its own checkpoint_path."
        )
    state, physics_carry, days = checkpoint_state(model, path)
    logger.info(
        "init=from_state: loaded %s (donor state carried %.0f sim-days); "
        "clock reset to 0", path, days,
    )
    return state, physics_carry


def _state_from_era5(model: Model, cfg: DictConfig):
    """Config adapter for the ``init.kind=era5`` initial condition.

    Resolves the ERA5 slice date from ``init.date``, else ``run.start_date``,
    else the 2000-01-01 default — matching the calendar the run integrates on
    — records provenance, then returns the regridded ``PhysicsState`` from
    :func:`jcm.data.era5.initial_state` for the caller to run.
    """
    from jcm.data import era5

    date = (cfg.get("init", {}).get("date", None)
            or cfg.get("run", {}).get("start_date", None)
            or "2000-01-01")
    provenance.record_fact("initial_condition", f"era5:{date}")
    return era5.initial_state(model.coords, str(date))


# ---------------------------------------------------------------------------
# Top-level model construction
# ---------------------------------------------------------------------------

def build_tracer_filter(cfg: DictConfig):
    """Build the optional dycore-side gridpoint tracer filter.

    Controlled by ``cfg.diffusion.tracer_positivity``. The only filter currently
    is mass-conserving positivity, which a spectral core applies as it projects
    to the physics gridpoint state so the sharp-source tracer fields of
    prognostic/prescribed aerosol emissions stay non-negative at the
    dynamics→physics boundary (Gibbs ringing otherwise NaNs the microphysics;
    see issue #521).

    Resolution of the config value:

    * ``true`` / ``false`` — force the filter on/off.
    * ``"auto"`` (or unset) — enable it only when the physics advects prognostic
      aerosol tracers (``physics.aerosol_module == "jam"``). This defaults the
      fix on for exactly the runs that need it while leaving non-aerosol runs
      bit-identical (the filter differs from the plain ``verify_state`` clip only
      where a tracer rings negative).

    Returns ``None`` when disabled — a no-op on the dycore.
    """
    diffusion = cfg.get("diffusion", None)
    tp = None if diffusion is None else diffusion.get("tracer_positivity", "auto")
    if isinstance(tp, bool):
        enabled = tp
    else:  # "auto" / null → on iff prognostic aerosols are advected
        physics = cfg.get("physics", None)
        aerosol_module = None if physics is None else physics.get("aerosol_module", None)
        enabled = (aerosol_module == "jam")
    if not enabled:
        return None
    from jcm.filters import MassConservingPositivity
    return MassConservingPositivity()


def _want_omega(cfg: DictConfig, physics=None) -> bool:
    """Resolve the dycore omega provider from the config and physics.

    An explicit ``dycore.compute_omega`` always wins. Left unset, the
    provider defaults ON when either (a) the composed physics REQUIRES
    the ``omega`` dycore field (the ``OmegaDiagnostic`` term, or Tiedtke
    convection with ECHAM's ``lmfmid`` mid-level trigger on) — ``Model``
    would switch the provider on anyway, so this only keeps the resolved
    config honest about what the run computes — or (b) the
    physics config runs the AeroCom ``plev`` group (``enable_aerocom``
    with ``plev`` in ``aerocom_groups``), whose wap/w500/w700 would
    otherwise be silently zero-filled: exactly the kind of
    valid-looking-but-empty submission file nobody catches until review.
    """
    explicit = cfg.get("dycore", {}).get("compute_omega", None)
    if explicit is not None:
        return bool(explicit)
    if physics is not None and "omega" in tuple(
            getattr(physics, "required_dycore_fields", lambda: ())()):
        return True
    phys = cfg.get("physics", {})
    return bool(phys.get("enable_aerocom", False)) and (
        "plev" in (phys.get("aerocom_groups") or ()))


def _resolve_start_date(cfg: DictConfig):
    """``run.start_date`` (ISO date string) as a ``jax_datetime.Datetime``.

    ``None``/unset keeps ``Model``'s default (2000-01-01). Transient
    (``BY_DATE``-aligned) forcing samples the file at the absolute model
    date, so a historical run must set this to place itself on the
    forcing's calendar (issue #610).
    """
    raw = cfg.get("run", {}).get("start_date", None)
    if raw in (None, "", "null"):
        return None
    import jax_datetime as jdt
    return jdt.to_datetime(str(raw))


def build_model(cfg: DictConfig) -> Model:
    """Build a fully-configured ``Model`` from a Hydra config.

    The ``dycore`` config group selects the backend: ``dinosaur`` (default,
    grid/diffusion/time_step from their own groups) or ``pyses`` (CAM-SE;
    resolution and timestep come from the dycore group itself — see
    ``config/dycore/pyses_ne30l47.yaml``).
    """
    from jcm.dycore.dinosaur.dycore import DEFAULT_OFF_CENTERING, DinosaurDycore

    dycore_name = cfg.get("dycore", {}).get("name", "dinosaur")
    if dycore_name == "pyses":
        init_kind = cfg.get("init", {}).get("kind", "isothermal")
        if init_kind not in ("isothermal", "from_state"):
            raise ValueError(
                f"init={init_kind!r} is dinosaur-specific; the pySES backend "
                "initializes from its resting USSA-1976 state (init="
                "isothermal) or a saved pySES state (init=from_state)."
            )
        if cfg.get("nudging", {}).get("enabled", False):
            raise ValueError(
                "nudging is dinosaur-only for now: the relaxation "
                "broadcasts over a 2-D lon/lat horizontal layout, not "
                "pySES physics columns."
            )
        return _build_pyses_model(cfg)
    if dycore_name != "dinosaur":
        raise ValueError(
            f"Unknown dycore config name {dycore_name!r} — expected "
            "'dinosaur' or 'pyses'."
        )

    coords = build_coords(cfg)
    physics = build_physics(cfg)
    physics = maybe_add_sponge(physics, cfg)
    physics = maybe_add_nudging(physics, cfg, coords)
    terrain = build_terrain(cfg, coords)
    diffusion = build_diffusion(cfg)
    tracer_filter = build_tracer_filter(cfg)

    log_level = getattr(logging, cfg.run.log_level.upper(), logging.WARNING)
    # Build the dycore explicitly so the diffusion config flows in via the
    # dycore constructor (Model itself no longer takes a diffusion kwarg —
    # that's a dinosaur-backend concern). The tracer filter is the same kind of
    # dycore-side knob.
    time_step = float(cfg.run.time_step)
    tracer_specs = {spec.name: spec for spec in physics.required_tracers()}
    sl_options = {"off_centering": float(
        cfg.get("sl_off_centering", DEFAULT_OFF_CENTERING))}
    dycore = DinosaurDycore(
        coords=coords,
        terrain=terrain,
        dt_seconds=time_step * 60.0,
        tracer_specs=tracer_specs,
        diffusion=diffusion,
        tracer_filter=tracer_filter,
        compute_omega=_want_omega(cfg, physics),
        sl_options=sl_options,
    )
    return Model(
        dycore,
        physics=physics,
        time_step=time_step,
        start_date=_resolve_start_date(cfg),
        log_level=log_level,
    )


def _build_pyses_model(cfg: DictConfig) -> Model:
    """Build a Model on the pySES CAM-SE backend from ``cfg.dycore``.

    Composition mirrors the production ne30 campaign driver this replaces:
    the backend owns resolution and timestep (``grid`` group and
    ``run.time_step`` are ignored — the Model adopts ``dt_seconds``), the
    physics runs float32 on the float64 core, and a finite-lid sponge term
    (USSA temperature relaxation + implicit Rayleigh wind friction, see the
    dycore config's ``lid_sponge``) is appended to the physics: the ~1 Pa
    lid sits outside the shipped radiation schemes' validity and both
    refrigerates and accelerates unbounded without it.
    """
    import jax.numpy as jnp

    from jcm.dycore.pyses import PysesCamSEDycore

    dc = cfg.dycore
    physics = build_physics(cfg)
    tracer_specs = {spec.name: spec for spec in physics.required_tracers()}

    dycore = PysesCamSEDycore(
        nx=int(dc.nx), npt=int(dc.npt), nlev=int(dc.nlev),
        dt_seconds=float(dc.dt_seconds),
        nu_top=float(dc.nu_top), n_sponge=int(dc.n_sponge),
        coupling=str(dc.coupling), hypervis=str(dc.hypervis),
        nu_div_factor=float(dc.get("nu_div_factor", 2.5)),
        tracer_substeps=int(dc.get("tracer_substeps", -1)),
        dyn_substeps_per_tracer=int(dc.get("dyn_substeps_per_tracer", -1)),
        compute_frontogenesis=bool(dc.get("compute_frontogenesis", False)),
        terrain_file=_resolve_data_path(dc.get("terrain_file", None))
        or _pyses_default_bc("terrain.nc"),
        tracer_specs=tracer_specs,
        physics_dtype=jnp.float32,
    )

    sponge = dc.get("lid_sponge", None)
    if sponge is not None and int(sponge.get("levels", 0)) > 0:
        physics = physics + _pyses_lid_sponge_term(dycore, sponge)

    log_level = getattr(logging, cfg.run.log_level.upper(), logging.WARNING)
    # No time_step: the Model adopts the dycore's dt_seconds (single source
    # of truth; a conflicting run.time_step would raise).
    return Model(dycore=dycore, physics=physics,
                 start_date=_resolve_start_date(cfg), log_level=log_level)


def _pyses_default_bc(filename: str) -> str:
    """Resolve the packaged T63 boundary file (temporary downscale)."""
    import jcm

    path = str(Path(jcm.__file__).resolve().parent / "data" / "bc" / "t63"
               / filename)
    # A fallback actually opened is provenance like any explicit file.
    provenance.record_input(path)
    return path


def _pyses_lid_sponge_term(dycore, sponge_cfg):
    """Config adapter for the pySES finite-lid sponge term.

    Reads the sponge config keys and builds the term via
    :meth:`jcm.physics.dissipation.upper_temperature_relaxation.UpperTemperatureRelaxation.from_ussa`,
    which evaluates the USSA-1976 reference temperature on the dycore's own
    hybrid grid.
    """
    from jcm.physics.dissipation.upper_temperature_relaxation import (
        UpperTemperatureRelaxation,
    )

    uv_hours = float(sponge_cfg.get("uv_hours", 0.0) or 0.0)
    return UpperTemperatureRelaxation.from_ussa(
        dycore.coords.vertical.a_boundaries,
        dycore.coords.vertical.b_boundaries,
        n_levels=int(sponge_cfg.get("levels", 8)),
        timescale_s=float(sponge_cfg.get("t_hours", 6.0)) * 3600.0,
        wind_timescale_s=(uv_hours * 3600.0 if uv_hours > 0 else None),
    )


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------

#: The ``auto`` emission-bundle naming convention lives in
#: :mod:`jcm.data.bundle_names` (a jcm-import-free module) so the build-time
#: resolver here and the benchmark's pre-GPU prefetch enumerator share one
#: source of truth. ``auto`` resolves to the per-grid HF bundle when a
#: prognostic-aerosol (JAM) package is active.
_EMISSION_AUTO_BUNDLES = bundle_names.EMISSION_AUTO_BUNDLES


def _resolve_one_emission_input(value, key, coords, jam, is_pyses):
    """Resolve one prescribed-emission forcing value (``auto`` / explicit).

    * ``auto`` → the per-grid HF bundle path, fetched *now* (so a cold cache
      fails loudly at build time, not mid-run) — but only on the spectral path
      with a JAM package active; pySES native grids are not the spectral-token
      bundles, and a non-JAM package consumes no emissions, so both give
      ``None``. ``auto`` is the only grid-portable mechanism: it composes the
      concrete bundle path from :mod:`jcm.data.bundle_names` + the grid token,
      so one config follows the grid without any user-facing path template.
    * explicit null (``None``/``""``/``"null"``) → ``None`` (opt-out).
    * an explicit path / ``hf://`` URL → returned unchanged for the attach
      helper's own ``_resolve_data_path`` to fetch. A ``{year}`` pattern is
      left intact for :func:`_expand_years` (there is no ``{grid}``/``{nlev}``
      substitution — use ``auto`` to let a config follow the grid).
    """
    if value == "auto":
        if is_pyses or not jam:
            return None
        if _grid_token(coords) not in bundle_names.PUBLISHED_GRIDS:
            # The mirror publishes emission bundles only for the grids in
            # ``PUBLISHED_GRIDS``; any other grid (e.g. echam_t42_l8_sigma,
            # ma-t119) has no ``bundles/<grid>/*.nc`` to fetch. Resolve ``auto``
            # to None so the run falls back to the null, emission-free baseline
            # automatically instead of aborting on a 404 — restoring the prior
            # behaviour for every non-mirrored grid. ``_resolve_emission_inputs``
            # emits one aggregated info log naming the keys and the reason.
            return None
        hf = bundle_names.emission_bundle_path(
            key, _grid_token(coords), int(coords.nodal_shape[0]))
        try:
            return _resolve_data_path(hf)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"forcing.{key}=auto resolved to {hf} — the online-aerosol "
                "default supplies the per-grid emission bundles — but it is "
                "not in the local Hugging Face cache and could not be "
                "downloaded. Prefetch on a node with internet (python -c "
                f"\"from jcm.data.remote import fetch; fetch('{hf[5:]}')\"), "
                f"point forcing.{key} at a local file, or set forcing.{key}"
                "=null to run without it (the runner then warns it is "
                "emission-free)."
            ) from e
    if value in (None, "", "null", "none"):
        return None
    return value


def _resolve_emission_inputs(forcing_cfg, cfg, coords, is_pyses):
    """Concretise the ``auto`` emission keys on ``forcing_cfg``.

    Returns ``forcing_cfg`` with the four prescribed-emission keys resolved (see
    :func:`_resolve_one_emission_input`). ``auto`` is keyed off whether the
    composed physics is the prognostic-aerosol (JAM) module — the only package
    that consumes prescribed emissions.
    """
    from omegaconf import OmegaConf

    jam = str((cfg.get("physics", {}) or {}).get("aerosol_module", "")) == "jam"
    auto_keys = [key for key in _EMISSION_AUTO_BUNDLES
                 if str(forcing_cfg.get(key, None)) == "auto"]
    updates = {
        key: _resolve_one_emission_input(
            forcing_cfg.get(key, None), key, coords, jam, is_pyses)
        for key in _EMISSION_AUTO_BUNDLES
    }
    # One info log when a JAM spectral run on a non-mirrored grid nulls its
    # ``auto`` emission keys (see _resolve_one_emission_input): naming the keys
    # and the reason once, rather than silently supplying nothing or aborting
    # per key. Guarded so ``_grid_token`` (spectral-only) is never touched on
    # the pySES path or the non-JAM path (where ``auto`` already nulled and
    # ``coords`` may be absent/native-grid).
    if jam and not is_pyses and auto_keys:
        token = _grid_token(coords)
        if token not in bundle_names.PUBLISHED_GRIDS:
            logging.info(
                "forcing.%s=auto resolved to None: grid %r is not one of the "
                "mirror's published grids (%s), so no per-grid emission bundle "
                "exists to fetch. The JAM run therefore uses only online "
                "sources (Gong sea salt) for these inputs; point each key at "
                "an on-grid file to supply prescribed emissions.",
                ", ".join(auto_keys), token,
                ", ".join(sorted(bundle_names.PUBLISHED_GRIDS)))
    return OmegaConf.merge(forcing_cfg, updates)


def build_forcing(cfg: DictConfig, coords, dycore=None):
    """Build a ``ForcingData`` from ``cfg.forcing``.

    ``kind: default`` returns ``None`` — ``Model.run`` then falls back to the
    aquaplanet ``default_forcing(coords.horizontal)``. ``kind: from_file``
    loads a netCDF boundary file via ``ForcingData.from_file``.

    Optionally attaches an ozone climatology (``cfg.forcing.ozone_file``),
    prescribed aerosol emissions (``cfg.forcing.emissions_file``), a seawater
    DMS climatology (``cfg.forcing.dms_file``), a dust source/erodibility map
    (``cfg.forcing.dust_file``) and an oxidant climatology
    (``cfg.forcing.oxidants_file``); all files must already be on the model
    horizontal grid (the HAMMOZ-style natural-emission files may have
    descending latitude — they are validated and flipped to model order).

    The four prescribed-emission keys default to ``auto`` (see
    ``forcing/default.yaml``): a JAM (online-aerosol) package then composes the
    per-grid HF emission bundles by itself, so
    ``physics=echam-jam grid=echam_t63_l47_hybrid`` is the documented-canonical
    run without nine lines of ``hf://`` overrides. Non-JAM packages and the
    pySES backend leave them empty; ``forcing.<key>=null`` opts out explicitly.
    """
    is_pyses = dycore is not None and hasattr(dycore, "colmap")
    _forcing_cfg = cfg.get("forcing", None)
    if _forcing_cfg is not None:
        _forcing_cfg = _resolve_emission_inputs(
            _forcing_cfg, cfg, coords, is_pyses)

    if is_pyses:
        # pySES backend: monthly lon/lat climatology + JAM aerosol inputs,
        # each bilinearly interpolated onto the physics columns at build
        # time by ``jcm.dycore.pyses.forcing`` (files may live on any
        # regular lon/lat grid). ``ozone_file: auto`` resolves the packaged
        # climatology — column sampling has no exact-grid requirement, so
        # the T63 file serves any pySES resolution.
        from jcm.dycore.pyses.forcing import build_forcing as pyses_build_forcing

        ozone_file = _forcing_cfg.get("ozone_file", None)
        if ozone_file == "auto":
            from importlib import resources

            cand = (Path(str(resources.files("jcm")))
                    / "data" / "bc" / "t63" / "ozone.nc")
            if cand.exists():
                ozone_file = str(cand)
            else:
                logging.warning(
                    "forcing.ozone_file=auto: packaged t63/ozone.nc missing "
                    "— pySES run falls back to the ANALYTIC ozone profile "
                    "(~12 W/m2 clear-sky OLR low bias)."
                )
                ozone_file = None
        elif ozone_file in ("", "null", "none"):
            ozone_file = None
        provenance.record_fact(
            "ozone_source",
            f"prescribed:{ozone_file}" if ozone_file
            else "analytic (no ozone file)")

        file = (_resolve_data_path(_forcing_cfg.get("file", None))
                or _pyses_default_bc("forcing.nc"))
        forcing = pyses_build_forcing(
            str(file), dycore,
            emissions_file=_resolve_data_path(
                _forcing_cfg.get("emissions_file", None)),
            dms_file=_resolve_data_path(_forcing_cfg.get("dms_file", None)),
            dust_file=_resolve_data_path(_forcing_cfg.get("dust_file", None)),
            oxidants_file=_resolve_data_path(
                _forcing_cfg.get("oxidants_file", None)),
            ozone_file=_resolve_data_path(ozone_file),
        )
        # MACv2-SP plume weights are the one dycore-agnostic attachment the
        # spectral tail below also performs that ``pyses_build_forcing`` does
        # NOT: ``aerosol_year_weight``/``aerosol_ann_cycle`` are plume-indexed
        # scalar time series with NO horizontal field, so they need none of the
        # column bilinear sampling ``attach_jam_forcing`` does for the gridded
        # inputs (ozone/emissions/dms/dust/oxidants, which it therefore
        # reimplements). Reuse the SAME ``_attach_macv2_weights`` helper here so
        # ``forcing=macv2_sp`` on pySES actually loads its mandatory
        # ``macv2_file`` instead of silently dropping it — the very silent-ignore
        # trap warning 4 recommends this config to escape. (The exact-grid
        # ``validate_emissions_grid``/``validate_oxidant_levels`` checks stay
        # dinosaur-only: pySES interpolates every field onto columns, so it has
        # no exact-grid requirement and asserts dim order in ``attach_jam_forcing``
        # instead. Nudging is likewise dinosaur-only — attached later in ``run``,
        # not here, and gated off on pySES.)
        return _attach_macv2_weights(forcing, _forcing_cfg, coords)

    forcing_cfg = _forcing_cfg
    if forcing_cfg is None or forcing_cfg.kind == "default":
        forcing = None
    elif forcing_cfg.kind == "from_file":
        from jcm.forcing import ForcingData
        files = _expand_years(forcing_cfg.file, forcing_cfg.get("years", None),
                              forcing_cfg.get("available_years", None))
        forcing = ForcingData.from_file(
            _resolve_data_path(files), coords=coords,
            align_mode=str(forcing_cfg.get("align", "auto")))
    else:
        raise ValueError(f"Unknown forcing.kind={forcing_cfg.kind!r}")
    forcing = _attach_ozone(forcing, forcing_cfg, coords)
    forcing = _attach_emissions(forcing, forcing_cfg, coords)
    forcing = _attach_dms(forcing, forcing_cfg, coords)
    forcing = _attach_dust(forcing, forcing_cfg, coords)
    forcing = _attach_oxidants(forcing, forcing_cfg, coords)
    forcing = _attach_macv2_weights(forcing, forcing_cfg, coords)
    return forcing


def _model_latlon_deg(coords):
    """Model nodal latitudes/longitudes in degrees (dinosaur stores radians)."""
    import numpy as np
    lat_deg = np.asarray(coords.horizontal.latitudes) * 180.0 / np.pi
    lon_deg = np.asarray(coords.horizontal.longitudes) * 180.0 / np.pi
    return lat_deg, lon_deg


def _ensure_parent_forcing(forcing, coords):
    """Build the aquaplanet parent ``ForcingData`` when ``kind: default``.

    Same rationale as ``_attach_ozone``: ``default_forcing`` preserves the
    cos²-latitude SST climatology that ``ForcingData.zeros`` would silently
    replace with a uniform 288.15 K placeholder.
    """
    if forcing is not None:
        return forcing
    from jcm.forcing import default_forcing
    return default_forcing(coords.horizontal)


def _grid_token(coords) -> str:
    """Mirror grid token (``"t63"``) for the model's horizontal grid.

    Derived from the spectral resolution (truncation =
    ``total_wavenumbers - 2``, the same relation ``utils.get_coords``
    uses), so no hand-maintained table can go stale; whether the mirror
    actually carries the grid is decided by the fetch itself.
    """
    return bundle_names.grid_token(
        int(coords.horizontal.total_wavenumbers) - 2)


def _resolve_auto_ozone(coords):
    """Find an ozone climatology matching the model grid.

    Two-stage discovery: (1) a packaged ``jcm/data/bc/*/ozone.nc`` whose
    (nlev, nlat, nlon) match; (2) the data mirror's per-grid file
    ``bundles/<grid>_l<nlev>/ozone_pd.nc`` (cache-first fetch — works
    offline once cached; the loader rejects any grid mismatch, so only
    an exact-grid file is worth returning). Grid identity is then fully
    validated by
    ``OzoneClimatology.from_file``. Returns ``None`` when neither stage
    finds a file — the caller warns and falls back to the analytic
    profile, whose ~7.6× tropospheric ozone column biases clear-sky OLR
    ~12 W/m² low.
    """
    from importlib import resources

    import xarray as xr

    nlon, nlat = (int(v) for v in coords.horizontal.nodal_shape)
    nlev = int(coords.nodal_shape[0])
    bc_root = Path(str(resources.files("jcm"))) / "data" / "bc"
    for cand in sorted(bc_root.glob("*/ozone.nc")):
        with xr.open_dataset(cand) as ds:
            sizes = ds.sizes
            if (sizes.get("level") == nlev and sizes.get("lat") == nlat
                    and sizes.get("lon") == nlon):
                return str(cand)
    token = _grid_token(coords)
    from jcm.data.remote import bundle_file
    try:
        return str(bundle_file(f"{token}_l{nlev}", "ozone_pd.nc"))
    except Exception as e:  # noqa: BLE001 — degrade, but LOUDLY
        # Warning, not info: the analytic-profile fallback biases
        # clear-sky OLR ~12 W/m² and the generic no-packaged-file
        # warning downstream does not mention the failed mirror fetch.
        logger.warning(
            "auto-ozone: mirror fetch bundles/%s_l%d/ozone_pd.nc failed "
            "(%s); falling back to the analytic ozone profile.",
            token, nlev, e,
        )
    return None


def _resolve_auto_terrain(coords):
    """Native-grid terrain path for ``terrain.kind: auto``.

    Terrain must be NATIVE to the model grid: horizontally interpolating
    a coarser file breaks the Lott-Miller SSO sub-grid orography fields
    (shape mismatch inside the column vmap). Stages: packaged
    ``jcm/data/bc/*/terrain.nc`` shape-matched on (nlat, nlon), then the
    mirror's ``bundles/<grid>/terrain.nc``. Raises when neither exists,
    because a silently substituted terrain corrupts the run.
    """
    from importlib import resources

    import xarray as xr

    nlon, nlat = (int(v) for v in coords.horizontal.nodal_shape)
    bc_root = Path(str(resources.files("jcm"))) / "data" / "bc"
    for cand in sorted(bc_root.glob("*/terrain.nc")):
        with xr.open_dataset(cand) as ds:
            if (ds.sizes.get("lat") == nlat and ds.sizes.get("lon") == nlon):
                return str(cand)
    token = _grid_token(coords)
    from jcm.data.remote import bundle_file
    try:
        return str(bundle_file(token, "terrain.nc"))
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            f"terrain.kind=auto: no packaged terrain matches "
            f"({nlon}x{nlat}) and the mirror fetch of "
            f"bundles/{token}/terrain.nc failed: {e}"
        ) from e


def _attach_ozone(forcing, forcing_cfg, coords):
    """Load the ozone climatology and attach to ``forcing``.

    ``ozone_file: auto`` (the shipped default) resolves a packaged
    climatology matching the grid via ``_resolve_auto_ozone``; no match
    degrades to the analytic ozone profile with a warning. An explicit
    path is loaded strictly (errors on any mismatch). ``null`` disables
    the climatology silently (analytic profile, no warning).

    When ``forcing`` is ``None`` (``kind: default``) and an ozone file IS
    given, build the parent struct via ``default_forcing(...)`` so the
    aquaplanet cos²-latitude SST climatology is preserved — using
    ``ForcingData.zeros`` here would silently swap it for the uniform
    288.15 K placeholder, materially changing the boundary conditions
    for any run configured with only ``ozone_file``.
    """
    if forcing_cfg is None:
        return forcing
    ozone_file = _resolve_data_path(_expand_years(
        forcing_cfg.get("ozone_file", None),
        forcing_cfg.get("years", None),
        _product_available_years(forcing_cfg, "ozone_available_years")))
    if isinstance(ozone_file, (list, tuple)):
        ozone_file = [str(p) for p in ozone_file]
    if ozone_file in (None, "", "null"):
        provenance.record_fact("ozone_source", "analytic (no ozone_file)")
        return forcing
    if ozone_file == "auto":
        ozone_file = _resolve_auto_ozone(coords)
        if ozone_file is None:
            provenance.record_fact(
                "ozone_source", "analytic (auto found no packaged match)")
            logging.warning(
                "forcing.ozone_file=auto: no packaged jcm/data/bc/*/ozone.nc "
                "matches this grid — falling back to the ANALYTIC ozone "
                "profile, whose ~7.6x tropospheric ozone column biases "
                "clear-sky OLR low by ~12 W/m2. Prepare a climatology with "
                "jcm.data.bc.interpolate_ozone for production radiation."
            )
            return forcing
        logging.info("forcing.ozone_file=auto resolved to %s", ozone_file)
    import numpy as np

    from jcm.forcing import default_forcing
    from jcm.ozone_climatology import OzoneClimatology
    nlon, nlat = coords.horizontal.nodal_shape
    nlev = coords.nodal_shape[0]
    # Pass the model's lat/lon (degrees) so the loader catches files
    # with the right shape but flipped/shifted grids — same N points,
    # wrong column mapping, would otherwise wire ozone into the wrong
    # latitudes silently. Dinosaur stores both in radians.
    lat_deg = np.asarray(coords.horizontal.latitudes) * 180.0 / np.pi
    lon_deg = np.asarray(coords.horizontal.longitudes) * 180.0 / np.pi
    climatology = OzoneClimatology.from_file(
        ozone_file,
        nlon=int(nlon), nlat=int(nlat), nlev=int(nlev),
        lat_deg=lat_deg, lon_deg=lon_deg,
    )
    provenance.record_fact("ozone_source", f"prescribed:{ozone_file}")
    provenance.record_input(ozone_file)
    if forcing is None:
        forcing = default_forcing(coords.horizontal)
    return forcing.copy(ozone_climatology=climatology)


def _attach_emissions(forcing, forcing_cfg, coords):
    """Attach prescribed aerosol emissions from ``cfg.forcing.emissions_file``.

    No-op when unset. ``emissions_file`` may be a single path or a **list** of
    paths (e.g. one file for biomass burning and one for the rest). Each list
    element is a **product** opened and time-aligned on its own (see
    :func:`_forcing_products`), so a transient ``{year}`` product and a 12-month
    climatology can be mixed without outer-joining their disjoint time axes; the
    per-variable ``TimeSeries`` leaves from all products merge into one
    ``ForcingData``, each keeping its own time axis. The fields auto-route by
    content: variables named ``emis_<sector>_<species>`` drive the bulk /
    in-model-speciated path (``anthropogenic_emissions``); ``aero_emis_<tracer>``
    variables drive the CAM6-faithful pre-speciated path
    (``prescribed_aerosol_emissions``). A file may carry either or both. The
    fields must already be on the model horizontal
    grid — this does **not** regrid (use :mod:`jcm.data.emissions.prepare`
    first); a grid mismatch raises rather than silently zeroing (the emission
    terms fall back to zero on a size mismatch, which from the CLI would look
    like the file "did nothing"). Like ozone, when ``kind: default`` supplies no
    parent struct one is built via ``default_forcing`` so the aquaplanet SST
    climatology is preserved.

    The matching emission term must also be in the physics package (e.g.
    ``physics=echam-jam``) for the fields to be consumed.
    """
    if forcing_cfg is None:
        return forcing
    raw = forcing_cfg.get("emissions_file", None)
    if raw in (None, "", "null"):
        return forcing
    years = forcing_cfg.get("years", None)
    available = forcing_cfg.get("available_years", None)

    from jcm.forcing import (
        default_forcing,
        read_anthropogenic_emissions,
        read_prescribed_aerosol_emissions,
        validate_emissions_grid,
    )

    # Read each product independently and merge the per-variable TimeSeries
    # leaves. Each product keeps its own time axis / align_mode, so mixing a
    # transient product with a climatology does not force one shared axis.
    #
    # This per-product merge is meaningful HERE but not for oxidants (contrast
    # _attach_oxidants, which handles a list as ONE product): emission products
    # carry DISJOINT variables (different ``emis_<sector>_<species>`` /
    # ``aero_emis_<tracer>`` sets), so ``dict.update`` across products unions
    # genuinely distinct keys. Oxidant files must each carry the IDENTICAL four
    # gases, so an analogous update would be pure last-one-wins.
    anthro: dict = {}
    speciated: dict = {}
    for product in _forcing_products(raw, years, available):
        path = _resolve_data_path(product)
        if path in (None, "", "null"):
            continue
        ds = _open_forcing_dataset(path)
        try:
            a = read_anthropogenic_emissions(ds)
            s = read_prescribed_aerosol_emissions(ds)
        finally:
            ds.close()
        if a is None and s is None:
            raise ValueError(
                f"forcing.emissions_file {path!r} has no emissions variables: "
                "expected ``emis_<sector>_<species>`` (bulk) or "
                "``aero_emis_<tracer>`` (pre-speciated). See the emissions-file "
                "contract in docs/design/jam.md."
            )
        if a:
            anthro.update(a)
        if s:
            speciated.update(s)
    if not anthro and not speciated:
        return forcing
    validate_emissions_grid({**anthro, **speciated}, coords, raw)
    if forcing is None:
        forcing = default_forcing(coords.horizontal)
    return forcing.copy(anthropogenic_emissions=anthro or None,
                        prescribed_aerosol_emissions=speciated or None)


def _attach_dms(forcing, forcing_cfg, coords):
    """Attach the seawater-DMS climatology from ``cfg.forcing.dms_file``.

    No-op when unset. Loads a HAMMOZ-style monthly ``DMS_sea (time, lat, lon)``
    climatology (nmol/L, converted to kg/m³ — see
    :func:`jcm.forcing.read_dms_seawater`) as a ``WRAP_YEAR`` ``TimeSeries``
    on ``forcing.dms_seawater``, which :class:`DmsEmissions` consumes. The
    file must already be on the model horizontal grid; lat/lon values are
    validated (a descending-latitude file is flipped) and a mismatch raises —
    the term otherwise falls back to zero on a size mismatch, which from the
    CLI would look like the file "did nothing". Needs a JAM physics package
    (e.g. ``physics=echam-jam``) for the field to be consumed.
    """
    if forcing_cfg is None:
        return forcing
    path = _resolve_data_path(forcing_cfg.get("dms_file", None))
    if path in (None, "", "null"):
        return forcing
    import xarray as xr

    from jcm.forcing import read_dms_seawater
    lat_deg, lon_deg = _model_latlon_deg(coords)
    with xr.open_dataset(str(path)) as ds:
        ts = read_dms_seawater(ds, lat_deg=lat_deg, lon_deg=lon_deg)
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(dms_seawater=ts)


def _attach_dust(forcing, forcing_cfg, coords):
    """Attach the dust-source/erodibility map from ``cfg.forcing.dust_file``.

    No-op when unset. Loads a HAMMOZ-style monthly ``pot_source
    (time, lat, lon)`` climatology (clipped to the [0, 1] erodibility contract
    of :class:`DustEmissions` — see :func:`jcm.forcing.read_dust_source`) as a
    ``WRAP_YEAR`` ``TimeSeries`` on ``forcing.dust_source``. Grid handling as
    in :func:`_attach_dms`.
    """
    if forcing_cfg is None:
        return forcing
    path = _resolve_data_path(forcing_cfg.get("dust_file", None))
    if path in (None, "", "null"):
        return forcing
    import xarray as xr

    from jcm.forcing import read_dust_source
    lat_deg, lon_deg = _model_latlon_deg(coords)
    with xr.open_dataset(str(path)) as ds:
        ts = read_dust_source(ds, lat_deg=lat_deg, lon_deg=lon_deg)
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(dust_source=ts)


def _attach_oxidants(forcing, forcing_cfg, coords):
    """Attach the oxidant climatology from ``cfg.forcing.oxidants_file``.

    No-op when unset. Loads a HAMMOZ/MACC-style monthly
    ``OH/NO3/O3/H2O2_VMR_avrg (time, mlev, lat, lon)`` mole-fraction
    climatology on ECHAM hybrid model levels into ``forcing.oxidant_vmr`` as
    ``WRAP_YEAR`` ``TimeSeries`` leaves; :class:`PrescribedOxidants` converts
    VMR → molec cm⁻³ in-term where T and p are available. The file's levels
    are mapped **one-to-one** onto the model levels: the level count is
    asserted in :func:`jcm.forcing.read_oxidant_vmr`, and when the model runs
    hybrid vertical coordinates the file's ``hyam``/``hybm`` are additionally
    cross-checked against the model's coefficients here, so a file on
    different 47 levels can't be wired in silently. Horizontal grid handling
    as in :func:`_attach_dms`.

    Supports the same yearly-expansion the anthropogenic emissions path does, so
    the year-matched transient oxidant product recommended for a transient run
    (``oxidants_file=.../{year}.nc`` with ``forcing.years``) actually loads: a
    ``{year}`` pattern expands to one file per year, which are concatenated
    along the time axis (``open_mfdataset``, by-coords) and read with ``auto``
    alignment — a single 12-month climatology stays ``WRAP_YEAR`` while a
    multi-year axis becomes ``BY_DATE``. The level-for-level vertical mapping
    is unchanged (the yearly files share the model's hybrid grid).

    Oxidants are handled as **one product**, unlike the per-product emissions
    path. A user-supplied **list** ``oxidants_file`` means the yearly files of a
    *single* oxidant product (exactly what ``{year}`` expansion produces): the
    whole file set is opened together (``open_mfdataset``, by-coords) along one
    time axis and read once. This differs deliberately from
    :func:`_attach_emissions`, whose per-product merge is meaningful because
    emission products carry **disjoint** variables (different sectors/species).
    :func:`jcm.forcing.read_oxidant_vmr` instead requires *every* oxidant file
    to carry **all four** gases (oh/no3/o3/h2o2), so distinct products fully
    overlap — a per-product ``dict.update`` would be pure last-one-wins and
    silently keep only the final file. Genuinely incompatible members in one
    file set (e.g. an integer-month climatology mixed with datetime transients)
    are rejected up front by :func:`_assert_uniform_oxidant_time_axis` rather
    than left to ``open_mfdataset`` to NaN-fill or clash cryptically.
    """
    if forcing_cfg is None:
        return forcing
    raw = forcing_cfg.get("oxidants_file", None)
    if raw in (None, "", "null"):
        return forcing
    years = forcing_cfg.get("years", None)
    available = forcing_cfg.get("available_years", None)

    import xarray as xr
    from omegaconf import ListConfig

    from jcm.forcing import read_oxidant_vmr, validate_oxidant_levels
    lat_deg, lon_deg = _model_latlon_deg(coords)
    nlev = int(coords.nodal_shape[0])

    # A `{year}` pattern expands to the product's yearly file list; an explicit
    # list is taken verbatim as that one product's file set. Either way the set
    # shares a single time axis and is read once (see docstring).
    files = _resolve_data_path(_expand_years(raw, years, available))
    if isinstance(files, (list, tuple, ListConfig)):
        paths = [str(p) for p in files
                 if str(p) not in ("", "null", "none", "None")]
    elif files in (None, "", "null"):
        return forcing
    else:
        paths = [str(files)]
    if not paths:
        return forcing
    _assert_uniform_oxidant_time_axis(paths)
    ds = (xr.open_mfdataset(paths, combine="by_coords") if len(paths) > 1
          else xr.open_dataset(paths[0]))
    ref = paths if len(paths) > 1 else paths[0]
    try:
        mapping = read_oxidant_vmr(ds, nlev=nlev, lat_deg=lat_deg,
                                   lon_deg=lon_deg, align_mode="auto")
        validate_oxidant_levels(ds, coords, ref)
    finally:
        ds.close()
    if not mapping:
        return forcing
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(oxidant_vmr=mapping)


def _assert_uniform_oxidant_time_axis(paths) -> None:
    """Reject an oxidant file set whose members carry incompatible time axes.

    A list ``oxidants_file`` (or a ``{year}`` expansion) is the yearly files of
    ONE product, opened together along a single time axis (see
    :func:`_attach_oxidants`). Mixing an integer-month climatology (numeric
    ``time``) with a datetime transient in that set would either silently
    NaN-fill the non-overlapping steps under ``open_mfdataset(by_coords)`` or
    clash cryptically on dtype — the silent-ignore class this hardening
    abolishes. Classify each member's ``time`` axis and raise a targeted error,
    naming both axes and the supported forms, when the set is mixed. A
    single-file set (a lone climatology, the common case) is trivially uniform.
    """
    if len(paths) <= 1:
        return
    import numpy as np
    import xarray as xr
    kinds: dict[str, list[str]] = {}
    for p in paths:
        with xr.open_dataset(p) as ds:
            if "time" not in ds.variables and "time" not in ds.dims:
                continue
            dt = np.asarray(ds["time"].values).dtype
            # datetime64 or object (decoded cftime) => a real calendar axis;
            # anything numeric is an integer/float month-index climatology.
            is_datetime = np.issubdtype(dt, np.datetime64) or dt == np.object_
            kind = "datetime" if is_datetime else "integer-month"
            kinds.setdefault(kind, []).append(p)
    if len(kinds) > 1:
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))
        raise ValueError(
            "forcing.oxidants_file mixes incompatible time axes within one "
            f"file set ({detail}). A list (or a {{year}} expansion) is the "
            "yearly files of a SINGLE oxidant product and must share one time "
            "axis. Supported forms: a single 12-month climatology file, or one "
            "transient product's yearly files (all on a datetime axis). Do not "
            "mix an integer-month climatology with datetime transient files."
        )


def _attach_macv2_weights(forcing, forcing_cfg, coords):
    """Attach time-varying MACv2-SP plume weights from ``forcing.macv2_file``.

    No-op when unset. Loads the ``year_weight``/``ann_cycle`` scalings from a
    ``MACv2.0-SP_v1.nc`` file (via :func:`jcm.forcing.read_macv2_weights`) onto
    ``forcing.aerosol_year_weight`` / ``aerosol_ann_cycle`` — the fields the
    MACv2-SP aerosol term reads for per-year amplitude and the seasonal cycle.
    Without a file these default to all-ones (perpetual year-2005 amplitude, no
    seasonal cycle); ``forcing=macv2_sp`` sets the key. The weights are plume-
    indexed and grid-independent, so no horizontal regridding is needed here.
    """
    if forcing_cfg is None:
        return forcing
    path = _resolve_data_path(forcing_cfg.get("macv2_file", None))
    if path in (None, "", "null"):
        return forcing
    from jcm.forcing import read_macv2_weights
    year_weight, ann_cycle = read_macv2_weights(str(path))
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(aerosol_year_weight=year_weight,
                        aerosol_ann_cycle=ann_cycle)


# ---------------------------------------------------------------------------
# Run + save
# ---------------------------------------------------------------------------

def maybe_enable_compilation_cache() -> None:
    """Enable JAX's persistent compilation cache (#592) — on by default.

    Safe to share across code edits: entries are keyed on the compiled HLO
    plus backend/jaxlib, so a source change that alters the computation
    *misses* rather than wrongly hits — the failure mode is recompilation,
    never staleness. Benchmarks discard the compile chunk deliberately, so
    caching only shortens their spin-up.

    ``JCM_CACHE_DIR`` relocates the cache; set it to ``off`` (or ``0`` /
    ``none``) to disable. Default: ``$SCRATCH/jcm-jax-cache`` when
    ``SCRATCH`` is set (fast scratch on HPC), else ``~/.cache/jcm/jax``.
    """
    val = os.environ.get("JCM_CACHE_DIR", "")
    if val.lower() in ("0", "off", "none", "false"):
        return
    if val:
        cache_dir = val
    elif os.environ.get("SCRATCH"):
        cache_dir = os.path.join(os.environ["SCRATCH"], "jcm-jax-cache")
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "jcm",
                                 "jax")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    logger.info("JAX persistent compilation cache: %s", cache_dir)


def run(cfg: DictConfig, model: Model | None = None):
    """Dispatch to the appropriate runtime mode.

    ``cfg.run.mode`` selects between:

    * ``full`` — the standard dynamical-core integration (``Model.run`` /
      ``Model.resume``). Honours ``cfg.init.kind`` and ``cfg.run.chunk_days``.
    * ``prescribed`` — load a full-grid state time series from
      ``cfg.run.state_file`` and run :class:`PrescribedStateModel`. No
      dynamical core; just diagnostic physics tendencies per snapshot.
    * ``scm`` — load a state time series, slice the column nearest to
      ``cfg.run.column.{lat_deg,lon_deg}``, and run :class:`SingleColumnModel`
      for tracer evolution at that column.
    """
    # Best-effort raise of the CPU device count, looked up from
    # ``grid.host_device_count`` with a top-level ``host_device_count``
    # fallback; no-op on a single device or on GPU. Note that importing the
    # model stack already initialises the JAX backend, so under the CLI this
    # can no longer change the count — it then only validates that the running
    # device count matches and warns (pointing at the ``XLA_FLAGS`` env var,
    # which is the reliable lever before the process starts). The ``spmd_mesh``
    # product must equal the device count either way.
    configure_host_device_count(
        cfg.get("host_device_count", None)
        or cfg.get("grid", {}).get("host_device_count", None)
    )
    maybe_enable_compilation_cache()

    # Apply any physical-constant overrides BEFORE the model is built, so the
    # dynamical core (which reads the live jcm.constants singleton at
    # construction) and the attribute-access physics both pick them up. Only base
    # fields may be set; derived constants (rd, cvd, rgrav, vtmpc*) follow.
    # Reset the provenance registries before the model build so every
    # input file the build resolves lands in them (#591); the code/env
    # probe and the summary log happen after the build, once the
    # config-selected libraries are actually imported.
    provenance.start_run(cfg)

    if cfg.init.kind == "from_state":
        # Resolve before the (minutes-scale at high resolution) model build
        # so a typo'd path fails immediately and names the config key.
        _p = _resolve_data_path(cfg.init.file)
        if not str(_p).startswith("hf://") and not Path(_p).exists():
            raise FileNotFoundError(
                f"init.file={cfg.init.file!r} resolved to {_p}, which does "
                "not exist."
            )

    constants_overrides = cfg.get("constants", None)
    if constants_overrides:
        import jcm.constants as _jcm_constants
        _jcm_constants.set_constants(
            **{k: float(v) for k, v in dict(constants_overrides).items()}
        )

    mode = cfg.run.get("mode", "full")
    if mode == "full":
        return _run_full(cfg, model)
    if mode == "prescribed":
        return _run_prescribed(cfg)
    if mode == "scm":
        return _run_scm(cfg)
    raise ValueError(
        f"Unknown run.mode={mode!r}; expected 'full', 'prescribed' or 'scm'."
    )


def _has_jam(physics) -> bool:
    """Report whether the physics package carries the JAM aerosol chain.

    Every JAM term names itself ``jam_*`` (emissions, deposition, chemistry,
    microphysics, optics, cloud-borne exchange); the two activation/reset
    helpers do not, but they never appear without the rest of the chain. So a
    single ``jam_``-prefixed term is a reliable, config-style detector that does
    not need to reach into ``cfg.physics`` (which only exists for the
    factory-built presets, not term-list ones).
    """
    return any(getattr(t, "name", "").startswith("jam_") for t in physics.terms)


def _has_macv2sp(physics) -> bool:
    """Report whether the MACv2-SP simple-plumes aerosol term is present."""
    return any(getattr(t, "name", "") == "macv2_sp_aerosol"
               for t in physics.terms)


def warn_on_config_traps(cfg: DictConfig, physics, forcing) -> None:
    """Warn (never raise) about config combinations that run but mislead.

    Config-layer cross-validation belongs in the runner (#640): it reads the
    composed ``cfg`` plus the already-built ``physics``/``forcing`` objects and
    calls no science. Every finding here is a :func:`logging.Logger.warning`,
    not an error — the combinations all *run*, they just quietly produce
    something other than what the config name suggests, and the maintainer
    chose to keep them runnable (e.g. for controlled idealized experiments).

    ``forcing`` may be ``None`` (``forcing.kind: default``, or the SCM/
    prescribed paths that build none): the aquaplanet ``default_forcing`` the
    model then falls back to carries the same all-ones MACv2-SP weights, so it
    is treated as the all-ones case for warning 4.
    """
    import numpy as np

    from jcm.forcing import TimeSeries

    terrain_kind = cfg.get("terrain", {}).get("kind", None)
    forcing_kind = cfg.get("forcing", {}).get("kind", None)
    has_jam = _has_jam(physics)

    # 1. Prognostic aerosol over a flat all-ocean planet: Gong sea-salt emits
    #    everywhere (including where land should be), there is no orography to
    #    source dust, and the idealized cos²-lat SSTs are not a real surface.
    if has_jam and terrain_kind == "aquaplanet":
        logger.warning(
            "config trap: JAM prognostic aerosol with terrain=aquaplanet — a "
            "flat all-ocean planet has no land-sea mask, so Gong sea-salt "
            "emission fires over cells that should be land and there is no "
            "orography to source dust. Use terrain=auto (native-grid mask) or "
            "terrain=from_file for a realistic surface."
        )

    # 2. The inverse mismatch (#640): a real-world boundary file's land-sea
    #    mask over aquaplanet terrain — SSTs land on cells the terrain calls
    #    ocean and vice-versa.
    if terrain_kind == "aquaplanet" and forcing_kind == "from_file":
        logger.warning(
            "config trap: forcing.kind=from_file over terrain=aquaplanet — the "
            "boundary file's real-world SST/land fields carry a land-sea mask "
            "that disagrees with the flat all-ocean terrain. Pair from_file "
            "forcing with terrain=from_file (or terrain=auto) so the two masks "
            "agree (issue #640)."
        )

    # 3. Prognostic aerosol with every prescribed-emission input nulled: only
    #    online Gong sea-salt has a source. After commit 3 the JAM presets
    #    supply the bundles by default, so this fires only when a user has
    #    explicitly nulled them.
    if has_jam:
        forcing_cfg = cfg.get("forcing", {})
        emission_keys = ("emissions_file", "dms_file", "dust_file",
                         "oxidants_file")
        unset = [k for k in emission_keys
                 if forcing_cfg.get(k, None) in (None, "", "null", "none")]
        if len(unset) == len(emission_keys):
            logger.warning(
                "config trap: zero-emission JAM baseline — %s are all unset, so "
                "the only online aerosol source is Gong sea salt; sulfur, dust "
                "and carbonaceous species stay at zero. Leave them at their "
                "'auto' default (the per-grid HF bundles) or set an explicit "
                "path (e.g. forcing.emissions_file=hf://bundles/<grid>/"
                "emissions_pd.nc).",
                ", ".join(unset),
            )

    # 4. MACv2-SP driven by the all-ones default weights: perpetual year-2005
    #    plume amplitude with no seasonal cycle — not historical forcing. Only
    #    for a pure MACv2-SP run (the echam* default); on the JAM path MACv2-SP
    #    is a passive optics fudge whose weights are not the concern.
    def _is_allones_static(x) -> bool:
        # A loaded MACv2 timeseries is a ``TimeSeries`` leaf; the untouched
        # default is a plain all-ones array (ForcingData.zeros).
        if isinstance(x, TimeSeries):
            return False
        return bool(np.allclose(np.asarray(x), 1.0))

    if _has_macv2sp(physics) and not has_jam:
        # forcing=None → the aquaplanet default_forcing, all-ones weights.
        all_ones = forcing is None or (
            _is_allones_static(forcing.aerosol_year_weight)
            and _is_allones_static(forcing.aerosol_ann_cycle))
        if all_ones:
            logger.warning(
                "config trap: MACv2-SP with the default all-ones "
                "aerosol_year_weight/aerosol_ann_cycle — this is perpetual "
                "year-2005 plume amplitude with no seasonal cycle, not "
                "historical aerosol forcing. Use forcing=macv2_sp (with "
                "macv2_file set) for real time-varying MACv2-SP weights."
            )

    # 5. Transient (by-date) surface forcing driving JAM off the present-day
    #    emission bundles. amip/era5 (forcing/{amip,era5}.yaml) are transient
    #    from_file presets — one file per year, ``align: by_date_interp``, a
    #    required ``years`` range — so the run's SST/sea-ice track real calendar
    #    dates. But ``emissions_file``/``oxidants_file: auto`` resolve the
    #    *present-day* ``*_pd`` bundle (``_resolve_one_emission_input`` — the
    #    ``dms``/``dust`` climatologies are natural and roughly time-invariant,
    #    so only these two anthropogenic products are the concern). The result
    #    is a historical/AMIP circulation breathing present-day aerosol
    #    emissions. Fires only when ``auto`` is still in place (an explicit
    #    year-matched path silences it); on the spectral+JAM path ``auto`` is
    #    exactly what resolved to the ``_pd`` bundle.
    if has_jam:
        forcing_cfg = cfg.get("forcing", {})
        years = forcing_cfg.get("years", None)
        align = str(forcing_cfg.get("align", "") or "")
        is_transient = bool(years) or align in ("by_date", "by_date_interp")
        pd_auto_keys = [k for k in ("emissions_file", "oxidants_file")
                        if forcing_cfg.get(k, None) == "auto"]
        if is_transient and pd_auto_keys:
            logger.warning(
                "config trap: transient (by-date) forcing with present-day JAM "
                "emissions — the surface forcing tracks real calendar dates "
                "(amip/era5: per-year files, by_date_interp) but %s are still "
                "'auto', which resolved to the present-day *_pd emission "
                "bundles. A historical/AMIP run is therefore using present-day "
                "aerosol emissions. For emissions, override with the mirror's "
                "transient product using a year-matched {year} pattern (the "
                "same yearly-file expansion the SST forcing uses) — e.g. "
                "forcing.emissions_file=hf://bundles/<grid>/emissions_amip/"
                "{year}.nc, with the run's forcing.years range. The mirror "
                "publishes NO transient oxidants product (only oxidants_pi/"
                "oxidants_pd climatologies), so transient oxidants must come "
                "from a separately prepared dataset; forcing.oxidants_file "
                "accepts a {year} pattern once you have one.",
                ", ".join(pd_auto_keys),
            )


def _run_full(cfg: DictConfig, model: Model | None = None) -> ModelPredictions:
    if model is None:
        model = build_model(cfg)

    forcing = build_forcing(cfg, model.coords, dycore=getattr(model, "dycore", None))
    forcing = _maybe_attach_nudging_target(forcing, cfg, model)
    guard_emulator_ghg_forcing(model.physics, forcing)
    warn_on_config_traps(cfg, model.physics, forcing)
    # After model + forcing construction: config-selected libraries are
    # imported and the ozone source is decided, so the summary is accurate.
    logger.info("provenance: %s", provenance.summary())
    chunk_days = float(cfg.run.get("chunk_days", 0.0) or 0.0)
    if chunk_days > 0:
        return run_chunked(
            cfg,
            chunk_days=chunk_days,
            output_prefix=cfg.run.get("output_prefix", "chunked_run"),
            model=model,
            forcing=forcing,
        )

    if cfg.init.kind == "isothermal":
        return model.run(
            forcing=forcing,
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
            snapshot_interval=cfg.run.get("snapshot_interval"),
            snapshot_variables=tuple(cfg.run.get("snapshot_variables") or ()),
        )
    # A warm start (from_state) carries the donor's physics carry through to
    # ``run``; every other init builds a fresh carry (initial_physics_state
    # stays None).
    initial_physics_state = None
    if cfg.init.kind == "jw":
        initial_state = jw_state(model, rh=float(cfg.init.get("rh", 0.6)))
    elif cfg.init.kind == "balanced_isothermal":
        initial_state = balanced_isothermal_state(model)
    elif cfg.init.kind == "from_state":
        initial_state, initial_physics_state = _state_from_file(model, cfg)
    elif cfg.init.kind == "era5":
        initial_state = _state_from_era5(model, cfg)
    else:
        raise ValueError(f"Unknown init.kind={cfg.init.kind!r}")
    return model.run(
        initial_state=initial_state,
        initial_physics_state=initial_physics_state,
        forcing=forcing,
        save_interval=cfg.run.save_interval,
        total_time=cfg.run.total_time,
        output_averages=cfg.run.output_averages,
        snapshot_interval=cfg.run.get("snapshot_interval"),
        snapshot_variables=tuple(cfg.run.get("snapshot_variables") or ()),
    )


def _load_states_from_cfg(cfg: DictConfig, physics):
    """Open ``cfg.run.state_file`` and return a stacked ``PhysicsState``.

    ``state_file`` is a netCDF from a previous JCM run, i.e. surface-first;
    ``load_states_from_xarray`` detects that and returns a top-first
    physics-frame state (#741), which is what the SCM / prescribed-state
    runners expect, and rejects a file whose orientation disagrees with its
    own pressures rather than handing physics an inverted column (#718).

    ``physics`` is the configured package (``None`` for a caller that has
    none). Which tracers to load comes from it: with ``run.tracer_vars``
    unset, every tracer the physics declares via ``required_tracers()`` and
    that the file actually carries is loaded. Without that the condensate a
    saved state holds was silently dropped, so cloud-aware physics ran
    against a clear sky — the second half of #718. ``run.tracer_vars: {}``
    opts out explicitly; an explicit mapping still wins outright.
    """
    state_file = _resolve_data_path(cfg.run.get("state_file", None))
    if not state_file:
        raise ValueError(
            f"run.mode={cfg.run.mode!r} requires run.state_file to point "
            "at a netCDF written by a previous JCM run."
        )
    import xarray as xr
    from omegaconf import OmegaConf
    from jcm.utils import load_states_from_xarray

    tracer_vars = cfg.run.get("tracer_vars", None)
    if tracer_vars is not None:
        # Keep an explicit empty mapping distinct from ``null``: the former
        # means "no tracers", the latter "take them from the physics".
        tracer_vars = OmegaConf.to_container(tracer_vars, resolve=True)
    ds = xr.open_dataset(state_file)
    return ds, load_states_from_xarray(
        ds,
        tracer_vars=tracer_vars,
        required_tracers=(
            physics.required_tracers() if physics is not None else None),
    )


def _run_prescribed(cfg: DictConfig):
    """Diagnose physics tendencies from a JCM state-file time series."""
    from jcm.prescribed_state_model import PrescribedStateModel

    coords = build_coords(cfg)
    physics = build_physics(cfg)
    terrain = build_terrain(cfg, coords)
    forcing = build_forcing(cfg, coords)
    guard_emulator_ghg_forcing(physics, forcing)
    warn_on_config_traps(cfg, physics, forcing)
    _, states = _load_states_from_cfg(cfg, physics)

    model = PrescribedStateModel(
        physics=physics,
        coords=coords,
        terrain=terrain,
        dt_seconds=float(cfg.run.time_step) * 60.0,
    )
    return model.run(states, forcing=forcing)


#: Nearest-column selection; the science lives in
#: :func:`jcm.single_column_model.select_column`. Aliased for the SCM runner.
_select_column = select_column


def _run_scm(cfg: DictConfig):
    """Run the single-column model on the column nearest to the user's lat/lon."""
    from jcm.single_column_model import SingleColumnModel

    column_cfg = cfg.run.get("column", None)
    if column_cfg is None:
        raise ValueError(
            "run.mode='scm' requires run.column.{lat_deg,lon_deg} to pick the column."
        )
    lat_deg = float(column_cfg.lat_deg)
    lon_deg = float(column_cfg.lon_deg)

    physics = build_physics(cfg)
    # Build coords just to grab the vertical coord; horizontal grid is unused.
    coords = build_coords(cfg)
    # The SCM builds no ForcingData; pass None so the config-trap check still
    # covers this mode (terrain/emission-key traps read cfg, and the None
    # forcing is the all-ones MACv2-SP default for warning 4).
    warn_on_config_traps(cfg, physics, None)
    ds, states = _load_states_from_cfg(cfg, physics)
    column_states, (i_lon, i_lat, actual_lat, actual_lon) = _select_column(
        states, ds, lat_deg=lat_deg, lon_deg=lon_deg,
    )
    logger.info(
        "SCM: requested (lat=%.2f, lon=%.2f) → grid cell (i_lon=%d, i_lat=%d) "
        "at (lat=%.2f, lon=%.2f)",
        lat_deg, lon_deg, i_lon, i_lat, actual_lat, actual_lon,
    )

    scm = SingleColumnModel(
        physics=physics,
        vertical=coords.vertical,
        lat_deg=actual_lat,
        lon_deg=actual_lon,
        dt_seconds=float(cfg.run.time_step) * 60.0,
    )
    return scm.run(column_states)


def run_chunked(
    cfg: DictConfig,
    chunk_days: float,
    output_prefix: str,
    model: Model | None = None,
    forcing=None,
):
    """Long-running integration broken into ``chunk_days``-day pieces.

    Each chunk is dumped to ``{output_prefix}_day{N}.nc`` and run through
    ``jcm.diagnostics.check_health``. The loop stops early on the first
    failed health check. Returns the per-chunk reports.

    When ``cfg.run.checkpoint_path`` is set, the model state and elapsed
    sim-day count are persisted after each chunk and (if the file
    already exists at startup) restored before the loop begins, so a
    preempted run resumes at the chunk boundary it last reached without
    redoing the integration. See :mod:`jcm.checkpoint` and issue #128.
    """
    import time

    from jcm.diagnostics import (
        aerosol_budget_report,
        check_health,
        print_report,
    )

    if model is None:
        model = build_model(cfg)
    if forcing is None:
        forcing = build_forcing(cfg, model.coords, dycore=getattr(model, "dycore", None))

    save_interval = float(cfg.run.save_interval)
    total_time = float(cfg.run.total_time)

    ckpt_path = cfg.run.get("checkpoint_path", None)

    reports: list[dict] = []
    elapsed_sim_days = 0.0
    total_wall = 0.0
    resumed_from_ckpt = False

    if ckpt_path and Path(ckpt_path).exists():
        from jcm.checkpoint import load_checkpoint

        # Build state templates without integrating so flax.serialization
        # has pytrees of the right shape and dtype to deserialize against.
        # Mirrors the init-kind branching of the fresh-start path below;
        # the template values are immediately overwritten by the
        # checkpoint's contents.
        # ``bootstrap_state`` populates both ``_final_dycore_state`` and the
        # physics carry (eagerly), which ``load_checkpoint`` needs as
        # deserialization templates; their values are immediately overwritten
        # by the checkpoint's contents, so the init-kind only decides the
        # template's pytree structure.
        if cfg.init.kind == "jw":
            model.bootstrap_state(jw_state(model, rh=float(cfg.init.get("rh", 0.6))))
        elif cfg.init.kind == "balanced_isothermal":
            model.bootstrap_state(balanced_isothermal_state(model))
        else:
            model.bootstrap_state()

        elapsed_sim_days = load_checkpoint(model, ckpt_path)
        resumed_from_ckpt = True
        print(
            f"Resumed from checkpoint {ckpt_path} at sim-day "
            f"{elapsed_sim_days:.1f}"
        )

    chunk_idx = int(elapsed_sim_days // chunk_days)
    started_at_days = elapsed_sim_days
    while elapsed_sim_days < total_time:
        cur_chunk = min(chunk_days, total_time - elapsed_sim_days)
        if cur_chunk <= 0:
            break

        t0 = time.perf_counter()
        first_fresh_chunk = chunk_idx == 0 and not resumed_from_ckpt
        if first_fresh_chunk:
            # First fresh chunk: bootstrap from the configured initial state
            # and integrate. ``model.run`` = bootstrap_state + resume, so the
            # cross-step physics carry is built exactly as the plain
            # (isothermal) path's ``model.run`` does. ``init=era5`` must be
            # handled here too: the chunked dispatch returns before
            # ``_run_full``'s init ladder runs.
            # from_state warm starts thread the donor's physics carry into
            # the first chunk's ``run``; all other inits build a fresh carry.
            initial_physics_state = None
            if cfg.init.kind == "jw":
                initial_state = jw_state(model, rh=float(cfg.init.get("rh", 0.6)))
            elif cfg.init.kind == "balanced_isothermal":
                initial_state = balanced_isothermal_state(model)
            elif cfg.init.kind == "era5":
                initial_state = _state_from_era5(model, cfg)
            elif cfg.init.kind == "from_state":
                initial_state, initial_physics_state = _state_from_file(model, cfg)
            else:
                initial_state = None
            preds = model.run(
                initial_state=initial_state,
                initial_physics_state=initial_physics_state,
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
                snapshot_interval=cfg.run.get("snapshot_interval"),
                snapshot_variables=tuple(
                    cfg.run.get("snapshot_variables") or ()),
            )
        else:
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
                snapshot_interval=cfg.run.get("snapshot_interval"),
                snapshot_variables=tuple(
                    cfg.run.get("snapshot_variables") or ()),
            )

        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            preds._predictions,
        )
        chunk_wall = time.perf_counter() - t0
        total_wall += chunk_wall
        elapsed_sim_days += cur_chunk

        ds = preds.to_xarray()
        ok, report = check_health(ds, chunk_idx, elapsed_sim_days)
        report["wall_seconds"] = chunk_wall
        reports.append(report)
        print_report(report)

        # #713: one greppable aerosol-budget line per species per chunk
        # (jcm.diagnostics.aerosol_budget_report). pySES presets omit
        # run.time_step (the Model adopts the dycore's dt); the closure
        # floor is an order-of-magnitude guide, so a nominal 900 s stands
        # in rather than reaching into the dycore from here.
        _dt_cfg = cfg.get("run", {}).get("time_step", None)
        _dt_s = float(_dt_cfg) * 60.0 if _dt_cfg else 900.0
        for _line in aerosol_budget_report(ds, _dt_s):
            print(_line)

        nc_path = f"{output_prefix}_day{int(elapsed_sim_days)}.nc"
        # The parameters ride on the predictions object, not the module
        # registry, so the record belongs to the model that produced THIS
        # chunk; pass them to both calls or the sidecar's run_hash will not
        # match the one in the attributes.
        params = getattr(preds, "params", None)
        ds.attrs.update(provenance.attrs(params))
        ds.attrs["jcm_prov_chunk_wall_seconds"] = round(chunk_wall, 1)
        ds.to_netcdf(nc_path)
        provenance.write_sidecar(nc_path, params)
        print(f"  Saved {nc_path}")
        snap_ds = getattr(preds, "snapshot_dataset", lambda: None)()
        if snap_ds is not None:
            snap_path = (f"{output_prefix}_day{int(elapsed_sim_days)}"
                         "_snapshots.nc")
            snap_ds.attrs.update(provenance.attrs(params))
            snap_ds.to_netcdf(snap_path)
            print(f"  Saved {snap_path}")

        # Checkpoint only after a PASSING health check, keeping the previous
        # checkpoint as ``.prev``: an unhealthy chunk must not overwrite the
        # only restartable state (with save-interval-averaged outputs the
        # netCDFs cannot reconstruct one — the first ne30 campaign lost a
        # 120-day run to exactly this ordering). ``archive_ckpt_every`` (days,
        # 0 = off) additionally keeps permanent copies so a later experiment
        # can restart from before a slowly-developing failure, not just from
        # the last two chunk boundaries.
        if ckpt_path and ok:
            from jcm.checkpoint import save_checkpoint

            cp = Path(ckpt_path)
            if cp.exists():
                cp.replace(f"{ckpt_path}.prev")
            save_checkpoint(model, ckpt_path, elapsed_days=elapsed_sim_days)
            print(f"  Saved checkpoint to {ckpt_path}")
            archive_every = float(cfg.run.get("archive_ckpt_every", 0.0) or 0.0)
            if archive_every > 0 and abs(elapsed_sim_days % archive_every) < 1e-6:
                import shutil

                archive = f"{output_prefix}_day{int(elapsed_sim_days)}.ckpt"
                shutil.copyfile(ckpt_path, archive)
                print(f"  Archived checkpoint {archive}")
        elif ckpt_path:
            print("  Checkpoint NOT updated (unhealthy chunk) — restart from "
                  f"{ckpt_path}")

        if not ok:
            # Honour ``run.bail_on_unhealthy`` (default True). The full-year
            # T63L47 ECHAM-1M run hits a single-column q-max excursion at
            # day 30 that doesn't propagate globally — bailing on the first
            # such excursion truncates a usable year of climatology to a
            # single chunk. With the flag set to False, log a warning and
            # keep going so we still get the rest of the integration.
            bail = bool(cfg.run.get("bail_on_unhealthy", True))
            msg = (
                f"\n*** atmosphere unhealthy at "
                f"day {elapsed_sim_days:.0f}: {report.get('reasons', [])} ***"
            )
            if bail:
                print(msg + "\nSTOPPING.")
                break
            print(msg + "\nContinuing (bail_on_unhealthy=False).")

        # Throughput is reported over the post-resume window so the
        # number reflects the run actually happening on this host.
        days_this_invocation = elapsed_sim_days - started_at_days
        if total_wall > 0:
            sdph = days_this_invocation / (total_wall / 3600)
            print(
                f"  Wall: {chunk_wall:.1f}s this chunk, {total_wall:.0f}s total "
                f"({sdph:.0f} sim days/hr)"
            )

        chunk_idx += 1

    return reports


def resolve_output_path(cfg: DictConfig, hydra_cfg: Any) -> Path:
    """Compute the netCDF output path, mirroring the legacy main.py behaviour."""
    output_name = cfg.run.get("output", "model_state.nc")
    if Path(output_name).is_absolute():
        return Path(output_name)

    base_dir = Path("outputs") / hydra_cfg.run.dir.split("outputs/")[-1]
    if str(hydra_cfg.mode) == "RunMode.MULTIRUN":
        out_dir = base_dir / "multirun" / str(hydra_cfg.job.num)
    else:
        out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / output_name


def save_predictions(predictions, output_path: Path) -> None:
    """Persist a run's outputs.

    ``run_chunked`` already writes one netCDF per chunk and returns the
    list of health-check reports. Skip the final dump in that case (the
    list of dicts has no ``to_xarray`` method, and the per-chunk files
    are the actual data).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(predictions, list):
        logger.info(
            "Chunked run: per-chunk netCDFs already written; skipping "
            "aggregate save_predictions for %s", output_path,
        )
        return
    params = getattr(predictions, "params", None)
    ds = predictions.to_xarray()
    ds.attrs.update(provenance.attrs(params))
    ds.to_netcdf(str(output_path))
    provenance.write_sidecar(output_path, params)
    logger.info("Wrote %s", output_path)
    # Interval-instantaneous snapshot stream (jax-gcm#586): a separate
    # file with its own (finer) time axis — folding a second cadence into
    # the main dataset would force a ragged time dimension.
    snap_ds = getattr(predictions, "snapshot_dataset", lambda: None)()
    if snap_ds is not None:
        snap_path = output_path.with_name(output_path.stem + "_snapshots.nc")
        snap_ds.attrs.update(provenance.attrs(params))
        snap_ds.to_netcdf(str(snap_path))
        logger.info("Wrote %s", snap_path)
