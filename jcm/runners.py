"""Build models and run simulations from a Hydra ``DictConfig``.

This is the bridge between the Hydra config groups in ``jcm/config/`` and the
construction of ``Model``, ``TerrainData``, ``DiffusionFilter`` and the various
physics packages. Keeps ``main.py`` minimal so other harnesses (notebooks,
integration tests) can import the same builders directly without going through
Hydra's CLI machinery.
"""

from __future__ import annotations

import logging
import os
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from jcm import provenance
from jcm.diffusion import ECHAM_LMIDATM_LAYERS, DiffusionFilter
from jcm.model import Model, ModelPredictions
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
        # ICON ships pre-tuned hybrid tables for 40 / 47 levels; for any
        # other count the user has to drop the table in by hand. Keep the
        # error chatty so the failure mode is obvious.
        from jcm.physics.echam.echam_levels import get_echam_levels
        try:
            vert = get_echam_levels(layers)
        except ValueError as exc:
            raise ValueError(
                f"hybrid coords with {layers} levels are not pre-configured. "
                "Use one of the supported counts (40, 47, 95) or extend "
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
        band_config=_band_config_for_terms(terms),
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


def _band_config_for_terms(terms):
    """Pick a ``RadiationBandConfig`` to match the active radiation backend.

    Walks the term list for an ``RRTMGPRadiation`` instance and reads its
    band centers; otherwise returns the broadband (single 550 nm SW band)
    fallback. Centralised here so every wavelength-dependent term — not
    just the aerosol scheme — sees the same band structure as whatever
    radiation backend is actually running. The band config is owned by
    ``ComposablePhysics`` and injected into ``diagnostics["_band_config"]``
    each step (same pattern as ``_dt_seconds``).
    """
    from jcm.physics.radiation.band_config import RadiationBandConfig
    from jcm.physics.radiation.rrtmgp import RRTMGPRadiation, _ensure_rrtmgp

    for t in terms:
        if isinstance(t, RRTMGPRadiation):
            return RadiationBandConfig.from_rrtmgp(_ensure_rrtmgp())
    return RadiationBandConfig.broadband()


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
    """Per-level inverse-timescale profiles from the ``nudging`` config.

    One ``1/tau`` value masked to zero (a) in the bottom ``pbl_levels``
    layers, and (b) above ``min_pressure_hpa`` — the WB2 ERA5 stores
    stop at 50 hPa, and values above that clamp, so relaxing the
    stratosphere toward them would drag it to 50-hPa winds.
    """
    import numpy as np
    if hasattr(vertical, "a_centers"):
        p_ref = (np.asarray(vertical.a_centers)
                 + np.asarray(vertical.b_centers) * 101325.0)
    else:
        p_ref = np.asarray(vertical.centers) * 101325.0
    nlev = p_ref.size
    mask = np.ones(nlev)
    mask[p_ref < float(nudging_cfg.get("min_pressure_hpa", 60.0)) * 100.0] = 0.0
    pbl = int(nudging_cfg.get("pbl_levels", 0))
    if pbl > 0:
        mask[nlev - pbl:] = 0.0
    inv_tau = mask / (float(nudging_cfg.get("tau_hours", 6.0)) * 3600.0)
    return inv_tau, nlev


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


def _expand_years(file_spec, years, available=None):
    """Expand a ``{year}`` file pattern into the yearly-bundle file list.

    The transient AMIP bundles are one file per year (issue #610:
    download only what you run, append new years without rewriting
    history), so config points at a pattern plus an inclusive range:
    ``file: hf://bundles/t63/forcing_amip/{year}.nc`` with
    ``years: [1979, 1983]``. A pattern without ``years`` raises rather
    than silently running with a literal ``{year}`` path. Non-pattern
    specs (plain paths, lists, ``None``) pass through untouched even when
    ``years`` is set — a run may mix yearly SST files with a static dust
    climatology, all sharing one ``forcing.years`` range.

    ``available`` (``forcing.available_years``, the product's inclusive
    source coverage) widens the expansion by one year on each side,
    clipped to that coverage: the yearly files hold *mid-month* samples,
    so a run starting Jan 1 needs the previous December's sample (and a
    run ending Dec 31 the next January's) for ``by_date_interp`` to
    bracket the boundary instead of clamping to the nearest mid-month
    value for ~half a month.
    """
    has_pattern = isinstance(file_spec, str) and "{year}" in file_spec
    if not has_pattern:
        return file_spec
    if years is None:
        raise ValueError(
            f"forcing file pattern {file_spec!r} contains {{year}} but "
            "no year range is set — add e.g. forcing.years=[1979,1983]")
    first, last = int(years[0]), int(years[-1])
    if last < first:
        raise ValueError(f"forcing.years range is reversed: {years!r}")
    if available is not None:
        lo, hi = int(available[0]), int(available[-1])
        first, last = max(first - 1, lo), min(last + 1, hi)
        # A requested range entirely outside coverage would invert here
        # and expand to nothing; clamp to the nearest edge file instead
        # (the time lookup then clamps to its first/last sample).
        first, last = min(first, hi), max(last, lo)
    return [file_spec.format(year=y) for y in range(first, last + 1)]


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
        # Match on the (vertical=hybrid, layers) pair so this fires for every
        # ECHAM-family grid at any truncation — and stays inert for SPEEDY
        # T31L8 / Held-Suarez, which have their own tuned uniform damping.
        if vertical == "hybrid" and layers in ECHAM_LMIDATM_LAYERS:
            base = DiffusionFilter.echam_lmidatm(truncation, layers)
        else:
            if vertical == "hybrid":
                logger.warning(
                    "diffusion.kind=auto found no ECHAM lmidatm profile for a "
                    "hybrid grid with %d levels, so this run uses the uniform "
                    "SPEEDY profile (24h temp / 12h vor_q / 2h div). ECHAM "
                    "profiles exist for %s levels. Set diffusion.kind "
                    "explicitly to silence this.",
                    layers, sorted(ECHAM_LMIDATM_LAYERS),
                )
            base = DiffusionFilter.default()
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

    # A level-dependent profile is a per-level array, so pinning one whose
    # length does not match the grid fails later inside the spectral filter as
    # an opaque broadcast error ("(95, 213, 108) vs (47,)"). Catch it here,
    # where the actual mismatch can be named (#579).
    n_orders = None if base.level_orders_temp is None else len(base.level_orders_temp)
    if n_orders is not None and layers and n_orders != layers:
        raise ValueError(
            f"diffusion.kind={kind!r} builds a {n_orders}-level hyperdiffusion "
            f"profile but the grid has {layers} levels. Use "
            "diffusion.kind=auto (or 'echam_lmidatm') to get the profile "
            "matching this grid, or 'default' for the uniform SPEEDY profile."
        )

    if scale == 1.0:
        return base
    return DiffusionFilter(
        div_timescale=base.div_timescale * scale,
        div_order=base.div_order,
        vor_q_timescale=base.vor_q_timescale * scale,
        vor_q_order=base.vor_q_order,
        temp_timescale=base.temp_timescale * scale,
        temp_order=base.temp_order,
        level_orders_div=base.level_orders_div,
        level_orders_vor_q=base.level_orders_vor_q,
        level_orders_temp=base.level_orders_temp,
    )


# ---------------------------------------------------------------------------
# Initial state injection (JW-style lapse-rate profile)
# ---------------------------------------------------------------------------

# Standard-atmosphere lapse rate and surface temperature for the JW init.
_JW_T_SFC = 288.0       # K, mid-latitude mean surface T
_JW_LAPSE = 6.5e-3      # K/m, ICAO standard tropospheric lapse rate
_JW_T_FLOOR = 250.0     # K, cold-tail cap so semi-implicit reference T stays
                        # close (dycore goes unstable for ΔT ~ 50 K).
# Reference temperature used for the column-mean hydrostatic balance applied
# to surface pressure over orography. ~ midpoint between troposphere and
# stratosphere — exact value matters very little for the surface-pressure
# field, but the nondimensionalisation is sensitive to changes here.
_HYDROSTATIC_T_REF = 260.0

# Tetens / Bolton coefficients for saturation vapour pressure over water.
_ES0 = 611.2     # Pa
_ES_A = 17.67
_ES_B = 29.65    # K offset
_T0_C = 273.15   # K, melting point reference

# Tropopause cap above which we set RH = 0 in the JW humidity profile.
_RH_CAP_PRESSURE_PA = 20000.0   # 200 hPa


def inject_balanced_isothermal_profile(model: Model) -> None:
    """Inject an isothermal-rest atmosphere with orography-balanced ``ps``.

    Same ps-rebalance logic as :func:`inject_jw_profile` (so air doesn't
    end up below ground over tall topography), but keeps the temperature
    field at a uniform 288 K and humidity at zero. Useful as a robust
    starting state for moist-physics runs over real terrain when the
    full JW lapse-rate profile is unstable at the chosen resolution.

    Mutates ``model._final_dycore_state`` in place. Follow with
    ``model.resume(...)`` rather than ``model.run(...)``.
    """
    from dinosaur.scales import units
    from jcm.constants import grav, p0s1_bg, rd

    model._final_dycore_state = model._prepare_initial_dycore_state(
        physics_state=None, random_seed=0,
    )
    state = model._final_dycore_state
    p0_pa = p0s1_bg

    orog = jnp.asarray(model.terrain.orog)
    if jnp.any(orog > 1.0):
        # Hydrostatic balance with the actual isothermal T (288 K), not
        # ``_HYDROSTATIC_T_REF`` (260 K which is appropriate for the
        # JW lapse-rate profile). Using the matching T avoids an
        # initial-step pressure-temperature inconsistency.
        ps_pa_nodal = p0_pa * jnp.exp(-grav * orog / (rd * _JW_T_SFC))
        scale = float(model.dycore.physics_specs.nondimensionalize(1.0 * units.pascal))
        log_ps_nodal = jnp.log(ps_pa_nodal * scale)
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            log_ps_nodal[None, ...]
        )
    model._final_dycore_state = state


def inject_jw_profile(model: Model, rh: float = 0.6) -> None:
    """Inject a Jablonowski-Williamson-style lapse-rate initial condition.

    Replaces ``model._final_dycore_state`` (set up by the default isothermal
    rest atmosphere) with a vertical profile suitable for moist physics:

    * Temperature: 288 K at the surface, ICAO standard lapse 6.5 K/km, capped
      at 250 K so the semi-implicit reference temperature stays close.
    * Surface pressure: hydrostatically balanced against the model's
      orography when present (otherwise the isothermal init places air below
      ground on tall mountains and the run blows up).
    * Humidity: ``rh`` × q_sat(T) below ~200 hPa, zero above; clipped to a
      sensible range for q.

    Mutates ``model._final_dycore_state`` in place. Follow with
    ``model.resume(...)`` rather than ``model.run(...)``.
    """
    from dinosaur.hybrid_coordinates import HybridCoordinates
    from dinosaur.scales import units

    from jcm.constants import grav, p0s1_bg, rd

    model._final_dycore_state = model._prepare_initial_dycore_state(
        physics_state=None, random_seed=0,
    )
    state = model._final_dycore_state

    nlon, nlat = model.coords.horizontal.nodal_shape
    p0_pa = p0s1_bg
    if isinstance(model.coords.vertical, HybridCoordinates):
        sigma = jnp.asarray(model.coords.vertical.get_sigma_centers(p0_pa))
    else:
        sigma = jnp.asarray(model.coords.vertical.centers)
    nlev = sigma.size

    # Hypsometric height for an isothermal column at T = 288 K. The scale
    # height H = R_d * T / g comes out to ~ 8400 m; we use it to convert
    # sigma to z so the lapse-rate profile can be evaluated.
    p = sigma * p0_pa
    scale_height = rd * _JW_T_SFC / grav
    z = scale_height * jnp.log(p0_pa / p)
    T_profile = jnp.maximum(_JW_T_SFC - _JW_LAPSE * z, _JW_T_FLOOR)

    # Hydrostatically rebalance surface pressure when there's nontrivial
    # orography, otherwise the isothermal-rest init produces air below ground.
    orog = jnp.asarray(model.terrain.orog)
    if jnp.any(orog > 1.0):
        ps_pa_nodal = p0_pa * jnp.exp(-grav * orog / (rd * _HYDROSTATIC_T_REF))
        scale = float(model.dycore.physics_specs.nondimensionalize(1.0 * units.pascal))
        log_ps_nodal = jnp.log(ps_pa_nodal * scale)
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            log_ps_nodal[None, ...]
        )

    # Humidity: rh * q_sat(T) below the tropopause cap, dry above.
    es = _ES0 * jnp.exp(_ES_A * (T_profile - _T0_C) / (T_profile - _ES_B))
    q_sat = 0.622 * es / jnp.maximum(p - es, 1.0)
    rh_profile = jnp.where(p > _RH_CAP_PRESSURE_PA, rh, 0.0)
    q_profile = jnp.clip(rh_profile * q_sat, 1e-8, 0.03)

    # Preserve the dry-balanced VIRTUAL temperature. The dynamical core's mass
    # field is driven by ``Tv = T*(1 + 0.61 q - q_cloud)`` (dinosaur
    # ``primitive_equations``). Injecting moisture onto the dry-balanced ``T``
    # raises Tv and breaks the hydrostatic balance the resting state was built
    # for, seeding a moisture-magnitude-dependent gravity-wave blow-up (rh=0.5
    # NaNs in ~3 h, rh=0.2 by day 2; the dry init is stable because Tv=T).
    # Lowering T so the moist Tv equals the dry-balanced value makes the
    # dynamics see the *identical*, stable state while the moisture is carried
    # transparently. The temperature change is tiny (~1 K at q~6 g/kg) but it
    # is exactly what restores the balance. Physics then evolves from a
    # consistent moist resting state.
    T_balanced_profile = T_profile / (1.0 + 0.61 * q_profile)

    T_ref = jnp.asarray(model.dycore.primitive.reference_temperature)
    T_var_profile = T_balanced_profile - T_ref
    T_var_nodal = jnp.broadcast_to(
        T_var_profile[:, None, None], (nlev, nlon, nlat)
    ).astype(state.temperature_variation.dtype)
    state.temperature_variation = model.coords.horizontal.to_modal(T_var_nodal)

    # Nondimensionalize the humidity exactly as the canonical physics→dynamics
    # bridge does (``state_bridge.physics_state_to_dynamics_state`` line ~149:
    # ``nondimensionalize(specific_humidity * gram/kilogram)``). The dynamics
    # ``State`` stores the *nondimensional* tracer; the forward bridge then
    # re-dimensionalizes with ``dimensionalize(q, gram/kilogram)`` (≈ ×1000)
    # when handing the gridpoint state to physics. Injecting the raw kg/kg
    # ``q_profile`` straight into ``state.tracers`` skipped this scaling, so
    # the physics saw ``q`` 1000× too large (~5 kg/kg) — the cloud saturation
    # adjustment (qs ~ 0.008) then read it as ~650× supersaturated, condensed
    # the whole column, and dumped L·Δq/cp ≈ 7000 K of latent heat in a single
    # step → instantaneous blow-up of every moist init. Mirroring the bridge's
    # nondimensionalization makes the gridpoint physics see the intended
    # ``q_profile`` value and the moist resting state is stable.
    q_nondim = model.dycore.physics_specs.nondimensionalize(
        q_profile * units.gram / units.kilogram
    )
    q_dtype = state.tracers["specific_humidity"].dtype
    q_nodal = jnp.broadcast_to(
        q_nondim[:, None, None], (nlev, nlon, nlat)
    ).astype(q_dtype)
    # Preserve the other prognostic tracers (qc, qi, qnc, qni, qr, qs, GHG VMRs,
    # aerosol modes, ...) that ``bootstrap_state`` seeded — only the JW analytic
    # humidity profile is injected here. Overwriting the whole dict used to drop
    # the cloud tracers, so radiation saw zero cloud water for the entire run
    # (CRE ≡ 0). Cloud water now persists and accumulates; the RRTMGP in-cloud
    # inflation that previously made this unstable is handled by the mo_psrad
    # in-cloud zeroing (mcica.in_cloud_path).
    state.tracers = {
        **state.tracers,
        "specific_humidity": model.coords.horizontal.to_modal(q_nodal),
    }
    model._final_dycore_state = state


def inject_state_file(model: Model, cfg: DictConfig) -> None:
    """Warm-start: load a saved model state as the initial condition.

    ``init.file`` takes a local or ``hf://`` path to a state written by
    :func:`jcm.checkpoint.save_checkpoint` (e.g. the hosted equilibrated
    states under ``bundles/<grid>_<levels>/init_states/``). Unlike a
    ``run.checkpoint_path`` resume, the recorded elapsed-day count is
    DISCARDED — the clock starts at zero / ``run.start_date`` — so a
    hosted state skips the ~9-month from-cold spin-up (#638) without
    inheriting the donor run's calendar. The state's pytree must match
    the composed model (grid, levels, physics tracer set);
    ``load_checkpoint`` fails loudly on any mismatch.
    """
    from jcm.checkpoint import load_checkpoint

    path = _resolve_data_path(cfg.init.file)
    ckpt = cfg.run.get("checkpoint_path", None)
    if ckpt and Path(ckpt).resolve() == Path(path).resolve():
        raise ValueError(
            "init.file and run.checkpoint_path point at the same file: the "
            "first chunk checkpoint would overwrite the donor init state. "
            "Give the run its own checkpoint_path."
        )
    # bootstrap_state builds both state pytrees, which load_checkpoint
    # needs as deserialization templates (their values are overwritten).
    model.bootstrap_state()
    days = load_checkpoint(model, path)
    # The checkpoint's dycore state carries the donor's sim_time, and dates,
    # forcing time-interpolation and output timestamps all derive from it
    # (Model._date_from_sim_time) — without this reset a day-730 donor
    # would run with forcing at start_date + 730 d.
    model._final_dycore_state = model.dycore.with_sim_time(
        model._final_dycore_state,
        jnp.zeros_like(model.dycore.sim_time(model._final_dycore_state)),
    )
    logger.info(
        "init=from_state: loaded %s (donor state carried %.0f sim-days); "
        "clock reset to 0", path, days,
    )


def inject_era5_state(model: Model, cfg: DictConfig) -> None:
    """Seed the model from ERA5 (WeatherBench2) at the run start date.

    ``init.date`` overrides; otherwise ``run.start_date`` (else the
    2000-01-01 default) — matching the calendar the run integrates on.
    The regridded slice comes from :mod:`jcm.data.era5` (cached; needs
    internet or a prefetched cache). Mutates
    ``model._final_dycore_state`` — follow with ``model.resume(...)``.
    """
    from jcm.data.era5 import initial_state

    date = (cfg.get("init", {}).get("date", None)
            or cfg.get("run", {}).get("start_date", None)
            or "2000-01-01")
    state = initial_state(model.coords, str(date))
    provenance.record_fact("initial_condition", f"era5:{date}")
    model._final_dycore_state = model._prepare_initial_dycore_state(
        physics_state=state)


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
    the ``omega`` dycore field (e.g. the model-agnostic
    ``OmegaDiagnostic`` term), so an explicitly requested diagnostic
    never dies on the construction-time contract check, or (b) the
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

    log_level = getattr(logging, cfg.run.log_level.upper(), logging.CRITICAL)
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

    log_level = getattr(logging, cfg.run.log_level.upper(), logging.CRITICAL)
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
    """Finite-lid sponge: USSA T relaxation + implicit Rayleigh wind drag.

    The USSA-1976 reference temperature is evaluated at the level reference
    mid-pressures of the dycore's own hybrid grid — the same profile the
    backend's resting initial state uses, so the relaxation target is
    consistent with the initialization.
    """
    import numpy as np

    from jcm.initial_states.ussa1976 import ussa_pressure, ussa_temperature
    from jcm.physics.dissipation.upper_temperature_relaxation import (
        UpperTemperatureRelaxation,
    )

    a = np.asarray(dycore.coords.vertical.a_boundaries, dtype=float)
    b = np.asarray(dycore.coords.vertical.b_boundaries, dtype=float)
    p_mid = 0.5 * (a[:-1] + a[1:]) + 0.5 * (b[:-1] + b[1:]) * 101325.0
    zs = np.linspace(0.0, 84000.0, 4000)
    ps = np.asarray(ussa_pressure(zs))
    z_of_p = np.interp(np.log(p_mid), np.log(ps[::-1]), zs[::-1])
    t_ref = np.asarray(ussa_temperature(z_of_p))

    uv_hours = float(sponge_cfg.get("uv_hours", 0.0) or 0.0)
    return UpperTemperatureRelaxation(
        t_ref,
        n_levels=int(sponge_cfg.get("levels", 8)),
        timescale_s=float(sponge_cfg.get("t_hours", 6.0)) * 3600.0,
        wind_timescale_s=(uv_hours * 3600.0 if uv_hours > 0 else None),
    )


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------

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
    """
    if dycore is not None and hasattr(dycore, "colmap"):
        # pySES backend: monthly lon/lat climatology + JAM aerosol inputs,
        # each bilinearly interpolated onto the physics columns at build
        # time by ``jcm.dycore.pyses.forcing`` (files may live on any
        # regular lon/lat grid). ``ozone_file: auto`` resolves the packaged
        # climatology — column sampling has no exact-grid requirement, so
        # the T63 file serves any pySES resolution.
        from jcm.dycore.pyses.forcing import build_forcing as pyses_build_forcing

        ozone_file = cfg.forcing.get("ozone_file", None)
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

        file = (_resolve_data_path(cfg.forcing.get("file", None))
                or _pyses_default_bc("forcing.nc"))
        return pyses_build_forcing(
            str(file), dycore,
            emissions_file=_resolve_data_path(
                cfg.forcing.get("emissions_file", None)),
            dms_file=_resolve_data_path(cfg.forcing.get("dms_file", None)),
            dust_file=_resolve_data_path(cfg.forcing.get("dust_file", None)),
            oxidants_file=_resolve_data_path(
                cfg.forcing.get("oxidants_file", None)),
            ozone_file=_resolve_data_path(ozone_file),
        )

    forcing_cfg = cfg.get("forcing", None)
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


def _resolve_auto_ozone(coords):
    """Find a packaged ``jcm/data/bc/*/ozone.nc`` matching the model grid.

    Shape-based discovery (nlev, nlat, nlon); grid identity is then fully
    validated by ``OzoneClimatology.from_file``. Returns ``None`` when no
    packaged file fits — the caller warns and falls back to the analytic
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
    return None


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
    paths (e.g. one file for biomass burning and one for the rest) — multiple
    files are merged by coordinates via ``xr.open_mfdataset``, so each can carry
    a disjoint set of channels on the same grid. The fields auto-route by
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
    path = _resolve_data_path(_expand_years(
        forcing_cfg.get("emissions_file", None),
        forcing_cfg.get("years", None),
        forcing_cfg.get("available_years", None)))
    if path in (None, "", "null"):
        return forcing

    import xarray as xr
    from omegaconf import ListConfig

    from jcm.forcing import (
        default_forcing,
        read_anthropogenic_emissions,
        read_prescribed_aerosol_emissions,
    )

    # One path → open_dataset; several → merge by coords (disjoint channels on a
    # shared grid, e.g. biomass-burning + anthropogenic files).
    if isinstance(path, (list, tuple, ListConfig)):
        paths = [str(p) for p in path]
        ds = xr.open_mfdataset(paths, combine="by_coords") if len(paths) > 1 \
            else xr.open_dataset(paths[0])
    else:
        ds = xr.open_dataset(path)
    anthro = read_anthropogenic_emissions(ds)
    speciated = read_prescribed_aerosol_emissions(ds)
    if anthro is None and speciated is None:
        raise ValueError(
            f"forcing.emissions_file {path!r} has no emissions variables: "
            "expected ``emis_<sector>_<species>`` (bulk) or "
            "``aero_emis_<tracer>`` (pre-speciated). See the emissions-file "
            "contract in docs/design/jam.md."
        )
    _validate_emissions_grid({**(anthro or {}), **(speciated or {})},
                             coords, path)
    if forcing is None:
        forcing = default_forcing(coords.horizontal)
    return forcing.copy(anthropogenic_emissions=anthro,
                        prescribed_aerosol_emissions=speciated)


def _validate_emissions_grid(mapping, coords, path):
    """Raise if any emission field's horizontal shape != the model grid."""
    from jcm.forcing import TimeSeries
    nodal = tuple(coords.horizontal.nodal_shape)
    for name, leaf in mapping.items():
        arr = leaf.values if isinstance(leaf, TimeSeries) else leaf
        spatial = tuple(arr.shape[-2:])
        if spatial != nodal:
            raise ValueError(
                f"forcing.emissions_file {path!r}: field {name!r} has "
                f"horizontal shape {spatial}, but the model grid is {nodal}. "
                "Regrid the file with jcm.data.emissions.prepare first."
            )


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
    """
    if forcing_cfg is None:
        return forcing
    path = _resolve_data_path(forcing_cfg.get("oxidants_file", None))
    if path in (None, "", "null"):
        return forcing
    import xarray as xr

    from jcm.forcing import read_oxidant_vmr
    lat_deg, lon_deg = _model_latlon_deg(coords)
    nlev = int(coords.nodal_shape[0])
    with xr.open_dataset(str(path)) as ds:
        mapping = read_oxidant_vmr(ds, nlev=nlev,
                                   lat_deg=lat_deg, lon_deg=lon_deg)
        _validate_oxidant_levels(ds, coords, path)
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(oxidant_vmr=mapping)


def _validate_oxidant_levels(ds, coords, path):
    """Cross-check the oxidant file's hybrid coefficients against the model.

    Only applies when both the file (``hyam``/``hybm``, plus ``p0`` in Pa for
    files storing normalized ``hyam``) and the model
    (:class:`dinosaur.hybrid_coordinates.HybridCoordinates`) define hybrid
    levels; a sigma-coordinate model only gets the level-count assert in
    ``read_oxidant_vmr`` (documented assumption: the file matches the model
    levels). Full-level model coefficients are boundary midpoints, matching
    the ECHAM ``hyam/hybm`` convention.
    """
    import numpy as np
    from dinosaur.hybrid_coordinates import HybridCoordinates
    vertical = coords.vertical
    if not isinstance(vertical, HybridCoordinates):
        return
    if "hyam" not in ds or "hybm" not in ds:
        return
    hyam = np.asarray(ds["hyam"].values, dtype=float)
    hybm = np.asarray(ds["hybm"].values, dtype=float)
    # HAMMOZ files store hyam normalized by the reference pressure p0 [Pa];
    # dinosaur's a_boundaries are in Pa.
    if "p0" in ds:
        hyam = hyam * float(ds["p0"].values)
    a = np.asarray(vertical.a_boundaries, dtype=float)
    b = np.asarray(vertical.b_boundaries, dtype=float)
    a_full = 0.5 * (a[:-1] + a[1:])
    b_full = 0.5 * (b[:-1] + b[1:])
    if (not np.allclose(a_full, hyam, atol=1.0)          # Pa
            or not np.allclose(b_full, hybm, atol=1e-5)):
        raise ValueError(
            f"forcing.oxidants_file {path!r}: hybrid-level coefficients "
            "(hyam/hybm) don't match the model's vertical grid — the file "
            "must be on the model levels (no vertical interpolation is "
            "done). Use the matching L-grid file (e.g. the T63L47 MACC file "
            "with grid=echam_t63_l47_hybrid) or re-interpolate it."
        )


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


def _run_full(cfg: DictConfig, model: Model | None = None) -> ModelPredictions:
    if model is None:
        model = build_model(cfg)

    forcing = build_forcing(cfg, model.coords, dycore=getattr(model, "dycore", None))
    forcing = _maybe_attach_nudging_target(forcing, cfg, model)
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
    if cfg.init.kind == "jw":
        inject_jw_profile(model, rh=float(cfg.init.get("rh", 0.6)))
        return model.resume(
            forcing=forcing,
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
            snapshot_interval=cfg.run.get("snapshot_interval"),
            snapshot_variables=tuple(cfg.run.get("snapshot_variables") or ()),
        )
    if cfg.init.kind == "balanced_isothermal":
        inject_balanced_isothermal_profile(model)
        return model.resume(
            forcing=forcing,
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
            snapshot_interval=cfg.run.get("snapshot_interval"),
            snapshot_variables=tuple(cfg.run.get("snapshot_variables") or ()),
        )
    if cfg.init.kind == "from_state":
        inject_state_file(model, cfg)
        return model.resume(
            forcing=forcing,
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
            snapshot_interval=cfg.run.get("snapshot_interval"),
            snapshot_variables=tuple(cfg.run.get("snapshot_variables") or ()),
        )
    if cfg.init.kind == "era5":
        inject_era5_state(model, cfg)
        return model.resume(
            forcing=forcing,
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
            snapshot_interval=cfg.run.get("snapshot_interval"),
            snapshot_variables=tuple(cfg.run.get("snapshot_variables") or ()),
        )
    raise ValueError(f"Unknown init.kind={cfg.init.kind!r}")


def _load_states_from_cfg(cfg: DictConfig):
    """Open ``cfg.run.state_file`` and return a stacked ``PhysicsState``."""
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
        tracer_vars = OmegaConf.to_container(tracer_vars, resolve=True)
    ds = xr.open_dataset(state_file)
    return ds, load_states_from_xarray(ds, tracer_vars=tracer_vars or None)


def _run_prescribed(cfg: DictConfig):
    """Diagnose physics tendencies from a JCM state-file time series."""
    from jcm.prescribed_state_model import PrescribedStateModel

    coords = build_coords(cfg)
    physics = build_physics(cfg)
    terrain = build_terrain(cfg, coords)
    forcing = build_forcing(cfg, coords)
    _, states = _load_states_from_cfg(cfg)

    model = PrescribedStateModel(
        physics=physics,
        coords=coords,
        terrain=terrain,
        dt_seconds=float(cfg.run.time_step) * 60.0,
    )
    return model.run(states, forcing=forcing)


def _select_column(states, ds, lat_deg: float, lon_deg: float):
    """Return the column of ``states`` nearest to ``(lat_deg, lon_deg)``.

    The state's xarray ``ds`` carries ``lat`` / ``lon`` coordinates from the
    JCM run that wrote it; pick by nearest neighbour so users can give
    physical degrees rather than grid indices.
    """
    import numpy as np
    from jax.tree_util import tree_map

    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    i_lat = int(np.argmin(np.abs(lat - lat_deg)))
    i_lon = int(np.argmin(np.abs(lon - lon_deg)))

    def slice_field(arr):
        # JCM xarray output is laid out (time, level, lon, lat) for column
        # variables and (time, lon, lat) for surface scalars.
        if arr.ndim == 4:
            return arr[:, :, i_lon, i_lat]
        if arr.ndim == 3:
            return arr[:, i_lon, i_lat]
        return arr

    return tree_map(slice_field, states), (i_lon, i_lat, float(lat[i_lat]), float(lon[i_lon]))


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
    ds, states = _load_states_from_cfg(cfg)
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

    from jcm.diagnostics import check_health, print_report

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
        if cfg.init.kind == "jw":
            inject_jw_profile(model, rh=float(cfg.init.get("rh", 0.6)))
        elif cfg.init.kind == "balanced_isothermal":
            inject_balanced_isothermal_profile(model)
        else:
            model.bootstrap_state()

        # ``inject_*_profile`` only populates ``_final_dycore_state`` —
        # the physics carry is normally built lazily by ``resume``.
        # ``load_checkpoint`` needs both pytrees as deserialization
        # templates, so build the carry now if the inject path took it.
        if model._final_physics_state is None:
            model._final_physics_state = model._build_initial_physics_carry()

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
        if first_fresh_chunk and cfg.init.kind == "jw":
            inject_jw_profile(model, rh=float(cfg.init.get("rh", 0.6)))
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
                snapshot_interval=cfg.run.get("snapshot_interval"),
                snapshot_variables=tuple(
                    cfg.run.get("snapshot_variables") or ()),
            )
        elif first_fresh_chunk and cfg.init.kind == "balanced_isothermal":
            inject_balanced_isothermal_profile(model)
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
                snapshot_interval=cfg.run.get("snapshot_interval"),
                snapshot_variables=tuple(
                    cfg.run.get("snapshot_variables") or ()),
            )
        elif first_fresh_chunk and cfg.init.kind == "era5":
            # Without this branch a chunked run silently ignored init=era5
            # (the chunked dispatch returns before _run_full's init ladder).
            inject_era5_state(model, cfg)
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
                snapshot_interval=cfg.run.get("snapshot_interval"),
                snapshot_variables=tuple(
                    cfg.run.get("snapshot_variables") or ()),
            )
        elif first_fresh_chunk and cfg.init.kind == "from_state":
            inject_state_file(model, cfg)
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
                snapshot_interval=cfg.run.get("snapshot_interval"),
                snapshot_variables=tuple(
                    cfg.run.get("snapshot_variables") or ()),
            )
        elif first_fresh_chunk:
            preds = model.run(
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

        nc_path = f"{output_prefix}_day{int(elapsed_sim_days)}.nc"
        ds.attrs.update(provenance.attrs())
        ds.attrs["jcm_prov_chunk_wall_seconds"] = round(chunk_wall, 1)
        ds.to_netcdf(nc_path)
        provenance.write_sidecar(nc_path)
        print(f"  Saved {nc_path}")
        snap_ds = getattr(preds, "snapshot_dataset", lambda: None)()
        if snap_ds is not None:
            snap_path = (f"{output_prefix}_day{int(elapsed_sim_days)}"
                         "_snapshots.nc")
            snap_ds.attrs.update(provenance.attrs())
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
    ds = predictions.to_xarray()
    ds.attrs.update(provenance.attrs())
    ds.to_netcdf(str(output_path))
    provenance.write_sidecar(output_path)
    logger.info("Wrote %s", output_path)
    # Interval-instantaneous snapshot stream (jax-gcm#586): a separate
    # file with its own (finer) time axis — folding a second cadence into
    # the main dataset would force a ragged time dimension.
    snap_ds = getattr(predictions, "snapshot_dataset", lambda: None)()
    if snap_ds is not None:
        snap_path = output_path.with_name(output_path.stem + "_snapshots.nc")
        snap_ds.attrs.update(provenance.attrs())
        snap_ds.to_netcdf(str(snap_path))
        logger.info("Wrote %s", snap_path)
