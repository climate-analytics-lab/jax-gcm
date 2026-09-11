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
from pathlib import Path
from typing import Any

import jax
from omegaconf import DictConfig

from jcm import provenance
from jcm.data import mirror_manifest as mm
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
        params_obj = base.__class__(
            **{**base.__dict__, **dict(overrides)}
        )
        # ``default()`` runs any config-time cross-field validation, but this
        # direct constructor bypasses it — so a YAML override could re-create
        # an illegal field COMBINATION (e.g. echam_1m's legacy ccraut-as-
        # KK2000-threshold, #674) that the defaults alone never trip. Re-run
        # the opt-in ``validate`` hook on the post-override object so ANY
        # Parameters class can guard both construction doors. Config-time,
        # concrete values only — never called under a jit trace.
        validate = getattr(params_obj, "validate", None)
        if callable(validate):
            validate()
        init_kwargs[kwarg_name] = params_obj

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

#: Yearly ``{year}`` file-pattern expansion; the science lives in
#: :func:`jcm.forcing.expand_yearly_files`. Aliased for the many call sites
#: (and TestYearExpansionAndStartDate) that reference the private name.
_expand_years = expand_yearly_files


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

# The forcing-assembly science — ``auto`` resolution, path/provenance
# resolution, the attach chain, the merge-compatibility guard — lives in
# :mod:`jcm.forcing_assembly` next to the readers it drives; the runner keeps
# only the cfg dispatch (``build_forcing``) plus the pySES column-sampling
# branch. These names are re-exported so ``runners.<name>`` call sites (incl.
# the terrain/init/pySES paths below) keep resolving; tests that STUB them
# patch :mod:`jcm.forcing_assembly` so the stub reaches both doors.
from jcm import forcing_assembly  # noqa: E402
from jcm.forcing_assembly import (  # noqa: E402
    _assert_uniform_time_axis as _assert_uniform_time_axis,
    _attach_dms as _attach_dms,
    _attach_dust as _attach_dust,
    _attach_emissions as _attach_emissions,
    _attach_macv2_weights as _attach_macv2_weights,
    _attach_oxidants as _attach_oxidants,
    _attach_ozone as _attach_ozone,
    _emission_auto_resolves_to_none as _emission_auto_resolves_to_none,
    _ensure_parent_forcing as _ensure_parent_forcing,
    _forcing_products as _forcing_products,
    _grid_token as _grid_token,
    _merge_disjoint_emissions as _merge_disjoint_emissions,
    _model_latlon_deg as _model_latlon_deg,
    _open_forcing_dataset as _open_forcing_dataset,
    _product_available_years as _product_available_years,
    _product_time_axis as _product_time_axis,
    _reject_year_pattern as _reject_year_pattern,
    _resolve_auto_ozone as _resolve_auto_ozone,
    _resolve_auto_terrain as _resolve_auto_terrain,
    _resolve_data_path as _resolve_data_path,
    _resolve_emission_inputs as _resolve_emission_inputs,
    _resolve_one_emission_input as _resolve_one_emission_input,
    _resolve_oxidant_paths as _resolve_oxidant_paths,
    _resolve_pyses_emission_paths as _resolve_pyses_emission_paths,
    _vertical_kind as _vertical_kind,
    assemble_spectral_forcing as assemble_spectral_forcing,
)


def build_forcing(cfg: DictConfig, coords, dycore=None):
    """Build a ``ForcingData`` from ``cfg.forcing`` (the CLI door).

    Adapter-side dispatch only: a pySES ``dycore`` routes to the runner-held
    column branch (:func:`_build_pyses_forcing`) after the same ``auto``
    emission resolution the engine applies; every other dycore delegates to the
    forcing-side engine :func:`jcm.forcing_assembly.build_forcing` — the one the
    Python door ``ForcingData.from_bundles`` drives too, so the two doors
    provably agree (#751). The engine never depends on this adapter.
    """
    if dycore is not None and hasattr(dycore, "colmap"):
        _forcing_cfg = cfg.get("forcing", None)
        if _forcing_cfg is not None:
            _forcing_cfg = _resolve_emission_inputs(
                _forcing_cfg, cfg, coords, is_pyses=True)
        return _build_pyses_forcing(_forcing_cfg, dycore, coords)
    return forcing_assembly.build_forcing(cfg, coords)


def _build_pyses_forcing(_forcing_cfg, dycore, coords):
    """Build pySES-backend forcing: bilinear column sampling of the inputs.

    Kept in the runner (not unified with the spectral assembly, #751): the pySES
    core interpolates every gridded field (ozone / emissions / dms / dust /
    oxidants) onto its physics columns at build time via ``attach_jam_forcing``
    (files may live on any regular lon/lat grid — no exact-grid requirement), so
    it delegates to ``jcm.dycore.pyses.forcing.build_forcing`` rather than the
    dinosaur attach helpers. It DOES share the resolution helpers
    (``_resolve_pyses_emission_paths`` / ``_resolve_oxidant_paths`` / year
    expansion) so the two paths cannot drift.
    """
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
    if isinstance(ozone_file, str) and "{year}" in ozone_file:
        # Transient ozone is genuinely unsupported on the pySES path (unlike
        # oxidants/emissions below): the column ozone leaf is a 12-month
        # WRAP_YEAR climatology and ``attach_jam_forcing`` rejects any
        # non-12-month file. Raise the clear limitation here rather than let
        # the literal-brace path reach ``xr.open_dataset`` as a file-not-
        # found. Run the spectral dinosaur backend for transient ozone.
        raise ValueError(
            f"forcing.ozone_file={ozone_file!r} has a {{year}} pattern, but "
            "transient ozone is not supported on the pySES backend (the "
            "column ozone climatology is a 12-month WRAP_YEAR field). "
            "Provide a single 12-month climatology file, or use the "
            "spectral dinosaur backend for transient ozone."
        )
    provenance.record_fact(
        "ozone_source",
        f"prescribed:{ozone_file}" if ozone_file
        else "analytic (no ozone file)")

    raw_file = _forcing_cfg.get("file", None)
    if isinstance(raw_file, str) and "{year}" in raw_file:
        # Transient surface forcing is genuinely unsupported on pySES (as
        # for ozone above): the column forcing reader opens a SINGLE
        # 12-month climatology (``jcm.dycore.pyses.forcing.build_forcing``
        # → one ``xr.open_dataset``), not a multi-year concatenation. Raise
        # the clear limitation here rather than let the literal-brace path
        # reach ``_resolve_data_path`` and surface as a confusing hf:// 404
        # / file-not-found. Use forcing=amip/era5 on the spectral dinosaur
        # backend for transient surface forcing.
        raise ValueError(
            f"forcing.file={raw_file!r} has a {{year}} pattern, but "
            "transient surface forcing is not supported on the pySES "
            "backend (the column forcing reader opens a single 12-month "
            "climatology). Provide a single climatology file, or use the "
            "spectral dinosaur backend (forcing=amip/era5) for transient "
            "surface forcing."
        )
    file = (_resolve_data_path(raw_file)
            or _pyses_default_bc("forcing.nc"))
    # Year expansion must happen on the pySES path too, so the documented
    # transient forms (``oxidants_file``/``emissions_file=.../{year}.nc``
    # with ``forcing.years``) resolve to real yearly files rather than a
    # literal-brace path (or an unfetched ``hf://`` URL) reaching
    # ``xr.open_dataset``. Oxidants and emissions each go through a shared
    # resolver that expands ``{year}`` patterns, resolves ``hf://`` and
    # runs the uniform-time-axis check, so neither can drift from the
    # spectral path. Both open their file set as ONE combined dataset in
    # ``attach_jam_forcing`` (pySES has no per-product alignment machinery),
    # so a genuine multi-product emissions *list* is flattened for that
    # single open with every ``{year}`` element expanded — and a list that
    # mixes a climatology with a transient product is rejected there rather
    # than mis-aligned (the spectral per-product path is the only one that
    # can carry both in one list). ``dms_file`` / ``dust_file`` take no year
    # expansion because they are climatology-only on BOTH backends (their
    # readers are WRAP_YEAR; ``_attach_dms`` / ``_attach_dust`` likewise
    # never expand); a ``{year}`` there is rejected loudly by
    # ``_reject_year_pattern`` on both paths rather than reaching
    # ``open_dataset`` as a literal-brace file-not-found.
    forcing = pyses_build_forcing(
        str(file), dycore,
        emissions_file=_resolve_pyses_emission_paths(_forcing_cfg),
        dms_file=_resolve_data_path(_reject_year_pattern(
            _forcing_cfg.get("dms_file", None), "dms_file")),
        dust_file=_resolve_data_path(_reject_year_pattern(
            _forcing_cfg.get("dust_file", None), "dust_file")),
        oxidants_file=_resolve_oxidant_paths(_forcing_cfg),
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


def apply_constants_overrides(cfg: DictConfig) -> None:
    """Apply any ``cfg.constants`` physical-constant overrides to the global singleton.

    Shared by the CLI door (:func:`run`) and the Python door
    (:func:`jcm.configurations.load`) so both build the model against the SAME
    constants — the dynamical core reads the live :mod:`jcm.constants` singleton
    at construction, so this MUST run before ``build_model``. Only base fields
    may be set; derived constants (rd, cvd, rgrav, vtmpc*) recompute. Note
    ``set_constants`` is process-global — the override persists for the whole
    interpreter, not just this build.
    """
    constants_overrides = cfg.get("constants", None)
    if constants_overrides:
        import jcm.constants as _jcm_constants
        _jcm_constants.set_constants(
            **{k: float(v) for k, v in dict(constants_overrides).items()}
        )


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

    apply_constants_overrides(cfg)

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


def _resolved_emission_value(literal, key, coords, jam, is_pyses):
    """Resolution OUTCOME of one emission ``key``, for warnings (no fetch).

    Returns None when the key resolves to no prescribed source (an explicit
    null, or ``auto`` on a grid that has no bundle for THIS key — see
    :func:`_emission_auto_resolves_to_none`, now key-specific so the level-
    dependent ``oxidants_file`` nulls on an unpublished layer count while the
    level-free keys still resolve, F2); otherwise the literal path (or the
    ``"auto"`` sentinel when ``auto`` resolves to a real per-grid bundle).
    Mirrors :func:`_resolve_one_emission_input`'s None-decision WITHOUT fetching
    so :func:`warn_on_config_traps` reasons about the same values the build
    applies rather than the raw ``"auto"`` cfg.
    """
    if literal in (None, "", "null", "none"):
        return None
    if literal == "auto":
        return None if _emission_auto_resolves_to_none(
            key, coords, jam, is_pyses) else "auto"
    return literal


def _forcing_tracks_calendar(forcing) -> bool:
    """Report whether the RESOLVED surface forcing is date-aligned (transient).

    The config keys ``forcing.years`` / ``forcing.align`` miss the common case
    of a single multi-year netCDF under the default ``align: auto``:
    ``ForcingData.from_file``'s span-based auto-detection resolves it to
    ``BY_DATE`` at build time, yet the config still reads ``align: auto`` /
    ``years: null``. So classify from what the resolution actually produced —
    any surface ``TimeSeries`` leaf (SST / sea-ice / snow / soil / land T) whose
    ``align_mode`` is ``BY_DATE`` / ``BY_DATE_INTERP`` means those fields track
    real calendar dates. A 12-month climatology resolves to ``WRAP_YEAR`` and is
    not transient. ``forcing`` may be ``None`` (default/prescribed path builds
    none) — then there is no date-aligned surface forcing to flag.
    """
    if forcing is None:
        return False
    from jcm.forcing import BY_DATE, BY_DATE_INTERP, TimeSeries
    # ``getattr`` defaults so a caller passing a partial forcing stand-in (with
    # only the fields the check it targets needs) is treated as non-transient
    # rather than raising — a real ForcingData always carries all five.
    for name in ("sea_surface_temperature", "sice_am", "snowc_am",
                 "soilw_am", "stl_am"):
        field = getattr(forcing, name, None)
        if isinstance(field, TimeSeries) and int(field.align_mode) in (
                BY_DATE, BY_DATE_INTERP):
            return True
    return False


def warn_emission_config_traps(*, has_jam, is_pyses, is_scm, forcing_cfg,
                               coords, forcing) -> None:
    """Emission-family config-trap warnings (traps 3 & 5) from RESOLVED values.

    Shared home so the CLI runner (:func:`warn_on_config_traps`) and the Python
    door (:func:`jcm.forcing.ForcingData.from_bundles`) fire IDENTICAL messages
    once (#751). The terrain / forcing-kind / MACv2-weight traps read cfg only
    and stay in :func:`warn_on_config_traps`. ``forcing_cfg`` is the ``forcing``
    config mapping; ``coords`` the built grid; ``forcing`` the built struct
    (``None`` on the default/prescribed path).
    """
    # 3. Prognostic aerosol with no RESOLVED prescribed-emission source: only
    #    online Gong sea-salt then has one. Reads the RESOLVED values, not the
    #    raw cfg (F2): ``auto`` on pySES / a non-published grid resolves to None,
    #    so a JAM run there is silently emission-free. Under scm the trap
    #    MIS-SUPPRESSES (auto reads "real bundle" but the SCM attaches no
    #    forcing), so scm gets one honest mode-specific message instead.
    if has_jam and is_scm:
        logger.warning(
            "config trap: JAM prognostic aerosol in single-column mode "
            "(run.mode=scm) — the single-column model builds no boundary "
            "ForcingData (it runs on ForcingData.zeros) and consumes NONE of "
            "the prescribed emission inputs: emissions_file, dms_file, "
            "dust_file and oxidants_file are all ignored in SCM regardless of "
            "grid. The column is therefore zero-emission apart from any online "
            "sources (e.g. wind-driven Gong sea salt); prescribed sulfur, dust "
            "and carbonaceous emissions stay at zero. This is expected for an "
            "SCM process study — prescribed emissions require the full "
            "(gridded) model."
        )
    elif has_jam:
        emission_keys = ("emissions_file", "dms_file", "dust_file",
                         "oxidants_file")
        # The mirror manifest is the read-side single source for what is
        # published: Gaussian grids (top-level ``grids`` with a real nlat — the
        # column ne30pg3 carries None) and the level-resolved layer counts.
        _man = mm.load_manifest()
        _pub_grids = sorted(g for g, n in _man["grids"].items() if n is not None)
        _pub_levels = sorted(_man["levels"])
        resolved = {k: _resolved_emission_value(
                        forcing_cfg.get(k, None), k, coords, has_jam, is_pyses)
                    for k in emission_keys}
        unset = [k for k in emission_keys if resolved[k] is None]
        # Keys the user left at ``auto`` that nonetheless resolved to None —
        # i.e. the silent-degrade case (pySES / non-mirrored grid), distinct
        # from an explicit opt-out null.
        auto_nulled = [k for k in emission_keys
                       if str(forcing_cfg.get(k, None)) == "auto"
                       and resolved[k] is None]
        if len(unset) == len(emission_keys):
            if auto_nulled:
                reason = (
                    "the pySES backend publishes no per-grid emission bundles"
                    if is_pyses else
                    f"grid {_grid_token(coords)!r} is not one of the mirror's "
                    f"published grids ({', '.join(_pub_grids)})")
                logger.warning(
                    "config trap: zero-emission JAM baseline — the 'auto' "
                    "emission key(s) %s resolved to None because %s, so the "
                    "only online aerosol source is Gong sea salt; sulfur, dust "
                    "and carbonaceous species stay at zero. Point each key at "
                    "an on-grid file (e.g. forcing.emissions_file=<path>) to "
                    "supply prescribed emissions.",
                    ", ".join(auto_nulled), reason,
                )
            else:
                logger.warning(
                    "config trap: zero-emission JAM baseline — %s are all "
                    "unset, so the only online aerosol source is Gong sea "
                    "salt; sulfur, dust and carbonaceous species stay at zero. "
                    "Leave them at their 'auto' default (the per-grid HF "
                    "bundles) or set an explicit path (e.g. "
                    "forcing.emissions_file=hf://bundles/<grid>/"
                    "emissions_pd.nc).",
                    ", ".join(unset),
                )
        elif auto_nulled:
            # Partial silent-degrade (F2): SOME 'auto' keys nulled while others
            # resolved — the LEVEL-dependent oxidants_file on a published
            # horizontal grid lacking a level-resolved bundle (unpublished layer
            # count e.g. t63_l8, or a sigma vertical). The level-free keys still
            # supply their emissions, so flag exactly the nulled keys.
            logger.warning(
                "config trap: partial zero-emission JAM baseline — the 'auto' "
                "emission key(s) %s resolved to None because the mirror "
                "publishes no bundle for this grid: level-resolved products "
                "such as oxidants exist only for hybrid verticals at L%s, and "
                "this grid is %s at L%d. The remaining keys resolved, so those "
                "species alone stay at zero; point each nulled key at an "
                "on-grid file, or run a published hybrid layer count, to "
                "supply them.",
                ", ".join(auto_nulled),
                "/L".join(str(n) for n in _pub_levels),
                _vertical_kind(coords) if coords is not None else "unknown-vertical",
                int(coords.nodal_shape[0]) if coords is not None else -1,
            )

    # 5. Transient (by-date) surface forcing driving JAM off the present-day
    #    emission bundles. Transience is read off the RESOLVED forcing's surface
    #    alignment (:func:`_forcing_tracks_calendar`) — a single multi-year
    #    netCDF under ``align: auto`` resolves to BY_DATE while the config still
    #    reads ``auto``/``years: null``, which keying only on those keys misses —
    #    with ``years``/``align`` as OR fallbacks for a forcing-less caller.
    #    Only ``auto`` that RESOLVED to a real present-day *_pd bundle is the
    #    concern (F2); an auto that nulled is warning 3's case, not this one.
    if has_jam and not is_scm:
        years = forcing_cfg.get("years", None)
        align = str(forcing_cfg.get("align", "") or "")
        is_transient = (_forcing_tracks_calendar(forcing)
                        or bool(years)
                        or align in ("by_date", "by_date_interp"))
        pd_auto_keys = [
            k for k in ("emissions_file", "oxidants_file")
            if str(forcing_cfg.get(k, None)) == "auto"
            and _resolved_emission_value(
                forcing_cfg.get(k, None), k, coords, has_jam,
                is_pyses) is not None]
        if is_transient and pd_auto_keys:
            logger.warning(
                "config trap: transient (by-date) forcing with present-day JAM "
                "emissions — the surface forcing tracks real calendar dates "
                "(amip/era5: per-year files, by_date_interp) but %s are still "
                "'auto', which resolved to the present-day *_pd emission "
                "bundles. A historical/AMIP run is therefore using present-day "
                "aerosol emissions. For emissions, override with the mirror's "
                "transient product using a year-matched {year} pattern (the "
                "same yearly-file expansion the SST forcing uses). A bare "
                "'{' is Hydra override syntax, so the value must be quoted for "
                "Hydra AND protected from the shell — wrap the whole argument "
                "in single quotes with the value in double quotes (or set it "
                "in a forcing yaml, where the brace needs no escaping) — e.g. "
                "'forcing.emissions_file=\"hf://bundles/<grid>/emissions_amip/"
                "{year}.nc\"', with the run's forcing.years range. The "
                "emissions_amip bundle spans 1950-2022 (ends before era5's "
                "2024 surface coverage), so also set "
                "forcing.emissions_available_years=[1950,2022] to clamp the "
                "expansion to the built files (era5 already ships this). The "
                "mirror publishes NO transient oxidants product (only "
                "oxidants_pi/oxidants_pd climatologies), so transient oxidants "
                "must come from a separately prepared dataset; "
                "forcing.oxidants_file accepts a {year} pattern (and "
                "forcing.oxidants_available_years its coverage) once you have "
                "one.",
                ", ".join(pd_auto_keys),
            )


def warn_on_config_traps(cfg: DictConfig, physics, forcing,
                         coords=None, dycore=None) -> None:
    """Warn (never raise) about config combinations that run but mislead.

    Config-layer cross-validation belongs in the runner (#640): it reads the
    composed ``cfg`` plus the already-built ``physics``/``forcing`` objects and
    calls no science. Every finding here is a :func:`logging.Logger.warning`,
    not an error — the combinations all *run*, they just quietly produce
    something other than what the config name suggests, and the maintainer
    chose to keep them runnable (e.g. for controlled idealized experiments).

    ``coords`` and ``dycore`` let the emission-key checks (3 and 5) read the
    RESOLVED emission values rather than the raw ``"auto"`` cfg (F2): ``auto``
    resolves to None on the pySES path or a non-mirrored grid, so a JAM run
    there is silently emission-free — the exact case warning 3 must catch.
    ``dycore`` decides the pySES path (``hasattr(dycore, "colmap")``); when
    both are omitted (e.g. a caller that builds no forcing) the resolution
    falls back to treating ``auto`` conservatively via the shared predicate.

    ``forcing`` may be ``None`` (``forcing.kind: default``, or the prescribed
    path that builds none): the aquaplanet ``default_forcing`` the model then
    falls back to carries the same all-ones MACv2-SP weights, so it is treated
    as the all-ones case for warning 4.

    ``run.mode=scm`` is handled specially. The single-column model
    (:func:`_run_scm`) builds NO gridded surface: it runs on
    ``TerrainData.single_column()`` (flat ocean) and ``ForcingData.zeros`` and
    consumes none of ``cfg.terrain``/``cfg.forcing`` — no gridded land-sea mask,
    no transient surface forcing, and (critically) none of the prescribed JAM
    emission inputs. The gridded-surface traps (1, 2), the transient-emission
    trap (5), and the MACv2-SP-weight trap (4) therefore either fire on config
    the SCM ignores or point at a remedy the SCM cannot apply, so they are gated
    off under ``scm``. The zero-emission trap (3) would MIS-SUPPRESS there — on a
    published grid ``auto`` resolves to a "real bundle" and stays silent, yet the
    SCM attaches no forcing so the column genuinely has no prescribed emissions —
    so ``scm`` replaces it with one honest, mode-specific warning.
    """
    import numpy as np

    from jcm.forcing import TimeSeries

    terrain_kind = cfg.get("terrain", {}).get("kind", None)
    forcing_kind = cfg.get("forcing", {}).get("kind", None)
    has_jam = _has_jam(physics)
    is_pyses = dycore is not None and hasattr(dycore, "colmap")
    # See the docstring: the SCM builds no gridded terrain/forcing and attaches
    # no prescribed emissions, so the gridded-surface / transient / MACv2-weight
    # traps below are gated off under ``scm`` and the zero-emission trap is
    # replaced by one honest SCM-specific message.
    is_scm = str(cfg.get("run", {}).get("mode", "full") or "full") == "scm"

    # 1. Prognostic aerosol over a flat all-ocean planet: Gong sea-salt emits
    #    everywhere (including where land should be), there is no orography to
    #    source dust, and the idealized cos²-lat SSTs are not a real surface.
    #    Gated off under scm: the SCM ignores cfg.terrain and always runs on a
    #    single flat-ocean column, so the gridded land-sea-mask concern is moot.
    if has_jam and terrain_kind == "aquaplanet" and not is_scm:
        logger.warning(
            "config trap: JAM prognostic aerosol with terrain=aquaplanet — a "
            "flat all-ocean planet has no land-sea mask, so Gong sea-salt "
            "emission fires over cells that should be land and there is no "
            "orography to source dust. Use terrain=auto (native-grid mask) or "
            "terrain=from_file for a realistic surface."
        )

    # 2. The inverse mismatch (#640): a real-world boundary file's land-sea
    #    mask over aquaplanet terrain — SSTs land on cells the terrain calls
    #    ocean and vice-versa. Gated off under scm: the SCM builds no forcing
    #    and uses a single-column terrain, so no gridded masks can disagree.
    if terrain_kind == "aquaplanet" and forcing_kind == "from_file" and not is_scm:
        logger.warning(
            "config trap: forcing.kind=from_file over terrain=aquaplanet — the "
            "boundary file's real-world SST/land fields carry a land-sea mask "
            "that disagrees with the flat all-ocean terrain. Pair from_file "
            "forcing with terrain=from_file (or terrain=auto) so the two masks "
            "agree (issue #640)."
        )

    # 3 & 5. Emission-family traps (zero/partial-emission JAM baseline;
    #    transient forcing with present-day emissions) — computed from the
    #    RESOLVED values in the shared home both doors traverse (#751).
    warn_emission_config_traps(
        has_jam=has_jam, is_pyses=is_pyses, is_scm=is_scm,
        forcing_cfg=cfg.get("forcing", {}), coords=coords, forcing=forcing)

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

    #    Gated off under scm: the SCM never builds forcing from cfg (it always
    #    runs on ForcingData.zeros, i.e. all-ones weights), so the remedy this
    #    warning offers — forcing=macv2_sp for time-varying weights — cannot be
    #    applied in SCM. Firing it would advertise an inapplicable fix.
    if _has_macv2sp(physics) and not has_jam and not is_scm:
        # forcing=None → the aquaplanet default_forcing, all-ones weights.
        all_ones = forcing is None or (
            _is_allones_static(forcing.aerosol_year_weight)
            and _is_allones_static(forcing.aerosol_ann_cycle))
        if all_ones:
            logger.warning(
                "config trap: MACv2-SP with the default all-ones "
                "aerosol_year_weight/aerosol_ann_cycle — this is perpetual "
                "year-2005 plume amplitude with no seasonal cycle, not "
                "historical aerosol forcing. Use forcing=macv2_sp for real "
                "time-varying MACv2-SP weights — it now loads the repo-packaged "
                "SPv2.1 file out of the box (no macv2_file needed)."
            )


def _run_full(cfg: DictConfig, model: Model | None = None) -> ModelPredictions:
    if model is None:
        model = build_model(cfg)

    forcing = build_forcing(cfg, model.coords, dycore=getattr(model, "dycore", None))
    forcing = _maybe_attach_nudging_target(forcing, cfg, model)
    guard_emulator_ghg_forcing(model.physics, forcing)
    warn_on_config_traps(cfg, model.physics, forcing, coords=model.coords,
                         dycore=getattr(model, "dycore", None))
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
    warn_on_config_traps(cfg, physics, forcing, coords=coords)
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
    # The SCM builds no ForcingData; pass None so the config-trap check runs in
    # its scm-aware branch (it reads run.mode=scm from cfg). In scm mode the
    # gridded-surface/transient/MACv2-weight traps are gated off and a JAM run
    # gets one honest "column is emission-free" warning — see
    # ``warn_on_config_traps``. ``coords`` is still passed for symmetry with the
    # other run paths (the scm branch does not use it).
    warn_on_config_traps(cfg, physics, None, coords=coords)
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
            # Archive at the first chunk boundary past each interval multiple,
            # so the cadence need not divide chunk_days (30-day chunks with
            # archive_ckpt_every=100 archive at days 120, 210, 300, ...). The
            # relative tolerance keeps a fractional cadence on schedule:
            # elapsed accumulates by summing chunks, so a nominal 0.9 arrives
            # as 0.8999999999999999 and would otherwise slip a whole chunk.
            tol = 1e-6 * archive_every
            if archive_every > 0 and (
                int((elapsed_sim_days + tol) // archive_every)
                > int((elapsed_sim_days - cur_chunk + tol) // archive_every)
            ):
                import shutil

                # ``:g`` keeps whole-day archives named ``_day30`` while giving
                # sub-day cadences a distinct name instead of colliding on the
                # truncated integer day.
                day = f"{elapsed_sim_days:g}".replace(".", "p")
                archive = f"{output_prefix}_day{day}.ckpt"
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
