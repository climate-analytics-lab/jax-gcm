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
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from jcm.diffusion import DiffusionFilter
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
                "Use one of the supported counts (40, 47) or extend "
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

    return ComposablePhysics(
        terms=terms,
        checkpoint_terms=physics_cfg.get("checkpoint_terms", True),
        vectorize_columns=physics_cfg.get("vectorize_columns", False),
        band_config=_band_config_for_terms(terms),
    )


#: Physics ``builder`` names → factory callables returning a ``ComposablePhysics``
#: with its own validated term ordering (and band_config/vectorize handled
#: internally). The factory already orders the JAM aerosol chain (incl. the
#: pre/post-cloud split), so the preset YAML only carries scalar flags.
def _physics_factories():
    from jcm.physics.echam.echam_terms import echam_physics
    return {"echam_physics": echam_physics}


def _build_physics_from_factory(physics_cfg):
    """Build physics by delegating to a factory named by ``physics.builder``.

    Only the factory keyword args present in the YAML are forwarded (filtered
    against the factory signature), so config-only keys consumed elsewhere
    (``builder``, ``radiation_chunk_size``) are ignored here.
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


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def build_terrain(cfg: DictConfig, coords) -> TerrainData:
    terrain_cfg = cfg.terrain
    kind = terrain_cfg.kind
    if kind == "aquaplanet":
        return TerrainData.aquaplanet(coords)
    if kind == "from_file":
        return TerrainData.from_coords(
            coords,
            terrain_file=terrain_cfg.file,
            interpolate=terrain_cfg.get("interpolate", True),
        )
    if kind == "from_file_enveloped":
        return TerrainData.from_file(
            terrain_cfg.file, coords=coords,
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
    default) and the grid is an L47 hybrid, return the matching
    ECHAM ``lmidatm`` level-dependent profile (del² at top 4 levels, del⁴
    at 5-7, del⁶ at 8-9, del⁸ below) — that's the stability stack the
    grid was tuned for in ECHAM. T63L47 picks the 7-hour base timescale;
    T85L47 picks 3 h. Set ``cfg.diffusion.kind: default`` to force the
    uniform SPEEDY del² profile (24h temp / 12h vor_q / 2h div), or
    ``cfg.diffusion.kind: echam_t63_l47`` / ``echam_t85_l47`` to pin a
    specific factory regardless of grid. ``cfg.diffusion.scale`` still
    multiplies the chosen profile's timescales — keep the existing
    SPEEDY-tuned configs working unchanged.
    """
    diffusion = cfg.get("diffusion", None)
    kind = "auto" if diffusion is None else str(diffusion.get("kind", "auto"))
    scale = 1.0 if diffusion is None else float(diffusion.get("scale", 1.0))

    if kind == "auto":
        # Pick by grid: ECHAM lmidatm profile for L47 hybrid grids, SPEEDY
        # default otherwise. Match on the (vertical=hybrid, layers, truncation)
        # triple so this fires for both echam_t63_l47_hybrid and
        # echam_t85_l47_hybrid — and stays inert for SPEEDY T31L8 / Held-Suarez.
        grid_cfg = cfg.get("grid", None)
        layers = int(grid_cfg.get("layers", 0)) if grid_cfg is not None else 0
        truncation = int(grid_cfg.get("spectral_truncation", 0)) if grid_cfg is not None else 0
        vertical = str(grid_cfg.get("vertical", "")) if grid_cfg is not None else ""
        if vertical == "hybrid" and layers == 47:
            if truncation == 63:
                base = DiffusionFilter.echam_t63_l47()
            elif truncation == 85:
                base = DiffusionFilter.echam_t85_l47()
            else:
                base = DiffusionFilter.default()
        else:
            base = DiffusionFilter.default()
    elif kind == "default":
        base = DiffusionFilter.default()
    elif kind == "echam_t63_l47":
        base = DiffusionFilter.echam_t63_l47()
    elif kind == "echam_t85_l47":
        base = DiffusionFilter.echam_t85_l47()
    else:
        raise ValueError(
            f"Unknown diffusion.kind={kind!r}; expected one of "
            "'auto', 'default', 'echam_t63_l47', 'echam_t85_l47'."
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

    T_ref = jnp.asarray(model.dycore.primitive.reference_temperature)
    T_var_profile = T_profile - T_ref
    T_var_nodal = jnp.broadcast_to(
        T_var_profile[:, None, None], (nlev, nlon, nlat)
    ).astype(state.temperature_variation.dtype)
    state.temperature_variation = model.coords.horizontal.to_modal(T_var_nodal)

    # Humidity: rh * q_sat(T) below the tropopause cap, dry above.
    es = _ES0 * jnp.exp(_ES_A * (T_profile - _T0_C) / (T_profile - _ES_B))
    q_sat = 0.622 * es / jnp.maximum(p - es, 1.0)
    rh_profile = jnp.where(p > _RH_CAP_PRESSURE_PA, rh, 0.0)
    q_profile = jnp.clip(rh_profile * q_sat, 1e-8, 0.03)
    # Nondimensionalize q exactly as the canonical physics->dynamics bridge does
    # (``state_bridge.physics_state_to_dynamics_state``:
    # ``nondimensionalize(specific_humidity * gram/kilogram)``). The dynamics
    # ``State`` stores the NONDIMENSIONAL tracer; the forward bridge then
    # re-dimensionalizes with ``dimensionalize(q, gram/kilogram)`` (~x1000)
    # before handing the gridpoint state to physics. Storing the raw kg/kg
    # ``q_profile`` skipped this scaling, so physics saw q ~1000x too large
    # (~5 kg/kg); the cloud saturation adjustment (qs ~ 0.008) read it as hugely
    # supersaturated and dumped L*dq/cp ~ 7000 K of latent heat in a single step
    # -> every moist init blew up at step 1. Mirroring the bridge keeps the
    # gridpoint physics at the intended q and the moist resting state is stable.
    q_nondim = model.dycore.physics_specs.nondimensionalize(
        q_profile * units.gram / units.kilogram
    )
    q_dtype = state.tracers["specific_humidity"].dtype
    q_nodal = jnp.broadcast_to(
        q_nondim[:, None, None], (nlev, nlon, nlat)
    ).astype(q_dtype)
    state.tracers = {
        "specific_humidity": model.coords.horizontal.to_modal(q_nodal),
    }
    model._final_dycore_state = state


# ---------------------------------------------------------------------------
# Top-level model construction
# ---------------------------------------------------------------------------

def build_tracer_filter(cfg: DictConfig):
    """Build the optional dycore-side gridpoint tracer filter.

    Enabled via ``cfg.diffusion.tracer_positivity`` (default off). The only
    filter currently is mass-conserving positivity, which a spectral core
    applies as it projects to the physics gridpoint state so the sharp-source
    tracer fields of prescribed aerosol emissions stay non-negative at the
    dynamics→physics boundary (Gibbs ringing otherwise NaNs the microphysics;
    see issue #521). Returns ``None`` when disabled — a no-op on the dycore.
    """
    diffusion = cfg.get("diffusion", None)
    enabled = (False if diffusion is None
               else bool(diffusion.get("tracer_positivity", False)))
    if not enabled:
        return None
    from jcm.filters import MassConservingPositivity
    return MassConservingPositivity()


def build_model(cfg: DictConfig) -> Model:
    """Build a fully-configured ``Model`` from a Hydra config."""
    from jcm.dycore.dinosaur.dycore import DinosaurDycore

    coords = build_coords(cfg)
    physics = build_physics(cfg)
    physics = maybe_add_sponge(physics, cfg)
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
    dycore = DinosaurDycore(
        coords=coords,
        terrain=terrain,
        dt_seconds=time_step * 60.0,
        tracer_specs=tracer_specs,
        diffusion=diffusion,
        tracer_filter=tracer_filter,
    )
    return Model(
        dycore,
        physics=physics,
        time_step=time_step,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------

def build_forcing(cfg: DictConfig, coords):
    """Build a ``ForcingData`` from ``cfg.forcing``.

    ``kind: default`` returns ``None`` — ``Model.run`` then falls back to the
    aquaplanet ``default_forcing(coords.horizontal)``. ``kind: from_file``
    loads a netCDF boundary file via ``ForcingData.from_file``.

    Optionally attaches an ozone climatology (``cfg.forcing.ozone_file``) and
    prescribed aerosol emissions (``cfg.forcing.emissions_file``); both files
    must already be on the model horizontal grid.
    """
    forcing_cfg = cfg.get("forcing", None)
    if forcing_cfg is None or forcing_cfg.kind == "default":
        forcing = None
    elif forcing_cfg.kind == "from_file":
        from jcm.forcing import ForcingData
        forcing = ForcingData.from_file(forcing_cfg.file, coords=coords)
    else:
        raise ValueError(f"Unknown forcing.kind={forcing_cfg.kind!r}")
    forcing = _attach_ozone(forcing, forcing_cfg, coords)
    forcing = _attach_emissions(forcing, forcing_cfg, coords)
    return forcing


def _attach_ozone(forcing, forcing_cfg, coords):
    """Load the ozone climatology and attach to ``forcing``.

    No-op when the cfg has no ``ozone_file`` or the path is null. When
    ``forcing`` is ``None`` (``kind: default``) and an ozone file IS
    given, build the parent struct via ``default_forcing(...)`` so the
    aquaplanet cos²-latitude SST climatology is preserved — using
    ``ForcingData.zeros`` here would silently swap it for the uniform
    288.15 K placeholder, materially changing the boundary conditions
    for any run configured with only ``ozone_file``.
    """
    if forcing_cfg is None:
        return forcing
    ozone_file = forcing_cfg.get("ozone_file", None)
    if ozone_file in (None, "", "null"):
        return forcing
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
    path = forcing_cfg.get("emissions_file", None)
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


# ---------------------------------------------------------------------------
# Run + save
# ---------------------------------------------------------------------------

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

    # Apply any physical-constant overrides BEFORE the model is built, so the
    # dynamical core (which reads the live jcm.constants singleton at
    # construction) and the attribute-access physics both pick them up. Only base
    # fields may be set; derived constants (rd, cvd, rgrav, vtmpc*) follow.
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

    forcing = build_forcing(cfg, model.coords)
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
        )
    if cfg.init.kind == "jw":
        inject_jw_profile(model, rh=float(cfg.init.get("rh", 0.6)))
        return model.resume(
            forcing=forcing,
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
        )
    if cfg.init.kind == "balanced_isothermal":
        inject_balanced_isothermal_profile(model)
        return model.resume(
            forcing=forcing,
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
        )
    raise ValueError(f"Unknown init.kind={cfg.init.kind!r}")


def _load_states_from_cfg(cfg: DictConfig):
    """Open ``cfg.run.state_file`` and return a stacked ``PhysicsState``."""
    state_file = cfg.run.get("state_file", None)
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
        forcing = build_forcing(cfg, model.coords)

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
            )
        elif first_fresh_chunk and cfg.init.kind == "balanced_isothermal":
            inject_balanced_isothermal_profile(model)
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
            )
        elif first_fresh_chunk:
            preds = model.run(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
            )
        else:
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
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
        ds.to_netcdf(nc_path)
        print(f"  Saved {nc_path}")

        if ckpt_path:
            from jcm.checkpoint import save_checkpoint
            save_checkpoint(model, ckpt_path, elapsed_days=elapsed_sim_days)
            print(f"  Saved checkpoint to {ckpt_path}")

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
    ds.to_netcdf(str(output_path))
    logger.info("Wrote %s", output_path)
