"""Build models and run simulations from a Hydra ``DictConfig``.

This is the bridge between the Hydra config groups in ``jcm/config/`` and the
construction of ``Model``, ``TerrainData``, ``DiffusionFilter`` and the various
physics packages. Keeps ``main.py`` minimal so other harnesses (notebooks,
integration tests) can import the same builders directly without going through
Hydra's CLI machinery.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from jcm.diffusion import DiffusionFilter
from jcm.model import Model, ModelPredictions
from jcm.terrain import TerrainData
from jcm.utils import get_coords


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------

def build_coords(cfg: DictConfig):
    """Build a ``CoordinateSystem`` from ``cfg.grid``.

    ``cfg.grid.vertical`` selects the vertical-coordinate family:
        ``speedy``      -- SPEEDY sigma boundaries (8 levels by default)
        ``held_suarez`` -- Held-Suarez sigma boundaries
        ``icon_hybrid`` -- ICON hybrid (a + b·P_s) coordinates
        ``sigma``       -- equidistant sigma coordinates
    """
    grid = cfg.grid
    layers = grid.layers
    truncation = grid.spectral_truncation

    vertical = grid.vertical
    if vertical == "speedy":
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        return get_speedy_coords(layers=layers, spectral_truncation=truncation)
    if vertical == "held_suarez":
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        return get_held_suarez_coords(layers=layers, spectral_truncation=truncation)
    if vertical == "icon_hybrid":
        from jcm.physics.icon.icon_levels import get_icon_levels
        return get_coords(
            vertical_coords=get_icon_levels(layers),
            spectral_truncation=truncation,
        )
    if vertical == "sigma":
        from dinosaur.sigma_coordinates import SigmaCoordinates
        return get_coords(
            vertical_coords=SigmaCoordinates.equidistant(layers),
            spectral_truncation=truncation,
        )
    raise ValueError(
        f"Unknown grid.vertical={vertical!r}; "
        "expected one of: speedy, held_suarez, icon_hybrid, sigma"
    )


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def build_physics(cfg: DictConfig):
    """Build the physics package from ``cfg.physics``."""
    name = cfg.physics.name
    if name == "speedy":
        from jcm.physics.speedy.speedy_terms import speedy_physics
        return speedy_physics(
            checkpoint_terms=cfg.physics.get("checkpoint_terms", True),
        )
    if name == "held_suarez":
        from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
        return held_suarez_physics()
    if name == "icon":
        from jcm.physics.icon.icon_terms import icon_physics
        return icon_physics(
            radiation_scheme=cfg.physics.radiation,
            cloud_scheme=cfg.physics.get("cloud_scheme", "1m"),
            checkpoint_terms=cfg.physics.get("checkpoint_terms", True),
        )
    raise ValueError(f"Unknown physics.name={name!r}")


def maybe_add_sponge(physics, cfg: DictConfig):
    """Append an ``UpperSponge`` term if ``cfg.run.sponge.levels > 0``."""
    sponge = cfg.run.get("sponge", None)
    if sponge is None or sponge.get("levels", 0) <= 0:
        return physics
    from jcm.physics.dissipation import UpperSponge
    return physics + UpperSponge(
        n_sponge_levels=int(sponge.levels),
        sponge_timescale_s=float(sponge.timescale_h) * 3600.0,
        enspodi=float(sponge.enspodi),
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
    raise ValueError(f"Unknown terrain.kind={kind!r}")


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def build_diffusion(cfg: DictConfig) -> DiffusionFilter:
    base = DiffusionFilter.default()
    diffusion = cfg.get("diffusion", None)
    scale = 1.0 if diffusion is None else float(diffusion.get("scale", 1.0))
    if scale == 1.0:
        return base
    return DiffusionFilter(
        div_timescale=base.div_timescale * scale,
        div_order=base.div_order,
        vor_q_timescale=base.vor_q_timescale * scale,
        vor_q_order=base.vor_q_order,
        temp_timescale=base.temp_timescale * scale,
        temp_order=base.temp_order,
    )


# ---------------------------------------------------------------------------
# Initial state injection (JW-style lapse-rate profile)
# ---------------------------------------------------------------------------

def inject_jw_profile(model: Model) -> None:
    """Replace the default isothermal init with a JW-style lapse-rate profile.

    Mirrors ``utils/run_icon_simulation.py:inject_realistic_profile`` so that
    moist ICON runs see a realistic T/q sounding instead of the isothermal
    rest atmosphere. Modifies ``model._final_modal_state`` in place; the
    caller is expected to follow with ``model.resume(...)``.
    """
    from dinosaur.hybrid_coordinates import HybridCoordinates
    from dinosaur.scales import units

    model._final_modal_state = model._prepare_initial_modal_state(
        physics_state=None, random_seed=0,
    )
    state = model._final_modal_state

    nlon, nlat = model.coords.horizontal.nodal_shape
    p0_pa = 101325.0
    if isinstance(model.coords.vertical, HybridCoordinates):
        sigma = jnp.asarray(model.coords.vertical.get_sigma_centers(p0_pa))
    else:
        sigma = jnp.asarray(model.coords.vertical.centers)
    nlev = sigma.size

    # Standard-atmosphere T(sigma); cap the cold tail so the semi-implicit
    # reference temperature stays close.
    p = sigma * p0_pa
    T_sfc, gamma = 288.0, 6.5e-3
    z = 8400.0 * jnp.log(p0_pa / p)
    T_profile = jnp.maximum(T_sfc - gamma * z, 250.0)

    # Hydrostatic-balance the surface pressure when there's nontrivial
    # orography, otherwise the isothermal-rest init produces air below ground.
    orog = jnp.asarray(model.terrain.orog)
    if jnp.any(orog > 1.0):
        Rd, grav, T_ref_avg = 287.04, 9.80665, 260.0
        ps_pa_nodal = p0_pa * jnp.exp(-grav * orog / (Rd * T_ref_avg))
        scale = float(model.physics_specs.nondimensionalize(1.0 * units.pascal))
        log_ps_nodal = jnp.log(ps_pa_nodal * scale)
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            log_ps_nodal[None, ...]
        )

    T_ref = jnp.asarray(model.primitive.reference_temperature)
    T_var_profile = T_profile - T_ref
    T_var_nodal = jnp.broadcast_to(
        T_var_profile[:, None, None], (nlev, nlon, nlat)
    ).astype(state.temperature_variation.dtype)
    state.temperature_variation = model.coords.horizontal.to_modal(T_var_nodal)

    # Humidity: 60% RH below 200 hPa
    es = 611.2 * jnp.exp(17.67 * (T_profile - 273.15) / (T_profile - 29.65))
    q_sat = 0.622 * es / jnp.maximum(p - es, 1.0)
    rh = jnp.where(p > 20000.0, 0.6, 0.0)
    q_profile = jnp.clip(rh * q_sat, 1e-8, 0.03)
    q_dtype = state.tracers["specific_humidity"].dtype
    q_nodal = jnp.broadcast_to(
        q_profile[:, None, None], (nlev, nlon, nlat)
    ).astype(q_dtype)
    state.tracers = {
        "specific_humidity": model.coords.horizontal.to_modal(q_nodal),
    }
    model._final_modal_state = state


# ---------------------------------------------------------------------------
# Top-level model construction
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> Model:
    """Build a fully-configured ``Model`` from a Hydra config."""
    coords = build_coords(cfg)
    physics = build_physics(cfg)
    physics = maybe_add_sponge(physics, cfg)
    terrain = build_terrain(cfg, coords)
    diffusion = build_diffusion(cfg)

    log_level = getattr(logging, cfg.run.log_level.upper(), logging.CRITICAL)
    return Model(
        coords=coords,
        physics=physics,
        terrain=terrain,
        diffusion=diffusion,
        time_step=cfg.run.time_step,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# Run + save
# ---------------------------------------------------------------------------

def run(cfg: DictConfig, model: Model | None = None) -> ModelPredictions:
    """Build the model (if not supplied) and run a simulation.

    Honours ``cfg.init.kind``: ``isothermal`` uses the default rest atmosphere;
    ``jw`` injects a JW-style profile and resumes.
    """
    if model is None:
        model = build_model(cfg)

    if cfg.init.kind == "isothermal":
        return model.run(
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
        )
    if cfg.init.kind == "jw":
        inject_jw_profile(model)
        return model.resume(
            save_interval=cfg.run.save_interval,
            total_time=cfg.run.total_time,
            output_averages=cfg.run.output_averages,
        )
    raise ValueError(f"Unknown init.kind={cfg.init.kind!r}")


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


def save_predictions(predictions: ModelPredictions, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = predictions.to_xarray()
    ds.to_netcdf(str(output_path))
    logger.info("Wrote %s", output_path)
