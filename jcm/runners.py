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
from omegaconf import DictConfig

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

def _apply_param_overrides(base, overrides: dict | None):
    """Apply a ``{subgroup: {field: value}}`` override dict to a Parameters-like object.

    Works for any ``tree_math.struct``-style container whose subgroups are
    themselves field-based dataclasses (which covers both
    ``jcm.physics.speedy.params.Parameters`` and
    ``jcm.physics.echam.parameters.Parameters``). Unknown subgroups raise
    ``ValueError`` so typos don't silently no-op.
    """
    if not overrides:
        return base
    base_fields = dict(base.__dict__)
    for subgroup, subdict in overrides.items():
        if subgroup not in base_fields:
            raise ValueError(
                f"Unknown physics parameter subgroup {subgroup!r}; "
                f"choices: {sorted(base_fields)}"
            )
        sub = base_fields[subgroup]
        base_fields[subgroup] = sub.__class__(**{**sub.__dict__, **dict(subdict)})
    return base.__class__(**base_fields)


def _physics_param_overrides(cfg: DictConfig) -> dict:
    """Pull ``cfg.physics.params`` out of OmegaConf into a plain nested dict."""
    raw = cfg.physics.get("params", None)
    if raw is None:
        return {}
    from omegaconf import OmegaConf
    return OmegaConf.to_container(raw, resolve=True) or {}


def build_physics(cfg: DictConfig):
    """Build the physics package from ``cfg.physics``.

    Each package's own ``Parameters.default()`` is the source of truth for
    tunables. ``cfg.physics.params`` (a free-form nested dict) is walked at
    build time and applied via ``_apply_param_overrides``, so users can
    poke individual fields from the CLI without having to mirror the
    Parameters structure in YAML, e.g.::

        python -m jcm.main physics.params.convection.entrpen=4e-4
    """
    name = cfg.physics.name
    overrides = _physics_param_overrides(cfg)

    if name == "speedy":
        from jcm.physics.speedy.params import Parameters as SpeedyParameters
        from jcm.physics.speedy.speedy_terms import speedy_physics
        params = _apply_param_overrides(SpeedyParameters.default(), overrides)
        return speedy_physics(
            parameters=params,
            checkpoint_terms=cfg.physics.get("checkpoint_terms", True),
        )
    if name == "held_suarez":
        if overrides:
            raise ValueError(
                "Held-Suarez has no Parameters object; cfg.physics.params "
                "must be empty."
            )
        from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
        return held_suarez_physics()
    if name == "echam":
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.echam.parameters import Parameters as EchamParameters
        params = _apply_param_overrides(EchamParameters.default(), overrides)
        return echam_physics(
            parameters=params,
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

    Mutates ``model._final_modal_state`` in place. Follow with
    ``model.resume(...)`` rather than ``model.run(...)``.
    """
    from dinosaur.scales import units
    from jcm.constants import grav, p0s1_bg, rd

    model._final_modal_state = model._prepare_initial_modal_state(
        physics_state=None, random_seed=0,
    )
    state = model._final_modal_state
    p0_pa = p0s1_bg

    orog = jnp.asarray(model.terrain.orog)
    if jnp.any(orog > 1.0):
        # Hydrostatic balance with the actual isothermal T (288 K), not
        # ``_HYDROSTATIC_T_REF`` (260 K which is appropriate for the
        # JW lapse-rate profile). Using the matching T avoids an
        # initial-step pressure-temperature inconsistency.
        ps_pa_nodal = p0_pa * jnp.exp(-grav * orog / (rd * _JW_T_SFC))
        scale = float(model.physics_specs.nondimensionalize(1.0 * units.pascal))
        log_ps_nodal = jnp.log(ps_pa_nodal * scale)
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            log_ps_nodal[None, ...]
        )
    model._final_modal_state = state


def inject_jw_profile(model: Model, rh: float = 0.6) -> None:
    """Inject a Jablonowski-Williamson-style lapse-rate initial condition.

    Replaces ``model._final_modal_state`` (set up by the default isothermal
    rest atmosphere) with a vertical profile suitable for moist physics:

    * Temperature: 288 K at the surface, ICAO standard lapse 6.5 K/km, capped
      at 250 K so the semi-implicit reference temperature stays close.
    * Surface pressure: hydrostatically balanced against the model's
      orography when present (otherwise the isothermal init places air below
      ground on tall mountains and the run blows up).
    * Humidity: ``rh`` × q_sat(T) below ~200 hPa, zero above; clipped to a
      sensible range for q.

    Mutates ``model._final_modal_state`` in place. Follow with
    ``model.resume(...)`` rather than ``model.run(...)``.
    """
    from dinosaur.hybrid_coordinates import HybridCoordinates
    from dinosaur.scales import units

    from jcm.constants import grav, p0s1_bg, rd

    model._final_modal_state = model._prepare_initial_modal_state(
        physics_state=None, random_seed=0,
    )
    state = model._final_modal_state

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

    # Humidity: rh * q_sat(T) below the tropopause cap, dry above.
    es = _ES0 * jnp.exp(_ES_A * (T_profile - _T0_C) / (T_profile - _ES_B))
    q_sat = 0.622 * es / jnp.maximum(p - es, 1.0)
    rh_profile = jnp.where(p > _RH_CAP_PRESSURE_PA, rh, 0.0)
    q_profile = jnp.clip(rh_profile * q_sat, 1e-8, 0.03)
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
# Forcing
# ---------------------------------------------------------------------------

def build_forcing(cfg: DictConfig, coords):
    """Build a ``ForcingData`` from ``cfg.forcing``.

    ``kind: default`` returns ``None`` — ``Model.run`` then falls back to the
    aquaplanet ``default_forcing(coords.horizontal)``. ``kind: from_file``
    loads a netCDF boundary file via ``ForcingData.from_file``.
    """
    forcing_cfg = cfg.get("forcing", None)
    if forcing_cfg is None or forcing_cfg.kind == "default":
        return None
    if forcing_cfg.kind == "from_file":
        from jcm.forcing import ForcingData
        return ForcingData.from_file(forcing_cfg.file, coords=coords)
    raise ValueError(f"Unknown forcing.kind={forcing_cfg.kind!r}")


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
        inject_jw_profile(model)
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
    """
    import time

    from jcm.diagnostics import check_health, print_report

    if model is None:
        model = build_model(cfg)
    if forcing is None:
        forcing = build_forcing(cfg, model.coords)

    save_interval = float(cfg.run.save_interval)
    total_time = float(cfg.run.total_time)
    n_chunks = int(total_time / chunk_days) + 1

    reports: list[dict] = []
    elapsed_sim_days = 0.0
    total_wall = 0.0

    for i in range(n_chunks):
        remaining = total_time - elapsed_sim_days
        cur_chunk = min(chunk_days, remaining)
        if cur_chunk <= 0:
            break

        t0 = time.perf_counter()
        if i == 0 and cfg.init.kind == "jw":
            inject_jw_profile(model)
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
            )
        elif i == 0 and cfg.init.kind == "balanced_isothermal":
            inject_balanced_isothermal_profile(model)
            preds = model.resume(
                forcing=forcing,
                save_interval=save_interval,
                total_time=cur_chunk,
                output_averages=cfg.run.output_averages,
            )
        elif i == 0:
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
        ok, report = check_health(ds, i, elapsed_sim_days)
        report["wall_seconds"] = chunk_wall
        reports.append(report)
        print_report(report)

        nc_path = f"{output_prefix}_day{int(elapsed_sim_days)}.nc"
        ds.to_netcdf(nc_path)
        print(f"  Saved {nc_path}")

        if not ok:
            print(
                f"\n*** STOPPING: atmosphere unhealthy at "
                f"day {elapsed_sim_days:.0f} ***"
            )
            break

        sdph = elapsed_sim_days / (total_wall / 3600)
        print(
            f"  Wall: {chunk_wall:.1f}s this chunk, {total_wall:.0f}s total "
            f"({sdph:.0f} sim days/hr)"
        )

    return reports

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


def save_predictions(predictions: ModelPredictions, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = predictions.to_xarray()
    ds.to_netcdf(str(output_path))
    logger.info("Wrote %s", output_path)
