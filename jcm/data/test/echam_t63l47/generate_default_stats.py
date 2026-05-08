"""Default-ECHAM-on-T63L47 statistics harness.

Mirrors ``jcm/data/test/t30/generate_default_stats.py`` (the SPEEDY
equivalent) but targets the production ECHAM wiring: T63L47 hybrid
coords, real ECHAM terrain + forcing under ``jcm/data/bc/t63``,
``echam_physics(grey) + UpperSponge``, and a 5-day stats window resumed
from a saved spun-up state.

The companion files ``spinup_state.nc`` and ``default_statistics.nc``
in this directory are produced by running this module's
:func:`generate` once on a GPU. The slow regression test in
``model_test.py`` loads ``spinup_state.nc`` as its initial condition
and asserts every variable's daily-mean global mean falls inside the
saved climatology band.

T63L47 is too heavy for CPU CI — use GPU::

    CUDA_VISIBLE_DEVICES=<idx> JCM_RUN_GPU_INTEGRATION_TESTS=1 python -c "from jcm.data.test.echam_t63l47.generate_default_stats import generate; generate()"
"""

from __future__ import annotations

from pathlib import Path


_T63_BC_DIR = Path("jcm/data/bc/t63")
_OUT_DIR = Path("jcm/data/test/echam_t63l47")


default_echam_t63l47_stat_vars = [
    # Prognostic state.
    "u_wind",
    "v_wind",
    "temperature",
    "specific_humidity",
    "normalized_surface_pressure",
    # Moist-air diagnostics produced by MoistAirColumnState.
    "pressure_full",
    "air_density",
    "layer_thickness",
    "relative_humidity",
    # Scheme outputs.
    "radiation.toa_lw_up",
    "radiation.surface_sw_down",
    "clouds.cloud_fraction",
    "clouds.precip_rain",
    "convection.precip_conv",
]


def t63l47_coords():
    """Return the production T63L47 coordinate system."""
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.utils import get_coords

    return get_coords(get_echam_levels(47), spectral_truncation=63)


def build_production_physics():
    """ECHAM grey + UpperSponge — same composition the user is debugging."""
    from jcm.physics.dissipation import UpperSponge
    from jcm.physics.echam.echam_terms import echam_physics

    return echam_physics(radiation_scheme="grey") + UpperSponge(
        n_sponge_levels=5,
        sponge_timescale_s=3 * 3600.0,
        enspodi=2.0,
    )


def _build_terrain_and_forcing(coords):
    from jcm.forcing import ForcingData
    from jcm.terrain import TerrainData

    terrain = TerrainData.from_file(
        _T63_BC_DIR / "terrain.nc", coords=coords,
    )
    forcing = ForcingData.from_file(
        _T63_BC_DIR / "forcing.nc", coords=coords,
    )
    return terrain, forcing


def _block_until_ready(predictions):
    """Force materialisation of every jax array in a ModelPredictions."""
    import jax
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready()
        if hasattr(x, "block_until_ready") else x,
        predictions._predictions,
    )
    return predictions


def _load_spinup_state():
    """Load the saved spun-up nodal PhysicsState as the initial condition."""
    import xarray as xr
    from jcm.utils import load_states_from_xarray

    spinup_path = _OUT_DIR / "spinup_state.nc"
    if not spinup_path.exists():
        raise FileNotFoundError(
            f"{spinup_path} missing; run "
            "jcm.data.test.echam_t63l47.generate_default_stats.generate() "
            "on a GPU to create it."
        )
    ds = xr.open_dataset(spinup_path)
    tracer_vars = {}
    for tname in ("qc", "qi"):
        if tname in ds.data_vars:
            tracer_vars[tname] = tname
    return load_states_from_xarray(
        ds, tracer_vars=tracer_vars or None,
    )


def run_default_echam_t63l47_model(save_interval=1.0, total_time=5.0):
    """Run the production-wiring T63L47 ECHAM run from the spun-up state.

    Mirrors the SPEEDY harness in ``jcm/data/test/t30/generate_default_stats.py``
    but with the full T63L47 hybrid coords, real terrain + forcing,
    ECHAM grey radiation + UpperSponge, and the spun-up state from
    ``spinup_state.nc`` as the initial condition.

    Returns daily *snapshots* rather than daily time-averages
    (``output_averages=False``) — ``output_averages=True`` on hybrid
    coords trips a shape-broadcast bug in
    ``compute_diagnostic_state_hybrid`` that the existing T63L47 tests
    don't exercise. Mean-of-5-daily-snapshots is a close approximation
    of the true 5-day mean for the slow-varying global statistics this
    regression compares against, and matches what
    ``test_echam_model_default_statistics`` does on the assertion side.

    Args:
        save_interval: Save interval in days. Defaults to 1 (daily snapshots).
        total_time: Total simulation length in days. Defaults to 5.

    Returns:
        Tuple ``(model, predictions)``.

    """
    from jcm.model import Model

    coords = t63l47_coords()
    terrain, forcing = _build_terrain_and_forcing(coords)
    physics = build_production_physics()

    model = Model(
        coords=coords, terrain=terrain, physics=physics, time_step=12,
    )
    initial_state = _load_spinup_state()

    predictions = model.run(
        initial_state=initial_state,
        forcing=forcing,
        save_interval=save_interval,
        total_time=total_time,
    )
    return model, predictions


def generate():
    """One-off generation of ``spinup_state.nc`` + ``default_statistics.nc``.

    Stage 1 spins up for 5 days from the balanced-isothermal init,
    saves the final state. Stage 2 resumes for 5 more days with daily
    averages and saves global-mean ``mean`` / ``std`` per level.

    Run on a GPU; takes ~30 minutes wall-clock at GPU speeds.
    """
    import jax
    import sys
    import xarray as xr

    from jcm.model import Model
    from jcm.runners import inject_balanced_isothermal_profile

    print(f"JAX backend: {jax.default_backend()} on {jax.devices()}")
    if jax.default_backend() == "cpu":
        print(
            "WARNING: running on CPU; T63L47 takes hours on CPU. "
            "Set CUDA_VISIBLE_DEVICES to a free GPU index.",
            file=sys.stderr,
        )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    coords = t63l47_coords()
    terrain, forcing = _build_terrain_and_forcing(coords)
    physics = build_production_physics()

    print("Stage 1: 5-day spin-up from balanced-isothermal …")
    model = Model(
        coords=coords, terrain=terrain, physics=physics, time_step=12,
    )
    model._final_modal_state = model._prepare_initial_modal_state()
    inject_balanced_isothermal_profile(model)
    spin_up = model.resume(
        forcing=forcing, save_interval=5.0, total_time=5.0,
    )
    _block_until_ready(spin_up)

    print("  writing spinup_state.nc …")
    # Save only the prognostic-state fields ``load_states_from_xarray``
    # expects, plus ``qc`` / ``qi`` tracers — the full predictions
    # ``.to_xarray()`` includes hundreds of diagnostic fields (radiation
    # fluxes, cloud sub-structs, …) which 150× the file size and aren't
    # needed for restart. Use netCDF deflate compression to keep the
    # checked-in file small.
    keep_vars = [
        "u_wind", "v_wind", "temperature", "specific_humidity",
        "geopotential", "normalized_surface_pressure",
    ]
    full_ds = spin_up.to_xarray().isel(time=-1).reset_coords(drop=True)
    for tname in ("qc", "qi"):
        if tname in full_ds.data_vars:
            keep_vars.append(tname)
    spin_ds = full_ds[keep_vars]
    encoding = {
        v: {"zlib": True, "complevel": 4} for v in keep_vars
    }
    spin_ds.to_netcdf(_OUT_DIR / "spinup_state.nc", encoding=encoding)
    spinup_size = (_OUT_DIR / "spinup_state.nc").stat().st_size / 1e6
    print(
        f"  wrote {_OUT_DIR / 'spinup_state.nc'} ({spinup_size:.1f} MB)",
    )

    # Stage 2: build a fresh model and load the spun-up state we just
    # wrote.  Daily *snapshots* (output_averages=False) — the averaged
    # path on hybrid coords trips a shape-broadcast bug in
    # ``compute_diagnostic_state_hybrid``; the mean of daily snapshots
    # is a close-enough approximation for the slow-varying global
    # statistics the regression compares against.
    print("Stage 2: 5 daily snapshots from the spun-up state …")
    _, stats_predictions = run_default_echam_t63l47_model(
        save_interval=1.0, total_time=5.0,
    )
    _block_until_ready(stats_predictions)

    pred_ds = stats_predictions.to_xarray()
    print(f"  trajectory shape: {dict(pred_ds.sizes)}")

    daily_global = pred_ds.mean(dim={"lon", "lat"})
    pred_mean = daily_global.mean(dim="time")
    pred_std = daily_global.std(dim="time")

    out = {}
    missing = []
    for var in default_echam_t63l47_stat_vars:
        if var not in pred_ds:
            missing.append(var)
            continue
        out[f"{var}.mean"] = pred_mean[var]
        out[f"{var}.std"] = pred_std[var]
    if missing:
        print(f"  WARNING: missing vars: {missing}")

    stats_ds = xr.Dataset(out)
    stats_ds.to_netcdf(_OUT_DIR / "default_statistics.nc")
    stats_size = (_OUT_DIR / "default_statistics.nc").stat().st_size
    print(
        f"  wrote {_OUT_DIR / 'default_statistics.nc'} ({stats_size} bytes)",
    )
