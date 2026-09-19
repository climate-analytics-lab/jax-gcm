"""Default-ECHAM-on-T63L47 statistics harness.

Mirrors ``jcm/data/test/t30/generate_default_stats.py`` (the SPEEDY
equivalent) on the T63L47 hybrid grid: real ECHAM terrain + forcing
under ``jcm/data/bc/t63``, ``echam_physics(grey) + UpperSponge``, and a
5-day stats window resumed from a saved spun-up state.

Radiation here is **grey**, which is deliberately not the
``physics=echam`` production composition — that composes RRTMGP, and
``jcm/config/physics/echam.yaml`` records grey-ECHAM as an unsupported
hybrid with no CLI route. Grey is the right choice for *this* harness
because what it regresses is the hybrid-coordinate dynamics–physics
coupling, which a cheap radiation scheme exercises just as well while
keeping the run affordable; the bands in this directory describe the
grey composition and nothing else. Composing RRTMGP here would be a new
fixture with its own bands, not a regeneration of these.

The companion files ``spinup_state.nc`` and ``default_statistics.nc``
in this directory are produced by running this module's
:func:`generate` once on a GPU. The slow regression test in
``model_test.py`` loads ``spinup_state.nc`` as its initial condition
and asserts every variable's daily-mean global mean falls inside the
saved climatology band. Both files are written by the same
:func:`generate` call and are only meaningful as a pair: the bands
describe the five days that *follow* the saved state, so regenerating
one without the other leaves the test comparing a trajectory against
bands drawn from a different starting point.

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


def _load_spinup_state(physics):
    """Load the saved spun-up nodal PhysicsState as the initial condition.

    Returns the state in the top-first physics frame the model expects;
    ``load_states_from_xarray`` handles the file's on-disk orientation.

    ``physics`` supplies the tracer list: the loader picks up whatever
    ``required_tracers()`` declares and the file carries, so the spun-up
    ``qc``/``qi`` arrive rather than starting the run from a clear sky
    (#718). This used to be a local ``qc``/``qi`` scan here, which left every
    other caller — and every other tracer — unprotected.
    """
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
    return load_states_from_xarray(
        ds, required_tracers=physics.required_tracers(),
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
    initial_state = _load_spinup_state(physics)

    predictions = model.run(
        initial_state=initial_state,
        forcing=forcing,
        save_interval=save_interval,
        total_time=total_time,
    )
    return model, predictions


def stage2_global_mean():
    """Global mean of the stats window, exactly as the regression compares it.

    ``(time, lon, lat)``-mean of every variable in
    :data:`default_echam_t63l47_stat_vars`, i.e. the quantity
    ``test_echam_model_default_statistics`` checks against the stored band.
    """
    _, predictions = run_default_echam_t63l47_model(
        save_interval=1.0, total_time=5.0,
    )
    _block_until_ready(predictions)
    pred_ds = predictions.to_xarray()
    means = pred_ds.mean(dim={"time", "lon", "lat"})
    present = [v for v in default_echam_t63l47_stat_vars if v in means]
    return means[present]


def write_stage2_global_mean(path):
    """Subprocess entry point for one reproducibility repeat.

    Kept module-level so :func:`generate` can invoke it with
    ``python -c`` in a *separate process*; see the ``.noise`` discussion
    in :func:`generate`.
    """
    stage2_global_mean().to_netcdf(path)


def _measure_reproducibility(in_process_mean, n_repeats, tmp_dir):
    """Peak-to-peak spread of the stats window over independent repeats.

    Returns a ``Dataset`` of per-variable, per-level ``max - min`` across
    ``n_repeats + 1`` runs of the *same* five days: the one already run in
    this process plus ``n_repeats`` run in fresh subprocesses.

    Separate processes are the point. Two runs in one process agree to
    ~3e-3 m/s in ``u_wind``; a run in a different process disagrees by
    ~4e-2 m/s, an order of magnitude more, and it is the larger number the
    band has to survive. Repeating in-process would measure the wrong
    thing and produce a floor that is too small by 10x.
    """
    import subprocess
    import sys

    import xarray as xr

    members = [in_process_mean]
    for i in range(n_repeats):
        out = Path(tmp_dir) / f"repeat_{i}.nc"
        print(f"  reproducibility repeat {i + 1}/{n_repeats} …", flush=True)
        subprocess.run(
            [sys.executable, "-c",
             "from jcm.data.test.echam_t63l47.generate_default_stats "
             "import write_stage2_global_mean as w; "
             f"w({str(out)!r})"],
            check=True,
        )
        members.append(xr.open_dataset(out).load())

    stacked = xr.concat(members, dim="_repeat")
    return stacked.max(dim="_repeat") - stacked.min(dim="_repeat")


def generate(n_reproducibility_repeats=3):
    """One-off generation of ``spinup_state.nc`` + ``default_statistics.nc``.

    Stage 1 spins up for 5 days from the balanced-isothermal init and
    saves the final state. Stage 2 resumes for 5 more days of daily
    snapshots and saves the global-mean ``mean`` / ``std`` per level.
    Stage 3 repeats stage 2 in fresh subprocesses and saves ``noise``,
    the peak-to-peak spread of the result across those repeats.

    Why ``noise`` exists
    --------------------
    ``std`` is the temporal spread of five daily snapshots of a field
    that is still trending, so it measures the trend, not the
    reproducibility of the measurement — and wherever the trend turns
    over, it collapses. Two real examples in these bands: ``u_wind``
    near sigma 0.24, where sigma falls to 5.1e-3 m/s while its
    neighbours sit at 1.4e-2 - 4.4e-2, and ``pressure_full`` on the
    near-pure-``a`` levels, where sigma is 4.9e-4 Pa — one float32 ULP
    at 7405.9 Pa. A +/-3 sigma band there is narrower than the noise
    floor of the computation, so the test fails on a new GPU, a new XLA
    version or simply a different process, and the failure is
    indistinguishable from the physics regression the test exists to
    catch.

    The floor therefore has to be the *measured* reproducibility, per
    variable and per level. Cheaper scales were tried against four
    independent runs of these same five days and none of them works:

    * ``rel * |mean|`` fails for ``u_wind`` and ``v_wind``, whose
      global-mean profile passes through zero — the floor vanishes
      exactly where the band is pinched. Sizing it for those variables
      instead needs ``rel ~ 2e-2``, which on ``temperature`` is a
      +/-5.5 K band and on ``pressure_full`` a +/-600 Pa one.
    * a fraction of the column-maximum sigma blinds any variable with a
      large vertical dynamic range: for ``specific_humidity`` it sets
      the stratospheric floor from a tropospheric sigma, widening those
      bands 10-30x and hiding any stratospheric moisture error.

    The measured spread has none of those failure modes because it is
    taken where the band is used.

    ``noise`` is one of three things the regression does to a stored
    ``std`` before treating it as a band, and on these fixtures it is
    the weakest of them: repeats spawned from one parent process agree
    far more closely than runs launched differently do, so it sizes the
    *within-harness* floor only. The other two live in
    ``model_test.test_echam_model_default_statistics``: the band uses
    the ``std`` of a small vertical neighbourhood rather than of the
    single level, which is what actually absorbs a trend-crossing pinch
    such as ``u_wind``'s, and it treats a ``std`` at or below float32
    resolution as carrying no information, which is what catches
    ``pressure_full``'s one-ULP levels. Measured against an independent
    reproduction of these five days, the three together take the worst
    excursion from 2.29 band half-widths to 0.43, widening the typical
    band by 1.0-1.4x (3.1x for ``specific_humidity``, whose vertical
    ``std`` profile is steepest).

    Run on a GPU. Stage 1 and each stage-2 run take ~90 s, so the
    default three repeats put the whole call at roughly 8 minutes.

    Give the card room for two processes. This one keeps its device pool
    while each repeat runs beside it, so stage 3 wants headroom for
    ~10 GB twice over; on an otherwise-busy A100 XLA logs
    ``CUDA_ERROR_OUT_OF_MEMORY`` and retries into a smaller allocation,
    and on a fuller card it would fail outright rather than retry.

    Args:
        n_reproducibility_repeats: Stage-2 repeats used to size
            ``noise``. Each runs in its own process; 0 skips stage 3 and
            writes no ``noise``, which the regression test then rejects.

    """
    import jax
    import sys
    import tempfile

    import xarray as xr

    from jcm.initial_states import balanced_isothermal_state
    from jcm.model import Model

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
    spin_up = model.run(
        initial_state=balanced_isothermal_state(model),
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

    # Stage 3: how far apart do independent repeats of stage 2 land?
    # That spread, not the five-snapshot ``std``, is what the band must
    # never be narrower than — see this function's docstring.
    noise = None
    if n_reproducibility_repeats:
        print(
            f"Stage 3: {n_reproducibility_repeats} independent repeats of "
            "stage 2 for the band floor …",
        )
        present = [v for v in default_echam_t63l47_stat_vars if v in pred_ds]
        with tempfile.TemporaryDirectory() as tmp_dir:
            noise = _measure_reproducibility(
                pred_ds[present].mean(dim={"time", "lon", "lat"}),
                n_reproducibility_repeats,
                tmp_dir,
            )

    out = {}
    missing = []
    for var in default_echam_t63l47_stat_vars:
        if var not in pred_ds:
            missing.append(var)
            continue
        out[f"{var}.mean"] = pred_mean[var]
        out[f"{var}.std"] = pred_std[var]
        if noise is not None:
            out[f"{var}.noise"] = noise[var]
    if missing:
        print(f"  WARNING: missing vars: {missing}")

    stats_ds = xr.Dataset(out)
    stats_ds.to_netcdf(_OUT_DIR / "default_statistics.nc")
    stats_size = (_OUT_DIR / "default_statistics.nc").stat().st_size
    print(
        f"  wrote {_OUT_DIR / 'default_statistics.nc'} ({stats_size} bytes)",
    )
