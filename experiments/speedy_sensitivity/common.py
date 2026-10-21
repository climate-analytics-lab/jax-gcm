"""Shared setup for the SPEEDY sensitivity harness (growth, GFMIP FD atlas, jvp chains).

Builds SPEEDY at any truncation / level count on either dinosaur transport
scheme, with an optional hyperdiffusion multiplier, plus realistic boundary
conditions, a state library, GFMIP patches and the ocean-mean observables the
scripts share. See README.md in this directory.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import pickle
from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np

from jcm.diffusion import DiffusionFilter
from jcm.dycore.dinosaur.dycore import DinosaurDycore
from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.terrain import TerrainData

from config import add_config_args, config_tag, log  # noqa: F401  (re-exported for the scripts)

BC = resources.files("jcm.data.bc")
#: Time-varying boundary-condition fields frozen at their annual mean by ``frozen_bcs``.
SEASONAL_FIELDS = ("sea_surface_temperature", "sice_am", "snowc_am", "soilw_am", "stl_am")
#: Ocean-mean observables (same set as the published campaign figures).
OBS_NAMES = ("olr", "swtoa", "precnv", "precls", "cloudc", "cloudstr", "gse")


def build_model(truncation: int = 31, nlev: int = 8, advection: str = "eulerian",
                diffusion_scale: float = 1.0, time_step: float | None = None,
                sl_vertical_interpolation: str | None = None):
    """SPEEDY ``Model`` on realistic terrain; returns ``(model, coords, terrain)``.

    ``advection``: ``eulerian``, ``semi_lagrangian`` (dinosaur's native SL), or ``hybrid``
    (Eulerian horizontal + the campaign's semi-Lagrangian vertical step).
    """
    coords = get_speedy_coords(layers=nlev, spectral_truncation=truncation)
    terrain = TerrainData.from_coords(coords, terrain_file=BC / "t63" / "terrain.nc", interpolate=True)
    physics = speedy_physics()
    if time_step is None:
        # SPEEDY's own stable step for this grid (thin bottom layers at high nlev
        # need a shorter step); the Model picks it when it builds its own dycore.
        time_step = float(Model(coords=coords, terrain=terrain, physics=physics).dt_si.m) / 60.0
    d0 = DiffusionFilter.default()
    diffusion = dataclasses.replace(
        d0,
        div_timescale=d0.div_timescale / diffusion_scale,
        vor_q_timescale=d0.vor_q_timescale / diffusion_scale,
        temp_timescale=d0.temp_timescale / diffusion_scale,
    )
    dycore = DinosaurDycore(coords=coords, terrain=terrain, dt_seconds=time_step * 60.0,
                            diffusion=diffusion,
                            advection="eulerian" if advection == "hybrid" else advection,
                            eulerian_vertical_sl=advection == "hybrid",
                            sl_options=({"vertical_interpolation_order": sl_vertical_interpolation}
                                        if sl_vertical_interpolation else None))
    model = Model(dycore, terrain=terrain, physics=physics)
    return model, coords, terrain


def build_model_from_args(args):
    return build_model(args.truncation, args.nlev, args.advection, args.diffusion_scale, args.time_step,
                       getattr(args, "sl_vertical_interpolation", None))


def load_forcing(coords, frozen_bcs: bool = True) -> ForcingData:
    """Climatological SPEEDY forcing; ``frozen_bcs`` replaces each seasonal field by its annual mean.

    Frozen BCs + ``JCM_PERPETUAL_EQUINOX=1`` give a stationary background, which
    tangent / Jacobian / chain work needs. The GFMIP FD atlas uses the seasonal
    cycle (``frozen_bcs=False``), following the GFMIP protocol.
    """
    f = ForcingData.from_file(BC / "t30" / "clim" / "forcing.nc", coords=coords, align_mode="wrap_year")
    if frozen_bcs:
        for fld in SEASONAL_FIELDS:
            f = f.replace(**{fld: jnp.asarray(getattr(f, fld).values).mean(axis=0)})
    return f


def with_sst_delta(forcing: ForcingData, delta):
    """Return ``forcing`` with ``delta`` (lon, lat) added to the SST, frozen or time-varying."""
    sst = forcing.sea_surface_temperature
    if hasattr(sst, "values"):          # TimeSeries (seasonal BCs)
        return forcing.replace(sea_surface_temperature=sst.replace(values=sst.values + delta[None]))
    return forcing.replace(sea_surface_temperature=sst + delta)


def ocean_weights(coords, terrain, forcing) -> np.ndarray:
    """cos(lat) x ocean fraction x (1 - annual-mean sea ice), normalised to sum 1; shape (lon, lat)."""
    lats = np.rad2deg(np.asarray(coords.horizontal.latitudes))
    ocean = 1.0 - np.asarray(terrain.fmask)
    sic = forcing.sice_am
    sic = np.asarray(sic.values).mean(axis=0) if hasattr(sic, "values") else np.asarray(sic)
    if sic.ndim == 3:
        sic = sic.mean(axis=0)
    w = np.cos(np.deg2rad(lats))[None, :] * ocean * np.clip(1.0 - sic, 0.0, 1.0)
    return (w / w.sum()).astype(np.float32)


def obs_fields(predictions):
    """Window-mean 2-D fields of :data:`OBS_NAMES`, stacked: (nobs, lon, lat). Precip in mm/day."""
    ph = predictions.physics
    return jnp.stack([
        ph["_longwave_rad"].ftop[0],
        ph["_shortwave_rad"].ftop[0],
        86.4 * ph["_convection"].precnv[0],
        86.4 * ph["_condensation"].precls[0],
        ph["_shortwave_rad"].cloudc[0],
        ph["_shortwave_rad"].cloudstr[0],
        ph["_shortwave_rad"].gse[0],
    ])


# ---------------------------------------------------------------------------
# State library: spun-up dycore states, stored as host arrays in chunk files.
# ---------------------------------------------------------------------------

def spin_library(model, forcing, n: int, days: float, batch: int, seed0: int = 0):
    """``n`` dycore states spun ``days`` from independent random initial states (batched with vmap)."""
    spin = jax.jit(jax.vmap(lambda s: model.run_from_state(
        initial_state=s, forcing=forcing, save_interval=days, total_time=days)[0].dynamics))
    out = []
    for i in range(seed0, seed0 + n, batch):
        seeds = range(i, min(i + batch, seed0 + n))
        st = [model._prepare_initial_dycore_state(random_seed=s) for s in seeds]
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *st)
        out.append(jax.tree_util.tree_map(np.asarray, jax.block_until_ready(spin(stacked))))
    return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *out)


def save_library(states, directory: pathlib.Path, chunk: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    n = jax.tree_util.tree_leaves(states)[0].shape[0]
    for k, i in enumerate(range(0, n, chunk)):
        part = jax.tree_util.tree_map(lambda x, i=i: x[i:i + chunk], states)
        with open(directory / f"chunk_{k:03d}.pkl", "wb") as fh:
            pickle.dump(part, fh)


def load_library(directory: pathlib.Path, n: int | None = None):
    """Concatenate chunk files into one batched state pytree (device arrays), first ``n`` states."""
    files = sorted(pathlib.Path(directory).glob("chunk_*.pkl"))
    if not files:
        raise FileNotFoundError(f"no chunk_*.pkl in {directory}; run spinup.py first")
    parts = []
    for f in files:
        with open(f, "rb") as fh:
            parts.append(pickle.load(fh))
    st = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *parts)
    if n is not None:
        st = jax.tree_util.tree_map(lambda x: x[:n], st)
    return jax.tree_util.tree_map(jnp.asarray, st)


def take(states, idx):
    idx = jnp.asarray(idx)
    return jax.tree_util.tree_map(lambda x: x[idx], states)


def library_dir(root: str, tag: str, frozen: bool) -> pathlib.Path:
    return pathlib.Path(root) / f"statelib-{tag}-{'frozen' if frozen else 'seasonal'}"


def env_summary() -> str:
    return (f"JCM_PERPETUAL_EQUINOX={os.environ.get('JCM_PERPETUAL_EQUINOX', '0')} "
            f"JCM_TANGENT_DECAY={os.environ.get('JCM_TANGENT_DECAY', '0')}")
