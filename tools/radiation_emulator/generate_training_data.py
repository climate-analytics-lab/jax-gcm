r"""Offline RRTMGP training-data generator for the radiation NN emulator.

Drives the in-repo ``radiation_scheme_rrtmgp`` over a batch of
atmospheric columns and writes the RAW PHYSICAL INPUTS alongside the
resulting flux profiles. Raw fields (not network features) are stored
deliberately: the emulator's feature layout is still being tuned and must
be changeable without regenerating the (expensive) labels.

McICA seed averaging
--------------------
RRTMGP samples clouds stochastically (one binary sub-column per g-point),
so a single call carries several W/m^2 of sampling noise on the all-sky
fluxes. Labels are therefore averaged over ``--n-seeds`` independent
draws. The draw is varied through ``model_step``, the *traced* int32 that
``mcica.column_key`` folds into the PRNG key — not through ``base_seed``,
which is a Python static and would force a full XLA re-trace (~40 s) per
seed. ``mcica_freeze_step`` must stay 0 (the default) or ``model_step``
is ignored and every "seed" returns the same draw.

Column sources are pluggable: a source is a function returning a batch of
raw column arrays, and the RRTMGP driving/labelling below is entirely
source-agnostic. Two are implemented (``trajectory``, ``perturbation``);
see ``COLUMN_SOURCES``.

Usage::

    python tools/radiation_emulator/generate_training_data.py \
        --source perturbation --n-columns 4096 --nlev 47 \
        --n-seeds 8 --out training_data.nc

    python tools/radiation_emulator/generate_training_data.py \
        --source trajectory --state-file run/output.nc \
        --n-columns 20000 --out traj_labels.nc
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# Vertical convention for every array in a column batch: level index 0 is
# the model top (TOA-first), matching the jcm physics column convention.
# ``radiation_scheme_rrtmgp`` detects the input orientation and returns
# flux profiles in the same order, so normalising here means the stored
# labels are TOA-first too.

# Per-column fields. Split by shape so the netCDF writer and the batch
# validator both work off one declaration.
PROFILE_FIELDS = (
    "temperature", "specific_humidity", "pressure_levels",
    "layer_thickness", "air_density", "cloud_water", "cloud_ice",
    "cloud_fraction", "ozone_vmr",
)
INTERFACE_FIELDS = ("pressure_interfaces",)
SCALAR_FIELDS = (
    "co2_vmr", "surface_temperature", "surface_albedo_vis",
    "surface_albedo_nir", "surface_emissivity", "latitude", "longitude",
    # Solar geometry is stored as well as the cos_zenith it produces:
    # cos_zenith is what the scheme actually uses, but the phases are
    # what make a generation reproducible.
    "orbital_phase", "synodic_phase",
)
SW_BAND_FIELDS = ("aod_sw_per_band", "ssa_sw_per_band", "asy_sw_per_band")
LW_BAND_FIELDS = ("aod_lw_per_band", "ssa_lw_per_band", "asy_lw_per_band")

# Flux labels, all on interfaces (nlev+1), TOA-first. The clear-sky half
# only exists when the scheme is called with compute_cre=True.
LABEL_FIELDS = (
    "sw_flux_up", "sw_flux_down", "lw_flux_up", "lw_flux_down",
    "sw_flux_up_clear", "sw_flux_down_clear",
    "lw_flux_up_clear", "lw_flux_down_clear",
)

# Lowest pressure (Pa) at which the synthetic sweep will place cloud.
_MIN_CLOUD_PRESSURE = 1.0e4


# ---------------------------------------------------------------------------
# RRTMGP driving / labelling (source-agnostic)
# ---------------------------------------------------------------------------


def band_counts():
    """Return ``(n_bnd_sw, n_bnd_lw, sw_centers_nm, lw_centers_nm)``.

    Loads the ~25 MB k-distribution tables on first call, so this (like
    every other use of ``_ensure_rrtmgp``) must happen outside any trace.
    """
    from jcm.physics.radiation.band_config import RadiationBandConfig
    from jcm.physics.radiation.rrtmgp import _ensure_rrtmgp

    cfg = RadiationBandConfig.from_rrtmgp(_ensure_rrtmgp())
    return (
        len(cfg.sw_band_centers_nm), len(cfg.lw_band_centers_nm),
        np.asarray(cfg.sw_band_centers_nm),
        np.asarray(cfg.lw_band_centers_nm),
    )


def make_labeller(base_seed: int = 0):
    """Return ``run(batch, model_step) -> dict`` of per-seed flux labels.

    The returned callable is jitted and vmapped over the leading column
    axis; ``model_step`` is the traced McICA draw selector (see the
    module docstring). ``base_seed`` and ``compute_cre`` are Python
    statics and take ``in_axes=None``, mirroring ``_compute_full``.
    """
    import jax
    import jax.numpy as jnp

    from jcm.forcing import SolarGeometry
    from jcm.physics.aerosol.aerosol_types import AerosolData
    from jcm.physics.radiation.radiation_types import RadiationParameters
    from jcm.physics.radiation.rrtmgp import (
        _ensure_rrtmgp, radiation_scheme_rrtmgp,
    )

    # Table load must happen before the trace: doing it inside would leak
    # tracers into the module-global RRTMGP instance.
    _ensure_rrtmgp()
    parameters = RadiationParameters.default()

    # Column-leading in_axes, in the order of the positional signature.
    # ``solar`` is vmapped here (unlike in ``_compute_full``, where one
    # date serves the whole grid) because the sweep source sets the solar
    # geometry per column to hit a target cos_zenith.
    in_axes = (
        0, 0, 0, 0, 0,       # T, q, p_full, p_half, dz
        0, 0, 0, 0,          # rho, qc, qi, cf
        0, 0, 0, 0,          # T_sfc, alb_vis, alb_nir, emis
        0, 0, 0,             # solar, lat, lon
        None, 0,             # parameters, aerosol
        0, None, None, None,  # col_index, model_step, base_seed, compute_cre
        0, 0,                # ozone_vmr, co2_vmr
    )
    vmapped = jax.vmap(radiation_scheme_rrtmgp, in_axes=in_axes)

    def _f32(x):
        return jnp.asarray(x, dtype=jnp.float32)

    @jax.jit
    def run(batch, model_step):
        ncol, nlev = batch["temperature"].shape
        zeros_col = jnp.zeros((ncol,), dtype=jnp.float32)
        zeros_prof = jnp.zeros((ncol, nlev), dtype=jnp.float32)
        # Only ``cdnc_factor`` and the per-band optics are read by the
        # scheme; the rest of AerosolData is required by the struct but
        # unused, so it carries neutral values.
        aerosol = AerosolData(
            aod_profile=zeros_prof,
            ssa_profile=zeros_prof,
            asy_profile=zeros_prof,
            aod_total=zeros_col,
            aod_anthropogenic=zeros_col,
            aod_background=zeros_col,
            cdnc_factor=jnp.ones((ncol,), dtype=jnp.float32),
            Nccn=zeros_col,
            angstrom=zeros_col,
            aod_sw_per_band=_f32(batch["aod_sw_per_band"]),
            ssa_sw_per_band=_f32(batch["ssa_sw_per_band"]),
            asy_sw_per_band=_f32(batch["asy_sw_per_band"]),
            aod_lw_per_band=_f32(batch["aod_lw_per_band"]),
            ssa_lw_per_band=_f32(batch["ssa_lw_per_band"]),
            asy_lw_per_band=_f32(batch["asy_lw_per_band"]),
        )
        solar = SolarGeometry(
            tyear=_f32(batch["orbital_phase"]) / (2.0 * jnp.pi),
            orbital_phase=_f32(batch["orbital_phase"]),
            synodic_phase=_f32(batch["synodic_phase"]),
        )
        _, diag = vmapped(
            _f32(batch["temperature"]), _f32(batch["specific_humidity"]),
            _f32(batch["pressure_levels"]), _f32(batch["pressure_interfaces"]),
            _f32(batch["layer_thickness"]), _f32(batch["air_density"]),
            _f32(batch["cloud_water"]), _f32(batch["cloud_ice"]),
            _f32(batch["cloud_fraction"]),
            _f32(batch["surface_temperature"]),
            _f32(batch["surface_albedo_vis"]),
            _f32(batch["surface_albedo_nir"]),
            _f32(batch["surface_emissivity"]),
            solar, _f32(batch["latitude"]), _f32(batch["longitude"]),
            parameters, aerosol,
            jnp.arange(ncol, dtype=jnp.int32), jnp.int32(model_step),
            base_seed, True,
            _f32(batch["ozone_vmr"]),
            jnp.broadcast_to(_f32(batch["co2_vmr"])[:, None], (ncol, nlev)),
        )
        out = {k: getattr(diag, k) for k in LABEL_FIELDS}
        # The scheme carries cos_zenith as a length-1 per-column vector.
        out["cos_zenith"] = diag.cos_zenith.reshape(ncol)
        out["total_cloud_cover"] = diag.total_cloud_cover
        return out

    return run


def label_batch(batch, n_seeds: int, base_seed: int = 0, labeller=None):
    """Label one column batch, averaging the fluxes over ``n_seeds`` draws.

    Returns ``(labels, per_seed)`` where ``labels`` holds the seed-mean
    flux profiles plus the deterministic diagnostics, and ``per_seed``
    stacks the raw draws ``(n_seeds, ncol, nlev+1)`` so callers can
    quantify the residual McICA noise.
    """
    if labeller is None:
        labeller = make_labeller(base_seed)
    draws = [
        {k: np.asarray(v) for k, v in labeller(batch, seed).items()}
        for seed in range(n_seeds)
    ]
    # A frozen McICA step makes every draw identical, so the averaging
    # silently becomes a no-op that still returns plausible labels. Catch
    # it on the data rather than trusting the parameter default.
    if n_seeds > 1 and np.array_equal(draws[0]["sw_flux_up"],
                                      draws[-1]["sw_flux_up"]):
        raise ValueError(
            "McICA draws are identical across seeds, so seed averaging is "
            "doing nothing. Check that RadiationParameters.mcica_freeze_step "
            "is 0 and that the columns are not all cloud-free."
        )
    per_seed = {
        k: np.stack([d[k] for d in draws], axis=0) for k in LABEL_FIELDS
    }
    labels = {k: per_seed[k].mean(axis=0) for k in LABEL_FIELDS}
    # cos_zenith is a pure function of the geometry and total_cloud_cover
    # is a McICA statistic; both come from the first draw / its mean.
    labels["cos_zenith"] = draws[0]["cos_zenith"]
    labels["total_cloud_cover"] = np.mean(
        [d["total_cloud_cover"] for d in draws], axis=0,
    )
    return labels, per_seed


# ---------------------------------------------------------------------------
# Shared column-building helpers
# ---------------------------------------------------------------------------


def _air_density(pressure, temperature):
    """Ideal-gas air density rho = p / (Rd T), kg/m3."""
    import jcm.constants as c

    return pressure / (float(c.rd) * temperature)


def _layer_thickness(pressure, temperature):
    """Hydrostatic layer thickness dz = dp / (rho g), m, TOA-first.

    The top layer reuses the thickness of the one below it, matching the
    helper the radiation tests build synthetic columns with.
    """
    import jcm.constants as c

    rho = _air_density(pressure, temperature)
    dp = np.diff(pressure, axis=-1)
    rho_mid = 0.5 * (rho[..., 1:] + rho[..., :-1])
    dz = np.zeros_like(pressure)
    dz[..., 1:] = dp / (rho_mid * float(c.grav))
    dz[..., 0] = dz[..., 1]
    return dz


def _interfaces_from_levels(pressure):
    """Half-level pressures (nlev+1) bracketing TOA-first full levels."""
    mid = 0.5 * (pressure[..., :-1] + pressure[..., 1:])
    top = np.maximum(2.0 * pressure[..., :1] - mid[..., :1], 1.0)
    sfc = 2.0 * pressure[..., -1:] - mid[..., -1:]
    return np.concatenate([top, mid, sfc], axis=-1)


def _per_band_optics(aod_550, ssa550, asy550, angstrom, band_centers_nm):
    """Per-band ``(ncol, n_bnd, nlev)`` optics from 550 nm references.

    Wraps the MACv2-SP closed-form wavelength scaling
    (``per_band_optical_properties``) so both column sources build their
    per-band aerosol the same way the online scheme does.
    """
    import jax.numpy as jnp

    from jcm.physics.aerosol.macv2_sp import per_band_optical_properties

    # The helper broadcasts a (n_bnd, 1, 1) wavelength axis against
    # (ncol, nlev) references, so per-column scalars gain a level axis.
    def _profile(x):
        x = jnp.asarray(x)
        return x[..., None] if x.ndim == 1 else x

    aod, ssa, asy = per_band_optical_properties(
        jnp.asarray(aod_550), _profile(ssa550), _profile(asy550),
        _profile(angstrom), jnp.asarray(band_centers_nm),
    )
    shape = (len(band_centers_nm),) + aod_550.shape
    to_col = lambda a: np.moveaxis(  # noqa: E731
        np.asarray(np.broadcast_to(a, shape)), 0, 1,
    )
    return to_col(aod), to_col(ssa), to_col(asy)


def _finalize_batch(batch, n_bnd_sw, n_bnd_lw):
    """Validate shapes and cast a raw column batch to float32/float64 numpy."""
    ncol, nlev = batch["temperature"].shape
    expected = {}
    for name in PROFILE_FIELDS:
        expected[name] = (ncol, nlev)
    for name in INTERFACE_FIELDS:
        expected[name] = (ncol, nlev + 1)
    for name in SCALAR_FIELDS:
        expected[name] = (ncol,)
    for name in SW_BAND_FIELDS:
        expected[name] = (ncol, n_bnd_sw, nlev)
    for name in LW_BAND_FIELDS:
        expected[name] = (ncol, n_bnd_lw, nlev)

    out = {}
    for name, shape in expected.items():
        if name not in batch:
            raise KeyError(f"column source did not provide {name!r}")
        arr = np.asarray(batch[name], dtype=np.float64)
        if arr.shape != shape:
            raise ValueError(
                f"{name}: expected shape {shape}, got {arr.shape}",
            )
        out[name] = arr
    return out


def _solar_geometry_for_cos_zenith(rng, target_cos_zenith, n_scan=2880):
    """Build solar geometry that realises a target cos(zenith) per column.

    The scheme derives cos(zenith) from (lat, lon, orbital phase, synodic
    phase) rather than taking it as an input, so the sweep must solve for
    a geometry instead of setting mu0 directly. Two steps:

    1. Draw the latitude uniformly from the band where the target is
       reachable at some hour of that day. Over a day mu0 sweeps
       ``[-cos(lat + decl), cos(lat - decl)]``, so reachability needs
       ``|lat - decl| <= A`` and ``|lat + decl| <= 180 - A`` with
       ``A = arccos(target)`` — the second bound is what excludes
       polar-day columns, whose sun never drops to a low target. Latitude
       stays free to vary within that band: solving for latitude alone
       would make mu0 a deterministic function of it, a shortcut the
       emulator could learn in place of the physics.
    2. Scan one day of synodic phases and keep the closest hour angle.

    Returns ``(latitude, longitude, orbital_phase, synodic_phase, mu0)``
    with the ACHIEVED mu0 (grid-resolution close to the target).
    """
    import jax.numpy as jnp
    from jax_solar import OrbitalTime, get_declination, get_solar_sin_altitude

    n = len(target_cos_zenith)
    orbital_phase = 2.0 * np.pi * rng.random(n)
    longitude = 360.0 * rng.random(n)
    declination = np.degrees(np.asarray(get_declination(
        jnp.asarray(orbital_phase, jnp.float32))))
    offset = np.degrees(np.arccos(np.clip(target_cos_zenith, -1.0, 1.0)))
    lo = np.maximum(declination - offset, offset - 180.0 - declination)
    hi = np.minimum(declination + offset, 180.0 - offset - declination)
    latitude = np.clip(lo + rng.random(n) * (hi - lo), -90.0, 90.0)

    scan = np.linspace(0.0, 2.0 * np.pi, n_scan, endpoint=False)
    mu = np.asarray(get_solar_sin_altitude(
        OrbitalTime(
            orbital_phase=jnp.asarray(orbital_phase, jnp.float32)[:, None],
            synodic_phase=jnp.asarray(scan, jnp.float32)[None, :],
        ),
        jnp.asarray(longitude, jnp.float32)[:, None],
        jnp.asarray(latitude, jnp.float32)[:, None],
    ))
    best = np.argmin(np.abs(mu - np.asarray(target_cos_zenith)[:, None]), axis=1)
    return (latitude, longitude, orbital_phase, scan[best],
            mu[np.arange(n), best])


# ---------------------------------------------------------------------------
# Column source: designed perturbation sweep
# ---------------------------------------------------------------------------


def _latin_hypercube(rng, n_samples, n_dim):
    """Stratified LHS draws in [0, 1) — one sample per stratum per axis.

    A full outer product over ~12 physical axes is unaffordable; LHS
    keeps every 1-D marginal uniformly covered at any sample count.
    """
    cut = (np.arange(n_samples)[:, None] + rng.random((n_samples, n_dim)))
    u = cut / n_samples
    for d in range(n_dim):
        rng.shuffle(u[:, d])
    return u


def _loguniform(u, lo, hi):
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def perturbation_sweep(n_columns, nlev, rng, n_bnd_sw, n_bnd_lw,
                       sw_centers_nm, lw_centers_nm, **_ignored):
    """Designed synthetic sweep over the radiatively active parameters.

    Base state is the standard-lapse-rate reference atmosphere the
    radiation tests use (100 Pa -> 1013 hPa over ~20 km), perturbed along
    12 LHS axes chosen to span the physically realisable envelope:
    surface albedo (ocean to fresh snow), solar elevation, condensate
    amount and vertical placement, cloud fraction, aerosol loading and
    type, surface temperature, humidity and CO2.
    """
    u = _latin_hypercube(rng, n_columns, 12)

    # Fixed pressure/height grid: perturbations act on the state, not the
    # discretisation, so all columns share one vertical grid.
    pressure = np.logspace(np.log10(100.0), np.log10(101325.0), nlev)
    height = np.linspace(20000.0, 0.0, nlev)
    pressure_levels = np.broadcast_to(pressure, (n_columns, nlev)).copy()
    height_levels = np.broadcast_to(height, (n_columns, nlev)).copy()

    surface_temperature = 220.0 + 100.0 * u[:, 0]
    lapse_rate = (4.0 + 5.0 * u[:, 1]) * 1e-3            # 4-9 K/km
    temperature = np.maximum(
        surface_temperature[:, None] - lapse_rate[:, None] * height_levels,
        190.0,
    )

    # Humidity: exponential decay from a surface value spanning polar
    # desert to deep tropics, capped below saturation so the sweep never
    # asks the gas optics for a supersaturated column.
    q_surface = _loguniform(u[:, 2], 2.0e-5, 3.0e-2)
    specific_humidity = np.maximum(
        q_surface[:, None] * np.exp(-height_levels / 8000.0), 1.0e-7,
    )
    e_sat = 610.78 * np.exp(
        17.269 * (temperature - 273.16) / (temperature - 35.86),
    )
    q_sat = 0.622 * e_sat / np.maximum(pressure_levels - 0.378 * e_sat, 1.0)
    specific_humidity = np.minimum(specific_humidity, 0.98 * q_sat)

    air_density = _air_density(pressure_levels, temperature)
    layer_thickness = _layer_thickness(pressure_levels, temperature)
    pressure_interfaces = _interfaces_from_levels(pressure_levels)

    # Cloud slab: a contiguous band of levels with a sampled top, depth,
    # fraction and in-cloud condensate. The liquid/ice split follows the
    # slab's mid temperature, and cloud_* are GRID-MEAN (fraction times
    # in-cloud), which is what the scheme expects.
    levels = np.arange(nlev)[None, :]
    # Cloud tops are confined to p > 100 hPa. Above that the sweep would
    # be sampling stratospheric "clouds" that do not occur, and RRTMGP
    # answers them with a small negative downward LW at the top interface
    # — nonsense labels the emulator would have to fit.
    k_top = int(np.argmax(pressure >= _MIN_CLOUD_PRESSURE))
    top_idx = k_top + np.floor(u[:, 3] * (nlev - k_top)).astype(int)
    depth = 1 + np.floor(u[:, 4] * max(nlev // 3, 1)).astype(int)
    in_slab = (levels >= top_idx[:, None]) & (levels < (top_idx + depth)[:, None])
    cloud_fraction = np.where(in_slab, u[:, 5][:, None], 0.0)
    # Snap the near-clear tail to exactly zero: the scheme's in-cloud
    # conversion zeroes cf <= 2e-3 anyway, so labels there are clear-sky.
    cloud_fraction = np.where(cloud_fraction < 2.0e-3, 0.0, cloud_fraction)
    condensate = _loguniform(u[:, 6], 1.0e-6, 3.0e-3)[:, None] * in_slab
    ice_fraction = np.clip((273.15 - temperature) / 40.0, 0.0, 1.0)
    cloud_water = cloud_fraction * condensate * (1.0 - ice_fraction)
    cloud_ice = cloud_fraction * condensate * ice_fraction

    # Ozone: analytic stratospheric profile (mole fraction), scaled to
    # span the observed column range.
    p_mb = np.maximum(pressure_levels / 100.0, 1e-3)
    ozone_vmr = np.where(
        p_mb < 100.0,
        8.0e-6 * np.exp(-((np.log(p_mb) - np.log(30.0)) ** 2) / 2.0),
        5.0e-8,
    ) * (0.6 + 0.8 * u[:, 7])[:, None]

    # Aerosol: exponentially distributed 550 nm extinction normalised to a
    # sampled column AOD spanning clean background to a heavy plume, then
    # scaled to every band with the MACv2-SP wavelength relations. LW
    # bands get the same Angstrom extrapolation, which is crude but keeps
    # the LW aerosol channel non-zero and physically ordered.
    aod_total = _loguniform(u[:, 8], 5.0e-3, 3.0)
    scale_height = 1000.0 + 3000.0 * u[:, 9]
    shape = np.exp(-height_levels / scale_height[:, None]) * layer_thickness
    aod_550 = aod_total[:, None] * shape / shape.sum(axis=1, keepdims=True)
    ssa550 = 0.80 + 0.20 * u[:, 10]
    asy550 = 0.55 + 0.20 * rng.random(n_columns)
    angstrom = 0.3 + 2.2 * rng.random(n_columns)
    aod_sw, ssa_sw, asy_sw = _per_band_optics(
        aod_550, ssa550, asy550, angstrom, sw_centers_nm,
    )
    aod_lw, ssa_lw, asy_lw = _per_band_optics(
        aod_550, ssa550, asy550, angstrom, lw_centers_nm,
    )

    # Surface: broadband albedo from ocean to fresh snow, with a sampled
    # vis/nir contrast (snow and vegetation sit at opposite signs of it).
    albedo = 0.05 + 0.80 * u[:, 11]
    contrast = -0.3 + 0.6 * rng.random(n_columns)
    surface_albedo_vis = np.clip(albedo * (1.0 + contrast), 0.02, 0.95)
    surface_albedo_nir = np.clip(albedo * (1.0 - contrast), 0.02, 0.95)

    # Solar elevation is an LHS axis in its own right (u[:, 0] is already
    # spent on surface temperature), so mu0 spans overhead sun to grazing
    # twilight uniformly rather than following a latitude distribution.
    target_mu0 = 0.05 + 0.95 * rng.random(n_columns)
    (latitude, longitude, orbital_phase,
     synodic_phase, _) = _solar_geometry_for_cos_zenith(rng, target_mu0)

    return dict(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        ozone_vmr=ozone_vmr,
        co2_vmr=(280.0 + 920.0 * rng.random(n_columns)) * 1e-6,
        surface_temperature=surface_temperature,
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=0.94 + 0.06 * rng.random(n_columns),
        latitude=latitude,
        longitude=longitude,
        orbital_phase=orbital_phase,
        synodic_phase=synodic_phase,
        aod_sw_per_band=aod_sw,
        ssa_sw_per_band=ssa_sw,
        asy_sw_per_band=asy_sw,
        aod_lw_per_band=aod_lw,
        ssa_lw_per_band=ssa_lw,
        asy_lw_per_band=asy_lw,
    )


# ---------------------------------------------------------------------------
# Column source: existing JCM trajectory
# ---------------------------------------------------------------------------


def _first_var(ds, names, default=None, shape=None):
    """Return the first present variable, else a constant-filled fallback.

    JCM output content depends on the run's diagnostic configuration, so
    the trajectory source treats everything except the core state as
    optional and says (in the returned provenance) what it substituted.
    """
    for name in names:
        if name in ds:
            return np.asarray(ds[name].values), name
    if default is None:
        raise KeyError(f"state file has none of {names!r}")
    return np.full(shape, default, dtype=np.float64), f"<default {default}>"


def trajectory_columns(n_columns, nlev, rng, n_bnd_sw, n_bnd_lw,
                       sw_centers_nm, lw_centers_nm, state_file=None,
                       **_ignored):
    """Sample columns from a JCM output netCDF.

    Randomly subsamples over (time, lon, lat). Fields the run did not
    write fall back to documented constants; the per-band aerosol optics
    are rebuilt from the broadband ``aerosol.*_profile`` diagnostics with
    the MACv2-SP wavelength scaling, because JCM output never carries the
    (n_bnd, nlev, ncols) per-band arrays.

    Point the source at SNAPSHOT output. Solar geometry is reconstructed
    from the ``time`` coordinate, so on time-AVERAGED output the columns
    are still internally consistent but get the geometry of the window's
    timestamp rather than of the averaged state.
    """
    import xarray as xr

    from jcm.utils import load_states_from_xarray

    if state_file is None:
        raise ValueError("--state-file is required for source='trajectory'")
    ds = xr.open_dataset(state_file)
    # The core state goes through the shared loader so this source stays
    # consistent with the SCM/prescribed runners in jcm.runners.
    states = load_states_from_xarray(ds)
    temperature_all = np.asarray(states.temperature)
    nt, file_nlev, nlon, nlat = temperature_all.shape
    if nlev not in (file_nlev, None):
        raise ValueError(
            f"state file has {file_nlev} levels, --nlev says {nlev}; "
            "the trajectory source cannot regrid."
        )

    idx_t = rng.integers(0, nt, n_columns)
    idx_x = rng.integers(0, nlon, n_columns)
    idx_y = rng.integers(0, nlat, n_columns)
    take = lambda a: a[idx_t, :, idx_x, idx_y]         # noqa: E731
    take_2d = lambda a: a[idx_t, idx_x, idx_y]         # noqa: E731

    shape_3d = (nt, file_nlev, nlon, nlat)
    shape_2d = (nt, nlon, nlat)
    temperature = take(temperature_all)
    specific_humidity = take(np.asarray(states.specific_humidity))
    pressure_levels, _ = _first_var(ds, ["pressure_full"])
    pressure_levels = take(pressure_levels)
    pressure_interfaces, _ = _first_var(ds, ["pressure_half"])
    pressure_interfaces = pressure_interfaces[idx_t, :, idx_x, idx_y]

    cloud_water = take(_first_var(
        ds, ["clouds.qc", "qc"], 0.0, shape_3d)[0])
    cloud_ice = take(_first_var(
        ds, ["clouds.qi", "qi"], 0.0, shape_3d)[0])
    cloud_fraction = take(_first_var(
        ds, ["clouds.cloud_fraction"], 0.0, shape_3d)[0])
    # chemistry.ozone_vmr is ppmv on output (see rrtmgp._compute_full).
    ozone_vmr = take(_first_var(
        ds, ["chemistry.ozone_vmr"], 1.0, shape_3d)[0]) * 1e-6

    surface_temperature = take_2d(_first_var(
        ds, ["surface.surface_temperature", "surface.skin_temperature"],
        288.0, shape_2d)[0])
    surface_albedo_vis = take_2d(_first_var(
        ds, ["radiation.surface_albedo_vis"], 0.07, shape_2d)[0])
    surface_albedo_nir = take_2d(_first_var(
        ds, ["radiation.surface_albedo_nir"], 0.07, shape_2d)[0])
    surface_emissivity = take_2d(_first_var(
        ds, ["radiation.surface_emissivity"], 0.98, shape_2d)[0])

    aod_profile = take(_first_var(
        ds, ["aerosol.aod_profile"], 0.0, shape_3d)[0])
    ssa_profile = take(_first_var(
        ds, ["aerosol.ssa_profile"], 0.9, shape_3d)[0])
    asy_profile = take(_first_var(
        ds, ["aerosol.asy_profile"], 0.7, shape_3d)[0])
    angstrom = take_2d(_first_var(
        ds, ["aerosol.angstrom"], 1.5, shape_2d)[0])

    lat = np.asarray(ds["lat"].values)[idx_y]
    lon = np.asarray(ds["lon"].values)[idx_x]
    times = np.asarray(ds["time"].values)[idx_t]
    orbital_phase, synodic_phase = _solar_phases_from_datetime64(times)

    batch = dict(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        ozone_vmr=ozone_vmr,
        surface_temperature=surface_temperature,
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=surface_emissivity,
        latitude=lat,
        longitude=lon,
        orbital_phase=orbital_phase,
        synodic_phase=synodic_phase,
        co2_vmr=np.full(n_columns, 400e-6),
    )
    batch = _orient_toa_first(batch, aux=dict(
        aod_profile=aod_profile, ssa_profile=ssa_profile,
        asy_profile=asy_profile,
    ))
    aod_profile = batch.pop("aod_profile")
    ssa_profile = batch.pop("ssa_profile")
    asy_profile = batch.pop("asy_profile")

    # layer_thickness / air_density are recomputed rather than read from
    # the file so they are guaranteed consistent with the (re-oriented)
    # pressure and temperature actually handed to RRTMGP.
    batch["air_density"] = _air_density(
        batch["pressure_levels"], batch["temperature"])
    batch["layer_thickness"] = _layer_thickness(
        batch["pressure_levels"], batch["temperature"])

    aod_sw, ssa_sw, asy_sw = _per_band_optics(
        aod_profile, ssa_profile, asy_profile, angstrom, sw_centers_nm,
    )
    aod_lw, ssa_lw, asy_lw = _per_band_optics(
        aod_profile, ssa_profile, asy_profile, angstrom, lw_centers_nm,
    )
    batch.update(
        aod_sw_per_band=aod_sw, ssa_sw_per_band=ssa_sw,
        asy_sw_per_band=asy_sw, aod_lw_per_band=aod_lw,
        ssa_lw_per_band=ssa_lw, asy_lw_per_band=asy_lw,
    )
    return batch


def _solar_phases_from_datetime64(times):
    """Orbital / synodic phase (radians) from numpy datetimes.

    Mirrors ``jcm.forcing._solar_from_date``: orbital phase is 2*pi times
    the fraction of the year elapsed, synodic phase 2*pi times the
    fraction of the day.
    """
    t = np.asarray(times, dtype="datetime64[s]")
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    next_year = (t.astype("datetime64[Y]") + 1).astype("datetime64[s]")
    year_len = (next_year - year_start).astype(np.float64)
    frac_year = (t - year_start).astype(np.float64) / year_len
    day_start = t.astype("datetime64[D]").astype("datetime64[s]")
    frac_day = (t - day_start).astype(np.float64) / 86400.0
    return 2.0 * np.pi * frac_year, 2.0 * np.pi * frac_day


def _orient_toa_first(batch, aux):
    """Flip any surface-first columns so level index 0 is the model top.

    JCM output files are written in whatever order the run's vertical
    coordinate used, and both orders occur in practice — including within
    one file, where ``pressure_full`` and ``pressure_half`` need not
    agree. So the full-level and interface groups are each oriented
    against their OWN pressure array rather than a single file-wide flag.
    RRTMGP is orientation-agnostic internally, but it requires its inputs
    to be mutually consistent, and the stored training set needs one
    fixed convention for the labels to be comparable across files.
    """
    merged = dict(batch)
    merged.update(aux)
    p_full = merged["pressure_levels"]
    p_half = merged["pressure_interfaces"]
    flip_full = p_full[:, 0] > p_full[:, -1]
    flip_half = p_half[:, 0] > p_half[:, -1]

    def _oriented(arr, flip):
        arr = np.asarray(arr)
        return np.where(flip[:, None], arr[:, ::-1], arr)

    for name, value in merged.items():
        arr = np.asarray(value)
        if arr.ndim != 2:
            continue
        merged[name] = _oriented(
            arr, flip_half if name == "pressure_interfaces" else flip_full,
        )
    return merged


COLUMN_SOURCES = {
    "perturbation": perturbation_sweep,
    "trajectory": trajectory_columns,
    # Extension point: an ERA5 source slots in here with the same
    # signature. Not implemented yet — the WeatherBench2 store this repo
    # already caches carries no cloud liquid/ice, cloud fraction or
    # ozone, four of the nine required profiles. They ARE available (and
    # readable anonymously from this machine) in ARCO-ERA5
    # ``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``
    # as fraction_of_cloud_cover / specific_cloud_{liquid,ice}_water_content
    # / ozone_mass_mixing_ratio on 37 pressure levels, alongside
    # surface_pressure, skin_temperature and forecast_albedo. A source
    # built on it must convert ozone mass to mole fraction, interpolate
    # the pressure levels onto a model grid (ERA5's top level is 1 hPa,
    # so the stratosphere above needs extending), and supply aerosol
    # optics from elsewhere — ERA5 has none.
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build_dataset(batch, labels, attrs):
    """Assemble the inputs + labels into an xarray Dataset.

    netCDF (not npz) because it is the repo standard and because it keeps
    the per-band arrays self-describing: ``(column, band_sw, level)`` with
    named dimensions survives a round trip, whereas npz would leave the
    axis order to a README.
    """
    import xarray as xr

    ncol, nlev = batch["temperature"].shape
    n_bnd_sw = batch["aod_sw_per_band"].shape[1]
    n_bnd_lw = batch["aod_lw_per_band"].shape[1]

    data = {}
    for name in PROFILE_FIELDS:
        data[name] = (("column", "level"), batch[name].astype(np.float32))
    for name in INTERFACE_FIELDS:
        data[name] = (("column", "interface"), batch[name].astype(np.float32))
    for name in SCALAR_FIELDS:
        data[name] = (("column",), batch[name].astype(np.float32))
    for name in SW_BAND_FIELDS:
        data[name] = (
            ("column", "band_sw", "level"), batch[name].astype(np.float32),
        )
    for name in LW_BAND_FIELDS:
        data[name] = (
            ("column", "band_lw", "level"), batch[name].astype(np.float32),
        )
    for name in LABEL_FIELDS:
        data[name] = (
            ("column", "interface"), np.asarray(labels[name], np.float32),
        )
    data["cos_zenith"] = (
        ("column",), np.asarray(labels["cos_zenith"], np.float32),
    )
    data["total_cloud_cover"] = (
        ("column",), np.asarray(labels["total_cloud_cover"], np.float32),
    )

    return xr.Dataset(
        data,
        coords={
            "column": np.arange(ncol),
            "level": np.arange(nlev),
            "interface": np.arange(nlev + 1),
            "band_sw": np.arange(n_bnd_sw),
            "band_lw": np.arange(n_bnd_lw),
        },
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate(source, n_columns, nlev, n_seeds, base_seed, batch_size,
             rng_seed, state_file=None):
    """Generate ``n_columns`` labelled columns and return ``(batch, labels)``."""
    rng = np.random.default_rng(rng_seed)
    n_bnd_sw, n_bnd_lw, sw_centers, lw_centers = band_counts()
    labeller = make_labeller(base_seed)

    batches, label_sets = [], []
    remaining = n_columns
    while remaining > 0:
        n_here = min(batch_size, remaining)
        raw = COLUMN_SOURCES[source](
            n_here, nlev, rng, n_bnd_sw, n_bnd_lw, sw_centers, lw_centers,
            state_file=state_file,
        )
        chunk = _finalize_batch(raw, n_bnd_sw, n_bnd_lw)
        labels, _ = label_batch(chunk, n_seeds, base_seed, labeller)
        batches.append(chunk)
        label_sets.append(labels)
        remaining -= n_here

    batch = {
        k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]
    }
    labels = {
        k: np.concatenate([s[k] for s in label_sets], axis=0)
        for k in label_sets[0]
    }
    return batch, labels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", default="perturbation",
                   choices=sorted(COLUMN_SOURCES))
    p.add_argument("--n-columns", type=int, default=1024)
    p.add_argument("--nlev", type=int, default=47,
                   help="levels per column (must match --state-file for "
                        "source=trajectory)")
    p.add_argument("--n-seeds", type=int, default=8,
                   help="independent McICA draws averaged per column")
    p.add_argument("--base-seed", type=int, default=0,
                   help="McICA base seed (Python static; changing it "
                        "re-traces, so use --n-seeds for averaging)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="columns per RRTMGP call, to bound peak memory")
    p.add_argument("--rng-seed", type=int, default=0,
                   help="seed for the column sampler")
    p.add_argument("--state-file", default=None,
                   help="JCM output netCDF for source=trajectory")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    started = time.time()
    batch, labels = generate(
        args.source, args.n_columns, args.nlev, args.n_seeds,
        args.base_seed, args.batch_size, args.rng_seed, args.state_file,
    )
    elapsed = time.time() - started

    n_bnd_sw = batch["aod_sw_per_band"].shape[1]
    n_bnd_lw = batch["aod_lw_per_band"].shape[1]
    attrs = dict(
        title="RRTMGP training data for the JCM radiation NN emulator",
        git_commit=_git_commit(),
        source=args.source,
        state_file=str(args.state_file or ""),
        n_columns=int(args.n_columns),
        n_levels=int(batch["temperature"].shape[1]),
        n_mcica_seeds=int(args.n_seeds),
        mcica_seed_mechanism="model_step (traced) folded into column_key",
        base_seed=int(args.base_seed),
        rng_seed=int(args.rng_seed),
        n_bands_sw=int(n_bnd_sw),
        n_bands_lw=int(n_bnd_lw),
        rrtmgp_config="rrtmgp-gas-lw-g128 / rrtmgp-gas-sw-g112, "
                      "McICA exponential overlap, compute_cre=True",
        vertical_convention="level 0 = model top (TOA-first)",
        generation_seconds=round(elapsed, 2),
        jax_platform=os.environ.get("JAX_PLATFORMS", "default"),
    )
    ds = build_dataset(batch, labels, attrs)
    ds.to_netcdf(args.out)

    per_col = elapsed / max(args.n_columns, 1)
    print(f"wrote {args.out}: {args.n_columns} columns x "
          f"{args.n_seeds} seeds in {elapsed:.1f} s "
          f"({per_col * 1e3:.1f} ms/column, "
          f"{args.n_columns / elapsed:.1f} columns/s)")
    toa_sw_up = labels["sw_flux_up"][:, 0]
    sfc_sw_down = labels["sw_flux_down"][:, -1]
    olr = labels["lw_flux_up"][:, 0]
    for label, arr in (("TOA SW up", toa_sw_up),
                       ("surface SW down", sfc_sw_down),
                       ("OLR", olr)):
        print(f"  {label:<16} min {arr.min():8.2f}  mean {arr.mean():8.2f}  "
              f"max {arr.max():8.2f} W/m2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
