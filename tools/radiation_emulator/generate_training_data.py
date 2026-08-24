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
        --source trajectory --state-file 'run/out_day*.nc' \
        --n-columns 20000 --out traj_labels.nc
"""

from __future__ import annotations

import argparse
import functools
import glob
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
    # RESOLVED effective radii (um): the microphysical value where the source
    # provides one, the diagnostic fallback elsewhere. Stored resolved, and
    # RRTMGP is driven with exactly these, so a feature and its label always
    # describe the same cloud (see _resolved_effective_radii).
    "r_eff_liq", "r_eff_ice",
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

# Top of the synthetic pressure grid, matching the model's own model top
# (1 Pa for the L47 hybrid levels).
_TOP_PRESSURE = 1.0

# Stratopause pressure (Pa): the temperature maximum between the tropopause
# and the mesosphere.
_STRATOPAUSE_PRESSURE = 1.0e2


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
        None, None,          # ch4_vmr, n2o_vmr (unset: RRTMGP's own defaults)
        0, 0,                # r_eff_liq_um, r_eff_ice_um
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
            # ch4/n2o stay unprescribed, as before; the radii are positional
            # after them, and they are the batch's RESOLVED values so the
            # labels describe exactly the cloud the stored features do.
            None, None,
            _f32(batch["r_eff_liq"]), _f32(batch["r_eff_ice"]),
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
    # silently becomes a no-op that still returns plausible labels. Catch it
    # on the data rather than trusting the parameter default -- but only where
    # the draws are *required* to differ. A cloud-free batch legitimately
    # yields identical draws, and it does occur for small batches, so
    # conditioning on cloud is what keeps this a correctness check rather than
    # a random abort partway through a long generation. The comparison is on
    # the longwave over the meaningfully-cloudy columns only: the shortwave is
    # identically zero at night whatever the draw, so comparing it would trip
    # on any batch whose cloudy columns all happen to be in darkness.
    cloudy = np.max(batch["cloud_fraction"], axis=1) > 0.01
    if n_seeds > 1 and cloudy.any() and np.array_equal(
            draws[0]["lw_flux_up"][cloudy], draws[-1]["lw_flux_up"][cloudy]):
        raise ValueError(
            "McICA draws are identical across seeds for a batch containing "
            "cloud, so seed averaging is doing nothing. Check that "
            "RadiationParameters.mcica_freeze_step is 0."
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


def label_quality_mask(batch, labels):
    """Return ``(keep, reasons)`` rejecting columns with unphysical labels.

    Input clipping removes the causes of solver blow-up that are known, but a
    training set must not depend on knowing them all: one column emitting
    1000 W/m2 of OLR would dominate a mean-squared-error loss over thousands of
    good ones. So the labels are checked against bounds that any correct
    radiative transfer must satisfy, and violating columns are dropped rather
    than silently trained on.
    """
    import jcm.constants as c

    keep = np.ones(batch["temperature"].shape[0], dtype=bool)
    reasons = {}

    def reject(name, bad):
        nonlocal keep
        n = int((bad & keep).sum())
        if n:
            reasons[name] = reasons.get(name, 0) + n
        keep = keep & ~bad

    finite = np.ones_like(keep)
    for name in LABEL_FIELDS:
        finite &= np.isfinite(labels[name]).all(axis=1)
    reject("non-finite flux", ~finite)

    for name in LABEL_FIELDS:
        reject("negative flux", (labels[name] < -1.0e-3).any(axis=1))

    # Shortwave cannot reflect more than it receives at any interface.
    for up, down in (("sw_flux_up", "sw_flux_down"),
                     ("sw_flux_up_clear", "sw_flux_down_clear")):
        reject("SW up exceeds SW down",
               (labels[up] > labels[down] + 1.0e-3).any(axis=1))

    # No interface can carry more longwave than a black body at the warmest
    # temperature anywhere in the column (10 K of headroom for the surface
    # skin sitting above the lowest model level).
    t_max = np.maximum(batch["temperature"].max(axis=1),
                       batch["surface_temperature"]) + 10.0
    lw_ceiling = (float(c.sbc) * t_max ** 4)[:, None]
    for name in ("lw_flux_up", "lw_flux_down",
                 "lw_flux_up_clear", "lw_flux_down_clear"):
        reject("longwave above black body",
               (labels[name] > lw_ceiling).any(axis=1))

    return keep, reasons


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------

# Physically admissible ranges for the fields a column source supplies.
# Model-output diagnostics do leave these bounds: surface emissivity reaches
# 1.9 over polar land (jax-gcm#703, an unclipped sea-ice fraction in the
# radiative surface-optics blend) and the time-averaged 550 nm SSA exceeds 1 by
# a few ppm through the float32 diagnostic accumulator. Neither is benign here,
# because MACv2-SP's per-band scaling divides by
# ``ssa550*l^4 + (1-ssa550)*l``, whose sign flips just above ssa550 = 1, so a
# 550 nm SSA of 1.0003 becomes a per-band SSA of ~1e21 in the far infrared.
# RRTMGP then returns >1000 W/m2 OLR for an otherwise unremarkable clear-sky
# column. Bounds are therefore enforced on input rather than trusted, and every
# clip is counted and reported so a bad source cannot pass silently.

# Keep the MACv2-SP SSA denominator positive (see above). SSA = 1 exactly is
# fine and common — MACv2-SP sets it there wherever the plume AOD vanishes —
# so only genuine excursions above 1 are clipped.
_SSA550_MAX = 1.0

INPUT_BOUNDS = {
    # 550 nm references, clipped BEFORE the per-band scaling so the bands
    # get the right small value rather than a clipped-to-1.0 wrong one.
    "ssa550": (0.0, _SSA550_MAX),
    "asy550": (-1.0, 1.0),
    "temperature": (150.0, 350.0),
    "surface_temperature": (150.0, 350.0),
    "specific_humidity": (0.0, 0.1),
    "cloud_water": (0.0, 0.05),
    "cloud_ice": (0.0, 0.05),
    "cloud_fraction": (0.0, 1.0),
    # Effective radii, generous on both ends: the bound exists to catch a
    # corrupt source, not to shape the distribution. The jax-rrtmgp optics
    # clip to their own LUT limits internally.
    "r_eff_liq": (0.0, 100.0),
    "r_eff_ice": (0.0, 500.0),
    "ozone_vmr": (0.0, 5.0e-5),
    "surface_albedo_vis": (0.0, 1.0),
    "surface_albedo_nir": (0.0, 1.0),
    "surface_emissivity": (0.0, 1.0),
    "ssa_sw_per_band": (0.0, 1.0),
    "ssa_lw_per_band": (0.0, 1.0),
    "asy_sw_per_band": (-1.0, 1.0),
    "asy_lw_per_band": (-1.0, 1.0),
    "aod_sw_per_band": (0.0, 10.0),
    "aod_lw_per_band": (0.0, 10.0),
}


def _clip_to_bounds(arrays, stats=None):
    """Clip ``arrays`` into ``INPUT_BOUNDS``, tallying violations in ``stats``.

    ``stats`` maps a field name to ``[n_violating, worst_low, worst_high]`` and
    accumulates across batches so the driver can report once at the end.
    """
    out = dict(arrays)
    for name, (lo, hi) in INPUT_BOUNDS.items():
        if name not in out:
            continue
        arr = np.asarray(out[name], dtype=np.float64)
        n_bad = int(((arr < lo) | (arr > hi)).sum())
        if not n_bad:
            continue
        out[name] = np.clip(arr, lo, hi)
        if stats is not None:
            entry = stats.setdefault(name, [0, np.inf, -np.inf])
            entry[0] += n_bad
            entry[1] = min(entry[1], float(arr.min()))
            entry[2] = max(entry[2], float(arr.max()))
    return out


def report_clips(stats):
    """Print the accumulated input-clipping tally (empty tally prints nothing)."""
    for name, (n_bad, lo, hi) in sorted(stats.items()):
        bound = INPUT_BOUNDS[name]
        print(f"  clipped {name}: {n_bad} values outside "
              f"[{bound[0]:g}, {bound[1]:g}] (input range {lo:g}..{hi:g})")


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


def _resolved_effective_radii(batch):
    """Return ``batch`` with ``r_eff_liq`` / ``r_eff_ice`` resolved.

    A source supplies RAW radii (um), zero meaning "not provided". This
    replaces them with the resolved values ``resolve_effective_radii``
    produces — microphysical where given, the diagnostic fallback elsewhere —
    and those are what get stored AND what the labeller hands RRTMGP. Storing
    the raw values instead would let a stored feature (0 outside cloud)
    describe a different cloud from the label RRTMGP computed from the
    fallback. The resolved radii are strictly positive, so RRTMGP's own
    fallback never re-triggers on them.

    The ice fallback is a power law in the IN-CLOUD ice water content, so the
    grid-mean condensate goes through the same ``in_cloud_path`` division and
    ``_MAX_IN_CLOUD_CONDENSATE`` cap that ``radiation_scheme_rrtmgp`` applies
    before building its radiation state.

    Requires ``air_density`` and ``layer_thickness`` to be present already.
    """
    import jax
    import jax.numpy as jnp

    from jcm.physics.radiation.cloud_optics import resolve_effective_radii
    from jcm.physics.radiation.mcica import in_cloud_path
    from jcm.physics.radiation.rrtmgp import _MAX_IN_CLOUD_CONDENSATE

    ice_in_cloud = jnp.minimum(
        in_cloud_path(jnp.asarray(batch["cloud_ice"]),
                      jnp.asarray(batch["cloud_fraction"])),
        _MAX_IN_CLOUD_CONDENSATE,
    )
    in_cloud_ice_path = (
        jnp.maximum(ice_in_cloud, 0.0)
        * jnp.asarray(batch["air_density"])
        * jnp.asarray(batch["layer_thickness"])
    )
    # cdnc_factor = 1 and land_fraction = 0.5 are what the labeller's
    # AerosolData and prepare_rrtmgp_data's default hand the liquid fallback.
    r_liq, r_ice = jax.vmap(
        resolve_effective_radii, in_axes=(0, 0, None, None, 0, 0),
    )(
        jnp.asarray(batch["r_eff_liq"]), jnp.asarray(batch["r_eff_ice"]),
        jnp.asarray(1.0), 0.5,
        in_cloud_ice_path, jnp.asarray(batch["layer_thickness"]),
    )
    out = dict(batch)
    out["r_eff_liq"] = np.asarray(r_liq, dtype=np.float64)
    out["r_eff_ice"] = np.asarray(r_ice, dtype=np.float64)
    return out


def _interfaces_from_levels(pressure):
    """Half-level pressures (nlev+1) bracketing TOA-first full levels."""
    mid = 0.5 * (pressure[..., :-1] + pressure[..., 1:])
    top = np.maximum(2.0 * pressure[..., :1] - mid[..., :1], 1.0)
    sfc = 2.0 * pressure[..., -1:] - mid[..., -1:]
    return np.concatenate([top, mid, sfc], axis=-1)


def _per_band_optics(aod_550, ssa550, asy550, angstrom, band_centers_nm,
                     clip_stats=None):
    """Per-band ``(ncol, n_bnd, nlev)`` optics from 550 nm references.

    Wraps the MACv2-SP closed-form wavelength scaling
    (``per_band_optical_properties``) so both column sources build their
    per-band aerosol the same way the online scheme does. The 550 nm SSA is
    clipped below 1 first: the scaling's denominator changes sign there and
    an SSA a whisker above 1 produces a ~1e21 far-infrared SSA.
    """
    import jax.numpy as jnp

    from jcm.physics.aerosol.macv2_sp import per_band_optical_properties

    clipped = _clip_to_bounds(
        {"ssa550": ssa550, "asy550": asy550}, clip_stats)
    ssa550, asy550 = clipped["ssa550"], clipped["asy550"]

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


def _absent_lw_optics(n_bnd_lw, aod_550):
    """Zero longwave per-band optics, matching what MACv2-SP actually emits.

    ``Macv2SpAerosol`` writes only the SHORTWAVE per-band trio
    (``macv2_sp.py``, the ``aerosol_data.copy(...)`` return); the longwave
    slots keep their ``AerosolData.zeros()`` default. Only the JAM optics term
    ever fills them.

    Generating labels with a wavelength-scaled longwave aerosol therefore fed
    the emulator 32 features that are identically zero at run time but
    averaged ssa 0.49 / asy 0.09 in training -- a ~1 unit offset on 32 of the
    56 longwave inputs, 100% outside the training distribution. The network
    responded with +41 W/m2 of surface downward longwave, which warmed the
    surface, dried the boundary layer and destroyed over half the cloud; the
    -22 W/m2 shortwave bias that showed up at TOA was the consequence, not
    the cause.

    The scaled values were meaningless anyway: MACv2-SP's longwave AOD is
    ~5e-6, so the ssa/asy attached to it carry no radiative information.

    Revisit when generating labels from a JAM run -- there the longwave
    optics are real and must be sampled from the run, not zeroed.
    """
    ncol, nlev = aod_550.shape
    zeros = np.zeros((ncol, n_bnd_lw, nlev))
    return zeros, zeros.copy(), zeros.copy()


def _finalize_batch(batch, n_bnd_sw, n_bnd_lw, clip_stats=None):
    """Validate shapes, clip to physical bounds, and cast to float64 numpy."""
    batch = _clip_to_bounds(batch, clip_stats)
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


@functools.lru_cache(maxsize=4)
def _model_level_pressures(nlev, surface_pressure=101325.0):
    """Full-level pressures (TOA-first) on the model's own hybrid levels.

    Falls back to a log-spaced grid spanning the same range if the level
    definitions are unavailable, so the sweep still runs for level counts
    the ECHAM coefficients do not define.
    """
    try:
        from jcm.physics.echam.echam_levels import get_echam_levels

        coords = get_echam_levels(nlev)
        a = np.asarray(coords.a_boundaries, dtype=np.float64)
        b = np.asarray(coords.b_boundaries, dtype=np.float64)
        p_half = a + b * surface_pressure
        p_full = 0.5 * (p_half[:-1] + p_half[1:])
        if p_full[0] > p_full[-1]:          # keep the TOA-first convention
            p_full = p_full[::-1]
        return np.maximum(p_full, _TOP_PRESSURE)
    except Exception:
        return np.logspace(
            np.log10(_TOP_PRESSURE), np.log10(surface_pressure), nlev)


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
                       sw_centers_nm, lw_centers_nm, clip_stats=None,
                       **_ignored):
    """Designed synthetic sweep over the radiatively active parameters.

    Base state is the standard-lapse-rate reference atmosphere the
    radiation tests use (100 Pa -> 1013 hPa over ~20 km), perturbed along
    12 LHS axes chosen to span the physically realisable envelope:
    surface albedo (ocean to fresh snow), solar elevation, condensate
    amount and vertical placement, cloud fraction, aerosol loading and
    type, surface temperature, humidity and CO2.
    """
    u = _latin_hypercube(rng, n_columns, 15)

    # The MODEL's own hybrid levels, not an invented grid. Two earlier
    # versions got this wrong in opposite directions: a logspace grid from
    # 100 Pa left the top eight L47 levels with no coverage at all (and a
    # coupled run then failed exactly there), while a logspace grid from
    # 1 Pa spread 47 levels over five decades instead of three, thickening
    # every tropospheric layer until per-layer aerosol optical depth reached
    # 21 and ~9% of columns came back with unphysical fluxes. Using the real
    # coefficients keeps layer thicknesses right at every altitude.
    pressure = _model_level_pressures(nlev)
    pressure_levels = np.broadcast_to(pressure, (n_columns, nlev)).copy()
    # Pressure-derived height, so the two stay consistent over five decades
    # of pressure rather than only over a 20 km troposphere.
    height_levels = -8000.0 * np.log(pressure_levels / 101325.0)

    # Temperature is piecewise-linear in log(p) through three sampled control
    # points -- surface, tropopause, stratopause. A single tropospheric lapse
    # rate floored at 190 K (the earlier form) has no stratospheric inversion,
    # so every column piled up at exactly 190 K near the model top while real
    # ones sit at 204-252 K. That gap between the synthetic pile and the
    # trajectory data is a hole the network has to extrapolate across.
    surface_temperature = 220.0 + 100.0 * u[:, 0]
    lapse_rate = (4.0 + 5.0 * u[:, 1]) * 1e-3            # 4-9 K/km
    p_tropopause = _loguniform(u[:, 10], 8.0e3, 3.0e4)
    t_tropopause = np.maximum(
        surface_temperature - lapse_rate * (-8000.0 * np.log(
            p_tropopause / 101325.0)),
        185.0,
    )
    # Ozone heating warms the stratopause, near 100 Pa, back to ~240-285 K;
    # above it the mesosphere COOLS again, so the model top at 1 Pa is a
    # separate control point rather than the profile's maximum. Getting this
    # wrong in either direction leaves a gap the network must extrapolate
    # across: a single tropospheric lapse floored at 190 K put every column
    # at 190 there, and running the warm branch all the way to 1 Pa put them
    # all at 240-285. The coupled model actually sits at ~205-246.
    t_stratopause = 240.0 + 45.0 * u[:, 11]
    t_model_top = 195.0 + 65.0 * u[:, 12]

    log_p = np.log(pressure_levels)
    log_sfc, log_top = np.log(101325.0), np.log(_TOP_PRESSURE)
    log_trop = np.log(p_tropopause)[:, None]
    log_strat = np.log(_STRATOPAUSE_PRESSURE)

    def _blend(lo_log, hi_log, lo_t, hi_t):
        """Linear in log(p) between two control points."""
        f = (log_p - lo_log) / (hi_log - lo_log)
        return lo_t + np.clip(f, 0.0, 1.0) * (hi_t - lo_t)

    temperature = np.where(
        pressure_levels >= p_tropopause[:, None],
        _blend(log_trop, log_sfc, t_tropopause[:, None],
               surface_temperature[:, None]),
        np.where(
            pressure_levels >= _STRATOPAUSE_PRESSURE,
            _blend(log_trop, log_strat, t_tropopause[:, None],
                   t_stratopause[:, None]),
            _blend(log_strat, log_top, t_stratopause[:, None],
                   t_model_top[:, None]),
        ),
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
    # answers them with a negative downward LW at the top interface
    # (jax-gcm#711) — nonsense labels the emulator would have to fit.
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

    # Effective radii on the two spare LHS axes, spanning the observed ranges
    # (drizzling marine stratocumulus to continental haze; small cirrus
    # crystals to large aggregates). Zero outside cloud means "not provided",
    # so the clear part of every column exercises the diagnostic fallback the
    # coupled model uses wherever the microphysics writes nothing.
    cloudy = cloud_fraction > 0.0
    r_eff_liq = np.where(cloudy, (2.0 + 18.0 * u[:, 13])[:, None], 0.0)
    r_eff_ice = np.where(cloudy, (10.0 + 140.0 * u[:, 14])[:, None], 0.0)

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
    aod_lw, ssa_lw, asy_lw = _absent_lw_optics(len(lw_centers_nm), aod_550)

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

    return _resolved_effective_radii(dict(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        r_eff_liq=r_eff_liq,
        r_eff_ice=r_eff_ice,
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
    ))


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


@functools.lru_cache(maxsize=1)
def _load_trajectory_fields(state_file):
    """Read every field the trajectory source needs from one JCM output file.

    Cached one file deep because :func:`generate` drives the files in major
    order: the fields used here are ~700 MB for a 20-frame T63L47 file, and
    re-reading them for every 256-column batch would dominate the run.

    Fields the run did not write fall back to documented constants, since
    JCM output content depends on the run's diagnostic configuration.
    """
    import xarray as xr

    from jcm.utils import load_states_from_xarray

    ds = xr.open_dataset(state_file)
    # The core state goes through the shared loader so this source stays
    # consistent with the SCM/prescribed runners in jcm.runners.
    states = load_states_from_xarray(ds)
    out = {
        "temperature": np.asarray(states.temperature),
        "specific_humidity": np.asarray(states.specific_humidity),
    }
    shape_3d = out["temperature"].shape
    shape_2d = shape_3d[:1] + shape_3d[2:]
    out["pressure_levels"] = _first_var(ds, ["pressure_full"])[0]
    out["pressure_interfaces"] = _first_var(ds, ["pressure_half"])[0]
    out["cloud_water"] = _first_var(ds, ["clouds.qc", "qc"], 0.0, shape_3d)[0]
    out["cloud_ice"] = _first_var(ds, ["clouds.qi", "qi"], 0.0, shape_3d)[0]
    out["cloud_fraction"] = _first_var(
        ds, ["clouds.cloud_fraction"], 0.0, shape_3d)[0]
    # Microphysical radii (um), written by the 2M scheme. Zero -- both as the
    # fallback for a 1M run that wrote none, and level-by-level within a 2M
    # run -- means "not provided" and selects the diagnostic parameterisation.
    out["r_eff_liq"] = _first_var(
        ds, ["clouds.r_eff_liq"], 0.0, shape_3d)[0]
    out["r_eff_ice"] = _first_var(
        ds, ["clouds.r_eff_ice"], 0.0, shape_3d)[0]
    # chemistry.ozone_vmr is ppmv on output (see rrtmgp._compute_full).
    out["ozone_vmr"] = _first_var(
        ds, ["chemistry.ozone_vmr"], 1.0, shape_3d)[0] * 1e-6
    out["surface_temperature"] = _first_var(
        ds, ["surface.surface_temperature", "surface.skin_temperature"],
        288.0, shape_2d)[0]
    out["surface_albedo_vis"] = _first_var(
        ds, ["radiation.surface_albedo_vis"], 0.07, shape_2d)[0]
    out["surface_albedo_nir"] = _first_var(
        ds, ["radiation.surface_albedo_nir"], 0.07, shape_2d)[0]
    out["surface_emissivity"] = _first_var(
        ds, ["radiation.surface_emissivity"], 0.98, shape_2d)[0]
    out["aod_profile"] = _first_var(
        ds, ["aerosol.aod_profile"], 0.0, shape_3d)[0]
    out["ssa_profile"] = _first_var(
        ds, ["aerosol.ssa_profile"], 0.9, shape_3d)[0]
    out["asy_profile"] = _first_var(
        ds, ["aerosol.asy_profile"], 0.7, shape_3d)[0]
    out["angstrom"] = _first_var(
        ds, ["aerosol.angstrom"], 1.5, shape_2d)[0]
    out["lat"] = np.asarray(ds["lat"].values)
    out["lon"] = np.asarray(ds["lon"].values)
    out["time"] = np.asarray(ds["time"].values)
    ds.close()
    return out


def trajectory_columns(n_columns, nlev, rng, n_bnd_sw, n_bnd_lw,
                       sw_centers_nm, lw_centers_nm, state_file=None,
                       clip_stats=None, **_ignored):
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
    if state_file is None:
        raise ValueError("--state-file is required for source='trajectory'")
    f = _load_trajectory_fields(state_file)
    nt, file_nlev, nlon, nlat = f["temperature"].shape
    if nlev not in (file_nlev, None):
        raise ValueError(
            f"state file has {file_nlev} levels, --nlev says {nlev}; "
            "the trajectory source cannot regrid."
        )

    idx_t = rng.integers(0, nt, n_columns)
    idx_x = rng.integers(0, nlon, n_columns)
    idx_y = rng.integers(0, nlat, n_columns)
    take = lambda name: f[name][idx_t, :, idx_x, idx_y]   # noqa: E731
    take_2d = lambda name: f[name][idx_t, idx_x, idx_y]   # noqa: E731

    temperature = take("temperature")
    specific_humidity = take("specific_humidity")
    pressure_levels = take("pressure_levels")
    pressure_interfaces = take("pressure_interfaces")
    cloud_water = take("cloud_water")
    cloud_ice = take("cloud_ice")
    cloud_fraction = take("cloud_fraction")
    ozone_vmr = take("ozone_vmr")
    r_eff_liq = take("r_eff_liq")
    r_eff_ice = take("r_eff_ice")

    surface_temperature = take_2d("surface_temperature")
    surface_albedo_vis = take_2d("surface_albedo_vis")
    surface_albedo_nir = take_2d("surface_albedo_nir")
    surface_emissivity = take_2d("surface_emissivity")

    aod_profile = take("aod_profile")
    ssa_profile = take("ssa_profile")
    asy_profile = take("asy_profile")
    angstrom = take_2d("angstrom")

    lat = f["lat"][idx_y]
    lon = f["lon"][idx_x]
    orbital_phase, synodic_phase = _solar_phases_from_datetime64(
        f["time"][idx_t])

    batch = dict(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        # Raw here; resolved below, once dz and rho exist. Carried in the
        # batch (not in ``aux``) so _orient_toa_first flips them with the same
        # per-column mask as pressure -- a misaligned radius profile would be
        # silent and severe. ``aux`` is only for fields consumed and dropped
        # before the batch is returned, which these are not.
        r_eff_liq=r_eff_liq,
        r_eff_ice=r_eff_ice,
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
    batch = _resolved_effective_radii(batch)

    aod_sw, ssa_sw, asy_sw = _per_band_optics(
        aod_profile, ssa_profile, asy_profile, angstrom, sw_centers_nm,
        clip_stats,
    )
    aod_lw, ssa_lw, asy_lw = _absent_lw_optics(
        len(lw_centers_nm), aod_profile,
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


def expand_state_files(state_file):
    """Expand a comma-separated list and/or glob into a sorted list of paths.

    Returns ``[None]`` when no state file is given, so a source that does not
    need one (the synthetic sweep) still drives the loop exactly once.
    """
    if not state_file:
        return [None]
    paths = []
    for piece in str(state_file).split(","):
        piece = piece.strip()
        if not piece:
            continue
        matched = sorted(glob.glob(piece))
        if not matched:
            raise FileNotFoundError(f"no state file matches {piece!r}")
        paths.extend(matched)
    return paths


def generate(source, n_columns, nlev, n_seeds, base_seed, batch_size,
             rng_seed, state_file=None, progress_every=20):
    """Generate ``n_columns`` labelled columns and return ``(batch, labels)``.

    Columns whose labels fail :func:`label_quality_mask` are dropped and
    replaced, so the returned batch always holds exactly ``n_columns`` good
    columns; the rejection tally is printed at the end.
    """
    rng = np.random.default_rng(rng_seed)
    n_bnd_sw, n_bnd_lw, sw_centers, lw_centers = band_counts()
    labeller = make_labeller(base_seed)

    clip_stats, reject_stats = {}, {}
    batches, label_sets = [], []
    n_kept = n_attempted = n_batches = n_empty = 0
    started = time.time()
    # A model run writes one file per chunk, so the trajectory source is
    # driven file-major: each file supplies an equal share of the columns and
    # is opened once, rather than being reopened per batch.
    files = expand_state_files(state_file)
    quotas = [n_columns * (i + 1) // len(files) for i in range(len(files))]
    file_index = 0
    while n_kept < n_columns:
        while file_index + 1 < len(files) and n_kept >= quotas[file_index]:
            file_index += 1
        n_here = min(batch_size, n_columns - n_kept)
        raw = COLUMN_SOURCES[source](
            n_here, nlev, rng, n_bnd_sw, n_bnd_lw, sw_centers, lw_centers,
            state_file=files[file_index], clip_stats=clip_stats,
        )
        chunk = _finalize_batch(raw, n_bnd_sw, n_bnd_lw, clip_stats)
        labels, _ = label_batch(chunk, n_seeds, base_seed, labeller)
        keep, reasons = label_quality_mask(chunk, labels)
        for name, count in reasons.items():
            reject_stats[name] = reject_stats.get(name, 0) + count
        batches.append({k: v[keep] for k, v in chunk.items()})
        label_sets.append({k: v[keep] for k, v in labels.items()})
        n_kept += int(keep.sum())
        n_attempted += n_here
        n_batches += 1
        # Resampling only terminates if the source can produce good columns
        # at all; a systematically broken one would otherwise spin forever.
        n_empty = n_empty + 1 if not keep.any() else 0
        if n_empty >= 5:
            raise RuntimeError(
                f"5 consecutive batches rejected entirely ({reject_stats}); "
                "the column source or the radiation configuration is broken."
            )
        if progress_every and n_batches % progress_every == 0:
            rate = n_kept / max(time.time() - started, 1e-9)
            print(f"  {n_kept}/{n_columns} columns "
                  f"({rate:.1f}/s, {n_attempted - n_kept} rejected)",
                  flush=True)

    batch = {
        k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]
    }
    labels = {
        k: np.concatenate([s[k] for s in label_sets], axis=0)
        for k in label_sets[0]
    }
    report_clips(clip_stats)
    for name, count in sorted(reject_stats.items()):
        print(f"  rejected {count} columns: {name}")
    return batch, labels, dict(n_attempted=n_attempted,
                               n_rejected=n_attempted - n_kept)


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
                   help="JCM output netCDF(s) for source=trajectory: a path, "
                        "a glob, or a comma-separated list. Each file "
                        "supplies an equal share of the columns.")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    started = time.time()
    batch, labels, quality = generate(
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
        # A non-zero rejection count is a signal about the source, not just
        # bookkeeping: it belongs with the data it describes.
        n_columns_attempted=int(quality["n_attempted"]),
        n_columns_rejected=int(quality["n_rejected"]),
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
