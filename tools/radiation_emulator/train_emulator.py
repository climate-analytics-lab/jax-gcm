r"""Train the JCM radiation NN emulator against RRTMGP labels.

Consumes the files written by ``generate_training_data.py`` and produces a
weight file that ``NNEmulatorRadiation(weights_file=...)`` can load.

Two properties matter more than anything else here and drive the design:

*Feature parity.* The inputs are built by calling the very same
``preprocess_sw_inputs`` / ``preprocess_lw_inputs`` the online scheme calls,
from the same raw fields, with the same derived quantities (water-vapour
mixing ratio, in-cloud water paths). Re-deriving features here would let the
trained network drift away from what it is fed at run time, which is the
classic silent failure of an emulator.

*Honest validation.* Columns drawn from one model snapshot are strongly
correlated with each other, so a column-wise random split would put near-copies
of training columns into validation and report a skill the emulator does not
have. The split is therefore by SOLAR GEOMETRY GROUP: every column sharing an
(orbital_phase, synodic_phase) pair comes from one snapshot and lands wholly in
one partition. The synthetic sweep randomises its geometry per column, so the
same rule degenerates to a plain random split there, which is correct because
those columns are independent by construction.

Usage::

    python tools/radiation_emulator/train_emulator.py \
        --data 'training/*.nc' --out emulator_weights.nc --epochs 40

    python tools/radiation_emulator/train_emulator.py \
        --data 'training/*.nc' --out best.nc --sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import optax                                                    # noqa: E402

import jcm.constants as c                                       # noqa: E402
from jcm.physics.radiation.nn_emulator import (                 # noqa: E402
    EmulatorWeights,
    apply_input_scaling,
    InputScaling,
    flux_to_heating_rate,
    lw_flux_scale,
    init_lw_emulator_weights,
    lw_emulator_column,
    n_input_features,
    preprocess_lw_inputs,
    preprocess_sw_inputs,
    reconstruct_sw_interface_fluxes,
    save_emulator_weights,
)

SECONDS_PER_DAY = 86400.0

# Interface flux channels, in the order reconstruct_*_interface_fluxes expects.
CHANNELS = ("down", "up", "down_clear", "up_clear")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_columns(pattern):
    """Concatenate every generated training file matching ``pattern``."""
    import xarray as xr

    paths = []
    for piece in str(pattern).split(","):
        matched = sorted(glob.glob(piece.strip()))
        if not matched:
            raise FileNotFoundError(f"no training file matches {piece!r}")
        paths.extend(matched)

    parts = [xr.open_dataset(p) for p in paths]
    nlev = {int(d.sizes["level"]) for d in parts}
    if len(nlev) != 1:
        raise ValueError(f"training files disagree on level count: {nlev}")
    counts = [int(d.sizes["column"]) for d in parts]
    ds = xr.concat(parts, dim="column", combine_attrs="drop_conflicts")
    source_ids = np.repeat(np.arange(len(paths)), counts)
    print(f"loaded {ds.sizes['column']} columns from {len(paths)} file(s)")
    return ds, paths, source_ids


def solar_group_ids(ds):
    """Group columns by the snapshot they came from.

    Columns sharing a solar geometry came from one model snapshot and are
    spatially correlated; the sweep source randomises geometry per column, so
    there each column is its own group. See the module docstring.
    """
    phase = np.stack([
        np.round(np.asarray(ds["orbital_phase"].values), 9),
        np.round(np.asarray(ds["synodic_phase"].values), 9),
    ], axis=1)
    _, ids = np.unique(phase, axis=0, return_inverse=True)
    return ids


def split_by_group(group_ids, fractions, seed):
    """Split column indices into train/val/test without splitting a group.

    Groups are assigned greedily to fill each partition's share of the COLUMNS
    (not of the groups): a trajectory snapshot contributes thousands of columns
    while a sweep column contributes one, so an even split of groups would be a
    wildly uneven split of data.

    Largest group first. Taking them in random order instead lets a run of
    small groups fill the training quota and a single large group arrive while
    training is still the emptiest partition — which put every column in
    training and left validation empty.
    """
    rng = np.random.default_rng(seed)
    groups, sizes = np.unique(group_ids, return_counts=True)
    # Shuffle first so equal-sized groups (the sweep's per-column groups) are
    # ordered randomly rather than by label, then sort by descending size.
    shuffle = rng.permutation(len(groups))
    order = shuffle[np.argsort(-sizes[shuffle], kind="stable")]

    quotas = [f * len(group_ids) for f in fractions]
    assigned, filled = {}, [0, 0, 0]
    for i in order:
        # Whichever partition is furthest below its quota takes the group.
        k = int(np.argmax([q - n for q, n in zip(quotas, filled)]))
        assigned[groups[i]] = k
        filled[k] += int(sizes[i])

    membership = np.array([assigned[g] for g in group_ids])
    splits = [np.where(membership == k)[0] for k in range(3)]
    if any(len(s) == 0 for s in splits):
        raise ValueError(
            f"split produced an empty partition ({[len(s) for s in splits]} "
            f"columns from {len(groups)} groups). With so few solar-geometry "
            "groups a held-out split is not meaningful -- generate columns "
            "from more model snapshots."
        )
    return splits


def split_by_source_and_group(group_ids, source_ids, fractions, seed):
    """Split within each source file, so every source reaches val and test.

    Splitting the pooled set by group alone concentrated the realistic columns
    in training. The sources have wildly different group granularity -- a
    trajectory file is ~10^5 columns in 8-40 solar-geometry groups, while the
    synthetic sweep randomises geometry per column and so is one group per
    column. Largest-group-first then hands every trajectory snapshot to
    training (its quota is 8x validation's, so it stays furthest below quota
    longest), and the sweep's singletons are left to fill validation and test.

    The result was a test set that was 95% synthetic and contained ZERO ERA5
    trajectory columns, so the offline metric was measured on columns the
    coupled model never visits -- it reported a -0.13 W/m2 shortwave bias for
    a network that ran a -21.7 W/m2 bias in the GCM.

    Splitting per source keeps each one's own 80/10/10, so held-out skill is
    reported on realistic columns too.
    """
    splits = [[], [], []]
    for source in np.unique(source_ids):
        idx = np.where(source_ids == source)[0]
        # Vary the seed per source so sources do not share a group ordering.
        sub = split_by_group(group_ids[idx], fractions, seed + int(source))
        for k in range(3):
            splits[k].append(idx[sub[k]])
    return [np.concatenate(p) for p in splits]


def build_features(ds, band_mode):
    """Build network inputs and normalized targets for every column.

    Returns a dict of unscaled inputs, per-column auxiliary scalars, targets,
    and the physical quantities the heating-rate loss needs.
    """
    f32 = lambda name: jnp.asarray(ds[name].values, dtype=jnp.float32)  # noqa: E731

    q = f32("specific_humidity")
    # Identical to radiation_scheme_emulated: mass mixing ratio to volume.
    h2o_vmr = q / (c.eps * (1.0 - q) + q)
    ozone_vmr = f32("ozone_vmr")
    cf = f32("cloud_fraction")
    # GRID-MEAN water paths, exactly as the scheme derives them
    # (``cloud_water`` in the training files is the grid-mean mixing ratio;
    # multiplying by cf again would make the feature scale as cf^2).
    cwp = f32("cloud_water") * f32("air_density") * f32("layer_thickness")
    cip = f32("cloud_ice") * f32("air_density") * f32("layer_thickness")

    # Already RESOLVED by the generator (microphysical where the source had a
    # value, diagnostic fallback elsewhere) and strictly positive, and the
    # RRTMGP labels in the same file were produced from these very numbers.
    r_eff_liq = f32("r_eff_liq")
    r_eff_ice = f32("r_eff_ice")

    temperature = f32("temperature")
    pressure = f32("pressure_levels")
    n_sw = n_input_features(band_mode, ds.sizes["band_sw"])
    n_lw = n_input_features(band_mode, ds.sizes["band_lw"])
    unit_sw = InputScaling(x_max=jnp.ones(n_sw))
    unit_lw = InputScaling(x_max=jnp.ones(n_lw))

    # Unscaled here; the divide-by-max scaling is fitted on the TRAIN split
    # alone and applied afterwards, so validation cannot leak into it.
    # Same feature clip as radiation_scheme_emulated: inference feeds
    # maximum(sin_altitude, min_cos_zenith), so training on the raw cosine
    # would fit twilight/night columns to feature values the deployed
    # network never receives (and skew the fitted max-scaling).
    from jcm.physics.radiation.radiation_types import RadiationParameters
    min_mu0 = float(RadiationParameters.default().min_cos_zenith)
    x_sw = jax.vmap(
        lambda *a: preprocess_sw_inputs(*a[:8], unit_sw, *a[8:], band_mode)
    )(temperature, pressure, h2o_vmr, ozone_vmr, cwp, cip, cf,
      jnp.maximum(f32("cos_zenith"), min_mu0), r_eff_liq, r_eff_ice,
      f32("aod_sw_per_band"), f32("ssa_sw_per_band"),
      f32("asy_sw_per_band"))
    x_lw = jax.vmap(
        lambda *a: preprocess_lw_inputs(*a[:8], unit_lw, *a[8:], band_mode)
    )(temperature, pressure, h2o_vmr, ozone_vmr, cwp, cip, cf,
      f32("co2_vmr"), r_eff_liq, r_eff_ice,
      f32("aod_lw_per_band"), f32("ssa_lw_per_band"),
      f32("asy_lw_per_band"))

    # Auxiliary scalar fed to the surface dense layer of each network.
    albedo = 0.46 * f32("surface_albedo_vis") + 0.54 * f32("surface_albedo_nir")
    emissivity = f32("surface_emissivity")
    surface_temperature = f32("surface_temperature")

    # Normalisation scales, matching reconstruct_*_interface_fluxes exactly:
    # the TOA insolation for shortwave, the surface emission for longwave.
    toa_sw_down = f32("sw_flux_down")[:, 0]
    lw_scale = jax.vmap(lw_flux_scale)(surface_temperature, temperature)

    def stack(prefix):
        return jnp.stack([f32(f"{prefix}_flux_{ch}") for ch in CHANNELS], -1)

    sw_labels, lw_labels = stack("sw"), stack("lw")
    lit = toa_sw_down > 1.0
    # A dark column carries no shortwave information: every flux is zero
    # whatever the network says, so it is excluded from the SW loss rather
    # than divided by a zero normalisation.
    safe_sw = jnp.where(lit, toa_sw_down, 1.0)

    return dict(
        x_sw=x_sw, x_lw=x_lw,
        aux_sw=albedo[:, None], aux_lw=emissivity[:, None],
        y_sw=sw_labels / safe_sw[:, None, None],
        y_lw=lw_labels / lw_scale[:, None, None],
        sw_scale=safe_sw, lw_scale=lw_scale, lit=lit,
        pressure_interfaces=f32("pressure_interfaces"),
        sw_labels=sw_labels, lw_labels=lw_labels,
    )


def report_target_range(data):
    """Report how much of the target distribution the output layer can reach.

    The output dense layer ends in a sigmoid, so predictions are confined to
    (0, 1). Longwave upward flux at the surface is emission PLUS reflected
    downwelling, which normalised by the emission alone exceeds 1 whenever the
    emissivity is below 1 — an error the network cannot train away. Quantify it
    rather than discovering it as an unexplained bias later.
    """
    for band in ("sw", "lw"):
        y = np.asarray(data[f"y_{band}"])
        if band == "sw":
            # The TOA downward interface normalizes to exactly 1 and is set by
            # construction in reconstruct_sw_interface_fluxes, not predicted,
            # so counting it here would misreport a ceiling problem.
            y = y[np.asarray(data["lit"])][:, 1:, :]
        over = float(np.mean(y > 1.0))
        print(f"{band} normalized targets: min {y.min():.4f} "
              f"max {y.max():.4f}, {over * 100:.3f}% above the sigmoid "
              f"ceiling of 1")


def fit_scaling(x_train):
    """Affine input scaling fitted on the training split only.

    Centres each feature on its mean and divides by the largest absolute
    deviation, so every feature lands in [-1, 1] with mean 0. The floor on the
    denominator keeps a constant feature (CO2, in a fixed-concentration
    dataset) at exactly 0 instead of dividing rounding noise by ~0.
    """
    x_offset = jnp.mean(x_train, axis=(0, 1))
    x_max = jnp.max(jnp.abs(x_train - x_offset), axis=(0, 1))
    return InputScaling(x_max=jnp.maximum(x_max, 1e-6), x_offset=x_offset)


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------


EVAL_CHUNK = 4096


def _predict(weights, x, aux):
    """Run one network over a batch of columns."""
    return jax.vmap(lw_emulator_column, in_axes=(0, 0, None))(x, aux, weights)


def _predict_chunked(weights, x, aux, chunk=EVAL_CHUNK):
    """Predict over a whole split without materialising every activation.

    A vmapped GRU over tens of thousands of columns holds all three hidden
    sequences at once, which is a multi-GB allocation the training batches
    never provoke.
    """
    return jnp.concatenate([
        _predict(weights, x[i:i + chunk], aux[i:i + chunk])
        for i in range(0, x.shape[0], chunk)
    ], axis=0)


def _heating(pred_norm, scale, p_half, is_sw):
    """All-sky heating rate (K/day) from normalized interface predictions."""
    if is_sw:
        down, up, _, _ = jax.vmap(reconstruct_sw_interface_fluxes)(
            pred_norm, scale)
    else:
        down, up = pred_norm[..., 0] * scale[:, None], \
            pred_norm[..., 1] * scale[:, None]
    return jax.vmap(flux_to_heating_rate)(down, up, p_half) * SECONDS_PER_DAY


def mass_weights(p_half):
    """Per-level layer-mass weights summing to 1 along each column.

    Reported, but NOT used for training -- see :func:`uniform_weights`. Mass
    weighting measures the energy error, which is what conservation cares
    about, and it is the right lens for asking whether the emulator loses
    energy. It is the wrong lens for asking whether the model survives.
    """
    dp = jnp.diff(p_half, axis=-1)
    return dp / jnp.sum(dp, axis=-1, keepdims=True)


def uniform_weights(p_half):
    """Equal weight per level, normalised to sum to 1 along each column.

    This is what the heating loss trains on, because heating rate -- not
    energy -- is what the model integrates, and a level's temperature does not
    care how little mass it holds.

    Mass weighting was tried first and produced a model that NaN'd the GCM in
    under five days. Heating is (g/cp) dF/dp and the topmost layer is ~2 Pa
    thick, so a 0.3 W/m2 flux-difference error there is 130 K/day; mass
    weighting gave that level ~1e-5 of the loss and the trained emulator
    reached 130 K/day RMSE at level 0 while sitting at 0.02-0.17 K/day through
    the whole troposphere. The headline mass-weighted score was 0.72 K/day and
    said nothing about it.
    """
    n = p_half.shape[-1] - 1
    return jnp.full(p_half.shape[:-1] + (n,), 1.0 / n)


# Below this normalised mean a channel/level carries essentially no signal
# (longwave downward at TOA is ~1e-4), and 1/mean would hand it a weight of
# 1e4 and let it dominate the flux term outright.
_WEIGHT_PROF_FLOOR = 1.0e-2


def channel_weights(y_train):
    """Per-level, per-channel ``1/mean`` weights for the flux term.

    Ukkonen's ``weight_prof = 1/y_tr.mean(axis=0)``: it equalises the
    contribution of each level and each output channel, so the large
    downward flux near TOA cannot swamp the small upward one. Fitted on the
    training split only, like the input scaling.
    """
    return 1.0 / jnp.maximum(jnp.mean(y_train, axis=0), _WEIGHT_PROF_FLOOR)


def make_loss(is_sw, alpha, weight_prof):
    """Build the loss for one band, following Ukkonen's hybrid formulation.

    ``alpha * RMSE(heating) + (1 - alpha) * MSE(weighted flux)``, with the
    heating term in K/day computed from denormalised fluxes and the TRUE dp,
    so the model top's amplification enters the loss exactly as it enters the
    physics. The asymmetry is deliberate: RMSE on the heating term keeps it
    from vanishing as it converges, while alpha = 1e-4 is unit-balancing --
    the flux MSE is O(1e-4) on O(1) targets and the heating RMSE is O(1) in
    K/day, so the two contribute comparably rather than one dominating.
    """
    def loss_fn(weights, batch):
        pred = _predict(weights, batch["x"], batch["aux"])
        mask = batch["mask"][:, None, None]
        n = jnp.maximum(jnp.sum(batch["mask"]), 1.0)
        err = (pred - batch["y"]) * weight_prof
        n_elem = pred.shape[1] * pred.shape[2]
        if is_sw:
            # Interface-0 down / down_clear are overwritten with the exact
            # incoming flux at inference (reconstruct_sw_interface_fluxes),
            # and their normalised target is exactly 1 — unreachable for a
            # sigmoid. Training on them feeds an irreducible error into the
            # shared output layer and the checkpoint-selection metric for
            # predictions that are never used (PR #730 review).
            elem = jnp.ones((pred.shape[1], pred.shape[2]))
            elem = elem.at[0, 0].set(0.0).at[0, 2].set(0.0)
            err = err * elem
            n_elem = n_elem - 2
        flux_mse = jnp.sum((err ** 2) * mask) / (n * n_elem)
        if alpha == 0.0:
            return flux_mse, flux_mse
        hr_pred = _heating(pred, batch["scale"], batch["p_half"], is_sw)
        hr_true = _heating(batch["y"], batch["scale"], batch["p_half"], is_sw)
        w = uniform_weights(batch["p_half"]) * batch["mask"][:, None]
        hr_rmse = jnp.sqrt(jnp.sum(((hr_pred - hr_true) ** 2) * w) / n)
        return alpha * hr_rmse + (1.0 - alpha) * flux_mse, flux_mse
    return loss_fn


def band_metrics(pred_norm, data, idx, is_sw):
    """Physical error metrics on a held-out split, in W/m^2 and K/day."""
    scale = data["sw_scale" if is_sw else "lw_scale"][idx]
    truth = data["sw_labels" if is_sw else "lw_labels"][idx]
    mask = np.asarray(data["lit"][idx]) if is_sw else np.ones(len(idx), bool)
    pred = np.asarray(pred_norm) * np.asarray(scale)[:, None, None]
    truth = np.asarray(truth)

    def err(a, b):
        d = (a - b)[mask]
        return float(np.sqrt(np.mean(d ** 2))), float(np.mean(d))

    p_half = data["pressure_interfaces"][idx]
    hr_pred = np.asarray(jax.vmap(flux_to_heating_rate)(
        jnp.asarray(pred[..., 0]), jnp.asarray(pred[..., 1]), p_half,
    )) * SECONDS_PER_DAY
    hr_true = np.asarray(jax.vmap(flux_to_heating_rate)(
        jnp.asarray(truth[..., 0]), jnp.asarray(truth[..., 1]), p_half,
    )) * SECONDS_PER_DAY
    # Both weightings are reported because they answer different questions:
    # the mass-weighted one is the energy error, the raw one exposes the
    # near-vacuum top levels where a small flux error is a large K/day error
    # that the model still has to integrate.
    w = np.asarray(mass_weights(p_half))[mask]
    d_hr = (hr_pred - hr_true)[mask]
    hr_rmse = float(np.sqrt(np.sum(w * d_hr ** 2) / np.sum(w)))
    hr_bias = float(np.sum(w * d_hr) / np.sum(w))
    hr_rmse_raw, _ = err(hr_pred, hr_true)
    # The single worst level, because a column mean of any weighting hides one
    # catastrophic layer -- and one catastrophic layer is what NaNs the model.
    per_level = np.sqrt(np.mean(d_hr ** 2, axis=0))
    hr_rmse_worst = float(per_level.max())
    hr_worst_level = int(np.argmax(per_level))

    # TOA is interface 0 and the surface is the last: the stored columns are
    # TOA-first (see the generator's vertical_convention attribute).
    toa_rmse, toa_bias = err(pred[:, 0, 1], truth[:, 0, 1])
    sfc_rmse, sfc_bias = err(pred[:, -1, 0], truth[:, -1, 0])
    return dict(
        toa_up_rmse=toa_rmse, toa_up_bias=toa_bias,
        sfc_down_rmse=sfc_rmse, sfc_down_bias=sfc_bias,
        heating_rmse=hr_rmse, heating_bias=hr_bias,
        heating_rmse_raw=hr_rmse_raw,
        heating_rmse_worst_level=hr_rmse_worst,
        heating_worst_level=hr_worst_level,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_band(data, splits, is_sw, config, key, log_prefix=""):
    """Train one band's network; return ``(weights, scaling, history)``."""
    train_idx, val_idx = splits[0], splits[1]
    x_all = data["x_sw" if is_sw else "x_lw"]
    aux_all = data["aux_sw" if is_sw else "aux_lw"]
    y_all = data["y_sw" if is_sw else "y_lw"]
    scale_all = data["sw_scale" if is_sw else "lw_scale"]
    mask_all = data["lit"] if is_sw else jnp.ones(x_all.shape[0], bool)

    scaling = fit_scaling(x_all[train_idx])
    x_all = apply_input_scaling(x_all, scaling)

    def gather(idx):
        idx = jnp.asarray(idx)
        return dict(x=x_all[idx], aux=aux_all[idx], y=y_all[idx],
                    scale=scale_all[idx], p_half=data["pressure_interfaces"][idx],
                    mask=mask_all[idx].astype(jnp.float32))

    val_batches = [gather(val_idx[i:i + EVAL_CHUNK])
                   for i in range(0, len(val_idx), EVAL_CHUNK)]
    weights = init_lw_emulator_weights(
        n_features=x_all.shape[-1], units=config["units"],
        n_outputs=len(CHANNELS), key=key,
    )
    # Fitted on the columns the loss actually sees: every shortwave flux is
    # zero at night, so including dark columns would drag each mean down and
    # inflate the weights by roughly 1/(lit fraction).
    train_sel = jnp.asarray(train_idx)[mask_all[jnp.asarray(train_idx)]]
    weight_prof = channel_weights(y_all[train_sel])
    loss_fn = make_loss(is_sw, config["alpha"], weight_prof)
    n_steps = max(1, len(train_idx) // config["batch_size"]) * config["epochs"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config["lr"] * 0.1, peak_value=config["lr"],
        warmup_steps=max(1, n_steps // 20), decay_steps=n_steps,
        end_value=config["lr"] * 0.01,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adam(schedule))
    opt_state = optimizer.init(weights)

    @jax.jit
    def step(weights, opt_state, batch):
        (total, flux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            weights, batch)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        return optax.apply_updates(weights, updates), opt_state, total, flux

    @jax.jit
    def evaluate(weights, batch):
        # The FULL objective, including the heating term -- not the flux part
        # alone. Selecting the best epoch by flux while ranking and reporting
        # by heating lets a longer run pick a checkpoint that is better on the
        # selection metric and worse on the reported one, which is how a
        # 300-epoch 128-unit model ended up with worse longwave heating than a
        # 40-epoch 64-unit one.
        return loss_fn(weights, batch)[0]

    rng = np.random.default_rng(config["seed"])
    best = (np.inf, weights)
    history = []
    started = time.time()
    for epoch in range(config["epochs"]):
        order = rng.permutation(train_idx)
        n_batches = max(1, len(order) // config["batch_size"])
        for b in range(n_batches):
            batch = gather(order[b * config["batch_size"]:
                                 (b + 1) * config["batch_size"]])
            weights, opt_state, _, _ = step(weights, opt_state, batch)
        val_loss = float(np.mean([evaluate(weights, b) for b in val_batches]))
        history.append(val_loss)
        if val_loss < best[0]:
            best = (val_loss, weights)
        print(f"{log_prefix}epoch {epoch + 1:3d}/{config['epochs']}  "
              f"val loss {val_loss:.3e}  "
              f"({time.time() - started:.0f}s)", flush=True)
    return best[1], scaling, history


def evaluate_split(weights, scaling, data, idx, is_sw):
    """Physical metrics for one trained band on one split."""
    x = data["x_sw" if is_sw else "x_lw"][jnp.asarray(idx)]
    x = apply_input_scaling(x, scaling)
    aux = data["aux_sw" if is_sw else "aux_lw"][jnp.asarray(idx)]
    return band_metrics(_predict_chunked(weights, x, aux), data, idx, is_sw)


def run_config(data, splits, config, verbose=True):
    """Train both bands under one hyperparameter configuration."""
    key_sw, key_lw = jax.random.split(jax.random.PRNGKey(config["seed"]))
    w_sw, s_sw, _ = train_band(data, splits, True, config, key_sw,
                               "  [sw] " if verbose else "")
    w_lw, s_lw, _ = train_band(data, splits, False, config, key_lw,
                               "  [lw] " if verbose else "")
    val = dict(
        sw=evaluate_split(w_sw, s_sw, data, splits[1], True),
        lw=evaluate_split(w_lw, s_lw, data, splits[1], False),
    )
    return dict(weights=EmulatorWeights(sw=w_sw, lw=w_lw),
                sw_scaling=s_sw, lw_scaling=s_lw, val=val)


def score(val):
    """Single number to rank configurations by.

    The WORST LEVEL's heating RMSE, summed over bands. Column-mean heating
    error ranks models by how well they do on average; what determines whether
    the GCM survives is the single worst layer, and ranking on the mean once
    selected a model that scored 0.72 K/day and NaN'd the model in five days
    off a 130 K/day error at the top level alone.
    """
    return (val["sw"]["heating_rmse_worst_level"]
            + val["lw"]["heating_rmse_worst_level"])


# Centred on 32 units, which is where this literature actually operates:
# Ukkonen's production shortwave model is 3 GRU layers x 16 units -- 5,698
# parameters -- and reaches 0.16 K/day, and nothing published exceeds 96.
# Earlier sweeps here ran 128-256, one to two orders of magnitude over, which
# buys nothing at run time (the network is already free against RRTMGP) and
# costs exactly the thing that matters: an over-parameterised network
# extrapolates wildly where the training distribution is thin, and that is
# what killed the coupled runs.
SWEEP = [
    dict(units=16, lr=3e-3, alpha=1e-4),
    dict(units=24, lr=3e-3, alpha=1e-4),
    dict(units=32, lr=3e-3, alpha=1e-4),
    dict(units=64, lr=3e-3, alpha=1e-4),
    dict(units=32, lr=1e-3, alpha=1e-4),
    dict(units=32, lr=3e-3, alpha=1e-3),
    dict(units=32, lr=3e-3, alpha=0.0),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True,
                   help="generated training file(s): path, glob or list")
    p.add_argument("--out", required=True, help="weight file to write")
    p.add_argument("--band-mode", default="per_band",
                   choices=("none", "broadband", "per_band"))
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--alpha", type=float, default=1e-4,
                   help="weight on the heating RMSE term; 1e-4 balances "
                        "it against the flux MSE (Ukkonen)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--sweep-epochs", type=int, default=12,
                   help="shorter budget per sweep candidate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", action="store_true",
                   help="rank SWEEP candidates on validation, then retrain "
                        "the winner for the full --epochs budget")
    p.add_argument("--report", default=None, help="JSON metrics output")
    args = p.parse_args(argv)

    ds, paths, source_ids = load_columns(args.data)
    data = build_features(ds, args.band_mode)
    groups = solar_group_ids(ds)
    splits = split_by_source_and_group(
        groups, source_ids, (0.8, 0.1, 0.1), args.seed,
    )
    print(f"split: {len(splits[0])} train / {len(splits[1])} val / "
          f"{len(splits[2])} test columns from "
          f"{len(np.unique(groups))} solar-geometry groups")
    # Per-source composition of the held-out sets: the failure this guards is
    # a test set that looks big but contains none of the realistic columns.
    for k, name in enumerate(("train", "val", "test")):
        by_source = [
            int(np.sum(source_ids[splits[k]] == s))
            for s in range(len(paths))
        ]
        parts = ", ".join(
            f"{pathlib.Path(p).name}:{n}" for p, n in zip(paths, by_source)
        )
        print(f"  {name:5s} {parts}")
    print(f"lit fraction: {float(jnp.mean(data['lit'])):.3f}")
    report_target_range(data)

    base = dict(batch_size=args.batch_size, seed=args.seed,
                epochs=args.epochs, units=args.units, lr=args.lr,
                alpha=args.alpha)

    sweep_results = []
    if args.sweep:
        for i, candidate in enumerate(SWEEP):
            config = {**base, **candidate, "epochs": args.sweep_epochs}
            print(f"\n=== sweep {i + 1}/{len(SWEEP)}: {candidate} ===",
                  flush=True)
            result = run_config(data, splits, config)
            sweep_results.append(dict(config=candidate, val=result["val"],
                                      score=score(result["val"])))
            print(f"  score (total heating RMSE) {score(result['val']):.4f} "
                  "K/day")
        best = min(sweep_results, key=lambda r: r["score"])
        print(f"\nbest sweep config: {best['config']} "
              f"(score {best['score']:.4f} K/day)")
        base = {**base, **best["config"]}

    print(f"\n=== final training: {base} ===", flush=True)
    final = run_config(data, splits, base)
    test = dict(
        sw=evaluate_split(final["weights"].sw, final["sw_scaling"],
                          data, splits[2], True),
        lw=evaluate_split(final["weights"].lw, final["lw_scaling"],
                          data, splits[2], False),
    )

    metadata = dict(
        band_mode=args.band_mode, units=int(base["units"]),
        n_outputs=len(CHANNELS),
        n_bnd_sw=int(ds.sizes["band_sw"]), n_bnd_lw=int(ds.sizes["band_lw"]),
        n_levels=int(ds.sizes["level"]),
        trained_on=";".join(pathlib.Path(p).name for p in paths),
        n_train=int(len(splits[0])),
        learning_rate=float(base["lr"]),
        alpha=float(base["alpha"]),
        epochs=int(base["epochs"]),
    )
    save_emulator_weights(args.out, final["weights"], final["sw_scaling"],
                          final["lw_scaling"], metadata)
    print(f"\nwrote {args.out}")

    for band in ("sw", "lw"):
        m = test[band]
        print(f"TEST {band.upper()}: "
              f"TOA up {m['toa_up_rmse']:7.2f} W/m2 RMSE "
              f"({m['toa_up_bias']:+.2f} bias) | "
              f"sfc down {m['sfc_down_rmse']:7.2f} "
              f"({m['sfc_down_bias']:+.2f}) | "
              f"heating {m['heating_rmse']:.4f} K/day mass-weighted, "
              f"{m['heating_rmse_raw']:.2f} unweighted, "
              f"WORST LEVEL {m['heating_rmse_worst_level']:.1f} "
              f"at k={m['heating_worst_level']}")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(dict(metadata=metadata, test=test,
                           val=final["val"], sweep=sweep_results), fh,
                      indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
