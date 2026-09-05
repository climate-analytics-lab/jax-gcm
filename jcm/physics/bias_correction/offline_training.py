"""Offline warm-up for the NN bias-correction term (issue #356).

The correction network starts as a zero-initialised, trainable no-op. This stage
fits its weights offline so it stops being a no-op. The training target is the
nudging tendency: run SPEEDY nudged toward ERA5 (``NudgingConfig.temp_humidity``
+ an ERA5 ``NudgingTarget``), then for every saved step recompute the nudge it
needed, ``dX/dt = inv_tau * (ERA5 - state)``. That recorded nudge is the
estimate of what SPEEDY got wrong, and the network learns to reproduce it from
the model's own state alone. This is the Rasp (2020) / Watt-Meyer (2021) recipe.

Everything here is cloud-free and array-only so it unit-tests without ERA5. The
notebook (``notebooks/07_bias_correction_offline_training.ipynb``) handles the
ERA5 download, the nudged run, and the plots; it calls these helpers for the
recompute, stats, and training.

The feature layout (T, q, u, v concatenated per level) comes from
:data:`jcm.physics.bias_correction.nn_bias_correction.FIELD_ORDER`, so the
trained network sees exactly what the live term feeds it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jcm.nudging import nudging_tendency
from jcm.physics.bias_correction.nn_bias_correction import (
    DEFAULT_ACTIVATION,
    FIELD_ORDER,
    NNBiasCorrection,
    mlp,
)
from jcm.physics.bias_correction.optim import adam_init, adam_update

_N_FIELDS = len(FIELD_ORDER)


# ---------------------------------------------------------------------------
# Recompute the training target from a saved nudged trajectory
# ---------------------------------------------------------------------------


def recompute_target_tendencies(states, target, config):
    """Per-step nudging tendency for a whole saved trajectory.

    Applies the pure :func:`jcm.nudging.nudging_tendency` to each saved step,
    so the recorded target is identical to the nudge the live run applied.

    Args:
        states: trajectory state with a leading time axis, fields shaped
            ``(ntime, nlev, *horiz)`` (e.g. ``predictions.dynamics``).
        target: :class:`jcm.nudging.NudgingTarget` of ERA5 fields on the model
            grid at the same saved times, same leading time axis.
        config: the :class:`jcm.nudging.NudgingConfig` used for the run.

    Returns:
        A :class:`jcm.physics_interface.PhysicsTendency` with the per-step
        target tendencies, leading time axis preserved.

    """
    return jax.vmap(lambda s, t: nudging_tendency(s, t, config))(states, target)


def _trajectory_columns(obj) -> jnp.ndarray:
    """Stack ``(ntime, nlev, *horiz)`` fields into ``(N, 4*nlev)`` rows.

    Every (time, *horiz) point becomes one row; fields are concatenated in
    :data:`FIELD_ORDER`. Works on a state or a tendency (same field names).
    """
    cols = []
    for name in FIELD_ORDER:
        field = getattr(obj, name)                       # (ntime, nlev, *horiz)
        nlev = field.shape[1]
        rows = jnp.moveaxis(field, 1, -1).reshape(-1, nlev)  # (N, nlev)
        cols.append(rows)
    return jnp.concatenate(cols, axis=1)                 # (N, 4*nlev)


def assemble_training_arrays(states, tendencies):
    """Flatten a trajectory into matched ``(N, 4*nlev)`` features and targets.

    Args:
        states: trajectory state (input features), leading time axis.
        tendencies: matching target tendencies from
            :func:`recompute_target_tendencies`, same leading time axis.

    Returns:
        ``(feats, targets)``, both ``(N, 4*nlev)`` with rows aligned column for
        column. ``feats`` are raw state values; ``targets`` are per-second
        tendencies. Standardise/normalise with :func:`compute_norm_stats`.

    """
    return _trajectory_columns(states), _trajectory_columns(tendencies)


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------


def corrected_feature_mask(nlev: int, correct) -> np.ndarray:
    """Boolean mask (length ``4*nlev``) selecting the slots in ``correct``.

    The feature/output vector packs the four fields of :data:`FIELD_ORDER` in
    blocks of ``nlev``; this marks the blocks named in ``correct`` so the loss
    and stats ignore the variables the term does not correct.
    """
    mask = np.zeros(_N_FIELDS * nlev, dtype=bool)
    for i, name in enumerate(FIELD_ORDER):
        if name in correct:
            mask[i * nlev:(i + 1) * nlev] = True
    return mask


def compute_norm_stats(feats, targets,
                       correct=("temperature", "specific_humidity")):
    """Per-feature input standardisation and output scale, length ``4*nlev``.

    ``in_mean``/``in_std`` standardise the inputs (T ~ 250, q ~ 1e-3 differ by
    orders of magnitude). ``out_scale`` is the per-output std of the recorded
    tendencies, so the network learns a dimensionless map and the term scales
    it back to per-second units. Uncorrected slots (never read by the term)
    and any zero-variance slot get scale 1 so the normalisation is well posed.

    Deliberately NO variance floor on ``out_scale`` (July 10 lesson).
    ``out_scale`` is dual-use: it normalises the offline loss targets AND
    scales the term's live output. The loss side needs no floor because
    ``targets / std(targets)`` is unit-variance per slot by construction. The
    output side must keep the raw std: flooring it handed the network ~150x
    more authority over near-zero-variance levels (stratospheric humidity,
    with huge radiative leverage) and the free-running climate blew up
    +30 K. Tiny observed signal -> tiny correction authority is a safety
    property, not a bug. (The online loss floors its own separate
    normalisation in ``stds_from_stats``; that one never touches the output.)

    Returns:
        ``(in_mean, in_std, out_scale)``, each ``(4*nlev,)``, ready to pass to
        :class:`NNBiasCorrection`.

    """
    nlev = feats.shape[1] // _N_FIELDS
    in_mean = jnp.mean(feats, axis=0)
    in_std = jnp.std(feats, axis=0)
    in_std = jnp.where(in_std > 0, in_std, 1.0)

    mask = jnp.asarray(corrected_feature_mask(nlev, correct))
    out_scale = jnp.std(targets, axis=0)
    out_scale = jnp.where(mask & (out_scale > 0), out_scale, 1.0)
    return in_mean, in_std, out_scale


# ---------------------------------------------------------------------------
# Supervised training (jax.grad + minibatch Adam; no optax in the repo)
# ---------------------------------------------------------------------------


def normalized_mse(layers, feats, targets, in_mean, in_std, out_scale,
                   correct=("temperature", "specific_humidity"),
                   activation=jax.nn.tanh) -> float:
    """Mean squared error in the dimensionless target space, over ``correct``.

    The quantity the training minimises, evaluated on a full array (no
    minibatching). Compare a zero-initialised ``layers`` (the no-op baseline)
    against the trained one to show the offline error drop.

    ``activation`` MUST match what :func:`train` used. Scoring a net under a
    different nonlinearity than it was fitted with reports nonsense: a
    gelu-trained 512-wide net evaluated as tanh came out at ratio 4.0, i.e.
    four times WORSE than the no-op baseline, while its true ratio was 0.30.
    """
    nlev = feats.shape[1] // _N_FIELDS
    feats_std = (feats - in_mean) / in_std
    target_norm = targets / out_scale
    weight = jnp.asarray(corrected_feature_mask(nlev, correct), dtype=feats_std.dtype)
    pred = jax.vmap(lambda f: mlp(f, layers, activation))(feats_std)
    return float(jnp.sum(weight * (pred - target_norm) ** 2)
                 / (feats.shape[0] * jnp.sum(weight)))


def train(layers, feats, targets, in_mean, in_std, out_scale,
          correct=("temperature", "specific_humidity"), *,
          steps: int = 2000, batch_size: int = 4096, lr: float = 1e-3,
          key=None, activation=jax.nn.tanh):
    """Fit ``layers`` to the recorded tendency with minibatch Adam.

    The network learns the dimensionless map (standardised state) ->
    (target / out_scale) over the slots in ``correct``, which is exactly what
    ``bias_correction_tendency`` reverses at run time. Adam is hand-rolled
    (shared with the online trainer via
    :mod:`jcm.physics.bias_correction.optim`) because the repo carries no
    optax dependency.

    Args:
        layers: initial MLP weights (tuple of :class:`DenseWeights`). Start
            from a non-zero init so gradients are non-trivial; the trained
            result is loaded into the term afterwards.
        feats, targets: ``(N, 4*nlev)`` arrays from
            :func:`assemble_training_arrays`. Subsample upstream if memory is
            tight.
        in_mean, in_std, out_scale: buffers from :func:`compute_norm_stats`.
        correct: variables to fit; others are masked out of the loss.
        steps, batch_size, lr: optimisation controls.
        key: PRNG key for minibatch sampling (defaults to ``key(0)``).
        activation: hidden-layer nonlinearity, matching :func:`mlp`'s own
            default. Exposed so an architecture sweep can vary it with the
            training code otherwise identical; a term's saved weights carry no
            activation, so anything other than the default is a DIAGNOSTIC
            until :class:`NNBiasCorrection` learns to store it.

    Returns:
        ``(layers, history)`` where ``history`` is a list of
        ``(step, loss)`` samples for plotting.

    """
    if key is None:
        key = jax.random.key(0)

    nlev = feats.shape[1] // _N_FIELDS
    feats_std = (feats - in_mean) / in_std
    target_norm = targets / out_scale
    weight = jnp.asarray(corrected_feature_mask(nlev, correct), dtype=feats_std.dtype)
    norm = jnp.sum(weight)

    def loss(layers, fb, tb):
        pred = jax.vmap(lambda f: mlp(f, layers, activation))(fb)
        return jnp.sum(weight * (pred - tb) ** 2) / (fb.shape[0] * norm)

    m, v = adam_init(layers)

    @jax.jit
    def step(layers, m, v, t, fb, tb):
        grads = jax.grad(loss)(layers, fb, tb)
        return adam_update(layers, grads, m, v, t, lr)

    n = feats_std.shape[0]
    history = []
    log_every = max(1, steps // 20)
    for i in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, n)
        fb, tb = feats_std[idx], target_norm[idx]
        layers, m, v = step(layers, m, v, jnp.asarray(i + 1, jnp.float32), fb, tb)
        if i % log_every == 0 or i == steps - 1:
            history.append((i, float(loss(layers, fb, tb))))
    return layers, history


def build_term(layers, in_mean, in_std, out_scale,
               correct=("temperature", "specific_humidity"),
               output_cap=None, polar_taper=None, surface_taper=None,
               context_features=(),
               activation=DEFAULT_ACTIVATION) -> NNBiasCorrection:
    """Wrap trained ``layers`` and buffers into a live :class:`NNBiasCorrection`.

    Save it with ``term.save(path)`` and reload with
    :func:`jcm.physics.bias_correction.load_bias_correction`.
    ``output_cap`` passes through to the term (soft tanh bound, ``None`` =
    unbounded; train and save with the same value you will run with).
    ``polar_taper`` passes through the ``(lat0_deg, lat1_deg)`` taper edges
    (``None`` = no taper); see
    :func:`jcm.physics.bias_correction.nn_bias_correction.polar_taper_factor`.
    ``context_features`` passes through the per-column context inputs
    (``()`` = profiles only); the layers' input width must already match
    (see :func:`jcm.physics.bias_correction.nn_bias_correction.widen_first_layer`).
    ``activation`` is the hidden-layer nonlinearity NAME and must match what
    :func:`train` was given, or the saved term runs a different network than
    the one that was fitted.
    """
    return NNBiasCorrection(layers, in_mean, in_std, out_scale, tuple(correct),
                            output_cap=output_cap, polar_taper=polar_taper,
                            surface_taper=surface_taper,
                            context_features=tuple(context_features),
                            activation=activation)
