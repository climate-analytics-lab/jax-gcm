"""Online fine-tuning of the NN bias correction by backprop through rollouts.

Issue #356. The offline stage fits the network to the recorded nudge,
which teaches it what SPEEDY got wrong on nudged states but never makes it
answer for its own downstream consequences; composed with free-running SPEEDY
the offline term destabilises the model (the standard offline-training
failure, Rasp 2020). This stage closes the loop: start from a saved near-ERA5 state,
run the model N steps with the term active, compare the final state to ERA5
at that lead time, and backprop the error through the whole rollout into the
weights (stage 2 of the Rasp recipe).

Design choices, and why:

* The loss closes over a ``build_model(layers) -> Model`` callable supplied by
  the caller and rebuilds the term and Model around the ``layers`` argument
  each evaluation. That is the same pattern ``model_test.py`` uses to
  differentiate ``model.run`` w.r.t. physics ``Parameters``: the weights enter
  as a traced argument of the jitted train step, every other parameter is a
  closed-over constant, so gradients reach the network and nothing else. It
  also keeps this module dycore-agnostic (the fast tests drive it with the
  fake cubed-sphere dycore; the notebook wires up real SPEEDY).

* One rollout length per compiled train step. Each curriculum stage compiles
  once (the weights and states are traced, ``n_steps`` is baked in); the
  model's inner ``lax.scan`` steps already carry ``jax.checkpoint``, so
  backprop memory stays flat even at the longest stage.

* The loss is the area-weighted, per-level-normalized MSE on temperature and
  humidity at the final step only. Normalizing by the per-level stds from the
  the offline stats makes 1 sigma of T error and 1 sigma of q error commensurate;
  stds are floored at a fraction of their column maximum so near-zero
  variance levels (stratospheric humidity) cannot dominate the loss.

* The NaN guard lives inside the jitted step: a non-finite loss or gradient
  norm keeps the previous weights and moments (``jnp.where`` on the ``ok``
  flag) instead of poisoning the run, and the driver aborts a stage after
  ``max_bad_steps`` consecutive failures. Long-rollout fine-tuning does blow
  up occasionally; the guard turns that into a logged skip, not a corrupted
  checkpoint.

The curriculum lengths are chosen for the 30-minute T31 step against 6-hourly
ERA5 targets: 12, 24, 48 steps = 6 h, 12 h, 24 h lead times. (The project
guide sketched 1 -> 2 -> 8 -> 40 before the data cadence was pinned down;
sub-6 h horizons have no instantaneous ERA5 target to compare against.)
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from jcm.physics.bias_correction.nn_bias_correction import FIELD_ORDER
from jcm.physics.bias_correction.optim import (
    adam_init,
    adam_update,
    clip_by_global_norm,
)


# ---------------------------------------------------------------------------
# Loss ingredients
# ---------------------------------------------------------------------------


def lat_weights(coords, pole_weight_floor: float = 0.0) -> jnp.ndarray:
    """Area weights for the loss, broadcastable against ``(..., nlat)`` fields.

    cos(latitude) normalized to mean 1, so losses stay comparable across
    grids. Falls back to a scalar 1.0 when the coordinate system exposes no
    latitudes (the fake test dycore), where equal weighting is exact anyway.

    Args:
        coords: model CoordinateSystem.
        pole_weight_floor: floor applied to cos(latitude) before the mean-1
            normalization, i.e. ``max(cos(lat), floor)``. The default 0.0 is a
            no-op, so every existing caller is byte-identical. A positive floor
            (e.g. 0.3) lifts the polar rows off ~0 weight so the poles actually
            enter the loss and training reduces the bias there instead of
            ignoring it.

    """
    lats = getattr(coords.horizontal, "latitudes", None)
    if lats is None:
        return jnp.ones(())
    w = jnp.cos(jnp.asarray(lats))
    if pole_weight_floor > 0.0:
        w = jnp.maximum(w, pole_weight_floor)
    return w / jnp.mean(w)


def level_weights(spec, nlev: int) -> jnp.ndarray:
    """Vertical weights for the loss, one per model level.

    Normalized to mean 1 exactly like :func:`lat_weights`, so the loss keeps
    its magnitude across settings and a uniform spec reproduces the plain
    vertical mean. ``None`` returns a scalar 1.0, which is what the loss did
    before this existed, so every existing caller stays byte-identical.

    The vertical is the one axis the loss has never weighted. Latitude
    weighting (and its pole floor) has always been available, so every
    configuration tried so far traded near-surface accuracy against the
    mid-troposphere with no way to state a preference between them: all levels
    entered the mean equally. Passing a non-uniform spec here is that missing
    dial, e.g. weights rising toward mid-levels to protect the 500 hPa skill
    that the surface-focused runs give up.

    Host-side setup helper: ``spec`` must be concrete, not a tracer.

    Args:
        spec: per-level weights, length ``nlev``, or None for uniform.
        nlev: number of model levels, used to validate ``spec``.

    """
    if spec is None:
        return jnp.ones(())
    values = [float(x) for x in spec]
    if len(values) != nlev:
        raise ValueError(f"level weights need {nlev} values, got {len(values)}")
    if any(v < 0.0 for v in values):
        raise ValueError(f"level weights must be non-negative, got {values}")
    if sum(values) <= 0.0:
        raise ValueError("level weights must not be all zero")
    w = jnp.asarray(values)
    return w / jnp.mean(w)


def stds_from_stats(in_std, nlev: int, *, floor_frac: float = 0.05):
    """Per-level T and q stds sliced out of the offline ``in_std`` buffer.

    ``in_std`` is the feature-space vector from
    :func:`jcm.physics.bias_correction.offline_training.compute_norm_stats`,
    laid out per :data:`FIELD_ORDER`. The stds are floored at ``floor_frac``
    of their column maximum: without the floor, near-zero-variance levels
    (stratospheric humidity) get their errors amplified by 1/std and drown
    out the levels the correction actually matters for.

    Returns:
        ``(T_std, q_std)``, each ``(nlev,)``.

    """
    def slice_for(name):
        i = FIELD_ORDER.index(name)
        s = jnp.asarray(in_std[i * nlev:(i + 1) * nlev])
        return jnp.maximum(s, floor_frac * jnp.max(s))

    return slice_for("temperature"), slice_for("specific_humidity")


def state_error(pred, target_T, target_q, T_std, q_std, weights,
                level_w=None) -> jnp.ndarray:
    """Area-weighted normalized MSE on T and q at a single time.

    ``pred`` is a gridpoint :class:`PhysicsState`; targets are plain arrays in
    model-state units (T in K, q in g/kg) with the same ``(nlev, *horiz)``
    layout. Broadcasting-native: the stds reshape against the vertical axis 0
    and ``weights`` broadcasts on the trailing axes, so any horizontal layout
    works.

    ``level_w`` is the optional ``(nlev,)`` vertical weighting from
    :func:`level_weights`, reshaped here against axis 0 for the same reason the
    stds are. Both weight arrays carry mean 1 over their own axis, so their
    product does too and the loss magnitude is unchanged by either. None keeps
    the plain vertical mean.
    """
    vshape = (-1,) + (1,) * (pred.temperature.ndim - 1)
    err_T = (pred.temperature - target_T) / T_std.reshape(vshape)
    err_q = (pred.specific_humidity - target_q) / q_std.reshape(vshape)
    w = weights if level_w is None else weights * jnp.asarray(level_w).reshape(vshape)
    return 0.5 * (jnp.mean(w * err_T ** 2) + jnp.mean(w * err_q ** 2))


# ---------------------------------------------------------------------------
# Rollout loss and train step
# ---------------------------------------------------------------------------


def make_rollout_loss(build_model, forcing, *, n_steps: int,
                      time_step_minutes: float = 30.0,
                      T_std, q_std, weights, level_w=None):
    """Build ``loss(layers, init_state, target_T, target_q) -> scalar``.

    Args:
        build_model: callable ``layers -> Model`` that wires the term with
            those weights into the physics (see the module docstring for why
            the rebuild happens inside the loss).
        forcing: :class:`ForcingData` for the rollout.
        n_steps: rollout length in model steps (static; one compile each).
        time_step_minutes: must match the Model's ``time_step``.
        T_std, q_std, weights: from :func:`stds_from_stats` and
            :func:`lat_weights`.

    """
    dt_days = time_step_minutes / (60.0 * 24.0)
    # The tiny epsilon guards int(save_interval / dt) truncating N*dt/dt to
    # N-1 through float roundoff; it can never add a step.
    rollout_days = n_steps * dt_days * (1.0 + 1e-9)

    def loss(layers, init_state, target_T, target_q):
        model = build_model(layers)
        final_state, _ = model.run_from_state(
            init_state, forcing,
            save_interval=rollout_days, total_time=rollout_days)
        pred = model.dycore.to_physics_state(final_state)
        return state_error(pred, target_T, target_q, T_std, q_std, weights,
                           level_w)

    return loss


def make_climatology_loss(build_model, forcing, *, n_steps: int,
                          time_step_minutes: float = 30.0,
                          T_std, q_std, weights, level_w=None):
    """Build ``loss(layers, init_state, target_T, target_q) -> scalar`` on the
    free-running TIME-MEAN, not a short-lead snapshot.

    The rollout loss (:func:`make_rollout_loss`) only ever scores a 6-48 h
    forecast, so it cannot see the free-running model's long-run / seasonal mean
    bias (the residual winter near-surface error that short rollouts never
    penalise). This loss runs the model FREE for ``n_steps`` (no nudging target
    in ``forcing``) with ``output_averages=True`` -- which accumulates a
    differentiable running mean of the state inside the checkpointed scan, so no
    per-frame activations are stored -- and scores the area-weighted,
    std-normalised bias of that time-mean against an ERA5 time-mean over the same
    window. Same signature as the rollout loss, so :func:`make_train_step` and
    :func:`train_online` reuse it unchanged.

    Args:
        build_model: ``layers -> Model`` (the term is rebuilt inside the loss so
            the gradient reaches only its weights).
        forcing: the FREE :class:`ForcingData` (no ``nudging_target``).
        n_steps: free-run length in model steps (static; one compile each). The
            averaged window is exactly these ``n_steps`` (1440 = 30 days).
        time_step_minutes: must match the Model's ``time_step``.
        T_std, q_std: per-level stds from :func:`stds_from_stats`.
        weights: area weights from :func:`lat_weights`; pass
            ``pole_weight_floor > 0`` so the poles enter the loss.

    """
    dt_days = time_step_minutes / (60.0 * 24.0)
    horizon_days = n_steps * dt_days * (1.0 + 1e-9)   # same eps guard as the rollout loss

    def loss(layers, init_state, target_T, target_q):
        model = build_model(layers)
        # output_averages=True makes the saved frame a running time-mean that is
        # already a gridpoint PhysicsState, so -- unlike the rollout loss -- we do
        # NOT call to_physics_state again. save_interval == total_time gives one
        # frame, so index [0].
        _final, preds = model.run_from_state(
            init_state, forcing,
            save_interval=horizon_days, total_time=horizon_days,
            output_averages=True)
        mean_state = jtu.tree_map(lambda a: a[0], preds.dynamics)
        return state_error(mean_state, target_T, target_q, T_std, q_std, weights,
                           level_w)

    return loss


def make_combined_loss(build_model, forcing, *, n_rollout_steps: int,
                       n_clim_steps: int, lam: float = 1.0,
                       time_step_minutes: float = 30.0,
                       T_std, q_std, weights, level_w=None):
    """Build ``loss(layers, init_state, target_T, target_q)`` = rollout + lam*clim.

    The two objectives pull in complementary directions: the short-lead rollout
    (:func:`make_rollout_loss`) is what earns the free-running model's aloft /
    humidity skill (a term that keeps the 6-24 h forecast accurate keeps the
    500 hPa and humidity climatology unbiased), while the free-running time-mean
    (:func:`make_climatology_loss`) is what corrects the long-run near-surface /
    seasonal bias that short rollouts never see. A pure climatology loss fixes
    the surface but overshoots the mid-troposphere (it can only warm the cold
    surface by warming the column); adding the rollout term penalises exactly
    that 500 hPa drift, so the network must fix the surface WITHOUT wrecking
    aloft.

    Both terms roll the SAME ``init_state`` forward under the SAME free
    ``forcing`` -- one ``build_model(layers)``, two ``run_from_state`` calls (a
    short forecast and the long averaged free run). Targets are PACKED as
    ``target_T = (rollout_T, clim_T)`` and ``target_q = (rollout_q, clim_q)`` so
    :func:`make_train_step` and its NaN guard are reused byte-for-byte -- the
    step never inspects the targets, it only threads them into the loss.

    Args:
        build_model: ``layers -> Model`` (rebuilt inside so the gradient reaches
            only the network weights).
        forcing: the FREE :class:`ForcingData` (no ``nudging_target``); shared by
            both rollouts.
        n_rollout_steps: short-lead forecast length in model steps (e.g. 48 =
            24 h at the 30-minute step). Its ERA5 target is the state at that
            lead, a single snapshot.
        n_clim_steps: free-run length whose time-mean is scored (e.g. 4320 =
            90 days). Its ERA5 target is the window mean.
        lam: weight on the climatology term. Larger pushes the surface harder;
            smaller protects the rollout-earned aloft/humidity skill more.
        time_step_minutes: must match the Model's ``time_step``.
        T_std, q_std, weights: shared loss ingredients (:func:`stds_from_stats`,
            :func:`lat_weights`); pass ``pole_weight_floor > 0`` in ``weights``.

    """
    dt_days = time_step_minutes / (60.0 * 24.0)
    # Same eps guard as the rollout / climatology losses (see make_rollout_loss).
    rollout_days = n_rollout_steps * dt_days * (1.0 + 1e-9)
    horizon_days = n_clim_steps * dt_days * (1.0 + 1e-9)

    def loss(layers, init_state, target_T, target_q):
        roll_T, clim_T = target_T
        roll_q, clim_q = target_q
        model = build_model(layers)

        # short-lead forecast term (keeps aloft/humidity skill)
        final_state, _ = model.run_from_state(
            init_state, forcing,
            save_interval=rollout_days, total_time=rollout_days)
        roll_pred = model.dycore.to_physics_state(final_state)
        roll_err = state_error(roll_pred, roll_T, roll_q, T_std, q_std, weights,
                               level_w)

        # free-running time-mean term (corrects long-run surface/seasonal bias)
        _final, preds = model.run_from_state(
            init_state, forcing,
            save_interval=horizon_days, total_time=horizon_days,
            output_averages=True)
        mean_state = jtu.tree_map(lambda a: a[0], preds.dynamics)
        clim_err = state_error(mean_state, clim_T, clim_q, T_std, q_std, weights,
                               level_w)

        return roll_err + lam * clim_err

    return loss


def make_train_step(loss_fn, *, lr: float, clip_norm: float):
    """Jitted Adam step over ``loss_fn`` with clipping and a NaN guard.

    Returns ``step(layers, m, v, t, init_state, target_T, target_q) ->
    (layers, m, v, loss, grad_norm, ok)``. When the loss or gradient norm is
    non-finite, ``ok`` is False and the weights and moments pass through
    unchanged, so one bad rollout cannot corrupt the training state.
    """

    @jax.jit
    def step(layers, m, v, t, init_state, target_T, target_q):
        loss, grads = jax.value_and_grad(loss_fn)(
            layers, init_state, target_T, target_q)
        grads, grad_norm = clip_by_global_norm(grads, clip_norm)
        new_layers, new_m, new_v = adam_update(layers, grads, m, v, t, lr)
        ok = jnp.isfinite(loss) & jnp.isfinite(grad_norm)

        def keep(new, old):
            return jtu.tree_map(lambda a, b: jnp.where(ok, a, b), new, old)

        return (keep(new_layers, layers), keep(new_m, m), keep(new_v, v),
                loss, grad_norm, ok)

    return step


# ---------------------------------------------------------------------------
# Curriculum driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CurriculumStage:
    """One stage of the rollout-length curriculum (static config)."""

    n_steps: int          # rollout length in model steps
    updates: int          # Adam updates in this stage
    lr: float = 1e-4      # lower than the offline 1e-3: this is fine-tuning
    clip_norm: float = 1.0


def default_curriculum() -> tuple[CurriculumStage, ...]:
    """6 h -> 12 h -> 24 h leads at the 30-minute T31 step."""
    return (
        CurriculumStage(n_steps=12, updates=300),
        CurriculumStage(n_steps=24, updates=300),
        CurriculumStage(n_steps=48, updates=600),
    )


def train_online(layers, build_model, forcing, samples, stages=None, *,
                 T_std, q_std, weights, level_w=None,
                 time_step_minutes: float = 30.0,
                 key=None, max_bad_steps: int = 5, stage_end_hook=None):
    """Fine-tune ``layers`` through the rollout-length curriculum.

    Args:
        layers: initial MLP weights, normally the offline term's (warm
            start). A zero-init output layer has zero gradient, same caveat
            as the offline trainer.
        build_model: callable ``layers -> Model`` (see
            :func:`make_rollout_loss`).
        forcing: :class:`ForcingData` shared by every rollout.
        samples: mapping ``n_steps -> (init_states, target_T, target_q)``
            where ``init_states`` is a dycore-state pytree stacked on a
            leading sample axis and the targets are
            ``(n_samples, nlev, *horiz)`` arrays in model-state units. Each
            stage draws from ``samples[stage.n_steps]``.
        stages: curriculum, default :func:`default_curriculum`.
        T_std, q_std, weights: loss ingredients (:func:`stds_from_stats`,
            :func:`lat_weights`).
        level_w: optional vertical weighting from :func:`level_weights`;
            None weights every level equally.
        time_step_minutes: must match the Model's ``time_step``.
        key: PRNG key for sample shuffling.
        max_bad_steps: consecutive non-finite steps before a stage aborts.
            The guarded step already kept the last good weights; aborting
            just stops wasting rollouts (shorten the stage or lower its lr
            and rerun).
        stage_end_hook: optional ``hook(stage_idx, stage, layers)`` called
            after each stage finishes (host-side, outside jit). The place
            for held-out validation losses and free-run probes between
            stages; its cost is the caller's business, training state is
            untouched.

    Returns:
        ``(layers, history)`` where ``history`` is a list of dicts with keys
        ``stage``, ``update``, ``n_steps``, ``loss``, ``grad_norm``, ``ok``.

    Notes:
        Batching is one start time per update (shuffled each pass): with
        Adam smoothing over updates this is plain SGD over snapshots, keeps
        memory flat, and compiles exactly once per stage. Adam moments and
        the step counter carry across stages, one continuous fine-tune.

    """
    if stages is None:
        stages = default_curriculum()
    if key is None:
        key = jax.random.key(0)

    m, v = adam_init(layers)
    t = 0
    history = []

    for stage_idx, stage in enumerate(stages):
        init_states, target_T, target_q = samples[stage.n_steps]
        n_samples = int(target_T.shape[0])
        loss_fn = make_rollout_loss(
            build_model, forcing, n_steps=stage.n_steps,
            time_step_minutes=time_step_minutes,
            T_std=T_std, q_std=q_std, weights=weights, level_w=level_w)
        step = make_train_step(loss_fn, lr=stage.lr, clip_norm=stage.clip_norm)

        bad_streak = 0
        order = jnp.arange(n_samples)
        for u in range(stage.updates):
            if u % n_samples == 0:
                key, sub = jax.random.split(key)
                order = jax.random.permutation(sub, n_samples)
            i = int(order[u % n_samples])
            init_i = jtu.tree_map(lambda a: a[i], init_states)
            t += 1
            layers, m, v, loss, grad_norm, ok = step(
                layers, m, v, jnp.asarray(float(t)),
                init_i, target_T[i], target_q[i])
            ok = bool(ok)
            history.append({
                "stage": stage_idx, "update": u, "n_steps": stage.n_steps,
                "loss": float(loss), "grad_norm": float(grad_norm), "ok": ok,
            })
            bad_streak = 0 if ok else bad_streak + 1
            if bad_streak >= max_bad_steps:
                # Weights are still the last good ones (the guard kept them).
                break

        if stage_end_hook is not None:
            stage_end_hook(stage_idx, stage, layers)

    return layers, history
