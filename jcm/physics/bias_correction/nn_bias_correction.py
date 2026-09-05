"""State-dependent neural-network bias correction as a composable PhysicsTerm.

Issue #356. SPEEDY's tendencies are systematically wrong (see the M0 bias
maps). This term adds a learned correction so the running model looks more
like ERA5:

    dX/dt_total = dX/dt_dynamics + dX/dt_SPEEDY + N_theta(X)

``N_theta`` is a small MLP applied independently to each atmospheric column.
Its input is that column's profiles of (temperature, specific humidity, u, v);
its output is a per-second correction tendency for the chosen variables. This
is the single-column ("local") version. Non-local inputs (a column seeing its
neighbours) are a later stretch goal and would only change how ``feats`` is
assembled in :func:`bias_correction_tendency`.

Design choices, and why:

* Two-piece split, mirroring ``jcm/nudging.py``: a pure
  :func:`bias_correction_tendency` that takes plain arrays plus weights and
  returns a :class:`PhysicsTendency`, and a thin :class:`NNBiasCorrection`
  wrapper. The pure helper has no Model dependency so tests call it directly.

* The network is the *learned* analog of nudging's fixed ``(X_ref - X)/tau``.
  Nudging needs ERA5 every step; this term reads only ``state``. That is the
  whole point of issue #356: a correction you can run without the truth.

* The output layer is zero-initialised, so a freshly constructed term outputs
  exactly zero and adding it to SPEEDY changes nothing. This is a deliberate
  safety choice: the term cannot destabilise the model before it has learned
  anything. Training moves the weights off zero. The gradient still flows
  (the weights are wired to the output), which is what makes the term
  trainable despite being a no-op at construction.

* Weights live in ``nnx.Param`` so ``jax.grad`` / optimizers see them.
  Frozen normalisation buffers live in ``nnx.Variable``. Storing a tunable as
  a plain attribute would hide it from gradients.

* Broadcasting-native, vertical on axis 0. The helper reshapes the horizontal
  to a flat column axis via ``state.temperature.shape``, so the identical code
  runs on a single ``(kx,)`` column, a ``(kx, ncols)`` block, or the whole
  ``(kx, nlon, nlat)`` grid. SPEEDY hands this term the 3D grid
  (``vectorize_columns=False``); the reshape round-trip is shape-agnostic.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tree_math
from flax import nnx

from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.radiation.speedy_shortwave import solar
from jcm.physics.speedy.physical_constants import solc
from jcm.physics.speedy.speedy_coords import SpeedyCoords
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

# The prognostic profiles fed in and corrected out, in this fixed order.
# Shared with the offline training pipeline so both build identical features.
FIELD_ORDER = ("temperature", "specific_humidity", "u_wind", "v_wind")
_N_FIELDS = len(FIELD_ORDER)


# ---------------------------------------------------------------------------
# Neural network: pure functions + tree_math weight leaves
# ---------------------------------------------------------------------------


@tree_math.struct
class DenseWeights:
    """One dense layer's weights as a JAX pytree (so gradients flow through)."""

    kernel: jnp.ndarray  # (in, out)
    bias: jnp.ndarray    # (out,)


def dense(x: jnp.ndarray, w: DenseWeights, activation=None) -> jnp.ndarray:
    """Affine layer ``x @ kernel + bias`` with optional activation.

    Works on a single feature vector ``(in,)`` or a batch ``(..., in)`` by
    numpy broadcasting of the matmul.
    """
    y = x @ w.kernel + w.bias
    return activation(y) if activation is not None else y


def mlp(x: jnp.ndarray, layers: Sequence[DenseWeights],
        activation=jax.nn.tanh) -> jnp.ndarray:
    """Plain MLP on one feature vector. Hidden layers squash, last is linear.

    Pure function. ``layers`` is an ordered sequence of :class:`DenseWeights`;
    the last one is the linear output layer (no activation) so the network can
    represent corrections of either sign and any magnitude.
    """
    for w in layers[:-1]:
        x = dense(x, w, activation)
    return dense(x, layers[-1])


def init_mlp(key, sizes: Sequence[int], *,
             zero_last_layer: bool = True) -> tuple[DenseWeights, ...]:
    """Glorot-initialise an MLP, optionally zeroing the output layer.

    Args:
        key: PRNG key.
        sizes: layer sizes ``(n_in, *hidden, n_out)``.
        zero_last_layer: if True (default), the final layer's kernel and bias
            are set to zero so the network is a no-op at construction. Set
            False only to exercise gradient flow in tests (a zero-output layer
            has zero gradient by construction, which would mask wiring bugs).

    Returns:
        A tuple of :class:`DenseWeights`, one per layer.

    """
    layers = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        # Glorot/Xavier scale keeps activations from vanishing or exploding
        # across layers; bias starts at zero.
        scale = jnp.sqrt(2.0 / (din + dout))
        layers.append(
            DenseWeights(jax.random.normal(k, (din, dout)) * scale,
                         jnp.zeros(dout))
        )
    if zero_last_layer:
        last = layers[-1]
        layers[-1] = DenseWeights(jnp.zeros_like(last.kernel),
                                  jnp.zeros_like(last.bias))
    return tuple(layers)


# ---------------------------------------------------------------------------
# Feature assembly (shared by the live term and the offline trainer)
# ---------------------------------------------------------------------------


def state_to_features(state) -> jnp.ndarray:
    """Flatten a ``(nlev, *horiz)`` state into ``(ncols, 4*nlev)`` rows.

    One row per column, fields concatenated in :data:`FIELD_ORDER`. The
    horizontal is flattened C-order, so this runs unchanged on a single
    column, a column block, or a full grid. The offline training pipeline
    reuses it so the trained network sees the same feature layout the live
    term feeds it.
    """
    nlev = state.temperature.shape[0]
    cols = [getattr(state, name).reshape(nlev, -1).T for name in FIELD_ORDER]
    return jnp.concatenate(cols, axis=1)


# Per-column context features the network can read alongside the profiles.
# Each is pre-normalised to O(1) here, so the in_mean/in_std buffers (which
# cover only the 4*nlev profile block) never need to change.
#
# APPEND-ONLY. A term's feature order is re-derived by filtering this tuple
# (training drivers do this when they add a feature to an existing term; see
# `remap_first_layer`), and `widen_first_layer` can only add kernel rows at the
# bottom. Inserting a name in the middle would reorder the features of an
# already-trained term without moving its weights, silently pairing each input
# with the wrong row.
CONTEXT_FEATURES = ("fmask", "orog", "sin_lat", "cos_lat", "ps", "insol",
                    "sice", "snowc", "stl", "soilw")

# Hidden-layer nonlinearities, keyed by name because a saved term stores the
# NAME: a .npz cannot hold a Python callable, and a term whose activation is
# implicit would silently run under whatever the default happened to be when it
# was reloaded. `tanh` is the shipped default and every artifact predating this
# registry is tanh by construction.
#
# gelu fits the offline target better than tanh (held-out MSE ratio 0.3087
# against 0.3422 at 512 wide) but scored worse on the free-run climate metrics,
# so the offline error did not select it and tanh stays the default. Inputs
# must be standardised before the first tanh: raw T ~ 250 saturates it exactly
# (tanh' == 0 in float32) and every first-layer gradient vanishes.
ACTIVATIONS = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
    "softplus": jax.nn.softplus,
}
DEFAULT_ACTIVATION = "tanh"



# Time-varying SURFACE STATE, read from the model's own boundary conditions.
# ERA5 itself is not available in a free run, so these are the runnable form of
# "give the network the land fields": the same quantities, from the forcing the
# model already carries. Each maps to (forcing field, scale, offset) chosen to
# land the feature near O(1) with roughly zero mean, as the others are.
#
# `sice` was built as the time-varying counterpart to `fmask`: a static land
# mask calls winter sea ice "ocean" exactly where the DJF bias lives, while
# sea-ice fraction says it is frozen. Tested at 148k parameters it beat plain
# SPEEDY on all five metrics but did not beat the insolation term on DJF
# (3.44 against 3.32 K on the 2016-2022 holdout), so it ships as an available
# input, not as the recommended one.
_SURFACE_FEATURES = {
    # fraction in [0, 1] -> centred
    "sice": ("sice_am", 1.0, -0.5),
    # snow depth in mm; 100 mm is a deep pack, so this saturates sensibly
    "snowc": ("snowc_am", 1.0 / 100.0, -0.5),
    # land surface temperature in K, centred on freezing and scaled by a
    # typical seasonal swing
    "stl": ("stl_am", 1.0 / 30.0, -273.15 / 30.0),
    # soil wetness, already O(1)
    "soilw": ("soilw_am", 1.0, -0.5),
}


def build_context(state, features: tuple[str, ...], terrain=None,
                  lat: jnp.ndarray | None = None,
                  speedy_coords=None,
                  tyear: jnp.ndarray | None = None,
                  forcing=None) -> jnp.ndarray:
    """Assemble per-column context features as ``(ncols, len(features))``.

    The near-surface temperature pattern is set by things a single column
    cannot see from its own profiles: land vs ocean, terrain height,
    latitude, and what time of year it is. These features hand the per-column
    MLP that missing context. Each is normalised to O(1) at build time:

    * ``fmask``: fractional land-sea mask, centred (``terrain.fmask - 0.5``).
    * ``orog``: mean orography in km-ish units (``terrain.orog / 3000``).
    * ``sin_lat`` / ``cos_lat``: from ``lat`` (radians, broadcast over the
      horizontal), a smooth position encoding.
    * ``ps``: ``state.normalized_surface_pressure - 1`` (terrain-driven
      surface-pressure deficit).
    * ``insol``: daily-mean top-of-atmosphere insolation as ``topsr/solc - 1``.
      ``solc`` is the area-averaged solar input, so the feature has global
      annual mean zero by construction and lands in about [-1, +0.6]. This is
      the only *time-varying* feature. It is a function of latitude and time of
      year together, so unlike a bare day-of-year phase (a global scalar) it
      separates northern winter from southern summer, and it saturates at -1
      through the polar night, which is the regime the winter-over-land bias
      lives in. Computed by
      :func:`jcm.physics.radiation.speedy_shortwave.solar`, reused rather than
      re-derived so the two cannot drift apart (that function already clips its
      ``arccos`` argument to keep gradients finite).

    Fields broadcast numpy-style against the state's horizontal shape and are
    flattened C-order, matching :func:`state_to_features` column order.

    Args:
        state: gridpoint :class:`PhysicsState` (sets the horizontal shape).
        features: names to build, a sub-sequence of :data:`CONTEXT_FEATURES`.
        terrain: :class:`TerrainData`, needed by ``fmask`` / ``orog``.
        lat: latitudes in radians, needed by ``sin_lat`` / ``cos_lat``.
        speedy_coords: :class:`SpeedyCoords`, needed by ``insol``.
        tyear: fraction of the year in [0, 1), needed by ``insol``. Read from
            ``forcing.solar.tyear``, which the model repopulates every step.
        forcing: :class:`ForcingData`, needed by the surface-state features in
            :data:`_SURFACE_FEATURES` (``sice``, ``snowc``, ``stl``,
            ``soilw``). Like ``insol`` these vary in time, because
            ``ForcingData.select`` repopulates them every step.

    """
    horiz = state.temperature.shape[1:]

    def col(x):
        return jnp.broadcast_to(jnp.asarray(x), horiz).reshape(-1)

    cols = []
    for name in features:
        if name == "fmask":
            cols.append(col(terrain.fmask - 0.5))
        elif name == "orog":
            cols.append(col(terrain.orog / 3000.0))
        elif name == "sin_lat":
            cols.append(col(jnp.sin(lat)))
        elif name == "cos_lat":
            cols.append(col(jnp.cos(lat)))
        elif name == "ps":
            cols.append(col(state.normalized_surface_pressure - 1.0))
        elif name == "insol":
            if tyear is None or speedy_coords is None:
                raise ValueError(
                    "context feature 'insol' needs `tyear` and `speedy_coords`")
            cols.append(col(solar(tyear, speedy_coords) / solc - 1.0))
        elif name in _SURFACE_FEATURES:
            if forcing is None:
                raise ValueError(
                    f"context feature {name!r} needs `forcing`")
            field, scale, offset = _SURFACE_FEATURES[name]
            cols.append(col(jnp.asarray(getattr(forcing, field)) * scale
                            + offset))
        else:
            raise ValueError(f"unknown context feature: {name!r}")
    return jnp.stack(cols, axis=1)


def widen_first_layer(layers: Sequence[DenseWeights],
                      n_new: int) -> tuple[DenseWeights, ...]:
    """Widen the input layer by ``n_new`` zero-weighted context inputs.

    Context features are appended *after* the profile block, so the new
    kernel rows go at the bottom. Zero rows make the widened network
    bit-identical to the original until training moves them (the same
    safety idea as the zero-initialised output layer), while gradients still
    flow into the new rows because the context inputs are non-zero.
    """
    first = layers[0]
    pad = jnp.zeros((n_new, first.kernel.shape[1]), first.kernel.dtype)
    widened = DenseWeights(jnp.concatenate([first.kernel, pad], axis=0),
                           first.bias)
    return (widened,) + tuple(layers[1:])


def remap_first_layer(layers: Sequence[DenseWeights], n_profile: int,
                      old_names: Sequence[str],
                      new_names: Sequence[str]) -> tuple[DenseWeights, ...]:
    """Move each named context row to its position in ``new_names``.

    :func:`widen_first_layer` can only append at the bottom, which is correct
    only when the new layout is the old one plus a suffix. Context order is
    re-derived by filtering :data:`CONTEXT_FEATURES`, so a feature that sorts
    EARLIER in the registry than one already present shifts every later row:
    warm-starting ``("insol",)`` with ``--context fmask`` gives the layout
    ``("fmask", "insol")`` while the trained ``insol`` row is still sitting at
    the first context slot, silently feeding it to ``fmask``. Keying the copy
    by name instead of position makes any layout change safe, including
    reorders and inserts.

    Rows named in ``new_names`` but not ``old_names`` are left at zero, so the
    result is still bit-identical to the warm start until training moves them.
    """
    first = layers[0]
    if first.kernel.shape[0] != n_profile + len(old_names):
        raise ValueError(
            f"first kernel has {first.kernel.shape[0]} rows, expected "
            f"{n_profile} profile + {len(old_names)} context "
            f"{tuple(old_names)}")

    kernel = jnp.zeros((n_profile + len(new_names), first.kernel.shape[1]),
                       first.kernel.dtype)
    kernel = kernel.at[:n_profile].set(first.kernel[:n_profile])
    for i, name in enumerate(old_names):
        if name in new_names:
            kernel = kernel.at[n_profile + new_names.index(name)].set(
                first.kernel[n_profile + i])
    return (DenseWeights(kernel, first.bias),) + tuple(layers[1:])


def polar_taper_factor(latitudes_rad: jnp.ndarray,
                       lat0_deg: float, lat1_deg: float) -> jnp.ndarray:
    """Latitude taper: 1 equatorward of ``lat0``, 0 poleward of ``lat1``.

    A C2-smooth (smootherstep) ramp on ``|lat|`` between the two edge
    latitudes, symmetric between hemispheres. Multiplying the applied
    correction by this factor switches the network off near the poles, where
    it is systematically wrong (the free-running polar warm band), while
    leaving the correction untouched in the well-constrained mid-latitudes and
    tropics. The C2 smoothness (zero first and second derivative at both edges)
    avoids a kink in the applied tendency at the taper latitude.

    Args:
        latitudes_rad: latitudes in radians, any shape. SPEEDY passes a
            ``(nlat,)`` row; a per-gridpoint ``(nlon, nlat)`` also broadcasts.
        lat0_deg: full-strength edge, degrees. ``|lat| <= lat0`` -> factor 1.
        lat1_deg: zero edge, degrees. ``|lat| >= lat1`` -> factor 0.

    Returns:
        A taper factor in ``[0, 1]`` with the same shape as ``latitudes_rad``.

    """
    lo = jnp.deg2rad(lat0_deg)
    hi = jnp.deg2rad(lat1_deg)
    t = jnp.clip((jnp.abs(latitudes_rad) - lo) / (hi - lo), 0.0, 1.0)
    smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # smootherstep
    return 1.0 - smooth


def surface_taper_factor(sigma: jnp.ndarray,
                         sigma0: float, sigma1: float) -> jnp.ndarray:
    """Vertical taper: 1 aloft (``sigma <= sigma0``), 0 near surface (``>= sigma1``).

    The vertical mirror of :func:`polar_taper_factor`, on the model's sigma
    coordinate instead of latitude, used to fade only the *temperature*
    correction near the ground. The near-surface temperature correction is the
    one piece that hurts (it drives the free-running polar warm band and a
    milder mid-latitude cold bias), while the correction aloft (mid-troposphere
    temperature) and the humidity correction both help. Tapering the T output
    to zero over the near-surface levels lets those levels revert to plain
    SPEEDY, which is accurate there, without touching the beneficial
    corrections. C2-smooth (smootherstep) so the applied tendency has no
    vertical kink.

    Args:
        sigma: level centres in sigma (~0 at the model top, ~1 at the surface),
            shape ``(nlev,)``.
        sigma0: full-strength edge. ``sigma <= sigma0`` -> factor 1 (aloft).
        sigma1: zero edge. ``sigma >= sigma1`` -> factor 0 (surface).

    Returns:
        A per-level factor in ``[0, 1]`` with the same shape as ``sigma``.

    """
    t = jnp.clip((sigma - sigma0) / (sigma1 - sigma0), 0.0, 1.0)
    smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # smootherstep
    return 1.0 - smooth


# ---------------------------------------------------------------------------
# Pure tendency helper (callable without a Model, mirrors nudging_tendency)
# ---------------------------------------------------------------------------


def bias_correction_tendency(
    state: PhysicsState,
    layers: Sequence[DenseWeights],
    in_mean: jnp.ndarray,
    in_std: jnp.ndarray,
    out_scale: jnp.ndarray,
    correct: tuple[str, ...] = ("temperature", "specific_humidity"),
    output_cap: float | None = None,
    taper: jnp.ndarray | None = None,
    t_level_weight: jnp.ndarray | None = None,
    context: jnp.ndarray | None = None,
    activation=None,
) -> PhysicsTendency:
    """State-dependent correction tendency, computed per column.

    The horizontal is flattened to a column axis, the MLP runs once per
    column via ``vmap``, and the result is reshaped back to the input layout.
    Fields not named in ``correct`` get an exact-zero tendency.

    Args:
        state: current gridpoint state. The vertical is axis 0; any trailing
            axes are horizontal and are flattened/restored via
            ``state.temperature.shape``.
        layers: MLP weights from :func:`init_mlp`.
        in_mean, in_std: per-feature input standardisation (length
            ``4 * nlev``). ``make_bias_correction`` seeds these with
            zeros/ones placeholders;
            ``offline_training.compute_norm_stats`` computes the real
            climatological values.
        out_scale: per-output scale (length ``4 * nlev``) mapping the
            dimensionless network output to per-second tendency units.
        correct: which prognostic variables receive a correction. Others are
            returned as zeros.
        output_cap: optional soft bound on the dimensionless output, in
            units of ``out_scale`` (roughly "sigmas of the training
            target"). ``None`` (default) leaves the linear head unbounded.
            Set it to tame off-distribution extrapolation: the tanh
            saturates the tail smoothly, so unlike a hard clip the gradient
            survives near the bound. Static config, part of the saved term.
        taper: optional latitude taper factor (see
            :func:`polar_taper_factor`) broadcast against the trailing
            horizontal axis of each corrected field. ``None`` (default) applies
            no taper. Only meaningful on the live 3D grid, where latitude is
            the trailing axis; ignored on a bare column (no horizontal axis).
        context: optional ``(ncols, n_ctx)`` per-column context features from
            :func:`build_context`, already O(1)-normalised. Appended to the
            standardised profile features, so the MLP's input layer must be
            ``4*nlev + n_ctx`` wide (see :func:`widen_first_layer`).

    Returns:
        A :class:`PhysicsTendency` in per-second units.

    """
    feats = state_to_features(state)  # (ncols, 4*nlev), fields in FIELD_ORDER

    # Standardise inputs: T ~ 250 and q ~ 1e-3 differ by orders of magnitude,
    # so an MLP cannot learn on the raw values. (No-op while in_mean/in_std are
    # the zeros/ones placeholders.)
    feats = (feats - in_mean) / in_std
    if context is not None:
        # Context is pre-normalised in build_context, so it bypasses the
        # profile-only in_mean/in_std buffers.
        feats = jnp.concatenate([feats, context], axis=1)

    act = ACTIVATIONS[DEFAULT_ACTIVATION] if activation is None else activation
    out = jax.vmap(lambda f: mlp(f, layers, act))(feats)  # (ncols, 4*nlev), dimensionless
    if output_cap is not None:
        # Static Python branch: output_cap is trace-time config, not data.
        out = output_cap * jnp.tanh(out / output_cap)
    out = out * out_scale                            # -> per-second units

    dT, dq, du, dv = jnp.split(out, _N_FIELDS, axis=1)  # each (ncols, nlev)
    if t_level_weight is not None:
        # Vertical taper: fade the temperature correction by level (only the
        # near-surface T correction is harmful). The (nlev,) factor broadcasts
        # over the (ncols, nlev) block; humidity and wind outputs are untouched.
        dT = dT * t_level_weight

    def to_grid(a: jnp.ndarray) -> jnp.ndarray:
        # (ncols, nlev) -> (nlev, *horiz), inverse of to_columns.
        g = a.T.reshape(state.temperature.shape)
        # Optional polar taper: a (nlat,) factor broadcasts on the trailing
        # horizontal axis, scaling the applied correction to zero near the
        # poles. Skipped on a bare (nlev,) column, which has no horizontal
        # axis for a latitude factor to act on (static shape branch,
        # jit-safe). On a flattened (nlev, ncols) block the taper is only
        # meaningful if the trailing axis really is latitude; the live
        # SPEEDY path always hands this term the full 3D grid.
        if taper is None or g.ndim == 1:
            return g
        return g * taper

    zero = jnp.zeros_like(state.temperature)
    return PhysicsTendency(
        u_wind=to_grid(du) if "u_wind" in correct else zero,
        v_wind=to_grid(dv) if "v_wind" in correct else zero,
        temperature=to_grid(dT) if "temperature" in correct else zero,
        specific_humidity=to_grid(dq) if "specific_humidity" in correct else zero,
        # SPEEDY carries no extra tracers; emit a matching (empty) tracer dict
        # so the summed PhysicsTendency pytree structure stays consistent.
        tracers={name: jnp.zeros_like(t) for name, t in state.tracers.items()},
    )


# ---------------------------------------------------------------------------
# Composable PhysicsTerm wrapper
# ---------------------------------------------------------------------------


class NNBiasCorrection(PhysicsTerm):
    """State-dependent NN bias correction added on top of the SPEEDY tendencies.

    Single-column: each column's correction depends only on that column's
    profiles of (T, q, u, v). ``requires``/``provides`` are empty - the term
    reads only the prognostic state and writes only a tendency, so it composes
    anywhere in the stack (append it last):

        physics = speedy_physics() + make_bias_correction(coords)

    Zero-initialised, it is a no-op until trained.
    """

    name: ClassVar[str] = "nn_bias_correction"
    category: ClassVar[str] = "bias_correction"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, layers: Sequence[DenseWeights],
                 in_mean: jnp.ndarray, in_std: jnp.ndarray,
                 out_scale: jnp.ndarray,
                 correct: tuple[str, ...] = ("temperature", "specific_humidity"),
                 output_cap: float | None = None,
                 polar_taper: tuple[float, float] | None = None,
                 surface_taper: tuple[float, float] | None = None,
                 context_features: tuple[str, ...] = (),
                 activation: str = DEFAULT_ACTIVATION):
        """Hold trainable weights and frozen normalisation buffers.

        Args:
            layers: MLP weights (trainable). Stored in ``nnx.Param``. With
                ``context_features``, the input layer must be
                ``4*nlev + len(context_features)`` wide (see
                :func:`widen_first_layer`).
            in_mean, in_std, out_scale: normalisation buffers (frozen).
                Stored in ``nnx.Variable``. They cover only the profile block.
            correct: prognostic variables to correct (static config).
            output_cap: optional soft tanh bound on the dimensionless
                output (static config; see
                :func:`bias_correction_tendency`). ``None`` = unbounded.
            polar_taper: optional ``(lat0_deg, lat1_deg)`` edges for a latitude
                taper that scales the applied correction to zero poleward of
                ``lat1`` (see :func:`polar_taper_factor`). ``None`` = no taper.
                Static config; the ``(nlat,)`` factor itself is precomputed in
                :meth:`cache_coords`.
            context_features: per-column context inputs appended after the
                standardised profiles (see :func:`build_context`). ``()`` =
                profiles only (static config).

        """
        self.weights = nnx.Param(tuple(layers))   # trainable leaves
        self.in_mean = nnx.Variable(in_mean)      # frozen buffers
        self.in_std = nnx.Variable(in_std)
        self.out_scale = nnx.Variable(out_scale)
        self.correct = tuple(correct)             # static config
        self.output_cap = None if output_cap is None else float(output_cap)
        self.polar_taper = (None if polar_taper is None
                            else (float(polar_taper[0]), float(polar_taper[1])))
        self.surface_taper = (None if surface_taper is None
                              else (float(surface_taper[0]), float(surface_taper[1])))
        self.context_features = tuple(context_features)
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; "
                             f"choose from {sorted(ACTIVATIONS)}")
        # Stored as a name so `save` can round-trip it; resolved to the callable
        # once here rather than per call.
        self.activation = activation
        self._activation_fn = ACTIVATIONS[activation]
        # The (nlat,) taper/latitude buffers are created only in cache_coords
        # (mirroring HeldSuarezPhysics): pre-declaring them here as None would
        # pin them as static nnx attributes and reject the later nnx.Variable
        # assignment.

    def cache_coords(self, coords) -> None:
        """Precompute the polar taper / latitudes (no-op if neither is used).

        Called once by ``ComposablePhysics.cache_coords`` (via
        ``Model.__init__``), outside jit, on both the training and eval paths.
        The buffers are frozen ``nnx.Variable``s so the online trainer, which
        differentiates only w.r.t. the weights, never touches them.
        """
        needs_lat = any(f in ("sin_lat", "cos_lat")
                        for f in self.context_features)
        if self.polar_taper is not None or needs_lat:
            lat = jnp.asarray(coords.horizontal.latitudes)  # radians
            if self.polar_taper is not None:
                self._taper = nnx.Variable(
                    polar_taper_factor(lat, *self.polar_taper))
            if needs_lat:
                self._lat = nnx.Variable(lat)
        if self.surface_taper is not None:
            sigma = jnp.asarray(coords.vertical.centers)
            self._surface_taper = nnx.Variable(
                surface_taper_factor(sigma, *self.surface_taper))
        if "insol" in self.context_features:
            # `solar` reads only .sia / .coa off this, but caching the whole
            # struct follows the SpeedyPhysics pattern (speedy_terms.py:163)
            # and keeps it a pytree the jitted `solar` accepts.
            self._speedy_coords = nnx.Variable(
                SpeedyCoords.from_coordinate_system(coords))

    def _taper_value(self) -> jnp.ndarray | None:
        """Return the cached taper factor, or None if unset / not yet cached."""
        if self.polar_taper is None or not hasattr(self, "_taper"):
            return None
        return self._taper.get_value()

    def _lat_value(self) -> jnp.ndarray | None:
        """Return the cached latitudes, or None if not needed / not yet cached."""
        if not hasattr(self, "_lat"):
            return None
        return self._lat.get_value()

    def _surface_taper_value(self) -> jnp.ndarray | None:
        """Return the cached ``(nlev,)`` vertical temperature taper, or None if unset."""
        if self.surface_taper is None or not hasattr(self, "_surface_taper"):
            return None
        return self._surface_taper.get_value()

    def _speedy_coords_value(self):
        """Return the cached SpeedyCoords, or None if ``insol`` is not in use."""
        if not hasattr(self, "_speedy_coords"):
            return None
        return self._speedy_coords.get_value()

    def __call__(self, state: PhysicsState, diagnostics: dict,
                 forcing: ForcingData, terrain: TerrainData):
        """Return the correction tendency; pass diagnostics through untouched."""
        context = None
        if self.context_features:
            # `forcing.solar` is repopulated every step by ForcingData.select,
            # so this is what makes `insol` time-varying. Only read it when the
            # term actually asked for it, so context-less terms and the test
            # doubles never need a populated forcing.
            tyear = (forcing.solar.tyear
                     if "insol" in self.context_features else None)
            # Same reasoning for the surface-state features: only reach into
            # `forcing` when one was actually requested, so context-less terms
            # and the test doubles never need a populated forcing.
            surface_forcing = (
                forcing if any(f in _SURFACE_FEATURES
                               for f in self.context_features) else None)
            context = build_context(state, self.context_features,
                                    terrain=terrain, lat=self._lat_value(),
                                    speedy_coords=self._speedy_coords_value(),
                                    tyear=tyear, forcing=surface_forcing)
        tend = bias_correction_tendency(
            state,
            self.weights.get_value(),
            self.in_mean.get_value(),
            self.in_std.get_value(),
            self.out_scale.get_value(),
            self.correct,
            output_cap=self.output_cap,
            taper=self._taper_value(),
            t_level_weight=self._surface_taper_value(),
            context=context,
            activation=self._activation_fn,
        )
        return tend, diagnostics

    def config(self) -> dict:
        """Every static (non-array) constructor argument, as kwargs.

        Static config is hand-listed at each of the five places that rebuild a
        term from a warm start, and dropping one there is silent: the rebuilt
        term simply runs with the default. That has bitten three separate
        times -- `activation` reset to tanh at every stage, `polar_taper` and
        `surface_taper` lost in the rollout driver, and the taper scripts
        rebuilding without either. Pairing this with :meth:`rebuild` makes the
        config travel as one object so a new field cannot be forgotten at one
        site and kept at the others.

        Array state (weights and the normalisation buffers) is deliberately
        excluded: those are what a caller is replacing.
        """
        return {
            "correct": self.correct,
            "output_cap": self.output_cap,
            "polar_taper": self.polar_taper,
            "surface_taper": self.surface_taper,
            "context_features": self.context_features,
            "activation": self.activation,
        }

    def rebuild(self, layers, in_mean=None, in_std=None, out_scale=None,
                **overrides) -> "NNBiasCorrection":
        """Return a new term with ``layers`` and this config, plus ``overrides``.

        Buffers default to this term's, so a stage that only trains weights
        does not have to restate them.
        """
        cfg = self.config()
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError(f"unknown config field(s) {sorted(unknown)}; "
                             f"choose from {sorted(cfg)}")
        cfg.update(overrides)
        return NNBiasCorrection(
            layers,
            self.in_mean.get_value() if in_mean is None else in_mean,
            self.in_std.get_value() if in_std is None else in_std,
            self.out_scale.get_value() if out_scale is None else out_scale,
            **cfg)

    def save(self, path) -> None:
        """Write the weights and normalisation buffers to ``path`` (.npz).

        Round-trips through :meth:`from_file`. ``correct`` is stored as a
        delimited string so the reload reconstructs an identical term.
        """
        layers = self.weights.get_value()
        arrays = {}
        for i, w in enumerate(layers):
            arrays[f"kernel_{i}"] = np.asarray(w.kernel)
            arrays[f"bias_{i}"] = np.asarray(w.bias)
        if self.output_cap is not None:
            # Written only when set, so files from cap-less terms (and the
            # pre-context artifacts) keep their exact old format.
            arrays["output_cap"] = np.float64(self.output_cap)
        if self.polar_taper is not None:
            # Same guard: taper-less terms keep their exact old format.
            arrays["polar_taper"] = np.asarray(self.polar_taper, np.float64)
        if self.surface_taper is not None:
            # Same guard: surface-taper-less terms keep their exact old format.
            arrays["surface_taper"] = np.asarray(self.surface_taper, np.float64)
        if self.context_features:
            # Same guard: context-less terms keep their exact old format.
            arrays["context_features"] = "|".join(self.context_features)
        if self.activation != DEFAULT_ACTIVATION:
            # Same guard: every artifact predating this key is tanh, so a
            # tanh term's file stays byte-identical to the old format.
            arrays["activation"] = self.activation
        np.savez(
            path,
            n_layers=len(layers),
            in_mean=np.asarray(self.in_mean.get_value()),
            in_std=np.asarray(self.in_std.get_value()),
            out_scale=np.asarray(self.out_scale.get_value()),
            correct="|".join(self.correct),
            **arrays,
        )

    @classmethod
    def from_file(cls, path) -> "NNBiasCorrection":
        """Rebuild a term saved by :meth:`save`."""
        data = np.load(path)
        n_layers = int(data["n_layers"])
        layers = tuple(
            DenseWeights(jnp.asarray(data[f"kernel_{i}"]),
                         jnp.asarray(data[f"bias_{i}"]))
            for i in range(n_layers)
        )
        # "".split("|") is ("",), not (): guard the empty-correct round-trip.
        correct_str = str(data["correct"])
        correct = tuple(correct_str.split("|")) if correct_str else ()
        # Absent in files saved before the cap existed: unbounded.
        cap = float(data["output_cap"]) if "output_cap" in data else None
        # Absent in files saved before the taper existed: no taper.
        taper = (tuple(map(float, data["polar_taper"]))
                 if "polar_taper" in data else None)
        surface_taper = (tuple(map(float, data["surface_taper"]))
                         if "surface_taper" in data else None)
        # Absent in files saved before context features existed: profiles only.
        ctx_str = str(data["context_features"]) if "context_features" in data else ""
        context_features = tuple(ctx_str.split("|")) if ctx_str else ()
        # Absent in every artifact predating the activation registry, and all of
        # those are tanh by construction.
        activation = (str(data["activation"]) if "activation" in data
                      else DEFAULT_ACTIVATION)
        return cls(layers,
                   jnp.asarray(data["in_mean"]),
                   jnp.asarray(data["in_std"]),
                   jnp.asarray(data["out_scale"]),
                   correct,
                   output_cap=cap,
                   polar_taper=taper,
                   surface_taper=surface_taper,
                   context_features=context_features,
                   activation=activation)


def make_bias_correction(
    coords=None, *,
    nlev: int | None = None,
    hidden: tuple[int, ...] = (64, 64),
    key=None,
    correct: tuple[str, ...] = ("temperature", "specific_humidity"),
    zero_last_layer: bool = True,
    out_scale_per_day: float = 1.0,
    output_cap: float | None = None,
    polar_taper: tuple[float, float] | None = None,
    surface_taper: tuple[float, float] | None = None,
    context_features: tuple[str, ...] = (),
    activation: str = DEFAULT_ACTIVATION,
) -> NNBiasCorrection:
    """Build an :class:`NNBiasCorrection` with default (placeholder) buffers.

    Args:
        coords: model CoordinateSystem; ``nlev`` is read from
            ``coords.nodal_shape[0]`` when ``nlev`` is not given directly.
        nlev: number of vertical levels (overrides ``coords``). Lets tests
            build the term without a full coordinate system.
        hidden: hidden layer widths.
        key: PRNG key (defaults to ``jax.random.key(0)``).
        correct: prognostic variables to correct.
        zero_last_layer: keep the term a no-op at construction (default True).
        out_scale_per_day: rough magnitude of the correction in units/day;
            converted to per-second. Only matters once the term is trained.
        output_cap: optional soft tanh bound on the dimensionless output
            (see :func:`bias_correction_tendency`).
        polar_taper: optional ``(lat0_deg, lat1_deg)`` latitude taper edges
            (see :func:`polar_taper_factor`); ``None`` = no taper.
        context_features: per-column context inputs appended after the
            profiles (see :func:`build_context`); widens the input layer by
            ``len(context_features)``. ``()`` = profiles only.

    Returns:
        A constructed :class:`NNBiasCorrection`.

    """
    if nlev is None:
        if coords is None:
            raise ValueError("make_bias_correction needs either `coords` or `nlev`")
        nlev = coords.nodal_shape[0]
    if key is None:
        key = jax.random.key(0)

    n_io = _N_FIELDS * nlev
    n_in = n_io + len(context_features)
    layers = init_mlp(key, (n_in, *hidden, n_io), zero_last_layer=zero_last_layer)

    # Placeholders. The offline stage replaces these with real climatological statistics from
    # a nudged-to-ERA5 run. Identity normalisation and a ~1/day output scale
    # are harmless while the output layer is zero.
    in_mean = jnp.zeros(n_io)
    in_std = jnp.ones(n_io)
    out_scale = jnp.ones(n_io) * (out_scale_per_day / 86400.0)

    return NNBiasCorrection(layers, in_mean, in_std, out_scale, correct,
                            output_cap=output_cap, polar_taper=polar_taper,
                            surface_taper=surface_taper,
                            context_features=context_features,
                            activation=activation)


def load_bias_correction(path) -> NNBiasCorrection:
    """Load a trained term from an ``.npz`` written by ``NNBiasCorrection.save``.

    The trained-term counterpart to :func:`make_bias_correction` (which builds
    a fresh no-op term). Compose it the same way::

        term = load_bias_correction(
            "jcm/data/bias_correction/online_term_t31_big_insol_vt.npz")
        physics = speedy_physics() + term
    """
    return NNBiasCorrection.from_file(path)
