"""Monte Carlo Independent Column Approximation (McICA) sub-column generator.

Implements the Räisänen et al. (2004) generalized exponential-random
overlap stochastic sub-column generator used by RRTMGP-class radiation
schemes to handle subgrid cloud variability + vertical overlap. Each
sub-column is a binary cloud profile: cloudy or clear at every level.
Radiation is then run *as if the column were homogeneous* in each
sub-column; averaging across many sub-columns (or across radiation
g-points, which is the whole point of McICA) recovers the true
overlap-aware fluxes with no extra cost beyond the homogeneous case.

The classical reference is

    Räisänen, P., Barker, H. W., Khairoutdinov, M., Li, J., Randall, D. A.,
    "Stochastic generation of subgrid-scale cloudy columns for large-scale
    models", QJRMS 130, 2047-2067 (2004).

Three overlap rules are supported:

- ``"random"``      independent draws at each level (no overlap).
- ``"maximum_random"`` maximum within continuous cloud banks, random
  across clear layers (Geleyn-Hollingsworth 1979).
- ``"exponential"``  generalised-exponential overlap with a configurable
  decorrelation length (ECHAM6 default ~2 km).

Determinism: the caller is responsible for constructing a PRNG key that
reflects whatever stochastic axes it cares about (model step, column
index, g-point index, ...). The recommended pattern is to compose
``jax.random.fold_in`` calls on those axes; the result is bit-exact
reproducible across runs.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp


_OverlapRule = Literal["random", "maximum_random", "exponential"]


def in_cloud_path(
    grid_mean_path: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    eps: float = 1.0e-3,
) -> jnp.ndarray:
    """Convert a grid-mean condensate path to its in-cloud value.

    Grid-mean ``LWP_grid = f * LWP_in_cloud``, so the in-cloud value is
    ``LWP_grid / max(f, eps)``. The eps floor avoids division by zero in
    clear cells; downstream code should multiply by a sub-column mask
    that vanishes in clear cells anyway, so the eps choice is cosmetic.
    """
    return grid_mean_path / jnp.maximum(cloud_fraction, eps)


def _alpha_from_overlap(
    cloud_fraction: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    overlap: _OverlapRule,
    decorrelation_km: float,
) -> jnp.ndarray:
    """Return per-interface decorrelation factors α_k.

    ``α_k`` is the probability that the rank random number at layer k
    inherits its value from layer k-1 (full correlation). Random
    overlap → all zeros; maximum overlap → all ones; the
    generalised-exponential rule blends them via
    ``α_k = exp(-Δz_k / L_cld)``.
    """
    nlev = cloud_fraction.shape[0]
    if overlap == "random":
        return jnp.zeros((nlev - 1,) + cloud_fraction.shape[1:])
    if overlap == "maximum_random":
        # α = 1 between two cloudy layers (so they share a rank within
        # one cloud bank), 0 across a clear layer that separates banks.
        return jnp.where(cloud_fraction[:-1] > 0, 1.0, 0.0)
    if overlap == "exponential":
        decorrelation_m = decorrelation_km * 1000.0
        # Use the layer thickness at level k as the displacement between
        # the centres of layers k-1 and k. Slightly approximate (the
        # exact distance would average the two thicknesses) but cheap and
        # well within the noise floor for any realistic L_cld.
        dz = layer_thickness[1:]
        return jnp.exp(-dz / decorrelation_m)
    raise ValueError(
        f"Unknown overlap rule {overlap!r}; "
        "choose 'random', 'maximum_random', or 'exponential'."
    )


def _rank_chain(u: jnp.ndarray, y: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Build the per-level rank random number r_k via Räisänen's chain.

    ``r_0 = u_0``; for k ≥ 1, ``r_k = r_{k-1}`` with probability
    ``α_k`` (decorrelation decision drawn from y), else ``r_k = u_k``.
    Sequential dependency in k → ``lax.scan``.
    """

    def step(r_prev, inputs):
        u_k, y_k, alpha_k = inputs
        r_k = jnp.where(y_k < alpha_k, r_prev, u_k)
        return r_k, r_k

    _, r_rest = jax.lax.scan(step, u[0], (u[1:], y, alpha))
    return jnp.concatenate([u[:1], r_rest], axis=0)


def generate_subcolumns(
    cloud_fraction: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    *,
    n_subcols: int,
    overlap: _OverlapRule = "exponential",
    decorrelation_km: float = 2.0,
    key: jax.Array,
) -> jnp.ndarray:
    """Generate ``n_subcols`` binary cloud masks for one column.

    Args:
        cloud_fraction: ``[nlev]`` grid-mean cloud fraction (TOA-first).
        layer_thickness: ``[nlev]`` layer thickness in metres.
        n_subcols: number of sub-columns to draw. Pass 1 per RRTMGP
            g-point for canonical McICA; pass a larger value for
            schemes (like grey two-stream) that don't have enough
            spectral subdivision to absorb the stochastic noise.
        overlap: overlap assumption.
        decorrelation_km: vertical decorrelation length for
            ``"exponential"`` overlap. ECHAM6 default ≈ 2 km.
        key: a JAX PRNG key. Construct deterministically via
            ``jax.random.fold_in`` over whatever stochastic axes the
            caller wants reproducible (model_step, column index,
            g-point index, ...).

    Returns:
        ``[n_subcols, nlev]`` array of 0/1 floats — 1 where cloud is
        present in that sub-column, 0 elsewhere.

    """
    nlev = cloud_fraction.shape[0]
    alpha = _alpha_from_overlap(
        cloud_fraction, layer_thickness, overlap, decorrelation_km,
    )

    def per_subcol(s_key):
        u_key, y_key = jax.random.split(s_key)
        u = jax.random.uniform(u_key, (nlev,))
        y = jax.random.uniform(y_key, (nlev - 1,))
        r = _rank_chain(u, y, alpha)
        return (r < cloud_fraction).astype(jnp.float32)

    subcol_keys = jax.random.split(key, n_subcols)
    return jax.vmap(per_subcol)(subcol_keys)


def column_key(
    base_key: jax.Array,
    *,
    model_step: jax.Array | int,
    column_index: jax.Array | int,
) -> jax.Array:
    """Compose a deterministic per-column PRNG key from model + column.

    This is the recommended seeding pattern for McICA: the same
    ``(base_key, model_step, column_index)`` always reproduces the same
    sub-columns, so simulation reruns are bit-exact regardless of the
    physics-block layout. G-point indices fold in further inside the
    radiation backend (one extra ``fold_in`` per g-point).
    """
    k = jax.random.fold_in(base_key, jnp.asarray(model_step, jnp.int32))
    return jax.random.fold_in(k, jnp.asarray(column_index, jnp.int32))


def column_total_cover(
    cloud_fraction: jnp.ndarray,
    overlap_code: int,
) -> jnp.ndarray:
    """Compute a scalar column-integrated cloud fraction.

    Used by the grey two-stream beam-split path, which combines a
    fully-clear and a fully-cloudy radiative-transfer call as
    ``F = (1 - c_col) F_clear + c_col F_cloudy``. Three closed-form
    options, dispatched on the ``RadiationParameters`` overlap code:

    - random (0):                ``c_col = 1 - ∏ (1 - f_k)``
    - maximum_random / max (1):  ``c_col = max_k f_k``
    - exponential (2):           ``c_col = max_k f_k``

    Both max-random and exponential reduce to ``max f_k`` here. The
    difference between max-random's continued-bank product and a plain
    max is small for stratiform clouds (the dominant grey-scheme use
    case) and within the noise floor of the two-call beam-split itself
    — neither captures the full McICA-with-sub-columns behaviour. For
    that, use the RRTMGP path (Phase 3) where the gpoint count makes
    proper McICA effectively free.
    """
    f = jnp.clip(cloud_fraction, 0.0, 1.0)
    c_random = 1.0 - jnp.prod(1.0 - f, axis=0)
    c_max = jnp.max(f, axis=0)

    return jax.lax.switch(
        overlap_code,
        [
            lambda: c_random,    # 0 random
            lambda: c_max,       # 1 maximum_random (max approximation)
            lambda: c_max,       # 2 exponential (max approximation)
        ],
    )
