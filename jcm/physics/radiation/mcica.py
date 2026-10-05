"""Monte Carlo Independent Column Approximation (McICA) sub-column generator.

Stochastic sub-column generators for RRTMGP-class radiation schemes, which
handle subgrid cloud variability and vertical overlap: the Räisänen et al.
(2004) generator for the random and generalised-exponential rules, and
ECHAM6.3's ``mo_cld_sampling.f90::sample_cld_state`` chain for
maximum-random. Each
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
- ``"maximum_random"`` ECHAM6.3's rule and jcm's default (``i_overlap =
  1``, ``mo_radiation_parameters.f90`` l.71), the rank chain of
  ``mo_cld_sampling.f90::sample_cld_state`` (l.66-83). ECHAM runs that chain
  top-down on the surface-first column ``psrad_interface`` hands its
  radiation (``mo_psrad_interface.f90`` l.221-227, 459, 476): each level
  keeps the rank of the level above where the sub-column is cloudy there and
  otherwise draws a fresh rank in that level's clear part. jcm runs the same
  rule bottom-up on its top-first column: each level keeps the rank of the
  level below where the sub-column is cloudy there, and otherwise redraws in
  that level's clear part. The two directions give every sub-column cloud
  pattern the same probability (verified exactly, by enumerating the
  patterns of random profiles, and by ``mcica_test.py``), so in both
  adjacent cloudy layers overlap maximally and layers separated by clear air
  overlap randomly. Its expected total cover is the adjacent-layer
  Geleyn-Hollingsworth (1979) product ECHAM reports as ``cld_cvr``
  (``mo_radiation.f90`` l.436-442; :func:`expected_total_cover`), the same
  read from either end.
- ``"exponential"``  generalised-exponential overlap with a configurable
  decorrelation length (jcm default 2 km), a jcm option: ECHAM6.3's
  sampler has no exponential rule.

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
    ``LWP_grid / max(f, eps)`` — but the in-cloud path is **zeroed in
    (essentially) clear cells** (``cloud_fraction <= 2*eps``).

    This mirrors ECHAM ``mo_psrad_interface.f90:224-237``::

        cld_frc_vr        = MAX(EPSILON, cld_frc)
        zlwgkg_vr         = xm_liq * 1000 / cld_frc_vr
        WHERE (cld_frc_vr <= 2*EPSILON) zlwgkg_vr = 0

    Without the zeroing, a vanishing cloud fraction inflates the in-cloud
    water without bound (grid-mean / eps), which drives a runaway cloud
    optical depth and, via ``inf * 0`` once the clear sub-columns are
    masked, NaNs the radiation. ECHAM relies on its cloud routine having
    already evaporated condensate in clear cells (mirrored here for the 2M
    scheme by the clear-sky evaporation in ``lohmann_2m``); the zeroing is
    the radiation-side half of that contract and protects every scheme
    (1M and 2M) against any residual decorrelated condensate. The
    ``cloud_fraction > 2*eps`` guard below zeros exactly the (essentially)
    clear cells and leaves every resolved cloud untouched.
    """
    in_cloud = grid_mean_path / jnp.maximum(cloud_fraction, eps)
    return jnp.where(cloud_fraction > 2.0 * eps, in_cloud, 0.0)


#: In-cloud condensate path [kg/m²] below which a cell passes no cloud to the
#: radiative transfer. Both schemes combine the single-scattering albedo and the
#: asymmetry by dividing by the optical depth, and a cloud whose optical depth is
#: a nonzero float32 far below ``1e-19`` has a derivative whose reciprocal square
#: overflows (``inf·0 = NaN``) while its value is a negligible ``1e-12`` or less
#: of an optical depth. The jax-rrtmgp library's optical depth is the extinction
#: times the path times a presence gate that is a cubic of the path below
#: ``2e-6 g/m²``, so it sits in that band for paths between about ``1e-16`` and
#: ``1e-8 g/m²``; the grey scheme's reaches it with condensate near ``1e-34``. The
#: floor, ``1e-8 g/m²``, is far under any radiatively relevant path and far above
#: both bands. ECHAM tests ``xq > 0``; below the floor the cloud has no radiative
#: effect at float32 precision, so the fluxes are those of ``xq > 0``.
NEGLIGIBLE_CLOUD_PATH_KG_M2 = 1.0e-11


def resolvable_path(path: jnp.ndarray) -> jnp.ndarray:
    """Zero the in-cloud paths below :data:`NEGLIGIBLE_CLOUD_PATH_KG_M2`.

    A hard test with the reference derivative (zero below the floor, the path's
    own above), like ECHAM's ``xq > 0`` mask it refines.
    """
    return jnp.where(path > NEGLIGIBLE_CLOUD_PATH_KG_M2, path, 0.0)


# NaN guard on the PHYSICAL in-cloud condensate (kg/kg) that sets the effective
# radii and the cloud optical depth. A thin but resolved cloud carrying large
# grid-mean condensate gives a huge in-cloud water (grid_mean / cf), and the
# resulting optical depth NaNs the two-stream solver. Applied by
# :func:`in_cloud_condensate` right after ``in_cloud_path``.
#
# This is a one-sided clip -- the identity almost everywhere, flattening
# everything above the threshold to the same value -- and is NOT the sub-grid
# inhomogeneity treatment. The inhomogeneity factor (ECHAM ``zinhoml``/
# ``zinhomi``) is a separate FIXED multiplicative reduction applied to the
# per-gpoint optical-depth paths (see ``RadiationParameters.cloud_inhomogeneity_*``
# and the ``in_cloud_*_lib`` scaling in ``radiation_scheme_rrtmgp``). Measured on
# T63L47 output this clip binds in ~0.003% of cloudy cells, so it is inert in
# practice; keep it strictly as a NaN guard (#678).
_MAX_IN_CLOUD_CONDENSATE = 1.0e-2


def in_cloud_condensate(
    grid_mean: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    eps: float = 1.0e-3,
) -> jnp.ndarray:
    """In-cloud condensate the radiation sees: :func:`in_cloud_path` plus the NaN cap.

    The one definition shared by the RRTMGP solve (its cloud paths) and the
    effective radii (``cloud_optics.radiation_effective_radii``), so the
    radius a layer radiates with is formed from exactly the condensate that
    layer's optical depth is.
    """
    return jnp.minimum(
        in_cloud_path(grid_mean, cloud_fraction, eps=eps),
        _MAX_IN_CLOUD_CONDENSATE,
    )


def effective_cloud_fraction(
    cloud_fraction: jnp.ndarray,
    eps: float = 1.0e-3,
) -> jnp.ndarray:
    """Zero the cloud fraction in the cells :func:`in_cloud_path` zeros.

    ``in_cloud_path`` zeros the in-cloud condensate wherever
    ``cloud_fraction <= 2*eps``, so those cells are radiatively **empty**. The
    fraction that drives the McICA sub-column sampler and the total-cover
    diagnostic must agree with that: an optically-empty cell must not be
    reported as cloud cover, nor act as a cloudy layer that bridges
    maximum-random overlap across an otherwise-clear gap. Returning
    ``where(cloud_fraction > 2*eps, cloud_fraction, 0)`` ties the two together
    on the **same** criterion.

    This mirrors ECHAM ``mo_psrad_interface.f90:232`` — the WHERE that zeros
    the in-cloud water (``ziwgkg_vr``/``zlwgkg_vr``) ALSO clears the
    layer-cloudy flag ``icldlyr`` on the identical ``cld_frc_vr > 2*EPSILON``
    test, so the sampler and the optics see a consistent "this cell is clear".
    In ECHAM ``EPSILON`` is machine epsilon, so this only ever bites exactly
    empty cells; here ``eps`` is a *physical* threshold (``cld_frac_min``,
    default 1e-3), so the tie must be made explicit.
    """
    return jnp.where(cloud_fraction > 2.0 * eps, cloud_fraction, 0.0)


def _alpha_from_overlap(
    cloud_fraction: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    overlap: _OverlapRule,
    decorrelation_km: float,
) -> jnp.ndarray:
    """Return per-interface decorrelation factors α_k.

    ``α_k`` is the probability that the rank random number at layer k
    inherits its value from layer k-1 (full correlation), for the two rules
    whose correlation does not depend on the sampled sub-column: random
    overlap → all zeros; the generalised-exponential rule
    ``α_k = exp(-Δz_k / L_cld)``. Maximum-random keeps a rank depending on
    the sub-column's own cloud (:func:`_maximum_random_ranks`), so it has no
    such factor.
    """
    nlev = cloud_fraction.shape[0]
    if overlap == "random":
        return jnp.zeros((nlev - 1,) + cloud_fraction.shape[1:])
    if overlap == "exponential":
        decorrelation_m = decorrelation_km * 1000.0
        # Use the layer thickness at level k as the displacement between
        # the centres of layers k-1 and k. Slightly approximate (the
        # exact distance would average the two thicknesses) but cheap and
        # well within the noise floor for any realistic L_cld.
        dz = layer_thickness[1:]
        return jnp.exp(-dz / decorrelation_m)
    raise ValueError(
        f"No decorrelation factor for overlap rule {overlap!r}; "
        "'random' and 'exponential' have one, 'maximum_random' keeps ranks "
        "by the sub-column's cloud (_maximum_random_ranks)."
    )


def _maximum_random_ranks(u: jnp.ndarray, cloud_fraction: jnp.ndarray) -> jnp.ndarray:
    """ECHAM's maximum-random rank rule for one sub-column, TOA-first.

    The rule of ``mo_cld_sampling.f90::sample_cld_state`` (l.66-83), in this
    module's convention (a cell is cloudy where its rank is below the cover;
    ECHAM tests ``rank > 1 - cover``, the same rule for ``1 - rank``), run
    from the bottom up: the lowest level takes ``u`` as its rank, and each
    level above keeps the rank of the level below where the sub-column is
    cloudy there and otherwise takes ``cf_below + u_k·(1 − cf_below)``,
    uniform over the clear part of the level below. ECHAM runs the chain
    from the top down on its surface-first column (see the module
    docstring); the two directions give every cloud pattern the same
    probability. ``u`` is ``[nlev]`` uniforms. Sequential in k from the
    bottom → reversed ``lax.scan``.

    The ranks depend on the cover only through the redraw's clear-part
    offset and through the comparisons that pick keep or redraw; the masks
    built from them are ``r < cf`` comparisons again, piecewise constant in
    the cover, so the sampler carries no cover gradient, as for the other
    two rules.
    """

    def step(r_below, inputs):
        u_k, cf_below = inputs
        r_k = jnp.where(r_below < cf_below, r_below,
                        cf_below + u_k * (1.0 - cf_below))
        return r_k, r_k

    _, r_upper = jax.lax.scan(step, u[-1], (u[:-1], cloud_fraction[1:]),
                              reverse=True)
    return jnp.concatenate([r_upper, u[-1:]], axis=0)


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
    overlap: _OverlapRule = "maximum_random",
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
        decorrelation_km: vertical decorrelation length [km] of the
            ``"exponential"`` rule; inert under the other two.
        key: a JAX PRNG key. Construct deterministically via
            ``jax.random.fold_in`` over whatever stochastic axes the
            caller wants reproducible (model_step, column index,
            g-point index, ...).

    Returns:
        ``[n_subcols, nlev]`` array of 0/1 floats — 1 where cloud is
        present in that sub-column, 0 elsewhere.

    """
    nlev = cloud_fraction.shape[0]
    if overlap == "maximum_random":
        def ranks(u, y):
            return _maximum_random_ranks(u, cloud_fraction)
    else:
        alpha = _alpha_from_overlap(
            cloud_fraction, layer_thickness, overlap, decorrelation_km,
        )

        def ranks(u, y):
            return _rank_chain(u, y, alpha)

    def per_subcol(s_key):
        u_key, y_key = jax.random.split(s_key)
        u = jax.random.uniform(u_key, (nlev,))
        y = jax.random.uniform(y_key, (nlev - 1,))
        r = ranks(u, y)
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


def expected_total_cover(
    cloud_fraction: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    overlap: _OverlapRule = "maximum_random",
    decorrelation_km: float = 2.0,
) -> jnp.ndarray:
    """Closed-form expectation of the sub-column total cloud cover.

    The diagnostic counterpart of :func:`generate_subcolumns`: the EXACT
    expectation of the total cover under the rank chain that
    ``per_subcol`` samples for the configured rule — so a scheme that
    cannot afford sub-column draws (the NN emulator) publishes the cover the
    McICA sampler reports in expectation, for any rule.

    Under maximum-random that is ECHAM's ``cld_cvr`` (``mo_radiation.f90``
    l.436-442), the adjacent-layer Geleyn-Hollingsworth product
    ``1 − (1 − c_1)·Π_k (1 − max(c_k, c_{k−1}))/(1 − min(c_{k−1}, 1 − ε))``:
    once a sub-column is clear in a level its next rank is a fresh draw in
    that level's clear part, so the probability of staying clear one level
    up depends on the two covers alone, ``(1 − max)/(1 − c_below)``, and the
    product over the column is the same read from the top or the bottom.

    For random and exponential overlap the chain inherits the previous rank with probability ``a_k`` and
    refreshes it otherwise, so a column partitions into rank *segments*
    sharing one uniform; a segment spanning layers ``s..k`` is clear with
    probability ``1 - max(cf_s..cf_k)``. Conditioning on the start of the
    final segment gives the O(nlev^2) recursion (B_j = P(first j layers
    clear))::

        B_{k+1} = sum_s [start_s * prod(a_{s..k-1})] B_s (1 - max cf_{s..k})

    A pairwise-conditional product (Hogan & Illingworth style) is NOT this
    expectation: an inherited rank keeps its history across several
    interfaces, and the pairwise form can overstate cover by several
    points on three-layer profiles (PR #730 review). At a = 0 this reduces
    to the random product.

    Unlike :func:`column_total_cover` (the grey beam-split's deliberate
    ``max`` approximation), this is for DIAGNOSTIC output (CMIP ``clt``).
    """
    cf = jnp.clip(cloud_fraction, 0.0, 1.0)
    if overlap == "maximum_random":
        # ECHAM bounds the divisor with 1 - EPSILON; a level with cover 1
        # makes every sub-column cloudy, and its numerator is then 0.
        eps = jnp.finfo(cf.dtype).eps
        stay_clear = ((1.0 - jnp.maximum(cf[1:], cf[:-1]))
                      / (1.0 - jnp.minimum(cf[:-1], 1.0 - eps)))
        return 1.0 - (1.0 - cf[0]) * jnp.prod(stay_clear, axis=0)
    alpha = _alpha_from_overlap(cf, layer_thickness, overlap, decorrelation_km)
    nlev = cf.shape[0]
    if nlev == 1:
        return 1.0 - (1.0 - cf[0])
    # B[j] = P(first j layers all clear). Keep the descending sum and
    # repeated maximum order, including the subgradient at tied fractions.
    # Scans keep the traced program compact as the number of layers grows.
    before = jnp.concatenate((jnp.ones_like(cf[:1]), alpha), axis=0)
    one = jnp.ones(cf.shape[1:])
    zero = jnp.zeros(cf.shape[1:])
    clear = jnp.zeros((nlev + 1,) + cf.shape[1:], dtype=jnp.result_type(cf, one))
    clear = clear.at[0].set(one).at[1].set(1.0 - cf[0])

    def layer(clear, k):
        def segment(carry, s):
            contrib, prod_a, seg_max = carry
            active = s <= k
            seg_max = jnp.where(active, jnp.maximum(seg_max, cf[s]), seg_max)
            start = jnp.where(s > 0, 1.0 - before[s], 1.0)
            addition = start * prod_a * clear[s] * (1.0 - seg_max)
            contrib = contrib + jnp.where(active, addition, 0.0)
            prod_a = jnp.where(active & (s > 0), prod_a * before[s], prod_a)
            return (contrib, prod_a, seg_max), None

        (contrib, _, _), _ = jax.lax.scan(
            segment, (zero, one, cf[k]), jnp.arange(nlev - 1, -1, -1)
        )
        return clear.at[k + 1].set(contrib), None

    clear, _ = jax.lax.scan(layer, clear, jnp.arange(1, nlev))
    return 1.0 - clear[nlev]
