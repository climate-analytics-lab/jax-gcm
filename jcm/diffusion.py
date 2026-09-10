"""Horizontal-diffusion configuration and filter builders.

``DiffusionFilter`` stores the time-scales and spectral orders used to damp
divergence, vorticity+humidity, and temperature after each dynamics step.

``level_dependent_scaling`` builds a ``(nlev, 1, lat_modes)`` scaling array
suitable for elementwise multiplication against a spectral state of shape
``(nlev, lon_modes, lat_modes)``. Use it to mimic ECHAM's per-level
hyperdiffusion order — del² at TOA, del⁴/⁶/⁸ going down — which keeps the
stratosphere well-damped without over-smoothing the troposphere.

:meth:`DiffusionFilter.echam_lmidatm` reproduces those ECHAM ``lmidatm``
profiles for a ``(truncation, layers)`` grid; see its docstring for how the
reference tables are indexed and what happens off them.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import jax.numpy as jnp
import tree_math
from jax import tree_util

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ECHAM6 lmidatm hyperdiffusion reference tables
# ---------------------------------------------------------------------------
#
# Base damping timescale ``dampth`` in hours, from ECHAM6.3 ``setdyn.f90``.
# This is the e-folding time applied at the truncation limit: the damping
# built below reduces exactly to ``exp(-dt/dampth)`` at ``n = nn`` (see
# :func:`level_dependent_scaling`), so the table is directly comparable with
# the SPEEDY-style uniform timescales in :meth:`DiffusionFilter.default`.
_ECHAM_DAMPTH_HOURS = {31: 12.0, 63: 7.0, 127: 1.5, 255: 0.5}

# Per-level hyperdiffusion orders from ECHAM6.3 ``mo_hdiff.f90::sudif``,
# keyed by ``(truncation, layers)`` and stored top-first as
# ``(n_levels, order)`` runs — order 1 is del², 2 del⁴, 3 del⁶, 4 del⁸.
#
# These are *level-index* tables in the reference, and they transfer verbatim
# because jcm's hybrid grids ARE ECHAM's: the pressures our L47/L95 tables put
# at each transition reproduce the ones ECHAM annotates in ``sudif`` (L47
# jk=4/7/9 → 0.2315/1.2153/2.9571 hPa vs "~0.23/1.22/2.96"; L95 jk=10/20/25 →
# 0.1503/0.7680/1.5001 hPa vs "~0.15/0.77/1.50"). A pressure-based rule would
# be *less* faithful, not more — ECHAM does not place these transitions at the
# same pressures on L47 and L95, so no single pressure rule reproduces both.
#
# Note that the profile depends on truncation as well as level count: on L95
# ECHAM stops at del⁶ for nn >= 127 but goes to del⁸ at nn = 63. Higher
# horizontal resolution gets the *lower* maximum order.
_ECHAM_LMIDATM_ORDERS = {
    (31, 47): ((4, 1), (3, 2), (2, 3), (2, 4), (36, 5)),
    (63, 47): ((4, 1), (3, 2), (2, 3), (38, 4)),
    (63, 95): ((10, 1), (10, 2), (5, 3), (70, 4)),
    (127, 95): ((10, 1), (15, 2), (70, 3)),
    (255, 95): ((10, 1), (15, 2), (70, 3)),
}

#: Level counts for which an ECHAM ``lmidatm`` order profile exists.
ECHAM_LMIDATM_LAYERS = frozenset(nlev for _, nlev in _ECHAM_LMIDATM_ORDERS)


def _check_truncation(truncation: int) -> None:
    """Reject non-positive truncations before they reach ``math.log``.

    Both selection rules are logarithmic in truncation, so a missing or zero
    ``grid.spectral_truncation`` would otherwise surface as a bare
    ``math domain error`` with no indication of which config key was empty.
    """
    if truncation <= 0:
        raise ValueError(
            f"spectral truncation must be positive to select an ECHAM "
            f"hyperdiffusion profile, got {truncation!r}; check "
            "grid.spectral_truncation."
        )


def echam_dampth_hours(truncation: int) -> float:
    """ECHAM ``dampth`` for ``truncation``, in hours.

    Exact for the truncations ECHAM tabulates (T31/T63/T127/T255); otherwise
    a power law in truncation fitted to the bracketing pair, which is how the
    reference values themselves scale (T63→T127 is ``τ ∝ nn**-2.20``). Off
    the ends of the table the nearest segment's slope is extrapolated.

    Deriving rather than hard-coding is deliberate: the previously hard-coded
    T85 value (3 h) was an eyeballed extrapolation that did not sit on ECHAM's
    own T63→T127 slope, and there was nothing at all for T106/T119.

    Raises:
        ValueError: if ``truncation`` is not positive.

    """
    _check_truncation(truncation)
    if truncation in _ECHAM_DAMPTH_HOURS:
        return _ECHAM_DAMPTH_HOURS[truncation]
    anchors = sorted(_ECHAM_DAMPTH_HOURS)
    below = [nn for nn in anchors if nn < truncation]
    above = [nn for nn in anchors if nn > truncation]
    if not below:                       # below T31: extrapolate the first leg
        lo, hi = anchors[0], anchors[1]
    elif not above:                     # above T255: extrapolate the last leg
        lo, hi = anchors[-2], anchors[-1]
    else:
        lo, hi = below[-1], above[0]
    exponent = (math.log(_ECHAM_DAMPTH_HOURS[hi] / _ECHAM_DAMPTH_HOURS[lo])
                / math.log(hi / lo))
    return _ECHAM_DAMPTH_HOURS[lo] * (truncation / lo) ** exponent


def echam_lmidatm_orders(truncation: int, layers: int) -> jnp.ndarray:
    """Per-level hyperdiffusion orders for ``(truncation, layers)``.

    Truncations ECHAM does not tabulate borrow the profile of the nearest
    tabulated truncation *in log space* — the natural metric when the tables
    themselves are spaced by factors of two. That rule keeps T85L47 on the
    T63L47 profile (as before this function existed) and puts T106L95 and
    T119L95 on the T127L95 profile.

    Raises:
        ValueError: if ``truncation`` is not positive, or no ECHAM profile
            exists for ``layers`` at all.

    """
    _check_truncation(truncation)
    candidates = [nn for nn, nlev in _ECHAM_LMIDATM_ORDERS if nlev == layers]
    if not candidates:
        raise ValueError(
            f"No ECHAM lmidatm hyperdiffusion profile for {layers} levels; "
            f"mo_hdiff.f90::sudif defines profiles for "
            f"{sorted(ECHAM_LMIDATM_LAYERS)} levels only."
        )
    nearest = min(candidates, key=lambda nn: abs(math.log(truncation / nn)))
    orders = []
    for count, order in _ECHAM_LMIDATM_ORDERS[(nearest, layers)]:
        orders.extend([order] * count)
    return jnp.asarray(orders, dtype=jnp.int32)


@tree_math.struct
class DiffusionFilter:
    """Hyperdiffusion configuration.

    The three (timescale, order) pairs control divergence, vorticity+humidity,
    and temperature damping respectively.

    Set ``level_orders_div`` / ``..._vor_q`` / ``..._temp`` to a 1-D array of
    per-level orders (length ``nlev``) to activate the ECHAM-style
    level-dependent hyperdiffusion. When left ``None`` the scalar ``..._order``
    is used for every level.
    """

    vor_q_timescale: jnp.float_  # s
    vor_q_order: jnp.int_        # uniform order when vor_q level_orders is None
    temp_timescale: jnp.float_
    temp_order: jnp.int_
    div_timescale: jnp.float_
    div_order: jnp.int_

    # Optional per-level orders. Shape (nlev,). Leave None for uniform order.
    level_orders_div: Optional[jnp.ndarray] = None
    level_orders_vor_q: Optional[jnp.ndarray] = None
    level_orders_temp: Optional[jnp.ndarray] = None

    @classmethod
    def default(cls):
        """SPEEDY defaults (temp 24h, vor_q 12h, div 2h); uniform order."""
        return cls(
            div_timescale=2 * 60 * 60,
            div_order=1,
            vor_q_timescale=12 * 60 * 60,
            vor_q_order=2,
            temp_timescale=24 * 60 * 60,
            temp_order=2,
        )

    @classmethod
    def echam_t85_l47(cls):
        """ECHAM ``lmidatm`` profile for T85 x 47 levels.

        Thin wrapper over :meth:`echam_lmidatm` retained for callers that pin
        a named profile.
        """
        return cls.echam_lmidatm(truncation=85, layers=47)

    @classmethod
    def echam_t63_l47(cls):
        """ECHAM ``lmidatm`` profile for T63 x 47 levels — the tuned target.

        Thin wrapper over :meth:`echam_lmidatm`. This is the one combination
        the reference tabulates exactly on both axes (``dampth = 7 h``, orders
        ``[del²]*4 + [del⁴]*3 + [del⁶]*2 + [del⁸]*38``), so it doubles as the
        fidelity anchor for the derived cases.
        """
        return cls.echam_lmidatm(truncation=63, layers=47)

    @classmethod
    def echam_lmidatm(cls, truncation: int, layers: int):
        """ECHAM6.3 ``lmidatm`` level-dependent hyperdiffusion for a grid.

        Combines the ``setdyn.f90`` base timescale (:func:`echam_dampth_hours`)
        with the ``mo_hdiff.f90::sudif`` per-level order profile
        (:func:`echam_lmidatm_orders`) — del² near the model top grading to
        del⁶/del⁸ below, which damps the stratosphere hard without
        over-smoothing the troposphere.

        Args:
            truncation: spectral truncation (``nn``).
            layers: number of vertical levels (``nlev``). Must be one of
                :data:`ECHAM_LMIDATM_LAYERS`.

        Raises:
            ValueError: if ECHAM defines no profile for ``layers``.

        """
        level_orders = echam_lmidatm_orders(truncation, layers)
        base_tau = echam_dampth_hours(truncation) * 3600.0
        return cls(
            # Effective timescale for each variable is ``base_tau * factor``;
            # factors match ECHAM's difvo / difd / dift proportions
            # (``mo_hdiff.f90``: ``difd = 5*difvo``, ``dift = 0.4*difvo``).
            div_timescale=base_tau / 5.0,        # divergence 5x stronger
            div_order=1,
            vor_q_timescale=base_tau,            # vorticity baseline
            vor_q_order=2,
            temp_timescale=base_tau / 0.4,       # temperature 2.5x weaker
            temp_order=2,
            level_orders_div=level_orders,
            level_orders_vor_q=level_orders,
            level_orders_temp=level_orders,
        )

    @classmethod
    def auto(cls, truncation: int, layers: int, vertical: str):
        """Resolution-aware default filter for a grid.

        When the grid is a hybrid grid with a level count ECHAM tabulates
        (L47 or L95), return the ECHAM ``lmidatm`` level-dependent profile
        for that ``(truncation, layers)`` — del² near the model top grading
        to del⁶/del⁸ below, with the ``setdyn.f90`` base timescale. That's
        the stability stack these grids were tuned for in ECHAM, and it is
        what the L95 middle-atmosphere grids exist to exploit. Any other grid
        — SPEEDY T31L8, Held-Suarez, a hybrid grid at an untabulated level
        count — gets the uniform SPEEDY del² profile, with a warning in the
        hybrid case since that is unlikely to be what was intended (#579).

        ``vertical`` is the grid's vertical-coordinate kind (``"hybrid"`` or
        ``"sigma"``).
        """
        # Match on the (vertical=hybrid, layers) pair so this fires for every
        # ECHAM-family grid at any truncation — and stays inert for SPEEDY
        # T31L8 / Held-Suarez, which have their own tuned uniform damping.
        if vertical == "hybrid" and layers in ECHAM_LMIDATM_LAYERS:
            return cls.echam_lmidatm(truncation, layers)
        if vertical == "hybrid":
            logger.warning(
                "diffusion.kind=auto found no ECHAM lmidatm profile for a "
                "hybrid grid with %d levels, so this run uses the uniform "
                "SPEEDY profile (24h temp / 12h vor_q / 2h div). ECHAM "
                "profiles exist for %s levels. Set diffusion.kind "
                "explicitly to silence this.",
                layers, sorted(ECHAM_LMIDATM_LAYERS),
            )
        return cls.default()

    def validate_layers(self, layers: int) -> None:
        """Raise if a level-dependent profile does not match the grid.

        A level-dependent profile is a per-level array, so pinning one whose
        length does not match the grid fails later inside the spectral filter
        as an opaque broadcast error ("(95, 213, 108) vs (47,)"). Catch it
        here, where the actual mismatch can be named (#579).
        """
        n_orders = (None if self.level_orders_temp is None
                    else len(self.level_orders_temp))
        if n_orders is not None and layers and n_orders != layers:
            raise ValueError(
                f"diffusion profile has {n_orders} levels but the grid has "
                f"{layers} levels. Use diffusion.kind=auto (or "
                "'echam_lmidatm') to get the profile matching this grid, or "
                "'default' for the uniform SPEEDY profile."
            )

    def scaled(self, scale: float) -> "DiffusionFilter":
        """Return a copy with all three timescales multiplied by ``scale``.

        ``scale == 1.0`` returns ``self`` unchanged (identity), so the
        SPEEDY-tuned configs stay bit-for-bit as before.
        """
        if scale == 1.0:
            return self
        return DiffusionFilter(
            div_timescale=self.div_timescale * scale,
            div_order=self.div_order,
            vor_q_timescale=self.vor_q_timescale * scale,
            vor_q_order=self.vor_q_order,
            temp_timescale=self.temp_timescale * scale,
            temp_order=self.temp_order,
            level_orders_div=self.level_orders_div,
            level_orders_vor_q=self.level_orders_vor_q,
            level_orders_temp=self.level_orders_temp,
        )

    def isnan(self):
        return tree_util.tree_map(
            lambda x: jnp.isnan(x) if hasattr(x, "shape") else jnp.asarray(False),
            self,
        )


def level_dependent_scaling(
    eigenvalues: jnp.ndarray,
    timescale: float,
    orders_per_level: jnp.ndarray,
    time_step: float,
) -> jnp.ndarray:
    """Build a per-level spectral damping scaling.

    Returns an array of shape ``(nlev, 1, lat_modes)`` such that element-wise
    multiplication against a spectral state of shape
    ``(nlev, lon_modes, lat_modes)`` applies the correct level-dependent
    hyperdiffusion damping per time step.

    For each level ``k`` with order ``p_k``:

        scaling[k, 0, n] = exp( -(dt/timescale) * (|eig[n]| / max|eig|) ** p_k )

    Algebraically equivalent to the textbook formulation
    ``exp(-dt/(τ·|eig_max|^p) · |eig|^p)`` but float-stable: the
    eigenvalues ``|eig|`` are O(1e-10) (nondimensional Laplacian
    eigenvalues for spherical harmonics), so for ``p=4`` the textbook
    form computes ``|eig_max|^4 ≈ 1e-40``, which underflows in float32
    to 0 → ``dt/0 = inf`` → ``inf · 0 = NaN`` in the leading-edge
    coefficient. Computing the ``|eig|/|eig_max|`` ratio first keeps the
    intermediate in ``[0, 1]``.

    Args:
        eigenvalues: Negative-definite Laplacian eigenvalues from
            ``grid.laplacian_eigenvalues``; shape ``(lat_modes,)``.
        timescale: Damping timescale in seconds (applied at the largest
            wavenumber).
        orders_per_level: Integer array of per-level orders; shape ``(nlev,)``.
        time_step: Model time step in seconds.

    Returns:
        ``(nlev, 1, lat_modes)`` scaling.

    """
    pos_eig = jnp.abs(eigenvalues)                                  # (lat_modes,)
    # ``.max()`` rather than ``[-1]``: under SPMD the modal axis is padded with
    # zeros so it divides evenly across devices, and those zeros land on the
    # last index. ``[-1]`` would then read 0 and blow up the normalisation;
    # ``max(|eig|)`` is the true largest-wavenumber eigenvalue either way
    # (padding zeros never exceed a real magnitude).
    pos_eig_max = pos_eig.max()                                     # scalar
    p = orders_per_level[:, None].astype(jnp.float32)               # (nlev, 1)
    norm_eig = pos_eig[None, :] / pos_eig_max                       # (1, lat_modes), in [0, 1]
    pow_norm = norm_eig ** p                                        # (nlev, lat_modes)
    return jnp.exp(-(time_step / timescale) * pow_norm)[:, None, :]


def uniform_scaling(
    eigenvalues: jnp.ndarray,
    timescale: float,
    order: int,
    time_step: float,
) -> jnp.ndarray:
    """Uniform-order damping scaling, shape ``(lat_modes,)``.

    Equivalent to ``dinosaur.filtering.horizontal_diffusion_filter`` with a
    single ``(timescale, order)``. Float-stable rewrite — see
    :func:`level_dependent_scaling` for the underflow note (matters
    once ``order >= 4``).
    """
    pos_eig = jnp.abs(eigenvalues)
    # ``.max()`` not ``[-1]`` — robust to the zero-padding of the modal axis
    # under SPMD sharding (see :func:`level_dependent_scaling`).
    norm_eig = pos_eig / pos_eig.max()
    return jnp.exp(-(time_step / timescale) * norm_eig ** order)
