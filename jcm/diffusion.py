"""Horizontal-diffusion configuration and filter builders.

``DiffusionFilter`` stores the time-scales and spectral orders used to damp
divergence, vorticity+humidity, and temperature after each dynamics step.

``level_dependent_scaling`` builds a ``(nlev, 1, lat_modes)`` scaling array
suitable for elementwise multiplication against a spectral state of shape
``(nlev, lon_modes, lat_modes)``. Use it to mimic ECHAM's per-level
hyperdiffusion order — del² at TOA, del⁴/⁶/⁸ going down — which keeps the
stratosphere well-damped without over-smoothing the troposphere.

Known issue: the level-dependent path currently triggers NaN at order >= 4
under JIT (eager and orders 1-3 are fine). The uniform-order path
(``level_orders_* = None``) is unaffected. Use the upper sponge layer
(``jcm.physics.dissipation.UpperSponge``) as an alternative stabiliser
until the JIT / order=4 interaction is diagnosed.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import tree_math
from jax import tree_util


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
        """Level-dependent hyperdiffusion profile tuned for T85 x 47 levels.

        Based on the ECHAM T63L47 order profile (see mo_hdiff.f90::sudif)
        extrapolated to T85 by shortening the base timescale from 7 h (T63)
        toward 3 h (T85 sits between T63 and T127). Levels 1-4 use del²,
        5-7 del⁴, 8-9 del⁶, 10+ del⁸. Applied equally to div/vor_q/temp.
        """
        orders = [1] * 4 + [2] * 3 + [3] * 2 + [4] * 38  # 4+3+2+38 = 47
        level_orders = jnp.asarray(orders, dtype=jnp.int32)
        base_tau = 3 * 60 * 60  # 3 h -- between T63 (7h) and T127 (1.5h)
        return cls(
            # Effective timescale for each variable is ``base_tau * factor``;
            # factors match ECHAM's difvo / difd / dift proportions.
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

        scale_k = time_step / (timescale * |eigenvalues[-1]| ** p_k)
        scaling[k, 0, n] = exp( - scale_k * (-eigenvalues[n]) ** p_k )

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
    pos_eig = -eigenvalues                                          # (lat_modes,)
    pos_eig_max = jnp.abs(eigenvalues[-1])                          # scalar
    p = orders_per_level[:, None].astype(jnp.float32)               # (nlev, 1)
    scale_per_level = time_step / (timescale * pos_eig_max ** p)    # (nlev, 1)
    pow_eig = pos_eig[None, :] ** p                                 # (nlev, lat_modes)
    # Insert lon-modes singleton so we can broadcast against (nlev, M, N)
    return jnp.exp(-scale_per_level * pow_eig)[:, None, :]


def uniform_scaling(
    eigenvalues: jnp.ndarray,
    timescale: float,
    order: int,
    time_step: float,
) -> jnp.ndarray:
    """Uniform-order damping scaling, shape ``(lat_modes,)``.

    Equivalent to ``dinosaur.filtering.horizontal_diffusion_filter`` with a
    single ``(timescale, order)``.
    """
    pos_eig_max = jnp.abs(eigenvalues[-1])
    scale = time_step / (timescale * pos_eig_max ** order)
    return jnp.exp(-scale * (-eigenvalues) ** order)
