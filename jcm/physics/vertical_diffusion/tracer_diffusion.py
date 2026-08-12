"""``TracerVerticalDiffusion`` — implicit turbulent mixing of tracers.

ECHAM's vdiff diffuses every tracer with the heat exchange coefficient
(``cfh``, the ``pxtte`` update in ``mo_vdiff_solver``) and CAM likewise
diffuses all constituents; jcm's TTE-TKE term solves only its fixed
8-variable block (u, v, T, q, qc, qi, TKE, TTE), so until now nothing in
the physics mixed aerosol or gas tracers vertically at all — the dycore
was the sole transporter (#602 item 2). This term closes that gap
generically: an unconditionally stable backward-Euler diffusion of an
explicit tracer list, using the ``kh`` exchange-coefficient profile the
TTE-TKE term publishes in the ``vertical_diffusion`` diagnostic.

Like the other consumers of that diagnostic (ARG's updraft, dry
deposition's u*), the profile comes from the previous step's carry
because the vdiff term runs after the aerosol block in the ECHAM
ordering; on the very first step the diagnostic is absent and the term is
a no-op. Boundaries are zero-flux: the surface exchange is dry
deposition's job, and emission injection is the emission terms' job, so
this term is pure interior mixing and conserves each tracer's column mass
exactly.

The solve is one batched Thomas algorithm over the stacked tracers (two
``lax.scan`` sweeps over levels), so cost is independent of how many
tracers ride along.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

#: Physical floor on the layer mass [kg/m²] in divisions (a 1e-3 kg/m²
#: layer is far thinner than any real grid); keeps the guarded-division
#: VJPs clear of the float32 squared-underflow window.
_DM_FLOOR = 1.0e-3


@tree_math.struct
class TracerDiffusionParameters:
    """Tunable knob for tracer turbulent mixing (differentiable)."""

    diffusion_scale: jnp.ndarray   # multiplies the exchange coefficient

    @classmethod
    def default(cls) -> "TracerDiffusionParameters":
        return cls(diffusion_scale=jnp.asarray(1.0))


def diffuse_tracers_implicit(
    q: jnp.ndarray,             # (K, nlev, ncols) tracer stack
    kh: jnp.ndarray,            # (nlev, ncols) exchange coefficient [m²/s]
    air_density: jnp.ndarray,   # (nlev, ncols)
    layer_thickness: jnp.ndarray,  # (nlev, ncols) [m]
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Backward-Euler vertical diffusion of a tracer stack.

    Interface conductances ``g = ρ·K/Δz`` [kg/m²/s] from arithmetic
    means of the layer values; zero-flux at top and bottom. The implicit
    operator is in flux form, so each tracer's column integral
    ``Σ q·ρ·Δz`` is conserved to solver round-off for ANY ``K·Δt``, and
    the solution is positivity-preserving (an M-matrix). Returns the
    post-diffusion stack.
    """
    dm = air_density * layer_thickness
    k_int = 0.5 * (kh[:-1] + kh[1:])
    rho_int = 0.5 * (air_density[:-1] + air_density[1:])
    dz_int = 0.5 * (layer_thickness[:-1] + layer_thickness[1:])
    g_int = rho_int * k_int / dz_int                     # (nlev-1, ncols)

    zero_row = jnp.zeros_like(g_int[:1])
    g_up = jnp.concatenate([zero_row, g_int], axis=0)    # g_{k-1/2}
    g_dn = jnp.concatenate([g_int, zero_row], axis=0)    # g_{k+1/2}
    dm_safe = jnp.maximum(dm, _DM_FLOOR)
    alpha = dt * g_up / dm_safe
    gamma = dt * g_dn / dm_safe

    a = -alpha                       # sub-diagonal (coupling to k-1)
    b = 1.0 + alpha + gamma          # diagonal
    c = -gamma                       # super-diagonal (coupling to k+1)

    # Thomas forward sweep over levels; carries are the modified
    # super-diagonal (ncols,) and RHS (K, ncols).
    def forward(carry, xs):
        c_prev, d_prev = carry
        a_k, b_k, c_k, d_k = xs
        denom = b_k - a_k * c_prev
        c_new = c_k / denom
        d_new = (d_k - a_k[jnp.newaxis] * d_prev) / denom[jnp.newaxis]
        return (c_new, d_new), (c_new, d_new)

    d0 = jnp.moveaxis(q, 1, 0)                            # (nlev, K, ncols)
    (_, _), (c_mod, d_mod) = jax.lax.scan(
        forward,
        (jnp.zeros_like(a[0]), jnp.zeros_like(d0[0])),
        (a, b, c, d0),
    )

    def backward(x_next, xs):
        c_k, d_k = xs
        x_k = d_k - c_k[jnp.newaxis] * x_next
        return x_k, x_k

    _, x_rev = jax.lax.scan(
        backward,
        jnp.zeros_like(d0[0]),
        (c_mod, d_mod),
        reverse=True,
    )
    return jnp.moveaxis(x_rev, 0, 1)                      # (K, nlev, ncols)


class TracerVerticalDiffusion(PhysicsTerm):
    """Implicit turbulent vertical mixing of an explicit tracer list."""

    name: ClassVar[str] = "tracer_vertical_diffusion"
    category: ClassVar[str] = "tracer_transport"
    requires: ClassVar[tuple[str, ...]] = (
        "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        tracer_names: tuple[str, ...],
        params: TracerDiffusionParameters | None = None,
    ):
        """Hold the tracer list and params."""
        if not tracer_names:
            raise ValueError(
                "TracerVerticalDiffusion needs a non-empty tracer list."
            )
        self._tracer_names = tuple(tracer_names)
        self.params = nnx.Param(
            params or TracerDiffusionParameters.default()
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        vd = diagnostics.get("vertical_diffusion")
        zeros = jnp.zeros_like(state.temperature)
        if vd is None:
            # First step / no vdiff scheme composed: nothing to mix with.
            tracer_tends = {nm: zeros for nm in self._tracer_names}
        else:
            dt = diagnostics.get("_dt_seconds", 1800.0)
            q = jnp.stack([
                state.tracers.get(nm, zeros) for nm in self._tracer_names
            ])
            q_new = diffuse_tracers_implicit(
                q,
                params.diffusion_scale * jnp.maximum(vd.kh, 0.0),
                diagnostics["air_density"],
                diagnostics["layer_thickness"],
                dt,
            )
            dq = (q_new - q) / dt
            tracer_tends = {
                nm: dq[k] for k, nm in enumerate(self._tracer_names)
            }

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
