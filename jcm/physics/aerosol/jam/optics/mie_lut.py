"""Mie lookup table: build once (NumPy), interpolate per step (JAX).

``build_mie_lut`` tabulates the Bohren–Huffman efficiencies over a grid of
(size parameter, real index, imaginary index); ``interp_mie`` does a
differentiable trilinear interpolation into that table. The table is built at
term construction, so the expensive Mie evaluation never enters the per-step
jitted path — only the cheap, autodiff-friendly interpolation does.

Axes are uniform in ``log10(x)``, ``mr`` and ``log10(mi)`` so the fractional
grid coordinate is a simple affine map.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from jcm.physics.aerosol.jam.optics.mie import X_MAX, mie_efficiencies

_MI_FLOOR = 1.0e-9   # imaginary-index floor (non-absorbing → log axis)


@dataclasses.dataclass(frozen=True)
class MieLUT:
    """Tabulated Mie efficiencies and the affine grid mapping."""

    q_ext: jnp.ndarray   # (nx, nmr, nmi)
    ssa: jnp.ndarray
    g: jnp.ndarray
    logx0: float
    dlogx: float
    mr0: float
    dmr: float
    logmi0: float
    dlogmi: float
    shape: tuple[int, int, int]


def build_mie_lut(
    nx: int = 64, nmr: int = 24, nmi: int = 24,
    x_min: float = 0.01, x_max: float = X_MAX,
    mr_min: float = 1.30, mr_max: float = 2.60,
    mi_min: float = _MI_FLOOR, mi_max: float = 5.0,
) -> MieLUT:
    """Tabulate ``(q_ext, ssa, g)`` over the (x, mr, mi) grid (NumPy)."""
    logx = np.linspace(np.log10(x_min), np.log10(x_max), nx)
    mr = np.linspace(mr_min, mr_max, nmr)
    logmi = np.linspace(np.log10(mi_min), np.log10(mi_max), nmi)

    qe = np.empty((nx, nmr, nmi))
    ss = np.empty((nx, nmr, nmi))
    gg = np.empty((nx, nmr, nmi))
    for i, lx in enumerate(logx):
        x = 10.0 ** lx
        for j, r in enumerate(mr):
            for k, lmi in enumerate(logmi):
                qe[i, j, k], ss[i, j, k], gg[i, j, k] = mie_efficiencies(
                    x, r, 10.0 ** lmi
                )

    return MieLUT(
        q_ext=jnp.asarray(qe, jnp.float32),
        ssa=jnp.asarray(ss, jnp.float32),
        g=jnp.asarray(gg, jnp.float32),
        logx0=float(logx[0]), dlogx=float(logx[1] - logx[0]),
        mr0=float(mr[0]), dmr=float(mr[1] - mr[0]),
        logmi0=float(logmi[0]), dlogmi=float(logmi[1] - logmi[0]),
        shape=(nx, nmr, nmi),
    )


def interp_mie(lut: MieLUT, x, mr, mi):
    """Differentiable trilinear interpolation → ``(q_ext, ssa, g)``.

    ``x``, ``mr``, ``mi`` are arrays of matching shape; returns three arrays of
    that shape. Inputs are clamped to the table range.
    """
    nx, nmr, nmi = lut.shape
    cx = (jnp.log10(jnp.clip(x, 10.0 ** lut.logx0, X_MAX)) - lut.logx0) / lut.dlogx
    cr = (jnp.clip(mr, lut.mr0, lut.mr0 + (nmr - 1) * lut.dmr) - lut.mr0) / lut.dmr
    cm = (
        jnp.log10(jnp.clip(mi, 10.0 ** lut.logmi0, 10.0 ** (lut.logmi0 + (nmi - 1) * lut.dlogmi)))
        - lut.logmi0
    ) / lut.dlogmi

    shape = x.shape
    coords = jnp.stack([cx.ravel(), cr.ravel(), cm.ravel()], axis=0)

    def _interp(table):
        return map_coordinates(table, coords, order=1, mode="nearest").reshape(shape)

    return _interp(lut.q_ext), _interp(lut.ssa), _interp(lut.g)
