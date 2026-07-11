"""U.S. Standard Atmosphere 1976 profiles for CAM-SE initial states.

Why USSA-1976 and not the analytic baroclinic-wave base state
-------------------------------------------------------------
The full ICON/ECHAM L47 column (finite-top variant, see
:func:`jcm.dycore.pyses.coords.full_echam_hybrid`) tops out near ~1 Pa
(~78 km geometric). The Jablonowski-Williamson-style analytic baroclinic base
state extrapolates to sub-100 K — and at some columns *negative* —
temperatures that high, which NaNs the pressure→height inversion in pyses's
``init_analytic_state``. The piecewise-linear USSA-1976 temperature profile
is positive and well defined all the way to its 84.852 km table top, so a
horizontally uniform USSA column **at rest** (u = v = 0, dry) over the real
orography is the default initial state for this backend: the model spins its
own circulation and moisture up from the prescribed SST / radiation.

Both :func:`ussa_temperature` and :func:`ussa_pressure` are jnp-traceable and
vectorised over ``z``; :func:`ussa_pressure` is monotone-decreasing in ``z``
so the initializer's bisection (pressure→height) is well posed.

The USSA constants are kept module-local (not read from ``jcm.constants``):
they are part of the *definition* of the published 1976 standard atmosphere
(g0 = 9.80665, Rs = 287.053, effective radius 6356766 m), not tunables of the
running model, and must not drift if a user overrides the model's constants.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# --- U.S. Standard Atmosphere 1976 definition -------------------------------
_USSA_G0 = 9.80665       # standard gravity (m/s^2)
_USSA_RS = 287.053       # dry-air specific gas constant (J/kg/K)
_USSA_RE = 6356766.0     # US-1976 effective Earth radius for z -> geopotential H
_USSA_T0 = 288.15        # sea-level temperature (K)
_USSA_P0 = 101325.0      # sea-level pressure (Pa)
_USSA_H_TOP = 84852.0    # top of the tabulated profile (geopotential m)
# Layer base geopotential heights (m) and lapse rates dT/dH (K/m).
_USSA_HB = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
_USSA_LR = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])


def _ussa_base_tables():
    """Temperature/pressure at every USSA layer edge (hydrostatic, host-side)."""
    edges = np.concatenate([_USSA_HB, [_USSA_H_TOP]])
    Tb = [_USSA_T0]
    pb = [_USSA_P0]
    for i, L in enumerate(_USSA_LR):
        dH = edges[i + 1] - edges[i]
        T_next = Tb[i] + L * dH
        if L == 0.0:
            p_next = pb[i] * np.exp(-_USSA_G0 * dH / (_USSA_RS * Tb[i]))
        else:
            p_next = pb[i] * (T_next / Tb[i]) ** (-_USSA_G0 / (_USSA_RS * L))
        Tb.append(T_next)
        pb.append(p_next)
    return np.array(Tb), np.array(pb), edges


_USSA_TB, _USSA_PB, _USSA_EDGES = _ussa_base_tables()


def _ussa_geopotential_height(z):
    """Geometric height ``z`` (m) -> USSA geopotential height ``H`` (m)."""
    return _USSA_RE * z / (_USSA_RE + z)


def ussa_temperature(z):
    """U.S. Standard Atmosphere 1976 temperature (K) at geometric height ``z`` (m).

    Piecewise linear in geopotential height; clamped to the table range so
    values above 84.852 km hold the top-layer temperature (positive
    everywhere — min ~187 K).
    """
    H = jnp.clip(_ussa_geopotential_height(z), 0.0, _USSA_H_TOP)
    T = jnp.full_like(H, float(_USSA_TB[-1]))
    for i in range(len(_USSA_LR)):
        in_layer = (H >= _USSA_EDGES[i]) & (H < _USSA_EDGES[i + 1])
        T = jnp.where(
            in_layer,
            float(_USSA_TB[i]) + float(_USSA_LR[i]) * (H - float(_USSA_EDGES[i])),
            T,
        )
    return T


def ussa_pressure(z):
    """U.S. Standard Atmosphere 1976 pressure (Pa) at geometric height ``z`` (m).

    Monotone-decreasing in ``z`` (so the initializer's pressure->height
    inversion is well posed); above the table top the profile is held at the
    top-edge pressure, which the finite-top L47 grid never reaches.
    """
    H = jnp.clip(_ussa_geopotential_height(z), 0.0, _USSA_H_TOP)
    p = jnp.full_like(H, float(_USSA_PB[-1]))
    for i in range(len(_USSA_LR)):
        L = float(_USSA_LR[i])
        Tb = float(_USSA_TB[i])
        pb = float(_USSA_PB[i])
        Hb = float(_USSA_EDGES[i])
        if L == 0.0:
            p_layer = pb * jnp.exp(-_USSA_G0 * (H - Hb) / (_USSA_RS * Tb))
        else:
            p_layer = pb * ((Tb + L * (H - Hb)) / Tb) ** (-_USSA_G0 / (_USSA_RS * L))
        in_layer = (H >= _USSA_EDGES[i]) & (H < _USSA_EDGES[i + 1])
        p = jnp.where(in_layer, p_layer, p)
    return p
