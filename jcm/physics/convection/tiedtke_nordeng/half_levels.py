"""Half-level environment for the Tiedtke-Nordeng scheme (ECHAM ``cuini``).

ECHAM's convection is a finite-volume scheme on the model's HALF levels: the
plume, its mass fluxes and every flux it carries live on the interfaces
between layers, and the tendency of a layer is the difference of the fluxes
through its two bounding interfaces divided by the layer's air mass
(``cudtdq``). The environment the plume entrains, is tested against and is
measured relative to is therefore the environment interpolated to those
interfaces, built once per call by ``cuini`` (mo_cuinitialize.f90:31-230).
This module is that routine.

Index convention (the physics-internal TOP-FIRST frame, vertical on axis 0):

* ``paph`` has ``nlev + 1`` entries; ``paph[i]`` is interface ``i``, with
  ``paph[0]`` the model top and ``paph[nlev]`` the surface.
* Every other half-level array has ``nlev`` entries, and entry ``i`` is the
  value at interface ``i`` — the TOP interface of layer ``i``. This is
  exactly ECHAM's ``klev``-long half-level arrays (1-based ``jk`` is 0-based
  ``i = jk - 1``); the surface interface carries no convective flux, so it
  needs no slot.
* Layer ``i`` lies between ``paph[i]`` and ``paph[i + 1]``; its air mass per
  unit area is ``dp[i] / g``.

The code is written against axis 0 only, so it runs unchanged on a single
``(nlev,)`` column or a broadcast ``(nlev, *horiz)`` block.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jcm.constants as c
from jcm.physics.convection.saturation import saturation_mixing_ratio
from .adjustment import cuadjtq


class HalfLevelEnvironment(NamedTuple):
    """The ``cuini`` environment on full and half levels (top-first).

    Attributes:
        paph: Interface pressures [Pa], ``(nlev + 1, *horiz)``; ``paph[0]`` is
            the model top, ``paph[nlev]`` the surface (ECHAM ``paphp1``).
        dp: True layer pressure thickness ``paph[1:] - paph[:-1]`` [Pa].
        geo: Full-level geopotential above the surface [m²/s²] (``pgeo``).
        dse: Full-level dry static energy ``pcpen·pten + pgeo`` [J/kg].
        geoh: Geopotential of each layer's TOP interface above the surface
            [m²/s²] (``pgeoh``); ``geoh[0] = geo[0]`` as in ``cuini``.
        tenh: Half-level environmental temperature [K] (``ptenh``).
        qenh: Half-level environmental specific humidity [kg/kg] (``pqenh``).
        qsenh: Half-level saturation specific humidity [kg/kg] (``pqsenh``).
        qsen: Full-level saturation specific humidity [kg/kg] (``pqsen``).
        cpcu: Half-level moist heat capacity [J/kg/K] (``pcpcu``).
        alvsh: Half-level latent heat keyed to ``tenh`` [J/kg] (``palvsh``).

    """

    paph: jnp.ndarray
    dp: jnp.ndarray
    geo: jnp.ndarray
    dse: jnp.ndarray
    geoh: jnp.ndarray
    tenh: jnp.ndarray
    qenh: jnp.ndarray
    qsenh: jnp.ndarray
    qsen: jnp.ndarray
    cpcu: jnp.ndarray
    alvsh: jnp.ndarray


def reconstruct_pressure_half(
    pressure: jnp.ndarray,
    layer_mass: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Interface pressures for column callers that supply only full levels.

    The model path always hands the scheme the host's true ``pressure_half``;
    this fallback serves standalone column callers (unit tests, offline
    probes). With ``layer_mass`` (``Δp/g`` per layer) the interfaces are
    stacked from the top so the layer masses are reproduced exactly, anchored
    so the top layer's full level sits mid-layer (``p = a + b·p_s`` full
    levels are interface midpoints on a hybrid grid). Without it the interior
    interfaces are full-level midpoints and the two boundaries are
    extrapolated by half a layer, clamped at zero pressure at the top.

    Args:
        pressure: Full-level pressure [Pa], ``(nlev, *horiz)``, top-first.
        layer_mass: Optional per-layer air mass ``Δp/g`` [kg/m²].

    Returns:
        ``(nlev + 1, *horiz)`` interface pressures, top-first.

    """
    if layer_mass is not None:
        dp = layer_mass * c.grav
        top = jnp.maximum(pressure[:1] - 0.5 * dp[:1], 0.0)
        return jnp.concatenate([top, top + jnp.cumsum(dp, axis=0)], axis=0)
    mid = 0.5 * (pressure[1:] + pressure[:-1])
    top = jnp.maximum(pressure[:1] - 0.5 * (pressure[1:2] - pressure[:1]), 0.0)
    bottom = pressure[-1:] + 0.5 * (pressure[-1:] - pressure[-2:-1])
    return jnp.concatenate([top, mid, bottom], axis=0)


def _log_ratio(p_lo: jnp.ndarray, p_hi: jnp.ndarray) -> jnp.ndarray:
    """``ln(p_lo / p_hi)`` for interfaces, finite where ``p_hi == 0``.

    Only the model-top interface can be at zero pressure; ECHAM never forms
    the logarithm there (``cuini`` stops at ``jk = 2`` and the top layer uses
    ``α = ln 2``). The safe substitution keeps the unused branch — and its
    VJP — finite.
    """
    positive = p_hi > 0.0
    safe_hi = jnp.where(positive, p_hi, 1.0)
    safe_lo = jnp.where(positive, p_lo, 1.0)
    return jnp.where(positive, jnp.log(safe_lo / safe_hi), 0.0)


def half_level_environment(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    pressure_half: jnp.ndarray,
    cp_moist: jnp.ndarray,
    condensate: jnp.ndarray | None = None,
) -> HalfLevelEnvironment:
    """ECHAM ``cuini``: interpolate the environment to half levels.

    Faithful port of mo_cuinitialize.f90:31-230 in the top-first frame
    described in the module docstring:

    * **Geopotential.** ``pgeoh`` is the hydrostatic integral
      ``Σ R_d·T_v·ln(p_{k+1}/p_k)`` up from the surface with the virtual
      temperature ``T·(1 + vtmpc1·q − x)`` (``ztvp1``, condensate-loaded),
      and ``pgeoh(1) = pgeo(1)``. The full-level ``pgeo`` is ECHAM's
      ``geopot`` of the same column, ``pgeoh(k+1) + R_d·T_v·α_k`` with
      ``α_k = 1 − p_k/Δp_k·ln(p_{k+1}/p_k)`` and ``α = ln 2`` for a top layer
      that reaches zero pressure — so the half and full geopotentials are
      one self-consistent hydrostatic column, both above the surface.
    * **Heat capacity.** ``pcpcu`` is the mean of the two adjacent full-level
      ``pcpen`` (``pcpcu(1) = pcpen(1)``).
    * **Temperature.** ``ptenh`` carries the larger of the two adjacent
      full-level dry static energies to the interface. It is then saturation
      adjusted (``cuadjtq`` with ``kcall = 0``) together with the saturation
      humidity of the level above, which yields the half-level saturation
      humidity ``pqsenh``, and finally a bottom-up running maximum of the
      half-level dry static energy is enforced (the ``zzs`` loop), so the
      interface profile is never dry-unstable.
    * **Humidity.** ``pqenh = min(q, qs)`` of the level above plus the change
      in saturation humidity from that level to the interface — the moist
      interpolation — floored at zero.
    * **Boundaries.** At the lowest interface (top of the bottom layer) the
      bottom full level's dry static energy is carried up to it and
      ``pqenh = pqen``; at the model top the full-level values are used.
    * ``palvsh`` is ``alv`` above ``tmelt`` and ``als`` below, keyed to
      ``ptenh``.

    Args:
        temperature: Full-level temperature [K], ``(nlev, *horiz)``.
        humidity: Full-level specific humidity [kg/kg].
        pressure: Full-level pressure [Pa] (``papp1``), for ``pqsen``.
        pressure_half: Interface pressures [Pa], ``(nlev + 1, *horiz)``.
        cp_moist: Full-level moist heat capacity [J/kg/K] (``pcpen``).
        condensate: Full-level cloud condensate ``qc + qi`` [kg/kg] for the
            virtual temperature (``zxp1``); ``None`` means condensate-free.

    Returns:
        The :class:`HalfLevelEnvironment`.

    """
    paph = pressure_half
    dp = paph[1:] - paph[:-1]
    x = jnp.zeros_like(temperature) if condensate is None else jnp.maximum(
        condensate, 0.0)
    tv = temperature * (1.0 + c.vtmpc1 * humidity - x)

    # --- geopotential (cuini lines 97-110, and ECHAM geopot) -------------
    # ln(p_bottom/p_top) of every layer; zero for a layer whose top is at
    # zero pressure, whose geoh contribution ECHAM never forms.
    lnp = _log_ratio(paph[1:], paph[:-1])
    dgeo = c.rd * tv * lnp
    # geoh_bottom[k] = geopotential of layer k's BOTTOM interface: the sum of
    # the increments of every layer below it (zero for the surface layer).
    below = jnp.cumsum(dgeo[::-1], axis=0)[::-1]
    geoh_bottom = below - dgeo
    top_is_zero = paph[:-1] <= 0.0
    safe_dp = jnp.where(dp > 0.0, dp, 1.0)
    alpha = jnp.where(
        top_is_zero, jnp.log(2.0), 1.0 - paph[:-1] / safe_dp * lnp,
    )
    geo = geoh_bottom + c.rd * tv * alpha
    geoh = jnp.concatenate([geo[:1], below[1:]], axis=0)

    # --- heat capacity at half levels ------------------------------------
    cpcu = jnp.concatenate(
        [cp_moist[:1], 0.5 * (cp_moist[1:] + cp_moist[:-1])], axis=0,
    )

    qsen = saturation_mixing_ratio(pressure, temperature)

    # --- interior interfaces 1..nlev-1 (cuini lines 111-151) -------------
    dse = cp_moist * temperature + geo
    tenh_int = (jnp.maximum(dse[:-1], dse[1:]) - geoh[1:]) / cpcu[1:]
    # cuadjtq(kcall=0) on (ptenh, qsen of the level above) at the interface
    # pressure — adjusts BOTH in place, as the Fortran does.
    tenh_adj, qsenh_int, _ = cuadjtq(
        tenh_int, qsen[:-1], paph[1:-1], kcall=0,
    )
    qenh_int = jnp.maximum(
        jnp.minimum(humidity[:-1], qsen[:-1]) + (qsenh_int - qsen[:-1]), 0.0,
    )

    # --- boundaries (cuini lines 153-167) ---------------------------------
    tenh_bottom = (dse[-1:] - geoh[-1:]) / cp_moist[-1:]
    tenh = jnp.concatenate(
        [temperature[:1], tenh_adj[:-1], tenh_bottom], axis=0,
    )
    qenh = jnp.concatenate([humidity[:1], qenh_int[:-1], humidity[-1:]],
                           axis=0)
    qsenh = jnp.concatenate([qsen[:1], qsenh_int], axis=0)

    # --- dry-static-energy running maximum from below (lines 175-181) ----
    # zzs = MAX(s_h(jk), s_h(jk+1)) sequentially from klevm1 up to 2 is a
    # cumulative maximum of the half-level DSE seeded at the bottom
    # interface; the top interface is left as the full-level value.
    sh = cpcu * tenh + geoh
    sh_max = jnp.flip(
        _cummax(jnp.flip(sh[1:], axis=0)), axis=0,
    )
    tenh = jnp.concatenate(
        [tenh[:1], (sh_max - geoh[1:]) / cpcu[1:]], axis=0,
    )

    alvsh = jnp.where(tenh > c.tmelt, c.alhc, c.alhs)
    return HalfLevelEnvironment(
        paph=paph, dp=dp, geo=geo, dse=dse, geoh=geoh, tenh=tenh, qenh=qenh,
        qsenh=qsenh, qsen=qsen, cpcu=cpcu, alvsh=alvsh,
    )


def _cummax(a: jnp.ndarray) -> jnp.ndarray:
    """Cumulative maximum along axis 0 (``lax.cummax`` for any rank)."""
    from jax import lax
    return lax.cummax(a, axis=0)
