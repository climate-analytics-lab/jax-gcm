"""Saturation of the Tiedtke-Nordeng scheme: ECHAM's ``ua`` table and ``cuadjtq``.

ECHAM's convection reads saturation from the ``ua`` lookup table everywhere
(``cumastr``, ``cuini``, ``cubase``, ``cuasc``, ``cudlfs``, ``cuddraf`` and
``cuadjtq``: ``lookup_ua_spline`` / ``lookup_ua_list_spline``, e.g.
``mo_cumastr.f90`` l.220-221 and 601-603, ``mo_cuadjust.f90`` l.85-86): Sonntag
(1990) over ice at and below the melting point and over water above
(:func:`jcm.physics.thermodynamics.es_ua`). ``cuadjtq`` takes ``L/cp`` from
``lookup_ubc`` with the same switch (``als/cpd`` at and below ``tmelt``,
``alv/cpd`` above; ``mo_echam_convect_tables.f90`` l.329-333).

:func:`cuadjtq` is the port of ``mo_cuadjust.f90::cuadjtq`` (l.83-213) and the
one saturation adjustment of the scheme: ``cuini`` (``kcall = 0``), ``cubase``
and ``cuasc`` (``kcall = 1``, through :func:`cuadjtq_newton`) and ``cudlfs``
and ``cuddraf`` (``kcall = 2``, :func:`cuadjtq_newton_evap` for the wet bulb).
It is verified against ECHAM's compiled routine in ``cuadjtq_test.py``
(``jcm/data/test/echam_cuadjtq_reference/``).

The functions live in this leaf module, which imports only
:mod:`jcm.physics.thermodynamics`, so that ``tiedtke_nordeng.py`` (trigger,
CAPE), ``updraft.py``, ``downdraft.py`` and ``half_levels.py`` can all call them
without an import cycle.

They are intentionally not ``@jit``-ed: they run inside the model's outer jit
and inline there.
"""

import jax.numpy as jnp

import jcm.constants as c
from jcm.physics import thermodynamics


def saturation_mixing_ratio(pressure: jnp.ndarray,
                            temperature: jnp.ndarray) -> jnp.ndarray:
    """Saturation specific humidity [kg/kg] from ECHAM's ``ua`` table.

    ``x = MIN(ua/p, 0.5)``, ``qs = x/(1 − vtmpc1·x)`` (``mo_cumastr.f90``
    l.220-226, ``mo_cuadjust.f90`` l.107-110). The argument order
    ``(pressure, temperature)`` is the scheme's historical one.
    """
    return thermodynamics.saturation_specific_humidity(
        temperature, pressure, phase="auto")


def lcp_ua(temperature):
    """``L/cp`` of ``lookup_ubc``: ``als/cpd`` at and below ``tmelt``, else ``alv/cpd``.

    DRY ``cpd`` is the reference: ``cuadjtq`` reads ``L/cp`` from ``uc``,
    built with ``zalvdcp = alv/cpd``, ``zalsdcp = als/cpd``
    (``mo_echam_convect_tables.f90`` l.214-215, 323-324), not the moist
    ``zcpq`` of the ``cumastr`` static-energy ledger. The switch is the ``ua``
    table's, so the latent heat always pairs with the saturation surface.
    """
    return jnp.where(thermodynamics.ua_ice_phase(temperature),
                     c.alhs, c.alhc) / c.cpd


def _newton_step(temperature, humidity, pressure):
    """One ``cuadjtq`` Newton step: ``(zcond, L/cp)`` at ``(T, q, p)``, unclipped.

    ``zcond = (q − qs)/(1 + zlcdqsdt)`` with ``zlcdqsdt = (L/cp)·dqs/dT``
    (``mo_cuadjust.f90`` l.107-115). ECHAM forms ``zlcdqsdt`` as
    ``zdqsdt·uc`` where ``zes < 0.4`` and as ``zqsat·zcor·ub`` above; both are
    ``(L/cp)·dqs/dT`` of the capped ``zes``, which is what
    :func:`~jcm.physics.thermodynamics.dqsat_dT_from_es` returns.
    """
    l_cp = lcp_ua(temperature)
    qs, dqs_dt = thermodynamics.saturation_specific_humidity_and_derivative(
        temperature, pressure, phase="auto")
    return (humidity - qs) / (1.0 + l_cp * dqs_dt), l_cp


def cuadjtq(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    kcall: int = 1,
    refine: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ECHAM's convective saturation adjustment, ``mo_cuadjust.f90::cuadjtq``.

    Two Newton steps towards ``q = qs(T)`` along ``cp·dT + L·dq = 0``, each

        zcond = (q − qs(T)) / (1 + (L/cp)·dqs/dT),
        T ← T + (L/cp)·zcond,   q ← q − zcond,

    with ``qs``, ``dqs/dT`` and ``L/cp`` of the ``ua`` table at the current
    temperature (:func:`lcp_ua`). The first step is clipped by ``kcall``
    (l.97-172):

    * ``kcall = 0`` — ``cuini``'s half-level environment: both signs;
    * ``kcall = 1`` — ``cubase``/``cuasc`` updraft: ``MAX(zcond, 0)``,
      condensation only;
    * ``kcall = 2`` — ``cudlfs``/``cuddraf`` downdraft: ``MIN(zcond, 0)``,
      evaporation only.

    The second step refines the linearisation (l.176-211). It runs only where
    the first step was non-zero (``ncond``), so air the clip left untouched
    stays untouched, and it is **unclipped** for every ``kcall``: it corrects
    the first step's overshoot in either direction.

    Args:
        temperature: Temperature [K].
        specific_humidity: Specific humidity [kg/kg].
        pressure: Pressure [Pa].
        kcall: ``0``, ``1`` or ``2`` as above; a static Python int.
        refine: Take the second step. ``False`` returns the first step alone,
            which ECHAM never does; it exists for tests of that step.

    Returns:
        ``(T_adj, q_adj, condensate)`` with ``condensate = q − q_adj``, the
        vapour the two steps removed (ECHAM's callers form it as
        ``zqold − pqu``). It has the sign of the first step, except within
        ~1 mK of ``tmelt``, where a first step that crosses the melting
        point is refined on the other phase's table (``es`` steps by ~1e-4
        there) and the unclipped refinement can outweigh it.

    """
    if kcall not in (0, 1, 2):
        raise ValueError(f"kcall must be 0, 1 or 2, got {kcall!r}")
    cond1, l_cp1 = _newton_step(temperature, specific_humidity, pressure)
    if kcall == 1:
        cond1 = jnp.maximum(cond1, 0.0)
    elif kcall == 2:
        cond1 = jnp.minimum(cond1, 0.0)
    t1 = temperature + l_cp1 * cond1
    q1 = specific_humidity - cond1
    if refine:
        # ``ncond = INT(FSEL(-ABS(zcond), 0, 1))``: refine where zcond /= 0.
        active = cond1 != 0.0
        cond2, l_cp2 = _newton_step(t1, q1, pressure)
        cond2 = jnp.where(active, cond2, 0.0)
        t1 = t1 + l_cp2 * cond2
        q1 = q1 - cond2
    return t1, q1, specific_humidity - q1


def cuadjtq_newton(
    temperature: jnp.ndarray,
    total_water: jnp.ndarray,
    pressure: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Condense an updraft parcel to saturation: :func:`cuadjtq`, ``kcall = 1``.

    A parcel lifted with vapour ``total_water`` and no condensate is brought
    to saturation as ``cubase`` (``mo_cuinitialize.f90`` l.302-322) and
    ``cuasc`` (``mo_cuascent.f90`` l.431-446) do: ``cuadjtq`` with
    ``kcall = 1``, and the condensate gains the vapour it removed,
    ``plu + zqold − pqu``, only ``IF (pqu < zqold)`` — the test that also
    marks the level as condensing (``klab = 2``). A subsaturated parcel is
    returned unchanged. Total water is conserved wherever vapour was removed;
    within ~1 mK of ``tmelt``, where the refinement can return more vapour
    than the parcel had (see :func:`cuadjtq`), ECHAM keeps the adjusted
    temperature and vapour and leaves the condensate alone, and so does
    this.

    Args:
        temperature: Temperature [K].
        total_water: Vapour of the unadjusted parcel [kg/kg].
        pressure: Pressure [Pa].

    Returns:
        ``(T_adj, vapour, liquid)``: ``liquid = total_water − vapour`` where
        that is positive, else 0.

    """
    t_adj, vapour, removed = cuadjtq(temperature, total_water, pressure,
                                     kcall=1)
    liquid = jnp.where(vapour < total_water, removed, 0.0)
    return t_adj, vapour, liquid


def cuadjtq_newton_evap(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bring air to its wet bulb: :func:`cuadjtq` with ``kcall = 2``.

    ``cudlfs`` brings the half-level environment to its wet bulb this way
    (``mo_cudescent.f90`` l.135-140): evaporation towards saturation, with
    ``cp·ΔT + L·Δq = 0`` in each step, so moist static energy (with the
    ``ua`` table's ``L``) is conserved; air at or above saturation comes back
    unchanged, because the first step's evaporation-only clip zeroes it and
    the refinement then does not run.

    Args:
        temperature: Temperature [K].
        humidity: Specific humidity [kg/kg].
        pressure: Pressure [Pa].

    Returns:
        ``(T_wb, q_wb)``.

    """
    t_wb, q_wb, _ = cuadjtq(temperature, humidity, pressure, kcall=2)
    return t_wb, q_wb
