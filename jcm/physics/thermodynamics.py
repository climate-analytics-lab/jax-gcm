"""Saturation thermodynamics of jcm's ECHAM physics: Sonntag (1990).

The one place ECHAM physics in jcm takes its saturation vapour pressure,
saturation specific humidity and their temperature derivatives from:
Tiedtke-Nordeng convection (including its ``cuadjtq`` saturation adjustment),
the Sundqvist cloud cover, the 1M and 2M cloud schemes, the TTE-TKE vertical
diffusion and the ECHAM surface tiles. No ECHAM scheme carries coefficients of
its own; the one other saturation form in the ECHAM surface,
``surface/echam/turbulent_fluxes.py::compute_surface_humidity``, feeds only
tile diagnostics that no tendency reads.

**The formula.** ECHAM6.3 (r7492) takes saturation from lookup tables built in
``mo_echam_convect_tables.f90::init_convect_tables`` (l.223-309). They tabulate
the five-term fit of Sonntag (1990, *Z. Meteorol.* 70, 340-344)

    ln es(T) = a1/T + a2 + a3·0.01·T + a4·1e-5·T² + a5·ln T        [es in Pa]

with separate coefficients over liquid water (``cavl1..5``, l.42-46) and over
ice (``cavi1..5``, l.48-52), and interpolate them with a cubic Hermite spline on
0.025 K knots whose knot slopes are the analytic ones (``fetch_ua_spline``,
l.394-437); the 2M scheme reads the 0.001 K tables at the nearest knot. The
functions here evaluate the fit itself. Against ECHAM's own spline tables on a
1e-4 K scan from 150 to 330 K they differ by at most 4e-12 in ``es`` and
1.8e-9 in ``d ln es/dT`` (``jcm/data/test/echam_saturation_tables/``,
``thermodynamics_test.py``), so they are ECHAM's formulation.

**The two tables and the phase rule.** ECHAM stores two tables, both holding
``es·rd/rv``:

* ``ua`` (``tlucua``, the spline table ``tlucu``) takes the **ice** fit at and
  below the melting point and the water fit above it: :func:`es_ua`,
  :func:`ua_ice_phase`. Convection, the vertical diffusion, the surface tiles
  and every "ice" saturation of the 2M scheme read this table.
* ``uaw`` (``tlucuaw``, ``tlucuw``) takes the **water** fit at all
  temperatures: :func:`es_water`.

The cloud cover and the condensation of the cloud schemes choose between the
two per cell with ECHAM's ``lo2`` switch (ice where ``T < cthomi``, or where
``T < tmelt`` and the scheme's ice criterion holds); those schemes form the
switch themselves and pick :func:`es_ice` or :func:`es_water` with it.

**The jump at the melting point.** The two fits meet at 273.16002 K (the
triple point), not at ``tmelt = 273.15 K``. Switching there, as ECHAM does,
steps ``es`` by −9.7e-5 of its value (0.059 Pa, the change a 1.3 mK warming
makes) and ``d ln es/dT`` by +13 %. Automatic differentiation returns each
side's analytic slope, which is the slope ECHAM's derivative tables hold at
that point. Both one-sided derivatives exist and are bounded, and the step is
too small to carry information a smooth surrogate gradient could add, so the
switch is differentiated as it stands.

**Saturation specific humidity** is formed as ECHAM forms it
(``mo_cuadjust.f90`` l.107-111, ``mo_cover.f90`` l.221-223,
``mo_cloud_micro_2m.f90::sat_spec_hum``)::

    x  = MIN(es·rd/rv / p, 0.5)
    qs = x / (1 − vtmpc1·x)

which equals ``eps·es/(p − (1 − eps)·es)`` with ``eps = rd/rv`` below the cap
(:func:`qsat_from_es`), and its temperature derivative as ``cuadjtq`` and
``mo_cloud`` form it, ``dqs/dT = (1/p)·zcor²·d(es·rd/rv)/dT`` with
``zcor = 1/(1 − vtmpc1·x)`` (:func:`dqsat_dT_from_es`). The constants are read
from :mod:`jcm.constants` at call time, so ``set_constants`` overrides are
honoured.

All functions are pure JAX and broadcasting-native. They are not ``@jit``-ed:
they always run inside a caller's compiled graph and inline there. ``phase``
arguments are static Python strings resolved at trace time.
"""

import jax.numpy as jnp

import jcm.constants as c

#: Sonntag (1990) over liquid water: ``cavl1..cavl5``,
#: ``mo_echam_convect_tables.f90`` l.42-46.
WATER_COEFFICIENTS = (-6096.9385, 21.2409642, -2.711193, 1.673952, 2.433502)
#: Sonntag (1990) over ice: ``cavi1..cavi5``, ``mo_echam_convect_tables.f90``
#: l.48-52.
ICE_COEFFICIENTS = (-6024.5282, 29.32707, 1.0613868, -1.3198825, -0.49382577)

#: Bounds of ECHAM's tables (``tlbound``/``tubound``, l.108-109). ECHAM stops
#: with a lookup error outside them; here the temperature is clipped to them,
#: which changes nothing inside and keeps the fit finite outside (the clip
#: also zeroes the temperature gradient there).
ECHAM_TABLE_T_MIN = 50.0
ECHAM_TABLE_T_MAX = 400.0

# ECHAM's cap on es·rd/rv/p (``MIN(0.5, zes)``, mo_cuadjust.f90 l.108): it
# keeps qs finite where es approaches p (very warm air at very low pressure);
# 0.5 is far above any physical value, so it is inactive in the atmosphere.
_X_MAX = 0.5

# Floor under the pressure the qs forms divide by [Pa]. Callers hand in the
# model-top interface, where p = 0; there ECHAM's form is already at its cap,
# but a bare 1/p would make the derivative through the cap 0·inf = NaN. The
# floor is far below the top full level (~1 Pa), so no value changes.
_P_MIN = 1.0e-3


# --- Sonntag (1990), the fit ECHAM's tables hold ---------------------------

def _ln_es(temperature, coefficients):
    a1, a2, a3, a4, a5 = coefficients
    t = jnp.clip(temperature, ECHAM_TABLE_T_MIN, ECHAM_TABLE_T_MAX)
    return a1 / t + a2 + a3 * 0.01 * t + a4 * 1.0e-5 * t * t + a5 * jnp.log(t)


def _dln_es_dT(temperature, coefficients):
    a1, _, a3, a4, a5 = coefficients
    t = jnp.clip(temperature, ECHAM_TABLE_T_MIN, ECHAM_TABLE_T_MAX)
    return -a1 / (t * t) + a3 * 0.01 + a4 * 2.0e-5 * t + a5 / t


def es_water(temperature):
    """Saturation vapour pressure over liquid water [Pa] (ECHAM ``uaw``·rv/rd).

    Sonntag (1990), the fit ``mo_echam_convect_tables.f90`` tabulates in
    ``tlucuaw``/``tlucuw``. Temperature [K] is clipped to ECHAM's table range
    [50, 400] K.
    """
    return jnp.exp(_ln_es(temperature, WATER_COEFFICIENTS))


def es_ice(temperature):
    """Saturation vapour pressure over ice [Pa], Sonntag (1990).

    The ice fit ECHAM tabulates in ``tlucua``/``tlucu`` at and below the
    melting point, here evaluated at every temperature. Temperature [K] is
    clipped to ECHAM's table range [50, 400] K.
    """
    return jnp.exp(_ln_es(temperature, ICE_COEFFICIENTS))


def dlnes_dT_water(temperature):
    """``d ln(es_water)/dT`` [1/K], the analytic slope ECHAM tabulates."""
    return _dln_es_dT(temperature, WATER_COEFFICIENTS)


def dlnes_dT_ice(temperature):
    """``d ln(es_ice)/dT`` [1/K], the analytic slope ECHAM tabulates."""
    return _dln_es_dT(temperature, ICE_COEFFICIENTS)


# --- ECHAM's ``ua`` table ---------------------------------------------------

def ua_ice_phase(temperature):
    """Return ``True`` where ECHAM's ``ua`` table holds the ice fit: ``T <= tmelt``.

    ``init_convect_tables`` builds the ice branch where ``T − tmelt <= 0``
    (l.226) and ``prepare_ua_index_spline`` shifts the index with
    ``FSEL(tmelt − T, 1, 0)`` (l.657) so that ``T == tmelt`` reads ice;
    ``lookup_ubc`` switches its latent heat with the same ``FSEL`` (l.329-333).
    Schemes that pair a latent heat with a ``ua`` saturation switch it here.
    """
    return temperature <= c.tmelt


def es_ua(temperature):
    """Saturation vapour pressure of ECHAM's ``ua`` table [Pa].

    Ice at and below the melting point, water above (:func:`ua_ice_phase`).
    """
    return jnp.where(ua_ice_phase(temperature),
                     es_ice(temperature), es_water(temperature))


def dlnes_dT_ua(temperature):
    """``d ln(es_ua)/dT`` [1/K]: each side's analytic slope (ECHAM ``dua/ua``)."""
    return jnp.where(ua_ice_phase(temperature),
                     dlnes_dT_ice(temperature), dlnes_dT_water(temperature))


# --- Saturation specific humidity -------------------------------------------

def qsat_from_es(es, pressure):
    """Saturation specific humidity [kg/kg] from ``es`` [Pa], as ECHAM forms it.

    ``x = MIN(es·rd/rv/p, 0.5)``, ``qs = x/(1 − vtmpc1·x)``
    (``mo_cover.f90`` l.221-223). With ``vtmpc1 = rv/rd − 1`` the denominator
    is at least ``1 − 0.5·vtmpc1 > 0``, so no further guard is needed.
    """
    x = jnp.minimum(es * (c.rd / c.rv) / jnp.maximum(pressure, _P_MIN),
                    _X_MAX)
    return x / (1.0 - c.vtmpc1 * x)


def dqsat_dT_from_es(es, des_dT, pressure):
    """``dqs/dT`` [kg/kg/K] of :func:`qsat_from_es`, as ECHAM forms it.

    ``cuadjtq``'s slope (``mo_cuadjust.f90`` l.107-113): ``zdqsdt =
    (1/p)·zcor²·dua`` with ``dua = des/dT·rd/rv`` and ``zcor = 1/(1 −
    vtmpc1·x)`` from the capped ``x`` where ``x < 0.4``, and
    ``qs·zcor·d ln es/dT`` (the ``ub`` branch) above, which keeps the capped
    ``x`` in the slope. Below the cap both are the analytic derivative, and
    equal the ``(1/p)·zcor²·dua`` that ``mo_cloud`` and ``precalc_land`` use
    everywhere; at the cap (the top few levels) those keep the
    uncapped ``dua/p``, as the 1M scheme's condensation does
    (``mo_cloud.f90`` l.700-704).

    Args:
        es: Saturation vapour pressure [Pa].
        des_dT: Its temperature derivative [Pa/K].
        pressure: Pressure [Pa].

    """
    k = (c.rd / c.rv) / jnp.maximum(pressure, _P_MIN)
    uncapped = es * k < _X_MAX
    x = jnp.where(uncapped, es * k, _X_MAX)
    zcor = 1.0 / (1.0 - c.vtmpc1 * x)
    # Both of ECHAM's branches are zcor²·des/dT·(x/es): ``dua/p`` carries
    # x/es = rd/rv/p, and ``qs·zcor·ub/uc`` carries the capped x/es = 0.5/es.
    # The double ``where`` keeps the capped branch from dividing by an ``es``
    # that underflows to zero in float32 at the coldest temperatures, whose
    # 0/0 would poison reverse-mode AD even where it is not selected.
    es_capped = jnp.where(uncapped, 1.0, es)
    x_over_es = jnp.where(uncapped, k, _X_MAX / es_capped)
    return zcor * zcor * des_dT * x_over_es


# --- Phase-selected interface -----------------------------------------------

def _validate_phase(phase: str) -> None:
    if phase not in ("auto", "water", "ice"):
        raise ValueError(
            f"phase must be 'auto', 'water' or 'ice', got {phase!r}")


def _es_and_dlnes(temperature, phase):
    if phase == "water":
        return es_water(temperature), dlnes_dT_water(temperature)
    if phase == "ice":
        return es_ice(temperature), dlnes_dT_ice(temperature)
    return es_ua(temperature), dlnes_dT_ua(temperature)


def saturation_vapor_pressure(temperature: jnp.ndarray,
                              phase: str = "auto") -> jnp.ndarray:
    """Saturation vapour pressure ``es(T)`` [Pa], Sonntag (1990).

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K]. Clipped to ECHAM's table range [50, 400] K.
    phase : str
        ``"auto"``: ECHAM's ``ua`` table, ice at and below ``tmelt`` and
        water above (:func:`es_ua`). ``"water"``: ECHAM's ``uaw``, liquid
        water at all temperatures. ``"ice"``: the ice fit at all
        temperatures.

    Returns
    -------
    jnp.ndarray
        Saturation vapour pressure [Pa].

    """
    _validate_phase(phase)
    return _es_and_dlnes(temperature, phase)[0]


def saturation_specific_humidity(temperature: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 phase: str = "auto") -> jnp.ndarray:
    """Saturation specific humidity ``qs(T, p)`` [kg/kg], as ECHAM forms it.

    :func:`qsat_from_es` of :func:`saturation_vapor_pressure`.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K].
    pressure : jnp.ndarray
        Pressure [Pa].
    phase : str
        Saturation surface, see :func:`saturation_vapor_pressure`.

    Returns
    -------
    jnp.ndarray
        Saturation specific humidity [kg/kg].

    """
    return qsat_from_es(saturation_vapor_pressure(temperature, phase=phase),
                        pressure)


def saturation_specific_humidity_and_derivative(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    phase: str = "auto",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(qs, dqs/dT)`` with ECHAM's analytic temperature derivative.

    The slope is the one ECHAM's Newton adjustments use
    (:func:`dqsat_dT_from_es` of the phase's ``es`` and ``des/dT``), in
    closed form so a Newton step is reproducible under JIT without
    differentiating through the cap.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K].
    pressure : jnp.ndarray
        Pressure [Pa].
    phase : str
        Saturation surface, see :func:`saturation_vapor_pressure`.

    Returns
    -------
    tuple of jnp.ndarray
        ``(qs, dqs_dT)`` in [kg/kg] and [kg/kg/K].

    """
    _validate_phase(phase)
    es, dlnes = _es_and_dlnes(temperature, phase)
    return qsat_from_es(es, pressure), dqsat_dT_from_es(es, es * dlnes,
                                                        pressure)


# --- Phase blending and in-cloud helpers ------------------------------------

def mixed_phase_weight(temperature: jnp.ndarray,
                       t_min: float = 238.15,
                       t_max: float | None = None) -> jnp.ndarray:
    """Linear liquid fraction for mixed-phase blending.

    Returns ``clip((T − t_min) / (t_max − t_min), 0, 1)`` — 1 for pure
    liquid at/above ``t_max``, 0 for pure ice at/below ``t_min``. Pair it
    with the pure-phase :func:`es_water` / :func:`es_ice`.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K].
    t_min : float
        Temperature at/below which the cloud is all ice [K]. Default
        238.15 K (−35 °C, near the homogeneous-freezing threshold).
    t_max : float, optional
        Temperature at/above which the cloud is all liquid [K]. Defaults
        to ``c.tmelt`` (read at call time so constant overrides apply).

    Returns
    -------
    jnp.ndarray
        Liquid fraction in [0, 1].

    """
    if t_max is None:
        t_max = c.tmelt
    return jnp.clip((temperature - t_min) / (t_max - t_min), 0.0, 1.0)


def grid_mean_to_in_cloud(x: jnp.ndarray,
                          cloud_fraction: jnp.ndarray,
                          eps: float = 1e-12) -> jnp.ndarray:
    """Convert a grid-mean quantity to its in-cloud value.

    ``x / cloud_fraction`` where a cloud is present (``cf > eps``), 0
    elsewhere. The ``maximum`` in the denominator keeps the masked-out
    branch's gradient finite (a bare ``x / cf`` would divide by ~0 there
    and poison reverse-mode AD even though the value is masked).

    Parameters
    ----------
    x : jnp.ndarray
        Grid-mean quantity (e.g. cloud water [kg/kg]).
    cloud_fraction : jnp.ndarray
        Cloud fraction in [0, 1].
    eps : float
        Presence threshold and division floor.

    Returns
    -------
    jnp.ndarray
        In-cloud value, 0 where ``cloud_fraction <= eps``.

    """
    return jnp.where(cloud_fraction > eps,
                     x / jnp.maximum(cloud_fraction, eps),
                     0.0)


def moist_isobaric_heat_capacity(specific_humidity: jnp.ndarray) -> jnp.ndarray:
    """Isobaric specific heat of moist air ``cp = cpd·(1 + vtmpc2·max(q, 0))``.

    ECHAM's humidity-weighted heat capacity, written exactly as the
    reference forms it — ``zcpq = cpd·(1 + vtmpc2·MAX(pqm1, 0))``
    (``mo_cumastr.f90:229``) for convection and ``zcair = cpd + cpd·vtmpc2·
    MAX(qm1, 0)`` (``physc.f90:289``) for the cloud schemes; ``cpd·vtmpc2``
    is ``cpv − cpd``. ECHAM evaluates it at the step-start humidity and
    divides every latent-heat and static-energy conversion of those schemes
    by it, so using dry ``cpd`` instead over-heats by ``vtmpc2·q``
    (~1.5 % at 18 g/kg). The ``max(q, 0)`` clamp mirrors the Fortran and
    keeps ``cp`` physical against spectral-ringing undershoots.

    Parameters
    ----------
    specific_humidity : jnp.ndarray
        Specific humidity [kg/kg]; any shape (broadcasting-native).

    Returns
    -------
    jnp.ndarray
        Moist isobaric specific heat [J/kg/K], same shape.

    """
    return c.cpd * (1.0 + c.vtmpc2 * jnp.maximum(specific_humidity, 0.0))
