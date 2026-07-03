"""Shared saturation thermodynamics (ECHAM 6.3 conventions).

Single source of truth for saturation vapour pressure, saturation specific
humidity (and its analytic temperature derivative), the mixed-phase liquid
fraction, and the grid-mean → in-cloud conversion. All coefficients follow
ECHAM 6.3 ``mo_convect_tables`` (the Tetens/Magnus family):

    es(T) = c1es · exp(c3 · (T − tmelt) / (T − c4))

with ``c1es = 610.78 Pa`` and, over liquid water, ``c3les = 17.269`` /
``c4les = 35.86 K``; over ice, ``c3ies = 21.875`` / ``c4ies = 7.66 K``.
``tmelt`` and ``eps`` are read by attribute access from :mod:`jcm.constants`,
so runtime ``set_constants`` overrides are honoured.

This module exists to end the historical divergence of six-plus independent
qsat implementations across the physics packages (convection, clouds,
surface tiles, aerosol), several of which had drifted onto inconsistent —
in one case outright broken — coefficient sets. New code must call these
functions instead of inlining coefficients. The inline saturation forms in
:mod:`jcm.physics.clouds.lohmann_2m` (beyond the liquid/ice pair used by the
Bergeron-Findeisen block) are intentionally **not yet** migrated — that
scheme has a coordinated set of pending fixes and will be moved over as one
piece.

Phase selection: ``phase="auto"`` is the *simple* switch — liquid-water
saturation at ``T ≥ tmelt`` and ice saturation below. Schemes that need a
gradual liquid/ice transition (mixed-phase weighting) must do the blending
themselves via :func:`mixed_phase_weight`; ``"auto"`` deliberately does not
attempt it.

All functions are pure JAX and broadcasting-native (no Python branching on
traced values; ``phase`` is a static Python string resolved at trace time).
They are intentionally not ``@jit``-ed: they always run inside a caller's
compiled graph and inline there.
"""

import jax.numpy as jnp

import jcm.constants as c

# ECHAM 6.3 mo_convect_tables coefficients.
C1ES = 610.78     # Pa — saturation vapour pressure at the melting point
C3LES = 17.269    # over liquid water
C4LES = 35.86     # K
C3IES = 21.875    # over ice
C4IES = 7.66      # K

# Math-safety temperature clip. The exponent denominator (T − c4) vanishes
# near 8-36 K, far below any physical temperature; a loose [50, 500] K bound
# keeps the math finite without masking genuine upstream bugs. Note the clip
# also zeroes the temperature gradient outside these bounds.
_T_MIN = 50.0
_T_MAX = 500.0

# ECHAM caps the saturation specific humidity lookup at MIN(..., 0.5): at
# very high T / low p the denominator p − (1−eps)·es shrinks toward zero and
# qs would blow up; 0.5 is far above any physical qs, so the cap is inactive
# in normal conditions.
_QS_MAX = 0.5


def _validate_phase(phase: str) -> None:
    if phase not in ("auto", "water", "ice"):
        raise ValueError(
            f"phase must be 'auto', 'water' or 'ice', got {phase!r}")


def saturation_vapor_pressure(temperature: jnp.ndarray,
                              phase: str = "auto") -> jnp.ndarray:
    """Saturation vapour pressure ``es(T)`` [Pa], ECHAM 6.3 coefficients.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K]. Clipped to [50, 500] K for math safety.
    phase : str
        ``"water"`` (liquid-water saturation at all temperatures),
        ``"ice"`` (ice saturation at all temperatures), or ``"auto"``
        (water at ``T ≥ tmelt``, ice below — the simple switch; callers
        needing a gradual mixed-phase transition must blend the two pure
        phases themselves, e.g. with :func:`mixed_phase_weight`).

    Returns
    -------
    jnp.ndarray
        Saturation vapour pressure [Pa].

    """
    _validate_phase(phase)
    temperature = jnp.clip(temperature, _T_MIN, _T_MAX)
    tc = temperature - c.tmelt
    if phase == "water":
        return C1ES * jnp.exp(C3LES * tc / (temperature - C4LES))
    if phase == "ice":
        return C1ES * jnp.exp(C3IES * tc / (temperature - C4IES))
    es_water = C1ES * jnp.exp(C3LES * tc / (temperature - C4LES))
    es_ice = C1ES * jnp.exp(C3IES * tc / (temperature - C4IES))
    return jnp.where(temperature >= c.tmelt, es_water, es_ice)


def saturation_specific_humidity(temperature: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 phase: str = "auto") -> jnp.ndarray:
    """Saturation specific humidity ``qs(T, p)`` [kg/kg].

    ``qs = eps·es / (p − (1−eps)·es)`` with the denominator guarded against
    non-positivity and the result capped at 0.5 (ECHAM's ``MIN(..., 0.5)``
    lookup guard — inactive at physical temperatures).

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
        Saturation specific humidity [kg/kg], in ``(0, 0.5]``.

    """
    es = saturation_vapor_pressure(temperature, phase=phase)
    denom = jnp.maximum(pressure - (1.0 - c.eps) * es, c.epsilon)
    return jnp.minimum(c.eps * es / denom, _QS_MAX)


def saturation_specific_humidity_and_derivative(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    phase: str = "auto",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(qs, dqs/dT)`` with the analytic temperature derivative.

    The closed form keeps Newton saturation-adjustment steps (cuadjtq,
    updraft ascent) bit-reproducible under JIT without differentiating
    through the guard/cap logic. From ``es = c1es·exp(c3·(T−tmelt)/(T−c4))``:

        d(es)/dT  = es · c3·(tmelt − c4) / (T − c4)²
        d(qs)/dT  = qs · c3·(tmelt − c4)/(T − c4)² · p / (p − (1−eps)·es)

    (the second factor comes from differentiating the ``qs(es)`` mapping:
    ``dqs/des = eps·p/denom²``). The derivative uses the same c3/c4
    coefficients as the value, per phase; where the 0.5 cap or the
    denominator guard is active the derivative is set to 0 for consistency
    with the capped value.

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
    es = saturation_vapor_pressure(temperature, phase=phase)
    denom_raw = pressure - (1.0 - c.eps) * es
    denom = jnp.maximum(denom_raw, c.epsilon)
    qs_raw = c.eps * es / denom
    qs = jnp.minimum(qs_raw, _QS_MAX)

    # d(ln es)/dT per phase, evaluated on the same clipped temperature the
    # vapour pressure used so the pair stays self-consistent.
    t_safe = jnp.clip(temperature, _T_MIN, _T_MAX)
    dlnes_water = C3LES * (c.tmelt - C4LES) / (t_safe - C4LES) ** 2
    dlnes_ice = C3IES * (c.tmelt - C4IES) / (t_safe - C4IES) ** 2
    if phase == "water":
        dlnes_dT = dlnes_water
    elif phase == "ice":
        dlnes_dT = dlnes_ice
    else:
        dlnes_dT = jnp.where(t_safe >= c.tmelt, dlnes_water, dlnes_ice)

    dqs_dT = qs_raw * dlnes_dT * pressure / denom
    # Guard/cap regions: the returned qs is flat there, so report a zero
    # slope rather than the (meaningless) uncapped one.
    dqs_dT = jnp.where((qs_raw < _QS_MAX) & (denom_raw > c.epsilon),
                       dqs_dT, 0.0)
    return qs, dqs_dT


def mixed_phase_weight(temperature: jnp.ndarray,
                       t_min: float = 238.15,
                       t_max: float | None = None) -> jnp.ndarray:
    """Linear liquid fraction for mixed-phase blending.

    Returns ``clip((T − t_min) / (t_max − t_min), 0, 1)`` — 1 for pure
    liquid at/above ``t_max``, 0 for pure ice at/below ``t_min``. This is
    the standard linear ramp used to blend water/ice saturation (and latent
    heats) across the mixed-phase range; pair it with the pure-phase
    ``"water"`` / ``"ice"`` saturation functions.

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
