"""ECHAM6.3's diagnostic cloud cover (``mo_cover.f90::cover``).

The cover is the Sundqvist (1989) / Lohmann and Roeckner (1996) relative-
humidity closure as ECHAM6.3-HAM2.3 r7492 codes it in ``mo_cover.f90``
(l.101-261), called once per step before radiation (``physc.f90`` l.543):

1. saturation specific humidity in ECHAM's form, over ice or over water per
   ECHAM's ``lo2`` switch (l.215-224), with ECHAM's vapour pressure, the
   Sonntag (1990) fit its tables hold
   (:mod:`jcm.physics.clouds.echam_saturation`);
2. critical relative humidity ``rhc = crt + (crs - crt)·exp(1 - (p_s/p)^nex)``
   (l.233);
3. over ice-free ocean without convection, a stratocumulus enhancement at the
   low-level inversion found by a level search between ECHAM's ``jbmin`` and
   the surface (l.179-207, 234-247; :func:`stratocumulus_saturation_factor`);
4. ``b0 = (q/(qs·zsat) - rhc)/(1 - rhc)`` clipped to ``[0, 1]`` and
   ``cover = 1 - sqrt(1 - b0)`` (l.248-251).

The values are ECHAM's: the cover is exactly 0 where ``b0 <= 0`` and exactly 1
where ``b0 >= 1``, at every level (ECHAM computes all levels, ``ktdia = 1``,
``physc.f90`` l.444). Where the reference derivative is useless (the clip's
plateaux, the square root's infinite slope at saturation, the inversion
test), the derivative is that of a named smooth surrogate, through
:func:`jcm.physics.surrogate_gradient.with_surrogate_gradient`; see
``docs/source/design/surrogate_gradients.md`` and the cloud-cover section of
``docs/source/science/clouds_microphysics.md``.

The module also holds :func:`saturation_specific_humidity`, a linear
mixed-phase blend of the Sonntag (1990) fits that tests build humidity
profiles from. It is not ECHAM's phase rule, and neither the cover nor the 1M
scheme uses it.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

import jcm.constants as c
from jcm.physics import thermodynamics
from jcm.physics.clouds import echam_saturation as es
from jcm.physics.clouds.echam_cloud_defaults import (
    CTHOMI_BELOW_TMELT,
    echam_cloud_defaults,
    inversion_levels,
)
from jcm.physics.resolution_defaults import (
    check_defaults_grid,
    spectral_truncation,
)
from jcm.physics.surrogate_gradient import with_surrogate_gradient


@struct.dataclass
class CloudParameters:
    """Parameters of the ECHAM cloud cover.

    The numeric fields are differentiable pytree leaves. ``crt``, ``crs``,
    ``nex``, ``csatsc``, ``cinv``, ``csecfrl`` and ``nadd`` have
    resolution-dependent defaults (ECHAM's ``mo_echam_cloud_params.f90``
    table, :func:`~jcm.physics.clouds.echam_cloud_defaults.echam_cloud_defaults`):
    build them with :meth:`default` ``(truncation=...)`` or :meth:`for_grid`.

    ``csecfrl`` and ``t_ice`` are ECHAM's ``csecfrl`` and ``cthomi``
    (``mo_echam_cloud_params.f90`` l.76, l.54), one value each, which
    ECHAM's cover and cloud scheme share. jcm holds a second copy in the
    cloud scheme's parameters (``MicrophysicsParameters.csecfrl``/``cthomi``;
    the 2M's ``CloudParams2M.cthomi``); the defaults agree, and an override of
    one copy leaves the other unchanged. ``echam_physics`` warns when the
    copies it builds differ.

    Static fields (``pytree_node=False``):

    * ``nadd`` selects which extra level below the inversion is enhanced, a
      level index, so it chooses a code path rather than scaling a value.
    * ``smooth_b0`` and ``smooth_inv_thr`` are the widths of the surrogates
      that define the derivatives; the value does not depend on them, so a
      gradient with respect to them would mean nothing. Zero selects the
      reference derivative.
    * ``defaults_truncation`` records the truncation whose defaults the
      fields were built from (``None`` for a non-spectral grid), so that the
      term can warn when it runs on a different grid.
    """

    crt: jnp.ndarray       # critical relative humidity aloft
    crs: jnp.ndarray       # critical relative humidity at the surface
    nex: jnp.ndarray       # exponent of the critical-RH profile
    csatsc: jnp.ndarray    # stratocumulus saturation factor at an inversion
    cinv: jnp.ndarray      # inversion stability threshold, fraction of g/cpd
    csecfrl: jnp.ndarray   # cloud ice [kg/kg] above which lo2 selects ice
    t_ice: jnp.ndarray     # cthomi [K]: below it lo2 always selects ice
    nadd: int = struct.field(pytree_node=False, default=0)
    # Width of the softplus surrogate of the b0 clip [1] (see
    # ``_cover_surrogate``).
    smooth_b0: float = struct.field(pytree_node=False, default=0.02)
    # Width of the sigmoid surrogate of the inversion stability test [K/m]
    # (see ``_zsat_surrogate``).
    smooth_inv_thr: float = struct.field(pytree_node=False, default=2.0e-4)
    defaults_truncation: int | None = struct.field(
        pytree_node=False, default=63)

    @classmethod
    def default(cls, *, truncation: int | None = 63,
                **overrides) -> "CloudParameters":
        """Defaults for a spectral truncation, with field overrides on top.

        Args:
            truncation: the run's triangular truncation; ``None`` means a grid
                that is not spectral (T63 defaults, with a warning). ECHAM's
                values at T31/T63/T127/T255, interpolated between them (see
                :mod:`jcm.physics.clouds.echam_cloud_defaults`).
            **overrides: field values that replace the defaults.

        Returns:
            The parameters. ``defaults_truncation`` is ``truncation``.

        """
        table = echam_cloud_defaults(truncation)
        values = dict(
            crt=table["crt"], crs=table["crs"], nex=float(table["nex"]),
            csatsc=table["csatsc"], cinv=table["cinv"],
            csecfrl=table["csecfrl"],
            t_ice=c.tmelt - CTHOMI_BELOW_TMELT,
            nadd=int(table["nadd"]),
            defaults_truncation=truncation,
        )
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise ValueError(
                f"unknown CloudParameters field(s) {unknown}; valid fields: "
                f"{sorted(valid)}")
        values.update(overrides)
        static = {"nadd", "smooth_b0", "smooth_inv_thr", "defaults_truncation"}
        params = cls(**{
            name: (value if name in static or isinstance(value, jax.Array)
                   else jnp.asarray(value))
            for name, value in values.items()})
        params.validate()
        return params

    @classmethod
    def for_grid(cls, coords, **overrides) -> "CloudParameters":
        """:meth:`default` for the truncation of ``coords``."""
        return cls.default(truncation=spectral_truncation(coords), **overrides)

    def validate(self) -> None:
        """Reject static fields outside their domain."""
        if int(self.nadd) != self.nadd or self.nadd < 0:
            raise ValueError(f"nadd must be a non-negative integer, got "
                             f"{self.nadd!r}")
        for name in ("smooth_b0", "smooth_inv_thr"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0 (0 selects the "
                                 "reference derivative)")


def critical_relative_humidity(
    pressure: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    config: CloudParameters,
) -> jnp.ndarray:
    """``rhc = crt + (crs - crt)·exp(1 - (p_s/p)^nex)`` (``mo_cover.f90`` l.233).

    Args:
        pressure: full-level pressure [Pa], shape ``(nlev, *horiz)``.
        surface_pressure: surface pressure [Pa] (ECHAM's ``paphm1(klevp1)``),
            shape ``(*horiz)``.
        config: the cover parameters.

    Returns:
        The critical relative humidity, shape ``(nlev, *horiz)``.

    """
    ps = jnp.asarray(surface_pressure)[None]
    return config.crt + (config.crs - config.crt) * jnp.exp(
        1.0 - (ps / pressure) ** config.nex)


def cover_saturation_vapor_pressure(
    temperature: jnp.ndarray, ice: jnp.ndarray,
) -> jnp.ndarray:
    """Return the saturation vapour pressure [Pa] the cover uses.

    The one place the cover takes ``es`` from: ECHAM's Sonntag (1990) fit,
    :func:`jcm.physics.clouds.echam_saturation.es_water` / ``es_ice``.

    Args:
        temperature: [K].
        ice: ``True`` where the ice surface applies (ECHAM's ``lo2``).

    """
    return jnp.where(ice, es.es_ice(temperature), es.es_water(temperature))


def cover_saturation_specific_humidity(
    temperature: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    pressure: jnp.ndarray,
    config: CloudParameters,
) -> jnp.ndarray:
    """Return the saturation specific humidity the cover divides by.

    ECHAM's ``lo2`` phase choice (ice where ``T < cthomi``, or ``T < tmelt``
    and ``xi > csecfrl``) and its ``MIN(ua/p, 0.5)/(1 - vtmpc1·...)`` form
    (``mo_cover.f90`` l.215-224), with ``es`` from
    :func:`cover_saturation_vapor_pressure`. The switch is ECHAM's hard
    switch, value and derivative alike: the derivative is that of the branch
    in use.
    """
    ice = es.lo2_ice_phase(temperature, cloud_ice, config.csecfrl,
                           config.t_ice)
    return es.qsat_from_es(
        cover_saturation_vapor_pressure(temperature, ice), pressure)


# ---------------------------------------------------------------------------
# Stratocumulus enhancement at the low-level inversion
# ---------------------------------------------------------------------------

def _inversion_lapse(temperature, geopotential):
    """ECHAM's ``zdtdz`` per level [K/m] (``mo_cover.f90`` l.194 and l.244).

    Level ``k`` owns the lapse across the interface above it,
    ``(T[k-1] - T[k])·g/(Φ[k-1] - Φ[k])``. Level 0 has none and is never in
    the search range (``jbmin >= 1``); it is set to 0.
    """
    lapse = ((temperature[:-1] - temperature[1:]) * c.grav
             / (geopotential[:-1] - geopotential[1:]))
    return jnp.concatenate([jnp.zeros_like(lapse[:1]), lapse], axis=0)


def _inversion_selection(lapse, jbmin, jbmax, nadd):
    """ECHAM's inversion search, the discrete part (``mo_cover.f90`` l.188-247).

    ECHAM scans from the lowest level up to ``jbmin`` and keeps the level with
    the largest ``min(0, zdtdz)``, updating only on strict improvement, so on
    a tie the lowest level wins. It enhances only if that level is at or above
    ``jbmax``, at the level itself and ``nadd`` levels below it.

    Returns:
        ``best``: the largest clipped lapse in the range, ``(*horiz)``;
        ``levels``: a ``(nlev, *horiz)`` 0/1 mask of the enhanced levels,
        already zero where the chosen level lies below ``jbmax``;
        ``lapse_at_choice``: ``zdtdz`` at the chosen level, ``(*horiz)``.
        The existence test ``best > -cinv·g/cpd`` is left to the caller.

    """
    kx = lapse.shape[0]
    level = jnp.arange(kx).reshape((kx,) + (1,) * (lapse.ndim - 1))
    score = jnp.where(level >= jbmin, jnp.minimum(lapse, 0.0), -jnp.inf)
    best = jnp.max(score, axis=0)
    # ``argmax`` returns the first maximum; over the reversed axis that is the
    # largest index, i.e. the lowest level, as ECHAM's upward scan keeps.
    choice = kx - 1 - jnp.argmax(score[::-1], axis=0)
    lapse_at_choice = jnp.take_along_axis(lapse, choice[None], axis=0)[0]
    enhanced = (level == choice[None]) | (level == choice[None] + nadd)
    in_range = (choice <= jbmax)[None]
    return best, (enhanced & in_range).astype(lapse.dtype), lapse_at_choice


def _zsat_from(found, levels, lapse_at_choice, csatsc, enhance):
    """``zsat = min(1, csatsc + max(0, -zdtdz·cpd/g))`` at the enhanced levels.

    ``mo_cover.f90`` l.244-246; 1 elsewhere. ``found`` is 1 where the search
    found a level more stable than the threshold (a float, so that the
    surrogate can make it smooth).
    """
    zgam = jnp.maximum(0.0, -lapse_at_choice * c.cpd / c.grav)
    reduction = found * enhance * (1.0 - jnp.minimum(1.0, csatsc + zgam))
    return 1.0 - levels * reduction[None]


def _zsat_exact(lapse, csatsc, cinv, enhance, *, jbmin, jbmax, nadd):
    """ECHAM's ``zsat`` exactly: the stability test is a hard ``>``."""
    best, levels, lapse_at_choice = _inversion_selection(
        lapse, jbmin, jbmax, nadd)
    found = (best > -cinv * c.grav / c.cpd).astype(lapse.dtype)
    return _zsat_from(found, levels, lapse_at_choice, csatsc, enhance)


def _zsat_surrogate(lapse, csatsc, cinv, enhance, *, jbmin, jbmax, nadd,
                    width):
    """Return the surrogate of :func:`_zsat_exact` that defines its derivative.

    Identical except that the stability test ``best > -cinv·g/cpd`` is the
    sigmoid ``σ((best + cinv·g/cpd)/width)``. That gives the cover a
    derivative with respect to ``cinv`` and to the lapse rate at the chosen
    level where a column is near the threshold, instead of none. The level
    choice itself keeps its reference derivative, zero: which level is chosen
    is piecewise constant in the temperature profile, and a smooth selection
    (a softmax over levels) would differ from ECHAM's value by the whole
    enhancement wherever two levels compete, so its derivative would describe
    a different function. The derivative through the chosen level's own
    ``zgam`` and through ``csatsc`` is the reference one.
    """
    best, levels, lapse_at_choice = _inversion_selection(
        lapse, jbmin, jbmax, nadd)
    found = jax.nn.sigmoid((best + cinv * c.grav / c.cpd) / width)
    return _zsat_from(found, levels, lapse_at_choice, csatsc, enhance)


def stratocumulus_saturation_factor(
    temperature: jnp.ndarray,
    geopotential: jnp.ndarray,
    config: CloudParameters,
    inversion_range: tuple[int, int],
    enhance_allowed: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """ECHAM's stratocumulus saturation factor ``zsat`` per level.

    ``mo_cover.f90`` l.179-207 and 234-247: over ice-free ocean without
    convection (``enhance_allowed``), the level between ``jbmin`` and the
    surface with the largest ``min(0, dT/dz)`` (the lowest one on a tie) is
    the inversion, provided that value exceeds ``-cinv·g/cpd``. If that level
    is at or above ``jbmax``, ``zsat = min(1, csatsc + max(0, -dT/dz·cpd/g))``
    there and ``nadd`` levels below it; ``zsat = 1`` everywhere else. The
    cover then uses ``q/(qs·zsat)``.

    The derivative is that of :func:`_zsat_surrogate` (width
    ``config.smooth_inv_thr``), or ECHAM's own with a zero width.

    Args:
        temperature: ``(nlev, *horiz)`` [K], top first.
        geopotential: full-level geopotential ``(nlev, *horiz)`` [m2 s-2];
            only differences between levels enter.
        config: the cover parameters.
        inversion_range: ECHAM's ``(jbmin, jbmax)`` as 0-based top-first
            level indices, from
            :func:`~jcm.physics.clouds.echam_cloud_defaults.inversion_levels`.
        enhance_allowed: ``(*horiz)`` boolean gate; ``None`` allows it
            everywhere.

    Returns:
        ``zsat``, shape ``(nlev, *horiz)``, in ``[csatsc, 1]``.

    """
    jbmin, jbmax = inversion_range
    lapse = _inversion_lapse(temperature, geopotential)
    if enhance_allowed is None:
        enhance = jnp.ones(lapse.shape[1:], lapse.dtype)
    else:
        enhance = jnp.broadcast_to(
            jnp.asarray(enhance_allowed), lapse.shape[1:]).astype(lapse.dtype)
    static = dict(jbmin=int(jbmin), jbmax=int(jbmax), nadd=int(config.nadd))

    def exact(lapse_, csatsc_, cinv_, enhance_):
        return _zsat_exact(lapse_, csatsc_, cinv_, enhance_, **static)

    if config.smooth_inv_thr == 0.0:
        return exact(lapse, config.csatsc, config.cinv, enhance)

    def surrogate(lapse_, csatsc_, cinv_, enhance_):
        return _zsat_surrogate(lapse_, csatsc_, cinv_, enhance_, **static,
                               width=float(config.smooth_inv_thr))

    return with_surrogate_gradient(exact, surrogate)(
        lapse, config.csatsc, config.cinv, enhance)


# ---------------------------------------------------------------------------
# The closure
# ---------------------------------------------------------------------------

def _cover_exact(b0_raw):
    """ECHAM's ``1 - sqrt(1 - clip(b0, 0, 1))`` (``mo_cover.f90`` l.249-251).

    Written with a safe square root so that the reference derivative (the
    zero-width case) is finite on the saturated plateau; the value is exactly
    ECHAM's everywhere.
    """
    arg = 1.0 - jnp.clip(b0_raw, 0.0, 1.0)
    positive = arg > 0.0
    return jnp.where(positive, 1.0 - jnp.sqrt(jnp.where(positive, arg, 1.0)),
                     1.0)


#: Floor under the surrogate's ``1 - b0``: below it the surrogate is held
#: flat. The true surrogate slope there is below ``sqrt(floor)/(2·width)``,
#: i.e. zero at any float precision.
_SURROGATE_ARG_FLOOR = 1.0e-30


def _cover_surrogate(b0_raw, width):
    """Return the smooth cover whose derivative the cover carries.

    ``1 - sqrt(1 - b0_s)`` with the softplus clip
    ``b0_s = w·softplus(x/w) - w·softplus((x - 1)/w)`` of width ``w``, which
    equals the identity inside ``[0, 1]`` away from the edges and approaches
    0 and 1 exponentially. ``1 - b0_s`` is formed as
    ``w·(softplus((1 - x)/w) - softplus(-x/w))``, which stays accurate where
    it is small. Because ``b0_s < 1`` for every finite ``x``, the square
    root's slope is bounded: the surrogate's ``d cover/d b0`` peaks at about
    ``1/(2·sqrt(w·ln 2))`` (4.2 at ``w = 0.02``), where the reference is
    unbounded as ``b0 -> 1`` and zero on both plateaux. The surrogate differs
    from the reference by at most ``sqrt(w·ln 2)`` (0.12 at ``w = 0.02``),
    at ``b0 = 1``.
    """
    arg = width * (jax.nn.softplus((1.0 - b0_raw) / width)
                   - jax.nn.softplus(-b0_raw / width))
    above = arg > _SURROGATE_ARG_FLOOR
    return jnp.where(above, 1.0 - jnp.sqrt(jnp.where(above, arg, 1.0)),
                     1.0 - jnp.sqrt(_SURROGATE_ARG_FLOOR))


def cover_from_b0(b0_raw: jnp.ndarray, width: float) -> jnp.ndarray:
    """Return the cover from the unclipped ``b0``: ECHAM's value, surrogate slope.

    Args:
        b0_raw: ``(q/(qs·zsat) - rhc)/(1 - rhc)``, unclipped.
        width: the static surrogate width ``smooth_b0``; 0 selects the
            reference derivative.

    Returns:
        ``1 - sqrt(1 - clip(b0_raw, 0, 1))``, exactly.

    """
    if width == 0.0:
        return _cover_exact(b0_raw)
    return with_surrogate_gradient(
        _cover_exact, lambda x: _cover_surrogate(x, width))(b0_raw)


def calculate_cloud_fraction(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    pressure: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    geopotential: jnp.ndarray,
    config: CloudParameters,
    inversion_range: tuple[int, int],
    enhance_allowed: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ECHAM's cloud cover, ``mo_cover.f90::cover`` (r7492 l.101-261).

    Broadcasting-native: the level is axis 0 and any trailing axes are
    horizontal, so a ``(nlev,)`` column, a ``(nlev, ncols)`` block and a
    ``(nlev, nlon, nlat)`` grid run the same code.

    Args:
        temperature: ``(nlev, *horiz)`` [K], top first (ECHAM ``ptm1``).
        specific_humidity: ``(nlev, *horiz)`` [kg/kg] (``pqm1``).
        cloud_ice: ``(nlev, *horiz)`` [kg/kg] (``pxim1``), for ``lo2``.
        pressure: full-level pressure ``(nlev, *horiz)`` [Pa] (``papm1``).
        surface_pressure: ``(*horiz)`` [Pa] (``paphm1(klevp1)``).
        geopotential: full-level geopotential ``(nlev, *horiz)`` [m2 s-2]
            (``pgeo``); only level differences enter.
        config: the cover parameters.
        inversion_range: ECHAM's ``(jbmin, jbmax)``, 0-based top-first.
        enhance_allowed: ``(*horiz)`` gate of the stratocumulus enhancement
            (ECHAM: water fraction > 0.5, ice fraction < 1e-12, ``ktype == 0``);
            ``None`` allows it everywhere.

    Returns:
        ``(cloud_fraction, relative_humidity)``, each ``(nlev, *horiz)``.
        ``relative_humidity`` is ``q/qs`` with the cover's ``qs`` (ice where
        ``lo2``), without the inversion factor and unclipped.

    """
    qs = cover_saturation_specific_humidity(
        temperature, cloud_ice, pressure, config)
    rhc = critical_relative_humidity(pressure, surface_pressure, config)
    zsat = stratocumulus_saturation_factor(
        temperature, geopotential, config, inversion_range, enhance_allowed)
    zqr = specific_humidity / (qs * zsat)
    b0_raw = (zqr - rhc) / (1.0 - rhc)
    cloud_fraction = cover_from_b0(b0_raw, float(config.smooth_b0))
    return cloud_fraction, specific_humidity / qs


# ---------------------------------------------------------------------------
# Mixed-phase saturation for other callers (the cover does not use it)
# ---------------------------------------------------------------------------

def saturation_specific_humidity(
    pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    t_mix_min: float = 238.15,
) -> jnp.ndarray:
    """Saturation specific humidity [kg/kg] on a linear mixed-phase blend.

    The vapour pressure blends Sonntag (1990) over water and over ice
    (:func:`jcm.physics.thermodynamics.es_water` / ``es_ice``) linearly in
    temperature between ``t_mix_min`` and ``tmelt``, and ``qs`` is formed as
    ECHAM forms it (:func:`jcm.physics.thermodynamics.qsat_from_es`). The
    blend is not ECHAM's phase rule: the cover
    (:func:`cover_saturation_specific_humidity`) and the 1M scheme choose ice
    or water per cell with ``lo2``. Tests build humidity profiles from it.

    Args:
        pressure: pressure [Pa].
        temperature: temperature [K].
        t_mix_min: lower end of the blend [K].

    Returns:
        Saturation specific humidity [kg/kg].

    """
    weight = jnp.clip(
        (temperature - t_mix_min) / (c.tmelt - t_mix_min), 0.0, 1.0)
    vapour = (weight * thermodynamics.es_water(temperature)
              + (1.0 - weight) * thermodynamics.es_ice(temperature))
    return thermodynamics.qsat_from_es(vapour, pressure)


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.clouds.cloud_data import (  # noqa: E402
    CLOUD_OUTPUT_ATTRS,
    CloudData,
)
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class SundqvistCloudFraction(PhysicsTerm):
    """ECHAM6.3's diagnostic cloud cover, ``mo_cover.f90::cover``.

    Operates on a ``(nlev, ncols)`` block or any broadcastable layout. Reads
    ``pressure_full`` / ``surface_pressure`` from the moist-air diagnostics,
    the temperature, humidity, geopotential and ``qi`` from ``state`` (ECHAM
    ``ptm1``, ``pqm1``, ``pgeo``, ``pxim1``), the land fraction from
    ``terrain`` and the sea-ice fraction from ``forcing``.

    **Time level.** ``state`` is the state the physics receives this step,
    which already contains this step's dynamics. ECHAM's ``cover`` reads the
    ``t - Δt`` fields with no tendency of any kind (``physc.f90`` l.543-548),
    a state one dynamics step earlier. Reading ECHAM's state would take the
    previous step's post-physics state from the carry, which does not hold it.
    Like ECHAM's, this term runs first in the step, before radiation.
    Writes ``cloud_fraction``, plus a pass-through of the ``qc`` / ``qi`` the
    downstream microphysics starts from, into the public ``"clouds"`` key
    (:class:`CloudData`), and publishes ``"cover_relative_humidity"``: the
    ``q / qs`` the cover closure sees, with ``qs`` over ice where ECHAM's
    ``lo2`` selects it. That is a scheme-internal closure variable, so it
    does NOT overwrite the public water-saturation ``"relative_humidity"``
    from :class:`~jcm.physics.diagnostics.moist_air_state.MoistAirColumnState`.

    **No q <-> qc/qi tendency is emitted**; condensation belongs to the
    downstream microphysics term (the 1M
    :class:`~jcm.physics.clouds.echam_1m.Echam1MMicrophysics` or the 2M
    :class:`~jcm.physics.clouds.lohmann_2m.Lohmann2MMicrophysics`), as in
    ECHAM, where ``cover`` diagnoses and ``cloud`` condenses.

    ``cache_coords`` must run before the term is called: it computes ECHAM's
    inversion-search levels ``jbmin``/``jbmax`` from the model's levels. It
    also checks that the parameters' resolution defaults were built for the
    model's truncation and warns once if they were not, unless the caller
    supplied the parameters explicitly.
    """

    name: ClassVar[str] = "sundqvist_cloud_fraction"
    category: ClassVar[str] = "cloud_fraction"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "surface_pressure",
    )
    provides: ClassVar[tuple[str, ...]] = ("clouds", "cover_relative_humidity")
    # CF/units metadata for the ``clouds.*`` output fields (#740), shared with
    # the microphysics terms that fill the rest of the CloudData struct, plus
    # this term's own humidity. No CF ``standard_name``: ``relative_humidity``
    # is the water-referenced quantity, which this one is not.
    output_attrs: ClassVar[dict[str, dict[str, str]]] = {
        **CLOUD_OUTPUT_ATTRS,
        "cover_relative_humidity": {
            "units": "1",
            "long_name": (
                "relative humidity seen by the ECHAM cloud cover "
                "(ice saturation where ECHAM's lo2 switch selects it)"),
        },
    }
    # Carry seeded as zeros; cloud fraction / qc / qi are rebuilt every
    # step from RH and the dynamics tracers, so the zero seed is
    # overwritten on the first compute call. Downstream microphysics
    # terms write ``precip_*`` / TOA-flux fields on the same key so the
    # carry shape stays stable after step 1.
    carry_slots: ClassVar[dict[str, type]] = {"clouds": CloudData}

    def __init__(self, params: CloudParameters | None = None, *,
                 params_are_defaults: bool = False):
        """Hold the scheme's :class:`CloudParameters`.

        Args:
            params: the parameters; ``None`` takes the T63 defaults.
            params_are_defaults: ``True`` when ``params`` are resolution
                defaults built by the physics factory or the runner (with or
                without field overrides), rather than an object the user
                supplied. Only then does ``cache_coords`` warn if they were
                built for another truncation.

        """
        self._params_user_supplied = (params is not None
                                      and not params_are_defaults)
        self.params = nnx.Param(params if params is not None
                                else CloudParameters.default())
        self._inversion_range: tuple[int, int] | None = None

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """``qc`` / ``qi`` are read each step; declared so dynamics carries them."""
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
        )

    def cache_coords(self, coords) -> None:
        """Compute ``jbmin``/``jbmax`` and check the parameters' grid.

        ``jbmin``/``jbmax`` are grid geometry (``mo_echam_cloud_params.f90``
        l.132-162), computed from the model's own levels here. The tunable
        parameters are not touched: they were fixed at construction.
        """
        jbmin, jbmax = inversion_levels(coords)
        if jbmin < 1:
            raise ValueError(
                "ECHAM's inversion search needs a level above jbmin; the grid "
                f"gives jbmin={jbmin} (0-based).")
        self._inversion_range = (jbmin, jbmax)
        if not self._params_user_supplied:
            check_defaults_grid(
                type(self).__name__,
                self.params.get_value().defaults_truncation, coords)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Diagnose the cover and its humidity; no tendency."""
        if self._inversion_range is None:
            raise RuntimeError(
                f"{type(self).__name__}.cache_coords(coords) must run before "
                "the term is called: it computes ECHAM's inversion-search "
                "levels from the model grid.")
        nlev = state.temperature.shape[0]
        horiz = state.temperature.shape[1:]
        params = self.params.get_value()
        zeros = jnp.zeros_like(state.temperature)

        # The condensate the downstream microphysics starts from. At this
        # term's position (before vertical diffusion) ``thermo_run`` still
        # holds the step-start tracers.
        tr = diagnostics.get("thermo_run") or {}
        qc = tr.get("qc", state.tracers.get("qc", zeros))
        qi = tr.get("qi", state.tracers.get("qi", zeros))
        # ECHAM's cover reads the step-start cloud ice ``xim1`` for lo2.
        qi_m1 = state.tracers.get("qi", zeros)

        # Stratocumulus gate (mo_cover.f90 l.181): water fraction > 0.5,
        # ice fraction < 1e-12 and ktype == 0, with ECHAM's fractions
        # frw = (1 - frl)(1 - seaice) and fri = 1 - frl - frw
        # (physc.f90 l.402-403).
        land = jnp.reshape(jnp.asarray(terrain.fmask), horiz)
        sice = getattr(forcing, "sice_am", None)
        sea_ice = (jnp.reshape(jnp.asarray(sice), horiz) if sice is not None
                   else jnp.zeros_like(land))
        frw = (1.0 - land) * (1.0 - sea_ice)
        fri = 1.0 - land - frw
        # ``ktype`` is the previous step's, read from the ``convection``
        # carry: this term runs before convection within the step. ECHAM's
        # ``cover`` likewise reads the previous step's type: ``itype`` is set
        # from ``rtype`` (physc.f90 l.528) before ``cucall`` (l.987) and
        # ``rtype`` is written back only after ``cloud`` (l.1124). Step 0 has
        # no carry and allows the enhancement, as ECHAM's initial rtype = 0
        # does. Deliberately not in ``requires`` (that would force this term
        # after convection); it is a cross-step read.
        conv = diagnostics.get("convection")
        no_convection = (
            jnp.reshape(conv.ktype, horiz) == 0
            if conv is not None and hasattr(conv, "ktype")
            else jnp.ones(horiz, dtype=bool))
        enhance_allowed = (frw > 0.5) & (fri < 1.0e-12) & no_convection

        cloud_fraction, rel_humidity = calculate_cloud_fraction(
            state.temperature, state.specific_humidity, qi_m1,
            diagnostics["pressure_full"], diagnostics["surface_pressure"],
            state.geopotential, params, self._inversion_range,
            enhance_allowed)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=zeros,
            specific_humidity=zeros,
            tracers={"qc": zeros, "qi": zeros},
        )
        # The convective-detrainment fields are reset to zero here because
        # ``prev_clouds`` is the PREVIOUS step's carry: this term seeds the
        # step's ``clouds`` upstream of convection, and those fields must
        # hold only what convection detrains THIS step (TiedtkeConvection
        # rewrites them). Without the reset, the gap before convection runs
        # would expose last step's values, and a stack that composes no
        # convection term — including a restart from a checkpoint written by
        # one that did — would feed a stale detrainment to the microphysics
        # on every step.
        prev_clouds = diagnostics.get(
            "clouds", CloudData.zeros(horiz, nlev),
        )
        clouds = prev_clouds.copy(
            cloud_fraction=cloud_fraction,
            qc=qc,
            qi=qi,
            conv_detrainment_qc=zeros,
            conv_detrainment_qi=zeros,
        )
        return tendency, {
            **diagnostics,
            "clouds": clouds,
            "cover_relative_humidity": rel_humidity,
        }
