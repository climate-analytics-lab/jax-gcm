"""Moist, cloud-weighted buoyancy of ECHAM's vertical diffusion.

ECHAM6's ``vdiff.f90`` forms the buoyancy of every interior half level from
the liquid-water potential temperature, the total water and the saturation
specific humidity of the two adjacent full levels, weighted by the cloud cover
of that half level (l.658-700 and 777-799). The one ``zbuoy`` it obtains is
both the numerator of the interior Richardson number ``zri = zbuoy /
max(zshear, zepshr)`` and the buoyancy term of the TKE budget
``zzb = zshear*zsm - zbuoy*zsh``; this module is that quantity, so the two
consumers here (``turbulence_coefficients.compute_richardson_number`` and the
TKE source in ``vertical_diffusion.vertical_diffusion_column``) read the same
value, as ECHAM's do.

The cloud-weighted multipliers :func:`cloud_weighted_buoyancy_multipliers` are
the same arithmetic in ``vdiff.f90`` l.782-789 and in the surface layer's bulk
Richardson number (``mo_surface_land.f90::precalc_land`` l.224-230 and the
ocean and ice analogues), so :mod:`.surface_layer` calls this one function.

Arrays are ``(ncol, nlev)`` with level 0 at the model top, like
:class:`~.vertical_diffusion_types.VDiffState`. A half-level quantity has
``nlev - 1`` entries and ``[:, i]`` is the interface between full levels ``i``
and ``i + 1`` (ECHAM's ``jk = i + 1``).
"""

from typing import NamedTuple

import jax.numpy as jnp

import jcm.constants as c
from jcm.physics.thermodynamics import saturation_specific_humidity
from .vertical_diffusion_types import VDiffState

#: ECHAM's security floor under the squared shear in ``zri``
#: (``zepshr = 1.e-5``, ``vdiff.f90`` l.573) [1/s²]. A floor under a
#: denominator, so it stays in the value.
SHEAR_FLOOR = 1.0e-5

# Floor under the layer-mean temperature in the multipliers' latent-heat
# ratios [K]. Far below any atmospheric temperature (and below ECHAM's table
# range), so it changes no value; it keeps ``L/(cp·T)`` finite for a
# degenerate state.
_T_FLOOR = 100.0


def cloud_weighted_buoyancy_multipliers(latent_heat, temperature, total_water,
                                        saturation_humidity, cloud_cover):
    """Cloud-cover-weighted buoyancy multipliers ``(zdus1, zdus2)``.

    The clear-air multipliers are ``(1 + vtmpc1·q_t, vtmpc1)``; the saturated
    ones add the latent heating of the condensation that a unit of vertical
    displacement causes (``vdiff.f90`` l.780-789)::

        zfux  = L/(cpd·T)                zfox = L/(rd·T)
        zmult1 = 1 + vtmpc1·q_t          zmult2 = zfux·zmult1 − rv/rd
        zmult3 = (rd/rv)·zfox·q_s / (1 + (rd/rv)·zfux·zfox·q_s)
        zmult5 = zmult1 − zmult2·zmult3  zmult4 = zfux·zmult5 − 1
        zdus1  = cc·zmult5 + (1 − cc)·zmult1
        zdus2  = cc·zmult4 + (1 − cc)·vtmpc1

    The buoyancy is then ``zdus1·Δθ_l + zdus2·θ·Δq_t`` (times ``g/θ_v``).
    All arguments broadcast against each other.

    Args:
        latent_heat: ``L`` [J/kg], the condensation heat above the melting
            point and the sublimation heat at and below it.
        temperature: Layer-mean temperature [K].
        total_water: ``q_t``, vapour plus cloud condensate [kg/kg].
        saturation_humidity: ``q_s`` of the layer [kg/kg].
        cloud_cover: Cloud cover ``cc`` of the layer [-].

    Returns:
        ``(zdus1, zdus2)``, dimensionless.

    """
    vtmpc1 = c.vtmpc1
    rv_over_rd = vtmpc1 + 1.0
    rd_over_rv = 1.0 / rv_over_rd
    t_safe = jnp.maximum(temperature, _T_FLOOR)
    fux = latent_heat / (c.cpd * t_safe)
    fox = latent_heat / (c.rd * t_safe)
    mult1 = 1.0 + vtmpc1 * total_water
    mult2 = fux * mult1 - rv_over_rd
    mult3 = (rd_over_rv * fox * saturation_humidity
             / (1.0 + rd_over_rv * fux * fox * saturation_humidity))
    mult5 = mult1 - mult2 * mult3
    mult4 = fux * mult5 - 1.0
    dus1 = cloud_cover * mult5 + (1.0 - cloud_cover) * mult1
    dus2 = cloud_cover * mult4 + (1.0 - cloud_cover) * vtmpc1
    return dus1, dus2


class InteriorBuoyancyTerms(NamedTuple):
    """ECHAM's interior half-level buoyancy and the quantities it is built from.

    Every field is ``(ncol, nlev - 1)``; ``[:, i]`` is the interface between
    full levels ``i`` and ``i + 1``. The names are ECHAM's.
    """

    buoyancy: jnp.ndarray        # zbuoy [1/s²], positive when stably stratified
    shear: jnp.ndarray           # zshear [1/s²]
    dus1: jnp.ndarray            # zdus1, multiplier of the θ_l difference
    dus2: jnp.ndarray            # zdus2, multiplier of the q_t difference
    teldif: jnp.ndarray          # zteldif, ∂θ_l/∂z [K/m]
    qddif: jnp.ndarray           # zqddif, ∂q_t/∂z [kg/kg/m]
    saturation_humidity: jnp.ndarray  # zqssm [kg/kg]


def interior_buoyancy_terms(state: VDiffState) -> InteriorBuoyancyTerms:
    """ECHAM's interior half-level buoyancy, ``vdiff.f90`` l.658-700 and 777-799.

    ``zbuoy`` [1/s²] is the moist, cloud-weighted buoyancy frequency squared
    (positive when stably stratified) and ``zshear`` [1/s²] the squared wind
    shear, both of the interface between adjacent full levels. Every
    thermodynamic quantity is formed on the full levels and then averaged to
    the interface with the mass weights of the two layers
    (``zsdep1 = dp_k/(dp_k + dp_{k+1})``, l.687-690), the saturation humidity
    ``zqssm`` included: it is the average of the full levels' ``ua``-table
    values (:func:`~jcm.physics.thermodynamics.saturation_specific_humidity`),
    not the saturation at the mean temperature. The latent heat is the
    condensation heat at and above the melting point and the sublimation heat
    below it (``FSEL(T - tmelt, alv, als)``, l.674), averaged like the rest,
    so an interface between a liquid and an ice level carries the blend.

    Args:
        state: the column state; reads ``temperature``, ``qv``, ``qc``,
            ``qi``, ``cloud_fraction``, ``u``, ``v``, ``pressure_full``,
            ``pressure_half`` and ``geopotential``.

    Returns:
        :class:`InteriorBuoyancyTerms`.

    """
    t = state.temperature
    p = state.pressure_full
    ph = state.pressure_half
    q = state.qv
    x = state.qc + state.qi                                   # zx: total cloud water
    vtmpc1 = c.vtmpc1

    theta = t * (c.p0 / p) ** (c.rd / c.cpd)                  # zteta1
    theta_v = theta * (1.0 + vtmpc1 * q - x)                  # ztvir1
    latent = jnp.where(t >= c.tmelt, c.alhc, c.alhs)          # zfaxe
    theta_l = theta - (latent / c.cpd) * theta / t * x        # zlteta1
    qs = saturation_specific_humidity(t, p)                   # zqss

    # zsdep1/zsdep2 are formed from differences of the interface pressures
    # exactly as ECHAM forms them.
    w_up = (ph[:, :-2] - ph[:, 1:-1]) / (ph[:, :-2] - ph[:, 2:])
    w_dn = (ph[:, 1:-1] - ph[:, 2:]) / (ph[:, :-2] - ph[:, 2:])

    def mid(field):
        return w_up * field[:, :-1] + w_dn * field[:, 1:]

    zhh = state.geopotential[:, :-1] - state.geopotential[:, 1:]
    qs_mid = mid(qs)
    dus1, dus2 = cloud_weighted_buoyancy_multipliers(
        mid(latent), mid(t), mid(x) + mid(q), qs_mid, mid(state.cloud_fraction))

    g = c.grav
    q_total = q + x
    teldif = (theta_l[:, :-1] - theta_l[:, 1:]) / zhh * g
    qddif = (q_total[:, :-1] - q_total[:, 1:]) / zhh * g
    buoyancy = (teldif * dus1 + mid(theta) * dus2 * qddif) * g / mid(theta_v)
    shear = (((state.u[:, :-1] - state.u[:, 1:]) ** 2
              + (state.v[:, :-1] - state.v[:, 1:]) ** 2) * (g / zhh) ** 2)
    return InteriorBuoyancyTerms(buoyancy, shear, dus1, dus2, teldif, qddif,
                                 qs_mid)


def interior_buoyancy_and_shear(state: VDiffState):
    """``(zbuoy, zshear)`` of :func:`interior_buoyancy_terms`, each ``(ncol, nlev - 1)``."""
    terms = interior_buoyancy_terms(state)
    return terms.buoyancy, terms.shear


def richardson_number(buoyancy, shear):
    """ECHAM's ``zri = zbuoy / MAX(zshear, zepshr)`` (``vdiff.f90`` l.799)."""
    return buoyancy / jnp.maximum(shear, SHEAR_FLOOR)


def interfaces_to_levels(interface_field):
    """Assign each full level the interface above it: ``(ncol, nlev - 1) -> (ncol, nlev)``.

    Level ``k`` takes the value of the interface between levels ``k - 1`` and
    ``k``; the top level repeats the first interface. This is the alignment
    of the mixing-length stability factor (``compute_mixing_length``) and of
    the TKE source, so ``zbuoy`` and ``zshear`` enter the TKE budget at the
    same level as ``ri`` enters the mixing length.
    """
    return jnp.concatenate([interface_field[:, :1], interface_field], axis=1)
