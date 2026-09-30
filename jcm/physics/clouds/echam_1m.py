"""ECHAM6.3 single-moment stratiform cloud scheme (``mo_cloud.f90::cloud``).

:func:`cloud_microphysics_column_sweep` is a transcription of ECHAM6.3-HAM2.3
r7492 ``mo_cloud.f90`` subroutine ``cloud`` (Lohmann and Roeckner 1996;
Roeckner et al. 2003, the ECHAM5 model description, section 10). Every level
runs ECHAM's sections in ECHAM's order, top to bottom, with the falling rain
and snow fluxes, the precipitating fraction and the sedimenting ice flux
carried from level to level within the step:

- 3.1 melting of the incoming snow and of the cloud ice above ``tmelt``;
- 3.2 sublimation of the incoming snow (Lin et al. 1983);
- 3.3 evaporation of the incoming rain (Rotstayn 1997);
- 4   sedimentation of cloud ice, the ``lo2`` phase switch, and the in-cloud
  condensate, with all condensate of a clear cell returned to vapour;
- 5   condensation driven by the step's humidity and temperature increments
  in the cloudy part, 5.4 the whole-box supersaturation check, 5.5 the
  in-cloud update and the promotion of a clear cell with new condensate;
- 6.1 freezing of cloud water below ``cthomi``; 6.2 Bigg and contact
  freezing between ``cthomi`` and ``tmelt``;
- 7.1 Beheng (1994) autoconversion and accretion by rain; 7.2 Levkov et al.
  (1992) aggregation, accretion of ice and riming by snow; 7.3 the flux and
  precipitating-fraction update;
- 8.3 the tendencies and 8.4 the return of condensate below ``ccwmin`` to
  vapour, with the cover write-back.

Section 9 (wet chemistry) belongs to the HAM submodels and section 10 to the
accumulated diagnostics; the ``ktype`` re-typing of section 10 is
:func:`shallow_liquid_convection_type`. Fortran line numbers (``F:``) refer to
r7492 ``src/mo_cloud.f90``.

The test module ``echam_fortran_reference_test.py`` compares this function,
output by output and intermediate by intermediate, with the unmodified
Fortran routine run on the same columns.
"""

import math

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional
from flax import struct

import jcm.constants as c
from jcm.physics.clouds.cloud_utils import (
    moist_isobaric_heat_capacity,
    prescribed_droplet_number,
    sundqvist_condensation,
)
# Saturation vapour pressure (ECHAM's Sonntag (1990) fit) and the lo2 phase
# rule of the ECHAM cloud schemes: one module, shared with the cover. Every
# saturation value and slope in the sweep comes from it.
from jcm.physics.clouds import echam_saturation as _saturation
from jcm.physics.clouds.echam_saturation import lo2_ice_phase
from jcm.physics.surrogate_gradient import with_surrogate_gradient

# Defaults shared by the ``default`` factory's signature AND its legacy-config
# guard (#674): the Beheng rate prefactor and the KK2000 in-cloud qc
# threshold. Kept as module constants so the guard tests the SAME literals the
# signature ships, and they can never drift apart.
_BEHENG_CCRAUT_DEFAULT = 15.0
_KK2000_QC_THRESHOLD_DEFAULT = 1.0e-5

# ECHAM's resolution-dependent defaults at T63 (mo_echam_cloud_params.f90
# lines 208-216), used when no truncation is given.
_T63_CVTFALL = 2.5
_T63_CSECFRL = 5.0e-6
_T63_CLWPRAT = 4.0

# ECHAM's security constants (F:336-338) and ``EPSILON(1._wp)`` of its
# double-precision ``wp`` (the sedimentation floor, F:583).
_ZEPSEC = 1.0e-12
_ZXSEC = 1.0 - _ZEPSEC
_ECHAM_EPSILON = 2.220446049250313e-16

# ECHAM cloud-ice bulk density (mo_echam_cloud_params.f90:56).
_CRHOI = 500.0


@struct.dataclass
class MicrophysicsParameters:
    """Parameters of the ECHAM 1-moment cloud scheme.

    Numeric fields are differentiable pytree leaves. The fields marked static
    configure only the derivative (surrogate widths and cutoffs, see
    ``docs/source/design/surrogate_gradients.md``) or select a jcm option; the
    value of the scheme never depends on a width.

    ``csecfrl`` and ``cthomi`` are ECHAM's single ``csecfrl`` and ``cthomi``
    (``mo_echam_cloud_params.f90`` l.76, l.54), which ECHAM's cover and cloud
    scheme share; jcm's cover holds its own copy
    (``CloudParameters.csecfrl``/``t_ice``). The defaults agree, an override
    of one copy leaves the other unchanged, and ``echam_physics`` warns when
    the copies it builds differ.
    """

    # --- Warm-phase precipitation (mo_echam_cloud_params.f90) ---
    ccraut: jnp.ndarray       # Beheng (1994) autoconversion prefactor (15.0).
                              # Read only in Beheng mode.
    ccraut_kk_threshold: jnp.ndarray  # KK2000 in-cloud qc threshold [kg/kg];
                              # read only in KK2000 mode (a jcm option, #674).
    ccracl: jnp.ndarray       # accretion of cloud water by rain (6.0)
    cauloc: jnp.ndarray       # local-rain accretion factor; 0.0 in ECHAM6.3,
                              # which disables zrac2 and the in-layer snow
                              # (F:918, 995, 1050)
    clmin: jnp.ndarray        # lower bound on zauloc (0.0)
    clmax: jnp.ndarray        # upper bound on zauloc (0.5)
    ceffmin: jnp.ndarray      # minimum effective ICE radius [um] (10.0, F:1030)
    ceffmax: jnp.ndarray      # maximum effective ICE radius [um] (150.0)

    # --- Ice phase ---
    cn0s: jnp.ndarray         # snow intercept parameter [1/m^4] (3e6)
    crhosno: jnp.ndarray      # snow bulk density [kg/m^3] (100)
    ccsaut: jnp.ndarray       # Levkov ice->snow coefficient (95.0)
    ccsacl: jnp.ndarray       # riming efficiency of snow collecting cloud
                              # water (0.10)
    cvtfall: jnp.ndarray      # ice and snow fall-speed factor; resolution
                              # dependent (2.5 at T63, 3.0 at T31/T127/T255)
    csecfrl: jnp.ndarray      # cloud-ice amount above which T < tmelt selects
                              # the ice phase (lo2, F:648); resolution
                              # dependent (5e-6 at T63) [kg/kg]
    cthomi: jnp.ndarray       # homogeneous freezing temperature, tmelt - 35 K

    # --- Thresholds (mo_echam_cloud_params.f90) ---
    cqtmin: jnp.ndarray       # total-water minimum (1e-12) [kg/kg]
    ccwmin: jnp.ndarray       # condensate below which a cell holds no cloud
                              # and its condensate returns to vapour (1e-7,
                              # F:1271-1288) [kg/kg]
    clwprat: jnp.ndarray      # shallow-liquid re-typing ratio (F:1452); a
                              # discrete threshold, zero gradient; resolution
                              # dependent (4.0 at T63)
    epsilon: jnp.ndarray      # cloud-fraction floor of the standalone rate
                              # helpers only; the sweep uses ECHAM's criteria

    # --- Static: jcm options ---
    # Autoconversion scheme, a code-path selector (a Python int, not traced):
    # 0 = Beheng (1994) implicit form, ECHAM's 1M scheme (default);
    # 1 = Khairoutdinov & Kogan (2000), a jcm option.
    autoconversion_scheme: int = struct.field(pytree_node=False, default=0)
    # Whether the MACv2-SP Twomey factor scales the droplet number of the
    # autoconversion. On by default: jcm's aerosol-cloud interaction acts on
    # the precipitation formation as well as on the radiation, a documented
    # departure from ECHAM with simple plumes (MPI-ESM1.2 scales the
    # radiation's droplet number only, Mauritsen et al. 2019, section 2.2),
    # decided by the maintainer (#932, 2026-09-30). The droplet number of the
    # freezing (section 6.2) is ECHAM's unscaled ``acdnc`` either way.
    autoconversion_twomey: bool = struct.field(pytree_node=False, default=True)

    # The truncation whose resolution defaults the parameters were built from
    # (metadata for the host's grid check; not read by the scheme).
    defaults_truncation: Optional[int] = struct.field(pytree_node=False, default=63)

    # --- Static: surrogate-derivative widths (0 = reference derivative) ---
    # Temperature width [K] of the logistic surrogates of the phase switches:
    # lo2 (F:647-650, re-evaluated F:763-764), the melt of all cloud ice above
    # tmelt (F:438-439) and the freezing of all cloud water at or below
    # cthomi (F:821-828).
    phase_switch_width: float = struct.field(pytree_node=False, default=1.0)
    # Width of the logistic surrogate of lo2's ice-memory criterion
    # (ice > csecfrl), as a fraction of csecfrl; 0 keeps that criterion's
    # reference (zero) derivative.
    phase_switch_ice_width: float = struct.field(pytree_node=False, default=0.1)
    # Below this ice content [kg/m^3] the derivative of the ice fall speed
    # cvtfall*(rho*xi)**0.16 is that of a parabola through the origin, C1 at
    # the cutoff, instead of the unbounded slope of the power law, and below
    # zero (negative provisional ice) that of the parabola's tangent at the
    # origin; the largest slope is then (2 - 0.16)*cutoff**(-0.84), about
    # 1.4e6, for every ice content. Thin cirrus holds about 1e-6 to
    # 1e-4 kg/m^3, so 1e-7 lies below real cloud.
    ice_fall_speed_gradient_cutoff: float = struct.field(pytree_node=False,
                                                         default=1.0e-7)
    # Below this in-cloud liquid [kg/kg] the derivative of the contact-freezing
    # radius (zradl, F:866-869) is that of the same C1 parabola; 1e-10 kg/kg is
    # a droplet radius of about 0.08 um at the prescribed droplet numbers.
    contact_freezing_liquid_cutoff: float = struct.field(pytree_node=False,
                                                         default=1.0e-10)
    # Logistic width [kg/kg] of the surrogate of the KK2000 threshold gate.
    smooth_ccraut: float = struct.field(pytree_node=False, default=5.0e-5)

    SCHEME_BEHENG = 0
    SCHEME_KK2000 = 1
    # Documented string aliases -> canonical int flag. A class attribute (no
    # annotation, so not a dataclass field), so both the ``default()`` door and
    # the Hydra-override door (``with_field_overrides``) map through it.
    _SCHEME_ALIASES = {"beheng": SCHEME_BEHENG, "kk2000": SCHEME_KK2000}

    @classmethod
    def _normalize_scheme(cls, scheme):
        """Map a string alias to the canonical int flag; pass ints through."""
        if isinstance(scheme, str):
            try:
                return cls._SCHEME_ALIASES[scheme]
            except KeyError:
                raise ValueError(
                    f"Unknown autoconversion_scheme {scheme!r}; expected one of "
                    f"{sorted(cls._SCHEME_ALIASES)} or the int constants "
                    f"{cls.SCHEME_BEHENG} (Beheng) / {cls.SCHEME_KK2000} (KK2000)."
                )
        return scheme

    def __post_init__(self):
        """Store the scheme selector as its canonical int, whichever door built it."""
        object.__setattr__(
            self, "autoconversion_scheme",
            self._normalize_scheme(self.autoconversion_scheme),
        )

    @classmethod
    def default(cls, ccraut=_BEHENG_CCRAUT_DEFAULT,
                ccraut_kk_threshold=_KK2000_QC_THRESHOLD_DEFAULT,
                ccracl=6.0, cauloc=0.0, clmin=0.0, clmax=0.5,
                ceffmin=10.0, ceffmax=150.0, cn0s=3.0e6,
                crhosno=100.0, ccsaut=95.0, ccsacl=0.1,
                cvtfall=None, csecfrl=None, cthomi=None,
                cqtmin=1.0e-12, ccwmin=1.0e-7, clwprat=None,
                epsilon=1.0e-12, autoconversion_scheme=0,
                truncation: int | None = 63, nlev: int | None = None,
                **static) -> 'MicrophysicsParameters':
        """Return the default parameters for a grid.

        ``cvtfall``, ``csecfrl`` and ``clwprat`` are tunable parameters whose
        ECHAM values depend on resolution (``mo_echam_cloud_params.f90``
        lines 197-240). A value passed here is used as given. A value left
        ``None`` is the resolution default for the spectral ``truncation``:
        ECHAM's T63 values for 63 (the default, also used without a grid),
        otherwise the ECHAM table interpolated in truncation between ECHAM's
        rows (``echam_cloud_defaults``); ``None`` is a non-spectral grid, for
        which that table returns the T63 row with a warning. ``nlev`` is
        accepted for a uniform grid signature; no 1M value depends on it.
        ``cthomi`` defaults to ``tmelt - 35`` with the live ``tmelt``. Keyword
        arguments naming a static field set it.
        """
        del nlev
        if cvtfall is None or csecfrl is None or clwprat is None:
            if truncation == 63:
                row = {"cvtfall": _T63_CVTFALL, "csecfrl": _T63_CSECFRL,
                       "clwprat": _T63_CLWPRAT}
            else:
                from jcm.physics.clouds.echam_cloud_defaults import (
                    echam_cloud_defaults)
                row = echam_cloud_defaults(truncation)
            cvtfall = row["cvtfall"] if cvtfall is None else cvtfall
            csecfrl = row["csecfrl"] if csecfrl is None else csecfrl
            clwprat = row["clwprat"] if clwprat is None else clwprat
        if cthomi is None:
            cthomi = c.tmelt - 35.0
        params = cls(
            ccraut=jnp.array(ccraut),
            ccraut_kk_threshold=jnp.array(ccraut_kk_threshold),
            ccracl=jnp.array(ccracl),
            cauloc=jnp.array(cauloc),
            clmin=jnp.array(clmin),
            clmax=jnp.array(clmax),
            ceffmin=jnp.array(ceffmin),
            ceffmax=jnp.array(ceffmax),
            cn0s=jnp.array(cn0s),
            crhosno=jnp.array(crhosno),
            ccsaut=jnp.array(ccsaut),
            ccsacl=jnp.array(ccsacl),
            cvtfall=jnp.array(cvtfall),
            csecfrl=jnp.array(csecfrl),
            cthomi=jnp.array(cthomi),
            cqtmin=jnp.array(cqtmin),
            ccwmin=jnp.array(ccwmin),
            clwprat=jnp.array(clwprat),
            epsilon=jnp.array(epsilon),
            autoconversion_scheme=autoconversion_scheme,
            defaults_truncation=truncation,
            **static,
        )
        # Run the cross-field validation on this door too (the Hydra door runs
        # it through ``with_field_overrides``).
        params.validate()
        return params

    def validate(self) -> None:
        """Raise on an illegal field combination (config time, concrete values).

        Legacy-config guard (#674): ``ccraut`` was once the single overloaded
        field, and legacy KK2000 configs documented it as the qc threshold.
        Such a config still composes because ``ccraut`` remains the Beheng
        field, but the KK2000 branch reads ``ccraut_kk_threshold``, so the
        override would be silently ignored. ``rel_tol`` accepts the float32
        round trip of the stored defaults.
        """
        if (int(self.autoconversion_scheme) == self.SCHEME_KK2000
                and not math.isclose(float(self.ccraut),
                                     _BEHENG_CCRAUT_DEFAULT, rel_tol=1e-6)
                and math.isclose(float(self.ccraut_kk_threshold),
                                 _KK2000_QC_THRESHOLD_DEFAULT, rel_tol=1e-6)):
            raise ValueError(
                "autoconversion_scheme='kk2000' with a non-default "
                f"ccraut={float(self.ccraut)} but ccraut_kk_threshold left at "
                f"its default ({_KK2000_QC_THRESHOLD_DEFAULT}). Legacy configs "
                "documented ccraut AS the KK2000 threshold, but the KK2000 "
                "branch now reads the dedicated 'ccraut_kk_threshold' field "
                "(in-cloud qc, kg/kg); 'ccraut' is the Beheng (1994) rate "
                "prefactor and is UNUSED under kk2000. Migrate this config: "
                "rename ccraut -> ccraut_kk_threshold."
            )


class SweepIntermediates(NamedTuple):
    """Per-level intermediates of the sweep under their ECHAM names.

    Each is ``(nlev, *horiz)``. Amounts are per step (kg/kg over ``dt``), as
    in the Fortran. These are what the Fortran comparison checks beside the
    outputs.
    """

    zevp: jnp.ndarray      # rain evaporation (3.3)
    zsub: jnp.ndarray      # snow sublimation (3.2)
    zsmlt: jnp.ndarray     # snow melt, incl. the bottom-level melt (3.1, 7.3)
    zimlt: jnp.ndarray     # cloud-ice melt (3.1)
    zqsed: jnp.ndarray     # sedimentation change of cloud ice (4)
    zlo2: jnp.ndarray      # phase switch of section 5, 1 = ice (4)
    zxlevap: jnp.ndarray   # clear-cell liquid returned to vapour (4)
    zxievap: jnp.ndarray   # clear-cell ice returned to vapour (4)
    zcnd: jnp.ndarray      # condensation, after 5.4 (5)
    zdep: jnp.ndarray      # deposition, after 5.4 (5)
    zclcaux: jnp.ndarray   # cover the microphysics saw, after 5.5
    zfrl: jnp.ndarray      # freezing of cloud water (6.1, 6.2)
    zrpr: jnp.ndarray      # rain production (7.1)
    zspr: jnp.ndarray      # snow production (7.2)
    zsacl: jnp.ndarray     # riming (7.2)
    zclcpre: jnp.ndarray   # precipitating fraction leaving the level (7.3)
    zdxlcor: jnp.ndarray   # liquid correction of 8.4, per second
    zdxicor: jnp.ndarray   # ice correction of 8.4, per second
    prelhum: jnp.ndarray   # relative humidity diagnostic (5)


class MicrophysicsState(NamedTuple):
    """Fluxes and diagnostics of the sweep. Per-level fields are ``(nlev, *horiz)``."""

    # Grid-mean precipitation fluxes LEAVING each level [kg/m^2/s]; the bottom
    # row is the surface precipitation. ``snow_flux`` adds the sedimenting
    # cloud-ice flux, so it is the total falling frozen water.
    rain_flux: jnp.ndarray
    snow_flux: jnp.ndarray
    rain_source: jnp.ndarray     # rain production of the level (zcons2*zdp*zrpr)
    snow_source: jnp.ndarray     # snow production incl. riming
    rain_evap_flux: jnp.ndarray  # rain evaporation (zcons2*zdp*zevp)
    snow_sublimation_flux: jnp.ndarray  # snow sublimation (zcons2*zdp*zsub)
    qc_in_cloud: jnp.ndarray     # input liquid / cover (0 where clear)
    qi_in_cloud: jnp.ndarray     # input ice / cover (0 where clear)
    autoconv_rate: jnp.ndarray   # grid-mean autoconversion [kg/kg/s]
    accretion_rate: jnp.ndarray  # grid-mean accretion by rain [kg/kg/s]
    melting_rate: jnp.ndarray    # snow + cloud-ice melt [kg/kg/s]
    freezing_rate: jnp.ndarray   # freezing of cloud water [kg/kg/s]
    precip_rain: jnp.ndarray     # surface rain [kg/m^2/s] (*horiz)
    precip_snow: jnp.ndarray     # surface snow [kg/m^2/s] (*horiz)
    cloud_fraction: jnp.ndarray  # cover after the 8.4 write-back (paclc)
    intermediates: SweepIntermediates
    echam_locals: dict           # ECHAM-named locals (``LevelOutputs``)


class MicrophysicsTendencies(NamedTuple):
    """Tendencies of the sweep, ``(nlev, *horiz)``."""

    dtedt: jnp.ndarray          # temperature [K/s]
    dqdt: jnp.ndarray           # specific humidity [kg/kg/s]
    dqcdt: jnp.ndarray          # cloud water [kg/kg/s]
    dqidt: jnp.ndarray          # cloud ice [kg/kg/s]
    dqrdt: jnp.ndarray          # rain (structural zero: rain is a flux)
    dqsdt: jnp.ndarray          # snow (structural zero: snow is a flux)


# ---------------------------------------------------------------------------
# Saturation (ECHAM's lookup tables) and the lo2 phase switch
# ---------------------------------------------------------------------------

def _es_and_derivative(temperature, ice):
    """``(e_s, de_s/dT)`` [Pa, Pa/K] over ice or over water at every temperature.

    From :mod:`echam_saturation`; the slope is the analytic one ECHAM
    tabulates beside the value.
    """
    if ice:
        es = _saturation.es_ice(temperature)
        return es, es * _saturation.dlnes_dT_ice(temperature)
    es = _saturation.es_water(temperature)
    return es, es * _saturation.dlnes_dT_water(temperature)


def _ua(temperature):
    """ECHAM's mixed table ``ua``, ``dua`` (convect_tables:262-309).

    ``e_s·rd/rv`` and its derivative, over ice for ``T <= tmelt`` and over
    water above.
    """
    ice = temperature <= c.tmelt
    es_i, des_i = _es_and_derivative(temperature, ice=True)
    es_w, des_w = _es_and_derivative(temperature, ice=False)
    return (jnp.where(ice, es_i, es_w) * (c.rd / c.rv),
            jnp.where(ice, des_i, des_w) * (c.rd / c.rv))


def _uaw(temperature):
    """ECHAM's water table ``uaw``, ``duaw`` at every temperature."""
    es_w, des_w = _es_and_derivative(temperature, ice=False)
    return es_w * (c.rd / c.rv), des_w * (c.rd / c.rv)


def _ub(temperature):
    """ECHAM ``lookup_ubc``: ``(alv or als)/cpd · d ln e_s/dT`` (convect_tables:339-346).

    Ice and ``als`` for ``T <= tmelt``. Dry ``cpd``, as in ECHAM.
    """
    ice = temperature <= c.tmelt
    es_i, des_i = _es_and_derivative(temperature, ice=True)
    es_w, des_w = _es_and_derivative(temperature, ice=False)
    return jnp.where(ice, c.alhs / c.cpd * (des_i / es_i),
                     c.alhc / c.cpd * (des_w / es_w))


# ---------------------------------------------------------------------------
# Switches and power laws: ECHAM's values, bounded surrogate derivatives
# ---------------------------------------------------------------------------

def _step(distance, inclusive):
    """1 where ``distance > 0`` (``>= 0`` if ``inclusive``), else 0."""
    on = distance >= 0.0 if inclusive else distance > 0.0
    return jnp.where(on, jnp.ones_like(distance), jnp.zeros_like(distance))


def temperature_switch_pair(width, inclusive=False):
    """``(exact, surrogate)`` of a 0/1 switch on a temperature distance.

    ``exact(d)`` is 1 past the threshold (``d > 0``, or ``d >= 0`` if
    ``inclusive``); ``surrogate(d) = sigmoid(d/width)``.
    """
    def exact(d):
        return _step(d, inclusive)

    def surrogate(d):
        return jax.nn.sigmoid(d / width)

    return exact, surrogate


def temperature_switch(distance, width, inclusive=False):
    """ECHAM's 0/1 switch on a temperature distance, surrogate derivative.

    ``width = 0`` returns the step with its reference (zero) derivative.
    """
    exact, surrogate = temperature_switch_pair(width, inclusive)
    if width == 0:
        return exact(distance)
    return with_surrogate_gradient(exact, surrogate)(distance)


def ice_phase_pair(width, ice_width):
    """``(exact, surrogate)`` of ECHAM's ``lo2`` as a number (1 = ice).

    Both take ``(T, xi, csecfrl, cthomi)``. ``exact`` is
    :func:`lo2_ice_phase` (F:647-650). ``surrogate`` is the
    same logical formula with each comparison a logistic:
    ``s_cold + (1 - s_cold)·s_warm·s_ice``, ``s_cold`` in
    ``(cthomi - T)/width``, ``s_warm`` in ``(tmelt - T)/width``, ``s_ice`` in
    ``(xi - csecfrl)/(ice_width·csecfrl)``. ``ice_width = 0`` keeps the
    reference step ``xi > csecfrl`` in ``s_ice``, so the ice criterion has its
    reference (zero) derivative while the temperature comparisons keep theirs
    of width ``width``.
    """
    def exact(t, xi, csec, cth):
        ice = lo2_ice_phase(t, xi, csec, cth)
        return jnp.where(ice, jnp.ones_like(t), jnp.zeros_like(t))

    def surrogate(t, xi, csec, cth):
        s_cold = jax.nn.sigmoid((cth - t) / width)
        s_warm = jax.nn.sigmoid((c.tmelt - t) / width)
        if ice_width == 0:
            # The ice criterion keeps its reference (zero) derivative.
            s_ice = jnp.where(xi > csec, jnp.ones_like(t), jnp.zeros_like(t))
        else:
            s_ice = jax.nn.sigmoid((xi - csec) / (ice_width * csec))
        return s_cold + (1.0 - s_cold) * s_warm * s_ice

    return exact, surrogate


def ice_phase_weight(temperature, cloud_ice, csecfrl, cthomi, width, ice_width):
    """ECHAM's ``lo2`` as 1 (ice) or 0 (liquid), with a surrogate derivative.

    See :func:`ice_phase_pair`. ``width = 0`` keeps the reference derivative
    of the whole switch; ``ice_width = 0`` that of its ice criterion alone.
    """
    exact, surrogate = ice_phase_pair(width, ice_width)
    if width == 0:
        return exact(temperature, cloud_ice, csecfrl, cthomi)
    shape = jnp.broadcast_shapes(jnp.shape(temperature), jnp.shape(cloud_ice))
    dtype = jnp.result_type(temperature)
    args = [jnp.broadcast_to(jnp.asarray(a, dtype=dtype), shape)
            for a in (temperature, cloud_ice, csecfrl, cthomi)]
    return with_surrogate_gradient(exact, surrogate)(*args)


def _c1_power(y, exponent, cutoff):
    """``y**exponent`` above ``cutoff``; below it the parabola through the origin
    that matches the value and slope at ``cutoff``, continued below the origin
    by its tangent there.

    With ``a = exponent``, ``t = min(y, cutoff)/cutoff`` and
    ``t₊ = max(t, 0)``: ``cutoff**a·((2 - a)·t + (a - 1)·t₊²)`` below the
    cutoff and ``max(y, cutoff)**a`` above it. The ``min``/``max`` keep the
    unselected branch free of a singular slope. The slope is
    ``(2 - a)·cutoff**(a - 1)`` at and below the origin and falls to
    ``a·cutoff**(a - 1)`` at the cutoff, so that is its bound for every
    ``y``, negative arguments included: the provisional ice of the sweep is
    negative after an advective undershoot, where the reference value is
    flat on ECHAM's floor. On ``[0, cutoff]`` the parabola differs from
    ``y**a`` by less than ``cutoff**a``.
    """
    t = jnp.minimum(y, cutoff) / cutoff
    t_pos = jnp.maximum(t, 0.0)
    low = cutoff ** exponent * ((2.0 - exponent) * t
                                + (exponent - 1.0) * t_pos * t_pos)
    high = jnp.maximum(y, cutoff) ** exponent
    return jnp.where(y < cutoff, low, high)


def ice_fall_speed_pair(cutoff):
    """``(exact, surrogate)`` of the power law of the ice fall speed.

    Both take ``(ice_density, floor)``, the ice content ``ρ·xi`` [kg/m^3] and
    ECHAM's floor on it. ``exact = max(ice_density, floor)**0.16``, the
    ``(ρ·zxip1)**0.16`` of F:583-592 with ``zxip1 = max(xi, EPSILON(1._wp))``
    when the caller passes ``floor = ρ·EPSILON(1._wp)``. ``surrogate`` is
    :func:`_c1_power` of ``ice_density`` with the static ``cutoff``; it does
    not read ``floor``, so the derivative with respect to the floor, a
    numerical guard, is zero, and a cell with no ice gets the parabola's slope
    at the origin like a cell with a trace of ice.
    """
    def exact(ice_density, floor):
        return jnp.maximum(ice_density, floor) ** 0.16

    def surrogate(ice_density, floor):
        del floor
        return _c1_power(ice_density, 0.16, cutoff)

    return exact, surrogate


def ice_fall_speed(air_density, cloud_ice, cvtfall, cutoff):
    """ECHAM's ice fall speed ``cvtfall·(ρ·max(xi, EPSILON))**0.16`` [m/s].

    F:581-592, with ECHAM's double-precision ``EPSILON(1._wp)`` floor. The
    slope of the power law is unbounded towards zero ice; below the static
    ``cutoff`` [kg/m^3] the derivative is that of the C1 parabola of
    :func:`ice_fall_speed_pair`. ``cutoff = 0`` keeps the reference
    derivative.
    """
    exact, surrogate = ice_fall_speed_pair(cutoff)
    power = exact if cutoff == 0.0 else with_surrogate_gradient(exact, surrogate)
    ice_density = jnp.asarray(air_density) * jnp.asarray(cloud_ice)
    floor = jnp.broadcast_to(jnp.asarray(air_density) * _ECHAM_EPSILON,
                             jnp.shape(ice_density)).astype(ice_density.dtype)
    return cvtfall * power(ice_density, floor)


def contact_radius_pair(liquid_cutoff):
    """``(exact, surrogate)`` of the contact-freezing droplet radius [m].

    Both take the in-cloud liquid ``zxlb`` [kg/kg] and ``zfrho = ρ/(ρw·N)``.
    ``exact = (0.75·zxlb·zfrho/π)**(1/3)`` (F:866-869); ``surrogate`` is
    ``(0.75·zfrho/π)**(1/3)`` times :func:`_c1_power` of ``zxlb`` below
    ``liquid_cutoff``. The surrogate acts on the liquid, not on the droplet
    volume (~1e-31 m^3): a cutoff on the volume would put ``1/cutoff**2``
    beyond float32 range, and XLA's reassociation of ``(y/c)·(y/c)`` then
    turns the parabola into ``inf·0``.
    """
    def exact(zxlb, zfrho):
        base = 0.75 * zxlb * zfrho / jnp.pi
        positive = base > 0.0
        return jnp.where(positive,
                         jnp.where(positive, base, 1.0) ** (1.0 / 3.0), 0.0)

    def surrogate(zxlb, zfrho):
        prefactor = (0.75 * zfrho / jnp.pi) ** (1.0 / 3.0)
        return prefactor * _c1_power(zxlb, 1.0 / 3.0, liquid_cutoff)

    return exact, surrogate


def contact_freezing_radius(zxlb, zfrho, liquid_cutoff):
    """Mean droplet radius of contact freezing [m], bounded surrogate slope.

    See :func:`contact_radius_pair`; ``liquid_cutoff = 0`` keeps the
    reference derivative.
    """
    exact, surrogate = contact_radius_pair(liquid_cutoff)
    if liquid_cutoff == 0:
        return exact(zxlb, zfrho)
    shape = jnp.broadcast_shapes(jnp.shape(zxlb), jnp.shape(zfrho))
    dtype = jnp.result_type(zxlb)
    args = [jnp.broadcast_to(jnp.asarray(a, dtype=dtype), shape) for a in (zxlb, zfrho)]
    return with_surrogate_gradient(exact, surrogate)(*args)


# ---------------------------------------------------------------------------
# Precipitation formation rates
# ---------------------------------------------------------------------------

def _beheng_depletion(zxlb, air_density, droplet_number_m3, dt, ccraut):
    """In-cloud liquid autoconverted over ``dt``, Beheng (1994) implicit (F:969-993).

    ``zraut = zxlb·(1 − (1 + c·dt·3.7·zxlb**3.7)**(−1/3.7))`` with
    ``c = ccraut·1.2e27/ρ·(N·1e-6)**−3.3·(ρ·1e-3)**4.7``. The droplet number
    is floored at 1 cm^-3 so the ``N**-3.3`` stays finite; ECHAM's prescribed
    ``acdnc`` is at least 20 cm^-3, so the floor never acts on its values.
    """
    zexm1 = 4.7 - 1.0
    zexp = -1.0 / zexm1
    ztmp1 = (ccraut * 1.2e27) / air_density
    ztmp2 = jnp.maximum(droplet_number_m3 * 1.0e-6, 1.0) ** (-3.3)
    ztmp3 = (air_density * 1.0e-3) ** 4.7
    ztmp4 = jnp.maximum(zxlb, 0.0) ** zexm1
    rate = ztmp1 * ztmp2 * ztmp3
    ztmp4 = (1.0 + rate * dt * zexm1 * ztmp4) ** zexp
    return zxlb * (1.0 - ztmp4)


def autoconversion_beheng(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters,
) -> jnp.ndarray:
    """Grid-mean Beheng (1994) autoconversion rate [kg/kg/s] (F:969-993).

    The implicit form keeps the depletion in ``[0, qc]`` for any ``dt``.

    Args:
        cloud_water: grid-mean cloud water [kg/kg].
        cloud_fraction: cloud fraction.
        air_density: air density [kg/m^3].
        droplet_number: droplet number PER KG of air [1/kg].
        dt: time step [s].
        config: parameters (``ccraut``, ``epsilon``).

    """
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / jnp.maximum(cloud_fraction, config.epsilon),
        0.0,
    )
    depletion = _beheng_depletion(
        qc_in_cloud, air_density, droplet_number * air_density, dt, config.ccraut)
    return depletion / dt * cloud_fraction


def _kk2000_gate(qc_in_cloud, threshold, width):
    """0/1 gate ``qc > threshold`` with a logistic surrogate derivative."""
    exact, surrogate = temperature_switch_pair(width, inclusive=False)
    distance = qc_in_cloud - threshold
    if width == 0:
        return exact(distance)
    return with_surrogate_gradient(exact, surrogate)(distance)


def autoconversion_kk2000(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters,
) -> jnp.ndarray:
    """Grid-mean Khairoutdinov and Kogan (2000) autoconversion [kg/kg/s].

    ``1350·qc**2.47·N**−1.79`` (qc in-cloud [kg/kg], N [cm^-3]; KK2000 eq. 29)
    above the in-cloud threshold ``ccraut_kk_threshold``. A jcm option (ECHAM's
    1M scheme uses Beheng), off by default. The threshold is a hard gate in
    the value; its derivative is that of a logistic of width
    ``smooth_ccraut`` (static), so the threshold stays calibratable. The
    explicit rate can exceed ``qc/dt``; the sweep caps the depletion at the
    in-cloud liquid.

    Args:
        cloud_water: grid-mean cloud water [kg/kg].
        cloud_fraction: cloud fraction.
        air_density: air density [kg/m^3].
        droplet_number: droplet number PER KG of air [1/kg].
        dt: unused (explicit rate); kept for the dispatcher signature.
        config: parameters.

    """
    del dt
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / jnp.maximum(cloud_fraction, config.epsilon),
        0.0,
    )
    nc_cm3 = droplet_number * air_density * 1e-6
    has_qc = qc_in_cloud > 0.0
    qc_safe = jnp.where(has_qc, qc_in_cloud, 1.0)
    gate = _kk2000_gate(qc_in_cloud, config.ccraut_kk_threshold,
                        config.smooth_ccraut)
    rate = jnp.where(
        has_qc,
        gate * 1350.0 * qc_safe ** 2.47 * (nc_cm3 + config.epsilon) ** (-1.79),
        0.0,
    )
    return rate * cloud_fraction


def autoconversion(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters,
) -> jnp.ndarray:
    """Dispatch to Beheng or KK2000 by ``config.autoconversion_scheme``."""
    scheme = (autoconversion_kk2000
              if config.autoconversion_scheme == MicrophysicsParameters.SCHEME_KK2000
              else autoconversion_beheng)
    return scheme(cloud_water, cloud_fraction, air_density, droplet_number, dt,
                  config)


def _levkov_depletion(zxib, air_density, dt, config, return_radius=False):
    """In-cloud ice aggregated to snow over ``dt`` (F:996-1001, 1026-1052).

    Levkov et al. (1992) with the Moss (1995) effective radius
    ``zrieff = 83.8·(IWC g/m^3)**0.216`` clipped to ``[ceffmin, ceffmax]``,
    integrated implicitly: ``zxib·(1 − 1/(1 + ccsaut/zdt2·dt·zxib))``. With
    ``return_radius`` the pair ``(zsaut, zrieff)``.
    """
    ztmp3 = (zxib * air_density * 1000.0) ** 0.216
    zrieff = jnp.minimum(jnp.maximum(83.8 * ztmp3, config.ceffmin), config.ceffmax)
    zrih = jnp.log10(jnp.sqrt(5113188.0 + 2809.0 * zrieff * zrieff * zrieff) - 2261.0)
    zqrho_033 = (1.3 / air_density) ** 0.33
    zc1 = 17.5 * air_density / _CRHOI * zqrho_033
    zdt2 = -6.0 / zc1 * (zrih / 3.0 - 2.0)
    rate = config.ccsaut / zdt2
    zsaut = zxib * (1.0 - 1.0 / (1.0 + rate * dt * zxib))
    return (zsaut, zrieff) if return_radius else zsaut


def ice_autoconversion(
    cloud_ice: jnp.ndarray,
    temperature: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters,
    air_density: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Grid-mean Levkov ice-to-snow conversion rate [kg/kg/s] (F:1026-1052).

    Args:
        cloud_ice: grid-mean ice [kg/kg].
        temperature: unused; kept for callers of the former signature.
        cloud_fraction: cloud fraction.
        dt: time step [s].
        config: parameters (``ccsaut``, ``ceffmin``, ``ceffmax``).
        air_density: air density [kg/m^3]; 1 if omitted.

    """
    del temperature
    # Built at call time: a jax array as a ``def`` default would initialise
    # the JAX backend on import (#859).
    if air_density is None:
        air_density = jnp.array(1.0)
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / jnp.maximum(cloud_fraction, config.epsilon),
        0.0,
    )
    has_ice = qi_in_cloud > 0.0
    depletion = _levkov_depletion(
        jnp.where(has_ice, qi_in_cloud, 1.0), air_density, dt, config)
    return cloud_fraction * jnp.where(has_ice, depletion, 0.0) / dt


def lonacc_levels(inversion_level, vertical_velocity, jbmin, jbmax, nlev):
    """Levels where ECHAM sets ``zauloc = 0`` (``lonacc``, F:917-928).

    At the inversion level ``knvb`` of the cover and the level below it, when
    ``knvb`` lies in ``[jbmin, jbmax]`` and the air subsides there
    (``pvervel > 0``). Level indices are 0-based and top first, as the
    sweep's. ``lonacc`` is ``.TRUE.`` in ECHAM6.3.

    Args:
        inversion_level: ``knvb`` per column (*horiz), int.
        vertical_velocity: ``pvervel`` [Pa/s] (nlev, *horiz).
        jbmin, jbmax: ECHAM's inversion-level search bounds (0-based).
        nlev: number of levels.

    Returns:
        Boolean mask (nlev, *horiz).

    """
    k = jnp.arange(nlev).reshape((nlev,) + (1,) * jnp.ndim(inversion_level))
    jb = jnp.asarray(inversion_level)[jnp.newaxis]
    in_window = (jb >= jbmin) & (jb <= jbmax)
    return in_window & (vertical_velocity > 0.0) & ((k == jb) | (k == jb + 1))


# ---------------------------------------------------------------------------
# The column sweep: mo_cloud.f90::cloud
# ---------------------------------------------------------------------------

def _blend(weight, if_one, if_zero):
    """``weight·if_one + (1 − weight)·if_zero``: ECHAM's ``FSEL`` for a 0/1 weight.

    For a weight of exactly 0 or 1 the result is the selected operand bit for
    bit, so a switch built with a surrogate derivative selects exactly as
    ECHAM does while its derivative flows through the weight.
    """
    return weight * if_one + (1.0 - weight) * if_zero


def _safe_pow(base, exponent, active):
    """``base**exponent`` where ``active`` and ``base > 0``, 0 elsewhere.

    ECHAM evaluates these powers of a flux wherever its gate holds, including
    a flux of exactly zero (``zxrp1`` wherever ``zclcpre > 0``, F:934-947),
    where the value is 0. A zero base with a fractional exponent has an
    infinite derivative, so the power is evaluated on a base of 1 there
    (double ``where``) and the value set to 0, which is the value.
    """
    positive = active & (base > 0.0)
    safe = jnp.where(positive, base, 1.0)
    return jnp.where(positive, safe ** exponent, 0.0)


class LevelInputs(NamedTuple):
    """One level of the sweep's inputs, in ECHAM's terms (see the sweep)."""

    tm1: jnp.ndarray         # ptm1
    qm1: jnp.ndarray         # pqm1
    dtemp: jnp.ndarray       # ztmst*ptte
    dq: jnp.ndarray          # ztmst*pqte
    xlp: jnp.ndarray         # pxlm1 + ztmst*(pxlte + pxtecl)
    xip: jnp.ndarray         # pxim1 + ztmst*(pxite + pxteci)
    paclc: jnp.ndarray       # cover
    p: jnp.ndarray           # papm1
    dp: jnp.ndarray          # layer pressure thickness
    rho: jnp.ndarray         # air density
    dz: jnp.ndarray          # layer depth [m]
    cdnc: jnp.ndarray        # pacdnc [1/m^3]
    cdnc_aut: jnp.ndarray    # droplet number of the autoconversion [1/m^3]
    pcair: jnp.ndarray       # moist heat capacity
    zauloc_off: jnp.ndarray  # lonacc: zauloc = 0 here
    top: jnp.ndarray         # first level (ECHAM jk = 1)
    bottom: jnp.ndarray      # last level (ECHAM jk = klev)


class LevelOutputs(NamedTuple):
    """One level of the sweep's outputs."""

    ztte: jnp.ndarray            # temperature tendency [K/s], incl. 8.4
    zqvte: jnp.ndarray           # humidity tendency [1/s]
    zxlte: jnp.ndarray           # liquid tendency [1/s]
    zxite: jnp.ndarray           # ice tendency [1/s]
    rain_source: jnp.ndarray     # zcons2*zdp*zrpr [kg/m^2/s]
    snow_source: jnp.ndarray     # zcons2*zdp*(zspr + zsacl)
    rain_evap_flux: jnp.ndarray  # zcons2*zdp*zevp
    snow_sub_flux: jnp.ndarray   # zcons2*zdp*zsub
    autoconv_rate: jnp.ndarray   # grid-mean autoconversion [kg/kg/s]
    accretion_rate: jnp.ndarray  # grid-mean accretion by rain [kg/kg/s]
    rain_flux: jnp.ndarray       # rain flux leaving the level
    snow_flux: jnp.ndarray       # snow + sedimenting-ice flux leaving the level
    cloud_fraction: jnp.ndarray  # cover after the 8.4 write-back
    intermediates: SweepIntermediates
    # Every local the Fortran reference harness records, under its ECHAM
    # name (the reference data's ``diag/<variant>/*`` keys), at the same point
    # of the routine, zero where ECHAM does not reach it; plus ``prelhum``.
    echam_locals: dict


def _sweep_level(carry, inputs: LevelInputs, config: MicrophysicsParameters, dt):
    """One level of ``mo_cloud.f90::cloud``: sections 1.3 to 8.4.

    ``carry`` is ``(zrfl, zsfl, zclcpre, zxiflux)``, the rain and snow fluxes,
    the precipitating fraction and the sedimenting ice flux arriving from the
    level above. Returns the carry for the level below and the level's
    outputs.
    """
    phase_width = config.phase_switch_width
    ice_width = config.phase_switch_ice_width
    qsec = 1.0 - config.cqtmin
    use_kk2000 = config.autoconversion_scheme == MicrophysicsParameters.SCHEME_KK2000
    tiny = jnp.finfo(jnp.result_type(inputs.tm1)).tiny
    zrfl, zsfl, zclcpre, zxiflux = carry
    # ECHAM's locals at the points the reference harness records them (see
    # ``LevelOutputs.echam_locals``); zero where ECHAM does not reach them.
    loc = {"zclcpre_in": zclcpre, "zrfl_in": zrfl, "zsfl_in": zsfl,
           "zxiflux_in": zxiflux}
    (tm1, qm1, dtemp, dq, xlp, xip, paclc, p, dp, rho, dz, cdnc, cdnc_aut,
     pcair, zauloc_off, top, bottom) = inputs

    # ---- 1.3 / 2: density factors, lookup at ptm1, latent heats ----
    zqrho = 1.3 / rho
    zqrho_sqrt = jnp.sqrt(zqrho)
    zpapm1_inv = 1.0 / p
    ua_m1, dua_m1 = _ua(tm1)
    uaw_m1, duaw_m1 = _uaw(tm1)
    zrc = 1.0 / pcair
    zlvdcp = c.alhc * zrc
    zlsdcp = c.alhs * zrc
    zlfdcp = zlsdcp - zlvdcp
    loc.update(ua_m1=ua_m1, dua_m1=dua_m1, uaw_m1=uaw_m1, duaw_m1=duaw_m1)
    zcons2 = 1.0 / (dt * c.grav)
    zmass = zcons2 * dp          # zcons2*zdp: kg/m^2/s per kg/kg
    zdpg = dp / c.grav

    # ---- 3.1 Melting of the incoming snow and of cloud ice (jk > 1) ----
    # Both at the step-start temperature ptm1 (F:432-439).
    not_top = ~top
    ztdif = jnp.maximum(0.0, tm1 - c.tmelt)
    zcons = zcons2 * (dp / zlfdcp)
    zsnmlt = jnp.where(not_top, jnp.minimum(_ZXSEC * zsfl, zcons * ztdif), 0.0)
    zrfl = zrfl + zsnmlt
    zsfl = zsfl - zsnmlt
    zsmlt = zsnmlt / zmass
    # All provisional cloud ice melts where ptm1 > tmelt (F:438-439). The
    # switch carries a surrogate derivative in temperature.
    melt_all = temperature_switch(tm1 - c.tmelt, phase_width, inclusive=False)
    zimlt = jnp.where(not_top, melt_all * jnp.maximum(0.0, xip), 0.0)

    # ---- 3.2 / 3.3 Sublimation of snow, evaporation of rain ----
    # Only below a precipitating level (nclcpre, F:442), on the incoming
    # fluxes after melting, at the step-start (ptm1, pqm1).
    has_pre = zclcpre > 0.0
    # 1/zclcpre reaches the level only through products with the incoming
    # rain or snow flux (3.2, 3.3 and the contents of section 7), so its value
    # is discarded where neither is positive. The division is guarded by that
    # condition too: a tiny positive zclcpre (a weighted mean of 7.3 above can
    # underflow to ~1e-25) would otherwise overflow the reverse-mode rule of
    # the division to 0·inf in float32 where the value is not used.
    inv_used = has_pre & ((zrfl > 0.0) | (zsfl > 0.0))
    zclcpre_inv = jnp.where(inv_used, 1.0 / jnp.where(inv_used, zclcpre, 1.0), 0.0)

    # 3.2 Lin et al. (1983), over ice saturation from the mixed table (F:451-506).
    zesi = jnp.minimum(ua_m1 * zpapm1_inv, 0.5)
    zqsi = zesi / (1.0 - c.vtmpc1 * zesi)
    zsusati = jnp.minimum(qm1 / zqsi - 1.0, 0.0)
    zb1 = zlsdcp ** 2 / (2.43e-2 * c.rv * (tm1 ** 2))
    zb2 = 1.0 / (rho * zqsi * 0.211e-4)
    zcoeff = 3.0e6 * 2.0 * jnp.pi * (zsusati / (rho * (zb1 + zb2)))
    snow_on = has_pre & (zsfl > config.cqtmin)
    s_tmp1 = jnp.sqrt(zqrho_sqrt)
    s_tmp2 = _safe_pow(zsfl * zclcpre_inv / config.cvtfall, 1.0 / 1.16, snow_on)
    s_tmp2 = _safe_pow(s_tmp2 / (jnp.pi * config.crhosno * config.cn0s), 0.5,
                       snow_on)
    s_tmp3 = _safe_pow(s_tmp2, 1.3125, snow_on)
    zcfac4c = 0.78 * s_tmp2 + 232.19 * s_tmp1 * s_tmp3
    zzeps = jnp.maximum(-_ZXSEC * zsfl * zclcpre_inv, zcoeff * zcfac4c * zdpg)
    zsub = -(zzeps / zdpg) * dt * zclcpre
    zsub = jnp.minimum(zsub, jnp.maximum(_ZXSEC * (zqsi - qm1), 0.0))
    zsub = jnp.maximum(zsub, 0.0)
    zsub = jnp.minimum(zsub, zsfl / zmass)
    zsub = jnp.where(snow_on, zsub, 0.0)

    # 3.3 Rotstayn (1997), over water saturation (F:520-548).
    rain_on = has_pre & (zrfl > config.cqtmin)
    zesw = uaw_m1 * zpapm1_inv
    zesat = uaw_m1 / c.rd
    zesw = jnp.minimum(zesw, 0.5)
    zqsw = zesw / (1.0 - c.vtmpc1 * zesw)
    zsusatw = jnp.minimum(qm1 / zqsw - 1.0, 0.0)
    zdv = 2.21 * zpapm1_inv
    zptm1_inv = 1.0 / tm1
    zast = c.alhc * (c.alhc * zptm1_inv / c.rv - 1.0) * zptm1_inv / 0.024
    zbst = tm1 / (zdv * zesat)
    r_tmp2 = _safe_pow(zrfl * zclcpre_inv, 0.61, rain_on)
    zzepr = 870.0 * zsusatw * r_tmp2 * zqrho_sqrt / jnp.sqrt(1.3)
    zzepr = zzepr / (zast + zbst)
    zzepr = jnp.maximum(-_ZXSEC * zrfl * zclcpre_inv, zzepr * zdpg)
    zevp = -(zzepr / zdpg) * dt * zclcpre
    zevp = jnp.minimum(zevp, jnp.maximum(_ZXSEC * (zqsw - qm1), 0.0))
    zevp = jnp.maximum(zevp, 0.0)
    zevp = jnp.minimum(zevp, zrfl / zmass)
    zevp = jnp.where(rain_on, zevp, 0.0)
    loc.update(zsmlt=zsmlt, zimlt=zimlt, zsub=zsub, zevp=zevp,
               zrfl_melt=zrfl, zsfl_melt=zsfl)

    # ---- 4. Sedimentation of cloud ice (F:580-611) ----
    # The ice relaxes towards the influx-fed equilibrium
    # zal2 = zxitop/(rho·v): zxised = zxip1·zal1 + zal2·(1 − zal1) with
    # zal1 = exp(−x), x = v·g·rho·dt/dp. The influx term is evaluated as
    # zxitop·g·dt/dp·phi(x), phi(x) = (1 − e^−x)/x, the same number, so
    # its derivative stays bounded where v is small.
    zxip1_raw = xip - zimlt
    zxip1 = jnp.maximum(zxip1_raw, _ECHAM_EPSILON)
    zxifall = ice_fall_speed(rho, zxip1_raw, config.cvtfall,
                             config.ice_fall_speed_gradient_cutoff)
    sed_x = zxifall * c.grav * rho * (dt / dp)
    zal1 = jnp.exp(-sed_x)
    sed_x_safe = jnp.maximum(sed_x, 1.0e-8)
    sed_phi = jnp.where(sed_x > 1.0e-8, -jnp.expm1(-sed_x_safe) / sed_x_safe,
                        1.0 - 0.5 * sed_x)
    zxitop = zxiflux
    zxised = jnp.maximum(0.0, zxip1 * zal1 + zxitop * (c.grav * dt / dp) * sed_phi)
    zqsed = zxised - zxip1
    zxibot = jnp.maximum(0.0, zxitop - zqsed * zmass)
    zqsed = (zxitop - zxibot) / zmass
    zxised = zxip1 + zqsed
    zxiflux = zxibot
    loc.update(zqsed=zqsed, zxised=zxised, zxiflux_sed=zxiflux)

    # ---- 4. lo2 on the provisional temperature (F:647-650) ----
    zlo2 = ice_phase_weight(tm1 + dtemp, zxised, config.csecfrl,
                            config.cthomi, phase_width, ice_width)

    # ---- 4. In-cloud condensate; clear cells evaporate all (F:661-685) ----
    locc = paclc > 0.0
    cf_safe = jnp.where(locc, paclc, 1.0)
    zclcauxi = jnp.where(locc, 1.0 / cf_safe, 0.0)
    zxip1c = xip + zqsed - zimlt
    zxlp1c = xlp + zimlt
    zxievap = jnp.where(locc, 0.0, jnp.maximum(0.0, zxip1c))
    zxlevap = jnp.where(locc, 0.0, jnp.maximum(0.0, zxlp1c))
    zxib = jnp.where(locc, zxip1c * zclcauxi, 0.0)
    zxlb = jnp.where(locc, zxlp1c * zclcauxi, 0.0)
    loc.update(lo2=zlo2, zclcaux_in=paclc, zxlevap=zxlevap, zxievap=zxievap,
               zxlb_in=zxlb, zxib_in=zxib)

    # ---- 5. Condensation in the cloudy part (F:696-750) ----
    zlc = _blend(zlo2, zlsdcp, zlvdcp)
    zua = _blend(zlo2, ua_m1, uaw_m1)
    zdua = _blend(zlo2, dua_m1, duaw_m1)
    zqsm1 = jnp.minimum(zua * zpapm1_inv, 0.5)
    zcor = 1.0 / (1.0 - c.vtmpc1 * zqsm1)
    zqsm1 = zqsm1 * zcor
    zdqsdt = zpapm1_inv * zcor ** 2 * zdua
    zlcdqsdt = zlc * zdqsdt
    zdtdt = (dtemp - zlvdcp * (zevp + zxlevap) - zlsdcp * (zsub + zxievap)
             - zlfdcp * (zsmlt + zimlt))
    zstar1 = paclc * zlc * dq
    zdqsat1 = zdqsdt / (1.0 + paclc * zlcdqsdt)
    zqvdt = dq + zevp + zsub + zxievap + zxlevap
    zqp1 = jnp.maximum(qm1 + zqvdt, 0.0)
    ztp1 = tm1 + zdtdt
    zxib = jnp.maximum(zxib, 0.0)
    zxlb = jnp.maximum(zxlb, 0.0)
    prelhum = jnp.maximum(jnp.minimum(qm1 / zqsm1, 1.0), 0.0)
    zdqsat = (zdtdt + zstar1) * zdqsat1
    zqcdif, zcnd, zdep = sundqvist_condensation(
        dq, zdqsat, paclc, zxib, zxlb, zqp1, zlo2, qsec, _ZEPSEC)
    loc.update(zqsm1=zqsm1, zdtdt=zdtdt, zqp1=zqp1, ztp1=ztp1, zdqsat1=zdqsat1,
               zqcdif=zqcdif, zcnd_pre54=zcnd, zdep_pre54=zdep)

    # ---- 5.4 Supersaturation of the whole box (F:754-784) ----
    # Saturation and the phase switch are re-evaluated at the temperature
    # after the condensation of section 5, with the ice after deposition.
    ztp1tmp = ztp1 + zlvdcp * zcnd + zlsdcp * zdep
    zqp1tmp = zqp1 - zcnd - zdep
    zxip1_54 = jnp.maximum(xip + zqsed - zimlt - zxievap + zdep, 0.0)
    ub = _ub(ztp1tmp)
    zlo2_54 = ice_phase_weight(ztp1tmp, zxip1_54, config.csecfrl,
                               config.cthomi, phase_width, ice_width)
    ua_p1, dua_p1 = _ua(ztp1tmp)
    uaw_p1, duaw_p1 = _uaw(ztp1tmp)
    zua_54 = _blend(zlo2_54, ua_p1, uaw_p1)
    zdua_54 = _blend(zlo2_54, dua_p1, duaw_p1)
    loc.update(ztp1tmp_pre54=ztp1tmp, zqp1tmp_pre54=zqp1tmp, ua_54=zua_54,
               dua_54=zdua_54, ub_54=ub, lo2_54=zlo2_54)
    zes = jnp.minimum(zua_54 * zpapm1_inv, 0.5)
    zcor = 1.0 / (1.0 - c.vtmpc1 * zes)
    zqsp1tmp = zes * zcor
    zoversat = zqsp1tmp * 0.01
    zdqsdt = zpapm1_inv * zcor ** 2 * zdua_54
    zlc = _blend(zlo2_54, zlsdcp, zlvdcp)
    zlcdqsdt = jnp.where(zes - 0.4 >= 0.0, zqsp1tmp * zcor * ub, zlc * zdqsdt)
    zqcon = 1.0 / (1.0 + zlcdqsdt)
    zcor = jnp.maximum((zqp1tmp - zqsp1tmp - zoversat) * zqcon, 0.0)
    zupdate = _blend(zlo2_54, zdep, zcnd) + zcor
    zdep = _blend(zlo2_54, zupdate, zdep)
    zcnd = _blend(zlo2_54, zcnd, zupdate)
    loc.update(zqsp1tmp=zqsp1tmp, zcnd=zcnd, zdep=zdep)

    # ---- 5.5 In-cloud update; a clear cell with new condensate (F:793-813) ----
    # A clear cell that gained condensate in 5.4 is treated as fully
    # cloudy by the microphysics of this step (zclcaux = 1); the cover
    # itself is not changed.
    zxib = jnp.where(locc, jnp.maximum(zxib + zdep * zclcauxi, 0.0), zxib)
    zxlb = jnp.where(locc, jnp.maximum(zxlb + zcnd * zclcauxi, 0.0), zxlb)
    zdepos = jnp.maximum(zdep, 0.0)
    zcond = jnp.maximum(zcnd, 0.0)
    promote = (~locc) & ((zdepos > 0.0) | (zcond > 0.0))
    zclcaux = jnp.where(promote, 1.0, paclc)
    zxib = jnp.where(promote, zdepos, zxib)
    zxlb = jnp.where(promote, zcond, zxlb)
    ztp1tmp = ztp1 + zlvdcp * zcnd + zlsdcp * zdep
    loc.update(zclcaux=zclcaux, zxlb_55=zxlb, zxib_55=zxib, ztp1tmp=ztp1tmp)

    # ---- 6.1 Freezing of all cloud water at or below cthomi (F:821-828) ----
    freeze_all = temperature_switch(config.cthomi - ztp1tmp, phase_width,
                                    inclusive=True)
    zfrl = freeze_all * zxlb * zclcaux
    zxib = zxib + freeze_all * zxlb
    zxlb = (1.0 - freeze_all) * zxlb
    loc.update(zfrl_hom=zfrl)

    # ---- 6.2 Bigg and contact freezing between cthomi and tmelt (F:832-885) ----
    mixed = (ztp1tmp > config.cthomi) & (ztp1tmp < c.tmelt) & (zxlb > 0.0)
    t_frz = jnp.where(mixed, ztp1tmp, c.tmelt)
    zxlb_frz = jnp.where(mixed, zxlb, 0.0)
    zfrho = rho / (c.rhow * cdnc)
    zfrl_b = 100.0 * (jnp.exp(0.66 * (c.tmelt - t_frz)) - 1.0) * zfrho
    zfrl_b = zxlb_frz * (1.0 - 1.0 / (1.0 + zfrl_b * dt * zxlb_frz))
    zradl = contact_freezing_radius(zxlb_frz, zfrho, config.contact_freezing_liquid_cutoff)
    zval = 4.0 * jnp.pi * zradl * cdnc * 2.0e5 * (c.tmelt - 3.0 - t_frz)
    zf1 = jnp.maximum(0.0, zval / rho)
    zfrl_62 = zfrl_b + dt * 1.4e-20 * zf1
    zfrl_62 = jnp.maximum(0.0, jnp.minimum(zfrl_62, zxlb_frz))
    zxlb = jnp.where(mixed, zxlb - zfrl_62, zxlb)
    zxib = jnp.where(mixed, zxib + zfrl_62, zxib)
    zfrl = jnp.where(mixed, zfrl_62 * zclcaux, zfrl)
    loc.update(zfrl=zfrl, zxlb_6=zxlb, zxib_6=zxib)

    # ---- 7. Precipitation formation (F:911-948) ----
    zxlb = jnp.maximum(zxlb, 1.0e-20)
    zxib = jnp.maximum(zxib, 1.0e-20)
    zauloc = jnp.maximum(jnp.minimum(config.cauloc * dz / 5000.0, config.clmax),
                         config.clmin)
    zauloc = jnp.where(zauloc_off, 0.0, zauloc)
    # Rain and snow water contents of the incoming fluxes (Marshall-Palmer),
    # after melting and before evaporation (F:934-948).
    zxrp1 = _safe_pow(zrfl * zclcpre_inv / (12.45 * zqrho_sqrt), 8.0 / 9.0, has_pre)
    zxsp1 = _safe_pow(zsfl * zclcpre_inv / config.cvtfall, 1.0 / 1.16, has_pre)
    loc.update(zauloc=zauloc, zxrp1=zxrp1, zxsp1=zxsp1)

    active = (zclcaux > 0.0) & ((zxlb > config.cqtmin) | (zxib > config.cqtmin))
    zclcstar = jnp.minimum(zclcaux, zclcpre)

    # 7.1 Warm phase: autoconversion, accretion by rain (F:968-1017).
    if use_kk2000:
        rate = autoconversion_kk2000(zxlb, jnp.ones_like(zxlb), rho,
                                     cdnc_aut / rho, dt, config)
        zraut = jnp.minimum(rate * dt, zxlb)
    else:
        zraut = _beheng_depletion(zxlb, rho, cdnc_aut, dt, config.ccraut)
    zraut = jnp.where(active, zraut, 0.0)
    zxlb_w = zxlb - zraut
    zrac1 = jnp.where(active, zxlb_w * (1.0 - jnp.exp(-config.ccracl * zxrp1 * dt)), 0.0)
    zxlb_w = zxlb_w - zrac1
    zrac2 = jnp.where(
        active,
        zxlb_w * (1.0 - jnp.exp(-config.ccracl * zauloc * rho * zraut * dt)),
        0.0)
    zxlb_w = zxlb_w - zrac2
    zrpr = zclcaux * (zraut + zrac2) + zclcstar * zrac1

    # 7.2 Cold phase: aggregation, accretion of ice and riming by snow
    # (F:1026-1100). The effective radius uses the ice before zsaut.
    zsaut, zrieff = _levkov_depletion(zxib, rho, dt, config, return_radius=True)
    zsaut = jnp.where(active, zsaut, 0.0)
    zxib_w = zxib - zsaut
    zxsp2 = zauloc * rho * zsaut
    zcolleffi = jnp.exp(0.025 * (ztp1tmp - c.tmelt))

    def _sweep_out(content, present):
        zlamsm = _safe_pow(content / (jnp.pi * config.crhosno * config.cn0s),
                           0.8125, present)
        return jnp.pi * config.cn0s * 3.078 * zlamsm * zqrho_sqrt

    pass1 = active & (zxsp1 > config.cqtmin)
    k1 = _sweep_out(zxsp1, pass1)
    zsacl1 = jnp.where(pass1, zxlb_w * (1.0 - jnp.exp(-k1 * config.ccsacl * dt)), 0.0)
    zxlb_w = zxlb_w - zsacl1
    zsacl1 = zclcstar * zsacl1
    zsaci1 = jnp.where(pass1, zxib_w * (1.0 - jnp.exp(-(k1 * zcolleffi * dt))), 0.0)
    zxib_w = zxib_w - zsaci1

    pass2 = active & (zxsp2 > config.cqtmin)
    k2 = _sweep_out(zxsp2, pass2)
    zsacl2 = jnp.where(pass2, zxlb_w * (1.0 - jnp.exp(-k2 * config.ccsacl * dt)), 0.0)
    zxlb_w = zxlb_w - zsacl2
    zsacl2 = zclcaux * zsacl2
    zsaci2 = jnp.where(pass2, zxib_w * (1.0 - jnp.exp(-(k2 * zcolleffi * dt))), 0.0)
    zxib_w = zxib_w - zsaci2
    zsacl = zsacl1 + zsacl2
    zspr = jnp.where(active, zclcaux * (zsaut + zsaci2) + zclcstar * zsaci1, 0.0)
    zrpr = jnp.where(active, zrpr, 0.0)
    loc.update(zraut=zraut, zrac1=zrac1, zrac2=zrac2,
               zrieff=jnp.where(active, zrieff, 0.0), zsaut=zsaut,
               zsaci1=zsaci1, zsaci2=zsaci2, zsacl1=zsacl1, zsacl2=zsacl2,
               zcolleffi=jnp.where(active, zcolleffi, 0.0),
               zrpr=zrpr, zspr=zspr, zsacl=zsacl, zxlb_7=zxlb_w, zxib_7=zxib_w)

    # ---- 7.3 Flux and precipitating-fraction update (F:1112-1224) ----
    zzdrr = zmass * zrpr
    zzdrs = zmass * (zspr + zsacl)
    rain_source = zzdrr
    snow_source = zzdrs
    # Lowest level: the remaining sedimenting ice joins the snow, and the
    # snow produced in the level melts at ztp1tmp (F:1119-1126).
    zzdrs = jnp.where(bottom, zzdrs + zxiflux, zzdrs)
    zsnmlt_b = jnp.minimum(_ZXSEC * zzdrs,
                           (zmass / zlfdcp) * jnp.maximum(0.0, ztp1tmp - c.tmelt))
    zsnmlt_b = jnp.where(bottom, zsnmlt_b, 0.0)
    zzdrr = zzdrr + zsnmlt_b
    zzdrs = zzdrs - zsnmlt_b
    zsmlt = zsmlt + zsnmlt_b / zmass
    # ECHAM leaves zxiflux set at the lowest level (F:1121); the carry and
    # the frozen-flux profile drop it there, where it has joined the snow.
    zxiflux_echam = zxiflux
    zxiflux = jnp.where(bottom, 0.0, zxiflux)

    zpretot = zrfl + zsfl
    zpredel = zzdrr + zzdrs
    # Where the level's own production is at least the incoming flux the
    # precipitation is taken to come from this level's cloud (F:1129, 1177).
    zclcpre = jnp.where(zpredel - zpretot >= 0.0, zclcaux, zclcpre)
    zpresum = zpretot + zpredel
    # ECHAM sets zclcpre to 0 where zpresum <= cqtmin (F:1148, 1196). The
    # division is guarded by that same condition, not only by the dtype's
    # tiny: between the two the quotient is discarded, but its reverse-mode
    # rule -(0·x)·zpresum**-2 overflows to 0·inf in float32.
    discard = config.cqtmin - zpresum >= 0.0
    no_division = discard | (zpresum < tiny)
    zclcpre1 = jnp.where(
        no_division, 0.0,
        (zclcaux * zpredel + zclcpre * zpretot)
        / jnp.where(no_division, 1.0, zpresum))
    zclcpre1 = jnp.maximum(zclcpre, zclcpre1)
    zclcpre1 = jnp.minimum(1.0, jnp.maximum(0.0, zclcpre1))
    zclcpre = jnp.where(discard, 0.0, zclcpre1)

    rain_evap_flux = zmass * zevp
    snow_sub_flux = zmass * zsub
    zrfl = zrfl + zzdrr - rain_evap_flux
    zsfl = zsfl + zzdrs - snow_sub_flux
    loc.update(zclcpre_out=zclcpre, zrfl_out=zrfl, zsfl_out=zsfl,
               zsmlt_final=zsmlt, zxiflux_final=zxiflux_echam)

    # ---- 8.3 Tendencies (F:1242-1251) ----
    zqvte = (-zcnd + zevp + zxlevap - zdep + zsub + zxievap) / dt
    zxlte = (zimlt - zfrl - zrpr - zsacl + zcnd - zxlevap) / dt
    zxite = (zfrl - zspr + zdep - zxievap - zimlt + zqsed) / dt
    ztte = (zlvdcp * (zcnd - zevp - zxlevap)
            + zlsdcp * (zdep - zsub - zxievap)
            + zlfdcp * (-zsmlt - zimlt + zfrl + zsacl)) / dt

    # ---- 8.4 Condensate below ccwmin returns to vapour (F:1264-1288) ----
    zxlp1 = xlp + zxlte * dt
    zxip1_end = xip + zxite * dt
    zxlp1_d = config.ccwmin - zxlp1
    zxip1_d = config.ccwmin - zxip1_end
    zxlp1_new = jnp.where(-zxlp1_d >= 0.0, zxlp1, 0.0)
    zxip1_new = jnp.where(-zxip1_d >= 0.0, zxip1_end, 0.0)
    zdxlcor = (zxlp1_new - zxlp1) / dt
    zdxicor = (zxip1_new - zxip1_end) / dt
    zxlp1_d = jnp.maximum(zxlp1_d, 0.0)
    paclc_out = jnp.where(-(zxlp1_d * zxip1_d) >= 0.0, paclc, 0.0)
    zxlte = zxlte + zdxlcor
    zxite = zxite + zdxicor
    zqvte = zqvte - zdxlcor - zdxicor
    ztte = ztte + zlvdcp * zdxlcor + zlsdcp * zdxicor
    loc.update(zdxlcor=zdxlcor, zdxicor=zdxicor, prelhum=prelhum)
    echam_locals = {k: jnp.broadcast_to(v, jnp.shape(tm1)).astype(tm1.dtype)
                    for k, v in loc.items()}

    inter = SweepIntermediates(
        zevp=zevp, zsub=zsub, zsmlt=zsmlt, zimlt=zimlt, zqsed=zqsed,
        zlo2=zlo2, zxlevap=zxlevap, zxievap=zxievap, zcnd=zcnd, zdep=zdep,
        zclcaux=zclcaux, zfrl=zfrl, zrpr=zrpr, zspr=zspr, zsacl=zsacl,
        zclcpre=zclcpre, zdxlcor=zdxlcor, zdxicor=zdxicor, prelhum=prelhum,
    )
    out = LevelOutputs(
        ztte, zqvte, zxlte, zxite,
        rain_source, snow_source, rain_evap_flux, snow_sub_flux,
        zclcaux * zraut / dt, (zclcaux * zrac2 + zclcstar * zrac1) / dt,
        zrfl, zsfl + zxiflux, paclc_out, inter, echam_locals,
    )
    return (zrfl, zsfl, zclcpre, zxiflux), out


def cloud_microphysics_column_sweep(
    temperature_m1: jnp.ndarray,
    specific_humidity_m1: jnp.ndarray,
    cloud_water_m1: jnp.ndarray,
    cloud_ice_m1: jnp.ndarray,
    temperature_increment: jnp.ndarray,
    humidity_increment: jnp.ndarray,
    cloud_water_increment: jnp.ndarray,
    cloud_ice_increment: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    pressure: jnp.ndarray,
    pressure_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt,
    config: Optional[MicrophysicsParameters] = None,
    *,
    detrained_liquid: Optional[jnp.ndarray] = None,
    detrained_ice: Optional[jnp.ndarray] = None,
    autoconversion_droplet_number: Optional[jnp.ndarray] = None,
    heat_capacity: Optional[jnp.ndarray] = None,
    lonacc_mask: Optional[jnp.ndarray] = None,
) -> tuple[MicrophysicsTendencies, MicrophysicsState]:
    """ECHAM6.3 ``mo_cloud.f90::cloud`` on a column or a block of columns.

    Every array is ``(nlev, *horiz)`` with level 0 at the model top, and the
    function is broadcasting-native: one column ``(nlev,)`` and a block
    ``(nlev, ncols)`` give the same per-column result.

    What the sweep receives, in ECHAM's terms (``ztmst = dt``). The anchor
    and the increments are separate arguments, as in ECHAM's argument list:

    - ``temperature_m1``, ``specific_humidity_m1``, ``cloud_water_m1``,
      ``cloud_ice_m1``: the anchor ``ptm1``, ``pqm1``, ``pxlm1``, ``pxim1``.
      Sections 3.1-3.3 (melt, snow sublimation, rain evaporation) and the
      saturation humidity of section 5 are evaluated at the anchor, and
      section 5 assumes the cloudy part saturated there.
    - ``temperature_increment``, ``humidity_increment``,
      ``cloud_water_increment``, ``cloud_ice_increment``: ``ztmst·ptte``,
      ``ztmst·pqte``, ``ztmst·pxlte``, ``ztmst·pxite``, the change made in
      this step, before the cloud scheme, by every process that ran since the
      anchor. Section 5 condenses the part of the humidity increment that the
      saturation humidity, moved by the temperature increment, does not
      absorb (F:706-734), and ``lo2`` reads ``ptm1 + ztmst·ptte``
      (F:647-650).
    - ``detrained_liquid``, ``detrained_ice``: ``ztmst·pxtecl``,
      ``ztmst·pxteci``, the convective detrainment (zero if omitted), clipped
      at zero as ECHAM clips it (F:384-385). ECHAM uses the condensate only as
      ``pxlm1 + ztmst·(pxlte + pxtecl)``, so a detrainment already inside the
      condensate increments gives the same result.
    - ``cloud_fraction``: ``paclc`` from the cover scheme. A cell is clear if
      and only if ``paclc`` is not positive (F:621).
    - ``droplet_number``: ``pacdnc`` [1/m^3], the droplet number of Bigg and
      contact freezing (section 6.2) and, unless
      ``autoconversion_droplet_number`` is given, of the Beheng
      autoconversion (section 7.1).
    - ``heat_capacity``: ``pcair`` [J/kg/K], the moist heat capacity of the
      latent-heat factors ``zlvdcp = alv/pcair``, ``zlsdcp = als/pcair``;
      defaults to ``cpd + (cpv − cpd)·max(pqm1, 0)`` (ECHAM physc.f90).
    - ``pressure``, ``pressure_thickness``, ``air_density``,
      ``layer_thickness``: ``papm1``, ``paphm1(k+1) − paphm1(k)``,
      ``papm1/(rd·ptvm1)`` and the layer depth ``zdz`` [m] (read only by
      ``zauloc``).
    - ``lonacc_mask``: the levels where ``zauloc`` is zeroed
      (:func:`lonacc_levels`); none if omitted. Inert while ``cauloc = 0``.

    What the composable term passes as the increments is stated in
    :class:`Echam1MMicrophysics`.

    The returned tendencies are the cloud scheme's own (ECHAM's ``zqvte``,
    ``zxlte``, ``zxite``, ``ztte`` plus the section 8.4 corrections). Adding
    ``dt`` times them to the provisional state gives the state after the
    cloud scheme.

    Returns:
        ``(MicrophysicsTendencies, MicrophysicsState)``.

    """
    if config is None:
        config = MicrophysicsParameters.default()

    dtype = jnp.result_type(temperature_m1)
    # The scheme runs in the precision of the state: float parameter leaves
    # (float64 under ``jax_enable_x64``) are cast to it, differentiably.
    config = jax.tree.map(
        lambda leaf: leaf.astype(dtype)
        if jnp.issubdtype(jnp.result_type(leaf), jnp.floating) else leaf,
        config)
    (specific_humidity_m1, cloud_water_m1, cloud_ice_m1, temperature_increment,
     humidity_increment, cloud_water_increment, cloud_ice_increment,
     cloud_fraction, pressure, pressure_thickness, air_density, layer_thickness,
     droplet_number) = (
        jnp.asarray(a).astype(dtype) for a in (
            specific_humidity_m1, cloud_water_m1, cloud_ice_m1,
            temperature_increment, humidity_increment, cloud_water_increment,
            cloud_ice_increment, cloud_fraction, pressure, pressure_thickness,
            air_density, layer_thickness, droplet_number))
    dt = jnp.asarray(dt).astype(dtype)
    nlev = temperature_m1.shape[0]
    horiz = temperature_m1.shape[1:]
    if heat_capacity is None:
        heat_capacity = moist_isobaric_heat_capacity(specific_humidity_m1)
    if detrained_liquid is None:
        detrained_liquid = jnp.zeros_like(cloud_water_m1)
    if detrained_ice is None:
        detrained_ice = jnp.zeros_like(cloud_ice_m1)
    # ECHAM clips the detrainment at zero on entry (F:384-385) and reads the
    # condensate only as pxlm1 + ztmst*(pxlte + pxtecl) (F:438, 581, 666-680).
    detrained_liquid = jnp.maximum(jnp.asarray(detrained_liquid).astype(dtype), 0.0)
    detrained_ice = jnp.maximum(jnp.asarray(detrained_ice).astype(dtype), 0.0)
    cloud_water = cloud_water_m1 + (cloud_water_increment + detrained_liquid)
    cloud_ice = cloud_ice_m1 + (cloud_ice_increment + detrained_ice)
    if autoconversion_droplet_number is None:
        autoconversion_droplet_number = droplet_number
    autoconversion_droplet_number = jnp.asarray(
        autoconversion_droplet_number).astype(dtype)
    heat_capacity = jnp.asarray(heat_capacity).astype(dtype)
    if lonacc_mask is None:
        lonacc_mask = jnp.zeros(temperature_m1.shape, dtype=bool)

    level = jnp.arange(nlev).reshape((nlev,) + (1,) * len(horiz))
    is_top = jnp.broadcast_to(level == 0, temperature_m1.shape)
    is_bottom = jnp.broadcast_to(level == nlev - 1, temperature_m1.shape)

    zero = jnp.zeros(horiz, dtype=dtype)
    inputs = LevelInputs(
        temperature_m1, specific_humidity_m1,
        temperature_increment, humidity_increment,
        cloud_water, cloud_ice, cloud_fraction, pressure, pressure_thickness,
        air_density, layer_thickness,
        jnp.broadcast_to(droplet_number, temperature_m1.shape),
        jnp.broadcast_to(autoconversion_droplet_number, temperature_m1.shape),
        jnp.broadcast_to(heat_capacity, temperature_m1.shape),
        jnp.broadcast_to(lonacc_mask, temperature_m1.shape),
        is_top, is_bottom,
    )
    (zrfl_sfc, zsfl_sfc, _, _), per_level = jax.lax.scan(
        lambda carry, level: _sweep_level(carry, level, config, dt),
        (zero, zero, zero, zero), inputs)
    (dtedt, dqdt, dqcdt, dqidt, rain_source, snow_source, rain_evap_flux,
     snow_sub_flux, autoconv_rate, accretion_rate, rain_flux, snow_flux,
     cloud_fraction_out, inter, echam_locals) = per_level

    tendencies = MicrophysicsTendencies(
        dtedt=dtedt, dqdt=dqdt, dqcdt=dqcdt, dqidt=dqidt,
        dqrdt=jnp.zeros_like(dtedt), dqsdt=jnp.zeros_like(dtedt),
    )
    cloudy = cloud_fraction > 0.0
    cf_safe = jnp.where(cloudy, cloud_fraction, 1.0)
    state = MicrophysicsState(
        rain_flux=rain_flux, snow_flux=snow_flux,
        rain_source=rain_source, snow_source=snow_source,
        rain_evap_flux=rain_evap_flux, snow_sublimation_flux=snow_sub_flux,
        qc_in_cloud=jnp.where(cloudy, cloud_water / cf_safe, 0.0),
        qi_in_cloud=jnp.where(cloudy, cloud_ice / cf_safe, 0.0),
        autoconv_rate=autoconv_rate, accretion_rate=accretion_rate,
        melting_rate=(inter.zsmlt + inter.zimlt) / dt,
        freezing_rate=inter.zfrl / dt,
        precip_rain=zrfl_sfc, precip_snow=zsfl_sfc,
        cloud_fraction=cloud_fraction_out,
        intermediates=inter,
        echam_locals=echam_locals,
    )
    return tendencies, state
# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.clouds.cloud_data import CLOUD_OUTPUT_ATTRS  # noqa: E402
from jcm.physics.clouds.cloud_inputs import cloud_scheme_inputs  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


def shallow_liquid_convection_type(
    ktype: jnp.ndarray,
    cloud_top: jnp.ndarray,
    pressure_full: jnp.ndarray,
    cloud_water: jnp.ndarray,
    pressure_thickness: jnp.ndarray,
    clwprat,
) -> jnp.ndarray:
    """ECHAM's radiation convective type: ``ktype`` with shallow-liquid 4s.

    ``mo_cloud.f90`` (1M ``cloud``, lines 1439-1455): with ``W_above`` the
    step-start liquid water path of the levels above the convective cloud top
    and ``W_below`` that at and below it, a shallow column (``ktype = 2``)
    becomes ``ktype = 4`` where ``W_below > clwprat·W_above``.

    Broadcasting-native: level on axis 0, any trailing horizontal axes.
    "Above the top" is decided by pressure (``p < p(cloud_top)``), so the
    result does not depend on the level axis's orientation. ECHAM's ``pxlm1``
    is non-negative; jcm's advected ``qc`` can carry small negative ringing, so
    the path above the top is floored at zero and a column needs positive
    liquid at/below the top to re-type — identical to ECHAM for any
    non-negative ``qc`` (where ``0 > clwprat·0`` is already false), and it
    keeps a liquid-free column from reading as "shallow liquid".

    Args:
        ktype: convection type per column (*horiz), int.
        cloud_top: convective cloud-top level index per column (*horiz), on
            the same level axis as ``pressure_full``.
        pressure_full: full-level pressure [Pa] (nlev, *horiz).
        cloud_water: grid-mean cloud liquid [kg/kg] (nlev, *horiz).
        pressure_thickness: layer Δp [Pa] (nlev, *horiz).
        clwprat: ECHAM ``clwprat`` threshold ratio.

    Returns:
        ``ktype`` with qualifying shallow columns set to 4, same dtype.

    """
    ktype = jnp.asarray(ktype)
    top = jnp.clip(cloud_top, 0, pressure_full.shape[0] - 1).astype(jnp.int32)
    p_top = jnp.take_along_axis(pressure_full, top[jnp.newaxis], axis=0)
    liquid = cloud_water * pressure_thickness / c.grav
    lwp_above_raw = jnp.sum(
        jnp.where(pressure_full < p_top, liquid, 0.0), axis=0)
    lwp_below = jnp.sum(liquid, axis=0) - lwp_above_raw
    lwp_above = jnp.maximum(lwp_above_raw, 0.0)
    shallow_liquid = (
        (ktype == 2) & (lwp_below > 0.0) & (lwp_below > clwprat * lwp_above)
    )
    return jnp.where(shallow_liquid, jnp.asarray(4, ktype.dtype), ktype)


class Echam1MMicrophysics(PhysicsTerm):
    """ECHAM6.3's 1-moment cloud scheme as a composable PhysicsTerm.

    Runs :func:`cloud_microphysics_column_sweep` on the ECHAM cover
    ``clouds.cloud_fraction`` of
    :class:`~jcm.physics.clouds.sundqvist.SundqvistCloudFraction`. The
    droplet number is ECHAM's prescribed ``acdnc``
    (:func:`~jcm.physics.clouds.cloud_utils.prescribed_droplet_number`).

    Anchor and increments (:func:`~jcm.physics.clouds.cloud_inputs.cloud_scheme_inputs`).
    The anchor is the previous step's post-physics state, which the model
    carries (the term declares ``requires_post_physics_fields``), so the
    increments are the dynamics of the step plus ``dt`` times the running
    tendency of every physics term composed before this one: in the ECHAM
    stack radiation, vertical diffusion (with its condensate), the surface
    and convection. These are what ECHAM's ``ptte``/``pqte``/``pxlte``/
    ``pxite`` hold at ``cloud``. The convective detrainment is passed
    separately, as ECHAM's ``pxtecl``/``pxteci``. Where no carried state is
    valid (the first step, a host without a dynamical core) the anchor is the
    state the physics receives and the dynamics increment is zero.

    Reads ``pressure_full``, ``pressure_thickness``, ``air_density`` and
    ``layer_thickness`` from the moist-air diagnostics and the timestep from
    ``diagnostics["_dt_seconds"]``. Writes the cover after ECHAM's section 8.4
    write-back, the surface precipitation, the flux profiles, the process
    rates and ``droplet_number`` into ``"clouds"``. When a convection term has
    published ``"convection"`` upstream, it re-types that step's shallow
    columns whose liquid sits below the convective cloud top as ``ktype = 4``
    for the next step's radiation (:func:`shallow_liquid_convection_type`).
    ``"convection"`` is not in ``provides``: the term only amends it.
    """

    name: ClassVar[str] = "echam_1m_microphysics"
    category: ClassVar[str] = "clouds"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "air_density", "layer_thickness",
        "clouds", "aerosol",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "autoconv", "accretn", "wbf", "clouds",
    )
    # CF/units metadata for the ``clouds.*`` output fields (#740). Shared with
    # the cover term; this term fills the precip/process-rate fields.
    output_attrs: ClassVar[dict[str, dict[str, str]]] = CLOUD_OUTPUT_ATTRS
    # The fields of the previous step's post-physics state this term takes
    # as ECHAM's anchor when the host carries one (the dynamics of the step
    # then joins the increments).
    requires_post_physics_fields: ClassVar[tuple[str, ...]] = (
        "temperature", "specific_humidity", "qc", "qi")

    def __init__(self, params: MicrophysicsParameters | None = None, *,
                 params_are_defaults: bool = False):
        """Hold the scheme-native :class:`MicrophysicsParameters`.

        ``params_are_defaults`` marks parameters that are the resolution
        defaults of a grid rather than the caller's own choice (the factory
        and the Hydra runner set it), so a host can check them against the
        grid; parameters left ``None`` are defaults too.
        """
        self.params = nnx.Param(
            params or MicrophysicsParameters.default(),
        )
        self.params_are_defaults = params is None or params_are_defaults

    def cache_coords(self, coords) -> None:
        """Warn if default parameters were built for another grid.

        The parameters were fixed at construction and are not re-resolved;
        parameters the caller supplied are not checked.
        """
        if self.params_are_defaults:
            from jcm.physics.resolution_defaults import check_defaults_grid
            check_defaults_grid(type(self).__name__,
                                self.params.get_value().defaults_truncation, coords)

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """``qc`` / ``qi`` are read each step; declared so dynamics carries them."""
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
        )

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute the cloud-scheme tendencies and the precipitation diagnostics."""
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        layer_thickness = diagnostics["layer_thickness"]
        clouds = diagnostics["clouds"]
        # Every ECHAM stack has MoistAirColumnState's exact layer Δp; the
        # ρ·g·dz fallback only serves hand-built diagnostics without it.
        pressure_thickness = diagnostics.get(
            "pressure_thickness",
            diagnostics["air_density"] * layer_thickness * c.grav)

        # ECHAM's cloud routine receives an anchor state, the increments
        # accumulated since it and the convective detrainment
        # (``cloud_inputs.cloud_scheme_inputs``). Everything ECHAM evaluates
        # at the step start reads the anchor, including the air density
        # papm1/(rd*ptvm1) with ECHAM's virtual temperature
        # T*(1 + vtmpc1*q - (xl + xi)) (physc.f90:267, mo_cloud.f90:382).
        inputs = cloud_scheme_inputs(state, diagnostics, tracers=("qc", "qi"))
        anchor, increment = inputs.anchor, inputs.increment
        anchor_qc, anchor_qi = anchor.tracers["qc"], anchor.tracers["qi"]
        virtual_temperature = anchor.temperature * (
            1.0 + c.vtmpc1 * anchor.specific_humidity - (anchor_qc + anchor_qi))
        air_density = pressure_full / (c.rd * virtual_temperature)

        # Droplet number. ECHAM passes its prescribed ``acdnc`` (physc.f90
        # section 3.12) to the cloud routine as ``pacdnc``, which both the
        # freezing of section 6.2 and the autoconversion of section 7.1 read.
        # The MACv2-SP Twomey factor scales the radiation's droplet number
        # (``prescribed_droplet_number``, the call the radiation makes, which
        # is also what ``clouds.droplet_number`` publishes) and, with
        # ``autoconversion_twomey`` (the default), the autoconversion's too:
        # jcm's aerosol-cloud interaction acts on precipitation formation,
        # which ECHAM with simple plumes does not do (#932).
        acdnc = prescribed_droplet_number(pressure_full, terrain, forcing, 1.0)
        cdnc_radiation = prescribed_droplet_number(
            pressure_full, terrain, forcing, diagnostics["aerosol"].cdnc_factor)
        cdnc_autoconversion = (
            cdnc_radiation if params.autoconversion_twomey else acdnc)

        micro_tend, micro_state = cloud_microphysics_column_sweep(
            anchor.temperature, anchor.specific_humidity, anchor_qc, anchor_qi,
            increment.temperature, increment.specific_humidity,
            increment.tracers["qc"], increment.tracers["qi"],
            clouds.cloud_fraction,
            pressure_full, pressure_thickness, air_density, layer_thickness,
            acdnc, dt, params,
            detrained_liquid=inputs.detrained_qc, detrained_ice=inputs.detrained_qi,
            autoconversion_droplet_number=cdnc_autoconversion,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=micro_tend.dtedt,
            specific_humidity=micro_tend.dqdt,
            tracers={
                "qc": micro_tend.dqcdt,
                "qi": micro_tend.dqidt,
            },
        )

        # AeroCom process rates (jax-gcm#585: both schemes, zero where a
        # pathway is absent), mass-weighted to kg/m^2/s. WBF stays zero: the
        # 1M scheme has no explicit Wegener-Bergeron-Findeisen transfer, and
        # the key is published so the diagnostic set is scheme-independent.
        dm = pressure_thickness / c.grav
        autoconv_col = jnp.sum(micro_state.autoconv_rate * dm, axis=0)
        accretn_col = jnp.sum(micro_state.accretion_rate * dm, axis=0)
        diagnostics = {**diagnostics, "autoconv": autoconv_col,
                       "accretn": accretn_col,
                       "wbf": jnp.zeros_like(autoconv_col)}

        clouds = clouds.copy(
            # ECHAM's section 8.4 write-back: a cell whose end-of-step
            # condensate is below ``ccwmin`` in both phases has no cloud
            # (F:1280). The terms after the cloud scheme (COSP, AeroCom, the
            # JAM cloud terms) read this cover, and it is the saved
            # ``clouds.cloud_fraction``. Radiation does not: it runs before
            # the cloud scheme and reads the cover term's value, which the
            # next step recomputes, as ECHAM's cover recomputes aclc before
            # radiation (physc.f90:543, 566).
            cloud_fraction=micro_state.cloud_fraction,
            # The published fluxes are floored at zero. ECHAM's flux update
            # (F:1211-1212) leaves a round-off remainder, of either sign, where
            # evaporation or sublimation takes the whole incoming flux; the
            # sweep keeps it, and the consumers of these fields (the surface,
            # COSP) take precipitation as non-negative.
            precip_rain=jnp.maximum(micro_state.precip_rain, 0.0),
            precip_snow=jnp.maximum(micro_state.precip_snow, 0.0),
            # Flux profiles for the satellite simulators (COSP/CloudSat).
            rain_flux=jnp.maximum(micro_state.rain_flux, 0.0),
            snow_flux=jnp.maximum(micro_state.snow_flux, 0.0),
            # Process rates for JAM wet scavenging (#499), grid-mean kg/kg/s:
            # formation is the condensate-to-precipitation ledger, evaporation
            # the rain evaporation plus the snow sublimation.
            precip_formation_rate=(
                micro_state.rain_source + micro_state.snow_source) / dm,
            precip_evaporation_rate=(
                micro_state.rain_evap_flux + micro_state.snow_sublimation_flux) / dm,
            droplet_number=cdnc_radiation,
        )

        # Advance the running view so terms downstream (the satellite
        # simulators, the AeroCom diagnostics) describe the post-cloud
        # atmosphere. ``thermo_run`` is a diagnostic view, never the
        # prognostic state (see ``advance_thermo_run``).
        from jcm.physics.diagnostics.moist_air_state import (
            advance_thermo_run)
        diagnostics = advance_thermo_run(
            diagnostics, dt,
            d_temperature=tendency.temperature,
            d_specific_humidity=tendency.specific_humidity,
            d_qc=tendency.tracers.get("qc"), d_qi=tendency.tracers.get("qi"))

        # ECHAM section 10 (F:1439-1455): re-type a shallow convective column
        # (ktype 2) as 4 when its liquid water path at and below the
        # convective cloud top exceeds ``clwprat`` x the path above it. Nothing
        # in the cloud scheme uses it; it is stored for the next step's
        # radiation, which then uses the shallow liquid inhomogeneity
        # ``zinhoml2`` (``mo_cloud_optics.f90``). Only ECHAM's 1M ``cloud``
        # re-types; its 2M ``cloud_micro_interface`` does not. The liquid is
        # the anchor's ``pxlm1``, as in ECHAM.
        conv = diagnostics.get("convection")
        if conv is not None and hasattr(conv, "cloud_top"):
            diagnostics = {**diagnostics, "convection": conv.replace(
                ktype=shallow_liquid_convection_type(
                    conv.ktype, conv.cloud_top, pressure_full, anchor_qc,
                    pressure_thickness,
                    params.clwprat,
                ),
            )}

        return tendency, {**diagnostics, "clouds": clouds}
