"""ECHAM-HAM's aerosol inputs to mixed-phase freezing, for a JAM population (#953).

Port of ``mo_ham_freezing.f90`` (ECHAM6.3-HAM2.3 r7492): ``get_aerofreez_nc``
(lines 300-465) with its helpers ``aero_massvolratio`` (188-263) and
``aero_nc_surfw`` (278-289), and the mixed-phase part of ``ham_IN_setup``
(53-120). HAM describes the aerosol that can freeze cloud droplets by the
dust and black-carbon number in two populations:

* **Soluble classes, immersion freezing.** The droplets activated on a
  soluble class hold its dust and BC. HAM counts them by surface weighting,
  ``(volume ratio of the species in the class)^(2/3) × activated number of
  the class``, summed over the classes (``ndusol_strat``, ``nbcsol_strat``),
  and divides by the activated CDNC: ``fracdusol``, ``fracbcsol``.
* **Insoluble classes, Brownian contact freezing.** Bare particles of an
  insoluble class, counted as ``(mass ratio)^(2/3) × class number``
  (``nduinsolai``, ``nduinsolci``, ``nbcinsol``), over the number of all
  insoluble classes (``naerinsol``): ``fracduai``, ``fracduci``,
  ``fracbcinsol``, with the wet radii of the insoluble Aitken, accumulation
  and coarse classes.

Each fraction is ``MIN(n/(N + EPSILON), 1)``.

Which classes of a population play HAM's roles is declared by
:class:`HamFreezingClasses`. For MAM4 (:data:`MAM4_FREEZING_CLASSES`):

* the **accumulation** and **coarse** modes are the soluble classes. MAM4
  emits dust into them directly and treats every particle as internally
  mixed, so they carry the aged dust and BC; they correspond to HAM's
  soluble accumulation and coarse modes (``iaccs``, ``icoas``). MAM4's Aitken
  mode carries neither dust nor BC, so HAM's exclusion of the soluble Aitken
  BC (``DN #295``, line 427) changes nothing here.
* the **primary-carbon** mode is the fresh, hydrophobic, insoluble carbon,
  HAM's insoluble Aitken mode (``iaiti``, BC and OC); it is the one
  insoluble class, so it is also the whole of ``naerinsol``.
* MAM4 has **no insoluble dust** (HAM's ``iacci``, ``icoai``): all its dust is
  in the soluble modes. The dust contact fractions and radii are therefore
  zero, and because ECHAM disables black-carbon contact freezing
  (``zfrzcntbc = 0``, mo_cloud_micro_2m.f90 line 2784), contact freezing is
  identically zero with MAM4; the mixed-phase heterogeneous freezing is
  immersion freezing of dust- and BC-bearing droplets.

The class composition is the class's whole population, interstitial plus
cloud-borne: HAM has one phase per class, which jcm's explicit cloud-borne
store splits in two. The caller (:class:`~jcm.physics.aerosol.jam.ice_nucleation.ice_term.IceNucleation`)
floors the masses and numbers at zero, because transport can leave round-off
negatives, which have no composition meaning; HAM's tracers enter unfloored.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.clouds.lohmann_2m.types import HeterogeneousFreezingAerosol

# ``zeps = EPSILON(1.0_dp)`` of mo_ham_freezing.f90 (line 49): the empty-class
# threshold of the ratios and the guard of the fraction denominators.
ZEPS = 2.220446049250313e-16


@dataclasses.dataclass(frozen=True)
class HamFreezingClasses:
    """The population classes that play HAM's freezing roles.

    Attributes:
        soluble: classes whose activated droplets count for immersion
            freezing (HAM's soluble accumulation and coarse modes).
        insoluble_aitken: the class of HAM's insoluble Aitken mode (BC contact
            nuclei and ``prwetki``), or ``None``.
        insoluble_accumulation: the class of HAM's insoluble accumulation
            dust (``nduinsolai``, ``prwetai``), or ``None``.
        insoluble_coarse: the class of HAM's insoluble coarse dust
            (``nduinsolci``, ``prwetci``), or ``None``.

    """

    soluble: tuple[str, ...]
    insoluble_aitken: str | None = None
    insoluble_accumulation: str | None = None
    insoluble_coarse: str | None = None

    def validate(self, spec: ModalAerosolSpec) -> None:
        """Raise if a named class is missing or has the wrong solubility."""
        for short in self.soluble:
            if not spec.mode(short).soluble:
                raise ValueError(f"HAM soluble freezing class {short!r} is insoluble in the spec.")
        for short in (self.insoluble_aitken, self.insoluble_accumulation, self.insoluble_coarse):
            if short is not None and spec.mode(short).soluble:
                raise ValueError(f"HAM insoluble freezing class {short!r} is soluble in the spec.")


#: MAM4 -> HAM mapping (see the module docstring for the reasoning).
MAM4_FREEZING_CLASSES = HamFreezingClasses(
    soluble=("acc", "cor"), insoluble_aitken="pcm",
    insoluble_accumulation=None, insoluble_coarse=None,
)


def _ratio_pow(spec, short, species, masses, volume):
    """``aero_massvolratio`` then ``aero_nc_surfw``'s ``ratio**(2/3)``.

    ``volume`` weights each mass by ``1000/density`` (HAM's ``zdens_rcp``,
    lines 225-231), so the empty-class threshold ``zdenom > zeps`` sees the
    same number as HAM's. Zero where the class is empty or the ratio is
    below ``zeps`` (lines 248-257); the power runs on a safe base there.
    """
    mode = spec.mode(short)
    if species not in mode.species:
        return None

    def w(sp):
        return 1000.0 / spec.species_props(sp).density if volume else 1.0

    num = w(species) * masses[(species, short)]
    den = sum(w(sp) * masses[(sp, short)] for sp in mode.species)
    ok_den = den > ZEPS
    ratio = jnp.where(ok_den, num / jnp.where(ok_den, den, 1.0), 0.0)
    ok = ratio >= ZEPS
    return jnp.where(ok, jnp.where(ok, ratio, 1.0) ** (2.0 / 3.0), 0.0)


def _fraction(n, total):
    """``MIN(n/(total + zeps), 1)`` with a derivative that stays finite.

    Where the quotient is clipped at 1 its derivative is discarded, so the
    quotient runs on a benign denominator there (a clipped ``n/zeps`` would
    otherwise send ``n/zeps**2`` through reverse mode).
    """
    den = total + ZEPS
    below = n < den
    return jnp.where(below, n / jnp.where(below, den, 1.0), 1.0)


def ham_freezing_aerosol(
    spec: ModalAerosolSpec,
    classes: HamFreezingClasses,
    masses: dict[tuple[str, str], jnp.ndarray],
    number: dict[str, jnp.ndarray],
    activated_number: dict[str, jnp.ndarray],
    wet_radius: dict[str, jnp.ndarray],
    air_density: jnp.ndarray,
    activated_cdnc: jnp.ndarray,
) -> HeterogeneousFreezingAerosol:
    """HAM's mixed-phase freezing inputs for one population.

    Args:
        spec: the population.
        classes: which classes play HAM's roles.
        masses: ``(species, class short) -> mass mixing ratio [kg/kg]`` for
            every species of every class named in ``classes`` (the class's
            whole population; non-negative).
        number: ``class short -> number [1/kg]`` for every insoluble class of
            the spec (HAM's ``pxtm1`` number tracers).
        activated_number: ``class short -> activated droplets [1/m3]`` for
            the soluble classes (HAM's ``nact_strat``).
        wet_radius: ``class short -> wet radius [m]`` for the insoluble
            classes named in ``classes`` (HAM's ``rwet``).
        air_density: [kg/m3].
        activated_cdnc: the activated CDNC [1/m3] (HAM's ``pcdncact``).

    Returns:
        :class:`HeterogeneousFreezingAerosol` shaped like ``air_density``.

    """
    zeros = jnp.zeros_like(air_density)

    # Soluble classes: immersion (lines 399-404, 427-431).
    n_du_sol, n_bc_sol = zeros, zeros
    for short in classes.soluble:
        for sp, acc in (("du", "du"), ("bc", "bc")):
            rp = _ratio_pow(spec, short, sp, masses, volume=True)
            if rp is None:
                continue
            term = rp * activated_number[short]
            if acc == "du":
                n_du_sol = n_du_sol + term
            else:
                n_bc_sol = n_bc_sol + term

    # Insoluble classes: contact (lines 406-411, 423-425), over the number of
    # all insoluble classes (lines 455-463).
    def insoluble(short, sp):
        if short is None:
            return zeros
        rp = _ratio_pow(spec, short, sp, masses, volume=False)
        return zeros if rp is None else rp * number[short] * air_density

    n_bc_ki = insoluble(classes.insoluble_aitken, "bc")
    n_du_ai = insoluble(classes.insoluble_accumulation, "du")
    n_du_ci = insoluble(classes.insoluble_coarse, "du")
    n_insol = zeros
    for mode in spec.modes:
        if not mode.soluble:
            n_insol = n_insol + number[mode.short] * air_density

    def radius(short):
        return zeros if short is None else wet_radius[short]

    # ham_IN_setup, lines 116-120.
    return HeterogeneousFreezingAerosol(
        dust_soluble=_fraction(n_du_sol, activated_cdnc),
        dust_insoluble_accumulation=_fraction(n_du_ai, n_insol),
        dust_insoluble_coarse=_fraction(n_du_ci, n_insol),
        bc_soluble=_fraction(n_bc_sol, activated_cdnc),
        bc_insoluble=_fraction(n_bc_ki, n_insol),
        wet_radius_insoluble_aitken=radius(classes.insoluble_aitken),
        wet_radius_insoluble_accumulation=radius(classes.insoluble_accumulation),
        wet_radius_insoluble_coarse=radius(classes.insoluble_coarse),
    )


#: ``crdiv(3)`` (mo_ham_m7ctl.f90:164): the floor ``ham_IN_setup`` applies to
#: the cirrus aerosol radius ``paprx`` (M7's accumulation-mode lower radius
#: bound, 0.05 um).
_CRDIV3_CM = 0.05e-4


def ham_cirrus_aerosol(
    spec: ModalAerosolSpec,
    number: dict[str, jnp.ndarray],
    wet_radius_accumulation_soluble: jnp.ndarray,
    air_density: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the cirrus-freezing inputs of ``mo_ham_freezing.f90::ham_IN_setup``.

    Port of the ``ld_het = .FALSE.`` branch (lines 122-175) -- the only
    reachable one: ``lhetfreeze`` is an ``em_error`` unless ECHAM is
    compiled with ``-DWITH_LHET`` (``mo_ham.f90:616-622``), so
    ``ld_het = lhetfreeze`` is always ``.FALSE.``. The ``ld_het = .TRUE.``
    branch (feeding ``ndusol_strat`` instead) is not ported for the same
    reason :mod:`jcm.physics.clouds.lohmann_2m.cirrus` does not port
    ``XFRZHET``.

    Args:
        spec: the population; ``spec.cirrus_aerosol_modes`` names which
            classes are soluble and not the nucleation mode (``None`` is a
            caller error -- a population with no cirrus wiring should not
            reach this function at all).
        number: ``class short -> number mixing ratio [1/kg]`` for every class
            named in ``spec.cirrus_aerosol_modes``. This is the END-OF-STEP
            value (HAM's own ``pxtm1 + pxtte*ztmst``, lines 130), NOT the
            step-start value :func:`ham_freezing_aerosol`'s ``masses``/
            ``number`` use for the mixed-phase fractions (HAM's own bare
            ``pxtm1`` there) -- a genuine difference between the two
            subroutines' own conventions, not an inconsistency introduced
            here.
        wet_radius_accumulation_soluble: wet radius [m] of the population's
            own accumulation-sized soluble mode (``spec.accumulation_mode``;
            HAM's ``rwet(iaccs)``).
        air_density: [kg/m3].

    Returns:
        ``(pascs, papnx, paprx)``:

        * ``pascs`` [1/kg] -- despite ``ham_IN_setup``'s own comment calling
          it a "number conc." (implying m⁻³), the CODE sums bare number
          MIXING ratios (``pxtm1``/``pxtte``, both [1/kg]); this ports what
          the code does, not what its comment claims.
        * ``papnx`` [1/m3] = ``air_density * pascs``.
        * ``paprx`` [cm] = ``max(100*wet_radius_accumulation_soluble,
          crdiv(3))``. Numerically inert through
          :func:`jcm.physics.clouds.lohmann_2m.cirrus.xfrzmstr` (``NOSIZE``
          is always true there too), returned for interface completeness
          with ``ham_IN_setup``'s own signature.

        ``papsigx`` is always exactly ``1.0`` (lines 164, 173) -- a
        constant, not derived, so it is not returned.

    """
    if spec.cirrus_aerosol_modes is None:
        raise ValueError(
            f"{spec.family!r} population has no cirrus_aerosol_modes; "
            "ham_cirrus_aerosol is only for a population wired to "
            "nic_cirrus=2 (M7).")
    pascs = jnp.zeros_like(air_density)
    for short in spec.cirrus_aerosol_modes:
        pascs = pascs + number[short]
    pascs = jnp.maximum(pascs, 1.0e7)
    papnx = air_density * pascs
    paprx = jnp.maximum(100.0 * wet_radius_accumulation_soluble, _CRDIV3_CM)
    return pascs, papnx, paprx
