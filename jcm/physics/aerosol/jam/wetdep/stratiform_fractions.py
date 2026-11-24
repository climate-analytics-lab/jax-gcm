"""HAMMOZ's mode-wise stratiform in-cloud scavenging fractions, mapped onto JAM.

ECHAM-HAM offers two stratiform nucleation-scavenging options, selected by
``kscavICtype`` in ``mo_ham_wetdep.f90::get_icscavfrac``. ``kscavICtype = 3``
derives the scavenged fraction from the aerosol's own activated/impacted size
spectrum (``ic_scav_nuc``, the M7 ``scheme="ham"`` path in ``wetdep_term.py``,
and the equivalent ARG-activated-fraction weighting on the default MAM4
population — see ``WetScavenging``'s ``stratiform_nucleation`` parameter).
This module ports the OTHER option, ``kscavICtype = 2`` ("mode-wise
fractions"): every mode converts a FIXED fraction of its in-cloud mass to
precipitation whenever the cloud's own water converts, with that fraction set
only by the cloud's thermodynamic phase — not by how much of the mode
actually activated. ``mo_ham_m7ctl.f90:515-518`` tabulates it per M7 class:

====================  ====  ====  ====  ====  ====  ====  ====
M7 class              NS    KS    AS    CS    KI    AI    CI
====================  ====  ====  ====  ====  ====  ====  ====
``csr_strat_wat``     0.10  0.25  0.85  0.99  0.20  0.40  0.40
``csr_strat_mix``     0.10  0.40  0.75  0.75  0.10  0.40  0.40
``csr_strat_ice``     0.10  0.10  0.10  0.10  0.10  0.10  0.10
====================  ====  ====  ====  ====  ====  ====  ====

JAM maps MAM4's four modes onto the M7 class of matching solubility and dry
size, reusing EXACTLY the mapping ``wetdep.convective_fractions`` already
uses for HAMMOZ's convective ``csr_conv`` from the same table (same
reasoning, see that module's docstring for the size/solubility
justification): accumulation -> AS, coarse -> CS, Aitken -> KS, primary
carbon -> KI. MAM4 carries no nucleation (NS) mode. M7's own seven modes
(``microphysics.m7_data.M7_SPEC``) read the table directly at their own
class — no mapping needed, since M7 *is* HAM's native mode set — which is
what lets ``WetScavenging(spec=M7_SPEC, scheme="ham_below_cloud")`` or
``scheme="jcm"`` (an ablation/testing composition, never the shipped M7
preset, which uses ``scheme="ham"``'s own ``ic_scav_nuc`` instead) size and
fill its table from the population it was actually composed with.

Which fraction applies is chosen by GRID temperature, not by which of the
cloud's own liquid/ice pools is currently converting: a tracer's in-droplet
(HAM's ``kwat_phase=1``) and in-crystal (``kwat_phase=2``) reservoirs are
scavenged with possibly different fractions, but the SAME temperature rule
picks both (``mo_ham_wetdep.f90::get_icscavfrac`` CASE(2), lines 597-623):

* ``T > tmelt`` (warm, liquid-only regime): ``csr_strat_wat`` for both
  reservoirs.
* ``cthomi < T <= tmelt`` (mixed-phase regime): ``csr_strat_mix`` for both.
* ``T <= cthomi`` (homogeneous-freezing threshold, ``tmelt - 35 K``,
  ``mo_echam_cloud_params.f90:54``): ``csr_strat_wat`` for the in-droplet
  reservoir (unchanged from the warm/mixed boundary case — any liquid
  surviving at this temperature is treated like the warm regime) but
  ``csr_strat_ice`` for the in-crystal one.

This is JAM's issue #1080 default for the stratiform nucleation-scavenging
WEIGHT applied to ``f_wat``/``f_ice`` (the per-step liquid/ice condensate->
precipitation conversion, ``wetdep_term.phase_split``): see
``WetScavenging``'s ``stratiform_nucleation`` parameter, which also keeps the
previous ARG-activated-fraction weighting available as
``"activated_fraction"`` for comparison and bit-for-bit reproduction of
pre-#1080 runs. It does not touch the M7 ``scheme="ham"`` path, which
already runs HAM's size-dependent ``ic_scav_nuc`` (``kscavICtype = 3``).
"""

from __future__ import annotations

import jax.numpy as jnp

import jcm.constants as c

#: ECHAM's homogeneous-freezing threshold, ``mo_echam_cloud_params.f90:54``
#: (``cthomi = tmelt - 35``): below it ``get_icscavfrac`` treats the in-crystal
#: reservoir with the mode's ice-phase fraction instead of its mixed-phase one.
_CTHOMI_OFFSET = 35.0

#: Per-mode-``short`` ``(M7 class, csr_strat_wat, csr_strat_mix,
#: csr_strat_ice)`` from ``mo_ham_m7ctl.f90:515-518``. Keyed by
#: ``AerosolMode.short``, not ``.name``: MAM4's four short codes
#: (``acc``/``ait``/``cor``/``pcm``) take the M7 class they map onto (see the
#: module docstring and ``wetdep.convective_fractions`` for the shared
#: mapping and its justification); M7's own seven short codes
#: (``ns``/``ks``/``as``/``cs``/``ki``/``ai``/``ci``) ARE the M7 class codes
#: and read the table natively, with no mapping. The two code sets are
#: disjoint, so one dict serves both populations.
HAM_CSR_STRAT: dict[str, tuple[str, float, float, float]] = {
    # MAM4 (mapped to the M7 class of matching solubility and dry size).
    "acc": ("AS", 0.85, 0.75, 0.10),
    "cor": ("CS", 0.99, 0.75, 0.10),
    "ait": ("KS", 0.25, 0.40, 0.10),
    "pcm": ("KI", 0.20, 0.10, 0.10),
    # M7 (native: each short code already IS its M7 class).
    "ns": ("NS", 0.10, 0.10, 0.10),
    "ks": ("KS", 0.25, 0.40, 0.10),
    "as": ("AS", 0.85, 0.75, 0.10),
    "cs": ("CS", 0.99, 0.75, 0.10),
    "ki": ("KI", 0.20, 0.10, 0.10),
    "ai": ("AI", 0.40, 0.40, 0.10),
    "ci": ("CI", 0.40, 0.40, 0.10),
}


def _lookup(mode_short: str, index: int) -> float:
    try:
        return HAM_CSR_STRAT[mode_short][index]
    except KeyError:
        raise KeyError(
            f"No HAMMOZ csr_strat mapping for mode short code {mode_short!r}; "
            "add it to HAM_CSR_STRAT with the M7 class it corresponds to."
        ) from None


def strat_wat(mode_short: str) -> float:
    """HAMMOZ's warm-cloud (and sub-``cthomi``, in-droplet) stratiform fraction."""
    return _lookup(mode_short, 1)


def strat_mix(mode_short: str) -> float:
    """HAMMOZ's mixed-phase-cloud stratiform fraction."""
    return _lookup(mode_short, 2)


def strat_ice(mode_short: str) -> float:
    """HAMMOZ's sub-``cthomi`` in-crystal stratiform fraction."""
    return _lookup(mode_short, 3)


def regime_fractions(
    csr_wat: jnp.ndarray,
    csr_mix: jnp.ndarray,
    csr_ice: jnp.ndarray,
    temperature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(f_liquid, f_ice)``: HAM's temperature-regime fraction select.

    ``csr_wat``/``csr_mix``/``csr_ice`` are one mode's three tabulated
    fractions (read from the differentiable ``WetDepParameters`` leaves, not
    this module's plain-float table, so a gradient can reach them); see the
    module docstring for ``get_icscavfrac``'s regime rule. ``f_liquid`` is the
    weight for the in-droplet (``kwat_phase=1``) reservoir, ``f_ice`` for the
    in-crystal (``kwat_phase=2``) one; both are broadcastable against
    ``temperature``.
    """
    cthomi = c.tmelt - _CTHOMI_OFFSET
    mixed = (temperature <= c.tmelt) & (temperature > cthomi)
    cold = temperature <= cthomi
    f_liquid = jnp.where(mixed, csr_mix, csr_wat)        # cold -> wat, same as warm
    f_ice = jnp.where(cold, csr_ice, jnp.where(mixed, csr_mix, csr_wat))
    return f_liquid, f_ice
