"""ECHAM-HAM's own in-cloud impaction scavenging: ``ic_scav_imp`` (#1017 follow-up B).

Stratiform aerosol-size-dependent impaction scavenging (``kscavICtype=3``,
same selector as follow-up A's nucleation pathway -- see
``get_icscavfrac``, mo_ham_wetdep.f90:560-681) computes a per-mode,
per-phase (water/ice), per-tracer-phase (number/mass) collection
coefficient by bilinear interpolation of Croft et al. (2010)'s look-up
tables against (collector radius, aerosol radius): cloud-droplet radius
(``reffl``) for the water phase, ice-plate radius (``reffi``) for ice.
``get_icscavfrac`` sums this with follow-up A's nucleation fraction,
``pfrac = clip(pfrac_nuc + pfrac_imp, 0, 1)`` -- see ``wetdep_term.py``'s
module docstring for how ``WetScavenging`` combines the two.

ECHAM input -> jcm source
-------------------------
======================  =========================================  ==========================================================
ECHAM (``mo_ham_wetdep.f90`` unless noted)                          jcm source
======================  =========================================  ==========================================================
``reffl``/``reffi``        the 2M scheme's OWN effective liquid/ice     ``Lohmann2MMicrophysics``'s own ``preffl``/``preffi``
                           radius streams [um] (``mo_activ.f90``,       (``eff_liquid_droplet_radius``/``eff_ice_crystal_radius``),
                           read via ``USE mo_activ, ONLY: reffl,        published as diagnostics ``"reffl"``/``"reffi"`` by
                           reffi``) -- NOT the radiation's own          ``configure_wetdep_hydro_diagnostics(True)`` (extended by
                           independently-formed radius.                this follow-up) -- the scheme's own value the real
                                                                        ``USE mo_activ`` resolves to, not a second estimate.
``mr``                     wet radius, clipped to 50 um, converted to  ``aer.r_wet[i]`` / ``aer.r_wet[i]*cmedr2mmedr(sigma_g)``,
                           um, mass-median-scaled for mass tracers     exactly as follow-up A's ``ham_below_cloud.py`` already
                           (:272-273, the SAME ``zrad_fac``/``indexy``  derives it -- ``aerosol_radius_bin`` is reused, not
                           block follow-up A's nucleation path and      re-derived, since ic_scav_imp reads the identical
                           the below-cloud path both also use)         ``indexy1``/``indexy2`` the real module computes once.
``idt_icnc``               in-cloud ICNC mixing ratio (ice phase       ``state.tracers["qni"]`` * ``clouds.cloud_fraction`` *
                           only, :852's ``zicnc = pxtp1c(idt_icnc)      ``air_density`` -- the SAME ``icnc_incloud`` follow-up A's
                           *prhop1``)                                  ``ice_phase_xie`` already builds, times density.
``cdroprad``/``caerorad``/  Croft tables, extracted as NUMBERS (never  ``jcm/data/wetdep/croft_ic_tables.npz`` (cdroprad/
``cplaterad``/             as Fortran text) by compiling the           cplaterad/scavdropn/scavdropm/scaviceplate) +
``scavdropn``/             unmodified ``mo_ham_wetdep_data.f90`` with  ``croft_bc_tables.npz``'s ``caerorad`` (the shared
``scavdropm``/             a print driver                              aerosol-radius axis, reused from follow-up A/the
``scaviceplate``                                                       below-cloud slice, not duplicated).
======================  =========================================  ==========================================================

Index formulas and the corner-swap quirk
-----------------------------------------
The aerosol-radius (Y) axis index pair is ``aerosol_radius_bin`` from
``ham_below_cloud.py`` -- the SAME ``indexy1``/``indexy2`` formula
(mo_ham_wetdep.f90:264-284), reused rather than re-derived, since
``ic_scav_imp`` reads the identical module-level arrays the below-cloud
pathway and follow-up A's nucleation pathway's radius selection both
populate.

The collector-radius (X) axis is NEW to this slice:

- water (``drop_radius_bin``): ``FLOOR(reffl/5)`` clipped to ``[0, 9]``
  (mo_ham_wetdep.f90:861-865), zeroed where ``reffl <= eps``.
- ice (``plate_radius_bin``): a THREE-regime index (mo_ham_wetdep.f90:
  908-927) -- ``reffi<1`` or ``icnc<eps``: index 0 (dead, the final
  ``icnc`` multiplication already zeroes the result); ``1<=reffi<50``:
  ``FLOOR(reffi/5)`` clipped to ``[0, 10]``; ``reffi>=50``:
  ``8+FLOOR(reffi/50)`` clipped to ``[0, 34]`` -- note the discontinuous
  "+8" offset into ``cplaterad``'s SAME array (its first 11 entries are
  5-um-spaced 0..50, the rest 25-um-spaced 50..650: cplaterad[8]=40,
  cplaterad[9]=45, cplaterad[10]=50 overlap the coarse branch's own
  cplaterad[10]=50 -- the two regimes' index-8..10 ranges genuinely
  overlap by construction, not a bug).

Both phases fill their four interpolation corners (Q11..Q22) with the
SAME (row2,col1)/(row1,col2) swap follow-up A's below-cloud ``bc_rain``
already found and ported AS COMPILED (``ham_below_cloud.py``'s
``lookup_swapped_corners`` docstring) -- confirmed the identical quirk,
not a coincidence, by the compiled full-chain harness, so it is reused
rather than re-derived.

Water's table value is the scavenged FRACTION directly (no further
transform, mo_ham_wetdep.f90:904); ice's is a collection COEFFICIENT that
still needs the exponential transform against ICNC and the timestep
(mo_ham_wetdep.f90:955-956) -- this asymmetry is real, ported as compiled,
not simplified to match water's form.

``cdroprad(6)`` reads 0.0, not 30.0
------------------------------------
``cdroprad = (0, 5, 10, 15, 20, 25, [0], 35, 40, 45, 50)`` -- index 6 breaks
the otherwise-regular 5-um spacing, confirmed in the COMPILED module
output (``jcm/data/wetdep/croft_ic_tables_provenance.json``), not merely
the source listing. It is almost certainly an upstream typo for 30.0, and
it matters: liquid in-cloud impaction reads this node for effective radii
in [25, 35) um, an ordinary warm-cloud range, where the water impaction
fraction differs from the 30.0 reading by up to 22 % (see this module's
``measure_cdroprad_bug_6_effect``). The maintainer decided (2026-10-06)
that jcm uses the corrected axis ``CDROPRAD_UM_TYPO_CORRECTED`` as the
default; the compiled r7492 axis ``CDROPRAD_UM_AS_COMPILED`` stays
available through ``WetDepParameters.cdroprad_um`` so a like-for-like
comparison with ECHAM-HAM can reproduce its numbers exactly.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.wetdep.ham_below_cloud import (
    aerosol_radius_bin,
    bilinear_interp,
    lookup_swapped_corners,
)

_TABLE_PATH = Path(__file__).resolve().parents[4] / "data" / "wetdep" / "croft_ic_tables.npz"
_EPS = float(np.finfo(np.float64).eps)
_MR_CAP_M = 50.0e-6  # mo_ham_wetdep.f90:272, the SAME clip the below-cloud path applies.

#: ECHAM's own axis, AS COMPILED (mo_ham_wetdep_data.f90:299-301) -- index 6
#: reads 0.0 where the regular 5-um spacing implies 30.0. Kept as the
#: reproduce-r7492 override (select it through ``WetDepParameters.cdroprad_um``,
#: ``water_impaction_fraction``'s ``cdroprad_um`` argument) and as the axis the
#: compiled-reference tests compare against; the functions below take it as
#: their own default so those tests exercise the port against the compiled
#: tables unchanged.
CDROPRAD_UM_AS_COMPILED = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 0.0, 35.0, 40.0, 45.0, 50.0)
#: The regular-spacing reading with 30.0 at index 6: jcm's DEFAULT
#: (``WetDepParameters.default()``) by maintainer decision 2026-10-06, see the
#: module docstring.
CDROPRAD_UM_TYPO_CORRECTED = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)


@dataclasses.dataclass(frozen=True)
class ImpactionTables:
    """The three NEW Croft in-cloud impaction arrays, as jnp arrays.

    ``caerorad`` (the shared aerosol-radius Y axis) is NOT duplicated here
    -- callers already hold it via ``ham_below_cloud.CroftTables`` (the
    below-cloud slice loaded it first) and pass it in separately.
    """

    cplaterad: jnp.ndarray     # (35,)
    scavdropn: jnp.ndarray     # (10, 61) number, water
    scavdropm: jnp.ndarray     # (10, 61) mass, water
    scaviceplate: jnp.ndarray  # (35, 61)


_DEFAULT_TABLES_NP: dict[str, np.ndarray] | None = None


def load_impaction_tables(path: Path | str = _TABLE_PATH) -> ImpactionTables:
    """Load the in-cloud impaction tables from the packaged NumPy archive."""
    with np.load(path) as z:
        return ImpactionTables(
            cplaterad=jnp.asarray(z["cplaterad"]),
            scavdropn=jnp.asarray(z["scavdropn"]),
            scavdropm=jnp.asarray(z["scavdropm"]),
            scaviceplate=jnp.asarray(z["scaviceplate"]),
        )


def default_impaction_tables() -> ImpactionTables:
    """Process-wide memoised default tables (see ``default_croft_tables``'s
    docstring for why the memo caches plain NumPy, not ``jnp``, arrays).
    """
    global _DEFAULT_TABLES_NP
    if _DEFAULT_TABLES_NP is None:
        with np.load(_TABLE_PATH) as z:
            _DEFAULT_TABLES_NP = {k: np.asarray(z[k]) for k in z.files}
    d = _DEFAULT_TABLES_NP
    return ImpactionTables(
        cplaterad=jnp.asarray(d["cplaterad"]),
        scavdropn=jnp.asarray(d["scavdropn"]),
        scavdropm=jnp.asarray(d["scavdropm"]),
        scaviceplate=jnp.asarray(d["scaviceplate"]),
    )


def drop_radius_bin(reffl: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cloud-droplet radius bin pair (``mo_ham_wetdep.f90:859-865``)."""
    live = reffl > _EPS
    safe = jnp.where(live, reffl, 1.0)
    frac = jnp.floor(safe / 5.0)
    idx1 = jnp.where(live, jnp.clip(frac, 0, 9), 0.0).astype(jnp.int32)
    idx2 = jnp.where(live, jnp.clip(frac + 1.0, 0, 9), 0.0).astype(jnp.int32)
    return idx1, idx2


def plate_radius_bin(reffi: jnp.ndarray, icnc: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ice-plate radius bin pair, the three-regime index (``mo_ham_wetdep.f90:908-927``).

    ``icnc`` is the ice crystal NUMBER CONCENTRATION [m-3] (``zicnc``), not
    a mixing ratio -- multiply by air density before calling, as
    ``ice_impaction_rate`` does.
    """
    have_ice = icnc >= _EPS
    fine = have_ice & (reffi < 50.0) & (reffi >= 1.0)
    coarse = have_ice & (reffi >= 50.0)

    safe_fine = jnp.where(fine, reffi, 1.0)
    fine1 = jnp.clip(jnp.floor(safe_fine / 5.0), 0, 10)
    fine2 = jnp.clip(jnp.floor(safe_fine / 5.0) + 1.0, 0, 10)

    safe_coarse = jnp.where(coarse, reffi, 1.0)
    coarse1 = jnp.clip(8.0 + jnp.floor(safe_coarse / 50.0), 0, 34)
    coarse2 = jnp.clip(9.0 + jnp.floor(safe_coarse / 50.0), 0, 34)

    idx1 = jnp.where(coarse, coarse1, jnp.where(fine, fine1, 0.0)).astype(jnp.int32)
    idx2 = jnp.where(coarse, coarse2, jnp.where(fine, fine2, 0.0)).astype(jnp.int32)
    return idx1, idx2


def water_impaction_fraction(
    reffl: jnp.ndarray, mr_m: jnp.ndarray, *, phase: str, caerorad: jnp.ndarray,
    tables: ImpactionTables | None = None,
    cdroprad_um: tuple[float, ...] | jnp.ndarray = CDROPRAD_UM_AS_COMPILED,
) -> jnp.ndarray:
    """In-cloud water-phase impaction scavenged fraction [-], ONE tracer phase.

    ``ic_scav_imp``'s ``CASE(1)`` (``mo_ham_wetdep.f90:857-904``): the table
    value IS the scavenged fraction, no further transform. ``mr_m`` is this
    tracer's own wet radius [m] (mass-median-scaled by the caller for
    ``phase="mass"``, exactly as ``ham_below_cloud.bc_rain_rate`` does).
    """
    if phase not in ("number", "mass"):
        raise ValueError(f"phase must be 'number' or 'mass', got {phase!r}")
    tables = tables or default_impaction_tables()
    cdroprad = jnp.asarray(cdroprad_um)
    table = tables.scavdropn if phase == "number" else tables.scavdropm
    mr_m = jnp.minimum(mr_m, _MR_CAP_M)
    dx1, dx2 = drop_radius_bin(reffl)
    ry1, ry2 = aerosol_radius_bin(mr_m)
    x1, x2 = cdroprad[dx1], cdroprad[dx2]
    y1, y2 = caerorad[ry1], caerorad[ry2]
    q11, q12, q21, q22 = lookup_swapped_corners(table, dx1, dx2, ry1, ry2)
    return bilinear_interp(reffl, mr_m * 1.0e6, x1, x2, y1, y2, q11, q12, q21, q22)


def ice_impaction_fraction(
    reffi: jnp.ndarray, icnc: jnp.ndarray, mr_m: jnp.ndarray, dt: jnp.ndarray,
    caerorad: jnp.ndarray, tables: ImpactionTables | None = None,
) -> jnp.ndarray:
    """In-cloud ice-phase impaction scavenged fraction [-].

    ``ic_scav_imp``'s ``CASE(2)`` (``mo_ham_wetdep.f90:906-956``): the table
    value is a collection COEFFICIENT, rescaled by the exponential transform
    against ICNC and the timestep -- NOT the same form as the water branch
    (ported as compiled, not unified). ``icnc`` is the ice NUMBER
    CONCENTRATION [m-3] (``zicnc = pxtp1c(idt_icnc)*prhop1`` -- multiply the
    mixing ratio by air density before calling). ``mr_m`` is this tracer's
    own wet radius [m] (same convention as the water branch and follow-up A).
    """
    tables = tables or default_impaction_tables()
    mr_m = jnp.minimum(mr_m, _MR_CAP_M)
    px1, px2 = plate_radius_bin(reffi, icnc)
    ry1, ry2 = aerosol_radius_bin(mr_m)
    x1, x2 = tables.cplaterad[px1], tables.cplaterad[px2]
    y1, y2 = caerorad[ry1], caerorad[ry2]
    q11, q12, q21, q22 = lookup_swapped_corners(tables.scaviceplate, px1, px2, ry1, ry2)
    coef = bilinear_interp(reffi, mr_m * 1.0e6, x1, x2, y1, y2, q11, q12, q21, q22)
    return -jnp.expm1(-coef * 1.0e-6 * icnc * dt)


def measure_cdroprad_bug_6_effect(
    reffl: np.ndarray, mr_m: np.ndarray, *, phase: str, caerorad: np.ndarray,
    tables: ImpactionTables | None = None,
) -> dict:
    """How often ``cdroprad(6)``'s 0.0-vs-30.0 reading matters, and by how much.

    For each ``reffl`` value, reports whether EITHER drop-radius bin index
    (``drop_radius_bin``) equals 6 (i.e. the lookup's X-axis interpolation
    actually reads the disputed node), and the relative difference in
    :func:`water_impaction_fraction` between the AS-COMPILED axis
    (``CDROPRAD_UM_AS_COMPILED``) and the typo-corrected one
    (``CDROPRAD_UM_TYPO_CORRECTED``). Takes plain NumPy in, returns plain
    NumPy/Python out -- a one-off diagnostic, not part of the traced model
    path.

    Returns:
        A dict with ``"hits_node_6"`` (bool array, same shape as ``reffl``)
        and ``"relative_difference"`` (float array; 0 where the node is not
        touched).

    """
    reffl = np.asarray(reffl, dtype=np.float64)
    mr_m = np.asarray(mr_m, dtype=np.float64)
    idx1, idx2 = (np.asarray(a) for a in drop_radius_bin(reffl))
    hits_node_6 = (idx1 == 6) | (idx2 == 6)

    as_compiled = np.asarray(water_impaction_fraction(
        reffl, mr_m, phase=phase, caerorad=caerorad, tables=tables,
        cdroprad_um=CDROPRAD_UM_AS_COMPILED))
    corrected = np.asarray(water_impaction_fraction(
        reffl, mr_m, phase=phase, caerorad=caerorad, tables=tables,
        cdroprad_um=CDROPRAD_UM_TYPO_CORRECTED))
    denom = np.where(np.abs(as_compiled) > 0.0, np.abs(as_compiled), 1.0)
    relative_difference = np.where(
        np.abs(as_compiled) > 0.0, np.abs(corrected - as_compiled) / denom,
        np.abs(corrected - as_compiled))
    return {"hits_node_6": hits_node_6, "relative_difference": relative_difference}
