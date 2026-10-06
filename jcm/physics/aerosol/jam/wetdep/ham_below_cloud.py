"""ECHAM-HAM's own below-cloud scavenging: the Croft size-dependent tables (#1017).

``WetScavenging(scheme="ham_below_cloud")`` replaces the default CAM/Slinn
below-cloud impaction pathway (``impaction.py``) with ECHAM-HAM's
``nwetdep=3`` below-cloud scheme (``ham_setscav`` maps ``nwetdep=3`` to
``kscavBCtype=3``, ``mo_ham_wetdep.f90:1407-1411`` -- it does NOT read the
mode-wise ``csr_strat_*``/``cbcr``/``cbcs`` tables ``nwetdep=1``/``2`` use).
Only this one pathway is ported here; the in-cloud pathways (nucleation,
impaction) still run exactly as the ``"jcm"`` scheme's ``incloud_*``
machinery does today under either setting -- see ``wetdep_term.py``'s
module docstring and jax-gcm#1017's follow-ups A/B.

ECHAM input -> jcm source
-------------------------
======================  =========================================  ==========================================================
ECHAM (``mo_ham_wetdep.f90`` unless noted)                          jcm source
======================  =========================================  ==========================================================
``pfrain``/``pfsnow``  rain/snow flux falling INTO this layer       ``CloudData``-adjacent ``"pfrain"``/``"pfsnow"``
                       [kg/m2/s] (:147-148) -- the in-cloud,         diagnostics, published by ``Lohmann2MMicrophysics``
                       cover-normalised, pre-evaporation flux        only when ``configure_wetdep_hydro_diagnostics(True)``
                       ``update_precip_fluxes`` computes every       is set -- the SAME call's own ``pfrain``/``pfsnow``
                       level (mo_cloud_micro_2m.f90), threaded to    OUTPUTs threaded out, not a derived or approximated
                       ``cloud_subm_2`` as ``zfrain``/``zfsnow``     quantity.
                       (:1813); ``cloud_subm_2``'s own comment at
                       mo_submodel_interface.f90:1676-1677 reads
                       "rain/snow flux before evaporation", matching
                       exactly.
``pclc``               fraction of grid box covered by precip,      ``CloudData``-adjacent ``"precip_cover"`` diagnostic,
                       STRATIFORM case (:147; traced to              published by ``Lohmann2MMicrophysics`` only when
                       ``mo_submodel_interface.f90:1771``'s          ``configure_wetdep_hydro_diagnostics(True)`` is set
                       ``pclcpre``, itself an argument of            (``echam_physics`` turns this on exactly when
                       ``cloud_subm_2`` supplied by ECHAM's cloud    ``jam_wetdep_scheme="ham_below_cloud"``); the POST-
                       microphysics, outside this file)              update value (``mo_cloud_micro_2m.f90:1719-1742``)
``mr``                 wet radius, clipped to 50 um, converted to   ``aer.r_wet[i]`` (number tracers) or
                       um, mass-median-scaled for mass tracers      ``aer.r_wet[i]*cmedr2mmedr(mode.geom_std_dev)`` (mass
                                                                     tracers) -- ``cmedr2mmedr = exp(3 ln^2(sigma_g))``
                       (:272-273, ``zrad_fac``)                     (``mo_ham_m7ctl.f90:419``)
======================  =========================================  ==========================================================

``pclc`` and ``pfrain``/``pfsnow`` are the inputs this slice could not map to
an existing jcm diagnostic outright; all three are now wired exactly (jcm's
2M scheme already computes each identically to ECHAM internally, it just
discarded them before this PR) -- see
``docs/source/design/jam_aerosol_removal.md``'s "The HAM below-cloud scheme"
section, plus the two reference-harness findings (the 50 um clip and the
``Q12``/``Q21`` corner swap) below.

Croft tables, lookup and index formulas
----------------------------------------
``crainrate``/``caerorad``/``cscavbcrn``/``cscavbcrm``/``csnowcolleff`` are
ECHAM6.3-HAM2.3 r7492's ``mo_ham_wetdep_data.f90`` tables, extracted as
NUMBERS (never as Fortran text) by compiling the unmodified data module with
a print driver -- see ``jcm/data/wetdep/croft_bc_tables.npz`` and
``/scr/dwatsonparris/ham-m7/w2/wetdep/harness/`` (private scratch, not
pushed). ``crainrate`` has 10 points, ``caerorad``/the table's aerosol axis
61, ``cscavbcrn``/``cscavbcrm`` (number/mass, rain) are ``(10, 61)``,
``csnowcolleff`` is ``(2, 61)`` with row 0 provably dead (``bc_snow`` always
reads row 1, ``mo_ham_wetdep.f90:1116-1119``).

The aerosol-radius bin index is ``FLOOR(3*log(1e4*mr_m)/log(2) + 1)``
(``mr_m`` already converted to um, :278), clipped to ``[0, 60]`` and forced
to 0 where ``mr <= eps`` (:275-284); the rain-rate bin is
``FLOOR(2*log10(3600*pfrain) + 5)`` clipped to ``[0, 9]``, forced to 0 where
``pfrain <= 0`` (:1004-1017). Both indices are read as a PAIR (``idx``,
``idx+1``, clipped independently) and fed to the same bilinear interpolation
(``scavcoef_bilinterp``, ``mo_ham_tools.f90:424-527``, ported verbatim in
:func:`_bilinear_interp` including its four degenerate branches -- two of
which read corner values that are provably identical to the "other" corner
whenever that branch is taken, not a latent bug).

``bc_rain``'s table VALUE is the final [1/s] removal rate directly (no
further scaling: the rain-rate axis IS ``crainrate``, so the rate's flux
dependence is already baked into the table, ``mo_ham_wetdep.f90:1053-1057``).
``bc_snow``'s table value is a collection EFFICIENCY that still needs
scaling by the snow flux and the literal constant ``0.6/0.027``
(``mo_ham_wetdep.f90:1133``, un-named in the source -- a snowflake fall-
speed/density ratio by inspection, not otherwise documented there).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_TABLE_PATH = Path(__file__).resolve().parents[4] / "data" / "wetdep" / "croft_bc_tables.npz"
_EPS = float(np.finfo(np.float64).eps)
_SNOW_SCALE = 0.6 / 0.027  # mo_ham_wetdep.f90:1133, literal in the source
# mo_ham_wetdep.f90:272: ``mr = MIN(rwet_p*zrad_fac, 50.E-6_dp)`` -- the SAME
# clipped value then feeds BOTH the bin-index formula and the bilinear
# interpolation's own radius argument (mo_ham_wetdep.f90:275-284, :1030-1032).
# caerorad's top node is 83.23 um (see croft_bc_tables.npz), well above this
# 50 um cap, so without it a large wet radius (e.g. a swollen coarse mode at
# high RH, especially for a MASS tracer after the cmedr2mmedr up-scaling)
# would both pick a too-high radius bin AND extrapolate the bilinear
# interpolation past its capped node instead of reading the clamped value
# ECHAM-HAM actually uses. Caught by the compiled-reference harness
# (bc_rain/bc_snow disagreed above 50 um before this clip was added).
_MR_CAP_M = 50.0e-6


@dataclasses.dataclass(frozen=True)
class CroftTables:
    """The five Croft below-cloud arrays, as jnp arrays."""

    crainrate: jnp.ndarray     # (10,)
    caerorad: jnp.ndarray      # (61,)
    cscavbcrn: jnp.ndarray     # (10, 61) number, rain
    cscavbcrm: jnp.ndarray     # (10, 61) mass, rain
    csnowcolleff: jnp.ndarray  # (2, 61); row 0 is dead data, row 1 is read


_DEFAULT_TABLES_NP: dict[str, np.ndarray] | None = None


def load_croft_tables(path: Path | str = _TABLE_PATH) -> CroftTables:
    """Load the Croft tables from the packaged NumPy archive."""
    with np.load(path) as z:
        return CroftTables(
            crainrate=jnp.asarray(z["crainrate"]),
            caerorad=jnp.asarray(z["caerorad"]),
            cscavbcrn=jnp.asarray(z["cscavbcrn"]),
            cscavbcrm=jnp.asarray(z["cscavbcrm"]),
            csnowcolleff=jnp.asarray(z["csnowcolleff"]),
        )


def default_croft_tables() -> CroftTables:
    """Process-wide memoised default tables.

    The memo caches plain NumPy arrays, not ``jnp`` ones: wrapping to
    ``jnp`` happens fresh on every call. ``default_croft_tables()`` can be
    called eagerly OR from inside a jit trace (e.g. as the tables-less
    fallback of ``bc_rain_rate``/``bc_snow_rate``, direct callers like
    tests); caching a ``jnp`` array instead would bind it to whichever
    trace happened to call this FIRST, and reusing that stale tracer from a
    later, different trace raises ``jax.errors.UnexpectedTracerError`` (hit
    once already -- the production path in ``WetScavenging.__init__`` now
    avoids this global entirely by loading once at construction, outside
    any trace; this fixes the lazy default for every other caller).
    """
    global _DEFAULT_TABLES_NP
    if _DEFAULT_TABLES_NP is None:
        with np.load(_TABLE_PATH) as z:
            _DEFAULT_TABLES_NP = {k: np.asarray(z[k]) for k in z.files}
    d = _DEFAULT_TABLES_NP
    return CroftTables(
        crainrate=jnp.asarray(d["crainrate"]),
        caerorad=jnp.asarray(d["caerorad"]),
        cscavbcrn=jnp.asarray(d["cscavbcrn"]),
        cscavbcrm=jnp.asarray(d["cscavbcrm"]),
        csnowcolleff=jnp.asarray(d["csnowcolleff"]),
    )


def cmedr2mmedr(geom_std_dev: float) -> float:
    """Count-median-to-mass-median radius ratio (``mo_ham_m7ctl.f90:419``)."""
    return math.exp(3.0 * math.log(geom_std_dev) ** 2)


def aerosol_radius_bin(mr_m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Aerosol wet-radius bin pair (``mo_ham_wetdep.f90:264-284``).

    ``mr_m`` is the wet radius already clipped to 50 um and scaled by
    ``cmedr2mmedr`` for a mass tracer (still in METRES here; this function
    does the m->um conversion, matching ``*1.E+6_dp`` at :273).
    """
    mr_um = mr_m * 1.0e6
    live = mr_um > _EPS
    safe = jnp.where(live, mr_um, 1.0)
    frac = jnp.floor(3.0 * jnp.log(1.0e4 * safe) / math.log(2.0) + 1.0)
    idx1 = jnp.where(live, jnp.clip(frac, 0, 60), 0.0).astype(jnp.int32)
    idx2 = jnp.where(live, jnp.clip(frac + 1.0, 0, 60), 0.0).astype(jnp.int32)
    return idx1, idx2


def rain_rate_bin(pfrain: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rain-rate bin pair (``mo_ham_wetdep.f90:1004-1017``)."""
    live = pfrain > 0.0
    safe = jnp.where(live, pfrain, 1.0)
    frac = jnp.floor(2.0 * jnp.log10(3600.0 * safe) + 5.0)
    idx1 = jnp.where(live, jnp.clip(frac, 0, 9), 0.0).astype(jnp.int32)
    idx2 = jnp.where(live, jnp.clip(frac + 1.0, 0, 9), 0.0).astype(jnp.int32)
    return idx1, idx2


def _bilinear_interp(x, y, x1, x2, y1, y2, q11, q12, q21, q22):
    """Faithful port of ``scavcoef_bilinterp`` (``mo_ham_tools.f90:424-527``).

    Four branches, matching the native ``MERGE`` cascade exactly (including
    which corners the two degenerate branches read -- ``lint1``'s pair
    (Q11, Q21) and ``lint2``'s pair (Q21, Q22) are each provably identical
    to the "other" corner pair whenever that branch fires, since the fixed
    axis's two bin indices are then equal by construction; ported as
    written rather than simplified).
    """
    lint1 = (x1 != x2) & (y1 == y2)
    lint2 = (x1 == x2) & (y1 != y2)
    lint3 = (x1 == x2) & (y1 == y2)
    lint4 = (x1 != x2) & (y1 != y2)

    dx = jnp.where(lint1 | lint4, x2 - x1, 1.0)
    dy = jnp.where(lint2 | lint4, y2 - y1, 1.0)

    x_only = ((x2 - x) / dx) * q11 + ((x - x1) / dx) * q21
    y_only = ((y2 - y) / dy) * q21 + ((y - y1) / dy) * q22
    none = q11
    both = (((y2 - y) / dy) * (((x2 - x) / dx) * q11 + ((x - x1) / dx) * q21)
            + ((y - y1) / dy) * (((x2 - x) / dx) * q12 + ((x - x1) / dx) * q22))

    out = jnp.zeros_like(x_only)
    out = jnp.where(lint1, x_only, out)
    out = jnp.where(lint2, y_only, out)
    out = jnp.where(lint3, none, out)
    out = jnp.where(lint4, both, out)
    return out


def _lookup(table, row_idx1, row_idx2, col_idx1, col_idx2):
    """Gather the four corners for ``_bilinear_interp``, in ``bc_rain``'s OWN
    (non-"intuitive") corner convention.

    ``_bilinear_interp``'s ``lint4`` (full bilinear) formula needs
    ``q12 = f(row1, col2)`` and ``q21 = f(row2, col1)`` to be the standard
    interpolation it reads as (derive it from the formula: the ``(Y2-y)/dy``
    term pairs ``q11``/``q21`` while varying X at ``row1``, i.e. ``q21`` must
    vary ``row`` not ``col``). ``bc_rain``'s own data-filling loop
    (``mo_ham_wetdep.f90:1024-1032``) instead sets
    ``Q12(jl,jk) = cscavbcrn(indexbcrx2, indexy1)`` -- ``row2, col1`` -- and
    ``Q21(jl,jk) = cscavbcrn(indexbcrx1, indexy2)`` -- ``row1, col2`` -- the
    OPPOSITE pairing. Fed through ``scavcoef_bilinterp``'s unchanged formula,
    this makes ``bc_rain``'s actual compiled behaviour silently swap the two
    off-diagonal corners relative to a textbook bilinear read; confirmed
    bit-for-bit against the compiled reference (a "textbook" lookup instead
    gave the wrong rate on every case exercising the ``lint4``/``lint1``
    branches -- bc_snow's own Q11..Q22 filling is immune since its X axis is
    always the ``X1=X2=1`` dummy, which is why it alone did not catch this).
    Ported AS COMPILED, not as "corrected": this is reference fidelity, and
    is flagged to the maintainer as a likely-unintentional upstream quirk
    rather than silently fixed (see the PR description / jax-gcm#1017).
    """
    q11 = table[row_idx1, col_idx1]
    q12 = table[row_idx2, col_idx1]
    q21 = table[row_idx1, col_idx2]
    q22 = table[row_idx2, col_idx2]
    return q11, q12, q21, q22


def bc_rain_rate(pfrain: jnp.ndarray, mr_m: jnp.ndarray, *, phase: str,
                 tables: CroftTables | None = None):
    """Below-cloud rain removal rate [1/s] for ONE tracer phase.

    ``bc_rain``'s ``kscavBCtype=3`` branch (``mo_ham_wetdep.f90:999-1058``):
    the table VALUE is the final rate directly, no further flux scaling.
    ``mr_m`` is THIS tracer's own wet radius [m] (already ``cmedr2mmedr``-
    scaled by the caller for ``phase="mass"`` -- the native routine is
    called once per tracer, with ``mr`` and the table (``cscavbcrn`` or
    ``cscavbcrm``) both selected by the SAME ``ktrac_phase``; number and
    mass are never mixed within one call, so this function takes one
    ``phase`` rather than returning both from a single ``mr_m``).
    """
    if phase not in ("number", "mass"):
        raise ValueError(f"phase must be 'number' or 'mass', got {phase!r}")
    tables = tables or default_croft_tables()
    table = tables.cscavbcrn if phase == "number" else tables.cscavbcrm
    mr_m = jnp.minimum(mr_m, _MR_CAP_M)
    rx1, rx2 = rain_rate_bin(pfrain)
    ry1, ry2 = aerosol_radius_bin(mr_m)
    x1, x2 = tables.crainrate[rx1], tables.crainrate[rx2]
    y1, y2 = tables.caerorad[ry1], tables.caerorad[ry2]
    q11, q12, q21, q22 = _lookup(table, rx1, rx2, ry1, ry2)
    return _bilinear_interp(pfrain, mr_m * 1.0e6, x1, x2, y1, y2, q11, q12, q21, q22)


def bc_snow_rate(pfsnow: jnp.ndarray, mr_m: jnp.ndarray,
                 tables: CroftTables | None = None):
    """Below-cloud snow removal rate [1/s] (same for number and mass).

    ``bc_snow``'s ``kscavBCtype=3`` branch (``mo_ham_wetdep.f90:1106-1138``):
    the table (``csnowcolleff``, row 1 only -- row 0 is dead, X1=X2=1
    disables X-interpolation, ``mo_ham_wetdep.f90:1112-1119``) gives a
    collection EFFICIENCY, mass/number-agnostic AT THE TABLE LEVEL -- but
    ``mr_m`` (hence the aerosol-radius bin) still differs between a number
    and a mass tracer exactly as for rain, so this must still be called
    once per tracer phase with that phase's own ``mr_m``; there is no
    ``phase`` argument only because the table itself does not branch on it.
    Rescaled by the snow flux and the literal ``0.6/0.027``
    (``mo_ham_wetdep.f90:1133``), and exactly zero where ``pfsnow<=eps``
    (the native ``MERGE``, :1131-1135) rather than wherever the bin index
    gate would otherwise leave a nonzero efficiency at zero flux.
    """
    tables = tables or default_croft_tables()
    mr_m = jnp.minimum(mr_m, _MR_CAP_M)
    y1, y2 = aerosol_radius_bin(mr_m)
    yv1, yv2 = tables.caerorad[y1], tables.caerorad[y2]
    q11 = tables.csnowcolleff[1, y1]
    q12 = tables.csnowcolleff[1, y2]
    # X1=X2=1 (a dummy, matching the native MERGE(... 1., 1. ...)): disables
    # X-interpolation, so every corner reads the SAME row (both Q columns
    # of the X pair are identical) and only the Y-only branch (lint2) can
    # fire -- lint1/lint4 never trigger since x1==x2 always here.
    x_dummy = jnp.ones_like(mr_m)
    efficiency = _bilinear_interp(
        jnp.zeros_like(mr_m), mr_m * 1.0e6, x_dummy, x_dummy, yv1, yv2,
        q11, q12, q11, q12,
    )
    live = pfsnow > _EPS
    rate = _SNOW_SCALE * pfsnow * efficiency
    return jnp.where(live, rate, 0.0)
