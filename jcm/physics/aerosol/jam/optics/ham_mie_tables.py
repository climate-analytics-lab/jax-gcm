r"""HAM-faithful Mie lookup tables (#1017).

Builds the four lookup tables ``mo_ham_rad_data.f90`` defines for the M7
aerosol submodel -- SW fine (lognormal :math:`\sigma_g=1.59`), SW coarse
(:math:`\sigma_g=2.0`), LW fine, LW coarse -- on HAM's own axes, using
jcm's own Bohren-Huffman kernel (:mod:`mie`) to fill them, since the
authentic LUT files are not available (SALSA's copies in this tree are
header-only stubs; the maintainer decision, #1017, is to build the LUT
ourselves rather than go without one). :func:`ham_rad_fitplus` is a faithful
port of the native nearest-neighbour lookup that reads them.

Axes (``mo_ham_rad_data.f90:95-102,420-433``, M7 column of the per-submodel
table): size parameter ``x = 2*pi*r/lambda``, log-spaced, 101 points, table-
specific range (fine SW ``[0.001, 25]``, coarse SW ``[0.4, 40]``, fine LW
``[5e-6, 3]``, coarse LW ``[0.0015, 4]``); real refractive index, linear,
101 points (SW ``[1.33, 2.00]``, LW ``[1.00, 3.00]``); imaginary refractive
index, log-spaced, 201 points (SW ``[1e-9, 1.00]``, LW ``[1e-9, 2.00]``).

Normalisation (not stated as a source comment; derived from the consuming
arithmetic -- ``mo_ham_rad.f90:1070,1219`` multiply the table value by
``lambda**2`` to get the extinction cross-section ``Cext``, and
``Cext = Qext*pi*r**2 = Qext*x**2*lambda**2/(4*pi)`` with
``x = 2*pi*r/lambda``, so the table stores ``Qext*x**2/(4*pi)`` -- a
size-parameter-normalised, wavelength-independent extinction efficiency).
SW tables additionally store plain SSA and asymmetry factor ``g`` (no extra
scaling, ``mo_ham_rad.f90:1016-1018,1113-1115``); LW tables store extinction
only (``mo_ham_rad.f90``'s LW lookup table declarations carry no ``omega``/
``g`` arrays at all).

Known, quantified limitation of jcm's Mie kernel against this table's own
needs (not a blocker; the design doc records it): :mod:`mie`'s ``X_MAX=100``
clip is shared, pinned default-path behaviour this module does not touch.
The lognormal quadrature's Gauss-Hermite tail node (largest abscissa
``t=2.9306``) at ``sigma=2.0`` scales a base ``x`` by ``growth=e^{sqrt(2)
ln(sigma) t} ~ 17.7``, so at the top of ``sw_coarse``'s axis (``x=40``) that
node's TRUE size parameter is ~707, clipped to 100. The true (unclipped)
Mie efficiency has not fully converged at ``x=100`` -- comparing x=100 vs
x=707 directly (bypassing the clip) at representative (n_r, n_i) shows a
~4% relative difference on THAT node's own value, but the node's
Gauss-Hermite weight there is tiny (``growth**2``-boosted contribution is
still only ~1.3% of the full quadrature sum, dominated by the mid-range
nodes), so the net effect on the table ENTRY at ``x_axis=40`` is of order
0.05-0.1%, not 4%. It is largest exactly where it matters least -- the
single hottest corner of one table's axis -- and does not meaningfully
change at smaller ``x_axis`` or at ``sigma=1.59``'s smaller growth factors.

Each table entry is the SAME lognormal-mode-integrated Gauss-Hermite
quadrature :meth:`JamOpticsTerm._mode_optics`'s default backend runs per
mode per step (identical 8-node quadrature, identical growth-factor
formula) -- evaluated once, at build time, over every (x, real RI, imag RI)
grid point rather than once per mode per step. A mode queried through this
table at its own median-radius size parameter is the lognormal integral at
HAM's fixed table sigma, not at the mode's own ``geom_std_dev`` (see
``ham_lut_optics_term.py`` for the fine/coarse table selection this implies
for a non-M7 population).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.optics.mie import mie_efficiencies_grid

# mo_ham_rad_data.f90:95-97 (Ndismax, Nnrmax, Nnimax). Arrays are declared
# 0:Ndismax etc (mo_ham_rad.f90:1304-1306), i.e. NDISMAX+1 points.
NDISMAX = 100
NNRMAX = 100
NNIMAX = 200
NX, NMR, NMI = NDISMAX + 1, NNRMAX + 1, NNIMAX + 1

# 8-node Gauss-Hermite quadrature, identical to
# ``optics_term.py``'s ``_GH_NODES``/``_GH_WEIGHTS`` -- same lognormal
# integral, evaluated here once per table point instead of once per mode
# per step.
_GH_NODES, _GH_WEIGHTS = (
    tuple(float(v) for v in arr) for arr in np.polynomial.hermite.hermgauss(8)
)


@dataclasses.dataclass(frozen=True)
class HamTableAxes:
    """One table's axis ranges and the lognormal sigma it is integrated at."""

    name: str
    sigma: float
    sw: bool          # SW table (stores ssa, g) vs LW (extinction only)
    x_min: float
    x_max: float
    nr_min: float
    nr_max: float
    ni_min: float
    ni_max: float


# M7 submodel column, mo_ham_rad_data.f90:420-431 (x0_min/x0_max) and
# :99-102 (nr_min/max, ni_min/max; shared by the fine/coarse pair of a given
# SW or LW table).
HAM_TABLE_AXES: dict[str, HamTableAxes] = {
    "sw_fine": HamTableAxes("sw_fine", 1.59, True,
                            0.001, 25.0, 1.33, 2.00, 1.0e-9, 1.00),
    "sw_coarse": HamTableAxes("sw_coarse", 2.00, True,
                              0.4, 40.0, 1.33, 2.00, 1.0e-9, 1.00),
    "lw_fine": HamTableAxes("lw_fine", 1.59, False,
                            5.0e-6, 3.0, 1.00, 3.00, 1.0e-9, 2.00),
    "lw_coarse": HamTableAxes("lw_coarse", 2.00, False,
                              0.0015, 4.0, 1.00, 3.00, 1.0e-9, 2.00),
}


@dataclasses.dataclass(frozen=True)
class HamRadLUT:
    """One tabulated table plus the affine grid mapping ``ham_rad_fitplus`` needs."""

    q_ext: jnp.ndarray         # (NX, NMR, NMI); Qext*x**2/(4*pi)
    ssa: jnp.ndarray | None    # (NX, NMR, NMI) or None for an LW table
    g: jnp.ndarray | None
    sw: bool
    # Both the linear bound and its log are stored independently (mirroring
    # mo_ham_rad_data.f90's own x0_min/log_x0_min pair, set once at init and
    # never reconstructed from each other) -- reconstructing x_min as
    # exp(log_x_min) at lookup time is off by ~1 ULP from the literal bound
    # (log then exp is not a perfect round-trip), which flips the in-range
    # test at exactly x==x_min and disagreed with the compiled reference at
    # that exact boundary (caught by hamrad_lookup's designed edge cases).
    x_min: float
    x_max: float
    log_x_min: float
    log_x_max: float
    nr_min: float
    nr_max: float
    inc_nr: float
    ni_min: float
    ni_max: float
    log_ni_min: float
    inc_ni: float


def _cache_dir() -> Path:
    """Local cache directory for built tables (same pattern as ``data/era5.py``)."""
    env = os.environ.get("JCM_HAM_MIE_CACHE")
    if env:
        return Path(env)
    scratch = os.environ.get("SCRATCH")
    if scratch and Path(scratch).is_dir():
        return Path(scratch) / "jcm-ham-mie-cache"
    return Path.home() / ".cache" / "jcm" / "ham_mie_tables"


# Bump when the table-building algorithm changes (axis layout, quadrature,
# normalisation) so a stale cache from before the change is never reused.
_CODE_VERSION = "1"


def _cache_key(axes: HamTableAxes) -> str:
    payload = "|".join(str(v) for v in (
        _CODE_VERSION, axes.name, axes.sigma, axes.sw,
        axes.x_min, axes.x_max, axes.nr_min, axes.nr_max,
        axes.ni_min, axes.ni_max, NX, NMR, NMI,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _lognormal_table(axes: HamTableAxes) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build one table (NumPy; this is the ~30 s-to-minutes-scale build)."""
    log_x = np.linspace(math.log(axes.x_min), math.log(axes.x_max), NX)
    nr_axis = np.linspace(axes.nr_min, axes.nr_max, NMR)
    log_ni_axis = np.linspace(math.log(axes.ni_min), math.log(axes.ni_max), NMI)
    nr_grid, ni_grid = np.meshgrid(nr_axis, np.exp(log_ni_axis), indexing="ij")
    ln_sigma = math.log(axes.sigma)

    qe = np.empty((NX, NMR, NMI))
    ssa = np.empty((NX, NMR, NMI)) if axes.sw else None
    g = np.empty((NX, NMR, NMI)) if axes.sw else None
    for i, lx in enumerate(log_x):
        x = math.exp(lx)
        sec = np.zeros((NMR, NMI))
        sec_scat = np.zeros((NMR, NMI))
        sec_gscat = np.zeros((NMR, NMI))
        for t_k, w_k in zip(_GH_NODES, _GH_WEIGHTS):
            growth = math.exp(math.sqrt(2.0) * ln_sigma * t_k)
            q_k, ssa_k, g_k = mie_efficiencies_grid(x * growth, nr_grid, ni_grid)
            wgt = (w_k / math.sqrt(math.pi)) * growth ** 2
            sec += wgt * q_k
            sec_scat += wgt * q_k * ssa_k
            sec_gscat += wgt * q_k * ssa_k * g_k
        qe[i] = sec * x * x / (4.0 * math.pi)
        if axes.sw:
            safe_sec = np.where(sec > 1.0e-300, sec, 1.0)
            ssa[i] = np.where(sec > 1.0e-300, sec_scat / safe_sec, 0.0)
            safe_scat = np.where(sec_scat > 1.0e-300, sec_scat, 1.0)
            g[i] = np.where(sec_scat > 1.0e-300, sec_gscat / safe_scat, 0.0)
    return qe, ssa, g


def _build_or_load(axes: HamTableAxes) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    path = _cache_dir() / f"{axes.name}_{_cache_key(axes)}.npz"
    if path.exists():
        with np.load(path) as z:
            qe = np.asarray(z["q_ext"])
            ssa = np.asarray(z["ssa"]) if "ssa" in z.files else None
            g = np.asarray(z["g"]) if "g" in z.files else None
        return qe, ssa, g
    qe, ssa, g = _lognormal_table(axes)
    payload = {"q_ext": qe}
    if ssa is not None:
        payload["ssa"] = ssa
        payload["g"] = g
    path.parent.mkdir(parents=True, exist_ok=True)
    # The name must still end in ".npz": np.savez silently appends that
    # suffix to any name that lacks it, which would otherwise write
    # "<tmp>.npz" instead of "<tmp>" and break the rename below.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)  # atomic within the cache dir; no partial file on crash
    return qe, ssa, g


def build_ham_mie_tables() -> dict[str, HamRadLUT]:
    """Build (or load from cache) all four HAM Mie tables."""
    out: dict[str, HamRadLUT] = {}
    for name, axes in HAM_TABLE_AXES.items():
        qe, ssa, g = _build_or_load(axes)
        out[name] = HamRadLUT(
            q_ext=jnp.asarray(qe, jnp.float32),
            ssa=jnp.asarray(ssa, jnp.float32) if ssa is not None else None,
            g=jnp.asarray(g, jnp.float32) if g is not None else None,
            sw=axes.sw,
            x_min=axes.x_min, x_max=axes.x_max,
            log_x_min=math.log(axes.x_min), log_x_max=math.log(axes.x_max),
            nr_min=axes.nr_min, nr_max=axes.nr_max,
            inc_nr=(axes.nr_max - axes.nr_min) / NNRMAX,
            ni_min=axes.ni_min, ni_max=axes.ni_max,
            log_ni_min=math.log(axes.ni_min),
            inc_ni=(math.log(axes.ni_max) - math.log(axes.ni_min)) / NNIMAX,
        )
    return out


_DEFAULT_TABLES: dict[str, HamRadLUT] | None = None


def default_ham_mie_tables() -> dict[str, HamRadLUT]:
    """Process-wide memoised default tables (disk-cached across processes)."""
    global _DEFAULT_TABLES
    if _DEFAULT_TABLES is None:
        _DEFAULT_TABLES = build_ham_mie_tables()
    return _DEFAULT_TABLES


def _nint_nonneg(v):
    """Fortran ``NINT`` (round-half-AWAY-from-zero) for an always-nonnegative

    argument, where it reduces to round-half-up: ``floor(v + 0.5)``. Every
    call site here feeds a fraction scaled from an in-range axis value, so
    ``v`` is never negative; ``jnp.round`` would instead give NumPy/JAX's
    round-half-to-EVEN and silently disagree with the compiled reference at
    exact half-steps (the designed test cases this module's tests use).
    """
    return jnp.floor(v + 0.5)


def ham_rad_fitplus(lut: HamRadLUT, x, mr, mi):
    """Faithful port of ``mo_ham_rad.f90::ham_rad_fitplus``'s nearest-

    neighbour branch (``loint=.FALSE.``, lines 1381-1415): returns
    ``(q_ext, ssa, g)`` -- ``ssa``/``g`` are exactly zero for an LW table (it
    carries none). Exactly zero on EVERY output, not clamped to the nearest
    table edge, when ``x``, ``mr`` or ``mi`` falls outside the table's range
    on any axis (lines 1387-1389, 1406-1410) -- a deliberate "no answer"
    convention, not an approximation of one.

    Inside range, the index is ``NINT`` of the affinely-scaled coordinate
    (lines 1391-1399), then clamped with ``MIN(Ndismax-1, MAX(0, ...))`` --
    note the native clamp's upper bound is one less than the array's actual
    top index (``Ndismax``, not ``Ndismax+1``'s last slot): the topmost
    table entry on every axis is populated at build time but never
    selected by this routine. That is the native routine's own behaviour,
    reproduced here faithfully rather than "fixed" (see the design doc).
    """
    # Natural dtype of the caller's inputs (float32 under jcm's default
    # physics precision); forcing float64 here would silently truncate
    # back down without jax_enable_x64, which the rest of this term's
    # arithmetic does not require either.
    x = jnp.asarray(x)
    mr = jnp.asarray(mr)
    mi = jnp.asarray(mi)

    # The literal bounds (lut.x_min/x_max/ni_min/ni_max), not exp(their own
    # stored logs): log-then-exp is not a bit-exact round-trip, and the
    # reconstructed bound can land ~1 ULP away from the literal one used to
    # build the table -- which flipped this exact in-range test at x==x_min
    # against the compiled reference (hamrad_lookup's designed edge cases).
    in_range = ((x >= lut.x_min) & (x <= lut.x_max)
                & (mr >= lut.nr_min) & (mr <= lut.nr_max)
                & (mi >= lut.ni_min) & (mi <= lut.ni_max))

    # Substitute before the logs so the discarded (out-of-range) branch
    # never evaluates log(0) or log(negative) -- both arms of the later
    # ``jnp.where`` are computed, so a non-finite value here would poison
    # the reverse pass even though the gate selects the other side.
    safe_x = jnp.where(in_range, x, lut.x_min)
    safe_mi = jnp.where(in_range, mi, lut.ni_min)

    ndis = _nint_nonneg((jnp.log(safe_x) - lut.log_x_min)
                         / (lut.log_x_max - lut.log_x_min) * NDISMAX)
    ndis = jnp.clip(ndis, 0, NDISMAX - 1).astype(jnp.int32)
    nnr = _nint_nonneg((mr - lut.nr_min) / lut.inc_nr)
    nnr = jnp.clip(nnr, 0, NNRMAX - 1).astype(jnp.int32)
    nni = _nint_nonneg((jnp.log(safe_mi) - lut.log_ni_min) / lut.inc_ni)
    nni = jnp.clip(nni, 0, NNIMAX - 1).astype(jnp.int32)

    q_ext = jnp.where(in_range, lut.q_ext[ndis, nnr, nni], 0.0)
    if lut.sw:
        ssa = jnp.where(in_range, lut.ssa[ndis, nnr, nni], 0.0)
        g = jnp.where(in_range, lut.g[ndis, nnr, nni], 0.0)
    else:
        ssa = jnp.zeros_like(q_ext)
        g = jnp.zeros_like(q_ext)
    return q_ext, ssa, g
