r"""HAM's Mie lookup tables (#1017): authentic files, or jcm's own build.

Two ways to get the four lookup tables ``mo_ham_rad.f90::ham_rad_initialize``
reads for the M7 aerosol submodel -- SW fine (lognormal
:math:`\sigma_g=1.59`), SW coarse (:math:`\sigma_g=2.0`), LW fine, LW coarse:

- :func:`load_ham_mie_tables` reads HAM's own authentic NetCDF files
  (``lut_optical_properties_M7.nc``, ``lut_optical_properties_lw_M7.nc``).
  Preferred whenever they are available (``HamLutOpticsTerm`` uses this
  path first; see its own docstring for the fallback order).
- :func:`default_ham_mie_tables`/:func:`build_ham_mie_tables` build an
  approximation with jcm's own Bohren-Huffman kernel (:mod:`mie`) on these
  same axes, for when the authentic files are not available (the no-data
  fallback) and for CI's lookup-arithmetic tests (data-free by design). See
  "The built tables are an approximation" below for the measured
  differences and why they are large enough to matter in places.

:func:`ham_rad_fitplus` is a faithful port of the native nearest-neighbour
lookup that reads either table (``mo_ham_rad.f90::ham_rad_fitplus``, lines
1264-1419) -- both expose the identical ``HamRadLUT`` layout, so the lookup
code is oblivious to which source filled it.

Axes (``mo_ham_rad_data.f90::ham_rad_data_initialize``, lines 95-102 and
420-431, the ``HAM_M7`` branch -- array index 1=SW fine, 2=SW coarse, 3=LW
fine, 4=LW coarse): size parameter ``x = 2*pi*r/lambda``, log-spaced, 101
points, table-specific range (fine SW ``[0.001, 25]``, coarse SW
``[0.4, 40]``, fine LW ``[5e-6, 3]``, coarse LW ``[0.0015, 4]``); real
refractive index, linear, 101 points (SW ``[1.33, 2.00]``, LW
``[1.00, 3.00]``); imaginary refractive index, log-spaced, 201 points (SW
``[1e-9, 1.00]``, LW ``[1e-9, 2.00]``). These ranges were verified directly
against the authentic files' own four corners (#1017 W1): the small-x/small-
nr corner agrees with a fresh Mie calculation to ~1e-6 relative on every
table, confirming both the axis bounds and jcm's reading of them are right.

On-disk layout and the axis order ``ham_rad_fitplus`` needs
-------------------------------------------------------------
Both authentic NetCDF files store each variable with dims ``(dis, ni, nr)``
= (101, 201, 101) and carry no coordinate variables for those dims (no
x/nr/ni axis *values* in the file -- only ``HAM_TABLE_AXES`` above gives
those, and they are a fixed property of the table's construction, not
something read per file). The in-memory layout this module (and
``ham_rad_fitplus``) uses is ``(dis, nr, ni)`` -- :func:`load_ham_mie_tables`
transposes by dimension NAME, which is robust to whichever order the file
actually stores on disk.

This differs from the Fortran reader's own in-memory layout, which is
``(nr, ni, dis)``: ``read_var_nf77_3d(file, "nr", "ni", "dis", var, lut, ierr)``
places the dimension named first (``"nr"``) first in the returned array
regardless of on-disk order (``mo_read_netcdf77.f90:261-368``, see its own
comment at lines 270-271: "The order of the dimensions in the netCDF file is
irrelevant"), and ``lut1_sigma`` etc. are declared and indexed accordingly
(``mo_ham_rad.f90:107-109`` declares ``lut1_sigma(0:Nnrmax,0:Nnimax,0:Ndismax)``;
``ham_rad_fitplus`` indexes ``plut1(Nnr,Nni,Ndis)`` at line 1401). Both are
just different axis orderings of the SAME conceptual 3-D table; jcm keeps its
own pre-existing ``(dis, nr, ni)`` convention (set when this module only
built the tables itself, before the authentic files were available) rather
than matching the Fortran's ``(nr, ni, dis)``, so nothing downstream of
either table source needs to change.

Normalisation (verified against the consuming Fortran arithmetic, not stated
as a file attribute or source comment): ``mo_ham_rad.f90`` multiplies the
raw table value by ``lambda**2`` to get the per-particle optical cross
section for BOTH SW (line 1070) and LW (line 1219) before the one shared
``tau = N_column * cross_section`` summation (line 1085 SW, the LW analogue
a few lines later) -- and ``Cext = Qext*pi*r**2 = Qext*x**2*lambda**2/(4*pi)``
with ``x = 2*pi*r/lambda``, so the raw table stores ``Qext*x**2/(4*pi)``, a
size-parameter-normalised, wavelength-independent extinction (SW) or
absorption (LW) efficiency. The authentic files' own ``units`` attribute on
these variables says ``cm+2 part-1`` (a dimensional per-particle cross
section) -- inconsistent with the dimensionless quantity the model code
actually reads and uses, and almost certainly a stale label surviving from
an earlier stage of HAM's offline table-building pipeline (not in this
source tree) rather than a correction this module needs to apply: the #1017
W1 comparison confirms jcm's own normalised build reproduces the authentic
file's numeric VALUES at the small-x corner to ~1e-6, which would not hold
under the units attribute's literal (dimensional) reading. SW tables
additionally store plain SSA and asymmetry factor ``g`` (no extra scaling,
``mo_ham_rad.f90:1016-1018,1113-1115``).

LW: absorption, not extinction
-------------------------------
The authentic LW files' own ``long_name`` attribute says it plainly: SW
``sigma_1``/``sigma_2`` are "extinction cross section fine/coarse mode",
while LW ``sigma_1_lw``/``sigma_2_lw`` are "absorption cross section fine/
coarse mode" (read directly off the files, #1017 W1). The Fortran code
applies the identical ``tau = N*sigma*lambda**2`` formula to both
(mo_ham_rad.f90:1070 SW, :1219 LW) with no separate scattering correction
for LW, which is exactly what ECHAM's non-scattering LW radiative transfer
wants: the LW optical depth IS the absorption optical depth there, not full
extinction. :func:`load_ham_mie_tables` reads the authentic LW files' values
as-is -- they already ARE the absorption efficiency HAM intends; no
transform is applied or needed. :func:`build_ham_mie_tables`'s own lognormal
quadrature instead computes the ABSORPTION integral directly for its LW
tables (``sec - sec_scat`` -- exact by linearity of the quadrature sum, the
lognormal integral of the per-node absorption efficiency
``q_k*(1 - ssa_k)``, not an approximation) since #1017 W1 found an earlier
version of this module built LW as extinction instead, which overstated LW
aerosol optical depth by orders of magnitude wherever scattering actually
matters (quantified spot check, #1017 W1: at x=3, real RI=2, imag RI=1e-9,
the built-as-extinction value is 2.73 while the authentic file holds
4.6e-8; recomputing the SAME Mie calculation as absorption instead
reproduces the file to five figures: 3.7519e-13 vs the file's 3.7526e-13).

The built tables are an approximation
---------------------------------------
:func:`build_ham_mie_tables` uses jcm's own Bohren-Huffman kernel
(:func:`mie.mie_efficiencies_grid`) and an 8-node Gauss-Hermite lognormal
quadrature -- the SAME integral :class:`~.optics_term.JamOpticsTerm`'s
default backend runs per mode per step, evaluated once per table point
instead. It is NOT a literal port of whatever offline tool built HAM's own
authentic tables (that tool is not in this source tree), so even after the
LW-as-absorption fix it disagrees with the authentic tables at a level that
matters for some fields/table corners -- see
``ham_mie_tables_test.py::test_built_tables_vs_authentic_measured_tolerance``
for the measured median/p99/max differences per table and field, and
``HamLutOpticsTerm``'s own docstring for how a run picks between the two.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import xarray as xr
from flax import struct

from jcm.physics.aerosol.jam.optics.mie import mie_efficiencies_grid

# mo_ham_rad_data.f90:95-97 (Ndismax, Nnrmax, Nnimax). Arrays are declared
# 0:Ndismax etc (mo_ham_rad.f90:1304-1306), i.e. NDISMAX+1 points.
NDISMAX = 100
NNRMAX = 100
NNIMAX = 200
NX, NMR, NMI = NDISMAX + 1, NNRMAX + 1, NNIMAX + 1

# The two files ``mo_ham_rad.f90::ham_rad_initialize`` reads (there generically
# as "lut_optical_properties[_lw].nc"; HAMMOZ ships per-submodel-named copies
# on disk, and the M7 config uses these).
_SW_FILE = "lut_optical_properties_M7.nc"
_LW_FILE = "lut_optical_properties_lw_M7.nc"

# 8-node Gauss-Hermite quadrature, identical to ``optics_term.py``'s
# ``_GH_NODES``/``_GH_WEIGHTS`` -- same lognormal integral, evaluated here
# once per table point instead of once per mode per step.
_GH_NODES, _GH_WEIGHTS = (
    tuple(float(v) for v in arr) for arr in np.polynomial.hermite.hermgauss(8)
)


@dataclasses.dataclass(frozen=True)
class HamTableAxes:
    """One table's axis ranges and the lognormal sigma it is integrated at."""

    name: str
    sigma: float
    sw: bool          # SW table (stores ssa, g) vs LW (absorption only)
    x_min: float = struct.field(pytree_node=False)
    x_max: float = struct.field(pytree_node=False)
    nr_min: float = struct.field(pytree_node=False)
    nr_max: float = struct.field(pytree_node=False)
    ni_min: float = struct.field(pytree_node=False)
    ni_max: float = struct.field(pytree_node=False)


# M7 submodel column, mo_ham_rad_data.f90:420-431 (x0_min/x0_max, HAM_M7
# branch) and :99-102 (nr_min/max, ni_min/max; shared by the fine/coarse pair
# of a given SW or LW table). Verified byte-for-byte against the source
# (#1017 W1): these are literal Fortran constants, not derived from the data
# files (which carry no axis-value coordinates at all).
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


@struct.dataclass
class HamRadLUT:
    """One tabulated table plus the affine grid mapping ``ham_rad_fitplus`` needs.

    A pytree: the three tables are leaves, so a term holds them as module
    data and passes them into the compiled step; the axis constants are
    static.
    """

    q_ext: jnp.ndarray         # (NX, NMR, NMI); Qext*x**2/(4*pi) (SW) or Qabs*x**2/(4*pi) (LW)
    ssa: jnp.ndarray | None    # (NX, NMR, NMI) or None for an LW table
    g: jnp.ndarray | None
    sw: bool = struct.field(pytree_node=False)
    # Both the linear bound and its log are stored independently (mirroring
    # mo_ham_rad_data.f90's own x0_min/log_x0_min pair, set once at init and
    # never reconstructed from each other) -- reconstructing x_min as
    # exp(log_x_min) at lookup time is off by ~1 ULP from the literal bound
    # (log then exp is not a perfect round-trip), which flips the in-range
    # test at exactly x==x_min and disagreed with the compiled reference at
    # that exact boundary (caught by hamrad_lookup's designed edge cases).
    x_min: float = struct.field(pytree_node=False)
    x_max: float = struct.field(pytree_node=False)
    log_x_min: float = struct.field(pytree_node=False)
    log_x_max: float = struct.field(pytree_node=False)
    nr_min: float = struct.field(pytree_node=False)
    nr_max: float = struct.field(pytree_node=False)
    inc_nr: float = struct.field(pytree_node=False)
    ni_min: float = struct.field(pytree_node=False)
    ni_max: float = struct.field(pytree_node=False)
    log_ni_min: float = struct.field(pytree_node=False)
    inc_ni: float = struct.field(pytree_node=False)


def _resolve_directory(directory: str | os.PathLike | None) -> Path:
    if directory is None:
        directory = os.environ.get("HAM_INPUT_DIR")
        if directory is None:
            raise FileNotFoundError(
                "HAM's authentic Mie LUT files need a directory: pass "
                "`directory` explicitly or set the HAM_INPUT_DIR environment "
                "variable (the same variable the Kazil/GCR work uses) to the "
                f"directory holding {_SW_FILE} and {_LW_FILE}.")
    return Path(directory)


def _read_table_var(path: Path, var: str) -> np.ndarray:
    """One variable, transposed by dimension NAME to this module's (dis, nr, ni).

    Robust to whichever order the file actually stores its dims in (the
    files hold ``(dis, ni, nr)``, see the module docstring) -- ``xarray``'s
    named transpose, not a positional one, is what makes that not matter.
    """
    with xr.open_dataset(path) as ds:
        if var not in ds.variables:
            raise KeyError(f"{path} has no variable {var!r} (found {sorted(ds.variables)})")
        arr = ds[var].transpose("dis", "nr", "ni").to_numpy()
    return np.asarray(arr, dtype=np.float32)


# (sw/lw file, q_ext variable, ssa variable, g variable) per table; ssa/g are
# None for the LW tables, which carry no SW scattering quantities at all
# (their own ham_rad_fitplus call, mo_ham_rad.f90:1199-1207, passes no
# plut2/plut3).
_TABLE_VARS: dict[str, tuple[str, str, str | None, str | None]] = {
    "sw_fine": (_SW_FILE, "sigma_1", "omega_1", "asym_1"),
    "sw_coarse": (_SW_FILE, "sigma_2", "omega_2", "asym_2"),
    "lw_fine": (_LW_FILE, "sigma_1_lw", None, None),
    "lw_coarse": (_LW_FILE, "sigma_2_lw", None, None),
}


def load_ham_mie_tables(directory: str | os.PathLike | None = None) -> dict[str, HamRadLUT]:
    """Load HAM's own four Mie LUTs from its authentic NetCDF files.

    See the module docstring for the axis layout, the on-disk-to-in-memory
    transpose, the shared Qext/Qabs normalisation and why the LW tables are
    read as-is despite their stale ``cm+2 part-1`` units attribute.

    Parameters
    ----------
    directory
        Directory holding ``lut_optical_properties_M7.nc`` and
        ``lut_optical_properties_lw_M7.nc``. Defaults to the
        ``HAM_INPUT_DIR`` environment variable; raises ``FileNotFoundError``
        naming it when neither is given, or naming the missing file when
        the directory does not hold one of the two.

    """
    directory = _resolve_directory(directory)
    paths = {_SW_FILE: directory / _SW_FILE, _LW_FILE: directory / _LW_FILE}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"HAM Mie LUT file {name!r} not found at {path} (set HAM_INPUT_DIR "
                "to its directory, or pass `directory` explicitly).")

    out: dict[str, HamRadLUT] = {}
    for name, (file_name, q_var, ssa_var, g_var) in _TABLE_VARS.items():
        axes = HAM_TABLE_AXES[name]
        path = paths[file_name]
        q_ext = _read_table_var(path, q_var)
        expected_shape = (NX, NMR, NMI)
        if q_ext.shape != expected_shape:
            # The authentic file's axes disagreeing with HAM_TABLE_AXES is a
            # STOP per #1017 W1's task brief -- surfaced as a loud, specific
            # error rather than a silent reshape/truncation.
            raise ValueError(
                f"{path}::{q_var} has shape {q_ext.shape}, expected {expected_shape} "
                f"= (dis={NX}, nr={NMR}, ni={NMI}) from HAM_TABLE_AXES[{name!r}]. "
                "The authentic table's axes do not match the Fortran-verified "
                "constants -- do not silently reshape; see ham_mie_tables.py's "
                "module docstring.")
        ssa = _read_table_var(path, ssa_var) if ssa_var else None
        g = _read_table_var(path, g_var) if g_var else None
        out[name] = HamRadLUT(
            q_ext=q_ext, ssa=ssa, g=g, sw=axes.sw,
            x_min=axes.x_min, x_max=axes.x_max,
            log_x_min=math.log(axes.x_min), log_x_max=math.log(axes.x_max),
            nr_min=axes.nr_min, nr_max=axes.nr_max,
            inc_nr=(axes.nr_max - axes.nr_min) / NNRMAX,
            ni_min=axes.ni_min, ni_max=axes.ni_max,
            log_ni_min=math.log(axes.ni_min),
            inc_ni=(math.log(axes.ni_max) - math.log(axes.ni_min)) / NNIMAX,
        )
    return out


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
# "2": LW now builds absorption (sec - sec_scat), not extinction (sec) --
# #1017 W1 found the latter overstated LW aerosol optical depth by orders of
# magnitude against HAM's authentic tables wherever scattering dominates.
_CODE_VERSION = "2"


def _cache_key(axes: HamTableAxes) -> str:
    payload = "|".join(str(v) for v in (
        _CODE_VERSION, axes.name, axes.sigma, axes.sw,
        axes.x_min, axes.x_max, axes.nr_min, axes.nr_max,
        axes.ni_min, axes.ni_max, NX, NMR, NMI,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _lognormal_table(axes: HamTableAxes) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build one table (NumPy; this is the ~30 s-to-minutes-scale build).

    SW: extinction efficiency, SSA and asymmetry factor. LW: ABSORPTION
    efficiency (``sec - sec_scat``, exact by linearity of the quadrature
    sum -- see the module docstring's "LW: absorption, not extinction").
    """
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
        if axes.sw:
            qe[i] = sec * x * x / (4.0 * math.pi)
            safe_sec = np.where(sec > 1.0e-300, sec, 1.0)
            ssa[i] = np.where(sec > 1.0e-300, sec_scat / safe_sec, 0.0)
            safe_scat = np.where(sec_scat > 1.0e-300, sec_scat, 1.0)
            g[i] = np.where(sec_scat > 1.0e-300, sec_gscat / safe_scat, 0.0)
        else:
            qe[i] = (sec - sec_scat) * x * x / (4.0 * math.pi)
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
    """Build (or load from cache) all four HAM Mie tables with jcm's own kernel.

    The no-data fallback :class:`HamLutOpticsTerm` uses when HAM's authentic
    files are not reachable -- see the module docstring's "The built tables
    are an approximation" for the measured differences against the real
    thing.
    """
    out: dict[str, HamRadLUT] = {}
    for name, axes in HAM_TABLE_AXES.items():
        qe, ssa, g = _build_or_load(axes)
        out[name] = HamRadLUT(
            # NumPy, never jax: the memo below is process-global, and a jax
            # array created while a caller is being traced would leak that
            # trace's tracer into every later call.
            q_ext=np.asarray(qe, np.float32),
            ssa=np.asarray(ssa, np.float32) if ssa is not None else None,
            g=np.asarray(g, np.float32) if g is not None else None,
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
    """Process-wide memoised default built tables (disk-cached across processes)."""
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
