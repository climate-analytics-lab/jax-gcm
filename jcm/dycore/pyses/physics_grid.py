"""The pg2 finite-volume physics grid and the GLL ↔ column bridge.

Why a separate physics grid at all
----------------------------------
Evaluating column physics directly on the GLL quadrature nodes imprints the
element structure onto the forcing (GLL nodes cluster at element edges and
carry very unequal quadrature weights), which CAM-SE inherits as grid-scale
noise. Following Hannah et al. (2021, JAMES, doi:10.1029/2020MS002419) —
and the developer prototype — physics runs instead on a quasi-equal-area
finite-volume grid of ``nf × nf`` sub-cells per element (``pg2``: nf = 2).

pyses 0.1.3a2 **ships** the element-local remap machinery
(:mod:`pyses.dynamical_cores.finite_volume_grid`): ``init_fv_grid`` builds
the reference operators + per-element metric, ``gll_to_fv`` is the
area-weighted GLL→cell average and ``fv_to_gll`` its density-weighted
pseudo-inverse satisfying the paper's R1 identity — the FV→GLL→FV round trip
is the identity to machine precision. This module therefore only *wraps*
that machinery into the layout jcm physics wants:

* ``gather_3d``: ``(E, npt, npt, nlev)`` GLL field → ``(nlev, 1, ncol)``
  physics field (level axis first, columns flattened element-major so that
  ``ncol = num_elem * nf * nf``);
* ``scatter_3d``: the inverse, in float64 (physics tendencies come back
  float32; the cast up happens here, at the dynamics boundary);
* ``dss``: direct stiffness summation (pyses ``project_scalar_3d``) restoring
  C0 continuity after a scatter — ``fv_to_gll`` is element-local and hence
  element-discontinuous, and CAM-SE's explicit terms assume C0 inputs.

Cell coordinates
----------------
``init_fv_grid`` also area-averages the GLL (lat, lon) coordinates onto the
cells, but averaging *longitude* directly is wrong across the 0/2π seam (a
cell straddling the seam averages ~π instead of ~0). We instead average the
unit Cartesian position over each cell with the same ``gll_to_fv`` operator
and convert back — seam-safe and pole-safe (the pg2 cells never average to
the exact sphere centre).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jcm.dycore.pyses._pyses import require_pyses


class FVPhysicsGrid:
    """pg-``nf`` finite-volume physics grid over a pyses SE horizontal grid.

    Args:
        h_grid: pyses ``SpectralElementGrid`` struct (from
            ``init_quasi_uniform_grid_elem_local``).
        dims: pyses grid-dims frozendict (``npt``, ``num_elem``, ...).
        nf: FV sub-cells per element edge (default 2 → pg2).

    Attributes:
        num_cols: Total physics columns ``num_elem * nf * nf``.
        latitudes / longitudes: Per-column cell-centre coordinates
            (radians, numpy float64, shape ``(ncol,)``; longitudes in
            ``[0, 2π)``), computed via the seam-safe Cartesian average.

    """

    def __init__(self, h_grid, dims, nf: int = 2):
        """Build the FV grid struct and seam-safe cell coordinates."""
        backend = require_pyses()
        from pyses.dynamical_cores.finite_volume_grid import init_fv_grid

        self.h_grid = h_grid
        self.dims = dims
        self.nf = int(nf)
        self.num_elem = int(dims["num_elem"])
        self.npt = int(dims["npt"])
        self.num_cols = self.num_elem * self.nf * self.nf
        self.fv_grid = init_fv_grid(h_grid, dims, nf=self.nf)
        # Multi-device: pyses shards the element axis across devices under an
        # *explicit* mesh (jax.set_mesh in its backend). Columns are flattened
        # element-major here (ncol = E·nf·nf), so the element sharding and a
        # block sharding of the column axis are the SAME partitioning — but
        # explicit-mode reshapes that merge/split the sharded axis must state
        # the output sharding (see _reshape). Single device: plain reshapes.
        self._do_sharding = bool(getattr(backend, "do_sharding", False))
        self._elem_axis = getattr(backend, "_elem_axis_name", "f")

        # Seam-safe cell centres: area-average the unit Cartesian position
        # of every GLL node over each FV cell, then re-normalise onto the
        # sphere. (Averaging lon directly wraps wrongly across 0/2π.)
        gll = np.asarray(h_grid["physical_coords"], dtype=np.float64)
        lat, lon = gll[..., 0], gll[..., 1]
        xyz = np.stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
            axis=-1,
        )
        xyz_fv = np.asarray(self.gather_cells(jnp.asarray(xyz)))  # (E, nf, nf, 3)
        xyz_fv = xyz_fv / np.linalg.norm(xyz_fv, axis=-1, keepdims=True)
        flat = xyz_fv.reshape(self.num_cols, 3)
        self.latitudes = np.arcsin(np.clip(flat[:, 2], -1.0, 1.0))
        self.longitudes = np.mod(np.arctan2(flat[:, 1], flat[:, 0]), 2.0 * np.pi)

    # ------------------------------------------------------------------
    # Raw cell-layout remaps (element-shaped FV fields)
    # ------------------------------------------------------------------

    def gather_cells(self, field):
        """GLL ``(E, npt, npt[, K])`` → FV cells ``(E, nf, nf[, K])`` (area average)."""
        from pyses.dynamical_cores.finite_volume_grid import gll_to_fv

        return gll_to_fv(field, self.fv_grid)

    def scatter_cells(self, field_fv):
        """FV cells ``(E, nf, nf[, K])`` → GLL ``(E, npt, npt[, K])``.

        Element-local and hence element-*discontinuous*; run :meth:`dss` on
        the result before it enters the dynamics. ``gather_cells ∘
        scatter_cells`` is the identity (Hannah et al. R1), which is what
        keeps the physics→dynamics→physics tendency round trip from
        smearing element means.
        """
        from pyses.dynamical_cores.finite_volume_grid import fv_to_gll

        return fv_to_gll(field_fv, self.fv_grid)

    # ------------------------------------------------------------------
    # Physics-layout gathers/scatters ((nlev, 1, ncol) on the physics side)
    # ------------------------------------------------------------------

    def _reshape(self, x, shape, sharded_axes):
        """Reshape, stating the element sharding of the output when active.

        ``sharded_axes`` marks which OUTPUT axes carry the (block) element
        sharding — always the axis that contains the element dimension,
        since columns are element-major. Explicit-mesh JAX requires
        merge/split reshapes of a sharded axis to state their output
        sharding; on a single device this is a plain reshape.
        """
        if not self._do_sharding:
            return x.reshape(shape)
        import jax
        from jax.sharding import PartitionSpec

        spec = PartitionSpec(
            *[self._elem_axis if s else None for s in sharded_axes])
        return jax.lax.reshape(x, shape, out_sharding=spec)

    def gather_3d(self, field):
        """GLL ``(E, npt, npt, nlev)`` → physics ``(nlev, 1, ncol)``."""
        fv = self.gather_cells(field)                       # (E, nf, nf, nlev)
        cols = self._reshape(fv, (self.num_cols, fv.shape[-1]),
                             (True, False))                 # (ncol, nlev)
        return jnp.moveaxis(cols, 0, 1)[:, None, :]         # (nlev, 1, ncol)

    def gather_2d(self, field):
        """GLL ``(E, npt, npt)`` → physics ``(1, ncol)``."""
        fv = self.gather_cells(field)                       # (E, nf, nf)
        return self._reshape(fv, (1, self.num_cols), (False, True))

    def scatter_3d(self, cols):
        """Physics ``(nlev, 1, ncol)`` → GLL ``(E, npt, npt, nlev)`` in float64.

        The float32→float64 cast happens here — this is the single seam where
        physics-precision arrays cross back into the float64 dynamics. The
        result is element-discontinuous; apply :meth:`dss` afterwards.
        """
        nlev = cols.shape[0]
        flat = self._reshape(jnp.asarray(cols, dtype=jnp.float64),
                             (nlev, self.num_cols), (False, True))
        fv = self._reshape(jnp.moveaxis(flat, 0, 1),
                           (self.num_elem, self.nf, self.nf, nlev),
                           (True, False, False, False))
        return self.scatter_cells(fv)

    def scatter_2d(self, cols):
        """Physics ``(1, ncol)`` → GLL ``(E, npt, npt)`` in float64 (discontinuous)."""
        fv = self._reshape(jnp.asarray(cols, dtype=jnp.float64),
                           (self.num_elem, self.nf, self.nf),
                           (True, False, False))
        return self.scatter_cells(fv)

    # ------------------------------------------------------------------
    # Direct stiffness summation
    # ------------------------------------------------------------------

    def dss(self, field):
        """Project a GLL field onto the C0-continuous subspace (DSS).

        Accepts ``(E, npt, npt)`` or ``(E, npt, npt, K)`` (K vmapped).
        Idempotent on already-continuous fields; this is the same projection
        pyses applies to its own dynamics tendencies (``project_dynamics``).
        """
        from pyses.dynamical_cores.model_state import project_scalar_3d
        from pyses.operations_2d.local_assembly import project_scalar

        if field.ndim == 3:
            return project_scalar(field, self.h_grid, self.dims)
        return project_scalar_3d(field, self.h_grid, self.dims)
