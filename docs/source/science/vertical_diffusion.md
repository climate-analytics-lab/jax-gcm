# Vertical diffusion (boundary-layer turbulence)

**What we do.** Boundary-layer turbulent mixing uses a TKE-based (total-turbulent-
energy) closure ported from ICON/ECHAM ``vdiff``, wrapped as the composable term
``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion.py::TteTkeVerticalDiffusion``.
It carries a prognostic TKE (and θᵥ-variance) budget — shear + buoyancy production
− dissipation + transport — diagnoses exchange coefficients from a mixing length
and √TKE, and solves the diffusion implicitly (backward Euler) with a tridiagonal
Thomas solve (``matrix_solver.py``, following ``mo_vdiff_solver.f90``). The term
owns the whole turbulent column ECHAM-style: per-tile surface exchange velocities
enter the implicit solve as the bottom-row Robin boundary condition for u/v/T/qᵥ,
and the delivered surface fluxes are diagnosed from the implicit solution (the
``pev_vdiff`` identity — reported equals delivered by construction). Surface-layer
exchange coefficients use a faithful Louis (1979, unstable) / Mauritsen (2007,
stable) port (``surface_layer.py``, ``mo_turbulence_diag::sfc_exchange_coeff``).
**K floors** hold minimum diffusivities: exchange coefficients clip to ECHAM's
free-troposphere background, mixing length floors at 1 m, friction velocity at
0.01 m/s, TKE at the ECHAM lower bound. Tracer diffusion is a separate generic
term ``tracer_diffusion.py::TracerVerticalDiffusion`` that mixes an explicit
tracer list with the ``kh`` profile the TTE-TKE term publishes, via one batched
unconditionally-stable backward-Euler solve that conserves each tracer's column
mass exactly.

**What ECHAM/CAM does.** ICON/ECHAM6 ``vdiff`` (Brinkop & Roeckner 1995;
Mauritsen et al. 2007 total-turbulent-energy closure) — prognostic TKE,
Louis/Mauritsen surface-layer stability functions, and an implicit column solve
folding the surface exchange into a single tridiagonal (``mo_vdiff_solver.f90``,
``mo_turbulence_diag.f90``). ECHAM diffuses every tracer with the heat exchange
coefficient ``cfh`` (the ``pxtte`` update in ``mo_vdiff_solver``); CAM diffuses
all constituents likewise.

**Why we differ.**
- `science` — the surface-layer exchange uses a Louis (1979) / Mauritsen (2007)
  form matching ECHAM/ICON to order of magnitude across the Richardson-number
  range, not a bit-exact reproduction of every ECHAM branch.
- `compute` — the TTE-TKE column solve covers only its fixed variable block
  (u, v, T, qᵥ, qc, qi, TKE, θᵥ variance), so aerosol/gas tracers are mixed by the
  separate ``TracerVerticalDiffusion`` term rather than in the same tridiagonal
  ; its boundaries are zero-flux (surface exchange is dry deposition's job).

**Status & known limitations.** ``TracerVerticalDiffusion`` is a no-op on the
first step — it reads the previous step's ``kh`` carry, which is seeded to
zero on step 0 (zero exchange coefficient, zero tendency)
and reads the previous step's ``kh`` carry, because vdiff runs after the aerosol
block in the ECHAM ordering.

**Code pointers.**
- ``jcm/physics/vertical_diffusion/tte_tke/`` — ``vertical_diffusion.py``
  (``TteTkeVerticalDiffusion``, ``vertical_diffusion_column``),
  ``turbulence_coefficients.py`` (``compute_exchange_coefficients`` and the K
  floor), ``matrix_solver.py`` (``setup_matrix_system``,
  ``solve_tridiagonal_system``, ``diagnose_surface_fluxes``),
  ``surface_layer.py`` (``compute_surface_exchange_coefficients_echam_louis``),
  ``tke_budget.py``, ``vertical_diffusion_types.py`` (the floors).
- ``jcm/physics/vertical_diffusion/tracer_diffusion.py`` —
  ``TracerVerticalDiffusion``, ``diffuse_tracers_implicit``.

## Ten-metre wind diagnostic

**What we do.** The term publishes a grid-mean 10 m wind speed,
``VerticalDiffusionData.wind_10m``, alongside the exchange coefficients. It is
the surface-layer profile evaluated at 10 m rather than an interpolation
between levels: with ``bn = ln(z₁/z₀ₘ)`` the neutral profile factor and
``bm = bn·√(CMₙ|U| / CM|U|)`` its stability-corrected counterpart,

```
zrat = 10/z₁
cbn  = ln(1 + (e^bn − 1)·zrat)
red  = [cbn + (stable: −(bn − bm)·zrat | unstable: −ln(1 + (e^(bn−bm) − 1)·zrat))] / bm
```

and ``|U(10 m)| = red·|U(z₁)|``, area-weighted over the surface tiles. The
reduction is computed **inside** each surface-layer scheme's ``lax.cond``
branch, from that scheme's own neutral drag: the two schemes differ in
roughness (state-carried vs a hard-coded table), in the bound on ``z/z₀`` and
in whether the wind is ``zepdu2``-floored, and both branches build their
momentum drag from the same helper the reduction uses, so the pair cannot
drift. The profile factor uses the ``zepdu2``-floored speed the coefficients
themselves were built from (``max(|U|, 1 m/s)``), and the resulting reduction
multiplies the true wind.

**What ECHAM/CAM does.** ECHAM5 ``vdiff.f90`` / ICON
``mo_surface_diag::nsurf_diag`` compute the 10 m wind by exactly this
construction, from the ``pbn``/``pbm`` profile factors ``mo_turbulence_diag``
exports for the purpose; ``zepdu2 = 1 m²/s²`` is ECHAM's calm-wind floor on the
bulk Richardson number and the drag.

**Why we differ.** We do not. The stable/unstable branch is selected by
``CM|U| < CMₙ|U|`` rather than by the sign of ``Ri``, which is equivalent for
both schemes here (their stability factors are <1 stable, ≥1 unstable, and both
equal 1 — with equal stable and unstable expressions — at ``Ri = 0``) and keeps
the Richardson number inside the coefficient solve.

**Status & known limitations.** The diagnostic is a grid-mean over the tiles;
per-tile 10 m winds are computed internally but not published. The Businger-Dyer
branch's neutral reference uses that scheme's own hard-coded von Kármán constant
rather than the live ``jcm.constants`` value, so a ``set_constants`` override
changes the drag and its neutral reference together.

**Code pointers.**
- ``jcm/physics/vertical_diffusion/tte_tke/surface_layer.py`` —
  ``wind_10m_reduction``, ``echam_louis_neutral_drag``.
- ``jcm/physics/vertical_diffusion/tte_tke/turbulence_coefficients.py`` —
  ``businger_dyer_neutral_drag``, ``compute_turbulence_diagnostics``.
- ``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion_types.py`` —
  ``VerticalDiffusionData``.
- ``jcm/physics/surface/echam/turbulent_fluxes.py`` —
  ``compute_surface_diagnostics`` (the reported 10 m wind and its components).

**Validation evidence.**
``jcm/physics/vertical_diffusion/tte_tke/surface_layer_test.py`` — the neutral
limit against a hand-computed log profile (0.912/0.904/0.894 for
z₀ = 5e-5/1.5e-4/5e-4 m at z₁ = 32.6 m), monotonicity in stability, continuity
across the branch, and that each scheme uses its own neutral drag.

**Validation evidence.**
``jcm/physics/vertical_diffusion/tte_tke/`` test suite —
``vertical_diffusion_test.py``, ``surface_layer_test.py``,
``boundary_layer_cases_test.py``, ``scm_boundary_layer_cases_test.py``;
``tracer_diffusion_test.py`` (column-mass conservation).
