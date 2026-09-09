# Dynamical core

**What we do.** Dynamics run behind a backend-agnostic ``DynamicalCore``
protocol (``jcm/dycore/base.py::DynamicalCore``) that owns one dynamics step,
native tendency conversion, hyperdiffusion, spectral/remap filters and vertical
remapping. Two backends implement it. The shipped default is a spectral-transform
core wrapping the external ``dinosaur`` package
(``jcm/dycore/dinosaur/dycore.py::DinosaurDycore``): primitive equations on a
Gaussian grid, IMEX-RK (SIL3) time integration, on hybrid σ–p or pure-σ vertical
coordinates. The optional backend is the pySES CAM-SE spectral-element core on a
cubed sphere (``jcm/dycore/pyses/dycore.py::PysesCamSEDycore``,
``pip install jcm[pyses]``, registry name ``pyses_cam_se``), coupled to the
column physics through a pg2 finite-volume physics grid (see
{doc}`../design/pyses_cam_se_dycore`).

Tracer transport is **semi-Lagrangian only** — dinosaur's departure-point
transport with a Bermejo–Staniforth quasi-monotone limiter. Every jcm extra
tracer (aerosol mass/number, gases, cloud condensate) rides as a *nodal* tracer
while ``specific_humidity`` stays modal for the implicit q↔Tᵥ coupling.
Horizontal hyperdiffusion is configured by ``jcm/diffusion.py::DiffusionFilter``:
for hybrid L47/L95 grids the resolution-aware ``DiffusionFilter.auto`` selects the
ECHAM ``lmidatm`` level-dependent order profile (∇² near the model top grading to
∇⁶/∇⁸ below, base timescale from ``setdyn.f90``'s ``dampth``); any other grid gets
the uniform SPEEDY ∇²/∇⁴ profile (``DiffusionFilter.default``), with a warning for
unrecognised hybrid grids. An optional upper sponge (Rayleigh drag) is enabled via
the ``run`` group (``jcm/config/run/longrun.yaml``: ``levels: 10``,
``target_T_K: 250``, ``enspodi: 2``). Resolutions T21–T425 are supported.

**What ECHAM/CAM does.** ECHAM6 (Stevens et al. 2013, *JAMES*) is a spectral-
transform core with leapfrog + semi-implicit correction and spectral (∇²ⁿ)
hyperdiffusion whose order varies by level under ``lmidatm``
(``mo_hdiff.f90::sudif``, timescale in ``setdyn.f90``). CAM-SE / CAM-FV3 (HOMME
spectral element; Lauritzen et al. 2018) use explicit RK dynamics with tensor
hyperviscosity and a ``nu_top`` Laplacian sponge on a cubed sphere, with a
separate finite-volume physics grid (pg2; Hannah et al. 2021). Both use hybrid
σ–p vertical coordinates.

**Why we differ.**
- `compute` — semi-Lagrangian is the *only* tracer transport; the Eulerian
  spectral-transform tracer path was removed because it rang negative on sharp
  emission sources and NaN'd the aerosol microphysics (#521). This is a
  positivity/compute-motivated choice with no fallback. See
  {doc}`../design/dinosaur_sl_jam_configuration`.
- `compute` — the dinosaur backend integrates with IMEX-RK SIL3 rather than
  ECHAM's leapfrog + semi-implicit; the pySES backend runs float64 dynamics with
  a float32 physics seam (the SE core needs x64).

**Status & known limitations.** SPEEDY physics is generalised to arbitrary
vertical level counts; high-``nlev`` / high-truncation configurations need a
resolution-aware timestep to stay stable (see
{doc}`../design/speedy_variable_levels`). The ECHAM ``lmidatm`` hyperdiffusion
profiles exist only for L47/L95; other hybrid grids fall back to the uniform
SPEEDY profile with a warning (#579). The pySES backend has open production gaps:
per-column longitudes are collapsed to a reference longitude in the physics
``cache_coords``, full-float32 ECHAM physics is not yet dtype-stable (ECHAM
requires ``physics_dtype=float64`` there), and multi-GPU sharding of the element
and physics-column axes is unreconciled (see
{doc}`../design/pyses_cam_se_dycore`).

**Code pointers.**
- ``jcm/dycore/base.py`` — ``DynamicalCore`` protocol (``initial_state``,
  ``step``, ``to_physics_state``, ``Predictions``).
- ``jcm/dycore/dinosaur/dycore.py`` — ``DinosaurDycore``,
  ``semi_lagrangian_available`` / ``_require_semi_lagrangian`` (the #521 removal
  guard), transport build (nodal tracers), filter build.
- ``jcm/dycore/pyses/dycore.py`` — ``PysesCamSEDycore``.
- ``jcm/diffusion.py`` — ``DiffusionFilter`` and its ``auto`` / ``echam_lmidatm``
  / ``default`` constructors; ``_ECHAM_LMIDATM_ORDERS``.
- ``jcm/config/run/longrun.yaml`` — production sponge.

**Validation evidence.** ``jcm/dycore/dinosaur/dycore_test.py``,
``sharding_test.py``, ``state_bridge_test.py``;
``jcm/dycore/pyses/pyses_dycore_test.py``, ``physics_grid_test.py``,
``rrtmgp_x64_test.py``; ``jcm/dycore/base_test.py``; ``jcm/diffusion_test.py``.
Design references: {doc}`../design/pyses_cam_se_dycore`,
{doc}`../design/speedy_variable_levels`,
{doc}`../design/dinosaur_sl_jam_configuration`.
