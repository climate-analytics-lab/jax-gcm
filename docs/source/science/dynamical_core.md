# Dynamical core

**What we do.** Dynamics run behind a backend-agnostic ``DynamicalCore``
protocol (``jcm/dycore/base.py::DynamicalCore``) that owns one dynamics step,
native tendency conversion, hyperdiffusion, spectral/remap filters and vertical
remapping. Two backends implement it. The shipped default is a spectral-transform
core wrapping the external ``dinosaur`` package
(``jcm/dycore/dinosaur/dycore.py::DinosaurDycore``): primitive equations on a
Gaussian grid, integrated with the two-time-level **semi-Lagrangian
semi-implicit Crank–Nicolson two-stage (RK2)** step
(``semi_lagrangian_crank_nicolson_rk2``, off-centred 0.2), on hybrid σ–p or
pure-σ vertical coordinates. The optional backend is the pySES CAM-SE spectral-element core on a
cubed sphere (``jcm/dycore/pyses/dycore.py::PysesCamSEDycore``,
``pip install jcm[pyses]``, registry name ``pyses_cam_se``), coupled to the
column physics through a pg2 finite-volume physics grid (see
{doc}`../design/pyses_cam_se_dycore`).

On the **dinosaur** backend tracer transport is **semi-Lagrangian only** —
departure-point transport with a Bermejo–Staniforth quasi-monotone limiter.
Every jcm extra tracer (aerosol mass/number, gases, cloud condensate) rides as
a *nodal* tracer while ``specific_humidity`` stays modal for the implicit
q↔Tᵥ coupling; the condensate species additionally enter the dynamics through
the virtual temperature (below), for which Dinosaur converts them to modal
coefficients once per step without changing how they are stored or
transported. The **pySES** backend instead carries every declared tracer as a
pySES passive tracer in physical units — advected and vertically remapped by
the spectral-element dynamics itself (with sub-cycling for the tracer CFL) —
so transport differs between the backends by construction.
**Moist coupling and the mass mixing-ratio contract.** On hybrid coordinates the
dynamics is moist: the virtual temperature is

```
Tv = T · (1 + (Rv/Rd − 1)·q − Σ q_condensate)
```

which drives the geopotential, the temperature adiabatic tendency and the
humidity vorticity/divergence corrections
(``SemiLagrangianPrimitiveEquationsHybrid`` with ``humidity_key`` and
``cloud_keys``, set in ``jcm/dycore/dinosaur/dycore.py::_build_transport``).
The **same** virtual temperature builds the geopotential handed to physics
(``jcm/dycore/dinosaur/state_bridge.py::dynamics_state_to_physics_state``), so
dynamics and physics see one thermodynamic state. This follows ECHAM6, which
uses ``ztv = t·(1 + vtmpc1·q − (xl + xi))`` in the dynamics
(``dyn.f90::ztv``) and the identical expression for the physics geopotential
(``physc.f90::ztvm1``).

The condensate set is whatever the active composition declares out of
``qc``/``qi``/``qr``/``qs`` (``state_bridge.py::CONDENSATE_TRACERS``). Including
prognostic rain and snow is a deliberate departure from ECHAM6, whose
one-moment scheme carries no prognostic precipitation: suspended precipitation
loads a column exactly as suspended cloud water does, and which hydrometeors a
scheme makes prognostic is largely a modelling convention rather than a
physical distinction. A composition carrying no condensate couples humidity
only.

Because the dynamics reads these tracers directly, every mass mixing ratio —
humidity, condensate, aerosol and gas mass — crosses the Dinosaur boundary as
the **dimensionless kg/kg value**, unscaled (``TracerSpec`` with
``nondimensionalize=True``, the default). Quantities that are not mixing
ratios (number concentrations per kg, volume mixing ratios) declare
``nondimensionalize=False`` and pass through untouched. A scaled store would
silently weaken the Tᵥ coupling by the scale factor while leaving every
linear operation — transport, filters, the modal round trip — unchanged, so
the contract is what makes the coupling correct rather than merely present.

On **pure-σ** coordinates (the SPEEDY configurations) Dinosaur exposes no
``humidity_key``, so that dynamics remains dry; only the geopotential handed to
physics carries the moisture and condensate terms.

Horizontal hyperdiffusion is configured by ``jcm/diffusion.py::DiffusionFilter``:
for hybrid L47/L95 grids the resolution-aware ``DiffusionFilter.auto`` selects the
ECHAM ``lmidatm`` level-dependent order profile (∇² near the model top grading to
∇⁶/∇⁸ below, base timescale from ``setdyn.f90``'s ``dampth``); any other grid gets
the uniform SPEEDY ∇²/∇⁴ profile (``DiffusionFilter.default``), with a warning for
unrecognised hybrid grids. An optional upper sponge (Rayleigh drag on u/v plus temperature relaxation
toward the zonal mean and, in production, an absolute ``target_T_K``) is
enabled via the ``run`` group (``jcm/config/run/longrun.yaml``: ``levels: 10``,
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
  emission sources and NaN'd the aerosol microphysics. This is a
  positivity/compute-motivated choice with no fallback. See
  {doc}`../design/dinosaur_sl_jam_configuration`.
- `compute` — the dinosaur backend integrates with a two-time-level
  semi-Lagrangian semi-implicit Crank–Nicolson RK2 step rather than ECHAM's
  three-time-level leapfrog + semi-implicit (both are semi-implicit; the
  time-level structure and SL transport are the real contrast); the pySES
  backend runs float64 dynamics with a float32 physics seam (the SE core
  needs x64).

**Status & known limitations.** SPEEDY physics is generalised to arbitrary
vertical level counts; high-``nlev`` / high-truncation configurations need a
resolution-aware timestep to stay stable (see
{doc}`../design/speedy_variable_levels`). The ECHAM ``lmidatm`` hyperdiffusion
profiles exist only for L47/L95; other hybrid grids fall back to the uniform
SPEEDY profile with a warning. The pySES backend's open production gap is that
multi-GPU sharding of the element and physics-column axes
is unreconciled (see {doc}`../design/pyses_cam_se_dycore`). Its precision seam
is float64 dynamics driving float32 physics by default
(``physics_dtype=float32``; ``ComposablePhysics`` pins tendency dtypes to the
working state), which the canonical ``pyses_ne30l47``/``l95`` configurations
rely on.

**Code pointers.**
- ``jcm/dycore/base.py`` — ``DynamicalCore`` protocol (``initial_state``,
  ``step``, ``to_physics_state``, ``Predictions``).
- ``jcm/dycore/dinosaur/dycore.py`` — ``DinosaurDycore``,
  ``semi_lagrangian_available`` / ``_require_semi_lagrangian`` (the
  Eulerian-removal guard), transport build (nodal tracers), filter build.
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
