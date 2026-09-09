# Convection

**What we do.** Three interchangeable convection schemes:

- **Tiedtke-Nordeng mass-flux**
  (``jcm/physics/convection/tiedtke_nordeng/tiedtke_nordeng.py::TiedtkeConvection``)
  — the ECHAM/ICON scheme: deep (CAPE closure), shallow (moisture-convergence
  closure) and mid-level convection, convective momentum transport, and downdrafts
  (``updraft.py``, ``downdraft.py``, ``flux_tendencies.py``). Cloud-base closure
  selects between a moisture-budget flux ``E/(q_u−q_e)`` and a bounded CAPE flux,
  gated by ECHAM's ``zlo1`` validity test (``mo_cumastr.f90``). The saturation
  adjustment ``cuadjtq`` is a faithful linearised-Newton port of ``mo_cuadjust.f90``
  (``adjustment.py``) with the three ``kcall`` modes. The precipitation budget
  (rain/snow partition, snow melt, sub-cloud Kessler evaporation, proportional
  depletion) transcribes ECHAM ``cuflx`` (``flux_tendencies.py``,
  ``mo_cufluxdts.f90``).
- **SPEEDY convection** (``jcm/physics/convection/speedy_convection.py::diagnose_convection``)
  — SPEEDY's simplified Tiedtke (1993) mass-flux scheme with a
  conditional-instability trigger on saturation moist static energy.
- **Betts-Miller**
  (``jcm/physics/convection/betts_miller/betts_miller_terms.py::BettsMillerConvection``;
  core ``betts_miller.py``) — a faithful port of Isca's ``betts_miller.f90``
  (Frierson 2007 Simplified Betts-Miller), relaxing T and q toward a moist-adiabatic
  reference at RH ``rhbm`` over ``tau_bm``, with ``do_shallower`` / ``do_changeqref``
  siblings. Written broadcasting-native (vertical on axis 0).

Activation is a **smooth sigmoid trigger** on CAPE rather than a hard
``cape > threshold`` branch, so tau / entrainment / threshold parameters carry
nonzero gradients near the trigger. Saturation thermodynamics are shared
(``jcm/physics/convection/saturation.py``, Tetens).

**What ECHAM/CAM does.** ECHAM6-HAM2.3 uses the **Tiedtke (1989) bulk mass-flux
scheme with Nordeng (1994) CAPE closure** (``mo_cumastr.f90`` master driver,
``mo_cuasc.f90`` / ``mo_cuascn.f90`` updraft ascent, ``mo_cudlfs`` / ``mo_cuddraf``
downdrafts, ``mo_cuadjust.f90`` saturation and per-level adjustment,
``mo_cufluxdts.f90`` fluxes). References: Tiedtke, M. (1989), *Mon. Wea. Rev.* 117,
1779-1800; Nordeng, T.E. (1994), ECMWF Tech. Memo. 206. Betts-Miller's reference
is Betts & Miller (1986) as simplified by Frierson, D.M.W. (2007), *J. Atmos. Sci.*
64, 1959-1976 (Isca ``betts_miller.f90``).

**Why we differ.**
- `differentiability` — the hard ``ldcum`` activation and the deep/shallow/mid
  selection are replaced by smooth sigmoid weights, so convective parameters are
  differentiable.
- `science` / `compute` (stopgap) — the **per-level moist-adjustment limits in
  ``mo_cuadjust.f90`` are not yet ported**. The cloud-base mass-flux **CFL cap**
  ``zmfmax = layer_mass/dt`` bounds the column-integrated flux but not per-level
  latent-heat spikes inside the updraft loop. Until the per-level limits land, an
  explicitly-labelled stopgap caps the convective T-tendency at 5 K/hr
  (``_DTDT_MAX``) and rescales the thermodynamic ledger homogeneously — T, q,
  qc/qi, precipitation, and the mass fluxes with the tracer transport they
  drive — preserving column conservation by linearity, as ECHAM's ``zmfub1``
  amplitude scaling does. The **momentum tendencies are the exception**:
  ``dudt``/``dvdt`` are returned unscaled, so a capped plume's momentum
  transport keeps full amplitude (tracked with the other ledger gaps in #676).
  This cap is the documented cause of a cap-pinned single-layer heating
  artifact in pathological columns.

**Status & known limitations.**
- The 5 K/hr tendency cap is a **safety net, not physics**; it fires only where
  the parcel-vs-environment balance has gone pathological (healthy tropical deep
  convection is ~1 K/hr). It remains until the ``mo_cuadjust`` per-level limits are
  ported.
- Cloud-base closure falls back to the bounded CAPE flux (rather than ECHAM's tiny
  flux) when the moisture-budget denominator collapses under a near-saturated cloud
  base or spectral supersaturation ringing.
- SPEEDY and Betts-Miller are idealized alternatives; Betts-Miller is
  specific-humidity-formulated (Isca's mixing-ratio form differs at second order).

**Code pointers.**
- ``jcm/physics/convection/tiedtke_nordeng/`` — ``tiedtke_nordeng.py``
  (``TiedtkeConvection``, the CFL cap, the ``zlo1`` gate, ``_DTDT_MAX``, the
  unported-``mo_cuadjust`` note), ``adjustment.py`` (``cuadjtq``),
  ``flux_tendencies.py`` (``cuflx`` budget, CAPE mass-flux closure),
  ``updraft.py``, ``downdraft.py``.
- ``jcm/physics/convection/speedy_convection.py`` — ``diagnose_convection``.
- ``jcm/physics/convection/betts_miller/`` — ``betts_miller.py``,
  ``betts_miller_terms.py`` (``BettsMillerConvection``).
- Convective *tracer* transport and in-plume scavenging live with the aerosol
  chain — see {doc}`aerosol`.

**Validation evidence.** ``jcm/physics/convection/tiedtke_nordeng/`` test suite
(``tiedtke_nordeng_test.py``, ``adjustment_test.py``, ``updraft_test.py``,
``downdraft_test.py``, ``deep_shallow_test.py``, ``midlevel_trigger_test.py``,
``rce_integration_test.py``, ``convection_units_test.py``,
``smooth_gradients_test.py``, ``cloud_depth_test.py``);
``betts_miller/betts_miller_test.py``; ``speedy_convection_test.py``.
