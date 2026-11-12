ECHAM Physics Package
=====================

Overview
--------

The ECHAM physics package provides comprehensive atmospheric parameterizations originating from the MPI-M ECHAM6 general circulation model. ICON-A (the atmospheric component of the ICON Earth System Model) reused the same ECHAM6 physics with the icosahedral dynamical core; JAX-GCM's port follows the ICON-A wiring of the ECHAM scheme, which is why a number of source-attribution comments still reference ICON's ``atm_phy_echam`` Fortran modules. The implementation is a pure JAX translation that maintains the physical fidelity of the ECHAM physics while adding full differentiability and GPU/TPU acceleration.

The ECHAM physics in JAX-GCM follows the model description of Giorgetta et al. (2018), with the addition of JAM, the JAX Aerosol Module: a flexible, prognostic aerosol process harness with a swappable microphysics core (HAM's M7 or CAM's MAM4 population), coupled to radiation and to cloud-droplet activation and ice nucleation (see :doc:`science/aerosol` and :doc:`design/jam`). The prescribed MACv2-SP plumes (Stevens et al., 2017) remain available as a lightweight alternative.

Key Characteristics
^^^^^^^^^^^^^^^^^^^

- **Comprehensive Physics**: Full representation of radiation, convection, clouds, microphysics, turbulence, surface processes, gravity waves, aerosols, and chemistry
- **Prognostic Cloud Scheme**: Separate prognostic equations for cloud water and cloud ice, with single-moment (ECHAM6.3 ``mo_cloud``) or two-moment (Lohmann/HAM) microphysics; droplet and crystal numbers are prognostic in 2M
- **Flexible Vertical Resolution**: Designed for 47 or 95 vertical levels on hybrid sigma-pressure coordinates
- **Aerosol-Cloud Coupling**: prognostic JAM aerosol (M7 or MAM4) with ARG cloud-droplet activation and aerosol-dependent ice nucleation feeding the two-moment cloud microphysics; or the prescribed MACv2-SP plumes with their Twomey factor on cloud droplet number
- **Differentiability**: Fully compatible with JAX automatic differentiation (``jit``, ``grad``, ``vmap``)
- **Column-Based Vectorization**: Physics operates on ``(nlev, ncols)`` format with ``jax.vmap`` for efficient GPU/TPU execution

Physics Parameterizations
--------------------------

The ECHAM physics package includes the following components, executed in sequence:

1. Diagnostic Preparation
2. Forcing and Boundary Conditions
3. Aerosol (MACv2-SP prescribed plumes, or the prognostic JAM aerosol with the M7 or MAM4 population)
4. Chemistry (Ozone, CH4)
5. Cloud Fraction (Sundqvist)
6. Radiation (Shortwave + Longwave)
7. Vertical Diffusion (TTE-TKE)
8. Surface Physics
9. Convection (Tiedtke-Nordeng)
10. Cloud Microphysics (1M or 2M)
11. Surface Exchange
12. Gravity Wave Drag (Hines non-orographic, Lott-Miller SSO)

The exact term list and order for each preset is in its ``jcm/config/physics/*.yaml``.

Process Coupling
^^^^^^^^^^^^^^^^

Every ``ComposablePhysics`` term receives the same prognostic input state and
the returned tendencies are summed. Terms execute in ECHAM's ``physc`` order
(radiation, vertical diffusion, convection, cloud microphysics), except that
gravity-wave and orographic drag run after the cloud scheme
(``physc.f90`` runs them before ``cucall``), and see what earlier terms did
through the running tendency sum
(``_tendency_run``), the running thermodynamic state (``thermo_run``) and the
cross-step carry, so the coupling is partly sequential and the order is
load-bearing. The cloud microphysics receives ECHAM's inputs this way: the
previous step's post-physics state as its anchor, and the dynamics and every
upstream term as increments. See :doc:`design/operator_split_physics`.

Each parameterization is described in detail below.

Radiation
^^^^^^^^^

The ECHAM stack offers two radiation backends, selected via the ``radiation_scheme`` argument to :func:`echam_physics`:

- ``"rrtmgp"`` (default): the same RRTMGP correlated-k gas-optics package used by ICON, wrapped via the ``jax-rrtmgp`` library
- ``"emulated"``: a neural-network surrogate of RRTMGP, the fast option

The grey two-stream scheme (``jcm.physics.radiation.grey_two_stream``) is an
idealized scheme, not ECHAM physics, and ``radiation_scheme="grey"`` is
rejected; see :doc:`science/radiation` for what it is and how to compose it
explicitly.

**RRTMGP** (default)

The RRTMGP path provides physically correct heating rates suitable for multi-day to multi-year integrations.

*Configuration*: g128 longwave + g112 shortwave (the "reduced" RRTMGP variant designed for production GCMs).

.. list-table:: Radiation g-point comparison
   :header-rows: 1
   :widths: 35 25 25 15

   * - Scheme
     - LW g-points
     - SW g-points
     - Notes
   * - ECHAM-IFS (RRTM)
     - 140
     - 112
     - 16 LW + 14 SW bands; production ECHAM
   * - **JAX-GCM (RRTMGP reduced)**
     - **128**
     - **112**
     - Default ICON production config; what we use today
   * - ICON-A (RRTMGP reduced)
     - 128
     - 112
     - Same configuration as JAX-GCM
   * - RRTMGP reference
     - 256
     - 224
     - Used for line-by-line validation; ~2× the cost

The reference (g256 / g224) variant ships with the ``jax-rrtmgp`` package and can be selected by editing the file paths in :py:func:`jcm.physics.radiation.rrtmgp._ensure_rrtmgp` if higher spectral fidelity is needed; with the chunked-vmap strategy described below it still fits on a single 80 GiB A100.

*Memory & cost*: the call is launch- and gather-bound rather than flop-bound. By default the gas optics run over all g-points in blocks of ``radiation_gpt_chunk`` (64) with a scanned loop (``radiation_gas_optics_mode: all_gpts``, ``radiation_use_scan: true``, the measured A100 optimum; physics-neutral). The columns are one ``vmap``; setting ``JCM_RRTMGP_COL_CHUNKS=n`` splits them into ``n`` blocks for the reverse pass when memory is tight. Structure, options and measured costs are in :doc:`design/rrtmgp_gpu_performance`.

With the default 2-hour ``radiation_interval`` cache, T63L47 ECHAM with RRTMGP costs roughly 22.7 s per simulated day on one A100 (see :doc:`running_at_scale`), and radiation is the dominant term.

**Common features**

- Gas absorption for H2O, CO2, O3, CH4, N2O
- Mie scattering for liquid cloud droplets; ice crystal optics (Yang et al. 2013, Baum et al. 2014)
- Aerosol direct effects with Angstrom spectral scaling (AOD(λ) = AOD(550nm) · (λ/0.55)^(-α))
- Aerosol indirect (Twomey) effect via CDNC modification of droplet effective radius
- Solar geometry from ``jax_solar`` (zenith angle, day/night, orbital parameters)
- Shortwave transmissivity caching: by default radiation is solved every 7200 s (2 hr) and reused on intermediate dynamics steps, as in ECHAM/ICON where it is the slowest physics term. The sun does not hold still over the interval, so the solve is stored as **transmissivities** (each shortwave flux and heating rate divided by the incoming beam of the solve; ECHAM ``trsol``/``trsof``) and every step, solve or not, the shortwave is rebuilt as ``trans * S(t) * max(cos_zenith(t), 0)`` (ECHAM ``physc.f90`` ``zi0 * trsol``). The solve itself is made at the middle of the interval, at ECHAM's curvature-corrected zenith floored at 0.1, so a column that is dark when radiation runs still has a transmissivity and receives sunlight as soon as the sun reaches it. The time-mean incoming shortwave is therefore the exact insolation integral. Longwave is simply held. Set ``RadiationParameters.radiation_interval = 0`` to solve every step.

  One residual error is bounded by the interval rather than removed: a column whose cached shortwave is zero — dark when radiation last ran, or gone dark since — cannot be brought back by a multiplicative factor, so it stays dark until the next full solve even if the sun rises within the interval. That is the main reason not to push ``radiation_interval`` far past a couple of hours.

**Configurable parameters** (:py:class:`jcm.physics.radiation.radiation_types.RadiationParameters`)

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``radiation_interval``
     - Seconds between full radiation calls (cached on intermediate dynamics steps; 0 = every step)
     - 7200.0
   * - ``solar_constant``
     - Total solar irradiance (W/m²)
     - 1361.0

Greenhouse-gas concentrations are not radiation parameters: CO2 and N2O come from ``ForcingData`` (``co2_vmr``, …) and CH4 and O3 from the chemistry term.


Convection
^^^^^^^^^^

**Type**: Mass-flux scheme based on Tiedtke (1989) with modifications by Nordeng (1994)

**Description**: Represents subgrid-scale deep, shallow, and mid-level moist convection using a bulk mass-flux approach. Deep convection uses CAPE closure; shallow convection uses moisture convergence closure.

**Key Features**:

- Three convection types: deep (penetrative), shallow, and mid-level
- CAPE-based closure for deep convection with adjustable timescale
- Organized entrainment and detrainment in updrafts and downdrafts
- Convective momentum transport (vertical redistribution of horizontal winds)
- Evaporatively-driven downdrafts
- Convective precipitation (rain and snow)
- Convective transport of cloud water and ice tracers
- **Cuadjtq saturation adjustment** (ECHAM ``mo_cuadjust.f90::cuadjtq``, compared with the compiled routine): two damped Newton steps ``cond = (q - qs) / (1 + L/cp · dqs/dT)`` on ECHAM's ``ua`` saturation (Sonntag 1990, :py:mod:`jcm.physics.thermodynamics`), the first clipped by ``kcall`` (``0`` both signs for the half-level environment, ``1`` condensation only in the updraft and at cloud base, ``2`` evaporation only in the downdraft), the second unclipped where the first was non-zero, in :py:func:`jcm.physics.convection.tiedtke_nordeng.cuadjtq.cuadjtq`. The updraft and cloud-base parcels use its ``kcall=1`` wrapper :py:func:`~jcm.physics.convection.tiedtke_nordeng.cuadjtq.cuadjtq_newton`, the downdraft wet bulb its ``kcall=2`` wrapper :py:func:`~jcm.physics.convection.tiedtke_nordeng.cuadjtq.cuadjtq_newton_evap`; the ``1 + L/cp · dqs/dT`` denominator accounts for the warming that condensation causes, and every step conserves total water.
- **Mass-flux CFL cap**: at cloud base the updraft mass flux is capped to the layer-mass-per-timestep, ``mfu_cb ≤ rho_cb · dz_cb / dt``, to prevent the explicit transport step from violating CFL when the closure suggests an unphysically large mass flux. This is the JAX-side analogue of ECHAM's implicit upwind transport.

**Activation Criteria**:

``ktype`` records which trigger fired and how ECHAM's moisture-budget test
classified it:

1. **Deep convection** (``ktype=1``): a surface (``cubase``) plume in a column
   whose moisture convergence exceeds 1.1x the surface latent flux
   (``zdqcv > 1.1·E``, ``mo_cumastr.f90:571``) — large-scale convergence
   beyond what the surface itself supplies. Demoted to shallow if the
   realized cloud is thinner than 200 hPa.
2. **Shallow convection** (``ktype=2``): a surface plume fed essentially by
   its own surface flux (no excess convergence).
3. **Mid-level convection** (``ktype=3``): a ``cubasmc``-initiated plume with
   no surface connection (see the mid-level trigger above).

With the 1M cloud scheme a fourth value appears in the published
``convection.ktype``: the cloud step re-types a shallow column as ``ktype=4``
when its liquid water path at and below the convective cloud top
(``convection.cloud_top``) exceeds ``clwprat`` times the path above it, exactly
as ECHAM's ``mo_cloud.f90`` does, so that the next step's radiation applies the
shallow-convection liquid inhomogeneity factor.

The convergence integral uses ECHAM's ``pqte``: the same-step vdiff moisture
tendency plus the dynamics (advection + hyperdiffusion) tendency of the
just-completed dycore step, reconstructed one step lagged from the physics
carry — the same information provenance as ECHAM's leapfrog dynamics
tendency. Its integral over the layers below the cloud base, ``zdqpbl``,
decides whether a surface plume may convect at all (ECHAM's ``zlo1`` gate,
which also requires the cloud-base parcel to carry more than 1 % more water
than the environment there) and sets the plume's first-guess cloud-base flux
``zdqpbl/(g·zqumqe)``. ECHAM has no CAPE trigger, and neither does jcm; each
of these decisions is ECHAM's in the value and carries the derivative of a
logistic surrogate (:doc:`science/convection`).

**Configurable Parameters** (:py:class:`ConvectionParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``entrpen``
     - Entrainment rate for penetrative convection (1/m)
     - 1.0e-4
   * - ``entrscv``
     - Entrainment rate for shallow convection (1/m)
     - 3.0e-3
   * - ``entrmid``
     - Entrainment rate for mid-level convection (1/m)
     - 1.0e-4
   * - ``tau``
     - CAPE-consumption timescale of the Nordeng closure, ECHAM ``cmftau`` (s)
     - 7200.0
   * - ``cmfcmax``
     - Maximum cloud base mass flux (kg/m2/s)
     - 1.0
   * - ``cmfcmin``
     - Minimum cloud base mass flux (kg/m2/s)
     - 1.0e-10
   * - ``cmfdeps``
     - Fractional downdraft mass flux at LFS
     - 0.3
   * - ``cprcon``
     - Precipitation conversion coefficient (s²/m²), applied per unit of
       geopotential depth
     - 2.5e-4
   * - ``cevapcu``
     - Leading coefficient of ECHAM's sub-cloud rain-evaporation profile
       ``cevapcu(jk)``; scales the whole level-dependent profile
     - 1.93e-6
   * - ``ascent_buoyancy_width``, ``ascent_mass_flux_width``,
       ``ascent_condensate_width``
     - Widths of the surrogates of the ascent test's buoyancy (K), 1 %
       flux (fraction of the cloud-base flux) and condensation (kg/kg)
       conditions. Static; derivative only
     - 0.01, 2.0e-3, 1.0e-8
   * - ``precip_onset_width``
     - Width of the surrogate of the ``zdnoprc`` precipitation onset (Pa).
       Static; derivative only
     - 2000.0
   * - ``deep_convergence_width``, ``sub_cloud_supply_width``
     - Widths of the surrogates of the deep/shallow test and of the
       ``zdqpbl > 0`` gate (kg/m²/s). Static; derivative only
     - 2.0e-7, 2.0e-7
   * - ``cloud_base_excess_width``
     - Width of the surrogate of the ``zqumqe > zdqmin`` gate, as a fraction
       of ``zdqmin``. Static; derivative only
     - 0.1
   * - ``cu_cminbuoy``
     - Floor on the cloud-base sub-grid buoyancy excess ``zlift`` (K)
     - 0.2
   * - ``cu_cmaxbuoy``
     - Ceiling on ``zlift`` (K)
     - 1.0
   * - ``cu_cbfac``
     - Multiplier applied to ``thvsig`` when forming ``zlift`` (-)
     - 1.0
   * - ``cu_thvsig``
     - Fallback :math:`\sigma(\theta_v)` (K), used only when no vdiff term
       is present; the model path reads vdiff's prognostic value
     - 1.0
   * - ``cu_lmfmid``
     - ECHAM ``lmfmid``: enable the ``cubasmc`` mid-level trigger
     - ``True``
   * - ``cu_midlev_rh``
     - Environmental RH a mid-level cloud base must exceed (-)
     - 0.90
   * - ``cu_midlev_zmin``
     - Minimum height above the surface for a mid-level cloud base (m)
     - 1500.0
   * - ``cu_midlev_ptop``
     - Pressure floor for a mid-level cloud base, ECHAM ``nmctop`` (Pa)
     - 30000.0

**Cloud-base determination** follows ECHAM ``cubase``'s ``klab`` walk
(``mo_cuinitialize.f90``). A parcel carrying the lowest level's temperature and
humidity is walked upward one level at a time, conserving dry static energy. At
each level it must be positively buoyant — otherwise the column is dropped and
gets **no convection at all** — and the walk stops at the first level where the
parcel condenses. That level is the cloud base if, and only if, the parcel is
buoyant there with condensate loading included, crediting the sub-grid thermal
excess

.. math::

   z_\mathrm{lift} = \min\!\big(\max(c_\mathrm{minbuoy},
   \min(c_\mathrm{maxbuoy}, \sigma_{\theta_v} \cdot c_\mathrm{bfac})), 1\,\mathrm{K}\big)

The practical consequence is that this trigger requires a **well-mixed boundary
layer**: a sounding running 6.5 K/km down to the surface loses roughly 0.4 K of
parcel buoyancy per model layer, more than any physical ``zlift``, and cannot
trigger. That strictness is deliberate, because it is only half of ECHAM's
trigger.

**Mid-level convection** (ECHAM ``cubasmc``, ``mo_cuascent.f90``) is the other
half, and it starts a plume with no surface connection at all. Scanning upward,
the lowest level that is

* within 10 % of saturation (:math:`q > 0.90\,q_s`),
* being lifted by resolved ascent (:math:`\omega < 0`),
* more than 1500 m above the surface, and
* below the 300 hPa ``nmctop`` ceiling

becomes a cloud base, provided the plume seeded there survives its first ascent
step (ECHAM otherwise retries one level higher). The parcel is the
**environment at that level** — not a surface parcel lifted to it — and the
cloud-base mass flux is the resolved ascent itself,
:math:`\max(c_\mathrm{mfcmin}, \min(c_\mathrm{mfcmax}, -\omega/g))`, with no
CAPE or moisture-budget closure and no Nordeng rescale (ECHAM gates that on
``ktype == 1``). This is what covers elevated convection above a stable layer,
warm-conveyor and frontal ascent, and nocturnal elevated convection over land.

``ktype`` therefore records **which trigger fired**, not a diagnosis of the
column: 3 means ``cubasmc``, while 1/2 are ``cubase`` plumes split deep/shallow.

``zlift`` appears in the cloud-base test of both triggers, and in exactly one
ascent test: ECHAM adds it where the level below is still sub-cloud
(``klab == 1``), which happens only at the first step above a ``cubasmc`` base —
``cubase`` sets ``klab = 2`` at its own cloud base. Everywhere else the
updraft's termination test uses the parcel's true buoyancy.

.. note::

   The mid-level trigger needs :math:`\omega` from the dynamical core, so
   composing Tiedtke convection makes the ``omega`` dycore field a hard
   requirement. Both backends (Dinosaur and pySES) supply it, and ``Model``
   switches the provider on automatically. A backend that could not would
   fail at ``Model`` construction; the trigger can be disabled explicitly
   with ECHAM's own namelist switch,
   ``+physics.terms.tiedtke_convection.params.cu_lmfmid=false``, at the cost
   of all elevated convection.

:math:`\sigma_{\theta_v}` is **not** a tunable: it is the standard deviation of
virtual potential temperature at the lowest interior interface (between the
two lowest full levels), taken from the TTE-TKE scheme's prognostic
:math:`\theta_v` variance (ECHAM ``pthvsig = SQRT(pthvvar(klev-1))``,
``vdiff.f90:1338``) and published as ``vertical_diffusion.thv_sigma``. The
variance carries its own production/dissipation budget
(:func:`~jcm.physics.vertical_diffusion.tte_tke.closure.thv_variance_source_update`),
so convective onset follows the boundary layer's turbulent state. Measured on
a 3-day T63L47 run it spans 0.01–1.14 K with a land mean of 0.30 K against an
ocean mean of 0.15 K — structure a constant cannot produce.

``cu_thvsig`` remains only as the fallback for configurations with no vertical
diffusion term (column-mode tests, dry dynamical-core runs).

.. admonition:: Gap vs. ICON-A

   The implementation follows the Tiedtke-Nordeng scheme structure. The key difference is that ICON-A includes additional refinements for the transition between convection types and tuned entrainment profiles that have been calibrated against the full AMIP climatology.


Cloud Cover
^^^^^^^^^^^

**Type**: ECHAM6.3's diagnostic cloud cover, ``mo_cover.f90::cover`` (Sundqvist et al. 1989; Lohmann and Roeckner 1996)

**Description**: Diagnoses cloud cover from relative humidity: zero at or below a critical relative humidity, full at saturation, ``1 - sqrt(1 - b0)`` in between, with ECHAM's stratocumulus enhancement at a low-level inversion over ice-free ocean. The values are ECHAM's, checked column by column against the ECHAM Fortran; the derivatives of the clip and of the inversion test are those of smooth surrogates. See :doc:`science/clouds_microphysics` for the formulation, the vapour pressure and the time level.

**Key Features**:

- Critical relative humidity ``crt + (crs - crt)·exp(1 - (p_s/p)^nex)``
- Saturation over ice or water by ECHAM's ``lo2`` rule (ice below ``t_ice``, or below 0 °C where cloud ice exceeds ``csecfrl``)
- ECHAM's inversion search between its levels ``jbmin`` and the surface, computed from the model's own vertical grid
- No stratospheric cutoff, as in ECHAM; radiation and COSP use the cover only where there is condensate (``mo_radiation.f90``)
- Resolution-dependent defaults for ``crt``, ``crs``, ``nex``, ``nadd``, ``csatsc``, ``cinv`` and ``csecfrl`` (ECHAM's table with jcm's calibrated T63 values of ``crt``, ``crs``, ``nex``, ``csatsc`` and ``cinv``, interpolated between its truncations; built by ``echam_physics(coords=...)``). The T63 values are calibrated per droplet-number source: one set for prescribed droplets (1M, 2M) and one for droplets activated from the JAM aerosol

**Configurable Parameters** (:py:class:`~jcm.physics.clouds.sundqvist.CloudParameters`; T63 defaults shown, prescribed droplets / JAM):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default (T63)
   * - ``crt``
     - Critical RH aloft
     - 0.677 / 0.727 (ECHAM: 0.75)
   * - ``crs``
     - Critical RH at the surface
     - 0.908 / 0.930 (ECHAM: 0.975)
   * - ``nex``
     - Exponent of the critical-RH profile
     - 1.01 / 1.14 (ECHAM: 2)
   * - ``csatsc``
     - Stratocumulus saturation factor at an inversion
     - 0.737 / 0.848 (ECHAM: 0.7)
   * - ``cinv``
     - Inversion stability threshold, as a fraction of the dry adiabatic lapse rate
     - 0.275 / 0.265 (ECHAM: 0.25)
   * - ``csecfrl``
     - Cloud ice [kg/kg] above which ice saturation applies below 0 °C
     - 5e-6
   * - ``t_ice``
     - Homogeneous-freezing temperature ``cthomi`` (K)
     - 238.15
   * - ``nadd`` (static)
     - Offset of the second enhanced level below the inversion level (ECHAM enhances levels ``jb`` and ``jb + nadd`` only)
     - 0
   * - ``smooth_b0``, ``smooth_inv_thr`` (static)
     - Widths of the derivative surrogates (0 selects the reference derivative)
     - 0.02, 2e-4 K/m


Cloud Microphysics
^^^^^^^^^^^^^^^^^^

JAX-GCM ships two cloud microphysics schemes; the 1-moment scheme is the default ``echam_physics(cloud_scheme="1m")`` and the 2-moment scheme is selectable via ``cloud_scheme="2m"``.

**1-moment (default)** — :py:func:`jcm.physics.clouds.echam_1m.cloud_microphysics_column_sweep`

The ECHAM6.3 single-moment scheme, a transcription of ``mo_cloud.f90::cloud``
(r7492). Cloud liquid (``qc``) and cloud ice (``qi``) are prognostic tracers;
rain and snow are fluxes that fall through the column within the step and are
not advected. One top-down ``lax.scan`` runs ECHAM's sections at every level in
ECHAM's order:

1. **Melting** of the incoming snow and of cloud ice above ``tmelt``;
   **sublimation** of the incoming snow (Lin et al. 1983) and **evaporation** of
   the incoming rain (Rotstayn 1997), at the anchor state (below).
2. **Ice sedimentation** (the analytic exponential integral, with ECHAM's
   ``EPSILON(1._wp)`` floor on the ice), the ``lo2`` phase switch, and the return
   of all condensate of a cloud-free cell to vapour.
3. **Condensation** in the cloudy part driven by the step's humidity and
   temperature increments, ``zqcdif = (Δq − Δq_sat)·paclc``, then the whole-box
   1 % supersaturation check; a clear cell that gains condensate is cloudy for
   the rest of the step.
4. **Freezing**: all cloud water below ``cthomi``; Bigg and contact freezing
   between ``cthomi`` and ``tmelt``.
5. **Autoconversion** (cloud water → rain), selected by
   ``MicrophysicsParameters.autoconversion_scheme``:

   - ``"beheng"`` (default): Beheng (1994) implicit form, ECHAM's.
     ``ccraut`` is its rate prefactor (default 15.0).
   - ``"kk2000"``: Khairoutdinov & Kogan (2000) explicit form, a jcm option.
     ``ccraut_kk_threshold`` is the in-cloud qc threshold (default 1e-5 kg/kg).

   and **accretion** of cloud water by rain (``ccracl``, default 6.0).
6. **Aggregation** of cloud ice to snow (Levkov et al. 1992), accretion of ice
   and **riming** of cloud water by snow.
7. The **precipitation fluxes** and the precipitating fraction (reset to the
   local cover where the level's own production dominates), and the return of
   condensate below ``ccwmin`` to vapour with the cover write-back.

The anchor is the previous step's post-physics state and the increments are
the dynamics since then plus ``dt`` times the running tendency of the physics
terms composed before the scheme, with the convective detrainment passed
separately, as for the 2M (see :doc:`design/operator_split_physics`).

The scheme publishes no effective radius: as in ECHAM, the radiation forms the
droplet and crystal radii itself from the step's state
(:py:func:`~jcm.physics.radiation.cloud_optics.radiation_effective_radii`; see
the radiation page of the model description).

**Configurable parameters** (:py:class:`jcm.physics.clouds.echam_1m.MicrophysicsParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``autoconversion_scheme``
     - 0 (Beheng) or 1 (KK2000); accepts ``"beheng"`` / ``"kk2000"``
     - 0 (Beheng)
   * - ``ccraut``
     - Beheng autoconversion rate prefactor
     - 15.0
   * - ``ccraut_kk_threshold``
     - KK2000 in-cloud qc threshold for autoconversion onset (kg/kg)
     - 1.0e-5
   * - ``ccracl``
     - Accretion coefficient (cloud → rain)
     - 6.0
   * - ``cn0s``
     - Snow particle number density (1/m³)
     - 3.0e6
   * - ``crhosno``
     - Snow density (kg/m³)
     - 100.0
   * - ``cvtfall``
     - Ice/snow fall-speed factor; ECHAM's value depends on the truncation
     - 2.5 at T63
   * - ``csecfrl``
     - Cloud ice above which ``T < tmelt`` selects the ice phase (kg/kg);
       resolution dependent
     - 5e-6 at T63
   * - ``autoconversion_twomey``
     - Apply the MACv2-SP Twomey factor to the autoconversion's droplet number
       (jcm's aerosol-cloud interaction; MPI-ESM1.2 scales the radiation's
       number only)
     - True

The droplet number is not a parameter: it is ECHAM's prescribed ``acdnc``
profile. The radiation's droplet number
(:py:func:`~jcm.physics.clouds.cloud_utils.prescribed_droplet_number`) is that
profile times the MACv2-SP Twomey factor.

**2-moment** — :py:func:`jcm.physics.clouds.lohmann_2m.cloud_microphysics_2m`

Two-moment scheme based on ECHAM6.3-HAM with the SPA cloud-droplet activation closure (Lin et al., 2025, see Cloud–Aerosol Coupling below). Tracks ``qc``, ``qi``, ``Nc``, ``Ni``, ``qr``, ``qs`` as prognostic variables. Release-validated (``t63-echam-2m``, ``t106-echam-2m``) and required by JAM, where the droplet source is JAM's activated CDNC instead of SPA; the bare factory default is still ``cloud_scheme="1m"``.

.. admonition:: Notes

   - Both microphysics schemes condense in the cloudy part the humidity increment that the saturation humidity, moved by the temperature increment, does not absorb (ECHAM's ``zqcdif``), followed by a whole-box supersaturation check; neither relaxes the grid-mean state to saturation.
   - ``qc`` and ``qi`` are declared as prognostic tracers via ``Echam1MMicrophysics.required_tracers``; they survive between physics calls. ``qr`` and ``qs`` are NOT prognostic — they are downward column fluxes (ECHAM ``mo_cloud.f90`` ``zrfl``/``zsfl``, zero at the model top each call).


Cloud–Aerosol Coupling with prescribed aerosol (SPA floor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With JAM composed, the 2M scheme instead reads JAM's activated droplet number
(ARG activation for MAM4, HAM's Köhler A/B + ARG with an updraft PDF for M7)
and aerosol-dependent ice nucleation; see :doc:`science/aerosol`. This section
covers the prescribed-aerosol (MACv2-SP) case.

The two-moment microphysics scheme tracks the cloud droplet number ``Nc``
as a prognostic variable. Every step has processes that *remove* droplets
(autoconversion, freezing, …), so the scheme also needs a way to *make*
new ones — i.e. an aerosol activation step.

A textbook activation scheme would compute the local supersaturation from
the cloud-base updraft velocity and step through Köhler theory for the
local aerosol size distribution. With prescribed aerosol neither is available: MACv2-SP gives
column AOD, not aerosol numbers, and the hydrostatic core doesn't resolve
sub-grid updrafts. So instead of "compute activation", we apply a
calibrated *droplet-number floor* and let the microphysics deplete from
there. Following Lin et al. (2025):

.. math::

   N_c^\mathrm{min} = 2000 \cdot (N_\mathrm{CCN} \cdot C_f)^{0.55}

where ``Nccn`` is the cloud condensation nuclei concentration and ``Cf``
is the cloud fraction. The microphysics then enforces
``Nc ← max(Nc, Nc_min)`` at each step.

We get ``Nccn`` from MACv2-SP's column AOD via a Twomey-style empirical
fit, so an anthropogenic-plume increase translates into more cloud
droplets, smaller effective radius, and brighter clouds — the Twomey
indirect aerosol effect, with the right sign and roughly the right
magnitude.

Two things to keep in mind:

- The 0.55 exponent matters. A linear ``Nc ∝ Nccn`` overestimates the
  indirect effect by roughly half; observations constrain
  ``d ln Nc / d ln Nccn`` to the 0.3–0.8 band, and 0.55 sits in the
  middle of that.
- This is a calibrated floor, not a resolved activation. It captures
  first-order cloud-aerosol sensitivity but won't reproduce the details
  a HAM-style aerosol scheme would.


Vertical Diffusion
^^^^^^^^^^^^^^^^^^

**Type**: ECHAM6.3 ``vdiff`` TTE-TKE closure: prognostic TKE and θv variance, Louis stability functions, Blackadar/Holtslag–Boville mixing length (see :doc:`design/tte_tke_closure`)

**Description**: Computes turbulent exchange coefficients and applies vertical diffusion of momentum, heat, and moisture. Includes a prognostic TKE equation with shear production, buoyancy, and dissipation.

**Key Features**:

- Prognostic TKE budget: production (shear + buoyancy), dissipation, vertical transport
- Richardson number-dependent exchange coefficients (Km, Kh)
- Height-dependent mixing length
- Monin-Obukhov surface layer similarity
- Per-surface-type exchange coefficients (water, ice, land)
- Implicit time integration via tridiagonal matrix solver (unconditionally stable)

**Process**:

1. Compute Brunt-Vaisala frequency and Richardson number at each level
2. Derive mixing length from height and stability
3. Solve TKE budget equation (production - dissipation + diffusion)
4. Calculate eddy diffusivities Km (momentum) and Kh (heat/moisture)
5. Apply implicit vertical diffusion to u, v, T, qv, qc, qi
6. Compute surface exchange coefficients per surface type

**Configurable Parameters** (:py:class:`VDiffParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``tpfac1``
     - Implicitness factor for TKE equation
     - 1.5
   * - ``tpfac2``
     - Implicitness factor for diffusion (linear)
     - 0.667
   * - ``tpfac3``
     - Implicitness factor for diffusion (nonlinear)
     - 0.333
   * - ``totte_min``
     - Floor on TKE and the θv variance (ECHAM's ``ztkemin``, ``vdiff.f90``)
     - 1.0e-10
   * - ``surface_layer_scheme``
     - Surface-layer exchange-coefficient scheme: ``"echam_louis"`` (faithful ``precalc_land``/``_ocean``/``_ice`` port) or ``"businger_dyer"`` (simpler bulk form). Its per-tile ``CH·|U|`` / ``CE·|U|`` / ``CM·|U|`` drive the surface flux **and** damping (see Surface Physics).
     - ``"echam_louis"``

Surface Physics
^^^^^^^^^^^^^^^

**Type**: Multi-surface tile scheme with separate treatment of ocean, sea ice, and land

**Description**: Computes turbulent fluxes of momentum, heat, and moisture between the surface and lowest atmospheric level. Each surface type has independent prognostic temperature and flux calculations; grid-box mean fluxes are area-weighted across the three tile fractions (``water_fraction``, ``sea_ice_fraction``, ``land_fraction``) which sum to 1.

**Key features**:

- **Per-tile temperatures**: ocean uses ``forcing.sea_surface_temperature``; sea ice uses its prognostic surface temperature (``s_sicetemp``; starts at ``min(SST, ctfreez=271.38 K)``) where ``sice > 0``, else SST; land has a prognostic skin temperature solved from the implicit surface energy balance, over SLIM's soil column by default and over the prescribed soil temperature ``forcing.stl_am`` with ``land_model: prescribed`` (see :doc:`design/land_skin_energy_balance`)
- **Ocean**: Mixed layer with prescribed SST, Charnock roughness length parameterisation
- **Sea Ice**: Multi-layer thermodynamics (2 default layers), snow on ice, melting/freezing
- **Land**: prescribed-moisture land (soil moisture, snow cover and deep soil temperature are climatologies) with JSBACH's humidity factors for evaporation and a prognostic skin temperature (``jcm.physics.surface.echam.jsbach_land``)
- **Exchange coefficients**: The per-tile surface exchange velocities (``CH·|U|``, ``CE·|U|``, ``CM·|U|``, all in m/s) come from a **single** surface-layer scheme, computed in the vertical-diffusion term and threaded into this term via the ``vertical_diffusion`` diagnostics. The same per-tile coefficients drive **both** the bulk-aerodynamic flux (``flux = ρ·C·Δ``) **and** the implicit-damping factor below, so heat, moisture and momentum stay mutually consistent — one surface exchange coefficient, exactly as ECHAM's ``sfc_exchange_coeff`` feeds both the vdiff surface BC and the flux diagnosis. The scheme is selected by ``VDiffParameters.surface_layer_scheme``:

  - ``"echam_louis"`` (**default, faithful**): a port of ECHAM6.3's own ``mo_surface_land.f90::precalc_land`` (and the matching ``precalc_ocean``/``precalc_ice``) — Louis (1979) functions in both the unstable and stable branches (not ICON's own ``mo_turbulence_diag::sfc_exchange_coeff``, whose stable branch is Mauritsen (2007) instead), potential-temperature bulk Richardson with a moisture-buoyancy term, per-tile momentum/heat roughness and surface wetness. Implemented in ``jcm.physics.vertical_diffusion.tte_tke.surface_layer.compute_surface_exchange_coefficients_echam_louis``.
  - ``"businger_dyer"`` (option): a simpler bulk-Richardson Businger-Dyer form in ``turbulence_coefficients.compute_surface_exchange_coefficients``. Both schemes now return momentum as well as heat/moisture.

  A standalone surface-side bulk-Richardson scheme (``turbulent_fluxes.compute_exchange_coefficients`` / ``compute_stability_functions``) remains in the tree as a tested reference but is **no longer in the default flux path** — ``surface_physics_step`` consumes the upstream exchange coefficients rather than recomputing its own.
- **10 m wind**: ECHAM has one 10 m wind, the stability-corrected ``nsurf_diag`` reduction the vertical-diffusion term diagnoses per tile from the same surface-layer scheme (``vertical_diffusion.wind_10m`` / ``wind_10m_u`` / ``wind_10m_v`` and their per-tile counterparts). The surface-exchange contract publishes it as ``wind_u``/``wind_v`` (``wind_reference="10m"``) and the AeroCom ``uas``/``vas`` apply the same reduction to the post-physics lowest-level wind; the surface term's diagnostics carry no separate 10 m wind. See :doc:`design/surface_exchange`.
- **Implicit damping**: An implicit-Euler factor ``1 / (1 + K·dt/dz_sfc)`` is applied to the sensible-heat, latent-heat, and momentum tendencies at the lowest model level to keep the explicit step stable when ``K·dt/dz_sfc > 2`` (easily violated over rough terrain at ECHAM-tuned exchange coefficients). This stands in for the implicit surface BC of the vdiff tridiagonal solve that ECHAM/ICON use natively but that JCM's explicit pipeline can't currently express. ``K`` here is the same per-tile ``CH·|U|`` / ``CE·|U|`` / ``CM·|U|`` (m/s) that sets the flux, so ``K·dt/dz_sfc`` is properly dimensionless. (The momentum factor previously borrowed the interior diffusivity ``Km`` at the lowest level — units m²/s — which made the momentum damping dimensionally inconsistent and over-suppressed the surface stress by ~10×.)

**Configurable parameters** (:py:class:`jcm.physics.surface.SurfaceParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``z0_water``
     - Ocean roughness length (m)
     - 1.0e-4
   * - ``z0_ice``
     - Sea ice roughness length (m)
     - 1.0e-3
   * - ``z0_land``
     - Land roughness length (m)
     - 0.1

**Boundary conditions**

The ECHAM physics package consumes two NetCDF files at run time. T63 versions sized for the standard tests ship with the repo under ``jcm/data/bc/t63/``:

.. list-table:: Fields read by the ECHAM physics
   :header-rows: 1
   :widths: 22 20 58

   * - Field (NetCDF name)
     - Source / time axis
     - Used by
   * - ``orog`` (terrain.nc)
     - Static
     - Dynamics (modal orography), surface (lapse-correction lower bound), :py:class:`LottMillerSso`
   * - ``lsm`` (terrain.nc)
     - Static
     - ``terrain.fmask`` — tile-fraction split between ocean / sea-ice / land
   * - ``orostd``, ``orosig``, ``orogam``, ``orothe``, ``oropic``, ``oroval`` (terrain.nc)
     - Static (optional)
     - SSO descriptors for :py:class:`LottMillerSso`. Filled with zeros if absent.
   * - ``stl`` (forcing.nc) → ``forcing.stl_am``
     - 12-month climatology
     - Land-tile surface temperature in :py:class:`EchamSurface` (passed through unmodified). When generated from the JSBACH IC file (``ic_land_soil_T63GR15_*.nc``, field ``surf_temp``), this is the real ECHAM/JSBACH land T at the model's orography.
   * - ``sst`` (forcing.nc) → ``forcing.sea_surface_temperature``
     - 12-month climatology
     - Ocean-tile surface temperature; also caps the sea-ice tile via ``min(sst, ctfreez = 271.38 K)`` (the saline freezing point — this is a physical constraint, not a workaround).
   * - ``icec`` (forcing.nc) → ``forcing.sice_am``
     - 12-month climatology
     - Sea-ice tile fraction (clipped to ``[0, 1 − fmask]`` at apply time).
   * - ``alb`` (forcing.nc) → ``forcing.alb0``
     - Static
     - Snow-free background albedo of the non-glacier land (JSBACH ``update_land_surface_fast``, see :doc:`science/surface`).
   * - ``soilw_am`` (forcing.nc)
     - 12-month climatology
     - Soil-moisture initial state for the land-tile column.
   * - ``snowc`` (forcing.nc) → ``forcing.snowc_am``
     - 12-month climatology
     - Prescribed snow-cover fraction of the non-glacier land (total cover ``glac + (1 − glac)·snowc``): brightens the land albedo towards the temperature-dependent snow albedo and sets the sublimating share of the land latent heat flux.
   * - ``forest`` (forcing.nc) → ``forcing.forest_fraction``
     - Static (optional)
     - Forest share of the non-glacier land, masking the snow albedo under a canopy. ``None`` (no forest) if absent.
   * - ``lsm`` (forcing.nc)
     - Static (optional)
     - Land share of the cell: the weight any regrid of the file gives the land-conditional channels (``jcm.data.regridding.CONDITIONAL_FIELDS``).
   * - ``glac`` (forcing.nc) → ``forcing.glacier_fraction``
     - Static (optional)
     - Glacier share of the land: the ECHAM glacier albedo and a fully snow-covered (sublimating) surface. ``None`` (no glacier) if absent.

Generating BC files from ECHAM input
""""""""""""""""""""""""""""""""""""

``utils/convert_echam_bc.py`` converts the standard ECHAM input files into the JCM ``terrain.nc`` + ``forcing.nc`` pair. Typical invocation::

    python utils/convert_echam_bc.py \
        --surface T63GR15_jan_surf.nc \
        --sst     T63_amipsst_1979-2008_mean.nc \
        --sic     T63_amipsic_1979-2008_mean.nc \
        --land-init ic_land_soil_T63GR15_1976.nc \
        --out-dir jcm/data/bc/t63/

``--snow-cover FILE`` takes the 12-month ``snowc`` cover fraction of a jcm forcing file on the same grid instead of the single ECHAM snow snapshot; the packaged T63 file is built with the data-mirror ``forcing_pd.nc`` this way, since the land albedo reads the snow cover every month.

When ``--land-init`` is provided (the JSBACH initial-conditions file from a standard ECHAM dataset), the ``stl`` field uses the real monthly land-surface temperature climatology and the soil-moisture / snow fields use ``init_moist`` / ``snow`` rather than the AMIP-SST extrapolation. Without ``--land-init``, ``stl`` falls back to AMIP SST extrapolated over land — fine for short development runs, but the ~+30 K bias over the Tibetan and Antarctic plateaus has historically driven multi-day stability failures, so the JSBACH-backed file should be used for any climate-style integration.

``ForcingData.from_file`` runs a one-time sanity check on the loaded fields (range bounds, finiteness, and a heuristic that flags the AMIP-extrapolation case at high orography). Hard violations raise; soft ones print a warning and continue.

.. admonition:: Gap vs. ICON-A

   ICON-A couples to the JSBACH land surface model (Reick et al., 2013), which includes dynamic vegetation, phenology, carbon cycle, and detailed soil hydrology. JAX-GCM uses a prescribed-moisture land with JSBACH's evaporation form and a prognostic skin temperature, without the bucket, multi-layer soil, interactive vegetation or carbon. The ocean is prescribed (AMIP-style) rather than coupled to an ocean model.


Gravity Wave Drag
^^^^^^^^^^^^^^^^^

The gravity-wave drag system in JAX-GCM is split into four coexisting schemes
under :py:mod:`jcm.physics.gravity_waves`. The default ECHAM physics factory
(:py:func:`jcm.physics.echam.echam_terms.echam_physics`) wires :py:class:`HinesGwd`
and :py:class:`LottMillerSso` into the term list; ``gw_scheme="frontal"`` (or
``"both"``) selects the CAM frontal spectral scheme instead of (or alongside)
Hines (:doc:`design/frontal_gravity_wave_drag`); the :py:class:`SimpleGwd`
fallback is available but excluded from the default.

**Hines (1997) — non-orographic spectral GWD**
   Faithful port of ECHAM ``mo_gw_hines.f90`` (atm_phy_echam version,
   2326 lines), validated bit-exact against the Fortran reference on 8 column
   scenarios. Targets the production control flow: 8 azimuths, slope=1,
   ``lheatcal=true``, no exponential cutoff, no front/precip/lat-dependent
   sources, no orographic-wave coupling.

   Configurable parameters (:py:class:`HinesParameters`): ``rmscon`` (RMS
   launch wind, default 1.0 m/s), ``kstar`` (typical horizontal wavenumber,
   5e-5 1/m), ``m_min`` (minimum vertical wavenumber, 1e-4 1/m),
   ``f1``..``f6`` (Hines fudge factors), ``alt_cutoff`` (105 km),
   ``smco`` (smoothing coefficient, 2.0), ``lheatcal`` (compute heating
   + diffusion). Static loop knobs (``emiss_lev``, ``naz``, ``nsmax``)
   are passed as Python kwargs to :py:func:`hines_gwd`.

**Lott & Miller (1997) + Lott (1999) — sub-grid orographic GWD**
   Faithful port of ECHAM ``mo_ssodrag.f90`` (atm_phy_echam version,
   1564 lines), validated bit-exact on 4 column scenarios. Targets
   production: ``gkdrag=0.2``, ``gkwake=1.0``, ``gklift=0.0`` (mountain
   lift branch not ported — known gap).

   Configurable parameters (:py:class:`SSOParameters`): ``gpicmea`` /
   ``gstd`` (activation thresholds), ``gkdrag`` (wave-drag coefficient,
   0.2), ``gkwake`` (blocked-flow wake coefficient, 1.0), ``gklift``
   (mountain lift, 0.0). Static knobs (``nktopg``, ``ntop``) are passed
   as Python kwargs to :py:func:`sso_drag`. ``nktopg`` is a property of
   the vertical grid: :py:func:`echam_nktopg` computes it as ECHAM's
   ``sugwd`` does (the highest level whose sigma at an 800 hPa reference
   surface pressure is at least 0.94 — level 45 of L47, 93 of L95), and
   ``LottMillerSso.cache_coords`` derives it from the model grid. It
   raises the top of the blocked-flow / low-level averaging layer to at
   least that level; it is not a model-top setting.

   The sub-grid descriptors (``orostd``, ``orosig``, ``orogam``,
   ``orothe``, ``oropic``, ``oroval``) come from :py:class:`TerrainData`
   (preprocessed fields when the terrain file carries them). The
   activation gate (``ppic-pmea > gpicmea`` AND ``pstd > gstd``) keeps
   drag at zero over ocean automatically.

**Simple monochromatic GWD (legacy)**
   The original placeholder scheme that used to live under ``hines/``
   (:py:func:`simple_gwd`). Single-amplitude wave with Richardson-number
   breaking criterion. Not the actual Hines parameterisation. Kept as a
   cheap option for aquaplanet experiments where running the full
   spectral schemes is overkill.

.. admonition:: Gap vs. ICON-A

   The mountain-lift branch of ``mo_ssodrag`` (controlled by ``gklift``)
   is not ported; ICON-A leaves it disabled by default too, but it can
   be relevant for some configurations. The Hines port omits the
   ``lfront`` (frontal source), ``lozpr`` (precipitation-modulated
   source), and ``lrmscon_lat`` (latitude-dependent ``rmscon``) optional
   branches that the Fortran source has but leaves disabled by default.


Aerosol Scheme (MACv2-SP)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Type**: MACv2-SP Simple Plumes scheme (Stevens et al., 2017)

**Description**: Provides a computationally efficient representation of anthropogenic aerosol effects using 9 prescribed plumes representing major global emission regions. The natural aerosol the plumes are an increment on — the coarse mode and pre-industrial fine mode of the MACv2 climatology — is added to the radiation's aerosol optics by ``Macv2NaturalAerosol`` from the ``macv2_natural_aerosol`` boundary file (``forcing.natural_aerosol_file``); see the aerosol section of the model description.

**Key Features**:

- 9 anthropogenic plumes (East Asia, Europe, N. America East/West, Africa biomass burning, South America, India, Southeast Asia, Middle East)
- Gaussian spatial distribution with rotated elliptical extent
- Beta-function vertical distribution
- Optical properties at 550 nm: AOD, SSA, asymmetry parameter (per plume)
- Angstrom exponent spectral scaling: AOD(lambda) = AOD(550nm) * (lambda/0.55)^(-alpha)
- Twomey effect: CDNC = 1 + A * ln(B * AOD + 1), modifying cloud droplet effective radius
- Time-varying emissions via forcing data (``aerosol_year_weight``, ``aerosol_ann_cycle``)

**Aerosol Effects on Climate**:

1. **Direct effect**: Aerosol scattering and absorption modify shortwave and longwave radiation
2. **First indirect (Twomey) effect**: Aerosol-induced increase in CDNC reduces cloud droplet size, increasing cloud albedo
3. **Second indirect effect**: Modified droplet number affects autoconversion rate in microphysics (KK2000: P_aut ~ N_c^(-1.79))

**Configurable Parameters** (:py:class:`AerosolParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``aod_spmx``
     - Maximum AOD at 550 nm per plume
     - [0.148, 0.094, 0.636, ...]
   * - ``ssa550``
     - Single scattering albedo at 550 nm per plume
     - [0.93, 0.93, 0.93, 0.93, 0.87, ...]
   * - ``asy550``
     - Asymmetry parameter at 550 nm per plume
     - 0.63 (all plumes)
   * - ``angstrom``
     - Angstrom exponent per plume
     - 2.0 (all plumes)
   * - ``background_aod``
     - Natural background AOD at 550 nm
     - 0.02

.. admonition:: Note vs. ICON-A

   ICON-A typically uses the Kinne et al. (2013) aerosol climatology or the MACv2-SP scheme. The JAX-GCM implementation uses MACv2-SP with the addition of Angstrom spectral scaling (matching the Fortran implementation) and aerosol-cloud coupling through the CDNC modification of both cloud optics and microphysics autoconversion. MPI-ESM1.2 applies the MACv2-SP factor to the radiation's droplet number only and leaves the cloud microphysics' unperturbed (Mauritsen et al. 2019); the extra autoconversion path is jcm's recorded departure (#932).


Chemistry
^^^^^^^^^

**Type**: Simplified prescribed chemistry

**Description**: Provides ozone and methane to radiation. Ozone is the prescribed CMIP6 monthly climatology (``OzoneClimatology``, applied by ``EchamBoundaryConditions``; ``forcing.ozone_file: auto`` resolves the packaged or mirror file and raises if neither fits the grid). The analytic profile below is used only on explicit opt-in. CO2 and N2O are prescribed forcings (``forcing.co2_vmr``), read by radiation directly, not chemistry parameters.

**Key Features**:

- **Ozone**: prescribed 3D monthly climatology by default; an analytic interim profile (scale height, maximum, height of maximum) on opt-in
- **CH4**: surface VMR with exponential decay above, at a specified lifetime and OH scaling

**Configurable Parameters** (:py:class:`ChemistryParameters`):

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Parameter
     - Description
     - Default
   * - ``ozone_max_vmr``
     - Maximum ozone VMR of the analytic profile (ppmv)
     - 8.0
   * - ``ozone_scale_height``
     - Analytic ozone scale height (m)
     - 7000.0
   * - ``ozone_tropopause_height``
     - Height of the analytic ozone maximum (m)
     - 20000.0
   * - ``methane_surface_vmr``
     - Surface methane VMR (ppmv)
     - 1.9
   * - ``methane_lifetime``
     - Methane lifetime (s)
     - 9 years
   * - ``methane_oh_scaling``
     - OH scaling factor
     - 1.0

.. admonition:: Gap vs. ICON-A

   ICON-A can be coupled to interactive chemistry (ICON-ART). JAX-GCM prescribes ozone from a monthly climatology and has no interactive gas-phase chemistry beyond the JAM sulfur chemistry (see :doc:`science/chemistry`).


Forcing and Boundary Conditions
--------------------------------

**Description**: Manages time-varying boundary conditions that drive the model surface:

- **Sea Surface Temperature (SST)**: Prescribed from climatology or constant profile
- **Sea Ice Concentration**: Prescribed from climatology
- **Snow Cover**: Prescribed from climatology
- **Soil Moisture**: Prescribed from climatology
- **Surface Albedo**: ECHAM 6.3 tile albedos — land from the background ``alb`` with prescribed snow cover, forest masking and glaciers; temperature-dependent sea ice; zenith-dependent open water (:doc:`science/surface`)
- **Aerosol Temporal Weights**: Per-plume year and seasonal cycle weights for MACv2-SP

The forcing data system supports both realistic (from netCDF files with 365 daily time steps) and idealized (aquaplanet with cos2 SST profile) configurations.

**Aerosol Temporal Forcing**:

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Field
     - Description
     - Default
   * - ``aerosol_year_weight``
     - Per-plume emission weight for scenario year (nplumes,)
     - ones (present-day)
   * - ``aerosol_ann_cycle``
     - Per-plume seasonal cycle weight (nplumes,)
     - ones (no seasonality)

These fields enable time-slice experiments (e.g., pre-industrial vs. present-day aerosols) by scaling the plume emissions without modifying the aerosol parameters.

Using Custom Parameters
-----------------------

To customize physics parameters:

.. code-block:: python

   from jcm.physics.echam.echam_terms import echam_physics
   from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import ConvectionParameters
   from jcm.physics.radiation.radiation_types import RadiationParameters

   convection = ConvectionParameters.default(
       tau=3600.0,        # Faster Nordeng closure (1 hour; the default is 2)
       entrpen=2.0e-4     # Stronger entrainment
   )

   radiation = RadiationParameters.default(
       solar_constant=1360.0
   )

   # Pass per-scheme parameter structs to the ECHAM factory
   physics = echam_physics(convection=convection, radiation=radiation)


Composable Physics API
-----------------------

The ECHAM physics is composable: each parameterization is a
``PhysicsTerm`` (``flax.nnx.Module``), and ``echam_physics()`` returns a
``ComposablePhysics`` configured for column vectorization with the standard
ordering wired up. Schemes can be swapped in or out without touching the
orchestrator:

.. code-block:: python

   from jcm.physics.echam.echam_terms import echam_physics

   # Create composable ECHAM physics with all standard terms
   physics = echam_physics()

   # Use the neural-network emulator of RRTMGP, the fast radiation option
   physics = echam_physics(radiation_scheme="emulated")

   # Remove non-orographic Hines gravity-wave drag
   physics = echam_physics().remove("hines")

   # Replace a single term with a separately configured instance
   from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
       ConvectionParameters,
       TiedtkeConvection,
   )
   convection = TiedtkeConvection(
       params=ConvectionParameters.default(tau=10800.0),
   )
   physics = echam_physics().replace("convection", convection)

A different radiation term can either be passed to the factory,
``echam_physics(radiation_scheme=<term>)``, or swapped in afterward with
``physics.replace("radiation", <term>)``. Both routes derive the same
composition: the ``band_config`` (SW/LW band centers the aerosol optics read)
and the JAM optics cadence (``JamOpticsTerm.configure_radiation_gate``) are
re-derived from whichever term ends up in the radiation slot, so a
``replace("radiation", ...)`` composition is indistinguishable from the
equivalent ``radiation_scheme=<term>`` one. See
``docs/source/design/composable_physics.md`` for what gets re-derived and
why.

Each ECHAM term is a ``PhysicsTerm`` subclass (``flax.nnx.Module``) with lazy
imports — the underlying ECHAM physics functions are imported at call time,
keeping startup fast and avoiding circular dependencies.

The returned ``ComposablePhysics`` enables column vectorization: it reshapes
the 3D state to ``(nlev, ncols)`` before iterating terms, and reshapes
accumulated tendencies back to 3D afterward.

Module Locations
^^^^^^^^^^^^^^^^

After the directory reorganization, ECHAM process modules live under their
respective process directories:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Process
     - Module path
   * - Radiation (RRTMGP)
     - ``jcm.physics.radiation.rrtmgp``
   * - Radiation (NN emulator)
     - ``jcm.physics.radiation.nn_emulator_scheme``
   * - Convection
     - ``jcm.physics.convection.tiedtke_nordeng``
   * - Clouds (fraction)
     - ``jcm.physics.clouds.sundqvist``
   * - Microphysics (1-moment)
     - ``jcm.physics.clouds.echam_1m``
   * - Microphysics (2-moment)
     - ``jcm.physics.clouds.lohmann_2m``
   * - Surface
     - ``jcm.physics.surface.echam``
   * - Vertical Diffusion
     - ``jcm.physics.vertical_diffusion.tte_tke``
   * - Gravity Waves (Hines / SSO / frontal)
     - ``jcm.physics.gravity_waves.hines`` / ``.sso`` / ``.spectral``
   * - Aerosol (JAM; M7 and MAM4 cores)
     - ``jcm.physics.aerosol.jam`` (cores in ``jam.microphysics``)
   * - Aerosol (MACv2-SP)
     - ``jcm.physics.aerosol.macv2_sp``
   * - Chemistry
     - ``jcm.physics.chemistry.simple_chemistry``
   * - Diagnostics (WMO tropopause)
     - ``jcm.physics.diagnostics.wmo_tropopause``

ECHAM-specific infrastructure such as coordinate helpers and the package
factory remains at ``jcm.physics.echam``; individual scheme parameter
types live beside their implementations.


Scientific References
---------------------

The ECHAM physics parameterizations are based on the following key publications:

1. **Giorgetta, M. A., et al.** (2018). ICON-A, the atmosphere component of the ICON Earth System Model: I. Model description. *Journal of Advances in Modeling Earth Systems*, 10, 1613--1637. https://doi.org/10.1029/2017MS001242

2. **Tiedtke, M.** (1989). A comprehensive mass flux scheme for cumulus parameterization in large-scale models. *Monthly Weather Review*, 117, 1779--1800.

3. **Nordeng, T. E.** (1994). Extended versions of the convective parameterization scheme at ECMWF and their impact on the mean and transient activity of the model in the tropics. *ECMWF Technical Memorandum*, 206.

4. **Sundqvist, H., Berge, E., & Kristjansson, J. E.** (1989). Condensation and cloud parameterization studies with a mesoscale numerical weather prediction model. *Monthly Weather Review*, 117, 1641--1657.

5. **Lohmann, U. & Roeckner, E.** (1996). Design and performance of a new cloud microphysics scheme developed for the ECHAM general circulation model. *Climate Dynamics*, 12, 557--572.

6. **Khairoutdinov, M. & Kogan, Y.** (2000). A new cloud physics parameterization in a large-eddy simulation model of marine stratocumulus. *Monthly Weather Review*, 128, 229--243.

7. **Brinkop, S. & Roeckner, E.** (1995). Sensitivity of a general circulation model to parameterizations of cloud-turbulence interactions in the atmospheric boundary layer. *Tellus A*, 47, 197--220.

8. **Pincus, R. & Stevens, B.** (2013). Paths to accuracy for radiation parameterizations in atmospheric models. *Journal of Advances in Modeling Earth Systems*, 5, 225--233.

9. **Lott, F. & Miller, M. J.** (1997). A new subgrid-scale orographic drag parametrization: Its formulation and testing. *Quarterly Journal of the Royal Meteorological Society*, 123, 101--127.

10. **Hines, C. O.** (1997). Doppler-spread parameterization of gravity-wave momentum deposition in the middle atmosphere. Part 1: Basic formulation. *Journal of Atmospheric and Solar-Terrestrial Physics*, 59, 371--386.

11. **Stevens, B., Fiedler, S., Kinne, S., et al.** (2017). MACv2-SP: a parameterization of anthropogenic aerosol optical properties and an associated Twomey effect for use in CMIP6. *Geoscientific Model Development*, 10, 433--452.

12. **Lin, G., et al.** (2025). Simple Prescribed Aerosol scheme for E3SMv3. *Atmospheric Chemistry and Physics*, 25, 15105--15129. https://acp.copernicus.org/articles/25/15105/2025/


Assumptions and Limitations
----------------------------

**Vertical Resolution**:

- Designed for 47 or 95 vertical levels with hybrid sigma-pressure coordinates
- Parameters are tuned for this configuration; performance may vary at other resolutions
- Sub-grid processes (e.g., turbulence, convection) scale with level count

**Simplifications Relative to ICON-A**:

- Prescribed-moisture land with a prognostic skin temperature instead of full JSBACH with interactive vegetation
- No interactive gas-phase chemistry (ozone is a prescribed climatology)
- No sub-stepping for microphysics sedimentation

**Time Steps**:

- The validated ECHAM L47/L95 time step is 12 minutes (``run/default.yaml``)
- Physics terms read the model timestep from the enclosing
  ``ComposablePhysics`` at runtime.
- Shorter time steps needed for higher resolutions

**Forcing Data**:

- Supports daily climatological or constant boundary conditions
- Runs on a Gregorian calendar; climatological forcing is wrapped onto it (``forcing.align``)
- SST and sea ice are prescribed (AMIP-style), not predicted
- Aerosol emissions controllable via temporal forcing weights

**Domain**:

- Global model (no regional capability)
- Spectral dynamical core (Dinosaur) with a Gaussian grid for physics, or the optional pySES CAM-SE spectral-element core (``ma-ne30-*``)


Comparison with SPEEDY Physics
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - SPEEDY
     - JAX-GCM ECHAM
     - Full ICON-A
   * - Complexity
     - Intermediate
     - High
     - Very High
   * - Speed
     - Fast
     - Medium
     - Slow
   * - Vertical Levels
     - 8 (typical)
     - 47 (typical)
     - 47--95
   * - Radiation Bands
     - 2 SW, 3 LW
     - 14 SW, 16 LW (RRTMGP)
     - 14 SW, 16 LW
   * - Cloud Scheme
     - Diagnostic
     - Prognostic (single/two-moment)
     - Prognostic (single/two-moment)
   * - Convection
     - Simplified Tiedtke
     - Tiedtke-Nordeng
     - Tiedtke-Nordeng
   * - Turbulence
     - Relaxation-based
     - TTE-TKE (ECHAM6.3 ``vdiff``)
     - TTE (Mauritsen et al.)
   * - Surface
     - Bulk, 2 types
     - 3 tiles; prescribed-moisture land
     - JSBACH + HD model
   * - Aerosols
     - Fixed climatology
     - Prognostic JAM (M7 or MAM4), or MACv2-SP prescribed
     - MACv2-SP / HAM
   * - Chemistry
     - None (optional CO2)
     - Prescribed O3 climatology, CH4
     - Prescribed / ICON-ART
   * - Gravity Waves
     - None
     - Orographic + non-orographic
     - SSO + Hines
   * - Use Case
     - Dynamics, ML training
     - Research, ML, DA
     - Climate projections


Next Steps
----------

- See :doc:`getting_started` for examples of running models with ECHAM physics
- See :doc:`speedy_physics` for comparison with the simpler SPEEDY physics package
- See :doc:`api` for detailed API documentation of individual parameterizations
