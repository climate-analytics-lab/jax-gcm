ECHAM Physics Package
=====================

Overview
--------

The ECHAM physics package provides comprehensive atmospheric parameterizations originating from the MPI-M ECHAM6 general circulation model. ICON-A (the atmospheric component of the ICON Earth System Model) reused the same ECHAM6 physics with the icosahedral dynamical core; JAX-GCM's port follows the ICON-A wiring of the ECHAM scheme, which is why a number of source-attribution comments still reference ICON's ``atm_phy_echam`` Fortran modules. The implementation is a pure JAX translation that maintains the physical fidelity of the ECHAM physics while adding full differentiability and GPU/TPU acceleration.

The ECHAM physics in JAX-GCM follows the model description of Giorgetta et al. (2018), with simplifications to the radiation scheme (reduced spectral bands) and the addition of the MACv2-SP aerosol scheme (Stevens et al., 2017) with aerosol-cloud coupling.

Key Characteristics
^^^^^^^^^^^^^^^^^^^

- **Comprehensive Physics**: Full representation of radiation, convection, clouds, microphysics, turbulence, surface processes, gravity waves, aerosols, and chemistry
- **Prognostic Cloud Scheme**: Separate prognostic equations for cloud water and cloud ice with single-moment microphysics
- **Flexible Vertical Resolution**: Designed for 40+ vertical levels on hybrid sigma-pressure coordinates
- **Aerosol-Cloud Coupling**: MACv2-SP aerosol scheme with Twomey effect on cloud droplet number and Angstrom spectral scaling
- **Differentiability**: Fully compatible with JAX automatic differentiation (``jit``, ``grad``, ``vmap``)
- **Column-Based Vectorization**: Physics operates on ``(nlev, ncols)`` format with ``jax.vmap`` for efficient GPU/TPU execution

Physics Parameterizations
--------------------------

The ECHAM physics package includes the following components, executed in sequence:

1. Diagnostic Preparation
2. Forcing and Boundary Conditions
3. Aerosol Scheme (MACv2-SP)
4. Chemistry (Ozone, CO2, CH4)
5. Radiation (Shortwave + Longwave)
6. Convection (Tiedtke-Nordeng)
7. Cloud Diagnostics
8. Cloud Microphysics
9. Vertical Diffusion (TKE-based)
10. Surface Physics
11. Gravity Wave Drag

Process Coupling
^^^^^^^^^^^^^^^^

Following ICON-A, radiation, vertical diffusion with surface processes, and gravity wave drag operate in a **parallel split** (each receives the same input state, and tendencies are summed). These initial processes and convection/cloud processes are coupled via **serial split** (output of one feeds into the next).

Each parameterization is described in detail below.

Radiation
^^^^^^^^^

JAX-GCM offers three radiation backends, selected via the ``radiation_scheme`` argument to :func:`echam_physics`:

- ``"rrtmgp"`` (recommended): the same RRTMGP correlated-k gas-optics package used by ICON, wrapped via the ``jax-rrtmgp`` library
- ``"grey"`` (default): a fast, low-fidelity two-stream scheme intended for development and ML emulator training
- ``"emulated"``: a neural-network surrogate of RRTMGP

**RRTMGP** (recommended for production)

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

*Memory & cost*: RRTMGP has substantial intermediate-array memory needs. JAX-GCM evaluates it in chunks via ``lax.map``; the chunk size is auto-detected from device HBM (see :func:`jcm.physics.radiation.rrtmgp.chunk_budget`) and on a single 80 GiB A100 picks 9216 columns / chunk for T63L47 (2 chunks), with peak per-chunk memory around 33 GiB. The earlier 4608-column / 4-chunk default was ~74 % slower per call.

Per-call cost at T63L47 g128/g112 is dominated by gas-optics interpolation table lookups across ``ncols × nlev × ngpt`` cells, so it scales close to linear in horizontal resolution and in ``nlev``. Measured on an idle 80 GiB A100 (single steady-state RRTMGP call inside a single-step ``model.resume`` Python loop, after warm-up):

.. list-table:: Steady-state per-step cost on T63L47 (real terrain + sponge, dt = 12 min)
   :header-rows: 1
   :widths: 30 25 25 20

   * - Scheme
     - RRTMGP step (RAD)
     - cache step
     - rad/cache ratio
   * - 1M microphysics
     - 16.3 s
     - ≈ 98 ms
     - ~165×
   * - 2M microphysics
     - 15.8 s
     - ≈ 101 ms
     - ~155×

The microphysics choice has effectively no impact on RRTMGP per-call cost — the work is entirely radiation. With the default 2-hour ``radiation_interval`` cache (RRTMGP fires on 1 step in 10 at dt = 12 min) a 30-day T63L47 + sponge integration takes ~78 min wall vs ~6.5 min for grey radiation — i.e. **~12× the grey total**, not the ~4× a smaller-config measurement might suggest. That works out to ~1.5 sim-yr per wall-day on one A100 at climate-quality resolution.

**Grey two-stream** (development / ML training)

The grey scheme is a simplified two-stream scheme with hand-tuned band coefficients. It is fast (single GPU pass, no chunking) but NOT physically calibrated — comparison against the RRTMGP harness shows mid-tropospheric LW cooling is roughly 150× too weak and OLR is ~70 % too high on a tropical column. Grey is appropriate for short development runs, neural-network emulator training (the network learns the mapping anyway), and any test where radiation is a passive element of the run; for multi-day climate-style integrations use RRTMGP.

*Configuration*: 2 SW bands (visible 0.2-0.69 µm, near-IR 0.69-2.5 µm), 3 LW bands (window 10-350 cm⁻¹, CO2 350-500 cm⁻¹, H2O 500-2500 cm⁻¹).

**Common features (all backends)**

- Gas absorption for H2O, CO2, O3, CH4, N2O
- Mie scattering for liquid cloud droplets; ice crystal optics (Yang et al. 2013, Baum et al. 2014)
- Aerosol direct effects with Angstrom spectral scaling (AOD(λ) = AOD(550nm) · (λ/0.55)^(-α))
- Aerosol indirect (Twomey) effect via CDNC modification of droplet effective radius
- Solar geometry from ``jax_solar`` (zenith angle, day/night, orbital parameters)
- Heating-rate caching: by default radiation is recomputed every 7200 s (2 hr) and reused on intermediate dynamics steps via ``_radiation_with_caching``. This matches the ECHAM/ICON convention — radiation is the slowest physics term and varies on hour timescales, so caching is essentially free at the accuracy level. Set ``RadiationParameters.radiation_interval = 0`` to compute every step.

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
   * - ``co2_vmr``
     - CO2 volume mixing ratio
     - 400e-6
   * - ``ch4_vmr``
     - CH4 volume mixing ratio
     - 1.8e-6
   * - ``n2o_vmr``
     - N2O volume mixing ratio
     - 0.32e-6


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
- **Cuadjtq saturation adjustment** (faithful port of ECHAM ``mo_cuadjust.f90``): the iterative linearised Newton step ``cond = (q - qs) / (1 + L/cp · dqs/dT)`` with all three ``kcall`` modes (saturation, evaporation, downdraft) is in :py:mod:`jcm.physics.convection.tiedtke_nordeng.adjustment`. Used by the parcel-lifting and detrainment branches to keep the in-cloud thermodynamic state consistent.
- **Mass-flux CFL cap**: at cloud base the updraft mass flux is capped to the layer-mass-per-timestep, ``mfu_cb ≤ rho_cb · dz_cb / dt``, to prevent the explicit transport step from violating CFL when the closure suggests an unphysically large mass flux. This is the JAX-side analogue of ECHAM's implicit upwind transport.

**Activation Criteria**:

Convection activates based on the diagnosed convection type (``ktype``):

1. **Deep convection**: Triggered by conditional instability with sufficient CAPE; CAPE closure timescale ``tau``
2. **Shallow convection**: Triggered by moisture convergence in the boundary layer
3. **Mid-level convection**: Triggered by conditional instability above the boundary layer

**Configurable Parameters** (:py:class:`ConvectionParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``entrpen``
     - Entrainment rate for penetrative convection (1/Pa)
     - 1.0e-4
   * - ``entrscv``
     - Entrainment rate for shallow convection (1/Pa)
     - 3.0e-4
   * - ``entrmid``
     - Entrainment rate for mid-level convection (1/Pa)
     - 1.0e-4
   * - ``tau``
     - CAPE closure timescale (s)
     - 3600.0
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
     - Precipitation conversion coefficient (1/m)
     - 1.4e-3
   * - ``dt_conv``
     - Convection time step (s)
     - 3600.0

.. admonition:: Gap vs. ICON-A

   The implementation follows the Tiedtke-Nordeng scheme structure. The key difference is that ICON-A includes additional refinements for the transition between convection types and tuned entrainment profiles that have been calibrated against the full AMIP climatology.


Cloud Cover
^^^^^^^^^^^

**Type**: Diagnostic cloud cover scheme based on Sundqvist et al. (1989)

**Description**: Diagnoses cloud fraction from relative humidity using a threshold-based approach. Cloud fraction increases from zero at a critical relative humidity to full cover at saturation.

**Key Features**:

- RH-based diagnostic cloud fraction
- Critical RH varies with height (lower threshold at top of atmosphere, higher near surface)
- Power-law interpolation between surface and TOA thresholds
- Mixed-phase partitioning between liquid and ice based on temperature
- Separate treatment above and below freezing

**Configurable Parameters** (:py:class:`CloudParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``crt``
     - Critical RH at surface for cloud formation
     - 0.9
   * - ``crs``
     - Critical RH at model top for cloud formation
     - 0.7
   * - ``nex``
     - Power-law exponent for vertical RH profile
     - 4.0
   * - ``t_ice``
     - Temperature for pure ice phase (K)
     - 238.15
   * - ``csatsc``
     - Saturation factor for stratocumulus
     - 0.97

.. admonition:: Gap vs. ICON-A

   ICON-A uses the Sundqvist et al. (1989) scheme with additional tuning for the representation of marine stratocumulus and Arctic low clouds. The JAX-GCM implementation captures the core RH-based diagnostic but may lack some of the refined tuning parameters.


Cloud Microphysics
^^^^^^^^^^^^^^^^^^

JAX-GCM ships two cloud microphysics schemes; the 1-moment scheme is the default ``echam_physics(cloud_scheme="1m")`` and the 2-moment scheme is selectable via ``cloud_scheme="2m"``.

**1-moment (default)** — :py:func:`jcm.physics.clouds.echam_1m.cloud_microphysics`

Bulk single-moment scheme based on ICON's ``mo_cloud.f90`` (single-moment branch). Tracks cloud liquid (``qc``) and cloud ice (``qi``) as prognostic tracers; rain and snow fluxes are computed within each column and not advected.

Key processes:

1. **Autoconversion** (cloud water → rain). Two formulations are selectable via ``MicrophysicsParameters.autoconversion_scheme``:

   - ``"beheng"`` (default): Beheng (1994) implicit form, robust at large dt. ``ccraut`` is the rate prefactor (default 15.0).
   - ``"kk2000"``: Khairoutdinov & Kogan (2000) explicit form. ``ccraut`` is the qc threshold above which autoconversion fires (small g/kg-scale value, e.g. 1e-5).

2. **Accretion** of cloud droplets by raindrops (``ccracl`` coefficient, default 6.0)
3. **Ice autoconversion + aggregation** of cloud ice by snow (Levkov et al., 1992)
4. **Riming** of cloud water by falling snow
5. **Melting / freezing** at the 0 °C level
6. **Sedimentation** with terminal velocity parameterisations:

   - Ice uses the integral form ``zxised = xi · exp(-vt·dt/dz) + flux_in / (rho·vt) · (1 - exp(...))`` for stability at the long ECHAM-default ``dt = 12 min`` timestep
   - Rain and snow use the simpler instantaneous form

7. **Evaporation / sublimation** of precipitation in subsaturated layers

A faithful column-sweep variant :py:func:`jcm.physics.clouds.echam_1m.cloud_microphysics_column_sweep` is also implemented (top-down ``lax.scan`` propagation of rain and snow fluxes, ICON ``mo_cloud.f90:267-1080`` structure, with Rotstayn 1997 rain evaporation). It is currently NOT wired into the default factory because it interacts with Sundqvist condensation in a way that creates a positive moisture-feedback loop (rain re-evap → condensation → latent heat release → convection); a coupled implicit Newton step between the rain-evap source and the condensation sink is the planned fix. Until then ``apply_microphysics_1m`` calls the per-level helper.

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
     - Autoconversion rate prefactor (Beheng) or qc threshold (KK2000)
     - 15.0
   * - ``ccracl``
     - Accretion coefficient (cloud → rain)
     - 6.0
   * - ``cthomi``
     - Homogeneous ice nucleation temperature (K)
     - 233.15
   * - ``ccollec``
     - Collection efficiency rain/cloud
     - 0.7
   * - ``ccollei``
     - Collection efficiency snow/ice
     - 0.3
   * - ``cn0s``
     - Snow particle number density (1/m³)
     - 3.0e6
   * - ``crhosno``
     - Snow density (kg/m³)
     - 100.0
   * - ``cvtfall``
     - Ice content-dependent terminal velocity factor
     - 3.29
   * - ``vt_rain_a`` / ``vt_rain_b``
     - Rain terminal velocity coefficient and exponent
     - 386.0 / 0.67
   * - ``vt_snow_a`` / ``vt_snow_b``
     - Snow terminal velocity coefficient and exponent
     - 8.8 / 0.15
   * - ``base_cdnc``
     - Baseline CDNC in clean air (1/m³)
     - 100e6

**2-moment** — :py:func:`jcm.physics.clouds.lohmann_2m.cloud_microphysics_2m`

Two-moment scheme based on ECHAM6.3-HAM with the SPA cloud-droplet activation closure (Lin et al., 2025, see Cloud–Aerosol Coupling below). Tracks ``qc``, ``qi``, ``Nc``, ``Ni``, ``qr``, ``qs`` as prognostic variables. Currently in alpha — selectable but the 1M scheme is the default.

.. admonition:: Notes

   - The Sundqvist condensation that feeds the microphysics uses a linearised Newton step (``cond = (q - qs) / (1 + L/cp · dqs/dT)``) ported from ICON ``mo_cloud.f90`` with the Newton denominator that damps the per-step heating by ~6× in the warm troposphere — without it, single-step condensation at 100 % supersat produced ~+60 K heating spikes vs ECHAM's ~+12 K.
   - ``qc`` and ``qi`` are declared as prognostic tracers via ``EchamCloudsAndMicrophysics1M.required_tracers``; they survive between physics calls. ``qr`` and ``qs`` are NOT prognostic — they are downward column fluxes per ICON ``mo_cloud.f90:267-268`` (``zrfl/zsfl`` reset to 0 at TOA each call).


Cloud–Aerosol Coupling (SPA activation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two-moment microphysics scheme tracks the cloud droplet number ``Nc``
as a prognostic variable. Every step has processes that *remove* droplets
(autoconversion, freezing, …), so the scheme also needs a way to *make*
new ones — i.e. an aerosol activation step.

A textbook activation scheme would compute the local supersaturation from
the cloud-base updraft velocity and step through Köhler theory for the
local aerosol size distribution. JAX-GCM has neither: MACv2-SP gives
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

**Type**: Prognostic TKE (Turbulent Kinetic Energy) scheme based on Brinkop & Roeckner (1995)

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
     - Minimum TKE value (m2/s2)
     - 1.0e-6
   * - ``cchar``
     - Charnock constant for ocean roughness
     - 0.018

.. admonition:: Gap vs. ICON-A

   ICON-A uses the Total Turbulent Energy (TTE) scheme of Mauritsen et al. (2007), which tracks both kinetic and potential turbulent energy. The JAX-GCM implementation uses the simpler Brinkop & Roeckner (1995) TKE scheme (prognostic kinetic energy only). The TTE scheme provides better representation of stable boundary layers and the transition between convective and stable regimes.


Surface Physics
^^^^^^^^^^^^^^^

**Type**: Multi-surface tile scheme with separate treatment of ocean, sea ice, and land

**Description**: Computes turbulent fluxes of momentum, heat, and moisture between the surface and lowest atmospheric level. Each surface type has independent prognostic temperature and flux calculations; grid-box mean fluxes are area-weighted across the three tile fractions (``water_fraction``, ``sea_ice_fraction``, ``land_fraction``) which sum to 1.

**Key features**:

- **Per-tile temperatures**: ocean uses ``forcing.sea_surface_temperature``; sea ice uses ``min(SST, ctfreez=271.38 K)`` (saline freezing point) where ``sice > 0``, else SST; land uses ``forcing.stl_am`` with a 6.5 K/km dry orographic lapse correction (see "Boundary-condition workarounds" below)
- **Ocean**: Mixed layer with prescribed SST, Charnock roughness length parameterisation
- **Sea Ice**: Multi-layer thermodynamics (2 default layers), snow on ice, melting/freezing
- **Land**: Multi-layer soil model (4 default layers), vegetation temperature, soil moisture
- **Exchange coefficients**: Monin-Obukhov similarity theory with bulk Richardson number stability correction. Businger-Dyer Φ_m, Φ_h are < 1 under unstable conditions so that the bulk exchange coefficient (κ²/(ln·Φ_m·Φ_h)·U) is *enhanced* — the textbook free-convection result. (See ``jcm.physics.surface.echam.turbulent_fluxes.compute_stability_functions`` and the test ``test_unstable_stability_functions``.)
- **Implicit damping**: An implicit-Euler factor ``1 / (1 + K·dt/dz_sfc)`` is applied to the sensible-heat, latent-heat, and momentum tendencies at the lowest model level to keep the explicit step stable when ``K·dt/dz_sfc > 2`` (which is easily violated over rough terrain at ECHAM-tuned exchange coefficients). This stands in for the implicit surface BC of the vdiff tridiagonal solve that ECHAM/ICON use natively but that JCM's explicit pipeline can't currently express.

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
   * - ``ocean_depth``
     - Ocean mixed layer depth (m)
     - 50.0
   * - ``n_ice_layers``
     - Number of sea ice layers
     - 2
   * - ``n_soil_layers``
     - Number of soil layers
     - 4

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
     - Dynamics (modal orography), surface (lapse-correction lower bound), :py:class:`EchamSSO`
   * - ``lsm`` (terrain.nc)
     - Static
     - ``terrain.fmask`` — tile-fraction split between ocean / sea-ice / land
   * - ``orostd``, ``orosig``, ``orogam``, ``orothe``, ``oropic``, ``oroval`` (terrain.nc)
     - Static (optional)
     - SSO descriptors for :py:class:`EchamSSO`. Filled with zeros if absent.
   * - ``stl`` (forcing.nc) → ``forcing.stl_am``
     - 12-month climatology
     - Land-tile surface temperature in :func:`apply_surface` (passed through unmodified). When generated from the JSBACH IC file (``ic_land_soil_T63GR15_*.nc``, field ``surf_temp``), this is the real ECHAM/JSBACH land T at the model's orography.
   * - ``sst`` (forcing.nc) → ``forcing.sea_surface_temperature``
     - 12-month climatology
     - Ocean-tile surface temperature; also caps the sea-ice tile via ``min(sst, ctfreez = 271.38 K)`` (the saline freezing point — this is a physical constraint, not a workaround).
   * - ``icec`` (forcing.nc) → ``forcing.sice_am``
     - 12-month climatology
     - Sea-ice tile fraction (clipped to ``[0, 1 − fmask]`` at apply time).
   * - ``alb`` (forcing.nc) → ``forcing.alb0``
     - Static (annual mean)
     - Bare-land surface albedo for the radiation backends.
   * - ``soilw_am`` (forcing.nc)
     - 12-month climatology
     - Soil-moisture initial state for the land-tile column.
   * - ``snowc`` (forcing.nc) → ``forcing.snowc_am``
     - 12-month climatology
     - Snow cover (clipped to plausible range at load time).

Generating BC files from ECHAM input
""""""""""""""""""""""""""""""""""""

``utils/convert_echam_bc.py`` converts the standard ECHAM input files into the JCM ``terrain.nc`` + ``forcing.nc`` pair. Typical invocation::

    python utils/convert_echam_bc.py \
        --surface T63GR15_jan_surf.nc \
        --sst     T63_amipsst_1979-2008_mean.nc \
        --sic     T63_amipsic_1979-2008_mean.nc \
        --land-init ic_land_soil_T63GR15_1976.nc \
        --out-dir jcm/data/bc/t63/

When ``--land-init`` is provided (the JSBACH initial-conditions file from a standard ECHAM dataset), the ``stl`` field uses the real monthly land-surface temperature climatology and the soil-moisture / snow fields use ``init_moist`` / ``snow`` rather than the AMIP-SST extrapolation. Without ``--land-init``, ``stl`` falls back to AMIP SST extrapolated over land — fine for short development runs, but the ~+30 K bias over the Tibetan and Antarctic plateaus has historically driven multi-day stability failures, so the JSBACH-backed file should be used for any climate-style integration.

``ForcingData.from_file`` runs a one-time sanity check on the loaded fields (range bounds, finiteness, and a heuristic that flags the AMIP-extrapolation case at high orography). Hard violations raise; soft ones print a warning and continue.

.. admonition:: Gap vs. ICON-A

   ICON-A couples to the JSBACH land surface model (Reick et al., 2013), which includes dynamic vegetation, phenology, carbon cycle, and detailed soil hydrology. JAX-GCM uses a simplified multi-layer soil model without interactive vegetation or carbon. The ocean is prescribed (AMIP-style) rather than coupled to an ocean model.


Gravity Wave Drag
^^^^^^^^^^^^^^^^^

The gravity-wave drag system in JAX-GCM is split into three coexisting schemes
under :py:mod:`jcm.physics.gravity_waves`. The default ECHAM physics factory
(:py:func:`jcm.physics.echam.echam_terms.echam_physics`) wires :py:class:`EchamHines`
and :py:class:`EchamSSO` into the term list; the :py:class:`EchamSimpleGwd`
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
   as Python kwargs to :py:func:`sso_drag`.

   Real SSO descriptor data (``orostd``, ``orosig``, ``orogam``,
   ``orothe``, ``oropic``, ``oroval``) is not yet plumbed through
   :py:class:`TerrainData`; the wiring uses placeholders derived from
   ``terrain.orog`` and ``terrain.fmask``. The activation gate
   (``ppic-pmea > gpicmea`` AND ``pstd > gstd``) keeps drag at zero over
   ocean automatically.

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

**Description**: Provides a computationally efficient representation of anthropogenic aerosol effects using 9 prescribed plumes representing major global emission regions, plus a natural background component.

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
   * - ``nplumes``
     - Number of anthropogenic plumes
     - 9
   * - ``aod_spmx``
     - Maximum AOD at 550 nm per plume
     - [0.30, 0.15, ...]
   * - ``ssa550``
     - Single scattering albedo at 550 nm per plume
     - [0.92, 0.95, ...]
   * - ``asy550``
     - Asymmetry parameter at 550 nm per plume
     - [0.65, 0.68, ...]
   * - ``angstrom``
     - Angstrom exponent per plume
     - [1.8, 1.5, ...]
   * - ``background_aod``
     - Natural background AOD at 550 nm
     - 0.02

.. admonition:: Note vs. ICON-A

   ICON-A typically uses the Kinne et al. (2013) aerosol climatology or the MACv2-SP scheme. The JAX-GCM implementation uses MACv2-SP with the addition of Angstrom spectral scaling (matching the Fortran implementation) and aerosol-cloud coupling through the CDNC modification of both cloud optics and microphysics autoconversion.


Chemistry
^^^^^^^^^

**Type**: Simplified prescribed chemistry

**Description**: Provides trace gas concentrations for radiation. Ozone is prescribed from a relaxation profile; CO2 and CH4 are specified as well-mixed gases.

**Key Features**:

- **Ozone**: Vertical profile with stratospheric maximum, relaxed toward climatological values
- **CO2**: Well-mixed tracer with configurable growth rate
- **CH4**: Exponential decay with specified lifetime

**Configurable Parameters** (:py:class:`ChemistryParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``o3_max_vmr``
     - Maximum ozone VMR (ppbv)
     - 8000.0
   * - ``o3_scale_height``
     - Ozone scale height (km)
     - 7.0
   * - ``co2_vmr``
     - CO2 volume mixing ratio (ppmv)
     - 420.0
   * - ``ch4_surface_vmr``
     - Surface methane VMR (ppbv)
     - 1900.0
   * - ``ch4_lifetime``
     - Methane lifetime (years)
     - 9.0

.. admonition:: Gap vs. ICON-A

   ICON-A uses prescribed monthly-mean ozone climatologies (e.g., from CMIP6) or can be coupled to interactive chemistry (ICON-ART). JAX-GCM uses a simplified analytical ozone profile with relaxation. For climate applications requiring accurate stratospheric heating, prescribed 3D ozone fields would be preferable.


Forcing and Boundary Conditions
--------------------------------

**Description**: Manages time-varying boundary conditions that drive the model surface:

- **Sea Surface Temperature (SST)**: Prescribed from climatology or constant profile
- **Sea Ice Concentration**: Prescribed from climatology
- **Snow Cover**: Prescribed from climatology
- **Soil Moisture**: Prescribed from climatology
- **Surface Albedo**: Annual-mean bare-land albedo
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
   from jcm.physics.echam.parameters import Parameters

   # Get default parameters
   params = Parameters.default()

   # Modify convection parameters
   params = params.with_convection(
       tau=7200.0,        # Slower CAPE closure (2 hours)
       entrpen=2.0e-4     # Stronger entrainment
   )

   # Modify radiation parameters
   params = params.with_radiation(
       solar_constant=1360.0
   )

   # Ensure all physics timesteps match model dt
   params = params.with_timestep(dt_seconds=1800.0)

   # Create composable ECHAM physics with all standard terms
   physics = echam_physics(parameters=params)


Composable Physics API
-----------------------

The ECHAM physics is composable: each parameterization is a
``PhysicsTerm`` (``flax.nnx.Module``), and ``echam_physics()`` returns a
``ComposableEchamPhysics`` (a subclass of ``ComposablePhysics``) with the
standard ordering wired up. Schemes can be swapped in or out without
touching the orchestrator:

.. code-block:: python

   from jcm.physics.echam.echam_terms import echam_physics

   # Create composable ECHAM physics with all standard terms
   physics = echam_physics(parameters=params)

   # Use neural network radiation emulator instead of grey radiation
   physics = echam_physics(radiation_scheme="emulated")

   # Remove gravity waves for a simplified configuration
   physics = echam_physics().remove("gravity_waves")

   # Replace a single term
   from jcm.physics.echam.echam_terms import EchamRadiationRRTMGP
   physics = echam_physics().replace("radiation", EchamRadiationRRTMGP())

Each ECHAM term is a ``PhysicsTerm`` subclass (``flax.nnx.Module``) with lazy
imports — the underlying ECHAM physics functions are imported at call time,
keeping startup fast and avoiding circular dependencies.

The ``ComposableEchamPhysics`` subclass automatically handles ECHAM's column
vectorization: it reshapes the 3D state to ``(nlev, ncols)`` format before
iterating terms, and reshapes accumulated tendencies back to 3D afterward.

Module Locations
^^^^^^^^^^^^^^^^

After the directory reorganization, ECHAM process modules live under their
respective process directories:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Process
     - Module path
   * - Radiation (grey 2-stream)
     - ``jcm.physics.radiation.grey_two_stream``
   * - Radiation (RRTMGP)
     - ``jcm.physics.radiation.rrtmgp``
   * - Convection
     - ``jcm.physics.convection.tiedtke_nordeng``
   * - Clouds (fraction)
     - ``jcm.physics.clouds.sundqvist``
   * - Microphysics (1-moment)
     - ``jcm.physics.clouds.echam_1m``
   * - Surface
     - ``jcm.physics.surface.echam``
   * - Vertical Diffusion
     - ``jcm.physics.vertical_diffusion.tte_tke``
   * - Gravity Waves
     - ``jcm.physics.gravity_waves.hines``
   * - Aerosol (MACv2-SP)
     - ``jcm.physics.aerosol.macv2_sp``
   * - Chemistry
     - ``jcm.physics.chemistry.simple_chemistry``
   * - Diagnostics (WMO tropopause)
     - ``jcm.physics.diagnostics.wmo_tropopause``

ECHAM-specific infrastructure (parameters, coordinates, ``PhysicsData``)
remains at ``jcm.physics.echam``.


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

- Designed for 40 vertical levels with hybrid sigma-pressure coordinates
- Parameters are tuned for this configuration; performance may vary at other resolutions
- Sub-grid processes (e.g., turbulence, convection) scale with level count

**Simplifications Relative to ICON-A**:

- Radiation uses 2 SW + 3 LW bands instead of 14 SW + 16 LW bands with correlated-k optics
- TKE turbulence scheme instead of TTE (Total Turbulent Energy) scheme
- Simplified land surface model instead of JSBACH with interactive vegetation
- Analytical ozone profile instead of prescribed 3D monthly climatology
- No McICA sub-grid cloud variability treatment in radiation
- No sub-stepping for microphysics sedimentation

**Time Steps**:

- Recommended time step: 30 minutes for T31 resolution
- All physics sub-schemes share the model timestep (set via ``Parameters.with_timestep()``)
- Shorter time steps needed for higher resolutions

**Forcing Data**:

- Supports daily climatological or constant boundary conditions
- Assumes 365-day year for climatological forcing
- SST and sea ice are prescribed (AMIP-style), not predicted
- Aerosol emissions controllable via temporal forcing weights

**Domain**:

- Global model (no regional capability)
- Spectral dynamical core with Gaussian grid for physics


Comparison with SPEEDY Physics
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - SPEEDY
     - ICON
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
     - 40 (typical)
     - 47--95
   * - Radiation Bands
     - 2 SW, 3 LW
     - 2 SW, 3 LW
     - 14 SW, 16 LW
   * - Cloud Scheme
     - Diagnostic
     - Prognostic (single-moment)
     - Prognostic (single/two-moment)
   * - Convection
     - Simplified Tiedtke
     - Tiedtke-Nordeng
     - Tiedtke-Nordeng
   * - Turbulence
     - Relaxation-based
     - Prognostic TKE
     - TTE (Mauritsen et al.)
   * - Surface
     - Bulk, 2 types
     - Multi-layer, 3 types
     - JSBACH + HD model
   * - Aerosols
     - Fixed climatology
     - MACv2-SP interactive
     - MACv2-SP / HAM
   * - Chemistry
     - None (optional CO2)
     - Simplified (O3, CO2, CH4)
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
