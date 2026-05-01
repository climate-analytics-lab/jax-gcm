ICON Physics Package
=====================

Overview
--------

The ICON (ICOsahedral Non-hydrostatic) physics package provides comprehensive atmospheric parameterizations based on the MPI-M physics package used in the ICON Earth System Model. The parameterizations originate from the ECHAM6 general circulation model and have been adapted for the ICON dynamical core framework. JAX-GCM's implementation is a pure JAX translation that maintains the physical fidelity of the ICON physics while adding full differentiability and GPU/TPU acceleration.

The ICON physics in JAX-GCM follows the model description of Giorgetta et al. (2018), with simplifications to the radiation scheme (reduced spectral bands) and the addition of the MACv2-SP aerosol scheme (Stevens et al., 2017) with aerosol-cloud coupling.

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

The ICON physics package includes the following components, executed in sequence:

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

**Type**: Simplified two-stream radiative transfer with gas, cloud, and aerosol optics

**Reference Model**: PSrad (Pincus & Stevens, 2013), which wraps the RRTM gas optics (Iacono et al., 2008). The full ICON-A uses 14 shortwave bands with 112 g-points and 16 longwave bands with 140 g-points. JAX-GCM uses a reduced band configuration for computational efficiency.

**Description**: Computes shortwave (solar) and longwave (terrestrial) radiative heating rates at each model level. Includes explicit treatment of gas absorption (H2O, CO2, O3, CH4, N2O), cloud optical properties (liquid and ice), and aerosol effects.

**Key Features**:

- Two-stream solver for shortwave and longwave fluxes
- Shortwave: 2 bands (visible 0.2--0.69 um, near-IR 0.69--2.5 um)
- Longwave: 3 bands (window 10--350 cm-1, CO2 350--500 cm-1, H2O 500--2500 cm-1)
- Gas optics for H2O, CO2, O3, CH4, N2O
- Mie scattering for liquid cloud droplets with wavelength-dependent refractive index
- Ice crystal optics based on Yang et al. (2013) and Baum et al. (2014)
- Aerosol direct effects with Angstrom spectral scaling
- Aerosol indirect effects via CDNC modification of cloud droplet effective radius
- Solar geometry from ``jax_solar`` (zenith angle, day/night, orbital parameters)

**Process**:

1. Compute solar zenith angle and TOA insolation from date, latitude, longitude
2. Calculate gas optical depths for SW and LW bands
3. Calculate cloud optical properties (liquid + ice) with Mie/parameterized scattering
4. Scale aerosol AOD spectrally using Angstrom exponent: AOD(lambda) = AOD(550nm) * (lambda/0.55)^(-alpha)
5. Combine gas, cloud, and aerosol optical properties
6. Solve two-stream equations for upward and downward fluxes
7. Convert net fluxes to heating rates

**Configurable Parameters** (:py:class:`RadiationParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``dt_rad``
     - Radiation time step (s)
     - 3600.0
   * - ``solar_constant``
     - Total solar irradiance (W/m2)
     - 1361.0
   * - ``n_sw_bands``
     - Number of shortwave bands
     - 2
   * - ``n_lw_bands``
     - Number of longwave bands
     - 3
   * - ``co2_vmr``
     - CO2 volume mixing ratio
     - 400e-6
   * - ``ch4_vmr``
     - CH4 volume mixing ratio
     - 1.8e-6
   * - ``n2o_vmr``
     - N2O volume mixing ratio
     - 0.32e-6

.. admonition:: Gap vs. ICON-A

   The full ICON-A PSrad scheme uses 14 SW and 16 LW bands with correlated-k gas optics and the Monte Carlo Independent Column Approximation (McICA) for sub-grid cloud variability. JAX-GCM currently uses 2 SW and 3 LW bands with simplified gas optics. Upgrading to RRTMGP-compatible band structure would improve spectral accuracy.


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

**Type**: Single-moment bulk microphysics based on Lohmann & Roeckner (1996)

**Description**: Represents conversion processes between water vapor, cloud liquid, cloud ice, rain, and snow. Rain and snow fluxes are diagnosed within each column (not advected).

**Key Processes**:

1. **Autoconversion**: Cloud water to rain via Khairoutdinov & Kogan (2000) parameterization

   - P_aut = 1350 * qc^2.47 * N_c^(-1.79), where N_c is cloud droplet number concentration
   - Sensitive to aerosol-modified CDNC via the Twomey effect

2. **Accretion**: Collection of cloud droplets by raindrops
3. **Aggregation**: Collection of ice crystals by snow (Levkov et al., 1992)
4. **Ice nucleation**: Temperature-dependent ice crystal formation
5. **Melting/Freezing**: Temperature-dependent phase transitions at the melting level
6. **Sedimentation**: Terminal velocity parameterizations for rain, snow, and ice
7. **Evaporation/Sublimation**: Subsaturation-driven evaporation of precipitation

**Configurable Parameters** (:py:class:`MicrophysicsParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``ccraut``
     - Autoconversion rate coefficient
     - 1350.0
   * - ``ccraut_exp_qc``
     - Autoconversion exponent for cloud water
     - 2.47
   * - ``ccraut_exp_nc``
     - Autoconversion exponent for droplet number
     - -1.79
   * - ``cdnc_base``
     - Base cloud droplet number concentration (1/m3)
     - 100e6
   * - ``ccsacl``
     - Accretion coefficient
     - 0.1
   * - ``v_rain``
     - Rain terminal velocity (m/s)
     - 5.0
   * - ``v_snow``
     - Snow terminal velocity (m/s)
     - 1.0
   * - ``v_ice``
     - Ice crystal sedimentation velocity (m/s)
     - 0.1

.. admonition:: Gap vs. ICON-A

   ICON-A uses the Lohmann & Roeckner (1996) single-moment scheme with refinements from Lohmann (2004). A two-moment scheme (Seifert & Beheng, 2006) is available as an option but not default. The JAX-GCM implementation uses the KK2000 autoconversion, which is more modern than the original Sundqvist (1978) autoconversion in ECHAM6/ICON-A. This is a deliberate choice to improve aerosol-cloud sensitivity.


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

**Description**: Computes turbulent fluxes of momentum, heat, and moisture between the surface and lowest atmospheric level. Each surface type has independent prognostic temperature and flux calculations; grid-box mean fluxes are area-weighted.

**Key Features**:

- **Ocean**: Mixed layer with prescribed SST, Charnock roughness length parameterization
- **Sea Ice**: Multi-layer thermodynamics (2 default layers), snow on ice, melting/freezing
- **Land**: Multi-layer soil model (4 default layers), vegetation temperature, soil moisture
- **Exchange Coefficients**: Monin-Obukhov similarity theory with bulk Richardson number stability correction
- **Grid-Box Mean**: Area-weighted averaging across three surface types

**Configurable Parameters** (:py:class:`SurfaceParameters`):

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

.. admonition:: Gap vs. ICON-A

   ICON-A couples to the JSBACH land surface model (Reick et al., 2013), which includes dynamic vegetation, phenology, carbon cycle, and detailed soil hydrology. JAX-GCM uses a simplified multi-layer soil model without interactive vegetation or carbon. The ocean is prescribed (AMIP-style) rather than coupled to an ocean model.


Gravity Wave Drag
^^^^^^^^^^^^^^^^^

**Type**: Combined orographic (Lott & Miller, 1997; Lott, 1999) and non-orographic (Hines, 1997) gravity wave drag

**Description**: Parameterizes the effects of unresolved gravity waves on the resolved flow. Orographic waves are generated by flow over sub-grid topography; non-orographic waves represent spontaneous gravity wave radiation.

**Key Features**:

- Orographic source from sub-grid standard deviation of topography
- Non-orographic source with configurable launch spectrum
- Wave breaking based on critical Richardson number
- Momentum deposition creates wind and temperature tendencies
- Brunt-Vaisala frequency for wave propagation

**Configurable Parameters** (:py:class:`GravityWaveParameters`):

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Default
   * - ``gkdrag``
     - Orographic gravity wave drag coefficient
     - 0.05
   * - ``gkwake``
     - Orographic low-level wake drag coefficient
     - 0.05
   * - ``grcrit``
     - Critical non-dimensional mountain height
     - 0.5
   * - ``ric``
     - Critical Richardson number for wave breaking
     - 0.25
   * - ``ruwmax``
     - Maximum non-orographic wave stress (Pa)
     - 0.1

.. admonition:: Gap vs. ICON-A

   The implementation follows the Lott & Miller (1997) scheme for orographic drag and the Hines (1997) parameterization for non-orographic waves. ICON-A has additional refinements including directional sensitivity and low-level blocking effects.


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

   from jcm.physics.icon.icon_terms import icon_physics
   from jcm.physics.icon.parameters import Parameters

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

   # Create composable ICON physics with all standard terms
   physics = icon_physics(parameters=params)


Composable Physics API
-----------------------

The ICON physics is composable: each parameterization is a
``PhysicsTerm`` (``flax.nnx.Module``), and ``icon_physics()`` returns a
``ComposableIconPhysics`` (a subclass of ``ComposablePhysics``) with the
standard ordering wired up. Schemes can be swapped in or out without
touching the orchestrator:

.. code-block:: python

   from jcm.physics.icon.icon_terms import icon_physics

   # Create composable ICON physics with all standard terms
   physics = icon_physics(parameters=params)

   # Use neural network radiation emulator instead of grey radiation
   physics = icon_physics(radiation_scheme="emulated")

   # Remove gravity waves for a simplified configuration
   physics = icon_physics().remove("gravity_waves")

   # Replace a single term
   from jcm.physics.icon.icon_terms import IconRadiationRRTMGP
   physics = icon_physics().replace("radiation", IconRadiationRRTMGP())

Each ICON term is a ``PhysicsTerm`` subclass (``flax.nnx.Module``) with lazy
imports — the underlying ICON physics functions are imported at call time,
keeping startup fast and avoiding circular dependencies.

The ``ComposableIconPhysics`` subclass automatically handles ICON's column
vectorization: it reshapes the 3D state to ``(nlev, ncols)`` format before
iterating terms, and reshapes accumulated tendencies back to 3D afterward.

Module Locations
^^^^^^^^^^^^^^^^

After the directory reorganization, ICON process modules live under their
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
     - ``jcm.physics.surface.icon``
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

ICON-specific infrastructure (parameters, coordinates, ``PhysicsData``)
remains at ``jcm.physics.icon``.


Scientific References
---------------------

The ICON physics parameterizations are based on the following key publications:

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

- See :doc:`getting_started` for examples of running models with ICON physics
- See :doc:`speedy_physics` for comparison with the simpler SPEEDY physics package
- See :doc:`api` for detailed API documentation of individual parameterizations
