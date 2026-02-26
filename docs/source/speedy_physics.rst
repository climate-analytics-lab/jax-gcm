SPEEDY Physics Package
======================

Overview
--------

The SPEEDY (Simplified Parameterizations, primitivE-Equation DYnamics) physics package provides intermediate-complexity atmospheric parameterizations suitable for climate modeling and machine learning applications. SPEEDY was originally developed by Franco Molteni at ICTP and has been widely used for studying atmospheric dynamics and climate variability.

JAX-GCM's implementation is a pure JAX translation that maintains the physical fidelity of SPEEDY while adding full differentiability and GPU/TPU acceleration.

Key Characteristics
^^^^^^^^^^^^^^^^^^^

- **Computational Efficiency**: Simplified physics allows for fast simulations
- **Physical Realism**: Captures essential atmospheric processes despite simplifications
- **Vertical Resolution**: Designed for 8 vertical levels (can work with other configurations)
- **Time-Varying Forcing**: Supports daily climatological or constant boundary conditions
- **Differentiability**: Fully compatible with JAX automatic differentiation

Physics Parameterizations
--------------------------

The SPEEDY physics package includes the following components, executed in sequence:

1. Convection (Simplified Tiedtke Scheme)
2. Large-Scale Condensation
3. Cloud Diagnostics
4. Shortwave Radiation
5. Longwave Radiation
6. Surface Fluxes
7. Vertical Diffusion

Each parameterization is described in detail below.

Convection
^^^^^^^^^^

**Type**: Simplified mass-flux scheme based on Tiedtke (1993)

**Description**: Represents subgrid-scale moist convection using a bulk mass-flux approach. The scheme diagnoses convectively unstable grid boxes where saturation moist static energy decreases with height.

**Key Features**:

- Primary and secondary mass fluxes
- Entrainment and detrainment
- Convective precipitation
- Temperature and moisture tendencies

**Activation Criteria**:

Convection activates when:

1. Conditional instability exists (saturation moist static energy decreases upward)
2. Either:

   - Actual convective instability (moist static energy decreases upward), OR
   - Relative humidity exceeds threshold in boundary layer

**Configurable Parameters** (:py:class:`ConvectionParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``psmin``
     - Minimum surface pressure for convection
     - 0.8
   * - ``trcnv``
     - Relaxation time toward reference state (hours)
     - 6.0
   * - ``rhil``
     - RH threshold for secondary mass flux
     - 0.7
   * - ``rhbl``
     - RH threshold in boundary layer
     - 0.9
   * - ``entmax``
     - Maximum entrainment fraction
     - 0.5
   * - ``smf``
     - Secondary to primary mass flux ratio
     - 0.8

Large-Scale Condensation
^^^^^^^^^^^^^^^^^^^^^^^^

**Description**: Represents stratiform precipitation and clouds through a relaxation scheme toward saturated conditions when relative humidity exceeds a threshold.

**Process**:

1. Check if relative humidity exceeds threshold
2. Relax specific humidity toward saturation
3. Convert excess moisture to precipitation
4. Release latent heat

**Vertical Variation**: RH threshold varies from ``rhlsc`` at the surface to ``rhlsc + drhlsc`` at model top, ensuring more precipitation in upper levels.

**Configurable Parameters** (:py:class:`CondensationParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``trlsc``
     - Relaxation time for specific humidity (hours)
     - 4.0
   * - ``rhlsc``
     - Maximum RH threshold at surface
     - 0.9
   * - ``drhlsc``
     - Vertical range of RH threshold
     - 0.1
   * - ``rhblsc``
     - RH threshold for boundary layer
     - 0.95

Clouds
^^^^^^

**Description**: Diagnostic cloud cover based on relative humidity and precipitation. Two types of clouds are diagnosed:

**Convective Clouds**:

- Based on relative humidity
- Weight applied to square root of precipitation
- Cloud cover increases with RH and precipitation

**Stratiform Clouds**:

- Additional component based on static stability
- Diagnosed from vertical gradient of dry static energy
- Higher over land (minimum cover ``clsminl``)

**Cloud Top**: Determined as the highest level with significant cloud cover.

**Configurable Parameters** (in :py:class:`ShortwaveRadiationParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``rhcl1``
     - RH for zero cloud cover
     - 0.30
   * - ``rhcl2``
     - RH for full cloud cover
     - 1.00
   * - ``qacl``
     - Specific humidity threshold
     - 0.20
   * - ``wpcl``
     - Precipitation weight (mm/day)⁻⁰·⁵
     - 0.2
   * - ``pmaxcl``
     - Maximum precipitation contribution (mm/day)
     - 10.0
   * - ``clsmax``
     - Maximum stratiform cloud cover
     - 0.60
   * - ``clsminl``
     - Minimum stratiform cover over land
     - 0.15

Shortwave Radiation
^^^^^^^^^^^^^^^^^^^

**Description**: Two-band shortwave radiation scheme (visible and near-IR) with explicit treatment of:

- Rayleigh scattering
- Water vapor absorption
- Cloud albedo and absorption
- Aerosol absorption
- Surface albedo (land, ocean, ice, snow)

**Process**:

1. Compute solar zenith angle and TOA insolation
2. Calculate atmospheric absorption by water vapor and aerosols
3. Account for cloud reflection and absorption
4. Compute surface albedo (varies by surface type)
5. Calculate multiple reflections between surface and clouds
6. Distribute absorbed radiation to atmospheric layers

**Radiation Frequency**: Computed every ``nstrad`` timesteps (typically every 3 hours) to save computation.

**Configurable Parameters** (:py:class:`ShortwaveRadiationParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``albcl``
     - Cloud albedo
     - 0.43
   * - ``albcls``
     - Stratiform cloud albedo
     - 0.50
   * - ``absdry``
     - Dry air absorptivity (visible)
     - 0.033
   * - ``absaer``
     - Aerosol absorptivity (visible)
     - 0.033
   * - ``abswv1``
     - Water vapor absorptivity (band 1)
     - 0.022
   * - ``abswv2``
     - Water vapor absorptivity (band 2)
     - 15.0
   * - ``abscl1``
     - Cloud absorptivity (visible, max)
     - 0.015
   * - ``abscl2``
     - Cloud absorptivity (band 2)
     - 0.15

**Surface Albedo** (:py:class:`ModRadConParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``albsea``
     - Ocean albedo
     - 0.07
   * - ``albice``
     - Sea ice albedo
     - 0.60
   * - ``albsn``
     - Snow albedo
     - 0.60

Longwave Radiation
^^^^^^^^^^^^^^^^^^

**Description**: Three-band longwave scheme representing:

- Window region (transparent to water vapor)
- Water vapor band 1 (weak absorption)
- Water vapor band 2 (strong absorption)

**Process**:

1. Compute blackbody emission at each level
2. Calculate water vapor absorptivity in each band
3. Account for cloud emissivity
4. Integrate upward and downward fluxes separately
5. Apply surface emissivity

**Band Weights**: Fixed fractions of blackbody spectrum allocated to each band.

**Configurable Parameters** (:py:class:`ShortwaveRadiationParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``ablwin``
     - Window band absorptivity
     - 0.3
   * - ``ablwv1``
     - Water vapor absorptivity (band 1)
     - 0.7
   * - ``ablwv2``
     - Water vapor absorptivity (band 2)
     - 50.0
   * - ``ablcl1``
     - Thick cloud absorptivity (window)
     - 12.0
   * - ``ablcl2``
     - Thin cloud absorptivity
     - 0.6

**Surface Parameters** (:py:class:`ModRadConParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``epslw``
     - PBL emission fraction
     - 0.05
   * - ``emisfc``
     - Surface emissivity
     - 0.98

Surface Fluxes
^^^^^^^^^^^^^^

**Description**: Bulk aerodynamic formulation for turbulent fluxes of momentum, heat, and moisture between the surface and atmosphere.

**Separate Treatment**:

- **Land**: Prognostic skin temperature from energy balance; moisture availability factor
- **Ocean**: Prescribed SST; unlimited moisture availability

**Stability Correction**: Adjusts exchange coefficients based on static stability (Richardson number approach).

**Energy Balance** (land only):

- Solves for skin temperature satisfying: Net radiation = Sensible heat + Latent heat + Ground heat flux
- Includes diurnal cycle correction
- Heat conduction to soil layer

**Configurable Parameters** (:py:class:`SurfaceFluxParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``cdl``
     - Momentum drag coefficient (land)
     - 2.4×10⁻³
   * - ``cds``
     - Momentum drag coefficient (sea)
     - 1.0×10⁻³
   * - ``chl``
     - Heat exchange coefficient (land)
     - 1.2×10⁻³
   * - ``chs``
     - Heat exchange coefficient (sea)
     - 0.9×10⁻³
   * - ``vgust``
     - Subgrid wind gust speed (m/s)
     - 5.0
   * - ``dtheta``
     - Potential temp gradient for stability
     - 3.0 K
   * - ``fstab``
     - Stability correction amplitude
     - 0.67
   * - ``ctday``
     - Daily cycle correction
     - 0.01
   * - ``clambda``
     - Soil heat conductivity
     - 7.0
   * - ``lfluxland``
     - Compute land surface temperature
     - True
   * - ``lskineb``
     - Redefine skin temp from energy balance
     - True
   * - ``lscasym``
     - Use asymmetric stability coefficient
     - True

Vertical Diffusion
^^^^^^^^^^^^^^^^^^

**Description**: Represents subgrid-scale vertical mixing by:

1. **Shallow Convection**: Moisture redistribution in marginally unstable conditions
2. **Moisture Diffusion**: Removes sharp vertical RH gradients
3. **Dry Convective Adjustment**: Removes super-adiabatic lapse rates

**Process**:

- Shallow convection diagnosed where moist static energy is nearly neutral
- Reduced in regions of deep convection
- Moisture diffusion applied where RH gradient exceeds threshold
- Super-adiabatic adjustment maintains stability

**Configurable Parameters** (:py:class:`VerticalDiffusionParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``trshc``
     - Shallow convection time scale (hours)
     - 6.0
   * - ``trvdi``
     - Moisture diffusion time scale (hours)
     - 24.0
   * - ``trvds``
     - Super-adiabatic adjustment time scale (hours)
     - 6.0
   * - ``redshc``
     - Shallow convection reduction factor
     - 0.5
   * - ``rhgrad``
     - Maximum RH gradient (d_RH/d_σ)
     - 0.5
   * - ``segrad``
     - Minimum dry static energy gradient
     - 0.1

Forcing and Boundary Conditions
--------------------------------

**Description**: Manages time-varying and constant boundary conditions including:

- Sea surface temperature (SST)
- Sea ice concentration
- Snow cover
- Soil moisture
- Surface albedo
- Orographic parameters

**CO₂ Forcing**: Optional increasing CO₂ concentration over time.

**Configurable Parameters** (:py:class:`ForcingParameters`):

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``increase_co2``
     - Enable time-varying CO₂
     - True
   * - ``co2_year_ref``
     - Reference year for CO₂
     - 1950

Using Custom Parameters
-----------------------

To customize physics parameters:

.. code-block:: python

   from jcm.physics.speedy.speedy_physics import SpeedyPhysics
   from jcm.physics.speedy.params import Parameters
   from jcm.model import Model

   # Get default parameters
   params = Parameters.default()

   # Modify convection parameters
   params = params.copy(
       convection=params.convection.copy(
           trcnv=8.0,  # Slower convection relaxation
           rhbl=0.85   # Lower RH threshold
       )
   )

   # Modify radiation parameters
   params = params.copy(
       shortwave_radiation=params.shortwave_radiation.copy(
           albcl=0.50  # Higher cloud albedo
       )
   )

   # Create physics with custom parameters
   physics = SpeedyPhysics(parameters=params)

   # Use in model
   model = Model(physics=physics)

Viewing All Parameters
^^^^^^^^^^^^^^^^^^^^^^^

To see all parameter values:

.. code-block:: python

   from jcm.physics.speedy.params import Parameters

   params = Parameters.default()
   print(params)

Scientific References
---------------------

The SPEEDY physics parameterizations are based on the following key publications:

1. **Molteni, F.** (2003). Atmospheric simulations using a GCM with simplified physical parametrizations. I: Model climatology and variability in multi-decadal experiments. *Climate Dynamics*, 20, 175-191. https://doi.org/10.1007/s00382-002-0268-2

2. **Tiedtke, M.** (1993). Representation of clouds in large-scale models. *Monthly Weather Review*, 121(11), 3040-3061.

3. **Original SPEEDY Documentation**: `SPEEDY User Guide <https://users.ictp.it/~kucharsk/speedy-net.html>`_

4. **Fortran 90 Implementation**: Our JAX implementation references the `speedy.f90 <https://github.com/samhatfield/speedy.f90>`_ version by Sam Hatfield and Leo Saffin.

Assumptions and Limitations
---------------------------

**Vertical Resolution**:

- Designed for 8 vertical levels with specific σ-coordinates
- Performance may vary with different vertical resolutions
- Some parameters are tuned for the standard 8-level configuration

**Simplifications**:

- Two-band shortwave radiation (more sophisticated schemes use 4+ bands)
- Simplified cloud microphysics (no explicit ice/liquid separation)
- Bulk mass-flux convection (no explicit cloud dynamics)
- No aerosol indirect effects on clouds
- No chemistry or interactive trace gases (except optional CO₂ trend)

**Time Steps**:

- Recommended time step: 30 minutes for T31 resolution
- Shorter time steps needed for higher resolutions
- Radiation computed less frequently (typically every 3 hours)

**Forcing Data**:

- Requires either daily climatological or constant boundary conditions
- Assumes 365-day year for climatological forcing
- SST and other boundary conditions are prescribed (not predicted)

**Domain**:

- Global model (no regional capability)
- Assumes spherical geometry

Performance Characteristics
---------------------------

**Computational Cost**: SPEEDY is designed to be ~10-100× faster than state-of-the-art physics packages while maintaining reasonable climatology.

**Physical Realism**:

- Captures large-scale circulation patterns well
- Reasonable representation of tropical variability (e.g., MJO-like features)
- Mean climate biases comparable to some comprehensive GCMs
- Best suited for dynamics studies, sensitivity experiments, and ensemble applications

**Use Cases**:

- ✓ Large ensemble simulations
- ✓ Parameter sensitivity studies
- ✓ Dynamics and variability research
- ✓ ML/AI training data generation
- ✓ Educational purposes
- ✗ Detailed process studies (e.g., cloud microphysics)
- ✗ High-accuracy climate projections

Comparison with Other Physics Packages
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - SPEEDY
     - ICON (future)
     - Comprehensive GCMs
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
     - Flexible
     - 30-100+
   * - Radiation Bands
     - 2 (SW), 3 (LW)
     - Multi-band
     - 10-100+
   * - Cloud Scheme
     - Diagnostic
     - Prognostic
     - Prognostic+Microphysics
   * - Aerosols
     - Fixed climatology
     - Interactive
     - Fully interactive
   * - Use Case
     - Dynamics, ML
     - Research
     - Climate projections

Next Steps
----------

- See :doc:`getting_started` for examples of running models with SPEEDY physics
- See :doc:`api` for detailed API documentation of individual parameterizations
- See :doc:`developer` for information on implementing custom physics packages
