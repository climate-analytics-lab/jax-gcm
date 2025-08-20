Speedy Variable Translation
===========================

Physicsstate
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``geopotential``
     - m²/s²
     - ``phi``
     - geopotential
   * - ``normalized_surface_pressure``
     - 1
     - ``psa``
     - normalized surface pressure
   * - ``specific_humidity``
     - g/kg
     - ``qa``
     - Specific humidity
   * - ``temperature``
     - K
     - ``ta``
     - temperature
   * - ``u_wind``
     - m/s
     - ``ua``
     - U-wind
   * - ``v_wind``
     - m/s
     - ``va``
     - V-wind


Condensation
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``dqlsc``
     - g/kg/s
     - ``dqlsc``
     - Specific humidity tendency due to large-scale condensation
   * - ``dtlsc``
     - K/s
     - ``dtlsc``
     - Temperature tendency due to large-scale condensation
   * - ``precls``
     - kg/m²/s
     - ``precls``
     - Precipitation due to large-scale condensation


Convection
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``cbmf``
     - kg/m²/s
     - ``cbmf``
     - Cloud-base mass flux
   * - ``iptop``
     - model level index
     - ``itop``
     - Top of convection (layer index)
   * - ``precnv``
     - g/m²/s
     - ``precnv``
     - Convective precipitation [g/(m^2 s)]
   * - ``se``
     - J/kg
     - ``se``
     - dry static energy


Date
----

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``model_step``
     - 1
     - ``-``
     - Model step counter
   * - ``model_year``
     - year
     - ``model_datetime``
     - The model's current year
   * - ``tyear``
     - year
     - ``tyear``
     - Fractional time of year


Humidity
--------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``qsat``
     - g/kg
     - ``qsat``
     - saturation specific humidity
   * - ``rh``
     - 1
     - ``rh``
     - relative humidity


Land_model
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``stl_am``
     - K
     - ``stl_am``
     - Land surface temperature used by the atmospheric model
   * - ``stl_lm``
     - K
     - ``stl_lm``
     - Land surface temperature calculated by the land model


Longwave_rad
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``dfabs``
     - W/m²
     - ``dfabs``
     - Flux of long-wave radiation absorbed in each atmospheric layer
   * - ``ftop``
     - W/m²
     - ``ftop``
     - Net downward flux at top of atmosphere


Mod_radcon
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``ablco2``
     - 1
     - ``ablco2``
     - CO2 absorptivity
   * - ``alb_l``
     - 1
     - ``alb_l``
     - Daily-mean albedo over land (bare-land + snow)
   * - ``alb_s``
     - 1
     - ``alb_s``
     - Daily-mean albedo over sea (open sea + sea ice)
   * - ``albsfc``
     - 1
     - ``albsfc``
     - Combined surface albedo (land + sea)
   * - ``flux``
     - W/m²
     - ``flux``
     - Radiative flux in different spectral bands
   * - ``snowc``
     - 1
     - ``snowc``
     - Effective snow cover (fraction)
   * - ``st4a``
     - W/m²
     - ``st4a``
     - Blackbody emission from full and half atmospheric levels
   * - ``stratc``
     - W/m²
     - ``stratc``
     - Stratospheric correction term
   * - ``tau2``
     - 1
     - ``tau2``
     - Transmissivity of atmospheric layers


Shortwave_rad
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``cloudc``
     - 1
     - ``cloudc``
     - Total cloud cover
   * - ``cloudstr``
     - 1
     - ``clstr``
     - Stratiform cloud cover
   * - ``compute_shortwave``
     - bool
     - ``compute_shortwave``
     - Flag to compute shortwave radiation
   * - ``dfabs``
     - W/m²
     - ``dfabs``
     - Flux of long-wave radiation absorbed in each atmospheric layer
   * - ``fsol``
     - W/m²
     - ``fsol``
     - Solar radiation at the top
   * - ``ftop``
     - W/m²
     - ``ftop``
     - Net downward flux at top of atmosphere
   * - ``gse``
     - W/m²/K
     - ``gse``
     - Vertical gradient of dry static energy
   * - ``icltop``
     - model level index
     - ``icltop``
     - Cloud top level
   * - ``ozone``
     - ppmv
     - ``ozone``
     - Ozone concentration in lower stratosphere
   * - ``ozupp``
     - Dobson Unit
     - ``ozupp``
     - Ozone depth in upper stratosphere
   * - ``qcloud``
     - 1
     - ``qcloud``
     - Equivalent specific humidity of clouds
   * - ``rsds``
     - W/m²
     - ``fsfcd``
     - Total downward flux of short-wave radiation at the surface
   * - ``rsns``
     - W/m²
     - ``fsfc``
     - Net downward flux of short-wave radiation at the surface
   * - ``stratz``
     - W/m²
     - ``stratz``
     - Polar night cooling in the stratosphere
   * - ``zenit``
     - radian
     - ``zenit``
     - The zenith angle


Surface_flux
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Jax Variable
     - Units
     - Speedy Equivalent
     - Description
   * - ``evap``
     - kg/m²/s
     - ``evap``
     - Evaporation
   * - ``hfluxn``
     - W/m²
     - ``hfluxn``
     - Net downward heat flux
   * - ``rlds``
     - W/m²
     - ``slrd``
     - Downward flux of long-wave radiation at the surface
   * - ``rlns``
     - W/m²
     - ``-``
     - Net upward flux of long-wave radiation at the surface
   * - ``rlus``
     - W/m²
     - ``slru``
     - Upward flux of long-wave radiation at the surface
   * - ``shf``
     - W/m²
     - ``shf``
     - Sensible heat flux
   * - ``t0``
     - K
     - ``t0``
     - Near-surface temperature
   * - ``tsfc``
     - K
     - ``tsfc``
     - Surface temperature
   * - ``tskin``
     - K
     - ``tskin``
     - Skin surface temperature
   * - ``u0``
     - m/s
     - ``u0``
     - Near-surface u-wind
   * - ``ustr``
     - N/m²
     - ``ustr``
     - u-stress
   * - ``v0``
     - m/s
     - ``v0``
     - Near-surface v-wind
   * - ``vstr``
     - N/m²
     - ``vstr``
     - v-stress


