"""Data structures and types for surface physics.

This module defines the key data structures used in surface processes,
following the ICON model structure for land-atmosphere exchange.
"""

from typing import NamedTuple
import jax.numpy as jnp
import tree_math

# ECHAM6.3 has one longwave surface emissivity, ``cemiss = 0.996``
# (``mo_radiation_parameters.f90::cemiss``, a PARAMETER), and applies it to
# every surface: the radiation's surface boundary
# (``mo_psrad_interface.f90``, ``zsemiss = cemiss`` in all longwave bands), the
# downward longwave over the tile mix (``mo_surface_boundary.f90::
# longwave_down_rad``), the land balance (``mo_surface_land.f90::land_rad``,
# ``update_surfacetemp.f90``; JSBACH's ``Emissivity`` in
# ``mo_jsbach_constants.f90``) and the longwave re-emission
# (``radheat.f90``). Water, ice and land are not distinguished.
ECHAM_SURFACE_EMISSIVITY = 0.996


@tree_math.struct
class SurfaceParameters:
    """Parameters for surface physics schemes."""
    
    # Surface types. The tile COUNT (jax-gcm#999, removed) is not itself a
    # field: every tile-shaped array's ``nsfc_type`` axis comes from its own
    # shape, never from a stored count (see ``EchamSurface.__call__``'s
    # ``nsfc_type = 3`` — a layout constant, not read from here either).
    iwtr: int              # Index for water surface
    iice: int              # Index for ice surface
    ilnd: int              # Index for land surface

    # Removed (jax-gcm#999): ``ml_depth``/``rho_water``/``cp_water`` (mixed-
    # layer ocean) and ``rho_ice``/``cp_ice``/``conduct_ice`` (sea ice) had
    # no consumer. jcm's ocean and sea-ice surface temperatures are
    # prescribed boundary conditions (see ``ocean.py``'s module docstring:
    # "Ocean temperature itself is prescribed ... Slab/mixed-layer ocean
    # and ocean-atmosphere coupling fluxes live outside this repo"), so
    # there is no thermodynamic scheme for these constants to drive; they
    # were declared ahead of a scheme that was never built.

    # Surface radiation parameters
    emissivity: float      # Surface emissivity [-]
    stefan_boltzmann: float  # Stefan-Boltzmann constant [W/m²/K⁴]
    
    # Roughness lengths [m]
    z0_water: float        # Water surface
    z0_ice: float          # Ice surface
    z0_land: float         # Land surface (can be spatially varying)
    
    # Turbulent exchange parameters
    von_karman: float      # von Kármán constant [-]
    min_wind_speed: float  # Minimum wind speed for flux calculations [m/s]

    @classmethod
    def default(cls, iwtr=0, iice=1, ilnd=2,
                 emissivity=ECHAM_SURFACE_EMISSIVITY, stefan_boltzmann=5.67e-8,
                 z0_water=1e-4, z0_ice=1e-3, z0_land=0.1,
                 von_karman=0.4, min_wind_speed=1.0) -> 'SurfaceParameters':
        """Return default surface parameters"""
        return cls(
            iwtr=jnp.array(iwtr),
            iice=jnp.array(iice),
            ilnd=jnp.array(ilnd),
            emissivity=jnp.array(emissivity),
            stefan_boltzmann=jnp.array(stefan_boltzmann),
            z0_water=jnp.array(z0_water),
            z0_ice=jnp.array(z0_ice),
            z0_land=jnp.array(z0_land),
            von_karman=jnp.array(von_karman),
            min_wind_speed=jnp.array(min_wind_speed)
        )


class SurfaceState(NamedTuple):
    """Surface state variables."""
    
    # Surface temperatures
    temperature: jnp.ndarray       # Surface temperature [K] (ncol, nsfc_type)
    temperature_rad: jnp.ndarray   # Radiative temperature [K] (ncol,)
    
    # Surface fractions
    fraction: jnp.ndarray          # Surface type fraction [-] (ncol, nsfc_type)
    
    # Ocean/lake variables
    ocean_temp: jnp.ndarray        # Ocean temperature [K] (ncol,)
    ocean_u: jnp.ndarray           # Ocean u-velocity [m/s] (ncol,)
    ocean_v: jnp.ndarray           # Ocean v-velocity [m/s] (ncol,)
    
    # Sea ice variables
    ice_thickness: jnp.ndarray     # Ice thickness [m] (ncol, nice_layers)
    ice_temp: jnp.ndarray          # Ice temperature [K] (ncol, nice_layers)
    snow_depth: jnp.ndarray        # Snow depth [m] (ncol,)
    
    # Land surface variables
    soil_temp: jnp.ndarray         # Soil temperature [K] (ncol, nsoil_layers)
    soil_moisture: jnp.ndarray     # Soil moisture [-] (ncol, nsoil_layers)
    vegetation_temp: jnp.ndarray   # Vegetation temperature [K] (ncol,)
    
    # Surface properties
    roughness_momentum: jnp.ndarray  # Momentum roughness [m] (ncol, nsfc_type)
    roughness_heat: jnp.ndarray      # Heat roughness [m] (ncol, nsfc_type)
    albedo_visible_direct: jnp.ndarray    # Visible direct albedo [-] (ncol, nsfc_type)
    albedo_visible_diffuse: jnp.ndarray   # Visible diffuse albedo [-] (ncol, nsfc_type)
    albedo_nir_direct: jnp.ndarray        # NIR direct albedo [-] (ncol, nsfc_type)
    albedo_nir_diffuse: jnp.ndarray       # NIR diffuse albedo [-] (ncol, nsfc_type)


class AtmosphericForcing(NamedTuple):
    """Atmospheric forcing at the surface."""
    
    # State at lowest atmospheric level
    temperature: jnp.ndarray       # Air temperature [K] (ncol,)
    humidity: jnp.ndarray          # Specific humidity [kg/kg] (ncol,)
    u_wind: jnp.ndarray           # Zonal wind [m/s] (ncol,)
    v_wind: jnp.ndarray           # Meridional wind [m/s] (ncol,)
    pressure: jnp.ndarray         # Pressure [Pa] (ncol,)
    # Surface (lowest interface) pressure [Pa] (ncol,): the pressure ECHAM's
    # precalc_ocean/_ice/_land divide the tile saturation by (paphm1).
    surface_pressure: jnp.ndarray
    
    # Radiation fluxes
    sw_downward: jnp.ndarray      # Downward shortwave [W/m²] (ncol,)
    lw_downward: jnp.ndarray      # Downward longwave [W/m²] (ncol,)
    
    # Precipitation
    rain_rate: jnp.ndarray        # Rain rate [kg/m²/s] (ncol,)
    snow_rate: jnp.ndarray        # Snow rate [kg/m²/s] (ncol,)
    
    # Exchange coefficients from turbulence scheme
    exchange_coeff_heat: jnp.ndarray     # Heat exchange [m²/s] (ncol, nsfc_type)
    exchange_coeff_moisture: jnp.ndarray # Moisture exchange [m²/s] (ncol, nsfc_type)
    exchange_coeff_momentum: jnp.ndarray # Momentum exchange [m²/s] (ncol, nsfc_type)

    # Height of THIS forcing's temperature/humidity/wind above the surface
    # [m] (ncol,) — the model's lowest full level, not literally 10 m.
    # ``compute_surface_diagnostics``'s 2 m blend and bulk Richardson number
    # (jax-gcm#1140) divide by this; a review on that issue caught both
    # reading a hard-coded 10 m assumption while ``EchamSurface.__call__``
    # hands this struct the LOWEST MODEL LEVEL's values (tens to hundreds of
    # metres). Defaults to 10.0 for callers that genuinely supply a 10 m
    # input (and every pre-existing test that built this struct without an
    # opinion on height), so this is additive, not a breaking field.
    reference_height: jnp.ndarray = 10.0


class SurfaceFluxes(NamedTuple):
    """Surface fluxes computed by surface physics."""
    
    # Energy fluxes [W/m²]
    sensible_heat: jnp.ndarray     # Sensible heat flux (ncol, nsfc_type)
    latent_heat: jnp.ndarray       # Latent heat flux (ncol, nsfc_type)
    longwave_net: jnp.ndarray      # Net longwave flux (ncol, nsfc_type)
    shortwave_net: jnp.ndarray     # Net shortwave flux (ncol, nsfc_type)
    ground_heat: jnp.ndarray       # Ground heat flux (ncol, nsfc_type)
    
    # Momentum fluxes [N/m²]
    momentum_u: jnp.ndarray        # U-momentum flux (ncol, nsfc_type)
    momentum_v: jnp.ndarray        # V-momentum flux (ncol, nsfc_type)
    
    # Mass fluxes [kg/m²/s]
    evaporation: jnp.ndarray       # Evaporation rate (ncol, nsfc_type)
    transpiration: jnp.ndarray     # Transpiration rate (ncol, nsfc_type)
    
    # Grid-box mean fluxes [various units]
    sensible_heat_mean: jnp.ndarray    # Grid-box mean sensible heat [W/m²] (ncol,)
    latent_heat_mean: jnp.ndarray      # Grid-box mean latent heat [W/m²] (ncol,)
    momentum_u_mean: jnp.ndarray       # Grid-box mean u-momentum [N/m²] (ncol,)
    momentum_v_mean: jnp.ndarray       # Grid-box mean v-momentum [N/m²] (ncol,)
    evaporation_mean: jnp.ndarray      # Grid-box mean evaporation [kg/m²/s] (ncol,)


class SurfaceTendencies(NamedTuple):
    """Tendencies for surface prognostic variables."""
    
    # Temperature tendencies [K/s]
    surface_temp_tendency: jnp.ndarray     # Surface temperature (ncol, nsfc_type)
    ocean_temp_tendency: jnp.ndarray       # Ocean temperature (ncol,)
    ice_temp_tendency: jnp.ndarray         # Ice temperature (ncol, nice_layers)
    soil_temp_tendency: jnp.ndarray        # Soil temperature (ncol, nsoil_layers)
    
    # Mass tendencies
    ice_thickness_tendency: jnp.ndarray    # Ice thickness [m/s] (ncol, nice_layers)
    snow_depth_tendency: jnp.ndarray       # Snow depth [m/s] (ncol,)
    soil_moisture_tendency: jnp.ndarray    # Soil moisture [1/s] (ncol, nsoil_layers)


class SurfaceDiagnostics(NamedTuple):
    """Diagnostic variables from surface physics."""
    
    # Standard diagnostics
    temperature_2m: jnp.ndarray        # 2m temperature [K] (ncol,)
    humidity_2m: jnp.ndarray           # 2m specific humidity [kg/kg] (ncol,)
    dewpoint_2m: jnp.ndarray           # 2m dew point [K] (ncol,)
    
    # Surface layer properties
    friction_velocity: jnp.ndarray     # Friction velocity [m/s] (ncol,)
    richardson_number: jnp.ndarray     # Surface Richardson number [-] (ncol,)
    surface_layer_height: jnp.ndarray # Surface layer height [m] (ncol,)
    
    # Energy balance components [W/m²]
    net_radiation: jnp.ndarray         # Net radiation (ncol,)
    radiation_balance: jnp.ndarray     # Radiation balance (ncol,)
    energy_balance_residual: jnp.ndarray  # Energy balance residual (ncol,)
    
    # Tile-specific diagnostics
    temperature_2m_tile: jnp.ndarray   # 2m temperature per tile [K] (ncol, nsfc_type)
    humidity_2m_tile: jnp.ndarray      # 2m humidity per tile [kg/kg] (ncol, nsfc_type)


class SurfaceResistances(NamedTuple):
    """Surface resistances for flux calculations."""

    # Aerodynamic resistances [s/m]
    aerodynamic_heat: jnp.ndarray      # Heat transfer (ncol, nsfc_type)
    aerodynamic_moisture: jnp.ndarray  # Moisture transfer (ncol, nsfc_type)
    aerodynamic_momentum: jnp.ndarray  # Momentum transfer (ncol, nsfc_type)

    # Surface resistances [s/m]
    surface_moisture: jnp.ndarray      # Surface moisture resistance (ncol, nsfc_type)
    canopy_resistance: jnp.ndarray     # Canopy resistance (ncol,)
    soil_resistance: jnp.ndarray       # Soil resistance (ncol,)

    # Stability corrections [-]
    stability_heat: jnp.ndarray        # Heat stability function (ncol, nsfc_type)
    stability_momentum: jnp.ndarray    # Momentum stability function (ncol, nsfc_type)


@tree_math.struct
class SurfaceData:
    """Diagnostics produced by ``EchamSurface`` and seeded by
    ``EchamBoundaryConditions``.

    The flux fields are written by the surface term per step; the
    surface temperature, skin temperature, and roughness length are
    blended from the forcing-driven boundary conditions.
    """

    # Surface fluxes
    sensible_heat_flux: jnp.ndarray  # Sensible heat flux [W/m²] (ncols,)
    latent_heat_flux: jnp.ndarray    # Latent heat flux [W/m²] (ncols,)
    # Positive-down: the momentum flux INTO the surface, positive with
    # the wind (republished from the vdiff-delivered
    # VDiffSurfaceFluxes.stress_u/v — same sign as the #754
    # surface-exchange contract).
    momentum_flux_u: jnp.ndarray     # U momentum flux into surface [N/m²] (ncols,)
    momentum_flux_v: jnp.ndarray     # V momentum flux into surface [N/m²] (ncols,)

    # Surface temperatures
    surface_temperature: jnp.ndarray # Surface temperature [K] (ncols,)
    skin_temperature: jnp.ndarray    # Skin temperature [K] (ncols,)

    # Surface properties
    roughness_length: jnp.ndarray    # Surface roughness length [m] (ncols,)

    # Evaporation
    evaporation: jnp.ndarray         # Evaporation [kg/m²/s] (ncols,)
    # Since the surface exchange became the bottom boundary row of the vdiff
    # implicit solve, the published evaporation IS the moisture the column
    # received (the ECHAM ``pev_vdiff`` identity), so this field now always
    # equals ``evaporation``. Kept for API stability — the Tiedtke
    # moisture-budget closure anchors to it (there is no raw-vs-damped
    # distinction and no ``imp_moist`` factor).
    effective_evaporation: jnp.ndarray  # == evaporation [kg/m²/s] (ncols,)

    # Exchange coefficients
    ch: jnp.ndarray                  # Heat exchange coefficient [1] (ncols,)
    cm: jnp.ndarray                  # Momentum exchange coefficient [1] (ncols,)

    # --- The land tile (jcm.physics.surface.echam.jsbach_land, #979) -------
    # PROGNOSTIC: the land skin temperature [K], carried from step to step.
    # A value <= 0 means "not yet set" (a cold start, or a checkpoint written
    # before it existed): EchamBoundaryConditions then seeds it from stl_am.
    land_surface_temperature: jnp.ndarray
    # The land tile's energy balance this step [W/m2]:
    # net_radiation = sensible + latent + ground + melt + storage, exactly.
    land_net_radiation: jnp.ndarray       # SW + εLW↓ − εσT⁴, absorbed
    land_sensible_heat_flux: jnp.ndarray  # positive up
    land_latent_heat_flux: jnp.ndarray    # positive up
    ground_heat_flux: jnp.ndarray         # Λ·(T_s − stl_am), into the ground
    snow_melt_heat_flux: jnp.ndarray      # taken up by melt at a capped surface
    land_heat_storage: jnp.ndarray        # C_s·dT_s/dt
    land_evaporation: jnp.ndarray         # land tile [kg/m2/s], positive up
    # Rn − SH − LH: ground + melt + storage with a prognostic skin; with a
    # prescribed one, the heat the prescription supplies or removes.
    land_energy_residual: jnp.ndarray
    # JSBACH's humidity factors of the land tile and their ingredients [1]
    cair: jnp.ndarray
    csat: jnp.ndarray
    water_stress_factor: jnp.ndarray
    bare_soil_humidity: jnp.ndarray
    canopy_conductance: jnp.ndarray       # unstressed, [m/s]
    # The land albedo the radiation last SOLVED with [1]: written by the
    # radiation term on a solve and held between solves, as its
    # surface_sw_down/_sw_up are, so the land's net shortwave is that solve's.
    # A value <= 0 means "not yet set" (a cold start before the first solve, a
    # checkpoint written before it existed, or a radiation term that does not
    # hold it): the land then uses the current step's albedo.
    land_albedo_at_solve: jnp.ndarray

    # --- The reference bulk-aerodynamic estimate (jax-gcm#1118) -----------
    # ``EchamSurface`` also runs ``surface_physics_step`` — the explicit
    # ECHAM-style ocean/ice/land bulk-flux computation, grid-box averaged by
    # ``combine_surface_fluxes`` — using the SAME per-tile exchange
    # coefficients the implicit vdiff solve delivered its own fluxes with.
    # It is NOT fed back into the column (the implicit solve already
    # delivered the real fluxes above; adding a second delivery path would
    # double-count), but publishing it gives a cheap, always-available
    # diagnostic of how closely an explicit one-shot bulk formula would have
    # tracked the implicit solution — previously this computation ran every
    # step and its result was simply discarded (jax-gcm#1118's "finding
    # worth recording"). Grid-box mean only (``combine_surface_fluxes``'s
    # ``*_mean`` fields); the per-tile breakdown is not published here — see
    # the design doc for what remains.
    bulk_sensible_heat_flux: jnp.ndarray  # [W/m²], positive up
    bulk_latent_heat_flux: jnp.ndarray    # [W/m²], positive up
    bulk_evaporation: jnp.ndarray         # [kg/m²/s], positive up
    bulk_momentum_flux_u: jnp.ndarray     # [N/m²]
    bulk_momentum_flux_v: jnp.ndarray     # [N/m²]

    # --- The surface-layer diagnostic bundle (jax-gcm#1140) ---------------
    # ``EchamSurface`` also runs ``compute_surface_diagnostics`` as part of
    # ``surface_physics_step`` (the same bulk-aerodynamic reference pass the
    # BULK_FIELDS above come from — these use the bulk ``combined_fluxes``,
    # not the implicit-solve-delivered ones); its ``SurfaceDiagnostics``
    # bundle was computed every step and discarded entirely. Three of its
    # fields are deliberately NOT republished here, each because a better
    # (or the only non-fake) version of it is already public elsewhere, and
    # publishing a second, disagreeing one would be worse than publishing
    # nothing (see ``docs/source/science/surface.md``'s *Surface-layer
    # diagnostics* section):
    #   * ``friction_velocity`` — a second, bulk-estimate u* alongside the
    #     live ``vertical_diffusion.surface_friction_velocity`` (the one
    #     the water tile's Charnock roughness actually reads).
    #   * ``surface_layer_height`` — a bare ``100.0`` constant for every
    #     column, never computed from anything; the live PBL height is
    #     ``vertical_diffusion.pbl_height``.
    #   * ``net_radiation``/``radiation_balance`` — an exact duplicate pair
    #     (``radiation_balance`` is assigned ``net_radiation`` verbatim,
    #     read nowhere) that is also exactly recoverable from fields already
    #     public: ``net_radiation == energy_balance_residual +
    #     bulk_sensible_heat_flux + bulk_latent_heat_flux`` by the
    #     ``compute_surface_diagnostics`` definition of the residual, so
    #     publishing it would add zero information once the residual below
    #     is public.
    temperature_2m: jnp.ndarray            # 2 m temperature [K]
    humidity_2m: jnp.ndarray               # 2 m specific humidity [kg/kg]
    dewpoint_2m: jnp.ndarray                # 2 m dew point [K]
    richardson_number: jnp.ndarray         # bulk Richardson number [-]
    # Rn - SH - LH of the bulk-aerodynamic reference estimate (NOT the real
    # column, which closes by construction via the implicit solve): how far
    # that one-shot explicit formula sits from its own radiative balance.
    energy_balance_residual: jnp.ndarray   # [W/m²]
    temperature_2m_tile: jnp.ndarray       # per tile [K] (ncols, nsfc_type)
    humidity_2m_tile: jnp.ndarray          # per tile [kg/kg] (ncols, nsfc_type)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        nsfc_type = 3  # water, ice, land
        return cls(
            sensible_heat_flux=jnp.zeros(nodal_shape),
            latent_heat_flux=jnp.zeros(nodal_shape),
            momentum_flux_u=jnp.zeros(nodal_shape),
            momentum_flux_v=jnp.zeros(nodal_shape),
            surface_temperature=jnp.zeros(nodal_shape),
            skin_temperature=jnp.zeros(nodal_shape),
            roughness_length=jnp.zeros(nodal_shape),
            evaporation=jnp.zeros(nodal_shape),
            effective_evaporation=jnp.zeros(nodal_shape),
            ch=jnp.zeros(nodal_shape),
            cm=jnp.zeros(nodal_shape),
            **{name: jnp.zeros(nodal_shape) for name in LAND_FIELDS},
            **{name: jnp.zeros(nodal_shape) for name in BULK_FIELDS},
            **{name: jnp.zeros(nodal_shape) for name in DIAG_SCALAR_FIELDS},
            **{name: jnp.zeros(nodal_shape + (nsfc_type,))
               for name in DIAG_TILE_FIELDS},
        )

    def copy(self, **kwargs):
        new_data = {
            'sensible_heat_flux': self.sensible_heat_flux,
            'latent_heat_flux': self.latent_heat_flux,
            'momentum_flux_u': self.momentum_flux_u,
            'momentum_flux_v': self.momentum_flux_v,
            'surface_temperature': self.surface_temperature,
            'skin_temperature': self.skin_temperature,
            'roughness_length': self.roughness_length,
            'evaporation': self.evaporation,
            'effective_evaporation': self.effective_evaporation,
            'ch': self.ch,
            'cm': self.cm,
            **{name: getattr(self, name) for name in LAND_FIELDS},
            **{name: getattr(self, name) for name in BULK_FIELDS},
            **{name: getattr(self, name) for name in DIAG_SCALAR_FIELDS},
            **{name: getattr(self, name) for name in DIAG_TILE_FIELDS},
        }
        new_data.update(kwargs)
        return SurfaceData(**new_data)


#: The land-tile fields of :class:`SurfaceData`, in declaration order.
LAND_FIELDS = (
    "land_surface_temperature", "land_net_radiation", "land_sensible_heat_flux",
    "land_latent_heat_flux", "ground_heat_flux", "snow_melt_heat_flux",
    "land_heat_storage", "land_evaporation", "land_energy_residual", "cair", "csat",
    "water_stress_factor", "bare_soil_humidity", "canopy_conductance",
    "land_albedo_at_solve",
)

#: The reference bulk-aerodynamic-flux fields of :class:`SurfaceData`
#: (jax-gcm#1118), in declaration order.
BULK_FIELDS = (
    "bulk_sensible_heat_flux", "bulk_latent_heat_flux", "bulk_evaporation",
    "bulk_momentum_flux_u", "bulk_momentum_flux_v",
)

#: The grid-mean surface-layer diagnostic fields of :class:`SurfaceData`
#: (jax-gcm#1140), shaped ``(ncols,)``.
DIAG_SCALAR_FIELDS = (
    "temperature_2m", "humidity_2m", "dewpoint_2m", "richardson_number",
    "energy_balance_residual",
)

#: The per-tile surface-layer diagnostic fields of :class:`SurfaceData`
#: (jax-gcm#1140), shaped ``(ncols, nsfc_type)``.
DIAG_TILE_FIELDS = ("temperature_2m_tile", "humidity_2m_tile")

#: CF attributes of the bulk-flux outputs (``surface.<field>``).
BULK_OUTPUT_ATTRS = {
    "surface.bulk_sensible_heat_flux": {
        "units": "W m-2",
        "long_name": "grid-mean sensible heat flux from the explicit ECHAM "
                      "bulk-aerodynamic formula (reference diagnostic, not delivered "
                      "to the column), positive up"},
    "surface.bulk_latent_heat_flux": {
        "units": "W m-2",
        "long_name": "grid-mean latent heat flux from the explicit ECHAM "
                      "bulk-aerodynamic formula (reference diagnostic, not delivered "
                      "to the column), positive up"},
    "surface.bulk_evaporation": {
        "units": "kg m-2 s-1",
        "long_name": "grid-mean evaporation from the explicit ECHAM bulk-aerodynamic "
                      "formula (reference diagnostic, not delivered to the column), "
                      "positive up"},
    "surface.bulk_momentum_flux_u": {
        "units": "N m-2",
        "long_name": "grid-mean zonal momentum flux from the explicit ECHAM "
                      "bulk-aerodynamic formula (reference diagnostic, not delivered "
                      "to the column)"},
    "surface.bulk_momentum_flux_v": {
        "units": "N m-2",
        "long_name": "grid-mean meridional momentum flux from the explicit ECHAM "
                      "bulk-aerodynamic formula (reference diagnostic, not delivered "
                      "to the column)"},
}

#: CF attributes of the land-tile outputs (``surface.<field>``).
LAND_OUTPUT_ATTRS = {
    "surface.land_surface_temperature": {
        "units": "K", "long_name": "land skin temperature (prognostic)"},
    "surface.land_net_radiation": {
        "units": "W m-2", "long_name": "net radiation absorbed by the land tile"},
    "surface.land_sensible_heat_flux": {
        "units": "W m-2", "long_name": "land-tile sensible heat flux, positive up"},
    "surface.land_latent_heat_flux": {
        "units": "W m-2", "long_name": "land-tile latent heat flux, positive up"},
    "surface.ground_heat_flux": {
        "units": "W m-2", "long_name": "ground heat flux into the soil below the skin"},
    "surface.snow_melt_heat_flux": {
        "units": "W m-2",
        "long_name": "energy taken up by melt at a snow- or glacier-covered land surface"},
    "surface.land_heat_storage": {
        "units": "W m-2", "long_name": "heat storage rate of the land skin layer"},
    "surface.land_energy_residual": {
        "units": "W m-2",
        "long_name": "land-tile net radiation minus sensible and latent heat flux"},
    "surface.land_evaporation": {
        "units": "kg m-2 s-1", "long_name": "land-tile evaporation, positive up"},
    "surface.cair": {"units": "1", "long_name": "JSBACH land humidity factor of the air (cair)"},
    "surface.csat": {
        "units": "1", "long_name": "JSBACH land humidity factor of the surface saturation (csat)"},
    "surface.water_stress_factor": {
        "units": "1", "long_name": "JSBACH root-zone water-stress factor"},
    "surface.bare_soil_humidity": {
        "units": "1", "long_name": "relative humidity of the bare soil surface"},
    "surface.canopy_conductance": {
        "units": "m s-1", "long_name": "unstressed canopy conductance"},
    "surface.land_albedo_at_solve": {
        "units": "1",
        "long_name": "land albedo of the last radiation solve, which the land's net "
                     "shortwave uses until the next"},
}

#: CF attributes of the surface-layer diagnostic outputs (``surface.<field>``,
#: jax-gcm#1140). All from the same bulk-aerodynamic reference pass as
#: :data:`BULK_OUTPUT_ATTRS` (the SAME ``combined_fluxes``/exchange
#: coefficients, not a second computation), so "reference diagnostic, not
#: delivered to the column" applies here too, EXCEPT ``temperature_2m``/
#: ``humidity_2m``/``dewpoint_2m``/the ``_tile`` pair: those read only the
#: step's forcing and exchange coefficients, which (unlike the bulk fluxes)
#: ARE the ones that drove the real implicit solve, so they are not a
#: separate reference estimate the way the fluxes are.
DIAG_OUTPUT_ATTRS = {
    "surface.temperature_2m": {
        "units": "K",
        "long_name": "grid-mean 2 m air temperature, linearly interpolated "
                     "between the surface and the lowest-model-level values"},
    "surface.humidity_2m": {
        "units": "kg kg-1",
        "long_name": "grid-mean 2 m specific humidity: each tile's own 2 m "
                     "humidity (interpolated between its surface anchor and "
                     "the lowest-model-level value) fraction-weighted into "
                     "the grid mean"},
    "surface.dewpoint_2m": {
        "units": "K",
        "long_name": "grid-mean 2 m dew-point temperature: the exact inverse "
                     "of the Tetens saturation law applied to humidity_2m "
                     "(jcm.physics.convection.saturation.dewpoint_temperature)"},
    "surface.richardson_number": {
        "units": "1",
        "long_name": "grid-mean bulk Richardson number of the bulk-aerodynamic "
                     "reference surface layer (reference diagnostic, not the "
                     "Richardson number the live exchange coefficients use)"},
    "surface.energy_balance_residual": {
        "units": "W m-2",
        "long_name": "net radiation minus sensible and latent heat flux of the "
                     "explicit ECHAM bulk-aerodynamic reference estimate "
                     "(reference diagnostic; the real column's implicit solve "
                     "closes its own energy budget by construction)"},
    "surface.temperature_2m_tile": {
        "units": "K", "long_name": "2 m air temperature per surface tile"},
    "surface.humidity_2m_tile": {
        "units": "kg kg-1", "long_name": "2 m specific humidity per surface tile"},
}