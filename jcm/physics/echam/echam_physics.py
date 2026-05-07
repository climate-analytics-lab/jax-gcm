"""ECHAM physics term functions.

Standalone functions implementing the individual ECHAM parameterizations
(``apply_radiation``, ``apply_clouds_and_microphysics``, etc.). These are
wrapped by ``ComposablePhysics`` term classes in ``echam_terms.py``; there
is no monolithic orchestrator class — use ``echam_physics()`` from
``echam_terms`` to build a composable ECHAM physics package.

As schemes migrate to scheme-named ``PhysicsTerm`` classes living next to
their underlying numerics (Phase 2+ of the composable refactor), the
``apply_*`` functions in this file shrink. ``apply_convection`` has been
extracted to :class:`jcm.physics.convection.tiedtke_nordeng.TiedtkeConvection`.
"""

import logging

import jax
from jax import jit
import jax.numpy as jnp
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm import constants as physical_constants

# Import physics modules (will be implemented progressively)
from jcm.physics.clouds.sundqvist import shallow_cloud_scheme
from jcm.physics.clouds.echam_1m import cloud_microphysics
from jcm.physics.echam.parameters import Parameters
from jcm.physics.surface.echam import surface_physics_step, initialize_surface_state
from jcm.physics.surface.echam.surface_types import AtmosphericForcing
from jcm.physics.echam.echam_physics_data import PhysicsData
from jcm.physics.aerosol.spa import spa_activated_cdnc

logger = logging.getLogger(__name__)


def _column_vector(value: jnp.ndarray, ncols: int) -> jnp.ndarray:
    """Return a vmapped scalar diagnostic as one value per column."""
    return jnp.reshape(value, (ncols,))


@jit
def _prepare_common_physics_state(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Prepare common physics variables that are used by multiple physics modules.
    
    This reduces code duplication by computing pressure levels, heights, air density,
    and other commonly needed variables once for all physics modules.
    
    Args:
        state: Physics state variables (already in 2D format [nlev, ncols])
        boundaries: Boundary conditions (already updated with time-varying conditions)
        geometry: Model geometry
        
    Returns:
        Dictionary with common physics variables

    """
    p0 = physical_constants.p0
    
    # Calculate pressure levels from surface pressure and hybrid (a, b) coefficients.
    # Works for pure sigma (a=0, b=sigma) and ICON hybrid (a + b*P_s).
    surface_pressure = state.normalized_surface_pressure * p0  # Convert to Pa
    pressure_levels = physics_data.echam_coords.calculate_pressure_full(surface_pressure)
    pressure_half = physics_data.echam_coords.calculate_pressure_half(surface_pressure)
    
    # Convert geopotential to height
    height_levels = state.geopotential / physical_constants.grav

    # Calculate height at interfaces (half levels)
    # Internal interfaces are midpoints between full levels
    height_half_internal = (height_levels[1:] + height_levels[:-1]) / 2

    # Top interface: extrapolate using the same spacing as the top layer
    # This maintains consistent layer thickness at the top
    top_layer_thickness = height_levels[0] - height_half_internal[0]
    height_top = height_levels[0] + top_layer_thickness

    # Surface interface: use actual surface height (from geopotential at lowest level)
    # For sigma coordinates, assume surface is at orography height
    # A reasonable approximation is half the lowest layer below the lowest full level
    bottom_layer_thickness = height_half_internal[-1] - height_levels[-1]
    height_surface = height_levels[-1] - bottom_layer_thickness

    height_half = jnp.concatenate((
        height_top[jnp.newaxis],
        height_half_internal,
        height_surface[jnp.newaxis]), axis=0)

    # Calculate air density
    rho = pressure_levels / (physical_constants.rd * state.temperature)
    
    # Calculate layer thickness (clamp to minimum 10m for numerical stability
    # with thin uniform sigma layers)
    dp = jnp.diff(pressure_half, axis=0)
    dz_full = jnp.maximum(dp / (rho * physical_constants.grav), 10.0)
    
    # Calculate relative humidity (Tetens formula; clip T only enough to avoid
    # divide-by-zero at T=29.65K and exp overflow)
    # Wide math-safety clip; NOT a physical-range bound
    T_clip = jnp.clip(state.temperature, 50.0, 500.0)
    q_clip = jnp.maximum(state.specific_humidity, 0.0)
    es = 611.2 * jnp.exp(17.67 * (T_clip - 273.15) / (T_clip - 29.65))
    e = q_clip * pressure_levels / (0.622 + 0.378 * q_clip)
    rel_humidity = e / jnp.maximum(es, 1e-3)

    diagnostic_data = physics_data.diagnostics.copy(
        pressure_full=pressure_levels,
        pressure_half=pressure_half,
        height_full=height_levels,
        height_half=height_half,
        relative_humidity=rel_humidity,
        surface_pressure=surface_pressure,
        air_density=rho,
        layer_thickness=dz_full,
    )

    # Note: chemistry is intentionally not initialized here. ``apply_forcing_data``
    # (the next term in the physics sequence) unconditionally overwrites
    # ``physics_data.chemistry`` with constant GHG concentrations every step,
    # so any initialization work done here would be immediately discarded.
    updated_physics_data = physics_data.copy(diagnostics=diagnostic_data)

    zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return zero_tendencies, updated_physics_data

# Physics term methods


# ``apply_radiation`` was extracted to
# :class:`jcm.physics.radiation.grey_two_stream.GreyTwoStreamRadiation`
# (Phase 3 of the scheme-named-terms refactor). The shared caching gate
# now lives in
# :func:`jcm.physics.radiation.grey_two_stream.radiation_should_compute`
# / :func:`cached_radiation_tendency` so the RRTMGP and NN-emulator
# radiation terms can call them too.


# ``apply_radiation_rrtmgp`` was extracted to
# :class:`jcm.physics.radiation.rrtmgp.RRTMGPRadiation` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_radiation_emulated`` was extracted to
# :class:`jcm.physics.radiation.nn_emulator_scheme.NNEmulatorRadiation`
# (Phase 3 of the scheme-named-terms refactor).


# ``_apply_radiation_inner`` was moved into
# :class:`jcm.physics.radiation.grey_two_stream.GreyTwoStreamRadiation`
# as the term's ``_compute_full`` method.


# ``_apply_radiation_rrtmgp_inner`` was moved into
# :class:`jcm.physics.radiation.rrtmgp.RRTMGPRadiation` as the term's
# ``_compute_full`` method.


# ``_apply_radiation_emulated_inner`` was moved into
# :class:`jcm.physics.radiation.nn_emulator_scheme.NNEmulatorRadiation`
# as the term's ``_compute_full`` method.


# ``apply_convection`` was extracted to
# ``jcm.physics.convection.tiedtke_nordeng.TiedtkeConvection`` (Phase 2 of
# the scheme-named-terms refactor). The numerical implementation moved
# verbatim into the term ``__call__``.


def _cloud_and_microphysics_column(
    temperature, specific_humidity, pressure, qc, qi,
    surface_pressure, air_density, layer_thickness, droplet_number,
    dt, cloud_config, micro_config
):
    """Compute cloud and microphysics for a single column.

    Following ECHAM mo_cloud.f90: condensation, cloud fraction, autoconversion,
    accretion, and precipitation are all computed in a single column sweep.
    This avoids the coupling issues of splitting them into separate calls.

    Tendency accounting (no double counting):
        The cloud scheme computes condensation and applies it within the
        timestep to produce updated cloud water (cloud_state.cloud_water).
        Microphysics then acts on this updated cloud water.

        Both schemes return SEPARATE tendencies that are additive:
        - Cloud:  dqcdt = +condensation,  dqdt = -condensation,  dtedt = +L*condensation/cp
        - Micro:  dqcdt = -autoconversion, dqdt = +evaporation,  dtedt = micro heating/cooling

        The integrator applies: qc_new = qc_old + (cloud_dqcdt + micro_dqcdt) * dt
        This gives: qc_new = 0 + (condensation - autoconversion) * dt

        Moisture is conserved: dq + dqc + precip = 0
        (-condensation + evap) + (condensation - autoconv) + (autoconv - evap) = 0

        The within-timestep cloud water update is used ONLY to provide
        microphysics with a physically meaningful input — it does not
        affect the tendencies returned to the integrator.
    """
    # 1. Cloud fraction and condensation
    cloud_tendencies, cloud_state = shallow_cloud_scheme(
        temperature, specific_humidity, pressure,
        qc, qi, surface_pressure, dt, cloud_config
    )

    # 2. Microphysics acts on the condensation-updated cloud water/ice
    micro_tendencies, micro_state = cloud_microphysics(
        temperature, specific_humidity, pressure,
        cloud_state.cloud_water, cloud_state.cloud_ice,
        cloud_state.cloud_fraction, air_density, layer_thickness,
        droplet_number, dt, micro_config
    )

    return cloud_tendencies, cloud_state, micro_tendencies, micro_state


# ``apply_cloud_fraction`` was extracted to
# :class:`jcm.physics.clouds.sundqvist.SundqvistCloudFraction` (Phase 3
# of the scheme-named-terms refactor).


# ``apply_microphysics_1m`` was extracted to
# :class:`jcm.physics.clouds.echam_1m.Echam1MMicrophysics` (Phase 3 of
# the scheme-named-terms refactor).


@jit
def apply_microphysics_2m(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Run ECHAM 2-moment cloud microphysics.

    Consumes the post-condensation ``qc``/``qi``/``cloud_fraction`` emitted
    by :func:`apply_cloud_fraction` and returns tendencies for the full 2M
    tracer set ``{qc, qi, qnc, qni, qr, qs}``. The orchestrator
    :func:`jcm.physics.clouds.lohmann_2m.cloud_microphysics_2m` chains the
    full ECHAM6 process list: warm precip (KK2000) + mixed-phase
    deposition/condensation + freezing-below-238K + DeMott(2010) INP
    mixed-phase freezing (placeholder for ECHAM's ``het_mxphase_freezing``
    which would need HAM aerosol modes — see #436) + WBF + cold precip +
    a top-down ``lax.scan`` over levels for sedimentation / melting /
    sublimation+evap / precip-flux accumulation, then ECHAM's
    ``update_tendencies_and_important_vars`` for the final tendency
    bookkeeping. Heterogeneous freezing is intentionally simplified to
    DeMott(2010) since JCM does not yet ingest a real IN field.
    """
    from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m

    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    air_density = physics_data.diagnostics.air_density
    layer_thickness = physics_data.diagnostics.layer_thickness
    tke = physics_data.vertical_diffusion.tke
    params_2m = parameters.microphysics_2m

    qc_interim = physics_data.clouds.qc
    qi_interim = physics_data.clouds.qi
    cloud_fraction = physics_data.clouds.cloud_fraction

    # Default any declared-but-missing tracers to zero.
    zeros = jnp.zeros_like(state.temperature)
    qnc = state.tracers.get('qnc', zeros)
    qni = state.tracers.get('qni', zeros)
    qr = state.tracers.get('qr', zeros)
    qs = state.tracers.get('qs', zeros)

    # Aerosol-activated CDNC floor from the MACv2-SP plume CCN
    # concentration via the SPA sublinear power-law (Lin et al. 2025;
    # #374). Output is per-level `(nlev, ncols)` in m^-3 — the column-
    # mean Nccn is broadcast to every level (vertical aerosol structure
    # is not resolved by the simple-plumes scheme). The fit's prefactor
    # and exponent come from `parameters.aerosol` so they remain
    # differentiable for calibration work.
    Nccn = physics_data.aerosol.Nccn  # (ncols,), units cm^-3
    activated_cdnc = spa_activated_cdnc(
        Nccn=Nccn[jnp.newaxis, :],
        cloud_fraction=cloud_fraction,
        prefactor=parameters.aerosol.spa_prefactor,
        exponent=parameters.aerosol.spa_exponent,
    )

    tend_all, surface_rain_flux, surface_snow_flux = jax.vmap(
        cloud_microphysics_2m,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),
        out_axes=(0, 0, 0),
    )(state.temperature, state.specific_humidity, pressure_levels,
      qc_interim, qi_interim, qnc, qni, qr, qs,
      cloud_fraction, air_density, layer_thickness, tke,
      activated_cdnc, dt, params_2m)

    tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=tend_all.dtedt.T,
        specific_humidity=tend_all.dqdt.T,
        tracers={
            'qc': tend_all.dqcdt.T,
            'qi': tend_all.dqidt.T,
            'qnc': tend_all.dqncdt.T,
            'qni': tend_all.dqnidt.T,
            'qr': tend_all.dqrdt.T,
            'qs': tend_all.dqsdt.T,
        },
    )
    # Stash the current-step qnc/qni as the tm1 state so the next call of
    # this term (or downstream update_tendencies_and_important_vars) can
    # read previous-step number concentrations. PhysicsData.clouds is
    # carried forward across timesteps in ComposableEchamPhysics.__call__.
    # Also expose the large-scale surface precip from the column scan as
    # diagnostic ``precip_rain``/``precip_snow`` (kg/m^2/s) — these are the
    # gravitational-fall flux at the bottom of the orchestrator's
    # top-down ``lax.scan``, summing autoconv + accretion + melt - evap
    # contributions across all levels.
    clouds_next = physics_data.clouds.copy(
        qnc_prev=qnc, qni_prev=qni,
        precip_rain=surface_rain_flux,
        precip_snow=surface_snow_flux,
    )
    return tendencies, physics_data.copy(clouds=clouds_next)


@jit
def apply_clouds_and_microphysics(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply cloud scheme and microphysics in a single coupled step.

    Combines condensation → cloud fraction → autoconversion → precipitation
    in one vmapped column call, following ECHAM mo_cloud.f90.
    """
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    surface_pressure = physics_data.diagnostics.surface_pressure
    air_density = physics_data.diagnostics.air_density
    dz = physics_data.diagnostics.layer_thickness
    qc = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    qi = state.tracers.get('qi', jnp.zeros_like(state.temperature))

    # Droplet number concentration from aerosol scheme
    base_cdnc = parameters.microphysics.base_cdnc  # Clean-air baseline CDNC (1/m³)
    cdnc_factor = physics_data.aerosol.cdnc_factor  # (ncols,)
    cdnc_m3 = jnp.ones_like(state.temperature) * base_cdnc * cdnc_factor[jnp.newaxis, :]
    droplet_number_per_kg = cdnc_m3 / air_density  # 1/m³ → 1/kg (for microphysics)

    cloud_config = parameters.clouds
    micro_config = parameters.microphysics

    # Single vmap over columns: cloud + microphysics together
    cloud_tend_all, cloud_state_all, micro_tend_all, micro_state_all = jax.vmap(
        _cloud_and_microphysics_column,
        in_axes=(1, 1, 1, 1, 1, 0, 1, 1, 1, None, None, None),
        out_axes=(0, 0, 0, 0)
    )(state.temperature, state.specific_humidity, pressure_levels,
      qc, qi, surface_pressure, air_density, dz, droplet_number_per_kg,
      dt, cloud_config, micro_config)

    # Combine tendencies: cloud (condensation) + microphysics (autoconversion etc.)
    # These are separate physical processes — see _cloud_and_microphysics_column
    # docstring for the full accounting showing no double counting.
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=cloud_tend_all.dtedt.T + micro_tend_all.dtedt.T,
        specific_humidity=cloud_tend_all.dqdt.T + micro_tend_all.dqdt.T,
        tracers={
            'qc': cloud_tend_all.dqcdt.T + micro_tend_all.dqcdt.T,
            'qi': cloud_tend_all.dqidt.T + micro_tend_all.dqidt.T
        }
    )

    # Update physics data with cloud and microphysics diagnostics
    cloud_data = physics_data.clouds.copy(
        cloud_fraction=cloud_state_all.cloud_fraction.T,
        qc=cloud_state_all.cloud_water.T,
        qi=cloud_state_all.cloud_ice.T,
        precip_rain=micro_state_all.precip_rain,
        precip_snow=micro_state_all.precip_snow,
        droplet_number=cdnc_m3  # Store in 1/m³ for diagnostics/radiation
    )

    diagnostics = physics_data.diagnostics.copy(
        relative_humidity=cloud_state_all.rel_humidity.T,
    )

    updated_physics_data = physics_data.copy(clouds=cloud_data,
                                             diagnostics=diagnostics)

    return physics_tendencies, updated_physics_data


# ``apply_vertical_diffusion`` was extracted to
# :class:`jcm.physics.vertical_diffusion.tte_tke.TteTkeVerticalDiffusion`
# (Phase 3 of the scheme-named-terms refactor).


@jit
def apply_surface(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply surface physics and calculate surface fluxes"""
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    # Per-tile surface temperatures are read directly from forcing below
    # (``ocean_temp``, ``land_temp``, ``ice_surface_temp``); the upstream-
    # blended ``physics_data.surface.surface_temperature`` is no longer
    # consulted here — see comment by ``ocean_temp`` for the reason.

    # Surface tile fractions: water (0), sea ice (1), land (2).
    # Sea ice fraction is taken from prescribed boundary conditions and
    # constrained to the non-land area so that fractions sum to exactly 1.
    nsfc_type = 3
    surface_fractions = jnp.zeros((ncols, nsfc_type))
    land_fraction = terrain.fmask.reshape((ncols,))
    sea_ice_fraction = jnp.clip(forcing.sice_am.reshape((ncols,)), 0.0, 1.0 - land_fraction)
    water_fraction = 1.0 - land_fraction - sea_ice_fraction
    surface_fractions = surface_fractions.at[:, 0].set(water_fraction)
    surface_fractions = surface_fractions.at[:, 1].set(sea_ice_fraction)
    surface_fractions = surface_fractions.at[:, 2].set(land_fraction)

    # Per-tile surface temperatures: boundary SST for ocean, the saline
    # freezing point (``ctfreez = 271.38 K``, ECHAM ``iniphy.f90:71``)
    # capped by SST for sea ice, and ``forcing.stl_am`` for land. Sea
    # ice uses min(SST, ctfreez) because the underlying ocean caps the
    # ice surface temperature physically.
    #
    # Read ``ocean_temp`` and ``land_temp`` straight from the forcing rather
    # than the upstream-blended ``physics_data.surface.surface_temperature``,
    # which is snapped to one-or-the-other via ``where(fmask>0.5)`` in
    # ``EchamBoundaryConditions`` and would feed the wrong T into the minority tile
    # (e.g. the 40% ocean fraction of a fmask=0.6 cell would otherwise
    # use ``stl_am`` instead of ``sst``).
    ocean_temp = forcing.sea_surface_temperature.reshape(ncols)
    ctfreez = 271.38  # K, ECHAM ``iniphy.f90:71`` saline-water freezing
    # ``stl_am`` is the JSBACH land surface temperature climatology
    # (``surf_temp`` from ``ic_land_soil_T63GR15_*.nc``), already at the
    # model's orography — no lapse correction needed. (An earlier workaround
    # subtracted 6.5 K/km · orog because the bundled BCs used ``stl ≈ sst``
    # extrapolated over land — see ``utils/convert_echam_bc.py`` for the
    # path that picks the right field.)
    land_temp = forcing.stl_am.reshape(ncols)
    ice_surface_temp = jnp.where(sea_ice_fraction > 0.0,
                                 jnp.minimum(ocean_temp, ctfreez),
                                 ocean_temp)
    ice_temp = jnp.repeat(ice_surface_temp[:, jnp.newaxis], 2, axis=1)  # 2 ice layers
    soil_temp = jnp.repeat(land_temp[:, jnp.newaxis], 4, axis=1)         # 4 soil layers
    
    surface_state = initialize_surface_state(
        ncols, surface_fractions, ocean_temp, ice_temp, soil_temp, parameters.surface
    )
    
    # Prepare atmospheric forcing
    # Use lowest model level for surface conditions
    atm_temp = state.temperature[-1, :]  # Lowest model level
    atm_qv = state.specific_humidity[-1, :]
    atm_u = state.u_wind[-1, :]
    atm_v = state.v_wind[-1, :]
    atm_p = pressure_levels[-1, :]
    
    # Height of lowest model level above surface
    ref_height = physics_data.diagnostics.height_full[-1, :] - physics_data.diagnostics.height_full[-1, :].min()
    ref_height = jnp.maximum(ref_height, 10.0)  # At least 10m
    
    # Get exchange coefficients from vertical diffusion diagnostics
    nsfc_type = 3
    exchange_coeff_heat = physics_data.vertical_diffusion.surface_exchange_heat.reshape(ncols, nsfc_type)
    exchange_coeff_moisture = physics_data.vertical_diffusion.surface_exchange_moisture.reshape(ncols, nsfc_type)
    exchange_coeff_momentum = physics_data.vertical_diffusion.surface_exchange_momentum.reshape(ncols, nsfc_type)

    atm_forcing = AtmosphericForcing(
        temperature=atm_temp,
        humidity=atm_qv,
        u_wind=atm_u,
        v_wind=atm_v,
        pressure=atm_p,
        sw_downward=physics_data.radiation.surface_sw_down,
        lw_downward=physics_data.radiation.surface_lw_down,
        rain_rate=jnp.zeros(ncols),  # No rain for now
        snow_rate=jnp.zeros(ncols),  # No snow for now
        exchange_coeff_heat=exchange_coeff_heat,
        exchange_coeff_moisture=exchange_coeff_moisture,
        exchange_coeff_momentum=exchange_coeff_momentum
    )
    
    # Apply surface physics to all columns
    fluxes, tendencies, diagnostics = surface_physics_step(
        atm_forcing, surface_state, dt, parameters.surface
    )
    
    # Extract grid-box mean fluxes
    sensible_heat = fluxes.sensible_heat_mean
    latent_heat = fluxes.latent_heat_mean
    tau_u = fluxes.momentum_u_mean
    tau_v = fluxes.momentum_v_mean
    evaporation = fluxes.evaporation_mean
    
    # Convert fluxes to atmospheric tendencies
    # Only the lowest model level is directly affected by surface fluxes
    
    # Air density at surface
    rho_sfc = pressure_levels[-1, :] / (physical_constants.rd * state.temperature[-1, :])

    # Layer thickness at surface (approximate, clamp to minimum 50m to avoid
    # enormous tendencies from thin uniform sigma layers)
    dp_sfc = pressure_levels[-1, :] - pressure_levels[-2, :]
    dz_sfc = jnp.maximum(dp_sfc / (rho_sfc * physical_constants.grav), 50.0)

    # The surface-flux divergence at the bottom level is a linear relaxation
    # toward the surface value with timescale ``dz_sfc / K`` (K = exchange
    # velocity in m/s). An *explicit* time step of size ``dt`` is unstable
    # whenever ``K * dt / dz_sfc > 2`` — and over rough terrain at the
    # ECHAM-tuned exchange coefficients this CFL is easily violated, with
    # the wind flipping sign each step until the column blows up. ECHAM
    # itself avoids this by handling the surface as an implicit BC of the
    # vdiff tridiagonal solve. JCM's explicit pipeline can't do that
    # directly, but we can damp each explicit tendency by the same factor
    # an implicit Euler step would — ``1 / (1 + K*dt/dz_sfc)``. This is
    # exact for the simple linear-relaxation form, recovers the explicit
    # tendency in the small-K*dt limit, and is unconditionally stable.
    #
    # ``surface_fractions`` and ``exchange_coeff_*`` are per-tile (ocean,
    # ice, land); the grid-box-mean exchange velocity is the area-weighted
    # sum.
    ch_grid = jnp.sum(surface_fractions * exchange_coeff_heat, axis=1)
    cm_grid = jnp.sum(surface_fractions * exchange_coeff_momentum, axis=1)
    ce_grid = jnp.sum(surface_fractions * exchange_coeff_moisture, axis=1)
    imp_heat = 1.0 / (1.0 + ch_grid * dt / dz_sfc)
    imp_mom = 1.0 / (1.0 + cm_grid * dt / dz_sfc)
    imp_moist = 1.0 / (1.0 + ce_grid * dt / dz_sfc)

    # Surface flux tendencies (applied to lowest level only).
    temp_tend_sfc = imp_heat * sensible_heat / (rho_sfc * physical_constants.cp * dz_sfc)
    qv_tend_sfc = imp_moist * evaporation / (rho_sfc * dz_sfc)
    u_tend_sfc = imp_mom * (-tau_u) / (rho_sfc * dz_sfc)
    v_tend_sfc = imp_mom * (-tau_v) / (rho_sfc * dz_sfc)
    
    # Initialize tendencies (only surface level affected)
    temp_tend = jnp.zeros_like(state.temperature)
    qv_tend = jnp.zeros_like(state.specific_humidity)
    u_tend = jnp.zeros_like(state.u_wind)
    v_tend = jnp.zeros_like(state.v_wind)
    
    # Apply surface tendencies to lowest level
    temp_tend = temp_tend.at[-1, :].set(temp_tend_sfc)
    qv_tend = qv_tend.at[-1, :].set(qv_tend_sfc)
    u_tend = u_tend.at[-1, :].set(u_tend_sfc)
    v_tend = v_tend.at[-1, :].set(v_tend_sfc)
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=u_tend,
        v_wind=v_tend,
        temperature=temp_tend,
        specific_humidity=qv_tend,
        tracers={}
    )
    
    # Update physics data with surface diagnostics
    # Extract exchange coefficients from atmospheric forcing
    ch = atm_forcing.exchange_coeff_heat[:, 0]  # Heat exchange coefficient
    cm = atm_forcing.exchange_coeff_momentum[:, 0]  # Momentum exchange coefficient
    
    surface_data = physics_data.surface.copy(
        sensible_heat_flux=sensible_heat,
        latent_heat_flux=latent_heat,
        momentum_flux_u=tau_u,
        momentum_flux_v=tau_v,
        evaporation=evaporation,  # Use 'evaporation' not 'evaporation_flux'
        ch=ch,
        cm=cm,
    )
    
    updated_physics_data = physics_data.copy(surface=surface_data)
    
    return physics_tendencies, updated_physics_data

# ``apply_simple_gwd`` was extracted to
# :class:`jcm.physics.gravity_waves.simple.SimpleGwd` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_hines`` was extracted to
# :class:`jcm.physics.gravity_waves.hines.HinesGwd` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_sso`` was extracted to
# :class:`jcm.physics.gravity_waves.sso.LottMillerSso` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_chemistry`` was extracted to
# :class:`jcm.physics.chemistry.SimpleChemistry` (Phase 3 of the
# scheme-named-terms refactor).