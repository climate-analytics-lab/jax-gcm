"""Ocean surface diagnostic fluxes for the ECHAM multi-tile surface scheme.

Provides the bulk-flux pieces — albedo, Charnock roughness, and
turbulent / radiative surface fluxes — that the surface tile aggregator
needs over open water. Ocean temperature itself is prescribed (boundary
SST) in ECHAM's flow, so this module computes diagnostic fluxes only and
returns zero ocean-temperature / ice tendencies. Slab / mixed-layer
ocean and ocean-atmosphere coupling fluxes live outside this repo.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from jcm.physics import thermodynamics
from .surface_types import (
    SurfaceParameters, AtmosphericForcing,
    SurfaceFluxes, SurfaceTendencies
)


@jax.jit
def compute_ocean_albedo(
    solar_zenith_angle: jnp.ndarray,
    params: SurfaceParameters | None = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ocean surface albedo as a function of solar zenith angle.

    Args:
        solar_zenith_angle: Solar zenith angle [rad] (ncol,)
        params: Surface parameters

    Returns:
        Tuple of (albedo_vis_direct, albedo_vis_diffuse,
                 albedo_nir_direct, albedo_nir_diffuse)

    """
    if params is None:  # not a def default: it would build jax arrays at import (#859)
        params = SurfaceParameters.default()
    # Fresnel reflection formula for direct beam
    # Simplified parameterization based on zenith angle
    cos_theta = jnp.cos(solar_zenith_angle)

    # Direct beam albedo (increases with zenith angle)
    albedo_direct_vis = 0.05 + 0.15 * (1.0 - cos_theta)**2
    albedo_direct_nir = 0.05 + 0.10 * (1.0 - cos_theta)**2

    # Diffuse albedo (constant)
    albedo_diffuse_vis = jnp.full_like(albedo_direct_vis, 0.06)
    albedo_diffuse_nir = jnp.full_like(albedo_direct_nir, 0.06)

    return albedo_direct_vis, albedo_diffuse_vis, albedo_direct_nir, albedo_diffuse_nir


#: ECHAM's Charnock constant (``mo_physc2``/``iniphy.f90::cchar = 0.018``).
#: Used by BOTH consumers of the Charnock relation: the live water-tile
#: roughness (``TteTkeVerticalDiffusion``, which reads the differentiable
#: ``VDiffParameters.cchar`` leaf — reintroduced in jax-gcm#1118 together
#: with this becoming its real consumer) and this module's own
#: reference/tile-diagnostic bulk flux path (``compute_ocean_surface_fluxes``
#: / ``ocean_physics_step``), which still just reads this bare default since
#: it has no ``VDiffParameters`` of its own to carry a differentiable
#: override. Kept as a named constant rather than a bare literal so it reads
#: as ECHAM's value, not an arbitrary one.
CHARNOCK_ALPHA_DEFAULT = 0.018

#: ECHAM's floor on the ocean momentum roughness (``mo_surface_ocean.f90::
#: update_z0_ocean``'s ``zepzzo = 1.5e-5_wp``). The Fortran applies only this
#: lower bound — ``paz0w = MAX(zepzzo, ...)`` — with no upper cap even at
#: high wind, so neither does this port.
Z0_OCEAN_MIN = 1.5e-5


@jax.jit
def compute_ocean_roughness(
    friction_velocity: jnp.ndarray,
    cchar: jnp.ndarray = CHARNOCK_ALPHA_DEFAULT,
) -> jnp.ndarray:
    """Ocean momentum roughness from the Charnock relation, ``z0 = cchar·u*²/g``.

    This is ECHAM's ``update_z0_ocean`` (``mo_surface_ocean.f90``):

        zcons14 = cchar·rd / (grav²·Δt)
        paz0w   = MAX(zepzzo, zcons14·zcfmw·|U_diff|·Tv/paphm1(:,klevp1))

    which reduces to the textbook ``cchar·u*²/g`` once ``zcfmw·|U_diff|`` is
    read as the momentum flux ``τ/ρ`` and ``Tv/paphm1 = 1/(ρ·rd)`` from the
    ideal-gas law: ``zcons14·zcfmw·|U_diff|·Tv/paphm1 = cchar·(τ/(ρ·grav)) =
    cchar·u*²/grav`` with ``u*² = τ/ρ``. Passing the friction velocity
    directly (rather than re-deriving it from a flux and a density here)
    keeps this function usable by both the ECHAM-Louis surface layer's own
    ``u*`` (see ``turbulence_coefficients.py::compute_friction_velocity``)
    and this module's bulk-flux reference path.

    Args:
        friction_velocity: Surface-layer friction velocity u* [m/s] (ncol,)
        cchar: Charnock constant [-]. Defaults to ECHAM's
            :data:`CHARNOCK_ALPHA_DEFAULT`; the live water-tile roughness
            passes the differentiable ``VDiffParameters.cchar`` leaf.

    Returns:
        Ocean roughness length [m] (ncol,)

    """
    z0_ocean = cchar * friction_velocity**2 / c.grav
    return jnp.maximum(z0_ocean, Z0_OCEAN_MIN)


@jax.jit
def compute_ocean_surface_fluxes(
    atmospheric_state: AtmosphericForcing,
    ocean_temp: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray,
    solar_zenith_angle: jnp.ndarray,
    params: SurfaceParameters | None = None,
) -> Tuple[SurfaceFluxes, jnp.ndarray]:
    """Compute surface fluxes over ocean.

    Args:
        atmospheric_state: Atmospheric forcing
        ocean_temp: Ocean surface temperature [K] (ncol,)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        solar_zenith_angle: Solar zenith angle [rad] (ncol,)
        params: Surface parameters

    Returns:
        Tuple of (surface_fluxes, roughness_length)

    """
    if params is None:  # not a def default: it would build jax arrays at import (#859)
        params = SurfaceParameters.default()
    ncol = ocean_temp.shape[0]

    # Air density
    air_density = (atmospheric_state.pressure /
                  (c.rd * atmospheric_state.temperature))

    # Ocean surface saturation humidity: ECHAM's precalc_ocean reads the
    # ``ua`` table at the SST (mo_surface_ocean.f90 l.334-339), Sonntag (1990)
    # over ice at and below tmelt and over water above, and forms
    # ``zqsw = zes/(1 − vtmpc1·zes)`` with ``zes = ua/paphm1``, at the
    # surface pressure.
    e_sat = thermodynamics.es_ua(ocean_temp)  # Pa
    q_sat_ocean = thermodynamics.qsat_from_es(
        e_sat, atmospheric_state.surface_pressure)

    # Temperature and humidity differences
    delta_temp = ocean_temp - atmospheric_state.temperature
    delta_humidity = q_sat_ocean - atmospheric_state.humidity

    # Wind relative to ocean surface
    wind_rel_u = atmospheric_state.u_wind - ocean_u
    wind_rel_v = atmospheric_state.v_wind - ocean_v

    # Sensible heat flux [W/m²]
    sensible_heat = air_density * c.cpd * exchange_coeff_heat * delta_temp

    # Momentum fluxes [N/m²]
    momentum_u = air_density * exchange_coeff_momentum * wind_rel_u
    momentum_v = air_density * exchange_coeff_momentum * wind_rel_v

    # Evaporation rate [kg/m²/s]
    evaporation = air_density * exchange_coeff_moisture * delta_humidity

    # Latent heat flux [W/m²]
    latent_heat = c.alhc * evaporation

    # Ocean albedo
    albedo_vis_dir, albedo_vis_dif, albedo_nir_dir, albedo_nir_dif = compute_ocean_albedo(
        solar_zenith_angle, params
    )

    # Shortwave absorption (simplified - assume 50% visible, 50% NIR)
    albedo_mean = (albedo_vis_dir + albedo_vis_dif + albedo_nir_dir + albedo_nir_dif) / 4.0
    shortwave_net = atmospheric_state.sw_downward * (1.0 - albedo_mean)

    # Longwave flux
    lw_upward = params.emissivity * params.stefan_boltzmann * ocean_temp**4
    longwave_net = atmospheric_state.lw_downward - lw_upward

    # Ocean roughness. ``compute_friction_velocity`` is the ONE definition
    # of u* from a momentum flux (floor included — the #558 zero-wind
    # cotangent guard), shared with the live water-tile roughness's own u*
    # (turbulence_coefficients.py::compute_turbulence_diagnostics); deriving
    # it again here inline would both duplicate the formula and silently
    # drop that floor. Imported locally: a module-level import would cycle
    # back into this package's own ``__init__`` mid-import (``tte_tke``
    # imports ``vertical_diffusion.py``, which imports this module) —
    # deferred to call time, this module is always fully loaded by then,
    # the same pattern ``vertical_diffusion.py`` itself uses for its own
    # lazy ``echam_boundary_conditions`` import.
    #
    # ``momentum_u``/``momentum_v`` above use the wind RELATIVE to the ocean
    # current (``wind_rel_u/v``), while ``exchange_coeff_momentum`` (an
    # upstream input) is CM·|U| from the ABSOLUTE lowest-level wind
    # (``turbulence_coefficients.py``'s ``wind_speed_surface`` does not
    # subtract a current). ``momentum_u/v = τ/ρ`` exactly only while
    # ``ocean_u``/``ocean_v`` are zero. This module's ``ocean_u``/``ocean_v``
    # (``initialize_surface_state`` hardcodes them) stay zero DELIBERATELY,
    # even though the real current is now read elsewhere (jax-gcm#915,
    # ``TteTkeVerticalDiffusion``/``compute_turbulence_diagnostics``): this
    # whole bulk-aerodynamic path is a reference DIAGNOSTIC only (never
    # delivered to the column, see ``EchamSurface``'s docstring), and wiring
    # the real current into ``wind_rel_u/v`` here without ALSO recomputing
    # ``exchange_coeff_momentum`` against that same relative wind would make
    # this already-approximate estimate worse, not more faithful — CM stays
    # built from the absolute wind regardless, so the identity above would
    # simply break instead of holding. Not an open gap; a parked decision.
    from jcm.physics.vertical_diffusion.tte_tke.turbulence_coefficients import (
        compute_friction_velocity,
    )
    friction_velocity = compute_friction_velocity(momentum_u, momentum_v, air_density)
    roughness = compute_ocean_roughness(friction_velocity)

    # Create single-tile flux structure (ocean only)
    fluxes = SurfaceFluxes(
        sensible_heat=sensible_heat[:, None],
        latent_heat=latent_heat[:, None],
        longwave_net=longwave_net[:, None],
        shortwave_net=shortwave_net[:, None],
        ground_heat=jnp.zeros((ncol, 1)),  # No ground heat flux for ocean
        momentum_u=momentum_u[:, None],
        momentum_v=momentum_v[:, None],
        evaporation=evaporation[:, None],
        transpiration=jnp.zeros((ncol, 1)),  # No transpiration for ocean
        sensible_heat_mean=sensible_heat,
        latent_heat_mean=latent_heat,
        momentum_u_mean=momentum_u,
        momentum_v_mean=momentum_v,
        evaporation_mean=evaporation
    )

    return fluxes, roughness


@jax.jit
def ocean_physics_step(
    atmospheric_state: AtmosphericForcing,
    ocean_temp: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray,
    solar_zenith_angle: jnp.ndarray,
    dt: float,
    params: SurfaceParameters | None = None,
) -> Tuple[SurfaceFluxes, SurfaceTendencies, jnp.ndarray]:
    """Diagnostic ocean surface step: bulk fluxes only, no prognostic state.

    Ocean temperature is prescribed by the boundary SST in the ECHAM flow,
    so this step computes the bulk surface fluxes via
    ``compute_ocean_surface_fluxes`` and returns zero tendencies for
    every prognostic field. Slab-ocean evolution lives outside this repo.

    Args:
        atmospheric_state: Atmospheric forcing
        ocean_temp: Ocean temperature [K] (ncol,)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        solar_zenith_angle: Solar zenith angle [rad] (ncol,)
        dt: Time step [s] — accepted for interface compatibility, unused.
        params: Surface parameters

    Returns:
        Tuple of (surface_fluxes, zero_tendencies, roughness_length)

    """
    if params is None:  # not a def default: it would build jax arrays at import (#859)
        params = SurfaceParameters.default()
    ncol = ocean_temp.shape[0]
    del dt  # Unused — no prognostic step.

    # Compute surface fluxes
    surface_fluxes, roughness = compute_ocean_surface_fluxes(
        atmospheric_state, ocean_temp, ocean_u, ocean_v,
        exchange_coeff_heat, exchange_coeff_moisture, exchange_coeff_momentum,
        solar_zenith_angle, params
    )

    tendencies = SurfaceTendencies(
        surface_temp_tendency=jnp.zeros((ncol, 1)),
        ocean_temp_tendency=jnp.zeros(ncol),
        ice_temp_tendency=jnp.zeros((ncol, 1)),
        soil_temp_tendency=jnp.zeros((ncol, 1)),
        ice_thickness_tendency=jnp.zeros((ncol, 1)),
        snow_depth_tendency=jnp.zeros(ncol),
        soil_moisture_tendency=jnp.zeros((ncol, 1))
    )

    return surface_fluxes, tendencies, roughness
