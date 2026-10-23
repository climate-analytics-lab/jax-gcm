"""Surface exchange and turbulence diagnostics of the vertical diffusion.

The interior exchange coefficients are ECHAM's closure (:mod:`.closure`); this
module holds the interior Richardson number, the Businger-Dyer surface-layer
option, the friction velocities and the assembly of
:class:`~.vertical_diffusion_types.VDiffDiagnostics`.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from .moist_buoyancy import interior_buoyancy_and_shear, richardson_number
from .vertical_diffusion_types import VDiffParameters, VDiffState, VDiffDiagnostics


@jax.jit
def compute_richardson_number(state: VDiffState) -> jnp.ndarray:
    """Interior Richardson number of every interface between adjacent full levels.

    ECHAM6's ``zri = zbuoy / MAX(zshear, zepshr)`` (``vdiff.f90`` l.799):
    the moist, cloud-weighted buoyancy of :func:`~.moist_buoyancy.
    interior_buoyancy_and_shear` over the squared shear, floored at ECHAM's
    ``zepshr = 1e-5`` s⁻². Saturated, cloudy layers therefore have the
    stability of the moist-adiabatic lapse rate rather than the dry one.

    Args:
        state: Atmospheric state; reads the wind, temperature, humidity,
            condensate, cloud cover, pressures and geopotential.

    Returns:
        Richardson number [-] (ncol, nlev-1); ``[:, i]`` is the interface
        between full levels ``i`` and ``i + 1``.

    """
    return richardson_number(*interior_buoyancy_and_shear(state))


#: Businger-Dyer roughness tables [water, ice, land] — this scheme ignores
#: ``state.roughness_length``, so its neutral drag must be built from these.
#: Tuples of Python floats, materialised with ``jnp.asarray`` where used: a
#: module-level ``jnp.array`` initialises the JAX backend on import (#859).
_BD_Z0_HEAT = (1e-4, 1e-4, 1e-2)
_BD_Z0_MOMENTUM = (1e-4, 1e-3, 1e-1)


def businger_dyer_neutral_drag(state, wind_speed):
    """``(ln(z/z0), CM_n·|U|)`` of the Businger-Dyer scheme, per tile.

    THE definition: :func:`compute_surface_exchange_coefficients` builds its
    momentum drag as ``cfnc · stability_momentum`` from this, so the 10 m
    reduction's ``f_m = CM|U| / CM_n|U|`` is exactly that scheme's own
    ``stability_momentum`` rather than a re-derivation of it. Sliced to the
    state's tile count, as the scheme's per-tile loop is.
    """
    z_ref = state.height_full[:, -1] - state.height_half[:, -1]
    z0 = jnp.asarray(_BD_Z0_MOMENTUM[:state.roughness_length.shape[1]])
    bn = jnp.log(jnp.maximum(z_ref, 1.0)[:, None]
                 / jnp.maximum(z0, 1e-5)[None, :])
    # 0.4, not c.karman_const: it must match the von_karman this scheme
    # hard-codes, or a set_constants override desyncs the pair.
    return bn, wind_speed[:, None] * 0.4 ** 2 / bn ** 2


@jax.jit
def compute_surface_exchange_coefficients(
    state: VDiffState,
    params: VDiffParameters,
    wind_speed_surface: jnp.ndarray,
    temperature_surface: jnp.ndarray,
    temperature_air: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute surface exchange coefficients (Businger-Dyer option).

    The non-default surface-layer scheme (``surface_layer_scheme ==
    SCHEME_BUSINGER_DYER``). Bulk-Richardson Monin-Obukhov with Businger-Dyer
    stability functions on top of a κ²/[ln(z/z0)]² neutral drag.

    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        wind_speed_surface: Wind speed at surface [m/s] (ncol,)
        temperature_surface: Surface temperature [K] (ncol, nsfc_type)
        temperature_air: Air temperature at lowest level [K] (ncol,)

    Returns:
        Tuple of ``(surface_exchange_heat, surface_exchange_moisture,
        surface_exchange_momentum)`` = (CH·|U|, CE·|U|, CM·|U|), each
        [m/s] (ncol, nsfc_type). The momentum coefficient is returned so this
        scheme can drive the surface stress on equal footing with the faithful
        ECHAM-Louis default.

    """
    ncol, nsfc_type = temperature_surface.shape

    z0_heat = jnp.asarray(_BD_Z0_HEAT)
    z0_moisture = z0_heat
    # Momentum: one definition, shared with the 10 m reduction (below).
    _, cfn_m_all = businger_dyer_neutral_drag(state, wind_speed_surface)
    
    # Reference height (lowest model level)
    z_ref = state.height_full[:, -1] - state.height_half[:, -1]
    
    # von Kármán constant
    von_karman = 0.4
    
    # Compute exchange coefficients for each surface type
    surface_exchange_heat = jnp.zeros((ncol, nsfc_type))
    surface_exchange_moisture = jnp.zeros((ncol, nsfc_type))
    surface_exchange_momentum = jnp.zeros((ncol, nsfc_type))

    for isfc in range(nsfc_type):
        # Stability correction (simplified)
        theta_surface = temperature_surface[:, isfc]
        theta_air = temperature_air
        
        # Bulk Richardson number for surface layer
        ri_surface = (c.grav * z_ref *
                     (theta_air - theta_surface) / 
                     (0.5 * (theta_air + theta_surface) * 
                      jnp.maximum(wind_speed_surface**2, 0.01)))
        
        # Surface-layer stability multiplier on the heat exchange
        # coefficient CH, derived from Businger-Dyer. CH scales as the
        # inverse of the heat gradient function 1/Φh:
        #   unstable (Ri < 0): 1/Φh = (1 − 16 Ri)^(+1/2) > 1, so the
        #     multiplier enhances the turbulent flux when buoyancy
        #     destabilises the surface layer (e.g. cold air over warm
        #     ocean).
        #   stable   (Ri > 0): 1/Φh = 1 / (1 + 5 Ri)         < 1, so
        #     the multiplier damps the flux as buoyancy suppresses
        #     turbulence.
        # Using 1/Φh rather than Φh is essential here: Φh is the
        # gradient-to-flux ratio, while CH is proportional to the
        # diffusivity, which is its inverse.
        #
        # Each branch sees Ri only on its own side of zero (the double-where
        # on the base, as the ``zxrp1_base`` guard in ``clouds/echam_1m.py``
        # does it). ``where`` differentiates the branch it discards: on a
        # stable surface (Ri > 1/16) the unstable branch's base 1 − 16 Ri is
        # negative and its fractional power NaN, and at Ri = −0.2 the stable
        # branch divides by zero; the discarded branch's zero cotangent then
        # meets a NaN or infinite derivative and makes the exchange
        # coefficients' gradient NaN. The selected branch sees Ri itself, so
        # the value is the same.
        unstable = ri_surface < 0
        ri_unstable = jnp.where(unstable, ri_surface, 0.0)
        ri_stable = jnp.where(unstable, 0.0, ri_surface)
        stability_heat = jnp.where(
            unstable,
            (1.0 - 16.0 * ri_unstable)**(+0.5),  # Unstable: enhance
            1.0 / (1.0 + 5.0 * ri_stable),       # Stable: suppress
        )
        # Momentum stability multiplier 1/Φm: Businger-Dyer gives a weaker
        # unstable enhancement than heat ((1−16Ri)^(1/4) vs ^(1/2)); the stable
        # branch matches heat.
        stability_momentum = jnp.where(
            unstable,
            (1.0 - 16.0 * ri_unstable)**(+0.25),  # Unstable: enhance
            1.0 / (1.0 + 5.0 * ri_stable),        # Stable: suppress
        )

        # Exchange coefficient: C = κ²·|U|·(1/Φ) / [ln(z/z0)]²
        # Guard the log arguments against zero / negative values
        log_ratio_heat = jnp.log(jnp.maximum(z_ref, 1.0)
                                 / jnp.maximum(z0_heat[isfc], 1e-5))
        log_ratio_moisture = jnp.log(jnp.maximum(z_ref, 1.0)
                                     / jnp.maximum(z0_moisture[isfc], 1e-5))
        exchange_heat = (von_karman**2 * wind_speed_surface *
                        stability_heat / log_ratio_heat**2)
        exchange_moisture = (von_karman**2 * wind_speed_surface *
                           stability_heat / log_ratio_moisture**2)
        # CM·|U| = CM_n·|U| · (1/Φm): the neutral factor is the shared one, so
        # f_m in the 10 m reduction recovers stability_momentum exactly.
        exchange_momentum = cfn_m_all[:, isfc] * stability_momentum

        surface_exchange_heat = surface_exchange_heat.at[:, isfc].set(
            jnp.maximum(exchange_heat, 1e-6)
        )
        surface_exchange_moisture = surface_exchange_moisture.at[:, isfc].set(
            jnp.maximum(exchange_moisture, 1e-6)
        )
        surface_exchange_momentum = surface_exchange_momentum.at[:, isfc].set(
            jnp.maximum(exchange_momentum, 1e-6)
        )

    return (surface_exchange_heat, surface_exchange_moisture,
            surface_exchange_momentum)


@jax.jit
def compute_friction_velocity(
    momentum_flux_u: jnp.ndarray,
    momentum_flux_v: jnp.ndarray,
    air_density: jnp.ndarray
) -> jnp.ndarray:
    """Compute friction velocity from surface momentum flux.
    
    Args:
        momentum_flux_u: U-momentum flux [N/m²] (ncol,)
        momentum_flux_v: V-momentum flux [N/m²] (ncol,)
        air_density: Air density [kg/m³] (ncol,)
        
    Returns:
        Friction velocity [m/s] (ncol,)

    """
    # Floor the sum-of-squares inside the sqrt: at a zero-wind / zero-flux
    # state (e.g. the balanced isothermal start) ``sqrt(0)`` has an infinite
    # derivative, which poisons the reverse pass with a NaN cotangent even
    # though a downstream ``maximum`` masks the forward value (issue #558).
    momentum_flux_magnitude = jnp.sqrt(
        jnp.maximum(momentum_flux_u**2 + momentum_flux_v**2, 1.0e-30))
    friction_velocity = jnp.sqrt(
        jnp.maximum(momentum_flux_magnitude / air_density, 1.0e-30))
    
    return jnp.maximum(friction_velocity, 0.01)  # Minimum value


@jax.jit
def compute_turbulence_diagnostics(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    boundary_layer_height: jnp.ndarray,
    richardson_number: jnp.ndarray,
    mixing_length: jnp.ndarray,
) -> VDiffDiagnostics:
    """Compute the surface exchange and collect the turbulence diagnostics.

    The interior quantities come from the closure (:mod:`.closure`) and are
    passed through; this computes the per-tile surface exchange velocities,
    the friction velocities and the 10 m wind.

    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient of the
            interfaces [m²/s] (ncol, nlev), see :class:`VDiffDiagnostics`.
        exchange_coeff_heat: Heat exchange coefficient [m²/s] (ncol, nlev)
        exchange_coeff_moisture: Moisture exchange coefficient [m²/s] (ncol, nlev)
        boundary_layer_height: Height of the PBL extension [m] (ncol,)
        richardson_number: ``zri`` of the interior interfaces (ncol, nlev - 1)
        mixing_length: ``zmix`` of the interior interfaces [m] (ncol, nlev - 1)

    Returns:
        Complete diagnostics structure

    """
    ncol = state.u.shape[0]
    ri = richardson_number
    pbl_height = boundary_layer_height

    # Surface diagnostics — dispatch to the configured scheme via
    # lax.cond. Both branches are JIT-able and return identically
    # shaped (ncol, nsfc_type) tensors, so the false-branch tracing
    # is cheap. The scheme field on ``params`` is a tracer under
    # JIT, hence cond rather than a Python ``if``.
    from .surface_layer import (
        compute_surface_exchange_coefficients_echam_louis,
        echam_louis_neutral_drag,
        wind_10m_reduction,
    )

    wind_speed_surface = jnp.sqrt(
        jnp.maximum(state.u[:, -1]**2 + state.v[:, -1]**2, 1.0e-30))
    z_ref_sfc = state.height_full[:, -1] - state.height_half[:, -1]

    # The 10 m reduction is computed INSIDE each branch, from that scheme's own
    # neutral drag: the two schemes differ in roughness, in the bound on z/z0
    # and in whether the wind is zepdu2-floored, so a shared reference would
    # read stability where there is none (~13 % on the sea-salt wind term).
    def _louis():
        cfh, cfe, cfm = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind_speed_surface,
            state.surface_temperature, state.temperature[:, -1],
        )
        bn, cfnc = echam_louis_neutral_drag(state, params, wind_speed_surface)
        return cfh, cfe, cfm, wind_10m_reduction(cfm, cfnc, bn, z_ref_sfc)

    def _businger_dyer():
        cfh, cfe, cfm = compute_surface_exchange_coefficients(
            state, params, wind_speed_surface,
            state.surface_temperature, state.temperature[:, -1],
        )
        bn, cfnc = businger_dyer_neutral_drag(state, wind_speed_surface)
        return cfh, cfe, cfm, wind_10m_reduction(cfm, cfnc, bn, z_ref_sfc)

    (surface_exchange_heat, surface_exchange_moisture,
     surface_exchange_momentum, wind_10m_tile) = jax.lax.cond(
        params.surface_layer_scheme == VDiffParameters.SCHEME_ECHAM_LOUIS,
        _louis, _businger_dyer,
    )
    
    # Air density at surface
    air_density = (state.pressure_full[:, -1] /
                   (c.rd * state.temperature[:, -1]))

    # Per-tile momentum flux [N/m²] (ncol, nsfc_type): each tile's OWN
    # ``surface_exchange_momentum`` (CM·|U|, m/s) against the lowest-level
    # wind — the one definition of this flux, consumed two ways below.
    air_density_tile = air_density[:, None]
    surface_momentum_flux_u_tile = (
        air_density_tile * surface_exchange_momentum * state.u[:, -1:]
    )
    surface_momentum_flux_v_tile = (
        air_density_tile * surface_exchange_momentum * state.v[:, -1:]
    )
    # Per-tile friction velocity. Needed so the water tile's Charnock
    # roughness update (jax-gcm#1118, ``ocean.py::compute_ocean_roughness``)
    # reads the open-water u* alone; the grid-mean ``friction_velocity``
    # below would let a coastal cell's land/ice stress leak into the water
    # tile's own roughness, the same class of error already fixed for the
    # land tile's roughness itself (see ``vertical_diffusion.py``'s comment
    # on the public ``roughness_length`` field).
    surface_friction_velocity_tile = compute_friction_velocity(
        surface_momentum_flux_u_tile, surface_momentum_flux_v_tile, air_density_tile
    )

    # Grid-mean friction velocity: the SAME per-tile momentum flux above,
    # fraction-weighted down to one grid value — a single ECHAM-style
    # ``sfc_exchange_coeff``, not a separate bulk-Richardson solve, and the
    # same area weighting ``EchamSurface`` applies to form its own grid-mean
    # momentum stress (``tau = rho·⟨CM·|U|⟩·U``), so this reproduces that
    # stress exactly rather than re-deriving it from a separately
    # fraction-weighted CM·|U|. This keeps the u* consumed downstream by
    # aerosol dry deposition and wind-driven dust/sea-salt emission
    # consistent with the surface momentum exchange rather than a parallel
    # bulk estimate.
    surface_momentum_flux_u = jnp.sum(
        state.surface_fraction * surface_momentum_flux_u_tile, axis=1
    )
    surface_momentum_flux_v = jnp.sum(
        state.surface_fraction * surface_momentum_flux_v_tile, axis=1
    )
    friction_velocity = compute_friction_velocity(
        surface_momentum_flux_u, surface_momentum_flux_v, air_density
    )

    # 10 m wind (ECHAM ``nsurf_diag``), area-weighted over the tiles from the
    # same per-tile CM·|U|. Surface-flux parameterizations (sea salt, DMS) are
    # calibrated to u10, not to the lowest model level — ~33 m at L47.
    # ``wind_10m_tile`` is each tile's reduction factor ``zred``; ECHAM
    # scales the lowest-level wind by it component-wise, per tile
    # (mo_surface_ocean.f90::postproc_ocean ``zu10w = zred*pum1``, and the
    # ice/land analogues), and box-averages the tiles by fraction
    # (mo_surface.f90::surface_box_average -> ``u10``/``v10``/``wind10``). So
    # the grid-mean vector is ``u_low * sum_t f_t zred_t``: the
    # fraction-weighted sum of the per-tile vectors, and parallel to the
    # lowest-level wind with magnitude ``wind_10m``. ``u_low`` is the
    # step-start wind, as in ECHAM: vdiff.f90 passes ``pum1(:,klev)`` (t-dt),
    # not the implicitly updated wind, to update_surface (vdiff.f90:951-964)
    # and on to postproc_ocean/ice/land (mo_surface.f90:936-967).
    reduction_10m = jnp.sum(state.surface_fraction * wind_10m_tile, axis=1)
    wind_10m = wind_speed_surface * reduction_10m
    wind_10m_u = state.u[:, -1] * reduction_10m
    wind_10m_v = state.v[:, -1] * reduction_10m
    speed_10m_tile = wind_10m_tile * wind_speed_surface[:, None]
    u_10m_tile = wind_10m_tile * state.u[:, -1:]
    v_10m_tile = wind_10m_tile * state.v[:, -1:]

    # Open-water 10 m speed RELATIVE to the ocean surface current (ECHAM
    # ``wind10w``, ``mo_surface_ocean.f90::postproc_ocean`` l.699-713):
    # ``zred_water·sqrt((u_low−ocu)² + (v_low−ocv)²)``, the water tile's own
    # reduction factor (tile index 0 of the water/ice/land ordering) applied
    # to the CURRENT-relative lowest-level wind, not the absolute one. This
    # is a distinct diagnostic from ``speed_10m_tile[:, 0]`` above (which
    # stays absolute, matching ECHAM's separately-reported ``u10``/``v10``);
    # they coincide only when ``state.ocean_u/v`` are zero. ``state.ocean_u``/
    # ``ocean_v`` are the RAW ocean current (``ForcingData.ocean_u/v``, zero
    # when not supplied) — unlike the momentum Robin row's surface target
    # (``vertical_diffusion_column``), which scales the current by the
    # non-land fraction, this water-only diagnostic is not scaled: ECHAM's
    # ``postproc_ocean`` runs strictly under the water-type mask.
    wind_10m_current_relative = wind_10m_tile[:, 0] * jnp.sqrt(jnp.maximum(
        (state.u[:, -1] - state.ocean_u) ** 2
        + (state.v[:, -1] - state.ocean_v) ** 2,
        1.0e-30))

    # Convective velocity scale (simplified)
    convective_velocity = jnp.maximum(friction_velocity, 0.1)
    
    return VDiffDiagnostics(
        exchange_coeff_momentum=exchange_coeff_momentum,
        exchange_coeff_heat=exchange_coeff_heat,
        exchange_coeff_moisture=exchange_coeff_moisture,
        surface_exchange_heat=surface_exchange_heat,
        surface_exchange_moisture=surface_exchange_moisture,
        surface_exchange_momentum=surface_exchange_momentum,
        boundary_layer_height=pbl_height,
        friction_velocity=friction_velocity,
        convective_velocity=convective_velocity,
        surface_friction_velocity_tile=surface_friction_velocity_tile,
        wind_10m=wind_10m,
        wind_10m_u=wind_10m_u,
        wind_10m_v=wind_10m_v,
        wind_10m_reduction=reduction_10m,
        wind_10m_tile=speed_10m_tile,
        wind_10m_u_tile=u_10m_tile,
        wind_10m_v_tile=v_10m_tile,
        wind_10m_current_relative=wind_10m_current_relative,
        richardson_number=ri,
        mixing_length=mixing_length,
        kinetic_energy_dissipation=jnp.zeros(ncol)  # Will be computed by TKE budget
    )