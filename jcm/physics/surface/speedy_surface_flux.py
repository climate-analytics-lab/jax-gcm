"""SPEEDY bulk surface fluxes (port of SPEEDY's ``suflux.f90``).

Exchange of momentum, heat and moisture between the surface and the lowest
model level, using bulk aerodynamic formulae with a stability correction.

Two surface types
-----------------
Every flux is computed twice — once over the land fraction of the cell and
once over the sea fraction — because the two differ in surface temperature,
albedo, exchange coefficient and (over land) an orographic drag enhancement
and an interactive skin temperature. The atmosphere only ever feels the
area-weighted grid mean

    merged = sea + fmask * (land - sea)

so ``ustr``, ``vstr``, ``shf``, ``evap`` and ``rlus`` are stored as merged
2D maps only; the land and sea values are intermediates of this module.

``hfluxn`` is the exception. It is the net heat flux *into the surface
medium*, i.e. the term that drives a surface component's own temperature
(the ground below the land skin, or the ocean mixed layer). A coupled land
or ocean model needs its own tile's value, not the grid mean, so ``hfluxn``
is published per surface type (``hfluxn_land``, ``hfluxn_sea``) alongside
the merged grid mean that closes the column energy budget.

Near-surface extrapolation
--------------------------
The bulk formulae need air properties at the surface layer (sigma = 0.99),
not at the lowest model level. Two extrapolations are made:

``t1``  using the *actual* near-surface lapse rate, measured between the
        lowest layer and a fixed sigma (the sub-cloud-layer top). Anchoring
        the reference in sigma makes the diagnosed lapse rate — and the
        stable/unstable branch selected from it — independent of the vertical
        grid. On the 8-level reference grid the fixed sigma is the
        second-lowest layer centre, reproducing SPEEDY's original
        ``wvi[kx-1, 1]`` interpolation weight exactly.

``t2``  using the *dry-adiabatic* lapse rate, which is what the stability
        correction compares the surface temperature against.

The land and sea variants of each differ only by the orographic offset: sea
values are extrapolated down to z = 0, land values stay at the model
orography.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import jit

from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters, SurfaceFluxParameters
from jcm.physics_interface import PhysicsTendency, PhysicsState
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.smoothing import smooth_gate, smooth_pos
import jcm.constants as c
from jcm.physics.speedy.physical_constants import alhc
from jcm.physics.speedy.speedy_coords import PBL_TOP_SIGMA, interp_to_sigma
from jcm.physics.clouds.speedy_humidity import get_qsat, rel_hum_to_spec_hum


class SurfaceTypeFluxes(NamedTuple):
    """Fluxes over a single surface type, all ``(ix, il)``.

    Sign convention follows SPEEDY: ``ustr``/``vstr`` are the stress on the
    atmosphere, ``shf``/``evap``/``rlus`` are upward (surface to atmosphere),
    and ``hfluxn`` is downward (into the surface).
    """

    ustr: jnp.ndarray    # u-stress [N/m2]
    vstr: jnp.ndarray    # v-stress [N/m2]
    shf: jnp.ndarray     # sensible heat flux [W/m2]
    evap: jnp.ndarray    # evaporation [g/m2/s]
    rlus: jnp.ndarray    # upward longwave emission [W/m2]
    hfluxn: jnp.ndarray  # net downward heat flux into the surface [W/m2]


class NearSurfaceAir(NamedTuple):
    """Air properties extrapolated to the surface layer (sigma = 0.99).

    ``*_land`` / ``*_sea`` differ only through the orographic offset in the
    extrapolation (see the module docstring).
    """

    psa: jnp.ndarray        # normalised surface pressure
    u_wind: jnp.ndarray     # near-surface u-wind, fwind0-scaled [m/s]
    v_wind: jnp.ndarray     # near-surface v-wind, fwind0-scaled [m/s]
    u_bottom: jnp.ndarray   # lowest model level u-wind [m/s]
    v_bottom: jnp.ndarray   # lowest model level v-wind [m/s]
    t_land: jnp.ndarray    # temperature, actual lapse rate [K]
    t_sea: jnp.ndarray
    t2_land: jnp.ndarray   # temperature, dry-adiabatic lapse rate [K]
    t2_sea: jnp.ndarray
    q_land: jnp.ndarray    # specific humidity [g/kg]
    q_sea: jnp.ndarray
    rho_wind: jnp.ndarray  # density * wind speed incl. gustiness [kg/m2/s]


def _stability_factor(surface_temp, t2, sfp: SurfaceFluxParameters):
    """Stability correction multiplying the neutral density-wind product.

    Driven by the surface-to-air potential temperature excess, clipped to
    +/- ``dtheta``. With ``lscasym`` the stable (negative) side is damped by
    half, so a stable surface suppresses exchange less than an equally
    unstable one enhances it.
    """
    astab = jnp.where(sfp.lscasym, 0.5, 1.0)
    rdth = sfp.fstab / sfp.dtheta
    dth = jnp.where(
        surface_temp > t2,
        jnp.minimum(sfp.dtheta, surface_temp - t2),
        jnp.maximum(-sfp.dtheta, astab * (surface_temp - t2)),
    )
    return 1.0 + dth * rdth


def _near_surface_humidity(t_near, psa, rh_bottom, q_bottom, sfp: SurfaceFluxParameters):
    """Near-surface specific humidity [g/kg].

    ``fhum0`` blends between extrapolating at constant relative humidity
    (1) and holding the lowest-level specific humidity (0). The blend sits
    behind a ``cond`` rather than a ``where`` because ``fhum0`` defaults to
    0: the saturation calculation is then never evaluated, and — more to the
    point — its derivative never enters a reverse-mode pass, where a column
    driving ``get_qsat`` toward its singular denominator would contribute
    ``0 * inf`` to the gradient of the whole model.
    """
    return jax.lax.cond(
        sfp.fhum0 > 0.0,
        lambda _: (sfp.fhum0 * rel_hum_to_spec_hum(t_near, psa, 1.0, rh_bottom)[0]
                   + (1.0 - sfp.fhum0) * q_bottom),
        lambda _: q_bottom,
        operand=None,
    )


def _land_fluxes(
    air: NearSurfaceAir,
    sfp: SurfaceFluxParameters,
    forcing: ForcingData,
    terrain: TerrainData,
    physics_data: PhysicsData,
    esbc,
    rsds,
    rlds,
) -> tuple[SurfaceTypeFluxes, jnp.ndarray]:
    """Land fluxes and the skin temperature they are evaluated at."""
    stl_am = forcing.stl_am
    snowc = physics_data.mod_radcon.snowc
    alb_l = physics_data.mod_radcon.alb_l

    # Effective skin temperature: the daytime skin excess over the prescribed
    # land temperature, scaled by absorbed shortwave. Compensates for the
    # non-linearity of the heat/moisture fluxes within the averaging period.
    tskin = (stl_am + sfp.ctday * jnp.sqrt(physics_data.speedy_coords.coa)
             * rsds * (1.0 - alb_l) * air.psa)

    rho_wind = air.rho_wind * _stability_factor(tskin, air.t2_land, sfp)

    # Momentum drag over land takes the neutral density-wind product (no
    # stability correction) but is enhanced over orography; heat and moisture
    # exchange use the stability-corrected one. The stress acts on the lowest
    # level wind, while the density-wind product carries the fwind0-scaled
    # near-surface wind and the gustiness floor.
    forog = get_orog_land_sfc_drag(terrain.phis0, sfp.hdrag)
    cdldv = sfp.cdl * air.rho_wind * forog
    ustr = -cdldv * air.u_bottom
    vstr = -cdldv * air.v_bottom

    chlcp = sfp.chl * c.cpd
    shf = chlcp * rho_wind * (tskin - air.t_land)

    qsat_skin = get_qsat(tskin, air.psa, 1.0)

    # The soil-moisture-limited evaporation onset is a hinge that zeroes
    # every gradient through dry land columns (and gates the d(Evap)/d(Tskin)
    # term in the energy balance below on the same hard condition).
    # evap_smoothing > 0 [g/kg] rounds it with a softplus; 0 keeps the hard
    # maximum.
    evap_excess = forcing.soilw_am * qsat_skin - air.q_land
    evap = sfp.chl * rho_wind * smooth_pos(evap_excess, sfp.evap_smoothing)

    tsk3 = tskin ** 3.0
    drls = 4.0 * esbc * tsk3
    rlus = esbc * tsk3 * tskin
    hfluxn = rsds * (1.0 - alb_l) + rlds - (rlus + shf + alhc * evap)

    def skin_energy_balance(operand):
        """Redefine the skin temperature so the surface energy budget closes.

        One Newton step on the residual ``hfluxn``, treating the emission,
        sensible and latent terms as locally linear in the skin temperature.
        ``hfluxn`` then becomes the conductive flux into the soil below the
        skin, which is what drives the land reservoir.
        """
        tskin, shf, evap, rlus, hfluxn = operand

        clamb = sfp.clambda + snowc * (sfp.clambsn - sfp.clambda)
        residual = hfluxn - clamb * (tskin - stl_am)

        # d(Evap)/d(Tskin) for a 1-degree increment. The activity weight is
        # smooth_gate — the analytic derivative of the smooth_pos hinge above
        # — so a dry column whose evaporation is only the softplus tail gets
        # the matching fraction of the latent sensitivity rather than all of
        # it. At width 0 it reduces to the hard evap > 0 mask.
        evap_gate = smooth_gate(evap_excess, 0.0, sfp.evap_smoothing)
        dqsat = evap_gate * forcing.soilw_am * (
            get_qsat(tskin + 1.0, air.psa, 1.0) - qsat_skin)

        dtskin = residual / (
            clamb + drls + sfp.chl * rho_wind * (c.cpd + alhc * dqsat))
        tskin = tskin + dtskin

        return (
            tskin,
            shf + chlcp * rho_wind * dtskin,
            evap + sfp.chl * rho_wind * dqsat * dtskin,
            rlus + drls * dtskin,
            clamb * (tskin - stl_am),
        )

    tskin, shf, evap, rlus, hfluxn = jax.lax.cond(
        sfp.lskineb,
        skin_energy_balance,
        lambda operand: operand,
        operand=(tskin, shf, evap, rlus, hfluxn),
    )

    return SurfaceTypeFluxes(ustr, vstr, shf, evap, rlus, hfluxn), tskin


def _sea_fluxes(
    air: NearSurfaceAir,
    sfp: SurfaceFluxParameters,
    forcing: ForcingData,
    physics_data: PhysicsData,
    esbc,
    rsds,
    rlds,
) -> SurfaceTypeFluxes:
    """Sea fluxes, evaluated at the prescribed sea-surface temperature."""
    sst = forcing.sea_surface_temperature
    alb_s = physics_data.mod_radcon.alb_s

    rho_wind = air.rho_wind * _stability_factor(sst, air.t2_sea, sfp)

    cdsdv = sfp.cds * rho_wind
    ustr = -cdsdv * air.u_bottom
    vstr = -cdsdv * air.v_bottom

    shf = sfp.chs * c.cpd * rho_wind * (sst - air.t_sea)

    qsat_sea = get_qsat(sst, air.psa, 1.0)
    evap = sfp.chs * rho_wind * (qsat_sea - air.q_sea)

    rlus = esbc * sst ** 4.0
    hfluxn = rsds * (1.0 - alb_s) + rlds - (rlus + shf + alhc * evap)

    return SurfaceTypeFluxes(ustr, vstr, shf, evap, rlus, hfluxn)


def _extrapolate_to_surface(
    state: PhysicsState,
    physics_data: PhysicsData,
    sfp: SurfaceFluxParameters,
    terrain: TerrainData,
) -> tuple[NearSurfaceAir, jnp.ndarray]:
    """Extrapolate the lowest model level down to the surface layer.

    Returns the near-surface air properties and the merged near-surface
    temperature ``t0`` (also used for the near-surface density).
    """
    psa = state.normalized_surface_pressure
    ta = state.temperature
    phi0 = terrain.orog * c.grav
    sigl = physics_data.speedy_coords.sigl
    kx = ta.shape[0]

    u_bottom, v_bottom = state.u_wind[-1], state.v_wind[-1]
    u_wind = sfp.fwind0 * u_bottom
    v_wind = sfp.fwind0 * v_bottom

    ta_ref = interp_to_sigma(ta, physics_data.speedy_coords.fsg, PBL_TOP_SIGMA)
    dt1_fac = (jnp.log(0.99) - sigl[kx - 1]) / (sigl[kx - 1] - jnp.log(PBL_TOP_SIGMA))
    dt1 = dt1_fac * (ta[-1] - ta_ref)

    # Actual-lapse-rate extrapolation; the sea variant continues down to z=0.
    t1_land = ta[-1] + dt1
    t1_sea = t1_land - phi0 * dt1 / (c.rd * 288.0 * sigl[kx - 1])

    # Dry-adiabatic extrapolation.
    rcp = 1.0 / c.cpd
    t2_sea = ta[-1] + rcp * state.geopotential[-1]
    t2_land = t2_sea - rcp * phi0

    # Blend the two extrapolations, but only where the near-surface layer is
    # statically unstable; in a stable layer the lowest-level temperature is
    # carried down unchanged.
    unstable = ta[-1] > ta_ref
    blend = lambda t1, t2: jnp.where(
        unstable, sfp.ftemp0 * t1 + (1.0 - sfp.ftemp0) * t2, ta[-1])
    t1_land, t1_sea = blend(t1_land, t2_land), blend(t1_sea, t2_sea)

    t0 = t1_sea + terrain.fmask * (t1_land - t1_sea)
    rho_wind = ((c.p0 * psa / (c.rd * t0))
                * jnp.sqrt(u_wind ** 2 + v_wind ** 2 + sfp.vgust ** 2))

    rh_bottom = physics_data.humidity.rh[-1]
    q_bottom = state.specific_humidity[-1]
    air = NearSurfaceAir(
        psa=psa, u_wind=u_wind, v_wind=v_wind,
        u_bottom=u_bottom, v_bottom=v_bottom,
        t_land=t1_land, t_sea=t1_sea,
        t2_land=t2_land, t2_sea=t2_sea,
        q_land=_near_surface_humidity(t1_land, psa, rh_bottom, q_bottom, sfp),
        q_sea=_near_surface_humidity(t1_sea, psa, rh_bottom, q_bottom, sfp),
        rho_wind=rho_wind,
    )
    return air, t0


@jit
def get_surface_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData,
) -> tuple[PhysicsTendency, PhysicsData]:
    """Surface fluxes and the tendencies they impose on the lowest level.

    Args:
        state: Atmospheric state; the lowest level supplies the bulk-formula
            air properties.
        physics_data: Diagnostics; reads ``shortwave_rad.rsds``,
            ``surface_flux.rlds``, ``humidity.rh``, ``mod_radcon`` albedos and
            snow cover, and ``speedy_coords``.
        parameters: SPEEDY parameters; ``surface_flux`` and ``mod_radcon``.
        forcing: Boundary conditions; sea-surface temperature, land surface
            temperature ``stl_am`` and soil wetness ``soilw_am``.
        terrain: Orography, land fraction ``fmask``, and ``lfluxland``.

    Returns:
        The wind, temperature and humidity tendencies applied to the lowest
        model level, and ``physics_data`` with ``surface_flux`` updated.

    """
    sfp = parameters.surface_flux
    esbc = parameters.mod_radcon.emisfc * c.sbc
    rsds = physics_data.shortwave_rad.rsds
    rlds = physics_data.surface_flux.rlds
    fmask = terrain.fmask

    air, t0 = _extrapolate_to_surface(state, physics_data, sfp, terrain)

    # Skipping the land branch entirely (rather than masking it) keeps
    # aquaplanet runs free of the land forcing fields, which are absent there.
    # The zero branch is built from the shapes and dtypes the land branch
    # would produce: those follow the land forcing, which a caller may hand
    # us at a different precision from the state, and lax.cond requires the
    # two branches to agree exactly.
    land_fluxes = lambda: _land_fluxes(
        air, sfp, forcing, terrain, physics_data, esbc, rsds, rlds)
    no_land = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype),
                           jax.eval_shape(land_fluxes))
    land, tskin = jax.lax.cond(
        terrain.lfluxland, lambda _: land_fluxes(), lambda _: no_land, operand=None,
    )
    sea = _sea_fluxes(air, sfp, forcing, physics_data, esbc, rsds, rlds)

    merged = jax.tree.map(lambda over_land, over_sea:
                          over_sea + fmask * (over_land - over_sea), land, sea)

    sst = forcing.sea_surface_temperature
    surface_flux_out = physics_data.surface_flux.copy(
        ustr=merged.ustr, vstr=merged.vstr, shf=merged.shf, evap=merged.evap,
        rlus=merged.rlus, hfluxn=merged.hfluxn,
        hfluxn_land=land.hfluxn, hfluxn_sea=sea.hfluxn,
        tsfc=sst + fmask * (forcing.stl_am - sst),
        tskin=sst + fmask * (tskin - sst),
        u0=air.u_wind, v0=air.v_wind, t0=t0,
    )
    physics_data = physics_data.copy(surface_flux=surface_flux_out)

    # Tendencies on the lowest level (physics.f90:197-205).
    rps = 1.0 / state.normalized_surface_pressure
    grdsig = physics_data.speedy_coords.grdsig[-1]
    grdscp = physics_data.speedy_coords.grdscp[-1]
    physics_tendencies = PhysicsTendency(
        jnp.zeros_like(state.u_wind).at[-1].add(merged.ustr * rps * grdsig),
        jnp.zeros_like(state.v_wind).at[-1].add(merged.vstr * rps * grdsig),
        jnp.zeros_like(state.temperature).at[-1].add(merged.shf * rps * grdscp),
        jnp.zeros_like(state.specific_humidity).at[-1].add(merged.evap * rps * grdsig),
    )

    return physics_tendencies, physics_data


@jit
def get_orog_land_sfc_drag(phis0, hdrag):
    """Orographic enhancement of the land momentum drag coefficient.

    Args:
        phis0: Surface geopotential [m2/s2].
        hdrag: Height scale of the correction [m].

    Returns:
        Multiplicative factor >= 1 on the land drag coefficient.

    """
    rhdrag = 1 / (c.grav * hdrag)

    return 1.0 + rhdrag * (1.0 - jnp.exp(-jnp.maximum(phis0, 0.0) * rhdrag))
