import jax
import jax.numpy as jnp
from jax import jit

# importing custom functions from library
from jcm.geometry import Geometry
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics_interface import PhysicsTendency, PhysicsState
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.physical_constants import p0, rgas, cp, alhc, sbc, grav
from jcm.physics.speedy.humidity import get_qsat, rel_hum_to_spec_hum
from jcm.utils import pass_fn

@jit
def compute_land_surface_fluxes(
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    ua: jnp.ndarray,
    va: jnp.ndarray,
    ta: jnp.ndarray,
    qa: jnp.ndarray,
    rh: jnp.ndarray,
    phi: jnp.ndarray,
    phi0: jnp.ndarray,
    psa: jnp.ndarray,
    fmask: jnp.ndarray,
    stl_am: jnp.ndarray,
    soilw_am: jnp.ndarray,
    rsds: jnp.ndarray,
    rlds: jnp.ndarray,
    alb_l: jnp.ndarray,
    snowc: jnp.ndarray,
    phis0: jnp.ndarray,
    wvi: jnp.ndarray,
    sigl: jnp.ndarray,
    coa: jnp.ndarray,
    parameters: Parameters,
    esbc: float,
    ghum0: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute surface fluxes over land surfaces.

    This function calculates wind stress, sensible heat flux, evaporation,
    and longwave radiation for land surfaces using prescribed or computed
    skin temperatures with energy balance adjustments.

    Parameters
    ----------
    u0, v0 : 2D arrays
        Surface wind components (adjusted with gustiness factor)
    ua, va : 3D arrays
        Wind components at model levels
    ta : 3D array
        Temperature at model levels
    qa : 3D array
        Specific humidity at model levels
    rh : 3D array
        Relative humidity at model levels
    phi : 3D array
        Geopotential at model levels
    phi0 : 2D array
        Surface geopotential
    psa : 2D array
        Normalized surface pressure
    fmask : 2D array
        Fractional land-sea mask
    stl_am : 2D array
        Annual mean land surface temperature
    soilw_am : 2D array
        Soil moisture availability
    rsds : 2D array
        Downward shortwave radiation at surface
    rlds : 2D array
        Downward longwave radiation at surface
    alb_l : 2D array
        Land surface albedo
    snowc : 2D array
        Snow cover fraction
    phis0 : 2D array
        Surface geopotential for orographic drag
    wvi : 3D array
        Vertical interpolation weights
    sigl : 1D array
        Sigma levels
    coa : 2D array
        Cosine of latitude for diurnal cycle
    parameters : Parameters
        Model parameters
    esbc : float
        Emissivity * Stefan-Boltzmann constant
    ghum0 : float
        1.0 - fhum0 (humidity interpolation parameter)

    Returns
    -------
    ustr : 2D array
        U-component of wind stress over land
    vstr : 2D array
        V-component of wind stress over land
    shf : 2D array
        Sensible heat flux over land
    evap : 2D array
        Evaporation over land
    rlus : 2D array
        Upward longwave radiation over land
    hfluxn : 2D array
        Net heat flux into land surface
    tskin : 2D array
        Land skin temperature
    """
    kx = ta.shape[0]
    rcp = 1.0 / cp
    nl1 = kx - 1
    gtemp0 = 1.0 - parameters.surface_flux.ftemp0

    ix, il = psa.shape
    t1 = jnp.zeros((ix, il, 2))
    t2 = jnp.zeros((ix, il, 2))
    q1 = jnp.zeros((ix, il, 2))
    qsat0 = jnp.zeros((ix, il, 2))
    denvvs = jnp.zeros((ix, il, 3))

    # Temperature difference between lowest level and sfc
    dt1 = wvi[kx-1, 1, jnp.newaxis, jnp.newaxis] * (ta[kx-1] - ta[nl1-1])

    # Extrapolated temperature using actual lapse rate (0:land, 1:sea)
    t1 = t1.at[:, :, 0].add(ta[kx-1] + dt1)
    t1 = t1.at[:, :, 1].set(t1[:, :, 0] - phi0 * dt1 / (rgas * 288.0 * sigl[kx-1]))

    # Extrapolated temperature using dry-adiabatic lapse rate (0:land, 1:sea)
    t2 = t2.at[:, :, 1].set(ta[kx-1] + rcp * phi[kx-1])
    t2 = t2.at[:, :, 0].set(t2[:, :, 1] - rcp * phi0)

    # Select temperature based on stability
    t1 = jnp.where(
        (ta[kx-1] > ta[nl1-1])[:, :, jnp.newaxis],
        parameters.surface_flux.ftemp0 * t1 + gtemp0 * t2,
        ta[kx-1][:, :, jnp.newaxis]
    )

    t0 = t1[:, :, 1] + fmask * (t1[:, :, 0] - t1[:, :, 1])

    # Density * wind speed (including gustiness factor)
    denvvs = denvvs.at[:, :, 0].set(
        (p0 * psa / (rgas * t0)) * jnp.sqrt(u0**2 + v0**2 + parameters.surface_flux.vgust**2)
    )

    # Effective skin temperature (compensating for non-linearity of fluxes)
    tskin = stl_am + parameters.surface_flux.ctday * jnp.sqrt(coa) * rsds * (1.0 - alb_l) * psa

    # Stability correction
    rdth = parameters.surface_flux.fstab / parameters.surface_flux.dtheta
    astab = jax.lax.cond(
        parameters.surface_flux.lscasym,
        lambda _: jnp.array(0.5),
        lambda _: jnp.array(1.0),
        operand=None
    )

    dthl = jnp.where(
        tskin > t2[:, :, 0],
        jnp.minimum(parameters.surface_flux.dtheta, tskin - t2[:, :, 0]),
        jnp.maximum(-parameters.surface_flux.dtheta, astab * (tskin - t2[:, :, 0]))
    )

    denvvs = denvvs.at[:, :, 1].set(denvvs[:, :, 0] * (1.0 + dthl * rdth))

    # Wind stress
    forog = get_orog_land_sfc_drag(phis0, parameters.surface_flux.hdrag)
    cdldv = parameters.surface_flux.cdl * denvvs[:, :, 0] * forog
    ustr = -cdldv * ua[kx-1]
    vstr = -cdldv * va[kx-1]

    # Sensible heat flux
    chlcp = parameters.surface_flux.chl * cp
    shf = chlcp * denvvs[:, :, 1] * (tskin - t1[:, :, 0])

    # Evaporation - compute humidity
    def compute_evap_true(operand):
        q1, qsat0, idx = operand
        q1_val, qsat0_val = rel_hum_to_spec_hum(t1[:, :, idx], psa, 1.0, rh[kx-1])
        q1 = q1.at[:, :, idx].set(parameters.surface_flux.fhum0 * q1_val + ghum0 * qa[kx-1])
        qsat0 = qsat0.at[:, :, idx].set(qsat0_val)
        return q1, qsat0

    def compute_evap_false(operand):
        q1, qsat0, idx = operand
        q1 = q1.at[:, :, idx].set(qa[kx-1])
        return q1, qsat0

    q1, qsat0 = jax.lax.cond(
        parameters.surface_flux.fhum0 > 0.0,
        compute_evap_true,
        compute_evap_false,
        operand=(q1, qsat0, 0)
    )

    qsat0 = qsat0.at[:, :, 0].set(get_qsat(tskin, psa, 1.0))

    evap = parameters.surface_flux.chl * denvvs[:, :, 1] * jnp.maximum(
        0.0, soilw_am * qsat0[:, :, 0] - q1[:, :, 0]
    )

    # Longwave radiation from surface and net heat flux
    tsk3 = tskin ** 3.0
    drls = 4.0 * esbc * tsk3
    rlus = esbc * tsk3 * tskin

    hfluxn = rsds * (1.0 - alb_l) + rlds - (rlus + shf + (alhc * evap))

    # Energy balance adjustment of skin temperature
    def skin_temp(operand):
        hfluxn_in, rlus_in, evap_in, shf_in, tskin_in, qsat0_in = operand

        # Compute net heat flux including flux into ground
        clamb = parameters.surface_flux.clambda + (snowc * (parameters.surface_flux.clambsn - parameters.surface_flux.clambda))
        hfluxn_adj = hfluxn_in - (clamb * (tskin_in - stl_am))
        dtskin_test = tskin_in + 1.0

        # Compute d(Evap) for a 1-degree increment of Tskin
        qsat0_next = get_qsat(dtskin_test, psa, 1.0)
        dqsat = jnp.where(
            evap_in > 0.0,
            soilw_am * (qsat0_next - qsat0_in),
            0.0
        )

        # Redefine skin temperature to balance the heat budget
        dtskin = hfluxn_adj / (
            clamb + drls + (parameters.surface_flux.chl * denvvs[:, :, 1] * (cp + (alhc * dqsat)))
        )
        tskin_new = tskin_in + dtskin

        # Add linear corrections to heat fluxes
        shf_new = shf_in + chlcp * denvvs[:, :, 1] * dtskin
        evap_new = evap_in + parameters.surface_flux.chl * denvvs[:, :, 1] * dqsat * dtskin
        rlus_new = rlus_in + drls * dtskin
        hfluxn_new = clamb * (tskin_new - stl_am)

        return (hfluxn_new, rlus_new, evap_new, shf_new, tskin_new, qsat0_in)

    hfluxn, rlus, evap, shf, tskin, qsat0 = jax.lax.cond(
        parameters.surface_flux.lskineb,
        skin_temp,
        pass_fn,
        operand=(hfluxn, rlus, evap, shf, tskin, qsat0[:, :, 0])
    )

    return ustr, vstr, shf, evap, rlus, hfluxn, tskin


@jit
def compute_sea_surface_fluxes(
    ua: jnp.ndarray,
    va: jnp.ndarray,
    ta: jnp.ndarray,
    qa: jnp.ndarray,
    rh: jnp.ndarray,
    phi: jnp.ndarray,
    psa: jnp.ndarray,
    sea_surface_temperature: jnp.ndarray,
    rsds: jnp.ndarray,
    rlds: jnp.ndarray,
    alb_s: jnp.ndarray,
    t1_sea: jnp.ndarray,
    t2_sea: jnp.ndarray,
    denvvs_base: jnp.ndarray,
    parameters: Parameters,
    esbc: float,
    ghum0: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute surface fluxes over ocean surfaces.

    Parameters
    ----------
    ua, va : 3D arrays
        Wind components at model levels
    ta : 3D array
        Temperature at model levels
    qa : 3D array
        Specific humidity at model levels
    rh : 3D array
        Relative humidity at model levels
    phi : 3D array
        Geopotential at model levels
    psa : 2D array
        Normalized surface pressure
    sea_surface_temperature : 2D array
        Sea surface temperature
    rsds : 2D array
        Downward shortwave radiation at surface
    rlds : 2D array
        Downward longwave radiation at surface
    alb_s : 2D array
        Sea surface albedo
    t1_sea : 2D array
        Extrapolated temperature for sea (from land calculation)
    t2_sea : 2D array
        Dry-adiabatic extrapolated temperature for sea
    denvvs_base : 2D array
        Base density * wind speed
    parameters : Parameters
        Model parameters
    esbc : float
        Emissivity * Stefan-Boltzmann constant
    ghum0 : float
        1.0 - fhum0

    Returns
    -------
    ustr : 2D array
        U-component of wind stress over sea
    vstr : 2D array
        V-component of wind stress over sea
    shf : 2D array
        Sensible heat flux over sea
    evap : 2D array
        Evaporation over sea
    rlus : 2D array
        Upward longwave radiation over sea
    hfluxn : 2D array
        Net heat flux into sea surface
    """
    kx = ta.shape[0]
    ix, il = psa.shape
    q1 = jnp.zeros((ix, il, 2))
    qsat0 = jnp.zeros((ix, il, 2))

    # Stability correction for sea surface
    rdth = parameters.surface_flux.fstab / parameters.surface_flux.dtheta
    astab = jax.lax.cond(
        parameters.surface_flux.lscasym,
        lambda _: jnp.array(0.5),
        lambda _: jnp.array(1.0),
        operand=None
    )

    dths = jnp.where(
        sea_surface_temperature > t2_sea,
        jnp.minimum(parameters.surface_flux.dtheta, sea_surface_temperature - t2_sea),
        jnp.maximum(-parameters.surface_flux.dtheta, astab * (sea_surface_temperature - t2_sea))
    )

    denvvs_sea = denvvs_base * (1.0 + dths * rdth)

    # Humidity for sea surface
    def compute_evap_true(operand):
        q1, qsat0, idx = operand
        t1_val = jax.lax.cond(idx == 1, lambda: t1_sea, lambda: jnp.zeros_like(psa))
        q1_val, qsat0_val = rel_hum_to_spec_hum(t1_val, psa, 1.0, rh[kx-1])
        q1 = q1.at[:, :, idx].set(parameters.surface_flux.fhum0 * q1_val + ghum0 * qa[kx-1])
        qsat0 = qsat0.at[:, :, idx].set(qsat0_val)
        return q1, qsat0

    def compute_evap_false(operand):
        q1, qsat0, idx = operand
        q1 = q1.at[:, :, idx].set(qa[kx-1])
        return q1, qsat0

    q1, qsat0 = jax.lax.cond(
        parameters.surface_flux.fhum0 > 0.0,
        compute_evap_true,
        compute_evap_false,
        operand=(q1, qsat0, 1)
    )

    # Wind stress
    cdsdv = parameters.surface_flux.cds * denvvs_sea
    ustr = -cdsdv * ua[kx-1]
    vstr = -cdsdv * va[kx-1]

    # Sensible heat flux
    shf = parameters.surface_flux.chs * cp * denvvs_sea * (sea_surface_temperature - t1_sea)

    # Evaporation
    qsat0_sea = get_qsat(sea_surface_temperature, psa, 1.0)
    evap = parameters.surface_flux.chs * denvvs_sea * (qsat0_sea - q1[:, :, 1])

    # Longwave emission and net heat flux
    rlus = esbc * (sea_surface_temperature ** 4.0)
    hfluxn = rsds * (1.0 - alb_s) + rlds - rlus + shf + alhc * evap

    return ustr, vstr, shf, evap, rlus, hfluxn


@jit
def get_surface_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """
    Compute surface fluxes for land and sea surfaces.

    Parameters
    ----------
    state : PhysicsState
        Current atmospheric state
    physics_data : PhysicsData
        Physics diagnostic data
    parameters : Parameters
        Model parameters
    forcing : ForcingData
        Forcing data (SST, soil moisture, etc.)
    geometry : Geometry
        Grid geometry

    Returns
    -------
    physics_tendencies : PhysicsTendency
        Tendencies due to surface fluxes
    physics_data : PhysicsData
        Updated physics data with surface flux diagnostics
    """
    stl_am = forcing.stl_am
    lfluxland = forcing.lfluxland
    kx, ix, il = state.temperature.shape

    psa = state.normalized_surface_pressure
    ua = state.u_wind
    va = state.v_wind
    ta = state.temperature
    qa = state.specific_humidity
    phi = state.geopotential
    fmask = geometry.fmask

    rsds = physics_data.shortwave_rad.rsds
    rlds = physics_data.surface_flux.rlds

    rh = physics_data.humidity.rh
    phi0 = geometry.orog * grav  # surface geopotential

    snowc = physics_data.mod_radcon.snowc
    alb_l = physics_data.mod_radcon.alb_l
    alb_s = physics_data.mod_radcon.alb_s

    # Initialize variables
    esbc = parameters.mod_radcon.emisfc * sbc
    ghum0 = 1.0 - parameters.surface_flux.fhum0

    ustr = jnp.zeros((ix, il, 3))
    vstr = jnp.zeros((ix, il, 3))
    shf = jnp.zeros((ix, il, 3))
    evap = jnp.zeros((ix, il, 3))
    rlus = jnp.zeros((ix, il, 3))
    hfluxn = jnp.zeros((ix, il, 2))
    t1 = jnp.zeros((ix, il, 2))
    t2 = jnp.zeros((ix, il, 2))
    denvvs = jnp.zeros((ix, il, 3))

    u0 = parameters.surface_flux.fwind0 * ua[kx-1]
    v0 = parameters.surface_flux.fwind0 * va[kx-1]

    # Compute t1, t2, denvvs (needed for both land and sea surface calculations)
    rcp = 1.0 / cp
    nl1 = kx - 1
    gtemp0 = 1.0 - parameters.surface_flux.ftemp0

    dt1 = geometry.wvi[kx-1, 1, jnp.newaxis, jnp.newaxis] * (ta[kx-1] - ta[nl1-1])
    t1 = t1.at[:, :, 0].add(ta[kx-1] + dt1)
    t1 = t1.at[:, :, 1].set(t1[:, :, 0] - phi0 * dt1 / (rgas * 288.0 * geometry.sigl[kx-1]))

    t2 = t2.at[:, :, 1].set(ta[kx-1] + rcp * phi[kx-1])
    t2 = t2.at[:, :, 0].set(t2[:, :, 1] - rcp * phi0)

    t1 = jnp.where(
        (ta[kx-1] > ta[nl1-1])[:, :, jnp.newaxis],
        parameters.surface_flux.ftemp0 * t1 + gtemp0 * t2,
        ta[kx-1][:, :, jnp.newaxis]
    )

    t0 = t1[:, :, 1] + fmask * (t1[:, :, 0] - t1[:, :, 1])
    denvvs = denvvs.at[:, :, 0].set(
        (p0 * psa / (rgas * t0)) * jnp.sqrt(u0**2 + v0**2 + parameters.surface_flux.vgust**2)
    )

    # Compute land surface fluxes
    def land_fluxes_wrapper(operand):
        """Wrapper to call compute_land_surface_fluxes with jax.lax.cond signature."""
        (u0, v0, ustr, vstr, shf, evap, rlus, hfluxn, tskin) = operand

        ustr_land, vstr_land, shf_land, evap_land, rlus_land, hfluxn_land, tskin = compute_land_surface_fluxes(
            u0=u0,
            v0=v0,
            ua=ua,
            va=va,
            ta=ta,
            qa=qa,
            rh=rh,
            phi=phi,
            phi0=phi0,
            psa=psa,
            fmask=fmask,
            stl_am=stl_am,
            soilw_am=forcing.soilw_am,
            rsds=rsds,
            rlds=rlds,
            alb_l=alb_l,
            snowc=snowc,
            phis0=geometry.phis0,
            wvi=geometry.wvi,
            sigl=geometry.sigl,
            coa=geometry.coa,
            parameters=parameters,
            esbc=esbc,
            ghum0=ghum0,
        )

        ustr = ustr.at[:, :, 0].set(ustr_land)
        vstr = vstr.at[:, :, 0].set(vstr_land)
        shf = shf.at[:, :, 0].set(shf_land)
        evap = evap.at[:, :, 0].set(evap_land)
        rlus = rlus.at[:, :, 0].set(rlus_land)
        hfluxn = hfluxn.at[:, :, 0].set(hfluxn_land)

        return (u0, v0, ustr, vstr, shf, evap, rlus, hfluxn, tskin)

    tskin = jnp.zeros_like(stl_am)
    u0, v0, ustr, vstr, shf, evap, rlus, hfluxn, tskin = jax.lax.cond(
        lfluxland,
        land_fluxes_wrapper,
        pass_fn,
        operand=(u0, v0, ustr, vstr, shf, evap, rlus, hfluxn, tskin)
    )

    # Compute sea surface fluxes
    ustr_sea, vstr_sea, shf_sea, evap_sea, rlus_sea, hfluxn_sea = compute_sea_surface_fluxes(
        ua=ua,
        va=va,
        ta=ta,
        qa=qa,
        rh=rh,
        phi=phi,
        psa=psa,
        sea_surface_temperature=forcing.sea_surface_temperature,
        rsds=rsds,
        rlds=rlds,
        alb_s=alb_s,
        t1_sea=t1[:, :, 1],
        t2_sea=t2[:, :, 1],
        denvvs_base=denvvs[:, :, 0],
        parameters=parameters,
        esbc=esbc,
        ghum0=ghum0,
    )

    ustr = ustr.at[:, :, 1].set(ustr_sea)
    vstr = vstr.at[:, :, 1].set(vstr_sea)
    shf = shf.at[:, :, 1].set(shf_sea)
    evap = evap.at[:, :, 1].set(evap_sea)
    rlus = rlus.at[:, :, 1].set(rlus_sea)
    hfluxn = hfluxn.at[:, :, 1].set(hfluxn_sea)

    # Weighted average of surface fluxes and temperatures according to land-sea mask
    weighted_average = lambda var: var[:, :, 1] + fmask * (var[:, :, 0] - var[:, :, 1])

    def weight_avg_landfluxes(operand):
        ustr, vstr, shf, evap, rlus, t1, t0, tsfc, tskin = operand
        ustr = ustr.at[:, :, 2].set(weighted_average(ustr))
        vstr = vstr.at[:, :, 2].set(weighted_average(vstr))
        shf = shf.at[:, :, 2].set(weighted_average(shf))
        evap = evap.at[:, :, 2].set(weighted_average(evap))
        rlus = rlus.at[:, :, 2].set(weighted_average(rlus))

        t0 = weighted_average(t1)

        tsfc = forcing.sea_surface_temperature + fmask * (stl_am - forcing.sea_surface_temperature)
        tskin = forcing.sea_surface_temperature + fmask * (tskin - forcing.sea_surface_temperature)

        return (ustr, vstr, shf, evap, rlus, t1, t0, tsfc, tskin)

    t0 = jnp.zeros_like(t1[:, :, 0])
    tsfc = jnp.zeros_like(stl_am)
    ustr, vstr, shf, evap, rlus, t1, t0, tsfc, tskin = jax.lax.cond(
        lfluxland, weight_avg_landfluxes, pass_fn, operand=(ustr, vstr, shf, evap, rlus, t1, t0, tsfc, tskin)
    )

    surface_flux_out = physics_data.surface_flux.copy(
        ustr=ustr, vstr=vstr, shf=shf, evap=evap, rlus=rlus,
        hfluxn=hfluxn, tsfc=tsfc, tskin=tskin, u0=u0, v0=v0, t0=t0
    )
    physics_data = physics_data.copy(surface_flux=surface_flux_out)

    # Compute tendencies due to surface fluxes
    rps = 1.0 / state.normalized_surface_pressure
    utend = jnp.zeros_like(state.u_wind).at[-1].add(ustr[:, :, 2] * rps * geometry.grdsig[-1])
    vtend = jnp.zeros_like(state.v_wind).at[-1].add(vstr[:, :, 2] * rps * geometry.grdsig[-1])
    ttend = jnp.zeros_like(state.temperature).at[-1].add(shf[:, :, 2] * rps * geometry.grdscp[-1])
    qtend = jnp.zeros_like(state.specific_humidity).at[-1].add(evap[:, :, 2] * rps * geometry.grdsig[-1])
    physics_tendencies = PhysicsTendency(utend, vtend, ttend, qtend)

    return physics_tendencies, physics_data

@jit
def get_orog_land_sfc_drag(phis0, hdrag):
    """
    Parameters
    ----------
    phi0 : Array
        - Array used for calculating the forog
    """

    rhdrag = 1/(grav*hdrag)

    forog = 1.0 + rhdrag*(1.0 - jnp.exp(-jnp.maximum(phis0, 0.0)*rhdrag))

    return forog
