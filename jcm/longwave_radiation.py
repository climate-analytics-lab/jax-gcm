import jax
from jax import jit
import jax.numpy as jnp
from jcm.geometry import Geometry
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.physical_constants import sbc
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData

nband = 4

@jit
def get_downward_longwave_rad_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """
    Calculate the downward longwave radiation fluxes
    
    Args:
        ta: Absolute temperature - state.temperature
        fband: Energy fraction emitted in each LW band = f(T) - modradcon.fband
        st4a: Blackbody emission from full and half atmospheric levels - modradcon.st4a
        flux: Radiative flux in different spectral bands - modradcon.flux

    Returns:
        rlds: Downward flux of long-wave radiation at the surface
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
    """
    kx, ix, il = state.temperature.shape
    ta = state.temperature
    st4a = physics_data.mod_radcon.st4a
    flux = physics_data.mod_radcon.flux
    tau2 = physics_data.mod_radcon.tau2

    nl1 = kx - 1

    # 1. Blackbody emission from atmospheric levels.
    # The linearized gradient of the blakbody emission is computed
    # from temperatures at layer boundaries, which are interpolated
    # assuming a linear dependence of T on log_sigma.
    # Above the first (top) level, the atmosphere is assumed isothermal.
    
    # Temperature at level boundaries
    st4a = st4a.at[:nl1,:,:,0].set(ta[:nl1]+geometry.wvi[:nl1,1,jnp.newaxis,jnp.newaxis]*(ta[1:nl1+1]-ta[:nl1]))
    
    # Mean temperature in stratospheric layers
    st4a = st4a.at[0,:,:,1].set(0.75 * ta[0] + 0.25 * st4a[0,:,:,0])
    st4a = st4a.at[1,:,:,1].set(0.50 * ta[1] + 0.25 * (st4a[0,:,:,0] + st4a[1,:,:,0]))

    # Temperature gradient in tropospheric layers
    anis = 1
    
    st4a = st4a.at[2:nl1,:,:,1].set(0.5 * anis * jnp.maximum(st4a[2:nl1, :, :, 0] - st4a[1:nl1-1, :, :, 0], 0.0))
    st4a = st4a.at[kx-1,:,:,1].set(anis * jnp.maximum(ta[kx-1] - st4a[nl1-1,:,:,0], 0.0))
    
    # Blackbody emission in the stratosphere
    st4a = st4a.at[:2,:,:,0].set(sbc * st4a[:2, :, :, 1]**4.0)
    st4a = st4a.at[:2,:,:,1].set(0.0)

    # Blackbody emission in the troposphere
    st3a = sbc * ta[2:kx]**3.0
    st4a = st4a.at[2:kx,:,:,0].set(st3a * ta[2:kx])
    st4a =  st4a.at[2:kx,:,:,1].set(4.0 * st3a * st4a[2:kx,:,:,1])

    # 2. Initialization of fluxes
    rlds = jnp.zeros((ix, il))
    dfabs = jnp.zeros((kx, ix, il))

    # 3. Emission and absorption of longwave downward flux.
    #    For downward emission, a correction term depending on the
    #    local temperature gradient and on the layer transmissivity is
    #    added to the average (full-level) emission of each layer.
    
    # 3.1 Stratosphere
    k = 0
    emis = 1 - tau2
    brad = radset(ta, parameters.mod_radcon.epslw) * (st4a[:,:,:,0,jnp.newaxis] + emis*st4a[:,:,:,1,jnp.newaxis])
    emis_brad = emis * brad
    flux = emis_brad[k].at[:,:,2:nband].set(0.0)
    dfabs = dfabs.at[k].add(-jnp.sum(flux,axis=-1))

    # 3.2 Troposphere
    _flux_3d = jnp.zeros((kx, ix, il, nband)).at[0].set(flux)
    _flux_3d = _flux_3d.at[1:].set(jax.lax.scan(
        jax.checkpoint(lambda carry, k: 2*(tau2[k] * carry + emis_brad[k],)),
        flux,
        jnp.arange(1, kx) # scan from TOA to surface
    )[1])
    flux = _flux_3d[-1]
    
    dfabs = dfabs.at[1:].add(-jnp.diff(jnp.sum(_flux_3d, axis=-1), axis=0))

    rlds = jnp.sum(parameters.mod_radcon.emisfc * flux, axis=-1)

    corlw = parameters.mod_radcon.epslw * parameters.mod_radcon.emisfc * st4a[kx-1,:,:,0]
    dfabs = dfabs.at[-1].add(-corlw)
    rlds += corlw

    surface_flux_out = physics_data.surface_flux.copy(rlds=rlds)
    longwave_out = physics_data.longwave_rad.copy(dfabs=dfabs)
    mod_radcon_out = physics_data.mod_radcon.copy(st4a=st4a)
    physics_data = physics_data.copy(
        surface_flux=surface_flux_out, longwave_rad=longwave_out, mod_radcon=mod_radcon_out
    )
    physics_tendencies = PhysicsTendency(
        jnp.zeros_like(state.u_wind),
        jnp.zeros_like(state.v_wind),
        jnp.zeros_like(state.temperature),
        jnp.zeros_like(state.temperature)
    )

    return physics_tendencies, physics_data

@jit
def get_upward_longwave_rad_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """
    Calculate the upward longwave radiation fluxes
    
    Args:
        ta: Absolute temperature
        ts: Surface temperature - surface_fluxes.tsfc
        rlds: Downward flux of long-wave radiation at the surface
        fsfcu: Surface blackbody emission - taken from rlus from surface fluxes
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
        st4a: Blackbody emission from full and half atmospheric levels - mod_radcon.st4a
    
    Returns:
        fsfc: Net upward flux of long-wave radiation at the surface
        ftop: Outgoing flux of long-wave radiation at the top of the atmosphere
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
        st4a: Blackbody emission from full and half atmospheric levels - mod_radcon.st4a
    """
    kx, ix, il = state.temperature.shape
    ta = state.temperature
    dfabs = physics_data.longwave_rad.dfabs
    rlds = physics_data.surface_flux.rlds

    st4a = physics_data.mod_radcon.st4a
    flux = physics_data.mod_radcon.flux
    tau2 = physics_data.mod_radcon.tau2
    stratc = physics_data.mod_radcon.stratc

    fsfcu = physics_data.surface_flux.rlus[:,:,2] # FIXME
    ts = physics_data.surface_flux.tsfc # called tsfc in surface_fluxes.f90
    refsfc = 1.0 - parameters.mod_radcon.emisfc
    epslw = parameters.mod_radcon.epslw
    fsfc = fsfcu - rlds
    
    flux = radset(ts, epslw) * fsfcu[:,:,jnp.newaxis] + refsfc * flux

    # Troposphere
    # correction for 'black' band
    dfabs = dfabs.at[-1].add(parameters.mod_radcon.epslw * fsfcu)

    emis = 1. - tau2
    brad = radset(ta, epslw) * (st4a[:,:,:,0,jnp.newaxis] - emis*st4a[:,:,:,1,jnp.newaxis])
    emis_brad = emis * brad

    _flux_3d = jnp.zeros((kx, ix, il, nband)).at[-1].set(flux)
    _flux_3d = _flux_3d.at[:-1].set(jax.lax.scan(
        jax.checkpoint(lambda carry, k: 2*(tau2[k] * carry + emis_brad[k],)),
        flux,
        jnp.arange(kx-1, 0, -1) # scan from surface to TOA
    )[1][::-1])
    flux = _flux_3d[0]

    dfabs = dfabs.at[1:].add(jnp.diff(jnp.sum(_flux_3d, axis=-1), axis=0))

    flux = flux.at[:,:,:2].set((tau2[0] * flux + emis_brad[0])[:,:,:2])
    dfabs = dfabs.at[0].add(jnp.sum((_flux_3d[0] - flux)[:,:,:2], axis=-1))

    corlw1 = geometry.dhs[0] * stratc[:,:,1] * st4a[0,:,:,0] + stratc[:,:,0]
    corlw2 = geometry.dhs[1] * stratc[:,:,1] * st4a[1,:,:,0]
    dfabs = dfabs.at[0].add(-corlw1)
    dfabs = dfabs.at[1].add(-corlw2)
    ftop = corlw1 + corlw2

    ftop += jnp.sum(flux, axis = -1)

    surface_flux_out = physics_data.surface_flux.copy(rlns=fsfc)
    longwave_out = physics_data.longwave_rad.copy(ftop=ftop, dfabs=dfabs)
    mod_radcon_out = physics_data.mod_radcon.copy(st4a=st4a, flux=flux)
    physics_data = physics_data.copy(
        surface_flux=surface_flux_out, longwave_rad=longwave_out, mod_radcon=mod_radcon_out
    )
    
    # Compute temperature tendency due to absorbed lw flux: logic from physics.f90:182-184
    ttend_lwr = dfabs * geometry.grdscp[:, jnp.newaxis, jnp.newaxis] / physics_data.convection.psa[jnp.newaxis]
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape,temperature=ttend_lwr)
    
    return physics_tendencies, physics_data

@jit
def radset(temp, epslw):
    """
    Compute energy fractions in longwave bands as a function of temperature

    Args:
        temp: Absolute temperature
        epslw: Longwave emissivity in PBL

    Returns:
        fband: Energy fraction emitted in each LW band
    """
    jtemp = jnp.clip(temp, 200, 320) # To retain backwards compatibility with F90 code
    
    fband = jnp.stack((
        jnp.zeros_like(jtemp),
        0.148 - 3.0e-6 * (jtemp - 247) ** 2,
        0.356 - 5.2e-6 * (jtemp - 282) ** 2,
        0.314 + 1.0e-5 * (jtemp - 315) ** 2,
    ), axis=-1)
    fband = fband.at[..., 0].set(1. - fband.sum(axis=-1))
    
    return (1. - epslw) * fband