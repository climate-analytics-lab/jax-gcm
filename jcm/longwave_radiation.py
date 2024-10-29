import jax.numpy as jnp
from jcm.physical_constants import sbc, wvi
from jcm.mod_radcon import epslw, emisfc, nband, fband
from jcm.geometry import dhs
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData

nband = 4
fband = None

def get_downward_longwave_rad_fluxes(state: PhysicsState, physics_data: PhysicsData):

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
    ix, il, kx = state.temperature.shape
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
    st4a = st4a.at[:,:,:nl1,0].set(ta[:,:,:nl1]+wvi[:nl1,1]*(ta[:,:,1:nl1+1]-ta[:,:,:nl1]))
    
    # Mean temperature in stratospheric layers
    st4a = st4a.at[:,:,0,1].set(0.75 * ta[:,:,0] + 0.25 * st4a[:,:,0,0])
    st4a = st4a.at[:,:,1,1].set(0.50 * ta[:,:,1] + 0.25 * (st4a[:,:,0,0] + st4a[:,:,1,0]))

    # Temperature gradient in tropospheric layers
    anis = 1
    
    st4a = st4a.at[:,:,2:nl1,1].set(0.5 * anis * jnp.maximum(st4a[:, :, 2:nl1, 0] - st4a[:, :, 1:nl1-1, 0], 0.0))
    st4a = st4a.at[:,:,kx-1,1].set(anis * jnp.maximum(ta[:,:,kx-1] - st4a[:,:,nl1-1,0], 0.0))
    
    # Blackbody emission in the stratosphere
    st4a = st4a.at[:,:,:2,0].set(sbc * st4a[:, :, :2, 1]**4.0)
    st4a = st4a.at[:,:,:2,1].set(0.0)

    # Blackbody emission in the troposphere
    st3a = sbc * ta[:, :, 2:kx]**3.0
    st4a = st4a.at[:,:,2:kx,0].set(st3a * ta[:,:,2:kx])
    st4a =  st4a.at[:,:,2:kx,1].set(4.0 * st3a * st4a[:,:,2:kx,1])

    # 2. Initialization of fluxes
    rlds = jnp.zeros((ix, il))
    dfabs = jnp.zeros((ix, il, kx))

    # 3. Emission and absorption of longwave downward flux.
    #    For downward emission, a correction term depending on the 
    #    local temperature gradient and on the layer transmissivity is
    #    added to the average (full-level) emission of each layer.
    
    # 3.1 Stratosphere
    ta_rounded = jnp.round(ta).astype(int)
    k = 0
    for jb in range(2):
        emis = 1 - tau2[:,:,k,jb]
        brad = fband[ta_rounded[:,:,k]-100, jb] * (st4a[:,:,k,0] + emis*st4a[:,:,k,1])
        flux = flux.at[:,:,jb].set(emis * brad)
        dfabs = dfabs.at[:,:,k].set(dfabs[:,:,k] - flux[:,:,jb])
    
    flux = flux.at[:,:,2:nband].set(0.0)

    # 3.2 Troposhere
    for jb in range(nband):
        for k in range(1,kx):
            emis = 1 - tau2[:,:,k,jb]
            brad = fband[ta_rounded[:,:,k]-100, jb] * (st4a[:,:,k,0] + emis*st4a[:,:,k,1])
            dfabs = dfabs.at[:, :, k].add(flux[:, :, jb])  # Equivalent to dfabs[:,:,k] += flux[:,:,jb]
            flux = flux.at[:, :, jb].set((tau2[:, :, k, jb] * flux[:, :, jb]) + (emis * brad))  # Equivalent to flux[:,:,jb] = tau2[:,:,k,jb]*flux[:,:,jb] + emis*brad
            dfabs = dfabs.at[:, :, k].add(-flux[:, :, jb])

    rlds = jnp.sum(emisfc * flux, axis=-1)

    corlw = epslw * emisfc * st4a[:,:,kx-1,0]
    dfabs = dfabs.at[:,:,kx-1].add(-corlw)
    rlds = rlds + corlw

    longwave_out = physics_data.longwave_rad.copy(rlds=rlds, dfabs=dfabs)
    mod_radcon_out = physics_data.mod_radcon.copy(st4a=st4a)
    physics_data = physics_data.copy(longwave_rad=longwave_out, mod_radcon=mod_radcon_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))

    return physics_tendencies, physics_data

def get_upward_longwave_rad_fluxes(state: PhysicsState, physics_data: PhysicsData):
    """
    Calculate the upward longwave radiation fluxes
    
    Args:
        ta: Absolute temperature
        ts: Surface temperature - surface_fluxes.tsfc
        rlds: Downward flux of long-wave radiation at the surface
        fsfcu: Surface blackbody emission - taken from slru from surface fluxes
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
        st4a: Blackbody emission from full and half atmospheric levels - mod_radcon.st4a
    
    Returns:
        fsfc: Net upward flux of long-wave radiation at the surface
        ftop: Outgoing flux of long-wave radiation at the top of the atmosphere
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
        st4a: Blackbody emission from full and half atmospheric levels - mod_radcon.st4a
    
    """
    _, _, kx = state.temperature.shape
    ta = state.temperature
    dfabs = physics_data.longwave_rad.dfabs
    rlds = physics_data.longwave_rad.rlds

    st4a = physics_data.mod_radcon.st4a
    flux = physics_data.mod_radcon.flux
    tau2 = physics_data.mod_radcon.tau2
    stratc = physics_data.mod_radcon.stratc

    fsfcu = physics_data.surface_flux.slru[:,:,2] 
    ts = physics_data.surface_flux.tsfc # called tsfc in surface_fluxes.f90
    refsfc = 1.0 - emisfc
    fsfc = fsfcu - rlds
    
    ts_rounded = jnp.round(ts).astype(int)  # Rounded ts
    ta_rounded = jnp.round(ta).astype(int)  # Rounded ta

    flux = fband[ts_rounded-100,:] * fsfcu[:,:,jnp.newaxis] + refsfc * flux

    # Troposphere
    # correction for 'black' band
    dfabs = dfabs.at[:,:,-1].set(dfabs[:,:,-1] + epslw * fsfcu)

    for jb in range(nband):
        for k in range(kx-1, 0, -1):
            emis = 1.0 - tau2[:,:,k,jb]
            brad = fband[ta_rounded[:,:,k]-100, jb] * (st4a[:,:,k,0] - emis*st4a[:,:,k,1])
            dfabs = dfabs.at[:,:,k].add(flux[:,:,jb])
            flux = flux.at[:,:,jb].set(tau2[:,:,k,jb] * flux[:,:,jb] + emis * brad)
            dfabs = dfabs.at[:,:,k].add(-flux[:,:,jb])

    k = 0
    for jb in range(2):
        emis = 1.0 - tau2[:,:,k,jb]
        brad = fband[ta_rounded[:,:,k]-100, jb] * (st4a[:,:,k,0] - emis*st4a[:,:,k,1])
        dfabs = dfabs.at[:,:,k].add(flux[:,:,jb])
        flux = flux.at[:,:,jb].set(tau2[:,:,k,jb] * flux[:,:,jb] + emis * brad)
        dfabs = dfabs.at[:,:,k].add(-flux[:,:,jb])

    corlw1 = dhs[0] * stratc[:,:,1] * st4a[:,:,0,0] + stratc[:,:,0]
    corlw2 = dhs[1] * stratc[:,:,1] * st4a[:,:,1,0]
    dfabs = dfabs.at[:,:,0].set(dfabs[:,:,0] - corlw1)
    dfabs = dfabs.at[:,:,1].set(dfabs[:,:,1] - corlw2)
    ftop = corlw1 + corlw2

    ftop = jnp.sum(flux, axis = -1)

    longwave_out = physics_data.longwave_rad.copy(rlds=fsfc, ftop=ftop, dfabs=dfabs)
    mod_radcon_out = physics_data.mod_radcon.copy(st4a=st4a)
    physics_data = physics_data.copy(longwave_rad=longwave_out, mod_radcon=mod_radcon_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_tendencies, physics_data

def radset():
    """
    Set the energy fraction emitted in each LW band = f(T)
    """
    global fband

    fband = jnp.zeros((301, nband))  # Example shape (100:400, 4)

    eps1 = 1.0 - epslw

    t_min, t_max = 200, 320
    jtemp = jnp.arange(t_min, t_max + 1)
    fband_2 = (0.148 - 3.0e-6 * (jtemp - 247) ** 2) * eps1
    fband_3 = (0.356 - 5.2e-6 * (jtemp - 282) ** 2) * eps1
    fband_4 = (0.314 + 1.0e-5 * (jtemp - 315) ** 2) * eps1
    fband_1 = eps1 - (fband_2 + fband_3 + fband_4)
    fband = fband.at[jtemp - 100, :4].set(jnp.stack((fband_1, fband_2, fband_3, fband_4), axis=-1))

    jb = jnp.arange(4)
    fband = fband.at[:(t_min - 100), jb].set(fband[t_min - 100, jb])
    fband = fband.at[(t_max + 1 - 100):, jb].set(fband[t_max - 100, jb])