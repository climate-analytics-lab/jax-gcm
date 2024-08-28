import jax.numpy as jnp
from jcm.physical_constants import sbc, wvi
from jcm.mod_radcon import epslw, emisfc
from jcm.geometry import dhs
from jcm.physics import PhysicsData, PhysicsState, PhysicsTendency
import tree_math
nband = 4

#FIXME: fband should be moved here from mod_radcon.py? since this is where it is used and set?
@tree_math.struct
class LWRadiaitonData:
    fsfcd: jnp.ndarray
    dfabs: jnp.ndarray
    ftop: jnp.ndarray
    fsfc: jnp.ndarray
    
    def __init__(self, nodal_shape, node_levels, fsfcd=None, dfabs=None, ftop=None, fsfc=None) -> None:
        if fsfcd is not None:
            self.fsfcd = fsfcd
        else:
            self.fsfcd = jnp.zeros((nodal_shape))
        if dfabs is not None:
            self.dfabs = dfabs
        else:
            self.dfabs = jnp.zeros((nodal_shape + (node_levels,)))
        if ftop is not None:
            self.ftop = ftop
        else:
            self.ftop = jnp.zeros((nodal_shape))
        if fsfc is not None:
            self.fsfc = fsfc
        else:
            self.fsfc = jnp.zeros((nodal_shape))



    def copy(self, fsfcd=None, dfabs=None, ftop=None, fsfc=None):
        return LWRadiaitonData(
            (0,0), 
            0, 
            fsfcd=fsfcd if fsfcd is not None else self.fsfcd,
            dfabs=dfabs if dfabs is not None else self.dfabs,
            ftop=ftop if ftop is not None else self.ftop,
            fsfc=fsfc if fsfc is not None else self.fsfc
        )

def get_downward_longwave_rad_fluxes(physics_data: PhysicsData, state: PhysicsState):

    """
    Calculate the downward longwave radiation fluxes
    
    Args:
        ta: Absolute temperature - state.temperature
        fband: Energy fraction emitted in each LW band = f(T) - modradcon.fband
        st4a: Blackbody emission from full and half atmospheric levels - modradcon.st4a
        flux: Radiative flux in different spectral bands - modradcon.flux

    Returns:
        fsfcd: Downward flux of long-wave radiation at the surface
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
    """
    ix, il, kx = state.temperature.shape
    ta = state.temperature
    fband = physics_data.mod_radcon.fband
    st4a = physics_data.mod_radcon.st4a
    flux = physics_data.mod_radcon.flux
    tau2 = physics_data.mod_radcon.tau2

    nl1 = kx - 1
    # Temperature at level boundaries
    st4a = st4a.at[:,:,:nl1,0].set(ta[:,:,:nl1]+wvi[:nl1,1]*(ta[:,:,1:nl1+1]-ta[:,:,:nl1]))
    st4a = st4a.at[:,:,0,1].set(0.75 * ta[:,:,0] + 0.25 * st4a[:,:,0,0])
    st4a = st4a.at[:,:,1,1].set(0.50 * ta[:,:,1] + 0.25 * (st4a[:,:,0,0] + st4a[:,:,1,0]))
    
    anis = 1

    # Temperature gradient in tropospheric layers
    st4a = st4a.at[:,:,2:nl1,1].set(0.5 * anis * jnp.maximum(st4a[:, :, 2:nl1, 0] - st4a[:, :, 1:nl1-1, 0], 0.0))
    st4a = st4a.at[:,:,kx-1,1].set(anis * jnp.maximum(ta[:,:,kx-1] - st4a[:,:,nl1-1,0], 0.0))
    
    # Blackbody emission in the stratosphere
    st4a = st4a.at[:,:,:2,0].set(sbc * st4a[:, :, :2, 1]**4.0)
    st4a = st4a.at[:,:,:2,1].set(0.0)

    # Blackbody emission in the troposphere
    st3a = sbc * ta[:, :, 2:kx]**3.0
    st4a = st4a.at[:,:,2:kx,0].set(st3a * ta[:,:,2:kx])
    st4a =  st4a.at[:,:,2:kx,1].set(4.0 * st3a * st4a[:,:,2:kx,1])

    # Initialization of fluxes
    fsfcd = jnp.zeros((ix, il))
    dfabs = jnp.zeros((ix, il, kx))

    ta_rounded = jnp.round(ta[:, :, :]).astype(int)  # Rounded ta
    #stratosphere
    k = 0
    for jb in range(2):
        emis = 1 - tau2[:,:,k,jb]
        brad = fband[ta_rounded[:,:,k]-100, jb] * (st4a[:,:,k,0] + emis*st4a[:,:,k,1])
        flux = flux.at[:,:,jb].set(emis * brad)
        dfabs = dfabs.at[:,:,k].set(dfabs[:,:,k] - flux[:,:,jb])
    
    flux = flux.at[:,:,2:nband].set(0.0)

    #Troposhere
    for jb in range(nband):
        for k in range(1,kx):
            emis = 1 - tau2[:,:,k,jb]
            brad = fband[ta_rounded[:,:,k]-100, jb] * (st4a[:,:,k,0] + emis*st4a[:,:,k,1])
            dfabs = dfabs.at[:, :, k].add(flux[:, :, jb])  # Equivalent to dfabs[:,:,k] += flux[:,:,jb]
            flux = flux.at[:, :, jb].set((tau2[:, :, k, jb] * flux[:, :, jb]) + (emis * brad))  # Equivalent to flux[:,:,jb] = tau2[:,:,k,jb]*flux[:,:,jb] + emis*brad
            dfabs = dfabs.at[:, :, k].add(-flux[:, :, jb])

    fsfcd = jnp.sum(emisfc * flux, axis=-1)

    corlw = epslw * emisfc * st4a[:,:,kx-1,0]
    dfabs = dfabs.at[:,:,kx-1].add(-corlw)
    fsfcd = fsfcd + corlw

    longwave_out = physics_data.longwave_rad.copy(fsfcd=fsfcd, dfabs=dfabs)
    physics_data = physics_data.copy(longwave_rad=longwave_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))

    return physics_tendencies, physics_data

def get_upward_longwave_rad_fluxes(physics_data: PhysicsData, state: PhysicsState):
    """
    Calculate the upward longwave radiation fluxes
    
    Args:
        ta: Absolute temperature
        ts: Surface temperature - surface_fluxes.ts
        fsfcd: Downward flux of long-wave radiation at the surface
        fsfcu: Surface blackbody emission - slru from surface fluxes (FIXME: currently in auxiliaries.py, should probably be owned by surface fluxes)
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
    dfabs = physics_data.longwave_rad.dfab
    fsfcd = physics_data.longwave_rad.fsfcd

    fband = physics_data.mod_radcon.fband
    st4a = physics_data.mod_radcon.st4a
    flux = physics_data.mod_radcon.flux
    tau2 = physics_data.mod_radcon.tau2
    stratc = physics_data.mod_radcon.stratc

    fsfcu = physics_data.surface_fluxes.slru[:,:,2] # FIXME: slru should be owned by surface fluxes, no idea why this is hard coded to 2, see physics.f90:180
    ts = physics_data.surface_fluxes.ts # called tsfc in surface_fluxes.f90
    refsfc = 1.0 - emisfc
    fsfc = fsfcu - fsfcd
    
    ts_rounded = jnp.round(ts[:, :, :]).astype(int)  # Rounded ts
    ta_rounded = jnp.round(ta[:, :, :]).astype(int)  # Rounded ta

    flux = flux.at[:,:,:].set(fband[ts_rounded[:,:]-100,:] * fsfcu[:,:] + refsfc * flux[:,:,:])

    # Troposphere

    # correction for 'black' band

    dfabs[:,:,-1] = dfabs[:,:,-1] + epslw * fsfcu

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
    dfabs[:,:,0] = dfabs[:,:,0] - corlw1
    dfabs[:,:,1] = dfabs[:,:,1] - corlw2
    ftop = corlw1 + corlw2

    ftop = jnp.sum(flux, axis = -1)

    longwave_out = physics_data.longwave_rad.copy(fsfc=fsfc, ftop=ftop, dfabs=dfabs)
    mod_radcon_out = physics_data.mod_radcon.copy(st4a=st4a)
    physics_data = physics_data.copy(longwave_rad=longwave_out, mod_radcon=mod_radcon_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_data, physics_tendencies



def radset(physics_data: PhysicsData):
    """
    Set the energy fraction emitted in each LW band = f(T)
    """
    
    fband = jnp.zeros((301, nband))  # Example shape (100:400, 4)

    eps1 = 1.0 - epslw

    for jtemp in range(100, 220):
        fband_2 = (0.148 - 3.0e-6 * (jtemp + 100 - 247) ** 2) * eps1
        fband_3 = (0.356 - 5.2e-6 * (jtemp + 100 - 282) ** 2) * eps1
        fband_4 = (0.314 + 1.0e-5 * (jtemp + 100 - 315) ** 2) * eps1
        fband_1 = eps1 - (fband_2 + fband_3 + fband_4)
        fband = fband.at[jtemp, 1].set(fband_2)
        fband = fband.at[jtemp, 2].set(fband_3)
        fband = fband.at[jtemp, 3].set(fband_4)
        fband = fband.at[jtemp, 0].set(fband_1)
    
    for jb in range(4):
        fband = fband.at[:99, jb].set(fband[100, jb])
        fband = fband.at[221:300, jb].set(fband[220,jb])

    modradcon_out = physics_data.mod_radcon.copy(fband=fband)
    physics_data = physics_data.copy(mod_radcon=modradcon_out)
    return physics_data


