import jax.numpy as jnp
from physical_constants import sbc, wvi
from mod_radcon import epslw, emisfc, fband, tau2, st4a, flux, stratc
from geometry import dhs
from params import ix, il, iy, kx
import os
import jax
nband = 4

# wvi = jnp.zeros((kx, 2)) # Weights for vertical interpolation
# tau2 = jnp.zeros((ix, il, kx, 4))     # Transmissivity of atmospheric layers
# st4a = jnp.zeros((ix, il, kx, 2))     # Blackbody emission from full and half atmospheric levels
# stratc = jnp.zeros((ix, il, 2))       # Stratospheric correction term
# flux = jnp.zeros((ix, il, 4))         # Radiative flux in different spectral bands

os.environ['JAX_PLATFORMS'] = 'cpu'

def get_downward_longwave_rad_fluxes(ta, fband, st4a, flux):

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
    return fsfcd, dfabs

def get_upward_longwave_rad_fluxes(ta, ts, fsfcd, fsfcu, fsfc, ftop, dfabs):
    # ta(ix,il,kx)    !! Absolute temperature
    # ts(ix,il)       !! Surface temperature
    # fsfcd(ix,il)    !! Downward flux of long-wave radiation at the surface
    # fsfcu(ix,il)    !! Surface blackbody emission
    # fsfc(ix,il)     !! Net upward flux of long-wave radiation at the surface
    # ftop(ix,il)     !! Outgoing flux of long-wave radiation at the top of the atmosphere
    # dfabs(ix,il,kx) !! Flux of long-wave radiation absorbed in each atmospheric layer
    
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

    return fsfc, ftop, dfabs, st4a

def radset(fband):
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

    return fband

def initialize_arrays(ix, il, kx):
    # Initialize arrays
    ta = jnp.zeros((ix, il, kx))
    fsfcd = jnp.zeros((ix, il))
    dfabs = jnp.zeros((ix, il, kx))

    # Set the min and max values
    min_val = 130.0
    max_val = 250.0
    
    # Calculate step size
    total_elements = ix * il * kx
    step_size = (max_val - min_val) / (total_elements - 1)
    print(step_size)

    # Create a range of values and reshape to match the ta array shape
    values = jnp.arange(min_val, max_val + step_size, step_size)
    ta = ta.at[:,:,:].set(jnp.reshape(values, (ix, il, kx)))
    for k in range(kx):
        for j in range(il):
            for i in range(ix):
                val = i + (j)*ix + (k)*ix*il
                ta = ta.at[i,j,k].set(min_val + step_size*val)
    
    return ta, fsfcd, dfabs

# Example usage
# ix, il, kx = 96, 48, 8  # Define dimensions
ta, fsfcd, dfabs = initialize_arrays(ix, il, kx)

print(ta[0,0,:,])

fband = radset(fband)
fsfcd, dfabs = get_downward_longwave_rad_fluxes(ta, fband, st4a, flux)

print(fsfcd[:5, :5])

