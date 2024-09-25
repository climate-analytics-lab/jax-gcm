'''
Date: 2/11/2024
Parametrization of convection. Convection is modelled using a simplified 
version of the Tiedke (1993) mass-flux convection scheme.
'''
import jax
import jax.numpy as jnp
from jcm.physics import PhysicsData, PhysicsTendency, PhysicsState
from jcm.physical_constants import p0, alhc, wvi, grav, sigl, sigh
from jcm.geometry import dhs, fsg

psmin = jnp.array(0.8) # Minimum (normalised) surface pressure for the occurrence of convection
trcnv = jnp.array(6.0) # Time of relaxation (in hours) towards reference state
rhil = jnp.array(0.7) # Relative humidity threshold in intermeduate layers for secondary mass flux
rhbl = jnp.array(0.9) # Relative humidity threshold in the boundary layer
entmax = jnp.array(0.5) # Maximum entrainment as a fraction of cloud-base mass flux
smf = jnp.array(0.8) # Ratio between secondary and primary mass flux at cloud-base

if wvi[0, 1] == 0.:
    """
    wvi is the weights for vertical interpolation. It's calculated in physics f90, but doesn't seem to be calculated in new code. Below is the code I used to 
    calculate it offline, but sigl and sigh seem to be a bit different than in the original Speedy code. Hard coded the wvi values, but would be good to resolve this
    """
    #wvi = wvi.at[:-1, 0].set(1.0 / (sigl[1:] - sigl[:-1]))
    #wvi = wvi.at[:-1, 1].set((jnp.log(sigh[:-1]) - sigl[:-1]) * wvi[:-1, 0])
    #wvi = wvi.at[-1, 1].set((jnp.log(0.99) - sigl[-1]) * wvi[-2, 0])
    wvi=jnp.array([[0.74906313, 0.519211  ],[1.3432906,  0.52088195], [1.8845587,  0.49444085],[2.4663029,  0.5211523 ],[3.3897371,  0.5508966 ],[5.0501776,  0.59072757],[7.7501183,  0.58097243],[0.,         0.31963795]])

def diagnose_convection(psa, se, qa, qsat):
    """
    Diagnose convectively unstable gridboxes  

    Convection is activated in gridboxes with conditional instability. This
    is diagnosed by checking for any tropopsheric half level where the
    saturation moist static energy is lower than in the boundary-layer level.
    In gridboxes where this is true, convection is activated if either: there
    is convective instability - the actual moist static energy at the
    tropospheric level is lower than in the boundary-layer level, or, the
    relative humidity in the boundary-layer level and lowest tropospheric
    level exceed a set threshold (rhbl).

    Args:
    psa: Normalised surface pressure [p/p0]
    se: Dry static energy [c_p.T + g.z]
    qa: Specific humidity [g/kg]
    qsat: Saturation specific humidity [g/kg]

    Returns:
    iptop: Top of convection (layer index)
    qdif: Excess humidity in convective gridboxes

    """
    ix, il, kx = se.shape
    iptop = jnp.full((ix, il), kx + 1, dtype=int)  # Initialize iptop with nlp
    qdif = jnp.zeros((ix, il), dtype=float)

    # Saturation moist static energy
    mss = se + alhc * qsat

    rlhc = 1.0 / alhc

    # Minimum of moist static energy in the lowest two levels
    # Mask for psa > psmin
    mask_psa = psa > psmin 

    mse0 = jnp.where(mask_psa, 0, se[:, :, kx-1] + alhc * qa[:, :, kx-1]) #se[:, :, kx-1] + alhc * qa[:, :, kx-1]
    mse1 = jnp.where(mask_psa, 0, se[:, :, kx-2] + alhc * qa[:, :, kx-2]) #se[:, :, kx-2] + alhc * qa[:, :, kx-2]
    mse1 = jnp.minimum(mse0, mse1)

    # Saturation (or super-saturated) moist static energy in PBL
    mss0 = jnp.maximum(mse0, mss[:, :, kx-1])

    # Compute mss2 array for all k layers (3 to kx-3)
    k_indices = jnp.arange(3, kx-3, dtype=int)
    mss2 = mss[:, :, k_indices] + wvi[k_indices, 1] * (mss[:, :, k_indices + 1] - mss[:, :, k_indices])

    # Check 1: conditional instability (MSS in PBL > MSS at top level)
    mask_conditional_instability = mss0[:, :, None] > mss2
    ktop1 = jnp.full((ix, il), kx, dtype=int)
    ktop1 = k_indices[jnp.argmax(mask_conditional_instability, axis=2)]

    # Check 2: gradient of actual moist static energy between lower and upper 
    # troposphere
    mask_mse1_greater_mss2 = mse1[:, :, None] > mss2
    ktop2 = jnp.full((ix, il), kx, dtype=int)
    ktop2 = k_indices[jnp.argmax(mask_mse1_greater_mss2, axis=2)]
    msthr = jnp.zeros((ix, il), dtype=float)
    msthr = mss2[jnp.arange(ix)[:, None], jnp.arange(il), jnp.argmax(mask_mse1_greater_mss2, axis=2)]

    # Check 3: RH > RH_c at both k=kx and k=kx-1
    qthr0 = rhbl * qsat[:, :, kx-1]
    qthr1 = rhbl * qsat[:, :, kx-2]
    lqthr = (qa[:, :, kx-1] > qthr0) & (qa[:, :, kx-2] > qthr1)

    # Applying masks to iptop and qdif
    mask_ktop1_less_kx = ktop1 < kx
    mask_ktop2_less_kx = ktop2 < kx

    combined_mask1 = mask_ktop1_less_kx & mask_ktop2_less_kx
    iptop = jnp.where(combined_mask1, ktop1, iptop)
    qdif = jnp.where(combined_mask1, jnp.maximum(qa[:, :, kx-1] - qthr0, (mse0 - msthr) * rlhc), qdif)

    combined_mask2 = mask_ktop1_less_kx & lqthr & ~combined_mask1
    iptop = jnp.where(combined_mask2, ktop1, iptop)
    qdif = jnp.where(combined_mask2, qa[:, :, kx-1] - qthr0, qdif)

    return iptop, qdif

def get_convection_tendencies(physics_data: PhysicsData, state: PhysicsState):
    """
    Compute convective fluxes of dry static energy and moisture using a simplified mass-flux scheme.

    Args:
    psa: Normalised surface pressure [p/p0] - state.surface_pressure
    se: Dry static energy [c_p.T + g.z]
    qa: Specific humidity [g/kg] - state.specific_humidity
    qsat: Saturation specific humidity [g/kg] - humidity.qsat

    Returns:
    iptop: Top of convection (layer index)
    cbmf: Cloud-base mass flux
    precnv: Convective precipitation [g/(m^2 s)]
    dfse:  Net flux of dry static energy into each atmospheric layer
    dfqa: Net flux of specific humidity into each atmospheric layer

    """
    conv = physics_data.convection
    humidity = physics_data.humidity
    se = conv.se
    qa = state.specific_humidity
    qsat = humidity.qsat
    _, _, kx = se.shape
    psa = state.surface_pressure #FIXME: normalized surface pressure should be computed from state.surface_pressure, in physics.f90:104 its exp(surface_pressure)??
    # psa = jnp.exp(state.surface_pressure) 
    
    # 1. Initialization of output and workspace arrays

    dfse = jnp.zeros_like(se)
    dfqa = jnp.zeros_like(physics_data.state.specific_humidity)

    cbmf = jnp.zeros_like(psa)
    precnv = jnp.zeros_like(psa)

    #keep indexing consistent with original Speedy
    nl1 = kx - 1 
    nlp = kx + 1

    # Entrainment profile (up to sigma = 0.5)
    entr = jnp.maximum(0.0, fsg[1:kx-1] - 0.5)**2.0
    sentr = jnp.sum(entr)
    entr *= entmax / sentr

    fqmax = 5.0 #maximum mass flux, not sure why this is needed
    fm0 = p0*dhs[-1]/(grav*trcnv*3600.0) #prefactor for mass fluxes
    rdps=2.0/(1.0 - psmin)

    # 2. Check of conditions for convection
    iptop, qdif = diagnose_convection(psa, se, state.specific_humidity, humidity.qsat)

    # 3. Convection over selected grid-points
    mask = iptop < kx
    #3.1 Boundary layer (cloud base)
    k = kx - 1
    k1 = k - 1

    # Maximum specific humidity in the PBL
    qmax = jnp.maximum(1.01 * state.specific_humidity[:, :, -1], humidity.qsat[:, :, -1])

    # Dry static energy and moisture at upper boundary
    sb = se[:, :, k1] + wvi[k1, 1] * (se[:, :, k] - se[:, :, k1])
    qb = jnp.minimum(state.specific_humidity[:, :, k1] + wvi[k1, 1] * (state.specific_humidity[:, :, k] - state.specific_humidity[:, :, k1]), state.specific_humidity[:, :, k])
    
    fpsa = psa * jnp.minimum(1.0, (psa - psmin) * rdps)
    fmass = fm0 * fpsa * jnp.minimum(fqmax, qdif / (qmax - qb))
    cbmf = jnp.where(mask, fmass, cbmf)

    # Upward fluxes at upper boundary
    fus = fmass * se[:, :, -1]
    fuq = fmass * qmax

    # Downward fluxes at upper boundary
    fds = fmass*sb
    fdq = fmass*qb

    #Net flux of dry static energy and moisture
    dfse = dfse.at[:, :, k].set(fds - fus)
    dfqa = dfqa.at[:, :, k].set(fdq - fuq)

    # 3.2 intermediate layers (entrainment)
    #start by making entrainment profile:
    enmass = entr[jnp.newaxis, jnp.newaxis, :] * psa[:, :, jnp.newaxis] * cbmf[:, :, jnp.newaxis]

    #now get mass entrainment 
    """
    N.B in this and following loops, fluxes at one level depend on fluxes at level below, so have to iterate 
    (is there a better way of doing this?)
    """
    #mass flux
    def entrain_loop_fmass(i, var):
        a = kx - i - 1
        var = var.at[:, :, a].set(var[:, :, a + 1] + enmass[:, :, a - 1])
        return var   
    fmass_broadcast = jnp.tile(fmass[:, :, jnp.newaxis], (1, 1, kx))
    fmass_new = jax.lax.fori_loop(1, kx - 2, entrain_loop_fmass, fmass_broadcast)

    #upwards static energy flux
    def entrain_loop_fus(i, var):
        a = kx - i - 1
        var = var.at[:, :, a].set(var[:, :, a + 1] + entrain_se[:, :, a - 1])
        return var
    entrain_se = enmass*se[:, :, 1:kx-1]
    fus_broadcast = jnp.tile(fus[:, :, jnp.newaxis], (1, 1, kx))
    fus_new = jax.lax.fori_loop(1, kx - 2, entrain_loop_fus, fus_broadcast)

    #upwards moisture flux
    def entrain_loop_fuq(i, var):
        a = kx - i - 1
        var = var.at[:, :, a].set(var[:, :, a + 1] + entrain_qa[:, :, a - 1])
        return var
    entrain_qa = enmass*qa[:, :, 1:kx-1]
    fuq_broadcast = jnp.tile(fuq[:, :, jnp.newaxis], (1, 1, kx))
    fuq_new = jax.lax.fori_loop(1, kx - 2, entrain_loop_fuq, fuq_broadcast)

    #Downward fluxes
    sb = se[:, :, :kx - 2] + wvi[jnp.newaxis, jnp.newaxis, :kx-2, 1] * (se[:,:, 1:kx - 1] - se[:,:, :kx - 2])
    qb = qa[:, :, :kx - 2] + wvi[jnp.newaxis, jnp.newaxis, :kx-2, 1] * (qa[:,:, 1:kx - 1] - qa[:,:, :kx - 2])

    fds_new = jnp.tile(fds[:, :, jnp.newaxis], (1, 1, kx))
    fdq_new = jnp.tile(fdq[:, :, jnp.newaxis], (1, 1, kx))

    fds_new = fds_new.at[:, :, 1:kx-1].set(fmass_new[:, :, 1:kx-1]*sb)
    fdq_new = fdq_new.at[:, :, 1:kx-1].set(fmass_new[:, :, 1:kx-1]*qb)

    #Now mask out all values above iptop:
    # Expand the iptop array to match the shape of fus_new
    expanded_iptop = jnp.expand_dims(iptop, axis=-1)
    
    # Generate indices for the third dimension
    indices = jnp.arange(kx)
    
    # Create a boolean mask where we want to set values to 0
    new_mask = indices > expanded_iptop
    
    # Apply the mask to the fluxes
    fus_new_masked = fus_new * new_mask
    fuq_new_masked = fuq_new * new_mask
    fds_new_masked = fds_new * new_mask
    fdq_new_masked = fdq_new * new_mask
    
    dfse = dfse.at[:, :, :kx-1].set(fus_new_masked[:, :, 1:] - fus_new_masked[:, :, :-1] + fds_new_masked[:, :, :-1] - fds_new_masked[:, :, 1:])
    dfqa = dfqa.at[:, :, :kx-1].set(fuq_new_masked[:, :, 1:] - fuq_new_masked[:, :, :-1] + fdq_new_masked[:, :, :-1] - fdq_new_masked[:, :, 1:])

    # Secondary moisture flux -- copied this directly from old code. Hard to test
    delq_vals = rhil * qsat[:, :, 1:kx-1] - qa[:, :, 1:kx-1]
    fsq_vals = jnp.where(delq_vals > 0, smf * cbmf[:, :, jnp.newaxis] * delq_vals, 0.0)
    dfqa = dfqa.at[:, :, 1:kx-1].add(fsq_vals)
    dfqa = dfqa.at[:, :, kx].add(-jnp.sum(fsq_vals, axis=-1))

    # 3.3 Top layer (condensation and detrainment)
    k = iptop

    # Flux of convective precipitation
    i = jnp.arange(96)[:, jnp.newaxis]  # Shape (96, 1)
    j = jnp.arange(48)[jnp.newaxis, :]  # Shape (1, 48)

    qsatb = qsat[j, j, k] + wvi[k, 1] *(qsat[i, j, k+1]-qsat[i,j, k])
    precnv = jnp.where(mask, jnp.maximum(fuq_new[i, j, k] - fmass_new[i, j, k] * qsatb, 0.0), precnv)

    # Net flux of dry static energy and moisture
    dfse = dfse.at[i,j,k].set(fus_new[i,j, k +1] - fds_new[i,j, k+1] + alhc * precnv)
    dfqa = dfqa.at[i,j,k].set(fuq_new[i,j,k+1] - fdq_new[i,j,k+1] - precnv)

    # make a new physics_data struct. overwrite the appropriate convection bits that were calculated in this function
    # pass on the rest of physics_data that was not updated or needed in this function
    # since convection doesn't generate new tendencies, just return PhysicsTendency instance that is all 0's

    convection_out = physics_data.convection.copy(psa=psa, se=se, iptop=iptop, cbmf=cbmf, precnv=precnv, dfse=dfse, dfqa=dfqa)
    physics_data = physics_data.copy(convection=convection_out)
    physics_tendencies = PhysicsTendency(jnp.zeros_like(state.u_wind),jnp.zeros_like(state.v_wind),jnp.zeros_like(state.temperature),jnp.zeros_like(state.temperature))
    
    return physics_tendencies, physics_data