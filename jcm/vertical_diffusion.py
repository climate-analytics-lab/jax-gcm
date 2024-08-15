import jax.numpy as jnp

# Constants for vertical diffusion and shallow convection
trshc = jnp.array(6.0)  # Relaxation time (in hours) for shallow convection
trvdi = jnp.array(24.0)  # Relaxation time (in hours) for moisture diffusion
trvds = jnp.array(6.0)  # Relaxation time (in hours) for super-adiabatic conditions
redshc = jnp.array(0.5)  # Reduction factor of shallow convection in areas of deep convection
rhgrad = jnp.array(0.5)  # Maximum gradient of relative humidity (d_RH/d_sigma)
segrad = jnp.array(0.1)  # Minimum gradient of dry static energy (d_DSE/d_phi)

# Placeholder values for constants (you should define these based on your context)
# cp = 1004.0  # Specific heat capacity at constant pressure
# alhc = 2.5e6  # Latent heat of condensation (J/kg)
# sigh = jnp.linspace(0, 1, 100)  # Example sigma levels (for simplicity)
# dhs = jnp.ones(100)  # Example placeholder, needs real values

def get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv):
    ix, il, kx = se.shape
    
    # Initialize output arrays for tendencies
    utenvd = jnp.zeros_like(se)
    vtenvd = jnp.zeros_like(se)
    ttenvd = jnp.zeros_like(se)
    qtenvd = jnp.zeros_like(se)
    
    nl1 = kx - 1
    cshc = dhs[kx - 1] / 3600.0
    cvdi = (sigh[nl1] - sigh[0]) / ((nl1 - 1) * 3600.0)
    
    fshcq = cshc / trshc
    fshcse = cshc / (trshc * cp)
    
    fvdiq = cvdi / trvdi
    fvdise = cvdi / (trvds * cp)
    
    rsig = 1.0 / dhs
    rsig1 = 1.0 / (1.0 - sigh)
    
    # Step 1: Shallow convection
    drh0 = rhgrad * (1.0 - sigh[nl1])  # Using 1.0 for fsg as a placeholder
    fvdiq2 = fvdiq * sigh[nl1]

    # Calculate dmse and drh arrays
    dmse = se[:, :, kx - 1] - se[:, :, nl1] + alhc * (qa[:, :, kx - 1] - qsat[:, :, nl1])
    drh = rh[:, :, kx - 1] - rh[:, :, nl1]

    # Initialize fcnv array
    fcnv = jnp.ones((ix, il))

    # Apply condition where icnv > 0 and set fcnv to redshc
    fcnv[icnv > 0] = redshc

    # Calculate fluxse where dmse >= 0.0
    fluxse = jnp.where(dmse >= 0.0, fcnv * fshcse * dmse, 0)

    # Update ttenvd based on fluxse
    ttenvd[:, :, nl1] = jnp.where(dmse >= 0.0, fluxse * rsig[nl1], ttenvd[:, :, nl1])
    ttenvd[:, :, kx - 1] = jnp.where(dmse >= 0.0, -fluxse * rsig[kx - 1], ttenvd[:, :, kx - 1])

    # Calculate fluxq for the first condition (dmse >= 0.0 and drh >= 0.0)
    fluxq_condition1 = jnp.where((dmse >= 0.0) & (drh >= 0.0), fcnv * fshcq * qsat[:, :, kx - 1] * drh, 0)

    # Update qtenvd based on fluxq_condition1
    qtenvd[:, :, nl1] = jnp.where((dmse >= 0.0) & (drh >= 0.0), fluxq_condition1 * rsig[nl1], qtenvd[:, :, nl1])
    qtenvd[:, :, kx - 1] = jnp.where((dmse >= 0.0) & (drh >= 0.0), -fluxq_condition1 * rsig[kx - 1], qtenvd[:, :, kx - 1])

    # Calculate fluxq for the second condition (dmse < 0.0 and drh > drh0)
    fluxq_condition2 = jnp.where((dmse < 0.0) & (drh > drh0), fvdiq2 * qsat[:, :, nl1] * drh, 0)

    # Update qtenvd based on fluxq_condition2
    qtenvd[:, :, nl1] = jnp.where((dmse < 0.0) & (drh > drh0), fluxq_condition2 * rsig[nl1], qtenvd[:, :, nl1])
    qtenvd[:, :, kx - 1] = jnp.where((dmse < 0.0) & (drh > drh0), -fluxq_condition2 * rsig[kx - 1], qtenvd[:, :, kx - 1])


    # Step 2: Vertical diffusion of moisture above the PBL
    # Define the k range
    k_range = jnp.arange(2, kx - 2)

    # Create a boolean mask for sigh[k] > 0.5 across the k_range
    k_mask = sigh[k_range] > 0.5

    # Filter k_range to include only those indices where sigh[k] > 0.5
    k_selected = k_range[k_mask]

    # Vectorized calculation of drh0 and fvdiq2 for all selected k values
    drh0 = rhgrad * (1.0 - sigh[k_selected])  # Shape: (len(k_selected),)
    fvdiq2 = fvdiq * sigh[k_selected]  # Shape: (len(k_selected),)

    # Calculate drh for all selected k values across the entire ix and il dimensions
    drh = rh[:, :, k_selected + 1] - rh[:, :, k_selected]  # Shape: (ix, il, len(k_selected))

    # Calculate fluxq where drh >= drh0
    # The broadcasting of drh0 over the ix and il dimensions allows for the vectorized comparison
    fluxq = jnp.where(drh >= drh0[jnp.newaxis, jnp.newaxis, :], fvdiq2 * qsat[:, :, k_selected] * drh, 0)

    # Update qtenvd for all selected k values
    qtenvd[:, :, k_selected] += fluxq * rsig[k_selected][jnp.newaxis, jnp.newaxis, :]
    qtenvd[:, :, k_selected + 1] -= fluxq * rsig[k_selected + 1][jnp.newaxis, jnp.newaxis, :]


    # Step 3: Damping of super-adiabatic lapse rate
    
    # Calculate se0 for all k, i, and j
    se0 = se[:, :, 1:nl1+1] + segrad * (phi[:, :, :nl1] - phi[:, :, 1:nl1+1])

    # Calculate the condition where se < se0
    condition = se[:, :, :nl1] < se0

    # Calculate fluxse where the condition is True
    fluxse = jnp.where(condition, fvdise * (se0 - se[:, :, :nl1]), 0)

    # Update ttenvd for all k, i, and j where the condition is True
    ttenvd[:, :, :nl1] += fluxse * rsig[:nl1][jnp.newaxis, jnp.newaxis, :]

    # Accumulate the fluxse across the remaining k1 values
    # We need to add the fluxse to the subsequent layers in a cumulative way
    cumulative_fluxse = jnp.cumsum(fluxse[:, :, ::-1] * rsig1[:nl1][::-1][jnp.newaxis, jnp.newaxis, :], axis=2)[:, :, ::-1]

    # Subtract the cumulative fluxse from ttenvd for all k1 > k
    ttenvd[:, :, 1:nl1+1] -= cumulative_fluxse


    return utenvd, vtenvd, ttenvd, qtenvd
