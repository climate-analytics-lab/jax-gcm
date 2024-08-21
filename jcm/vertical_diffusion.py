import jax.numpy as jnp

# Constants for vertical diffusion and shallow convection
# trshc = jnp.array(6.0)  # Relaxation time (in hours) for shallow convection
# trvdi = jnp.array(24.0)  # Relaxation time (in hours) for moisture diffusion
# trvds = jnp.array(6.0)  # Relaxation time (in hours) for super-adiabatic conditions
# redshc = jnp.array(0.5)  # Reduction factor of shallow convection in areas of deep convection
# rhgrad = jnp.array(0.5)  # Maximum gradient of relative humidity (d_RH/d_sigma)
# segrad = jnp.array(0.1)  # Minimum gradient of dry static energy (d_DSE/d_phi)

# Constants for vertical diffusion and shallow convection

import dataclasses

@dataclasses.dataclass
class DiffusionConstants:
    shallow_convection_relax_time: float = 6
    moisture_diffusion_relax_time: float = 24
    super_adiabatic_relax_time: float = 6
    shallow_reduction_factor: float = 0.5
    relative_humidity_max_gradient: float = 0.5
    dry_static_energy_min_gradient: float = 0.1 

# Placeholder values for constants (you should define these based on your context)
# sigh = jnp.linspace(0, 1, 100)  # Example sigma levels (for simplicity)
# dhs = jnp.ones(100)  # Example placeholder, needs real values

def get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv,
                                diffusion_constants, fsg,
                                dhs, sigh
                                ):
    trshc = diffusion_constants.shallow_convection_relax_time
    trvdi = diffusion_constants.moisture_diffusion_relax_time
    trvds = diffusion_constants.super_adiabatic_relax_time
    redshc = diffusion_constants.shallow_reduction_factor
    rhgrad = diffusion_constants.relative_humidity_max_gradient
    segrad = diffusion_constants.dry_static_energy_min_gradient

    #### We define cp and alhc within functions
    cp = 1004.0  # Specific heat capacity at constant pressure
    alhc = 2501.0  # Latent heat of condensation (kJ/kg)

    ix, il, kx = se.shape
    
    # Initialize output arrays for tendencies
    utenvd = jnp.zeros_like(se)
    vtenvd = jnp.zeros_like(se)
    ttenvd = jnp.zeros_like(se)
    qtenvd = jnp.zeros_like(se)
    
    nl1 = kx - 1
    cshc = dhs[kx - 1] / 3600.0
    cvdi = (sigh[nl1-1] - sigh[0]) / ((nl1 - 1) * 3600.0)
    
    fshcq = cshc / trshc
    fshcse = cshc / (trshc * cp)
    
    fvdiq = cvdi / trvdi
    fvdise = cvdi / (trvds * cp)

    sigh = sigh[1:]

    rsig = 1.0 / dhs
    rsig1 = 1.0 / (1.0 - sigh)
    rsig1 = rsig1.at[-1].set(0.0)
    
    # Step 2: Shallow convection
    drh0 = rhgrad * (fsg[kx - 1] - fsg[nl1 - 1])  # 
    fvdiq2 = fvdiq * sigh[nl1 -1]

    # Calculate dmse and drh arrays
    dmse = se[:, :, kx - 1] - se[:, :, nl1 - 1] + alhc * (qa[:, :, kx - 1] - qsat[:, :, nl1 -1])
    drh = rh[:, :, kx - 1] - rh[:, :, nl1 -1]

    # Initialize fcnv array
    fcnv = jnp.ones((ix, il))

    # Apply condition where icnv > 0 and set fcnv to redshc
    fcnv = jnp.where(jnp.logical_and(icnv > 0, dmse >= 0), redshc, fcnv)

    # Calculate fluxse where dmse >= 0.0
    fluxse = jnp.where(dmse >= 0.0, fcnv * fshcse * dmse, 0)

    # Update ttenvd based on fluxse
    ttenvd = ttenvd.at[:, :, nl1 - 1].set(jnp.where(dmse >= 0.0, fluxse * rsig[nl1 - 1], ttenvd[:, :, nl1 - 1]))
    ttenvd = ttenvd.at[:, :, kx - 1].set(jnp.where(dmse >= 0.0, -fluxse * rsig[kx - 1], ttenvd[:, :, kx - 1]))

    # Calculate fluxq for the first condition (dmse >= 0.0 and drh >= 0.0)
    fluxq_condition1 = jnp.where((dmse >= 0.0) & (drh >= 0.0), fcnv * fshcq * qsat[:, :, kx - 1] * drh, 0)

    # Update qtenvd based on fluxq_condition1
    qtenvd = qtenvd.at[:, :, nl1 - 1].set(jnp.where((dmse >= 0.0) & (drh >= 0.0), fluxq_condition1 * rsig[nl1 - 1], qtenvd[:, :, nl1 - 1]))
    qtenvd = qtenvd.at[:, :, kx - 1].set(jnp.where((dmse >= 0.0) & (drh >= 0.0), -fluxq_condition1 * rsig[kx - 1], qtenvd[:, :, kx - 1])
            )

    # Calculate fluxq for the second condition (dmse < 0.0 and drh > drh0)
    fluxq_condition2 = jnp.where((dmse < 0.0) & (drh > drh0), fvdiq2 * qsat[:, :, nl1 - 1] * drh, 0)

    # Update qtenvd based on fluxq_condition2
    qtenvd = qtenvd.at[:, :, nl1 - 1].set(
                        jnp.where((dmse < 0.0) & (drh > drh0), fluxq_condition2 * rsig[nl1 - 1], qtenvd[:, :, nl1 - 1])
            )
    qtenvd = qtenvd.at[:, :, kx - 1].set(
                        jnp.where((dmse < 0.0) & (drh > drh0), -fluxq_condition2 * rsig[kx - 1], qtenvd[:, :, kx - 1])
            )

    # Step 3: Vertical diffusion of moisture above the PBL
    # Define the k range
    k_range = jnp.arange(2, kx - 2)

    # Create a boolean mask for sigh[k] > 0.5 across the k_range
    k_mask = sigh[k_range] > 0.5

    # Filter k_range to include only those indices where sigh[k] > 0.5
    k_selected = k_range[k_mask]

    # Vectorized calculation of drh0 and fvdiq2 for all selected k values
    drh0 = rhgrad * (fsg[k_selected + 1] - fsg[k_selected])  # Shape: (len(k_selected),)
    fvdiq2 = fvdiq * sigh[k_selected]  # Shape: (len(k_selected),)

    # Calculate drh for all selected k values across the entire ix and il dimensions
    drh = rh[:, :, k_selected + 1] - rh[:, :, k_selected]  # Shape: (ix, il, len(k_selected))

    # Calculate fluxq where drh >= drh0
    # The broadcasting of drh0 over the ix and il dimensions allows for the vectorized comparison
    fluxq = jnp.where(drh >= drh0[jnp.newaxis, jnp.newaxis, :], fvdiq2 * qsat[:, :, k_selected] * drh, 0)

    # Update qtenvd for all selected k values
    qtenvd = qtenvd.at[:, :, k_selected].add(
        fluxq * rsig[k_selected][jnp.newaxis, jnp.newaxis, :]
            )
    qtenvd = qtenvd.at[:, :, k_selected + 1].add(
        -fluxq * rsig[k_selected + 1][jnp.newaxis, jnp.newaxis, :]
            )
    
    # Step 4: Damping of super-adiabatic lapse rate
    se0 = se[:, :, 1:nl1+1] + segrad * (phi[:, :, :nl1] - phi[:, :, 1:nl1+1])

    condition = se[:, :, :nl1] < se0

    fluxse = jnp.where(condition, fvdise * (se0 - se[:, :, :nl1]), 0)

    ttenvd = ttenvd.at[:, :, :nl1].add(fluxse * rsig[:nl1])

    cumulative_fluxse = jnp.cumsum(fluxse * rsig1[:nl1], axis=2)

    ttenvd = ttenvd.at[:, :, 1:nl1+1].add(-cumulative_fluxse)

    return utenvd, vtenvd, ttenvd, qtenvd