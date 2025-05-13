import jax.numpy as jnp
from jax import jit
from jcm.geometry import Geometry
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.physical_constants import cp, alhc
from jcm.physics import PhysicsState, PhysicsTendency
from jcm.physics_data import PhysicsData

@jit
def get_vertical_diffusion_tend(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """
    Inputs:
        se(ix,il,kx)     !! Dry static energy
        rh(ix,il,kx)     !! Relative humidity
        qa(ix,il,kx)     !! Specific humidity [g/kg]
        qsat(ix,il,kx)   !! Saturated specific humidity [g/kg]
        phi(ix,il,kx)    !! Geopotential
        icnv(ix,il)      !! Sigma-level index of deep convection
    
    Returns:
        ttenvd(ix,il,kx) !! Temperature tendency
        qtenvd(ix,il,kx) !! Specific humidity tendency
    """

    se = physics_data.convection.se
    rh = physics_data.humidity.rh
    qsat = physics_data.humidity.qsat
    qa = state.specific_humidity
    phi = state.geopotential

    kx, ix, il = state.temperature.shape
    icnv = kx - physics_data.convection.iptop - 1 # this comes from physics.f90:132

    ttenvd = jnp.zeros((kx,ix,il))
    qtenvd = jnp.zeros((kx,ix,il))

    nl1 = kx - 1
    cshc = geometry.dhs[kx - 1] / 3600.0
    cvdi = (geometry.hsg[nl1] - geometry.hsg[1]) / ((nl1 - 1) * 3600.0)
    
    fshcq = cshc / parameters.vertical_diffusion.trshc
    fshcse = cshc / (parameters.vertical_diffusion.trshc * cp)
    
    fvdiq = cvdi / parameters.vertical_diffusion.trvdi
    fvdise = cvdi / (parameters.vertical_diffusion.trvds * cp)

    rsig = 1.0 / geometry.dhs
    rsig1 = jnp.zeros((kx,)).at[:-1].set(1.0 / (1.0 - geometry.hsg[1:-1]))
    rsig1 = rsig1.at[-1].set(0.0)
    
    # Step 2: Shallow convection
    drh0 = parameters.vertical_diffusion.rhgrad * (geometry.fsg[kx - 1] - geometry.fsg[nl1 - 1])
    fvdiq2 = fvdiq * geometry.hsg[nl1]

    # Calculate dmse and drh arrays
    dmse = se[kx - 1] - se[nl1 - 1] + alhc * (qa[kx - 1] - qsat[nl1 -1])
    drh = rh[kx - 1] - rh[nl1 -1]

    # Initialize fcnv array
    fcnv = jnp.ones((ix, il))

    # Apply condition where icnv > 0 and set fcnv to redshc
    fcnv = jnp.where(jnp.logical_and(icnv > 0, dmse >= 0), parameters.vertical_diffusion.redshc, fcnv)

    # Calculate fluxse where dmse >= 0.0
    fluxse = jnp.where(dmse >= 0.0, fcnv * fshcse * dmse, 0)

    # Update ttenvd based on fluxse
    ttenvd = ttenvd.at[nl1 - 1].set(jnp.where(dmse >= 0.0, fluxse * rsig[nl1 - 1], ttenvd[nl1 - 1]))
    ttenvd = ttenvd.at[kx - 1].set(jnp.where(dmse >= 0.0, -fluxse * rsig[kx - 1], ttenvd[kx - 1]))

    # Calculate fluxq for the first condition (dmse >= 0.0 and drh >= 0.0)
    fluxq_condition1 = jnp.where((dmse >= 0.0) & (drh >= 0.0), fcnv * fshcq * qsat[kx - 1] * drh, 0)

    # Update qtenvd based on fluxq_condition1
    qtenvd = qtenvd.at[nl1 - 1].set(jnp.where((dmse >= 0.0) & (drh >= 0.0), fluxq_condition1 * rsig[nl1 - 1], qtenvd[nl1 - 1]))
    qtenvd = qtenvd.at[kx - 1].set(jnp.where((dmse >= 0.0) & (drh >= 0.0), -fluxq_condition1 * rsig[kx - 1], qtenvd[kx - 1])
            )

    # Calculate fluxq for the second condition (dmse < 0.0 and drh > drh0)
    fluxq_condition2 = jnp.where((dmse < 0.0) & (drh > drh0), fvdiq2 * qsat[nl1 - 1] * drh, 0)

    # Update qtenvd based on fluxq_condition2
    qtenvd = qtenvd.at[nl1 - 1].set(
        jnp.where((dmse < 0.0) & (drh > drh0), fluxq_condition2 * rsig[nl1 - 1], qtenvd[nl1 - 1])
    )
    qtenvd = qtenvd.at[kx - 1].set(
        jnp.where((dmse < 0.0) & (drh > drh0), -fluxq_condition2 * rsig[kx - 1], qtenvd[kx - 1])
    )
    
    # Step 3: Vertical diffusion of moisture above the PBL
    k_range = jnp.arange(2, kx - 2)
    condition = geometry.hsg[k_range + 1] > 0.5

    # Vectorized calculation of drh0 and fvdiq2 for all selected k values
    drh0 = parameters.vertical_diffusion.rhgrad * (geometry.fsg[k_range + 1] - geometry.fsg[k_range])  # Shape: (len(k_range),)
    fvdiq2 = fvdiq * geometry.hsg[k_range + 1]  # Shape: (len(k_range),)

    # Calculate drh for all selected k values across the entire ix and il dimensions
    drh = rh[k_range + 1] - rh[k_range]  # Shape: (ix, il, len(k_range))

    # Calculate fluxq where drh >= drh0
    fluxq = jnp.where(
        (drh >= drh0[:, jnp.newaxis, jnp.newaxis]) & condition[:, jnp.newaxis, jnp.newaxis],
        fvdiq2[:, jnp.newaxis, jnp.newaxis] * qsat[k_range] * drh,
        0
    )

    # Update qtenvd for all selected k values
    qtenvd = qtenvd.at[k_range].add(fluxq * rsig[k_range][:, jnp.newaxis, jnp.newaxis])
    qtenvd = qtenvd.at[k_range + 1].add(-fluxq * rsig[k_range + 1][:, jnp.newaxis, jnp.newaxis])

    # Step 4: Damping of super-adiabatic lapse rate
    se0 = se[1:] - parameters.vertical_diffusion.segrad * jnp.diff(phi, axis=0)

    condition = se[:nl1] < se0
    
    fluxse = jnp.where(condition, fvdise * (se0 - se[:nl1]), 0)
    
    ttenvd = ttenvd.at[:nl1].add(fluxse * rsig[:nl1, jnp.newaxis, jnp.newaxis])
    
    cumulative_fluxse = jnp.cumsum(fluxse * rsig1[:nl1, jnp.newaxis, jnp.newaxis], axis=0)
    
    ttenvd = ttenvd.at[1:].add(-cumulative_fluxse)
    
    physics_tendencies = PhysicsTendency.zeros(shape=ttenvd.shape, temperature=ttenvd, specific_humidity=qtenvd)

    # have not updated physics_data, can just return the instance we were passed
    return physics_tendencies, physics_data