import jax.numpy as jnp
from jax import jit
from jcm.physics.speedy.speedy_coords import stratosphere_mask
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
import jcm.constants as c
# alhc is SPEEDY's latent heat in J/g (q is in g/kg) — a SPEEDY-specific value.
# cpd is shared and read as a module attribute from jcm.constants.
from jcm.physics.speedy.physical_constants import alhc
from jcm.physics.speedy.smoothing import (
    smooth_gate, smooth_pos, surrogate_gate, surrogate_pair, surrogate_pos,
)
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.speedy.physics_data import PhysicsData


def _pbl_moisture_flux(dmse, drh, a_shc, a_vdi, drh0):
    """SPEEDY's moisture flux across the lowest interface (``vdifsc``, step 2).

    Shallow convection where ``dmse > 0``, ``a_shc * drh`` if ``drh >= 0``;
    otherwise moisture diffusion, ``a_vdi * drh`` if ``drh > drh0``
    (speedy.f90 ``vertical_diffusion.f90``, whose shallow-convection test is
    ``dmse >= 0``: the two differ only at an exact tie). ``a_shc`` is
    ``fcnv * fshcq * qsat[kx]``, ``a_vdi`` is ``fvdiq2 * qsat[nl1]``.
    """
    shallow = a_shc * jnp.maximum(drh, 0.0)
    diffusion = jnp.where(drh > drh0, a_vdi * drh, 0.0)
    return jnp.where(dmse > 0.0, shallow, diffusion)


def _smooth_pbl_moisture_flux(dmse, drh, a_shc, a_vdi, drh0, widths):
    """Smooth stand-in whose derivatives ``_pbl_moisture_flux`` takes.

    The two branches crossfaded by a sigmoid in ``dmse`` of half-width
    ``widths[0]`` (``mse_gate_smoothing``, J/kg); the shallow-convection
    branch's ``max(drh, 0)`` as a softplus and the diffusion branch's
    ``drh > drh0`` onset as a sigmoid, both of half-width ``widths[1]``
    (``rh_gate_smoothing``, an RH fraction). At zero widths it is
    ``_pbl_moisture_flux`` itself, so its derivative is the reference's
    (at an exact ``drh == 0`` tie, half the shallow branch's slope: JAX's
    rule for ``maximum``, measure zero).
    """
    w_mse, w_rh = widths
    shallow_weight = smooth_gate(dmse, 0.0, w_mse)
    shallow = a_shc * smooth_pos(drh, w_rh)
    diffusion = smooth_gate(drh, drh0, w_rh) * a_vdi * drh
    return shallow_weight * shallow + (1.0 - shallow_weight) * diffusion


# The dmse >= 0 regime split is a value jump of the flux by the difference
# between the two branches' fluxes, so the reference derivative is blind to
# it. One smooth function for the whole block gives it the derivative
# sigmoid'(dmse/w)/w times exactly that difference, the jump the value
# makes. Where neither branch carries a flux (drh < 0, so both shallow
# convection's drh >= 0 and the diffusion onset fail), crossing dmse = 0
# changes nothing, and the dmse-derivative is zero (up to the
# rh_gate_smoothing tails, which have the damping sign: more flux above the
# switch, drying q[kx], which dmse grows with).
#
# Rejected: a sigmoid surrogate per decision, the branches' pairs chained
# with "the derivative of the first decision that fails" (chain_gates).
# Below the switch with drh < 0 that gave the tangent sigmoid'/w * a_shc *
# drh, the derivative of a REVERSED shallow-convection flux the scheme never
# produces, and a positive feedback: dmse grows with q[kx], and the reversed
# flux would moisten q[kx]. One T30 L8 column that lingered within a width
# or two of dmse = 0 with drh < 0 for four days multiplied the model's
# humidity tangent 2-10x every 12 hours, the forecast bit-identical (#1167). The
# shallow branch's drh test had the same reversal at drh < 0: max(drh, 0)
# is a kink, not a jump, and takes a softplus, as the dry static energy
# flux's dmse does, not a gate times drh.
_surrogate_pbl_moisture_flux = surrogate_pair(
    _pbl_moisture_flux, _smooth_pbl_moisture_flux)


@jit
def get_vertical_diffusion_tend(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Get vertical diffusion tendencies.
    
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
    icnv = kx - physics_data.convection.iptop # this comes from physics.f90:132

    ttenvd = jnp.zeros((kx,ix,il))
    qtenvd = jnp.zeros((kx,ix,il))

    nl1 = kx - 1
    cshc = physics_data.speedy_coords.dhs[kx - 1] / 3600.0
    cvdi = (physics_data.speedy_coords.hsg[nl1] - physics_data.speedy_coords.hsg[1]) / ((nl1 - 1) * 3600.0)
    
    fshcq = cshc / parameters.vertical_diffusion.trshc
    fshcse = cshc / (parameters.vertical_diffusion.trshc * c.cpd)
    
    fvdiq = cvdi / parameters.vertical_diffusion.trvdi
    fvdise = cvdi / (parameters.vertical_diffusion.trvds * c.cpd)

    rsig = 1.0 / physics_data.speedy_coords.dhs
    rsig1 = jnp.zeros((kx,)).at[:-1].set(1.0 / (1.0 - physics_data.speedy_coords.hsg[1:-1]))
    rsig1 = rsig1.at[-1].set(0.0)
    
    # Step 2: Shallow convection
    drh0 = parameters.vertical_diffusion.rhgrad * (physics_data.speedy_coords.fsg[kx - 1] - physics_data.speedy_coords.fsg[nl1 - 1])
    fvdiq2 = fvdiq * physics_data.speedy_coords.hsg[nl1]

    # Calculate dmse and drh arrays
    dmse = se[kx - 1] - se[nl1 - 1] + alhc * (qa[kx - 1] - qsat[nl1 -1])
    drh = rh[kx - 1] - rh[nl1 -1]

    w_mse = parameters.vertical_diffusion.mse_gate_smoothing
    w_rh = parameters.vertical_diffusion.rh_gate_smoothing

    # Shallow convection is damped by redshc where deep convection is
    # active; the deep-convection index is discrete and stays hard.
    fcnv = jnp.where(icnv > 0, parameters.vertical_diffusion.redshc, 1.0)

    # The dry static energy flux is proportional to dmse and has no branch
    # below the threshold, so its dmse >= 0 test is a kink, not a jump: a
    # one-sided hinge (with a softplus surrogate derivative), never a gate
    # times dmse, which would turn negative (a reversed heat flux) in stable
    # columns (Codex review, PR #567).
    fluxse = fcnv * fshcse * surrogate_pos(dmse, w_mse)
    ttenvd = ttenvd.at[nl1 - 1].set(fluxse * rsig[nl1 - 1])
    ttenvd = ttenvd.at[kx - 1].set(-fluxse * rsig[kx - 1])

    # The moisture flux across the lowest interface: one surrogate for the
    # whole block (_pbl_moisture_flux).
    fluxq_pbl = _surrogate_pbl_moisture_flux(
        dmse, drh, fcnv * fshcq * qsat[kx - 1], fvdiq2 * qsat[nl1 - 1], drh0,
        (w_mse, w_rh))
    qtenvd = qtenvd.at[nl1 - 1].set(fluxq_pbl * rsig[nl1 - 1])
    qtenvd = qtenvd.at[kx - 1].set(-fluxq_pbl * rsig[kx - 1])

    # Step 3: Vertical diffusion of moisture above the PBL.
    #
    # Diffusion acts across each interface between layer k and k+1. SPEEDY
    # restricted this to k=2..kx-3 (skipping the top-two stratosphere at the top
    # and the two shallow-convection/PBL layers handled in Step 2 at the bottom).
    # We replace the upper (stratosphere) bound with the sigma<0.2 mask so it
    # scales with nlev, keep the lower bound (interfaces above the bottom two
    # layers), and keep the original sigma>0.5 gate. The stratosphere mask is
    # static (fsg-derived) so the per-interface gate is a compile-time constant.
    k_range = jnp.arange(1, kx - 2)
    strat_mask = stratosphere_mask(physics_data.speedy_coords.fsg)
    # Skip an interface whose upper layer k is in the stratosphere.
    not_strat = ~strat_mask[k_range]
    condition = (physics_data.speedy_coords.hsg[k_range + 1] > 0.5) & not_strat

    # Vectorized calculation of drh0 and fvdiq2 for all selected k values
    drh0 = parameters.vertical_diffusion.rhgrad * (physics_data.speedy_coords.fsg[k_range + 1] - physics_data.speedy_coords.fsg[k_range])  # Shape: (len(k_range),)
    fvdiq2 = fvdiq * physics_data.speedy_coords.hsg[k_range + 1]  # Shape: (len(k_range),)

    # Calculate drh for all selected k values across the entire ix and il dimensions
    drh = rh[k_range + 1] - rh[k_range]  # Shape: (ix, il, len(k_range))

    # Moisture-diffusion onset: the flux is proportional to drh, so the
    # hard drh >= drh0 gate is a value jump; it keeps its hard value with
    # the same rh_gate_smoothing sigmoid as its derivative. The
    # per-interface sigma condition is static and stays hard.
    g_rh3 = surrogate_gate(drh, drh0[:, jnp.newaxis, jnp.newaxis], w_rh)
    fluxq = jnp.where(
        condition[:, jnp.newaxis, jnp.newaxis],
        g_rh3 * fvdiq2[:, jnp.newaxis, jnp.newaxis] * qsat[k_range] * drh,
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