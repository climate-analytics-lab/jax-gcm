"""For converting between specific and relative humidity, and computing the
saturation specific humidity.
"""

import jax
from jax import jit
import jax.numpy as jnp
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.speedy_coords import stratosphere_mask
from jcm.physics_interface import PhysicsState, PhysicsTendency

# Numerical-sanity cap on the *reported* ``rh`` diagnostic — not a physical
# bound. ``get_qsat`` follows the reference Fortran's ice-Tetens branch
# (``ta < t0``) verbatim (speedy.f90's humidity.f90::get_qsat, Hatfield,
# Kucharski & Molteni), which has no floor either: as T keeps falling, the
# exponential drives qsat down by many more orders of magnitude than any
# physically-resolvable specific humidity falls, so a near-zero but still
# non-zero q divided by a near-zero qsat is unbounded. SPEEDY's top two
# sigma layers (0.025 and 0.095, ``stratosphere_mask``) are exactly where
# this bites — deep polar night can put the lid's temperature near
# 125-150 K (a known, accepted characteristic of SPEEDY's crude top-level
# radiative balance, not this function's concern) — and the formula then
# reports RH in the hundreds of thousands of percent (#407).
#
# The cap is deliberately restricted to those two layers rather than
# applied everywhere ``rh`` exceeds it. Every other SPEEDY term that reads
# ``physics_data.humidity.rh`` already treats the stratosphere as off
# limits for this exact reason: ``speedy_condensation.py`` only ever sets
# ``dqlsc``/``dtlsc`` from level 1 down, ``speedy_vdiff.py``'s moisture
# step both masks ``stratosphere_mask`` and additionally gates on
# ``hsg > 0.5``, and ``speedy_shortwave.py::get_clouds`` excludes it from
# its own cloud-top ``rh`` search with the same mask. So capping exactly
# the stratosphere leaves every physics consumer's input bit-identical
# (verified: a full T31L8 year differs from an uncapped run in no field
# but ``humidity.rh`` itself, confined to these two levels and a handful of
# interim ones nothing reads). A milder echo of the same mechanism persists
# at sigma 0.2 and 0.34 (measured max ~30x and ~14x, against the top
# layer's 1.6e6x) — capping those too would touch ``get_clouds``'s cloud-top
# diagnosis, a physics change this fix does not make.
#
# 10.0 (1000 %) is the cap: real supersaturation (ice nucleation, cirrus)
# tops out around 1.5-2, so this is generously above anything physically
# meaningful while eliminating values that are many orders of magnitude
# beyond that.
_RH_SANITY_MAX = 10.0

@jit
def spec_hum_to_rel_hum(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Convert specific humidity to relative humidity, and also return saturation
    specific humidity.

    Args:
        ta: Absolute temperature [K] - PhysicsState.temperature
        ps: Normalized pressure (p/1000 hPa) - state.normalized_surface_pressure
        sig: Sigma level - fsg from geometry
        qa: Specific humidity [g/kg] from the SPEEDY compatibility adapter.

    Returns:
        rh: Relative humidity
        qsat: Saturation specific humidity

    """
    # compute thermodynamic variables: logic from physics.f90:110-114
    psa = state.normalized_surface_pressure
    
    # spec_hum_to_rel_hum logic
    map_qsat = jax.vmap(get_qsat, in_axes=(0, jnp.newaxis, 0), out_axes=0) # map over each input's z-axis and output to z-axis
    qsat = map_qsat(state.temperature, psa, physics_data.speedy_coords.fsg)
    # ``qsat`` itself is left exactly as the reference computes it (every
    # consumer, including the ones below, still reads the faithful value);
    # only the reported ratio is capped, and only in the two stratospheric
    # layers nothing else reads it in — see ``_RH_SANITY_MAX`` above for why.
    raw_rh = state.specific_humidity / qsat
    strat = stratosphere_mask(physics_data.speedy_coords.fsg)
    strat = strat.reshape((-1,) + (1,) * (raw_rh.ndim - 1))
    rh = jnp.where(strat, jnp.minimum(raw_rh, _RH_SANITY_MAX), raw_rh)
    humidity_out = physics_data.humidity.copy(rh=rh, qsat=qsat)

    physics_data = physics_data.copy(humidity=humidity_out)
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)
    
    return physics_tendencies, physics_data

@jit
def rel_hum_to_spec_hum(ta, ps, sig, rh):
    """Convert relative humidity to specific humidity, and also return saturation
    specific humidity.

    Args:
        ta: Absolute temperature
        ps: Normalized pressure (p/1000 hPa)
        sig: Sigma level
        rh: Relative humidity

    Returns:
        qa: Specific humidity
        qsat: Saturation specific humidity

    """
    qsat = get_qsat(ta, ps, sig)
    qa = rh * qsat
    return qa, qsat

@jit
def get_qsat(ta, ps, sig):
    """Compute saturation specific humidity.
    
    Args:
        ta: Absolute temperature [K]
        ps: Normalized pressure (p/1000 hPa)
        sig: Sigma level
    
    Returns:
        qsat: Saturation specific humidity (g/kg)

    """
    e0 = 6.108e-3
    c1 = 17.269
    c2 = 21.875
    t0 = 273.16
    t1 = 35.86
    t2 = 7.66

    # Computing qsat for each grid point
    # 1. Compute Qsat (g/kg) from T (degK) and normalized pres. P (= p/1000_hPa)
    
    qsat = jnp.where(ta >= t0, e0 * jnp.exp(c1 * (ta - t0) / (ta - t1)),
                      e0 * jnp.exp(c2 * (ta - t0) / (ta - t2)))
    
    # If sig > 0, P = Ps * sigma, otherwise P = Ps(1) = const
    qsat = jnp.where(sig <= 0.0, 622.0 * qsat / (ps[0,0] - 0.378 * qsat),
                      622.0 * qsat / (sig * ps - 0.378 * qsat))

    return qsat
