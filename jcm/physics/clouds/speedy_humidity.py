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
from jcm.physics_interface import PhysicsState, PhysicsTendency

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
    rh = state.specific_humidity / qsat
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

    Deliberately NOT delegated to a shared saturation module (#615's
    "decide and document" item for this file). ``speedy.f90``'s own
    ``get_qsat``/``shtorh`` hardcodes this exact Tetens form in these exact
    units (``ps`` as p/1000 hPa, ``qsat`` in g/kg, ``e0 = 6.108e-3`` already
    folded into those units) — the reference formulation SPEEDY's Fortran
    fidelity this port targets, not a quantity jcm is free to re-derive from
    :mod:`jcm.physics.thermodynamics` (ECHAM's Sonntag tables) or
    :mod:`jcm.physics.convection.saturation` (the shared Tetens module in
    SI units/kg-kg, used by Betts-Miller and JAM). Converting this call site
    would change speedy_vdiff/speedy_shortwave/speedy_surface_flux output
    (every reader of ``humidity.rh``) against the Fortran reference by
    whatever the units/constant reconciliation introduces — out of scope for
    a numerics-hygiene consolidation. See ``convection/saturation.py`` for
    the shared module every OTHER non-ECHAM Tetens caller uses instead of a
    private copy.

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
