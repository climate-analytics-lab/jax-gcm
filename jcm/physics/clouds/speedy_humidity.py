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

# Numerical-sanity ceiling for the *public* relative-humidity diagnostic
# (``capped_relative_humidity`` below) — not a physical bound, and not
# applied to ``physics_data.humidity.rh`` itself (see that function's
# docstring for why). 10.0 (1000 %) is generously above any physically
# meaningful supersaturation (ice nucleation / cirrus tops out around
# 1.5-2), so it only ever clips the numerical pathology this exists for.
_RH_SANITY_MAX = 10.0


def capped_relative_humidity(rh: jnp.ndarray) -> jnp.ndarray:
    """Sanity-capped relative humidity for the public diagnostic (#407).

    ``get_qsat`` follows the reference Fortran's ice-Tetens branch
    verbatim and, like the reference, floors neither the saturation
    specific humidity nor ``rh = q / qsat``: as T keeps falling the
    exponential drives ``qsat`` down by many more orders of magnitude
    than any physically-resolvable specific humidity falls, so a
    near-zero but non-zero ``q`` divided by a near-zero ``qsat`` is
    unbounded. SPEEDY's top sigma layers are exactly where this bites —
    deep polar night can put the model lid's temperature near 125-150 K
    (a known, accepted characteristic of SPEEDY's crude top-level
    radiative balance, not this function's concern) — and the raw ratio
    then reaches the hundreds of thousands of percent (measured 1.64e6 on
    the ``speedy-t31`` release-validation year).

    This is applied only to the separate ``"relative_humidity"`` output
    diagnostic (:class:`~jcm.physics.speedy.speedy_terms.SpeedyHumidity`),
    never to ``physics_data.humidity.rh`` itself: that field is also a
    *physics input* — ``speedy_vdiff.py``, ``speedy_shortwave.py``'s cloud
    diagnosis (including its fixed-sigma ``interp_to_sigma`` blends, which
    are not confined to any one level on every SPEEDY vertical grid) and
    ``speedy_surface_flux.py`` all read it — so capping it in place would
    change simulated clouds and fluxes on a config where some bracketing
    level's raw ratio exceeds the cap, not just the reported number. Two
    Codex review rounds on #407 found exactly this when the cap was first
    applied in place (restricting it to specific levels was still grid-
    dependent and unsafe). Keeping the reference quantity untouched and
    adding a distinct, always-sane public field mirrors the
    ``relative_humidity`` / ``cover_relative_humidity`` split already used
    for ECHAM's Sundqvist closure (:mod:`jcm.physics.diagnostics.moist_air_state`).
    """
    return jnp.minimum(rh, _RH_SANITY_MAX)


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
