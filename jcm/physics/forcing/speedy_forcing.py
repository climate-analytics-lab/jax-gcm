from jcm.terrain import TerrainData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import ablco2_ref, PhysicsData
from jcm.forcing import ForcingData, DEFAULT_CO2_VMR_PPMV
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.radiation.speedy_shortwave import get_zonal_average_fields
import jax.numpy as jnp


# Reference CO2 concentration (ppmv) at which `ablco2_ref` is defined. SPEEDY's
# original Fortran tuned `ablco2_ref` for ~1990s atmosphere (~360 ppmv); we
# anchor our linear scaling to that baseline so a `forcing.co2_vmr` of 360
# ppmv reproduces the legacy un-perturbed (`increase_co2=False`) behavior
# exactly.
CO2_VMR_REF_PPMV = DEFAULT_CO2_VMR_PPMV  # 360.0


def ablco2_from_co2_vmr(co2_vmr_ppmv: jnp.ndarray) -> jnp.ndarray:
    """CO2 absorptivity used by SPEEDY shortwave/longwave, derived from
    a CO2 mixing ratio (ppmv).

    Linear scaling against the reference: `ablco2 = ablco2_ref * C / C_ref`.
    Chosen because SPEEDY's legacy `exp(0.005 * dyears)` ramp grew the
    absorptivity at ~0.5%/yr, which closely tracks the historical CO2 growth
    rate near the 360 ppmv reference. A pure log-in-ratio mapping (CO2
    radiative-forcing physics) is also reasonable; the regression test in
    `speedy_co2_test.py` pins the linear form.
    """
    return ablco2_ref * co2_vmr_ppmv / CO2_VMR_REF_PPMV


def set_forcing(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData=None,
    terrain: TerrainData=None
) -> tuple[PhysicsTendency, PhysicsData]:
    # 2. daily-mean radiative forcing
    physics_data = get_zonal_average_fields(state, physics_data, forcing=forcing, terrain=terrain)

    # total surface albedo
    fmask = terrain.fmask

    snowc = jnp.minimum(1.0, forcing.snowc_am)
    alb_l = forcing.alb0 + snowc * (parameters.mod_radcon.albsn - forcing.alb0)
    alb_s = parameters.mod_radcon.albsea + forcing.sice_am * (parameters.mod_radcon.albice - parameters.mod_radcon.albsea)
    albsfc = alb_s + fmask * (alb_l - alb_s)

    ablco2 = ablco2_from_co2_vmr(forcing.co2_vmr)

    mod_radcon = physics_data.mod_radcon.copy(snowc=snowc, alb_l=alb_l, alb_s=alb_s, albsfc=albsfc, ablco2=ablco2)

    physics_data = physics_data.copy(mod_radcon=mod_radcon)
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendencies, physics_data
