from jcm.terrain import TerrainData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import ablco2_ref, PhysicsData
from jcm.forcing import ForcingData, DEFAULT_CO2_VMR_PPMV, land_snow_cover
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.radiation.speedy_shortwave import get_zonal_average_fields
import jax.numpy as jnp


# Reference CO2 concentration (ppmv) at which the CO2-band absorptivity
# `ModRadConParameters.ablco2_ref` applies: the forcing's default concentration
# (`jcm.forcing.DEFAULT_CO2_VMR_PPMV`, 420 ppmv), so a run with the default
# forcing absorbs with `ablco2_ref` itself -- SPEEDY's 6.0 unless tuned, the
# value its Fortran uses without the `increase_co2` ramp -- and any other
# concentration scales it linearly.
CO2_VMR_REF_PPMV = DEFAULT_CO2_VMR_PPMV


def ablco2_from_co2_vmr(co2_vmr_ppmv: jnp.ndarray, ablco2_ref=ablco2_ref) -> jnp.ndarray:
    """CO2 absorptivity used by SPEEDY shortwave/longwave, derived from
    a CO2 mixing ratio (ppmv).

    Linear scaling against the reference: `ablco2 = ablco2_ref * C / C_ref`,
    with `ablco2_ref` the absorptivity at the reference concentration (the
    `ModRadConParameters.ablco2_ref` tunable; SPEEDY's value by default).
    Chosen because SPEEDY's legacy `exp(0.005 * dyears)` ramp grew the
    absorptivity at ~0.5%/yr, close to the historical growth rate of the CO2
    concentration. A pure log-in-ratio mapping (CO2
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

    # Whole-land snow cover: ``snowc_am`` describes the non-glacier land and
    # a glacier is fully snow covered (the bundle convention, see
    # ``jcm.forcing.land_snow_cover``). With it, ``alb0 + S·(albsn − alb0)``
    # is exactly the tile average of the glacier at ``albsn`` and the
    # non-glacier land at its background plus snow. A forcing without a
    # glacier map (the T30 climatology) gives ``min(1, snowc_am)``.
    snowc = land_snow_cover(forcing.snowc_am,
                            getattr(forcing, "glacier_fraction", None))
    alb_l = forcing.alb0 + snowc * (parameters.mod_radcon.albsn - forcing.alb0)
    alb_s = parameters.mod_radcon.albsea + forcing.sice_am * (parameters.mod_radcon.albice - parameters.mod_radcon.albsea)
    albsfc = alb_s + fmask * (alb_l - alb_s)

    ablco2 = ablco2_from_co2_vmr(forcing.co2_vmr, parameters.mod_radcon.ablco2_ref)

    mod_radcon = physics_data.mod_radcon.copy(snowc=snowc, alb_l=alb_l, alb_s=alb_s, albsfc=albsfc, ablco2=ablco2)

    physics_data = physics_data.copy(mod_radcon=mod_radcon)
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendencies, physics_data
