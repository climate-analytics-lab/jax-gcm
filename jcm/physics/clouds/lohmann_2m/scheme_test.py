"""The 2M's inputs: anchor, increments and detrainment (``cloud_microphysics_2m``).

ECHAM's ``cloud_micro_interface`` receives the previous time level
(``ptm1``, ``pqm1``, ``pxlm1``, ``pxim1``), the tendencies accumulated since
(``ptte``, ``pqte``, ``pxlte``, ``pxite``) and the convective detrainment
separately (``pxtecl``, ``pxteci``; ``physc.f90:1073-1081``). These tests pin
how the column function consumes that split and that the term feeds it.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics import thermodynamics
from jcm.physics.clouds.cloud_utils import latent_heat_over_cp
from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
from jcm.physics.clouds.lohmann_2m_params import CloudParams2M

DT = 1800.0
NLEV = 16


def _column():
    """Build a column with a warm liquid cloud, a mixed-phase layer and a cirrus deck."""
    temperature = jnp.linspace(215.0, 295.0, NLEV)
    pressure = jnp.linspace(2.0e4, 1.0e5, NLEV)
    rho = pressure / (287.04 * temperature)
    qsat = thermodynamics.saturation_specific_humidity_and_derivative(
        temperature, pressure, phase="water")[0]
    q = 0.85 * qsat
    idx = jnp.arange(NLEV)
    qc = jnp.where((idx >= 9) & (idx < 14), 4e-4, 0.0)
    qi = jnp.where((idx >= 2) & (idx < 8), 6e-5, 0.0)
    cf = jnp.where((qc + qi) > 0, 0.6, 0.0)
    return dict(
        temperature_m1=temperature, specific_humidity_m1=q, pressure=pressure,
        qc_m1=qc, qi_m1=qi,
        qnc_m1=jnp.where(qc > 0, 5e7 / rho, 0.0),
        qni_m1=jnp.where(qi > 0, 5e4 / rho, 0.0),
        cloud_fraction=cf, air_density=rho,
        layer_thickness=jnp.full(NLEV, 500.0), tke=jnp.full(NLEV, 0.1),
        activated_cdnc=jnp.full(NLEV, 5e7), ice_nuclei=jnp.zeros(NLEV),
        ice_nuclei_deposition=jnp.zeros(NLEV),
    )


def _increments(scale=1.0):
    idx = jnp.arange(NLEV, dtype=jnp.float32)
    return dict(
        temperature_increment=scale * -0.4 * jnp.sin(idx),
        humidity_increment=scale * 2e-5 * jnp.cos(idx),
        qc_increment=scale * jnp.where((idx >= 9) & (idx < 14), -2e-5, 0.0),
        qi_increment=scale * jnp.where((idx >= 2) & (idx < 8), 5e-6, 0.0),
    )


def _detrainment():
    idx = jnp.arange(NLEV)
    return dict(
        detrained_qc=jnp.where((idx >= 8) & (idx < 12), 3e-5, 0.0),
        detrained_qi=jnp.where((idx >= 3) & (idx < 7), 1e-5, 0.0),
    )


def _call(inputs):
    col = dict(inputs)
    args = [col.pop(k) for k in (
        "temperature_m1", "specific_humidity_m1", "pressure", "qc_m1", "qi_m1",
        "qnc_m1", "qni_m1", "cloud_fraction", "air_density", "layer_thickness",
        "tke", "activated_cdnc", "ice_nuclei", "ice_nuclei_deposition")]
    return cloud_microphysics_2m(*args, DT, CloudParams2M.default(), **col)


def _split_and_lumped(detrainment):
    """Run the column with ``detrainment`` passed apart, and inside the increments."""
    column = _column()
    split = _call({**column, **_increments(), **detrainment})
    lumped_increments = _increments()
    for phase in ("qc", "qi"):
        lumped_increments[f"{phase}_increment"] = (
            lumped_increments[f"{phase}_increment"]
            + detrainment.get(f"detrained_{phase}", jnp.zeros(NLEV)))
    return split, _call({**column, **lumped_increments})


def test_warm_liquid_detrainment_is_part_of_the_condensate_increment():
    """Liquid detrained above ``tmelt`` gives the lumped result to the last bit.

    ECHAM's 2M gives the detrained condensate ``zxtec`` its own rules only
    where they can act: it keeps it out of the ice sedimentation, gives it a
    crystal number where ``ll_cv`` holds (below ``tmelt``) and re-splits it by
    ``lo2``, which puts all of it in the liquid where ``lo2`` fails (always
    above ``tmelt``). Liquid detrained above ``tmelt`` therefore enters
    exactly as ``pxlte + pxtecl``, and moving it into the increment changes
    nothing.
    """
    column = _column()
    warm = column["temperature_m1"] > c.tmelt
    cloudy_liquid = column["qc_m1"] > 0.0
    assert bool(jnp.any(warm & cloudy_liquid))
    detrained_qc = jnp.where(warm & cloudy_liquid, 3e-5, 0.0)
    split, lumped = _split_and_lumped({"detrained_qc": detrained_qc})
    for a, b in zip(jax.tree.leaves(split), jax.tree.leaves(lumped)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_cold_ice_detrainment_is_not_an_increment():
    """Ice detrained below ``tmelt`` follows its own rules (ICE-1/3/4, #941).

    As an increment it would sediment and keep the ice phase wherever the
    ice-memory criterion does; as detrainment it does not fall this step,
    brings its crystal number and is re-split by ``lo2``. The mixed-phase
    and cirrus levels that receive it end differently.
    """
    detrained_qi = _detrainment()["detrained_qi"]
    split, lumped = _split_and_lumped({"detrained_qi": detrained_qi})
    receiving = np.asarray(detrained_qi) > 0.0
    diff = np.abs(np.asarray(split[0].dqidt - lumped[0].dqidt))
    assert float(np.max(diff[receiving])) > 0.0


def test_no_increments_is_the_anchor_alone():
    """Omitted increments and detrainment are exactly zero ones."""
    column = _column()
    implicit = _call(column)
    zeros = {k: jnp.zeros(NLEV) for k in (*_increments(), *_detrainment())}
    explicit = _call({**column, **zeros})
    for a, b in zip(jax.tree.leaves(implicit), jax.tree.leaves(explicit)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _budgets(inputs, outputs):
    """Column water and moist-enthalpy residuals of the scheme's own tendencies.

    The tendencies are relative to the provisional state, so they must close
    against the surface precipitation alone whatever the anchor, the
    increments and the detrainment are. Enthalpy uses the moist heat capacity
    at the ANCHOR humidity, the one the scheme converts latent heat with.
    """
    tend, rain, snow = outputs[0], float(outputs[1]), float(outputs[2])
    mass = np.asarray(inputs["air_density"] * inputs["layer_thickness"])
    water = float(np.sum(mass * np.asarray(tend.dqdt + tend.dqcdt + tend.dqidt)))
    water_residual = water + rain + snow
    water_gross = float(np.sum(mass * np.abs(np.asarray(tend.dqdt)))) + rain + snow
    q_anchor = np.maximum(np.asarray(inputs["specific_humidity_m1"]), 0.0)
    cp = c.cpd + (c.cpv - c.cpd) * q_anchor
    heating = mass * cp * np.asarray(tend.dtedt)
    d_enthalpy = float(np.sum(heating - mass * c.alhc * np.asarray(tend.dqcdt)
                              - mass * c.alhs * np.asarray(tend.dqidt)))
    enthalpy_residual = d_enthalpy - (c.alhc * rain + c.alhs * snow)
    enthalpy_gross = float(np.sum(np.abs(heating)))
    return water_residual, water_gross, enthalpy_residual, enthalpy_gross


def _stage_two(column, increments, detr):
    """Move the anchor and keep the provisional state (a dynamics offset).

    Moves the anchor by ``offset`` and the increment by ``-offset``, so the
    provisional state is unchanged up to rounding while every quantity the
    scheme evaluates at the anchor moves.
    """
    offsets = {"temperature_m1": ("temperature_increment", 0.7),
               "specific_humidity_m1": ("humidity_increment", -3e-5),
               "qc_m1": ("qc_increment", 1e-5), "qi_m1": ("qi_increment", 2e-6)}
    column, increments = dict(column), dict(increments)
    for anchor_key, (inc_key, offset) in offsets.items():
        shift = jnp.where(column[anchor_key] > 0, offset, 0.0) \
            if anchor_key in ("qc_m1", "qi_m1") else offset
        column[anchor_key] = column[anchor_key] - shift
        increments[inc_key] = increments[inc_key] + shift
    return {**column, **increments, **detr}


@pytest.mark.parametrize("stage", ["first", "second"])
def test_column_water_and_enthalpy_close_with_detrainment(stage):
    column, increments, detr = _column(), _increments(), _detrainment()
    inputs = ({**column, **increments, **detr} if stage == "first"
              else _stage_two(column, increments, detr))
    outputs = _call(inputs)
    wres, wgross, eres, egross = _budgets(inputs, outputs)
    assert wgross > 0 and egross > 1.0, "column did nothing"
    assert abs(wres) < 1e-5 * wgross, (stage, wres, wgross)
    assert abs(eres) < 1e-5 * egross, (stage, eres, egross)


# ---------------------------------------------------------------------------
# The term: an upstream cooling tendency condenses in the same step
# ---------------------------------------------------------------------------

CLOUD_LEVEL = 5
TERM_NLEV = 8


def _composition(cooling_rate):
    """Compose [column diagnostics stub, a 'radiation' term that only cools, 2M]."""
    from jcm.physics.clouds.cloud_data import CloudData
    from jcm.physics.clouds.lohmann_2m import Lohmann2MMicrophysics
    from jcm.physics.composable_physics import ComposablePhysics
    from jcm.physics.physics_term import PhysicsTerm
    from jcm.physics_interface import PhysicsTendency

    nlev, k = TERM_NLEV, CLOUD_LEVEL
    pressure = jnp.linspace(5.0e4, 1.0e5, nlev)
    cover = jnp.zeros(nlev).at[k].set(0.5)

    class _Column(PhysicsTerm):
        name: ClassVar[str] = "column"
        category: ClassVar[str] = "diagnostics"
        requires: ClassVar[tuple[str, ...]] = ()
        provides: ClassVar[tuple[str, ...]] = (
            "pressure_full", "air_density", "layer_thickness", "clouds",
            "aerosol", "activated_cdnc")

        def __call__(self, state, diagnostics, forcing, terrain):
            p = jnp.broadcast_to(pressure[:, None], state.temperature.shape)
            ncols = state.temperature.shape[1]
            return PhysicsTendency.zeros(state.temperature.shape), {
                **diagnostics,
                "pressure_full": p,
                "air_density": p / (287.04 * state.temperature),
                "layer_thickness": jnp.full_like(p, 400.0),
                "clouds": CloudData.zeros((ncols,), nlev).copy(
                    cloud_fraction=jnp.broadcast_to(
                        cover[:, None], state.temperature.shape)),
                "aerosol": {},
                "activated_cdnc": jnp.full_like(p, 5.0e7),
            }

    class _Cooling(PhysicsTerm):
        name: ClassVar[str] = "cooling"
        category: ClassVar[str] = "radiation"
        requires: ClassVar[tuple[str, ...]] = ()
        provides: ClassVar[tuple[str, ...]] = ()

        def __call__(self, state, diagnostics, forcing, terrain):
            rate = jnp.zeros(nlev).at[k].set(cooling_rate)
            tend = PhysicsTendency.zeros(state.temperature.shape)
            return tend.copy(temperature=jnp.broadcast_to(
                rate[:, None], state.temperature.shape)), diagnostics

    return ComposablePhysics([_Column(), _Cooling(), Lohmann2MMicrophysics()],
                             vectorize_columns=True, dt_seconds=DT)


def _cloudy_state():
    from jcm.physics_interface import PhysicsState

    nlev, k = TERM_NLEV, CLOUD_LEVEL
    shape = (nlev, 2, 1)
    temperature = jnp.linspace(262.0, 292.0, nlev)
    pressure = jnp.linspace(5.0e4, 1.0e5, nlev)
    qsat = thermodynamics.saturation_specific_humidity_and_derivative(
        temperature, pressure, phase="water")[0]
    rh = jnp.full(nlev, 0.5).at[k].set(0.9)     # box subsaturated: no 5.4 correction
    qc = jnp.zeros(nlev).at[k].set(5e-5)
    rho = pressure / (287.04 * temperature)

    def grid(column):
        return jnp.broadcast_to(column[:, None, None], shape)

    return PhysicsState(
        u_wind=jnp.zeros(shape), v_wind=jnp.zeros(shape),
        temperature=grid(temperature), specific_humidity=grid(rh * qsat),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones(shape[1:]),
        tracers={"qc": grid(qc), "qi": jnp.zeros(shape),
                 "qnc": grid(jnp.where(qc > 0, 5e7 / rho, 0.0)),
                 "qni": jnp.zeros(shape)}), temperature, pressure, rh * qsat


def test_upstream_radiative_cooling_condenses_in_the_same_step():
    """A cooling tendency from an upstream term condenses zqcdif at once.

    ECHAM: radheat adds to ptte before cloud (physc.f90:776-794), so cloud-top
    cooling condenses in the step it happens: zqcdif = (0 - zdqsat)·paclc,
    zdqsat = ztmst·ptte·dqs/dT / (1 + paclc·L/cp·dqs/dT)
    (mo_cloud_micro_2m.f90 section 5). First step, so this is the running
    tendency alone (no carried anchor).
    """
    from jcm.forcing import ForcingData

    state, temperature, pressure, q = _cloudy_state()
    rate = -3.0e-4                                   # K/s, about -26 K/day
    k = CLOUD_LEVEL

    def vapour_tendency(cooling_rate):
        physics = _composition(cooling_rate)
        forcing = ForcingData.zeros(state.normalized_surface_pressure.shape)
        tend, _ = physics.compute_tendencies(
            state, forcing, _aquaplanet_like(state))
        return np.asarray(tend.specific_humidity[k, 0, 0])

    condensed = -DT * (vapour_tendency(rate) - vapour_tendency(0.0))
    _, dqsdt = thermodynamics.saturation_specific_humidity_and_derivative(
        temperature[k], pressure[k], phase="water")
    lvdcp, _ = latent_heat_over_cp(q[k])
    cf = 0.5
    zdqsat = DT * rate * float(dqsdt) / (1.0 + cf * float(lvdcp) * float(dqsdt))
    zqcdif = -zdqsat * cf
    assert zqcdif > 0.0
    np.testing.assert_allclose(condensed, zqcdif, rtol=1e-3)


def _aquaplanet_like(state):
    """Terrain for a bare (nlon, nlat) column block."""
    from jcm.terrain import TerrainData

    shape = state.normalized_surface_pressure.shape
    coords = type("_Coords", (), {})()
    coords.horizontal = type("_H", (), {"nodal_shape": shape})()
    return TerrainData.aquaplanet(coords)
