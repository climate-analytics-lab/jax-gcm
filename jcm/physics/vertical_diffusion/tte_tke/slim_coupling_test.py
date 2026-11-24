"""The SLIM land solved with the lowest level (``land_model: slim``).

``couple_surface_tiles`` couples SLIM's top soil layer to the atmosphere
through the land tile's Richtmyer–Morton relations, with the rest of the soil
column eliminated into the balance's capacity and ground flux, SLIM's melt and
its bucket evaporation form. These tests check what that builds: the land and
column energy budgets, the evaporation regimes, the melt and the derivatives.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.surface.slim import slim_land as sl
from jcm.testing import check_gradients
from .land_coupling_test import _column_integrals, _land_column
from .vertical_diffusion import vertical_diffusion_column
from .vertical_diffusion_types import LandBalanceInputs, SlimColumnInputs, VDiffParameters

DT = 720.0
P = sl.SlimParameters()


@contextlib.contextmanager
def float64():
    with jax.enable_x64(True):
        yield


def _slim_inputs(state, *, water=100.0, snow=0.0, cap=200.0, t_soil=None, glacier=0.0,
                 ice=None, sw=600.0, lw=350.0, r_evap=150.0):
    n = state.temperature.shape[0]
    t_top = state.surface_temperature[:, 2]
    z, dz, zh, maxw = sl.soil_layer_geometry(P)
    if t_soil is None:
        t_soil = t_top[None, :] + jnp.linspace(0.0, -6.0, 10)[:, None]
    t_soil = t_soil.at[0].set(t_top)
    if ice is None:
        ice = sl.seed_soil_ice(t_soil, maxw)
    g = jnp.full(n, glacier)
    _, cv, tkh = sl.soil_thermal_properties(jnp.full(n, 1.5), jnp.full(n, 2.0e6),
                                            jnp.full(n, 2.29), jnp.full(n, 2.1e6), g, z, zh)
    w = jnp.full(n, water)
    s = jnp.full(n, snow)
    beta = sl.bucket_beta(w, jnp.full(n, cap), P)
    snow_on = sl.snow_indicator(s, g, P)
    slim = SlimColumnInputs(
        soil_temperature=t_soil, soil_ice=ice, heat_capacity=cv, interface_conductivity=tkh,
        depth=z, thickness=dz, max_water=maxw, snow=s, glacier=g,
        beta_effective=snow_on + (1.0 - snow_on) * beta, snow_on=snow_on,
        evaporative_resistance=jnp.full(n, r_evap),
        available_water=jnp.where(g > 0.5, jnp.inf, w + s), params=P)
    zeros = jnp.zeros(n)
    from jcm.physics.surface.echam.jsbach_land import JsbachLandParameters
    return LandBalanceInputs(
        temperature=t_top, soil_temperature=zeros, saturation_slope=zeros,
        net_shortwave=jnp.full(n, sw), longwave_down=jnp.full(n, lw),
        emissivity=jnp.full(n, 0.996), heat_capacity=zeros, conductance=zeros,
        melt_capped=jnp.zeros(n, bool), params=JsbachLandParameters(), slim=slim)


def _state(t_skin=300.0, rh=0.3, beta=0.5, snow_on=0.0, wind=4.0):
    state = _land_column(t_skin=t_skin, rh=rh, wind=wind)
    n = state.temperature.shape[0]
    f = jnp.full(n, beta)
    ones = jnp.ones(n)
    return state._replace(
        surface_cair=jnp.stack([ones, ones, f], 1), surface_csat=jnp.stack([ones, ones, f], 1),
        surface_wetness=jnp.stack([ones, ones, f], 1),
        surface_sublimation_fraction=state.surface_sublimation_fraction.at[:, 2].set(snow_on * f))


def _run(state, land):
    return vertical_diffusion_column(state, VDiffParameters.default(), DT, land=land)


def _soil_energy(t, ice, land):
    s = land.slim
    return (jnp.sum(s.heat_capacity * s.thickness[:, None] * t, axis=0)
            - c.alhf * jnp.sum(ice, axis=0))


class TestSlimEnergyBudget:

    def test_land_energy_budget_closes_to_round_off(self):
        """d/dt Σ(cv·dz·T − L_f·ice) = Rn − SH − LH − L_f·melt, every step.

        Also: the top layer's storage and the conduction below it are the
        balance's storage and ground flux, and the atmosphere receives the
        tile's sensible heat and moisture exactly (pev_vdiff).
        """
        with float64():
            for t_skin, water, snow, t_deep in ((305.0, 120.0, 0.0, 296.0),
                                                (274.0, 50.0, 3.0, 271.0),
                                                (268.0, 50.0, 40.0, 272.5)):
                state = _state(t_skin=t_skin, beta=0.5)
                tsoil = (t_skin + jnp.linspace(0.0, t_deep - t_skin, 10))[:, None]
                land = _slim_inputs(state, water=water, snow=snow, t_soil=tsoil,
                                    sw=700.0 if t_skin > 273 else 400.0)
                tend, diag = _run(state, land)
                lb = diag.land_balance
                so = lb.slim
                closure = (lb.net_radiation - lb.sensible_heat_flux - lb.latent_heat_flux
                           - lb.heat_storage - lb.ground_heat_flux - lb.melt_heat_flux)
                np.testing.assert_allclose(closure, 0.0, atol=1e-8)
                d_energy = (_soil_energy(so.soil_temperature, so.soil_ice, land)
                            - _soil_energy(land.slim.soil_temperature, land.slim.soil_ice,
                                           land)) / DT
                np.testing.assert_allclose(
                    d_energy, lb.net_radiation - lb.sensible_heat_flux - lb.latent_heat_flux
                    - lb.melt_heat_flux, atol=1e-7)
                np.testing.assert_allclose(so.soil_temperature[0], lb.temperature)
                col_q, col_t = _column_integrals(state, tend)
                np.testing.assert_allclose(col_t, lb.sensible_heat_flux, rtol=1e-9)
                np.testing.assert_allclose(col_q, lb.evaporation, rtol=1e-9, atol=1e-15)

    def test_latent_heat_is_sublimation_over_snow_and_vaporisation_otherwise(self):
        with float64():
            for snow, latent in ((0.0, c.alhc), (20.0, c.alhs)):
                state = _state(t_skin=290.0, snow_on=float(snow > 0))
                _, diag = _run(state, _slim_inputs(state, snow=snow))
                lb = diag.land_balance
                np.testing.assert_allclose(lb.latent_heat_flux, latent * lb.evaporation,
                                           rtol=1e-10)


class TestSlimEvaporation:

    def test_dry_bucket_stops_evaporation_and_a_wet_one_does_not(self):
        with float64():
            state = _state(t_skin=308.0, rh=0.3)
            _, dry = _run(state, _slim_inputs(state, water=0.0, sw=800.0))
            _, wet = _run(state, _slim_inputs(state, water=180.0, sw=800.0))
            assert float(dry.land_balance.evaporation[0]) == 0.0
            assert float(wet.land_balance.evaporation[0]) > 3e-5       # > 2.6 mm/d
            # The dry land sends its energy up as sensible heat, hotter.
            assert float(dry.land_balance.temperature[0]) > float(wet.land_balance.temperature[0])
            assert float(dry.land_balance.sensible_heat_flux[0]) > float(
                wet.land_balance.sensible_heat_flux[0])

    def test_the_factor_is_slims_resistance_form(self):
        """E = ρ·β·(q_s − q_a)/(r_evap + r_a) with this step's r_a."""
        with float64():
            state = _state(t_skin=300.0)
            land = _slim_inputs(state, water=75.0)                       # β = 0.5
            _, diag = _run(state, land)
            chu = diag.surface_exchange_moisture[0, 2]
            expected = 0.5 / (1.0 + chu * 150.0)
            np.testing.assert_allclose(diag.land_balance.slim.evaporation_factor[0], expected,
                                       rtol=1e-12)

    def test_one_step_cannot_take_more_water_than_the_land_holds(self):
        with float64():
            state = _state(t_skin=315.0, rh=0.1)
            land = _slim_inputs(state, water=1e-3, cap=1e-3 / 0.75 * 1.0, sw=900.0, r_evap=0.0)
            _, diag = _run(state, land)
            e_dt = float(diag.land_balance.evaporation[0]) * DT
            assert 0.0 < e_dt <= 1e-3 * 1.05


class TestSlimMelt:

    def test_snow_melt_holds_the_surface_at_the_freezing_point(self):
        with float64():
            state = _state(t_skin=272.0, rh=0.5)
            deep = _slim_inputs(state, snow=200.0, water=50.0, sw=900.0, lw=360.0)
            _, d = _run(state, deep)
            assert float(d.land_balance.melt_heat_flux[0]) > 0.0
            np.testing.assert_allclose(d.land_balance.temperature[0], c.tmelt, atol=1e-9)
            # A thin snow pack melts away within the step and the surface
            # warms past the freezing point.
            thin = _slim_inputs(state, snow=0.02, water=50.0, sw=900.0, lw=360.0)
            _, t = _run(state, thin)
            np.testing.assert_allclose(t.land_balance.slim.snow_melt[0] * DT, 0.02, rtol=1e-10)
            assert float(t.land_balance.temperature[0]) > c.tmelt
            np.testing.assert_allclose(t.land_balance.melt_heat_flux,
                                       c.alhf * t.land_balance.slim.snow_melt, rtol=1e-12)

    def test_glacier_never_runs_out(self):
        with float64():
            state = _state(t_skin=272.5, rh=0.5)
            g = _slim_inputs(state, snow=0.0, glacier=1.0, sw=900.0, lw=360.0)
            _, d = _run(state, g)
            np.testing.assert_allclose(d.land_balance.temperature[0], c.tmelt, atol=1e-9)
            assert float(d.land_balance.melt_heat_flux[0]) > 0.0


class TestSlimGradients:

    def test_gradients_through_beta_melt_and_soil(self):
        with float64():
            state = _state(t_skin=300.0, rh=0.4)

            def f_beta(water, t_top):
                st = state._replace(surface_temperature=state.surface_temperature.at[:, 2].set(t_top))
                _, d = _run(st, _slim_inputs(st, water=water[0]))
                return d.land_balance.evaporation, d.land_balance.slim.soil_temperature

            check_gradients(f_beta, (jnp.full(1, 60.0), jnp.full(1, 300.0)), rtol=1e-4)

            cold = _state(t_skin=272.0, rh=0.5)

            def f_melt(snow):
                _, d = _run(cold, _slim_inputs(cold, snow=snow[0], sw=900.0, lw=360.0))
                return d.land_balance.melt_heat_flux, d.land_balance.temperature

            # Limited by the snow present (0.02 kg/m2 melts within the step).
            check_gradients(f_melt, (jnp.full(1, 0.02),), rtol=1e-3)
            g = jax.grad(lambda s: f_melt(s)[0][0])(jnp.full(1, 0.02))
            assert bool(jnp.all(jnp.isfinite(g))) and float(g[0]) > 0.0
