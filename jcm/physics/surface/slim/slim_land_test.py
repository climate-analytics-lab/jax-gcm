"""SLIM's column physics (``mml_main.F90``): soil column, phase change, buckets."""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.surface.slim import slim_land as sl
from jcm.testing import check_gradients

DT = 720.0
P = sl.SlimParameters()


@contextlib.contextmanager
def float64():
    with jax.enable_x64(True):
        yield


def _column(ncol=3, tk=1.5, cv=2.0e6, glacier=0.0):
    z, dz, zh, maxw = sl.soil_layer_geometry(P)
    _, cvs, tkh = sl.soil_thermal_properties(
        jnp.full(ncol, tk), jnp.full(ncol, cv), jnp.full(ncol, 2.29), jnp.full(ncol, 2.1e6),
        jnp.full(ncol, glacier), z, zh)
    return z, dz, zh, maxw, cvs, tkh


class TestSoilGrid:

    def test_geometry_is_slims_init_cold(self):
        with float64():
            z, dz, zh, maxw = sl.soil_layer_geometry(P)
            i = np.arange(1, 11)
            np.testing.assert_allclose(z, -0.025 * (np.exp(0.5 * (i - 0.5)) - 1.0))
            np.testing.assert_allclose(dz[0], 0.5 * abs(z[0] + z[1]))
            np.testing.assert_allclose(dz[1:-1], 0.5 * (z[:-2] - z[2:]))
            np.testing.assert_allclose(dz[-1], z[-2] - z[-1])
            np.testing.assert_allclose(zh, -np.cumsum(dz))
            assert float(maxw[0]) == 0.0
            np.testing.assert_allclose(maxw[1:], 300.0 * dz[1:])
            # The deepest node is at 2.87 m, the column ends at 3.43 m, and
            # the top layer is 1.75 cm thick.
            np.testing.assert_allclose(-float(z[-1]), 2.865, rtol=1e-3)
            np.testing.assert_allclose(-float(zh[-1]), 3.433, rtol=1e-3)
            np.testing.assert_allclose(float(dz[0]), 0.0175, rtol=0.01)

    def test_interface_conductivity_is_harmonic(self):
        with float64():
            z, dz, zh, _ = sl.soil_layer_geometry(P)
            _, _, tkh = sl.soil_thermal_properties(1.5, 2e6, 2.29, 2.1e6, 0.0, z, zh)
            np.testing.assert_allclose(tkh, 1.5)          # equal layers: tk itself
            _, cv, tkh_g = sl.soil_thermal_properties(1.5, 2e6, 2.29, 2.1e6, 1.0, z, zh)
            np.testing.assert_allclose(tkh_g, 2.29)       # glacier: ice
            np.testing.assert_allclose(cv, 2.1e6)


class TestSoilTridiagonal:

    def _step(self, t, cv, tkh, z, dz, flux, dt=DT):
        den, num, e, f = sl.eliminate_soil_column(t, cv, tkh, z, dz, dt)
        d1 = (num + flux) / den
        return sl.complete_soil_column(t, d1, e, f)

    def test_the_solution_satisfies_the_implicit_system(self):
        """Every row of SLIM's tridiagonal holds to round-off after the sweeps."""
        with float64():
            z, dz, zh, _, cv, tkh = _column(ncol=2)
            rng = np.random.default_rng(1)
            t = jnp.asarray(280.0 + 10.0 * rng.standard_normal((10, 2)))
            flux = jnp.asarray([150.0, -40.0])
            tn = self._step(t, cv, tkh, z, dz, flux)
            dT = tn - t
            dzz = (z[:-1] - z[1:])[:, None]
            cond = tkh * (tn[:-1] - tn[1:]) / dzz          # implicit i -> i+1
            lhs = cv * dz[:, None] * dT / DT
            rhs = jnp.concatenate([(flux - cond[0])[None], cond[:-1] - cond[1:],
                                   cond[-1:]], axis=0)
            np.testing.assert_allclose(lhs, rhs, atol=1e-9)
            # Energy: the column takes up exactly the surface flux.
            np.testing.assert_allclose(jnp.sum(lhs, axis=0), flux, atol=1e-9)

    def test_uniform_heating_rate_of_an_insulated_column(self):
        """Constant surface heating, no flux through the base: after the
        diffusive transient every layer warms at F/Σ(cv·dz), the analytic
        rate of the continuous problem.
        """
        with float64():
            z, dz, zh, _, cv, tkh = _column(ncol=1)
            t = jnp.full((10, 1), 280.0)
            flux = jnp.asarray([100.0])
            step = jax.jit(lambda t: self._step(t, cv, tkh, z, dz, flux, dt=3600.0))
            for _ in range(24 * 200):
                t = step(t)
            t1 = step(t)
            rate = (t1 - t)[:, 0] / 3600.0
            np.testing.assert_allclose(rate, 100.0 / float(jnp.sum(cv[:, 0] * dz)), rtol=1e-3)

    def test_decay_of_the_gravest_mode_matches_the_diffusion_equation(self):
        """An insulated column relaxes like ``exp(−κ(π/L)²t)``, L its depth.

        Ten stretched layers resolve the gravest cosine mode to a few per cent,
        so this checks the column's conduction against the analytic diffusion
        solution, not just its algebra.
        """
        with float64():
            z, dz, zh, _, cv, tkh = _column(ncol=1)
            depth = -float(zh[-1])
            kappa = 1.5 / 2.0e6
            t = (280.0 + jnp.cos(jnp.pi * (-z) / depth))[:, None]
            dt = 1800.0
            step = jax.jit(lambda t: self._step(t, cv, tkh, z, dz, jnp.zeros(1), dt=dt))
            # Project onto the mode by the heat-weighted cosine.
            w = (cv[:, 0] * dz) * jnp.cos(jnp.pi * (-z) / depth)
            a0 = float(jnp.sum(w * (t[:, 0] - 280.0)))
            nsteps = 48 * 20
            for _ in range(nsteps):
                t = step(t)
            mean = float(jnp.sum(cv[:, 0] * dz * t[:, 0]) / jnp.sum(cv[:, 0] * dz))
            a1 = float(jnp.sum(w * (t[:, 0] - mean)))
            rate = -np.log(a1 / a0) / (nsteps * dt)
            np.testing.assert_allclose(rate, kappa * (np.pi / depth) ** 2, rtol=0.1)

    def test_gradient_through_the_soil_solve(self):
        with float64():
            z, dz, zh, _, cv, tkh = _column(ncol=2)

            def f(t, flux, tk):
                _, cvs, tkh = sl.soil_thermal_properties(
                    tk, jnp.full(2, 2e6), 2.29, 2.1e6, jnp.zeros(2), z, zh)
                return self._step(t, cvs, tkh, z, dz, flux)

            t = jnp.asarray(np.linspace(270.0, 285.0, 20).reshape(10, 2))
            check_gradients(f, (t, jnp.asarray([120.0, -30.0]), jnp.asarray([1.2, 0.8])),
                            rtol=1e-6)


class TestPhaseChange:

    def test_freezing_and_thawing_conserve_energy(self):
        with float64():
            z, dz, zh, maxw, cv, tkh = _column(ncol=3)
            t = jnp.stack([jnp.full(10, 270.0), jnp.full(10, 275.0), jnp.full(10, c.tmelt)], 1)
            ice = jnp.stack([jnp.zeros(10), maxw, 0.5 * maxw], 1)   # liquid, frozen, half
            tn, icen, heat = sl.phase_change(t, ice, cv, dz, maxw, DT)
            sensible = jnp.sum(cv * dz[:, None] * (tn - t), axis=0) / DT
            latent = c.alhf * jnp.sum(icen - ice, axis=0) / DT
            np.testing.assert_allclose(sensible, latent, atol=1e-8)
            np.testing.assert_allclose(heat, latent, atol=1e-8)
            # Below freezing, liquid freezes and the layers warm toward T_f;
            # above it, ice melts and they cool toward it; the top layer holds
            # no water and keeps its temperature.
            assert bool(jnp.all(icen[1:, 0] > 0.0)) and bool(jnp.all(tn[1:, 0] > 270.0))
            assert bool(jnp.all(icen[1:, 1] < maxw[1:])) and bool(jnp.all(tn[1:, 1] < 275.0))
            assert float(tn[0, 0]) == 270.0 and float(tn[0, 1]) == 275.0
            np.testing.assert_array_equal(tn[:, 2], t[:, 2])         # at T_f: nothing
            assert bool(jnp.all((icen >= 0) & (icen <= maxw[:, None] + 1e-12)))

    def test_a_layer_holds_at_the_freezing_point_until_its_water_is_frozen(self):
        with float64():
            z, dz, zh, maxw, cv, tkh = _column(ncol=1)
            t = jnp.full((10, 1), 273.15 - 0.01)
            tn, icen, _ = sl.phase_change(t, jnp.zeros((10, 1)), cv, dz, maxw, DT)
            np.testing.assert_allclose(tn[1:, 0], c.tmelt)
            assert bool(jnp.all(icen[1:, 0] < maxw[1:]))


class TestEvaporationFactor:

    def test_factor_is_the_resistance_form(self):
        """E = ρ·C|U|·csat·Δq equals SLIM's ρ·β·Δq/(r_evap + r_a)."""
        with float64():
            chu, r_e, beta = 0.015, 150.0, 0.4
            f = sl.evaporation_factor(beta, 0.0, chu, r_e, P)
            np.testing.assert_allclose(chu * f, beta / (1.0 / chu + r_e))
            f_snow = sl.evaporation_factor(beta, 1.0, chu, r_e, P)
            np.testing.assert_allclose(chu * f_snow, 1.0 / (1.0 / chu + r_e))
            # The total resistance is capped at 1e5 s/m.
            f_calm = sl.evaporation_factor(1.0, 0.0, 1e-6, r_e, P)
            np.testing.assert_allclose(1e-6 * f_calm, 1.0 / 1e5)

    def test_beta_and_its_surrogate_derivative(self):
        with float64():
            cap = jnp.asarray([200.0, 200.0, 200.0])
            w = jnp.asarray([-5.0, 75.0, 151.5])
            np.testing.assert_allclose(sl.bucket_beta(w, cap, P), [0.0, 0.5, 1.0])
            g = jax.grad(lambda w: jnp.sum(sl.bucket_beta(w, cap, P)))(w)
            # Interior: the exact slope 1/(0.75 cap); a full bucket still sees
            # a small slope (the surrogate), never an exact zero.
            np.testing.assert_allclose(g[1], 1.0 / 150.0, rtol=1e-6)
            assert 0.0 < float(g[2]) < 1.0 / 150.0 and float(g[0]) > 0.0
            check_gradients(lambda w: sl.bucket_beta(w, cap, P),
                            (jnp.asarray([30.0, 75.0, 100.0]),), rtol=1e-5)

    def test_snow_switch(self):
        with float64():
            s = jnp.asarray([0.0, 1e-3, 30.0, 0.0])
            g = jnp.asarray([0.0, 0.0, 0.0, 1.0])
            np.testing.assert_array_equal(sl.snow_indicator(s, g, P), [0.0, 1.0, 1.0, 1.0])
            d = jax.grad(lambda s: jnp.sum(sl.snow_indicator(s, g, P)))(s)
            assert float(d[0]) > 0.1 and float(d[3]) == 0.0

    def test_cap_limits_the_step_to_the_water_present(self):
        with float64():
            f = sl.cap_evaporation_factor(jnp.asarray([0.5, 0.5, 0.5]),
                                          jnp.asarray([1e-4, 1e-4, -1e-4]),
                                          jnp.asarray([0.01, 100.0, 0.0]), DT)
            np.testing.assert_allclose(f[0] * 1e-4 * DT, 0.01)
            np.testing.assert_allclose(f[1:], 0.5)                     # enough water; dew


class TestMelt:

    def test_melt_returns_the_surface_to_the_freezing_point_within_the_snow(self):
        with float64():
            den = jnp.asarray([80.0, 80.0, 80.0, 80.0])
            t = c.tmelt + jnp.asarray([2.0, 2.0, -1.0, 2.0])
            snow = jnp.asarray([100.0, 0.05, 100.0, 0.0])
            glac = jnp.asarray([0.0, 0.0, 0.0, 1.0])
            m = sl.melt_energy(t, den, snow, glac, DT, P)
            np.testing.assert_allclose(m[0], 2.0 * 80.0)                     # enough snow
            np.testing.assert_allclose(m[1], 0.05 * c.alhf / DT)            # limited by S
            assert float(m[2]) == 0.0                                        # below T_f
            np.testing.assert_allclose(m[3], 2.0 * 80.0)                     # glacier: no limit
            check_gradients(lambda t, d, s: sl.melt_energy(t, d, s, glac[:2], DT, P),
                            (t[:2] + 1.0, den[:2], jnp.asarray([300.0, 0.1])), rtol=2e-2)


class TestBuckets:

    def _run(self, **kw):
        n = 1
        args = dict(water=100.0, snow=0.0, rain=0.0, snowfall=0.0, melt=0.0,
                    evaporation=0.0, capacity=200.0, glacier=0.0)
        args.update(kw)
        arrs = {k: jnp.full(n, float(v)) for k, v in args.items()}
        out = sl.update_buckets(arrs["water"], arrs["snow"], arrs["rain"], arrs["snowfall"],
                                arrs["melt"], arrs["evaporation"], arrs["capacity"],
                                arrs["glacier"], DT, P)
        budget = (out["water"] + out["snow"] - arrs["water"] - arrs["snow"]
                  - DT * (arrs["rain"] + arrs["snowfall"] - arrs["evaporation"]
                          - out["runoff"] + out["glacier_supply"]))
        np.testing.assert_allclose(budget, 0.0, atol=1e-9)
        return {k: float(v[0]) for k, v in out.items()}

    @pytest.mark.parametrize("case", [
        dict(rain=1e-3), dict(evaporation=2e-5, snow=1.0), dict(evaporation=-1e-5, snow=5.0),
        dict(melt=1e-4, snow=50.0, snowfall=1e-4), dict(glacier=1.0, evaporation=1e-5),
        dict(glacier=1.0, melt=1e-2, snow=1.0), dict(snow=4999.0, snowfall=1e-2),
    ])
    def test_water_budget_closes(self, case):
        with float64():
            self._run(**case)

    def test_saturated_bucket_runs_off(self):
        with float64():
            out = self._run(water=190.0, rain=1e-3)                 # +0.72 kg/m2 ... no overflow
            assert out["runoff"] == 0.0
            out = self._run(water=199.9, rain=1e-3)
            assert out["water"] == pytest.approx(200.0)
            assert out["runoff"] == pytest.approx((199.9 + 0.72 - 200.0) / DT)
            out = self._run(snow=4999.9, snowfall=1e-3)
            assert out["snow"] == pytest.approx(5000.0)

    def test_evaporation_comes_from_the_snow_first(self):
        with float64():
            out = self._run(water=10.0, snow=0.005, evaporation=1e-5)   # 7.2e-3 kg/m2
            assert out["snow"] == pytest.approx(0.0, abs=1e-12)
            assert out["water"] == pytest.approx(10.0 - (7.2e-3 - 0.005))
            out = self._run(glacier=1.0, water=10.0, evaporation=1e-5)
            assert out["water"] == 10.0 and out["glacier_supply"] > 0.0


class TestAlbedo:

    def test_snow_masked_blend(self):
        props = {k: jnp.asarray(v) for k, v in dict(
            alb_gvd=0.1, alb_svd=0.9, alb_gnd=0.3, alb_snd=0.6, alb_gvf=0.12, alb_svf=0.92,
            alb_gnf=0.32, alb_snf=0.62, snowmask=50.0).items()}
        vd, nd, vf, nf = sl.land_albedos(props, jnp.asarray(50.0))
        np.testing.assert_allclose([vd, nd, vf, nf], [0.5, 0.45, 0.52, 0.47], rtol=1e-6)
        vd0, *_ = sl.land_albedos(props, jnp.asarray(0.0))
        np.testing.assert_allclose(vd0, 0.1)
