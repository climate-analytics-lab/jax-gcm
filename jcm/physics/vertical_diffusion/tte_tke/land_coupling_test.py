"""ECHAM's per-tile surface coupling and the land skin energy balance (#979).

Heat and moisture couple to the surface tile by tile (``richtmyer_land``,
``_ocean``, ``_ice``, then ``blend_zq_zt``); the land tile's skin temperature
is solved with the lowest level (``update_surfacetemp``). The coefficients
themselves are pinned against the compiled Fortran in
``jcm/physics/surface/echam/jsbach_land_test.py``; this module checks what
the port builds from them: the column budget, the land budget, the
evaporation regimes and the derivatives.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.surface.echam import jsbach_land as jl
from jcm.testing import check_gradients
from .matrix_solver import (
    setup_matrix_system, solve_tridiagonal_single, vertical_diffusion_step,
)
from .vertical_diffusion import vertical_diffusion_column
from .vertical_diffusion_test import _make_marine_bl_state
from .vertical_diffusion_types import LandBalanceInputs, SurfaceTiles, VDiffParameters

DT = 720.0


@contextlib.contextmanager
def float64():
    with jax.enable_x64(True):
        yield


def _land_column(ncol=1, t_skin=305.0, rh=0.3, wind=4.0, land=1.0, water=0.0, ice=0.0):
    """Build a boundary-layer column over (mostly) land, the skin at ``t_skin``."""
    state = _make_marine_bl_state(ncol=ncol, sst_offset=0.0, wind=wind, rh=rh)
    frac = jnp.tile(jnp.asarray([[water, ice, land]], state.temperature.dtype), (ncol, 1))
    t_sfc = jnp.stack([state.surface_temperature[:, 0],
                       jnp.minimum(state.surface_temperature[:, 0], 271.38),
                       jnp.full(ncol, t_skin, state.temperature.dtype)], axis=1)
    return state._replace(
        surface_fraction=frac, surface_temperature=t_sfc,
        roughness_length=jnp.tile(jnp.asarray([[1e-4, 1e-3, 0.05]]), (ncol, 1)),
        roughness_heat=jnp.tile(jnp.asarray([[4.9e-5, 1e-3, 0.05]]), (ncol, 1)),
        surface_sublimation_fraction=jnp.zeros((ncol, 3)).at[:, 1].set(1.0),
    )


def _with_factors(state, *, w, h, veg=0.3, gc=0.015, chu=0.02):
    """Give the land tile JSBACH's factors for root/upper fill ``w``/``h``."""
    n = state.temperature.shape[0]
    qs = jnp.asarray(_qsat(state.surface_temperature[:, 2], state.pressure_half[:, -1]))
    cair, csat, _ = jl.humidity_factors(
        jnp.full(n, h), jnp.full(n, w), jnp.zeros(n), jnp.zeros(n), jnp.full(n, veg),
        jnp.full(n, gc), jnp.full(n, chu), state.qv[:, -1], qs, jl.JsbachLandParameters())
    ones = jnp.ones(n)
    return state._replace(surface_cair=jnp.stack([ones, ones, cair], 1),
                          surface_csat=jnp.stack([ones, ones, csat], 1),
                          surface_wetness=jnp.stack([ones, ones, csat], 1)), cair, csat


def _qsat(t, p):
    from jcm.physics.thermodynamics import saturation_specific_humidity
    return saturation_specific_humidity(t, p)


def _land_inputs(state, *, sw=600.0, lw=350.0, t_soil=300.0, snow=0.0, glacier=0.0, capped=False):
    n = state.temperature.shape[0]
    p = jl.JsbachLandParameters()
    cap, lam = jl.top_layer_thermal_properties(jnp.full(n, snow), jnp.full(n, glacier), p)
    return LandBalanceInputs(
        temperature=state.surface_temperature[:, 2], soil_temperature=jnp.full(n, t_soil),
        saturation_slope=jnp.zeros(n), net_shortwave=jnp.full(n, sw),
        longwave_down=jnp.full(n, lw), emissivity=jnp.full(n, 0.95),
        heat_capacity=cap, conductance=lam, melt_capped=jnp.full(n, capped), params=p)


def _with_snow(state, snow):
    """Set the land's sublimating share, the snow cover the balance charges with als."""
    return state._replace(surface_sublimation_fraction=(
        state.surface_sublimation_fraction.at[:, 2].set(snow)))


def _column_integrals(state, tend):
    dm = np.asarray(state.air_mass)
    return (np.sum(dm * np.asarray(tend.qv_tendency), axis=1),
            c.cpd * np.sum(dm * np.asarray(tend.temperature_tendency), axis=1))


class TestPerTileCoupling:

    def _tiles(self, state, frac, c_h=(0.03, 0.01, 0.02)):
        n = state.temperature.shape[0]
        ch = jnp.tile(jnp.asarray([c_h]), (n, 1))
        t_s = jnp.tile(jnp.asarray([[300.0, 271.38, 295.0]]), (n, 1))
        return SurfaceTiles(
            fraction=jnp.tile(jnp.asarray([frac]), (n, 1)), exchange_heat=ch,
            exchange_moisture=ch, cair=jnp.ones((n, 3)), csat=jnp.ones((n, 3)),
            temperature=t_s, saturation_humidity=_qsat(t_s, state.pressure_half[:, -1:]),
            sublimation_fraction=jnp.zeros((n, 3)).at[:, 1].set(1.0))

    def test_a_single_tile_is_the_robin_row(self):
        """With one tile, the Richtmyer–Morton elimination IS the Robin row.

        Re-derives the pre-#979 collapsed coupling for the cells where the two
        agree: ``(R + tp2·k·X_s)/(D + k)`` is the bottom row of the matrix with
        ``k`` on the diagonal and ``tp2·k·X_s`` on the right-hand side, so a
        pure-ocean (or pure-land, prescribed) column is unchanged to round-off.
        """
        with float64():
            state = _make_marine_bl_state(ncol=2, sst_offset=4.0, wind=8.0)
            params = VDiffParameters.default()
            kk = jnp.full(state.u.shape, 6.0)
            tiles = self._tiles(state, (1.0, 0.0, 0.0))
            tend, fluxes, _ = vertical_diffusion_step(
                state, params, kk, kk, kk, DT, kk, surface_tiles=tiles)

            ms = setup_matrix_system(state, params, kk, kk, kk, DT, kk)
            rho = state.pressure_half[:, -1] / (c.rd * state.temperature[:, -1])
            k = DT * params.tpfac1 * rho * tiles.exchange_heat[:, 0] / state.air_mass[:, -1]
            phi = c.grav * (state.height_full[:, -1] - state.height_half[:, -1])
            for ivar, imat, target in ((2, 1, tiles.temperature[:, 0] - phi / c.cpd),
                                       (3, 2, tiles.saturation_humidity[:, 0])):
                b = ms.matrix_coeffs[:, :, 1, imat].at[:, -1].add(k)
                d = ms.rhs_vectors[:, :, ivar].at[:, -1].add(params.tpfac2 * k * target)
                bb = solve_tridiagonal_single(ms.matrix_coeffs[:, :, 0, imat], b,
                                              ms.matrix_coeffs[:, :, 2, imat], d)
                x_old = state.temperature if ivar == 2 else state.qv
                robin = (bb + params.tpfac3 * x_old - x_old) / DT
                got = tend.temperature_tendency if ivar == 2 else tend.qv_tendency
                np.testing.assert_allclose(got, robin, rtol=1e-10, atol=1e-16)

    def test_mixed_tiles_close_the_column_budget_against_their_own_fluxes(self):
        """Each tile's flux is taken against its own lowest-level value, and the
        column receives exactly the fraction-weighted sum (ECHAM's ``pev_vdiff``).

        Also measures the gain over the collapsed row for a 60/40 water/ice
        cell: the fluxes differ at the O(k/D) the design page quotes.
        """
        with float64():
            state = _make_marine_bl_state(ncol=1, sst_offset=4.0, wind=8.0)
            params = VDiffParameters.default()
            kk = jnp.full(state.u.shape, 6.0)
            tiles = self._tiles(state, (0.6, 0.4, 0.0))
            tend, fluxes, _ = vertical_diffusion_step(
                state, params, kk, kk, kk, DT, kk, surface_tiles=tiles)
            col_q, col_t = _column_integrals(state, tend)
            np.testing.assert_allclose(col_q, fluxes.evaporation, rtol=1e-10)
            np.testing.assert_allclose(col_t, fluxes.sensible_heat, rtol=1e-10)
            # Collapsed row (k_grid = Σ f k, flux-weighted target): a different
            # bottom value, so a different flux.
            ms = setup_matrix_system(state, params, kk, kk, kk, DT, kk)
            rho = state.pressure_half[:, -1] / (c.rd * state.temperature[:, -1])
            k_t = (DT * params.tpfac1 * rho / state.air_mass[:, -1])[:, None] * tiles.exchange_moisture
            k_g = jnp.sum(tiles.fraction * k_t, axis=1)
            q_g = jnp.sum(tiles.fraction * k_t * tiles.saturation_humidity, axis=1) / k_g
            b = ms.matrix_coeffs[:, :, 1, 2].at[:, -1].add(k_g)
            d = ms.rhs_vectors[:, :, 3].at[:, -1].add(params.tpfac2 * k_g * q_g)
            bb = solve_tridiagonal_single(ms.matrix_coeffs[:, :, 0, 2], b,
                                          ms.matrix_coeffs[:, :, 2, 2], d)
            e_collapsed = float(rho[0] * k_g[0] / k_t[0, 0] * tiles.exchange_moisture[0, 0]
                                * params.tpfac1 * (params.tpfac2 * q_g[0] - bb[0, -1]))
            rel = abs(float(fluxes.evaporation[0]) - e_collapsed) / abs(e_collapsed)
            assert 1e-4 < rel < 0.2, rel


class TestLandEnergyBalance:

    def _run(self, state, land, params=None):
        return vertical_diffusion_column(state, params or VDiffParameters.default(), DT, land=land)

    @pytest.mark.parametrize("w,h", [(0.1, 0.05), (0.6, 0.6), (0.95, 0.99)])
    def test_budget_closes_per_step(self, w, h):
        """Storage = Rn − SH − LH − G to round-off; no melt term without snow.

        ``update_surfacetemp`` solves the balance with the fluxes the column
        then receives (same E, F, same implicit values), so the residual the
        melt diagnostic carries is zero away from a melt cap.
        """
        with float64():
            state, _, _ = _with_factors(_land_column(t_skin=305.0), w=w, h=h)
            tend, diag = self._run(state, _land_inputs(state))
            lb = diag.land_balance
            np.testing.assert_allclose(lb.melt_heat_flux, 0.0, atol=1e-8)
            closure = lb.net_radiation - lb.sensible_heat_flux - lb.latent_heat_flux \
                - lb.ground_heat_flux - lb.heat_storage
            np.testing.assert_allclose(closure, 0.0, atol=1e-8)
            # A pure-land column: the grid flux is the land tile's, and the
            # column receives it exactly.
            col_q, col_t = _column_integrals(state, tend)
            np.testing.assert_allclose(diag.surface_fluxes.sensible_heat, lb.sensible_heat_flux, rtol=1e-12)
            np.testing.assert_allclose(col_t, lb.sensible_heat_flux, rtol=1e-9)
            np.testing.assert_allclose(col_q, lb.evaporation, rtol=1e-9, atol=1e-15)
            assert abs(float(lb.temperature[0]) - 305.0) > 0.05   # the skin moved

    def test_a_dry_hot_column_stops_evaporating_and_heats_while_a_wet_one_cools(self):
        """The #979 mechanism, in one column under the same midday forcing.

        Dry (root zone below wilting, bare soil whose h·q_s < q_a): JSBACH's
        factors vanish, so E = 0 exactly and the absorbed energy leaves as
        sensible heat with a hotter skin. Wet: the soil and the canopy
        evaporate, the skin stays cooler and the Bowen ratio falls.
        """
        with float64():
            dry, cair_d, csat_d = _with_factors(_land_column(t_skin=308.0, rh=0.3), w=0.1, h=0.05)
            wet, cair_w, csat_w = _with_factors(_land_column(t_skin=308.0, rh=0.3), w=0.95, h=0.99)
            assert float(csat_d[0]) == 0.0 and float(cair_d[0]) == 0.0
            _, dd = self._run(dry, _land_inputs(dry, sw=800.0))
            _, dw = self._run(wet, _land_inputs(wet, sw=800.0))
            assert float(dd.land_balance.evaporation[0]) == 0.0
            assert float(dw.land_balance.evaporation[0]) > 5e-5      # > 4 mm/d
            assert float(dd.land_balance.temperature[0]) > float(dw.land_balance.temperature[0]) + 1.0
            assert float(dd.land_balance.sensible_heat_flux[0]) > float(dw.land_balance.sensible_heat_flux[0])

    def test_snow_holds_the_skin_at_the_melting_point(self):
        """A capped (snow-covered) surface under strong sunshine stays at tmelt and
        the excess goes into melt; without the cap it would warm past it.
        """
        with float64():
            state, _, _ = _with_factors(_land_column(t_skin=272.9, rh=0.6), w=0.9, h=0.9)
            state = _with_snow(state, 0.3)
            _, free = self._run(state, _land_inputs(state, sw=900.0, t_soil=272.0, snow=0.3))
            _, cap = self._run(state, _land_inputs(state, sw=900.0, t_soil=272.0, snow=0.3,
                                                   capped=True))
            assert float(free.land_balance.temperature[0]) > c.tmelt + 0.1
            np.testing.assert_allclose(cap.land_balance.temperature, c.tmelt, rtol=1e-12)
            assert float(cap.land_balance.melt_heat_flux[0]) > 10.0
            np.testing.assert_allclose(free.land_balance.melt_heat_flux, 0.0, atol=1e-8)

    def test_gradients_through_the_hinge_and_the_implicit_solve(self):
        """Derivatives of the skin temperature and fluxes w.r.t. the step-start skin,
        the lowest-level humidity and the soil fill.

        * dry column, hinge inactive: the reference derivative of E w.r.t. the
          soil fill is zero; with the surrogate it is finite and live
          (adjoint-consistent);
        * wet column, away from every switch, reference derivatives (widths
          0): AD agrees with a central difference through the whole implicit
          solve and the energy balance.
        """
        def make(w, h, widths):
            p = jl.JsbachLandParameters() if widths else jl.JsbachLandParameters(
                hinge_width=0.0, stress_width=0.0, melt_width=0.0)

            def f(t_skin, q_low, fill):
                base = _land_column(t_skin=308.0, rh=0.3)
                st = base._replace(
                    surface_temperature=base.surface_temperature.at[:, 2].set(t_skin),
                    qv=base.qv.at[:, -1].set(q_low))
                qs = _qsat(st.surface_temperature[:, 2], st.pressure_half[:, -1])
                n = 1
                hh = jl.bare_soil_relative_humidity(fill * h / w)
                cair, csat, _ = jl.humidity_factors(
                    hh, fill, jnp.zeros(n), jnp.zeros(n), jnp.full(n, 0.3), jnp.full(n, 0.015),
                    jnp.full(n, 0.02), q_low, qs, p)
                ones = jnp.ones(n)
                st = st._replace(surface_cair=jnp.stack([ones, ones, cair], 1),
                                 surface_csat=jnp.stack([ones, ones, csat], 1))
                land = _land_inputs(st)._replace(params=p)
                _, d = vertical_diffusion_column(st, VDiffParameters.default(), DT, land=land)
                lb = d.land_balance
                return lb.temperature, lb.evaporation, lb.sensible_heat_flux
            return f

        with float64():
            base = _land_column(t_skin=308.0, rh=0.3)
            q0 = base.qv[:, -1]
            f_dry = make(0.1, 0.05, widths=True)
            check_gradients(f_dry, (jnp.full(1, 308.0), q0, jnp.full(1, 0.1)),
                            reference="adjoint", live_inputs=("[0]", "[1]", "[2]"))
            f_wet = make(0.95, 0.99, widths=False)
            check_gradients(f_wet, (jnp.full(1, 308.0), q0, jnp.full(1, 0.95)), rtol=1e-4)


class TestTerm:
    """``TteTkeVerticalDiffusion`` with the land tile, as the ECHAM package runs it."""

    def _inputs(self, columns):
        """Term inputs for a list of (soil fill, snow cover, glacier, skin) land columns."""
        from types import SimpleNamespace

        from jcm.forcing import ForcingData
        from jcm.physics.radiation import SURFACE_OPTICS_KEY
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics_interface import PhysicsState
        from .vertical_diffusion_types import VerticalDiffusionData

        n = len(columns)
        fill, snow, glac, skin = (jnp.asarray([col[i] for col in columns]) for i in range(4))
        vs = _make_marine_bl_state(ncol=n, sst_offset=0.0, wind=5.0, rh=0.4)
        nlev = vs.u.shape[1]
        tc = lambda a: jnp.asarray(a).T  # noqa: E731
        state = PhysicsState(
            u_wind=tc(vs.u), v_wind=tc(vs.v), temperature=tc(vs.temperature),
            specific_humidity=tc(vs.qv), geopotential=tc(vs.geopotential),
            normalized_surface_pressure=jnp.ones((n,)),
            tracers={"qc": jnp.zeros((nlev, n)), "qi": jnp.zeros((nlev, n))})
        diagnostics = {
            "_dt_seconds": DT,
            "pressure_full": tc(vs.pressure_full), "pressure_half": tc(vs.pressure_half),
            "height_full": tc(vs.height_full), "height_half": tc(vs.height_half),
            "surface": SurfaceData.zeros((n,), nlev).copy(
                roughness_length=jnp.full((n,), 0.05), land_surface_temperature=skin),
            "vertical_diffusion": VerticalDiffusionData.zeros((n,), nlev).copy(
                tke=jnp.full((nlev, n), 1.0),
                surface_exchange_heat=jnp.full((n, 3), 0.02)),
            "radiation": SimpleNamespace(surface_sw_down=jnp.full((n,), 700.0),
                                         surface_lw_down=jnp.full((n,), 380.0)),
            SURFACE_OPTICS_KEY: {"land_albedo": jnp.full((n,), 0.25),
                                 "land_emissivity": jnp.full((n,), 0.95)},
        }
        forcing = ForcingData.zeros((n,)).copy(
            stl_am=jnp.full((n,), 298.0), soilw_am=fill, soilw_rel=fill, snowc_am=snow,
            glacier_fraction=glac, forest_fraction=jnp.full((n,), 0.4),
            sea_surface_temperature=jnp.full((n,), 295.0))
        terrain = SimpleNamespace(fmask=jnp.ones((n,)))
        return state, diagnostics, forcing, terrain

    COLUMNS = [(0.1, 0.0, 0.0, 310.0), (0.9, 0.0, 0.0, 300.0), (0.5, 0.6, 0.2, 272.0)]

    def test_columns_agree_alone_and_in_a_batch(self):
        """The land tile is column-local: a 3-column batch equals each column alone."""
        from .vertical_diffusion import TteTkeVerticalDiffusion

        term = TteTkeVerticalDiffusion()
        _, batch = term(*self._inputs(self.COLUMNS))
        for i, col in enumerate(self.COLUMNS):
            _, one = term(*self._inputs([col]))
            for field in ("land_surface_temperature", "land_latent_heat_flux",
                          "ground_heat_flux", "cair", "csat", "water_stress_factor"):
                np.testing.assert_allclose(getattr(one["surface"], field)[0],
                                           getattr(batch["surface"], field)[i],
                                           rtol=2e-5, atol=1e-6, err_msg=field)

    def test_term_publishes_a_closed_land_budget(self):
        """Every published land field is filled, and the budget closes from output alone."""
        from .vertical_diffusion import TteTkeVerticalDiffusion

        _, out = TteTkeVerticalDiffusion()(*self._inputs(self.COLUMNS))
        sf = out["surface"]
        residual = (sf.land_net_radiation - sf.land_sensible_heat_flux - sf.land_latent_heat_flux
                    - sf.ground_heat_flux - sf.snow_melt_heat_flux - sf.land_heat_storage)
        np.testing.assert_allclose(residual, 0.0, atol=1e-2)   # float32 W/m2
        # dry hot: no evaporation; wet: evaporates; snowy/glacier: capped at tmelt
        assert float(sf.land_evaporation[0]) == 0.0
        assert float(sf.land_evaporation[1]) > 1e-5
        assert float(sf.land_surface_temperature[2]) <= c.tmelt + 1e-3
        assert float(sf.snow_melt_heat_flux[2]) > 0.0
        assert np.all(np.asarray(sf.canopy_conductance) > 0.0)


class TestPrescribedLandTemperature:
    """``land_temperature="prescribed"``: the fixed-land-temperature configuration."""

    def _term(self, mode="prescribed"):
        from jcm.physics.surface.echam.jsbach_land import JsbachLandParameters
        from .vertical_diffusion import TteTkeVerticalDiffusion
        return TteTkeVerticalDiffusion(land_params=JsbachLandParameters(land_temperature=mode))

    def test_holds_the_forcing_temperature_and_publishes_an_open_budget(self):
        """Over several steps the skin is the forcing's land temperature exactly; the
        humidity factors are the prognostic tile's at the same skin; the budget
        diagnostics stay finite, with the residual ``Rn − SH − LH`` and no
        ground, melt or storage term.
        """
        inputs = TestTerm()._inputs(TestTerm.COLUMNS)
        state, diagnostics, forcing, terrain = inputs
        forcing = forcing.copy(stl_am=jnp.asarray([303.0, 297.0, 268.0]))
        term = self._term()
        for _ in range(3):
            _, out = term(state, diagnostics, forcing, terrain)
            sf = out["surface"]
            np.testing.assert_array_equal(np.asarray(sf.land_surface_temperature),
                                          np.asarray(forcing.stl_am))
            for name in ("land_net_radiation", "land_sensible_heat_flux",
                         "land_latent_heat_flux", "land_energy_residual", "land_evaporation"):
                assert np.all(np.isfinite(np.asarray(getattr(sf, name)))), name
            np.testing.assert_allclose(
                sf.land_energy_residual,
                sf.land_net_radiation - sf.land_sensible_heat_flux - sf.land_latent_heat_flux,
                atol=1e-3)
            for name in ("ground_heat_flux", "snow_melt_heat_flux", "land_heat_storage"):
                np.testing.assert_array_equal(np.asarray(getattr(sf, name)), 0.0)
            diagnostics = {**diagnostics, "surface": sf,
                           "vertical_diffusion": out["vertical_diffusion"]}
        # The same evaporation form: a prognostic tile carrying the same skin
        # forms the same factors.
        state0, diag0, forcing0, terrain0 = inputs
        diag0 = {**diag0, "surface": diag0["surface"].copy(
            land_surface_temperature=jnp.asarray([303.0, 297.0, 268.0]))}
        forcing0 = forcing0.copy(stl_am=jnp.asarray([303.0, 297.0, 268.0]))
        _, prog = self._term("prognostic")(state0, diag0, forcing0, terrain0)
        _, pres = term(state0, diag0, forcing0, terrain0)
        for name in ("cair", "csat", "water_stress_factor", "canopy_conductance"):
            np.testing.assert_array_equal(np.asarray(getattr(pres["surface"], name)),
                                          np.asarray(getattr(prog["surface"], name)), err_msg=name)
        # ... and with a prognostic skin the residual is the closed budget's G + melt + storage.
        sp = prog["surface"]
        np.testing.assert_allclose(
            sp.land_energy_residual,
            sp.ground_heat_flux + sp.snow_melt_heat_flux + sp.land_heat_storage, atol=1e-2)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="land_temperature"):
            self._term("fixed")

    def test_the_config_override_reaches_the_term(self):
        """The term-list override path (``physics.terms.<name>.land_params``)."""
        from jcm.runners import _build_term
        term = _build_term("tte_tke_vertical_diffusion", {
            "_target_": "jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion."
                        "TteTkeVerticalDiffusion",
            "land_params": {"land_temperature": "prescribed"}})
        assert term.land_params.get_value().land_temperature == "prescribed"
