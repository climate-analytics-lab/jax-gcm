"""Tests for the SPEEDY bulk surface fluxes.

The reference values are the fmask-weighted land/sea means the scheme
publishes, quoted as ``[max, min, mean]`` over the grid because the land
skin temperature varies with latitude through the daily-cycle term.
"""
import functools
import unittest

import jax
import jax.numpy as jnp
from jax.test_util import check_vjp, check_jvp

from jcm.constants import grav
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import (
    ConvectionData, HumidityData, LWRadiationData, PhysicsData,
    SurfaceFluxData, SWRadiationData,
)
from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords
from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
from jcm.physics.surface.speedy_surface_flux import (
    get_orog_land_sfc_drag, get_surface_fluxes)
from jcm.physics_interface import PhysicsState, PhysicsTendency

IX, IL, KX = 96, 48, 8
XY, ZXY = (IX, IL), (KX, IX, IL)


def build_inputs(
    *, ta=288.0, qa=5.0, rh=0.8, phi=5000.0, phi0=500.0, fmask=0.5, psa=1.0,
    ua=1.0, va=1.0, sst=290.0, rsds=400.0, rlds=400.0, stl_am=288.0,
    soilw_am=0.5, ones_forcing=False, aquaplanet=False, geopotential=None,
):
    """Assemble the five ``get_surface_fluxes`` arguments for a uniform column.

    ``aquaplanet`` selects ``TerrainData.aquaplanet`` (fmask = 0,
    ``lfluxland`` False) and drops the land forcing fields, so the sea
    branch has to produce the whole flux on its own.
    """
    from jcm.terrain import TerrainData

    coords = get_speedy_coords(layers=KX, nodal_shape=XY)
    speedy_coords = SpeedyCoords.from_coordinate_system(coords)

    phi_field = geopotential if geopotential is not None else phi * jnp.ones(ZXY)
    state = PhysicsState.zeros(
        ZXY, ua * jnp.ones(ZXY), va * jnp.ones(ZXY), ta * jnp.ones(ZXY),
        qa * jnp.ones(ZXY), phi_field, psa * jnp.ones(XY))

    if aquaplanet:
        terrain = TerrainData.aquaplanet(coords)
    else:
        terrain = TerrainData.from_coords(
            coords, orography=phi0 * jnp.ones(XY) / grav,
            fmask=fmask * jnp.ones(XY), lfluxland=True)
    terrain, speedy_coords = convert_to_speedy_latitudes(terrain, speedy_coords)

    physics_data = PhysicsData.zeros(
        XY, KX,
        convection=ConvectionData.zeros(XY, KX),
        humidity=HumidityData.zeros(XY, KX, rh=rh * jnp.ones(ZXY)),
        surface_flux=SurfaceFluxData.zeros(XY, rlds=rlds * jnp.ones(XY)),
        shortwave_rad=SWRadiationData.zeros(XY, KX, rsds=rsds * jnp.ones(XY)),
        longwave_rad=LWRadiationData.zeros(XY, KX),
        speedy_coords=speedy_coords,
    )

    forcing_kwargs = dict(sea_surface_temperature=sst * jnp.ones(XY))
    if not aquaplanet:
        forcing_kwargs.update(soilw_am=soilw_am * jnp.ones(XY),
                              stl_am=stl_am * jnp.ones(XY))
    forcing_cls = ForcingData.ones if ones_forcing else ForcingData.zeros

    return dict(state=state, physics_data=physics_data,
                parameters=Parameters.default(),
                forcing=forcing_cls(XY, **forcing_kwargs), terrain=terrain)


# Reference [max, min, mean] of every published surface-flux field, for the
# five configurations below. Fields are fmask-weighted land/sea means except
# hfluxn_land / hfluxn_sea, which are the per-surface-type components.
_FIELDS = ("ustr", "vstr", "shf", "evap", "rlus", "hfluxn", "hfluxn_land",
           "hfluxn_sea", "tsfc", "tskin", "u0", "v0", "t0")

_CASES = {
    # Warm sea under a cool near-isothermal column, with saturated forcing.
    "warm_sea": (
        dict(ta=290.0, qa=1.0, rh=0.5, phi0=0.0, sst=292.0, ones_forcing=True,
             geopotential=jnp.ones((KX, IX, IL))
             * (jnp.arange(KX))[::-1][:, jnp.newaxis, jnp.newaxis]),
        [[-1.19625032e-02, -1.19625032e-02, -1.19624995e-02],  # ustr
         [-1.19625032e-02, -1.19625032e-02, -1.19624995e-02],  # vstr
         [4.94021873e+01, 4.80357971e+01, 4.87642822e+01],     # shf
         [9.53914225e-02, 8.26347470e-02, 9.13820267e-02],     # evap
         [4.31850861e+02, 4.18786133e+02, 4.22756989e+02],     # rlus
         [1.12463333e+02, 9.46041260e+01, 9.99318924e+01],     # hfluxn
         [1.01222931e+02, 6.55045166e+01, 7.61595764e+01],     # hfluxn_land
         [1.23703735e+02, 1.23703735e+02, 1.23703262e+02],     # hfluxn_sea
         [2.90000000e+02, 2.90000000e+02, 2.90000000e+02],     # tsfc
         [2.97230225e+02, 2.94678894e+02, 2.95440155e+02],     # tskin
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],     # u0
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],     # v0
         [2.90000000e+02, 2.90000000e+02, 2.90000000e+02]],    # t0
    ),
    # Baseline: elevated orography, atmosphere cooler than the sea.
    "elevated": (
        dict(ta=288.0, phi0=500.0),
        [[-9.60592739e-03, -9.60592739e-03, -9.60589107e-03],
         [-9.60592739e-03, -9.60592739e-03, -9.60589107e-03],
         [5.79089394e+01, 4.82901459e+01, 5.54587822e+01],
         [3.71975675e-02, 3.02297361e-02, 3.52667645e-02],
         [4.40432190e+02, 4.29336853e+02, 4.32341431e+02],
         [2.35673111e+02, 2.19723160e+02, 2.23998611e+02],
         [1.37954346e+02, 1.06054443e+02, 1.14604996e+02],
         [3.33391876e+02, 3.33391876e+02, 3.33390808e+02],
         [2.89000000e+02, 2.89000000e+02, 2.89000000e+02],
         [2.98853882e+02, 2.96575317e+02, 2.97186798e+02],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [2.88000000e+02, 2.88000000e+02, 2.88000000e+02]],
    ),
    # Below sea level: exercises the max(phis0, 0) clamp in the drag factor.
    "below_sea_level": (
        dict(ta=288.0, phi0=-10.0),
        [[-9.60591808e-03, -9.60591808e-03, -9.60589014e-03],
         [-9.60591808e-03, -9.60591808e-03, -9.60589014e-03],
         [5.63907928e+01, 4.57414093e+01, 5.36825180e+01],
         [3.65183949e-02, 2.92323455e-02, 3.45011912e-02],
         [4.42618805e+02, 4.30757050e+02, 4.33960571e+02],
         [2.38529633e+02, 2.21519623e+02, 2.26068863e+02],
         [1.43667480e+02, 1.09647369e+02, 1.18746910e+02],
         [3.33391876e+02, 3.33391876e+02, 3.33390808e+02],
         [2.89000000e+02, 2.89000000e+02, 2.89000000e+02],
         [2.99261963e+02, 2.96831970e+02, 2.97482574e+02],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [2.88000000e+02, 2.88000000e+02, 2.88000000e+02]],
    ),
    # Atmosphere warmer than the sea: stable side of the stability correction.
    "stable": (
        dict(ta=300.0, phi0=500.0),
        [[-8.20686668e-03, -8.20686668e-03, -8.20684712e-03],
         [-8.20686668e-03, -8.20686668e-03, -8.20684712e-03],
         [8.05199432e+00, 7.09895515e+00, 7.37413263e+00],
         [1.97063759e-02, 1.81624368e-02, 1.92561075e-02],
         [4.57913269e+02, 4.57794006e+02, 4.57840332e+02],
         [2.88610413e+02, 2.85821365e+02, 2.86627014e+02],
         [1.83628235e+02, 1.78050110e+02, 1.79660461e+02],
         [3.93592621e+02, 3.93592621e+02, 3.93593597e+02],
         [2.89000000e+02, 2.89000000e+02, 2.89000000e+02],
         [3.02116302e+02, 3.01717865e+02, 3.01832672e+02],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [3.00000000e+02, 3.00000000e+02, 3.00000000e+02]],
    ),
    # Pure land (fmask = 1): the merge collapses onto the land branch, so
    # this pins the land bulk formulae and the skin energy balance on their
    # own. Every other case runs fmask = 0.5, where the weighting is
    # symmetric in land and sea and cannot detect a swapped merge order —
    # here tsfc must be stl_am (288) rather than the SST (290).
    "pure_land": (
        dict(ta=288.0, phi0=500.0, fmask=1.0),
        [[-1.50308944e-02, -1.50308944e-02, -1.50308656e-02],  # ustr
         [-1.50308944e-02, -1.50308944e-02, -1.50308656e-02],  # vstr
         [1.08257225e+02, 8.90196381e+01, 1.03357094e+02],     # shf
         [4.79898155e-02, 3.40541564e-02, 4.41281646e-02],     # evap
         [4.87856598e+02, 4.65665894e+02, 4.71672852e+02],     # rlus
         [1.37954346e+02, 1.06054443e+02, 1.14604996e+02],     # hfluxn
         [1.37954346e+02, 1.06054443e+02, 1.14604996e+02],     # hfluxn_land
         [3.33391876e+02, 3.33391876e+02, 3.33390808e+02],     # hfluxn_sea
         [2.88000000e+02, 2.88000000e+02, 2.88000000e+02],     # tsfc
         [3.07707764e+02, 3.03150635e+02, 3.04372498e+02],     # tskin
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],     # u0
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],     # v0
         [2.88000000e+02, 2.88000000e+02, 2.88000000e+02]],    # t0
    ),
    # Cold atmosphere: strongest unstable exchange of the set.
    "unstable": (
        dict(ta=285.0, phi0=500.0),
        [[-1.07752765e-02, -1.07752765e-02, -1.07752131e-02],
         [-1.07752765e-02, -1.07752765e-02, -1.07752131e-02],
         [9.26892319e+01, 7.85854797e+01, 8.86138916e+01],
         [4.69812490e-02, 4.07872051e-02, 4.52995971e-02],
         [4.27910583e+02, 4.15460693e+02, 4.18975739e+02],
         [1.91495117e+02, 1.74349945e+02, 1.79115692e+02],
         [1.05238403e+02, 7.09480591e+01, 8.04812698e+01],
         [2.77751831e+02, 2.77751831e+02, 2.77750305e+02],
         [2.89000000e+02, 2.89000000e+02, 2.89000000e+02],
         [2.96517029e+02, 2.94067719e+02, 2.94748505e+02],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [9.49999988e-01, 9.49999988e-01, 9.50007141e-01],
         [2.85000000e+02, 2.85000000e+02, 2.85000000e+02]],
    ),
}


class TestSurfaceFluxesUnit(unittest.TestCase):

    def test_regression_against_reference_fluxes(self):
        """Every published field, in five configurations.

        Tolerances: rtol 2e-5 / atol 0.1 covers the cp/rd unification to the
        high-precision ECHAM values (jcm/constants.py), which moves the heat
        fluxes by <0.06 (~1e-4 relative).
        """
        for name, (kwargs, expected) in _CASES.items():
            with self.subTest(case=name):
                _, physics_data = get_surface_fluxes(**build_inputs(**kwargs))
                sflux = physics_data.surface_flux
                actual = jnp.array([
                    [jnp.max(v), jnp.min(v), jnp.mean(v)]
                    for v in (getattr(sflux, f) for f in _FIELDS)])
                self.assertTrue(
                    jnp.allclose(actual, jnp.array(expected), rtol=2e-5, atol=0.1),
                    f"{name}: {actual} != {jnp.array(expected)}")

    def test_all_fields_are_grid_maps(self):
        """Every published field is a 2D map on the nodal grid.

        The scheme resolves land and sea internally and publishes only grid
        means (plus the two named hfluxn components), so no field carries a
        surface-type axis for a consumer to index into.
        """
        _, physics_data = get_surface_fluxes(**build_inputs())
        for name, value in vars(physics_data.surface_flux).items():
            self.assertEqual(value.shape, XY, f"{name} is not a 2D map")

    @unittest.skipUnless(jax.config.read("jax_enable_x64"),
                         "needs x64 to build a forcing that differs in precision "
                         "from the state")
    def test_land_branch_tolerates_higher_precision_forcing(self):
        """Forcing may arrive at a different precision from the state.

        Coupled drivers run in float64 and promote the forcing they hand
        back, while the model state stays float32. The land fluxes inherit
        the forcing dtype, so the zero arm of the land branch has to inherit
        it too or ``lax.cond`` rejects the pair outright.
        """
        args = build_inputs()
        to_f32 = lambda tree: jax.tree.map(
            lambda leaf: jnp.asarray(leaf, jnp.float32)
            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, tree)
        for key in ("state", "physics_data", "terrain"):
            args[key] = to_f32(args[key])
        args["forcing"] = jax.tree.map(
            lambda leaf: jnp.asarray(leaf, jnp.float64)
            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf,
            args["forcing"])
        self.assertEqual(args["forcing"].stl_am.dtype, jnp.float64)
        self.assertEqual(args["state"].temperature.dtype, jnp.float32)

        _, physics_data = get_surface_fluxes(**args)
        self.assertTrue(jnp.all(jnp.isfinite(physics_data.surface_flux.hfluxn)))

    def test_merged_hfluxn_is_the_area_weighted_mean(self):
        """The published hfluxn is the fmask weighting of its two components."""
        args = build_inputs(fmask=0.3)
        fmask = args["terrain"].fmask
        _, physics_data = get_surface_fluxes(**args)
        sflux = physics_data.surface_flux
        expected = sflux.hfluxn_sea + fmask * (sflux.hfluxn_land - sflux.hfluxn_sea)
        self.assertTrue(jnp.allclose(sflux.hfluxn, expected, rtol=1e-6))
        # The components must genuinely differ, or the check is vacuous.
        self.assertFalse(jnp.allclose(sflux.hfluxn_land, sflux.hfluxn_sea))

    def test_tendencies_use_the_merged_fluxes(self):
        """Lowest-level tendencies scale with the merged flux, not a component."""
        args = build_inputs()
        tendencies, physics_data = get_surface_fluxes(**args)
        sflux = physics_data.surface_flux
        speedy_coords = args["physics_data"].speedy_coords
        rps = 1.0 / args["state"].normalized_surface_pressure
        self.assertTrue(jnp.allclose(
            tendencies.u_wind[-1], sflux.ustr * rps * speedy_coords.grdsig[-1]))
        self.assertTrue(jnp.allclose(
            tendencies.temperature[-1], sflux.shf * rps * speedy_coords.grdscp[-1]))

    def test_grad_surface_flux(self):
        args = build_inputs()
        _, f_vjp = jax.vjp(
            get_surface_fluxes, args["state"], args["physics_data"],
            args["parameters"], args["forcing"], args["terrain"])

        cotangent = (
            PhysicsTendency.ones(ZXY),
            PhysicsData.ones(XY, KX,
                             speedy_coords=args["physics_data"].speedy_coords))
        df_dstate, df_ddatas, df_dparams, df_dforcing, _ = f_vjp(cotangent)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

    def test_surface_fluxes_drag_test(self):
        phi0 = 500. * jnp.ones(XY)
        forog = get_orog_land_sfc_drag(
            phi0, Parameters.default().surface_flux.hdrag)
        self.assertAlmostEqual(jnp.max(forog), 1.0000012824780082)
        self.assertAlmostEqual(jnp.min(forog), 1.0000012824780082)

    def test_surface_fluxes_gradient_check_test1(self):
        from jcm.utils import convert_back, convert_to_float

        args = build_inputs()
        state, physics_data = args["state"], args["physics_data"]
        parameters, forcing, terrain = (
            args["parameters"], args["forcing"], args["terrain"])

        def f(state_f, physics_data_f, parameters_f, forcing_f, terrain_f):
            _, data_out = get_surface_fluxes(
                state=convert_back(state_f, state),
                physics_data=convert_back(physics_data_f, physics_data),
                parameters=convert_back(parameters_f, parameters),
                forcing=convert_back(forcing_f, forcing),
                terrain=convert_back(terrain_f, terrain))
            return convert_to_float(data_out.surface_flux)

        float_args = tuple(convert_to_float(x) for x in
                           (state, physics_data, parameters, forcing, terrain))
        check_vjp(f, functools.partial(jax.vjp, f), args=float_args,
                  atol=None, rtol=1, eps=0.00001)
        check_jvp(f, functools.partial(jax.jvp, f), args=float_args,
                  atol=None, rtol=1, eps=0.000001)

    def test_surface_fluxes_drag_test_gradient_check(self):
        phi0 = 500. * jnp.ones(XY)
        hdrag = Parameters.default().surface_flux.hdrag
        check_vjp(get_orog_land_sfc_drag,
                  functools.partial(jax.vjp, get_orog_land_sfc_drag),
                  args=(phi0, hdrag), atol=None, rtol=1, eps=0.00001)
        check_jvp(get_orog_land_sfc_drag,
                  functools.partial(jax.jvp, get_orog_land_sfc_drag),
                  args=(phi0, hdrag), atol=None, rtol=1, eps=0.000001)


class TestAquaplanetSurfaceFluxes(unittest.TestCase):
    """Aquaplanet configuration (``lfluxland`` False, fmask = 0).

    With no land fraction the merged flux is exactly the sea flux, so these
    pin the sea branch in isolation — including that it still runs when the
    land branch is switched off.
    """

    def _run(self, **kwargs):
        args = build_inputs(aquaplanet=True, **kwargs)
        tendencies, physics_data = get_surface_fluxes(**args)
        return tendencies, physics_data.surface_flux, args

    def test_aquaplanet_ocean_evaporation_nonzero(self):
        tendencies, sflux, _ = self._run(ta=280.0, rh=0.7, ua=5.0, va=2.0,
                                         sst=300.0, rlds=350.0)
        self.assertFalse(jnp.any(jnp.isnan(sflux.evap)), "Evaporation contains NaNs")
        self.assertFalse(jnp.any(jnp.isnan(sflux.shf)), "Sensible heat flux contains NaNs")
        self.assertFalse(jnp.any(jnp.isnan(sflux.ustr)), "Wind stress contains NaNs")
        self.assertFalse(jnp.any(jnp.isnan(tendencies.specific_humidity)),
                         "Humidity tendency contains NaNs")

        self.assertTrue(jnp.all(sflux.evap > 0),
                        "Ocean evaporation should be positive with warm SST")
        self.assertTrue(jnp.all(tendencies.specific_humidity[-1] > 0),
                        "Humidity tendency should be positive from ocean evaporation")

    def test_aquaplanet_merged_flux_is_the_sea_flux(self):
        _, sflux, _ = self._run(ta=280.0, rh=0.7, ua=5.0, va=2.0, sst=300.0,
                                rlds=350.0)
        self.assertTrue(jnp.allclose(sflux.hfluxn, sflux.hfluxn_sea))
        # No land branch ran, so its component carries no flux.
        self.assertTrue(jnp.all(sflux.hfluxn_land == 0.0))

    def test_aquaplanet_sensible_heat_flux(self):
        tendencies, sflux, _ = self._run(ta=280.0, rh=0.7, ua=5.0, va=2.0,
                                         sst=300.0, rlds=350.0)
        self.assertTrue(jnp.all(sflux.shf > 0),
                        "Ocean sensible heat flux should be positive with warm SST")
        self.assertTrue(jnp.all(tendencies.temperature[-1] > 0),
                        "Temperature tendency should be positive from warm ocean")

    def test_aquaplanet_gradient_check(self):
        _, _, args = self._run(ta=290.0, rh=0.7, ua=5.0, va=2.0, sst=295.0,
                               rlds=350.0)
        _, f_vjp = jax.vjp(
            get_surface_fluxes, args["state"], args["physics_data"],
            args["parameters"], args["forcing"], args["terrain"])
        cotangent = (
            PhysicsTendency.ones(ZXY),
            PhysicsData.ones(XY, KX,
                             speedy_coords=args["physics_data"].speedy_coords))
        df_dstate, df_ddatas, df_dparams, df_dforcing, _ = f_vjp(cotangent)

        self.assertFalse(df_ddatas.isnan().any_true(), "Gradient w.r.t. physics_data contains NaNs")
        self.assertFalse(df_dstate.isnan().any_true(), "Gradient w.r.t. state contains NaNs")
        self.assertFalse(df_dparams.isnan().any_true(), "Gradient w.r.t. parameters contains NaNs")
        self.assertFalse(df_dforcing.isnan().any_true(), "Gradient w.r.t. forcing contains NaNs")
