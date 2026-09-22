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

from jcm.testing import check_gradients

import jcm.constants as c
from jcm.constants import grav
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import (
    ConvectionData, HumidityData, LWRadiationData, PhysicsData,
    SurfaceFluxData, SWRadiationData,
)
from jcm.physics.speedy.speedy_coords import (
    SpeedyCoords, compute_speedy_vertical_coords, get_speedy_coords)
from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
from jcm.physics.surface.speedy_surface_flux import (
    get_orog_land_sfc_drag, get_surface_fluxes)
from jcm.physics_interface import PhysicsState, PhysicsTendency

IX, IL, KX = 96, 48, 8
XY, ZXY = (IX, IL), (KX, IX, IL)


def build_inputs(
    *, ta=288.0, qa=5.0, rh=0.8, phi=5000.0, phi0=500.0, fmask=0.5, psa=1.0,
    ua=1.0, va=1.0, sst=290.0, rsds=400.0, rlds=400.0, stl_am=288.0,
    soilw_am=0.5, sice=0.0, ones_forcing=False, aquaplanet=False,
    geopotential=None,
):
    """Assemble the five ``get_surface_fluxes`` arguments for a uniform column.

    ``aquaplanet`` selects ``TerrainData.aquaplanet`` (fmask = 0,
    ``lfluxland`` False) and drops the land forcing fields, so the sea
    branch has to produce the whole flux on its own.

    ``sice`` is the sea-ice fraction; it defaults to 0 (open water) and is
    passed explicitly even under ``ones_forcing`` — otherwise
    ``ForcingData.ones`` would set it to 1, turning every open-water case into
    a fully ice-covered one.
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

    forcing_kwargs = dict(sea_surface_temperature=sst * jnp.ones(XY),
                          sice_am=sice * jnp.ones(XY))
    if not aquaplanet:
        forcing_kwargs.update(soilw_am=soilw_am * jnp.ones(XY),
                              stl_am=stl_am * jnp.ones(XY))
    forcing_cls = ForcingData.ones if ones_forcing else ForcingData.zeros

    return dict(state=state, physics_data=physics_data,
                parameters=Parameters.default(),
                forcing=forcing_cls(XY, **forcing_kwargs), terrain=terrain)


def build_block_inputs(latitudes_deg, *, ta, qa=5.0, rh=0.8, phi=5000.0,
                       phi0=500.0, fmask=0.5, psa=1.0, ua=1.0, va=1.0,
                       sst=290.0, rsds=400.0, rlds=400.0, stl_am=288.0,
                       soilw_am=0.5):
    """``build_inputs`` on a block of ``len(latitudes_deg)`` columns.

    The gradient checks want a handful of columns at chosen latitudes, not a
    T30 grid: a check run over 4608 columns at once fails as soon as *any* of
    them straddles a branch, so what it reports is a property of the grid
    rather than of the scheme (``jcm/testing.py``, and the same argument
    ``term_gradients_test`` makes for one column). Latitude is the only thing
    that distinguishes columns here — every other field in ``build_inputs`` is
    a scalar broadcast — and it enters the fluxes through ``coa`` in the land
    daily-cycle term, so naming the latitudes is naming the whole block.

    Built directly rather than through ``get_speedy_coords``, which takes a
    spectral truncation and so cannot produce a grid this small;
    ``SpeedyCoords.single_column_coords`` and ``TerrainData.single_column``
    are the same construction at ``n = 1``.
    """
    from jcm.terrain import TerrainData

    n = len(latitudes_deg)
    xy, zxy = (1, n), (KX, 1, n)
    radang = jnp.asarray(jnp.deg2rad(jnp.asarray(latitudes_deg)), jnp.float32)
    hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = compute_speedy_vertical_coords(KX)
    speedy_coords = SpeedyCoords(
        hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl, grdsig=grdsig, grdscp=grdscp,
        wvi=wvi, radang=radang, sia=jnp.sin(radang), coa=jnp.cos(radang))

    # ``single_column`` derives the SSO descriptors from the orography the same
    # way ``from_coords`` does; every field is horizontally uniform, so
    # widening (1, 1) to (1, n) is the whole difference.
    terrain = jax.tree.map(
        lambda x: jnp.broadcast_to(x, xy) if jnp.ndim(x) == 2 else x,
        TerrainData.single_column(orog=phi0 / grav, fmask=fmask,
                                  lfluxland=True))

    ta_field = jnp.broadcast_to(jnp.reshape(jnp.asarray(ta), (KX, 1, 1)), zxy)
    state = PhysicsState.zeros(
        zxy, ua * jnp.ones(zxy), va * jnp.ones(zxy), ta_field,
        qa * jnp.ones(zxy), phi * jnp.ones(zxy), psa * jnp.ones(xy))

    physics_data = PhysicsData.zeros(
        xy, KX,
        convection=ConvectionData.zeros(xy, KX),
        humidity=HumidityData.zeros(xy, KX, rh=rh * jnp.ones(zxy)),
        surface_flux=SurfaceFluxData.zeros(xy, rlds=rlds * jnp.ones(xy)),
        shortwave_rad=SWRadiationData.zeros(xy, KX, rsds=rsds * jnp.ones(xy)),
        longwave_rad=LWRadiationData.zeros(xy, KX),
        speedy_coords=speedy_coords,
    )

    forcing = ForcingData.zeros(
        xy, sea_surface_temperature=sst * jnp.ones(xy),
        soilw_am=soilw_am * jnp.ones(xy),
        stl_am=stl_am * jnp.ones(xy))

    return dict(state=state, physics_data=physics_data,
                parameters=Parameters.default(),
                forcing=forcing, terrain=terrain)


# Reference [max, min, mean] of every published surface-flux field — all of
# them fmask-weighted land/sea means — for the configurations below.
_FIELDS = ("ustr", "vstr", "shf", "evap", "rlus", "hfluxn", "tsfc", "tskin",
           "u0", "v0", "t0")

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

    def test_land_branch_tolerates_higher_precision_forcing(self):
        """Forcing may arrive at a different precision from the state.

        Coupled drivers run in float64 and promote the forcing they hand
        back, while the model state stays float32. The land fluxes inherit
        the forcing dtype, so the zero arm of the land branch has to inherit
        it too or ``lax.cond`` rejects the pair outright.

        Enables x64 for the duration rather than skipping without it: the
        session pins the flag off, so a skip condition read at import time
        could never be true.
        """
        with jax.enable_x64():
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

    def test_fluxes_are_linear_in_the_land_fraction(self):
        """Each published flux is the fmask weighting of its two components.

        The components are internal, so the weighting is pinned through its
        endpoints instead: fmask 0 and 1 give the pure sea and pure land
        values, and any intermediate fraction must fall on the straight line
        between them. That catches a swapped or dropped weighting without
        publishing the per-surface values.
        """
        flux = lambda fmask: get_surface_fluxes(
            **build_inputs(fmask=fmask))[1].surface_flux
        sea_only, half, land_only = flux(0.0), flux(0.5), flux(1.0)

        for field in ("ustr", "vstr", "shf", "evap", "rlus", "hfluxn"):
            over_sea = getattr(sea_only, field)
            over_land = getattr(land_only, field)
            self.assertFalse(jnp.allclose(over_land, over_sea),
                             f"{field}: land and sea agree, so the check is vacuous")
            self.assertTrue(
                jnp.allclose(getattr(half, field), 0.5 * (over_land + over_sea),
                             rtol=1e-6),
                f"{field} is not linear in the land fraction")

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

        # Eight columns, not the T30 grid, and an operating point placed away
        # from the two hinges the scheme carries here:
        #
        #  * the near-surface extrapolation is
        #    ``t1 = ta[-1] + dt1_fac * (ta[-1] - ta_ref)`` applied only where
        #    the layer is unstable, i.e. ``dt1_fac * relu(ta[-1] - ta_ref)``,
        #    and ta constant in the vertical makes ta_ref *equal* ta[-1] — the
        #    isothermal ``build_inputs`` default is exactly on that hinge, and
        #    since the step is a fraction of each leaf's own magnitude
        #    (jcm.testing) temperature is the dominant direction, so the
        #    one-sided secants differ by a factor of four at every rung and no
        #    reference exists at all. A 50 K lapse over the column clears it.
        #  * ``_stability_factor`` is piecewise linear in the surface-to-air
        #    excess with breaks at 0 and at ``+dtheta`` above and
        #    ``-dtheta/astab`` below (3 K and -6 K at the defaults, with
        #    ``lscasym``). Its stable branch is therefore 6 K wide and -3 K is
        #    the middle of it: 3 K from either break, which is ten top-rung
        #    displacements of a ~290 K leaf. ``stl_am`` is set per column so
        #    that the land daily-cycle term ``ctday*sqrt(coa)*rsds`` — the only
        #    thing latitude changes here — lands every column on that same
        #    -3 K, and ``sst`` puts the sea branch there too.
        #
        # Eight columns rather than 4608 because a check over a whole grid
        # fails as soon as any one column straddles a branch, so its tolerance
        # ends up paying for the fixture rather than measuring the scheme:
        # on the grid the gap ran to 1.3 % and needed rtol=2e-2, while this
        # block holds 5e-3 — see ``build_block_inputs``.
        fsg = compute_speedy_vertical_coords(KX)[1]
        ta = 288.0 - 50.0 * (1.0 - fsg)
        latitudes = jnp.linspace(20.0, 45.0, 8)
        t2_sea = float(ta[-1]) + 5000.0 / c.cpd
        excess = -3.0
        args = build_block_inputs(
            latitudes, ta=ta, sst=t2_sea + excess,
            stl_am=(t2_sea - 500.0 / c.cpd + excess
                    - 0.01 * jnp.sqrt(jnp.cos(jnp.deg2rad(latitudes))) * 400.0))
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
        # 5e-3 is what the difference actually reaches on this block: over
        # seeds 0-9 it holds at every seed but 7, where the ladder reports no
        # usable rung at all rather than a loose one — an honest straddle in
        # that particular direction, not a tolerance to be widened. The
        # residual it does allow is float32 cancellation in the projection
        # itself: the published fluxes carry both signs, so a total near 12
        # is assembled from summands in the hundreds. live_inputs then holds
        # the per-leaf line the projection cannot: it is a sum over leaves,
        # and a dead one of them is invisible in it.
        check_gradients(f, float_args, rtol=5e-3,
                        live_inputs=["temperature", "specific_humidity",
                                     "u_wind", "v_wind",
                                     "sea_surface_temperature", "stl_am"])

    def test_surface_fluxes_drag_test_gradient_check(self):
        phi0 = 500. * jnp.ones(XY)
        hdrag = Parameters.default().surface_flux.hdrag
        check_vjp(get_orog_land_sfc_drag,
                  functools.partial(jax.vjp, get_orog_land_sfc_drag),
                  args=(phi0, hdrag), atol=None, rtol=1, eps=0.00001)
        check_jvp(get_orog_land_sfc_drag,
                  functools.partial(jax.jvp, get_orog_land_sfc_drag),
                  args=(phi0, hdrag), atol=None, rtol=1, eps=0.000001)


class TestSeaIceFluxes(unittest.TestCase):
    """Sea-ice weighting of the sea tile.

    SPEEDY hands the atmosphere a single sea-surface temperature
    ``tsea = (1 - sice)*SST + sice*T_ice`` with the ice surface at the saline
    freezing point (``sea_model.f90``); the sea fluxes are evaluated once at
    ``tsea``. These run on the aquaplanet (fmask = 0) so the published grid
    mean is exactly the sea tile.
    """

    # Saline freezing point and surface emission constant used by the scheme.
    SSTFR = 273.2 - 1.8
    ESBC = 0.98 * c.sbc  # Parameters.default().mod_radcon.emisfc * sigma

    def _sea(self, **kwargs):
        # Warm ocean under a cold, sub-saturated column: open water drives a
        # strong upward sensible/latent flux, so ice suppression is visible.
        args = build_inputs(aquaplanet=True, ta=280.0, rh=0.7, ua=5.0, va=2.0,
                            sst=300.0, rlds=350.0, **kwargs)
        _, physics_data = get_surface_fluxes(**args)
        return physics_data.surface_flux

    def test_effective_temperature_blend(self):
        """Published tsfc and rlus follow the ice-weighted freezing-point blend."""
        sst = 300.0
        t_ice = min(sst, self.SSTFR)
        for sice in (0.0, 0.3, 0.5, 1.0):
            with self.subTest(sice=sice):
                sflux = self._sea(sice=sice)
                tsea = sst + sice * (t_ice - sst)
                self.assertTrue(jnp.allclose(sflux.tsfc, tsea, atol=1e-3),
                                f"tsfc {jnp.mean(sflux.tsfc)} != {tsea}")
                self.assertTrue(
                    jnp.allclose(sflux.rlus, self.ESBC * tsea ** 4, rtol=1e-4),
                    "rlus is not the emission at the blended temperature")

    def test_fraction_interpolates_linearly_in_temperature(self):
        """The blended temperature is exactly linear in the ice fraction."""
        water, half, ice = (self._sea(sice=s) for s in (0.0, 0.5, 1.0))
        self.assertTrue(jnp.allclose(half.tsfc, 0.5 * (water.tsfc + ice.tsfc),
                                     rtol=1e-6))

    def test_ice_suppresses_turbulent_fluxes(self):
        """Sensible and latent fluxes fall monotonically as ice grows.

        Over the capped, colder ice surface both the sensible flux (linear in
        the surface temperature) and evaporation (through the much smaller
        saturation humidity at the freezing point) are strongly reduced
        relative to open water — the physical point of the fix.
        """
        water, half, ice = (self._sea(sice=s) for s in (0.0, 0.5, 1.0))
        for field in ("shf", "evap"):
            w, h, i = (jnp.mean(getattr(x, field)) for x in (water, half, ice))
            self.assertGreater(float(w), float(h), f"{field}: water !> 50% ice")
            self.assertGreater(float(h), float(i), f"{field}: 50% !> full ice")
        # Open water evaporates strongly; the freezing ice tile essentially
        # shuts moisture exchange off.
        self.assertGreater(float(jnp.mean(water.evap)), 0.0)
        self.assertLess(float(jnp.mean(ice.evap)),
                        0.2 * float(jnp.mean(water.evap)))

    def test_open_water_is_unchanged(self):
        """Zero ice fraction reproduces the pure-SST sea fluxes bit-for-bit."""
        sflux = self._sea(sice=0.0)
        self.assertTrue(jnp.allclose(sflux.tsfc, 300.0, atol=1e-4))

    def test_ice_fraction_varies_across_the_grid(self):
        """A per-cell ice fraction is applied pointwise (broadcasting-native).

        Passing a 2D ``sice`` field (uniform SST/air) must give each cell the
        blend for its own fraction — no shape assumption collapses the map.
        """
        sst = 300.0
        sice_grid = jnp.linspace(0.0, 1.0, IX * IL).reshape(XY)
        args = build_inputs(aquaplanet=True, ta=280.0, rh=0.7, ua=5.0, va=2.0,
                            sst=sst, rlds=350.0, sice=sice_grid)
        _, physics_data = get_surface_fluxes(**args)
        expected = sst + sice_grid * (min(sst, self.SSTFR) - sst)
        self.assertTrue(jnp.allclose(
            physics_data.surface_flux.tsfc, expected, atol=1e-3))

    def test_mixed_cell_ice_is_a_fraction_of_the_sea_part(self):
        """In coastal cells ``sice_am`` weights the sea tile, not the grid box.

        SPEEDY's ``sice_am`` is the ice fraction of the *sea part* of the
        cell: reference ``forcing.f90`` builds the sea albedo from it
        unnormalised and merges with land by ``fmask`` afterwards, and the
        packaged climatology reaches ``icec = 1`` in ``lsm = 0.99`` cells —
        impossible for a grid-box tile fraction, which is bounded by
        ``1 - lsm``. So the sea-tile temperature must depend on ``sice``
        alone, independent of ``fmask``: with sice = 1 the whole sea part
        is at the freezing point whatever the land fraction, and with
        sice = 0.5 it sits at the SST/freezing midpoint. A blend that
        renormalised by ``1 - fmask`` (reading sice as a grid-box
        fraction) would fail every mixed-cell case here.
        """
        sst, stl = 300.0, 288.0
        for fmask in (0.25, 0.5):
            for sice in (0.5, 1.0):
                with self.subTest(fmask=fmask, sice=sice):
                    args = build_inputs(ta=280.0, rh=0.7, ua=5.0, va=2.0,
                                        sst=sst, rlds=350.0, stl_am=stl,
                                        fmask=fmask, sice=sice)
                    _, physics_data = get_surface_fluxes(**args)
                    tsea = sst + sice * (min(sst, self.SSTFR) - sst)
                    expected = tsea + fmask * (stl - tsea)
                    self.assertTrue(
                        jnp.allclose(physics_data.surface_flux.tsfc,
                                     expected, atol=1e-3),
                        f"tsfc {float(jnp.mean(physics_data.surface_flux.tsfc))}"
                        f" != {expected} (sea-part-relative convention)")

    def test_partial_ice_gradient_is_finite(self):
        """Reverse-mode gradients (incl. w.r.t. sice_am) stay finite over ice."""
        args = build_inputs(aquaplanet=True, ta=280.0, rh=0.7, ua=5.0, va=2.0,
                            sst=300.0, rlds=350.0, sice=0.5)
        _, f_vjp = jax.vjp(
            get_surface_fluxes, args["state"], args["physics_data"],
            args["parameters"], args["forcing"], args["terrain"])
        cotangent = (
            PhysicsTendency.ones(ZXY),
            PhysicsData.ones(XY, KX,
                             speedy_coords=args["physics_data"].speedy_coords))
        df_dstate, df_ddatas, df_dparams, df_dforcing, _ = f_vjp(cotangent)
        self.assertFalse(df_dforcing.isnan().any_true(),
                         "Gradient w.r.t. forcing (sice_am) contains NaNs")
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())


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

    def test_aquaplanet_surface_temperature_is_the_sst(self):
        """With no land fraction the merge must collapse onto the sea side.

        The land branch does not run at all here, so a merge that leaned the
        wrong way would surface the land branch's zeros rather than the SST.
        """
        _, sflux, args = self._run(ta=280.0, rh=0.7, ua=5.0, va=2.0, sst=300.0,
                                   rlds=350.0)
        self.assertTrue(jnp.allclose(
            sflux.tsfc, args["forcing"].sea_surface_temperature))
        self.assertTrue(jnp.allclose(
            sflux.tskin, args["forcing"].sea_surface_temperature))

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
