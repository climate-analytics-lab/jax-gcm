import jax
import unittest
import jax.numpy as jnp
import numpy as np

from jcm.testing import check_gradients

# Two reference soundings for the moisture half of the scheme, with the levels
# top-first (index 0 = model top, index -1 = surface) as the physics-internal
# frame requires. They are chosen against vdifsc.f90's own gate arithmetic
# rather than tuned until a branch opened. That arithmetic, on the SPEEDY
# 8-level sigma table:
#
#  * the lowest interface splits between a shallow-convection branch
#    (``dmse >= 0``, whose moisture flux is gated only on ``drh >= 0``) and a
#    stable moisture-diffusion branch (``dmse < 0``), whose onset is
#    ``drh0 = rhgrad * (fsg[-1] - fsg[-2]) = 0.0575``;
#  * step 3 runs on the interfaces inside ``1 .. kx-3`` whose own sigma
#    ``hsg[k+1]`` clears 0.5 -- interface ``k`` sits at the *bottom* of layer
#    ``k`` in this top-first frame -- which here is interfaces 4 and 5, with
#    ``drh0 = 0.0875`` and ``0.075``.
#
# Each sounding was built from a temperature profile by integrating the
# geopotential hydrostatically from ``phi_surface = 0`` with
# ``jcm.constants.rd``, taking ``qsat`` from
# :func:`jcm.physics.clouds.speedy_humidity.get_qsat` at ``ps = 1013 hPa``,
# then ``qa = rh * qsat`` and ``se = cpd * T + phi``. The results are frozen as
# literals so the test pins ``vdifsc`` alone, and does not move when the
# saturation formula or the hydrostatic integration does.

# Trade cumulus: a moist boundary layer (RH 0.88) under a drier free
# troposphere, with a near-moist-adiabatic lapse rate
# (T = 220, 200, 216, 240, 262, 279, 289, 297 K). This is conditionally
# unstable -- dmse = +6.3 kJ/kg -- so SPEEDY's *shallow convection* branch runs
# and carries both the dry-static-energy flux and, since drh = 0.13 > 0, the
# moisture flux. The column is stable to the dry-adiabatic test throughout, so
# step 4 contributes nothing and ttenvd here is the shallow-convection flux
# alone.
TRADE_CUMULUS = dict(
    se=[459777.814, 359213.287, 330841.218, 320225.549, 313115.064,
        307288.238, 301192.757, 298378.08],
    rh=[0.05, 0.1, 0.25, 0.3, 0.35, 0.5, 0.75, 0.88],
    qa=[0.0313958198, 0.000961144548, 0.0116021, 0.145869792, 0.987346447,
        4.16573572, 10.0059071, 17.005294],
    qsat=[0.627916396, 0.00961144548, 0.0464083999, 0.486232638, 2.82098985,
          8.33147144, 13.3412094, 19.3241978],
    phi=[238757.014, 158285.287, 113838.978, 79111.9494, 49899.3843,
         26993.6783, 10851.7969, 0.0],
)

# Stratocumulus: a cool moist boundary layer (RH 0.90) capped by a subsidence
# inversion -- the surface layer is *colder* than the one above it
# (T = ..., 283, 291, 286 K) -- under very dry subsiding air. dmse = -32 kJ/kg,
# so shallow convection is off and the stable moisture-diffusion branch runs
# instead on drh = 0.30 > drh0 = 0.0575. The deep dry layer above the inversion
# is close enough to dry-adiabatic that step 4 also fires at level 4, which is
# where this sounding's temperature tendency comes from.
STRATOCUMULUS = dict(
    se=[459951.019, 359386.493, 331014.423, 320398.755, 313288.27,
        311310.645, 303035.371, 287327.04],
    rh=[0.05, 0.08, 0.15, 0.2, 0.25, 0.35, 0.6, 0.9],
    qa=[0.0313958198, 0.000768915638, 0.00696125999, 0.0972465277,
        0.705247462, 3.83663354, 9.09775829, 8.67181692],
    qsat=[0.627916396, 0.00961144548, 0.0464083999, 0.486232638, 2.82098985,
          10.9618101, 15.1629305, 9.63535213],
    phi=[238930.219, 158458.493, 114012.183, 79285.1547, 50072.5896,
         26997.5253, 10685.1311, 0.0],
)

# The cases run against the Fortran. A sounding sitting far on the open side of
# a gate pins the flux but not the *threshold*: deleting the gate outright would
# not change its answer. So each gate additionally gets a pair of variants that
# straddle it, made by moving the relative humidity of one layer and nothing
# else. Perturbing ``rh`` alone is safe because none of the three thresholds
# depends on it: dmse is built from se, qa[-1] and qsat[-2], so the *branch*
# does not change under these overrides -- only which side of the moisture gate
# the column falls on. ``qa`` is kept consistent (``qa = rh * qsat``) even
# though vdifsc reads it at the surface level only.
#
# Index 6 is the layer above the surface, which sets the lowest interface's drh;
# index 4 is the layer above step-3 interface 4 -- the frame is top-first, and
# the scheme takes ``drh = rh[k+1] - rh[k]`` weighted by ``qsat[k]``.
MOISTURE_GATE_CASES = (
    # name, sounding, rh overrides, deep convection
    ("trade_cumulus", TRADE_CUMULUS, {}, False),
    ("trade_cumulus_deep_convection", TRADE_CUMULUS, {}, True),
    # drh = -0.02: below the shallow branch's drh >= 0, so the dry-static-energy
    # flux still fires but the moisture flux must not. This is the case that
    # would otherwise let a dropped guard drive a *reversed* moisture flux,
    # moistening the surface out of the PBL top.
    ("trade_cumulus_shallow_gate_closed", TRADE_CUMULUS, {6: 0.90}, False),
    # drh = +0.02: open, but below the stable branch's drh0 = 0.0575, so it also
    # pins that this branch's threshold is 0 and not drh0.
    ("trade_cumulus_shallow_gate_open", TRADE_CUMULUS, {6: 0.86}, False),
    ("stratocumulus", STRATOCUMULUS, {}, False),
    # Bit-identical to the line above, and that *is* what it pins: vdifsc sets
    # fcnv inside the dmse >= 0 block only, so redshc must not reach the stable
    # branch. Without this case, applying fcnv to the stable moisture flux too
    # would go unnoticed here.
    ("stratocumulus_deep_convection", STRATOCUMULUS, {}, True),
    # drh = 0.050 and 0.065 straddle the stable branch's drh0 = 0.0575.
    ("stratocumulus_stable_gate_closed", STRATOCUMULUS, {6: 0.85}, False),
    ("stratocumulus_stable_gate_open", STRATOCUMULUS, {6: 0.835}, False),
    # drh = 0.085 at interface 4, just below its drh0 = 0.0875; the unmodified
    # sounding sits just above at 0.10.
    ("stratocumulus_free_trop_gate_closed", STRATOCUMULUS, {4: 0.265}, False),
    # Interface 3's RH gate thrown wide open (drh = 0.15 against drh0 = 0.085)
    # where only ``hsg[4] = 0.42 <= 0.5`` keeps it shut. Without this the sigma
    # condition does no work in any case -- both soundings leave interface 3 at
    # drh = 0.05, already shut on RH -- and deleting it outright changes
    # nothing. The expected tendencies are the unmodified sounding's.
    ("stratocumulus_sigma_gate_shut", STRATOCUMULUS, {3: 0.10}, False),
)

def iptop_for(deep_convection, nlev):
    """Return the cloud-top index a column with or without deep convection has.

    SPEEDY's convection initialises ``iptop`` to the ``nlev + 1`` sentinel and
    lowers it to a cloud top only where it triggers, so those are the two kinds
    of value a real column presents. The scheme branches on
    ``icnv = nlev - iptop > 0``, so the cloud top itself is arbitrary -- as
    long as it stays below ``nlev``, which the straddle test asserts, since a
    short sounding would otherwise turn the deep-convection cases into silent
    duplicates of their base case and drop the redshc coverage with it.

    Public because ``tools/regenerate_speedy_vdiff_reference.py`` hands the same
    ``icnv`` to the Fortran; deriving it twice is how the reference and the port
    would come to be given different inputs.
    """
    return 3 if deep_convection else nlev + 1


def _with_rh_overrides(sounding, rh_overrides):
    """Return ``(rh, qa)`` with a case's relative-humidity overrides applied.

    Shared by the input builder and the straddle guard so the two cannot
    disagree about what a case actually presents to the scheme.
    """
    rh = list(sounding["rh"])
    qa = list(sounding["qa"])
    for index, value in rh_overrides.items():
        rh[index] = value
        qa[index] = value * sounding["qsat"][index]
    return rh, qa


# Tendencies from SPEEDY's ``vertical_diffusion.f90`` (samhatfield/speedy.f90),
# compiled unmodified in double precision against stub modules supplying
# cp = 1004.64 and alhc = 2501.0 -- jcm's values, not SPEEDY's own cp = 1004.0
# -- so that only the formulation, and not the choice of constants, is being
# compared. Regenerate with
# ``python tools/regenerate_speedy_vdiff_reference.py``; compiling the file
# rather than transcribing it also keeps details a transcription silently
# "fixes", such as its single-precision ``segrad = 0.1`` literal widening to
# 0.10000000149 rather than 0.1 (worth 1e-7 in the step-4 tendencies).
#
# Note that the *inputs* are constants-independent but these expected
# tendencies are not: they scale with the live ``c.cpd`` through fshcse/fvdise
# and with alhc through dmse. The same 0.06% cpd revision that forced the
# tolerance note in test_get_vertical_diffusion_tend would put these outside
# rtol = 2e-4 as well, so a constants change means regenerating them rather
# than hunting for a port bug.
FORTRAN_REFERENCE = {
    "trade_cumulus": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2506675098057929e-04,
                -2.9258677627475314e-04],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 2.3127328052662036e-06,
                1.3020053681520057e-05, 6.9235409235962714e-05,
                -1.1630304231481483e-04],
    ),
    "trade_cumulus_deep_convection": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1253337549028965e-04,
                -1.4629338813737657e-04],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 2.3127328052662036e-06,
                1.3020053681520057e-05, 2.4503469884110871e-05,
                -5.8151521157407417e-05],
    ),
    "trade_cumulus_shallow_gate_closed": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2506675098057929e-04,
                -2.9258677627475314e-04],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 2.3127328052662036e-06,
                2.2301351437307094e-05, -3.2365551148385563e-05, 0.0],
    ),
    "trade_cumulus_shallow_gate_open": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2506675098057929e-04,
                -2.9258677627475314e-04],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 2.3127328052662036e-06,
                1.9826338702430550e-05, -1.5365322386823345e-05,
                -1.7892775740740759e-05],
    ),
    "stratocumulus": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 1.1964343331322881e-05,
                -5.3839544990952970e-06, -5.3839544990952970e-06,
                -5.3839544990952970e-06],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 1.5418218701774688e-06,
                1.8719994726562494e-05, 2.5021758845819989e-05,
                -6.7127556901041690e-05],
    ),
    "stratocumulus_deep_convection": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 1.1964343331322881e-05,
                -5.3839544990952970e-06, -5.3839544990952970e-06,
                -5.3839544990952970e-06],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 1.5418218701774688e-06,
                1.8719994726562494e-05, 2.5021758845819989e-05,
                -6.7127556901041690e-05],
    ),
    "stratocumulus_stable_gate_closed": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 1.1964343331322881e-05,
                -5.3839544990952970e-06, -5.3839544990952970e-06,
                -5.3839544990952970e-06],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 1.5418218701774688e-06,
                3.9072506727430539e-05, -5.3229646771501056e-05, 0.0],
    ),
    "stratocumulus_stable_gate_open": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 1.1964343331322881e-05,
                -5.3839544990952970e-06, -5.3839544990952970e-06,
                -5.3839544990952970e-06],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 1.5418218701774688e-06,
                3.7851356007378451e-05, -4.0444831218182400e-05,
                -1.4544303995225712e-05],
    ),
    "stratocumulus_free_trop_gate_closed": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 1.1964343331322881e-05,
                -5.3839544990952970e-06, -5.3839544990952970e-06,
                -5.3839544990952970e-06],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 0.0, 2.0352512000868048e-05,
                2.5021758845819989e-05, -6.7127556901041690e-05],
    ),
    "stratocumulus_sigma_gate_shut": dict(
        ttenvd=[0.0, 0.0, 0.0, 0.0, 1.1964343331322881e-05,
                -5.3839544990952970e-06, -5.3839544990952970e-06,
                -5.3839544990952970e-06],
        qtenvd=[0.0, 0.0, 0.0, 0.0, 1.5418218701774688e-06,
                1.8719994726562494e-05, 2.5021758845819989e-05,
                -6.7127556901041690e-05],
    ),
}

class Test_VerticalDiffusion_Unit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 1, 1, 8

        global HumidityData, ConvectionData, PhysicsData, PhysicsState, PhysicsTendency, get_vertical_diffusion_tend, \
            parameters, speedy_coords, terrain, ForcingData
        from jcm.physics.speedy.physics_data import HumidityData, ConvectionData, PhysicsData
        from jcm.physics.speedy.params import Parameters
        from jcm.forcing import ForcingData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.vertical_diffusion.speedy_vdiff import get_vertical_diffusion_tend
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords

        speedy_coords = SpeedyCoords.single_column_coords(num_levels=kx)
        parameters = Parameters.default()
        terrain = TerrainData.single_column()

    def test_get_vertical_diffusion_tend(self):
        se = jnp.ones((ix,il)) * jnp.linspace(400,300,kx)[:, jnp.newaxis, jnp.newaxis]
        rh = jnp.ones((ix,il)) * jnp.linspace(0.1,0.9,kx)[:, jnp.newaxis, jnp.newaxis]
        qa = jnp.ones((ix,il)) * jnp.array([1, 4, 7.3, 8.8, 12, 18, 24, 26])[:, jnp.newaxis, jnp.newaxis]
        qsat = jnp.ones((ix,il)) * jnp.array([5, 8, 10, 13, 16, 21, 28, 31])[:, jnp.newaxis, jnp.newaxis]
        phi = jnp.ones((ix,il)) * jnp.linspace(150000,0,kx)[:, jnp.newaxis, jnp.newaxis]
        iptop = jnp.ones((ix,il), dtype=int)*1
        
        zxy = (kx, ix, il)
        xy = (ix, il)
        humidity_data = HumidityData.zeros((ix,il), kx, rh=rh, qsat=qsat)
        convection_data = ConvectionData.zeros((ix,il), kx, iptop=iptop, se=se)
        physics_data = PhysicsData.zeros((ix,il), kx, humidity=humidity_data, convection=convection_data, speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=phi)
        forcing = ForcingData.ones(xy)
        
        # utenvd, vtenvd, ttenvd, qtenvd = get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv)
        physics_tendencies, _ = get_vertical_diffusion_tend(state, physics_data, parameters, forcing, terrain)

        utenvd, vtenvd, ttenvd, qtenvd = physics_tendencies.u_wind, physics_tendencies.v_wind, physics_tendencies.temperature, physics_tendencies.specific_humidity

        self.assertTrue(np.allclose(utenvd, np.zeros_like(utenvd), atol=1e-9))
        self.assertTrue(np.allclose(vtenvd, np.zeros_like(vtenvd), atol=1e-9))
        # Tolerance loosened from 1e-9 to 1e-6: cp was unified to the high-precision
        # ECHAM value (1004.64; was 1004.0) and rd to 287.04 (see jcm/constants.py).
        # That ~0.06% change shifts these reference tendencies by ~2e-7 — far below
        # any physical significance, but above the original 1e-9 lock.
        self.assertTrue(np.allclose(ttenvd[:,0,0], np.array([ 2.78098357e-04,  1.39862334e-04,  8.50690617e-05,  3.73100450e-05,
        3.67983799e-06, -2.65383318e-05, -6.18272365e-05, -3.07837296e-04]), atol=1e-6))
        self.assertTrue(np.allclose(qtenvd[:,0,0], np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 9.99411916e-06,  7.24206425e-06,  1.30163815e-05, -4.72222083e-05]), atol=1e-6))

    def test_get_vertical_diffusion_gradients_isnan_ones(self):
        """Test that we can calculate gradients of vertical diffusion without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)

        # Calculate gradient
        primals, f_vjp = jax.vjp(get_vertical_diffusion_tend, state, physics_data, parameters, forcing, terrain)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)
        input = (tends, datas)
        df_dstate, df_ddatas, df_dparams, df_dforcing, df_dterrain = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

    def test_get_vertical_diffusion_gradient_check(self):
        """Test that we get correct gradient values"""
        from jcm.utils import convert_back, convert_to_float
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx, speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = get_vertical_diffusion_tend(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            # Only the temperature tendency carries a gradient at this
            # operating point; see the assertions below for the other two.
            return convert_to_float(tend_out.temperature)

        # No finite difference is usable here in any direction tried (seeds
        # 0-4): the diffusion coefficients are kinked in the bulk Richardson
        # number at this operating point, and a central difference across a
        # kink converges — stably, at every step — to the *mean* of the two
        # one-sided derivatives, which is not what AD computes. The one-sided
        # secants stay a factor ~4000 apart however far the step comes down.
        # An earlier rtol here passed only because one direction happened to
        # land near the mean; the adjoint identity plus a live, finite gradient
        # is what is actually verifiable.
        args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats)
        check_gradients(f, args, reference="adjoint")

        # Two of the three tendency components are dead here, for different
        # reasons, and both are asserted rather than left to pass silently
        # inside a projection:
        #
        #  * wind - SPEEDY's vertical diffusion is a heat and moisture scheme
        #    (vdifsc.f90 returns ttenvd/qtenvd only) and momentum is handled by
        #    the surface drag, so this is structural.
        #  * specific humidity - the moisture branch is gated on
        #    ``drh = rh[surface] - rh[nl1] > drh0``, and PhysicsData.ones()
        #    makes rh uniform, so drh is exactly 0 and the gate never opens.
        #    That is a property of this operating point, not of the scheme;
        #    the moisture half is checked on soundings that do open the gate
        #    in test_moisture_branch_gradients.
        def f_dead(physics_data_f, state_f, parameters_f, forcing_f, terrain_f):
            tend_out, _ = get_vertical_diffusion_tend(physics_data=convert_back(physics_data_f, physics_data),
                                       state=convert_back(state_f, state),
                                       parameters=convert_back(parameters_f, parameters),
                                       forcing=convert_back(forcing_f, forcing),
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return (convert_to_float(tend_out.u_wind),
                    convert_to_float(tend_out.v_wind),
                    convert_to_float(tend_out.specific_humidity))

        primal, dead_vjp = jax.vjp(f_dead, *args)
        grads = dead_vjp(tuple(jnp.ones_like(x) for x in primal))
        self.assertTrue(all(jnp.all(g == 0) for g in jax.tree.leaves(grads)),
                        "wind and moisture tendencies are expected to be dead here")

    # -- The moisture branch ------------------------------------------------
    #
    # Everything below exercises the half of vdifsc that a uniform-rh state
    # cannot reach. ``PhysicsData.ones()`` gives drh == 0 exactly, so neither
    # the shallow-convection moisture flux (dmse >= 0, drh > 0) nor the stable
    # diffusion branch (dmse < 0, drh > drh0) ever ran under a gradient check,
    # and the whole dmse >= 0 branch -- its heat flux and the redshc damping
    # included -- had no value coverage at all.

    def _sounding_inputs(self, sounding, rh_overrides, deep_convection):
        """Assemble the scheme's arguments from a sounding literal.

        See :func:`iptop_for` for the two ``iptop`` values a real column
        presents.
        """
        iptop = iptop_for(deep_convection, kx)
        rh, qa = _with_rh_overrides(sounding, rh_overrides)

        def col(values):
            return jnp.asarray(values, dtype=jnp.float32)[:, jnp.newaxis, jnp.newaxis] \
                * jnp.ones((ix, il))

        humidity = HumidityData.zeros((ix, il), kx, rh=col(rh),
                                      qsat=col(sounding["qsat"]))
        convection = ConvectionData.zeros(
            (ix, il), kx, iptop=jnp.full((ix, il), iptop, dtype=int),
            se=col(sounding["se"]))
        physics_data = PhysicsData.zeros(
            (ix, il), kx, humidity=humidity, convection=convection,
            speedy_coords=speedy_coords)
        state = PhysicsState.zeros((kx, ix, il), specific_humidity=col(qa),
                                   geopotential=col(sounding["phi"]))
        return state, physics_data

    def _step3_interfaces(self):
        """How step 3 selects its interfaces, derived rather than written out.

        Mirrors the scheme's own selection (``jcm.physics.vertical_diffusion.
        speedy_vdiff``): interfaces ``1 .. kx-3`` whose own sigma ``hsg[k+1]``
        clears 0.5 -- the frame is top-first, so interface ``k`` sits at the
        *bottom* of layer ``k`` -- and whose upper layer is not stratospheric.

        Returns ``(active, sigma_shut, strat_only)``: the interfaces step 3
        diffuses across, those inside the same index range that the sigma
        condition alone excludes, and those the stratosphere mask excludes but
        the sigma condition does not.
        """
        from jcm.physics.speedy.speedy_coords import stratosphere_mask

        hsg = np.asarray(speedy_coords.hsg)
        strat = np.asarray(stratosphere_mask(speedy_coords.fsg))
        candidates = range(1, kx - 2)
        return ([k for k in candidates if hsg[k + 1] > 0.5 and not strat[k]],
                [k for k in candidates if hsg[k + 1] <= 0.5 and not strat[k]],
                [k for k in candidates if hsg[k + 1] > 0.5 and strat[k]])

    def test_moisture_branch_matches_fortran(self):
        """Every moisture path and every gate, against the reference Fortran.

        The two soundings split the lowest interface between them: trade
        cumulus takes ``dmse > 0`` (shallow convection, and under deep
        convection the redshc damping), stratocumulus takes ``dmse < 0`` with
        ``drh > drh0`` (stable diffusion). The variants straddle each gate, so
        the thresholds are pinned and not just the fluxes they admit.

        The tolerance is float32, and what sets it is step 4: its
        ``se0 - se[k]`` is 330 J/kg out of dry static energies of 3.1e5, so it
        loses about three decimal digits to cancellation, and the stratocumulus
        temperature tendency -- which is step 4 alone, the shallow heat flux
        being off at ``dmse < 0`` -- agrees with the double-precision Fortran to
        1.98e-5 relative. The ``dmse`` path is milder (trade-cumulus
        temperature, 3.9e-6) and every moisture tendency milder still (8.6e-7).
        A shut gate must give *exactly* zero, which ``atol`` covers.
        """
        for name, sounding, rh_overrides, deep in MOISTURE_GATE_CASES:
            with self.subTest(case=name):
                state, physics_data = self._sounding_inputs(
                    sounding, rh_overrides, deep)
                tend, _ = get_vertical_diffusion_tend(
                    state, physics_data, parameters, ForcingData.ones((ix, il)),
                    terrain)
                expected = FORTRAN_REFERENCE[name]
                np.testing.assert_allclose(
                    np.asarray(tend.temperature[:, 0, 0]),
                    expected["ttenvd"], rtol=2e-4, atol=1e-12)
                np.testing.assert_allclose(
                    np.asarray(tend.specific_humidity[:, 0, 0]),
                    expected["qtenvd"], rtol=2e-4, atol=1e-12)

    def test_moisture_gate_cases_straddle_their_gates(self):
        """Each case sits on the side of its gate that it is named for.

        Asserted separately from the value test so that a case which silently
        stopped straddling ``drh0`` -- after a change to rhgrad or to the sigma
        table -- fails as the coverage regression it is, rather than as an
        unexplained match of two arrays of zeros.

        The comparisons are strict because the scheme's width-0 gates are:
        ``smooth_gate(x, thr, 0)`` is ``x > thr``. vdifsc matches that exactly
        on the step-2 *stable* branch, which it writes ``else if (drh > drh0)``;
        it is the shallow branch (``drh >= 0``) and step 3 (``drh >= drh0``)
        that are non-strict there, so those two differ from the port on a
        column landing exactly on a threshold. Unreachable in practice, and the
        strict form is what these assertions must describe if they are to
        predict what the scheme does.
        """
        from jcm.physics.speedy.physical_constants import alhc

        fsg = np.asarray(speedy_coords.fsg)
        rhgrad = float(parameters.vertical_diffusion.rhgrad)
        # A deep-convection case is only a deep-convection case while the
        # cloud top iptop_for picks sits below kx; see that function.
        self.assertGreater(kx - iptop_for(True, kx), 0,
                           "the deep-convection cases no longer set icnv > 0, "
                           "so they duplicate their base case and redshc is "
                           "uncovered")
        interfaces, sigma_shut, strat_only = self._step3_interfaces()
        self.assertTrue(interfaces, "step 3 has no active interfaces to check")
        # The port's one deliberate divergence from vdifsc is here. The Fortran
        # hard-codes the range as ``do k = 3, kx-2``; the port keeps the
        # numeric upper bound and replaces the ``k = 3`` start -- the
        # upper-atmosphere end of the range, and numerically its lower bound --
        # with ``1`` plus the sigma<0.2 stratosphere mask, so the range scales
        # with nlev. On this 8-level table the two coincide, which is why no
        # case above can see the substitution. State that rather than leave it
        # implicit: at an nlev where they part company these cases stop being a
        # Fortran comparison at the interfaces concerned, and that should
        # surface here rather than as a silent change of behaviour.
        self.assertEqual(strat_only, [],
                         "the stratosphere mask now excludes an interface the "
                         "sigma condition keeps, so the cases here no longer "
                         "pin step 3 against the Fortran range")
        # And the other direction: the port's index range starts at 1, which the
        # Fortran's ``do k = 3, kx-2`` (0-based: k >= 2) excludes outright. Here
        # interface 1 is shut twice over -- ``hsg[2] = 0.14`` and
        # ``fsg[1] = 0.095 < 0.2`` -- so it takes an nlev where *both* open to
        # put the port outside the Fortran's range.
        self.assertGreaterEqual(min(interfaces), 2,
                                "step 3 now diffuses across an interface below "
                                "vdifsc's own k >= 3 lower bound")
        self.assertTrue(sigma_shut, "no interface is excluded by hsg <= 0.5, so "
                                    "stratocumulus_sigma_gate_shut pins nothing")

        for name, sounding, rh_overrides, _ in MOISTURE_GATE_CASES:
            with self.subTest(case=name):
                se = np.asarray(sounding["se"])
                qsat = np.asarray(sounding["qsat"])
                # Read the *overridden* rh and qa, the same arrays the scheme
                # is handed. None of the present cases moves qa[-1], so dmse is
                # in practice a property of the sounding alone -- but computing
                # it from the pristine literals would keep this assertion
                # passing against stale data if one ever did, and dmse is only
                # +6.3 kJ/kg on trade cumulus, close enough to flip the branch.
                rh, qa = (np.asarray(a, dtype=float)
                          for a in _with_rh_overrides(sounding, rh_overrides))
                dmse = se[-1] - se[-2] + alhc * (qa[-1] - qsat[-2])
                shallow = sounding is TRADE_CUMULUS
                self.assertEqual(bool(dmse > 0.0), shallow)

                pbl_drh = rh[-1] - rh[-2]
                pbl_drh0 = 0.0 if shallow else rhgrad * (fsg[-1] - fsg[-2])
                if "shallow_gate_closed" in name or "stable_gate_closed" in name:
                    self.assertLessEqual(pbl_drh, pbl_drh0)
                else:
                    self.assertGreater(pbl_drh, pbl_drh0)

                # A free-troposphere case shuts exactly the interface whose
                # upper layer it overrode; derived from the override so the two
                # cannot drift apart if the sigma table changes which
                # interfaces are active.
                shut = (set(rh_overrides) & set(interfaces)
                        if "free_trop_gate_closed" in name else set())
                for k in interfaces:
                    drh = rh[k + 1] - rh[k]
                    drh0 = rhgrad * (fsg[k + 1] - fsg[k])
                    if k in shut:
                        self.assertLessEqual(drh, drh0)
                    else:
                        self.assertGreater(drh, drh0)
                if "free_trop_gate_closed" in name:
                    self.assertTrue(shut, "the override no longer shuts an "
                                          "active step-3 interface")

                # The sigma-gate case has to leave an interface whose RH gate is
                # wide open and which only hsg <= 0.5 keeps shut; otherwise the
                # sigma condition is doing no work and its case pins nothing.
                if "sigma_gate_shut" in name:
                    self.assertTrue(
                        any(rh[k + 1] - rh[k] > rhgrad * (fsg[k + 1] - fsg[k])
                            for k in sigma_shut),
                        "no sigma-excluded interface has its RH gate open")

    def test_moisture_branch_gradients(self):
        """Gradients where the moisture tendency is live.

        Unlike the uniform-rh operating point above, a real finite difference
        is usable here: within a branch the fluxes are smooth, and the
        perturbations stay far from the gates (the nearest margin on these two
        soundings is ``drh - drh0 = 0.012``). The ``adjoint`` pass runs first
        for its per-output-leaf guard -- a difference contracts the outputs
        onto one projection, where an identically zero specific-humidity
        gradient, the exact defect this test exists for, would be invisible.

        What the difference does *not* constrain is which inputs the answer is
        sensitive to. The projection here is dominated by the sigma-grid
        metrics, and ``check_gradients`` steps by an absolute 1e-3, which is
        below a float32 ulp of ``se ~ 3e5``, so an ``se`` perturbation rounds
        away entirely. See test_moisture_branch_input_sensitivities.
        """
        from jcm.utils import convert_back, convert_to_float

        for name, sounding in (("trade_cumulus", TRADE_CUMULUS),
                               ("stratocumulus", STRATOCUMULUS)):
            with self.subTest(sounding=name):
                state, physics_data = self._sounding_inputs(
                    sounding, {}, deep_convection=False)
                forcing = ForcingData.ones((ix, il))

                def f(physics_data_f, state_f, parameters_f, forcing_f, terrain_f):
                    tend, _ = get_vertical_diffusion_tend(
                        physics_data=convert_back(physics_data_f, physics_data),
                        state=convert_back(state_f, state),
                        parameters=convert_back(parameters_f, parameters),
                        forcing=convert_back(forcing_f, forcing),
                        terrain=convert_back(terrain_f, terrain))
                    return (convert_to_float(tend.temperature),
                            convert_to_float(tend.specific_humidity))

                args = (convert_to_float(physics_data), convert_to_float(state),
                        convert_to_float(parameters), convert_to_float(forcing),
                        convert_to_float(terrain))
                check_gradients(f, args, reference="adjoint")
                # Four directions, not one: a single random direction can miss
                # a term it happens to be nearly orthogonal to. 5e-3 is a 10x
                # margin -- these agree with the central difference to 5e-4.
                for seed in range(4):
                    check_gradients(f, args, rtol=5e-3, seed=seed)

    # Which inputs the moisture branch is differentiably sensitive to. This is
    # the mirror of check_gradients' per-output-leaf guard, and it is needed
    # because the central difference cannot check it here: the projection is
    # dominated by the sigma-grid metrics, and an absolute 1e-3 step on
    # ``se ~ 3e5`` (float32 ulp 0.03) rounds away, so a stop_gradient on ``se``
    # or ``phi`` slips through the difference untouched. That blind spot is a
    # property of the step ladder rather than of this scheme -- tracked in
    # issue #820.
    #
    # The dead entries are as much of the point as the live ones, and each has
    # a reason in the scheme:
    #
    #  * geopotential is read only by step 4's super-adiabatic damping, and the
    #    trade-cumulus column is stable to the dry-adiabatic test everywhere,
    #    so there is nothing to damp.
    #  * specific humidity is read only through ``dmse``, and in the stable
    #    branch both of its consumers are flat: ``g_mse`` is a step, and the
    #    dry-static-energy hinge ``smooth_pos`` is zero below its threshold.
    #  * fsg enters only the gate thresholds ``drh0`` and the static
    #    stratosphere mask. At the default zero smoothing width the gates are
    #    steps, so a threshold has no derivative -- which is exactly the
    #    differentiability that ``rh_gate_smoothing`` exists to restore.
    LIVE_INPUT_GRADIENTS = {
        "trade_cumulus": dict(se=True, rh=True, qsat=True, qa=True, phi=False,
                              dhs=True, hsg=True, fsg=False),
        "stratocumulus": dict(se=True, rh=True, qsat=True, qa=False, phi=True,
                              dhs=True, hsg=True, fsg=False),
    }

    def test_moisture_branch_input_sensitivities(self):
        """The tendencies depend on the inputs the scheme actually reads."""
        # The table below is read at the default zero smoothing widths.
        # ``rh_gate_smoothing > 0`` -- which is what it is for -- makes fsg live
        # in both soundings, so say that here rather than let an intended
        # improvement surface as "fsg: gradient liveness changed".
        #
        # ``mse_gate_smoothing`` does not move this table at any usable width.
        # It reaches fsg not at all, and it revives qa in stratocumulus only
        # once the width approaches dmse = -32 kJ/kg itself: measured
        # sum|d qa| is exactly 0 at w = 100 J/kg, 7e-18 at 1000 and 4e-7 at
        # 5000, because sigmoid(dmse/w) and softplus(dmse/w) saturate to
        # float32 zero far above that. So a contributor who turns it on and
        # sees this table unchanged is seeing the right answer, not a crossfade
        # that failed to wire up.
        self.assertEqual(
            (float(parameters.vertical_diffusion.mse_gate_smoothing),
             float(parameters.vertical_diffusion.rh_gate_smoothing)), (0.0, 0.0),
            "LIVE_INPUT_GRADIENTS describes the hard (zero-width) gates")
        for name, sounding in (("trade_cumulus", TRADE_CUMULUS),
                               ("stratocumulus", STRATOCUMULUS)):
            with self.subTest(sounding=name):
                state, physics_data = self._sounding_inputs(
                    sounding, {}, deep_convection=False)
                forcing = ForcingData.ones((ix, il))

                def f(physics_data_f, state_f):
                    tend, _ = get_vertical_diffusion_tend(
                        state_f, physics_data_f, parameters, forcing, terrain)
                    return tend.temperature, tend.specific_humidity

                # One cotangent of ones would contract all levels of each
                # tendency into a single sum, and the scheme writes flux
                # *divergences* (+f/dhs[k] at k, -f/dhs[k+1] at k+1): on a grid
                # with uniform dhs those cancel exactly, so a live gradient
                # would read as dead. Accumulating |grad| over one-hot
                # cotangents instead asks the question directly -- an input is
                # live iff some single output element depends on it.
                primal, vjp = jax.vjp(f, physics_data, state)

                def sensitivities(cotangent):
                    grad_data, grad_state = vjp(cotangent)
                    return dict(
                        se=grad_data.convection.se,
                        rh=grad_data.humidity.rh,
                        qsat=grad_data.humidity.qsat,
                        qa=grad_state.specific_humidity,
                        phi=grad_state.geopotential,
                        dhs=grad_data.speedy_coords.dhs,
                        hsg=grad_data.speedy_coords.hsg,
                        fsg=grad_data.speedy_coords.fsg,
                    )

                gradients = None
                for leaf_index, leaf in enumerate(primal):
                    for flat in range(leaf.size):
                        cotangent = [jnp.zeros_like(x) for x in primal]
                        cotangent[leaf_index] = jnp.zeros_like(leaf).reshape(
                            -1).at[flat].set(1.0).reshape(leaf.shape)
                        one = sensitivities(tuple(cotangent))
                        gradients = (
                            {k: np.abs(np.asarray(v)) for k, v in one.items()}
                            if gradients is None else
                            {k: gradients[k] + np.abs(np.asarray(v))
                             for k, v in one.items()})
                for field, expect_live in self.LIVE_INPUT_GRADIENTS[name].items():
                    values = np.asarray(gradients[field])
                    self.assertTrue(np.all(np.isfinite(values)),
                                    f"{field}: gradient is not finite")
                    self.assertEqual(bool(np.any(values != 0.0)), expect_live,
                                     f"{field}: gradient liveness changed")
