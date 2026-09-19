import jax
import unittest
import jax.numpy as jnp
import numpy as np

from jcm.testing import check_gradients

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
        # The adjoint identity plus a live, finite gradient is what is
        # actually verifiable.
        #
        # live_inputs names the three leaves the temperature tendency is built
        # from, checked one at a time on the reverse gradient. The projection
        # cannot do it: contracted over the whole tree, qsat contributes
        # +5.7e-5 of it and se only +4.2e-9, so a stop_gradient on se would
        # move the projection by parts in ten thousand and pass. se also earns
        # the check on its own account — it is a dry static energy, O(3e5) at
        # any realistic operating point, and the schemes only ever take
        # differences of it (see jcm/testing.py on why the step is relative).
        args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats)
        check_gradients(f, args, reference="adjoint",
                        live_inputs=["convection/se", "humidity/qsat",
                                     "specific_humidity"])

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
        #    That is a property of this operating point, not of the scheme:
        #    the moisture half of vdiff has therefore never been covered by a
        #    gradient check. Tracked in issue #814.
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

        
