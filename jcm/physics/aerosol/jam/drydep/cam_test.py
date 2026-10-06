"""Reference CAM collection and prescribed-map integration checks."""
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.drydep.drydep_term import CAMDryDeposition, DryDepParameters
from jcm.physics.aerosol.jam.drydep.resistances import cam_collection_velocity
from jcm.physics.aerosol.jam.sedimentation.sedi_term import stokes_velocity


class CAMCollectionTest(unittest.TestCase):
    def test_against_compiled_cam_all_classes_both_moments(self):
        """CAM 21a7829 aero_model.F90 compiled unmodified with module stubs.

        Independent Fortran values at r_num=.7 um, sigma=1.8, rho=2500,
        T=288 K, p=1000 hPa, u*=.3 m/s, neutral ra=95.941 s/m. The
        tolerance covers the host's slightly different viscosity constants.
        """
        expected = np.array([
            [4.079381637e-4,5.215900493e-4,5.215900493e-4,4.063524646e-4,
             4.067317605e-4,4.063221979e-4,9.029068627e-4,3.744188098e-4,
             5.431077548e-4,5.215900493e-4,3.744188098e-4],
            [1.951159937e-4,3.584625962e-4,3.584625962e-4,3.154425502e-4,
             6.680123724e-4,3.204706440e-4,5.660592788e-4,2.268317766e-4,
             2.715704076e-4,3.584625962e-4,2.268317766e-4],
        ])
        for row, moment in enumerate((0,3)):
            r = jnp.full(11, .7e-6)
            grav = stokes_velocity(r,2500.,288.,1e5,geom_std_dev=1.8,
                                   moment=moment,aspherical=True)
            actual = cam_collection_velocity(r,grav,.3,288.,1e5,1e5/(287.05*288),
                                             jnp.eye(11),geom_std_dev=1.8,moment=moment)
            np.testing.assert_allclose(actual,expected[row],rtol=.03)

    def test_mixed_cover_is_area_weighted_velocity(self):
        def run(frac):
            return cam_collection_velocity(jnp.full(2,2e-6),jnp.full(2,.01),.4,
                                           288.,1e5,1.2,frac,geom_std_dev=1.,moment=3)
        a=jnp.eye(11)[6,:,None]*jnp.ones((1,2))
        b=jnp.eye(11)[7,:,None]*jnp.ones((1,2))
        np.testing.assert_allclose(run(.3*a+.7*b),.3*run(a)+.7*run(b),rtol=1e-6)
        # Wet water and dry bare ground cannot inherit one collection rate.
        self.assertGreater(float(run(a)[0]),float(run(b)[0]))

    def test_calm_wind_value_and_gradient_are_finite(self):
        def run(u):
            return cam_collection_velocity(jnp.array([1e-7]),jnp.array([0.]),u,
                                           288.,1e5,1.2,jnp.eye(11)[7,:,None],
                                           geom_std_dev=1.8,moment=3).sum()
        self.assertTrue(np.isfinite(jax.jit(run)(0.)))
        self.assertTrue(np.isfinite(jax.jit(jax.grad(run))(0.)))

    def test_cached_map_and_aquaplanet_selection(self):
        from jcm.utils import get_coords
        from jcm.physics.echam.echam_levels import get_echam_levels
        term=CAMDryDeposition()
        term.cache_coords(get_coords(vertical_coords=get_echam_levels(47),
                                     spectral_truncation=21))
        frac=np.asarray(term._fractions.get_value())
        self.assertEqual(frac.shape,(11,64*32))
        self.assertTrue((frac>=0).all())
        np.testing.assert_allclose(frac.sum(axis=0),1.,atol=2e-7)
        kw=dict(mode=SimpleNamespace(geom_std_dev=1.8),moment=3,
                params=DryDepParameters.default())
        water=term._velocity(jnp.ones(64*32)*.7e-6,.002,.3,288.,1e5,1.2,
                             terrain=SimpleNamespace(fmask=jnp.zeros(64*32)),**kw)
        world=term._velocity(jnp.ones(64*32)*.7e-6,.002,.3,288.,1e5,1.2,
                             terrain=SimpleNamespace(fmask=jnp.ones(64*32)),**kw)
        np.testing.assert_allclose(water,water[0],rtol=1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(world-water))),1e-5)
