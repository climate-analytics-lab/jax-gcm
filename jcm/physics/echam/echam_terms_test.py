"""Tests for composable ECHAM physics (echam_terms.py).

Tests mixed-package composition, replacement of individual terms, and
roundtripping through nnx.split / nnx.merge.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.utils import get_coords
from jcm.physics.physics_term import PhysicsTerm


class DummyRadiationTerm(PhysicsTerm):
    """Minimal radiation term used to test ECHAM factory dispatch."""

    name = "dummy_radiation"
    category = "radiation"


class DummyConvectionTerm(PhysicsTerm):
    """Minimal non-radiation term used to test factory validation."""

    name = "dummy_convection"
    category = "convection"


def _make_echam_test_setup(nlev=8, nlat=64, nlon=32):
    """Create test setup matching ECHAM conventions."""
    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(nlat, nlon))
    terrain = TerrainData.aquaplanet(coords)
    forcing = ForcingData.zeros((nlat, nlon))

    shape_3d = (nlev, nlat, nlon)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    state = PhysicsState(
        temperature=250.0 + 20.0 * jax.random.normal(
            keys[0], shape_3d,
        ),
        specific_humidity=jnp.abs(
            3.0 * jax.random.normal(keys[1], shape_3d),
        ),
        u_wind=5.0 * jax.random.normal(keys[2], shape_3d),
        v_wind=5.0 * jax.random.normal(keys[3], shape_3d),
        geopotential=jnp.broadcast_to(
            jnp.linspace(50000, 0, nlev)[:, None, None],
            shape_3d,
        ),
        normalized_surface_pressure=(
            1.0
            + 0.01 * jax.random.normal(keys[4], (nlat, nlon))
        ),
        tracers={
            "qc": jnp.abs(
                1e-4 * jax.random.normal(keys[5], shape_3d),
            ),
            "qi": jnp.zeros(shape_3d),
        },
    )

    return coords, state, forcing, terrain


class TestEchamComposablePhysics(unittest.TestCase):
    """Test composable ECHAM physics wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.coords, self.state, self.forcing, self.terrain = (
            _make_echam_test_setup()
        )

    def test_echam_physics_factory(self):
        """echam_physics() creates composable physics with correct terms."""
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(checkpoint_terms=False)
        # Cloud fraction and microphysics are separate terms; the GWD
        # category split adds Hines + SSO (the simple-GWD scheme is kept
        # available but excluded from the default factory); the terminal
        # EchamSurfaceExchange publishes the #754 coupling struct.
        self.assertEqual(len(physics.terms), 13)
        categories = [t.category for t in physics.terms]
        self.assertIn("radiation", categories)
        self.assertIn("convection", categories)
        self.assertIn("surface", categories)
        self.assertIn("cloud_fraction", categories)
        self.assertIn("clouds", categories)
        self.assertIn("hines", categories)
        self.assertIn("sso", categories)
        self.assertNotIn("simple_gwd", categories)
        # Cloud fraction must precede microphysics so the microphysics term
        # can read the post-condensation qc/qi/cloud_fraction diagnostics.
        self.assertLess(
            categories.index("cloud_fraction"),
            categories.index("clouds"),
        )
        # Cloud fraction must also precede radiation so radiation sees the
        # current step's cloud field (matches ECHAM6's cov→rad ordering).
        self.assertLess(
            categories.index("cloud_fraction"),
            categories.index("radiation"),
        )
        # ECHAM physc ordering (radheat → vdiff → cucall → cloud): vertical
        # diffusion and the surface term that republishes its delivered
        # fluxes must precede convection, so the Tiedtke zdqpbl closure
        # reads the SAME-STEP vdiff moisture tendency and evaporation. A
        # convection-first ordering forces a one-step-lagged supply, which
        # compounds the convergence→convection feedback (onset7 NaN).
        self.assertLess(
            categories.index("radiation"),
            categories.index("vertical_diffusion"),
        )
        self.assertLess(
            categories.index("vertical_diffusion"),
            categories.index("surface"),
        )
        self.assertLess(
            categories.index("surface"),
            categories.index("convection"),
        )
        self.assertLess(
            categories.index("convection"),
            categories.index("clouds"),
        )

    def test_cu_lmfmid_flag_toggles_the_omega_requirement(self):
        """The scalar cu_lmfmid knob controls the dycore omega contract.

        With the mid-level trigger on (the default) TiedtkeConvection
        declares an ``omega`` dycore requirement, which fails Model
        construction on a backend that cannot supply omega (pySES, #698).
        Setting cu_lmfmid=False drops that requirement so the ne30
        experiments compose (#715).
        """
        from jcm.physics.echam.echam_terms import echam_physics

        on = echam_physics(checkpoint_terms=False)
        self.assertIn("omega", on.required_dycore_fields())

        off = echam_physics(checkpoint_terms=False, cu_lmfmid=False)
        self.assertNotIn("omega", off.required_dycore_fields())

    def test_updraft_precip_cover_follows_the_ham_submodel(self):
        """The sub-cloud rain-evaporation cover tracks the JAM chain (#812).

        ECHAM keys ``zcucov`` on ``lham`` (mo_cufluxdts.f90:414-420); jcm's
        ``lham`` is "the JAM aerosol chain is composed", so the convection
        term takes the updraft-area cover for a JAM run and the constant 0.05
        otherwise. An explicit ``convective_updraft_precip_cover`` pins it.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            TiedtkeConvection,
        )

        def cover_flag(physics):
            (conv,) = (t for t in physics.terms
                       if isinstance(t, TiedtkeConvection))
            return conv._updraft_precip_cover

        # Non-JAM ECHAM: the constant 0.05 (flag off).
        self.assertFalse(cover_flag(echam_physics(checkpoint_terms=False)))
        # Pinned on without JAM (the A/B escape hatch).
        self.assertTrue(cover_flag(echam_physics(
            checkpoint_terms=False, convective_updraft_precip_cover=True)))
        # A JAM run defaults the flag on; pinning it off recovers 0.05. Build
        # only the convection term list cheaply via the placeholder core.
        jam_on = echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            radiation_scheme="grey", jam_microphysics="placeholder")
        self.assertTrue(cover_flag(jam_on))
        jam_off = echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            radiation_scheme="grey", jam_microphysics="placeholder",
            convective_updraft_precip_cover=False)
        self.assertFalse(cover_flag(jam_off))

    def test_jam_takes_ham_ice_inhomogeneity(self):
        """JAM (2M + ARG) defaults to ECHAM-HAM's ``zinhomi = 0.7``; every
        other stack keeps ECHAM6's 0.8, and an explicit override wins.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.radiation_types import RadiationParameters

        def zinhomi(physics):
            rad = next(t for t in physics.terms if t.category == "radiation")
            return float(rad.params.get_value().cloud_inhomogeneity_ice)

        jam = dict(checkpoint_terms=False, aerosol_module="jam",
                   cloud_scheme="2m", radiation_scheme="grey",
                   jam_microphysics="placeholder")
        self.assertAlmostEqual(zinhomi(echam_physics(**jam)), 0.7, places=6)
        self.assertAlmostEqual(
            zinhomi(echam_physics(checkpoint_terms=False)), 0.8, places=6)
        self.assertAlmostEqual(zinhomi(echam_physics(
            checkpoint_terms=False, cloud_scheme="2m")), 0.8, places=6)
        explicit = RadiationParameters.default(cloud_inhomogeneity_ice=0.9)
        self.assertAlmostEqual(
            zinhomi(echam_physics(**jam, radiation=explicit)), 0.9, places=6)

    def test_cu_lmfmid_rejects_a_simultaneous_convection_override(self):
        """cu_lmfmid and an explicit convection Parameters are exclusive."""
        from jcm.physics.convection.tiedtke_nordeng import ConvectionParameters
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            echam_physics(
                checkpoint_terms=False,
                cu_lmfmid=False,
                convection=ConvectionParameters.default(),
            )

    def test_echam_physics_accepts_custom_radiation_term(self):
        """A radiation PhysicsTerm can be passed directly."""
        from jcm.physics.echam.echam_terms import echam_physics

        custom_rad = DummyRadiationTerm()
        physics = echam_physics(
            checkpoint_terms=False,
            radiation_scheme=custom_rad,
        )

        rad_term = next(t for t in physics.terms if t.category == "radiation")
        self.assertIs(rad_term, custom_rad)

    def test_echam_physics_rejects_non_radiation_custom_term(self):
        """Custom radiation terms must declare the radiation category."""
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, "category 'radiation'"):
            echam_physics(
                checkpoint_terms=False,
                radiation_scheme=DummyConvectionTerm(),
            )

    def test_column_vector_handles_vmap_scalar_shapes(self):
        """Radiation scalar diagnostics are normalized to [ncols]."""
        from jcm.physics.radiation.grey_two_stream.radiation_scheme import (
            _column_vector,
        )

        self.assertEqual(_column_vector(jnp.arange(3), 3).shape, (3,))
        self.assertEqual(
            _column_vector(jnp.arange(3).reshape(3, 1), 3).shape,
            (3,),
        )

    def test_composable_with_model(self):
        """Composable ECHAM physics works with Model."""
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        composable = echam_physics()
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=composable,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    def test_replace_radiation(self):
        """Can replace radiation with a different scheme."""
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.grey_two_stream import (
            GreyTwoStreamRadiation,
        )

        composable = echam_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)

        # Replace radiation with a fresh instance
        new_rad = GreyTwoStreamRadiation()
        replaced = composable.replace("radiation", new_rad)
        replaced.cache_coords(self.coords)

        tend, _ = replaced.compute_tendencies(self.state, self.forcing, self.terrain,
        )
        # Check shape is correct (NaNs expected with random state)
        self.assertEqual(
            tend.temperature.shape, self.state.temperature.shape,
        )

    def test_nnx_split_merge(self):
        """nnx.split/merge works for ECHAM composable physics."""
        from jcm.physics.echam.echam_terms import echam_physics

        composable = echam_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)

        # Verify split/merge roundtrip works
        graphdef, state = nnx.split(composable)
        restored = nnx.merge(graphdef, state)
        self.assertEqual(len(restored.terms), 13)

class TestAerosolFreeValidation(unittest.TestCase):
    """The mode/interval contract must hold for every radiation scheme.

    Regression for a hole found in adversarial review: the guard used to
    live only in ``RRTMGPRadiation.__init__``, which the grey and emulated
    branches never construct — so a nonsensical interval was accepted in
    silence on exactly the paths that cannot produce *noa fluxes at all.
    """

    def setUp(self):
        from jcm.physics.echam.echam_terms import echam_physics
        self.echam_physics = echam_physics

    def test_interval_is_rejected_on_non_rrtmgp_schemes(self):
        for scheme in ("grey", "emulated"):
            with self.assertRaises(ValueError) as cm:
                self.echam_physics(radiation_scheme=scheme,
                                   aerosol_free_interval=1)
            self.assertIn("radiation_scheme='rrtmgp'", str(cm.exception))

    def test_emulated_composition_carries_the_rrtmgp_bands(self):
        # The per-band emulator expects the RRTMGP band structure
        # (14 SW / 16 LW); the broadband 1-SW/0-LW layout passes
        # composition and then fails the emulator's band-count check at
        # first compute (PR #730 review). Only the Hydra runner path had
        # the emulator in its band selection; the Python factory must too.
        physics = self.echam_physics(radiation_scheme="emulated")
        bc = physics.band_config
        self.assertEqual(len(bc.sw_band_centers_nm), 14)
        self.assertEqual(len(bc.lw_band_centers_nm), 16)

    def test_nonsensical_interval_is_rejected_before_the_scheme_check(self):
        # A meaningless spacing must name the real problem rather than
        # complain about the radiation scheme, which would send the reader
        # down a blind alley.
        with self.assertRaises(ValueError) as cm:
            self.echam_physics(radiation_scheme="grey",
                               aerosol_free_interval=0)
        self.assertIn("must be >= 1", str(cm.exception))


class TestEmulatorWeightsFile(unittest.TestCase):
    """The factory must load TRAINED emulator weights by default (#640 trap).

    ``echam_physics(radiation_scheme="emulated")`` used to build the term with
    random untrained weights, which the scheme's own docs say NaN within a step;
    it now defaults to the packaged trained checkpoint.
    """

    def setUp(self):
        from jcm.physics.echam.echam_terms import echam_physics
        self.echam_physics = echam_physics

    def _rad_term(self, physics):
        return next(t for t in physics.terms
                    if getattr(t, "name", "") == "nn_emulator_radiation")

    def test_default_loads_packaged_trained_weights(self):
        term = self._rad_term(self.echam_physics(radiation_scheme="emulated"))
        # ``_weights_file`` is set only on the load-from-file path (the random
        # init leaves it None), and the packaged default is the u64 checkpoint.
        self.assertIsNotNone(term._weights_file)
        self.assertTrue(
            str(term._weights_file).endswith(
                "emulator_weights_per_band_u64.nc"))

    def test_random_reaches_the_random_init_path(self):
        # The explicit "random" sentinel is the train-from-scratch value:
        # weights_file=None on the term (``_weights_file`` stays None).
        term = self._rad_term(
            self.echam_physics(radiation_scheme="emulated",
                               emulator_weights_file="random"))
        self.assertIsNone(term._weights_file)

    def test_none_falls_back_to_auto(self):
        # None is treated as the "auto" default (an omitted/null config key,
        # which the Hydra builder strips, must NOT reach random init) — it
        # loads the packaged trained checkpoint, exactly like the default.
        term = self._rad_term(
            self.echam_physics(radiation_scheme="emulated",
                               emulator_weights_file=None))
        self.assertIsNotNone(term._weights_file)
        self.assertTrue(
            str(term._weights_file).endswith(
                "emulator_weights_per_band_u64.nc"))

    def test_rejected_on_non_emulated_scheme(self):
        # An explicit value with grey/rrtmgp is a silently-ignored argument —
        # the factory rejects it (same contract as aerosol_free_interval).
        with self.assertRaises(ValueError) as cm:
            self.echam_physics(radiation_scheme="grey",
                               emulator_weights_file="some_ckpt.nc")
        self.assertIn("radiation_scheme='emulated'", str(cm.exception))

    def test_auto_default_does_not_trip_non_emulated_schemes(self):
        # The "auto" default must stay silent for other schemes (it is the
        # unset state, not a user choice).
        self.echam_physics(radiation_scheme="grey")  # no raise


class TestRadiationReadsLaggedConvectionType(unittest.TestCase):
    """End to end through a composed ECHAM stack (#870).

    Step 1 publishes a ``convection`` carry; the radiation term, run on that
    carry with ``ktype`` forced per column to 0 / 2 / 4, must thin the liquid
    cloud only in the ktype-4 columns — the carry → term → scheme path the
    scheme-level tests do not exercise.
    """

    def test_only_shallow_liquid_columns_change(self):
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        physics = echam_physics(radiation_scheme="grey",
                                checkpoint_terms=False)
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        physics.cache_coords(coords)
        nlev = 8
        nlon, nlat = coords.horizontal.nodal_shape
        ncols = nlon * nlat
        qc = jnp.zeros((nlev, nlon, nlat)).at[5:7].set(2e-4)
        state = PhysicsState.zeros(
            (nlev, nlon, nlat),
            temperature=jnp.full((nlev, nlon, nlat), 285.0),
            specific_humidity=jnp.full((nlev, nlon, nlat), 8e-3),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={**{spec.name: jnp.zeros((nlev, nlon, nlat))
                        for spec in physics.required_tracers()}, "qc": qc},
        )
        forcing = ForcingData.zeros((nlon, nlat))
        _, diag = physics.compute_tendencies(
            state, forcing, TerrainData.aquaplanet(coords))

        # Column view of the state and a fixed half-cover liquid cloud where
        # qc sits, so every column carries the same optically thick liquid.
        cols = lambda a: jnp.reshape(a, a.shape[:1] + (ncols,))  # noqa: E731
        state_cols = PhysicsState.zeros(
            (nlev, ncols),
            temperature=cols(state.temperature),
            specific_humidity=cols(state.specific_humidity),
            normalized_surface_pressure=jnp.ones(ncols),
            tracers={"qc": cols(qc), "qi": jnp.zeros((nlev, ncols))},
        )
        clouds = diag["clouds"].copy(
            cloud_fraction=jnp.where(cols(qc) > 0, 0.5, 0.0))
        pattern = jnp.tile(jnp.array([0, 2, 4], jnp.int32),
                           ncols // 3 + 1)[:ncols]
        rad = next(t for t in physics.terms if t.category == "radiation")
        params = rad.params.get_value()

        def solve(ktype):
            d = {**diag, "clouds": clouds,
                 "convection": diag["convection"].replace(ktype=ktype)}
            _, out = rad._compute_full(state_cols, d, forcing, params)
            return (np.asarray(out.sw_heating_rate),        # (nlev, ncols)
                    np.asarray(out.toa_sw_up), np.asarray(out.cos_zenith))

        base, base_up, mu0 = solve(jnp.zeros(ncols, jnp.int32))
        mixed, mixed_up, _ = solve(pattern)
        pattern = np.asarray(pattern)
        self.assertTrue(np.all(np.isfinite(mixed)))
        # 0 and 2 share the 0.8 liquid factor: bit-identical columns.
        np.testing.assert_array_equal(mixed[:, pattern != 4],
                                      base[:, pattern != 4])
        # ktype 4 in daylight: the 0.4 liquid factor thins the cloud, so the
        # column's SW heating changes and less SW is reflected on average.
        # (The grey LW is saturated by this much liquid, hence the SW check;
        # a few grey columns are float32-insensitive to the change.)
        lit4 = (pattern == 4) & (mu0 > 0.1)
        self.assertGreater(int(lit4.sum()), 0)
        changed = np.abs(mixed[:, lit4] - base[:, lit4]).max(axis=0) > 0.0
        self.assertGreater(float(changed.mean()), 0.9)
        self.assertLess(float(mixed_up[lit4].mean()),
                        float(base_up[lit4].mean()))


if __name__ == "__main__":
    unittest.main()
