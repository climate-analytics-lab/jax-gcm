"""Phase 0 tests: population contract + κ-Köhler placeholder core."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import (
    MAM4_SPEC,
    PlaceholderMicrophysics,
    mass_name,
    number_name,
    tracer_specs,
)
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    equilibrium_modal_state,
    saturation_ratio,
)


class PopulationTest(unittest.TestCase):
    def test_mam4_shape(self):
        self.assertEqual(MAM4_SPEC.n_modes(), 4)
        self.assertEqual(
            MAM4_SPEC.mode_names,
            ("accum", "aitken", "coarse", "primary_carbon"),
        )
        self.assertEqual(MAM4_SPEC.family, "modal")

    def test_mode_species_counts_match_mam4(self):
        # nspec_amode = (7, 4, 7, 3)
        self.assertEqual(
            tuple(len(m.species) for m in MAM4_SPEC.modes), (7, 4, 7, 3)
        )

    def test_sigmas(self):
        sig = tuple(m.geom_std_dev for m in MAM4_SPEC.modes)
        self.assertEqual(sig, (1.8, 1.6, 1.8, 1.6))

    def test_primary_carbon_insoluble(self):
        pcm = MAM4_SPEC.mode("primary_carbon")
        self.assertFalse(pcm.soluble)
        self.assertFalse(pcm.can_activate)

    def test_lookup_helpers(self):
        self.assertEqual(MAM4_SPEC.mode_index("aitken"), 1)
        self.assertEqual(MAM4_SPEC.mode_index("cor"), 2)  # by short
        self.assertAlmostEqual(MAM4_SPEC.species_props("ss").hygroscopicity, 1.16)

    def test_bad_family_rejected(self):
        from jcm.physics.aerosol.jam.population import ModalAerosolSpec
        with self.assertRaises(ValueError):
            ModalAerosolSpec(modes=(), species=(), family="bogus")


class TracerLayoutTest(unittest.TestCase):
    def test_spec_count_and_uniqueness(self):
        specs = tracer_specs(MAM4_SPEC)
        # per mode: (1 number + nspec mass) * 2 (interstitial + cloud-borne)
        expected = sum(2 * (1 + len(m.species)) for m in MAM4_SPEC.modes)
        self.assertEqual(len(specs), expected)
        names = [s.name for s in specs]
        self.assertEqual(len(names), len(set(names)))

    def test_number_specs_not_nondimensionalized(self):
        specs = {s.name: s for s in tracer_specs(MAM4_SPEC)}
        self.assertFalse(specs[number_name("acc")].nondimensionalize)
        self.assertTrue(specs[mass_name("so4", "acc")].nondimensionalize)
        # cloud-borne mirror present
        self.assertIn(mass_name("so4", "acc", cloud_borne=True), specs)
        self.assertIn(number_name("acc", cloud_borne=True), specs)


def _uniform_population(nlev=3, ncols=2, mass=1.0e-9, number=1.0e8):
    """Build mass/number dicts with a uniform loading for every mode."""
    masses, numbers = {}, {}
    for mode in MAM4_SPEC.modes:
        numbers[number_name(mode.short)] = jnp.full((nlev, ncols), number)
        for sp in mode.species:
            masses[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), mass)
    return masses, numbers


class PlaceholderPhysicsTest(unittest.TestCase):
    def test_saturation_ratio_clipped(self):
        t = jnp.array([[290.0]])
        q = jnp.array([[0.05]])  # very moist → would exceed 1
        p = jnp.array([[1.0e5]])
        rh = saturation_ratio(t, q, p)
        self.assertTrue(bool(jnp.all(rh <= 0.99)))
        self.assertTrue(bool(jnp.all(rh >= 0.0)))

    def test_equilibrium_radii_in_bounds(self):
        masses, numbers = _uniform_population()
        sat = jnp.full((3, 2), 0.8)
        st = equilibrium_modal_state(masses, numbers, MAM4_SPEC, sat)
        self.assertEqual(st.r_dry.shape, (4, 3, 2))
        # dry radius within [dgnum_lo/2, dgnum_hi/2] per mode
        for i, mode in enumerate(MAM4_SPEC.modes):
            self.assertTrue(bool(jnp.all(st.r_dry[i] >= mode.dgnum_lo / 2 - 1e-12)))
            self.assertTrue(bool(jnp.all(st.r_dry[i] <= mode.dgnum_hi / 2 + 1e-12)))
        # wet >= dry, density positive, kappa in [0, ~1.2]
        self.assertTrue(bool(jnp.all(st.r_wet >= st.r_dry)))
        self.assertTrue(bool(jnp.all(st.rho > 0)))
        self.assertTrue(bool(jnp.all(st.kappa >= 0)))
        self.assertTrue(np.all(np.isfinite(np.asarray(st.r_wet))))

    def test_empty_mode_falls_back_to_reference(self):
        # All-zero loading: radii must be finite and equal the reference.
        masses, numbers = _uniform_population(mass=0.0, number=0.0)
        sat = jnp.full((3, 2), 0.5)
        st = equilibrium_modal_state(masses, numbers, MAM4_SPEC, sat)
        self.assertTrue(np.all(np.isfinite(np.asarray(st.r_dry))))
        for i, mode in enumerate(MAM4_SPEC.modes):
            np.testing.assert_allclose(
                np.asarray(st.r_dry[i]), mode.dgnum / 2, rtol=1e-6
            )

    def test_dry_radius_grows_with_mass(self):
        sat = jnp.full((1, 1), 0.5)
        small, n = _uniform_population(1, 1, mass=1e-10, number=1e8)
        big, _ = _uniform_population(1, 1, mass=1e-8, number=1e8)
        rs = equilibrium_modal_state(small, n, MAM4_SPEC, sat).r_dry
        rb = equilibrium_modal_state(big, n, MAM4_SPEC, sat).r_dry
        self.assertTrue(bool(jnp.all(rb >= rs)))

    def test_jit_and_grad_finite(self):
        masses, numbers = _uniform_population(2, 2)
        sat = jnp.full((2, 2), 0.7)

        def loss(scale):
            scaled = {k: v * scale for k, v in masses.items()}
            st = equilibrium_modal_state(scaled, numbers, MAM4_SPEC, sat)
            return jnp.sum(st.r_wet)

        g = jax.jit(jax.grad(loss))(jnp.array(1.0))
        self.assertTrue(np.isfinite(float(g)))


class PlaceholderTermTest(unittest.TestCase):
    def test_term_smoke(self):
        from jcm.physics_interface import PhysicsState

        nlev, ncols = 4, 3
        term = PlaceholderMicrophysics()
        masses, numbers = _uniform_population(nlev, ncols)
        tracers = {**masses, **numbers}
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 285.0),
            specific_humidity=jnp.full((nlev, ncols), 0.005),
            tracers=tracers,
        )
        diagnostics = {"pressure_full": jnp.full((nlev, ncols), 9.0e4)}
        tend, diag = term(state, diagnostics, None, None)
        # zero tendency
        self.assertTrue(bool(jnp.all(tend.temperature == 0.0)))
        self.assertIn("_jam_state", diag)
        self.assertEqual(diag["_jam_state"].r_dry.shape, (4, nlev, ncols))
        self.assertTrue(
            np.all(np.isfinite(np.asarray(diag["_jam_state"].r_wet)))
        )

    def test_required_tracers_match_population(self):
        term = PlaceholderMicrophysics()
        self.assertEqual(
            set(s.name for s in term.required_tracers()),
            set(s.name for s in tracer_specs(MAM4_SPEC)),
        )


if __name__ == "__main__":
    unittest.main()
