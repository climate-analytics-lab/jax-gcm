"""Tests for the Tegen-physics dust emission scheme."""

import dataclasses
import math
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.aerosol.jam import mass_name, number_name
from jcm.physics.aerosol.jam.emissions.dust import (
    DustEmissions,
    DustParameters,
    horizontal_flux,
    mobilization_fraction,
    source_weight,
    wet_threshold_factor,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC


class HorizontalFluxTest(unittest.TestCase):
    def test_zero_below_threshold(self):
        g = horizontal_flux(
            jnp.asarray(0.1), jnp.asarray(0.3), jnp.asarray(1.2), jnp.asarray(1.0)
        )
        self.assertAlmostEqual(float(g), 0.0)

    def test_grows_above_threshold(self):
        lo = horizontal_flux(jnp.asarray(0.4), jnp.asarray(0.2), jnp.asarray(1.2), jnp.asarray(1.0))
        hi = horizontal_flux(jnp.asarray(0.8), jnp.asarray(0.2), jnp.asarray(1.2), jnp.asarray(1.0))
        self.assertGreater(float(hi), float(lo))
        self.assertGreater(float(lo), 0.0)


class SourceGatingTest(unittest.TestCase):
    """CAM ``dust_model.F90`` erodibility handling and CLM ``lnd_frc_mbl``."""

    def test_cam_kind_thresholds_and_does_not_clip(self):
        s = source_weight(jnp.asarray([0.05, 0.1, 0.5, 3.0, -1.0]),
                          "cam_erodibility", jnp.asarray(0.1))
        # Below soil_erod_threshold -> 0; at/above it the factor passes through
        # UNBOUNDED (CAM's basin factor runs to 5.7 in the big source basins).
        np.testing.assert_allclose(np.asarray(s), [0.0, 0.1, 0.5, 3.0, 0.0])

    def test_tegen_kind_is_a_fraction(self):
        s = source_weight(jnp.asarray([0.05, 0.5, 3.0]), "tegen_potential",
                          jnp.asarray(0.1))
        np.testing.assert_allclose(np.asarray(s), [0.05, 0.5, 1.0])

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            DustEmissions(source_kind="mystery_map")

    def test_mobilization_masks_ocean_snow_and_frozen_ground(self):
        # ocean, snow-covered land, frozen land, half-land, bare warm land
        m = mobilization_fraction(
            land_fraction=jnp.asarray([0.0, 1.0, 1.0, 0.5, 1.0]),
            snow_cover=jnp.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
            land_temperature=jnp.asarray([290.0, 275.0, 240.0, 290.0, 290.0]),
            freeze_range=DustParameters.default().freeze_range,
        )
        np.testing.assert_allclose(np.asarray(m), [0.0, 0.0, 0.0, 0.5, 1.0])

    def test_fecan_wetness_raises_threshold_only_above_gwc_threshold(self):
        scale = float(DustParameters.default().soil_water_gwc_scale)
        f = wet_threshold_factor(jnp.asarray([0.0, 0.1, 1.0]),
                                 jnp.asarray(0.04), jnp.asarray(scale))
        # 0.1 of the index -> gwc 0.02 < 0.04: still the dry threshold.
        np.testing.assert_allclose(np.asarray(f[:2]), [1.0, 1.0])
        expect = math.sqrt(1.0 + 1.21 * (100.0 * (scale - 0.04)) ** 0.68)
        self.assertAlmostEqual(float(f[2]), expect, places=4)

    def test_gwc_scale_is_field_capacity_not_porosity(self):
        """soilw_am saturates at field capacity, so the mapping must too."""
        # swcap = 0.30 vol over a (1 - watsat)*2700 = 1485 kg/m3 bulk density.
        self.assertAlmostEqual(
            float(DustParameters.default().soil_water_gwc_scale),
            0.30 * 1000.0 / 1485.0, places=3)

    def test_fecan_gradient_finite_at_the_threshold(self):
        scale = float(DustParameters.default().soil_water_gwc_scale)
        g = jax.grad(lambda w: jnp.sum(wet_threshold_factor(
            w, jnp.asarray(0.04), jnp.asarray(scale))))(
                jnp.asarray([0.04 / scale, 0.5]))
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))


def _inputs(nlev=3, ncols=2, u_star=0.6, source=0.5, land=1.0, snow=0.0,
            soilw=0.0, land_temp=295.0):
    from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
        VerticalDiffusionData,
    )

    def col(v):
        return jnp.full((ncols,), v) if np.isscalar(v) else jnp.asarray(v)

    state = __import__(
        "jcm.physics_interface", fromlist=["PhysicsState"]
    ).PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 295.0),
    )
    vd = VerticalDiffusionData.zeros((ncols,), nlev).copy(
        surface_friction_velocity=col(u_star),
    )
    diagnostics = {
        "air_density": jnp.full((nlev, ncols), 1.2),
        "layer_thickness": jnp.full((nlev, ncols), 100.0),
        "vertical_diffusion": vd,
    }
    forcing = types.SimpleNamespace(
        dust_source=col(source), snowc_am=col(snow), soilw_am=col(soilw),
        stl_am=col(land_temp),
    )
    terrain = types.SimpleNamespace(fmask=col(land))
    return state, diagnostics, forcing, terrain


class DustTermTest(unittest.TestCase):
    def test_emits_with_source_and_wind(self):
        term = DustEmissions()
        tend, _ = term(*_inputs(u_star=0.6, source=0.8))
        key = mass_name("du", "cor")
        self.assertGreater(float(tend.tracers[key][-1, 0]), 0.0)

    def test_zero_without_source(self):
        term = DustEmissions()
        tend, _ = term(*_inputs(source=0.0))
        key = mass_name("du", "cor")
        self.assertAlmostEqual(float(tend.tracers[key][-1, 0]), 0.0)

    def test_zero_below_threshold(self):
        term = DustEmissions()  # default u_threshold 0.2
        tend, _ = term(*_inputs(u_star=0.1, source=0.8))
        key = mass_name("du", "cor")
        self.assertAlmostEqual(float(tend.tracers[key][-1, 0]), 0.0)

    def test_gated_map_emits_only_from_the_bare_warm_land_cell(self):
        """Antarctic, ocean, sub-threshold and >1 erodibility cells (#768)."""
        state, diagnostics, forcing, terrain = _inputs(
            ncols=4, u_star=0.8,
            source=[1.0, 1.0, 0.05, 3.0],      # antarctic, ocean, sub-thr, basin
            land=[1.0, 0.0, 1.0, 1.0],
            snow=0.0,
            land_temp=[235.0, 290.0, 290.0, 290.0],
        )
        tend, _ = DustEmissions()(state, diagnostics, forcing, terrain)
        flux = np.asarray(tend.tracers[mass_name("du", "cor")][-1])
        np.testing.assert_allclose(flux[:3], 0.0)
        self.assertGreater(flux[3], 0.0)
        # The >1 basin factor is NOT clipped: 3.0 emits 3x a unit-source cell.
        unit, _ = DustEmissions()(*_inputs(ncols=4, u_star=0.8, source=1.0,
                                           land=1.0, land_temp=290.0))
        self.assertAlmostEqual(
            flux[3] / float(unit.tracers[mass_name("du", "cor")][-1, 3]),
            3.0, places=4)

    def test_mode_split_and_number_match_cam(self):
        """CAM ``dust_emis_sclfctr`` and ``x_mton = 6/(pi rho D_vwr^3)`` (#768)."""
        tend, _ = DustEmissions()(*_inputs(u_star=0.8, source=1.0))
        acc = float(tend.tracers[mass_name("du", "acc")][-1, 0])
        cor = float(tend.tracers[mass_name("du", "cor")][-1, 0])
        self.assertAlmostEqual(acc / (acc + cor), 0.021, places=6)
        self.assertAlmostEqual(cor / (acc + cor), 0.979, places=6)

        rho = MAM4_SPEC.species_props("du").density
        p = DustParameters.default()
        for mode, mass, diameter in (("acc", acc, p.emission_diameter[0]),
                                     ("cor", cor, p.emission_diameter[1])):
            x_mton = 6.0 / (math.pi * rho * float(diameter) ** 3)
            self.assertAlmostEqual(
                float(tend.tracers[number_name(mode)][-1, 0]) / mass / x_mton,
                1.0, places=4)

    def test_cam_emission_diameters(self):
        p = DustParameters.default()
        np.testing.assert_allclose(np.asarray(p.emission_diameter),
                                   [0.7806e-6, 3.8983e-6], rtol=1e-4)
        # ~1.5e15 #/kg for accumulation dust, not the 1.2e17 the mode's
        # equilibrium size would give. The density is jcm's own 2600 (MAM4
        # specdens_dust), deliberately, not CAM dust_model's mo_constants
        # 2500 — mass and number must use one density inside this model, and
        # the 3.8 % difference is a CAM-internal inconsistency.
        rho = MAM4_SPEC.species_props("du").density
        self.assertAlmostEqual(
            6.0 / (math.pi * rho * float(p.emission_diameter[0]) ** 3) / 1.5445e15,
            1.0, places=3)

    def test_population_split_matches_the_term_default(self):
        (fine, fine_frac), (coarse, coarse_frac) = MAM4_SPEC.primary_split("du")
        self.assertEqual((fine.short, coarse.short), ("acc", "cor"))
        self.assertAlmostEqual(fine_frac,
                               float(DustParameters.default().accum_fraction))
        self.assertAlmostEqual(fine_frac + coarse_frac, 1.0)

    def test_column_and_batch_agree(self):
        """One column and a batch of columns give the same per-column flux."""
        single, _ = DustEmissions()(*_inputs(ncols=1, u_star=0.7, source=0.8,
                                             soilw=0.3))
        batch, _ = DustEmissions()(*_inputs(ncols=5, u_star=0.7, source=0.8,
                                            soilw=0.3))
        key = mass_name("du", "cor")
        np.testing.assert_allclose(
            np.asarray(batch.tracers[key][-1]),
            np.full(5, float(single.tracers[key][-1, 0])), rtol=1e-6)

    def test_wet_soil_suppresses_emission(self):
        dry, _ = DustEmissions()(*_inputs(u_star=0.5, source=1.0, soilw=0.0))
        wet, _ = DustEmissions()(*_inputs(u_star=0.5, source=1.0, soilw=1.0))
        key = mass_name("du", "cor")
        self.assertGreater(float(dry.tracers[key][-1, 0]),
                           float(wet.tracers[key][-1, 0]))

    def test_tegen_source_kind_is_a_fraction_on_its_own_map(self):
        """Each kind on the map it is for — not the CAM map read as a fraction.

        A [0, 1] potential-source map is used as-is by ``tegen_potential`` and
        thresholded by ``cam_erodibility``; that difference is the point of the
        option. Feeding the CAM map (max 5.7) to ``tegen_potential`` would clip
        the basins #768 exists to keep, which is a misconfiguration
        ``warn_on_config_traps`` now catches rather than something to pin here.
        """
        args = _inputs(u_star=0.8, source=0.5)
        key = mass_name("du", "cor")
        tegen, _ = DustEmissions(source_kind="tegen_potential")(*args)
        cam, _ = DustEmissions()(*args)          # threshold 0.1 < 0.5: passes
        self.assertAlmostEqual(
            float(cam.tracers[key][-1, 0]) / float(tegen.tracers[key][-1, 0]),
            1.0, places=6)
        # Below CAM's threshold the two genuinely differ: CAM zeroes, Tegen
        # keeps the fraction.
        sub_args = _inputs(u_star=0.8, source=0.05)
        self.assertAlmostEqual(
            float(DustEmissions()(*sub_args)[0].tracers[key][-1, 0]), 0.0)
        self.assertGreater(
            float(DustEmissions(source_kind="tegen_potential")(*sub_args)[0]
                  .tracers[key][-1, 0]), 0.0)

    def test_grad_through_alpha(self):
        state, diagnostics, forcing, terrain = _inputs()
        base = DustParameters.default()

        def loss(alpha):
            term = DustEmissions(params=dataclasses.replace(base, alpha=alpha))
            tend, _ = term(state, diagnostics, forcing, terrain)
            return jnp.sum(tend.tracers[mass_name("du", "cor")])

        g = jax.grad(loss)(jnp.asarray(1.0e-5))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreater(float(g), 0.0)

    def test_grad_through_gating_and_emission_size(self):
        """The new knobs are differentiable leaves, not static config.

        ``source_threshold`` is the exception by construction: it appears only
        as a ``jnp.where`` predicate (CAM's hard cut-off), so its gradient is
        identically zero and only finiteness is asserted for it.
        """
        state, diagnostics, forcing, terrain = _inputs(u_star=0.8, source=1.0,
                                                       soilw=0.5)
        base = DustParameters.default()

        def loss(threshold, gwc_thr, diameter, gwc_scale, freeze_range):
            p = dataclasses.replace(
                base, source_threshold=threshold,
                soil_moisture_threshold=gwc_thr, emission_diameter=diameter,
                soil_water_gwc_scale=gwc_scale, freeze_range=freeze_range)
            tend, _ = DustEmissions(params=p)(state, diagnostics, forcing,
                                              terrain)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
            jnp.asarray(0.1), jnp.asarray(0.04),
            jnp.asarray([0.7806e-6, 3.8983e-6]), jnp.asarray(0.202),
            jnp.asarray(2.0))
        for leaf in jax.tree_util.tree_leaves(g):
            self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))
        # The wetness threshold, the emission size and the gwc scale move the
        # answer; the freezing ramp does not here (the column is warm).
        self.assertGreater(abs(float(g[1])), 0.0)
        self.assertTrue(np.any(np.abs(np.asarray(g[2])) > 0.0))
        self.assertGreater(abs(float(g[3])), 0.0)

    def test_no_gating_fields_still_emits(self):
        """Absent boundary fields (aquaplanet/unit tests) fall back to bare land."""
        state, diagnostics, _, _ = _inputs(u_star=0.8)
        forcing = types.SimpleNamespace(dust_source=jnp.full((2,), 1.0))
        tend, _ = DustEmissions()(state, diagnostics, forcing, None)
        self.assertGreater(float(tend.tracers[mass_name("du", "cor")][-1, 0]), 0.0)

    def test_freezing_ramp_uses_the_melting_point(self):
        m = mobilization_fraction(jnp.asarray(1.0), jnp.asarray(0.0),
                                  jnp.asarray(c.tmelt),
                                  DustParameters.default().freeze_range)
        self.assertAlmostEqual(float(m), 1.0)


if __name__ == "__main__":
    unittest.main()
