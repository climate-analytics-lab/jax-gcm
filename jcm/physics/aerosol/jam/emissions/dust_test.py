"""Tests for the Tegen/HAMMOZ dust emission scheme (issue #802).

Reference values are hand-computed from the constants of
``mo_ham_dust.f90`` (Marticorena-Bergametti 1995 equations 5-7, 15, 17, 28-33),
not from a saved model run.
"""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import mass_name, number_name
from jcm.physics.aerosol.jam.emissions.surface_wind import MODEL_LEVEL_WIND_KEY
from jcm.physics.aerosol.jam.emissions.dust import (
    ACCUM_UM,
    CD,
    COARSE_UM,
    DSTEP,
    DUST_SUPERCOARSE_KEY,
    HIGH_WIND_MS,
    MIXTURE_ROWS,
    NCLASS,
    SOIL_TYPE_VARS,
    VK,
    ZZ,
    DustEmissions,
    DustParameters,
    emission_weight_matrix,
    soil_diameters,
    soil_size_distributions,
    threshold_friction_velocity,
)

_P = DustParameters.preset(4)
_D = jnp.asarray(soil_diameters())
_UTH = threshold_friction_velocity(_D, _P.a_rnolds, _P.b_rnolds, _P.x_rnolds,
                                   _P.d_thrsld, _P.uth_coeff)
_SREL, _SRELV, _CUM = soil_size_distributions(_P.soil_table, _D)


def _class_for(diameter_um):
    return int(np.argmin(np.abs(np.asarray(_D) * 1.0e4 - diameter_um)))


def _reference_spectrum(flux_type, size_type, u_star, nduscale=0.86, utsc=1.0,
                        feff=1.0):
    """Direct transcription of the Fortran's per-class flux + redistribution.

    Deliberately written as the two nested loops ``bgc_dust_calc_emis`` runs,
    so it is independent of the matrix form the module uses.
    """
    uth = np.asarray(_UTH)
    srel, srel_v, cum = (np.asarray(_SREL), np.asarray(_SRELV), np.asarray(_CUM))
    alpha = float(_P.soil_table[flux_type - 1, 12])
    ratio = uth * nduscale * utsc / (feff * u_star)
    base = np.where(1.0 - ratio > 0.0, (1.0 + ratio) ** 2 * (1.0 - ratio), 0.0)
    fluxdiam = srel[flux_type - 1] * base * CD * u_star ** 3 * alpha
    flux = np.zeros(NCLASS)
    flux[0] += fluxdiam[0]
    js = size_type - 1
    for k in range(1, NCLASS):
        denom = cum[js, k] - srel_v[js, 0]
        flux[:k + 1] += fluxdiam[k] * srel_v[js, :k + 1] / denom
    return flux


def _inputs(nlev=3, ncols=2, u10=9.0, source=1.0, soil=None, psrc=0.0,
            regions=1, snow=0.0, wetness=0.0):
    from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
        VerticalDiffusionData,
    )
    from jcm.physics_interface import PhysicsState

    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 295.0))
    vd = VerticalDiffusionData.zeros((ncols,), nlev).copy(
        wind_10m=jnp.broadcast_to(jnp.asarray(u10, dtype=float),
                                  (ncols,)).astype(float))
    diagnostics = {"air_density": jnp.full((nlev, ncols), 1.2),
                   "layer_thickness": jnp.full((nlev, ncols), 100.0),
                   "vertical_diffusion": vd}
    fractions = {name: jnp.zeros((ncols,)) for name in SOIL_TYPE_VARS}
    for name, value in (soil or {"type2": 1.0}).items():
        fractions[name] = jnp.broadcast_to(
            jnp.asarray(value, dtype=float), (ncols,)).astype(float)
    bc = lambda v: jnp.broadcast_to(  # noqa: E731
        jnp.asarray(v, dtype=float), (ncols,)).astype(float)
    forcing = types.SimpleNamespace(
        dust_source=bc(source), dust_preferential=bc(psrc),
        dust_soil_types=fractions, dust_regions=bc(regions),
        snowc_am=bc(snow), soilw_am=bc(wetness))
    return state, diagnostics, forcing, None


def _total_mass(tend, ncols=2):
    """Return the emitted accumulation + coarse mass flux [kg/m²/s] per column."""
    rho_dz = 1.2 * 100.0
    return np.asarray(
        (tend.tracers[mass_name("du", "acc")][-1]
         + tend.tracers[mass_name("du", "cor")][-1]) * rho_dz)


class ThresholdVelocityTest(unittest.TestCase):
    """MB95 (5)-(7): u*t is a pure function of diameter."""

    def test_hand_computed_values(self):
        expected = {0.2: 1091.0756, 1.0024: 325.6716, 10.0237: 57.8920,
                    20.0: 35.0955, 76.0379: 20.4502, 200.0: 25.2082,
                    502.3773: 39.1337, 1261.9147: 66.3005}
        for diameter, value in expected.items():
            k = _class_for(diameter)
            self.assertAlmostEqual(float(_UTH[k]), value, places=3,
                                   msg=f"D = {diameter} µm")

    def test_reynolds_branch_switches_between_200_and_502_um(self):
        reynolds = (float(_P.a_rnolds) * np.asarray(_D) ** float(_P.x_rnolds)
                    + float(_P.b_rnolds))
        self.assertLess(reynolds[_class_for(200.0)], 10.0)
        self.assertGreater(reynolds[_class_for(502.3773)], 10.0)

    def test_minimum_is_the_umin_pre_gate(self):
        k = int(np.argmin(np.asarray(_UTH)))
        self.assertAlmostEqual(float(_D[k]) * 1.0e4, 76.0379, places=3)
        self.assertAlmostEqual(float(_UTH[k]), 20.4502, places=3)
        # r_dust_umin = 21 cm/s is that minimum rounded up.
        self.assertLess(float(_UTH[k]), float(_P.r_dust_umin))
        self.assertLess(float(_P.r_dust_umin) - float(_UTH[k]), 1.0)


class SoilGridTest(unittest.TestCase):
    def test_grid_spans_191_classes(self):
        d = np.asarray(_D)
        self.assertEqual(d.size, 191)
        self.assertAlmostEqual(d[0] * 1.0e4, 0.2, places=6)
        self.assertAlmostEqual(d[-1] * 1.0e4, 1261.9147, places=3)
        self.assertAlmostEqual(DSTEP, np.log(10.0) / 50.0, places=12)
        # The next class would exceed Dmax, which is why the Fortran's 192nd
        # allocated class stays empty.
        self.assertGreater(d[-1] * np.exp(DSTEP), 0.130)

    def test_relative_surface_and_mass_close(self):
        for row in (1, 2, 3, 4, 6, 10, 11, 13, 14, 15, 16, 17):
            self.assertAlmostEqual(float(jnp.sum(_SREL[row - 1])), 1.0, places=5)
            self.assertAlmostEqual(float(jnp.sum(_SRELV[row - 1])), 1.0, places=5)
        cumulative = np.asarray(_CUM[9])
        self.assertTrue(np.all(np.diff(cumulative) >= 0.0))


class SingleClassFluxTest(unittest.TestCase):
    """MB95 (28)/(33) at one class, against the hand-computed chain."""

    def test_soil_type_2_at_20_um(self):
        k = _class_for(20.0)
        ratio = float(_UTH[k]) * 0.86 / 40.0
        self.assertAlmostEqual(ratio, 0.754553, places=6)
        self.assertAlmostEqual((1.0 + ratio) ** 2, 3.078457, places=6)
        self.assertAlmostEqual(1.0 - ratio, 0.245447, places=6)
        self.assertAlmostEqual(CD, 1.2511918e-6, places=13)
        flux = (float(_SREL[1, k]) * (1.0 + ratio) ** 2 * (1.0 - ratio)
                * CD * 40.0 ** 3 * 4.0e-6)
        self.assertAlmostEqual(flux / 4.4758e-10, 1.0, places=4)


class RedistributionTest(unittest.TestCase):
    """The matrix form must reproduce the Fortran's sandblasting loops exactly."""

    def test_matrix_matches_direct_loops(self):
        weights = np.asarray(emission_weight_matrix(_SREL, _SRELV, _CUM, _D))
        u_star = 40.0
        ratio = np.asarray(_UTH) * 0.86 / u_star
        base = np.where(1.0 - ratio > 0.0,
                        (1.0 + ratio) ** 2 * (1.0 - ratio), 0.0) * CD * u_star ** 3
        d_um = np.asarray(_D) * 1.0e4
        windows = {"acc": (d_um >= ACCUM_UM[0]) & (d_um < ACCUM_UM[1]),
                   "cor": (d_um >= COARSE_UM[0]) & (d_um < COARSE_UM[1]),
                   "all": np.ones_like(d_um, dtype=bool)}
        combos = (("acc", 0), ("acc", 1), ("cor", 0), ("cor", 1), ("all", 0))
        for row, (flux_type, size_type) in enumerate(MIXTURE_ROWS):
            spectrum = _reference_spectrum(flux_type, size_type, u_star)
            alpha = float(_P.soil_table[flux_type - 1, 12])
            for col, (name, power) in enumerate(combos):
                sel = windows[name]
                expected = np.sum(spectrum[sel]
                                  * np.asarray(_D)[sel] ** (-3 * power))
                got = float(np.dot(weights[row, col], base)) * alpha
                np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_weights_sum_slightly_above_one(self):
        # The Fortran's numerator includes class 1 while its denominator does
        # not; reproduced deliberately rather than "fixed".
        js = 1
        k = 100
        denom = float(_CUM[js, k] - _SRELV[js, 0])
        total = float(jnp.sum(_SRELV[js, :k + 1])) / denom
        self.assertGreater(total, 1.0)
        self.assertLess(total, 1.0001)


class WindAndGateTest(unittest.TestCase):
    def test_friction_velocity_from_ten_metre_wind(self):
        # u* = vk·U10/ln(ZZ/z0) with z0 = z0s = 0.001 cm.
        factor = VK * 100.0 / np.log(ZZ / 0.001)
        self.assertAlmostEqual(factor, 2.895297, places=6)

    def test_feff_is_exactly_one_at_the_default_roughness(self):
        term = DustEmissions()
        state, diagnostics, forcing, terrain = _inputs(u10=9.0)
        z0 = term._roughness(forcing, 2, _P)
        np.testing.assert_allclose(np.asarray(z0), float(_P.ndurough))
        feff = 1.0 - np.log(float(_P.ndurough) / float(_P.r_dust_z0s)) / np.log(
            float(_P.aeff) * (float(_P.xeff) / float(_P.r_dust_z0s)) ** 0.8)
        self.assertAlmostEqual(feff, 1.0, places=12)

    def test_emission_onset_at_6_2377_m_per_s(self):
        # u* >= r_dust_umin·nduscale/feff at nduscale = 0.86, feff = 1.
        params = DustParameters.preset(3, truncation=63)
        term = DustEmissions(params=params)
        below, _ = term(*_inputs(u10=6.2377 - 1e-3))
        above, _ = term(*_inputs(u10=6.2377 + 1e-2))
        np.testing.assert_allclose(_total_mass(below), 0.0)
        self.assertTrue(np.all(_total_mass(above) > 0.0))

    def test_potential_source_gates_and_scales(self):
        term = DustEmissions(params=DustParameters.preset(3))
        gated, _ = term(*_inputs(source=1e-12))       # below r_dust_lai = 1e-10
        np.testing.assert_allclose(_total_mass(gated), 0.0)
        half, _ = term(*_inputs(source=0.5))
        full, _ = term(*_inputs(source=1.0))
        np.testing.assert_allclose(_total_mass(half) * 2.0, _total_mass(full),
                                   rtol=1e-6)

    def test_ham2_preset_raises_the_gate_to_a_vegetation_threshold(self):
        self.assertAlmostEqual(float(DustParameters.preset(4).r_dust_lai), 0.1)
        term = DustEmissions()
        gated, _ = term(*_inputs(source=0.05))
        np.testing.assert_allclose(_total_mass(gated), 0.0)


class SnowAndMoistureTest(unittest.TestCase):
    def test_snow_cover_scales_the_flux_and_one_kills_it(self):
        term = DustEmissions()
        dry, _ = term(*_inputs(snow=0.0))
        half, _ = term(*_inputs(snow=0.5))
        buried, _ = term(*_inputs(snow=1.0))
        np.testing.assert_allclose(_total_mass(half) * 2.0, _total_mass(dry),
                                   rtol=1e-6)
        np.testing.assert_allclose(_total_mass(buried), 0.0)

    def test_saturated_soil_emits_nothing_in_every_preset(self):
        for ndust in (2, 3, 4):
            term = DustEmissions(params=DustParameters.preset(ndust))
            wet, _ = term(*_inputs(wetness=1.0))
            np.testing.assert_allclose(_total_mass(wet), 0.0,
                                       err_msg=f"ndust={ndust}")

    def test_damp_soil_still_emits_with_fecan_off(self):
        for ndust in (3, 4):
            term = DustEmissions(params=DustParameters.preset(ndust))
            damp, _ = term(*_inputs(wetness=0.5))
            self.assertTrue(np.all(_total_mass(damp) > 0.0),
                            msg=f"ndust={ndust}")

    def test_fecan_raises_the_threshold_when_switched_on(self):
        # k_dust_smst = 0: uth -> uth·sqrt(1 + 1.21·(w − w_res)^0.68). Off in
        # every preset but ndust=2, and jcm has no ECHAM ws/wsmx (#787).
        params = DustParameters.preset(4).replace(fecan_moisture=True)
        dry, _ = DustEmissions(params=params)(*_inputs(wetness=0.0))
        damp, _ = DustEmissions(params=params)(*_inputs(wetness=0.5))
        self.assertTrue(np.all(_total_mass(dry) > 0.0))
        np.testing.assert_allclose(_total_mass(damp), 0.0)


class PreferentialSourceTest(unittest.TestCase):
    def test_area_weighting_is_exactly_linear(self):
        term = DustEmissions()
        pure_soil, _ = term(*_inputs(psrc=0.0))
        pure_lake, _ = term(*_inputs(psrc=1.0))
        half, _ = term(*_inputs(psrc=0.5))
        np.testing.assert_allclose(
            _total_mass(half), 0.5 * (_total_mass(pure_soil)
                                      + _total_mass(pure_lake)), rtol=1e-6)

    def test_a_lake_without_a_potential_source_emits_nothing(self):
        # The Amazon case: pref = 0.245 but pot_source below r_dust_lai.
        term = DustEmissions()
        tend, _ = term(*_inputs(psrc=0.25, source=0.0))
        np.testing.assert_allclose(_total_mass(tend), 0.0)

    def test_high_wind_switches_the_emitted_spectrum_to_clay(self):
        term = DustEmissions()
        below, _ = term(*_inputs(psrc=1.0, u10=HIGH_WIND_MS - 1e-6))
        above, _ = term(*_inputs(psrc=1.0, u10=HIGH_WIND_MS + 1e-6))

        def submicron_fraction(tend, diag):
            rho_dz = 1.2 * 100.0
            acc = np.asarray(tend.tracers[mass_name("du", "acc")][-1]) * rho_dz
            cor = np.asarray(tend.tracers[mass_name("du", "cor")][-1]) * rho_dz
            return acc / (acc + cor + np.asarray(diag[DUST_SUPERCOARSE_KEY]))

        lo = submicron_fraction(
            below, term(*_inputs(psrc=1.0, u10=HIGH_WIND_MS - 1e-6))[1])
        hi = submicron_fraction(
            above, term(*_inputs(psrc=1.0, u10=HIGH_WIND_MS + 1e-6))[1])
        # Silt (type 10) -> clay (type 11) across the step, at u* = 28.953 cm/s
        # and nduscale = 1.05: a ~3400x jump in sub-micron mass.
        np.testing.assert_allclose(lo, 4.4189e-5, rtol=1e-2)
        np.testing.assert_allclose(hi, 0.151173, rtol=1e-2)


class EastAsiaOverlapTest(unittest.TestCase):
    """The soil file holds two OVERLAPPING partitions; both guards must hold."""

    GOBI = {"type2": 1.0, "type15": 0.679, "type17": 0.172}

    def _residual(self, east_asia):
        params = DustParameters.preset(4).replace(east_asia=east_asia)
        term = DustEmissions(params=params)
        _, _, forcing, _ = _inputs(soil=self.GOBI, psrc=0.3)
        weights, psrc = term._soil_weights(forcing, 2, params)
        return np.asarray(weights[0]), np.asarray(psrc)

    def test_naive_nine_way_sum_would_be_negative(self):
        # The two partitions overlap: the residual 1 − Σ goes to −0.851 here.
        self.assertAlmostEqual(1.0 - sum(self.GOBI.values()), -0.851, places=6)

    def test_both_guarded_branches_keep_the_residual_non_negative(self):
        for mode in (1, 2):
            residual, _ = self._residual(mode)
            self.assertTrue(np.all(residual >= -1e-12), msg=f"k_dust_easo={mode}")

    def test_replacement_branch_zeroes_the_preferential_source(self):
        residual, psrc = self._residual(2)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
        np.testing.assert_allclose(psrc, 0.0)

    def test_east_asian_threshold_scale_takes_the_last_match(self):
        params = DustParameters.preset(4)
        term = DustEmissions(params=params)
        _, _, forcing, _ = _inputs(soil={"type13": 1.0})
        np.testing.assert_allclose(
            np.asarray(term._threshold_scale(forcing, 2, params)), 0.6)
        _, _, both, _ = _inputs(soil={"type13": 1.0, "type17": 1.0})
        np.testing.assert_allclose(
            np.asarray(term._threshold_scale(both, 2, params)), 1.0)


class RegionTuningTest(unittest.TestCase):
    def test_free_running_t63_vector(self):
        np.testing.assert_allclose(
            np.asarray(DustParameters.preset(4, 63).nduscale_reg),
            [1.05, 1.45, 1.45, 1.05, 1.05, 1.05, 1.45, 1.05])
        np.testing.assert_allclose(
            np.asarray(DustParameters.preset(4, 63, nudged=True).nduscale_reg),
            [0.95, 1.25, 1.25, 0.95, 0.95, 0.95, 1.25, 0.95])
        # The ndust=3 polynomial equals 0.86 at T63 and is clamped above it.
        np.testing.assert_allclose(
            float(DustParameters.preset(3, 63).nduscale_reg[0]), 0.86, rtol=1e-4)
        np.testing.assert_allclose(
            float(DustParameters.preset(3, 106).nduscale_reg[0]), 0.86)

    def test_regions_scale_independently(self):
        term = DustEmissions()
        state, diagnostics, forcing, terrain = _inputs(u10=9.0, regions=1)
        forcing.dust_regions = jnp.asarray([4.0, 7.0])   # N Africa, Asia
        tend, _ = term(state, diagnostics, forcing, terrain)
        mass = _total_mass(tend)
        # nduscale scales the THRESHOLD, so region 7's larger value emits less.
        self.assertGreater(mass[0], mass[1])
        uniform = DustParameters.preset(4).replace(
            nduscale_reg=jnp.full((8,), 1.05))
        flat, _ = DustEmissions(params=uniform)(
            state, diagnostics, forcing, terrain)
        np.testing.assert_allclose(_total_mass(flat)[0], _total_mass(flat)[1],
                                   rtol=1e-6)


class EmittedSizeTest(unittest.TestCase):
    def test_mass_and_number_are_consistent_with_the_effective_diameter(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC

        term = DustEmissions()
        tend, _ = term(*_inputs(soil={"type2": 1.0}, u10=13.8))
        rho_dz = 1.2 * 100.0
        density = MAM4_SPEC.species_props("du").density
        expected = {"acc": 0.5698e-6, "cor": 1.9125e-6}
        for short, diameter in expected.items():
            mass = np.asarray(tend.tracers[mass_name("du", short)][-1]) * rho_dz
            number = np.asarray(tend.tracers[number_name(short)][-1]) * rho_dz
            got = (mass / number / (density * np.pi / 6.0)) ** (1.0 / 3.0)
            np.testing.assert_allclose(got, diameter, rtol=2e-3)

    def test_effective_diameter_is_not_the_mode_equilibrium_size(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC

        # MAM4's own volume-median diameters are 0.310 / 5.64 µm; using them
        # would misstate emitted number by 6-20x.
        accum = MAM4_SPEC.mode("acc")
        implied = accum.dgnum * np.exp(3.0 * np.log(accum.geom_std_dev) ** 2)
        self.assertGreater(abs(implied - 0.5698e-6) / 0.5698e-6, 0.3)

    def test_supercoarse_mass_is_reported_and_discarded(self):
        term = DustEmissions()
        tend, diag = term(*_inputs(soil={"type2": 1.0}, u10=13.8))
        rho_dz = 1.2 * 100.0
        emitted = _total_mass(tend)
        discarded = np.asarray(diag[DUST_SUPERCOARSE_KEY])
        self.assertTrue(np.all(discarded > 0.0))
        # Nothing renormalises: HAM discards it too.
        self.assertTrue(np.all(discarded / (emitted + discarded) > 0.2))
        del rho_dz


class GradientTest(unittest.TestCase):
    def _loss(self, params):
        term = DustEmissions(params=params)
        tend, _ = term(*_inputs(soil={"type2": 1.0}, psrc=0.4, u10=9.0))
        return jnp.sum(tend.tracers[mass_name("du", "cor")])

    def test_gradient_through_alpha_and_the_region_vector(self):
        base = DustParameters.preset(4)

        def by_alpha(alpha):
            table = base.soil_table.at[9, 12].set(alpha)    # type 10, paleolake
            return self._loss(base.replace(soil_table=table))

        g = jax.grad(by_alpha)(jnp.asarray(1.0e-5))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreater(float(g), 0.0)

        def by_region(vector):
            return self._loss(base.replace(nduscale_reg=vector))

        gr = jax.grad(by_region)(jnp.asarray(base.nduscale_reg))
        self.assertTrue(np.all(np.isfinite(np.asarray(gr))))
        # Region 1 is the test grid's region and scales the threshold, so more
        # scaling means less emission.
        self.assertLess(float(gr[0]), 0.0)

    def test_the_high_wind_switch_has_zero_gradient(self):
        # Documented limitation, asserted deliberately: the U10 = 10 m/s texture
        # switch is a hard step (#664) and no optimiser can see through it.
        base = DustParameters.preset(4)

        def by_wind(u10):
            state, diagnostics, forcing, terrain = _inputs(psrc=1.0, u10=9.0)
            from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
                VerticalDiffusionData,
            )
            vd = VerticalDiffusionData.zeros((2,), 3).copy(
                wind_10m=jnp.full((2,), u10))
            diagnostics = {**diagnostics, "vertical_diffusion": vd}
            term = DustEmissions(params=base)
            tend, _ = term(state, diagnostics, forcing, terrain)
            return jnp.sum(tend.tracers[mass_name("du", "acc")])

        step = jax.grad(by_wind)(jnp.asarray(HIGH_WIND_MS))
        self.assertTrue(np.isfinite(float(step)))
        lo = float(by_wind(jnp.asarray(HIGH_WIND_MS - 1e-4)))
        hi = float(by_wind(jnp.asarray(HIGH_WIND_MS + 1e-4)))
        # The finite-difference slope across the step is enormous while the
        # analytic derivative sees only the smooth u*³ dependence.
        self.assertGreater((hi - lo) / 2e-4, 1.0e3 * abs(float(step)))


class BroadcastTest(unittest.TestCase):
    def test_column_and_block_agree(self):
        term = DustEmissions()
        single, _ = term(*_inputs(ncols=1, u10=9.0, psrc=0.3))
        block, _ = term(*_inputs(ncols=5, u10=9.0, psrc=0.3))
        key = mass_name("du", "cor")
        np.testing.assert_allclose(
            np.asarray(block.tracers[key][-1]),
            np.full(5, float(single.tracers[key][-1, 0])), rtol=1e-6)

    def test_jit_compiles(self):
        term = DustEmissions()
        state, diagnostics, forcing, terrain = _inputs()
        run = jax.jit(lambda s, d: term(s, d, forcing, terrain)[0])
        tend = run(state, diagnostics)
        self.assertTrue(np.all(np.isfinite(
            np.asarray(tend.tracers[mass_name("du", "cor")]))))


class EmissionWindTest(unittest.TestCase):
    def test_no_diagnosed_ten_metre_wind_means_no_dust(self):
        # wind_10m falls back to the lowest model level (~33 m at L47) on a cold
        # start and without vdiff; feeding that to a 10 m-calibrated threshold
        # and u*^3 would over-emit, so those columns stay off.
        from jcm.physics_interface import PhysicsState
        state = PhysicsState.zeros((3, 2)).copy(
            temperature=jnp.full((3, 2), 295.0),
            u_wind=jnp.full((3, 2), 20.0))
        _, _, forcing, _ = _inputs()
        diagnostics = {"air_density": jnp.full((3, 2), 1.2),
                       "layer_thickness": jnp.full((3, 2), 100.0)}
        tend, diag = DustEmissions()(state, diagnostics, forcing, None)
        np.testing.assert_allclose(_total_mass(tend), 0.0)
        np.testing.assert_allclose(
            np.asarray(diag[MODEL_LEVEL_WIND_KEY]), 1.0)

    def test_a_diagnosed_wind_emits(self):
        tend, diag = DustEmissions()(*_inputs(u10=9.0))
        self.assertTrue(np.all(_total_mass(tend) > 0.0))
        np.testing.assert_allclose(
            np.asarray(diag[MODEL_LEVEL_WIND_KEY]), 0.0)


class InertTest(unittest.TestCase):
    def test_no_forcing_means_no_emission(self):
        from jcm.physics_interface import PhysicsState
        state = PhysicsState.zeros((3, 2)).copy(
            temperature=jnp.full((3, 2), 295.0))
        diagnostics = {"air_density": jnp.full((3, 2), 1.2),
                       "layer_thickness": jnp.full((3, 2), 100.0)}
        tend, _ = DustEmissions()(state, diagnostics, None, None)
        np.testing.assert_allclose(_total_mass(tend), 0.0)

    def test_unsupported_preset_raises(self):
        with self.assertRaisesRegex(ValueError, "ndust=5"):
            DustParameters.preset(5)


if __name__ == "__main__":
    unittest.main()
