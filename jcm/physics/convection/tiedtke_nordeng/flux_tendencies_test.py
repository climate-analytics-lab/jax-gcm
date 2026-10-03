"""Tests for the convective precipitation flux/tendency helpers.

Focus: the sub-cloud rain-evaporation precipitation cover selected by the
HAM submodel (jax-gcm#812). ECHAM ``cuflx`` uses the updraft area
``pmfu/(zwu·zrhou)`` under ``lham`` and the constant ``zcucov = 0.05``
otherwise (``mo_cufluxdts.f90:414-420``); :func:`updraft_area_cover` is the
shared implementation of that area (the JAM convective washout uses the same
function) and :func:`convective_precip_fluxes` routes its Kessler
evaporation through it when ``use_updraft_cover`` is set.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.flux_tendencies import (
    calculate_tendencies,
    cevapcu_profile,
    convective_precip_fluxes,
    updraft_area_cover,
)
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    saturation_mixing_ratio,
)
from jcm.physics.convection.tiedtke_nordeng.types import (
    ECHAM_CEVAPCU_COEFFICIENT,
    ConvectionParameters,
)
from jcm.testing import check_gradients


class TestUpdraftAreaCover(unittest.TestCase):
    """The updraft-area cover ``pmfu/(zwu·rho)`` with ECHAM's sub-cloud taper."""

    def setUp(self):
        # Top-first: a plume occupying levels 1-2, cloud base at level 2,
        # nothing below in the raw flux (the taper is rebuilt).
        self.mfu = jnp.array([0.0, 0.2, 0.2, 0.0])
        self.layer = jnp.full((4,), 200.0)   # equal air mass per layer
        self.rho = jnp.ones((4,))
        self.w_u = 2.0

    def test_deep_column_matches_hand_computed_area(self):
        # In-plume levels: 0.2/(2·1) = 0.1. Level 3 sits below cloud base
        # (level 2) with half the air mass below its top interface (200 of
        # 400), so the linear taper zzp = 0.5 gives 0.2·0.5/(2·1) = 0.05.
        cover = np.asarray(updraft_area_cover(
            self.mfu, self.rho, jnp.array(1), self.layer, self.w_u))
        np.testing.assert_allclose(cover, [0.0, 0.1, 0.1, 0.05], rtol=1e-6)

    def test_midlevel_taper_is_squared(self):
        # ktype == 3 squares the sub-cloud pressure ratio (0.5² = 0.25).
        cover = np.asarray(updraft_area_cover(
            self.mfu, self.rho, jnp.array(3), self.layer, self.w_u))
        np.testing.assert_allclose(cover, [0.0, 0.1, 0.1, 0.025], rtol=1e-6)

    def test_no_plume_gives_zero_cover(self):
        cover = updraft_area_cover(
            jnp.zeros((4,)), self.rho, jnp.array(1), self.layer, self.w_u)
        np.testing.assert_array_equal(np.asarray(cover), 0.0)

    def test_velocity_is_floored_not_divided_by_zero(self):
        # A zero assumed updraft speed falls back to the 0.01 m/s floor
        # rather than producing an infinite area.
        cover = np.asarray(updraft_area_cover(
            self.mfu, self.rho, jnp.array(1), self.layer, 0.0))
        self.assertTrue(np.all(np.isfinite(cover)))
        np.testing.assert_allclose(cover[1], 0.2 / (0.01 * 1.0), rtol=1e-6)

    def test_not_clipped_unlike_the_wetdep_cover(self):
        # cuflx applies NO clamp; an area above the whole box is returned
        # raw (the wet-deposition caller clips, the evaporation does not).
        huge = jnp.array([0.0, 10.0, 10.0, 0.0])
        cover = np.asarray(updraft_area_cover(
            huge, self.rho, jnp.array(1), self.layer, self.w_u))
        self.assertGreater(cover[1], 1.0)

    def test_taper_on_the_real_l47_grid(self):
        # ECHAM's sub-cloud taper is a ratio of HALF-LEVEL pressure
        # differences: zzp = (p_s - p_half(jk)) / (p_s - p_half(kcbot))
        # (mo_cufluxdts.f90:236). Reconstructed here by cumulatively
        # summing the layer weights, so the weights must be the true
        # half-level Δp of the production hybrid grid — the dual-grid
        # centre-to-centre spacing of the tendency ledger accumulates a
        # systematic p_s - p_half error wherever adjacent thicknesses
        # differ, which on L47 is everywhere.
        from jcm.physics.echam.echam_levels import get_echam_levels

        levels = get_echam_levels(47)
        ps = 101325.0
        phalf = np.asarray(levels.a_boundaries) + np.asarray(
            levels.b_boundaries) * ps          # (48,), TOA-first
        dp = jnp.asarray(np.diff(phalf))       # true layer Δp (47,)
        nlev = dp.shape[0]

        kbase = 40                             # ~793 hPa cloud base
        mfu = jnp.where(jnp.arange(nlev) <= kbase, 0.0, 0.0)
        mfu = mfu.at[35:kbase + 1].set(0.1)    # plume levels 35..40
        rho = jnp.ones((nlev,))
        w_u = 2.0

        cover = np.asarray(updraft_area_cover(
            mfu, rho, jnp.array(1), dp, w_u))

        # Hand-computed reference straight from the half-level pressures.
        zzp = (ps - phalf[:-1]) / (ps - phalf[kbase])
        expected = np.where(np.arange(nlev) > kbase,
                            0.1 * zzp / (w_u * 1.0), 0.0)
        expected[35:kbase + 1] = 0.1 / (w_u * 1.0)
        np.testing.assert_allclose(cover, expected, rtol=1e-5)

        # The dual-grid tendency spacing (centre-to-centre full-level
        # diffs, last value duplicated) is NOT a valid weight: on this
        # grid it visibly distorts the sub-cloud taper.
        pfull = 0.5 * (phalf[:-1] + phalf[1:])
        dp_dual = np.abs(np.diff(pfull))
        dp_dual = jnp.asarray(np.concatenate([dp_dual, dp_dual[-1:]]))
        cover_dual = np.asarray(updraft_area_cover(
            mfu, rho, jnp.array(1), dp_dual, w_u))
        sub = np.arange(nlev) > kbase
        self.assertGreater(
            float(np.max(np.abs(cover_dual[sub] - expected[sub])
                         / expected[sub])), 0.01)

    def test_broadcasting_native_column_equals_block(self):
        block_mfu = jnp.stack([self.mfu, self.mfu * 0.5], axis=1)
        block_rho = jnp.ones((4, 2))
        block_layer = jnp.full((4, 2), 200.0)
        ktype = jnp.array([1, 3])
        block = np.asarray(updraft_area_cover(
            block_mfu, block_rho, ktype, block_layer, self.w_u))
        col0 = np.asarray(updraft_area_cover(
            block_mfu[:, 0], block_rho[:, 0], ktype[0], block_layer[:, 0],
            self.w_u))
        np.testing.assert_allclose(block[:, 0], col0, rtol=1e-6)


class TestSubCloudEvaporationCover(unittest.TestCase):
    """Routing the cuflx Kessler evaporation through the two cover choices."""

    def setUp(self):
        # Warm column (all rain), dry sub-cloud air so qs − q > 0 drives
        # evaporation; precip generated in the plume (levels 1-2) then falls
        # through the sub-cloud levels 3-4 below cloud base (kbase = 2).
        self.temperature = jnp.array([278.0, 282.0, 286.0, 290.0, 294.0])
        self.humidity = jnp.full((5,), 1.0e-3)
        self.pressure = jnp.array([2.0e4, 4.0e4, 6.0e4, 8.0e4, 1.0e5])
        self.dp_lev = jnp.full((5,), 2.0e4)
        self.kbase = 2
        self.pdmfup = jnp.array([0.0, 2.0e-3, 2.0e-3, 0.0, 0.0])
        self.pdmfdp = jnp.zeros((5,))
        self.dt = 1800.0
        self.mfu = jnp.array([0.0, 0.05, 0.05, 0.0, 0.0])
        self.tu = jnp.array([0.0, 282.0, 286.0, 0.0, 0.0])

    def _run(self, use_updraft_cover, mfu=None, tu=None):
        return convective_precip_fluxes(
            self.temperature, self.humidity, self.pressure, self.dp_lev,
            self.kbase, self.pdmfup, self.pdmfdp, self.dt,
            updraft_temperature=self.tu if tu is None else tu,
            updraft_mass_flux=self.mfu if mfu is None else mfu,
            ktype=jnp.array(1),
            updraft_velocity=2.0,
            use_updraft_cover=use_updraft_cover,
            # Uniform column: the true layer mass Δp/g coincides with the
            # ledger spacing here, so the cover values are unaffected by
            # the weight choice (which only matters on stretched grids —
            # see TestUpdraftAreaCover.test_taper_on_the_real_l47_grid).
            updraft_layer_mass=self.dp_lev / 9.80665,
        )

    def test_evaporation_depletes_surface_rain_and_stays_finite(self):
        total_gen = float(jnp.sum(self.pdmfup))
        for flag in (False, True):
            rain_sfc, snow_sfc, _, _, pdmfup_adj, precip_flux, _, _ = self._run(flag)
            self.assertTrue(np.all(np.isfinite(np.asarray(rain_sfc))))
            self.assertTrue(np.all(np.isfinite(np.asarray(pdmfup_adj))))
            self.assertTrue(np.all(np.isfinite(np.asarray(precip_flux))))
            # Some rain evaporated below cloud base, none created.
            self.assertLess(float(rain_sfc), total_gen)
            self.assertGreaterEqual(float(rain_sfc), 0.0)
            self.assertAlmostEqual(float(snow_sfc), 0.0)

    def test_evap_fraction_is_ham_prevap(self):
        """HAMMOZ ``prevap``: the fraction of the falling precip evaporated.

        Below the precipitating levels no precip forms, so ``prevap`` of a
        sub-cloud layer is the fraction of the flux entering it that does
        not leave through its bottom (``zfevapr/zfrain`` of cuflx and
        ``prep_wetdep_hydro``), and it is zero where nothing evaporates.
        """
        for flag in (False, True):
            out = self._run(flag)
            flux, evap = np.asarray(out[5]), np.asarray(out[7])
            rain_sfc = float(out[0])
            below = np.append(flux[1:], rain_sfc)
            for k in (3, 4):
                np.testing.assert_allclose(
                    evap[k], (flux[k] - below[k]) / flux[k], rtol=1e-5)
            self.assertGreater(float(evap[3]), 0.0)
            np.testing.assert_array_equal(evap[:2], 0.0)
            self.assertTrue(np.all((evap >= 0.0) & (evap <= 1.0)))

    def test_non_ham_path_ignores_the_updraft_flux(self):
        # With the constant 0.05 cover the updraft fields are unused, so the
        # result must not depend on them (proves the switch is off).
        base = self._run(False)
        perturbed = self._run(
            False, mfu=jnp.array([0.0, 0.9, 0.9, 0.0, 0.0]),
            tu=jnp.array([0.0, 260.0, 260.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            np.asarray(base[0]), np.asarray(perturbed[0]), rtol=1e-12)

    def test_floor_source_is_the_water_the_flux_floor_creates(self):
        """Downdraft uptake beyond the plume's rain: the floor's created water.

        ``cuflx`` floors the rain flux at zero while ``cudtdq`` charges the
        unfloored ``pdmfup + pdmfdp`` to the vapour, so the column loses
        ``precip - floor_source`` of water (#912). With no downdraft the
        source is exactly zero; with an uptake exceeding the generated rain by
        ``d`` it is ``d`` and the surface rain is zero.
        """
        self.assertEqual(float(self._run(False)[6]), 0.0)
        excess = 1.5e-3
        uptake = -(jnp.sum(self.pdmfup) + excess)
        original = self.pdmfdp
        try:
            self.pdmfdp = jnp.array([0.0, 0.0, uptake, 0.0, 0.0])
            out = self._run(False)
        finally:
            self.pdmfdp = original
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[6]), excess, places=9)

    def test_ham_and_non_ham_covers_give_different_evaporation(self):
        # The updraft area here (≈0.025 at the surface) differs from 0.05,
        # so the evaporated amount — and the surface rain — must differ.
        rain_false = float(self._run(False)[0])
        rain_true = float(self._run(True)[0])
        self.assertNotAlmostEqual(rain_false, rain_true, places=7)

    def test_gradients_finite_including_zero_updraft_flux(self):
        def loss(w_u, temperature, humidity, mfu, tu):
            rain_sfc, *_ = convective_precip_fluxes(
                temperature, humidity, self.pressure, self.dp_lev,
                self.kbase, self.pdmfup, self.pdmfdp, self.dt,
                updraft_temperature=tu, updraft_mass_flux=mfu,
                ktype=jnp.array(1), updraft_velocity=w_u,
                use_updraft_cover=True,
                updraft_layer_mass=self.dp_lev / 9.80665)
            return jnp.sum(rain_sfc)

        grad = jax.grad(loss, argnums=(0, 1, 2))
        # Healthy plume: every gradient finite.
        g = grad(2.0, self.temperature, self.humidity, self.mfu, self.tu)
        for leaf in jax.tree_util.tree_leaves(g):
            self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))
        # Degenerate column — zero updraft flux drives the cover (and its
        # zwu·ρ_u denominators) to zero everywhere; the discarded-lane
        # substitution must keep every gradient finite (jax-gcm#558/#559).
        zeros = jnp.zeros((5,))
        g0 = grad(2.0, self.temperature, self.humidity, zeros, zeros)
        for leaf in jax.tree_util.tree_leaves(g0):
            self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))
        # No plume ⇒ no rain shaft ⇒ the assumed updraft speed is inert.
        self.assertEqual(float(g0[0]), 0.0)


class TestTaperWeightPlumbing(unittest.TestCase):
    """``calculate_tendencies`` feeds the cover the unfloored layer mass."""

    def _run(self, layer_mass, layer_thickness):
        import jcm.constants as c
        from jcm.physics.convection.tiedtke_nordeng.downdraft import (
            DowndraftState,
        )
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            ConvectionParameters,
        )
        from jcm.physics.convection.tiedtke_nordeng.updraft import (
            UpdatedraftState,
        )

        temperature = jnp.array([278.0, 282.0, 286.0, 290.0, 294.0])
        humidity = jnp.full((5,), 1.0e-3)
        pressure = jnp.array([2.0e4, 4.0e4, 6.0e4, 8.0e4, 1.0e5])
        rho = pressure / (c.rd * temperature)
        zeros = jnp.zeros((5,))
        updraft = UpdatedraftState(
            tu=jnp.array([0.0, 282.0, 286.0, 0.0, 0.0]),
            qu=zeros, lu=zeros,
            mfu=jnp.array([0.0, 0.05, 0.05, 0.0, 0.0]),
            entr=zeros, detr=zeros, buoy=zeros,
            pdmfup=jnp.array([0.0, 2.0e-3, 2.0e-3, 0.0, 0.0]),
            plude=zeros,
            uu=zeros, vu=zeros,
        )
        downdraft = DowndraftState(
            td=temperature, qd=zeros, mfd=zeros, pdmfdp=zeros,
            ud=zeros, vd=zeros,
            lfs=0, active=False,
        )
        return calculate_tendencies(
            temperature, humidity, zeros, zeros, pressure, rho,
            layer_thickness, updraft, downdraft, kbase=2, ktop=1,
            dt=1800.0, config=ConvectionParameters.default(),
            ktype=jnp.array(1), use_updraft_cover=True,
            layer_mass=layer_mass,
        )

    def test_layer_mass_wins_over_the_thickness_product(self):
        # The taper weight must be the SUPPLIED unfloored Δp/g, not
        # ρ·layer_thickness: with a deliberately wrong (floored-style)
        # thickness, supplying the true mass must still give the same
        # precipitation as the internally consistent column ...
        import jcm.constants as c

        temperature = jnp.array([278.0, 282.0, 286.0, 290.0, 294.0])
        pressure = jnp.array([2.0e4, 4.0e4, 6.0e4, 8.0e4, 1.0e5])
        rho = pressure / (c.rd * temperature)
        mass_true = jnp.full((5,), 2.0e4) / c.grav
        dz_true = mass_true / rho
        a = self._run(mass_true, dz_true)
        b = self._run(mass_true, jnp.full((5,), 10.0))
        np.testing.assert_allclose(
            float(a.precip_conv), float(b.precip_conv), rtol=1e-12)
        # ... and the fallback (layer_mass=None) reproduces ρ·Δz exactly,
        # which coincides with the truth only when Δz is unfloored.
        fb = self._run(None, dz_true)
        np.testing.assert_allclose(
            float(a.precip_conv), float(fb.precip_conv), rtol=1e-12)

    def test_a_different_layer_mass_changes_the_sub_cloud_cover(self):
        # Sanity that the weight is live: a non-uniform mass profile below
        # cloud base changes the taper, the cover, and hence the surface
        # precipitation after sub-cloud evaporation.
        import jcm.constants as c

        temperature = jnp.array([278.0, 282.0, 286.0, 290.0, 294.0])
        pressure = jnp.array([2.0e4, 4.0e4, 6.0e4, 8.0e4, 1.0e5])
        rho = pressure / (c.rd * temperature)
        mass_true = jnp.full((5,), 2.0e4) / c.grav
        dz_true = mass_true / rho
        a = self._run(mass_true, dz_true)
        skewed = mass_true * jnp.array([1.0, 1.0, 1.0, 3.0, 3.0])
        s = self._run(skewed, dz_true)
        self.assertNotAlmostEqual(
            float(a.precip_conv), float(s.precip_conv), places=9)


class TestCevapcuProfile(unittest.TestCase):
    """ECHAM's ``cevapcu(jk)`` (iniphy.f90:87-89) and its leading coefficient."""

    ETA = jnp.array([1.0e-5, 1.0e-3, 0.05, 0.3, 0.7, 0.99, 1.0])

    @staticmethod
    def _iniphy(eta):
        """Evaluate iniphy.f90:87-89 as transcribed, with the literal ``1.93E-6``."""
        return (
            1.93e-6 * 261.0
            * jnp.sqrt(1.0e3 / (38.3 * 0.293) * jnp.sqrt(jnp.clip(eta, 1e-4, 1.0)))
            * 0.5 / c.grav
        )

    def test_default_is_iniphys_expression_exactly(self):
        # Parameterising the coefficient must not move a bit of the default
        # profile. ``ConvectionParameters`` carries it as an array leaf, so
        # the equality has to hold for that form as well as for the float, in
        # both precisions the model runs in. Whether the transcription agrees
        # with compiled ECHAM is the job of the cumastr reference data
        # (``jcm/data/test/echam_cumastr_reference``, whose sub-cloud
        # evaporation columns run the profile), not of this comparison.
        for x64 in (False, True):
            with jax.enable_x64(x64):
                eta = self.ETA.astype(jnp.float64 if x64 else jnp.float32)
                want = np.asarray(self._iniphy(eta))
                leaf = jnp.asarray(ConvectionParameters.default().cevapcu)
                for got in (cevapcu_profile(eta),
                            cevapcu_profile(eta, ECHAM_CEVAPCU_COEFFICIENT),
                            cevapcu_profile(eta, leaf)):
                    np.testing.assert_array_equal(np.asarray(got), want)

    def test_coefficient_scales_the_whole_level_dependent_profile(self):
        base = np.asarray(cevapcu_profile(self.ETA))
        doubled = np.asarray(
            cevapcu_profile(self.ETA, 2.0 * ECHAM_CEVAPCU_COEFFICIENT))
        # A power-of-two factor is exact, so the ratio is 2 at every level.
        np.testing.assert_array_equal(doubled, 2.0 * base)
        # The shape stays ECHAM's: it grows with eta, and the clip at the top
        # keeps it positive.
        self.assertTrue(np.all(np.diff(base) > 0.0))
        self.assertGreater(float(base[0]), 0.0)

    def test_default_parameter_is_echams_coefficient(self):
        self.assertEqual(ECHAM_CEVAPCU_COEFFICIENT, 1.93e-6)
        self.assertEqual(
            float(ConvectionParameters.default().cevapcu),
            float(np.float32(ECHAM_CEVAPCU_COEFFICIENT)))


class TestSubCloudEvaporationCoefficient(unittest.TestCase):
    """``cevapcu`` sets how fast convective rain evaporates below cloud base.

    The column is chosen so that ECHAM's Kessler chain, not its cap, decides
    the evaporation. ``cuflx`` takes the weaker of two limits per layer:
    the chain ``(√(rain/cover) − cevapcu(k)·Δp·(qs − q))²·cover``, and a cap
    that moistens the layer to no more than 80 % of saturation in one step
    (``zrmin``). Where the cap binds the coefficient changes nothing — in the
    warm, dry column of ``TestSubCloudEvaporationCover`` it does not, at
    ``dt`` = 1800 s, anywhere between 0.5x and 2x ECHAM's value — so a test
    of the coefficient has to sit where the chain is the smaller limit:
    half-saturated sub-cloud air and a 600 s step.
    """

    DT = 600.0
    PRECIP_PER_PLUME_LAYER = 1.0e-3
    # Regression values (float32) of ``_run`` at ECHAM's coefficient: the
    # surface rain, then ``pdmfup_adj``, ``precip_flux`` and ``evap_fraction``
    # per layer, for the constant 0.05 cover (``False``) and the updraft-area
    # cover (``True``). Taken with the profile's ``1.93E-6`` as the literal of
    # iniphy.f90:87-89, which ``TestCevapcuProfile`` compares the profile
    # against exactly.
    PINNED = {
        False: (
            0.00044985744,
            [0.0, 0.001, 0.00039263256, -0.00051661086, -0.00042616433],
            [0.0, 0.0, 0.001, 0.0013926326, 0.0008760218],
            [0.0, 0.0, 0.30368373, 0.3709599, 0.48647687],
        ),
        True: (
            0.0011125670,
            [0.0, 0.001, 0.00048980233, -0.0002662827, -0.000110952766],
            [0.0, 0.0, 0.001, 0.0014898024, 0.0012235197],
            [0.0, 0.0, 0.25509885, 0.17873693, 0.09068327],
        ),
    }

    def setUp(self):
        self.temperature = jnp.array([278.0, 282.0, 286.0, 290.0, 294.0])
        self.pressure = jnp.array([2.0e4, 4.0e4, 6.0e4, 8.0e4, 1.0e5])
        self.dp_lev = jnp.full((5,), 2.0e4)
        self.humidity = 0.5 * jax.vmap(saturation_mixing_ratio)(
            self.pressure, self.temperature)
        self.pdmfup = jnp.array(
            [0.0, 1.0, 1.0, 0.0, 0.0]) * self.PRECIP_PER_PLUME_LAYER
        self.mfu = jnp.array([0.0, 0.05, 0.05, 0.0, 0.0])
        self.tu = jnp.array([0.0, 282.0, 286.0, 0.0, 0.0])

    def _run(self, use_updraft_cover=False, **kwargs):
        return convective_precip_fluxes(
            self.temperature, self.humidity, self.pressure, self.dp_lev, 2,
            self.pdmfup, jnp.zeros((5,)), self.DT,
            updraft_temperature=self.tu, updraft_mass_flux=self.mfu,
            ktype=jnp.array(1), updraft_velocity=2.0,
            use_updraft_cover=use_updraft_cover,
            updraft_layer_mass=self.dp_lev / 9.80665, **kwargs)

    def _evaporated(self, factor, **kwargs):
        """Column rain evaporated, i.e. what ``cudtdq`` charges as a sink."""
        out = self._run(
            cevapcu_coefficient=factor * ECHAM_CEVAPCU_COEFFICIENT, **kwargs)
        return float(jnp.sum(self.pdmfup - out[4]))

    def test_default_coefficient_gives_the_regression_fluxes(self):
        leaf = jnp.asarray(ConvectionParameters.default().cevapcu)
        for flag, (rain, adj, flux, evap) in self.PINNED.items():
            for kwargs in ({}, {"cevapcu_coefficient": ECHAM_CEVAPCU_COEFFICIENT},
                           {"cevapcu_coefficient": leaf}):
                out = self._run(flag, **kwargs)
                # Room for the platform's ``exp`` in the saturation humidity
                # (a change of the profile moves these by 1e-2 or more); the
                # profile itself is compared exactly in ``TestCevapcuProfile``.
                np.testing.assert_allclose(float(out[0]), rain, rtol=1e-5)
                np.testing.assert_allclose(
                    np.asarray(out[4]), adj, rtol=1e-5, atol=1e-12)
                np.testing.assert_allclose(
                    np.asarray(out[5]), flux, rtol=1e-5, atol=1e-12)
                np.testing.assert_allclose(
                    np.asarray(out[7]), evap, rtol=1e-5, atol=1e-12)

    def test_doubling_the_coefficient_changes_the_evaporation_tendency(self):
        total = float(jnp.sum(self.pdmfup))
        evaporated = {f: self._evaporated(f) for f in (0.0, 0.5, 1.0, 2.0)}
        # No coefficient, no evaporation: the chain depletes nothing.
        self.assertEqual(evaporated[0.0], 0.0)
        # More coefficient, more evaporation; doubling it is a first-order
        # change of the tendency (about a fifth of the column's rain here),
        # not a rounding one.
        self.assertGreater(evaporated[1.0], 1.2 * evaporated[0.5])
        self.assertGreater(evaporated[2.0], 1.2 * evaporated[1.0])
        self.assertLess(evaporated[2.0], total)
        # The per-layer ledger the tendencies are built from moves with it,
        # not only the column total: the first sub-cloud layer, which the
        # doubled coefficient evaporates more of, is charged a larger sink.
        doubled = self._run(
            cevapcu_coefficient=2.0 * ECHAM_CEVAPCU_COEFFICIENT)
        default = self._run()
        self.assertLess(float(doubled[4][3]), float(default[4][3]) - 1.0e-4)
        self.assertLess(float(doubled[0]), float(default[0]))
        # The updraft-area cover the HAM submodel selects shares the profile.
        self.assertGreater(
            self._evaporated(2.0, use_updraft_cover=True),
            self._evaporated(0.5, use_updraft_cover=True))

    def test_the_saturation_cap_leaves_the_coefficient_inert_where_it_binds(self):
        # ECHAM's ``zrmin``: a layer is not moistened beyond 80 % of
        # saturation in one step, whatever the chain asks for. In warm, dry
        # sub-cloud air under a 1800 s step the cap is the smaller limit at
        # every coefficient from 0.5x to 1e6x ECHAM's, so the evaporation is
        # the cap's and neither depends on the coefficient nor consumes all
        # the rain. This is ECHAM's formulation, not a plumbing failure; it is
        # why the tests above use half-saturated air and a 600 s step.
        self.humidity = jnp.full((5,), 1.0e-3)
        self.pdmfup = jnp.array([0.0, 2.0e-3, 2.0e-3, 0.0, 0.0])
        self.DT = 1800.0
        capped = [self._evaporated(f) for f in (0.5, 1.0, 2.0, 1.0e6)]
        self.assertEqual(len(set(capped)), 1)
        self.assertGreater(capped[0], 0.0)
        self.assertLess(capped[0], float(jnp.sum(self.pdmfup)))

    def test_coefficient_gradient_is_finite_and_not_zero(self):
        """d(fluxes)/d(cevapcu) is the AD derivative, and it is live.

        ``live_inputs`` asserts the coefficient's reverse gradient is finite
        and not identically zero, which is the failure this guards: a
        parameter the scheme never reads has a trivially finite zero for a
        gradient.
        """
        def fluxes(coefficient):
            out = self._run(cevapcu_coefficient=coefficient)
            return out[0], out[4], out[5], out[7]

        check_gradients(
            fluxes, (jnp.asarray(ECHAM_CEVAPCU_COEFFICIENT),), rtol=2e-2,
            live_inputs=("[0]",))
        # The sign is the physics: a larger coefficient evaporates more, so
        # less rain reaches the surface.
        slope = jax.grad(lambda x: fluxes(x)[0])(
            jnp.asarray(ECHAM_CEVAPCU_COEFFICIENT))
        self.assertTrue(np.isfinite(float(slope)))
        self.assertLess(float(slope), 0.0)


if __name__ == "__main__":
    unittest.main()
